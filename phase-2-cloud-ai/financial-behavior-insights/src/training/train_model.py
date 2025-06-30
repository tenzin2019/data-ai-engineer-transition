import argparse
import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import randint
import joblib
import mlflow
import mlflow.sklearn

# Azure ML imports for registration
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Model

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- Data Loading -----------------
def load_data(data_path: str):
    logger.info(f"Loading data from {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data not found: {data_path}")
    df = pd.read_csv(data_path)
    if "HighAmount" not in df.columns:
        raise ValueError("Target column ('HighAmount') not found in dataset.")
    X = df.drop(columns=["HighAmount"])
    y = df["HighAmount"]
    return X, y

# ---------------- Hyperparameter Tuning -----------------
def tune_hyperparameters(X, y, n_iter=20, cv=3, random_state=42):
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    rf = RandomForestClassifier(random_state=random_state)
    random_search = RandomizedSearchCV(
        rf, param_distributions=param_dist, n_iter=n_iter, cv=cv,
        scoring='roc_auc', random_state=random_state, n_jobs=-1, verbose=1
    )
    random_search.fit(X, y)
    logger.info(f"Best hyperparameters: {random_search.best_params_}")
    logger.info(f"Best CV ROC-AUC score: {random_search.best_score_:.4f}")
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

# ---------------- Model Training & Evaluation -----------------
def train_and_eval(model, X, y, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
    }
    logger.info(f"Test set metrics: {metrics}")
    return model, metrics

# ---------------- Model Registration (Azure ML) -----------------
def register_model_azureml(model_path, model_name="financial-behavior-insights-model", description="Random Forest for HighAmount prediction"):
    logger.info("Registering model to Azure ML Model Registry...")
    try:
        ml_client = MLClient(
            credential=DefaultAzureCredential(),
            subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
            resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
            workspace_name=os.environ["AZURE_WORKSPACE_NAME"]
        )
        azureml_model = Model(
            path=model_path,
            name=model_name,
            description=description,
            type="custom_model"
        )
        registered_model = ml_client.models.create_or_update(azureml_model)
        logger.info(f"Model registered: {registered_model.name} (version {registered_model.version})")
    except Exception as e:
        logger.error(f"Azure ML model registration failed: {e}")

# ---------------- Pipeline Orchestration -----------------
def main(args):
    mlflow.set_experiment("financial-behavior-insights")
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
    tags = {"project": "financial-behavior-insights", "stage": "training"}

    with mlflow.start_run(run_name="rf_hyperopt", tags=tags):
        X, y = load_data(args.input_data)

        # 1. Hyperparameter tuning
        best_model, best_params, best_cv_score = tune_hyperparameters(X, y)
        mlflow.log_metric("best_cv_roc_auc", best_cv_score)

        # 2. Train/evaluate with best model
        model, metrics = train_and_eval(best_model, X, y)
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
            mlflow.log_metric(k, v)

        # 3. Save model & metrics
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "model.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        mlflow.log_artifact(model_path)
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        pd.Series(metrics).to_json(metrics_path)
        mlflow.log_artifact(metrics_path)

        # 4. Register model in Azure ML Model Registry
        register_model_azureml(
            model_path=model_path,
            model_name="financial-behavior-insights-model",
            description="Random Forest for HighAmount prediction"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to input CSV file (local path or Azure ML data mount)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save trained model and metrics."
    )
    args = parser.parse_args()
    main(args)
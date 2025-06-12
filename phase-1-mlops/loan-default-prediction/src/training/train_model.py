import os
import logging
from typing import Tuple, Dict
import pandas as pd
import numpy as np
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from azure.ai.ml.entities import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from sklearn.exceptions import NotFittedError

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# ---- Config ----
EXPERIMENT_NAME = "loan-default-experiment"
MODEL_NAME = "loan-default-model"
DATA_PATH = "src/data/loan_data.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---- Helper Funtions ----

def load_and_preprocess_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and preprocesses the loan data.
    Drops missing rows, scales features, and checks for columns.
    """
    logger.info(f"Loading dataset from {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_csv(path)
    required_cols = ["age", "income", "loan_amount", "default"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Checking for missing valus (should drop those)
    if df.isnull().any().any():
        logger.warning("Missing values found. Dropping rows with NaNs.")
        df = df.dropna()

    X = df[["age", "income", "loan_amount"]].astype("float64")
    y = df["default"].astype("int")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    logger.info(f"Dataset shape: {X_scaled.shape}")
    logger.info("Class distribution: %s", y.value_counts(normalize=True).to_dict())
    return X_scaled, y

def plot_feature_importance(model: RandomForestClassifier, feature_names: list, run_id: str) -> None:
    """Plots and logs feature importances as artifact."""
    # simple featre importance plot, can improve with more featres
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='importance', y='feature')
    plt.title('Feature Importance')
    plt.tight_layout()
    path = f"feature_importance_{run_id}.png"
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()
    os.remove(path)

def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, run_id: str) -> None:
    """Logs confusion matrix plot to MLflow."""
    # logs conf matrix, TODO: support multiclass
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    path = f"confusion_matrix_{run_id}.png"
    plt.tight_layout()
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()
    os.remove(path)

def log_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, run_id: str) -> None:
    """Logs ROC curve plot to MLflow."""
    # need probabilities for roc, will break if not binary
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    path = f"roc_curve_{run_id}.png"
    plt.tight_layout()
    plt.savefig(path)
    mlflow.log_artifact(path)
    plt.close()
    os.remove(path)

def evaluate_model(model: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """
    Evaluates a classifier on X and y. Uses cross-validation when possible.
    Returns a dict of metrics.
    """
    metrics = {}
    min_samples_per_class = y.value_counts().min()
    if min_samples_per_class >= 3:
        n_splits = min(5, min_samples_per_class)
        logger.info("Using %d-fold cross-validation", n_splits)
        cv_scores = cross_val_score(model, X, y, cv=n_splits)
        metrics["cv_mean_accuracy"] = float(cv_scores.mean())
        metrics["cv_std_accuracy"] = float(cv_scores.std())
    else:
        # just use single train set metrics, small data
        logger.warning("Dataset too small for cross-validation.")
        metrics["cv_mean_accuracy"] = float("nan")
        metrics["cv_std_accuracy"] = float("nan")

    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
    except (AttributeError, NotFittedError):
        y_proba = y_pred
        metrics["roc_auc"] = float("nan")

    # standard metrics, f1 can be unstable with imbalnced
    metrics["accuracy"] = float(accuracy_score(y, y_pred))
    metrics["precision"] = float(precision_score(y, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y, y_pred, zero_division=0))
    metrics["f1_score"] = float(f1_score(y, y_pred, zero_division=0))
    metrics["cpu_percent"] = float(psutil.cpu_percent())
    metrics["memory_percent"] = float(psutil.virtual_memory().percent)
    return metrics

def get_ml_client() -> MLClient:
    """
    Initializes the Azure MLClient from env variables. Fails if required values are missing.
    """
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError("Missing AZURE_* env variables for Azure ML connection.")

    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        workspace = ml_client.workspaces.get(name=workspace_name)
        logger.info("Connected to workspace: %s", workspace.name)
        return ml_client
    except Exception as e:
        logger.error("Azure ML client init failed: %s", str(e))
        raise

# ---- Main Process ----

def main():
    """Run the ML training and tracking pipeline."""
    logger.info("Starting training pipeline...")

    try:
        ml_client = get_ml_client()
        tracking_uri = ml_client.workspaces.get(
            name=os.getenv("AZURE_WORKSPACE_NAME")
        ).mlflow_tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    except Exception as e:
        logger.error("Failed MLflow tracking setup: %s", str(e))
        raise

    try:
        experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info("Using experiment '%s' (ID: %s)", experiment.name, experiment.experiment_id)
    except Exception as e:
        logger.error("Failed to set up MLflow experiment: %s", str(e))
        raise

    run_name = f"loan_default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info("Starting new run: %s", run_name)
    try:
        run = mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment.experiment_id,
            nested=False
        )
        run_id = run.info.run_id
        logger.info("Run started: %s", run_id)

        X, y = load_and_preprocess_data(DATA_PATH)
        if len(y) < 10:
            # pretty bad to train/test on same set, just for demo
            logger.warning("Very small dataset. Using all data for training and testing.")
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        train_metrics = evaluate_model(model, X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)

        metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        plot_feature_importance(model, X.columns.tolist(), run_id)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        log_confusion_matrix(y_test, y_pred, run_id)
        log_roc_curve(y_test, y_proba, run_id)

        report = classification_report(y_test, y_pred)
        mlflow.log_text(report, "classification_report.txt")

        model_dir = "model"
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(
            model,
            model_dir,
            signature=signature,
            input_example=input_example
        )
        logger.info("Model logged in MLflow for run: %s", run_id)

        experiment_url = f"{tracking_uri}/#/experiments/{experiment.experiment_id}"
        run_url = f"{experiment_url}/runs/{run_id}"
        print(f"View run {run_name} at: {run_url}")
        print(f"Experiment at: {experiment_url}")

        model_path = f"runs:/{run.info.run_id}/{model_dir}"
        azureml_model = Model(
            path=model_path,
            name=MODEL_NAME,
            description="Loan default model registered at train time",
            type="mlflow_model"
        )
        registered_model = ml_client.models.create_or_update(azureml_model)
        logger.info("Registered Model: %s, version: %s", registered_model.name, registered_model.version)
        mlflow.end_run()
    except Exception as e:
        logger.error("Error in training run: %s", str(e))
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
    finally:
        if mlflow.active_run():
            logger.warning("Found active run. Ending it.")
            mlflow.end_run()

# ---- Run ----

if __name__ == "__main__":
    main()
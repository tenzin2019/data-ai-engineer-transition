import os
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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

# ------------------------------ Config ------------------------------
# ML experiment configuration
EXPERIMENT_NAME = "loan-default-experiment"
MODEL_NAME = "loan-default-model"
DATA_PATH = "src/data/loan_data.csv"

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# ---------------------------- Logging Setup --------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------- Helper Functions -----------------------

def load_and_preprocess_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and preprocess the loan dataset.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Tuple of features DataFrame and target Series
    """
    logger.info(f"📦 Loading dataset from {path}")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    
    df = pd.read_csv(path)
    
    # Basic data validation
    required_columns = ["age", "income", "loan_amount", "default"]
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    if df.isnull().any().any():
        logger.warning("Dataset contains missing values. Dropping rows with missing values.")
        df = df.dropna()
    
    X = df[["age", "income", "loan_amount"]].astype("float64")
    y = df["default"].astype("int")
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    logger.info(f"Dataset shape: {X_scaled.shape}")
    logger.info(f"Class distribution:\n{y.value_counts(normalize=True)}")
    
    return X_scaled, y

def plot_feature_importance(model: RandomForestClassifier, feature_names: list, run_id: str) -> None:
    """Plot and log feature importance."""
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
    """Plot and log confusion matrix."""
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
    """Plot and log ROC curve."""
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
    Evaluate model performance using cross-validation and various metrics.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Cross-validation scores (if dataset is large enough)
    min_samples_per_class = y.value_counts().min()
    if min_samples_per_class >= 3:
        n_splits = min(5, min_samples_per_class)
        logger.info(f"Using {n_splits}-fold cross-validation")
        cv_scores = cross_val_score(model, X, y, cv=n_splits)
        metrics.update({
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
        })
    else:
        logger.warning("Dataset too small for cross-validation. Using single train set metrics.")
        metrics.update({
            "cv_mean_accuracy": float("nan"),
            "cv_std_accuracy": float("nan"),
        })
    
    # Predictions on full dataset
    y_pred = model.predict(X)
    try:
        y_proba = model.predict_proba(X)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
    except (AttributeError, NotFittedError):
        y_proba = y_pred
        metrics["roc_auc"] = float("nan")
    
    # Classification metrics
    metrics.update({
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y, y_pred, zero_division=0)),
    })
    
    # System metrics
    metrics.update({
        "cpu_percent": float(psutil.cpu_percent()),
        "memory_percent": float(psutil.virtual_memory().percent)
    })
    
    return metrics

def get_ml_client() -> MLClient:
    """
    Initialize and return the Azure ML client with proper error handling.
    
    Returns:
        MLClient: Authenticated Azure ML client
    
    Raises:
        ValueError: If required environment variables are missing
        Exception: For other Azure ML client initialization errors
    """
    # Validate environment variables
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
    
    if not all([subscription_id, resource_group, workspace_name]):
        raise ValueError(
            "Missing required environment variables. Please ensure the following are set:\n"
            "- AZURE_SUBSCRIPTION_ID\n"
            "- AZURE_RESOURCE_GROUP\n"
            "- AZURE_WORKSPACE_NAME"
        )
    
    try:
        credential = DefaultAzureCredential()
        
        # Initialize MLClient
        ml_client = MLClient(
            credential=credential,
            subscription_id=subscription_id,
            resource_group_name=resource_group,
            workspace_name=workspace_name
        )
        
        # Verify connection by attempting to get the workspace
        workspace = ml_client.workspaces.get(name=workspace_name)
        logger.info(f"Successfully connected to workspace: {workspace.name}")
        
        return ml_client
    
    except Exception as e:
        logger.error(f"Failed to initialize Azure ML client: {str(e)}")
        raise

# ---------------------------- Main Process ---------------------------

def main():
    """Main training pipeline."""
    logger.info("🚀 Starting training pipeline...")
    
    # --- Get Azure ML MLflow tracking URI ---
    logger.info("🔑 Authenticating with Azure ML workspace...")
    try:
        ml_client = get_ml_client()
        tracking_uri = ml_client.workspaces.get(
            name=os.getenv("AZURE_WORKSPACE_NAME")
        ).mlflow_tracking_uri
        
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"🔗 MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        logger.info(f"📂 Working Directory: {os.getcwd()}")
    except Exception as e:
        logger.error(f"Failed to set up MLflow tracking: {str(e)}")
        raise
    
    # --- Create or get experiment in Azure ML ---
    try:
        experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(f"Using experiment '{experiment.name}' (ID: {experiment.experiment_id})")
    except Exception as e:
        logger.error(f"Failed to set up MLflow experiment: {str(e)}")
        raise
    
    # Start new run with timestamp
    run_name = f"loan_default_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger.info(f"Starting new run: {run_name}")
    
    try:
        # Force creation of a new run
        run = mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment.experiment_id,
            nested=False
        )
        run_id = run.info.run_id
        logger.info(f"Started run with ID: {run_id}")
        
        # --- Load and preprocess data ---
        X, y = load_and_preprocess_data(DATA_PATH)
        
        # Handle small dataset case
        if len(y) < 10:  # Very small dataset
            logger.warning("Dataset too small for train-test split. Using entire dataset for training.")
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
        
        # --- Train model ---
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight='balanced'  # Handle class imbalance
        )
        model.fit(X_train, y_train)
        
        # --- Evaluate model ---
        train_metrics = evaluate_model(model, X_train, y_train)
        test_metrics = evaluate_model(model, X_test, y_test)
        
        # Add prefix to metrics
        metrics = {
            f"train_{k}": v for k, v in train_metrics.items()
        }
        metrics.update({
            f"test_{k}": v for k, v in test_metrics.items()
        })
        
        # --- Log metrics and artifacts ---
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        
        # Log feature importance
        plot_feature_importance(model, X.columns.tolist(), run_id)
        
        # Log confusion matrix and ROC curve for test set
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        log_confusion_matrix(y_test, y_pred, run_id)
        log_roc_curve(y_test, y_proba, run_id)
        
        # Log classification report
        report = classification_report(y_test, y_pred)
        mlflow.log_text(report, "classification_report.txt")
        
        # --- Log model with MLflow ---
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]
        
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example
        )
        
        logger.info(f"✅ Model trained and logged for run ID: {run_id}")
        
        # Print links to MLflow UI
        experiment_url = f"{tracking_uri}/#/experiments/{experiment.experiment_id}"
        run_url = f"{experiment_url}/runs/{run_id}"
        
        print(f"🏃 View run {run_name} at: {run_url}")
        print(f"🧪 View experiment at: {experiment_url}")
        
        # End the run explicitly
        mlflow.end_run()
    
    except Exception as e:
        logger.error(f"Error during training run: {str(e)}")
        # Ensure run is ended even if there's an error
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
    
    finally:
        # Double-check no runs are left active
        if mlflow.active_run():
            logger.warning("Found active run in finally block. Ending it.")
            mlflow.end_run()

# ------------------------------ Run ------------------------------

if __name__ == "__main__":
    main()
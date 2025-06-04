import os
import logging
import pandas as pd
import psutil
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# ------------------------------ Config ------------------------------
EXPERIMENT_NAME = "loan-default-experiment"
MODEL_NAME = "loan-default-model"
DATA_PATH = "src/data/loan_data.csv"
TRACKING_URI = "http://localhost:9090"

# ---------------------------- Logging Setup --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------- Helper Functions -----------------------

def load_data(path):
    """Load and preprocess the dataset."""
    logging.info(f"📦 Loading dataset from {path}")
    df = pd.read_csv(path)
    X = df[["age", "income", "loan_amount"]].astype("float64")
    y = df["default"].astype("int")
    return X, y

def log_confusion_matrix(y_true, y_pred, run_id):
    """Generate and log the confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
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

def log_roc_curve(y_true, y_proba, run_id):
    """Generate and log the ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
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

# ---------------------------- Main Process ---------------------------

def main():
    logging.info("🚀 Starting training pipeline...")
    mlflow.set_tracking_uri(TRACKING_URI)
    logging.info(f"🔗 Tracking URI: {mlflow.get_tracking_uri()}")
    logging.info(f"📂 Working Directory: {os.getcwd()}")

    # Load data
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Setup experiment
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
        logging.info(f"🧪 Created new experiment ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logging.info(f"🔁 Using existing experiment ID: {experiment_id}")

    with mlflow.start_run(experiment_id=experiment_id) as run:
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_proba = y_pred

        # Compute and log metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=1),
            "recall": recall_score(y_test, y_pred, zero_division=1),
            "f1_score": f1_score(y_test, y_pred, zero_division=1),
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

        mlflow.log_metrics(metrics)
        log_confusion_matrix(y_test, y_pred, run.info.run_id)
        log_roc_curve(y_test, y_proba, run.info.run_id)

        # Log model
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=input_example
        )

        # Register model
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=MODEL_NAME
        )

        logging.info(f"✅ Model training and registration completed for run ID: {run.info.run_id}")

# ------------------------------ Run ------------------------------

if __name__ == "__main__":
    main()
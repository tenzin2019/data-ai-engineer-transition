import argparse
import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import mlflow

# ---- Logging Setup ----
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def load_data(data_path):
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    # Minimal cleaning – customize for your schema!
    if "target" not in df.columns:
        raise ValueError("Target column ('target') not found in dataset.")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan
    }
    return model, metrics

def main(args):
    mlflow.start_run()
    X, y = load_data(args.input_data)
    model, metrics = train_and_eval(X, y)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
        mlflow.log_metric(k, v)

    # Save model to outputs (required for Azure ML)
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    mlflow.log_artifact(model_path)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    pd.Series(metrics).to_json(metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")
    mlflow.log_artifact(metrics_path)

    mlflow.end_run()

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
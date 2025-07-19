"""
train_model.py

Trains a Random Forest model to predict high-amount transactions in the comprehensive banking dataset.
Supports hyperparameter tuning, evaluation, and optional registration with Azure ML.

Usage:
    python train_model.py --input-data <processed_csv> [--output-dir <dir>] [--register-model]

Arguments:
    --input-data: Path to the processed CSV file
    --output-dir: Directory to save model artifacts (default: outputs/)
    --register-model: Register the trained model with Azure ML (requires Azure config)

MLOps Best Practices:
    - Uses logging for traceability
    - Validates data and environment
    - Modularizes training, evaluation, and registration
    - Supports MLflow tracking
    - Handles errors gracefully
"""

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
import json
import sys
from pathlib import Path

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Azure ML imports for registration (optional)
try:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import Model
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False
    logger.warning("Azure ML dependencies not available. Model registration will be skipped.")

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------- Environment Validation -----------------
def validate_environment_variables():
    """
    Validate required environment variables for Azure ML integration.
    Raises:
        ValueError: If any required environment variable is missing.
    """
    required_vars = [
        "AZURE_SUBSCRIPTION_ID",
        "AZURE_RESOURCE_GROUP", 
        "AZURE_WORKSPACE_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("All required environment variables are set")

# ---------------- Data Validation -----------------
def validate_dataframe(df: pd.DataFrame, target_column: str = "HighAmount"):
    """
    Validate dataframe for training requirements.
    Args:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.
    Raises:
        ValueError: If validation fails.
    """
    if df.empty:
        raise ValueError("Dataframe is empty")
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    # Check for missing values in target column
    if df[target_column].isnull().any():
        raise ValueError(f"Missing values found in target column '{target_column}'")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numeric")
    
    # Check for duplicate columns robustly
    if df.columns.duplicated().any():
        raise ValueError("Duplicate columns found in dataset")
    
    logger.info(f"Data validation passed. Shape: {df.shape}")

# ---------------- Data Loading -----------------
def load_data(data_path: str, target_column: str = "HighAmount", chunk_size: int = None):
    """
    Load and validate data with optional chunking for large files.
    Args:
        data_path (str): Path to the CSV file.
        target_column (str): Name of the target column.
        chunk_size (int): Optional chunk size for large files.
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and target variable.
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If validation fails.
    """
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input data not found: {data_path}")
    
    # Check file size for memory considerations
    file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
    logger.info(f"File size: {file_size:.2f} MB")
    
    if file_size > 100 and chunk_size is None:  # If file > 100MB, suggest chunking
        logger.warning("Large file detected. Consider using chunk_size parameter for memory efficiency")
    
    try:
        if chunk_size and file_size > 100:
            # Read in chunks for large files
            chunks = []
            for chunk in pd.read_csv(data_path, chunksize=chunk_size):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_csv(data_path)
        
        # Validate the dataframe
        validate_dataframe(df, target_column)
        
        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Successfully loaded data. Features: {X.shape[1]}, Samples: {X.shape[0]}")
        return X, y
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# ---------------- Hyperparameter Tuning -----------------
def tune_hyperparameters(X, y, n_iter=20, cv=3, random_state=42, n_jobs=-1):
    """
    Tune hyperparameters for Random Forest using RandomizedSearchCV.
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        n_iter (int): Number of iterations for search.
        cv (int): Number of cross-validation folds.
        random_state (int): Random seed.
        n_jobs (int): Number of parallel jobs.
    Returns:
        Tuple[RandomForestClassifier, dict, float]: Best estimator, best params, best score.
    Raises:
        Exception: If tuning fails.
    """
    logger.info(f"Starting hyperparameter tuning with {n_iter} iterations")
    
    if X.shape[0] < cv * 2:
        logger.warning(f"Small dataset ({X.shape[0]} samples) may not be suitable for {cv}-fold CV")
        cv = min(cv, X.shape[0] // 2)
    
    param_dist = {
        'n_estimators': randint(50, 200),
        'max_depth': randint(3, 20),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'bootstrap': [True, False],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    rf = RandomForestClassifier(random_state=random_state)
    
    try:
        random_search = RandomizedSearchCV(
            rf, param_distributions=param_dist, n_iter=n_iter, cv=cv,
            scoring='roc_auc', random_state=random_state, n_jobs=n_jobs, verbose=1
        )
        random_search.fit(X, y)
        
        logger.info(f"Best hyperparameters: {random_search.best_params_}")
        logger.info(f"Best CV ROC-AUC score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_, random_search.best_params_, random_search.best_score_
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        raise

# ---------------- Model Training & Evaluation -----------------
def train_and_eval(model, X, y, random_state=42, test_size=0.2):
    """
    Train and evaluate model with comprehensive metrics.
    Args:
        model: The model to train.
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        random_state (int): Random seed.
        test_size (float): Proportion of test set.
    Returns:
        Tuple[model, dict, tuple]: Trained model, metrics, and test data/results.
    Raises:
        Exception: If training or evaluation fails.
    """
    logger.info("Starting model training and evaluation")
    
    if len(np.unique(y)) < 2:
        raise ValueError("Target variable has only one class. Cannot perform classification.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
    
    try:
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
        return model, metrics, (X_test, y_test, y_pred, y_proba)
        
    except Exception as e:
        logger.error(f"Model training/evaluation failed: {e}")
        raise

# ---------------- Model Validation -----------------
def validate_model(model, X_test, y_test):
    """
    Validate model can make predictions and save/load properly.
    Args:
        model: Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
    Raises:
        Exception: If validation fails.
    """
    try:
        # Test prediction
        _ = model.predict(X_test[:5])
        if hasattr(model, "predict_proba"):
            _ = model.predict_proba(X_test[:5])
        
        # Test serialization
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            joblib.dump(model, temp_path)
            loaded_model = joblib.load(temp_path)
            
            # Test loaded model
            _ = loaded_model.predict(X_test[:5])
            
            logger.info("Model validation passed")
            return True
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except Exception as e:
        logger.error(f"Model validation failed: {e}")
        return False

# ---------------- Model Registration (Azure ML) -----------------
def register_model_azureml(model_path, model_name="financial-behavior-insights-model", 
                          description="Random Forest for HighAmount prediction"):
    """Register model to Azure ML Model Registry with proper error handling."""
    if not AZURE_AVAILABLE:
        logger.warning("Azure ML not available. Skipping model registration.")
        return None
    
    logger.info("Registering model to Azure ML Model Registry...")
    
    try:
        # Validate environment variables
        validate_environment_variables()
        
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
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
        return registered_model
        
    except Exception as e:
        logger.error(f"Azure ML model registration failed: {e}")
        raise  # Re-raise to indicate failure

# ---------------- Output Directory Validation -----------------
def validate_output_directory(output_dir: str):
    """Validate and create output directory with proper permissions."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions
        test_file = output_path / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        
        logger.info(f"Output directory validated: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Output directory validation failed: {e}")
        return False

# ---------------- MLflow Setup -----------------
def setup_mlflow():
    """Setup MLflow with proper error handling."""
    try:
        # Try to set tracking URI if not already set
        if not mlflow.get_tracking_uri():
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            logger.info("Set MLflow tracking URI to local SQLite database")
        
        mlflow.set_experiment("financial-behavior-insights")
        logger.info("MLflow setup completed")
        return True
        
    except Exception as e:
        logger.warning(f"MLflow setup failed: {e}. Continuing without MLflow tracking.")
        return False

# ---------------- Pipeline Orchestration -----------------
def main(args):
    """Main training pipeline with comprehensive error handling."""
    logger.info("Starting training pipeline")
    
    try:
        # Setup MLflow
        mlflow_available = setup_mlflow()
        
        # Validate output directory
        if not validate_output_directory(args.output_dir):
            raise ValueError(f"Cannot write to output directory: {args.output_dir}")
        
        # Load and validate data
        X, y = load_data(args.input_data, chunk_size=args.chunk_size)
        
        # Start MLflow run if available
        if mlflow_available:
            mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
            tags = {"project": "financial-behavior-insights", "stage": "training"}
            mlflow_run = mlflow.start_run(run_name="rf_hyperopt", tags=tags)
        else:
            mlflow_run = None
        
        try:
            # 1. Hyperparameter tuning
            best_model, best_params, best_cv_score = tune_hyperparameters(
                X, y, n_iter=args.n_iter, cv=args.cv, random_state=args.random_state
            )
            
            if mlflow_available:
                mlflow.log_metric("best_cv_roc_auc", best_cv_score)
                mlflow.log_params(best_params)

            # 2. Train/evaluate with best model
            model, metrics, eval_data = train_and_eval(
                best_model, X, y, random_state=args.random_state, test_size=args.test_size
            )
            
            # Log metrics
            for k, v in metrics.items():
                logger.info(f"{k}: {v:.4f}")
                if mlflow_available:
                    mlflow.log_metric(k, v)

            # 3. Validate model
            if not validate_model(model, eval_data[0], eval_data[1]):
                raise ValueError("Model validation failed")

            # 4. Save model & metrics
            model_path = os.path.join(args.output_dir, "model.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            if mlflow_available:
                mlflow.log_artifact(model_path)
                # Log with MLflow custom pyfunc
                # Use self-contained model class to avoid import issues in deployment
                from mlflow.pyfunc.model import PythonModel
                
                class FinancialBehaviorModel(PythonModel):
                    """Self-contained MLflow model wrapper for deployment."""
                    
                    def load_context(self, context):
                        import joblib
                        self.model = joblib.load(context.artifacts["model_path"])
                    
                    def predict(self, context, model_input):
                        import numpy as np
                        import pandas as pd
                        
                        if isinstance(model_input, pd.DataFrame):
                            X = model_input.values
                        else:
                            X = np.array(model_input)
                        
                        if len(X.shape) == 1:
                            X = X.reshape(1, -1)
                        
                        if X.shape[1] != 12:
                            raise ValueError(f"Expected 12 features, got {X.shape[1]}")
                        
                        return self.model.predict(X)
                input_example = X.iloc[[0]].copy()
                signature = mlflow.models.infer_signature(input_example, model.predict(input_example.values))
                mlflow.pyfunc.log_model(
                    artifact_path="financial_behavior_model",
                    python_model=FinancialBehaviorModel(),
                    artifacts={"model_path": model_path},
                    input_example=input_example,
                    signature=signature
                )
            
            # Save metrics
            metrics_path = os.path.join(args.output_dir, "metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")
            
            if mlflow_available:
                mlflow.log_artifact(metrics_path)

            # 5. Register model in Azure ML Model Registry (if requested)
            if args.register_model:
                register_model_azureml(
                    model_path=model_path,
                    model_name=args.model_name,
                    description=args.model_description
                )

            logger.info("Training pipeline completed successfully")
            
        finally:
            if mlflow_run:
                mlflow.end_run()
                
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest model for financial behavior insights")
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
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility."
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of iterations for hyperparameter tuning."
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for testing."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for reading large CSV files (for memory efficiency)."
    )
    parser.add_argument(
        "--register-model",
        action="store_true",
        help="Register model in Azure ML Model Registry."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="financial-behavior-insights-model",
        help="Name for the model in Azure ML Model Registry."
    )
    parser.add_argument(
        "--model-description",
        type=str,
        default="Random Forest for HighAmount prediction",
        help="Description for the model in Azure ML Model Registry."
    )
    
    args = parser.parse_args()
    main(args)
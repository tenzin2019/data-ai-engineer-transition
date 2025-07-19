"""
register_model.py

Utility to register an existing trained model in MLflow.
This is useful when you have a trained model but haven't registered it yet.

Usage:
    python register_model.py
"""

import os
import sys
import mlflow
import mlflow.exceptions
import mlflow.tracking
import joblib
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import MLflow pyfunc base class
import mlflow.pyfunc
from mlflow.pyfunc.model import PythonModel

class FinancialBehaviorModel(PythonModel):
    """
    Custom MLflow model wrapper for the financial behavior prediction model.
    
    This wrapper ensures compatibility with MLflow's pyfunc interface and provides
    consistent prediction behavior across different deployment scenarios.
    """
    
    def load_context(self, context):
        """
        Load the scikit-learn model from the artifacts.
        
        Args:
            context: MLflow context containing artifacts path
        """
        import joblib
        self.model = joblib.load(context.artifacts["model_path"])
    
    def predict(self, context, model_input):
        """
        Make predictions on the input data.
        
        Args:
            context: MLflow context (not used in this implementation)
            model_input: Input data as pandas DataFrame or numpy array
            
        Returns:
            np.ndarray: Array of predictions
            
        Raises:
            ValueError: If input shape is incorrect
        """
        import numpy as np
        import pandas as pd
        
        # Convert input to numpy array if needed
        if isinstance(model_input, pd.DataFrame):
            X = model_input.values
        else:
            X = np.array(model_input)
        
        # Ensure 2D array
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Validate feature count
        if X.shape[1] != 12:
            raise ValueError(f"Expected 12 features, got {X.shape[1]}")
        
        # Make predictions
        return self.model.predict(X)


def register_existing_model():
    """Register the existing model in MLflow."""
    
    # Check if model exists
    model_path = Path("outputs/model.joblib")
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first using:")
        print("  python src/training/train_model.py --input-data data/processed/Comprehensive_Banking_Database_processed.csv")
        return False
    
    # Set MLflow tracking URI to current directory
    mlruns_path = Path.cwd() / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")
    
    # Handle experiment creation/restoration
    experiment_name = "financial-behavior-model-registration"
    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e:
        if "deleted experiment" in str(e):
            print(f"Experiment '{experiment_name}' was deleted. Creating new experiment...")
            # Get the experiment and restore or create new
            client = mlflow.tracking.MlflowClient()
            try:
                # Try to restore deleted experiment
                exp = client.get_experiment_by_name(experiment_name)
                if exp and exp.lifecycle_stage == "deleted":
                    client.restore_experiment(exp.experiment_id)
                    print(f"Restored experiment '{experiment_name}'")
                    mlflow.set_experiment(experiment_name)
                else:
                    raise e
            except:
                # If restore fails, create with a new name
                import time
                new_name = f"{experiment_name}-{int(time.time())}"
                print(f"Creating new experiment '{new_name}'")
                mlflow.set_experiment(new_name)
        else:
            raise e
    
    print(f"Registering model from {model_path}")
    
    # Load sample data to infer signature
    data_path = Path("data/processed/Comprehensive_Banking_Database_processed.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, nrows=5)
        X_sample = df.drop('HighAmount', axis=1)
        
        # Load model to get predictions for signature
        model = joblib.load(model_path)
        y_sample = model.predict(X_sample)
        
        # Infer signature
        from mlflow.models import infer_signature
        signature = infer_signature(X_sample, y_sample)
    else:
        print("Warning: Sample data not found, registering without signature")
        signature = None
        X_sample = None
    
    # Define conda environment
    conda_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.8",
            "scikit-learn=1.3.0",
            "xgboost=2.0.3",
            "joblib=1.2.0",
            "pandas=1.5.3",
            "numpy=1.23.5",
            {
                "pip": [
                    "mlflow==2.14.1",
                ]
            }
        ],
    }
    
    # Start MLflow run
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("task", "binary_classification")
        mlflow.log_param("target", "HighAmount")
        
        # Log metrics if available
        metrics_path = Path("outputs/metrics.json")
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
        
        # Log the model using pyfunc
        log_model_kwargs = {
            "artifact_path": "model",
            "python_model": FinancialBehaviorModel(),
            "artifacts": {"model_path": str(model_path)},
            "conda_env": conda_env,
            "registered_model_name": "financial-behavior-model"
        }
        
        # Add optional parameters only if they exist
        if signature is not None:
            log_model_kwargs["signature"] = signature
        if X_sample is not None:
            log_model_kwargs["input_example"] = X_sample
            
        mlflow.pyfunc.log_model(**log_model_kwargs)
        
        print(f"Model logged in run: {run.info.run_id}")
        
        # Get the latest model version
        client = mlflow.MlflowClient()
        model_version = client.search_model_versions(
            f"name='financial-behavior-model'",
            order_by=["version_number DESC"],
            max_results=1
        )[0]
        
        # Set alias for the model
        client.set_registered_model_alias(
            name="financial-behavior-model",
            alias="production",
            version=model_version.version
        )
        
        print(f"Model registered as 'financial-behavior-model' version {model_version.version}")
        print(f"Alias 'production' set for version {model_version.version}")
        print("\nYou can now test the model with:")
        print("  python src/serving/test_local.py")
        print("\nOr load it with:")
        print("  model = mlflow.pyfunc.load_model('models:/financial-behavior-model@production')")
        
    return True


if __name__ == "__main__":
    success = register_existing_model()
    sys.exit(0 if success else 1) 
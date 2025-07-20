import mlflow.pyfunc
import joblib
import numpy as np
import pandas as pd

class FinancialBehaviorModel(mlflow.pyfunc.PythonModel):
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
        preds = self.model.predict(X)
        
        # Return predictions as numpy array (standard MLflow pyfunc format)
        return preds 
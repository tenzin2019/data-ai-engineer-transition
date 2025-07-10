import os
import json
import joblib
import numpy as np
import logging
import pandas as pd
from typing import Dict, Any, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
feature_names = None

def init():
    """
    Called once when the endpoint starts up. Loads the model from the Azure ML mount path.
    """
    global model, feature_names
    
    try:
        # AzureML mounts the registered model under AZUREML_MODEL_DIR
        model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
        model_path = os.path.join(model_dir, "model.joblib")
        
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        model = joblib.load(model_path)
        
        # Validate model has required methods
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded model does not have 'predict' method")
        
        # Get feature names if available (for validation)
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_
            logger.info(f"Model loaded with {len(feature_names)} features: {list(feature_names)}")
        else:
            logger.warning("Model does not have feature names. Input validation will be limited.")
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def validate_input(data: Union[str, Dict], expected_features: int = 12) -> np.ndarray:
    """
    Validate and preprocess input data.
    
    Args:
        data: Input data as string or dict
        expected_features: Number of expected features
        
    Returns:
        Preprocessed numpy array
    """
    try:
        # Parse JSON if string
        if isinstance(data, str):
            data = json.loads(data)
        
        # Extract features
        if "data" not in data:
            raise ValueError("Input must contain 'data' field")
        
        X = np.array(data["data"])
        
        # Validate shape
        if len(X.shape) == 1:
            X = X.reshape(1, -1)  # Single sample
        
        if X.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
        
        # Convert to float if needed
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        
        # Check for NaN values
        if np.isnan(X).any():
            raise ValueError("Input contains NaN values")
        
        logger.info(f"Input validated: {X.shape}")
        return X
        
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        raise

def run(raw_data: Union[str, Dict]) -> Dict[str, Any]:
    """
    Called per request. Parses incoming JSON, makes a prediction, returns results.
    
    Expected input format:
    {
        "data": [
            [feature1, feature2, ..., feature12],
            [feature1, feature2, ..., feature12],
            ...
        ]
    }
    
    Returns:
    {
        "predictions": [0, 1, 0, ...],
        "probabilities": [0.1, 0.9, 0.2, ...],  # if available
        "status": "success"
    }
    """
    try:
        logger.info("Processing prediction request")
        
        # Validate model is loaded
        if model is None:
            raise RuntimeError("Model not loaded. Check initialization.")
        
        # Validate and preprocess input
        X = validate_input(raw_data)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Get probabilities if available
        result = {
            "predictions": predictions.tolist(),
            "status": "success",
            "input_shape": X.shape,
            "model_type": type(model).__name__
        }
        
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X)[:, 1].tolist()
                result["probabilities"] = probabilities
            except Exception as e:
                logger.warning(f"Could not get probabilities: {e}")
        
        logger.info(f"Prediction completed: {len(predictions)} samples")
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "error": str(e),
            "status": "error",
            "predictions": None,
            "probabilities": None
        }

def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to verify model is loaded and ready.
    """
    try:
        if model is None:
            return {"status": "error", "message": "Model not loaded"}
        
        # Test prediction with dummy data
        dummy_data = np.zeros((1, 12), dtype=np.float64)
        _ = model.predict(dummy_data)
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_type": type(model).__name__,
            "features": len(dummy_data[0])
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "model_loaded": model is not None
        }
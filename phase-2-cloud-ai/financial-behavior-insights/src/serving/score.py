#!/usr/bin/env python3
"""
score.py

Azure ML scoring script for the financial behavior prediction model.
This script loads the MLflow model and provides the scoring endpoint.

Based on Azure ML best practices for MLflow model deployment.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the model
model = None
scaler = None
feature_columns = None

def init():
    """
    Initialize the model. This function is called once when the container starts.
    """
    global model, scaler, feature_columns
    
    try:
        import joblib
        
        # Get the model path from environment variable
        model_path = os.getenv('AZUREML_MODEL_DIR')
        if not model_path:
            raise ValueError("AZUREML_MODEL_DIR environment variable not set")
        
        # The model is directly in the model directory (custom model, not MLflow)
        model_file = os.path.join(model_path, "model_compatible.joblib")
        
        logger.info(f"Loading model from: {model_file}")
        
        # Load the compatible model (which is a dictionary)
        model_dict = joblib.load(model_file)
        
        # Extract components from the dictionary
        model = model_dict['model']
        scaler = model_dict['scaler']
        feature_columns = model_dict['feature_columns']
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Scaler type: {type(scaler)}")
        logger.info(f"Feature columns: {len(feature_columns)} features")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

def run(raw_data):
    """
    Score the model with the provided data.
    
    Args:
        raw_data: JSON string containing the input data
        
    Returns:
        JSON string containing the prediction results
    """
    try:
        # Check if model is loaded
        if model is None:
            raise ValueError("Model not loaded")
        
        if scaler is None:
            raise ValueError("Scaler not loaded")
        
        if feature_columns is None:
            raise ValueError("Feature columns not loaded")
        
        # Parse the input data
        data = json.loads(raw_data)
        
        # Convert to DataFrame - handle both single row and multiple rows
        if isinstance(data, dict):
            # Single row - create DataFrame with index
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            # Multiple rows
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unexpected data format: {type(data)}")
        
        # Log input data for debugging
        logger.info(f"Input data shape: {df.shape}")
        logger.info(f"Input columns: {list(df.columns)}")
        logger.info(f"Expected features: {feature_columns}")
        
        # Ensure we have the correct feature columns
        if not all(col in df.columns for col in feature_columns):
            missing_cols = [col for col in feature_columns if col not in df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Select only the required features
        X = df[feature_columns]
        logger.info(f"Features shape: {X.shape}")
        
        # Apply scaling
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Convert predictions to list for JSON serialization
        prediction_list = predictions.tolist()
        
        # Return results
        return json.dumps({
            "predictions": prediction_list,
            "model_type": str(type(model)),
            "features_used": feature_columns
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return json.dumps({
            "error": str(e),
            "predictions": None
        })

def health_check() -> str:
    """
    Health check endpoint.
    """
    if model is None:
        return json.dumps({'status': 'unhealthy', 'error': 'Model not loaded'})
    else:
        return json.dumps({'status': 'healthy'}) 
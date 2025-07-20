#!/usr/bin/env python3
"""
retrain_compatible_model.py

Script to retrain the financial behavior model with scikit-learn 1.1.3
to make it compatible with Azure ML deployment.

This script ensures the model is trained with a compatible scikit-learn version
and saves it in the correct format for Azure ML deployment.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def check_sklearn_version():
    """Check if scikit-learn version is compatible."""
    try:
        import sklearn
        version = sklearn.__version__
        logger.info(f"Current scikit-learn version: {version}")
        
        # Check if version is compatible (1.0.2 or 1.1.3)
        if version.startswith(('1.0.2', '1.1.3')):
            logger.info("✅ scikit-learn version is compatible with Azure ML")
            return True
        else:
            logger.warning(f"⚠️ scikit-learn version {version} may not be compatible with Azure ML")
            logger.info("Recommended versions: 1.0.2 or 1.1.3")
            return False
    except ImportError:
        logger.error("❌ scikit-learn not found")
        return False

def load_and_validate_data(data_path):
    """Load and validate the training data."""
    logger.info(f"Loading data from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"❌ Data file not found: {data_path}")
        return None, None, None
    
    try:
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"✅ Loaded data with shape: {df.shape}")
        
        # Check for required columns
        if 'HighAmount' not in df.columns:
            logger.error("❌ Target column 'HighAmount' not found in dataset")
            return None, None, None
        
        # Prepare features and target
        feature_columns = [col for col in df.columns if col != 'HighAmount']
        X = df[feature_columns]
        y = df['HighAmount']
        
        # Validate data
        if X.empty or y.empty:
            logger.error("❌ Empty feature or target data")
            return None, None, None
        
        # Check for missing values
        missing_features = X.isnull().sum().sum()
        missing_target = y.isnull().sum()
        
        if missing_features > 0:
            logger.warning(f"⚠️ Found {missing_features} missing values in features")
            X = X.fillna(X.mean())  # Fill with mean for numeric columns
        
        if missing_target > 0:
            logger.error(f"❌ Found {missing_target} missing values in target")
            return None, None, None
        
        logger.info(f"✅ Features: {len(feature_columns)}")
        logger.info(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_columns
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        return None, None, None

def train_compatible_model(X, y, feature_columns):
    """Train the model with compatible scikit-learn version."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
        
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Scale features
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with compatible sklearn version
        logger.info("Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"✅ Model accuracy: {accuracy:.4f}")
        
        # Detailed evaluation
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 10 most important features:")
        logger.info(feature_importance.head(10))
        
        return model, scaler, feature_columns, accuracy, {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
    except ImportError as e:
        logger.error(f"❌ Failed to import sklearn: {e}")
        logger.error("Please ensure scikit-learn is installed")
        return None, None, None, None, None
    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")
        return None, None, None, None, None

def save_model_artifacts(model, scaler, feature_columns, accuracy, output_dir="outputs"):
    """Save model artifacts in the correct format."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("simple_model", exist_ok=True)
        
        logger.info("Saving model artifacts...")
        
        # Save complete model data (for Azure ML deployment)
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'accuracy': accuracy,
            'model_type': 'RandomForestClassifier',
            'sklearn_version': '1.1.3'
        }
        
        model_path = os.path.join(output_dir, "model_compatible.joblib")
        joblib.dump(model_data, model_path)
        logger.info(f"✅ Model saved to {model_path}")
        
        # Save simple model for basic deployment
        simple_model_path = os.path.join("simple_model", "model.pkl")
        joblib.dump(model, simple_model_path)
        logger.info(f"✅ Simple model saved to {simple_model_path}")
        
        # Save model info
        model_info = {
            'accuracy': accuracy,
            'feature_count': len(feature_columns),
            'model_type': 'RandomForestClassifier',
            'sklearn_version': '1.1.3',
            'feature_columns': feature_columns
        }
        
        import json
        info_path = os.path.join(output_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        logger.info(f"✅ Model info saved to {info_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving model artifacts: {e}")
        return False

def retrain_model(data_path="data/processed/Comprehensive_Banking_Database_processed.csv"):
    """Main function to retrain the model with compatible sklearn version."""
    logger.info("🔄 Starting model retraining for Azure ML compatibility...")
    
    # Check sklearn version
    if not check_sklearn_version():
        logger.warning("⚠️ Proceeding with current sklearn version")
    
    # Load and validate data
    X, y, feature_columns = load_and_validate_data(data_path)
    if X is None:
        logger.error("❌ Failed to load data")
        return False
    
    # Train model
    model, scaler, feature_columns, accuracy, evaluation = train_compatible_model(X, y, feature_columns)
    if model is None:
        logger.error("❌ Failed to train model")
        return False
    
    # Save model artifacts
    if not save_model_artifacts(model, scaler, feature_columns, accuracy):
        logger.error("❌ Failed to save model artifacts")
        return False
    
    logger.info("🎉 Model retraining completed successfully!")
    logger.info(f"📊 Final accuracy: {accuracy:.4f}")
    logger.info("📁 Model artifacts saved to outputs/ and simple_model/")
    
    return True

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Retrain model for Azure ML compatibility")
    parser.add_argument(
        "--data-path",
        default="data/processed/Comprehensive_Banking_Database_processed.csv",
        help="Path to the processed data file"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for model artifacts"
    )
    
    args = parser.parse_args()
    
    success = retrain_model(args.data_path)
    if success:
        logger.info("✅ Model retraining completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Model retraining failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
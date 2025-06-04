"""
Model trainer module for stock prediction models.
"""
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    """Stock price prediction model trainer."""
    
    def __init__(self, experiment_name: str = "asx-stock-prediction"):
        self.experiment_name = experiment_name
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_20', 'SMA_50', 'RSI'
        ]
        
    def prepare_data(
        self,
        df: pd.DataFrame,
        prediction_days: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with stock data
            prediction_days: Number of days to predict ahead
            
        Returns:
            Tuple of features and target arrays
        """
        # Create target variable (1 if price goes up, 0 if down)
        df['Target'] = (df['Close'].shift(-prediction_days) > df['Close']).astype(int)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Prepare features
        X = df[self.feature_columns].values
        y = df['Target'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
        
    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Train the prediction model.
        
        Args:
            df: DataFrame with stock data
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with model performance metrics
        """
        # Start MLflow run
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Prepare data
            X, y = self.prepare_data(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Log metrics
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score
            }
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                self.model,
                "model",
                registered_model_name=f"stock_predictor_{datetime.now().strftime('%Y%m%d')}"
            )
            
            return metrics
            
    def predict(
        self,
        df: pd.DataFrame,
        days_ahead: int = 5
    ) -> Dict[str, float]:
        """
        Make predictions for future price movements.
        
        Args:
            df: DataFrame with recent stock data
            days_ahead: Number of days to predict ahead
            
        Returns:
            Dictionary with prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # Prepare features
        X = df[self.feature_columns].values
        X = self.scaler.transform(X)
        
        # Make predictions
        probabilities = self.model.predict_proba(X)
        
        return {
            'up_probability': float(probabilities[-1, 1]),
            'down_probability': float(probabilities[-1, 0])
        }
        
    def save_model(self, path: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        mlflow.sklearn.save_model(self.model, path)
        
    def load_model(self, path: str):
        """Load a trained model from disk."""
        self.model = mlflow.sklearn.load_model(path) 
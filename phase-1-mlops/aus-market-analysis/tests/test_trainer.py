"""
Tests for the stock prediction model trainer.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.models.trainer import StockPredictor

def test_init():
    """Test StockPredictor initialization."""
    predictor = StockPredictor()
    assert predictor.experiment_name == "asx-stock-prediction"
    assert predictor.model is None
    assert len(predictor.feature_columns) > 0

def test_prepare_data(sample_stock_data):
    """Test data preparation for training."""
    predictor = StockPredictor()
    X, y = predictor.prepare_data(sample_stock_data)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    assert X.shape[1] == len(predictor.feature_columns)
    assert set(np.unique(y)) <= {0, 1}

@patch('mlflow.start_run')
@patch('mlflow.log_metrics')
@patch('mlflow.sklearn.log_model')
def test_train(mock_log_model, mock_log_metrics, mock_start_run, sample_stock_data):
    """Test model training."""
    predictor = StockPredictor()
    metrics = predictor.train(sample_stock_data)
    
    assert isinstance(metrics, dict)
    assert 'train_accuracy' in metrics
    assert 'test_accuracy' in metrics
    assert 0 <= metrics['train_accuracy'] <= 1
    assert 0 <= metrics['test_accuracy'] <= 1
    assert predictor.model is not None

def test_predict(sample_stock_data):
    """Test model prediction."""
    predictor = StockPredictor()
    predictor.train(sample_stock_data)
    
    predictions = predictor.predict(sample_stock_data)
    
    assert isinstance(predictions, dict)
    assert 'up_probability' in predictions
    assert 'down_probability' in predictions
    assert 0 <= predictions['up_probability'] <= 1
    assert 0 <= predictions['down_probability'] <= 1
    assert abs(predictions['up_probability'] + predictions['down_probability'] - 1.0) < 1e-10

def test_predict_without_training():
    """Test prediction without training."""
    predictor = StockPredictor()
    with pytest.raises(ValueError):
        predictor.predict(pd.DataFrame())

@patch('mlflow.sklearn.save_model')
def test_save_model(mock_save_model, sample_stock_data):
    """Test model saving."""
    predictor = StockPredictor()
    predictor.train(sample_stock_data)
    predictor.save_model('test_model')
    mock_save_model.assert_called_once()

def test_save_model_without_training():
    """Test saving without training."""
    predictor = StockPredictor()
    with pytest.raises(ValueError):
        predictor.save_model('test_model')

@patch('mlflow.sklearn.load_model')
def test_load_model(mock_load_model):
    """Test model loading."""
    predictor = StockPredictor()
    predictor.load_model('test_model')
    mock_load_model.assert_called_once_with('test_model')
    assert predictor.model is not None

def test_prepare_data_with_invalid_data():
    """Test data preparation with invalid data."""
    predictor = StockPredictor()
    invalid_df = pd.DataFrame({'Invalid': [1, 2, 3]})
    with pytest.raises(KeyError):
        predictor.prepare_data(invalid_df) 
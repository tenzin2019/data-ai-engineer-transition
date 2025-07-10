"""
Tests for serving components including scoring script and deployment utilities.
"""

import pytest
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add serving directory to path
sys.path.append(str(Path(__file__).parent.parent / "src" / "serving"))

from score import validate_input, health_check, run, init

class TestScoringScript:
    """Test the scoring script functionality."""
    
    def test_validate_input_valid_data(self):
        """Test input validation with valid data."""
        valid_data = {
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
        }
        
        result = validate_input(valid_data)
        assert result.shape == (1, 12)
        assert result.dtype == np.float64
    
    def test_validate_input_multiple_samples(self):
        """Test input validation with multiple samples."""
        valid_data = {
            "data": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 0.1]
            ]
        }
        
        result = validate_input(valid_data)
        assert result.shape == (2, 12)
        assert result.dtype == np.float64
    
    def test_validate_input_json_string(self):
        """Test input validation with JSON string."""
        valid_data = {
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
        }
        json_string = json.dumps(valid_data)
        
        result = validate_input(json_string)
        assert result.shape == (1, 12)
        assert result.dtype == np.float64
    
    def test_validate_input_single_sample(self):
        """Test input validation with single sample (1D array)."""
        valid_data = {
            "data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
        }
        
        result = validate_input(valid_data)
        assert result.shape == (1, 12)
        assert result.dtype == np.float64
    
    def test_validate_input_missing_data_field(self):
        """Test input validation with missing data field."""
        invalid_data = {
            "features": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
        }
        
        with pytest.raises(ValueError, match="Input must contain 'data' field"):
            validate_input(invalid_data)
    
    def test_validate_input_wrong_feature_count(self):
        """Test input validation with wrong number of features."""
        invalid_data = {
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]  # 11 features
        }
        
        with pytest.raises(ValueError, match="Expected 12 features, got 11"):
            validate_input(invalid_data)
    
    def test_validate_input_nan_values(self):
        """Test input validation with NaN values."""
        invalid_data = {
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, np.nan]]
        }
        
        with pytest.raises(ValueError, match="Input contains NaN values"):
            validate_input(invalid_data)
    
    def test_validate_input_custom_feature_count(self):
        """Test input validation with custom feature count."""
        valid_data = {
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5]]
        }
        
        result = validate_input(valid_data, expected_features=5)
        assert result.shape == (1, 5)
        assert result.dtype == np.float64

class TestHealthCheck:
    """Test the health check functionality."""
    
    @patch('score.model')
    def test_health_check_model_loaded(self, mock_model):
        """Test health check when model is loaded."""
        # Mock model with predict method
        mock_model.predict.return_value = np.array([0])
        
        result = health_check()
        
        assert result["status"] == "healthy"
        assert result["model_loaded"] is True
        assert "model_type" in result
        assert result["features"] == 12
    
    @patch('score.model')
    def test_health_check_model_not_loaded(self, mock_model):
        """Test health check when model is not loaded."""
        mock_model = None
        
        with patch('score.model', None):
            result = health_check()
        
        assert result["status"] == "error"
        assert result["message"] == "Model not loaded"
        # The function doesn't return model_loaded when model is None
        assert "model_loaded" not in result
    
    @patch('score.model')
    def test_health_check_model_error(self, mock_model):
        """Test health check when model prediction fails."""
        # Mock model that raises an exception
        mock_model.predict.side_effect = Exception("Model error")
        
        result = health_check()
        
        assert result["status"] == "error"
        assert "Model error" in result["message"]
        assert result["model_loaded"] is True

class TestRunFunction:
    """Test the run function for predictions."""
    
    @patch('score.model')
    @patch('score.validate_input')
    def test_run_success(self, mock_validate_input, mock_model):
        """Test successful prediction run."""
        # Mock input validation
        mock_validate_input.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        
        # Mock model predictions
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
        
        test_data = {"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]}
        
        result = run(test_data)
        
        assert result["status"] == "success"
        assert result["predictions"] == [1]
        assert result["probabilities"] == [0.8]
        assert result["input_shape"] == (1, 12)
        assert "model_type" in result
    
    @patch('score.model')
    @patch('score.validate_input')
    def test_run_no_probabilities(self, mock_validate_input, mock_model):
        """Test prediction run without probabilities."""
        # Mock input validation
        mock_validate_input.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])
        
        # Mock model without predict_proba
        mock_model.predict.return_value = np.array([1])
        del mock_model.predict_proba
        
        test_data = {"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]}
        
        result = run(test_data)
        
        assert result["status"] == "success"
        assert result["predictions"] == [1]
        assert "probabilities" not in result
    
    @patch('score.model')
    def test_run_model_not_loaded(self, mock_model):
        """Test run when model is not loaded."""
        mock_model = None
        
        with patch('score.model', None):
            test_data = {"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]}
            result = run(test_data)
        
        assert result["status"] == "error"
        assert "Model not loaded" in result["error"]
        assert result["predictions"] is None
        assert result["probabilities"] is None
    
    @patch('score.model')
    @patch('score.validate_input')
    def test_run_validation_error(self, mock_validate_input, mock_model):
        """Test run when input validation fails."""
        # Mock validation to raise an exception
        mock_validate_input.side_effect = ValueError("Invalid input")
        
        test_data = {"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]}
        
        result = run(test_data)
        
        assert result["status"] == "error"
        assert "Invalid input" in result["error"]
        assert result["predictions"] is None
        assert result["probabilities"] is None

class TestInitFunction:
    """Test the init function for model loading."""
    
    @patch('joblib.load')
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_init_success(self, mock_getenv, mock_exists, mock_load):
        """Test successful model initialization."""
        # Mock environment and file existence
        mock_getenv.return_value = "/path/to/model"
        mock_exists.return_value = True
        
        # Mock model with required attributes
        mock_model = MagicMock()
        mock_model.predict = MagicMock()
        mock_model.feature_names_in_ = np.array(['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12'])
        mock_load.return_value = mock_model
        
        # Mock os.path.join
        with patch('os.path.join', return_value="/path/to/model/model.joblib"):
            init()
        
        # Verify model was loaded
        mock_load.assert_called_once()
    
    @patch('os.getenv')
    @patch('os.path.exists')
    def test_init_model_not_found(self, mock_exists, mock_getenv):
        """Test init when model file is not found."""
        mock_getenv.return_value = "/path/to/model"
        mock_exists.return_value = False
        
        with patch('os.path.join', return_value="/path/to/model/model.joblib"):
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                init()
    
    @patch('joblib.load')
    @patch('os.path.exists')
    @patch('os.getenv')
    def test_init_invalid_model(self, mock_getenv, mock_exists, mock_load):
        """Test init with invalid model (no predict method)."""
        mock_getenv.return_value = "/path/to/model"
        mock_exists.return_value = True
        
        # Mock invalid model without predict method
        mock_model = MagicMock()
        del mock_model.predict
        mock_load.return_value = mock_model
        
        with patch('os.path.join', return_value="/path/to/model/model.joblib"):
            with pytest.raises(ValueError, match="Loaded model does not have 'predict' method"):
                init()

class TestIntegration:
    """Integration tests for the serving components."""
    
    def test_end_to_end_workflow(self):
        """Test the complete workflow from input to prediction."""
        # This would require a real model file
        # For now, we'll test the structure
        
        # Test input validation
        test_data = {
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
        }
        
        # Validate input
        X = validate_input(test_data)
        assert X.shape == (1, 12)
        assert X.dtype == np.float64
        
        # Test JSON serialization
        json_string = json.dumps(test_data)
        X_from_json = validate_input(json_string)
        np.testing.assert_array_equal(X, X_from_json)
    
    def test_error_handling_consistency(self):
        """Test that error handling is consistent across functions."""
        # Test with invalid input
        invalid_data = {"wrong_field": [[1, 2, 3]]}
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            validate_input(invalid_data)
        
        # Test with None model
        with patch('score.model', None):
            result = run(invalid_data)
            assert result["status"] == "error"
            assert "Model not loaded" in result["error"]

if __name__ == "__main__":
    pytest.main([__file__]) 
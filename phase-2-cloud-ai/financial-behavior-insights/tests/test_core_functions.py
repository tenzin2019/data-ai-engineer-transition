import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import only core functions that don't require Azure
from training.train_model import (
    validate_dataframe,
    load_data,
    train_and_eval,
    validate_model,
    validate_output_directory,
    setup_mlflow
)

class TestDataValidation:
    """Test data validation functions."""
    
    def test_validate_dataframe_success(self):
        """Test successful dataframe validation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'HighAmount': [0, 1, 0]
        })
        validate_dataframe(df)
    
    def test_validate_dataframe_empty(self):
        """Test validation fails for empty dataframe."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="Dataframe is empty"):
            validate_dataframe(df)
    
    def test_validate_dataframe_missing_target(self):
        """Test validation fails for missing target column."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        with pytest.raises(ValueError, match="Target column 'HighAmount' not found"):
            validate_dataframe(df)
    
    def test_validate_dataframe_missing_values(self):
        """Test validation fails for missing values in target."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'HighAmount': [0, np.nan, 1]
        })
        with pytest.raises(ValueError, match="Missing values found in target column"):
            validate_dataframe(df)
    
    def test_validate_dataframe_non_numeric_target(self):
        """Test validation fails for non-numeric target."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'HighAmount': ['low', 'high', 'low']
        })
        with pytest.raises(ValueError, match="Target column 'HighAmount' must be numeric"):
            validate_dataframe(df)
    
    def test_validate_dataframe_duplicate_columns(self, tmp_path):
        import pandas as pd
        from packaging import version
        if version.parse(pd.__version__) < version.parse("1.5.0"):
            import pytest
            pytest.skip("pandas < 1.5.0 does not support mangle_dupe_cols; cannot test duplicate columns from CSV")
        # Create a CSV file with duplicate headers
        csv_path = tmp_path / "dup_cols.csv"
        with open(csv_path, 'w') as f:
            f.write("feature1,feature1,HighAmount\n1,2,0\n3,4,1\n5,6,0\n")
        df = pd.read_csv(csv_path, mangle_dupe_cols=False)
        with pytest.raises(ValueError, match="Duplicate columns found"):
            validate_dataframe(df)

class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_data_success(self, tmp_path):
        """Test successful data loading."""
        # Create test CSV file
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10],
            'HighAmount': [0, 1, 0, 1, 0]
        })
        csv_path = tmp_path / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        X, y = load_data(str(csv_path))
        
        assert X.shape == (5, 2)
        assert y.shape == (5,)
        assert list(X.columns) == ['feature1', 'feature2']
    
    def test_load_data_file_not_found(self):
        """Test data loading fails for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_data("non_existent_file.csv")
    
    def test_load_data_invalid_format(self, tmp_path):
        """Test data loading fails for invalid CSV."""
        # Create invalid CSV file
        csv_path = tmp_path / "invalid.csv"
        with open(csv_path, 'w') as f:
            f.write("invalid,csv,format\n")
        
        with pytest.raises(Exception):
            load_data(str(csv_path))

class TestModelTraining:
    """Test model training and evaluation."""
    
    def test_train_and_eval_success(self):
        """Test successful model training and evaluation."""
        X = np.random.randn(100, 3)
        y = np.random.randint(0, 2, 100)
        
        model = MagicMock()
        model.fit.return_value = None
        model.predict.return_value = np.random.randint(0, 2, 20)
        model.predict_proba.return_value = np.random.rand(20, 2)
        
        with patch('sklearn.model_selection.train_test_split') as mock_split:
            mock_split.return_value = (X[:80], X[80:], y[:80], y[80:])
            
            result_model, metrics, eval_data = train_and_eval(model, X, y)
            
            assert isinstance(metrics, dict)
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1_score' in metrics
            assert 'roc_auc' in metrics
    
    def test_train_and_eval_single_class(self):
        """Test training fails for single class target."""
        X = np.random.randn(100, 3)
        y = np.zeros(100)  # All zeros - single class
        
        model = MagicMock()
        
        with pytest.raises(ValueError, match="Target variable has only one class"):
            train_and_eval(model, X, y)

class TestModelValidation:
    """Test model validation functionality."""
    
    def test_validate_model_success(self):
        """Test successful model validation."""
        X_test = np.random.randn(10, 3)
        y_test = np.random.randint(0, 2, 10)
        
        model = MagicMock()
        model.predict.return_value = np.random.randint(0, 2, 5)
        model.predict_proba.return_value = np.random.rand(5, 2)
        
        with patch('joblib.dump'), patch('joblib.load', return_value=model):
            result = validate_model(model, X_test, y_test)
            assert result is True
    
    def test_validate_model_failure(self):
        """Test model validation fails."""
        X_test = np.random.randn(10, 3)
        y_test = np.random.randint(0, 2, 10)
        
        model = MagicMock()
        model.predict.side_effect = Exception("Prediction failed")
        
        result = validate_model(model, X_test, y_test)
        assert result is False

class TestOutputDirectoryValidation:
    """Test output directory validation."""
    
    def test_validate_output_directory_success(self, tmp_path):
        """Test successful output directory validation."""
        output_dir = tmp_path / "test_output"
        result = validate_output_directory(str(output_dir))
        assert result is True
        assert output_dir.exists()
    
    def test_validate_output_directory_permission_error(self):
        """Test output directory validation fails for permission issues."""
        # Try to write to a non-existent path that would require root permissions
        with patch('pathlib.Path.mkdir', side_effect=PermissionError):
            result = validate_output_directory("/root/test_output")
            assert result is False

class TestMLflowSetup:
    """Test MLflow setup functionality."""
    
    @patch('mlflow.get_tracking_uri')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_setup_mlflow_success(self, mock_set_experiment, mock_set_uri, mock_get_uri):
        """Test successful MLflow setup."""
        mock_get_uri.return_value = None
        
        result = setup_mlflow()
        
        assert result is True
        mock_set_uri.assert_called_once_with("sqlite:///mlflow.db")
        mock_set_experiment.assert_called_once_with("financial-behavior-insights")
    
    @patch('mlflow.get_tracking_uri')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_setup_mlflow_failure(self, mock_set_experiment, mock_set_uri, mock_get_uri):
        """Test MLflow setup fails gracefully."""
        mock_get_uri.return_value = None
        mock_set_experiment.side_effect = Exception("MLflow error")
        
        result = setup_mlflow()
        
        assert result is False

class TestIntegration:
    """Integration tests for core functionality."""
    
    def test_data_validation_pipeline(self, tmp_path):
        """Test complete data validation pipeline."""
        # Create valid test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'HighAmount': np.random.randint(0, 2, 100)
        })
        
        # Test dataframe validation
        validate_dataframe(test_data)
        
        # Test data loading
        csv_path = tmp_path / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        X, y = load_data(str(csv_path))
        
        # Verify data structure
        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert len(np.unique(y)) == 2  # Binary classification
    
    def test_output_directory_creation(self, tmp_path):
        """Test output directory creation and validation."""
        output_dir = tmp_path / "outputs" / "models"
        
        # Test directory validation
        result = validate_output_directory(str(output_dir))
        assert result is True
        assert output_dir.exists()
        
        # Test file writing
        test_file = output_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
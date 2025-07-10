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

from training.train_model import (
    validate_environment_variables,
    validate_dataframe,
    load_data,
    tune_hyperparameters,
    train_and_eval,
    validate_model,
    register_model_azureml,
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
    
    def test_validate_dataframe_duplicate_columns(self):
        """Test validation fails for duplicate columns."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature1': [4, 5, 6],  # Duplicate column name
            'HighAmount': [0, 1, 0]
        })
        with pytest.raises(ValueError, match="Duplicate columns found"):
            validate_dataframe(df)

class TestEnvironmentValidation:
    """Test environment variable validation."""
    
    @patch.dict(os.environ, {
        'AZURE_SUBSCRIPTION_ID': 'test-sub',
        'AZURE_RESOURCE_GROUP': 'test-rg',
        'AZURE_WORKSPACE_NAME': 'test-ws'
    })
    def test_validate_environment_variables_success(self):
        """Test successful environment validation."""
        validate_environment_variables()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_validate_environment_variables_missing(self):
        """Test environment validation fails for missing variables."""
        with pytest.raises(ValueError, match="Missing required environment variables"):
            validate_environment_variables()
    
    @patch.dict(os.environ, {
        'AZURE_SUBSCRIPTION_ID': 'test-sub',
        'AZURE_RESOURCE_GROUP': 'test-rg'
        # Missing AZURE_WORKSPACE_NAME
    })
    def test_validate_environment_variables_partial(self):
        """Test environment validation fails for partial variables."""
        with pytest.raises(ValueError, match="Missing required environment variables"):
            validate_environment_variables()

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

class TestHyperparameterTuning:
    """Test hyperparameter tuning functionality."""
    
    def test_tune_hyperparameters_success(self):
        """Test successful hyperparameter tuning."""
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        
        with patch('sklearn.model_selection.RandomizedSearchCV') as mock_search:
            mock_search_instance = MagicMock()
            mock_search_instance.best_estimator_ = MagicMock()
            mock_search_instance.best_params_ = {'n_estimators': 100}
            mock_search_instance.best_score_ = 0.85
            mock_search.return_value = mock_search_instance
            
            best_model, best_params, best_score = tune_hyperparameters(X, y, n_iter=5)
            
            assert best_params == {'n_estimators': 100}
            assert best_score == 0.85
    
    def test_tune_hyperparameters_small_dataset(self):
        """Test hyperparameter tuning with small dataset."""
        X = np.random.randn(10, 3)  # Very small dataset
        y = np.random.randint(0, 2, 10)
        
        with patch('sklearn.model_selection.RandomizedSearchCV') as mock_search:
            mock_search_instance = MagicMock()
            mock_search_instance.best_estimator_ = MagicMock()
            mock_search_instance.best_params_ = {'n_estimators': 100}
            mock_search_instance.best_score_ = 0.85
            mock_search.return_value = mock_search_instance
            
            # Should not raise an error, but should log a warning
            best_model, best_params, best_score = tune_hyperparameters(X, y, n_iter=5, cv=3)

class TestAzureMLRegistration:
    """Test Azure ML model registration."""
    
    @patch.dict(os.environ, {
        'AZURE_SUBSCRIPTION_ID': 'test-sub',
        'AZURE_RESOURCE_GROUP': 'test-rg',
        'AZURE_WORKSPACE_NAME': 'test-ws'
    })
    @patch('azure.ai.ml.MLClient')
    def test_register_model_azureml_success(self, mock_ml_client, tmp_path):
        """Test successful Azure ML model registration."""
        # Create a dummy model file
        model_path = tmp_path / "model.joblib"
        model_path.write_text("dummy model content")
        
        mock_client_instance = MagicMock()
        mock_registered_model = MagicMock()
        mock_registered_model.name = "test-model"
        mock_registered_model.version = "1"
        mock_client_instance.models.create_or_update.return_value = mock_registered_model
        mock_ml_client.return_value = mock_client_instance
        
        result = register_model_azureml(str(model_path))
        
        assert result.name == "test-model"
        assert result.version == "1"
    
    def test_register_model_azureml_file_not_found(self):
        """Test Azure ML registration fails for non-existent model file."""
        with pytest.raises(FileNotFoundError):
            register_model_azureml("non_existent_model.joblib")

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
    """Integration tests for the training pipeline."""
    
    def test_end_to_end_pipeline_mock(self, tmp_path):
        """Test end-to-end pipeline with mocked components."""
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'HighAmount': np.random.randint(0, 2, 100)
        })
        
        csv_path = tmp_path / "test_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        output_dir = tmp_path / "outputs"
        
        # Mock all external dependencies
        with patch('training.train_model.setup_mlflow', return_value=False), \
             patch('training.train_model.validate_output_directory', return_value=True), \
             patch('training.train_model.tune_hyperparameters') as mock_tune, \
             patch('training.train_model.train_and_eval') as mock_train, \
             patch('training.train_model.validate_model', return_value=True), \
             patch('joblib.dump'), \
             patch('json.dump'):
            
            # Setup mocks
            mock_model = MagicMock()
            mock_tune.return_value = (mock_model, {'n_estimators': 100}, 0.85)
            mock_train.return_value = (mock_model, {'accuracy': 0.8}, (None, None, None, None))
            
            # Import and run main function
            from training.train_model import main
            import argparse
            
            # Create mock args
            args = argparse.Namespace(
                input_data=str(csv_path),
                output_dir=str(output_dir),
                random_state=42,
                n_iter=5,
                cv=3,
                test_size=0.2,
                chunk_size=None,
                register_model=False,
                model_name="test-model",
                model_description="Test model"
            )
            
            # Run pipeline
            main(args)
            
            # Verify outputs were created
            assert output_dir.exists()
            assert (output_dir / "model.joblib").exists()
            assert (output_dir / "metrics.json").exists()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
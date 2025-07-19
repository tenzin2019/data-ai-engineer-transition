#!/usr/bin/env python3
"""
test_local.py

Local testing script for the financial behavior prediction model.
Tests the model locally before deployment to catch issues early.

Usage:
    python test_local.py [--model-uri <uri>] [--test-data <path>]

MLOps Best Practices:
    - Comprehensive model validation
    - Performance testing
    - Edge case handling
    - Clear error reporting
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from typing import Dict, Any, Optional

# Suppress warnings for cleaner output in production
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*mlflow.*")
warnings.filterwarnings("ignore", message=".*sklearn.*")

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def load_test_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load test data for model validation.
    
    Args:
        data_path: Optional path to test data CSV
        
    Returns:
        pd.DataFrame: Test data
    """
    if data_path and os.path.exists(data_path):
        print(f"Loading test data from {data_path}")
        df = pd.read_csv(data_path)
        # Remove target column if present
        if 'HighAmount' in df.columns:
            df = df.drop(columns=['HighAmount'])
        return df
    else:
        print("Generating synthetic test data")
        # Generate synthetic test data with 12 features
        np.random.seed(42)
        n_samples = 10
        data = {
            f'feature_{i}': np.random.randn(n_samples) for i in range(12)
        }
        return pd.DataFrame(data)


def test_model_loading(model_uri: str) -> bool:
    """
    Test that the model loads correctly.
    
    Args:
        model_uri: URI of the model to test
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*50)
    print("Testing model loading...")
    print("="*50)
    
    try:
        start_time = time.time()
        model = mlflow.pyfunc.load_model(model_uri)
        load_time = time.time() - start_time
        
        print(f"✓ Model loaded successfully in {load_time:.2f} seconds")
        print(f"  Model type: {type(model)}")
        
        # Check model attributes
        if hasattr(model, '_model_impl'):
            print(f"  Implementation type: {type(model._model_impl)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction(model_uri: str, test_data: pd.DataFrame) -> bool:
    """
    Test model predictions on sample data.
    
    Args:
        model_uri: URI of the model to test
        test_data: Test data DataFrame
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*50)
    print("Testing model predictions...")
    print("="*50)
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Test single prediction
        print("\n1. Testing single sample prediction:")
        single_sample = test_data.iloc[[0]]
        print(f"   Input shape: {single_sample.shape}")
        
        start_time = time.time()
        single_pred = model.predict(single_sample)
        pred_time = time.time() - start_time
        
        print(f"✓ Single prediction successful in {pred_time*1000:.2f} ms")
        print(f"  Output type: {type(single_pred)}")
        print(f"  Output shape: {getattr(single_pred, 'shape', len(single_pred))}")
        print(f"  Prediction value: {single_pred}")
        
        # Test batch prediction
        print("\n2. Testing batch prediction:")
        batch_size = min(5, len(test_data))
        batch_sample = test_data.iloc[:batch_size]
        print(f"   Batch size: {batch_size}")
        
        start_time = time.time()
        batch_pred = model.predict(batch_sample)
        batch_time = time.time() - start_time
        
        print(f"✓ Batch prediction successful in {batch_time*1000:.2f} ms")
        print(f"  Average time per sample: {(batch_time/batch_size)*1000:.2f} ms")
        print(f"  Output shape: {getattr(batch_pred, 'shape', len(batch_pred))}")
        
        # Validate output format
        if isinstance(batch_pred, np.ndarray):
            print(f"  Output dtype: {batch_pred.dtype}")
            if len(batch_pred) != batch_size:
                print(f"✗ Warning: Expected {batch_size} predictions, got {len(batch_pred)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases(model_uri: str) -> bool:
    """
    Test model behavior with edge cases.
    
    Args:
        model_uri: URI of the model to test
        
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\n" + "="*50)
    print("Testing edge cases...")
    print("="*50)
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        all_passed = True
        
        # Define the correct feature names expected by the model
        feature_names = [
            'Age', 'Transaction Amount', 'Account Balance', 'AccountAgeDays',
            'TransactionHour', 'TransactionDayOfWeek', 'Transaction Type_Deposit',
            'Transaction Type_Transfer', 'Transaction Type_Withdrawal',
            'Gender_Female', 'Gender_Male', 'Gender_Other'
        ]
        
        # Test 1: Missing features (wrong number of features)
        print("\n1. Testing with wrong number of features:")
        try:
            # Create data with only 10 features instead of 12
            wrong_features = pd.DataFrame({feature_names[i]: [0.0] for i in range(10)})
            model.predict(wrong_features)
            print("✗ Model should have rejected input with wrong number of features")
            all_passed = False
        except Exception as e:
            print(f"✓ Model correctly rejected invalid input")
            print(f"  Error: {str(e)[:100]}...")
        
        # Test 2: Empty DataFrame
        print("\n2. Testing with empty DataFrame:")
        try:
            empty_df = pd.DataFrame()
            model.predict(empty_df)
            print("✗ Model should have rejected empty input")
            all_passed = False
        except Exception as e:
            print(f"✓ Model correctly rejected empty input")
            print(f"  Error: {str(e)[:100]}...")
        
        # Test 3: NaN values (but with correct feature names)
        print("\n3. Testing with NaN values:")
        nan_data = pd.DataFrame({name: [np.nan] for name in feature_names})
        try:
            nan_pred = model.predict(nan_data)
            print("✓ Model handled NaN values")
            print(f"  Prediction with NaN: {nan_pred}")
        except Exception as e:
            print(f"✗ Model failed with NaN values")
            print(f"  Error: {str(e)[:100]}...")
            all_passed = False
        
        # Test 4: Extreme values (but with correct feature names)
        print("\n4. Testing with extreme values:")
        extreme_data = pd.DataFrame({
            name: [1e10 if i % 2 == 0 else -1e10] 
            for i, name in enumerate(feature_names)
        })
        try:
            extreme_pred = model.predict(extreme_data)
            print("✓ Model handled extreme values")
            print(f"  Prediction with extremes: {extreme_pred}")
        except Exception as e:
            print(f"✗ Model failed with extreme values")
            print(f"  Error: {str(e)[:100]}...")
            all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"✗ Edge case testing failed: {e}")
        return False


def test_performance(model_uri: str, test_data: pd.DataFrame, n_iterations: int = 100) -> bool:
    """
    Test model performance and latency.
    
    Args:
        model_uri: URI of the model to test
        test_data: Test data DataFrame
        n_iterations: Number of iterations for performance testing
        
    Returns:
        bool: True if performance is acceptable, False otherwise
    """
    print("\n" + "="*50)
    print("Testing model performance...")
    print("="*50)
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Warm up
        for _ in range(5):
            model.predict(test_data.iloc[[0]])
        
        # Performance test
        single_sample = test_data.iloc[[0]]
        latencies = []
        
        print(f"\nRunning {n_iterations} predictions...")
        for i in range(n_iterations):
            start_time = time.time()
            _ = model.predict(single_sample)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{n_iterations}")
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        print("\nPerformance Statistics:")
        print(f"  Mean latency: {np.mean(latencies):.2f} ms")
        print(f"  Median latency: {np.median(latencies):.2f} ms")
        print(f"  Min latency: {np.min(latencies):.2f} ms")
        print(f"  Max latency: {np.max(latencies):.2f} ms")
        print(f"  95th percentile: {np.percentile(latencies, 95):.2f} ms")
        print(f"  99th percentile: {np.percentile(latencies, 99):.2f} ms")
        
        # Check if performance meets requirements
        mean_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        if mean_latency < 100 and p95_latency < 200:
            print("\n✓ Performance meets requirements")
            return True
        else:
            print("\n✗ Performance below requirements")
            print("  Expected: mean < 100ms, p95 < 200ms")
            return False
        
    except Exception as e:
        print(f"✗ Performance testing failed: {e}")
        return False


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test MLflow model locally")
    parser.add_argument(
        "--model-uri",
        type=str,
        default="models:/financial-behavior-model@production",
        help="URI of the model to test"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        help="Path to test data CSV file"
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Skip performance testing"
    )
    
    args = parser.parse_args()
    
    print(f"Testing model: {args.model_uri}")
    
    # Load test data
    test_data = load_test_data(args.test_data)
    print(f"\nTest data shape: {test_data.shape}")
    print(f"Features: {list(test_data.columns)}")
    
    # Run tests
    all_passed = True
    
    # Test 1: Model loading
    if not test_model_loading(args.model_uri):
        all_passed = False
        print("\n✗ Model loading test failed. Aborting further tests.")
        sys.exit(1)
    
    # Test 2: Basic predictions
    if not test_prediction(args.model_uri, test_data):
        all_passed = False
    
    # Test 3: Edge cases
    if not test_edge_cases(args.model_uri):
        all_passed = False
    
    # Test 4: Performance (optional)
    if not args.skip_performance:
        if not test_performance(args.model_uri, test_data):
            all_passed = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    if all_passed:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed. Please review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main() 

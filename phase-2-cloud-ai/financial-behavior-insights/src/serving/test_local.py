#!/usr/bin/env python3
"""
Local testing script for the scoring endpoint.
Tests the model locally before deployment to catch issues early.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add the serving directory to path
sys.path.append(os.path.dirname(__file__))

from score import init, run, health_check, validate_input

def test_model_loading():
    """Test that the model loads correctly."""
    print("Testing model loading...")
    
    try:
        # Set environment variable to point to our model
        model_path = Path(__file__).parent.parent.parent / "outputs" / "model.joblib"
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("Please run training first: python src/training/train_model.py --input-data data/processed/Comprehensive_Banking_Database_processed.csv")
            return False
        
        # Set environment variable for model loading
        os.environ["AZUREML_MODEL_DIR"] = str(model_path.parent)
        
        # Initialize the model
        init()
        print("✅ Model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_health_check():
    """Test the health check endpoint."""
    print("\nTesting health check...")
    
    try:
        result = health_check()
        print(f"Health check result: {result}")
        
        if result.get("status") == "healthy":
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_input_validation():
    """Test input validation with various scenarios."""
    print("\nTesting input validation...")
    
    test_cases = [
        {
            "name": "Valid single sample",
            "data": {"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]},
            "should_pass": True
        },
        {
            "name": "Valid multiple samples",
            "data": {"data": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 0.1]
            ]},
            "should_pass": True
        },
        {
            "name": "Missing data field",
            "data": {"features": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]},
            "should_pass": False
        },
        {
            "name": "Wrong number of features",
            "data": {"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]]},
            "should_pass": False
        },
        {
            "name": "NaN values",
            "data": {"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, np.nan]]},
            "should_pass": False
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for test_case in test_cases:
        try:
            validate_input(test_case["data"])
            if test_case["should_pass"]:
                print(f"✅ {test_case['name']}: Passed")
                passed += 1
            else:
                print(f"❌ {test_case['name']}: Should have failed but passed")
        except Exception as e:
            if test_case["should_pass"]:
                print(f"❌ {test_case['name']}: Should have passed but failed - {e}")
            else:
                print(f"✅ {test_case['name']}: Correctly failed - {e}")
                passed += 1
    
    print(f"Input validation: {passed}/{total} tests passed")
    return passed == total

def test_predictions():
    """Test prediction functionality."""
    print("\nTesting predictions...")
    
    # Load some real data for testing
    data_path = Path(__file__).parent.parent.parent / "data" / "processed" / "Comprehensive_Banking_Database_processed.csv"
    
    if not data_path.exists():
        print(f"❌ Test data not found at {data_path}")
        print("Using synthetic data instead...")
        
        # Use synthetic data
        test_data = {
            "data": [
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 0.1],
                [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 0.1, 0.2]
            ]
        }
    else:
        # Use real data
        print("Using real data for testing...")
        df = pd.read_csv(data_path)
        X = df.drop(columns=["HighAmount"]).head(3).values.tolist()
        test_data = {"data": X}
    
    try:
        result = run(test_data)
        
        print(f"Prediction result: {result}")
        
        if result.get("status") == "success":
            predictions = result.get("predictions", [])
            probabilities = result.get("probabilities", [])
            
            print(f"✅ Predictions: {predictions}")
            if probabilities:
                print(f"✅ Probabilities: {probabilities}")
            
            # Validate predictions
            if len(predictions) == len(test_data["data"]):
                print("✅ Prediction count matches input count")
                return True
            else:
                print(f"❌ Prediction count mismatch: expected {len(test_data['data'])}, got {len(predictions)}")
                return False
        else:
            print(f"❌ Prediction failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def test_json_serialization():
    """Test that the endpoint can handle JSON string input."""
    print("\nTesting JSON serialization...")
    
    test_data = {
        "data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]
    }
    
    try:
        # Test with JSON string
        json_string = json.dumps(test_data)
        result = run(json_string)
        
        if result.get("status") == "success":
            print("✅ JSON string input handled correctly")
            return True
        else:
            print(f"❌ JSON string input failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ JSON serialization test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Starting local endpoint tests...\n")
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Health Check", test_health_check),
        ("Input Validation", test_input_validation),
        ("Predictions", test_predictions),
        ("JSON Serialization", test_json_serialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed! The endpoint is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix issues before deployment.")
        return 1

if __name__ == "__main__":
    exit(main()) 

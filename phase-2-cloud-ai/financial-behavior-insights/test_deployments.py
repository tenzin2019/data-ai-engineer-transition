#!/usr/bin/env python3
"""
test_deployments.py

Comprehensive testing script for all deployments:
- Local model testing
- Azure ML deployment testing
- End-to-end validation

Usage:
    python test_deployments.py
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DeploymentTester:
    """Comprehensive deployment testing class."""
    
    def __init__(self):
        """Initialize the tester."""
        load_dotenv()
        
        self.subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = os.getenv("AZURE_RESOURCE_GROUP")
        self.workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
        
        if not all([self.subscription_id, self.resource_group, self.workspace_name]):
            raise ValueError("Missing required Azure ML environment variables")
        
        # Initialize Azure ML client
        try:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            
            self.ml_client = MLClient(
                DefaultAzureCredential(),
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
            logger.info(f"Connected to Azure ML workspace: {self.workspace_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure ML client: {e}")
            raise
    
    def test_local_model(self):
        """Test local model functionality."""
        logger.info("=" * 60)
        logger.info("TESTING LOCAL MODEL")
        logger.info("=" * 60)
        
        try:
            import mlflow
            
            # Test model loading
            model_uri = "models:/financial-behavior-model-fixed@production"
            logger.info(f"Loading model: {model_uri}")
            
            start_time = time.time()
            model = mlflow.pyfunc.load_model(model_uri)
            load_time = time.time() - start_time
            
            logger.info(f"✅ Model loaded successfully in {load_time:.2f} seconds")
            
            # Create test data
            test_data = self._create_test_data()
            logger.info(f"Created test data with shape: {test_data.shape}")
            
            # Test predictions
            start_time = time.time()
            predictions = model.predict(test_data)
            predict_time = time.time() - start_time
            
            logger.info(f"✅ Predictions successful in {predict_time:.2f} seconds")
            logger.info(f"Prediction shape: {predictions.shape}")
            logger.info(f"Sample predictions: {predictions[:5]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Local model test failed: {e}")
            return False
    
    def test_azure_deployment(self):
        """Test Azure ML deployment."""
        logger.info("=" * 60)
        logger.info("TESTING AZURE ML DEPLOYMENT")
        logger.info("=" * 60)
        
        try:
            endpoint_name = "financial-behavior-endpoint"
            
            # Get endpoint details
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            logger.info(f"Endpoint URL: {endpoint.scoring_uri}")
            
            # Get endpoint key
            keys = self.ml_client.online_endpoints.get_keys(endpoint_name)
            endpoint_key = keys.primary_key if hasattr(keys, 'primary_key') else keys.key1
            
            # Create test data
            test_data = self._create_test_data()
            
            # Convert to JSON for REST API call
            import json
            import requests
            
            payload = {
                "data": test_data.to_dict('records')
            }
            
            # Make prediction request
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {endpoint_key}'
            }
            
            logger.info("Making prediction request to Azure ML endpoint...")
            response = requests.post(
                f"{endpoint.scoring_uri}",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Azure ML prediction successful!")
                logger.info(f"Response: {result}")
                return True
            else:
                logger.error(f"❌ Azure ML prediction failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Azure ML deployment test failed: {e}")
            return False
    
    def _create_test_data(self, n_samples=5):
        """Create test data with correct feature names and types."""
        # Use the actual feature names expected by the model
        feature_names = [
            'Age', 'Transaction Amount', 'Account Balance', 'AccountAgeDays',
            'TransactionHour', 'TransactionDayOfWeek', 'Transaction Type_Deposit',
            'Transaction Type_Transfer', 'Transaction Type_Withdrawal',
            'Gender_Female', 'Gender_Male', 'Gender_Other'
        ]
        
        np.random.seed(42)
        data = {}
        
        for feature in feature_names:
            if feature in ['Transaction Type_Deposit', 'Transaction Type_Transfer', 'Transaction Type_Withdrawal', 
                          'Gender_Female', 'Gender_Male', 'Gender_Other']:
                # Binary features (0 or 1) as float
                data[feature] = np.random.choice([0.0, 1.0], n_samples)
            elif feature in ['TransactionHour']:
                # Hour of day (0-23) as float
                data[feature] = np.random.randint(0, 24, n_samples).astype(float)
            elif feature in ['TransactionDayOfWeek']:
                # Day of week (0-6) as float
                data[feature] = np.random.randint(0, 7, n_samples).astype(float)
            elif feature in ['AccountAgeDays']:
                # Account age in days (0-3650, about 10 years) as float
                data[feature] = np.random.randint(0, 3650, n_samples).astype(float)
            elif feature in ['Age']:
                # Age (18-80) as float
                data[feature] = np.random.randint(18, 81, n_samples).astype(float)
            else:
                # Continuous features (Transaction Amount, Account Balance)
                data[feature] = np.random.exponential(1000, n_samples)  # Realistic financial amounts
        
        # Ensure all columns are float64
        df = pd.DataFrame(data)
        for col in df.columns:
            df[col] = df[col].astype('float64')
        
        return df
    
    def get_deployment_status(self):
        """Get current deployment status."""
        logger.info("=" * 60)
        logger.info("DEPLOYMENT STATUS REPORT")
        logger.info("=" * 60)
        
        try:
            # Check endpoints
            endpoints = list(self.ml_client.online_endpoints.list())
            logger.info(f"Found {len(endpoints)} endpoints:")
            
            for ep in endpoints:
                logger.info(f"  - {ep.name} (state: {ep.provisioning_state})")
                
                # Check deployments for each endpoint
                deployments = list(self.ml_client.online_deployments.list(ep.name))
                if deployments:
                    for dep in deployments:
                        logger.info(f"    └─ {dep.name} (state: {dep.provisioning_state})")
                else:
                    logger.info(f"    └─ No deployments")
            
            # Check models
            models = list(self.ml_client.models.list())
            logger.info(f"\nFound {len(models)} models in workspace:")
            for model in models:
                logger.info(f"  - {model.name} (version: {model.version})")
                
        except Exception as e:
            logger.error(f"❌ Failed to get deployment status: {e}")
    
    def run_all_tests(self):
        """Run all deployment tests."""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE DEPLOYMENT TESTING")
        logger.info("=" * 60)
        
        results = {}
        
        # Step 1: Get current status
        self.get_deployment_status()
        
        # Step 2: Test local model
        logger.info("\n" + "=" * 60)
        results['local_model'] = self.test_local_model()
        
        # Step 3: Test Azure deployment
        logger.info("\n" + "=" * 60)
        results['azure_test'] = self.test_azure_deployment()
        
        # Step 4: Final status report
        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 60)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        # Overall result
        all_passed = all(results.values())
        if all_passed:
            logger.info("\n🎉 ALL TESTS PASSED!")
        else:
            logger.info("\n⚠️  SOME TESTS FAILED")
        
        return all_passed

def main():
    """Main function."""
    try:
        tester = DeploymentTester()
        success = tester.run_all_tests()
        
        if success:
            logger.info("\n✅ Deployment testing completed successfully!")
            return True
        else:
            logger.error("\n❌ Deployment testing failed!")
            return False
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return False

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1) 
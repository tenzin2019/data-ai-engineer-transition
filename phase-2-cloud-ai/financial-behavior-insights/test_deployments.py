#!/usr/bin/env python3
"""
test_deployments.py

Comprehensive testing script for all deployments:
- Local model testing
- Azure ML deployment testing
- End-to-end validation
- Model compatibility testing

Usage:
    python test_deployments.py [--test-type all|local|azure|status]
"""

import os
import sys
import logging
import time
import pandas as pd
import numpy as np
import argparse
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
            logger.info(f"✅ Connected to Azure ML workspace: {self.workspace_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Azure ML client: {e}")
            raise
    
    def test_model_compatibility(self):
        """Test model compatibility with Azure ML environment."""
        logger.info("=" * 60)
        logger.info("TESTING MODEL COMPATIBILITY")
        logger.info("=" * 60)
        
        try:
            # Check if compatible model exists
            model_path = "outputs/model_compatible.joblib"
            if not os.path.exists(model_path):
                logger.error(f"❌ Compatible model not found: {model_path}")
                return False
            
            # Load and test the compatible model
            import joblib
            model_data = joblib.load(model_path)
            
            logger.info("✅ Compatible model loaded successfully")
            logger.info(f"Model type: {type(model_data.get('model'))}")
            logger.info(f"Accuracy: {model_data.get('accuracy', 'N/A')}")
            logger.info(f"Feature count: {len(model_data.get('feature_columns', []))}")
            
            # Test model prediction locally
            model = model_data['model']
            scaler = model_data['scaler']
            feature_columns = model_data['feature_columns']
            
            # Create test data with correct features
            test_data = self._create_test_data(5, feature_columns)
            
            # Scale the data
            test_data_scaled = scaler.transform(test_data)
            
            # Make prediction
            prediction = model.predict(test_data_scaled)
            prediction_proba = model.predict_proba(test_data_scaled)
            
            logger.info(f"✅ Local prediction successful")
            logger.info(f"Predictions: {prediction}")
            logger.info(f"Prediction probabilities shape: {prediction_proba.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Model compatibility test failed: {e}")
            return False
    
    def test_local_model(self):
        """Test local model functionality."""
        logger.info("=" * 60)
        logger.info("TESTING LOCAL MODEL")
        logger.info("=" * 60)
        
        try:
            # Test the compatible model directly
            return self.test_model_compatibility()
            
        except Exception as e:
            logger.error(f"❌ Local model test failed: {e}")
            return False
    
    def test_azure_deployment(self):
        """Test Azure ML deployment."""
        logger.info("=" * 60)
        logger.info("TESTING AZURE ML DEPLOYMENT")
        logger.info("=" * 60)
        
        try:
            endpoint_name = "fin-behavior-ep-fixed"
            
            # Get endpoint details
            if not endpoint_name:
                raise ValueError("Endpoint name cannot be None")
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            logger.info(f"✅ Endpoint found: {endpoint_name}")
            logger.info(f"Endpoint URL: {endpoint.scoring_uri}")
            logger.info(f"Endpoint state: {endpoint.provisioning_state}")
            
            if endpoint.provisioning_state != "Succeeded":
                logger.warning(f"⚠️ Endpoint is not in Succeeded state: {endpoint.provisioning_state}")
            
            # Get endpoint key
            keys = self.ml_client.online_endpoints.get_keys(endpoint_name)
            # Use the first available key
            endpoint_key = str(keys)
            
            # Load model info to get correct feature columns
            model_info_path = "outputs/model_info.json"
            if os.path.exists(model_info_path):
                import json
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                feature_columns = model_info.get('feature_columns', [])
            else:
                # Fallback to default features
                feature_columns = [
                    'Age', 'Transaction Amount', 'Account Balance', 'AccountAgeDays',
                    'TransactionHour', 'TransactionDayOfWeek', 'Transaction Type_Deposit',
                    'Transaction Type_Transfer', 'Transaction Type_Withdrawal',
                    'Gender_Female', 'Gender_Male', 'Gender_Other'
                ]
            
            # Create test data with correct features
            test_data = self._create_test_data(5, feature_columns)
            
            # Convert to JSON for REST API call
            import json
            import requests
            
            # Format data as expected by the scoring script
            payload = test_data.to_dict('records')[0]  # Send single record
            
            # Make prediction request
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {endpoint_key}'
            }
            
            logger.info("Making prediction request to Azure ML endpoint...")
            logger.info(f"Payload: {payload}")
            
            response = requests.post(
                f"{endpoint.scoring_uri}",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Azure ML prediction successful!")
                logger.info(f"Response: {result}")
                return True
            else:
                logger.error(f"❌ Azure ML prediction failed: {response.status_code}")
                logger.error(f"Response text: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Azure ML deployment test failed: {e}")
            return False
    
    def _create_test_data(self, n_samples=5, feature_columns=None):
        """Create test data with correct feature names and types."""
        if feature_columns is None:
            feature_columns = [
                'Age', 'Transaction Amount', 'Account Balance', 'AccountAgeDays',
                'TransactionHour', 'TransactionDayOfWeek', 'Transaction Type_Deposit',
                'Transaction Type_Transfer', 'Transaction Type_Withdrawal',
                'Gender_Female', 'Gender_Male', 'Gender_Other'
            ]
        
        np.random.seed(42)
        data = {}
        
        for feature in feature_columns:
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
                        
                        # Get deployment logs if available
                        try:
                            import subprocess
                            if ep.name and dep.name and self.resource_group and self.workspace_name:
                                result = subprocess.run([
                                    'az', 'ml', 'online-deployment', 'get-logs',
                                    '--endpoint-name', str(ep.name),
                                    '--name', str(dep.name),
                                    '--resource-group', str(self.resource_group),
                                    '--workspace-name', str(self.workspace_name),
                                    '--lines', '10'
                                ], capture_output=True, text=True, timeout=30)
                            
                            if result.returncode == 0:
                                logger.info(f"        └─ Recent logs: {result.stdout.strip()[:100]}...")
                            else:
                                logger.info(f"        └─ Could not fetch logs")
                        except:
                            logger.info(f"        └─ Could not fetch logs")
                else:
                    logger.info(f"    └─ No deployments")
            
            # Check models
            models = list(self.ml_client.models.list())
            logger.info(f"\nFound {len(models)} models in workspace:")
            for model in models:
                logger.info(f"  - {model.name} (version: {model.version})")
                
        except Exception as e:
            logger.error(f"❌ Failed to get deployment status: {e}")
    
    def test_environment_setup(self):
        """Test environment setup and dependencies."""
        logger.info("=" * 60)
        logger.info("TESTING ENVIRONMENT SETUP")
        logger.info("=" * 60)
        
        try:
            # Test required packages
            import sklearn
            import pandas
            import numpy
            import joblib
            import mlflow
            
            logger.info(f"✅ scikit-learn: {sklearn.__version__}")
            logger.info(f"✅ pandas: {pandas.__version__}")
            logger.info(f"✅ numpy: {numpy.__version__}")
            logger.info(f"✅ joblib: {joblib.__version__}")
            logger.info(f"✅ mlflow: {mlflow.__version__}")
            
            # Test Azure ML packages
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            logger.info("✅ Azure ML packages available")
            
            # Test data files
            data_path = "data/processed/Comprehensive_Banking_Database_processed.csv"
            if os.path.exists(data_path):
                logger.info(f"✅ Data file exists: {data_path}")
            else:
                logger.warning(f"⚠️ Data file not found: {data_path}")
            
            # Test model files
            model_path = "outputs/model_compatible.joblib"
            if os.path.exists(model_path):
                logger.info(f"✅ Compatible model exists: {model_path}")
            else:
                logger.warning(f"⚠️ Compatible model not found: {model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all deployment tests."""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE DEPLOYMENT TESTING")
        logger.info("=" * 60)
        
        results = {}
        
        # Step 1: Test environment setup
        logger.info("\n" + "=" * 60)
        results['environment'] = self.test_environment_setup()
        
        # Step 2: Get current status
        self.get_deployment_status()
        
        # Step 3: Test model compatibility
        logger.info("\n" + "=" * 60)
        results['model_compatibility'] = self.test_model_compatibility()
        
        # Step 4: Test local model
        logger.info("\n" + "=" * 60)
        results['local_model'] = self.test_local_model()
        
        # Step 5: Test Azure deployment
        logger.info("\n" + "=" * 60)
        results['azure_test'] = self.test_azure_deployment()
        
        # Step 6: Final status report
        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            if result:
                passed += 1
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! Deployment is working correctly.")
        else:
            logger.warning("⚠️ Some tests failed. Check the logs above for details.")
        
        return results

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Test deployments comprehensively")
    parser.add_argument(
        "--test-type",
        choices=["all", "local", "azure", "status", "environment", "compatibility"],
        default="all",
        help="Type of test to run"
    )
    
    args = parser.parse_args()
    
    try:
        tester = DeploymentTester()
        
        if args.test_type == "all":
            tester.run_all_tests()
        elif args.test_type == "local":
            tester.test_local_model()
        elif args.test_type == "azure":
            tester.test_azure_deployment()
        elif args.test_type == "status":
            tester.get_deployment_status()
        elif args.test_type == "environment":
            tester.test_environment_setup()
        elif args.test_type == "compatibility":
            tester.test_model_compatibility()
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
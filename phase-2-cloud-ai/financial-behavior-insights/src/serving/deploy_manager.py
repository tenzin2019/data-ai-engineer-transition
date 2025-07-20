#!/usr/bin/env python3
"""
deploy_manager.py

Comprehensive deployment manager for Azure ML.
Handles model deployment, endpoint management, and testing.

Usage:
    python deploy_manager.py --action deploy --model-name financial-behavior-model
    python deploy_manager.py --action test --endpoint-name financial-behavior-endpoint
    python deploy_manager.py --action status
"""

import os
import sys
import logging
import argparse
import time
import json
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Comprehensive deployment manager for Azure ML."""
    
    def __init__(self):
        """Initialize the deployment manager."""
        load_dotenv()
        
        # Azure ML configuration
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
    
    def deploy_model(
        self,
        model_name: str,
        endpoint_name: str = "fin-behavior-ep-fixed",
        deployment_name: str = "blue",
        instance_type: str = "Standard_F4s_v2",
        instance_count: int = 1
    ) -> Dict[str, Any]:
        """
        Deploy model to Azure ML endpoint.
        
        Args:
            model_name: Name of the model to deploy
            endpoint_name: Name of the endpoint
            deployment_name: Name of the deployment
            instance_type: Azure VM instance type
            instance_count: Number of instances
            
        Returns:
            Dict with deployment results
        """
        try:
            from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment
            from azure.ai.ml.constants import AssetTypes
            
            # Step 1: Create or update endpoint
            logger.info(f"Creating/updating endpoint: {endpoint_name}")
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                description="Financial behavior prediction endpoint",
                auth_mode="key"
            )
            
            endpoint_result = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            logger.info(f"✅ Endpoint '{endpoint_name}' ready")
            
            # Step 2: Register MLflow model in Azure ML and use Azure ML model URI
            try:
                # First, register the MLflow model in Azure ML if not already registered
                from azure.ai.ml.entities import Model
                
                # Check if model already exists in Azure ML
                try:
                    azure_model = self.ml_client.models.get(model_name, label="latest")
                    logger.info(f"Model '{model_name}' already exists in Azure ML (version: {azure_model.version})")
                    # Force registration of new version to avoid compatibility issues
                    logger.info("Forcing registration of new model version for compatibility...")
                    raise Exception("Force new registration")
                except:
                    # Register the compatible model in Azure ML
                    logger.info(f"Registering model '{model_name}' in Azure ML...")
                    
                    # Create a custom environment that matches Azure ML sklearn-1.0 environment
                    from azure.ai.ml.entities import Environment
                    
                    # Define custom environment with exact Azure ML sklearn-1.0 specifications
                    custom_env = Environment(
                        name="custom-sklearn-1.0",
                        conda_file={
                            "name": "azureml-sklearn-1.0",
                            "channels": ["conda-forge"],
                            "dependencies": [
                                "python=3.8",
                                "pip",
                                {
                                    "pip": [
                                        "scikit-learn==1.0.2",
                                        "pandas>=1.1.0,<2.0.0",
                                        "numpy>=1.19.0,<2.0.0",
                                        "joblib>=1.0.0,<2.0.0",
                                        "mlflow>=1.20.0,<2.0.0",
                                        "azureml-inference-server-http>=0.7.0",
                                        "azureml-defaults>=1.44.0"
                                    ]
                                }
                            ]
                        }
                    )
                    
                    # Register the environment
                    try:
                        self.ml_client.environments.create_or_update(custom_env)
                        logger.info("Custom environment registered successfully")
                    except Exception as e:
                        logger.warning(f"Could not register custom environment: {e}")
                    
                    # Register the compatible model
                    model = Model(
                        path="outputs/model_compatible.joblib",
                        type="custom_model",
                        name=model_name,
                        description="Financial behavior prediction model (compatible with Azure ML)"
                    )
                    
                    azure_model = self.ml_client.models.create_or_update(model)
                    model_uri = f"azureml:{model_name}:{azure_model.version}"
                    logger.info(f"Model registered successfully with version: {azure_model.version}")
                
            except Exception as e:
                logger.error(f"Failed to register model: {e}")
                return {"success": False, "error": f"Model registration failed: {e}"}
            
            # Step 3: Create deployment using Azure CLI (more reliable)
            logger.info(f"Creating deployment: {deployment_name}")
            
            # Create temporary directory for code
            import tempfile
            import shutil
            
            temp_code_dir = tempfile.mkdtemp()
            scoring_script_src = os.path.join("src", "serving", "score.py")
            scoring_script_dst = os.path.join(temp_code_dir, "score.py")
            
            # Copy scoring script to temporary directory
            shutil.copy2(scoring_script_src, scoring_script_dst)
            logger.info(f"Copied scoring script to: {temp_code_dir}")
            
            # Create deployment YAML configuration with correct schema
            deployment_config = {
                "name": deployment_name,
                "endpoint_name": endpoint_name,
                "model": model_uri,
                "instance_type": instance_type,
                "instance_count": instance_count,
                "environment": "azureml:AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest",
                "code_configuration": {
                    "code": temp_code_dir,
                    "scoring_script": "score.py"
                }
            }
            
            # Use Azure CLI for deployment
            import subprocess
            import tempfile
            import yaml
            
            # Create temporary YAML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(deployment_config, f)
                yaml_file = f.name
            
            try:
                # Check if deployment already exists
                check_cmd = [
                    "az", "ml", "online-deployment", "show",
                    "--endpoint-name", endpoint_name,
                    "--name", deployment_name,
                    "--resource-group", self.resource_group,
                    "--workspace-name", self.workspace_name
                ]
                
                check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=30)
                
                if check_result.returncode == 0:
                    # Deployment exists, use update
                    logger.info(f"Deployment '{deployment_name}' exists, updating...")
                    cmd = [
                        "az", "ml", "online-deployment", "update",
                        "--endpoint-name", endpoint_name,
                        "--name", deployment_name,
                        "--file", yaml_file,
                        "--resource-group", self.resource_group,
                        "--workspace-name", self.workspace_name
                    ]
                else:
                    # Deployment doesn't exist, use create
                    logger.info(f"Creating new deployment '{deployment_name}'...")
                    cmd = [
                        "az", "ml", "online-deployment", "create",
                        "--endpoint-name", endpoint_name,
                        "--name", deployment_name,
                        "--file", yaml_file,
                        "--resource-group", self.resource_group,
                        "--workspace-name", self.workspace_name
                    ]
                
                logger.info("Starting deployment with Azure CLI...")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                
                if result.returncode == 0:
                    logger.info("✅ Deployment started successfully")
                else:
                    logger.error(f"❌ Deployment failed: {result.stderr}")
                    return {"success": False, "error": result.stderr}
                    
            finally:
                # Clean up temporary file
                os.unlink(yaml_file)
            
            # Monitor deployment status
            logger.info("Monitoring deployment status...")
            time.sleep(60)  # Wait for deployment to start
            
            # Check deployment status using Azure CLI
            status_cmd = [
                "az", "ml", "online-deployment", "show",
                "--endpoint-name", endpoint_name,
                "--name", deployment_name,
                "--resource-group", self.resource_group,
                "--workspace-name", self.workspace_name
            ]
            
            try:
                result = subprocess.run(status_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    import json
                    deployment_info = json.loads(result.stdout)
                    state = deployment_info.get("provisioning_state", "Unknown")
                    logger.info(f"Deployment state: {state}")
                    
                    if state == "Succeeded":
                        # Route traffic to new deployment
                        traffic_cmd = [
                            "az", "ml", "online-endpoint", "update",
                            "--name", endpoint_name,
                            "--traffic", f"{deployment_name}=100",
                            "--resource-group", self.resource_group,
                            "--workspace-name", self.workspace_name
                        ]
                        
                        traffic_result = subprocess.run(traffic_cmd, capture_output=True, text=True, timeout=30)
                        if traffic_result.returncode == 0:
                            logger.info("✅ Deployment successful and traffic routed!")
                            return {
                                "success": True,
                                "endpoint_name": endpoint_name,
                                "deployment_name": deployment_name,
                                "model_name": model_name,
                                "model_uri": model_uri,
                                "state": state
                            }
                        else:
                            logger.warning(f"Traffic routing failed: {traffic_result.stderr}")
                            return {
                                "success": True,
                                "endpoint_name": endpoint_name,
                                "deployment_name": deployment_name,
                                "model_name": model_name,
                                "model_uri": model_uri,
                                "state": state,
                                "warning": "Traffic routing failed"
                            }
                    else:
                        logger.error(f"❌ Deployment failed with state: {state}")
                        return {"success": False, "error": f"Deployment failed: {state}"}
                else:
                    logger.error(f"❌ Failed to get deployment status: {result.stderr}")
                    return {"success": False, "error": f"Status check failed: {result.stderr}"}
                    
            except Exception as e:
                logger.error(f"❌ Deployment monitoring failed: {e}")
                return {"success": False, "error": str(e)}
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return {"success": False, "error": str(e)}
    
    def test_deployment(
        self,
        endpoint_name: str = "fin-behavior-ep-fixed",
        n_samples: int = 5
    ) -> Dict[str, Any]:
        """
        Test deployment with sample data.
        
        Args:
            endpoint_name: Name of the endpoint to test
            n_samples: Number of test samples
            
        Returns:
            Dict with test results
        """
        try:
            # Get endpoint details
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            logger.info(f"Testing endpoint: {endpoint.scoring_uri}")
            
            # Get endpoint key
            keys = self.ml_client.online_endpoints.get_keys(endpoint_name)
            endpoint_key = keys.primary_key if hasattr(keys, 'primary_key') else keys.key1
            
            # Create test data
            test_data = self._create_test_data(n_samples)
            logger.info(f"Created test data with shape: {test_data.shape}")
            
            # Convert to JSON for REST API call
            payload = {
                "data": test_data.to_dict('records')
            }
            
            # Make prediction request
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {endpoint_key}'
            }
            
            logger.info("Making prediction request...")
            start_time = time.time()
            response = requests.post(
                f"{endpoint.scoring_uri}",
                json=payload,
                headers=headers,
                timeout=30
            )
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Prediction successful in {request_time:.2f}s")
                logger.info(f"Response: {result}")
                
                return {
                    "success": True,
                    "response_time": request_time,
                    "status_code": response.status_code,
                    "predictions": result,
                    "test_samples": n_samples
                }
            else:
                logger.error(f"❌ Prediction failed: {response.status_code} - {response.text}")
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "error": response.text
                }
                
        except Exception as e:
            logger.error(f"❌ Deployment test failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        try:
            # Get all endpoints
            endpoints = list(self.ml_client.online_endpoints.list())
            
            status = {
                "endpoints": [],
                "total_endpoints": len(endpoints),
                "total_deployments": 0
            }
            
            for endpoint in endpoints:
                endpoint_info = {
                    "name": endpoint.name,
                    "state": endpoint.provisioning_state,
                    "scoring_uri": endpoint.scoring_uri,
                    "deployments": []
                }
                
                # Get deployments for this endpoint
                deployments = list(self.ml_client.online_deployments.list(endpoint.name))
                endpoint_info["deployments"] = [
                    {
                        "name": dep.name,
                        "state": dep.provisioning_state,
                        "model": dep.model,
                        "instance_type": dep.instance_type,
                        "instance_count": dep.instance_count
                    }
                    for dep in deployments
                ]
                
                status["endpoints"].append(endpoint_info)
                status["total_deployments"] += len(deployments)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {"error": str(e)}
    
    def delete_deployment(
        self,
        endpoint_name: str,
        deployment_name: str
    ) -> Dict[str, bool]:
        """
        Delete a deployment.
        
        Args:
            endpoint_name: Name of the endpoint
            deployment_name: Name of the deployment to delete
            
        Returns:
            Dict with deletion result
        """
        try:
            logger.info(f"Deleting deployment '{deployment_name}' from endpoint '{endpoint_name}'")
            
            # Delete deployment
            self.ml_client.online_deployments.begin_delete(
                name=deployment_name,
                endpoint_name=endpoint_name
            ).result()
            
            logger.info(f"✅ Deployment '{deployment_name}' deleted successfully")
            return {"success": True}
            
        except Exception as e:
            logger.error(f"❌ Failed to delete deployment: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_endpoint(self, endpoint_name: str) -> Dict[str, bool]:
        """
        Delete an endpoint and all its deployments.
        
        Args:
            endpoint_name: Name of the endpoint to delete
            
        Returns:
            Dict with deletion result
        """
        try:
            logger.info(f"Deleting endpoint '{endpoint_name}' and all deployments")
            
            # Delete endpoint (this will also delete all deployments)
            self.ml_client.online_endpoints.begin_delete(endpoint_name).result()
            
            logger.info(f"✅ Endpoint '{endpoint_name}' deleted successfully")
            return {"success": True}
            
        except Exception as e:
            logger.error(f"❌ Failed to delete endpoint: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_test_data(self, n_samples: int = 5) -> pd.DataFrame:
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
    
    def _get_deployment_logs(self, deployment_name: str, endpoint_name: str):
        """Get deployment logs for troubleshooting."""
        try:
            logger.info("Retrieving deployment logs...")
            logs = self.ml_client.online_deployments.get_logs(
                name=deployment_name,
                endpoint_name=endpoint_name,
                lines=100
            )
            logger.error(f"Deployment logs:\n{logs}")
            
            # Also try to get storage initializer logs
            try:
                storage_logs = self.ml_client.online_deployments.get_logs(
                    name=deployment_name,
                    endpoint_name=endpoint_name,
                    lines=100,
                    container_type="storage-initializer"
                )
                logger.error(f"Storage initializer logs:\n{storage_logs}")
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Could not retrieve logs: {e}")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Azure ML Deployment Manager")
    parser.add_argument("--action", required=True,
                       choices=["deploy", "test", "status", "delete-deployment", "delete-endpoint"],
                       help="Action to perform")
    parser.add_argument("--model-name", default="financial-behavior-model-optimized",
                       help="Model name for deployment")
    parser.add_argument("--endpoint-name", default="fin-behavior-ep-fixed",
                       help="Endpoint name")
    parser.add_argument("--deployment-name", default="blue",
                       help="Deployment name")
    parser.add_argument("--instance-type", default="Standard_F4s_v2",
                       help="Azure VM instance type")
    parser.add_argument("--instance-count", type=int, default=1,
                       help="Number of instances")
    parser.add_argument("--test-samples", type=int, default=5,
                       help="Number of test samples")
    
    args = parser.parse_args()
    
    try:
        manager = DeploymentManager()
        
        if args.action == "deploy":
            logger.info("=" * 60)
            logger.info("DEPLOYING MODEL TO AZURE ML")
            logger.info("=" * 60)
            
            result = manager.deploy_model(
                model_name=args.model_name,
                endpoint_name=args.endpoint_name,
                deployment_name=args.deployment_name,
                instance_type=args.instance_type,
                instance_count=args.instance_count
            )
            
            if result["success"]:
                logger.info("🎉 DEPLOYMENT SUCCESSFUL!")
                logger.info(f"Endpoint: {result['endpoint_name']}")
                logger.info(f"Deployment: {result['deployment_name']}")
                logger.info(f"Model: {result['model_name']}:{result['model_version']}")
                logger.info(f"Scoring URI: {result['scoring_uri']}")
            else:
                logger.error(f"❌ DEPLOYMENT FAILED: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.action == "test":
            logger.info("=" * 60)
            logger.info("TESTING DEPLOYMENT")
            logger.info("=" * 60)
            
            result = manager.test_deployment(
                endpoint_name=args.endpoint_name,
                n_samples=args.test_samples
            )
            
            if result["success"]:
                logger.info("✅ DEPLOYMENT TEST SUCCESSFUL!")
                logger.info(f"Response time: {result['response_time']:.2f}s")
                logger.info(f"Test samples: {result['test_samples']}")
            else:
                logger.error(f"❌ DEPLOYMENT TEST FAILED: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.action == "status":
            logger.info("=" * 60)
            logger.info("DEPLOYMENT STATUS")
            logger.info("=" * 60)
            
            status = manager.get_deployment_status()
            
            if "error" in status:
                logger.error(f"❌ Failed to get status: {status['error']}")
                sys.exit(1)
            
            logger.info(f"Total endpoints: {status['total_endpoints']}")
            logger.info(f"Total deployments: {status['total_deployments']}")
            
            for endpoint in status["endpoints"]:
                logger.info(f"\nEndpoint: {endpoint['name']} ({endpoint['state']})")
                if endpoint['deployments']:
                    for dep in endpoint['deployments']:
                        logger.info(f"  └─ {dep['name']} ({dep['state']}) - {dep['model']}")
                else:
                    logger.info("  └─ No deployments")
        
        elif args.action == "delete-deployment":
            logger.info("=" * 60)
            logger.info("DELETING DEPLOYMENT")
            logger.info("=" * 60)
            
            result = manager.delete_deployment(args.endpoint_name, args.deployment_name)
            
            if result["success"]:
                logger.info("✅ Deployment deleted successfully")
            else:
                logger.error(f"❌ Failed to delete deployment: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        elif args.action == "delete-endpoint":
            logger.info("=" * 60)
            logger.info("DELETING ENDPOINT")
            logger.info("=" * 60)
            
            result = manager.delete_endpoint(args.endpoint_name)
            
            if result["success"]:
                logger.info("✅ Endpoint deleted successfully")
            else:
                logger.error(f"❌ Failed to delete endpoint: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
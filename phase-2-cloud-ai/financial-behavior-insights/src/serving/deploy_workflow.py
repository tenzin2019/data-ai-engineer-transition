#!/usr/bin/env python3
"""
Comprehensive deployment workflow for the financial behavior insights model.
Handles local testing, model registration, and deployment to Azure ML.
"""

import os
import sys
import subprocess
import logging
import time
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def run_command(command, description, check=True):
    """Run a shell command with logging."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, shell=True, check=check, capture_output=True, text=True
        )
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stdout:
            logger.info(f"Output: {e.stdout}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("Checking prerequisites...")
    
    # Check if we're in the right directory
    current_dir = Path.cwd()
    if not (current_dir / "src" / "serving").exists():
        logger.error("Must run from project root directory")
        return False
    
    # Check if model exists
    model_path = current_dir / "outputs" / "model.joblib"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        logger.info("Please run training first:")
        logger.info("python src/training/train_model.py --input-data data/processed/Comprehensive_Banking_Database_processed.csv")
        return False
    
    # Check if environment variables are set
    required_vars = ["AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_WORKSPACE_NAME"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("These are required for Azure ML deployment")
        logger.info("You can set them in a .env file or export them")
        return False
    
    logger.info("✅ Prerequisites check passed")
    return True

def run_local_tests():
    """Run local endpoint tests."""
    logger.info("Running local endpoint tests...")
    
    test_script = Path("src/serving/test_local.py")
    if not test_script.exists():
        logger.error(f"Test script not found: {test_script}")
        return False
    
    success = run_command(
        f"python3 {test_script}",
        "Local endpoint tests"
    )
    
    if success:
        logger.info("✅ Local tests passed")
    else:
        logger.error("❌ Local tests failed")
    
    return success

def register_model():
    """Register the model in Azure ML Model Registry."""
    logger.info("Registering model in Azure ML...")
    
    success = run_command(
        "python3 src/training/train_model.py --input-data data/processed/Comprehensive_Banking_Database_processed.csv --register-model",
        "Model registration"
    )
    
    if success:
        logger.info("✅ Model registered successfully")
    else:
        logger.error("❌ Model registration failed")
    
    return success

def deploy_endpoint():
    """Deploy the endpoint to Azure ML."""
    logger.info("Deploying endpoint to Azure ML...")
    
    success = run_command(
        "python3 src/training/deploy_model.py",
        "Endpoint deployment"
    )
    
    if success:
        logger.info("✅ Endpoint deployed successfully")
    else:
        logger.error("❌ Endpoint deployment failed")
    
    return success

def test_deployed_endpoint():
    """Test the deployed endpoint."""
    logger.info("Testing deployed endpoint...")
    
    # This would require the endpoint URL and key from deployment
    # For now, we'll just log that this step should be done manually
    logger.info("⚠️  Manual testing required:")
    logger.info("1. Get the endpoint URL and key from deployment output")
    logger.info("2. Test with curl or Python requests")
    logger.info("3. Example curl command:")
    logger.info('curl -X POST "ENDPOINT_URL" \\')
    logger.info('  -H "Authorization: Bearer PRIMARY_KEY" \\')
    logger.info('  -H "Content-Type: application/json" \\')
    logger.info('  -d \'{"data": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]}\'')
    
    return True

def cleanup_resources():
    """Clean up resources (optional)."""
    logger.info("Cleanup options:")
    logger.info("1. Delete endpoint: az ml online-endpoint delete --name ENDPOINT_NAME")
    logger.info("2. Delete model: az ml model delete --name MODEL_NAME")
    logger.info("3. Delete workspace resources: az ml workspace delete --name WORKSPACE_NAME")
    return True

def main():
    """Main deployment workflow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy financial behavior insights model")
    parser.add_argument(
        "--skip-local-tests",
        action="store_true",
        help="Skip local endpoint tests"
    )
    parser.add_argument(
        "--skip-registration",
        action="store_true",
        help="Skip model registration (if already registered)"
    )
    parser.add_argument(
        "--skip-deployment",
        action="store_true",
        help="Skip endpoint deployment"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run local tests"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting deployment workflow...")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed")
        return 1
    
    # Run local tests
    if not args.skip_local_tests:
        if not run_local_tests():
            logger.error("Local tests failed. Aborting deployment.")
            return 1
    
    if args.test_only:
        logger.info("Test-only mode. Exiting.")
        return 0
    
    # Register model
    if not args.skip_registration:
        if not register_model():
            logger.error("Model registration failed")
            return 1
    
    # Deploy endpoint
    if not args.skip_deployment:
        if not deploy_endpoint():
            logger.error("Endpoint deployment failed")
            return 1
        
        # Test deployed endpoint
        test_deployed_endpoint()
    
    logger.info("🎉 Deployment workflow completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main()) 
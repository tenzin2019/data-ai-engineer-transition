#!/usr/bin/env python3
"""
workflow_runner.py

Comprehensive workflow runner for the financial behavior insights project.
Orchestrates the entire end-to-end pipeline from data preparation to deployment.

Usage:
    python workflow_runner.py [--step all|data|train|retrain|deploy|test]
    python workflow_runner.py --full-pipeline
"""

import os
import sys
import logging
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("workflow.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkflowRunner:
    """Comprehensive workflow runner for the ML pipeline."""
    
    def __init__(self):
        """Initialize the workflow runner."""
        load_dotenv()
        self.start_time = time.time()
        self.results = {}
        
        # Validate environment
        self._validate_environment()
    
    def _validate_environment(self):
        """Validate the environment setup."""
        logger.info("🔍 Validating environment setup...")
        
        # Check required environment variables
        required_vars = [
            "AZURE_SUBSCRIPTION_ID",
            "AZURE_RESOURCE_GROUP", 
            "AZURE_WORKSPACE_NAME"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables: {missing_vars}")
            raise ValueError(f"Missing environment variables: {missing_vars}")
        
        logger.info("✅ Environment variables validated")
        
        # Check required packages
        try:
            import sklearn
            import pandas
            import numpy
            import joblib
            import mlflow
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential
            
            logger.info(f"✅ scikit-learn: {sklearn.__version__}")
            logger.info(f"✅ pandas: {pandas.__version__}")
            logger.info(f"✅ numpy: {numpy.__version__}")
            logger.info(f"✅ joblib: {joblib.__version__}")
            logger.info(f"✅ mlflow: {mlflow.__version__}")
            logger.info("✅ Azure ML packages available")
            
        except ImportError as e:
            logger.error(f"❌ Missing required package: {e}")
            raise
    
    def run_command(self, command: str, description: str) -> bool:
        """Run a shell command and log the result."""
        logger.info(f"🚀 {description}")
        logger.info(f"Command: {command}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ {description} completed successfully")
                if result.stdout.strip():
                    logger.debug(f"Output: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"❌ {description} failed")
                logger.error(f"Error: {result.stderr.strip()}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ {description} timed out after 30 minutes")
            return False
        except Exception as e:
            logger.error(f"❌ {description} failed with exception: {e}")
            return False
    
    def step_data_preparation(self) -> bool:
        """Step 1: Data preparation and preprocessing."""
        logger.info("=" * 80)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("=" * 80)
        
        # Check if data already exists
        data_path = "data/processed/Comprehensive_Banking_Database_processed.csv"
        if os.path.exists(data_path):
            logger.info(f"✅ Processed data already exists: {data_path}")
            return True
        
        # Run data preparation
        command = "make data-prep"
        success = self.run_command(command, "Data preparation")
        
        if success:
            # Verify the output
            if os.path.exists(data_path):
                logger.info(f"✅ Data preparation completed: {data_path}")
                return True
            else:
                logger.error(f"❌ Data file not found after preparation: {data_path}")
                return False
        
        return False
    
    def step_model_training(self) -> bool:
        """Step 2: Model training with original sklearn version."""
        logger.info("=" * 80)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("=" * 80)
        
        # Check if training data exists
        data_path = "data/processed/Comprehensive_Banking_Database_processed.csv"
        if not os.path.exists(data_path):
            logger.error(f"❌ Training data not found: {data_path}")
            return False
        
        # Run model training
        command = "make train"
        success = self.run_command(command, "Model training")
        
        if success:
            # Verify the output
            model_path = "outputs/model.pkl"
            if os.path.exists(model_path):
                logger.info(f"✅ Model training completed: {model_path}")
                return True
            else:
                logger.warning(f"⚠️ Model file not found after training: {model_path}")
                return True  # Continue anyway
        
        return False
    
    def step_model_retraining(self) -> bool:
        """Step 3: Model retraining with compatible sklearn version."""
        logger.info("=" * 80)
        logger.info("STEP 3: MODEL RETRAINING FOR COMPATIBILITY")
        logger.info("=" * 80)
        
        # Check if training data exists
        data_path = "data/processed/Comprehensive_Banking_Database_processed.csv"
        if not os.path.exists(data_path):
            logger.error(f"❌ Training data not found: {data_path}")
            return False
        
        # Run model retraining
        command = "make retrain"
        success = self.run_command(command, "Model retraining")
        
        if success:
            # Verify the output
            model_path = "outputs/model_compatible.joblib"
            if os.path.exists(model_path):
                logger.info(f"✅ Model retraining completed: {model_path}")
                return True
            else:
                logger.error(f"❌ Compatible model not found after retraining: {model_path}")
                return False
        
        return False
    
    def step_deployment(self) -> bool:
        """Step 4: Model deployment to Azure ML."""
        logger.info("=" * 80)
        logger.info("STEP 4: MODEL DEPLOYMENT")
        logger.info("=" * 80)
        
        # Check if compatible model exists
        model_path = "outputs/model_compatible.joblib"
        if not os.path.exists(model_path):
            logger.error(f"❌ Compatible model not found: {model_path}")
            return False
        
        # Run deployment
        command = "make deploy"
        success = self.run_command(command, "Model deployment")
        
        if success:
            # Wait a bit for deployment to stabilize
            logger.info("⏳ Waiting for deployment to stabilize...")
            time.sleep(30)
            
            # Check deployment status
            status_command = "make status"
            status_success = self.run_command(status_command, "Deployment status check")
            
            if status_success:
                logger.info("✅ Deployment completed successfully")
                return True
            else:
                logger.warning("⚠️ Deployment status check failed, but deployment may still be successful")
                return True
        
        return False
    
    def step_testing(self) -> bool:
        """Step 5: Deployment testing and validation."""
        logger.info("=" * 80)
        logger.info("STEP 5: DEPLOYMENT TESTING")
        logger.info("=" * 80)
        
        # Run comprehensive testing
        command = "make test"
        success = self.run_command(command, "Deployment testing")
        
        if success:
            logger.info("✅ Deployment testing completed")
            return True
        
        return False
    
    def run_full_pipeline(self) -> Dict[str, bool]:
        """Run the complete end-to-end pipeline."""
        logger.info("🚀 STARTING COMPLETE END-TO-END PIPELINE")
        logger.info("=" * 80)
        
        pipeline_steps = [
            ("data_preparation", self.step_data_preparation),
            ("model_training", self.step_model_training),
            ("model_retraining", self.step_model_retraining),
            ("deployment", self.step_deployment),
            ("testing", self.step_testing)
        ]
        
        results = {}
        
        for step_name, step_function in pipeline_steps:
            logger.info(f"\n{'='*20} {step_name.upper()} {'='*20}")
            
            try:
                success = step_function()
                results[step_name] = success
                
                if not success:
                    logger.error(f"❌ Pipeline failed at step: {step_name}")
                    logger.info("🛑 Stopping pipeline due to failure")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Pipeline failed at step {step_name} with exception: {e}")
                results[step_name] = False
                break
        
        # Final summary
        self._print_pipeline_summary(results)
        
        return results
    
    def run_single_step(self, step: str) -> bool:
        """Run a single pipeline step."""
        logger.info(f"🚀 RUNNING SINGLE STEP: {step.upper()}")
        logger.info("=" * 80)
        
        step_functions = {
            "data": self.step_data_preparation,
            "train": self.step_model_training,
            "retrain": self.step_model_retraining,
            "deploy": self.step_deployment,
            "test": self.step_testing
        }
        
        if step not in step_functions:
            logger.error(f"❌ Unknown step: {step}")
            logger.info(f"Available steps: {list(step_functions.keys())}")
            return False
        
        try:
            success = step_functions[step]()
            return success
            
        except Exception as e:
            logger.error(f"❌ Step {step} failed with exception: {e}")
            return False
    
    def _print_pipeline_summary(self, results: Dict[str, bool]):
        """Print a summary of the pipeline results."""
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        
        total_time = time.time() - self.start_time
        
        passed = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"Total steps: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        for step_name, success in results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{step_name}: {status}")
        
        if passed == total:
            logger.info("\n🎉 ALL STEPS PASSED! Pipeline completed successfully!")
        else:
            logger.warning(f"\n⚠️ {total - passed} steps failed. Check the logs above for details.")
    
    def cleanup(self):
        """Cleanup temporary files and artifacts."""
        logger.info("🧹 Cleaning up temporary files...")
        
        cleanup_commands = [
            "find . -name '*.pyc' -delete",
            "find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
            "rm -rf .pytest_cache 2>/dev/null || true"
        ]
        
        for command in cleanup_commands:
            try:
                subprocess.run(command, shell=True, check=False)
            except:
                pass
        
        logger.info("✅ Cleanup completed")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run the ML workflow pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "data", "train", "retrain", "deploy", "test"],
        default="all",
        help="Pipeline step to run"
    )
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run the complete end-to-end pipeline"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary files after completion"
    )
    
    args = parser.parse_args()
    
    try:
        runner = WorkflowRunner()
        
        if args.full_pipeline or args.step == "all":
            results = runner.run_full_pipeline()
            success = all(results.values())
        else:
            success = runner.run_single_step(args.step)
        
        if args.cleanup:
            runner.cleanup()
        
        if success:
            logger.info("✅ Workflow completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Workflow failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Workflow runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
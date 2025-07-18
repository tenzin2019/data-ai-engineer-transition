"""
register_azureml_data_asset.py

Registers a processed data asset with Azure ML using environment variables for configuration.

Usage:
    python register_azureml_data_asset.py

Environment Variables:
    AZURE_SUBSCRIPTION_ID: Azure subscription ID
    AZURE_RESOURCE_GROUP: Azure resource group name
    AZURE_WORKSPACE_NAME: Azure ML workspace name
    PROCESSED_DATA_PATH: Path to the processed data file

MLOps Best Practices:
    - Uses environment variables for configuration
    - Checks for missing configuration
    - Provides clear error messages
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve required environment variables
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
processed_data_path = os.getenv("PROCESSED_DATA_PATH")

# Check for missing environment variables
missing_vars = []
for var, value in [
    ("AZURE_SUBSCRIPTION_ID", subscription_id),
    ("AZURE_RESOURCE_GROUP", resource_group),
    ("AZURE_WORKSPACE_NAME", workspace_name),
    ("PROCESSED_DATA_PATH", processed_data_path)
]:
    if not value:
        missing_vars.append(var)
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {missing_vars}")

# Authenticate and create MLClient
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# Register the DVC-tracked CSV as a Data Asset
data_asset = Data(
    path=processed_data_path,
    type="uri_file",
    name="preprocessed-comprehensive-banking-database",
    version="1",
    description="Preprocessed Comprehensive transaction-level banking data"
)

ml_client.data.create_or_update(data_asset)
print("Registered data asset in Azure ML.")
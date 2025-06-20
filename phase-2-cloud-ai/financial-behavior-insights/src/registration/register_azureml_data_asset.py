from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)

# Register the DVC-tracked CSV as a Data Asset
data_asset = Data(
    path="phase-2-cloud-ai/financial-behavior-insights/data/Banking-Dataset/Comprehensive_Banking_Database.csv",
    type="uri_file",
    name="comprehensive-banking-database",
    version="1",
    description="Comprehensive transaction-level banking data"
)

ml_client.data.create_or_update(data_asset)
print("Registered data asset in Azure ML.")
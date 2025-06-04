from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id= os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME")
)

print(ml_client.workspaces.get(os.getenv("AZURE_WORKSPACE_NAME")).mlflow_tracking_uri)
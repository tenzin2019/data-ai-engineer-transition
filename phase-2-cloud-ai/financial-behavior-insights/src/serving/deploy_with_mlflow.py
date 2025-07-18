# This script performs the initial "blue" deployment for an MLflow model on AzureML.
# It loads Azure config from a .env file in the project root.

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Model
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your AzureML workspace details from .env
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace = os.getenv("AZURE_WORKSPACE_NAME")

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace,
)

# ---- Step 1: Register the MLflow model (if not already registered) ----
# Path to your local MLflow model directory (update as needed)
local_model_path = "./mlruns/418652773150309469/eb102ef99649497baddf6d80a7209cf5/artifacts/financial_behavior_model/artifacts"

# Register the model as an MLflow model
registered_model = ml_client.models.create_or_update(
    Model(
        name="financial_behavior_model",
        path=local_model_path,
        type="mlflow_model",  # This is critical for no-code deployment!
        description="Financial behavior model logged with MLflow",
    )
)
print(f"Registered model: {registered_model.name} v{registered_model.version}")

# ---- Step 2: Create the online endpoint (if it doesn't exist) ----
endpoint_name = "my-bluegreen-endpoint"  # Must be unique in workspace

try:
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    print(f"Endpoint '{endpoint_name}' already exists.")
except Exception:
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Blue-green endpoint for financial behavior model",
        auth_mode="key"
    )
    endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Created endpoint: {endpoint.name}")

# ---- Step 3: Create the "blue" deployment ----
blue_deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=f"{registered_model.name}:{registered_model.version}",  # Use name:version
    instance_type="Standard_F2s_v2",
    instance_count=1,
)

ml_client.online_deployments.begin_create_or_update(blue_deployment).result()
print("Blue deployment created.")

# ---- Step 4: Set traffic to blue ----
endpoint.traffic = {"blue": 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("100% traffic routed to blue deployment.")

print("Deployment complete! You can now test your endpoint.") 
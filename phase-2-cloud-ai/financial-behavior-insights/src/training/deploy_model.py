import os
import uuid
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use built-in AzureML sklearn environment by referencing its name and version
sklearn_env = "AzureML-sklearn-1.0-ubuntu20.04-py38-cpu:1"

# Retrieve Azure ML workspace details from environment variables
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

# Define model details (update as needed)
model_name = "financial-behavior-insights-model"
model_version = 1  # Use None for latest version

# Generate a unique endpoint name (max 32 chars)
base_name = "fbinsight"
endpoint_name = f"{base_name}-ep-{uuid.uuid4().hex[:6]}"

# Authenticate and create MLClient
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# --- Create the managed online endpoint ---
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description=f"{model_name} scoring endpoint",
    auth_mode="key"
)

ml_client.begin_create_or_update(endpoint).result()
print(f"Endpoint created: {endpoint_name}")

# --- Pick latest model version if not specified ---
if not model_version:
    model = ml_client.models.get(name=model_name, label="latest")
    model_version = model.version
else:
    model = f"{model_name}:{model_version}"

# --- Define and create deployment ---

deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=model,
    environment=sklearn_env,  # Pass the environment as a string reference
    instance_type="Standard_DS1_v2",  # Use a low-cost SKU for dev/testing
    instance_count=1
)

endpoint = ml_client.online_endpoints.get(name=endpoint_name)
ml_client.begin_create_or_update(endpoint).result()
print(f"Deployment complete. Endpoint: {endpoint_name}")

# --- Retrieve and print scoring URI and key ---
endpoint_obj = ml_client.online_endpoints.get(name=endpoint_name)
keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
print("Scoring URI:", endpoint_obj.scoring_uri)
print("Primary Key:", keys.primary_key)
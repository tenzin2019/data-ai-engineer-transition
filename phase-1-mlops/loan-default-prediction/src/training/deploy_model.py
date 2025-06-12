import os
import uuid
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# --- Get workspace config from env ---
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

# --- Model details ---
model_name = "loan-default-model"
model_version = 1

# Generate uniq endpoint name, Azure limit is 32 chars
short_name = "loandef"
endpoint_name = f"{short_name}-ep-{uuid.uuid4().hex[:6]}"

# --- Auth with Azure ML ---
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# --- Create managed online endpoint ---
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description=f"{model_name} serving endpoint",
    auth_mode="key"
)

# create or update endpoint (should be idempotent, re-runs ok)
ml_client.begin_create_or_update(endpoint).result()
print(f"Endpoint created: {endpoint_name}")

# --- Deployment config (blue is the default slot) ---
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=f"{model_name}:{model_version}",
    instance_type="Standard_F4s_v2",  # cheap VM for test only, prod should use diff
    instance_count=1
)

ml_client.begin_create_or_update(deployment).result()
print("Model deployed to endpoint.")

# --- Set traffic 100% to blue slot (default after first deploy) ---
endpoint = ml_client.online_endpoints.get(name=endpoint_name)
endpoint.traffic = {"blue": 100}

# "begin_create_or_update" works for both new and update, a bit confuzing
ml_client.begin_create_or_update(endpoint).result()
print(f"Deployment complete. Endpoint: {endpoint_name}")

# --- Print scoring URL and primary key for testing ---
endpoint_obj = ml_client.online_endpoints.get(name=endpoint_name)
keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
print("Scoring URI:", endpoint_obj.scoring_uri)
print("Primary Key:", keys.primary_key)
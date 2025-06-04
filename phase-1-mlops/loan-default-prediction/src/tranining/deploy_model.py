import os
import uuid
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# --- ENVIRONMENT SETUP ---
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")
model_name = "loan-default-model"
model_version = 6  # Change to your model version

ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# --- ENDPOINT CREATION ---
endpoint_name = f"loan-default-endpoint-{uuid.uuid4().hex[:6]}"
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description="Loan default model serving endpoint",
    auth_mode="key"
)
ml_client.begin_create_or_update(endpoint).result()
print(f"✅ Endpoint created: {endpoint_name}")

# --- DEPLOYMENT ---
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=f"{model_name}:{model_version}",
    instance_type="Standard_F2s_v2",  # <-- use a supported VM size
    instance_count=1
)
ml_client.begin_create_or_update(deployment).result()
print("✅ Model deployed to endpoint.")

# --- SET DEFAULT DEPLOYMENT ---
ml_client.online_endpoints.begin_update(
    endpoint_name=endpoint_name,
    default_deployment_name="blue"
).result()
print(f"🚀 Deployment complete. Endpoint: {endpoint_name}")
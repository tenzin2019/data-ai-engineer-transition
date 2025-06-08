import os
import uuid
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment

# Retrieve Azure ML workspace details from environment variables
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

# Define model details
model_name = "loan-default-model"
model_version = 1

# Generate a unique endpoint name (max 32 characters)
short_name = "loandef"
endpoint_name = f"{short_name}-ep-{uuid.uuid4().hex[:6]}"

# Authenticate and create MLClient
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name,
)

# Create the managed online endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name,
    description=f"{model_name} serving endpoint",
    auth_mode="key"
)

ml_client.begin_create_or_update(endpoint).result()
print(f"✅ Endpoint created: {endpoint_name}")

# Define the deployment
deployment = ManagedOnlineDeployment(
    name="blue",
    endpoint_name=endpoint_name,
    model=f"{model_name}:{model_version}",
    instance_type="Standard_F4s_v2",
    instance_count=1
)

ml_client.begin_create_or_update(deployment).result()
print("✅ Model deployed to endpoint.")

# Retrieve the endpoint to update traffic settings
endpoint = ml_client.online_endpoints.get(name=endpoint_name)

# Set 100% traffic to the 'blue' deployment
endpoint.traffic = {"blue": 100}

# Update the endpoint with the new traffic configuration
ml_client.begin_create_or_update(endpoint).result()
print(f"🚀 Deployment complete. Endpoint: {endpoint_name}")

# Retrieve and display the scoring URI and primary key
# Retrieve and display the scoring URI and primary key
endpoint_obj = ml_client.online_endpoints.get(name=endpoint_name)
keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
print("Scoring URI:", endpoint_obj.scoring_uri)
print("Primary Key:", keys.primary_key)
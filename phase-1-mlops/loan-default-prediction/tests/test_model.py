from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import os
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
    resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
    workspace_name=os.getenv("AZURE_WORKSPACE_NAME"),
)

models = ml_client.models.list()
for m in models:
    print(m.name, m.version)


from azure.ai.ml.entities import Model

model = Model(
    path="path/to/local/model",  # Directory containing MLmodel file
    name="loan-default-model",
    description="Loan default model for deployment",
    type="mlflow_model"
)

registered_model = ml_client.models.create_or_update(model)
print(registered_model.name, registered_model.version)
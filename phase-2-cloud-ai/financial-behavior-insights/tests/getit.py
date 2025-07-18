from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential
import os
from dotenv import load_dotenv

load_dotenv()
ml_client = MLClient(
    DefaultAzureCredential(),
    os.getenv("AZURE_SUBSCRIPTION_ID"),
    os.getenv("AZURE_RESOURCE_GROUP"),
    os.getenv("AZURE_WORKSPACE_NAME")
)

# Register model
model = ml_client.models.create_or_update(
    Model(
        name="my-mlflow-model",
        path="./mlruns/<exp_id>/<run_id>/artifacts/financial_behavior_model"
    )
)
print("Registered model:", model.name, model.version)

# List all models
for m in ml_client.models.list():
    print(m.name, m.version)
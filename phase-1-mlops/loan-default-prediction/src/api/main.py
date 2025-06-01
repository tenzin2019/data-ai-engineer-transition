from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os

# Define the experiment and model names
EXPERIMENT_NAME = "loan-default-experiment"
MODEL_NAME = "loan-default-model"
MODEL_ALIAS = "production"  # Use None if you prefer to load the latest version without an alias

# Initialize the MLflow client
client = MlflowClient()

# Retrieve the experiment by name
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' does not exist.")

# Retrieve the artifact location
artifact_location = experiment.artifact_location
print(f"Original artifact location: {artifact_location}")

# Determine the current working directory
current_dir = os.getcwd()
adjusted_artifact_path = os.path.join(current_dir, "mlruns")


# Set the MLflow tracking URI to the adjusted artifact path
mlflow.set_tracking_uri(f"file://{adjusted_artifact_path}")

# Load the model using the model name and alias or latest version
try:
    if MODEL_ALIAS:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    else:
        model_uri = f"models:/{MODEL_NAME}/latest"

    model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model '{MODEL_NAME}' loaded successfully from URI: {model_uri}")
except Exception as e:
    print(f"Model load failed: {e}")
    model = None

# Initialize FastAPI app
app = FastAPI(title="Loan Default Prediction API")

# Define input data schema
class InputData(BaseModel):
    age: int
    income: float
    loan_amount: float

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available.")
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}
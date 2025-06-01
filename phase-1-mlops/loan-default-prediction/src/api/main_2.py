import os
import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

class InputData(BaseModel):
    age: float
    income: float
    loan_amount: float

# Explicit tracking URI in Docker
mlflow.set_tracking_uri("file:///app/mlruns")

# Set experiment ID and run ID (from training output)
EXPERIMENT_ID = "0"
RUN_ID = "f13d9dc883e543b39bb8ea6e57d5f6f6"
MODEL_PATH = f"/app/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"

try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

@app.post("/predict")
def predict(input_data: InputData):
    try:
        input_df = pd.DataFrame([{
            "age": float(input_data.age),
            "income": float(input_data.income),
            "loan_amount": float(input_data.loan_amount)
        }])
        prediction = model.predict(input_df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
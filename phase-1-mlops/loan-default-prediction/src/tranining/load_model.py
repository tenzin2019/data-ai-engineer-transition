import mlflow
import mlflow.sklearn
import pandas as pd

# Sample input matching training features
sample_input = pd.DataFrame({
    "age": [30],
    "income": [60000],
    "loan_amount": [10000]
})

# Load the latest run from the local mlruns directory
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("Default") or client.get_experiment("0")
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)

if runs:
    latest_run_id = runs[0].info.run_id
    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    prediction = model.predict(sample_input)
    print(f"Predicted default risk: {prediction[0]}")
else:
    print("No MLflow runs found. Please train a model first.")
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Set the tracking URI to the MLflow tracking server
mlflow.set_tracking_uri("http://localhost:9090")

# Constants
EXPERIMENT_NAME = "loan-default-experiment"
MODEL_NAME = "loan-default-model"
MODEL_ALIAS = "production"

print(f"Using MLflow tracking URI: {mlflow.get_tracking_uri()}")
print(f"current directory: {os.getcwd()}")
# Load data
DATA_PATH = "phase-1-mlops/loan-default-prediction/src/data/loan_data.csv"
data = pd.read_csv(DATA_PATH)
X = data[["age", "income", "loan_amount"]].astype("float64")
y = data["default"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize MLflow client
client = MlflowClient()

# Ensure experiment exists
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
    print(f"Created experiment with ID: {experiment_id}")
else:
    experiment_id = experiment.experiment_id
    print(f"Using existing experiment with ID: {experiment_id}")

# Start new MLflow run
with mlflow.start_run(experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    print(f"Started run with ID: {run_id}")

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log parameters, metrics, and model
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    # Infer the signature and set input_example
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:5]

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=input_example
    )

    # Register model
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Registered model version: {mv.version}")

    # Update alias to point to latest version
    client.set_registered_model_alias(name=MODEL_NAME, alias=MODEL_ALIAS, version=mv.version)
    print(f"Updated alias '{MODEL_ALIAS}' to version {mv.version}")
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Set tracking URI to a local folder for MLflow
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

# Define experiment and model names
EXPERIMENT_NAME = "loan-default-experiment"
MODEL_NAME = "loan-default-model"
ALIAS_NAME = "production"

# Initialize MLflow client
client = MlflowClient()

# Check if the experiment exists; create it if not
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(name=EXPERIMENT_NAME)
    print(f"Created experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}")
else:
    experiment_id = experiment.experiment_id
    print(f"Experiment '{EXPERIMENT_NAME}' already exists with ID: {experiment_id}")

# Load or create your dataset
# For demonstration, we'll create a simple synthetic dataset
data = pd.DataFrame({
    "age": [25, 40, 35, 28, 60],
    "income": [40000, 60000, 50000, 42000, 75000],
    "loan_amount": [5000, 10000, 12000, 7000, 15000],
    "default": [0, 1, 0, 0, 1]
})

# Define features and target
X = data[["age", "income", "loan_amount"]].astype("float64")
y = data["default"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Start an MLflow run
with mlflow.start_run(experiment_id=experiment_id, run_name="random-forest-loan-default") as run:
    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", acc)

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log and register the model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train,
        registered_model_name=MODEL_NAME
    )

    print(f"Model registered as '{MODEL_NAME}' with run ID: {run.info.run_id}")

    # Retrieve the latest model version using the "latest" alias
    latest_version = max(
        [int(m.version) for m in client.search_model_versions(f"name='{MODEL_NAME}'")]
    )

    # Set or update the alias to point to the latest version
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=ALIAS_NAME,
        version=latest_version
    )

    print(f"Alias '{ALIAS_NAME}' set to version {latest_version} of model '{MODEL_NAME}'.")
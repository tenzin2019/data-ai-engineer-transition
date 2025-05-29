import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlflow.models.signature import infer_signature

def load_data():
    """Simulate loading loan default dataset."""
    # Create a small DataFrame to simulate loan default data
    return pd.DataFrame({
        "age": [25, 40, 50, 35, 28],
        "income": [50000, 80000, 120000, 60000, 52000],
        "loan_amount": [10000, 20000, 15000, 12000, 8000],
        "default": [0, 0, 1, 0, 1]
    })

def train_and_log_model(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=3):
    """Train RandomForest and log with MLflow."""
    # Enable automatic logging of sklearn parameters, metrics, and models
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="loan-default-experiment"):
        # Initialize and train the RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        clf.fit(X_train, y_train)
        # Make predictions on the test set
        preds = clf.predict(X_test)
        # Calculate accuracy
        acc = accuracy_score(y_test, preds)

        # Infer model signature for input/output schema
        signature = infer_signature(X_train, clf.predict(X_train))
        # Provide an input example for model logging
        input_example = X_train.iloc[:1]

        # Log the trained model to MLflow
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )
        # Log accuracy metric
        mlflow.log_metric("accuracy", acc)
        # Log model parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        # Set MLflow tags for project and phase
        mlflow.set_tag("project", "loan-default")
        mlflow.set_tag("phase", "phase-2")

def main():
    # Load the dataset
    data = load_data()
    # Select features and target variable
    X = data[["age", "income", "loan_amount"]].astype("float64")
    y = data["default"]
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train the model and log results
    train_and_log_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
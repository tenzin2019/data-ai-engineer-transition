import mlflow.sklearn
import pandas as pd

# Example input — same structure as training
sample_input = pd.DataFrame({
    "age": [30],
    "income": [60000],
    "loan_amount": [10000]
})

# Load the model from MLflow using the model URI
model_uri = "runs:/4c5fb8de9fe44e1cafe56fb47cf2dce7/model"

# Load the model
model = mlflow.sklearn.load_model(model_uri)

# Predict
prediction = model.predict(sample_input)
print(f"Predicted default risk: {prediction[0]}")
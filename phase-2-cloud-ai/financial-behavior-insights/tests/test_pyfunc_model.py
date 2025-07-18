import mlflow
from mlflow.models import Model
import joblib
import mlrun.frameworks.sklearn

model_uri = 'runs:/a5493291677d4165b18c6ef270c601d8/financial_behavior_model'
# The model is logged with an input example
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
input_data = pyfunc_model.input_example

# Verify the model with the provided input data using the logged dependencies.
# For more details, refer to:
# https://mlflow.org/docs/latest/models.html#validate-models-before-deployment
mlflow.models.predict(
    model_uri=model_uri,
    input_data=input_data,
    env_manager="uv",
)

sklearn_model = joblib.load("outputs/model.joblib")
model = mlrun.frameworks.sklearn.apply_mlrun(sklearn_model)


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/training')))
import mlflow
from mlflow.models import Model

logged_model = 'runs:/eb102ef99649497baddf6d80a7209cf5/financial_behavior_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(pd.DataFrame(data))
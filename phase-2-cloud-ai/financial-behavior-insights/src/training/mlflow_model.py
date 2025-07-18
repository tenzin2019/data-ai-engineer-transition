import mlflow.pyfunc
import joblib
import numpy as np
import pandas as pd

class FinancialBehaviorModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model_path"])

    def predict(self, context, model_input):
        # Accepts pandas DataFrame or numpy array
        if isinstance(model_input, pd.DataFrame):
            X = model_input.values
        else:
            X = np.array(model_input)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != 12:
            raise ValueError(f"Expected 12 features, got {X.shape[1]}")
        preds = self.model.predict(X)
        result = {"predictions": preds.tolist()}
        if hasattr(self.model, "predict_proba"):
            result["probabilities"] = self.model.predict_proba(X)[:, 1].tolist()
        return result 
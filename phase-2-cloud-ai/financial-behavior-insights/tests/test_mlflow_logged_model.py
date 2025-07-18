import unittest
import mlflow
import pandas as pd
import numpy as np
import os

class TestMLflowLoggedModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Update this to your actual MLflow model URI or registry path
        cls.model_uri = 'runs:/a5493291677d4165b18c6ef270c601d8/financial_behavior_model'  # Change as needed
        # Load the MLflow model
        cls.model = mlflow.pyfunc.load_model(cls.model_uri)
        # Path to sample data
        cls.data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../data/processed/Comprehensive_Banking_Database_processed.csv'))
        # Load a sample input (first 3 rows, drop target)
        df = pd.read_csv(cls.data_path)
        if 'HighAmount' in df.columns:
            df = df.drop(columns=['HighAmount'])
        cls.sample = df.head(3)

    def test_predict_output_shape(self):
        preds = self.model.predict(self.sample)
        print("Sample shape:", self.sample.shape)
        print("Sample columns:", self.sample.columns)
        print("Predictions:", preds)
        print("Predictions type:", type(preds))
        self.assertEqual(len(preds), len(self.sample))

    def test_predict_output_type(self):
        preds = self.model.predict(self.sample)
        print("Sample shape:", self.sample.shape)
        print("Sample columns:", self.sample.columns)
        print("Predictions:", preds)
        print("Predictions type:", type(preds))
        self.assertTrue(isinstance(preds, (np.ndarray, list, pd.Series, pd.DataFrame)))

    def test_predict_no_nan(self):
        preds = self.model.predict(self.sample)
        print("Sample shape:", self.sample.shape)
        print("Sample columns:", self.sample.columns)
        print("Predictions:", preds)
        print("Predictions type:", type(preds))
        if isinstance(preds, (np.ndarray, pd.Series, pd.DataFrame)):
            self.assertFalse(np.isnan(preds).any())

if __name__ == '__main__':
    unittest.main() 
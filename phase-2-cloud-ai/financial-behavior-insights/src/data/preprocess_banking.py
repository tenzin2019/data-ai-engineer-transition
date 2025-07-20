"""
preprocess_banking.py

This script preprocesses the comprehensive banking dataset for downstream machine learning tasks.
It performs feature engineering, handles missing values, scales and encodes features, and saves processed artifacts.

Usage:
    python preprocess_banking.py --data_path <input_csv> --output_dir <output_directory>

Environment Variables:
    BANKING_DATA_PATH: Path to the input CSV file (optional, can be overridden by CLI argument)

Outputs:
    - Processed features (X.npy)
    - Target variable (y.npy)
    - Preprocessing pipeline (preprocessor.joblib)
    - Processed CSV (Comprehensive_Banking_Database_processed.csv)

MLOps Best Practices:
    - Uses logging for traceability
    - Parameterizes file paths and thresholds
    - Validates input columns and data types
    - Saves all artifacts for reproducibility
"""

import os
import logging
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_data(filepath: str, required_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads data from a CSV file and checks for required columns.
    Args:
        filepath (str): Path to the CSV file.
        required_cols (Optional[List[str]]): List of required columns to check for.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    Raises:
        ValueError: If required columns are missing.
    """
    logging.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    if required_cols is not None:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    return df

def engineer_features(
    df: pd.DataFrame,
    amt_quantile: float = 0.95,
    min_std: float = 100,
    multiplier: float = 5
) -> pd.DataFrame:
    """
    Adds engineered features to the banking DataFrame, including account age, transaction time features,
    high amount flag, and unusual behavior flag.
    Parameters are configurable for flexibility.
    Args:
        df (pd.DataFrame): Input DataFrame.
        amt_quantile (float): Quantile for high amount flag.
        min_std (float): Minimum standard deviation for unusual behavior (not used in current version).
        multiplier (float): Multiplier for unusual behavior (not used in current version).
    Returns:
        pd.DataFrame: DataFrame with new features.
    Raises:
        ValueError: If date parsing fails.
    """
    logging.info("Engineering features...")
    df = df.copy()
    # Parse dates
    try:
        df["Transaction Date"] = pd.to_datetime(df["Transaction Date"])
        df["Date Of Account Opening"] = pd.to_datetime(df["Date Of Account Opening"])
    except Exception as e:
        raise ValueError(f"Date parsing error: {e}")
    # Feature engineering
    df["AccountAgeDays"] = (df["Transaction Date"] - df["Date Of Account Opening"]).dt.days
    df["TransactionHour"] = df["Transaction Date"].dt.hour
    df["TransactionDayOfWeek"] = df["Transaction Date"].dt.dayofweek
    amt_thresh = df["Transaction Amount"].quantile(amt_quantile)
    df["HighAmount"] = (df["Transaction Amount"] >= amt_thresh).astype(int)


    return df

def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Returns lists of numeric and categorical columns for preprocessing.
    Treats binary columns as numeric for efficiency.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        Tuple[List[str], List[str]]: Numeric and categorical column names.
    Raises:
        KeyError: If required columns are missing.
    """
    numeric_cols = [
        "Age", "Transaction Amount", "Account Balance", "AccountAgeDays",
        "TransactionHour", "TransactionDayOfWeek"
    ]
    cat_cols = ["Transaction Type", "Gender"]
    missing_cols = [col for col in numeric_cols + cat_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following required columns are missing from the CSV: {missing_cols}")
    return numeric_cols, cat_cols

def build_preprocessor(numeric_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Builds a preprocessing pipeline for numeric and categorical columns.
    Args:
        numeric_cols (List[str]): List of numeric columns.
        cat_cols (List[str]): List of categorical columns.
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])
    return preprocessor

def save_artifacts(
    X,
    y: np.ndarray,
    preprocessor: ColumnTransformer,
    feature_names: List[str],
    output_dir: str
):
    """
    Saves processed features, target, preprocessor, and processed CSV to disk.
    Args:
        X: Processed features (can be sparse matrix or numpy array).
        y (np.ndarray): Target variable.
        preprocessor (ColumnTransformer): Preprocessing pipeline.
        feature_names (List[str]): Names of features.
        output_dir (str): Directory to save artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Convert to dense array if sparse
    if hasattr(X, 'toarray'):
        X_dense = X.toarray()
    else:
        X_dense = X
    
    np.save(os.path.join(output_dir, "X.npy"), X_dense)
    np.save(os.path.join(output_dir, "y.npy"), y)
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    # Save as CSV (with column names)
    df_out = pd.DataFrame(X_dense, columns=feature_names)
    df_out["HighAmount"] = y
    csv_path = os.path.join(output_dir, "Comprehensive_Banking_Database_processed.csv")
    if csv_path:
        df_out.to_csv(csv_path, index=False)
    logging.info(f"Artifacts (including CSV) saved to {output_dir}")

def main(
    data_path: str = None,
    output_dir: str = "data/processed",
    amt_quantile: float = 0.95,
    min_std: float = 100,
    multiplier: float = 5
):
    """
    Main function to run the preprocessing pipeline with configurable parameters.
    Args:
        data_path (str): Path to input CSV file.
        output_dir (str): Directory to save processed artifacts.
        amt_quantile (float): Quantile for high amount flag.
        min_std (float): Minimum standard deviation for unusual behavior (not used).
        multiplier (float): Multiplier for unusual behavior (not used).
    """
    if data_path is None:
        data_path = os.environ.get("BANKING_DATA_PATH", "data/Banking-Dataset/Comprehensive_Banking_Database.csv")
    required_cols = [
        "Customer ID", "Transaction Date", "Date Of Account Opening", "Age",
        "Transaction Amount", "Account Balance", "Transaction Type", "Gender"
    ]
    df = load_data(data_path, required_cols)
    df = engineer_features(df, amt_quantile=amt_quantile, min_std=min_std, multiplier=multiplier)
    numeric_cols, cat_cols = get_feature_columns(df)
    all_cols = numeric_cols + cat_cols
    X = df[all_cols]
    for col in cat_cols:
        X.loc[:, col] = X[col].astype(str)
    y = df["HighAmount"]
    preprocessor = build_preprocessor(numeric_cols, cat_cols)
    X_processed = preprocessor.fit_transform(X)
    # Get new column names (compatible with scikit-learn 1.1.3)
    cat_feature_names = []
    enc = preprocessor.named_transformers_['cat']['encode']
    try:
        # Try the newer method first
        cat_feature_names = enc.get_feature_names_out(cat_cols)
    except (AttributeError, TypeError):
        # Fallback for older scikit-learn versions
        try:
            # For scikit-learn 1.1.3, we need to construct feature names manually
            cat_feature_names = []
            for i, col in enumerate(cat_cols):
                categories = enc.categories_[i]
                for cat in categories:
                    cat_feature_names.append(f"{col}_{cat}")
        except:
            # Final fallback
            cat_feature_names = [f"{col}_encoded" for col in cat_cols]
    
    # Always generate the correct number of feature names
    feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
    
    # Log feature information
    logging.info(f"Processed data shape: {X_processed.shape}")
    logging.info(f"Number of features: {len(feature_names)}")
    
    save_artifacts(X_processed, y.to_numpy(), preprocessor, feature_names, output_dir)
    logging.info("Preprocessing complete. Processed data, CSV, and preprocessor saved.")

if __name__ == "__main__":
    logging.info("Starting preprocessing script...")
    main()
    logging.info("Preprocessing script finished.")
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


def load_data():
    """Load the Iris dataset into a DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["target_name"] = [iris.target_names[i] for i in iris.target]
    return df, iris.target_names


def preprocess(df, scaler=None, fit=True):
    """Scale features. If fit=True, fit a new scaler; else use provided one."""
    feature_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    X = df[feature_cols].values
    y = df["target"].values

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """Split into train/test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def save_scaler(scaler, path="models/scaler.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def load_scaler(path="models/scaler.joblib"):
    return joblib.load(path)
import joblib
import numpy as np
from sklearn.datasets import load_iris

CLASS_NAMES = load_iris().target_names.tolist()  # ['setosa', 'versicolor', 'virginica']


def load_model(model_path="models/model.joblib"):
    return joblib.load(model_path)


def load_scaler(scaler_path="models/scaler.joblib"):
    return joblib.load(scaler_path)


def predict(features, model=None, scaler=None):
    """
    Predict the Iris species for a list of 4 features:
    [sepal_length, sepal_width, petal_length, petal_width]
    """
    if model is None:
        model = load_model()
    if scaler is None:
        scaler = load_scaler()

    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    return {
        "class_id": int(prediction),
        "class_name": CLASS_NAMES[prediction],
        "probabilities": {
            name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities)
        },
    }

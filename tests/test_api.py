import pytest
import os
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
model_exists = os.path.exists(model_path)

if model_exists:
    from api.app import app

    def test_root():
        with TestClient(app) as client:
            r = client.get("/")
            assert r.status_code == 200

    def test_health():
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200

    def test_predict_setosa():
        with TestClient(app) as client:
            r = client.post(
                "/predict",
                json={
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                },
            )
            assert r.status_code == 200
            assert r.json()["class_name"] == "setosa"

    def test_predict_virginica():
        with TestClient(app) as client:
            r = client.post(
                "/predict",
                json={
                    "sepal_length": 6.3,
                    "sepal_width": 3.3,
                    "petal_length": 6.0,
                    "petal_width": 2.5,
                },
            )
            assert r.status_code == 200
            assert r.json()["class_name"] == "virginica"

else:

    def test_skip():
        pytest.skip("Model not trained — run `make train` first")

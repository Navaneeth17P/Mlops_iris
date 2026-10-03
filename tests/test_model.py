import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import load_data, preprocess, split_data


def test_load_data():
    df, target_names = load_data()
    assert len(df) == 150, "Iris dataset must have 150 rows"
    assert "target" in df.columns
    assert len(target_names) == 3


def test_preprocess():
    df, _ = load_data()
    X, y, scaler = preprocess(df)
    assert X.shape == (150, 4), "Should have 4 features"
    assert len(y) == 150
    assert scaler is not None


def test_split_data():
    df, _ = load_data()
    X, y, _ = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_train) == 120
    assert len(X_test) == 30


def test_predict():
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
    if not os.path.exists(model_path):
        pytest.skip("Model not trained yet — run `make train` first")

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from predict import predict

    result = predict([5.1, 3.5, 1.4, 0.2])
    assert "class_id" in result
    assert "class_name" in result
    assert "probabilities" in result
    assert result["class_id"] in [0, 1, 2]
    assert result["class_name"] == "setosa"

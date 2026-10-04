import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_data, preprocess, split_data, save_scaler


def train(n_estimators=100, max_depth=5, random_state=42):
    mlflow.set_experiment("iris-classification")

    with mlflow.start_run():
        # --- Data ---
        df, target_names = load_data()
        X, y, scaler = preprocess(df)
        X_train, X_test, y_train, y_test = split_data(X, y)

        # --- Log hyperparameters ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)

        # --- Train ---
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        )

        # --- Log metrics ---
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

        # --- Save artifacts ---
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.joblib"
        joblib.dump(model, model_path)
        save_scaler(scaler)

        mlflow.sklearn.log_model(
            model, "model", skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        mlflow.log_artifact(model_path)

        metrics = {"accuracy": accuracy, "classification_report": report}
        with open("models/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact("models/metrics.json")

        print(f"\n✅ Training complete!")
        print(f"   Accuracy : {accuracy:.4f}")
        print(f"   F1 (macro): {report['macro avg']['f1-score']:.4f}")
        print(classification_report(y_test, y_pred, target_names=target_names))

        return model, scaler, accuracy


if __name__ == "__main__":
    train()

from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    """Load a saved model artifact from disk."""
    artifact = joblib.load(model_path)

    required_keys = {"model", "feature_columns", "threshold"}
    missing_keys = required_keys - artifact.keys()

    if missing_keys:
        raise ValueError(
            f"Model artifact is missing required keys: {missing_keys}"
        )

    return artifact


def predict_transaction(
    transaction: dict,
    artifact: dict,
) -> dict:
    """Predict the fraud probability and classification for one transaction."""
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact["threshold"])
    model = artifact["model"]

    missing_features = set(feature_columns) - transaction.keys()

    if missing_features:
        raise ValueError(
            f"Transaction is missing required features: {missing_features}"
        )

    X = pd.DataFrame(
        [{column: transaction[column] for column in feature_columns}]
    )

    probability = float(model.predict_proba(X)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "fraud_probability": probability,
        "is_fraud": prediction,
        "threshold": threshold,
    }

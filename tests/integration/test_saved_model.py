import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from fraud_detection.models.predict import load_model, predict_transaction


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"


def test_saved_random_forest_loads_and_predicts():
    # Load the actual model artifact saved during training.
    artifact = joblib.load(MODEL_PATH)

    assert "model" in artifact
    assert "threshold" in artifact
    assert "feature_columns" in artifact

    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]

    # Build a synthetic transaction using the expected features.
    X = pd.DataFrame(
        [{column: 0.0 for column in feature_columns}]
    )

    # Use a plausible transaction amount if Amount is expected.
    if "Amount" in X.columns:
        X["Amount"] = 100.0

    # Run inference with the real saved model.
    probabilities = model.predict_proba(X)[:, 1]

    assert probabilities.shape == (1,)
    assert np.isfinite(probabilities).all()
    assert 0.0 <= probabilities[0] <= 1.0

    # Verify that the saved threshold can produce a prediction.
    prediction = (probabilities >= threshold).astype(int)

    assert prediction.shape == (1,)
    assert prediction[0] in (0, 1)



def test_saved_random_forest_works_with_prediction_module():
    artifact = load_model(MODEL_PATH)

    transaction = {
        column: 0.0
        for column in artifact["feature_columns"]
    }

    if "Amount" in transaction:
        transaction["Amount"] = 100.0

    result = predict_transaction(transaction, artifact)

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["is_fraud"] in (0, 1)
    assert result["threshold"] == float(artifact["threshold"])

import numpy as np
import pytest

from fraud_detection.models.predict import predict_transaction

class FixedProbabilityModel:
    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def test_predict_transaction_returns_probability_and_classification():
    artifact = {
        "model": FixedProbabilityModel(),
        "feature_columns": ["Amount", "V1"],
        "threshold": 0.5,
    }

    transaction = {
        "Amount": 100.0,
        "V1": 0.2,
        "UnexpectedFeature": 999.0,
    }

    result = predict_transaction(transaction, artifact)

    assert result["fraud_probability"] == 0.8
    assert result["is_fraud"] == 1
    assert result["threshold"] == 0.5


def test_predict_transaction_classifies_below_threshold():
    artifact = {
        "model": FixedProbabilityModel(),
        "feature_columns": ["Amount", "V1"],
        "threshold": 0.9,
    }

    transaction = {"Amount": 100.0, "V1": 0.2}

    result = predict_transaction(transaction, artifact)

    assert result["is_fraud"] == 0


def test_predict_transaction_fails_when_feature_is_missing():
    artifact = {
        "model": FixedProbabilityModel(),
        "feature_columns": ["Amount", "V1"],
        "threshold": 0.5,
    }

    transaction = {"Amount": 100.0}

    with pytest.raises(ValueError, match="missing required features"):
        predict_transaction(transaction, artifact)

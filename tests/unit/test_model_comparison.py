
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from fraud_detection.models.compare_models import evaluate_model


def test_evaluate_model_calculates_metrics_and_costs():
    # Four transactions: two legitimate and two fraudulent.
    y_test = pd.Series([0, 0, 1, 1])

    X_test = pd.DataFrame(
        {
            "Amount": [10.0, 100.0, 50.0, 200.0],
            "V1": [0.1, 0.2, 0.3, 0.4],
        }
    )

    amounts = X_test["Amount"]

    # Fixed probabilities make the test deterministic.
    class FixedProbabilityModel:
        def predict_proba(self, X):
            return np.array(
                [
                    [0.9, 0.1],  # legitimate, correctly classified
                    [0.1, 0.9],  # legitimate, false positive
                    [0.1, 0.9],  # fraud, correctly classified
                    [0.9, 0.1],  # fraud, false negative
                ]
            )

    artifact = {
        "model": FixedProbabilityModel(),
        "threshold": 0.5,
        "feature_columns": ["Amount", "V1"],
    }

    result = evaluate_model(
        name="Test Model",
        artifact=artifact,
        X_test=X_test,
        y_test=y_test,
        amounts=amounts,
    )

    assert result["TP"] == 1
    assert result["FP"] == 1
    assert result["FN"] == 1
    assert result["TN"] == 1

    assert result["precision"] == 0.5
    assert result["recall"] == 0.5

    # Missed fraud: max(€200, €20) = €200.
    assert result["cost_fn_eur"] == 200.0

    # False alarm: max(10% of €100, €5) = €10.
    assert result["cost_fp_eur"] == 10.0

    assert result["total_cost_eur"] == 210.0


def test_evaluate_model_respects_saved_threshold():
    y_test = pd.Series([0, 1])
    X_test = pd.DataFrame(
        {
            "Amount": [50.0, 100.0],
            "V1": [0.1, 0.2],
        }
    )

    class FixedProbabilityModel:
        def predict_proba(self, X):
            return np.array(
                [
                    [0.6, 0.4],
                    [0.6, 0.4],
                ]
            )

    artifact = {
        "model": FixedProbabilityModel(),
        "threshold": 0.3,
        "feature_columns": ["Amount", "V1"],
    }

    result = evaluate_model(
        name="Test Model",
        artifact=artifact,
        X_test=X_test,
        y_test=y_test,
        amounts=X_test["Amount"],
    )

    # Both probabilities of fraud (0.4) exceed the saved threshold.
    assert result["TP"] == 1
    assert result["FP"] == 1
    assert result["FN"] == 0
    assert result["TN"] == 0
    
    
def test_evaluate_model_handles_no_positive_predictions():
    y_test = pd.Series([0, 1])
    X_test = pd.DataFrame(
        {
            "Amount": [50.0, 100.0],
            "V1": [0.1, 0.2],
        }
    )

    class FixedProbabilityModel:
        def predict_proba(self, X):
            return np.array([
                [0.8, 0.2],
                [0.9, 0.1],
            ])

    artifact = {
        "model": FixedProbabilityModel(),
        "threshold": 0.5,
        "feature_columns": ["Amount", "V1"],
    }

    result = evaluate_model(
        name="Test Model",
        artifact=artifact,
        X_test=X_test,
        y_test=y_test,
        amounts=X_test["Amount"],
    )

    assert result["TP"] == 0
    assert result["FP"] == 0
    assert result["FN"] == 1
    assert result["TN"] == 1

    # Nessuna frode rilevata: precision e recall sono zero.
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0

    # La frode mancata costa max(100, 20) = 100 euro.
    assert result["cost_fn_eur"] == 100.0
    assert result["cost_fp_eur"] == 0.0
    assert result["total_cost_eur"] == 100.0


def test_evaluate_model_handles_all_positive_predictions():
    y_test = pd.Series([0, 1])
    X_test = pd.DataFrame(
        {
            "Amount": [50.0, 100.0],
            "V1": [0.1, 0.2],
        }
    )

    class FixedProbabilityModel:
        def predict_proba(self, X):
            return np.array([
                [0.2, 0.8],
                [0.1, 0.9],
            ])

    artifact = {
        "model": FixedProbabilityModel(),
        "threshold": 0.5,
        "feature_columns": ["Amount", "V1"],
    }

    result = evaluate_model(
        name="Test Model",
        artifact=artifact,
        X_test=X_test,
        y_test=y_test,
        amounts=X_test["Amount"],
    )

    assert result["TP"] == 1
    assert result["FP"] == 1
    assert result["FN"] == 0
    assert result["TN"] == 0

    assert result["precision"] == 0.5
    assert result["recall"] == 1.0

    # Falso allarme: max(50 * 10%, 5) = 5 euro.
    assert result["cost_fn_eur"] == 0.0
    assert result["cost_fp_eur"] == 5.0
    assert result["total_cost_eur"] == 5.0


def test_evaluate_model_classifies_probability_equal_to_threshold():
    y_test = pd.Series([0, 1])
    X_test = pd.DataFrame(
        {
            "Amount": [50.0, 100.0],
            "V1": [0.1, 0.2],
        }
    )

    class FixedProbabilityModel:
        def predict_proba(self, X):
            return np.array([
                [0.5, 0.5],
                [0.5, 0.5],
            ])

    artifact = {
        "model": FixedProbabilityModel(),
        "threshold": 0.5,
        "feature_columns": ["Amount", "V1"],
    }

    result = evaluate_model(
        name="Test Model",
        artifact=artifact,
        X_test=X_test,
        y_test=y_test,
        amounts=X_test["Amount"],
    )

    # La funzione usa >=: anche una probabilità uguale
    # alla soglia viene classificata come frode.
    assert result["TP"] == 1
    assert result["FP"] == 1
    assert result["FN"] == 0
    assert result["TN"] == 0
    
def test_evaluate_model_uses_saved_feature_columns():
    y_test = pd.Series([0, 1])

    X_test = pd.DataFrame(
        {
            "Amount": [50.0, 100.0],
            "V1": [0.1, 0.2],
            "UnexpectedFeature": [999.0, 999.0],
        }
    )

    class FeatureCheckingModel:
        def predict_proba(self, X):
            assert list(X.columns) == ["Amount", "V1"]

            return np.array(
                [
                    [0.9, 0.1],
                    [0.1, 0.9],
                ]
            )

    artifact = {
        "model": FeatureCheckingModel(),
        "threshold": 0.5,
        "feature_columns": ["Amount", "V1"],
    }

    result = evaluate_model(
        name="Test Model",
        artifact=artifact,
        X_test=X_test,
        y_test=y_test,
        amounts=X_test["Amount"],
    )

    assert result["TP"] == 1
    assert result["FP"] == 0
    assert result["FN"] == 0
    assert result["TN"] == 1
    

def test_evaluate_model_fails_when_feature_is_missing():
    y_test = pd.Series([0, 1])

    X_test = pd.DataFrame(
        {
            "Amount": [50.0, 100.0],
        }
    )

    class DummyModel:
        def predict_proba(self, X):
            raise AssertionError(
                "The model should not be called when a feature is missing"
            )

    artifact = {
        "model": DummyModel(),
        "threshold": 0.5,
        "feature_columns": ["Amount", "V1"],
    }

    with pytest.raises(KeyError):
        evaluate_model(
            name="Test Model",
            artifact=artifact,
            X_test=X_test,
            y_test=y_test,
            amounts=X_test["Amount"],
        )

from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FN_MIN_COST = 20.0
FP_MIN_COST = 5.0
FP_AMOUNT_RATE = 0.10


def evaluate_thresholds(
    y_true: pd.Series,
    y_prob,
    amounts: pd.Series,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        # False negatives: missed frauds.
        missed_frauds = (y_true == 1) & (y_pred == 0)
        cost_fn = (
            amounts.loc[missed_frauds]
            .clip(lower=FN_MIN_COST)
            .sum()
        )

        # False positives: legitimate transactions flagged as fraud.
        false_alarms = (y_true == 0) & (y_pred == 1)
        cost_fp = (
            amounts.loc[false_alarms]
            .mul(FP_AMOUNT_RATE)
            .clip(lower=FP_MIN_COST)
            .sum()
        )

        total_cost = cost_fn + cost_fp

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_true, y_pred, zero_division=0
                ),
                "recall": recall_score(
                    y_true, y_pred, zero_division=0
                ),
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "cost_fn_eur": cost_fn,
                "cost_fp_eur": cost_fp,
                "cost_eur": total_cost,
            }
        )

    return pd.DataFrame(rows)

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = project_root / "data" / "raw" / "creditcard.csv"
    reports_dir = project_root / "reports"
    models_dir = project_root / "models"

    reports_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    df = pd.read_csv(dataset_path).sort_values("Time").reset_index(drop=True)

    # Chronological split: 60% train, 20% validation, 20% test.
    n = len(df)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    train_df = df.iloc[:train_end]
    validation_df = df.iloc[train_end:validation_end]
    test_df = df.iloc[validation_end:]

    feature_columns = [
        column for column in df.columns if column not in ["Class", "Time"]
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["Class"]

    X_validation = validation_df[feature_columns]
    y_validation = validation_df["Class"]

    X_test = test_df[feature_columns]
    y_test = test_df["Class"]

    print("--- DATA SPLIT ---")
    for name, subset in [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]:
        print(
            f"{name}: {len(subset)} transactions, "
            f"{int(subset['Class'].sum())} frauds"
        )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=42),
    )
    model.fit(X_train, y_train)

    # Select a threshold using validation data only.
    validation_prob = model.predict_proba(X_validation)[:, 1]
    thresholds = np.round(np.arange(0.01, 0.51, 0.01), 2).tolist()
    
    validation_results = evaluate_thresholds(
        y_validation,
        validation_prob,
        validation_df["Amount"],
        thresholds,
)
    

    print("\n--- VALIDATION: THRESHOLD COMPARISON ---")
    print(validation_results.round(4).to_string(index=False))

    # Example policy: maximize recall while keeping precision >= 50%.

# Select the threshold with the lowest estimated economic cost.
    best_row = validation_results.loc[
    validation_results["cost_eur"].idxmin()
]
    best_threshold = float(best_row["threshold"])

    print("\n--- 10 LOWEST-COST THRESHOLDS ---")
    print(
        validation_results.nsmallest(10, "cost_eur")
        .round(4)
        .to_string(index=False)
)

    print(f"\nSelected threshold: {best_threshold:.2f}")
    print(
        f"Estimated validation cost: "
        f"{best_row['cost_eur']:.2f} €"
)
    print(f"\nSelected threshold: {best_threshold:.2f}")

    # Final evaluation: the test set is not used to select the threshold.
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    print("\n--- FINAL TEST RESULTS ---")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_test, test_prob):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, test_prob):.4f}")
    
    false_alarms = (y_test == 0) & (test_pred == 1)

    fp_details = X_test.loc[false_alarms, ["Amount"]].copy()
    fp_details["predicted_probability"] = test_prob[false_alarms]
    fp_details["estimated_cost_eur"] = (
        fp_details["Amount"]
        .mul(FP_AMOUNT_RATE)
        .clip(lower=FP_MIN_COST)
)
    
    
    # False negatives: missed frauds.
    missed_frauds = (y_test == 1) & (test_pred == 0)

    test_cost_fn = (
        X_test.loc[missed_frauds, "Amount"]
        .clip(lower=FN_MIN_COST)
        .sum()
    )

    # False positives: legitimate transactions flagged as fraud.
    false_alarms = (y_test == 0) & (test_pred == 1)

    test_cost_fp = (
        X_test.loc[false_alarms, "Amount"]
        .mul(FP_AMOUNT_RATE)
        .clip(lower=FP_MIN_COST)
        .sum()
    )

    test_cost = test_cost_fn + test_cost_fp   
    print(f"True positives (frauds found): {tp}")
    print(f"False positives (false alarms): {fp}")
    print(f"False negatives (missed frauds): {fn}")
    print(f"True negatives: {tn}")
    print(f"Estimated economic cost: {test_cost:.2f} €")
    print(f"Estimated missed-fraud cost: {test_cost_fn:.2f} €")
    print(f"Estimated false-alarm cost:  {test_cost_fp:.2f} €")
    print(f"Estimated total cost:       {test_cost:.2f} €")
    # Save model and selected threshold for later reuse.
    model_path = models_dir / "logistic_regression.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": best_threshold,
        },
        model_path,
    )

    validation_results.to_csv(
        reports_dir / "threshold_comparison.csv", index=False
    )

    print(f"\nModel saved to: {model_path}")
    print(
        "Threshold comparison saved to: "
        f"{reports_dir / 'threshold_comparison.csv'}"
    )


if __name__ == "__main__":
    main()
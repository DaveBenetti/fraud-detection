
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

FN_MIN_COST = 20.0
FP_MIN_COST = 5.0
FP_AMOUNT_RATE = 0.10


def evaluate_model(
    name: str,
    artifact: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    amounts: pd.Series,
) -> dict:
    """Evaluate a saved model on the common test set."""
    model = artifact["model"]
    threshold = float(artifact["threshold"])
    feature_columns = artifact["feature_columns"]

    X = X_test[feature_columns]
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    missed_frauds = (y_test == 1) & (predictions == 0)
    false_alarms = (y_test == 0) & (predictions == 1)

    cost_fn = amounts.loc[missed_frauds].clip(
        lower=FN_MIN_COST
    ).sum()

    cost_fp = amounts.loc[false_alarms].mul(
        FP_AMOUNT_RATE
    ).clip(lower=FP_MIN_COST).sum()

    return {
        "model": name,
        "threshold": threshold,
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "pr_auc": average_precision_score(
            y_test, probabilities
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilities
        ),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "cost_fn_eur": float(cost_fn),
        "cost_fp_eur": float(cost_fp),
        "total_cost_eur": float(cost_fn + cost_fp),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "data" / "raw" / "creditcard.csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"

    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load the previously trained models and their thresholds.
    model_paths = {
        "Logistic Regression": (
            models_dir / "logistic_regression.joblib"
        ),
        "Random Forest": (
            models_dir / "random_forest.joblib"
        ),
    }

    artifacts = {}
    for name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model artifact: {path}"
            )
        artifacts[name] = joblib.load(path)

    # Reproduce the exact chronological split used in training.
    df = pd.read_csv(data_path)
    df = df.sort_values("Time").reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    test_df = df.iloc[validation_end:]

    y_test = test_df["Class"]
    amounts = test_df["Amount"]

    excluded_columns = {"Class", "Time"}
    feature_columns = [
        col for col in df.columns
        if col not in excluded_columns
    ]
    X_test = test_df[feature_columns]

    print("--- COMMON TEST SET ---")
    print(f"Transactions: {len(test_df)}")
    print(f"Frauds:       {int(y_test.sum())}")

    results = []
    for name, artifact in artifacts.items():
        result = evaluate_model(
            name=name,
            artifact=artifact,
            X_test=X_test,
            y_test=y_test,
            amounts=amounts,
        )
        results.append(result)

    comparison = pd.DataFrame(results).sort_values(
        "total_cost_eur"
    ).reset_index(drop=True)

    # Save the complete machine-readable comparison.
    csv_path = reports_dir / "model_comparison.csv"
    comparison.to_csv(csv_path, index=False)

    # Create a readable Markdown report.
    display_columns = [
        "model",
        "threshold",
        "precision",
        "recall",
        "pr_auc",
        "roc_auc",
        "TP",
        "FP",
        "FN",
        "TN",
        "cost_fn_eur",
        "cost_fp_eur",
        "total_cost_eur",
    ]

    display_df = comparison[display_columns].copy()

    metric_columns = [
        "threshold", "precision", "recall",
        "pr_auc", "roc_auc",
    ]
    cost_columns = [
        "cost_fn_eur", "cost_fp_eur", "total_cost_eur",
    ]

    for column in metric_columns:
        display_df[column] = display_df[column].map(
            lambda value: f"{value:.4f}"
        )

    for column in cost_columns:
        display_df[column] = display_df[column].map(
            lambda value: f"€{value:,.2f}"
        )

    markdown_table = (
        "| " + " | ".join(display_df.columns) + " |\n"
        + "| " + " | ".join(["---"] * len(display_df.columns))
        + " |\n"
    )

    for _, row in display_df.iterrows():
        markdown_table += (
            "| " + " | ".join(str(value) for value in row)
            + " |\n"
        )

    best = comparison.iloc[0]
    report = f"""# Fraud Detection — Model Comparison

## Evaluation setup

- Test transactions: {len(test_df)}
- Test frauds: {int(y_test.sum())}
- Split: chronological 60% training / 20% validation / 20% test
- Thresholds: selected previously using validation data
- Test set: used only for final evaluation
- Cost assumptions: FN = max(Amount, €{FN_MIN_COST:.2f});
  FP = max({FP_AMOUNT_RATE:.0%} of Amount, €{FP_MIN_COST:.2f})

## Results

{markdown_table}

## Lowest estimated cost

**{best['model']}** with estimated total cost
**€{best['total_cost_eur']:,.2f}** on the test set.

These costs are illustrative estimates based on the project's
assumptions, not verified real-world banking losses.

## Interpretation

Compare the estimated cost alongside precision, recall,
PR-AUC and the number of false negatives. The lowest-cost
model under these assumptions is not automatically the best
choice under every operational objective.

The thresholds were not retuned on the test set.
"""

    report_path = reports_dir / "model_comparison.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n--- MODEL COMPARISON ---")
    print(
        comparison[display_columns]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nCSV report:      {csv_path}")
    print(f"Markdown report: {report_path}")


if __name__ == "__main__":
    main()
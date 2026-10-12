# Fraud Detection — Model Comparison

## Evaluation setup

- Test transactions: 56962
- Test frauds: 75
- Split: chronological 60% training / 20% validation / 20% test
- Thresholds: selected previously using validation data
- Test set: used only for final evaluation
- Cost assumptions: FN = max(Amount, €20.00);
  FP = max(10% of Amount, €5.00)

## Results

| model | threshold | precision | recall | pr_auc | roc_auc | TP | FP | FN | TN | cost_fn_eur | cost_fp_eur | total_cost_eur |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.4700 | 0.8507 | 0.7600 | 0.8097 | 0.9779 | 57 | 10 | 18 | 56877 | €2,796.14 | €150.93 | €2,947.07 |
| Logistic Regression | 0.2100 | 0.7273 | 0.6400 | 0.6964 | 0.9730 | 48 | 18 | 27 | 56869 | €4,564.06 | €2,654.12 | €7,218.18 |


## Lowest estimated cost

**Random Forest** with estimated total cost
**€2,947.07** on the test set.

These costs are illustrative estimates based on the project's
assumptions, not verified real-world banking losses.

## Interpretation

Compare the estimated cost alongside precision, recall,
PR-AUC and the number of false negatives. The lowest-cost
model under these assumptions is not automatically the best
choice under every operational objective.

The thresholds were not retuned on the test set.

## Dataset
UCI **APS Failure**: Train 60k rows (1k positives), 170 numeric features + label; Test set provided separately.

## Preprocessing
- **Imputation** for missing values (documented method).
- Compute **CV** per feature; visualize top \(\lfloor \sqrt{170} \rfloor = 13\) for intuition (scatter/box plots).
- Correlation heatmap to spot redundancies.
- Quantify imbalance (positives vs negatives).

## Models & Experiments
### 1) Random Forest (uncompensated)
- Fit baseline RF; collect **confusion matrix**, **ROC**, **AUC**, **misclassification** on train/test.
- Report **OOB error** and compare to test error.

### 2) Random Forest (imbalanced-aware)
- Strategies: `class_weight='balanced'`, balanced subsampling, or threshold tuning.
- Repeat metrics; analyze precision–recall trade-offs.

### 3) XGBoost model trees
- Use logistic objective; treat inner nodes as linear separators (effective model-tree behavior via tree + linear booster if configured).
- Cross-validate **regularization** (\(\alpha\)), depth, learning rate.
- Estimate generalization via 5/10-fold or LOO where feasible; compare CV vs test.

### 4) SMOTE pipeline
- Apply **SMOTE** (training data only; within CV folds to avoid leakage).
- Re-train XGBoost; repeat evaluation; compare with uncompensated runs.

## Tech used
`pandas`, `numpy`, `scikit-learn` (RandomForest, metrics), `xgboost`, `imbalanced-learn` (SMOTE), `matplotlib`.

## ML tasks
**Imputation**, **class-imbalance mitigation**, **ensemble learning**, **cross-validation correctness**, **threshold analysis**.

## Notes
- Clearly separate **train-only resampling** from test evaluation.  
- RF OOB is a useful proxy but can be optimistic when heavy tuning is done—prefer held-out test confirmation.

## Task
Binary and multiclass classification of AReM activities using **time-domain features** engineered in HW3.

## Binary: bending vs others
1) **Visualization**  
Scatter plots for the 3 selected features from time series 1, 2, and 6; color: bending vs others.

2) **Temporal tiling**  
Split each of the 6 series into \(l \in \{1,2,\dots,20\}\) equal segments → feature set size scales by \(l\). Repeat (1) for \(l=2\) and discuss visual changes.

3) **Logistic regression (unregularized)**
- For each \(l\): fit LR on train, compute **p-values** (via `statsmodels.Logit`), prune non-significant features, **refit**.
- **Feature selection alternative**: sklearn **RFE** to select top-p features.
- **Validation**: **5-fold stratified CV** over \((l, p)\). Use *right way* CV: feature selection **within** folds (no leakage).

4) **Evaluation**
- Train-set confusion matrix; **ROC** & **AUC**.
- Coefficients \(\beta_i\) with p-values for interpretability.
- Test-set accuracy vs CV estimate; stability notes; class separability/instability diagnosis.

5) **Case-control sampling** (if imbalance observed)
- Refit LR under case-control sampling; adjust intercept; compare confusion/ROC/AUC.

## Regularized LR (L1)
- **L1-penalized logistic regression** (Liblinear/Saga), standardized features.
- Joint selection of \(l\) and \(\lambda\) via stratified 5-fold CV (package CV for \(\lambda\)); compare to p-value pruning: performance and ease.

## Multiclass
- **Multinomial LR** with L1; select \(l\) as above; report test error; macro/micro F1; multiclass ROC (one-vs-rest) if produced.
- **Naive Bayes** baselines (Gaussian, Multinomial); compare.

## Tech used
`pandas`, `numpy`, `statsmodels`, `scikit-learn` (LogisticRegression, RFE, metrics), `matplotlib`, `seaborn`.

## ML tasks
Feature-based time-series **classification**, **model selection** across feature granularity \(l\) and sparsity, **proper CV**, **calibration/diagnostics**.

## Notes
- Guard against leakage when segmenting series and re-computing features: apply the **same transform to test** with the chosen \(l\).

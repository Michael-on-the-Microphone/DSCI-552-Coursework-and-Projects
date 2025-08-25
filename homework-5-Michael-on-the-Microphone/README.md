## Part 1 — Decision Trees as Interpretable Models
**Dataset**: UCI Acute Inflammations (multi-label).  
**Steps**:
- Train a full decision tree (multi-label via label powerset or binary relevance per label).
- Visualize the tree; extract **IF–THEN** rules (path conditions → predictions).
- Apply **cost-complexity pruning** (ccp-alpha via CV) to derive a **minimal** yet high-fidelity subtree and corresponding compact rule set.

**Deliverables**: tree plot, post-pruning depth/leaf stats, rule list with coverage and accuracy per rule.

## Part 2 — Regularization & Boosting for Regression
**Dataset**: UCI Communities & Crime. Train: first 1,495 rows; Test: remainder.

**Preprocessing**
- Handle missing values via imputation (document strategy). Drop non-predictive features as per data description.
- Correlation matrix for multicollinearity scan.
- Compute **Coefficient of Variation** \(CV = s/m\) per feature; visualize top \(\lfloor \sqrt{128} \rfloor = 11\) features (scatter + box plots).

**Models**
- **OLS** baseline: report test error.
- **RidgeCV**: cross-validated \(\lambda\); test error.
- **LASSO**: cross-validated \(\lambda\); report test error and selected variables. Repeat with/without standardization (note: dataset says already normalized).
- **PCR**: choose number of PCs \(M\) by CV; test error.

**Boosted model trees**
- **XGBoost** for regression; enable L1 penalty on leaf models (proxy for model trees with sparsity). Cross-validate regularization (\(\alpha\)) and tree depth/learning rate.
- Compare against linear baselines; discuss bias–variance and nonlinearity capture.

## Tech used
`pandas`, `numpy`, `scikit-learn` (DecisionTree, RidgeCV, LassoCV, PCA, Pipeline), `xgboost`, plotting libs.

## ML tasks
Interpretable **rule extraction**, **pruning**, regression with **regularization** (L2/L1), **PCR**, **gradient boosting**.

## Notes
- For multi-label trees, document the transformation strategy (powerset vs binary relevance) and its implications for interpretability.

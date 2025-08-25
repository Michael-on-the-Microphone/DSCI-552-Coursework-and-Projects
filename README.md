# DSCI-552-Coursework-and-Projects

> USC DSCI 552 • Summer 2025 • Instructor: M. R. Rajati

This monorepo aggregates all coursework (9 homeworks + 1 final project). Each subfolder contains a self-contained report-style README that documents the dataset(s), techniques, ML tasks, modeling choices, and evaluation protocols used.

## At a glance

| Module | Topic focus | Primary datasets | Core methods |
|---|---|---|---|
| HW0 | Onboarding, Pandas/Numpy/Matplotlib/Seaborn/Scikit-learn, Git | Salaries.csv | Data I/O, indexing, filtering, `describe()`, vectorization, plotting, sklearn API survey |
| HW1 | k-Nearest Neighbors, distance metrics, learning curves | UCI Vertebral Column | KNN (Euclidean/Manhattan/Minkowski/Chebyshev/Mahalanobis), weighted voting, confusion metrics |
| HW2 | Linear vs Polynomial vs Interactions vs KNN on CCPP | UCI CCPP (Sheet 1) | Univariate & multivariate LR, cubic per-feature tests, pairwise interactions with hierarchical selection, 70/30 holdout MSE, KNN regression with/without scaling, error vs 1/k |
| HW3 | Time-series feature extraction | UCI AReM | Time-domain features, bootstrap CIs, feature screening |
| HW4 | Time-series classification | UCI AReM | Logistic regression, RFE, stratified CV, L1-penalized LR, multinomial LR, Naive Bayes |
| HW5 | Interpretable trees; regularization & boosting for regression | UCI Acute Inflammations; Communities & Crime | Decision trees, rule extraction, cost-complexity pruning, OLS, Ridge, LASSO, PCR, XGBoost |
| HW6 | Imbalanced learning with tree ensembles | UCI APS Failure | Imputation, RF (OOB), class weighting, XGBoost (model trees), SMOTE |
| HW7 | Multi-label SVMs; unsupervised clustering evaluation | UCI Anuran Calls (MFCCs) | One-vs-rest SVM (RBF, L1-linear), SMOTE, k-means model selection, Hamming/Exact-match |
| HW8 | Supervised vs semi-supervised vs unsupervised; Active Learning | UCI Breast Cancer (Diag); Banknote | L1-SVM, self-training, k-means, spectral clustering, active vs passive SVM learning curves |
| Final | Transfer learning for image classification | Waste images (9 classes) | ResNet50/ResNet100/EfficientNetB0/VGG16, frozen bases + head, augmentation, early stopping |

## Tech stack

- **Python** 3.11
- **Core**: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`
- **ML**: `scikit-learn`, `statsmodels` (LR p-values), `xgboost`, `imbalanced-learn` (SMOTE)
- **DL** (Final): `tensorflow`/`keras`, `opencv-python` (I/O & resizing)
- **Utilities**: `bootstrapped`/custom bootstrap, `tqdm`

## Evaluation playbook

- Classification: accuracy, precision, recall, F1, ROC–AUC, confusion matrix; stratified splits/CV where appropriate.
- Regression: MSE/RMSE/MAE; cross-validated hyperparameters.
- Model selection: grid search over problem-specific ranges; nested or repeated CV when selecting both features and hyperparameters.
- Learning curves and Monte-Carlo repeats to dampen variance.

See each sub-README for dataset-specific notes, artifacts, and lessons learned.


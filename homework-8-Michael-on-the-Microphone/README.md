## Part 1 — Breast Cancer (Diagnostic) — M=30 Monte-Carlo runs
**Splits**: per run, keep 20% of both classes as test; remaining for training.

### A) Supervised (L1-SVM)
- Standardize; 5-fold CV for penalty.
- Report **average** train/test: accuracy, precision, recall, F1, AUC; plot ROC & confusion matrix for a representative run.

### B) Semi-Supervised (Self-training)
- Within training set, mark **50% of each class** as labeled; remainder unlabeled.
- Train L1-SVM; iteratively add the unlabeled point farthest from the margin (highest confidence) with its predicted label; retrain until exhausted.
- Aggregate the same metrics across 30 runs.

### C) Unsupervised
1) **k-means (k=2)**  
   - Multiple random inits; guard against local minima (best inertia/consensus).
   - Label clusters via majority among **30 nearest** points to each center.
   - Compute metrics on train and test (test by proximity to centers).

2) **Spectral clustering (RBF)**
   - Choose gamma=1 or tune to match class proportions; label clusters with cluster-member majorities; evaluate as above using `fit_predict`.

### Comparison
- Summarize performance: supervised > semi-supervised > unsupervised expected; discuss cases where semi-supervised narrows the gap.

## Part 2 — Active Learning on Banknote Authentication
**Setup**: 472 test; 900 train.

- **Passive learner**: start with 10 random training points; increment by 10 without replacement to 900; 5-fold CV for linear L1-SVM at each step.
- **Active learner**: same start; at each step, **query** 10 training points **closest to the current hyperplane**; retrain; continue to 900.

**Monte-Carlo**: 50 repetitions each; plot **average test error vs training size** for passive vs active to obtain learning curves.

## Tech used
`scikit-learn` (LinearSVC/SVC, StandardScaler, kmeans, SpectralClustering, metrics), `numpy`, `pandas`, plotting libs.

## ML tasks
**Semi-supervised self-training**, **unsupervised labeling**, **active learning** with **learning curves** and repeated trials.

## Notes
- For spectral clustering, clusters may be **non-convex**—avoid center-based labeling; use `fit_predict` memberships.

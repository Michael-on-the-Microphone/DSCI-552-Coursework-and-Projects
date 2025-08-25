## Dataset
UCI **Vertebral Column**. Six biomechanical attributes; labels collapsed to binary: Normal (0) vs Abnormal (1).

## Splits & EDA
- Training: first 70 rows of class 0 and first 140 rows of class 1; test: remainder (deterministic partition as specified).
- EDA: scatterplots and boxplots per feature, color-coded by class; check scale drift/outliers.

## Modeling
### 1) Baseline KNN (Euclidean)
- Search **k** over `{208, 205, …, 7, 4, 1}` (descending) or finer grid; majority vote.
- Track **train/test error** vs **k**; select \(k^\*\) minimizing test error.
- Report **confusion matrix**, **TPR/TNR**, **precision**, **F1** at \(k^\*\).

### 2) Learning curve (subset training)
- For \(N \in \{10,20,\dots,210\}\): use first ⌊N/3⌋ from class 0 and the remainder from class 1 (within the original training set).
- For each \(N\), scan \(k \in \{1,6,11,\dots\}\); record **best test error**; plot **best error vs N**.

### 3) Distance metrics study
- **Minkowski** \(p\): 
  - Manhattan \(p=1\) → choose \(k^\*_{\text{Manhattan}}\).
  - Sweep \(\log_{10}(p) \in \{0.1,0.2,\dots,1.0\}\) using \(k^\*_{\text{Manhattan}}\); report best \(\log_{10}(p)\).
  - Chebyshev \(p \to \infty\).
- **Mahalanobis**: covariance-aware distance; handle singular/ill-conditioned \(\Sigma\) via pseudoinverse or dimensionality reduction.
- Summarize **test errors at \(k^\*\)** for all metrics in a comparison table.

### 4) Weighted voting
- Use inverse-distance weights with Euclidean/Manhattan/Chebyshev; scan \(k \in \{1,6,11,\dots,196\}\); report best **test error** per distance.

## Evaluation & Reporting
- Primary: misclassification rate; plus confusion-derived metrics.
- Diagnostic: error vs k; learning curve; stability of \(k^\*\) across metrics.

## Tech used
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (`KNeighborsClassifier`, `DistanceMetric`, metrics).

## ML tasks
Binary classification with **KNN**, **metric learning by choice**, **model selection** over \(k\), **learning curves**, **robust metrics**.

## Notes
- Class size imbalance in partitions is intentional per spec; focus on metric sensitivity and capacity control via **k**.

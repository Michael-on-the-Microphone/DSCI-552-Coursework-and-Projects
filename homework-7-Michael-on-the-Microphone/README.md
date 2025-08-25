## Dataset
UCI **Anuran Calls (MFCCs)**. Multi-label: **Family**, **Genus**, **Species**; each with multiple classes. Random 70% train.

## Multi-label classification (Binary Relevance)
### Metrics
- **Exact match** (subset accuracy) and **Hamming score/loss** for multi-label evaluation.

### Models
1) **SVM (RBF)** one-vs-rest per label
   - Tune penalty and RBF width with 10-fold CV.
   - Run with raw vs standardized features (report both if tried).

2) **L1-penalized linear SVM**
   - Standardize features; tune penalty via 10-fold CV.
   - Compare sparsity and performance to RBF SVM.

3) **Imbalance handling**
   - Apply SMOTE or class weighting; reevaluate metrics.

*(Optional)* Classifier Chains for label-dependence modeling.

## Unsupervised clustering as labeling
- **k-means** on full dataset (no train/test split here).
- Choose \(k \in \{1,\dots,50\}\) via an internal criterion (CH, Gap, Silhouette, scree).
- In each cluster, determine majority **(Family, Genus, Species)** triplet from true labels.
- Compute average **Hamming distance**, **Hamming score**, **Hamming loss** across samples.

## Monte-Carlo protocol
Repeat clustering experiments **50 times**; report mean ± std of Hamming metrics to assess stability.

## Tech used
`scikit-learn` (SVC, LinearSVC, StandardScaler, OneVsRest, kmeans, metrics), `imbalanced-learn` (SMOTE), plotting libs.

## ML tasks
**Multi-label classification**, **kernel vs sparse linear trade-offs**, **imbalance mitigation**, **unsupervised proxy labeling** with robust metrics.

## Notes
- For SVM grids, use log-spaced \(C\) and linear-spaced \(\gamma\); constrain ranges by quick over/under-fitting sanity checks.

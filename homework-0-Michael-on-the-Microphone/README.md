## Summary
Hands-on onboarding to the DSCI 552 toolchain: data I/O & wrangling with **Pandas**, vectorization & array ops with **NumPy**, plotting with **Matplotlib/Seaborn**, a survey of **scikit-learn** estimators/metrics, and Git/GitHub hygiene.

## Data + Tasks
- **Salaries.csv** (tabular): set `playerID` as index; first row as header; skip second row.
- Queries: filter ATL/HOU players with salary > \$1M; `describe()` ATL salary (std, quartiles, median, mean, min, max).
- Convert DataFrame → dict via row iteration (`iterrows`) and compare against `to_dict()`; rebuild a DataFrame and relabel columns `a..z`.

## NumPy practice
- Construct 2-D Python lists → `ndarray`; inspect `ndim`, `shape`, `size`, `dtype`, `itemsize`, buffer `data`.
- Reshape/flatten to learn axis semantics; slices for 1-D and 2-D; reductions and ufuncs: `argmin/argmax/min/max/mean/sum/std/dot/square/sqrt/abs/exp/sign/mod`.
- Constructors: `arange/ones/zeros/eye/linspace/concatenate`.

## Scikit-learn API survey (catalog & quick smoke-tests)
- **Preprocessing**: `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `OneHotEncoder`, `train_test_split`.
- **Models**: `KNeighborsClassifier`, `LinearRegression`, `LogisticRegression/CV`, `DecisionTree{Classifier,Regressor}`, `RandomForest{Classifier,Regressor}`, `AdaBoost{Classifier,Regressor}`, `LinearSVC/LinearSVR`, `KMeans`; Multiclass wrappers `OneVsOne`, `OneVsRest`.
- **Model selection**: `LassoCV`, `RidgeCV`, CV objects (`StratifiedKFold`, `RepeatedKFold`, `LOO`, `KFold`) and scoring (`cross_validate`, `cross_val_score`).
- **Metrics**: `accuracy_score`, `precision/recall/f1_score`, `roc_auc_score`, `auc`, `hamming_loss`.

## Visualization
- **Matplotlib**: single/multi-line plots, titles/labels/grid, legends, LaTeX eqn text, subplots, axis limits, log scales, `gca()`.
- **Seaborn**: DataFrame-native plots; `lmplot`, `catplot`, `relplot`; `boxplot`; `pairplot`/`jointplot`.

## Git workflow (practiced)
Init repo; branch (`dev`) → commit; merge to `master`; create a temporary GitHub remote; push/pull; iterate with feature branch changes.

## Tech used
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, Python 3.11, Git/GitHub.

## ML tasks
EDA, data filtering, summary statistics; quick model API familiarization (no formal benchmarking).

## Notes
- Emphasis on **vectorized operations** and **DataFrame indexing** for correctness + speed.
- Early exposure to **CV objects** and **metric selection** to standardize later experiments.


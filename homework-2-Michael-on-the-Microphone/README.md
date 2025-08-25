## Dataset
**Combined Cycle Power Plant (CCPP)** — 6 years of hourly data (2006–2011) at full load.  
**Predictors**: Temperature (T), Ambient Pressure (AP), Relative Humidity (RH), Exhaust Vacuum (V).  
**Target**: Net hourly electrical energy output (EP).  
Use **Sheet 1** (others are shuffled duplicates).

## 1) Exploratory Analysis
- **Cardinality**: Report total **rows** and **columns** (incl. EP), where each row is an hour, columns are ambient variables + EP.
- **Pairwise plots**: Scatter matrix (all predictors, plus each vs EP) to assess linearity, curvature, and heteroskedasticity.
- **Summary stats**: For each variable (T, AP, RH, V, EP), compute mean, median, range, Q1, Q3, and IQR; tabulate.
- **Collinearity scan**: Predictor–predictor correlations (notably T–V) to anticipate coefficient shrinkage/sign flips in multivariate models.

## 2) Simple (Univariate) Linear Regressions
For each predictor \(X \in \{T, AP, RH, V\}\):
- Fit \( \text{EP} = \beta_0 + \beta_1 X \).
- Report \(\hat\beta_1\), standard error, **t-stat**, **p-value**, \(R^2\), residual diagnostics (QQ, residuals vs fitted).
- Visualize regression line and confidence band; flag potential outliers/high leverage points (e.g., via Cook’s distance).
- Note any **statistically significant** associations and directionality.

## 3) Multiple Linear Regression (MLR)
Fit \( \text{EP} = \beta_0 + \beta_T T + \beta_{AP} AP + \beta_{RH} RH + \beta_V V \).
- Report coefficients, standard errors, **p-values**, \(R^2\)/Adj-\(R^2\).
- Compare **univariate** vs **multivariate** significance: which predictors remain significant after controlling for others?
- **Coefficient comparison plot**: x-axis = univariate \(\hat\beta_1\); y-axis = multivariate \(\hat\beta\) (one point per predictor). Discuss attenuation or sign changes due to collinearity.

## 4) Nonlinearity Check (Per-Predictor Cubics)
For each predictor \(X\), fit cubic:
\[
\text{EP} = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3
\]
- Use polynomial basis (degree=3) without interaction.
- Test \(\beta_2\) and \(\beta_3\) (jointly and individually) for significance; visualize partial dependence (EP vs X with cubic fit).
- Record cases where curvature improves fit materially (AIC/BIC or CV-MSE).

## 5) Pairwise Interactions
- Fit full linear model with all **pairwise interactions** (T·AP, T·RH, T·V, AP·RH, AP·V, RH·V) plus all main effects.
- Identify significant interaction terms (hierarchical principle: retain main effects for any significant interaction).
- Inspect VIFs and condition indices for instability.

## 6) Model Improvement via Nonlinearity + Interactions (Holdout Evaluation)
- **Split**: Random **70% train / 30% test** (seeded), stratification not needed for regression.
- **Model A (Linear)**: MLR with all four predictors (main effects only).
- **Model B (Enhanced)**: Main effects + all pairwise interactions + quadratic terms \(X^2\) for each predictor.  
  - Perform **p-value-based backward elimination** (retain hierarchy: if an interaction survives, keep its mains).
- **Metrics**: Report **train MSE** and **test MSE** for both models; include Adj-\(R^2\) (train) and residual diagnostics on train; note any generalization gap.

## 7) KNN Regression
- Fit **KNNRegressor** with \(k \in \{1,2,\dots,100\}\).
- Run **two pipelines**: (a) raw features, (b) standardized features (`StandardScaler`).
- For each \(k\), compute **train MSE** and **test MSE**; plot errors vs **1/k**.
- Select best \(k\) per pipeline by **lowest test MSE**; compare standardized vs raw (distance sensitivity).

## 8) Comparative Analysis
- Compare best **linear family** model (A or B) to **best KNN** (standardized vs raw).
- Discuss **bias–variance trade-off**, nonlinearity capture, and stability to noise.  
  Typical outcomes: linear models excel with additive structure; KNN can win if strong local nonlinearities exist but may overfit at low \(k\) and degrade without scaling.

## Artifacts
- `eda_summary.csv` (stats table), correlation heatmap.
- Coefficient comparison scatter plot (univariate vs multivariate).
- Cubic fits per predictor (plots).
- Interaction coefficients table with p-values.
- Learning curves for KNN (MSE vs 1/k).  
- Final report table: Train/Test MSE for Model A, Model B, KNN(raw), KNN(scaled).

## Tech used
`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (`LinearRegression`, `PolynomialFeatures`, `KNeighborsRegressor`, `StandardScaler`), `statsmodels` (for coefficient inference).

## ML tasks
Univariate & multivariate **linear regression**, **nonlinearity detection** (cubic per feature), **interaction modeling**, **feature scaling**, **KNN regression**, **model selection** via holdout MSE and inference.

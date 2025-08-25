## Dataset
UCI **Activity Recognition system based on Multisensor (AReM)**: 7 activities; 88 instances; each instance has **6 sensor time series** with **480** samples.

## Protocol
- **Train/Test split**: keep bending1/2 datasets 1–2 and other folders’ datasets 1–3 as **test**; remaining instances as **train**.
- **Focus**: *time-domain* feature extraction; no classifiers here (used in HW4).

## Feature extraction
For each of 6 series per instance:
- **Statistics**: min, max, mean, median, std, 1st quartile (Q1), 3rd quartile (Q3).
- **(Optional)** standardize or keep in native scale (record choice for HW4 consistency).

The resulting design matrix has one row per instance and 6×7 = **42** features.

## Uncertainty quantification
- For **each feature**, estimate **standard deviation** and compute **90% bootstrap CI** for the std (resampling instances with replacement).
- Report CI widths to gauge feature stability across activities.

## Feature screening
- From inspection (variance, separability), choose **three** most informative time-domain features (e.g., {min, mean, max}) as a compact subset for HW4 visualization & early models.

## Artifacts
- `features.csv` / `features.parquet` with 42 columns.
- `bootstrap_std_ci.csv` with lower/upper bounds per feature.
- Plots: per-feature distributions by activity; correlation heatmap of features.

## Tech used
`pandas`, `numpy`, `scipy.stats`/custom bootstrap, `matplotlib`, `seaborn`.

## ML tasks
Feature engineering for time series; **resampling-based uncertainty**; **feature selection (screening)**.

## Notes
- Preserve the **exact same preprocessing and feature definitions** for HW4 to avoid leakage/mismatch.

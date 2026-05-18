# Startup Success Prediction Using Machine Learning

## Overview

This repository contains the code and outputs for my MSc Data Science thesis, **Predicting Startup Success and Failure Using Machine Learning**.

The thesis investigates binary startup outcome prediction using machine learning. The prediction task distinguishes startups that were ultimately **acquired** from startups that were ultimately **closed**. The analysis compares Logistic Regression, XGBoost, and CatBoost, and further evaluates the selected XGBoost model through feature importance analysis, feature-group ablation, subgroup error analysis, and validation-strategy comparison.

## Dataset

The analysis uses the publicly available Kaggle [Startup Investments (Crunchbase)](https://www.kaggle.com/datasets/arindam235/startup-investments-crunchbase) dataset, which is derived from Crunchbase and released under a CC0 Public Domain license.

The raw dataset is included in:

```text
data/raw/investments_VC.csv
```

The processed and interim datasets are generated from the notebooks and are included to make the analysis easier to inspect and reproduce.

## Language and tools

The analysis was implemented in **Python 3.11.14** using Jupyter notebooks.

The main libraries used were:

```text
pandas 3.0.0
NumPy 2.4.2
scikit-learn 1.8.0
XGBoost 3.2.0
CatBoost 1.2.10
SHAP 0.51.0
Matplotlib 3.10.8
Seaborn 0.13.2
SciPy 1.17.0
Joblib 1.5.3
IPython 9.7.0
```

Package requirements are listed in:

```text
requirements.txt
```

## Repository structure

```text
data/
├── raw/           # Raw dataset
├── processed/     # Preprocessed modeling dataset
└── interim/       # Temporal train-test split files

notebooks/
├── 01_exploratory_data_analysis.ipynb
├── 02_preprocessing_feature_engineering.ipynb
├── 03_feature_diagnostics.ipynb
├── 04_temporal_split.ipynb
├── 05_model_comparison.ipynb
├── 06_feature_importance.ipynb
├── 07_feature_group_ablation.ipynb
├── 08_subgroup_error_analysis.ipynb
└── 09_validation_strategy_comparison.ipynb

outputs/
├── figures/       # Figures generated for the thesis analysis
├── logs/          # Grid-search logs
├── models/        # Trained model files
└── tables/        # Result and diagnostic tables
```

## Notebook workflow

The notebooks should be run in numerical order.

| Notebook | Purpose |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Explores the raw dataset, missing values, temporal coverage, funding-variable skewness, and categorical distributions. |
| `02_preprocessing_feature_engineering.ipynb` | Applies outcome filtering, data cleaning, missing-value handling, feature engineering, and feature selection. |
| `03_feature_diagnostics.ipynb` | Checks correlations among the final numeric and binary predictors. |
| `04_temporal_split.ipynb` | Creates the blocked temporal train-test split using `last_funding_at`. |
| `05_model_comparison.ipynb` | Compares Logistic Regression, XGBoost, and CatBoost on the fixed temporal holdout test set. |
| `06_feature_importance.ipynb` | Computes SHAP-based feature importance for the selected XGBoost model. |
| `07_feature_group_ablation.ipynb` | Evaluates feature-group contribution by removing one feature group at a time. |
| `08_subgroup_error_analysis.ipynb` | Compares false-positive and false-negative patterns for U.S.-based and non-U.S.-based startups. |
| `09_validation_strategy_comparison.ipynb` | Compares stratified cross-validation with time-aware validation. |

## Main outputs

Main result tables are stored in:

```text
outputs/tables/model_comparison/
outputs/tables/feature_importance/
outputs/tables/feature_group_ablation/
outputs/tables/subgroup_error_analysis/
outputs/tables/validation_strategy_comparison/
```

Main figures are stored in:

```text
outputs/figures/eda/
outputs/figures/feature_diagnostics/
outputs/figures/feature_importance/
outputs/figures/feature_group_ablation/
outputs/figures/subgroup_error_analysis/
```

Trained models are stored in:

```text
outputs/models/
```

Grid-search logs are stored in:

```text
outputs/logs/
```

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run the notebooks in order from `01` to `09`.

The notebooks assume the following project structure:

```text
data/raw/investments_VC.csv
notebooks/
outputs/
```

## Reproducibility notes

A fixed random seed of `2026` was used where applicable, including cross-validation procedures, model initialization, and hyperparameter tuning.

The output tables, figures, logs, processed datasets, split files, and trained models were generated from the notebooks and are included to support reproducibility and inspection of the thesis results.

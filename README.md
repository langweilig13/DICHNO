# Dissecting Characteristics Nonparametrically

## About

Reimplementation of Freyberger, Neuhierl, and Weber, *Dissecting Characteristics Nonparametrically*.

The project reproduces the paper's nonparametric return-prediction approach and compares it with several feature-selection methods.

## Models

- **Spline Regression** — additive quadratic spline baseline;
- **Adaptive Group LASSO** — main method from the paper;
- **Random Forest** — nonlinear benchmark with Random-Forest importance;
- **LassoNet** — neural-network alternative with feature sparsity and a time-aware selection path.

## Data

The project uses firm characteristics constructed from CRSP and Compustat data.

For each time period, every characteristic is converted to its cross-sectional rank. The target is the next-period excess stock return.

## Project structure

```text
.
├── configs/
│   └── random_forest_ranked.json
├── run_lassonet_pipeline.py
├── src/
│   ├── data/
│   │   └── panel.py
│   ├── evaluation/
│   │   └── rolling_ols.py
│   ├── model/
│   │   ├── adaptive_group_lasso.py
│   │   ├── baseline_model.py
│   │   ├── group_lasso.py
│   │   └── lassonet_selector.py
│   ├── selection/
│   │   ├── bic.py
│   │   └── random_forest.py
│   ├── notebooks/
│   │   └── init.ipynb
│   └── transforms/
│       └── ranking.py
└── README.md
```

## Random-Forest selector

`src/selection/random_forest.py` contains the alternative selector supplied in
`rf-model-development-log.zip`. It fits a Random Forest on the training period,
uses the model's feature importances to retain the top characteristics, and
returns their names for a downstream forecasting model. Its ranked-target
configuration is recorded in `configs/random_forest_ranked.json`.

This is a benchmark branch, not part of the paper's adaptive group-LASSO
estimator. It must select features using training data only; validation and test
periods must not enter `RandomForestFeatureSelector.fit()`.

## LassoNet and backtesting

The LassoNet contribution is integrated as a separate alternative pipeline:

- `src/data/panel.py` loads the characteristics panel, computes within-month
  ranks, aligns the risk-free rate, and applies the price filter. The default
  `price_filter="previous"` uses the price at `t-1`, avoiding a current-month
  look-ahead filter.
- `src/model/lassonet_selector.py` fits a LassoNet regularization path on the
  selection sample and applies pre-specified sparsity and seed-consensus rules.
- `src/evaluation/rolling_ols.py` estimates the selected-feature forecasting
  regression in rolling windows and turns forecasts into long-short portfolios.
- `run_lassonet_pipeline.py` connects those pieces without changing the spline
  / adaptive group-LASSO branch.

Install the optional LassoNet dependencies before running its pipeline:

```bash
.venv/bin/pip install -r requirements-lassonet.txt
```

Then provide the panel and Fama-French risk-free archive explicitly, for example:

```bash
.venv/bin/python run_lassonet_pipeline.py \
  --csv src/data/characteristics_data_feb2017.csv \
  --rf-zip /path/to/F-F_Research_Data_Factors_CSV.zip
```

The pipeline has a `--price-filter previous` default. `--price-filter current`
is retained only for matching descriptive specifications and must not be treated
as an honest out-of-sample trading filter.

## Experiments

- reproduce the main results from the original paper;
- evaluate the models on July 2015–December 2025;
- compare feature selection and out-of-sample forecasts.

## Reference

Freyberger, J., Neuhierl, A., Weber, M.  
*Dissecting Characteristics Nonparametrically*.

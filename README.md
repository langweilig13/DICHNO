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


## Experiments

- reproduce the main results from the original paper;
- compare feature selection and out-of-sample forecasts.

## Reference

Freyberger, J., Neuhierl, A., Weber, M.  
*Dissecting Characteristics Nonparametrically*.

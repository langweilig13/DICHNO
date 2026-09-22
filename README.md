# Dissecting Characteristics Nonparametrically

## About

Reimplementation of Freyberger, Neuhierl, and Weber, *Dissecting Characteristics Nonparametrically*.

The project reproduces the paper's spline-based return model and compares it with alternative feature-selection methods.

## Models

- **Spline regression** — baseline additive quadratic spline model;
- **Adaptive Group LASSO** — main method from the paper;
- **Random Forest** — nonlinear benchmark with permutation importance;
- **LassoNet** — neural network with feature sparsity.

## Data

The model uses firm characteristics from CRSP and Compustat.

For every time period, each characteristic is transformed into its cross-sectional rank:

$$
\widetilde{C}_{s,i,t}
=
\frac{\operatorname{rank}(C_{s,i,t})}{N_t + 1}.
$$

The target is the next-period excess stock return.

## Method

Each characteristic is represented with quadratic spline basis functions:

$$
m_s(x)=\sum_k \beta_{sk}p_k(x).
$$

The final prediction is additive:

$$
\hat R_{i,t}
=
\sum_{s=1}^{S} m_s(C_{s,i,t-1}).
$$

## Project structure

```text
src/
├── model/
│   ├── baseline_model.py
│   └── group_lasso.py
├── notebooks/
│   └── init.ipynb
└── transforms/
    └── ranking.py
```

## Experiments

- reproduce the main historical results from the paper;
- evaluate the model on July 2015–December 2025;
- compare feature selection and out-of-sample forecasts across models.

## Reference

Freyberger, J., Neuhierl, A., Weber, M.  
*Dissecting Characteristics Nonparametrically*.

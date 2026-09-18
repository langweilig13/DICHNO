# Implementation and comparison plan

## Research question

Reproduce the empirical results in Freyberger, Neuhierl, and Weber, *Dissecting Characteristics Nonparametrically*, then test whether its characteristic selection and return forecasts hold up from July 2015 through December 2025. Compare the paper's adaptive group LASSO with two specified alternatives: a random forest assessed with held-out permutation importance, and LassoNet. Use the same investable universe, predictor definitions, information dates, and evaluation protocol wherever the methods permit.

## 1. Reconstruct the data

1. Obtain CRSP monthly and daily stock data, Compustat fundamentals, the CRSP–Compustat link history, and the daily Fama–French factors needed for idiosyncratic volatility. Confirm licensing and coverage through December 2025 before setting the final endpoint.
2. Build a point-in-time firm-month panel of the paper's 36 characteristics. Implement the formulas in Section IV and Online Appendix I, including their daily-data requirements, accounting fallbacks, and July-to-June timing for annual fundamentals. Retain the raw inputs and the date on which each input becomes usable for prediction.
3. Apply the paper's universe: US-incorporated common stocks on NYSE, Amex, or Nasdaq, price above $5, and at least two years of Compustat history. Reconcile security-level returns with firm-level market equity, delisting returns, exchange changes, missing values, and cross-sectional coverage. Document every rule that the paper leaves ambiguous.
4. At each prediction date, transform each characteristic to its cross-sectional rank divided by the number of eligible stocks plus one. Check selected characteristic distributions and the sample counts against Table 1 before fitting models.

**Data checkpoint:** freeze a versioned panel, feature dictionary, coverage report, and a table of differences between reconstructed and published sample counts. Do not interpret a model discrepancy until the underlying data discrepancy is understood.

## 2. Reproduce the paper on its historical sample

1. Implement the additive model with quadratic spline functions of the ranked characteristics. Use equally spaced knots and the paper's two-stage adaptive group LASSO: group LASSO, adaptive group weights, BIC selection at each stage, and OLS refitting on selected characteristics.
2. Reproduce Table 4's selected variables and selected-variable counts, beginning with the 14-knot baseline and the nine-knot specification used for forecasting. Reproduce the paper's large-stock and period-split checks and the main conditional response curves where feasible.
3. Reproduce Table 5's forecasting experiment. Select features using only the initial pre-1991 period; fit coefficients on a rolling 120-month window; make one-month-ahead predictions from January 1991; form equal- and value-weighted long–short portfolios from the top and bottom predicted deciles. Replicate the paper's linear adaptive-LASSO benchmark.
4. Report the paper's Sharpe ratios and the reproduction results side by side, with differences in sample size, selected features, portfolio construction, and return handling. Add firm-level out-of-sample R² and forecast slopes to check that the comparison is not driven by a portfolio metric alone.

**Date reconciliation:** the text states a July 1963–June 2015 overall sample, while Tables 4 and 5 label their principal results through 2014. Match each published table to its stated endpoint first. Treat any extension through June 2015 as a separate run, with its exact endpoint recorded.

## 3. Evaluate 2015–2025 as a new period

1. Define the primary new evaluation period as July 2015–December 2025, with no overlap with the stated full historical sample. Check data availability before locking December 2025 as the last return month.
2. Run two versions of the paper's model: **fixed selection**, carrying forward the characteristics chosen before the new period while updating coefficients on a rolling 120-month window; and **updated selection**, rerunning the selection procedure using only information available before each forecast date. Keep these results separate: they answer persistence and adaptation questions, respectively.
3. Compare performance in 2015–2025 with the historical out-of-sample period on the same return definition and universe. Show annual and subperiod results, equal- and value-weighted portfolios, and a large-stock subset. Track which characteristics remain selected and whether their estimated response curves change.

## 4. Compare the two requested alternatives

| Method | Fit and feature assessment | Output for comparison |
| --- | --- | --- |
| Adaptive group LASSO | Two-stage spline selection and refit as in the paper | Selected characteristics, response curves, forecasts |
| Random forest | Fit a regression forest on the same training windows. In a held-out validation fold, reshuffle one feature at a time and measure the loss in forecast quality; repeat with several seeds. | Permutation-importance ranks and forecasts |
| LassoNet | Fit the feature-sparse neural-network regression model on the same inputs. Choose its regularization level and architecture using only training/validation data. | Selected characteristics, selection path, forecasts |

Use rolling, time-ordered validation for all tuning. Hold the final test months out of fitting, tuning, and decisions about which features to keep. Permutation importance on the final test months can be reported as a diagnostic after model choices are frozen; it must not feed back into model selection. Random-forest permutation importance measures reliance of the fitted forest on a feature, which is not the same estimand as the paper's conditional additive-effect selection. Because correlated characteristics can split or mask importance, add a grouped-permutation sensitivity check for strongly correlated predictors.

For the primary comparison, use all 36 characteristics for each method and identical monthly forecasts. Report annualized Sharpe ratio, mean long–short return, volatility, drawdown, turnover, firm-level out-of-sample R², and performance by year. Include gross results for comparison with the paper and a separately labeled trading-cost sensitivity analysis. Compare selected sets with overlap and rank stability, while interpreting each method's definition of “important” separately.

## 5. Decision rules and deliverables

Pre-register the historical and new-period endpoints, missing-data treatment, rolling-window length, tuning grids, random seeds, and primary metric before inspecting 2015–2025 results. Estimate uncertainty with time-series-aware intervals or block bootstrap and report the number of test months; avoid selecting a winner from a small difference in Sharpe ratios. Record changes in data definitions or model settings as deviations from the reproduction.

Deliver a data-construction specification, reproducible code and environment, a historical replication table, a 2015–2025 comparison table, characteristic-stability plots, and a short report explaining where the reproduction matches the paper, where it differs, and whether the differences persist in the new sample.

## References

- Freyberger, Neuhierl, and Weber, *Dissecting Characteristics Nonparametrically*, supplied PDF, especially Section III.D, Section IV, Section V.C, Tables 4–5, and Online Appendix I.
- [Lemhadri et al., *LassoNet: A Neural Network with Feature Sparsity*](https://www.jmlr.org/papers/volume22/20-848/20-848.pdf).
- [scikit-learn documentation: permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html).

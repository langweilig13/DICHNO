"""Rolling OLS forecasts and long-short portfolio evaluation."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd


def _monthly_sums(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray | float | int]:
    return {
        "n": len(y),
        "sum_x": X.sum(axis=0),
        "sum_y": float(y.sum()),
        "gram": X.T @ X,
        "cross": X.T @ y,
    }


def _window_coefficients(window: Iterable[dict], width: int) -> tuple[float, np.ndarray]:
    items = list(window)
    n = sum(int(item["n"]) for item in items)
    sum_x = sum((item["sum_x"] for item in items), np.zeros(width))
    sum_y = sum(float(item["sum_y"]) for item in items)
    gram = sum((item["gram"] for item in items), np.zeros((width, width)))
    cross = sum((item["cross"] for item in items), np.zeros(width))
    centered_gram = gram - np.outer(sum_x, sum_x) / n
    centered_cross = cross - sum_x * sum_y / n
    beta = np.linalg.lstsq(centered_gram, centered_cross, rcond=1e-10)[0]
    intercept = (sum_y - sum_x @ beta) / n
    return float(intercept), beta


def _hedge_return(
    predicted: np.ndarray,
    returns: np.ndarray,
    market_caps: np.ndarray,
) -> tuple[float, float, int]:
    order = np.argsort(predicted, kind="stable")
    decile = max(1, len(order) // 10)
    short = order[:decile]
    long = order[-decile:]
    ew = float(returns[long].mean() - returns[short].mean())
    caps = np.maximum(np.asarray(market_caps, dtype=np.float64), 0)
    if caps[long].sum() <= 0 or caps[short].sum() <= 0:
        raise ValueError("Value-weighted portfolio has zero total market capitalization")
    vw = float(
        np.average(returns[long], weights=caps[long])
        - np.average(returns[short], weights=caps[short])
    )
    return ew, vw, decile


def run_rolling_ols(
    data: pd.DataFrame,
    ranked_features: np.ndarray,
    excess: np.ndarray,
    *,
    selected_features: list[str],
    backtest_start: int = 199101,
    backtest_end: int = 201405,
    window_months: int = 120,
    universe: str = "all",
) -> list[dict]:
    """Fit OLS on the preceding window and predict each next cross section."""

    if universe not in {"all", "size_q10"}:
        raise ValueError("universe must be 'all' or 'size_q10'")
    if len(data) != len(ranked_features) or len(data) != len(excess):
        raise ValueError("data, ranked_features, and excess must have equal length")
    if ranked_features.shape[1] != len(selected_features):
        raise ValueError("ranked_features width does not match selected_features")

    history: deque[dict] = deque()
    output: list[dict] = []
    for period, indices in data.groupby("period", sort=True).indices.items():
        period = int(period)
        indices = np.asarray(indices)
        if universe == "size_q10":
            large = data.iloc[indices].lme.to_numpy() > data.iloc[indices].q10.to_numpy()
            indices = indices[large]
        if len(indices) < 20:
            continue
        X = ranked_features[indices].astype(np.float64, copy=False)
        y = excess[indices]
        raw = data.iloc[indices].ret.to_numpy(dtype=np.float64)

        if backtest_start <= period <= backtest_end:
            if len(history) != window_months:
                raise RuntimeError(
                    f"Expected {window_months} training months before {period}, "
                    f"found {len(history)}"
                )
            intercept, beta = _window_coefficients(history, X.shape[1])
            predicted = intercept + X @ beta
            ew, vw, decile = _hedge_return(
                predicted, raw, data.iloc[indices].lme.to_numpy(dtype=np.float64)
            )
            output.append(
                {
                    "period": period,
                    "firms": int(len(indices)),
                    "decile_firms": int(decile),
                    "ew_long_short": ew,
                    "vw_long_short": vw,
                }
            )

        if period < backtest_end:
            history.append(_monthly_sums(X, y))
            if len(history) > window_months:
                history.popleft()
    if not output:
        raise RuntimeError("The rolling backtest produced no out-of-sample months")
    return output


def _newey_west_t(values: np.ndarray, lags: int = 6) -> float | None:
    n = len(values)
    mean = float(values.mean())
    centered = values - mean
    long_run = float(centered @ centered / n)
    for lag in range(1, min(lags, n - 1) + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        long_run += 2 * (1 - lag / (lags + 1)) * covariance
    return mean / np.sqrt(long_run / n) if long_run > 0 else None


def _block_bootstrap(
    values: np.ndarray,
    *,
    seed: int = 20260925,
    draws: int = 2000,
    block: int = 12,
) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = rng.integers(0, n, size=(draws, (n + block - 1) // block))
    indices = (starts[:, :, None] + np.arange(block)) % n
    samples = values[indices.reshape(draws, -1)[:, :n]].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def summarize_returns(rows: list[dict]) -> dict:
    """Summarize EW and VW long-short returns with annualized Sharpe ratios."""

    summary: dict[str, dict] = {}
    for label, field in (("ew", "ew_long_short"), ("vw", "vw_long_short")):
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        volatility = float(values.std(ddof=1))
        summary[label] = {
            "months": int(len(values)),
            "mean_monthly": float(values.mean()),
            "volatility_monthly": volatility,
            "sharpe_annualized": (
                float(np.sqrt(12) * values.mean() / volatility) if volatility > 0 else None
            ),
            "newey_west_t": _newey_west_t(values),
            "positive_month_fraction": float(np.mean(values > 0)),
            "block_bootstrap_95pct_mean_ci": _block_bootstrap(values),
        }
    return summary

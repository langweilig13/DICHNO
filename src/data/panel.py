"""Load the CRSP/Compustat characteristic panel without look-ahead filters."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METADATA_COLUMNS = [
    "yy",
    "mm",
    "date",
    "permno",
    "ret",
    "q10",
    "q20",
    "q50",
    "prc",
]


def _feature_columns(path: Path) -> list[str]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if "a2me" not in columns:
        raise ValueError("The CSV does not contain the expected a2me characteristic")
    features = columns[columns.index("a2me") :]
    if len(features) != 36:
        raise ValueError(f"Expected 36 characteristics, found {len(features)}")
    return features


def load_characteristics(
    path: str | Path,
    *,
    start_period: int = 196307,
    end_period: int = 201405,
    price_filter: str = "previous",
    nrows: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Read and filter the complete-case panel.

    ``previous`` is the default price filter because the previous month price is
    known when the portfolio is formed. ``current`` is retained only for the
    descriptive Table 4 comparison with the paper.
    """

    path = Path(path)
    if price_filter not in {"previous", "current"}:
        raise ValueError("price_filter must be 'previous' or 'current'")
    features = _feature_columns(path)
    wanted = list(dict.fromkeys([*METADATA_COLUMNS, *features]))
    dtypes = {
        column: "float32"
        for column in wanted
        if column not in {"yy", "mm", "permno", "date"}
    }
    dtypes.update({"yy": "int16", "mm": "int8", "permno": "int32"})
    data = pd.read_csv(path, usecols=wanted, dtype=dtypes, nrows=nrows)
    data.sort_values(["permno", "yy", "mm"], kind="stable", inplace=True)

    period = data.yy.astype("int32") * 100 + data.mm.astype("int32")
    if price_filter == "current":
        eligible_price = data.prc.abs() > 5
    else:
        month_index = data.yy.astype("int32") * 12 + data.mm.astype("int32")
        grouped = data.groupby("permno", sort=False)
        previous_index = grouped.yy.shift().to_numpy() * 12 + grouped.mm.shift().to_numpy()
        previous_price = grouped.prc.shift().abs().to_numpy()
        eligible_price = (month_index.to_numpy() == previous_index + 1) & (previous_price > 5)

    finite = np.isfinite(data[["ret", *features]].to_numpy()).all(axis=1)
    valid = (
        (period.to_numpy() >= start_period)
        & (period.to_numpy() <= end_period)
        & np.asarray(eligible_price)
        & finite
    )
    data = data.loc[valid].copy()
    data["period"] = data.yy.astype("int32") * 100 + data.mm.astype("int32")
    data.reset_index(drop=True, inplace=True)
    if data.empty:
        raise ValueError("No observations remain after the requested filters")
    return data, features


def cross_sectional_ranks(
    data: pd.DataFrame,
    features: Iterable[str],
) -> np.ndarray:
    """Return monthly average ranks divided by the monthly cross-section size + 1."""

    feature_list = list(features)
    if not feature_list:
        raise ValueError("At least one feature is required")
    grouped = data.groupby("period", sort=False)
    ranks = grouped[feature_list].rank(method="average")
    sizes = data["period"].map(data["period"].value_counts(sort=False)).to_numpy()
    # pandas may expose a read-only view; ranking below normalizes in place.
    values = ranks.to_numpy(dtype=np.float32, copy=True)
    values /= (sizes[:, None] + 1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values appeared during rank transformation")
    return values


def sample_monthly(
    data: pd.DataFrame,
    arrays: list[np.ndarray],
    max_per_month: int,
    *,
    seed: int,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Take a reproducible within-month sample after ranks have been computed."""

    if max_per_month <= 0:
        return data, arrays
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for indices in data.groupby("period", sort=True).indices.values():
        indices = np.asarray(indices)
        if len(indices) > max_per_month:
            indices = np.sort(rng.choice(indices, max_per_month, replace=False))
        chosen.append(indices)
    selected = np.concatenate(chosen)
    sampled_data = data.iloc[selected].reset_index(drop=True)
    return sampled_data, [array[selected] for array in arrays]


def load_risk_free(path: str | Path) -> dict[int, float]:
    """Read monthly RF in decimal units from the Kenneth French factor ZIP."""

    rates: dict[int, float] = {}
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(member) as stream:
            rows = csv.reader(line.decode("utf-8-sig") for line in stream)
            for row in rows:
                key = row[0].strip() if row else ""
                if len(row) >= 5 and key.isdigit() and len(key) == 6:
                    rates[int(key)] = float(row[4]) / 100
    if not rates:
        raise ValueError(f"No monthly risk-free rates found in {path}")
    return rates


def excess_returns(data: pd.DataFrame, risk_free: dict[int, float]) -> np.ndarray:
    """Subtract the contemporaneous monthly risk-free rate from raw stock returns."""

    missing = sorted(set(data.period.unique()) - set(risk_free))
    if missing:
        raise ValueError(f"Missing risk-free rates for periods: {missing[:5]}")
    rf = data.period.map(risk_free).to_numpy(dtype=np.float64)
    return data.ret.to_numpy(dtype=np.float64) - rf

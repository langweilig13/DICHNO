"""Pre-specified LassoNet sensitivity analysis with rolling OLS portfolios."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.panel import (
    cross_sectional_ranks,
    excess_returns,
    load_characteristics,
    load_risk_free,
    sample_monthly,
)
from src.evaluation.rolling_ols import run_rolling_ols, summarize_returns
from src.model.lassonet_selector import (
    PATH_CHECKPOINT_SCHEMA_VERSION,
    SENSITIVITY_RULES,
    LassoNetSelection,
    apply_selection_rules,
    fit_lassonet_paths,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _elapsed_eta(started: float, completed: int, total: int) -> str:
    elapsed = time.perf_counter() - started
    if completed <= 0 or completed >= total:
        eta = 0.0 if completed >= total else None
    else:
        eta = elapsed / completed * (total - completed)
    elapsed_text = f"{elapsed / 60:.1f} min"
    eta_text = "unknown" if eta is None else f"{eta / 60:.1f} min"
    return f"elapsed={elapsed_text}, ETA={eta_text}"


def _group_rules_by_feature_set(
    selections: dict[str, LassoNetSelection],
) -> dict[tuple[str, ...], list[str]]:
    """Group rules so identical feature sets share one rolling OLS computation."""

    groups: dict[tuple[str, ...], list[str]] = {}
    for rule_name, selection in selections.items():
        groups.setdefault(tuple(selection.selected), []).append(rule_name)
    return groups


def run_sensitivity_backtests(
    data: pd.DataFrame,
    excess: np.ndarray,
    all_features: list[str],
    selections: dict[str, LassoNetSelection],
    *,
    backtest_start: int,
    backtest_end: int,
    window_months: int,
    progress: bool = True,
) -> tuple[dict[str, dict], list[dict], dict[str, dict]]:
    """Backtest each unique selected set once and fan results out to its rules."""

    grouped = _group_rules_by_feature_set(selections)
    selected_union = [
        feature
        for feature in all_features
        if any(feature in feature_set for feature_set in grouped)
    ]
    if not selected_union:
        raise RuntimeError("Sensitivity rules produced no selected features")
    union_ranks = cross_sectional_ranks(data, selected_union)
    union_index = {name: index for index, name in enumerate(selected_union)}

    rule_backtests: dict[str, dict] = {}
    monthly_rows: list[dict] = []
    feature_sets: dict[str, dict] = {}
    started = time.perf_counter()
    total = len(grouped)

    for completed, (feature_tuple, rule_names) in enumerate(grouped.items(), start=1):
        feature_set_id = f"feature_set_{completed}"
        selected = list(feature_tuple)
        columns = [union_index[name] for name in selected]
        ranked = union_ranks[:, columns]
        if progress:
            print(
                f"[5/5] Rolling OLS {completed}/{total}: {feature_set_id}, "
                f"{len(selected)} features, rules={','.join(rule_names)}",
                flush=True,
            )

        universes: dict[str, dict] = {}
        rows_by_universe: dict[str, list[dict]] = {}
        for universe in ("all", "size_q10"):
            rows = run_rolling_ols(
                data,
                ranked,
                excess,
                selected_features=selected,
                backtest_start=backtest_start,
                backtest_end=backtest_end,
                window_months=window_months,
                universe=universe,
            )
            rows_by_universe[universe] = rows
            universes[universe] = {
                "period": [backtest_start, backtest_end],
                "training_window_months": window_months,
                "portfolio_rule": "long top predicted-return decile, short bottom decile",
                "summary": summarize_returns(rows),
            }

        feature_sets[feature_set_id] = {
            "selected_features": selected,
            "rules": rule_names,
        }
        for rule_name in rule_names:
            rule_backtests[rule_name] = {
                "feature_set_id": feature_set_id,
                "reused_backtest": rule_name != rule_names[0],
                "universes": universes,
            }
            for universe, rows in rows_by_universe.items():
                monthly_rows.extend(
                    {
                        "selection_rule": rule_name,
                        "feature_set_id": feature_set_id,
                        "universe": universe,
                        **row,
                    }
                    for row in rows
                )

        if progress:
            print(
                f"[5/5] Completed {feature_set_id}; "
                f"{_elapsed_eta(started, completed, total)}",
                flush=True,
            )

    return rule_backtests, monthly_rows, feature_sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=Path("src/data/characteristics_data_feb2017.csv")
    )
    parser.add_argument(
        "--rf-zip",
        type=Path,
        default=Path("src/data/F-F_Research_Data_Factors_CSV.zip"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/lassonet_sensitivity.json")
    )
    parser.add_argument(
        "--monthly-output",
        type=Path,
        default=Path("results/lassonet_sensitivity_monthly.csv"),
    )
    parser.add_argument("--start-period", type=int, default=196307)
    parser.add_argument("--end-period", type=int, default=201405)
    parser.add_argument("--selection-end", type=int, default=199012)
    parser.add_argument("--validation-months", type=int, default=60)
    parser.add_argument("--backtest-start", type=int, default=199101)
    parser.add_argument("--backtest-end", type=int, default=201405)
    parser.add_argument("--window-months", type=int, default=120)
    parser.add_argument("--price-filter", choices=("previous", "current"), default="previous")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 29, 41, 53])
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.60,
        help="Legacy baseline flag; sensitivity baseline is fixed at 0.60",
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=0.01,
        help="Legacy baseline flag; sensitivity baseline is fixed at 0.01",
    )
    parser.add_argument("--selection-sample-per-month", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[32, 16])
    parser.add_argument("--path-multiplier", type=float, default=1.08)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-iters-init", type=int, default=100)
    parser.add_argument("--n-iters-path", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=1e-4)
    parser.add_argument("--gamma-skip", type=float, default=1e-4)
    parser.add_argument("--min-selected", type=int, default=1)
    parser.add_argument("--max-selected", type=int, default=36)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Full-path seed checkpoint JSON (defaults next to --output)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed full seed paths from a compatible schema-v2 checkpoint",
    )
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--nrows", type=int, help="Read only the first rows for diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_started = time.perf_counter()
    if args.price_filter == "current" and not args.skip_backtest:
        raise ValueError("current-month price is not allowed in the tradable backtest")
    if args.selection_end >= args.backtest_start and not args.skip_backtest:
        raise ValueError("selection_end must be strictly before backtest_start")
    if args.backtest_end > args.end_period:
        raise ValueError("backtest_end cannot be later than end_period")
    if not np.isclose(args.selection_tolerance, SENSITIVITY_RULES["baseline"].tolerance):
        raise ValueError(
            "The sensitivity baseline fixes --selection-tolerance at 0.01; "
            "remove the legacy override."
        )
    if not np.isclose(
        args.consensus_threshold,
        SENSITIVITY_RULES["baseline"].consensus_threshold,
    ):
        raise ValueError(
            "The sensitivity baseline fixes --consensus-threshold at 0.60; "
            "remove the legacy override."
        )

    stage_started = time.perf_counter()
    print("[1/5] Loading, filtering, and hashing the panel...", flush=True)
    data, all_features = load_characteristics(
        args.csv,
        start_period=args.start_period,
        end_period=args.end_period,
        price_filter=args.price_filter,
        nrows=args.nrows,
    )
    risk_free = load_risk_free(args.rf_zip)
    excess = excess_returns(data, risk_free)
    source_sha256 = _sha256(args.csv)
    checkpoint_path = args.checkpoint or args.output.with_name(
        args.output.stem + "_checkpoint.json"
    )
    print(
        f"[1/5] Data ready in {(time.perf_counter() - stage_started) / 60:.1f} min: "
        f"{len(data):,} observations",
        flush=True,
    )

    stage_started = time.perf_counter()
    print("[2/5] Building the pre-1991 ranked selection sample...", flush=True)
    selection_mask = data.period.to_numpy() <= args.selection_end
    selection_data = data.loc[selection_mask].reset_index(drop=True)
    selection_ranks = cross_sectional_ranks(selection_data, all_features)
    selection_y = excess[selection_mask]
    selection_data, [selection_ranks, selection_y] = sample_monthly(
        selection_data,
        [selection_ranks, selection_y],
        args.selection_sample_per_month,
        seed=args.seeds[0],
    )
    print(
        f"Selection sample: {len(selection_data):,} observations, "
        f"{selection_data.period.nunique()} months",
        flush=True,
    )
    if args.selection_sample_per_month:
        print(
            "Pilot sampling is enabled; do not report this as the final full-sample run.",
            flush=True,
        )
    print(
        f"[2/5] Selection sample ready in "
        f"{(time.perf_counter() - stage_started) / 60:.1f} min",
        flush=True,
    )

    stage_started = time.perf_counter()
    print("[3/5] Training or restoring one LassoNet path per seed...", flush=True)
    paths = fit_lassonet_paths(
        selection_ranks,
        selection_y,
        selection_data.period.to_numpy(),
        all_features,
        seeds=args.seeds,
        validation_months=args.validation_months,
        hidden_dims=tuple(args.hidden_dims),
        path_multiplier=args.path_multiplier,
        batch_size=args.batch_size,
        n_iters=(args.n_iters_init, args.n_iters_path),
        gamma=args.gamma,
        gamma_skip=args.gamma_skip,
        device=args.device,
        verbose=args.verbose,
        checkpoint_path=checkpoint_path,
        resume=args.resume,
        checkpoint_context={
            "source_sha256": source_sha256,
            "start_period": args.start_period,
            "end_period": args.end_period,
            "selection_end": args.selection_end,
            "price_filter": args.price_filter,
            "selection_sample_per_month": args.selection_sample_per_month,
        },
    )
    print(
        f"[3/5] LassoNet paths ready in "
        f"{(time.perf_counter() - stage_started) / 60:.1f} min",
        flush=True,
    )

    stage_started = time.perf_counter()
    print("[4/5] Applying pre-specified selection rules to saved paths...", flush=True)
    selections = apply_selection_rules(
        paths,
        all_features,
        SENSITIVITY_RULES,
        min_selected=args.min_selected,
        max_selected=args.max_selected,
    )
    for index, (rule_name, selection) in enumerate(selections.items(), start=1):
        rule = SENSITIVITY_RULES[rule_name]
        print(
            f"[4/5] Rule {index}/{len(selections)} {rule_name}: "
            f"tolerance={rule.tolerance:g}, consensus={rule.consensus_threshold:g}, "
            f"selected={len(selection.selected)} {selection.selected}",
            flush=True,
        )
    print(
        f"[4/5] Rules applied in {(time.perf_counter() - stage_started):.1f} sec",
        flush=True,
    )

    training_parameters = {
        "lassonet_package": "lassonet==0.0.20",
        "hidden_dims": args.hidden_dims,
        "lambda_start": "auto",
        "path_multiplier": args.path_multiplier,
        "hierarchy_M": 10,
        "dropout": 0,
        "batch_size": args.batch_size,
        "n_iters": [args.n_iters_init, args.n_iters_path],
        "patience": None,
        "val_size": 0,
        "optimizer": "official defaults: Adam initialization, SGD regularization path",
        "gamma": args.gamma,
        "gamma_skip": args.gamma_skip,
        "seeds": args.seeds,
        "device": args.device,
        "feature_scaling": "training-sample mean and standard deviation",
        "target_scaling": "training-sample mean and standard deviation",
        "path_trained_once_per_seed": True,
        "state_dicts_stored": False,
    }
    result = {
        "analysis_type": "pre-specified LassoNet feature-selection sensitivity analysis",
        "source_csv": str(args.csv),
        "source_sha256": source_sha256,
        "input_observations_after_filters": int(len(data)),
        "input_period": [int(data.period.min()), int(data.period.max())],
        "price_filter": args.price_filter,
        "price_known_at_formation": args.price_filter == "previous",
        "target": "monthly stock return minus Kenneth French monthly RF",
        "selection_observations": int(len(selection_data)),
        "selection_months": int(selection_data.period.nunique()),
        "selection_sample_per_month": args.selection_sample_per_month,
        "pilot_sampled_selection": bool(args.selection_sample_per_month),
        "checkpoint": {
            "path": str(checkpoint_path),
            "schema_version": PATH_CHECKPOINT_SCHEMA_VERSION,
            "contains_full_paths": True,
            "contains_state_dicts": False,
        },
        "periods": {
            "train": paths.train_period,
            "validation": paths.validation_period,
            "oos": [args.backtest_start, args.backtest_end],
        },
        "lassonet_training": training_parameters,
        "selection_rules_pre_specified_before_oos_evaluation": True,
        "oos_period_used_to_tune_rules": False,
        "returns": {
            "type": "gross",
            "transaction_costs_included": False,
            "borrow_fees_included": False,
        },
        "rules": {},
        "limits": [
            "Do not select a preferred rule by its 1991-2014 out-of-sample Sharpe ratio.",
            "No transaction costs, spreads, slippage, borrow fees, or short-sale "
            "constraints are included.",
            "LassoNet is non-convex; per-seed checkpoints and consensus frequencies are reported.",
            "The supplied historical CSV ends in May 2014.",
        ],
    }
    for rule_name, selection in selections.items():
        rule = SENSITIVITY_RULES[rule_name]
        result["rules"][rule_name] = {
            "parameters": {
                "selection_tolerance": rule.tolerance,
                "consensus_threshold": rule.consensus_threshold,
                "min_selected_per_seed": args.min_selected,
                "max_selected_per_seed": args.max_selected,
            },
            **selection.sensitivity_dict(),
        }

    monthly_rows: list[dict] = []
    if not args.skip_backtest:
        print("[5/5] Running rolling OLS for each unique feature set...", flush=True)
        backtests, monthly_rows, feature_sets = run_sensitivity_backtests(
            data,
            excess,
            all_features,
            selections,
            backtest_start=args.backtest_start,
            backtest_end=args.backtest_end,
            window_months=args.window_months,
        )
        result["unique_feature_sets"] = feature_sets
        for rule_name, backtest in backtests.items():
            result["rules"][rule_name]["backtest"] = backtest
    else:
        result["unique_feature_sets"] = {}
        print("[5/5] Backtest skipped by --skip-backtest", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if monthly_rows:
        args.monthly_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(monthly_rows).to_csv(args.monthly_output, index=False)
    print(f"Wrote {args.output}", flush=True)
    if monthly_rows:
        print(f"Wrote {args.monthly_output}", flush=True)
    print(
        f"Total elapsed: {(time.perf_counter() - run_started) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()

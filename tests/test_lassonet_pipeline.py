"""Invariant tests for the time-aware LassoNet and rolling OLS pipeline."""

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data.panel import cross_sectional_ranks, load_characteristics
from src.evaluation.rolling_ols import run_rolling_ols, summarize_returns
from src.model.lassonet_selector import (
    LassoNetPaths,
    LassoNetSelection,
    PathPoint,
    SENSITIVITY_RULES,
    SeedPath,
    SeedSelection,
    SelectionRule,
    _checkpoint_signature,
    _path_checkpoint_signature,
    _read_path_checkpoint,
    _read_seed_checkpoint,
    _write_path_checkpoint,
    _write_seed_checkpoint,
    apply_selection_rule,
    apply_selection_rules,
    choose_sparse_checkpoint,
    fit_lassonet_paths,
    temporal_train_validation_mask,
)
from src.transforms.ranking import rank_transform
from run_lassonet_pipeline import run_sensitivity_backtests


@dataclass
class Checkpoint:
    lambda_: float
    val_loss: float
    selected: np.ndarray


class PipelineTests(unittest.TestCase):
    def test_rank_transform_uses_average_tie_ranks(self):
        actual = rank_transform(np.array([3.0, 1.0, 1.0, 2.0]))
        np.testing.assert_allclose(actual, [0.8, 0.3, 0.3, 0.6])

    def test_cross_sectional_ranks_do_not_mix_months(self):
        frame = pd.DataFrame(
            {"period": [200001, 200001, 200002, 200002], "x": [1, 2, 10, 20]}
        )
        np.testing.assert_allclose(
            cross_sectional_ranks(frame, ["x"]).ravel(),
            [1 / 3, 2 / 3, 1 / 3, 2 / 3],
        )

    def test_temporal_split_places_validation_after_training(self):
        periods = np.repeat([200001, 200002, 200003, 200004], 2)
        train, validation, train_period, validation_period = temporal_train_validation_mask(
            periods, 2
        )
        self.assertEqual(train_period, (200001, 200002))
        self.assertEqual(validation_period, (200003, 200004))
        self.assertLess(periods[train].max(), periods[validation].min())

    def test_sparse_checkpoint_rule(self):
        path = [
            Checkpoint(0, 1.00, np.array([1, 1, 1, 1], dtype=bool)),
            Checkpoint(1, 1.005, np.array([1, 1, 0, 0], dtype=bool)),
            Checkpoint(2, 1.02, np.array([1, 0, 0, 0], dtype=bool)),
        ]
        baseline = SENSITIVITY_RULES["baseline"]
        self.assertEqual(baseline.consensus_threshold, 0.60)
        chosen = choose_sparse_checkpoint(
            path, tolerance=baseline.tolerance, min_selected=1, max_selected=4
        )
        self.assertEqual(int(chosen.selected.sum()), 2)

    def test_zero_tolerance_selects_minimum_validation_loss(self):
        path = [
            Checkpoint(0, 1.00, np.array([1, 1, 1], dtype=bool)),
            Checkpoint(1, 1.01, np.array([1, 0, 0], dtype=bool)),
        ]
        chosen = choose_sparse_checkpoint(
            path, tolerance=0, min_selected=1, max_selected=3
        )
        self.assertEqual(chosen.lambda_, 0)
        self.assertEqual(chosen.val_loss, 1.00)

    @staticmethod
    def _seed_path(seed, masks_and_losses):
        points = []
        for index, (mask, loss) in enumerate(masks_and_losses):
            points.append(
                PathPoint(
                    path_index=index,
                    lambda_=float(index),
                    validation_loss=loss,
                    validation_mse=loss / 100,
                    selected_count=sum(mask),
                    selected_mask=mask,
                )
            )
        return SeedPath(seed=seed, path=points)

    def test_different_tolerances_reuse_one_path_without_training(self):
        paths = LassoNetPaths(
            per_seed=[
                self._seed_path(7, [([True, True], 1.0), ([True, False], 1.005)])
            ],
            train_period=[196307, 198512],
            validation_period=[198601, 199012],
        )
        rules = {
            "baseline": SelectionRule(0.01, 0.60),
            "best_loss": SelectionRule(0.0, 0.60),
        }
        with patch("src.model.lassonet_selector.fit_lassonet_paths") as trainer:
            selections = apply_selection_rules(paths, ["x", "y"], rules)
        trainer.assert_not_called()
        self.assertEqual(selections["baseline"].selected, ["x"])
        self.assertEqual(selections["best_loss"].selected, ["x", "y"])

    def test_consensus_thresholds_040_and_060(self):
        masks = [
            [True, True, True],
            [True, True, True],
            [False, True, True],
            [False, False, True],
            [False, False, True],
        ]
        paths = LassoNetPaths(
            per_seed=[
                self._seed_path(seed, [(mask, 1.0)])
                for seed, mask in zip([7, 17, 29, 41, 53], masks)
            ],
            train_period=[196307, 198512],
            validation_period=[198601, 199012],
        )
        broad = apply_selection_rule(
            paths, ["x", "y", "z"], tolerance=0, consensus_threshold=0.40
        )
        strict = apply_selection_rule(
            paths, ["x", "y", "z"], tolerance=0, consensus_threshold=0.60
        )
        self.assertEqual(broad.selected, ["x", "y", "z"])
        self.assertEqual(strict.selected, ["y", "z"])
        self.assertEqual(broad.selection_frequency["x"], 0.4)
        self.assertEqual(strict.selection_frequency["y"], 0.6)

    def test_seed_checkpoint_round_trip_and_signature_guard(self):
        signature = {"version": 1, "seeds": [7, 17], "feature_names": ["x"]}
        completed = {
            7: SeedSelection(
                seed=7,
                lambda_=1.0,
                validation_mse=0.02,
                selected=["x"],
                selected_count=1,
                path_length=4,
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            _write_seed_checkpoint(path, signature, completed)
            restored = _read_seed_checkpoint(path, signature)
            self.assertEqual(restored[7], completed[7])
            with self.assertRaises(RuntimeError):
                _read_seed_checkpoint(path, {**signature, "seeds": [7]})

    def test_full_path_checkpoint_round_trip_and_resume_skips_training(self):
        feature_names = ["x", "y"]
        periods = np.repeat([200001, 200002, 200003, 200004], 2)
        X = np.arange(16, dtype=np.float32).reshape(8, 2)
        y = np.linspace(-0.1, 0.1, 8, dtype=np.float32)
        kwargs = {
            "validation_months": 1,
            "hidden_dims": (2,),
            "path_multiplier": 2.0,
            "batch_size": 4,
            "n_iters": (1, 1),
            "gamma": 0.0,
            "gamma_skip": 0.0,
            "device": "cpu",
        }
        signature = _path_checkpoint_signature(
            names=feature_names,
            seed_list=[7],
            observations=len(y),
            period_bounds=[200001, 200004],
            checkpoint_context=None,
            **kwargs,
        )
        completed = {
            7: self._seed_path(
                7,
                [([True, True], 1.0), ([True, False], 1.01)],
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            _write_path_checkpoint(path, signature, completed)
            payload = pd.read_json(path, typ="series")
            self.assertEqual(payload["schema_version"], 2)
            restored, legacy = _read_path_checkpoint(path, signature)
            self.assertFalse(legacy)
            self.assertEqual(restored, completed)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            with patch(
                "lassonet.LassoNetRegressor.path",
                side_effect=AssertionError("resume must not retrain a complete seed"),
            ):
                paths = fit_lassonet_paths(
                    X,
                    y,
                    periods,
                    feature_names,
                    seeds=[7],
                    verbose=0,
                    checkpoint_path=path,
                    resume=True,
                    **kwargs,
                )
        self.assertEqual(paths.per_seed, [completed[7]])

    def test_legacy_checkpoint_is_detected_for_safe_upgrade(self):
        common = {
            "names": ["x"],
            "seed_list": [7],
            "observations": 10,
            "period_bounds": [200001, 200005],
            "validation_months": 1,
            "hidden_dims": (2,),
            "path_multiplier": 2.0,
            "batch_size": 4,
            "n_iters": (1, 1),
            "gamma": 0.0,
            "gamma_skip": 0.0,
            "device": "cpu",
            "checkpoint_context": {"source_sha256": "abc"},
        }
        legacy_signature = _checkpoint_signature(
            selection_tolerance=0.01,
            min_selected=1,
            max_selected=1,
            **common,
        )
        current_signature = _path_checkpoint_signature(**common)
        completed = {
            7: SeedSelection(
                seed=7,
                lambda_=1.0,
                validation_mse=0.02,
                selected=["x"],
                selected_count=1,
                path_length=4,
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            _write_seed_checkpoint(path, legacy_signature, completed)
            restored, legacy = _read_path_checkpoint(path, current_signature)
        self.assertTrue(legacy)
        self.assertEqual(restored, {})

    def test_previous_price_filter_never_uses_current_price(self):
        features = ["a2me", *[f"feature_{i}" for i in range(35)]]
        rows = []
        for month, price in ((1, 4), (2, 6), (3, 4)):
            row = {
                "row": month,
                "yy": 2000,
                "mm": month,
                "date": f"2000-{month:02d}-28",
                "permno": 10001,
                "ret": 0.01,
                "q10": 1,
                "q20": 2,
                "q50": 3,
                "prc": price,
            }
            row.update({feature: 1 for feature in features})
            rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            current, _ = load_characteristics(
                path, start_period=200001, end_period=200003, price_filter="current"
            )
            previous, _ = load_characteristics(
                path, start_period=200001, end_period=200003, price_filter="previous"
            )
        self.assertEqual(current.mm.tolist(), [2])
        self.assertEqual(previous.mm.tolist(), [3])

    def test_rolling_ols_produces_positive_spread(self):
        rng = np.random.default_rng(3)
        periods = [200001, 200002, 200003, 200004, 200005]
        frames = []
        ranks = []
        for period in periods:
            x = np.linspace(0.01, 0.99, 100)
            ret = 0.05 * x + rng.normal(scale=0.001, size=100)
            frames.append(
                pd.DataFrame(
                    {
                        "period": period,
                        "ret": ret,
                        "lme": np.arange(1, 101),
                        "q10": np.repeat(10, 100),
                    }
                )
            )
            ranks.append(x[:, None])
        data = pd.concat(frames, ignore_index=True)
        ranked = np.vstack(ranks)
        rows = run_rolling_ols(
            data,
            ranked,
            data.ret.to_numpy(),
            selected_features=["x"],
            backtest_start=200004,
            backtest_end=200005,
            window_months=3,
        )
        self.assertEqual(len(rows), 2)
        self.assertGreater(min(row["ew_long_short"] for row in rows), 0)
        self.assertGreater(summarize_returns(rows)["ew"]["sharpe_annualized"], 0)

    def test_identical_feature_sets_reuse_backtest(self):
        rng = np.random.default_rng(4)
        frames = []
        for period in [200001, 200002, 200003, 200004, 200005]:
            x = np.linspace(0.01, 0.99, 100)
            frames.append(
                pd.DataFrame(
                    {
                        "period": period,
                        "x": x,
                        "ret": 0.05 * x + rng.normal(scale=0.001, size=100),
                        "lme": np.arange(1, 101),
                        "q10": np.repeat(10, 100),
                    }
                )
            )
        data = pd.concat(frames, ignore_index=True)
        selection = LassoNetSelection(
            selected=["x"],
            selection_frequency={"x": 1.0},
            per_seed=[],
            train_period=[200001, 200003],
            validation_period=[200003, 200003],
        )
        selections = {"baseline": selection, "mild": selection}
        with patch(
            "run_lassonet_pipeline.run_rolling_ols",
            wraps=run_rolling_ols,
        ) as rolling:
            backtests, monthly, feature_sets = run_sensitivity_backtests(
                data,
                data.ret.to_numpy(),
                ["x"],
                selections,
                backtest_start=200004,
                backtest_end=200005,
                window_months=3,
                progress=False,
            )
        self.assertEqual(rolling.call_count, 2)  # all and size_q10, not per rule
        self.assertEqual(len(feature_sets), 1)
        self.assertEqual(
            backtests["baseline"]["feature_set_id"],
            backtests["mild"]["feature_set_id"],
        )
        self.assertTrue(backtests["mild"]["reused_backtest"])
        self.assertEqual(len(monthly), 8)


if __name__ == "__main__":
    unittest.main()

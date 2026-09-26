"""Time-aware LassoNet paths, resumable checkpoints, and selection rules."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PATH_CHECKPOINT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SelectionRule:
    tolerance: float
    consensus_threshold: float


SENSITIVITY_RULES: dict[str, SelectionRule] = {
    "baseline": SelectionRule(tolerance=0.01, consensus_threshold=0.60),
    "mild": SelectionRule(tolerance=0.0025, consensus_threshold=0.60),
    "best_loss": SelectionRule(tolerance=0.0, consensus_threshold=0.60),
    "broad": SelectionRule(tolerance=0.0, consensus_threshold=0.40),
}


@dataclass
class SeedSelection:
    seed: int
    lambda_: float
    validation_mse: float
    selected: list[str]
    selected_count: int
    path_length: int
    path_index: int | None = None
    validation_loss: float | None = None

    def sensitivity_dict(self) -> dict:
        """Use publication-friendly keys while retaining the legacy dataclass API."""

        return {
            "seed": self.seed,
            "path_index": self.path_index,
            "lambda": self.lambda_,
            "validation_loss_standardized": self.validation_loss,
            "validation_mse": self.validation_mse,
            "selected_count": self.selected_count,
            "selected_features": self.selected,
            "path_length": self.path_length,
        }


@dataclass
class LassoNetSelection:
    selected: list[str]
    selection_frequency: dict[str, float]
    per_seed: list[SeedSelection]
    train_period: list[int]
    validation_period: list[int]

    def to_dict(self) -> dict:
        """Legacy single-rule result format."""

        return {
            "selected": self.selected,
            "selection_frequency": self.selection_frequency,
            "per_seed": [asdict(item) for item in self.per_seed],
            "train_period": self.train_period,
            "validation_period": self.validation_period,
        }

    def sensitivity_dict(self) -> dict:
        return {
            "selected_features": self.selected,
            "selection_frequency": self.selection_frequency,
            "selected_checkpoint_by_seed": [
                item.sensitivity_dict() for item in self.per_seed
            ],
        }


@dataclass
class PathPoint:
    path_index: int
    lambda_: float
    validation_loss: float
    validation_mse: float
    selected_count: int
    selected_mask: list[bool]

    def checkpoint_dict(self) -> dict:
        return {
            "path_index": self.path_index,
            "lambda": self.lambda_,
            "validation_loss": self.validation_loss,
            "validation_mse": self.validation_mse,
            "selected_count": self.selected_count,
            "selected_mask": self.selected_mask,
        }


@dataclass
class SeedPath:
    seed: int
    path: list[PathPoint]


@dataclass
class LassoNetPaths:
    per_seed: list[SeedPath]
    train_period: list[int]
    validation_period: list[int]


def temporal_train_validation_mask(
    periods: np.ndarray,
    validation_months: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], tuple[int, int]]:
    """Split by whole months, keeping validation strictly after training."""

    unique = np.unique(np.asarray(periods, dtype=np.int32))
    if validation_months <= 0 or validation_months >= len(unique):
        raise ValueError("validation_months must leave at least one training month")
    validation = unique[-validation_months:]
    train = unique[:-validation_months]
    train_mask = np.isin(periods, train)
    validation_mask = np.isin(periods, validation)
    return (
        train_mask,
        validation_mask,
        (int(train[0]), int(train[-1])),
        (int(validation[0]), int(validation[-1])),
    )


def _item_validation_loss(item) -> float:
    if hasattr(item, "validation_loss"):
        return float(item.validation_loss)
    return float(item.val_loss)


def _item_selected_count(item) -> int:
    if hasattr(item, "selected_count"):
        return int(item.selected_count)
    return int(item.selected.sum())


def choose_sparse_checkpoint(
    path: Sequence,
    *,
    tolerance: float,
    min_selected: int,
    max_selected: int,
):
    """Choose the sparsest path point within a relative loss tolerance of the best.

    With ``tolerance=0`` only checkpoints attaining the minimum validation loss are
    eligible. Ties are resolved toward fewer features, then the larger lambda.
    """

    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    eligible = [
        item
        for item in path
        if min_selected <= _item_selected_count(item) <= max_selected
        and np.isfinite(_item_validation_loss(item))
    ]
    if not eligible:
        raise RuntimeError("LassoNet path contains no eligible non-empty checkpoint")
    best_loss = min(_item_validation_loss(item) for item in eligible)
    limit = best_loss * (1 + tolerance)
    near_best = [item for item in eligible if _item_validation_loss(item) <= limit]
    return min(
        near_best,
        key=lambda item: (
            _item_selected_count(item),
            _item_validation_loss(item),
            -float(item.lambda_),
        ),
    )


def _checkpoint_signature(
    *,
    names: list[str],
    seed_list: list[int],
    observations: int,
    period_bounds: list[int],
    validation_months: int,
    selection_tolerance: float,
    min_selected: int,
    max_selected: int,
    hidden_dims: tuple[int, ...],
    path_multiplier: float,
    batch_size: int,
    n_iters: tuple[int, int],
    gamma: float,
    gamma_skip: float,
    device: str,
    checkpoint_context: dict | None,
) -> dict:
    """Build the legacy v1 signature for compatibility with existing callers."""

    return {
        "version": 1,
        "feature_names": names,
        "seeds": seed_list,
        "observations": observations,
        "period_bounds": period_bounds,
        "validation_months": validation_months,
        "selection_tolerance": selection_tolerance,
        "min_selected": min_selected,
        "max_selected": max_selected,
        "hidden_dims": list(hidden_dims),
        "path_multiplier": path_multiplier,
        "batch_size": batch_size,
        "n_iters": list(n_iters),
        "gamma": gamma,
        "gamma_skip": gamma_skip,
        "device": device,
        "context": checkpoint_context or {},
    }


def _read_seed_checkpoint(path: Path, signature: dict) -> dict[int, SeedSelection]:
    """Read the legacy v1 selected-seed checkpoint."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        raise RuntimeError(
            "Checkpoint configuration does not match this run. "
            "Use a different --checkpoint path or restart without --resume."
        )
    results = {}
    for item in payload.get("completed_seeds", []):
        result = SeedSelection(**item)
        results[result.seed] = result
    return results


def _write_seed_checkpoint(
    path: Path,
    signature: dict,
    results: dict[int, SeedSelection],
) -> None:
    """Write the legacy v1 format atomically (kept for API/test compatibility)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "completed_seeds": [
            asdict(results[seed]) for seed in signature["seeds"] if seed in results
        ],
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _path_checkpoint_signature(
    *,
    names: list[str],
    seed_list: list[int],
    observations: int,
    period_bounds: list[int],
    validation_months: int,
    hidden_dims: tuple[int, ...],
    path_multiplier: float,
    batch_size: int,
    n_iters: tuple[int, int],
    gamma: float,
    gamma_skip: float,
    device: str,
    checkpoint_context: dict | None,
) -> dict:
    """Signature only training inputs; post-training selection rules are excluded."""

    return {
        "schema_version": PATH_CHECKPOINT_SCHEMA_VERSION,
        "feature_names": names,
        "seeds": seed_list,
        "observations": observations,
        "period_bounds": period_bounds,
        "validation_months": validation_months,
        "hidden_dims": list(hidden_dims),
        "path_multiplier": path_multiplier,
        "batch_size": batch_size,
        "n_iters": list(n_iters),
        "gamma": gamma,
        "gamma_skip": gamma_skip,
        "device": device,
        "context": checkpoint_context or {},
    }


def _legacy_signature_matches(legacy: dict, current: dict) -> bool:
    """Check common training fields while ignoring legacy rule parameters."""

    common = (
        "feature_names",
        "seeds",
        "observations",
        "period_bounds",
        "validation_months",
        "hidden_dims",
        "path_multiplier",
        "batch_size",
        "n_iters",
        "gamma",
        "gamma_skip",
        "device",
        "context",
    )
    return all(legacy.get(key) == current.get(key) for key in common)


def _deserialize_path_point(item: dict, feature_count: int) -> PathPoint:
    mask = [bool(value) for value in item["selected_mask"]]
    if len(mask) != feature_count:
        raise RuntimeError("Checkpoint selected_mask has the wrong feature count")
    selected_count = int(item["selected_count"])
    if selected_count != sum(mask):
        raise RuntimeError("Checkpoint selected_count does not match selected_mask")
    return PathPoint(
        path_index=int(item["path_index"]),
        lambda_=float(item["lambda"]),
        validation_loss=float(item["validation_loss"]),
        validation_mse=float(item["validation_mse"]),
        selected_count=selected_count,
        selected_mask=mask,
    )


def _read_path_checkpoint(
    path: Path,
    signature: dict,
) -> tuple[dict[int, SeedPath], bool]:
    """Read v2 full paths; return ``legacy=True`` for compatible v1 files."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version is None:
        legacy_signature = payload.get("signature", {})
        if not _legacy_signature_matches(legacy_signature, signature):
            raise RuntimeError(
                "Legacy checkpoint configuration does not match this run. "
                "Use a different --checkpoint path or restart without --resume."
            )
        return {}, True
    if schema_version != PATH_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported checkpoint schema_version={schema_version}")
    if payload.get("signature") != signature:
        raise RuntimeError(
            "Checkpoint configuration does not match this run. "
            "Use a different --checkpoint path or restart without --resume."
        )

    feature_count = len(signature["feature_names"])
    results: dict[int, SeedPath] = {}
    for item in payload.get("completed_seeds", []):
        seed = int(item["seed"])
        if seed in results:
            raise RuntimeError(f"Checkpoint contains duplicate seed {seed}")
        points = [
            _deserialize_path_point(point, feature_count) for point in item.get("path", [])
        ]
        if not points:
            raise RuntimeError(f"Checkpoint seed {seed} has no path points")
        if [point.path_index for point in points] != list(range(len(points))):
            raise RuntimeError(f"Checkpoint seed {seed} has invalid path indices")
        results[seed] = SeedPath(seed=seed, path=points)
    return results, False


def _write_path_checkpoint(
    path: Path,
    signature: dict,
    results: Mapping[int, SeedPath],
) -> None:
    """Atomically save every completed seed's light-weight regularization path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": PATH_CHECKPOINT_SCHEMA_VERSION,
        "signature": signature,
        "completed_seeds": [
            {
                "seed": seed,
                "path": [point.checkpoint_dict() for point in results[seed].path],
            }
            for seed in signature["seeds"]
            if seed in results
        ],
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def fit_lassonet_paths(
    X: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    feature_names: Sequence[str],
    *,
    seeds: Iterable[int],
    validation_months: int = 60,
    hidden_dims: tuple[int, ...] = (32, 16),
    path_multiplier: float = 1.08,
    batch_size: int = 4096,
    n_iters: tuple[int, int] = (100, 30),
    gamma: float = 1e-4,
    gamma_skip: float = 1e-4,
    device: str = "auto",
    verbose: int = 1,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    checkpoint_context: dict | None = None,
) -> LassoNetPaths:
    """Train one LassoNet path per seed and checkpoint only light path metadata."""

    try:
        import torch
        from lassonet import LassoNetRegressor
    except ImportError as exc:
        raise RuntimeError(
            "LassoNet dependencies are missing. Install requirements-lassonet.txt."
        ) from exc

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    periods = np.asarray(periods, dtype=np.int32)
    names = list(feature_names)
    if X.shape != (len(y), len(names)):
        raise ValueError("X, y, and feature_names have inconsistent shapes")

    seed_list = list(seeds)
    if not seed_list:
        raise ValueError("At least one seed is required")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("Seeds must be unique")

    train_mask, val_mask, train_period, val_period = temporal_train_validation_mask(
        periods, validation_months
    )
    X_train = X[train_mask]
    X_val = X[val_mask]
    y_train = y[train_mask]
    y_val = y[val_mask]

    x_mean = X_train.mean(axis=0)
    x_scale = X_train.std(axis=0)
    x_scale[x_scale == 0] = 1
    y_mean = float(y_train.mean())
    y_scale = float(y_train.std()) or 1.0
    X_train = (X_train - x_mean) / x_scale
    X_val = (X_val - x_mean) / x_scale
    y_train = (y_train - y_mean) / y_scale
    y_val = (y_val - y_mean) / y_scale

    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    signature = _path_checkpoint_signature(
        names=names,
        seed_list=seed_list,
        observations=len(y),
        period_bounds=[int(periods.min()), int(periods.max())],
        validation_months=validation_months,
        hidden_dims=hidden_dims,
        path_multiplier=path_multiplier,
        batch_size=batch_size,
        n_iters=n_iters,
        gamma=gamma,
        gamma_skip=gamma_skip,
        device=device,
        checkpoint_context=checkpoint_context,
    )
    results_by_seed: dict[int, SeedPath] = {}
    if resume and checkpoint is not None and checkpoint.exists():
        results_by_seed, legacy = _read_path_checkpoint(checkpoint, signature)
        if legacy:
            print(
                "Legacy v1 checkpoint contains only chosen features, not full paths. "
                "All seeds will be trained once and the file will be upgraded to schema v2.",
                flush=True,
            )
        unknown = set(results_by_seed) - set(seed_list)
        if unknown:
            raise RuntimeError(f"Checkpoint contains unexpected seeds: {sorted(unknown)}")
        print(
            f"Resuming from {checkpoint}: {len(results_by_seed)}/{len(seed_list)} "
            "full seed paths complete",
            flush=True,
        )
        for seed in seed_list:
            if seed in results_by_seed:
                print(
                    f"Restored seed {seed}: {len(results_by_seed[seed].path)} path points",
                    flush=True,
                )

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None
    progress_enabled = verbose > 0 and tqdm is not None
    seed_bar = (
        tqdm(
            total=len(seed_list),
            initial=len(results_by_seed),
            desc="LassoNet seed paths",
            unit="seed",
            dynamic_ncols=True,
        )
        if progress_enabled
        else None
    )

    for seed in seed_list:
        if seed in results_by_seed:
            continue
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = LassoNetRegressor(
            hidden_dims=hidden_dims,
            path_multiplier=path_multiplier,
            batch_size=batch_size,
            n_iters=n_iters,
            patience=None,
            val_size=0,
            gamma=gamma,
            gamma_skip=gamma_skip,
            device=None if device == "auto" else device,
            random_state=seed,
            torch_seed=seed,
            verbose=max(0, verbose - 1),
        )
        path_bar = (
            tqdm(
                desc=f"seed {seed}: lambda path",
                unit="step",
                leave=False,
                dynamic_ncols=True,
            )
            if progress_enabled
            else None
        )
        previous_path_length = 0

        def update_path_progress(_model, history):
            nonlocal previous_path_length
            if path_bar is None or len(history) <= previous_path_length:
                return
            path_bar.update(len(history) - previous_path_length)
            previous_path_length = len(history)
            item = history[-1]
            path_bar.set_postfix(
                lambda_=f"{float(item.lambda_):.3g}",
                selected=int(item.selected.sum()),
            )

        try:
            raw_path = model.path(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                return_state_dicts=False,
                callback=update_path_progress,
            )
        finally:
            if path_bar is not None:
                path_bar.close()

        points: list[PathPoint] = []
        for path_index, item in enumerate(raw_path):
            selected = item.selected
            if hasattr(selected, "detach"):
                selected = selected.detach().cpu().numpy()
            mask = np.asarray(selected, dtype=bool).tolist()
            points.append(
                PathPoint(
                    path_index=path_index,
                    lambda_=float(item.lambda_),
                    validation_loss=float(item.val_loss),
                    validation_mse=float(item.val_loss) * y_scale**2,
                    selected_count=int(sum(mask)),
                    selected_mask=mask,
                )
            )
        result = SeedPath(seed=seed, path=points)
        results_by_seed[seed] = result
        if checkpoint is not None:
            _write_path_checkpoint(checkpoint, signature, results_by_seed)
        if seed_bar is not None:
            seed_bar.update(1)
            seed_bar.set_postfix(seed=seed, path_points=len(points))
        print(
            f"Seed {seed} path complete: {len(points)} points; "
            f"checkpoint={checkpoint if checkpoint is not None else 'disabled'}",
            flush=True,
        )

    if seed_bar is not None:
        seed_bar.close()

    return LassoNetPaths(
        per_seed=[results_by_seed[seed] for seed in seed_list],
        train_period=list(train_period),
        validation_period=list(val_period),
    )


def apply_selection_rule(
    paths: LassoNetPaths,
    feature_names: Sequence[str],
    *,
    tolerance: float,
    consensus_threshold: float,
    min_selected: int = 1,
    max_selected: int | None = None,
) -> LassoNetSelection:
    """Apply a rule to already-trained paths; this function never trains a model."""

    names = list(feature_names)
    if not 0 < consensus_threshold <= 1:
        raise ValueError("consensus_threshold must be in (0, 1]")
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    if max_selected is None:
        max_selected = len(names)
    if not paths.per_seed:
        raise ValueError("At least one completed seed path is required")

    counts = np.zeros(len(names), dtype=np.int64)
    seed_results: list[SeedSelection] = []
    for seed_path in paths.per_seed:
        chosen = choose_sparse_checkpoint(
            seed_path.path,
            tolerance=tolerance,
            min_selected=min_selected,
            max_selected=max_selected,
        )
        mask = np.asarray(chosen.selected_mask, dtype=bool)
        if len(mask) != len(names):
            raise RuntimeError("Path feature mask is incompatible with feature_names")
        counts += mask
        selected = [name for name, keep in zip(names, mask) if keep]
        seed_results.append(
            SeedSelection(
                seed=seed_path.seed,
                lambda_=chosen.lambda_,
                validation_mse=chosen.validation_mse,
                selected=selected,
                selected_count=chosen.selected_count,
                path_length=len(seed_path.path),
                path_index=chosen.path_index,
                validation_loss=chosen.validation_loss,
            )
        )

    frequencies = counts / len(paths.per_seed)
    selected = [
        name for name, frequency in zip(names, frequencies) if frequency >= consensus_threshold
    ]
    if not selected:
        raise RuntimeError(
            "No feature met the consensus threshold. Inspect per-seed selections or lower it."
        )
    frequency_map = {
        name: float(frequency)
        for name, frequency in sorted(
            zip(names, frequencies), key=lambda pair: (-pair[1], pair[0])
        )
    }
    return LassoNetSelection(
        selected=selected,
        selection_frequency=frequency_map,
        per_seed=seed_results,
        train_period=paths.train_period,
        validation_period=paths.validation_period,
    )


def apply_selection_rules(
    paths: LassoNetPaths,
    feature_names: Sequence[str],
    rules: Mapping[str, SelectionRule] = SENSITIVITY_RULES,
    *,
    min_selected: int = 1,
    max_selected: int | None = None,
) -> dict[str, LassoNetSelection]:
    """Apply several post-training rules to the same in-memory paths."""

    return {
        name: apply_selection_rule(
            paths,
            feature_names,
            tolerance=rule.tolerance,
            consensus_threshold=rule.consensus_threshold,
            min_selected=min_selected,
            max_selected=max_selected,
        )
        for name, rule in rules.items()
    }


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    periods: np.ndarray,
    feature_names: Sequence[str],
    *,
    seeds: Iterable[int],
    validation_months: int = 60,
    consensus_threshold: float = 0.6,
    selection_tolerance: float = 0.01,
    min_selected: int = 1,
    max_selected: int | None = None,
    hidden_dims: tuple[int, ...] = (32, 16),
    path_multiplier: float = 1.08,
    batch_size: int = 4096,
    n_iters: tuple[int, int] = (100, 30),
    gamma: float = 1e-4,
    gamma_skip: float = 1e-4,
    device: str = "auto",
    verbose: int = 1,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    checkpoint_context: dict | None = None,
) -> LassoNetSelection:
    """Backward-compatible single-rule API built on the reusable full paths."""

    paths = fit_lassonet_paths(
        X,
        y,
        periods,
        feature_names,
        seeds=seeds,
        validation_months=validation_months,
        hidden_dims=hidden_dims,
        path_multiplier=path_multiplier,
        batch_size=batch_size,
        n_iters=n_iters,
        gamma=gamma,
        gamma_skip=gamma_skip,
        device=device,
        verbose=verbose,
        checkpoint_path=checkpoint_path,
        resume=resume,
        checkpoint_context=checkpoint_context,
    )
    return apply_selection_rule(
        paths,
        feature_names,
        tolerance=selection_tolerance,
        consensus_threshold=consensus_threshold,
        min_selected=min_selected,
        max_selected=max_selected,
    )

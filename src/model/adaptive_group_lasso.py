import numpy as np

from src.model.baseline_model import build_matrix
from src.selection.bic import select_lambda_by_bic


def make_adaptive_weights(beta, group_size):
    groups = beta.reshape(-1, group_size)
    norms = np.linalg.norm(groups, axis=1)

    weights = np.full(len(norms), np.inf)
    alive = norms > 1e-8
    weights[alive] = 1 / norms[alive]

    return weights


def selected_columns(beta, group_size):
    groups = beta.reshape(-1, group_size)
    active_groups = np.linalg.norm(groups, axis=1) > 1e-8

    columns = []
    for group_index in np.flatnonzero(active_groups):
        start = group_index * group_size
        end = (group_index + 1) * group_size
        columns.extend(range(start, end))

    return np.array(columns)


def fit_adaptive_group_lasso(X, y, knots, lambda_grid, lr, n_iter):
    group_size = len(knots) + 3
    n_groups = X.shape[1]

    weights_1 = np.ones(n_groups)

    beta_1, lambda_1, bic_1 = select_lambda_by_bic(
        X, y, knots, lambda_grid, lr, n_iter, weights_1
    )

    weights_2 = make_adaptive_weights(beta_1, group_size)

    # Stage 2: weighted group LASSO
    beta_2, lambda_2, bic_2 = select_lambda_by_bic(
        X, y, knots, lambda_grid, lr, n_iter, weights_2
    )

    columns = selected_columns(beta_2, group_size)
    X_spline = build_matrix(X, knots)
    if not len(columns):
        return {
            "beta_stage_1": beta_1,
            "lambda_stage_1": lambda_1,
            "bic_stage_1": bic_1,
            "adaptive_weights": weights_2,
            "beta_stage_2": beta_2,
            "lambda_stage_2": lambda_2,
            "bic_stage_2": bic_2,
            "selected_columns": columns,
            "intercept_ols": float(np.mean(y)),
            "beta_ols": np.array([]),
        }
    X_selected = X_spline[:, columns]

    intercept_ols = float(np.mean(y))
    beta_ols = np.linalg.lstsq(X_selected, y - intercept_ols, rcond=None)[0]

    return {
        "beta_stage_1": beta_1,
        "lambda_stage_1": lambda_1,
        "bic_stage_1": bic_1,
        "adaptive_weights": weights_2,
        "beta_stage_2": beta_2,
        "lambda_stage_2": lambda_2,
        "bic_stage_2": bic_2,
        "selected_columns": columns,
        "intercept_ols": intercept_ols,
        "beta_ols": beta_ols,
    }

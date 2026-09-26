import numpy as np
from src.model.group_lasso import LassoSplineModel
from src.model.baseline_model import build_matrix


def bic(beta, X, y, group_size):
    mse = np.mean((y - X @ beta) ** 2)

    groups =  beta.reshape(-1, group_size)
    active_groups = np.linalg.norm(groups, axis=1) > 1e-8

    df = active_groups.sum() * group_size

    return X.shape[0] * np.log(mse) + df * np.log(X.shape[0])



def select_lambda_by_bic(X, y, knots, lambda_grid, lr, n_iter, weights=None):
    group_size = len(knots) + 3
    X_spline = build_matrix(X, knots)
    n_groups = X_spline.shape[1] // group_size
    if weights is None:
        weights = np.ones(n_groups)

    best_bic = np.inf
    best_beta = None
    best_lambda = None

    for lamb in lambda_grid:
        model = LassoSplineModel(knots)
        model.fit(X, y, lr=lr, lamb=lamb, n_iter=n_iter, weights=weights)

        score = bic(model.beta, X_spline, y, group_size)

        if score < best_bic:
            best_bic = score
            best_beta = model.beta.copy()
            best_lambda = lamb

    return best_beta, best_lambda, best_bic

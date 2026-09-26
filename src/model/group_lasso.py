import numpy as np
from src.model.baseline_model import build_matrix


def train_one_step(beta, X, y, lr, lamb, group_size, weights=None):
    n_groups = X.shape[1] // group_size

    if weights is None:
        weights = np.ones(n_groups)

    grad = X.T @ (X @ beta - y) / X.shape[0]
    z = beta - lr * grad
    new_beta = []

    for s, weight in enumerate(weights):
        start = s * group_size
        end = (s + 1) * group_size
        z_s = z[start:end]

        if not np.isfinite(weight):
            new_beta.append(np.zeros(group_size))
            continue

        norm = np.linalg.norm(z_s)
        shrink = max(0, 1 - lr * lamb * weight / norm) if norm > 0 else 0
        new_beta.append(shrink * z_s)

    return np.concatenate(new_beta)


def train(beta, X, y, lr, lamb, group_size, n_iter, weights=None):
    n_groups = X.shape[1] // group_size
    if weights is None:
        weights = np.ones(n_groups)

    train_loss = []
    for step in range(n_iter):
        beta = train_one_step(beta, X, y, lr, lamb, group_size, weights)
        data_loss = 0.5 * np.mean((X @ beta - y) ** 2)
        penalty = 0
        for s, weight in enumerate(weights):
            if np.isfinite(weight):
                start = group_size * s
                end = group_size * (s + 1)
                penalty += lamb * weight * np.linalg.norm(beta[start:end], 2)
        train_loss.append(data_loss + penalty)

    return beta, train_loss


class LassoSplineModel:
    def __init__(self, knots):
        self.knots = knots
        self.beta = None
        self.train_loss = []

    def fit(self, X, y, lr, lamb, n_iter, weights=None):
        X = build_matrix(X, self.knots)
        beta = np.zeros(X.shape[1])
        group_size = len(self.knots) + 3
        self.beta, self.train_loss = train(
            beta, X, y, lr, lamb, group_size, n_iter, weights
        )

    def predict(self, X):
        X = build_matrix(X, self.knots)
        return X @ self.beta

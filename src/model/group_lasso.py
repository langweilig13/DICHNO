import numpy as np
from src.model.baseline_model import build_matrix


def train_one_step(beta, X, y, lr, lamb, knots):
    grad = 1 / X.shape[0] * X.T @ (X @ beta - y)
    z = beta - lr * grad
    new_beta = []
    for s in range(X.shape[1] // (len(knots) + 3)):
        start = (len(knots) + 3) * s
        end = (len(knots) + 3) * (s + 1)
        z_s = z[start:end]

        nr = np.linalg.norm(z_s, 2)
        if nr == 0:
            coef = 0
        else:
            coef = max(0, 1 - lr * lamb / nr)
        new_beta.append(coef * z_s)

    return np.concatenate(new_beta)


def train(beta, X, y, lr, lamb, knots, n_iter):
    train_loss = []
    for step in range(n_iter):
        beta = train_one_step(beta, X, y, lr, lamb, knots)
        data_loss = 0.5 * np.mean((X @ beta - y) ** 2)
        penalty = 0
        for s in range(X.shape[1] // (len(knots) + 3)):
            start = (len(knots) + 3) * s
            end = (len(knots) + 3) * (s + 1)
            penalty += lamb * np.linalg.norm(beta[start:end], 2)
        train_loss.append(data_loss + penalty)

    return beta, train_loss


class LassoSplineModel:
    def __init__(self, knots):
        self.knots = knots
        self.beta = None
        self.train_loss = []

    def fit(self, X, y, lr, lamb, n_iter):
        X = build_matrix(X, self.knots)
        beta = np.zeros(X.shape[1])
        self.beta, self.train_loss = train(beta, X, y, lr, lamb, self.knots, n_iter)

    def predict(self, X):
        X = build_matrix(X, self.knots)
        return X @ self.beta

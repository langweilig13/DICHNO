import numpy as np
from src.transforms.ranking import rank_transform

def quadratic_spline_basis(x: list, knots:list) -> np.ndarray:
    """" x: list of n values of feature j"""
    x = np.asarray(x)
    x = rank_transform(x)

    features = [np.ones(x.shape[0]), x, x**2] # 1, x, x^2
    for knot in knots:
        features.append(np.maximum(x - knot, 0)**2)
    return np.column_stack(features)

def build_matrix(X: np.ndarray, knots: list) -> np.ndarray:
    matrices = []
    for j in range((X.shape[1])):
        x_j = X[:, j]
        basis_j = quadratic_spline_basis(x_j, knots)
        matrices.append(basis_j)
    return np.column_stack(matrices)


class BaselineSplineModel:
    def __init__(self, knots):
        self.knots = knots
        self.beta = None
        self.train_loss = []

    def fit(self, X, y):
        X = build_matrix(X, self.knots)
        self.beta  = np.linalg.lstsq(X, y, rcond=None)[0]
        self.train_loss.append(np.mean((X @ self.beta - y)**2))

    def predict(self, X):
        X = build_matrix(X, self.knots)
        return X @ self.beta










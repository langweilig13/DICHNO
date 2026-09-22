import numpy as np

def rank_transform(X: np.ndarray) -> np.ndarray:
    sorted_x = sorted(X)
    return [(sorted_x.index(value) + 1) / (len(X) + 1) for value in X]

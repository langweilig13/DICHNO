import numpy as np

def quadratic_spline_basis(x: list, knots:list) -> list:
    x = np.asarray(x)

    features = [x, x**2]
    for knot in knots:
        features.append(np.maximum(x - knot, 0)**2)
    return features


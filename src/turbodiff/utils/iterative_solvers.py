import numpy as np


def seidel(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    One Gauss–Seidel iteration step.
    a: (n, n) coefficient matrix
    x: (n,)  current solution (updated in place)
    b: (n,)  right-hand side
    """
    n = len(a)
    for j in range(n):
        # Compute dot product excluding the diagonal element -> diagonal becomes subject of equation
        d = b[j] - np.dot(a[j, :j], x[:j]) - np.dot(a[j, j + 1 :], x[j + 1 :])
        x[j] = d / a[j, j]
    return x

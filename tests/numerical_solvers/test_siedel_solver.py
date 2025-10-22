from turbodiff.utils.iterative_solvers import seidel
import numpy as np


def test_siedel_solver():
    A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)
    b = np.array([4, 7, 3], dtype=float)
    x = np.zeros_like(b)

    for _ in range(10):
        x = seidel(A, x, b)

    err = x - np.array([0.5, 1, 0.5])
    assert np.linalg.norm(err) < 1e-5

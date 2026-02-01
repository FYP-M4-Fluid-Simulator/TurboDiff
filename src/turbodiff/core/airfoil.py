"""
CST (Class-Shape Transformation) airfoil parametrization.

This module provides differentiable airfoil geometry generation using
Bernstein polynomials, enabling gradient-based shape optimization.
"""

import jax.numpy as jnp
from jax import Array
from typing import Tuple
from scipy.special import comb as scipy_comb


def bernstein_polynomial(n: int, i: int, x: Array) -> Array:
    """
    Compute the i-th Bernstein basis polynomial of degree n.

    B_{i,n}(x) = C(n,i) * x^i * (1-x)^(n-i)

    Args:
        n: Polynomial degree
        i: Index (0 <= i <= n)
        x: Evaluation points, shape (num_points,)

    Returns:
        Bernstein polynomial values, same shape as x
    """
    # Binomial coefficient (computed at trace time, constant for JIT)
    binom_coeff = scipy_comb(n, i, exact=True)
    return binom_coeff * (x**i) * ((1.0 - x) ** (n - i))


def cst_class_function(x: Array) -> Array:
    """
    Standard CST class function for airfoils.

    C(x) = sqrt(x) * (1 - x)

    This gives the characteristic rounded leading edge and
    sharp trailing edge of airfoils.

    Args:
        x: Normalized chord positions in [0, 1], shape (num_points,)

    Returns:
        Class function values, same shape as x
    """
    return jnp.sqrt(x) * (1.0 - x)


def cst_shape_function(x: Array, weights: Array) -> Array:
    """
    CST shape function using Bernstein polynomial basis.

    S(x) = sum_{i=0}^{n} w_i * B_{i,n}(x)

    Args:
        x: Normalized chord positions in [0, 1], shape (num_points,)
        weights: Bernstein coefficients, shape (n+1,)

    Returns:
        Shape function values, same shape as x
    """
    n = weights.shape[0] - 1

    # Vectorized Bernstein computation
    # Create index array [0, 1, ..., n]
    i_vals = jnp.arange(n + 1)

    # Binomial coefficients (constant array)
    binom_coeffs = jnp.array([scipy_comb(n, i, exact=True) for i in range(n + 1)])

    # Broadcast: x is (num_points,), i_vals is (n+1,)
    # x[:, None] -> (num_points, 1), i_vals[None, :] -> (1, n+1)
    # Result: (num_points, n+1)
    x_power_i = x[:, None] ** i_vals[None, :]
    one_minus_x_power = (1.0 - x[:, None]) ** (n - i_vals[None, :])

    # Bernstein basis: (num_points, n+1)
    B = binom_coeffs[None, :] * x_power_i * one_minus_x_power

    # Weighted sum over polynomial terms
    # (num_points, n+1) @ (n+1,) -> (num_points,)
    return B @ weights


def generate_cst_coords(
    weights_upper: Array,
    weights_lower: Array,
    num_points: int = 100,
    cosine_spacing: bool = True,
) -> Tuple[Array, Array, Array]:
    """
    Generate airfoil coordinates from CST parameters.

    Args:
        weights_upper: Upper surface Bernstein weights, shape (n+1,)
        weights_lower: Lower surface Bernstein weights, shape (n+1,)
        num_points: Number of points per surface
        cosine_spacing: Use cosine spacing for better LE/TE resolution

    Returns:
        Tuple of (x, y_upper, y_lower), each shape (num_points,)
    """
    if cosine_spacing:
        # Cosine spacing: finer resolution at leading/trailing edges
        beta = jnp.linspace(0.0, jnp.pi, num_points)
        x = 0.5 * (1.0 - jnp.cos(beta))
    else:
        x = jnp.linspace(0.0, 1.0, num_points)

    # Class function
    C = cst_class_function(x)

    # Shape functions
    S_upper = cst_shape_function(x, weights_upper)
    S_lower = cst_shape_function(x, weights_lower)

    # Final coordinates: y = C(x) * S(x)
    y_upper = C * S_upper
    y_lower = C * S_lower

    return x, y_upper, y_lower


def cst_to_closed_contour(
    x: Array, y_upper: Array, y_lower: Array
) -> Tuple[Array, Array]:
    """
    Combine upper and lower surfaces into a closed contour.

    Goes from trailing edge along upper surface to leading edge,
    then back along lower surface to trailing edge.

    Args:
        x: Chord positions, shape (num_points,)
        y_upper: Upper surface y-coordinates
        y_lower: Lower surface y-coordinates

    Returns:
        (x_contour, y_contour) for closed airfoil shape
    """
    x_contour = jnp.concatenate([x, x[::-1]])
    y_contour = jnp.concatenate([y_upper, y_lower[::-1]])
    return x_contour, y_contour


def thickness_at_x(y_upper: Array, y_lower: Array) -> Array:
    """
    Compute thickness distribution along chord.

    Args:
        y_upper: Upper surface y-coordinates
        y_lower: Lower surface y-coordinates

    Returns:
        Thickness at each chordwise location
    """
    return y_upper - y_lower


def max_thickness(y_upper: Array, y_lower: Array) -> float:
    """
    Compute maximum thickness of airfoil.

    Args:
        y_upper: Upper surface y-coordinates
        y_lower: Lower surface y-coordinates

    Returns:
        Maximum thickness value (scalar)
    """
    thickness = thickness_at_x(y_upper, y_lower)
    return jnp.max(thickness)


__all__ = [
    "bernstein_polynomial",
    "cst_class_function",
    "cst_shape_function",
    "generate_cst_coords",
    "cst_to_closed_contour",
    "thickness_at_x",
    "max_thickness",
]

"""
Class Shape Transformation (CST) parameterized airfoil generator.
"""

from typing import Callable, Union, List
import math

import jax
import jax.numpy as jnp
from jax import Array


def nCr_float(n: int, k: int) -> float:
    """Compute the binomial coefficient as a float."""
    return float(math.comb(n, k))


def create_cst_sdf(
    weights_upper: Union[List[float], Array],
    weights_lower: Union[List[float], Array],
    chord: float,
    cell_size: float,
    center_x: float = 0.0,
    center_y: float = 0.0,
    dz_te_u: float = 0.0,
    dz_te_l: float = 0.0,
    aoa_deg: float = 0.0,
) -> Callable[[Array, Array], Array]:
    """
    Create a Signed Distance Function for a CST airfoil.

    Args:
        weights_upper: Weights for the upper surface
        weights_lower: Weights for the lower surface (usually negative)
        chord: Chord length of the airfoil in physical units (m)
        cell_size: Grid cell size
        center_x: x-coordinate of the leading edge in physical units
        center_y: y-coordinate of the leading edge in physical units
        dz_te_u: Upper trailing edge gap
        dz_te_l: Lower trailing edge gap (usually negative)
        aoa_deg: Angle of attack in degrees

    Returns:
        sdf_fn: A function sdf(i_grid, j_grid) that returns approximate
                signed distances, negative inside the airfoil.
    """
    w_u = jnp.array(weights_upper)
    w_l = jnp.array(weights_lower)

    aoa_rad = math.radians(aoa_deg)
    cos_a = math.cos(aoa_rad)
    sin_a = math.sin(aoa_rad)

    N_u = len(weights_upper) - 1
    N_l = len(weights_lower) - 1

    K_u = jnp.array([nCr_float(N_u, i) for i in range(N_u + 1)])
    K_l = jnp.array([nCr_float(N_l, i) for i in range(N_l + 1)])

    def compute_shape_func(xi: Array, weights: Array, K: Array, N: int) -> Array:
        S = jnp.zeros_like(xi)
        for i in range(N + 1):
            term = weights[i] * K[i] * (xi**i) * ((1.0 - xi) ** (N - i))
            S += term
        return S

    @jax.jit
    def sdf(i_grid: Array, j_grid: Array) -> Array:
        # Convert grid indices to physical coordinates
        # Notice that typically j_grid is exactly the x-axis, i_grid is the y-axis
        cx = j_grid * cell_size
        cy = i_grid * cell_size

        # Translate relative to leading edge
        cx_rel = cx - center_x
        cy_rel = cy - center_y

        # Rotate INTO body frame (-aoa transforms point into unrotated frame)
        cx_body = cx_rel * cos_a - cy_rel * sin_a
        cy_body = cx_rel * sin_a + cy_rel * cos_a

        # Normalized chord position
        xi = cx_body / chord
        xi_c = jnp.clip(xi, 0.0, 1.0)

        # Class function
        C = jnp.sqrt(xi_c) * (1.0 - xi_c)

        # Shape functions
        S_u = compute_shape_func(xi_c, w_u, K_u, N_u)
        S_l = compute_shape_func(xi_c, w_l, K_l, N_l)

        # y coordinates of the surfaces in body frame
        y_u_body = (C * S_u + xi_c * dz_te_u) * chord
        y_l_body = (C * S_l + xi_c * dz_te_l) * chord

        # Vertical distance bounds (negative means inside)
        dist_u = cy_body - y_u_body  # Distance from upper surface
        dist_l = y_l_body - cy_body  # Distance from lower surface
        vert_dist = jnp.maximum(dist_u, dist_l)

        # Horizontal distance bounds
        dist_le = -cx_body
        dist_te = cx_body - chord
        horiz_dist = jnp.maximum(dist_le, dist_te)

        # The approximate SDF is the maximum of all boundary distances.
        # This creates a bounding box-like signed distance field, which is
        # highly effective for marking Immersed Boundary Method solid masks.
        approx_sdf = jnp.maximum(vert_dist, horiz_dist)

        return approx_sdf

    return sdf

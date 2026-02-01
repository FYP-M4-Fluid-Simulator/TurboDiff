"""
Tests for CST airfoil parametrization module.
"""

import pytest
import jax.numpy as jnp
from jax import grad
from turbodiff.core.airfoil import (
    bernstein_polynomial,
    cst_class_function,
    cst_shape_function,
    generate_cst_coords,
)


class TestBernsteinPolynomial:
    """Test Bernstein polynomial basis functions."""

    def test_partition_of_unity(self):
        """Bernstein polynomials should sum to 1 at any point."""
        n = 5
        x = jnp.linspace(0.0, 1.0, 50)

        # Sum all Bernstein polynomials
        total = sum(bernstein_polynomial(n, i, x) for i in range(n + 1))

        assert jnp.allclose(total, jnp.ones_like(x), atol=1e-6)

    def test_boundary_values(self):
        """Check Bernstein polynomial values at x=0 and x=1."""
        n = 4
        x = jnp.array([0.0, 1.0])

        # B_{0,n}(0) = 1, B_{n,n}(1) = 1
        assert jnp.isclose(bernstein_polynomial(n, 0, x)[0], 1.0)
        assert jnp.isclose(bernstein_polynomial(n, n, x)[1], 1.0)

        # B_{0,n}(1) = 0, B_{n,n}(0) = 0
        assert jnp.isclose(bernstein_polynomial(n, 0, x)[1], 0.0)
        assert jnp.isclose(bernstein_polynomial(n, n, x)[0], 0.0)


class TestCSTFunctions:
    """Test CST class and shape functions."""

    def test_class_function_endpoints(self):
        """Class function should be 0 at leading and trailing edges."""
        x = jnp.array([0.0, 1.0])
        C = cst_class_function(x)

        assert jnp.isclose(C[0], 0.0)  # LE
        assert jnp.isclose(C[1], 0.0)  # TE

    def test_class_function_positive_interior(self):
        """Class function should be positive in interior."""
        x = jnp.linspace(0.01, 0.99, 50)
        C = cst_class_function(x)

        assert jnp.all(C > 0)

    def test_shape_function_with_constant_weights(self):
        """Shape function with constant weights should equal that constant."""
        x = jnp.linspace(0.0, 1.0, 50)
        weights = jnp.ones(6) * 0.5  # n=5, constant weights of 0.5

        S = cst_shape_function(x, weights)

        # Due to partition of unity, S(x) = sum(w_i * B_i) = 0.5 * sum(B_i) = 0.5
        assert jnp.allclose(S, 0.5 * jnp.ones_like(x), atol=1e-6)


class TestGenerateCSTCoords:
    """Test airfoil coordinate generation."""

    def test_output_shapes(self):
        """Check coordinate array shapes."""
        weights_upper = jnp.array([0.15, 0.15, 0.15, 0.15])
        weights_lower = jnp.array([-0.15, -0.15, -0.15, -0.15])

        x, y_u, y_l = generate_cst_coords(weights_upper, weights_lower, num_points=100)

        assert x.shape == (100,)
        assert y_u.shape == (100,)
        assert y_l.shape == (100,)

    def test_upper_above_lower(self):
        """Upper surface should be above lower surface for positive weights."""
        weights_upper = jnp.array([0.2, 0.2, 0.2, 0.2])
        weights_lower = jnp.array([-0.1, -0.1, -0.1, -0.1])

        x, y_u, y_l = generate_cst_coords(weights_upper, weights_lower)

        # Exclude endpoints where both are 0
        interior = (x > 0.01) & (x < 0.99)
        assert jnp.all(y_u[interior] > y_l[interior])

    def test_endpoints_at_zero(self):
        """Leading and trailing edges should be at y=0."""
        weights_upper = jnp.array([0.15, 0.15, 0.15])
        weights_lower = jnp.array([-0.15, -0.15, -0.15])

        x, y_u, y_l = generate_cst_coords(weights_upper, weights_lower)

        # LE (x=0) and TE (x=1) should have y~=0
        assert jnp.isclose(y_u[0], 0.0, atol=1e-6)
        assert jnp.isclose(y_u[-1], 0.0, atol=1e-6)
        assert jnp.isclose(y_l[0], 0.0, atol=1e-6)
        assert jnp.isclose(y_l[-1], 0.0, atol=1e-6)


class TestDifferentiability:
    """Test gradient flow through CST functions."""

    def test_shape_function_gradients(self):
        """Gradients should flow through shape function."""

        def loss(weights):
            x = jnp.linspace(0.0, 1.0, 50)
            S = cst_shape_function(x, weights)
            return jnp.sum(S**2)

        weights = jnp.array([0.1, 0.2, 0.3, 0.1])
        gradients = grad(loss)(weights)

        assert gradients.shape == weights.shape
        assert jnp.all(jnp.isfinite(gradients))
        assert jnp.any(gradients != 0)

    def test_full_coords_gradients(self):
        """Gradients should flow through full coordinate generation."""

        def loss(weights_upper, weights_lower):
            x, y_u, y_l = generate_cst_coords(
                weights_upper, weights_lower, num_points=50
            )
            thickness = y_u - y_l
            return jnp.sum(thickness**2)

        weights_upper = jnp.array([0.15, 0.15, 0.15])
        weights_lower = jnp.array([-0.15, -0.15, -0.15])

        grad_upper = grad(loss, argnums=0)(weights_upper, weights_lower)
        grad_lower = grad(loss, argnums=1)(weights_upper, weights_lower)

        assert jnp.all(jnp.isfinite(grad_upper))
        assert jnp.all(jnp.isfinite(grad_lower))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

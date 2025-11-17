"""
Tests to verify JAX is properly set up and differentiable.
"""

import pytest
import turbodiff  # noqa: F401

import jax.numpy as jnp
from jax import jit, grad


class TestJAXBasics:
    """Test basic JAX functionality."""

    def test_array_creation(self):
        """Test JAX array creation."""
        arr = jnp.zeros((3, 4))
        assert arr.shape == (3, 4)
        assert isinstance(arr, jnp.ndarray)

    def test_jit_compilation(self):
        """Test JIT compilation works."""

        @jit
        def square(x):
            return x**2

        result = square(jnp.array([1.0, 2.0, 3.0]))
        expected = jnp.array([1.0, 4.0, 9.0])
        assert jnp.allclose(result, expected)

    def test_automatic_differentiation(self):
        """Test automatic differentiation."""

        def f(x):
            return jnp.sum(x**2)

        grad_f = grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        gradient = grad_f(x)

        # Gradient of sum(x^2) is 2x
        expected = jnp.array([2.0, 4.0, 6.0])
        assert jnp.allclose(gradient, expected)


class TestDifferentiableOperations:
    """Test that common operations are differentiable."""

    def test_where_differentiable(self):
        """Test jnp.where is differentiable."""

        def f(x):
            condition = x > 0
            return jnp.sum(jnp.where(condition, x**2, x**3))

        grad_f = grad(f)
        gradient = grad_f(1.0)
        assert jnp.isfinite(gradient)

    def test_clip_differentiable(self):
        """Test jnp.clip is differentiable."""

        def f(x):
            return jnp.sum(jnp.clip(x, 0, 10))

        grad_f = grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        gradient = grad_f(x)

        # In valid range, gradient is 1
        assert jnp.allclose(gradient, jnp.ones(3))

    def test_roll_differentiable(self):
        """Test jnp.roll is differentiable."""

        def f(x):
            rolled = jnp.roll(x, 1)
            return jnp.sum(rolled**2)

        grad_f = grad(f)
        x = jnp.array([1.0, 2.0, 3.0])
        gradient = grad_f(x)

        assert jnp.all(jnp.isfinite(gradient))

    def test_meshgrid_differentiable(self):
        """Test meshgrid + operations are differentiable."""

        def f(scale):
            x = jnp.arange(3.0)
            y = jnp.arange(3.0)
            xx, yy = jnp.meshgrid(x, y, indexing="ij")
            return jnp.sum((xx * scale) ** 2 + (yy * scale) ** 2)

        grad_f = grad(f)
        gradient = grad_f(1.0)
        assert jnp.isfinite(gradient)


class TestVectorization:
    """Test vectorized operations."""

    def test_vectorized_computation(self):
        """Test vectorized vs loop computation."""
        n = 100
        x = jnp.arange(n, dtype=jnp.float32)

        # Vectorized
        result_vec = x**2 + 2 * x - 1

        # Check result is correct
        assert result_vec.shape == (n,)
        assert jnp.all(jnp.isfinite(result_vec))

    def test_neighbor_sum(self):
        """Test neighbor summation pattern."""
        arr = jnp.arange(25, dtype=jnp.float32).reshape(5, 5)

        # Sum of neighbors using roll
        neighbors_sum = (
            jnp.roll(arr, -1, axis=0)
            + jnp.roll(arr, 1, axis=0)
            + jnp.roll(arr, -1, axis=1)
            + jnp.roll(arr, 1, axis=1)
        )

        assert neighbors_sum.shape == (5, 5)
        assert jnp.all(jnp.isfinite(neighbors_sum))


class TestSimulationPrimitives:
    """Test primitives needed for fluid simulation."""

    def test_backward_trace(self):
        """Test backward particle tracing pattern."""
        h, w = 10, 10

        # Grid coordinates
        i, j = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
        x = j + 0.5
        y = i + 0.5

        # Sample velocity (constant for test)
        u = jnp.ones((h, w))
        v = jnp.ones((h, w))

        # Backward trace
        dt = 0.1
        x_back = x - u * dt
        y_back = y - v * dt

        assert x_back.shape == (h, w)
        assert y_back.shape == (h, w)

    def test_bilinear_interpolation(self):
        """Test bilinear interpolation is differentiable."""

        def interpolate_at(field, x, y):
            """Simple bilinear interpolation."""
            h, w = field.shape

            # Get integer and fractional parts
            i = jnp.floor(y).astype(int)
            j = jnp.floor(x).astype(int)

            # Clamp to valid range
            i = jnp.clip(i, 0, h - 2)
            j = jnp.clip(j, 0, w - 2)

            # Fractional parts
            fi = y - i
            fj = x - j

            # Bilinear interpolation
            v00 = field[i, j]
            v01 = field[i, j + 1]
            v10 = field[i + 1, j]
            v11 = field[i + 1, j + 1]

            v0 = v00 * (1 - fj) + v01 * fj
            v1 = v10 * (1 - fj) + v11 * fj

            return v0 * (1 - fi) + v1 * fi

        def loss(field):
            x, y = 2.5, 3.5
            return interpolate_at(field, x, y)

        field = jnp.arange(25.0).reshape(5, 5)
        gradient = grad(loss)(field)

        # Check gradient is valid
        assert gradient.shape == (5, 5)
        assert jnp.any(gradient != 0)  # Some gradients should be non-zero

    def test_divergence_computation(self):
        """Test divergence computation pattern."""
        h, w = 10, 10
        dx = 0.1

        # Velocity components (staggered grid)
        u = jnp.ones((h, w + 1))
        v = jnp.ones((h + 1, w))

        # Compute divergence
        du_dx = (u[:, 1:] - u[:, :-1]) / dx
        dv_dy = (v[1:, :] - v[:-1, :]) / dx

        divergence = du_dx + dv_dy

        assert divergence.shape == (h, w)
        assert jnp.all(jnp.isfinite(divergence))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

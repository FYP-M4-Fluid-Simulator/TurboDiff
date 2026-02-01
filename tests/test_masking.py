"""
Tests for soft masking module.
"""

import pytest
import jax.numpy as jnp
from jax import grad
from turbodiff.core.masking import (
    soft_sigmoid_mask,
    soft_mask_from_bounds,
    create_airfoil_mask,
    interpolate_surface_to_grid,
)


class TestSoftSigmoidMask:
    """Test sigmoid-based soft masking."""

    def test_inside_outside_values(self):
        """Negative distance -> 1, positive distance -> 0."""
        distance = jnp.array([-1.0, 0.0, 1.0])
        mask = soft_sigmoid_mask(distance, sharpness=50.0)

        assert mask[0] > 0.99  # Inside (negative distance)
        assert jnp.isclose(mask[1], 0.5, atol=0.01)  # On boundary
        assert mask[2] < 0.01  # Outside (positive distance)

    def test_sharpness_effect(self):
        """Higher sharpness -> sharper transition."""
        distance = jnp.array([0.0])

        # At boundary (d=0), mask should always be ~0.5
        mask_low = soft_sigmoid_mask(distance, sharpness=10.0)
        mask_high = soft_sigmoid_mask(distance, sharpness=100.0)

        assert jnp.isclose(mask_low[0], 0.5, atol=0.01)
        assert jnp.isclose(mask_high[0], 0.5, atol=0.01)

    def test_differentiability(self):
        """Mask should have non-zero gradients."""

        def loss(distance):
            mask = soft_sigmoid_mask(distance, sharpness=50.0)
            return jnp.sum(mask)

        distance = jnp.array([0.1, 0.0, -0.1])
        gradients = grad(loss)(distance)

        assert jnp.all(jnp.isfinite(gradients))
        assert jnp.any(gradients != 0)


class TestSoftMaskFromBounds:
    """Test mask for values within bounds."""

    def test_inside_bounds(self):
        """Values inside bounds should have mask ~1."""
        values = jnp.array([5.0])
        mask = soft_mask_from_bounds(values, 0.0, 10.0, sharpness=50.0)

        assert mask[0] > 0.99

    def test_outside_bounds(self):
        """Values outside bounds should have mask ~0."""
        values = jnp.array([-5.0, 15.0])
        mask = soft_mask_from_bounds(values, 0.0, 10.0, sharpness=50.0)

        assert mask[0] < 0.01
        assert mask[1] < 0.01


class TestCreateAirfoilMask:
    """Test full airfoil mask creation."""

    def test_mask_shape(self):
        """Output mask should match grid shape."""
        H, W = 10, 20
        grid_x = jnp.ones((H, W)) * 0.5  # All at x=0.5
        grid_y = jnp.linspace(-0.1, 0.1, H)[:, None] * jnp.ones((1, W))

        y_upper = jnp.ones((H, W)) * 0.05
        y_lower = jnp.ones((H, W)) * (-0.05)

        mask = create_airfoil_mask(grid_x, grid_y, y_upper, y_lower)

        assert mask.shape == (H, W)

    def test_inside_outside(self):
        """Points inside airfoil should have mask ~1."""
        # Simple grid
        grid_x = jnp.array([[0.5]])
        grid_y = jnp.array([[0.0]])  # At centerline

        y_upper = jnp.array([[0.1]])
        y_lower = jnp.array([[-0.1]])

        mask = create_airfoil_mask(
            grid_x, grid_y, y_upper, y_lower, x_min=0.0, x_max=1.0
        )

        assert mask[0, 0] > 0.9  # Inside

    def test_outside_chord(self):
        """Points outside chord range should have mask ~0."""
        grid_x = jnp.array([[1.5]])  # Beyond chord
        grid_y = jnp.array([[0.0]])

        y_upper = jnp.array([[0.1]])
        y_lower = jnp.array([[-0.1]])

        mask = create_airfoil_mask(
            grid_x, grid_y, y_upper, y_lower, x_min=0.0, x_max=1.0
        )

        assert mask[0, 0] < 0.1  # Outside


class TestInterpolateSurfaceToGrid:
    """Test surface interpolation to grid."""

    def test_interpolation_accuracy(self):
        """Test linear interpolation is correct."""
        # Linear surface: y = 2*x
        x_coords = jnp.array([0.0, 0.5, 1.0])
        y_coords = jnp.array([0.0, 1.0, 2.0])

        grid_x = jnp.array([[0.25], [0.75]])

        y_interp = interpolate_surface_to_grid(grid_x, x_coords, y_coords)

        assert jnp.isclose(y_interp[0, 0], 0.5, atol=1e-5)
        assert jnp.isclose(y_interp[1, 0], 1.5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

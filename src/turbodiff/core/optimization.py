"""
Optimization utilities for differentiable shape optimization.

This module provides optimizer implementations for gradient-based optimization.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any, Tuple


# ============================================================================
# Optimizer Implementations
# ============================================================================


def create_optimizer(
    optimizer_type: str = "adam", learning_rate: float = 0.01, **kwargs
) -> Tuple[Dict[str, Any], Callable]:
    """
    Create optimizer state and update function.

    Supports Adam and SGD optimizers for gradient-based shape optimization.

    Args:
        optimizer_type: "adam" or "sgd"
        learning_rate: Learning rate for parameter updates
        **kwargs: Additional optimizer parameters (beta1, beta2, epsilon for Adam)

    Returns:
        (optimizer_state, update_fn) tuple
    """
    if optimizer_type == "adam":
        beta1 = kwargs.get("beta1", 0.9)
        beta2 = kwargs.get("beta2", 0.999)
        epsilon = kwargs.get("epsilon", 1e-8)

        opt_state = {
            "m": None,  # First moment
            "v": None,  # Second moment
            "t": 0,  # Time step
        }

        def adam_update(params, gradients, state):
            """Adam optimizer update with bias correction."""
            t = state["t"] + 1

            # Initialize moments on first call
            if state["m"] is None:
                m = jax.tree.map(jnp.zeros_like, gradients)
                v = jax.tree.map(jnp.zeros_like, gradients)
            else:
                m = state["m"]
                v = state["v"]

            # Update biased moments
            m = jax.tree.map(
                lambda m_i, g_i: beta1 * m_i + (1 - beta1) * g_i, m, gradients
            )
            v = jax.tree.map(
                lambda v_i, g_i: beta2 * v_i + (1 - beta2) * g_i**2, v, gradients
            )

            # Bias correction
            m_hat = jax.tree.map(lambda m_i: m_i / (1 - beta1**t), m)
            v_hat = jax.tree.map(lambda v_i: v_i / (1 - beta2**t), v)

            # Parameter update
            new_params = jax.tree.map(
                lambda p, mh, vh: p - learning_rate * mh / (jnp.sqrt(vh) + epsilon),
                params,
                m_hat,
                v_hat,
            )

            new_state = {"m": m, "v": v, "t": t}
            return new_params, new_state

        return opt_state, adam_update

    elif optimizer_type == "sgd":
        opt_state = {}

        def sgd_update(params, gradients, state):
            """SGD update (gradient descent with fixed learning rate)."""
            new_params = jax.tree.map(
                lambda p, g: p - learning_rate * g, params, gradients
            )
            return new_params, state

        return opt_state, sgd_update

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


__all__ = [
    "create_optimizer",
]

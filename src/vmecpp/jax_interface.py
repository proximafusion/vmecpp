# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""JAX bindings for solve-like equilibrium maps.

The VMEC solve itself is an opaque stateful operation, so it is represented as
two NumPy callbacks: a forward solve and its parameter VJP. The returned JAX
function has a normal ``custom_vjp`` contract and can therefore be composed
with arbitrary JAX objectives. The caller owns the solve and VJP callbacks;
VMEC++ applications can provide the VJP from the Enzyme-backed adjoint.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def make_custom_vjp_solve(
    solve: Callable[[np.ndarray], np.ndarray],
    parameter_vjp: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    output_shape: Sequence[int],
) -> Callable[[Any], Any]:
    """Wrap a NumPy solve and parameter VJP as a differentiable JAX function.

    ``solve(parameters)`` returns the state exposed to a JAX objective. The
    ``parameter_vjp(parameters, state, state_bar)`` callback returns the VJP of
    that solve with respect to ``parameters``. Both callbacks run outside JAX;
    the custom VJP is the interface that lets a JAX objective remain fully
    composable. ``jax.pure_callback`` also makes the wrapper usable under
    ``jax.jit`` (but not under ``jax.vmap`` over a stateful VMEC model).
    """
    shape = tuple(int(dimension) for dimension in output_shape)
    if any(dimension < 0 for dimension in shape):
        error_message = "output_shape must contain non-negative dimensions"
        raise ValueError(error_message)

    def solve_callback(parameters: Any) -> np.ndarray:
        state = np.asarray(solve(np.asarray(parameters)))
        if state.shape != shape:
            error_message = f"solve returned shape {state.shape}, expected {shape}"
            raise ValueError(error_message)
        return state

    def vjp_callback(parameters: Any, state: Any, state_bar: Any) -> np.ndarray:
        gradient = np.asarray(
            parameter_vjp(
                np.asarray(parameters), np.asarray(state), np.asarray(state_bar)
            )
        )
        if gradient.shape != np.asarray(parameters).shape:
            error_message = (
                f"parameter_vjp returned shape {gradient.shape}, expected "
                f"{np.asarray(parameters).shape}"
            )
            raise ValueError(error_message)
        return gradient

    @jax.custom_vjp
    def solve_jax(parameters: Any) -> Any:
        parameters = jnp.asarray(parameters)
        result = jax.ShapeDtypeStruct(shape, parameters.dtype)
        return jax.pure_callback(solve_callback, result, parameters)

    def solve_fwd(parameters: Any) -> tuple[Any, tuple[Any, Any]]:
        state = solve_jax(parameters)
        return state, (parameters, state)

    def solve_bwd(residual: tuple[Any, Any], state_bar: Any) -> tuple[Any]:
        parameters, state = residual
        result = jax.ShapeDtypeStruct(parameters.shape, parameters.dtype)
        parameter_bar = jax.pure_callback(
            vjp_callback, result, parameters, state, state_bar
        )
        return (parameter_bar,)

    solve_jax.defvjp(solve_fwd, solve_bwd)
    return solve_jax

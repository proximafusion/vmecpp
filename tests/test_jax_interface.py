# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Behavioral tests for the generic JAX custom-VJP solve wrapper."""

import jax
import jax.numpy as jnp
import numpy as np

from vmecpp import make_custom_vjp_solve

jax.config.update("jax_enable_x64", True)


def test_custom_vjp_composes_with_jax_objective_and_jit():
    matrix = np.array([[2.0, -1.0], [0.5, 3.0]])

    def solve(parameters):
        return matrix @ parameters

    def parameter_vjp(parameters, _state, state_bar):
        del parameters
        return matrix.T @ state_bar

    solve_jax = make_custom_vjp_solve(solve, parameter_vjp, output_shape=(2,))
    parameters = jnp.array([0.7, -0.2])

    def objective(p):
        state = solve_jax(p)
        return jnp.sum(jnp.sin(state) + state**2)

    value, gradient = jax.value_and_grad(objective)(parameters)
    compiled_value, compiled_gradient = jax.jit(jax.value_and_grad(objective))(
        parameters
    )
    state = matrix @ np.asarray(parameters)
    expected = matrix.T @ (np.cos(state) + 2.0 * state)
    np.testing.assert_allclose(value, np.sum(np.sin(state) + state**2))
    np.testing.assert_allclose(gradient, expected)
    np.testing.assert_allclose(compiled_value, value)
    np.testing.assert_allclose(compiled_gradient, gradient)

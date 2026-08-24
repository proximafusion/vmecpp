# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Compose a VMEC boundary solve with arbitrary JAX objectives.

The VMEC solve is deliberately kept outside JAX. ``make_boundary_solve`` wraps
the solve-like map in ``vmecpp.make_custom_vjp_solve``; the objective can then
be written entirely in JAX over the returned geometry state. On an Enzyme
build, the callback VJP is VMEC++'s exact transposed-HVP adjoint.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from vmecpp_adjoint import (
    DEFAULT_INPUT,
    adjoint_boundary_gradient,
    make_model,
    partition,
    solve_interior,
)

from vmecpp import make_custom_vjp_solve


class BoundarySolve:
    """Stateful, single-threaded solve map suitable for a JAX callback."""

    def __init__(self, input_path: Path = DEFAULT_INPUT, ns: int = 11):
        self.model = make_model(input_path, ns)
        self.model.solve()
        self.ns = ns
        self.interior, self.boundary = partition(self.model, ns)
        self.state = np.asarray(self.model.get_state(), float).copy()

    def solve(self, boundary: np.ndarray) -> np.ndarray:
        self.state = solve_interior(
            self.model,
            self.state,
            self.interior,
            self.boundary,
            np.asarray(boundary, float),
        )
        return self.state.copy()

    def parameter_vjp(
        self, boundary: np.ndarray, state: np.ndarray, state_bar: np.ndarray
    ) -> np.ndarray:
        del boundary
        self.state = np.asarray(state, float).copy()
        gradient, info = adjoint_boundary_gradient(
            self.model,
            self.state,
            self.interior,
            self.boundary,
            np.asarray(state_bar, float),
        )
        if info != 0:
            error_message = f"VMEC adjoint solve failed with info={info}"
            raise RuntimeError(error_message)
        return gradient

    def as_jax_function(self):
        """Return ``boundary -> full_state`` with the exact custom VJP."""
        return make_custom_vjp_solve(
            self.solve,
            self.parameter_vjp,
            output_shape=(self.state.size,),
        )


def boundary_geometry(
    state,
    ns: int,
    mpol: int,
    ntor: int,
    theta,
    zeta,
):
    """Evaluate the symmetric boundary R/Z Fourier geometry in JAX.

    This small pure-JAX map is an example objective layer. Magnetic QS objectives can
    replace it with their field or harmonic residual while keeping the same custom-VJP
    boundary solve.
    """
    n_modes = mpol * (ntor + 1)
    surface = ns - 1
    block = surface * n_modes
    rcc = state[block : block + n_modes].reshape(mpol, ntor + 1)
    rss_offset = ns * n_modes
    rss = state[rss_offset + block : rss_offset + block + n_modes].reshape(
        mpol, ntor + 1
    )
    zsc_offset = 2 * ns * n_modes
    zsc = state[zsc_offset + block : zsc_offset + block + n_modes].reshape(
        mpol, ntor + 1
    )
    zcs_offset = 3 * ns * n_modes
    zcs = state[zcs_offset + block : zcs_offset + block + n_modes].reshape(
        mpol, ntor + 1
    )
    m = jnp.arange(mpol)[:, None]
    n = jnp.arange(ntor + 1)[None, :]
    theta = jnp.asarray(theta)
    zeta = jnp.asarray(zeta)
    cos_m = jnp.cos(theta[..., None, None] * m)
    sin_m = jnp.sin(theta[..., None, None] * m)
    cos_n = jnp.cos(zeta[..., None, None] * n)
    sin_n = jnp.sin(zeta[..., None, None] * n)
    r = jnp.sum(rcc * cos_m * cos_n + rss * sin_m * sin_n, axis=(-2, -1))
    z = jnp.sum(zsc * sin_m * cos_n + zcs * cos_m * sin_n, axis=(-2, -1))
    return r, z

# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Gradients with respect to the power-series profile parameters, against central
differences of re-solved equilibria."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff, autodiff_wout
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)

requires_enzyme = pytest.mark.skipif(
    not _vmecpp.VMECPP_ENABLE_ENZYME, reason="needs the exact residual transpose"
)


def _cth_like(ncurr: int) -> vmecpp.VmecInput:
    source = vmecpp.VmecInput.from_file(
        Path(__file__).parents[1] / "examples" / "data" / "cth_like_fixed_bdy.json"
    )
    profile = {"ac": np.asarray([1.0, -1.0]), "pcurr_type": "power_series"}
    if ncurr == 0:
        profile = {"ai": np.asarray([0.35, 0.15]), "piota_type": "power_series"}
    return source.model_copy(
        update={
            "ncurr": ncurr,
            "pmass_type": "power_series",
            "am": np.asarray([1.0, -2.0, 1.0]),
            **profile,
            "ns_array": np.asarray([15]),
            "ftol_array": np.asarray([1.0e-16]),
            "niter_array": np.asarray([6000]),
        }
    )


@pytest.mark.parametrize("ncurr", [0, 1])
def test_half_grid_profiles_match_the_solver(ncurr: int) -> None:
    indata = _cth_like(ncurr)
    wout = vmecpp.run(indata, max_threads=1, verbose=False).wout
    mass, iota, current = autodiff_wout.half_grid_profiles(indata, wout.ns, {})
    np.testing.assert_allclose(mass, wout.mass[1:] * autodiff_wout.MU_0, rtol=1e-14)
    if ncurr == 0:
        np.testing.assert_allclose(iota, wout.iotas[1:], rtol=1e-14)
    else:  # <B_u> meets the prescribed current through the chi' solve
        np.testing.assert_allclose(current, wout.buco[1:], rtol=1e-10)


def _check_gradient(objective, parameters: dict) -> None:
    direction = {
        name: 1e-4 * (jnp.abs(value) + 1e-2 * jnp.max(jnp.abs(value)))
        for name, value in parameters.items()
    }
    gradient = jax.grad(objective)(parameters)
    directional = sum(float(jnp.sum(gradient[k] * direction[k])) for k in direction)
    plus, minus = (
        objective({k: v + sign * direction[k] for k, v in parameters.items()})
        for sign in (1.0, -1.0)
    )
    np.testing.assert_allclose(directional, 0.5 * float(plus - minus), rtol=1e-3)


@requires_enzyme
@pytest.mark.parametrize(
    ("ncurr", "name"),
    [(0, "am"), (0, "pres_scale"), (0, "ai"), (1, "am"), (1, "ac"), (1, "curtor")],
)
def test_solver_gradient(ncurr: int, name: str) -> None:
    indata = _cth_like(ncurr)
    solver = autodiff.make_solver(indata)
    boundary = jnp.stack([jnp.asarray(indata.rbc), jnp.asarray(indata.zbs)])

    def objective(parameters):
        geometry = solver(boundary, parameters)
        return jnp.sum(geometry.r_cc[7] ** 2) + 1e2 * geometry.poloidal_flux[-1]

    _check_gradient(objective, {name: jnp.asarray(getattr(indata, name), float)})


@requires_enzyme
@pytest.mark.parametrize("ncurr", [0, 1])
def test_run_gradient(ncurr: int) -> None:
    """Through vmecpp.run and the output stage, including the echoed coefficients."""
    indata = _cth_like(ncurr)
    names = ("am", "pres_scale", "ai") if ncurr == 0 else ("am", "ac", "curtor")

    def objective(parameters):
        wout = vmecpp.run(indata.model_copy(update=parameters), verbose=False).wout
        return wout.betatotal + wout.iotaf[-1] + 1e-3 * wout.am[0]

    parameters = {name: jnp.asarray(getattr(indata, name), float) for name in names}
    _check_gradient(objective, parameters)

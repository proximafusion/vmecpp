# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Gradients with respect to the pressure, iota and current profile parameters.

The profiles enter the force through the half-grid pressure presH, the prescribed-iota
chi' = iota phi' (ncurr = 0) and the prescribed current currH (ncurr = 1). profile_vjp
returns their cotangents from the Enzyme reverse pass of the force composition;
autodiff_wout.half_grid_profiles maps them to the power-series coefficients and scales.
The checks compare directional derivatives against central differences of fully re-
solved equilibria.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff, autodiff_wout
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _cth_like(ncurr: int, ns: int = 15) -> vmecpp.VmecInput:
    source = vmecpp.VmecInput.from_file(
        REPO_ROOT / "examples" / "data" / "cth_like_fixed_bdy.json"
    )
    update = {
        "pmass_type": "power_series",
        "am": np.asarray([1.0, -2.0, 1.0]),
        "ns_array": np.asarray([ns]),
        "ftol_array": np.asarray([1.0e-16]),
        "niter_array": np.asarray([6000]),
    }
    if ncurr == 1:
        update |= {
            "ncurr": 1,
            "pcurr_type": "power_series",
            "ac": np.asarray([1.0, -1.0]),
        }
    else:
        update |= {
            "ncurr": 0,
            "piota_type": "power_series",
            "ai": np.asarray([0.35, 0.15]),
        }
    return source.model_copy(update=update)


def _requires_exact_derivatives(indata: vmecpp.VmecInput) -> None:
    ns = int(np.asarray(indata.ns_array)[-1])
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    if not model.has_exact_force_jacobian:
        pytest.skip("needs an Enzyme-enabled build for the exact residual transpose")


def _boundary(indata: vmecpp.VmecInput) -> jax.Array:
    return jnp.stack([jnp.asarray(indata.rbc), jnp.asarray(indata.zbs)])


@pytest.mark.parametrize("ncurr", [0, 1])
def test_half_grid_profiles_match_the_solver(ncurr: int) -> None:
    indata = _cth_like(ncurr)
    wout = vmecpp.run(indata, max_threads=1, verbose=False).wout
    profiles = autodiff_wout.half_grid_profiles(indata, wout.ns, {})
    np.testing.assert_allclose(
        profiles["mass_half"], wout.mass[1:] * autodiff_wout.MU_0, rtol=1.0e-14
    )
    if ncurr == 0:
        np.testing.assert_allclose(profiles["iota_half"], wout.iotas[1:], rtol=1.0e-14)
    else:
        # <B_u> meets the prescribed current through the chi' solve
        np.testing.assert_allclose(
            profiles["current_half"], wout.buco[1:], rtol=1.0e-10
        )


def test_half_grid_profiles_reject_unsupported_parameterizations() -> None:
    indata = _cth_like(0).model_copy(update={"pmass_type": "two_power"})
    with pytest.raises(NotImplementedError, match="power_series"):
        autodiff_wout.half_grid_profiles(indata, 15, {"am": indata.am})
    with pytest.raises(NotImplementedError, match="gamma"):
        autodiff_wout.half_grid_profiles(
            _cth_like(0).model_copy(update={"gamma": 5.0 / 3.0}), 15, {"am": indata.am}
        )


def _central_difference(objective, parameters: dict, direction: dict) -> float:
    def shifted(sign):
        return {
            name: value + sign * direction.get(name, 0.0)
            for name, value in parameters.items()
        }

    return 0.5 * float(objective(shifted(1.0)) - objective(shifted(-1.0)))


def _directional(gradient: dict, direction: dict) -> float:
    return sum(float(jnp.sum(gradient[name] * direction[name])) for name in direction)


def _check_gradient(objective, parameters: dict, direction: dict) -> None:
    gradient = jax.grad(objective)(parameters)
    for name in parameters:
        assert np.all(np.isfinite(np.asarray(gradient[name])))
    directional = _directional(gradient, direction)
    finite_difference = _central_difference(objective, parameters, direction)
    np.testing.assert_allclose(directional, finite_difference, rtol=1.0e-3)


@pytest.mark.parametrize(
    ("ncurr", "name"),
    [(0, "am"), (0, "pres_scale"), (0, "ai"), (1, "am"), (1, "ac"), (1, "curtor")],
)
def test_solver_gradient_matches_central_differences(ncurr: int, name: str) -> None:
    """A geometry objective that depends on the state and on the poloidal flux."""
    indata = _cth_like(ncurr)
    _requires_exact_derivatives(indata)
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)
    value = jnp.asarray(getattr(indata, name), dtype=jnp.float64)
    direction = {name: 1.0e-4 * (jnp.abs(value) + 1.0e-3 * jnp.max(jnp.abs(value)))}

    def objective(parameters):
        geometry = solver(boundary, parameters)
        mid = geometry.r_cc.shape[0] // 2
        return (
            jnp.sum(geometry.r_cc[mid] ** 2)
            + jnp.sum(geometry.z_sc[mid] ** 2)
            + jnp.sum(geometry.lambda_sc[mid] ** 2)
            + 1.0e2 * geometry.poloidal_flux[-1]
        )

    _check_gradient(objective, {name: value}, direction)


@pytest.mark.parametrize(
    ("ncurr", "objective_name"),
    [(0, "betatotal"), (0, "iota_edge"), (1, "betatotal"), (1, "iota_edge")],
)
def test_run_gradient_matches_central_differences(
    ncurr: int, objective_name: str
) -> None:
    """jax.grad through vmecpp.run and the JAX output stage, with respect to the
    pressure and the iota or current profile at once."""
    indata = _cth_like(ncurr)
    _requires_exact_derivatives(indata)
    names = ("am", "pres_scale", "ai") if ncurr == 0 else ("am", "ac", "curtor")
    parameters = {
        name: jnp.asarray(getattr(indata, name), dtype=jnp.float64) for name in names
    }
    generator = np.random.default_rng(ncurr)
    direction = {
        name: 1.0e-4
        * jnp.asarray(generator.standard_normal(np.shape(value)))
        * (jnp.abs(value) + 1.0e-2 * jnp.max(jnp.abs(value)))
        for name, value in parameters.items()
    }

    def objective(parameters):
        wout = vmecpp.run(
            indata.model_copy(update=parameters), max_threads=1, verbose=False
        ).wout
        if objective_name == "betatotal":
            return wout.betatotal
        return wout.iotaf[-1]

    _check_gradient(objective, parameters, direction)


def test_boundary_only_gradient_ignores_unsupported_profile_types() -> None:
    """Profile leaves traced by jax.jit but not differentiated need no JAX port."""
    indata = _cth_like(1).model_copy(
        update={"pmass_type": "two_power", "am": np.asarray([1.0, 5.0, 10.0])}
    )
    _requires_exact_derivatives(indata)

    @jax.jit
    def aspect_gradient(vmec_input):
        def aspect(boundary):
            traced = vmec_input.model_copy(
                update={"rbc": boundary[0], "zbs": boundary[1]}
            )
            return vmecpp.run(traced, max_threads=1, verbose=False).wout.aspect

        return jax.grad(aspect)(_boundary(vmec_input))

    assert np.all(np.isfinite(np.asarray(aspect_gradient(indata))))

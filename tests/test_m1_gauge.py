# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The pinned m=1 poloidal-origin gauge and the boundary adjoint that relies on it.

``FourierCoeffs::m1Constraint`` stores, for every ``m = 1`` mode, the pair
``(rss + zcs) / 2`` and ``(rss - zcs) / 2``; the second one is a poloidal-angle
origin per toroidal harmonic. The native iteration lets it drift under its force
until ``fsqz < 1e-6`` and freezes it wherever it is, so the converged state is a
function of the iteration history. ``VmecModel.always_fix_m1_gauge`` zeroes that
force from the first iteration and sets the gauge from the boundary at every
multigrid step; the converged gauge is then the boundary gauge scaled by
``sqrt(s)``, and the fixed-gauge force Jacobian is the linearization of the
iterated system.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)

EXAMPLES_DATA = Path(__file__).resolve().parents[1] / "examples" / "data"

# Power-series fit of iotaf from examples/data/wout_cth_like_fixed_bdy.nc, so the
# ncurr=0 case below has the equilibrium's own rotational transform.
_CTH_LIKE_IOTA = np.asarray([1.239, 0.328, -0.768])


def _cth_like_input(ns_array, ftol: float = 1.0e-16, niter: int = 5000):
    source = vmecpp.VmecInput.from_file(EXAMPLES_DATA / "cth_like_fixed_bdy.json")
    ns_array = np.asarray(ns_array, dtype=np.int64)
    return source.model_copy(
        update={
            "ncurr": 0,
            "piota_type": "power_series",
            "ai": _CTH_LIKE_IOTA,
            "ns_array": ns_array,
            "ftol_array": np.full(ns_array.size, ftol),
            "niter_array": np.full(ns_array.size, niter, dtype=np.int64),
        }
    )


def _small_3d_input() -> vmecpp.VmecInput:
    """Solovev with one toroidal boundary mode (as in test_autodiff.py)."""
    source = vmecpp.VmecInput.from_file(EXAMPLES_DATA / "solovev.json")
    mpol = source.mpol
    rbc = np.zeros((mpol, 3))
    zbs = np.zeros((mpol, 3))
    rbc[:, 1] = np.asarray(source.rbc)[:, 0]
    zbs[:, 1] = np.asarray(source.zbs)[:, 0]
    rbc[1, 2] = 0.01
    zbs[1, 2] = 0.01
    return source.model_copy(
        update={
            "ntor": 1,
            "rbc": rbc,
            "zbs": zbs,
            "raxis_c": np.asarray([4.0, 0.0]),
            "zaxis_s": np.asarray([0.0, 0.0]),
            "ns_array": np.asarray([5]),
            "ftol_array": np.asarray([1.0e-15]),
            "niter_array": np.asarray([4000]),
        }
    )


def _boundary(indata: vmecpp.VmecInput) -> np.ndarray:
    return np.stack([np.asarray(indata.rbc), np.asarray(indata.zbs)], axis=0).astype(
        np.float64
    )


def _requires_exact_derivatives(model) -> None:
    if not model.has_exact_force_jacobian:
        pytest.skip("needs an Enzyme-enabled build for the exact force Jacobian")


def _solve(indata: vmecpp.VmecInput, always_fix_m1_gauge: bool):
    """Solve through every entry of ns_array, as autodiff._solve_model does."""
    cpp_indata = indata._to_cpp_vmecindata()
    model = None
    for ns in (int(value) for value in np.asarray(indata.ns_array)):
        if model is None:
            model = _vmecpp.VmecModel.create(cpp_indata, ns)
            model.always_fix_m1_gauge = always_fix_m1_gauge
        else:
            model.refine_to(ns)
        model.solve()
    assert model is not None
    return model


def test_pinned_gauge_stays_at_the_boundary_interpolation() -> None:
    indata = _cth_like_input([25])
    cpp_indata = indata._to_cpp_vmecindata()
    model = _vmecpp.VmecModel.create(cpp_indata, 25)
    model.always_fix_m1_gauge = True
    initial = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.solve()
    solved = np.asarray(model.get_state(), dtype=np.float64)

    gauge = autodiff._gauge_entries(model)
    assert gauge.size == (model.ns - 2) * model.ntor
    np.testing.assert_allclose(solved[gauge], initial[gauge], rtol=0.0, atol=1.0e-12)

    # The gauge is sqrt(s) times the boundary gauge on every surface.
    modes_per_surface = model.mpol * (model.ntor + 1)
    span = autodiff._span_slices(model)["z_cs"]
    edge = span.start + (model.ns - 1) * modes_per_surface
    boundary_gauge = solved[edge + (gauge - span.start) % modes_per_surface]
    np.testing.assert_allclose(
        solved[gauge],
        autodiff._gauge_radial_weight(model, gauge) * boundary_gauge,
        rtol=0.0,
        atol=1.0e-15,
    )

    # The native iteration moves the gauge by an amount comparable to its size.
    native = _vmecpp.VmecModel.create(cpp_indata, 25)
    native.solve()
    drift = np.abs(np.asarray(native.get_state())[gauge] - initial[gauge]).max()
    assert drift > 1.0e-5
    assert np.abs(initial[gauge]).max() < 1.0e-3


def test_pinned_gauge_solve_is_independent_of_the_multigrid_history() -> None:
    direct = _solve(_cth_like_input([25]), always_fix_m1_gauge=True)
    staged = _solve(_cth_like_input([13, 25]), always_fix_m1_gauge=True)
    direct_state = np.asarray(direct.get_state(), dtype=np.float64)
    staged_state = np.asarray(staged.get_state(), dtype=np.float64)
    gauge = autodiff._gauge_entries(direct)
    np.testing.assert_allclose(
        staged_state[gauge], direct_state[gauge], rtol=0.0, atol=1.0e-14
    )
    # The remaining difference is the convergence floor at ftol = 1e-16: the
    # R, Z blocks agree to 2e-8 and lambda (the slowest to converge) to 1.4e-6.
    slices = autodiff._span_slices(direct)
    for name, span in slices.items():
        floor = 1.0e-5 if name.startswith("lambda") else 1.0e-7
        np.testing.assert_allclose(
            staged_state[span], direct_state[span], rtol=0.0, atol=floor
        )

    # The native gauge carries the multigrid history into the geometry: 4e-5 in
    # z_cs (the gauge slot) and 5e-6 in the other R, Z blocks.
    native_direct = _solve(_cth_like_input([25]), always_fix_m1_gauge=False)
    native_staged = _solve(_cth_like_input([13, 25]), always_fix_m1_gauge=False)
    native_difference = np.abs(
        np.asarray(native_staged.get_state()) - np.asarray(native_direct.get_state())
    )
    assert native_difference[gauge].max() > 1.0e-6
    assert native_difference[slices["r_cc"]].max() > 1.0e-6


@pytest.mark.parametrize("always_fix_m1_gauge", [True, False])
def test_exact_hvp_differentiates_the_force_with_the_same_gauge_flag(
    always_fix_m1_gauge: bool,
) -> None:
    """At the initial guess fsqz is large, so the two flags select different forces:

    the free-gauge force has nonzero gauge rows, the fixed-gauge force has none.
    """
    indata = _small_3d_input()
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), 5)
    _requires_exact_derivatives(model)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    gauge = autodiff._gauge_entries(model)

    def raw_force(value):
        model.set_state(np.ascontiguousarray(value))
        model.evaluate(2, 2, False, always_fix_m1_gauge)
        return np.asarray(model.get_forces(), dtype=np.float64).copy()

    generator = np.random.default_rng(0)
    direction = generator.standard_normal(state.size)
    direction /= np.linalg.norm(direction)
    step = 1.0e-6
    difference = (
        raw_force(state + step * direction) - raw_force(state - step * direction)
    ) / (2.0 * step)

    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True, always_fix_m1_gauge)
    product = np.asarray(
        model.exact_hessian_vector_product(
            np.ascontiguousarray(direction), always_fix_m1_gauge
        ),
        dtype=np.float64,
    )
    assert np.linalg.norm(product - difference) < 1.0e-6 * np.linalg.norm(difference)
    gauge_rows = np.abs(product[gauge]).max()
    if always_fix_m1_gauge:
        assert gauge_rows == 0.0
    else:
        assert gauge_rows > 1.0e-3 * np.abs(product).max()


def _scalar_of_the_geometry(solver, mid: int):
    def objective(boundary):
        result = solver(boundary)
        return jnp.sum(result.r_cc[mid, 1, :] ** 2) + jnp.sum(result.lambda_sc**2)

    return objective


def _richardson_central_difference(function, boundary, index, step):
    def central(h):
        plus = boundary.copy()
        minus = boundary.copy()
        plus[index] += h
        minus[index] -= h
        return (function(plus) - function(minus)) / (2.0 * h)

    coarse = central(step)
    fine = central(step / 2.0)
    return (4.0 * fine - coarse) / 3.0, abs(fine - coarse)


def test_implicit_vjp_matches_the_pinned_gauge_re_solve() -> None:
    """The boundary gradient of a scalar of the geometry against Richardson-
    extrapolated central differences of the pinned-gauge solve, for the n != 0 m = 1
    modes whose linearized response the fixed-gauge Jacobian did not describe under the
    native gauge, and for rbc(1, 0).

    Measured: the adjoint agrees to 8e-6, 2e-6 and 6e-6 (relative), with the
    two central differences (h, h/2) spread over 5e-6, 8e-6 and 7e-7. The same
    differences of the native-gauge solve spread over 3e-5, 2e-5 and 3e-5 and
    sit 3e-4, 1.2e-3 and 5e-4 away from the adjoint.
    """
    indata = _cth_like_input([25])
    _requires_exact_derivatives(
        _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), 25)
    )
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)
    ntor = indata.ntor
    objective = _scalar_of_the_geometry(solver, mid=12)

    gradient = np.asarray(jax.grad(objective)(jnp.asarray(boundary)))

    def value(candidate):
        return float(objective(jnp.asarray(candidate)))

    for index, step in (
        ((0, 1, ntor + 1), 1.0e-4),  # rbc(1, +1)
        ((1, 1, ntor - 1), 1.0e-4),  # zbs(1, -1)
        ((0, 1, ntor), 1.0e-4),  # rbc(1, 0)
    ):
        reference, spread = _richardson_central_difference(value, boundary, index, step)
        assert abs(gradient[index] - reference) < 1.0e-4 * abs(reference)
        assert spread < 1.0e-4 * abs(reference)

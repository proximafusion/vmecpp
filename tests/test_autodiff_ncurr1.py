# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The implicit adjoint for ncurr=1 (prescribed toroidal current).

For ncurr=1 the half-grid chi' (chipH) is not a fixed input profile: it is
solved from the prescribed toroidal current at every force evaluation
(IdealMhdModel::computeBContra), so it depends on the state through the same
geometry (guu, bsupu, bsupv, gsqrt) the force densities do. The Enzyme kernel
that differentiates the force densities (local_force_composition.h) already
differentiates that chi' solve; chip_state_vjp exposes it as its own reverse
pass, and geometry_state_vjp routes the poloidal_flux cotangent through it.

These tests use examples/data/cth_like_fixed_bdy.json, an ncurr=1 case, at a
reduced ns/ftol for speed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp
from vmecpp import autodiff
from vmecpp.cpp import _vmecpp  # type: ignore

jax.config.update("jax_enable_x64", True)


def _ncurr1_input() -> vmecpp.VmecInput:
    source = vmecpp.VmecInput.from_file("examples/data/cth_like_fixed_bdy.json")
    assert source.ncurr == 1
    return source.model_copy(
        update={
            "ns_array": np.asarray([15]),
            "ftol_array": np.asarray([1.0e-16]),
            "niter_array": np.asarray([4000]),
        }
    )


def _requires_exact_derivatives(indata: vmecpp.VmecInput) -> None:
    ns = int(np.asarray(indata.ns_array)[-1])
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    if not model.has_exact_force_jacobian:
        pytest.skip("needs an Enzyme-enabled build for the exact residual transpose")


def _boundary(indata: vmecpp.VmecInput) -> np.ndarray:
    return np.stack([np.asarray(indata.rbc), np.asarray(indata.zbs)], axis=0).astype(
        np.float64
    )


def _solved_model(indata: vmecpp.VmecInput):
    ns = int(np.asarray(indata.ns_array)[-1])
    model = _vmecpp.VmecModel.create(indata._to_cpp_vmecindata(), ns)
    model.solve()
    return model


def test_exact_hvp_matches_finite_difference_with_prescribed_current() -> None:
    """The Enzyme kernel already differentiates chi' for ncurr=1 (PR #841's tcon branch
    shares the state-dependent chip), so the exact HVP should already be the derivative
    of the raw force -- not just of a frozen-iota force."""
    indata = _ncurr1_input()
    _requires_exact_derivatives(indata)
    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()

    def raw_force(x):
        model.set_state(np.ascontiguousarray(x))
        model.evaluate(2, 2, False)
        return np.asarray(model.get_forces(), dtype=np.float64).copy()

    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(state.size)
    direction /= np.linalg.norm(direction)

    hv = np.asarray(
        model.exact_hessian_vector_product(np.ascontiguousarray(direction)),
        dtype=np.float64,
    )

    errors = {}
    for h in (1.0e-4, 1.0e-5, 1.0e-6):
        fd = (raw_force(state + h * direction) - raw_force(state - h * direction)) / (
            2 * h
        )
        model.set_state(np.ascontiguousarray(state))
        model.evaluate(2, 2, True)
        errors[h] = np.linalg.norm(hv - fd) / np.linalg.norm(fd)
    # O(h^2) convergence of the central difference to the exact HVP.
    assert errors[1.0e-6] < 1.0e-6
    assert errors[1.0e-6] < errors[1.0e-5] < errors[1.0e-4]


def test_exact_hvp_differs_from_a_frozen_iota_model() -> None:
    """A converged ncurr=1 equilibrium re-run with ncurr=0 and its converged iota
    prescribed as a fixed profile is a different functional (chi' no longer responds to
    the state), so its HVP at the same state should disagree with the ncurr=1 exact HVP
    at a resolvable level, even though both reproduce the same force residual at that
    one state."""
    indata = _ncurr1_input()
    _requires_exact_derivatives(indata)
    output = vmecpp.run(indata, verbose=_vmecpp.OutputMode.SILENT)
    iotaf = np.asarray(output.wout.iotaf)
    s_full = np.linspace(0.0, 1.0, iotaf.size)

    frozen_indata = vmecpp.populate_raw_profile(
        indata, "iota", lambda s: np.interp(s, s_full, iotaf)
    ).model_copy(update={"ncurr": 0})

    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(state.size)
    direction /= np.linalg.norm(direction)
    hv_ncurr1 = np.asarray(
        model.exact_hessian_vector_product(np.ascontiguousarray(direction)),
        dtype=np.float64,
    )

    frozen_model = _vmecpp.VmecModel.create(frozen_indata._to_cpp_vmecindata(), 15)
    frozen_model.set_state(np.ascontiguousarray(state))
    frozen_model.evaluate(2, 2, True)
    # Both models reproduce the same converged residual at this shared state.
    assert frozen_model.fsqr < 1.0e-6
    hv_frozen = np.asarray(
        frozen_model.exact_hessian_vector_product(np.ascontiguousarray(direction)),
        dtype=np.float64,
    )
    relative_difference = np.linalg.norm(hv_ncurr1 - hv_frozen) / np.linalg.norm(
        hv_ncurr1
    )
    assert relative_difference > 5.0e-4


def test_chip_state_vjp_matches_finite_difference_of_chip_h() -> None:
    """chip_state_vjp is (dchi'/dx)^T seeded on an arbitrary chip_bar; check it against
    a central difference of chipH(x) along a random direction."""
    indata = _ncurr1_input()
    _requires_exact_derivatives(indata)
    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)

    rng = np.random.default_rng(1)
    chip_bar = rng.standard_normal(model.ns - 1)
    state_bar = np.asarray(
        model.chip_state_vjp(np.ascontiguousarray(chip_bar)), dtype=np.float64
    )

    # linearity of the reverse-mode map is a cheap, independent sanity check
    state_bar_2x = np.asarray(
        model.chip_state_vjp(np.ascontiguousarray(2.0 * chip_bar)), dtype=np.float64
    )
    np.testing.assert_allclose(state_bar_2x, 2.0 * state_bar, rtol=1.0e-12)

    direction = rng.standard_normal(state.size)
    direction /= np.linalg.norm(direction)

    def chip_h(x):
        model.set_state(np.ascontiguousarray(x))
        model.evaluate(2, 2, True)
        return np.asarray(model.chip_h, dtype=np.float64).copy()

    errors = {}
    for h in (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6):
        d_chip = (chip_h(state + h * direction) - chip_h(state - h * direction)) / (
            2 * h
        )
        model.set_state(np.ascontiguousarray(state))
        model.evaluate(2, 2, True)
        lhs = chip_bar @ d_chip
        rhs = state_bar @ direction
        errors[h] = abs(lhs - rhs) / abs(lhs)
    assert errors[1.0e-6] < 1.0e-6
    assert errors[1.0e-6] < errors[1.0e-5] < errors[1.0e-4] < errors[1.0e-3]


def test_geometry_state_vjp_poloidal_flux_route_matches_finite_difference() -> None:
    """geometry_state_vjp's ncurr=1 flux conversion (poloidal_flux_bar -> iota_bar ->
    chip_bar -> chip_state_vjp), checked end to end against a central difference of
    MakeGeometry's own poloidal_flux."""
    indata = _ncurr1_input()
    _requires_exact_derivatives(indata)
    model = _solved_model(indata)
    state = np.asarray(model.get_state(), dtype=np.float64).copy()
    model.set_state(np.ascontiguousarray(state))
    model.evaluate(2, 2, True)

    rng = np.random.default_rng(2)
    poloidal_flux_bar = rng.standard_normal(model.ns)
    coefficient_size = 12 * model.ns * model.mpol * (model.ntor + 1)
    coefficient_bar = np.zeros(coefficient_size)
    state_bar = np.asarray(
        model.geometry_state_vjp(
            np.ascontiguousarray(coefficient_bar),
            np.ascontiguousarray(poloidal_flux_bar),
        ),
        dtype=np.float64,
    )

    def poloidal_flux(x):
        model.set_state(np.ascontiguousarray(x))
        model.evaluate(2, 2, True)
        return np.asarray(model.get_geometry().poloidal_flux, dtype=np.float64).copy()

    direction = rng.standard_normal(state.size)
    direction /= np.linalg.norm(direction)

    errors = {}
    for h in (1.0e-3, 1.0e-4, 1.0e-5):
        d_flux = (
            poloidal_flux(state + h * direction) - poloidal_flux(state - h * direction)
        ) / (2 * h)
        model.set_state(np.ascontiguousarray(state))
        model.evaluate(2, 2, True)
        lhs = poloidal_flux_bar @ d_flux
        rhs = state_bar @ direction
        errors[h] = abs(lhs - rhs) / abs(lhs)
    assert errors[1.0e-5] < 1.0e-4
    assert errors[1.0e-5] < errors[1.0e-4] < errors[1.0e-3]


@pytest.mark.skipif(
    not _vmecpp.VMECPP_ENABLE_ENZYME,
    reason="needs an Enzyme-enabled build for make_solver's exact residual transpose",
)
def test_differentiable_vmec_accepts_ncurr1() -> None:
    indata = _ncurr1_input()
    solver = autodiff.make_solver(indata)
    assert solver.vmec_input.ncurr == 1


def test_quasisymmetry_gradient_through_a_prescribed_current_solve() -> None:
    """End to end: jax.grad of a poloidal_flux + boundary objective through an
    ncurr=1 solve, against a Richardson-extrapolated central difference of the
    re-solved equilibrium.

    VMEC's m=1 constraint gauge moves during the native transient, so small
    relative perturbations (with Richardson extrapolation) are needed for a
    stable finite-difference reference.
    """
    indata = _ncurr1_input()
    _requires_exact_derivatives(indata)
    solver = autodiff.make_solver(indata)
    boundary = _boundary(indata)
    ns = int(np.asarray(indata.ns_array)[-1])
    mid = ns // 2

    def objective(value):
        geometry = solver(value)
        dchi = geometry.poloidal_flux[mid + 1] - geometry.poloidal_flux[mid]
        dphi = geometry.toroidal_flux[mid + 1] - geometry.toroidal_flux[mid]
        iota_mid = dchi / dphi
        return iota_mid + 10.0 * geometry.r_cc[mid, 1, 0] ** 2

    value, gradient = jax.value_and_grad(objective)(jnp.asarray(boundary))
    gradient = np.asarray(gradient)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(gradient))

    def objective_np(value: np.ndarray) -> float:
        model = autodiff._solve_model(indata._to_cpp_vmecindata(), value)
        geometry = model.get_geometry()
        dchi = geometry.poloidal_flux[mid + 1] - geometry.poloidal_flux[mid]
        dphi = geometry.toroidal_flux[mid + 1] - geometry.toroidal_flux[mid]
        iota_mid = dchi / dphi
        r_cc = np.asarray(geometry.coefficients.r_cc).reshape(
            model.ns, model.mpol, model.ntor + 1
        )
        return float(iota_mid + 10.0 * r_cc[mid, 1, 0] ** 2)

    # the three largest-amplitude boundary modes (rbc[m=1,n=0], zbs[m=1,n=0],
    # rbc[m=0,n=0]); a relative perturbation on a near-zero mode collapses to
    # an absolute step far below the equilibrium's own noise floor
    modes = [(0, 1, 4), (1, 1, 4), (0, 0, 4)]
    for row, m, n in modes:
        base = boundary[row, m, n]
        deltas = {}
        for relative_step in (1.0e-3, 5.0e-4):
            h = relative_step * abs(base)
            plus = boundary.copy()
            plus[row, m, n] += h
            minus = boundary.copy()
            minus[row, m, n] -= h
            deltas[h] = (objective_np(plus) - objective_np(minus)) / (2 * h)
        h_coarse, h_fine = sorted(deltas, reverse=True)  # h_coarse = 2 * h_fine
        richardson = (4.0 * deltas[h_fine] - deltas[h_coarse]) / 3.0
        fd_floor = abs(deltas[h_fine] - richardson)  # FD stability at the finer step
        exact = gradient[row, m, n]
        relative_error = abs(richardson - exact) / max(abs(exact), 1.0e-300)
        assert relative_error < 2.0e-3, (
            f"mode {(row, m, n)}: richardson={richardson}, exact={exact}, "
            f"relerr={relative_error}, fd_floor={fd_floor}"
        )

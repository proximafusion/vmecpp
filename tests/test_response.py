# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The first-order response of an equilibrium to changes of its input."""

from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp._response import _combined

TEST_DATA = Path(__file__).resolve().parents[1] / "src" / "vmecpp" / "cpp" / "vmecpp"
TEST_DATA = TEST_DATA / "test_data"
THETA = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
ZETA = np.zeros(1)


def _lasym(vmec_input: vmecpp.VmecInput, **update) -> vmecpp.VmecInput:
    zeros = np.zeros_like(np.asarray(vmec_input.rbc))
    return vmec_input.model_copy(
        update={
            "lasym": True,
            "rbs": zeros.copy(),
            "zbc": zeros.copy(),
            "raxis_s": np.zeros(vmec_input.ntor + 1),
            "zaxis_c": np.zeros(vmec_input.ntor + 1),
            **update,
        }
    )


@pytest.fixture(scope="module")
def solovev():
    """The lasym Solov'ev tokamak at 13 surfaces, its output and its linearization."""
    base = _lasym(
        vmecpp.VmecInput.from_file(TEST_DATA / "solovev.json"),
        ns_array=np.array([13]),
        ftol_array=np.array([1.0e-16]),
        niter_array=np.array([200000]),
        always_fix_m1_gauge=True,
    )
    output = vmecpp.run(base, max_threads=1, verbose=False)
    return base, output, vmecpp.Linearization(output)


@pytest.fixture(scope="module")
def solovev_free():
    """The Solov'ev free-boundary tokamak at 16 surfaces, its output and its
    linearization."""
    base = vmecpp.VmecInput.from_file(TEST_DATA / "solovev_free_bdy.json")
    base = base.model_copy(
        update={
            "mgrid_file": str(TEST_DATA / "mgrid_solovev.nc"),
            "ns_array": np.array([16]),
            "ftol_array": np.array([1.0e-16]),
            "niter_array": np.array([200000]),
            "always_fix_m1_gauge": True,
        }
    )
    output = vmecpp.run(base, max_threads=1, verbose=False)
    return base, output, vmecpp.Linearization(output)


@pytest.fixture(scope="module")
def cth_like():
    """The CTH-like stellarator at 9 surfaces with its pressure raised eightfold, to a
    beta of 1.6 %, its output and its linearization."""
    base = vmecpp.VmecInput.from_file(TEST_DATA / "cth_like_fixed_bdy.json")
    base = base.model_copy(
        update={
            "ns_array": np.array([9]),
            "ftol_array": np.array([1.0e-16]),
            "niter_array": np.array([200000]),
            "always_fix_m1_gauge": True,
            "pres_scale": 8.0 * base.pres_scale,
        }
    )
    output = vmecpp.run(base, max_threads=1, verbose=False)
    return base, output, vmecpp.Linearization(output)


def _shaped(base: vmecpp.VmecInput, amplitude: float) -> vmecpp.VmecInput:
    """The boundary with its m = 2 and m = 3 coefficients changed."""
    shaped = base.model_copy(deep=True)
    shaped.rbc[2, base.ntor] += amplitude
    shaped.zbs[3, base.ntor] += 0.5 * amplitude
    return shaped


def _coil(base: vmecpp.VmecInput, factor: float) -> vmecpp.VmecInput:
    """The current of the second circuit scaled by factor."""
    scaled = base.model_copy(deep=True)
    scaled.extcur = np.asarray(base.extcur).copy()
    scaled.extcur[1] *= factor
    return scaled


def _force_ratios(linearization: vmecpp.Linearization, make, amplitudes) -> list[float]:
    """For the change ``make(1.0)`` of the input, at each amplitude: the largest force
    at the state the response predicts over the largest force with the interior unmoved.

    The first is second order in the amplitude, the second first order.
    """
    response = linearization.response_to(make(1.0))
    boundary_only = np.zeros_like(response.state)
    fixed = np.setdiff1d(np.arange(response.state.size), linearization.free)
    boundary_only[fixed] = response.state[fixed]
    ratios = []
    for eps in amplitudes:
        unmoved = linearization.force_of(
            make(eps), linearization.state + eps * boundary_only
        )
        predicted = linearization.predicted_force(make(1.0), eps)
        ratios.append(float(np.abs(predicted).max() / np.abs(unmoved).max()))
    return ratios


def test_the_combined_basis_reproduces_the_wout(solovev):
    base, output, linearization = solovev
    g = linearization._base_geometry
    for name, first, second, cosine in (
        ("rmnc", "r_cc", "r_ss", True),
        ("zmns", "z_sc", "z_cs", False),
        ("zmnc", "z_cc", "z_ss", True),
    ):
        combined = _combined(
            linearization.xm,
            linearization.xn,
            base.nfp,
            g[first],
            g[second],
            cosine=cosine,
        )
        # measured 1e-7, the hot restart's reconstruction of the state
        np.testing.assert_allclose(combined, getattr(output.wout, name), atol=1.0e-6)


def test_a_rigid_shift_moves_every_surface(solovev):
    base, _, linearization = solovev
    shifted = base.model_copy(deep=True)
    shifted.zbc[0, base.ntor] += 1.0
    response = linearization.response_to(shifted)
    k = response.mode(0, 0)
    # measured 1 + 5e-8 on every surface, 5e-10 elsewhere
    np.testing.assert_allclose(response.zmnc[k], 1.0, rtol=0.0, atol=1.0e-6)
    others = np.concatenate(
        [
            response.rmnc.ravel(),
            response.zmns.ravel(),
            response.rmns.ravel(),
            np.delete(response.zmnc, k, axis=0).ravel(),
            response.lmns_full.ravel(),
            response.lmnc_full.ravel(),
        ]
    )
    assert np.abs(others).max() < 1.0e-7
    # the shift along the normal is the z component of the unit normal
    assert np.abs(response.normal_displacement(THETA, ZETA)).max() < 1.0 + 1.0e-6


def test_the_response_is_the_derivative_of_the_equilibrium(solovev):
    """The secant of two nonlinear solves at +/- eps converges to the response as eps
    falls, compared in the normal displacement of the surfaces."""
    base, output, linearization = solovev
    linear = linearization.response_to(_shaped(base, 1.0)).normal_displacement(
        THETA, ZETA
    )
    scale = np.abs(linear).max()
    mismatch = []
    for eps in (4.0e-3, 2.0e-3):
        secants = []
        for sign in (1.0, -1.0):
            solved = vmecpp.run(
                _shaped(base, sign * eps),
                max_threads=1,
                verbose=False,
                restart_from=output,
            )
            secants.append(
                linearization.geometry_difference(solved).normal_displacement(
                    THETA, ZETA
                )
            )
        # measured 1.5e-3 and 8.0e-4
        mismatch.append(np.abs(secants[0] / eps - linear).max() / scale)
        # the central secant has no second-order term; measured 1.6e-4 and 1.9e-4
        central = (secants[0] - secants[1]) / (2.0 * eps)
        assert np.abs(central - linear).max() < 1.0e-3 * scale
    assert 1.7 < mismatch[0] / mismatch[1] < 2.3


def test_the_force_at_the_predicted_state_is_second_order(solovev):
    """With the boundary moved and the interior unmoved the force is first order in the
    amplitude; at the predicted state it is second order."""
    base, _, linearization = solovev
    ratios = _force_ratios(linearization, lambda a: _shaped(base, a), (1.0e-2, 5.0e-3))
    # measured 1.0e-4 and 5.1e-5
    assert ratios[0] < 1.0e-3
    assert 1.7 < ratios[0] / ratios[1] < 2.3


def test_the_free_boundary_responds_to_a_coil_current(solovev_free):
    """For a change of a coil current the force at the unmoved state is first order in
    the amplitude and the force at the predicted state second order; the secant of two
    nonlinear solves agrees with the response to the accuracy the solves have along the
    soft mode of the free boundary."""
    base, output, linearization = solovev_free
    coil = _coil(base, 2.0)
    ratios = _force_ratios(
        linearization, lambda a: _coil(base, 1.0 + a), (2.0e-2, 1.0e-2)
    )
    # measured 7.6e-2 and 3.8e-2
    assert ratios[0] < 0.15
    assert 1.7 < ratios[0] / ratios[1] < 2.3

    linear = linearization.response_to(coil).normal_displacement(THETA, ZETA)
    eps = 1.0e-2
    secants = []
    for sign in (1.0, -1.0):
        solved = vmecpp.run(
            _coil(base, 1.0 + sign * eps),
            max_threads=1,
            verbose=False,
            restart_from=output,
        )
        secants.append(
            linearization.geometry_difference(solved).normal_displacement(THETA, ZETA)
        )
    central = (secants[0] - secants[1]) / (2.0 * eps)
    # measured 3.5e-3
    assert np.abs(central - linear).max() < 1.0e-2 * np.abs(linear).max()


def test_a_three_dimensional_high_beta_equilibrium_responds_to_second_order(cth_like):
    """For a three-dimensional change of the boundary, a change of the pressure and a
    change of the current of the CTH-like stellarator at beta 1.6 %, the force at the
    predicted state is second order in the amplitude."""
    base, output, linearization = cth_like
    assert output.wout.betatotal > 0.015

    def shaped(amplitude):
        moved = base.model_copy(deep=True)
        moved.rbc[2, base.ntor + 1] += amplitude
        moved.zbs[1, base.ntor - 1] += 0.5 * amplitude
        return moved

    def pressure(amplitude):
        return base.model_copy(
            update={"pres_scale": base.pres_scale * (1.0 + amplitude)}
        )

    def current(amplitude):
        return base.model_copy(update={"curtor": base.curtor * (1.0 + amplitude)})

    # measured 5.4e-4 and 2.6e-4, 3.1e-3 and 1.5e-3, 7.0e-3 and 3.5e-3
    for make, amplitudes, bound in (
        (shaped, (2.0e-3, 1.0e-3), 2.0e-3),
        (pressure, (1.0e-2, 5.0e-3), 1.0e-2),
        (current, (1.0e-2, 5.0e-3), 2.0e-2),
    ):
        ratios = _force_ratios(linearization, make, amplitudes)
        assert ratios[0] < bound
        assert 1.7 < ratios[0] / ratios[1] < 2.3


def test_a_uniform_scaling_of_the_field_leaves_the_geometry(solovev_free):
    """Doubling every coil current and the toroidal flux and tripling the pressure scale
    changes the field by the same factor everywhere to first order, so the geometry does
    not respond."""
    base, _, linearization = solovev_free
    scaling = base.model_copy(deep=True)
    scaling.extcur = 2.0 * np.asarray(base.extcur)
    scaling.phiedge = 2.0 * base.phiedge
    scaling.pres_scale = 3.0 * base.pres_scale
    currents = base.model_copy(deep=True)
    currents.extcur = 2.0 * np.asarray(base.extcur)
    nothing = linearization.response_to(scaling).normal_displacement(THETA, ZETA)
    something = linearization.response_to(currents).normal_displacement(THETA, ZETA)
    # measured 3e-4
    assert np.abs(nothing).max() < 1.0e-3 * np.abs(something).max()


def _tokamak(ns: int) -> vmecpp.VmecInput:
    """An elongated tokamak with iota = 0.8 - 0.5 s, so that q = 2 at s = 0.6."""
    vmec_input = vmecpp.VmecInput(
        lasym=False,
        nfp=1,
        mpol=6,
        ntor=1,
        ns_array=np.array([ns]),
        ftol_array=np.array([1.0e-15]),
        niter_array=np.array([100000]),
        phiedge=5.0,
        ncurr=0,
        ai=np.array([0.8, -0.5]),
        pmass_type="power_series",
        am=np.array([1.0, -1.0]),
        pres_scale=1.0e3,
        gamma=0.0,
        always_fix_m1_gauge=True,
        raxis_c=np.array([1.5, 0.0]),
        zaxis_s=np.array([0.0, 0.0]),
    )
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(np.array([[1.5], [1.0]]), 6, 1)
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(np.array([[0.0], [1.4]]), 6, 1)
    return vmec_input


def _lambda_peak(response: vmecpp.GeometryResponse, m: int, n: int, ns: int):
    """The peak of |lambda_mn| over the interior surfaces: its height, its position in s
    from the parabola through the three surfaces around it, and the width in s over
    which |lambda_mn| exceeds half of it."""
    lam = np.abs(response.lmns_full[response.mode(m, n)])
    s = np.linspace(0.0, 1.0, ns)
    j = 1 + int(np.argmax(lam[1:-1]))
    a, b, c = lam[j - 1], lam[j], lam[j + 1]
    offset = 0.5 * (a - c) / (a - 2.0 * b + c)
    above = np.flatnonzero(lam > 0.5 * b)
    return b, s[j] + offset * (s[1] - s[0]), s[above[-1]] - s[above[0]]


def test_a_resonant_perturbation_draws_a_shielding_response_at_the_rational_surface():
    """The (2, 1) harmonic of lambda answers a cos(2 theta - zeta) boundary perturbation
    with a peak at the q = 2 surface that grows and narrows as the radial resolution
    doubles, the shielding current sheet of the resonant surface; the non-resonant
    (1, 1) response does neither."""
    peaks = {}
    for ns in (25, 51):
        vmec_input = _tokamak(ns)
        output = vmecpp.run(vmec_input, max_threads=1, verbose=False)
        linearization = vmecpp.Linearization(output)
        for m, n in ((2, 1), (1, 1)):
            perturbed = vmec_input.model_copy(deep=True)
            perturbed.rbc[m, 1 + n] += 1.0
            peaks[ns, m, n] = _lambda_peak(
                linearization.response_to(perturbed), m, n, ns
            )
    # measured 0.71 at s = 0.56 over a half width of 0.38 at ns 25, and 1.02 at s = 0.57
    # over 0.26 at ns 51
    for ns in (25, 51):
        assert abs(peaks[ns, 2, 1][1] - 0.6) < 0.05
    assert peaks[51, 2, 1][0] > 1.2 * peaks[25, 2, 1][0]
    assert peaks[51, 2, 1][2] < peaks[25, 2, 1][2]
    # measured 0.63 and 0.66
    assert peaks[51, 1, 1][0] < 1.15 * peaks[25, 1, 1][0]


def test_the_linearization_checks_its_input(solovev):
    base, output, linearization = solovev
    schedule = output.model_copy(
        update={"input": base.model_copy(update={"ns_array": np.array([5, 13])})}
    )
    with pytest.raises(ValueError, match="single ns_array"):
        vmecpp.Linearization(schedule)
    with pytest.raises(ValueError, match="cannot follow"):
        linearization.response_to(base.model_copy(update={"delt": 0.5 * base.delt}))
    response = linearization.response_to(_shaped(base, 1.0))
    with pytest.raises(ValueError, match="no mode"):
        response.mode(base.mpol, 0)

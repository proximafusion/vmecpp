# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Error-field correction from the linear response of an equilibrium."""

import numpy as np
import pytest

import vmecpp

MPOL, NTOR = 6, 1
THETA = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
ZETA = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
EPS = 1.0e-3
ERROR = {("rbc", 2, 1): 1.0}
KNOBS = [("zbs", 2, 1), ("rbc", 3, 1)]


def _tokamak(ns: int) -> vmecpp.VmecInput:
    """An elongated tokamak with iota = 0.8 - 0.5 s, so that q = 2 at s = 0.6."""
    vmec_input = vmecpp.VmecInput(
        lasym=False,
        nfp=1,
        mpol=MPOL,
        ntor=NTOR,
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
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[1.5], [1.0]]), MPOL, NTOR
    )
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[0.0], [1.4]]), MPOL, NTOR
    )
    return vmec_input


def _perturbed(base: vmecpp.VmecInput, coefficients: dict) -> vmecpp.VmecInput:
    moved = base.model_copy(deep=True)
    for (field, m, n), amplitude in coefficients.items():
        getattr(moved, field)[m, NTOR + n] += amplitude
    return moved


def _slope_jump(lam: np.ndarray, s: np.ndarray, s_resonant: float) -> float:
    """The jump of the radial slope of lam across s_resonant, from linear fits over the
    surfaces between 0.1 and 0.25 away on either side."""
    inside = (s_resonant - s >= 0.1) & (s_resonant - s <= 0.25)
    outside = (s - s_resonant >= 0.1) & (s - s_resonant <= 0.25)
    return (
        np.polyfit(s[outside], lam[outside], 1)[0]
        - np.polyfit(s[inside], lam[inside], 1)[0]
    )


def _peak(lam: np.ndarray) -> float:
    return float(np.abs(lam[1:-1]).max())


def _resonant_case(ns: int):
    """The tokamak at ns surfaces: its input, output and linearization, the radial grid
    and the position of its q = 2 surface."""
    base = _tokamak(ns)
    output = vmecpp.run(base, max_threads=1, verbose=False)
    s = np.linspace(0.0, 1.0, ns)
    iota = np.asarray(output.wout.iotaf)
    j = int(np.flatnonzero(np.diff(np.sign(iota - 0.5)))[0])
    s_resonant = s[j] + (0.5 - iota[j]) / (iota[j + 1] - iota[j]) * (s[1] - s[0])
    return base, output, vmecpp.Linearization(output), s, s_resonant


def _corrected(base, linearization, s, s_resonant, knob):
    """The perturbation corrected by the knob at the amplitude that nulls the slope jump
    of the error field's lambda(2,1), and the responses to the error field and to it."""
    response = linearization.response_to(_perturbed(base, ERROR))
    k = response.mode(2, 1)
    alone = linearization.response_to(_perturbed(base, {knob: 1.0}))
    amplitude = -_slope_jump(response.lmns_full[k], s, s_resonant) / _slope_jump(
        alone.lmns_full[k], s, s_resonant
    )
    corrected = {**ERROR, knob: amplitude}
    return corrected, response, linearization.response_to(_perturbed(base, corrected))


def _secant(linearization, base, output, coefficients):
    """The central secant of two solves at +/- EPS of the perturbation: lambda in the
    layout of a response, and the normal displacement."""
    solved = []
    for sign in (1.0, -1.0):
        scaled = {key: sign * EPS * value for key, value in coefficients.items()}
        out = vmecpp.run(
            _perturbed(base, scaled), max_threads=1, verbose=False, restart_from=output
        )
        solved.append(linearization.geometry_difference(out))
    lam = (solved[0].lmns_full - solved[1].lmns_full) / (2.0 * EPS)
    xi = (
        solved[0].normal_displacement(THETA, ZETA)
        - solved[1].normal_displacement(THETA, ZETA)
    ) / (2.0 * EPS)
    return lam, xi


@pytest.fixture(scope="module")
def coarse():
    """The case at 25 surfaces, with the secant of the solves for the error field."""
    base, output, linearization, s, s_resonant = _resonant_case(25)
    secant = _secant(linearization, base, output, ERROR)
    return base, output, linearization, s, s_resonant, secant


@pytest.fixture(scope="module")
def fine():
    """The case at 51 surfaces."""
    return _resonant_case(51)


def test_the_error_field_draws_a_slope_jump_that_the_solves_reproduce(coarse):
    base, _, linearization, s, s_resonant, (lam, _) = coarse
    response = linearization.response_to(_perturbed(base, ERROR))
    k = response.mode(2, 1)
    linear = _slope_jump(response.lmns_full[k], s, s_resonant)
    solved = _slope_jump(lam[k], s, s_resonant)
    # measured 3.89 and 3.87
    assert abs(linear) > 1.0
    assert abs(solved - linear) < 2.0e-2 * abs(linear)


@pytest.mark.parametrize("knob", KNOBS)
def test_a_correction_nulls_the_shielding_current_of_the_resonant_surface(coarse, knob):
    """A second boundary harmonic, scaled by the ratio of the slope jumps of lambda(2,1)
    that the error field and it draw, removes the jump: exactly in the linearization,
    and in the solves of the corrected perturbation to the accuracy of the secant, while
    the surfaces stay displaced."""
    base, output, linearization, s, s_resonant, (lam_error, xi_error) = coarse
    corrected, response, linear = _corrected(base, linearization, s, s_resonant, knob)
    k = response.mode(2, 1)
    jump_error = _slope_jump(response.lmns_full[k], s, s_resonant)
    # the response is linear in the perturbation; measured 3e-7 of the error field's jump
    assert abs(_slope_jump(linear.lmns_full[k], s, s_resonant)) < 1.0e-5 * abs(
        jump_error
    )
    lam, xi = _secant(linearization, base, output, corrected)
    # measured 0.021 and 0.023 against 3.87
    assert abs(_slope_jump(lam[k], s, s_resonant)) < 2.0e-2 * abs(
        _slope_jump(lam_error[k], s, s_resonant)
    )
    # measured 1.25 and 3.05 against 1.11
    assert np.abs(xi).max() > 0.5 * np.abs(xi_error).max()


def test_the_correction_removes_the_part_of_the_response_that_grows_with_the_resolution(
    coarse, fine
):
    """From 25 to 51 surfaces the peak of the error field's lambda(2,1) grows by 43 %;
    the peaks of the corrected responses, each corrected at its own resolution, by 10 %
    and 6 %."""
    peaks = {}
    for ns, case in ((25, coarse[:5]), (51, fine)):
        base, _, linearization, s, s_resonant = case
        for knob in KNOBS:
            _, response, linear = _corrected(base, linearization, s, s_resonant, knob)
            k = response.mode(2, 1)
            peaks[ns, "error"] = _peak(response.lmns_full[k])
            peaks[ns, knob] = _peak(linear.lmns_full[k])
    assert peaks[51, "error"] > 1.3 * peaks[25, "error"]
    for knob in KNOBS:
        assert peaks[51, knob] < 1.2 * peaks[25, knob]

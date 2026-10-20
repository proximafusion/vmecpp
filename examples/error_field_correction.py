# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Error-field correction from the linear response of an equilibrium.

The boundary perturbation ``cos(2 theta - zeta)`` of the tokamak of
``perturbed_equilibrium.py`` is resonant at its q = 2 surface, where the equilibrium
answers with a shielding current: the (2, 1) harmonic of lambda peaks there, and the
peak grows with the radial resolution while the rest of the response converges, the
singular part of the response. Its measure here is the jump of the radial slope of the
harmonic between linear fits over the surfaces 0.1 to 0.25 inside and outside the
resonant surface. Error-field correction nulls the resonant component of a field at the
rational surface with a correction field of another spectrum. Here a correction is a
second boundary harmonic, ``sin(2 theta - zeta)`` in Z, of the error's helicity, or
``cos(3 theta - zeta)`` in R, which the shaping couples to the (2, 1) harmonic.
``vmecpp.Linearization`` gives the slope jump per unit of the error and per unit of a
correction, so the amplitude that nulls the jump is the ratio of the two, and the
response to the corrected perturbation follows without another solve. The corrected
responses converge with the radial resolution where the uncorrected one grows, and
solved nonlinearly at +/- eps the corrected perturbations keep their displaced surfaces
without the jump.
"""

import numpy as np

import vmecpp

MPOL, NTOR = 6, 1
THETA = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
ZETA = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
ERROR = {("rbc", 2, 1): 1.0}
KNOBS = {
    "sin(2 theta - zeta) in Z": ("zbs", 2, 1),
    "cos(3 theta - zeta) in R": ("rbc", 3, 1),
}


def tokamak(ns: int) -> vmecpp.VmecInput:
    """An elongated tokamak with iota = 0.8 - 0.5 s, so q = 2 at s = 0.6."""
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


def perturbed(base: vmecpp.VmecInput, coefficients: dict) -> vmecpp.VmecInput:
    """The input with the boundary coefficients ``{(field, m, n): amplitude}`` added."""
    moved = base.model_copy(deep=True)
    for (field, m, n), amplitude in coefficients.items():
        getattr(moved, field)[m, NTOR + n] += amplitude
    return moved


def slope_jump(lam: np.ndarray, s: np.ndarray, s_resonant: float) -> float:
    """The jump of the radial slope of ``lam`` across ``s_resonant``, from linear fits
    over the surfaces between 0.1 and 0.25 away on either side, clear of the peak."""
    inside = (s_resonant - s >= 0.1) & (s_resonant - s <= 0.25)
    outside = (s - s_resonant >= 0.1) & (s - s_resonant <= 0.25)
    return (
        np.polyfit(s[outside], lam[outside], 1)[0]
        - np.polyfit(s[inside], lam[inside], 1)[0]
    )


def peak_and_width(lam: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """The largest |lam| over the interior surfaces and the width in s over which |lam|
    exceeds half of it."""
    magnitude = np.abs(lam)
    j = 1 + int(np.argmax(magnitude[1:-1]))
    above = np.flatnonzero(magnitude > 0.5 * magnitude[j])
    return float(magnitude[j]), float(s[above[-1]] - s[above[0]])


def resonant_case(ns: int):
    """The tokamak at ``ns`` surfaces: its input, output and linearization, the radial
    grid and the position of its q = 2 surface."""
    base = tokamak(ns)
    output = vmecpp.run(base, verbose=False)
    s = np.linspace(0.0, 1.0, ns)
    iota = np.asarray(output.wout.iotaf)
    j = int(np.flatnonzero(np.diff(np.sign(iota - 0.5)))[0])
    s_resonant = s[j] + (0.5 - iota[j]) / (iota[j + 1] - iota[j]) * (s[1] - s[0])
    return base, output, vmecpp.Linearization(output), s, s_resonant


def corrections(base, linearization, s, s_resonant):
    """The responses to the error field and to the perturbations corrected by each knob,
    with the amplitude of the knob that nulls the slope jump."""
    response = linearization.response_to(perturbed(base, ERROR))
    k = response.mode(2, 1)
    jump = slope_jump(response.lmns_full[k], s, s_resonant)
    cases = {"error field": (ERROR, response)}
    amplitudes = {}
    for name, knob in KNOBS.items():
        alone = linearization.response_to(perturbed(base, {knob: 1.0}))
        amplitudes[name] = -jump / slope_jump(alone.lmns_full[k], s, s_resonant)
        corrected = {**ERROR, knob: amplitudes[name]}
        cases[f"corrected by {name}"] = (
            corrected,
            linearization.response_to(perturbed(base, corrected)),
        )
    return k, cases, amplitudes


print(
    " ns   error field: peak, half width, slope jump   "
    + "   ".join(f"{name}: amplitude, corrected peak" for name in KNOBS)
)
coarse = None
for ns in (25, 51, 101):
    base, output, linearization, s, s_resonant = resonant_case(ns)
    k, cases, amplitudes = corrections(base, linearization, s, s_resonant)
    lam = cases["error field"][1].lmns_full[k]
    height, width = peak_and_width(lam, s)
    columns = "   ".join(
        f"{amplitudes[name]:9.3f}  {peak_and_width(cases[f'corrected by {name}'][1].lmns_full[k], s)[0]:14.3f}"
        + " " * (len(name) - 10)
        for name in KNOBS
    )
    print(
        f"{ns:3d}   {height:.3f}  {width:.3f}  {slope_jump(lam, s, s_resonant):.3f}                     {columns}"
    )
    if coarse is None:
        coarse = (base, output, linearization, s, s_resonant, k, cases)

# the perturbations solved at +/- eps on the coarse grid, the central secant in place of
# the response
base, output, linearization, s, s_resonant, k, cases = coarse
eps = 1.0e-3
print(
    f"\niota = 1/2 at s = {s_resonant:.3f}; at {len(s)} surfaces, the response and the solves:"
)
print(" " * 48 + "slope jump   peak of lambda(2,1)   normal displacement")
for name, (coefficients, response) in cases.items():
    lam = response.lmns_full[k]
    print(
        f"{name + ', linear':46s}  {slope_jump(lam, s, s_resonant):+8.4f}   {peak_and_width(lam, s)[0]:8.3f}"
        f"             {np.abs(response.normal_displacement(THETA, ZETA)).max():8.3f}"
    )
    solved = []
    for sign in (1.0, -1.0):
        scaled = {key: sign * eps * value for key, value in coefficients.items()}
        out = vmecpp.run(perturbed(base, scaled), verbose=False, restart_from=output)
        solved.append(linearization.geometry_difference(out))
    lam = (solved[0].lmns_full[k] - solved[1].lmns_full[k]) / (2.0 * eps)
    xi = (
        solved[0].normal_displacement(THETA, ZETA)
        - solved[1].normal_displacement(THETA, ZETA)
    ) / (2.0 * eps)
    print(
        f"{name + ', solved':46s}  {slope_jump(lam, s, s_resonant):+8.4f}   {peak_and_width(lam, s)[0]:8.3f}"
        f"             {np.abs(xi).max():8.3f}"
    )

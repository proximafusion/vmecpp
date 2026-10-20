# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The first-order response of an equilibrium to a resonant boundary perturbation.

``vmecpp.Linearization`` factorizes the force Jacobian of a converged equilibrium once
and then gives the first-order response of the geometry to any change of the input, a
boundary coefficient here, without another solve. The equilibrium is a tokamak whose
rotational transform falls from 0.8 to 0.3, through 1/2 at s = 0.6. A boundary
perturbation ``cos(2 theta - zeta)`` is resonant there, and the plasma answers with a
peak of the (2, 1) harmonic of lambda, the quantity that carries the surface current in
VMEC's representation, at the resonant surface. The peak sharpens as the radial
resolution grows: it stands for the shielding current sheet that an equilibrium with
nested flux surfaces needs at a resonant surface and that no grid resolves. The
non-resonant perturbation ``cos(theta - zeta)`` draws no such peak.
"""

import numpy as np

import vmecpp

MPOL, NTOR = 6, 1


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


print(
    "ns   perturbation         lambda peak   at s    half width in s   iota = 1/2 at s"
)
for ns in (25, 51):
    vmec_input = tokamak(ns)
    output = vmecpp.run(vmec_input, verbose=False)
    linearization = vmecpp.Linearization(output)
    s = np.linspace(0.0, 1.0, ns)
    iota = np.asarray(output.wout.iotaf)
    j = int(np.flatnonzero(np.diff(np.sign(iota - 0.5)))[0])
    s_resonant = s[j] + (0.5 - iota[j]) / (iota[j + 1] - iota[j]) * (s[1] - s[0])
    for label, m, n in (("resonant (2, 1)", 2, 1), ("non-resonant (1, 1)", 1, 1)):
        perturbed = vmec_input.model_copy(deep=True)
        perturbed.rbc[m, NTOR + n] += 1.0
        response = linearization.response_to(perturbed)
        lam = np.abs(response.lmns_full[response.mode(m, n)])
        # the peak over the interior surfaces, placed by the parabola through its
        # neighbours, and the width over which lambda exceeds half of it
        j = 1 + int(np.argmax(lam[1:-1]))
        offset = (
            0.5 * (lam[j - 1] - lam[j + 1]) / (lam[j - 1] - 2.0 * lam[j] + lam[j + 1])
        )
        above_half = np.flatnonzero(lam > 0.5 * lam[j])
        width = s[above_half[-1]] - s[above_half[0]]
        print(
            f"{ns:3d}  {label:19s}  {lam[j]:.3f}         {s[j] + offset * (s[1] - s[0]):.2f}"
            f"    {width:.2f}              {s_resonant:.3f}"
        )

# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""An equilibrium without a magnetic axis.

With ``VmecInput.inner_boundary_s`` in (0, 1), the plasma fills the shell between a
fixed inner boundary, given by ``rbc_inner`` and ``zbs_inner`` like the boundary by
``rbc`` and ``zbs``, and the boundary. The ``ns`` flux surfaces divide
``[inner_boundary_s, 1]`` evenly, and the profiles keep their meaning in the
normalized toroidal flux s.

This example solves the CTH-like equilibrium, takes its flux surface at s = 0.2 as
the inner boundary of a shell, solves the shell, and compares the two on the
surfaces they share.
"""

from pathlib import Path

import numpy as np

import vmecpp

ns = 51
cut = 10  # the surface at s = cut / (ns - 1) = 0.2

vmec_input = vmecpp.VmecInput.from_file(
    Path("examples") / "data" / "cth_like_fixed_bdy.json"
)
vmec_input.ns_array = np.array([ns])
vmec_input.ftol_array = np.array([1.0e-14])
vmec_input.niter_array = np.array([5000])
full = vmecpp.run(vmec_input, verbose=False)
wout = full.wout


def surface(j: int) -> tuple[np.ndarray, np.ndarray]:
    """The coefficients of flux surface j in the layout of rbc and zbs."""
    rbc = np.zeros((vmec_input.mpol, 2 * vmec_input.ntor + 1))
    zbs = np.zeros_like(rbc)
    for mn, (m, xn) in enumerate(zip(wout.xm, wout.xn, strict=True)):
        n = int(xn) // vmec_input.nfp
        rbc[m, vmec_input.ntor + n] = wout.rmnc[mn, j]
        zbs[m, vmec_input.ntor + n] = wout.zmns[mn, j]
    return rbc, zbs


shell_input = vmec_input.model_copy(deep=True)
# the boundary as the equilibrium carries it, in the same poloidal angle as the
# inner boundary
shell_input.rbc, shell_input.zbs = surface(ns - 1)
shell_input.rbc_inner, shell_input.zbs_inner = surface(cut)
shell_input.inner_boundary_s = cut / (ns - 1)
shell_input.ns_array = np.array([ns - cut])
shell = vmecpp.run(shell_input, verbose=False)

print(f"equilibrium with a magnetic axis: {wout.itfsq} iterations, ns = {ns}")
print(
    f"shell outside s = {shell_input.inner_boundary_s}: "
    f"{shell.wout.itfsq} iterations, ns = {ns - cut}"
)
for name in ("rmnc", "zmns"):
    shared = np.asarray(getattr(wout, name))[:, cut:]
    difference = np.abs(np.asarray(getattr(shell.wout, name)) - shared).max()
    print(
        f"largest difference of {name} on the shared surfaces: {difference:.1e} "
        f"(largest coefficient {np.abs(shared).max():.2f})"
    )
print(
    f"toroidal flux enclosed by the inner boundary: {shell.wout.phi[0]:.6f} Wb, "
    f"in the equilibrium {wout.phi[cut]:.6f} Wb"
)
print(f"plasma volume: shell {shell.wout.volume_p:.4f}, full {wout.volume_p:.4f} m^3")

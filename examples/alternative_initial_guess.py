# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Alternative initial guesses for the flux-surface geometry.

By default VMEC++ seeds its interior by interpolating between the magnetic axis
and the boundary (m=0 linear in the normalized flux s, higher poloidal modes as
s^(m/2)). Two published alternatives instead construct the whole interior
coordinate map up front:

  * zeno_guess     -- Tecchiolli's variational construction (arXiv:2405.08173):
                      minimize a Jacobian-based action over a Fourier-Zernike
                      basis with the boundary held fixed.
  * map2disc_guess -- conformal map of each cross-section to the unit disc
                      (Babin, PPCF 2025): a C2 diffeomorphism, so the interior
                      cannot fold (positive Jacobian by construction).

Both return a VmecOutput you pass to vmecpp.run via restart_from. They are most
useful when you want a known-valid interior independent of the linear guess
(single-resolution solves, guess-sensitivity studies); for typical boundaries
VMEC++'s radial multi-grid already supplies a valid coarse-grid geometry.
"""

from pathlib import Path

import vmecpp

vmec_input = vmecpp.VmecInput.from_file(
    Path("examples") / "data" / "cth_like_fixed_bdy.json"
)

# Reference solve with the default (linear) interior guess.
reference = vmecpp.run(vmec_input, verbose=False)
print(f"default guess : volume_p = {reference.wout.volume_p:.6f}")

# Build a variational (Zeno) interior and hot-restart VMEC++ from it.
guess = vmecpp.zeno_guess(vmec_input)
from_zeno = vmecpp.run(vmec_input, restart_from=guess, verbose=False)
print(f"zeno guess    : volume_p = {from_zeno.wout.volume_p:.6f}")

# map2disc is an optional dependency (gitlab.mpcdf.mpg.de/gvec-group/map2disc);
# when present it is a drop-in replacement:
#     guess = vmecpp.map2disc_guess(vmec_input)
#     from_map2disc = vmecpp.run(vmec_input, restart_from=guess, verbose=False)

# A valid (non-overlapping) interior converges to the same equilibrium.
assert abs(from_zeno.wout.volume_p - reference.wout.volume_p) < 1e-6
print("both guesses converge to the same equilibrium")

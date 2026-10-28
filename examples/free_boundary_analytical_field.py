# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Free-boundary run in an analytical magnetic field instead of an mgrid file."""

from pathlib import Path

import numpy as np
from scipy.special import ellipe, ellipk

import vmecpp

TEST_DATA_DIR = Path("src") / "vmecpp" / "cpp" / "vmecpp" / "test_data"
MU0 = 4.0e-7 * np.pi

# The coils of coils.solovev: a straight wire along the Z axis carrying extcur[0],
# and circular loops (radius R, height Z) carrying extcur[1:].
LOOPS = np.array(
    [
        [0.867845, 1.86006],
        [4.49628, 3.69583],
        [8.63810, -2.46839],
        [8.63847, 2.46836],
        [4.49620, -3.69589],
        [0.867842, -1.86005],
        [1.19984, 3.34106],
        [2.62980, 5.24924],
        [9.65488, -3.29570],
        [9.65501, 3.29541],
        [2.62966, -5.24925],
        [1.19984, -3.34102],
    ]
)

vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev_free_bdy.json")


def magnetic_field(r, _phi, z):
    """Cylindrical components of B at the points (r, phi, z) on the boundary."""
    # field of a circular loop of radius a at height z0, in complete elliptic integrals
    a = LOOPS[:, 0, None]
    dz = z - LOOPS[:, 1, None]
    alpha2 = a**2 + r**2 + dz**2 - 2 * a * r
    beta2 = a**2 + r**2 + dz**2 + 2 * a * r
    m = 1.0 - alpha2 / beta2
    scale = MU0 * vmec_input.extcur[1:, None] / (2 * np.pi * alpha2 * np.sqrt(beta2))
    b_r = scale * dz / r * ((a**2 + r**2 + dz**2) * ellipe(m) - alpha2 * ellipk(m))
    b_z = scale * ((a**2 - r**2 - dz**2) * ellipe(m) + alpha2 * ellipk(m))
    # field of the straight wire
    b_phi = MU0 * vmec_input.extcur[0] / (2 * np.pi * r)
    return b_r.sum(axis=0), b_phi, b_z.sum(axis=0)


output = vmecpp.run(vmec_input, magnetic_field=magnetic_field, verbose=False)

# The same run with the field interpolated from the mgrid file of these coils. In
# coils.solovev the wire closes 1e6 m away, which changes B_phi by 3e-6 relative and
# rmnc by 4e-4 m; with that return path in B_phi, the runs agree to 1e-7 m.
vmec_input.mgrid_file = str(TEST_DATA_DIR / "mgrid_solovev.nc")
reference = vmecpp.run(vmec_input, verbose=False)

print("analytical field:", output.wout.itfsq, "iterations")
print("mgrid file:      ", reference.wout.itfsq, "iterations")
print("max |rmnc - rmnc_mgrid| =", np.abs(output.wout.rmnc - reference.wout.rmnc).max())

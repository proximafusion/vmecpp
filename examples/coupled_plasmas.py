# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Two free-boundary plasmas in one vacuum field.

Two copies of the Solov'ev free-boundary tokamak sit on one axis, one above the
other, as the two plasma rings of merging-compression start-up do. Each has its own
coil loops, and a straight wire on the axis makes the toroidal field of both.
``vmecpp.run_coupled`` solves both plasmas in the field of the coils and of each
other's currents.

The effect of the lower plasma on the upper one is the difference between that
solution and the upper plasma solved in the field of both coil sets without the lower
plasma. The lower plasma's field opposes the vertical field that holds the upper one,
so the upper plasma expands outward, as the inverse cube of the separation; and the
upper plasma settles where the coils' radial field cancels the lower plasma's, which
with these coils lies away from the lower plasma, as the inverse fourth power.
"""

from pathlib import Path

import numpy as np
from scipy.special import ellipe, ellipk

import vmecpp

TEST_DATA_DIR = Path("src") / "vmecpp" / "cpp" / "vmecpp" / "test_data"
MU0 = 4.0e-7 * np.pi

# The coils of coils.solovev: a straight wire along the z axis carrying extcur[0],
# and circular loops (radius, height) carrying extcur[1:].
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

# The field of another plasma breaks the up-down symmetry of each one. One radial
# resolution keeps the solves short; the vertical position sits in a weak field
# gradient and needs the tight tolerance.
vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev_free_bdy.json")
zeros = np.zeros_like(np.asarray(vmec_input.rbc))
vmec_input = vmec_input.model_copy(
    update={
        "lasym": True,
        "rbs": zeros.copy(),
        "zbc": zeros.copy(),
        "raxis_s": np.zeros(1),
        "zaxis_c": np.zeros(1),
        "ns_array": np.array([16]),
        "ftol_array": np.array([1.0e-16]),
        "niter_array": np.array([20000]),
    }
)
extcur = np.asarray(vmec_input.extcur)


def coils(heights):
    """The field of the wire and of a copy of the loops around each of heights."""

    def magnetic_field(r, _phi, z):
        b_r = np.zeros_like(r)
        b_z = np.zeros_like(r)
        a = LOOPS[:, 0, None]
        for height in heights:
            dz = z - (LOOPS[:, 1, None] + height)
            alpha2 = (a - r) ** 2 + dz**2
            beta2 = (a + r) ** 2 + dz**2
            m = 1.0 - alpha2 / beta2
            scale = MU0 * extcur[1:, None] / (2 * np.pi * alpha2 * np.sqrt(beta2))
            e, k = ellipe(m), ellipk(m)
            b_r += np.sum(scale * dz / r * ((a**2 + r**2 + dz**2) * e - alpha2 * k), 0)
            b_z += np.sum(scale * ((a**2 - r**2 - dz**2) * e + alpha2 * k), 0)
        return b_r, MU0 * extcur[0] / (2 * np.pi * r), b_z

    return magnetic_field


print("separation   effect of the lower plasma on the upper one")
for separation in (40.0, 30.0, 20.0):
    heights = [separation / 2, -separation / 2]
    # the upper plasma in its own frame: its coils at 0, the lower plasma's at -D
    reference = vmecpp.run(vmec_input, coils([0.0, -separation]), verbose=False)
    upper, _lower = vmecpp.run_coupled(
        [vmec_input, vmec_input],
        [vmecpp.Placement(translation=np.array([0.0, 0.0, h])) for h in heights],
        coils(heights),
    )
    expansion = (
        np.asarray(upper.wout.rmnc)[0, -1] - np.asarray(reference.wout.rmnc)[0, -1]
    )
    shift = np.asarray(upper.wout.zmnc)[0, -1] - np.asarray(reference.wout.zmnc)[0, -1]
    print(f"{separation:6.0f} m    {expansion:+.2e} m outward, {shift:+.2e} m upward")

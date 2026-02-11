# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Comparison of VMEC++ with PARVMEC."""

from pathlib import Path

import h5py  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
import numpy as np

folder = Path("examples/data")

vmecpp_nestor = h5py.File(folder / "w7x_free_bdy_vac_nestor.out.h5", "r")
nfp = vmecpp_nestor["/wout/nfp"][()]
ns = vmecpp_nestor["/wout/ns"][()]
mnmax = vmecpp_nestor["/wout/mnmax"][()]
xm = vmecpp_nestor["/wout/xm"][()]
xn = vmecpp_nestor["/wout/xn"][()]
ref_rmnc = vmecpp_nestor["/wout/rmnc"][()]
ref_zmns = vmecpp_nestor["/wout/zmns"][()]

vmecpp_only_coils = h5py.File(folder / "w7x_free_bdy_vac_only_coils.out.h5", "r")
# assume the rest is consistent
tst_rmnc = vmecpp_only_coils["/wout/rmnc"][()]
tst_zmns = vmecpp_only_coils["/wout/zmns"][()]

# evaluate flux surface geometry from both runs and plot them
ntheta = 101
theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
for phi_degrees in [0, 18, 36]:
    phi = np.deg2rad(phi_degrees)
    kernel = np.outer(xm, theta) - np.outer(xn, phi)
    cos_kernel = np.cos(kernel)
    sin_kernel = np.sin(kernel)

    plt.figure()
    for j in [0, 2**2, 4**2, 6**2, 50]:
        ref_r = np.dot(ref_rmnc[j, :], cos_kernel)
        ref_z = np.dot(ref_zmns[j, :], sin_kernel)
        if j == 0:
            plt.plot(ref_r, ref_z, "ro", label="Nestor")
        else:
            plt.plot(ref_r, ref_z, "r-", lw=2)

        tst_r = np.dot(tst_rmnc[j, :], cos_kernel)
        tst_z = np.dot(tst_zmns[j, :], sin_kernel)
        if j == 0:
            plt.plot(tst_r, tst_z, "bx", label="only coils")
        else:
            plt.plot(tst_r, tst_z, "b--", lw=2)

    plt.axis("equal")
    plt.xlabel("R / m")
    plt.ylabel("Z / m")
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.title(f"$\\varphi = {phi_degrees}$ deg")

plt.show()

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
ref_nfp = vmecpp_nestor["/wout/nfp"][()]
ref_ns = vmecpp_nestor["/wout/ns"][()]
ref_mnmax = vmecpp_nestor["/wout/mnmax"][()]
ref_xm = vmecpp_nestor["/wout/xm"][()]
ref_xn = vmecpp_nestor["/wout/xn"][()]
ref_rmnc = vmecpp_nestor["/wout/rmnc"][()]
ref_zmns = vmecpp_nestor["/wout/zmns"][()]

vmecpp_only_coils = h5py.File(folder / "w7x_free_bdy_vac_only_coils.out.h5", "r")
tst_nfp = vmecpp_only_coils["/wout/nfp"][()]
tst_ns = vmecpp_only_coils["/wout/ns"][()]
tst_mnmax = vmecpp_only_coils["/wout/mnmax"][()]
tst_xm = vmecpp_only_coils["/wout/xm"][()]
tst_xn = vmecpp_only_coils["/wout/xn"][()]
tst_rmnc = vmecpp_only_coils["/wout/rmnc"][()]
tst_zmns = vmecpp_only_coils["/wout/zmns"][()]

# evaluate flux surface geometry from both runs and plot them
ntheta = 101
theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
for phi_degrees in [0, 18, 36]:
    phi = np.deg2rad(phi_degrees)

    ref_kernel = np.outer(ref_xm, theta) - np.outer(ref_xn, phi)
    ref_cos_kernel = np.cos(ref_kernel)
    ref_sin_kernel = np.sin(ref_kernel)

    tst_kernel = np.outer(tst_xm, theta) - np.outer(tst_xn, phi)
    tst_cos_kernel = np.cos(tst_kernel)
    tst_sin_kernel = np.sin(tst_kernel)

    plt.figure()
    for j in [0, 2**2, 4**2, 6**2, -1]:
        ref_r = np.dot(ref_rmnc[j, :], ref_cos_kernel)
        ref_z = np.dot(ref_zmns[j, :], ref_sin_kernel)
        if j == 0:
            plt.plot(ref_r, ref_z, "ro", label="Nestor")
        else:
            plt.plot(ref_r, ref_z, "r-", lw=2)

        tst_r = np.dot(tst_rmnc[j, :], tst_cos_kernel)
        tst_z = np.dot(tst_zmns[j, :], tst_sin_kernel)
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

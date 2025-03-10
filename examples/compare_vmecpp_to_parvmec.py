# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Comparison of VMEC++ with PARVMEC."""

from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np

import vmecpp

examples = Path("examples")

# reference produced using ORNL-Fusion/PARVMEC via
# `mpirun -np 4 xvmec input.w7x` in `examples/data`

# read PARVMEC reference output
with netCDF4.Dataset(examples / "data" / "wout_w7x.nc", "r") as reference_wout:
    reference_wout.set_always_mask(False)
    ref_nfp = reference_wout["nfp"][()]
    ref_ns = reference_wout["ns"][()]
    ref_mnmax = reference_wout["mnmax"][()]
    ref_xm = reference_wout["xm"][()]
    ref_xn = reference_wout["xn"][()]
    ref_rmnc = reference_wout["rmnc"][()]
    ref_zmns = reference_wout["zmns"][()]
    ref_iotas = reference_wout["iotas"][()][1:]

# read input file and run VMEC++
input_file = examples / "data" / "input.w7x"
vmecpp_input = vmecpp.VmecInput.from_file(input_file)
vmecpp_output = vmecpp.run(input=vmecpp_input)

# make sure runs are broadly consistent
assert ref_nfp == vmecpp_output.wout.nfp
assert ref_ns == vmecpp_output.wout.ns
assert ref_mnmax == vmecpp_output.wout.mnmax

# evaluate flux surface geometry from both runs and plot them
ntheta = 101
theta = np.linspace(0.0, 2.0 * np.pi, ntheta)
for phi_degrees in [0, 18, 36]:
    phi = np.deg2rad(phi_degrees)
    kernel = np.outer(ref_xm, theta) - np.outer(ref_xn, phi)
    cos_kernel = np.cos(kernel)
    sin_kernel = np.sin(kernel)

    plt.figure()
    for j in [0, 2**2, 4**2, 6**2, 8**2, 98]:
        ref_r = np.dot(ref_rmnc[j, :], cos_kernel)
        ref_z = np.dot(ref_zmns[j, :], sin_kernel)
        if j == 0:
            plt.plot(ref_r, ref_z, "ro", label="PARVMEC")
        else:
            plt.plot(ref_r, ref_z, "r-", lw=2)

        vmecpp_r = np.dot(vmecpp_output.wout.rmnc[:, j], cos_kernel)
        vmecpp_z = np.dot(vmecpp_output.wout.zmns[:, j], sin_kernel)
        if j == 0:
            plt.plot(vmecpp_r, vmecpp_z, "bx", label="VMEC++")
        else:
            plt.plot(vmecpp_r, vmecpp_z, "b--", lw=2)

    plt.axis("equal")
    plt.xlabel("R / m")
    plt.ylabel("Z / m")
    plt.grid(True)
    plt.legend(loc="upper right")

    plt.title(f"$\\varphi = {phi_degrees}°$")

# plot iota profile comparison
s_half = (0.5 + np.arange(ref_ns - 1)) / (ref_ns - 1.0)
plt.figure()
plt.plot(s_half, ref_iotas, "ro-", label="PARVMEC")
plt.plot(s_half, vmecpp_output.wout.iotas[1:], "bx--", label="VMEC++")
plt.xlabel("s / 1")
plt.ylabel("rotational transform / 1")
plt.grid(True)
plt.legend(loc="upper left")

plt.show()

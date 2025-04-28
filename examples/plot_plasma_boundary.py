# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Showcases how to plot the geometry of the plasma boundary computed using VMEC++.

Note, that this script requires matplotlib as an additional dependency.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import vmecpp


def plot_plasma_boundary():
    # We need a VmecInput, a Python object that corresponds
    # to the classic "input.*" files.
    # We can construct it from such a classic VMEC input file
    # (Fortran namelist called INDATA):
    input_file = Path(__file__).parent / "data" / "input.w7x"
    vmec_input = vmecpp.VmecInput.from_file(input_file)

    # Now we can run VMEC++:
    vmec_output = vmecpp.run(vmec_input)

    # The output object contains the Fourier coefficients of the geometry in R and Z
    # as a function of the poloidal (theta) and toroidal (phi) angle-like coordinates
    # for a number of discrete radial locations.

    # number of flux surfaces, i.e., final radial resolution
    ns = vmec_output.wout.ns

    # poloidal mode numbers: m
    xm = vmec_output.wout.xm

    # toroidal mode numbers: n * nfp
    xn = vmec_output.wout.xn

    # stellarator-symmetric Fourier coefficients of flux surface geometry R ~ cos(m * theta - n * nfp * phi)
    rmnc = vmec_output.wout.rmnc

    # stellarator-symmetric Fourier coefficients of flux surface geometry Z ~ sin(m * theta - n * nfp * phi)
    zmns = vmec_output.wout.zmns

    # plot the outermost (last) flux surface, which is the plasma boundary
    j = ns - 1

    # resolution over the flux surface
    num_theta = 101
    num_phi = 181

    # grid in theta and phi along the flux surface
    grid_theta = np.linspace(0.0, 2.0 * np.pi, num_theta, endpoint=True)
    grid_phi = np.linspace(0.0, 2.0 * np.pi, num_phi, endpoint=True)

    # compute Cartesian coordinates of flux surface geometry
    x = np.zeros([num_theta, num_phi])
    y = np.zeros([num_theta, num_phi])
    z = np.zeros([num_theta, num_phi])
    for idx_theta, theta in enumerate(grid_theta):
        for idx_phi, phi in enumerate(grid_phi):
            kernel = xm * theta - xn * phi
            r = np.dot(rmnc[:, j], np.cos(kernel))
            x[idx_theta, idx_phi] = r * np.cos(phi)
            y[idx_theta, idx_phi] = r * np.sin(phi)
            z[idx_theta, idx_phi] = np.dot(zmns[:, j], np.sin(kernel))

    # actually make the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Plot the surface
    ax.plot_surface(x, y, z)

    # Set an equal aspect ratio
    ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    plot_plasma_boundary()

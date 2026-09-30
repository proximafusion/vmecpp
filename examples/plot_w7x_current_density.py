# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Plot the magnitude of W7-X current density in four toroidal cross sections.

Requires matplotlib. Run with     MPLBACKEND=Agg python
examples/plot_w7x_current_density.py to save the figure without opening a window.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri

import vmecpp


def cross_section(wout, phi, theta):
    """Return R, Z and |J| on interior full-grid surfaces and at the axis."""
    if wout.lasym:
        msg = "This example expects a stellarator-symmetric W7-X equilibrium."
        raise ValueError(msg)

    # curr* is sqrt(g) J^angle on the full grid, but gmnc is on the half grid.
    # Evaluate both at interior full-grid nodes, never using the padded gmnc[:, 0]
    # or the extrapolated current coefficients on the magnetic axis.
    radial = np.arange(1, wout.ns - 1)
    angle = wout.xm[:, None] * theta - wout.xn[:, None] * phi
    angle_nyq = wout.xm_nyq[:, None] * theta - wout.xn_nyq[:, None] * phi
    cosine = np.cos(angle)
    sine = np.sin(angle)
    cosine_nyq = np.cos(angle_nyq)

    r = wout.rmnc[:, radial].T @ cosine
    z = wout.zmns[:, radial].T @ sine
    drdu = wout.rmnc[:, radial].T @ (-wout.xm[:, None] * sine)
    drdv = wout.rmnc[:, radial].T @ (wout.xn[:, None] * sine)
    dzdu = wout.zmns[:, radial].T @ (wout.xm[:, None] * cosine)
    dzdv = wout.zmns[:, radial].T @ (-wout.xn[:, None] * cosine)

    sqrtg_coeff = (wout.gmnc[:, radial] + wout.gmnc[:, radial + 1]) / 2
    sqrtg = sqrtg_coeff.T @ cosine_nyq
    curru = (wout.currumnc[:, radial].T @ cosine_nyq) / sqrtg
    currv = (wout.currvmnc[:, radial].T @ cosine_nyq) / sqrtg
    jr = curru * drdu + currv * drdv
    jphi = currv * r
    jz = curru * dzdu + currv * dzdv
    magnitude = np.sqrt(jr**2 + jphi**2 + jz**2) / 1e3  # kA/m^2

    # The axis is a single point, not a flux surface: its Fourier coefficients
    # for J and sqrt(g) are extrapolated/padded and cannot be divided pointwise.
    # Extrapolate the theta-averaged physical magnitude from the first two
    # resolved surfaces in s, then connect that point to the innermost ring.
    axis_r = np.dot(wout.rmnc[:, 0], cosine[:, 0])
    axis_z = np.dot(wout.zmns[:, 0], sine[:, 0])
    axis_j = max(0.0, 2 * magnitude[0].mean() - magnitude[1].mean())
    return r, z, magnitude, (axis_r, axis_z, axis_j)


def plot_current_density(output_path=None):
    vmec_input = vmecpp.VmecInput.from_file(
        Path(__file__).parent / "data" / "input.w7x"
    )
    vmec_input.lbsubs = True
    wout = vmecpp.run(vmec_input).wout

    theta = np.linspace(0, 2 * np.pi, 129, endpoint=False)
    sections = [
        cross_section(wout, phi, theta) for phi in np.arange(4) * np.pi / (2 * wout.nfp)
    ]
    vmax = max(np.max(magnitude) for _, _, magnitude, _ in sections)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    for index, (ax, (r, z, magnitude, axis)) in enumerate(
        zip(axes.flat, sections, strict=True)
    ):
        nrad, ntheta = r.shape
        points_r = np.r_[axis[0], r.ravel()]
        points_z = np.r_[axis[1], z.ravel()]
        values = np.r_[axis[2], magnitude.ravel()]
        triangles = []
        for t in range(ntheta):
            next_t = (t + 1) % ntheta
            triangles.append((0, 1 + t, 1 + next_t))
        for s in range(nrad - 1):
            for t in range(ntheta):
                a = 1 + s * ntheta + t
                b = 1 + s * ntheta + (t + 1) % ntheta
                c = a + ntheta
                d = b + ntheta
                triangles.extend(((a, b, c), (b, d, c)))
        mesh = tri.Triangulation(points_r, points_z, triangles)
        image = ax.tripcolor(mesh, values, shading="gouraud", vmin=0, vmax=vmax)
        ax.set_aspect("equal")
        ax.set_title(f"phi = {index} pi / (2 nfp)")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")

    fig.colorbar(image, ax=axes, label="|J| [kA/m^2]", shrink=0.8)
    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot_current_density(Path(__file__).parent / "w7x_current_density.png")

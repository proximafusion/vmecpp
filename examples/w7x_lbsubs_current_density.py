# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""W7-X toroidal current density with lbsubs = False and True.

The equilibrium is the W7-X standard configuration with zero net toroidal
current and p ~ (1 - s)(1 - s^4). Ampere's law in VMEC coordinates gives

    sqrt(g) J^v = (dB_u/ds - dB_s/du) / mu_0.

B_u is on the half grid and the wout bsubsmns is the full-grid B_s:
- lbsubs = False: the metric B_s averaged onto the full grid;
- lbsubs = True: the B_s that solves the radial force balance.
The cross sections show isocontours of J_phi = R J^v.

Use lbsubs=True for currents and quantities derived from them. The exported
currents use the same radial derivative as the B_s force-balance solve.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, tri

import vmecpp

N_THETA_PLOT = 256
N_LEVELS = 20

TITLES = {
    False: "lbsubs = False: B_s from the metric",
    True: "lbsubs = True: B_s from radial force balance",
}


def run(lbsubs):
    vmec_input = vmecpp.VmecInput.from_file(
        Path(__file__).parent / "data" / "input.w7x"
    )
    vmec_input.am = np.array([6.5e4, -6.5e4, 0.0, 0.0, -6.5e4, 6.5e4])
    vmec_input.ftol_array[-1] = 1e-15
    vmec_input.lbsubs = lbsubs
    return vmecpp.run(vmec_input, max_threads=4)


def toroidal_current_density(wout, zeta):
    """R, Z and J_phi in kA/m^2 on the interior full-grid surfaces at angle zeta.

    The exported coefficients are sqrt(g) J^v and include the factor 1/mu_0.
    """
    radial = np.arange(1, wout.ns - 1)
    currvmnc = wout.currvmnc[:, radial]

    theta = np.linspace(0, 2 * np.pi, N_THETA_PLOT, endpoint=False)
    cos = np.cos(wout.xm[:, None] * theta - wout.xn[:, None] * zeta)
    sin = np.sin(wout.xm[:, None] * theta - wout.xn[:, None] * zeta)
    cos_nyq = np.cos(wout.xm_nyq[:, None] * theta - wout.xn_nyq[:, None] * zeta)
    r = wout.rmnc[:, radial].T @ cos
    z = wout.zmns[:, radial].T @ sin
    sqrtg = (0.5 * (wout.gmnc[:, radial] + wout.gmnc[:, radial + 1])).T @ cos_nyq
    return r, z, r * (currvmnc.T @ cos_nyq) / sqrtg / 1e3


def triangulation(r, z):
    nrad, ntheta = r.shape
    t = np.arange(ntheta)
    a = (np.arange(nrad - 1)[:, None] * ntheta + t).ravel()
    b = (np.arange(nrad - 1)[:, None] * ntheta + (t + 1) % ntheta).ravel()
    triangles = np.r_[np.c_[a, b, a + ntheta], np.c_[b, b + ntheta, a + ntheta]]
    return tri.Triangulation(r.ravel(), z.ravel(), triangles)


def plot(wout, title):
    """Isocontours of J_phi in four toroidal planes of a field period.

    The levels span the 98th percentile of |J_phi|.
    """
    zetas = np.arange(4) * np.pi / (3 * wout.nfp)
    sections = [toroidal_current_density(wout, zeta) for zeta in zetas]
    jphi_all = np.concatenate([jphi.ravel() for _, _, jphi in sections])
    limit = np.percentile(np.abs(jphi_all), 98)
    norm = colors.Normalize(-limit, limit)
    levels = np.linspace(-limit, limit, 2 * N_LEVELS + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for index, (ax, (r, z, jphi)) in enumerate(zip(axes.flat, sections, strict=True)):
        ax.tricontour(
            triangulation(r, z),
            jphi.ravel(),
            levels=levels,
            cmap="RdBu_r",
            norm=norm,
            linewidths=0.7,
        )
        ax.plot(r[-1], z[-1], color="0.5", lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"zeta = {index} pi / (3 nfp)", fontsize=9)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
        ax=axes,
        label="J_phi = R J^v [kA/m^2]",
        shrink=0.8,
        extend="both",
    )
    fig.suptitle(title)


def main():
    for lbsubs in (False, True):
        plot(run(lbsubs).wout, TITLES[lbsubs])
    plt.show()


if __name__ == "__main__":
    main()

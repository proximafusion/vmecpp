# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Plot signed W7-X current densities in four toroidal cross sections.

Two figures are produced with a diverging red-blue colorscale:
- the toroidal current density J_phi, i.e. the component into (red) and out of
  (blue) the R-Z image plane, since phi points into the page for (R, phi, Z);
- the parallel current density J.B/|B|, i.e. the current along (red) and
  against (blue) the magnetic field lines.

A third figure shows the radial profiles of iota and of the flux-surface
averaged currents, with black lines where iota crosses a rational n/m, m <= 12.

Requires matplotlib. Run with     MPLBACKEND=Agg python
examples/plot_w7x_current_density.py to save the figures without opening a window.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri

import vmecpp


def cross_section(wout, phi, theta):
    """Return R, Z, J_phi and J_parallel on interior full-grid surfaces and the axis."""
    if wout.lasym:
        msg = "This example expects a stellarator-symmetric W7-X equilibrium."
        raise ValueError(msg)

    # curr* is sqrt(g) J^angle on the full grid, but gmnc, bsub*mnc and bmnc are
    # on the half grid. Evaluate all of them at interior full-grid nodes, never
    # using the padded half-grid index 0 or the extrapolated currents on the axis.
    radial = np.arange(1, wout.ns - 1)
    angle = wout.xm[:, None] * theta - wout.xn[:, None] * phi
    angle_nyq = wout.xm_nyq[:, None] * theta - wout.xn_nyq[:, None] * phi
    cosine = np.cos(angle)
    sine = np.sin(angle)
    cosine_nyq = np.cos(angle_nyq)

    def full_grid(half_coeff):
        return ((half_coeff[:, radial] + half_coeff[:, radial + 1]) / 2).T @ cosine_nyq

    r = wout.rmnc[:, radial].T @ cosine
    z = wout.zmns[:, radial].T @ sine

    sqrtg = full_grid(wout.gmnc)
    curru = (wout.currumnc[:, radial].T @ cosine_nyq) / sqrtg
    currv = (wout.currvmnc[:, radial].T @ cosine_nyq) / sqrtg
    # Only e_v = dX/dv has a component along phi_hat, equal to R.
    jphi = currv * r / 1e3  # kA/m^2
    jpar = (
        (curru * full_grid(wout.bsubumnc) + currv * full_grid(wout.bsubvmnc))
        / full_grid(wout.bmnc)
        / 1e3
    )  # kA/m^2

    # The axis is a single point, not a flux surface: its Fourier coefficients
    # for J and sqrt(g) are extrapolated/padded and cannot be divided pointwise.
    # Extrapolate the theta-averaged values from the first two resolved surfaces
    # in s, then connect that point to the innermost ring.
    axis_r = np.dot(wout.rmnc[:, 0], cosine[:, 0])
    axis_z = np.dot(wout.zmns[:, 0], sine[:, 0])
    axis_jphi = 2 * jphi[0].mean() - jphi[1].mean()
    axis_jpar = 2 * jpar[0].mean() - jpar[1].mean()
    return (
        r,
        z,
        {"jphi": (jphi, axis_jphi), "jpar": (jpar, axis_jpar)},
        (axis_r, axis_z),
    )


def triangulation(r, z, axis):
    nrad, ntheta = r.shape
    triangles = [(0, 1 + t, 1 + (t + 1) % ntheta) for t in range(ntheta)]
    for s in range(nrad - 1):
        for t in range(ntheta):
            a = 1 + s * ntheta + t
            b = 1 + s * ntheta + (t + 1) % ntheta
            triangles.extend(((a, b, a + ntheta), (b, b + ntheta, a + ntheta)))
    return tri.Triangulation(
        np.r_[axis[0], r.ravel()], np.r_[axis[1], z.ravel()], triangles
    )


def plot_signed(sections, key, label, output_path):
    vmax = max(np.max(np.abs(fields[key][0])) for _, _, fields, _ in sections)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    images = []
    for index, (ax, (r, z, fields, axis)) in enumerate(
        zip(axes.flat, sections, strict=True)
    ):
        values, axis_value = fields[key]
        images.append(
            ax.tripcolor(
                triangulation(r, z, axis),
                np.r_[axis_value, values.ravel()],
                shading="gouraud",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
        )
        ax.set_aspect("equal")
        ax.set_title(f"phi = {index} pi / (2 nfp)")
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")

    fig.colorbar(images[-1], ax=axes, label=label, shrink=0.8)
    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=110)
    plt.close(fig)


def rational_surfaces(s, iota, max_m=12):
    """Return (s, n, m) where iota(s) crosses n/m, m <= max_m, n/m in lowest
    terms."""
    crossings = []
    for m in range(1, max_m + 1):
        for n in range(int(np.ceil(iota.min() * m)), int(np.floor(iota.max() * m)) + 1):
            if np.gcd(n, m) != 1:
                continue
            delta = iota - n / m
            for i in np.nonzero(np.sign(delta[:-1]) != np.sign(delta[1:]))[0]:
                weight = delta[i] / (delta[i] - delta[i + 1])
                crossings.append((s[i] + weight * (s[i + 1] - s[i]), n, m))
    return crossings


def plot_profiles(wout, output_path):
    s = np.linspace(0, 1, wout.ns)
    # Drop the extrapolated axis and boundary values of the current profiles.
    interior = slice(1, wout.ns - 1)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    axes[0].plot(s, wout.iotaf, color="tab:green")
    axes[0].set_ylabel("iota")
    axes[1].plot(s[interior], wout.jcuru[interior] / 1e3, label="jcuru")
    axes[1].plot(s[interior], wout.jcurv[interior] / 1e3, label="jcurv")
    axes[1].set_ylabel("<J^u>, <J^v> [kA/m^2]")
    axes[1].legend()
    axes[2].plot(s[interior], wout.jdotb[interior], color="tab:red")
    axes[2].set_ylabel("<J.B> [T A/m^2]")
    axes[2].set_xlabel("s")

    for s_rational, n, m in rational_surfaces(s, wout.iotaf):
        for ax in axes:
            ax.axvline(s_rational, color="black", linewidth=0.8)
        axes[0].annotate(
            f"{n}/{m}",
            (s_rational, 1),
            xycoords=("data", "axes fraction"),
            xytext=(2, -12),
            textcoords="offset points",
            fontsize=8,
        )
    for ax in axes[1:]:
        ax.axhline(0, color="gray", linewidth=0.5)

    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=110)
    plt.close(fig)


def plot_current_density(output_dir=None):
    vmec_input = vmecpp.VmecInput.from_file(
        Path(__file__).parent / "data" / "input.w7x"
    )
    vmec_input.lbsubs = True
    vmec_input.ntheta = 96
    vmec_input.nzeta = 96
    wout = vmecpp.run(vmec_input).wout

    theta = np.linspace(0, 2 * np.pi, 129, endpoint=False)
    sections = [
        cross_section(wout, phi, theta) for phi in np.arange(4) * np.pi / (2 * wout.nfp)
    ]
    figures = {
        "jphi": (
            "J_phi [kA/m^2] (red: into, blue: out of the plane)",
            "w7x_toroidal_current_density.png",
        ),
        "jpar": (
            "J.B/|B| [kA/m^2] (red: along, blue: against B)",
            "w7x_parallel_current_density.png",
        ),
    }
    for key, (label, filename) in figures.items():
        path = None if output_dir is None else Path(output_dir) / filename
        plot_signed(sections, key, label, path)
    plot_profiles(
        wout,
        None if output_dir is None else Path(output_dir) / "w7x_current_profiles.png",
    )


if __name__ == "__main__":
    plot_current_density(Path(__file__).parent)

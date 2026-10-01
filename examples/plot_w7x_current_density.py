# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Compare W7-X current densities computed with lbsubs = False and True.

The equilibrium is run on a 96 x 96 (theta, zeta) real-space grid with both
lbsubs settings. lbsubs only changes the full-grid B_s used by the jxbout
diagnostic, so the current density is taken from jxbout:
J^u = jsupu3 / sqrtg3, J^v = jsupv3 / sqrtg3, J.B = jdotb_sqrtg / sqrtg3. The
wout currumnc/currvmnc/jdotb do not depend on lbsubs.

Three figures are produced:
- the toroidal current density J_phi = R J^v in four toroidal cross sections,
  red into and blue out of the R-Z image plane (phi points into the page);
- the parallel current density J.B/|B| in the same cross sections, red along
  and blue against the magnetic field;
- radial profiles of iota and of the surface-averaged currents, with black
  lines where iota crosses a rational n/m, m <= 12.
The cross-section figures have one row each for lbsubs = False, True and
True - False.

Requires matplotlib. Run with     MPLBACKEND=Agg python
examples/plot_w7x_current_density.py to save the figures without opening a window.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import tri

import vmecpp

N_THETA = 96
N_ZETA = 96


def cross_section(wout, jxbout, k):
    """Return R, Z, J_phi and J_parallel at toroidal grid index k.

    Only interior full-grid surfaces are returned: jxbout is zero on the axis and
    at the boundary.
    """
    if wout.lasym:
        msg = "This example expects a stellarator-symmetric W7-X equilibrium."
        raise ValueError(msg)

    ntheta3 = N_THETA // 2 + 1
    radial = np.arange(1, wout.ns - 1)
    # jxbout covers theta in [0, pi] with index k * ntheta3 + l. The quantities
    # used here are stellarator-even, f(2 pi - theta, -zeta) = f(theta, zeta),
    # which gives theta in (pi, 2 pi).
    l_full = np.arange(N_THETA)
    reflected = l_full >= ntheta3
    l_source = np.where(reflected, N_THETA - l_full, l_full)
    k_source = np.where(reflected, (N_ZETA - k) % N_ZETA, k)
    index = k_source * ntheta3 + l_source

    def real_space(field):
        return field[radial][:, index]

    sqrtg = real_space(jxbout.sqrtg3)
    jsupv = real_space(jxbout.jsupv3) / sqrtg
    jdotb = real_space(jxbout.jdotb_sqrtg) / sqrtg

    theta = l_full * 2 * np.pi / N_THETA
    phi = k * 2 * np.pi / (wout.nfp * N_ZETA)
    angle = wout.xm[:, None] * theta - wout.xn[:, None] * phi
    angle_nyq = wout.xm_nyq[:, None] * theta - wout.xn_nyq[:, None] * phi
    r = wout.rmnc[:, radial].T @ np.cos(angle)
    z = wout.zmns[:, radial].T @ np.sin(angle)
    # bmnc is on the half grid; average onto the full grid.
    modb = ((wout.bmnc[:, radial] + wout.bmnc[:, radial + 1]) / 2).T @ np.cos(angle_nyq)

    jphi = jsupv * r / 1e3  # kA/m^2
    jpar = jdotb / modb / 1e3  # kA/m^2

    # The axis is a single point; extrapolate the theta averages of the two
    # innermost surfaces in s.
    axis = (
        np.dot(wout.rmnc[:, 0], np.cos(angle[:, 0])),
        np.dot(wout.zmns[:, 0], np.sin(angle[:, 0])),
    )
    fields = {
        key: (values, 2 * values[0].mean() - values[1].mean())
        for key, values in (("jphi", jphi), ("jpar", jpar))
    }
    return r, z, fields, axis


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


def save(fig, output_path):
    if output_path is None:
        plt.show()
    else:
        fig.savefig(output_path, dpi=90)
    plt.close(fig)


def plot_signed(runs, key, label, output_path):
    """Plot `key` for lbsubs = False, True and their difference, one row each."""
    off = [fields[key] for _, _, fields, _ in runs[False]]
    on = [fields[key] for _, _, fields, _ in runs[True]]
    diff = [(a[0] - b[0], a[1] - b[1]) for a, b in zip(on, off, strict=True)]
    rows = [("lbsubs = False", off), ("lbsubs = True", on), ("True - False", diff)]
    vmax = max(np.max(np.abs(values)) for values, _ in off + on)
    vmax_diff = max(np.max(np.abs(values)) for values, _ in diff)

    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    images = []
    for row_index, (row_label, row) in enumerate(rows):
        limit = vmax_diff if row_index == 2 else vmax
        for index, (ax, (r, z, _, axis), (values, axis_value)) in enumerate(
            zip(axes[row_index], runs[True], row, strict=True)
        ):
            images.append(
                ax.tripcolor(
                    triangulation(r, z, axis),
                    np.r_[axis_value, values.ravel()],
                    shading="gouraud",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
            )
            ax.set_aspect("equal")
            ax.set_title(f"{row_label}, phi = {index} pi / (2 nfp)", fontsize=9)
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")

    fig.colorbar(images[4], ax=axes[:2], label=label, shrink=0.8)
    fig.colorbar(images[8], ax=axes[2], label="difference [kA/m^2]", shrink=0.8)
    save(fig, output_path)


def rational_surfaces(s, iota, max_m=12):
    """Return (s, n, m) where iota(s) crosses n/m, m <= max_m, n/m in lowest terms."""
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


def plot_profiles(outputs, output_path):
    """Plot profiles for lbsubs = True (solid) and False (dashed)."""
    wout = outputs[True].wout
    s = np.linspace(0, 1, wout.ns)
    # Drop the axis and boundary values, which are extrapolated or zero.
    interior = slice(1, wout.ns - 1)
    fig, axes = plt.subplots(
        4, 1, figsize=(8, 11), sharex=True, constrained_layout=True
    )
    axes[0].plot(s, wout.iotaf, color="tab:green")
    axes[0].set_ylabel("iota")
    axes[1].plot(s[interior], wout.jcuru[interior] / 1e3, label="jcuru")
    axes[1].plot(s[interior], wout.jcurv[interior] / 1e3, label="jcurv")
    axes[1].set_ylabel("wout <J^u>, <J^v> [kA/m^2]")
    axes[1].legend(fontsize=8, title="independent of lbsubs", title_fontsize=8)
    for lbsubs, style in ((False, "--"), (True, "-")):
        jxbout = outputs[lbsubs].jxbout
        axes[2].plot(
            s[interior],
            jxbout.jdotb[interior],
            style,
            color="tab:red",
            label=f"lbsubs = {lbsubs}",
        )
        axes[3].plot(
            s[interior],
            np.sqrt(jxbout.jpar2[interior]) / 1e3,
            style,
            color="tab:purple",
            label=f"lbsubs = {lbsubs}",
        )
    axes[2].set_ylabel("jxbout <J.B> [T A/m^2]")
    axes[3].set_ylabel("jxbout <J_par^2>^(1/2) [kA/m^2]")
    axes[3].set_xlabel("s")
    for ax in axes[2:]:
        ax.legend(fontsize=8)

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
    for ax in axes[1:3]:
        ax.axhline(0, color="gray", linewidth=0.5)
    save(fig, output_path)


def plot_current_density(output_dir=None):
    vmec_input = vmecpp.VmecInput.from_file(
        Path(__file__).parent / "data" / "input.w7x"
    )
    vmec_input.ntheta = N_THETA
    vmec_input.nzeta = N_ZETA
    outputs = {}
    for lbsubs in (False, True):
        vmec_input.lbsubs = lbsubs
        outputs[lbsubs] = vmecpp.run(vmec_input)

    runs = {
        lbsubs: [
            cross_section(output.wout, output.jxbout, k)
            for k in np.arange(4) * N_ZETA // 4
        ]
        for lbsubs, output in outputs.items()
    }
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
        plot_signed(runs, key, label, path)
    plot_profiles(
        outputs,
        None if output_dir is None else Path(output_dir) / "w7x_current_profiles.png",
    )


if __name__ == "__main__":
    plot_current_density(Path(__file__).parent)

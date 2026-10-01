# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""W7-X toroidal and parallel current density with lbsubs = False and True.

The equilibrium is the W7-X standard configuration at <beta> = 1% with zero net
toroidal current. Ampere's law in VMEC coordinates gives

    sqrt(g) J^u = (dB_s/dv - dB_v/ds) / mu_0,
    sqrt(g) J^v = (dB_u/ds - dB_s/du) / mu_0.

B_u and B_v are on the half grid and the wout bsubsmns is the full-grid B_s:
- lbsubs = False: the metric B_s averaged onto the full grid;
- lbsubs = True: the B_s that solves the radial force balance.
The currents use bsubsmns at the full-grid surface and plain radial differences
of B_u and B_v. These are the differences the force balance for B_s uses, and
with lbsubs = True the result equals the jxbout currents on the real-space grid.

The cross sections show isocontours of J_phi and J.B/|B|. The jxbout figures
draw the rational surfaces iota = p/q over theta in [0, pi], so that currents
peaking on a single surface can be compared with them.

Requires matplotlib. Run with
    MPLBACKEND=Agg python examples/w7x_lbsubs_current_density.py
to save the figures without opening a window.
"""

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, tri

import vmecpp

MU_0 = 4.0e-7 * np.pi
N_THETA_PLOT = 256
N_LEVELS = 20


def run(lbsubs):
    vmec_input = vmecpp.VmecInput.from_file(
        Path(__file__).parent / "data" / "input.w7x_beta1"
    )
    vmec_input.lbsubs = lbsubs
    return vmecpp.run(vmec_input)


def jxbout_grid(output):
    """Ntheta3, ntheta, nzeta of the jxbout real-space grid, theta in [0, pi]."""
    nzeta = output.input.nzeta
    ntheta3 = output.jxbout.bsubs3.shape[1] // nzeta
    return ntheta3, 2 * (ntheta3 - 1), nzeta


def currents_from_bsubs_full(wout, bsubsmns_full):
    """Sqrt(g) J^u, sqrt(g) J^v on the interior full grid, (mnmax_nyq, ns).

    B_s is truncated to m < mpol, |n| <= ntor, the resolution of B_u and B_v in the
    wout.
    """
    ds = 1.0 / (wout.ns - 1)
    xm, xn = wout.xm_nyq[:, None], wout.xn_nyq[:, None]
    resolved = (xm < wout.mpol) & (np.abs(xn) <= wout.ntor * wout.nfp)
    bsubsmns_full = np.where(resolved, bsubsmns_full, 0.0)
    dbsubu = np.diff(wout.bsubumnc[:, 1:], axis=1) / ds
    dbsubv = np.diff(wout.bsubvmnc[:, 1:], axis=1) / ds
    currumnc = np.zeros_like(bsubsmns_full)
    currvmnc = np.zeros_like(bsubsmns_full)
    currumnc[:, 1:-1] = (-xn * bsubsmns_full[:, 1:-1] - dbsubv) / MU_0
    currvmnc[:, 1:-1] = (-xm * bsubsmns_full[:, 1:-1] + dbsubu) / MU_0
    return currumnc, currvmnc


def cross_section(wout, currumnc, currvmnc, zeta):
    """R, Z, J_phi and J_parallel in kA/m^2 on the interior full-grid surfaces."""
    theta = np.linspace(0, 2 * np.pi, N_THETA_PLOT, endpoint=False)
    radial = np.arange(1, wout.ns - 1)
    cos = np.cos(wout.xm[:, None] * theta - wout.xn[:, None] * zeta)
    sin = np.sin(wout.xm[:, None] * theta - wout.xn[:, None] * zeta)
    cos_nyq = np.cos(wout.xm_nyq[:, None] * theta - wout.xn_nyq[:, None] * zeta)

    def full_grid(half):
        return (0.5 * (half[:, radial] + half[:, radial + 1])).T @ cos_nyq

    r = wout.rmnc[:, radial].T @ cos
    z = wout.zmns[:, radial].T @ sin
    sqrtg = full_grid(wout.gmnc)
    jsupu = (currumnc[:, radial].T @ cos_nyq) / sqrtg
    jsupv = (currvmnc[:, radial].T @ cos_nyq) / sqrtg
    jpar = (
        jsupu * full_grid(wout.bsubumnc) + jsupv * full_grid(wout.bsubvmnc)
    ) / full_grid(wout.bmnc)
    return r, z, {"jphi": r * jsupv / 1e3, "jpar": jpar / 1e3}


def jxbout_cross_section(output, k):
    """R, Z, J_phi and J_parallel in kA/m^2 at jxbout toroidal index k."""
    wout, jxbout = output.wout, output.jxbout
    ntheta3, ntheta, nzeta = jxbout_grid(output)
    radial = np.arange(1, wout.ns - 1)
    # J^u, J^v, J.B and |B| are stellarator-even: f(2 pi - theta, -zeta) = f(theta, zeta).
    poloidal = np.arange(ntheta)
    reflected = poloidal >= ntheta3
    index = np.where(reflected, (nzeta - k) % nzeta, k) * ntheta3 + np.where(
        reflected, ntheta - poloidal, poloidal
    )
    theta = 2 * np.pi * poloidal / ntheta
    zeta = 2 * np.pi * k / (nzeta * wout.nfp)
    angle = wout.xm[:, None] * theta - wout.xn[:, None] * zeta
    r = wout.rmnc[:, radial].T @ np.cos(angle)
    z = wout.zmns[:, radial].T @ np.sin(angle)
    sqrtg = jxbout.sqrtg3[radial][:, index]
    jsupv = jxbout.jsupv3[radial][:, index] / sqrtg
    angle_nyq = wout.xm_nyq[:, None] * theta - wout.xn_nyq[:, None] * zeta
    modb = (0.5 * (wout.bmnc[:, radial] + wout.bmnc[:, radial + 1])).T @ np.cos(
        angle_nyq
    )
    jpar = jxbout.jdotb_sqrtg[radial][:, index] / sqrtg / modb
    return r, z, {"jphi": r * jsupv / 1e3, "jpar": jpar / 1e3}


def triangulation(r, z):
    nrad, ntheta = r.shape
    t = np.arange(ntheta)
    a = (np.arange(nrad - 1)[:, None] * ntheta + t).ravel()
    b = (np.arange(nrad - 1)[:, None] * ntheta + (t + 1) % ntheta).ravel()
    triangles = np.r_[np.c_[a, b, a + ntheta], np.c_[b, b + ntheta, a + ntheta]]
    return tri.Triangulation(r.ravel(), z.ravel(), triangles)


def rational_surfaces(wout, max_q=12, max_q_resonant=24):
    """(iota = p/q, s) in the iota range: all p/q with q <= max_q, and p/q with p a
    multiple of nfp (resonant with the field periods) and q <= max_q_resonant."""
    s = np.linspace(0, 1, wout.ns)
    iota = wout.iotaf[1:]
    rationals = {
        Fraction(p, q)
        for q in range(1, max_q_resonant + 1)
        for p in range(int(iota.min() * q), int(iota.max() * q) + 2)
        if q <= max_q or p % wout.nfp == 0
    }
    out = []
    for value in sorted(rationals):
        delta = iota - float(value)
        for i in np.nonzero(np.sign(delta[:-1]) != np.sign(delta[1:]))[0]:
            out.append(
                (
                    value,
                    s[1 + i]
                    + delta[i] / (delta[i] - delta[i + 1]) * (s[2 + i] - s[1 + i]),
                )
            )
    return out


def flux_surface(wout, s_value, zeta, theta_max=2 * np.pi):
    theta = np.linspace(0, theta_max, N_THETA_PLOT + 1)
    s = np.linspace(0, 1, wout.ns)
    rmnc = np.array([np.interp(s_value, s, row) for row in wout.rmnc])
    zmns = np.array([np.interp(s_value, s, row) for row in wout.zmns])
    angle = wout.xm[:, None] * theta - wout.xn[:, None] * zeta
    return rmnc @ np.cos(angle), zmns @ np.sin(angle)


def plot_rows(rows, key, label, output_path, wout=None):
    """Isocontours in one row of four toroidal planes per (row label, sections).

    The levels span the 98th percentile of |value| per row. Rational surfaces are drawn
    for theta in [0, pi] only, leaving the other half of each section uncovered.
    """
    fig, axes = plt.subplots(
        len(rows),
        4,
        figsize=(17, 4.4 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    surfaces = [] if wout is None else rational_surfaces(wout)
    nfp = 1 if wout is None else wout.nfp
    for row_axes, (row_label, sections) in zip(axes, rows, strict=True):
        values = np.concatenate([fields[key].ravel() for _, _, fields, _ in sections])
        limit = np.percentile(np.abs(values), 98)
        norm = colors.Normalize(-limit, limit)
        levels = np.linspace(-limit, limit, 2 * N_LEVELS + 1)
        for index, (ax, (r, z, fields, zeta)) in enumerate(
            zip(row_axes, sections, strict=True)
        ):
            ax.tricontour(
                triangulation(r, z),
                fields[key].ravel(),
                levels=levels,
                cmap="RdBu_r",
                norm=norm,
                linewidths=0.7,
            )
            ax.plot(r[-1], z[-1], color="0.5", lw=0.5)
            for value, s_value in surfaces:
                resonant = value.numerator % nfp == 0
                ax.plot(
                    *flux_surface(wout, s_value, zeta, theta_max=np.pi),
                    color="tab:green",
                    lw=0.9 if resonant else 0.5,
                    ls="-" if resonant else "--",
                )
            ax.set_aspect("equal")
            ax.set_title(f"{row_label}, zeta = {index} pi / (3 nfp)", fontsize=9)
            ax.set_xlabel("R [m]")
            ax.set_ylabel("Z [m]")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
            ax=row_axes,
            label=label,
            shrink=0.85,
            extend="both",
        )
    if surfaces:
        text = ", ".join(f"{value} (s = {s_value:.2f})" for value, s_value in surfaces)
        fig.suptitle(
            f"iota = p/q surfaces for theta in [0, pi]: {text}; "
            "solid: p a multiple of nfp",
            fontsize=9,
        )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_profiles(output, output_path):
    """Iota and per-surface jxbout currents with the rational surfaces."""
    wout, jxbout = output.wout, output.jxbout
    s = np.linspace(0, 1, wout.ns)
    interior = slice(1, wout.ns - 1)
    jsupv = np.abs(jxbout.jsupv3 / np.where(jxbout.sqrtg3 == 0, np.inf, jxbout.sqrtg3))
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, constrained_layout=True)
    axes[0].plot(s[interior], wout.iotaf[interior], color="tab:green")
    axes[0].set_ylabel("iota")
    axes[1].semilogy(s[interior], jsupv.max(axis=1)[interior])
    axes[1].set_ylabel("max |J^v| on surface [A/m^2]")
    axes[2].plot(s[interior], np.sqrt(jxbout.jpar2[interior]) / 1e3)
    axes[2].set_ylabel("<J_par^2>^(1/2) [kA/m^2]")
    axes[2].set_xlabel("s")
    for value, s_value in rational_surfaces(wout):
        resonant = value.numerator % wout.nfp == 0
        for ax in axes:
            ax.axvline(
                s_value,
                color="k" if resonant else "0.6",
                lw=0.8,
                ls="-" if resonant else "--",
            )
        axes[0].annotate(
            str(value),
            (s_value, 1),
            xycoords=("data", "axes fraction"),
            xytext=(2, -12),
            textcoords="offset points",
            fontsize=8,
        )
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main(output_dir=Path()):
    outputs = {lbsubs: run(lbsubs) for lbsubs in (False, True)}
    wout = outputs[True].wout
    zetas = np.arange(4) * np.pi / (3 * wout.nfp)
    jxbout_k = np.arange(4) * outputs[True].input.nzeta // 6

    currents = {
        lbsubs: currents_from_bsubs_full(output.wout, output.wout.bsubsmns)
        for lbsubs, output in outputs.items()
    }
    fourier = {
        lbsubs: [
            (*cross_section(outputs[lbsubs].wout, *currents[lbsubs], zeta), zeta)
            for zeta in zetas
        ]
        for lbsubs in (False, True)
    }
    jxb = {
        lbsubs: [
            (*jxbout_cross_section(output, k), zeta)
            for k, zeta in zip(jxbout_k, zetas, strict=True)
        ]
        for lbsubs, output in outputs.items()
    }

    jphi_label = "J_phi = R J^v [kA/m^2]"
    jpar_label = "J.B/|B| [kA/m^2]"
    plot_rows(
        [("lbsubs = False", fourier[False]), ("lbsubs = True", fourier[True])],
        "jphi",
        jphi_label,
        output_dir / "w7x_jtor_lbsubs.png",
    )
    plot_rows(
        [("lbsubs = False, jxbout", jxb[False]), ("lbsubs = True, jxbout", jxb[True])],
        "jphi",
        jphi_label,
        output_dir / "w7x_jtor_lbsubs_jxbout.png",
    )
    for key, key_label in (("jphi", jphi_label), ("jpar", jpar_label)):
        plot_rows(
            [("lbsubs = True, jxbout", jxb[True])],
            key,
            key_label,
            output_dir / f"w7x_{key}_rational_surfaces.png",
            wout=wout,
        )
    plot_profiles(outputs[True], output_dir / "w7x_current_profiles.png")


if __name__ == "__main__":
    main(Path(__file__).parent)

# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Visualize a free-boundary dual-run dump (see DualSolver).

Reads the JSON-lines file written when VMEC++ runs with
VMECPP_FB_SHADOW=(nestor|biest) and VMECPP_FB_DUAL_DUMP=<path>, and produces:

1. Per-update side-by-side imshow panels of |B|^2/2 on the (theta, zeta)
   grid: primary | shadow | difference.
2. The evolution of the poloidal Fourier spectrum of |B|^2/2 for both
   solvers across vacuum updates (to expose high-k content building up).
3. Scalar traces across updates: RMS/max difference of the two fields, net
   current integrals of both solvers, and the spectral content of the
   boundary Fourier coefficients.

Usage:
    python tools/plot_fb_dual_run.py <dump.jsonl> [--out <dir>]
    python tools/plot_fb_dual_run.py <dump.jsonl> --updates 0 5 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    if not records:
        msg = f"no records found in {path}"
        raise ValueError(msg)
    return records


def infer_grid_shape(record: dict) -> tuple[int, int]:
    """The bsqvac buffers are indexed l * nZeta + k (theta slow, zeta fast).

    The dump does not carry the grid dimensions explicitly, so infer nZeta from the
    boundary coefficient array sizes is not possible either; instead require the user
    grid to be recoverable from the two most common cases.
    """
    n = len(record["primary_b_sq_vac"])
    # heuristics: try common aspect ratios by looking for integer factors
    # preferring nZeta > nTheta (VMEC reduced grids are wide in zeta)
    best = None
    for n_theta in range(2, int(np.sqrt(n)) + 1):
        if n % n_theta == 0:
            best = (n_theta, n // n_theta)
    if best is None:
        msg = f"cannot infer (nThetaEff, nZeta) from flat size {n}"
        raise ValueError(msg)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path, help="JSONL dump from DualSolver")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output directory (default: <dump-stem>_plots)",
    )
    parser.add_argument(
        "--updates",
        type=int,
        nargs="*",
        default=None,
        help="update indices to plot as imshow panels (default: ~8 evenly spaced)",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=None,
        help="poloidal grid size nThetaEff (inferred if omitted)",
    )
    args = parser.parse_args()

    records = load_records(args.dump)
    out_dir = args.out or args.dump.parent / f"{args.dump.stem}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ntheta is not None:
        n_flat = len(records[0]["primary_b_sq_vac"])
        shape = (args.ntheta, n_flat // args.ntheta)
    elif "n_theta_eff" in records[0]:
        shape = (records[0]["n_theta_eff"], records[0]["n_zeta"])
    else:
        shape = infer_grid_shape(records[0])
    n_theta, n_zeta = shape
    print(f"{len(records)} vacuum updates, grid (nThetaEff, nZeta) = {shape}")

    primary = np.array([r["primary_b_sq_vac"] for r in records]).reshape(
        len(records), n_theta, n_zeta
    )
    shadow = np.array([r["shadow_b_sq_vac"] for r in records]).reshape(
        len(records), n_theta, n_zeta
    )
    diff = primary - shadow
    update_idx = np.array([r["update_index"] for r in records])

    # ------------------------------------------------------------------
    # 1) side-by-side imshow panels for selected updates
    if args.updates is not None and len(args.updates) > 0:
        selected = [i for i in args.updates if i < len(records)]
    else:
        n_panels = min(8, len(records))
        selected = np.unique(
            np.linspace(0, len(records) - 1, n_panels).astype(int)
        ).tolist()

    vmin = min(primary.min(), shadow.min())
    vmax = max(primary.max(), shadow.max())
    dmax = np.abs(diff).max()

    for i in selected:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        im0 = axs[0].imshow(
            primary[i], origin="lower", aspect="auto", vmin=vmin, vmax=vmax
        )
        axs[0].set_title(f"primary |B|^2/2 (update {update_idx[i]})")
        im1 = axs[1].imshow(
            shadow[i], origin="lower", aspect="auto", vmin=vmin, vmax=vmax
        )
        axs[1].set_title("shadow |B|^2/2")
        im2 = axs[2].imshow(
            diff[i],
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-dmax,
            vmax=dmax,
        )
        axs[2].set_title("primary - shadow")
        for ax in axs:
            ax.set_xlabel("zeta index")
            ax.set_ylabel("theta index")
        fig.colorbar(im0, ax=axs[0])
        fig.colorbar(im1, ax=axs[1])
        fig.colorbar(im2, ax=axs[2])
        fig.savefig(out_dir / f"bsq_update_{update_idx[i]:04d}.png", dpi=150)
        plt.close(fig)

    # ------------------------------------------------------------------
    # 2) spectral evolution: poloidal FFT amplitude of bsqvac vs update
    # (theta is the reduced range for stellarator-symmetric runs; the DFT
    # along it still exposes relative high-k growth between the solvers)
    spec_primary = np.abs(np.fft.rfft(primary, axis=1)).mean(axis=2)
    spec_shadow = np.abs(np.fft.rfft(shadow, axis=1)).mean(axis=2)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    common = {
        "origin": "lower",
        "aspect": "auto",
        "norm": mpl.colors.LogNorm(
            vmin=max(spec_primary.min(), 1e-12), vmax=spec_primary.max()
        ),
    }
    im0 = axs[0].imshow(spec_primary.T, **common)
    axs[0].set_title("primary: poloidal spectrum of |B|^2/2")
    im1 = axs[1].imshow(spec_shadow.T, **common)
    axs[1].set_title("shadow: poloidal spectrum of |B|^2/2")
    ratio = spec_shadow / np.maximum(spec_primary, 1e-30)
    im2 = axs[2].imshow(
        np.log10(ratio).T,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
    )
    axs[2].set_title("log10(shadow / primary)")
    for ax in axs:
        ax.set_xlabel("vacuum update")
        ax.set_ylabel("poloidal mode m")
    fig.colorbar(im0, ax=axs[0])
    fig.colorbar(im1, ax=axs[1])
    fig.colorbar(im2, ax=axs[2])
    fig.savefig(out_dir / "bsq_spectral_evolution.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3) scalar traces
    rms = np.sqrt((diff**2).mean(axis=(1, 2)))
    maxabs = np.abs(diff).max(axis=(1, 2))
    scale = primary.mean(axis=(1, 2))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axs[0, 0].semilogy(update_idx, rms / scale, label="rms(diff)/mean")
    axs[0, 0].semilogy(update_idx, maxabs / scale, label="max|diff|/mean")
    axs[0, 0].set_xlabel("vacuum update")
    axs[0, 0].set_title("primary vs shadow |B|^2/2 difference")
    axs[0, 0].legend()

    axs[0, 1].plot(
        update_idx,
        [r["primary_b_sub_u_vac"] for r in records],
        label="primary bsubuvac",
    )
    axs[0, 1].plot(
        update_idx,
        [r["shadow_b_sub_u_vac"] for r in records],
        "--",
        label="shadow bsubuvac",
    )
    axs[0, 1].set_xlabel("vacuum update")
    axs[0, 1].set_title("net toroidal current integral")
    axs[0, 1].legend()

    # boundary spectral content: RMS of high-m half vs low-m half of rCC
    r_cc = np.array([r["rCC"] for r in records])
    half = r_cc.shape[1] // 2
    low = np.sqrt((r_cc[:, :half] ** 2).mean(axis=1))
    high = np.sqrt((r_cc[:, half:] ** 2).mean(axis=1))
    axs[1, 0].semilogy(update_idx, low, label="rCC low-mn RMS")
    axs[1, 0].semilogy(update_idx, high, label="rCC high-mn RMS")
    axs[1, 0].set_xlabel("vacuum update")
    axs[1, 0].set_title("boundary coefficient spectral content")
    axs[1, 0].legend()

    axs[1, 1].semilogy(
        update_idx,
        np.abs(spec_primary[:, -1]) + 1e-30,
        label="primary highest-m",
    )
    axs[1, 1].semilogy(
        update_idx,
        np.abs(spec_shadow[:, -1]) + 1e-30,
        "--",
        label="shadow highest-m",
    )
    axs[1, 1].set_xlabel("vacuum update")
    axs[1, 1].set_title("highest poloidal mode of |B|^2/2")
    axs[1, 1].legend()

    fig.savefig(out_dir / "dual_run_traces.png", dpi=150)
    plt.close(fig)

    print(f"wrote plots to {out_dir}/")


if __name__ == "__main__":
    main()

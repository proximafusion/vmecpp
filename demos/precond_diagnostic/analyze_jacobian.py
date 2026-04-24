# ruff: noqa: T201, B905, ICN001, F841, C401
"""Analyze a saved Jacobian extraction.

Computes effective condition numbers (numerical-rank truncated), distribution of
singular values, and mode-wise structure. Produces richer plots than the initial
extraction script.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def effective_cond(sv: np.ndarray, rel_tol: float = 1e-10) -> dict:
    """Condition number after truncating SVs below rel_tol * max.

    Returns numerical rank, effective kappa, and the truncation threshold used.
    """
    if sv.size == 0:
        return {"rank": 0, "cond_eff": float("inf"), "threshold": 0.0}
    smax = float(sv[0])
    threshold = rel_tol * smax
    kept = sv[sv > threshold]
    rank = int(kept.size)
    cond_eff = float(kept[0] / kept[-1]) if kept.size > 0 else float("inf")
    return {"rank": rank, "cond_eff": cond_eff, "threshold": threshold}


def per_mode_stiffness(
    matrix: np.ndarray, m_arr: np.ndarray, n_arr: np.ndarray, comp_arr: np.ndarray
) -> dict:
    """Sum |matrix[i, j]|^2 over (i, j) with matching (comp, m, n) indices.

    Gives the ratio of diagonal-band energy for R/Z/lambda per Fourier mode. This
    identifies which (m, n) modes the operator puts most energy on.
    """
    mpol = int(m_arr.max()) + 1
    ntorp1 = int(n_arr.max()) + 1
    per_mode = {
        comp: np.zeros((mpol, ntorp1), dtype=np.float64) for comp in ("R", "Z", "L")
    }
    comp_names = {0: "R", 1: "Z", 2: "L"}
    N = matrix.shape[0]
    for i in range(N):
        ci = comp_names[int(comp_arr[i])]
        mi = int(m_arr[i])
        ni = int(n_arr[i])
        per_mode[ci][mi, ni] += float(matrix[i, i])
    return per_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    if args.tag is None:
        args.tag = args.npz.stem
    if args.out_dir is None:
        args.out_dir = args.npz.parent

    data = np.load(args.npz, allow_pickle=True)
    J = data["J"]
    PiJ = data["PiJ"]
    m_arr = data["m"]
    n_arr = data["n"]
    comp_arr = data["comp"]
    jF_arr = data["jF"]
    basis_arr = data["basis"]

    print(f"N = {J.shape[0]}; J.shape = {J.shape}", flush=True)

    sv_J = np.linalg.svd(J, compute_uv=False)
    sv_PiJ = np.linalg.svd(PiJ, compute_uv=False)

    stats = {
        "N": int(J.shape[0]),
        "frobenius_J": float(np.linalg.norm(J)),
        "frobenius_PiJ": float(np.linalg.norm(PiJ)),
        "spectral_J": float(sv_J[0]),
        "spectral_PiJ": float(sv_PiJ[0]),
        "min_sv_J": float(sv_J[-1]),
        "min_sv_PiJ": float(sv_PiJ[-1]),
    }
    for rel_tol in (1e-6, 1e-8, 1e-10, 1e-12):
        key = f"rel_tol_{rel_tol:.0e}"
        stats[f"J_{key}"] = effective_cond(sv_J, rel_tol)
        stats[f"PiJ_{key}"] = effective_cond(sv_PiJ, rel_tol)
    with open(args.out_dir / f"{args.tag}_analysis.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Singular-value spectrum: log-log with relative normalization.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(
        sv_J / sv_J[0],
        "-",
        color="C0",
        label=f"sigma(J)/sigma_max,  kappa={stats['J_rel_tol_1e-08']['cond_eff']:.2e}",
    )
    ax.semilogy(
        sv_PiJ / sv_PiJ[0],
        "-",
        color="C3",
        label=f"sigma(P^-1 J)/sigma_max,  kappa={stats['PiJ_rel_tol_1e-08']['cond_eff']:.2e}",
    )
    ax.axhline(1e-8, color="gray", ls="--", lw=0.8, label="rel tol 1e-8")
    ax.axhline(1e-12, color="gray", ls=":", lw=0.8, label="rel tol 1e-12")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("sigma / sigma_max")
    ax.set_title(f"{args.tag}: normalized singular-value spectrum (N={J.shape[0]})")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"{args.tag}_sv_normalized.png", dpi=120)
    plt.close(fig)

    # Matrix structure log-magnitude.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, M, title in zip(
        axes, (J, PiJ), ("J (unpreconditioned)", "P^-1 J (preconditioned)")
    ):
        im = ax.imshow(
            np.log10(np.abs(M) + 1e-30),
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
            vmin=-6,
            vmax=max(np.log10(np.abs(J).max()), np.log10(np.abs(PiJ).max())),
        )
        ax.set_title(title)
        ax.set_xlabel("state index j")
        ax.set_ylabel("force index i")
        plt.colorbar(im, ax=ax, shrink=0.8, label="log10 |entry|")
    fig.suptitle(f"{args.tag}: matrix structure (log-magnitude)")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"{args.tag}_structure.png", dpi=120)
    plt.close(fig)

    # Diagonal stiffness per (m,n): how much does each Fourier mode
    # self-couple through the preconditioned operator? This is a cheap
    # proxy that tells us whether the preconditioner has made the
    # self-coupling roughly uniform across modes.
    for matrix, mtag in ((J, "J"), (PiJ, "PiJ")):
        per_mode = per_mode_stiffness(matrix, m_arr, n_arr, comp_arr)
        for comp in ("R", "Z", "L"):
            data_mode = per_mode[comp]
            if data_mode.max() == 0:
                continue
            fig, ax = plt.subplots(figsize=(6, 4.5))
            im = ax.imshow(
                np.log10(np.abs(data_mode) + 1e-30),
                cmap="plasma",
                aspect="auto",
                origin="lower",
            )
            ax.set_xlabel("n")
            ax.set_ylabel("m")
            ax.set_title(f"{args.tag}: diag({mtag}), component {comp}")
            plt.colorbar(im, ax=ax, label="log10 |sum_jF diag|")
            fig.tight_layout()
            fig.savefig(args.out_dir / f"{args.tag}_diag_{mtag}_{comp}.png", dpi=120)
            plt.close(fig)

    # Radial profile of diag entries summed over Fourier modes; shows
    # whether the operator is uniform across radial surfaces or spikes
    # at the axis / boundary.
    N = J.shape[0]
    for matrix, mtag in ((J, "J"), (PiJ, "PiJ")):
        diag = np.abs(np.diag(matrix))
        comp_names = {0: "R", 1: "Z", 2: "L"}
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for comp_id, cname in comp_names.items():
            mask = comp_arr == comp_id
            if not mask.any():
                continue
            # Bin by jF
            jf_vals = sorted(set(int(x) for x in jF_arr[mask]))
            binned = []
            for j in jf_vals:
                bm = mask & (jF_arr == j)
                binned.append(float(np.sqrt((diag[bm] ** 2).sum())))
            ax.semilogy(jf_vals, binned, "-", label=cname)
        ax.set_xlabel("radial surface jF")
        ax.set_ylabel(f"|diag({mtag})|_2 per component per surface")
        ax.set_title(f"{args.tag}: diag({mtag}) radial profile")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out_dir / f"{args.tag}_diag_{mtag}_radial.png", dpi=120)
        plt.close(fig)

    # Report.
    print("\n=== condition numbers ===")
    for rel_tol in (1e-6, 1e-8, 1e-10, 1e-12):
        print(
            f"rel_tol={rel_tol:.0e}:  "
            f"J kappa={stats[f'J_rel_tol_{rel_tol:.0e}']['cond_eff']:.3e} "
            f"(rank {stats[f'J_rel_tol_{rel_tol:.0e}']['rank']}),  "
            f"P^-1 J kappa={stats[f'PiJ_rel_tol_{rel_tol:.0e}']['cond_eff']:.3e} "
            f"(rank {stats[f'PiJ_rel_tol_{rel_tol:.0e}']['rank']})"
        )
    print(
        "\n||J||_F / ||P^-1 J||_F  =  "
        f"{stats['frobenius_J'] / stats['frobenius_PiJ']:.3e}"
    )
    print(
        "sigma_max(J) / sigma_max(P^-1 J)  =  "
        f"{stats['spectral_J'] / stats['spectral_PiJ']:.3e}"
    )


if __name__ == "__main__":
    main()

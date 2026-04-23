# ruff: noqa: T201, ICN001
"""Finite-difference Jacobian extraction for the VMEC++ preconditioner.

For a chosen VMEC input:
  1. Run to convergence in single-thread mode via VmecJacobianProbe.
  2. Snapshot the state x*.
  3. For each one-hot perturbation e_i, compute
       J[:, i]   = (f_unprec(x* + eps e_i) - f_unprec(x*)) / eps
       PiJ[:, i] = (f_prec(x* + eps e_i)   - f_prec(x*))   / eps
     where f_unprec is the force before M1+RZ+lambda preconditioning,
     and f_prec is the force after. P^-1 J = PiJ.
  4. Save J and PiJ as .npz together with the mode-index metadata and
     plot condition-number diagnostics.

Usage:
    python demos/precond_diagnostic/compute_jacobian.py \
        --input examples/data/solovev.json --out solovev_baseline
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import vmecpp
from vmecpp.cpp import _vmecpp as cpp  # type: ignore[attr-defined]


def load_indata(
    path: Path,
    mpol_override: int | None = None,
    ntor_override: int | None = None,
    ns_override: int | None = None,
):
    pi = vmecpp.VmecInput.from_file(str(path))
    if mpol_override is not None or ntor_override is not None:
        new_mpol = mpol_override if mpol_override is not None else pi.mpol
        new_ntor = ntor_override if ntor_override is not None else pi.ntor
        # Trim rbc/zbs to the new (mpol, ntor) envelope.
        pi.rbc = pi.rbc[:new_mpol, : 2 * new_ntor + 1]
        pi.zbs = pi.zbs[:new_mpol, : 2 * new_ntor + 1]
        if pi.rbs is not None:
            pi.rbs = pi.rbs[:new_mpol, : 2 * new_ntor + 1]
        if pi.zbc is not None:
            pi.zbc = pi.zbc[:new_mpol, : 2 * new_ntor + 1]
        if len(pi.raxis_c) > new_ntor + 1:
            pi.raxis_c = pi.raxis_c[: new_ntor + 1]
        if len(pi.zaxis_s) > new_ntor + 1:
            pi.zaxis_s = pi.zaxis_s[: new_ntor + 1]
        if pi.raxis_s is not None and len(pi.raxis_s) > new_ntor + 1:
            pi.raxis_s = pi.raxis_s[: new_ntor + 1]
        if pi.zaxis_c is not None and len(pi.zaxis_c) > new_ntor + 1:
            pi.zaxis_c = pi.zaxis_c[: new_ntor + 1]
        pi.mpol = new_mpol
        pi.ntor = new_ntor
    if ns_override is not None:
        pi.ns_array = np.array([ns_override], dtype=np.int32)
        pi.ftol_array = np.array([pi.ftol_array[-1]], dtype=np.float64)
        pi.niter_array = np.array([pi.niter_array[-1]], dtype=np.int32)
    return pi._to_cpp_vmecindata()


def build_probe(
    indata_path: Path,
    mpol_override: int | None = None,
    ntor_override: int | None = None,
    ns_override: int | None = None,
) -> cpp.VmecJacobianProbe:
    indata = load_indata(indata_path, mpol_override, ntor_override, ns_override)
    probe = cpp.VmecJacobianProbe(indata)
    probe.run_to_convergence()
    probe.snapshot_state()
    return probe


def extract_jacobian(
    probe: cpp.VmecJacobianProbe,
    preconditioned: bool,
    eps: float,
) -> np.ndarray:
    """Build the finite-difference Jacobian by one-hot probing.

    Uses central differences: J[:, i] = (f(x+eps e_i) - f(x-eps e_i)) / (2 eps).
    """
    n = probe.num_state_vars()
    x0 = np.asarray(probe.get_state_vector(), dtype=np.float64)
    probe.set_state_vector(x0)
    f0 = np.asarray(probe.evaluate_forces(preconditioned), dtype=np.float64)
    assert f0.shape == (n,), (f0.shape, n)
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xp = x0.copy()
        xp[i] += eps
        probe.set_state_vector(xp)
        fp = np.asarray(probe.evaluate_forces(preconditioned), dtype=np.float64)
        xm = x0.copy()
        xm[i] -= eps
        probe.set_state_vector(xm)
        fm = np.asarray(probe.evaluate_forces(preconditioned), dtype=np.float64)
        J[:, i] = (fp - fm) / (2.0 * eps)
        if i % max(1, n // 40) == 0 or i == n - 1:
            print(
                f"  column {i:5d}/{n}  |f_col|={np.linalg.norm(J[:, i]):.3e}",
                flush=True,
            )
    probe.restore_state()
    return J


def plot_diagnostics(
    J: np.ndarray,
    PiJ: np.ndarray,
    out_dir: Path,
    tag: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = J.shape[0]

    sv_J = np.linalg.svd(J, compute_uv=False)
    sv_PiJ = np.linalg.svd(PiJ, compute_uv=False)

    def cond(sv):
        sv_nonzero = sv[sv > 1e-300]
        if sv_nonzero.size == 0:
            return float("inf")
        return float(sv_nonzero[0] / sv_nonzero[-1])

    cond_J = cond(sv_J)
    cond_PiJ = cond(sv_PiJ)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(sv_J, "-", color="C0", label=f"sigma(J), kappa={cond_J:.2e}")
    ax.semilogy(sv_PiJ, "-", color="C3", label=f"sigma(P^-1 J), kappa={cond_PiJ:.2e}")
    ax.set_xlabel("singular value index")
    ax.set_ylabel("singular value")
    ax.set_title(f"{tag}: singular-value spectrum (N={n})")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_singular_values.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, M, title in zip(
        axes,
        (J, PiJ),
        ("J (unpreconditioned)", "P^-1 J (preconditioned)"),
        strict=True,
    ):
        im = ax.imshow(
            np.log10(np.abs(M) + 1e-30),
            cmap="viridis",
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("state index j")
        ax.set_ylabel("force index i")
        plt.colorbar(im, ax=ax, shrink=0.8, label="log10 |.|")
    fig.suptitle(f"{tag}: matrix structure")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_structure.png", dpi=120)
    plt.close(fig)

    eigs_PiJ_sym = np.linalg.eigvalsh(0.5 * (PiJ + PiJ.T))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sorted(eigs_PiJ_sym), "-", color="C2")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue (symmetric part)")
    ax.set_title(f"{tag}: symmetric-part eigenvalues of P^-1 J")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_eigs_symmetric_part.png", dpi=120)
    plt.close(fig)

    norms = {
        "frobenius_J": float(np.linalg.norm(J, "fro")),
        "frobenius_PiJ": float(np.linalg.norm(PiJ, "fro")),
        "spectral_J": float(sv_J[0]) if sv_J.size else 0.0,
        "spectral_PiJ": float(sv_PiJ[0]) if sv_PiJ.size else 0.0,
        "min_sv_J": float(sv_J[-1]) if sv_J.size else 0.0,
        "min_sv_PiJ": float(sv_PiJ[-1]) if sv_PiJ.size else 0.0,
        "cond_J": cond_J,
        "cond_PiJ": cond_PiJ,
        "min_eig_sym_PiJ": float(eigs_PiJ_sym.min()),
        "max_eig_sym_PiJ": float(eigs_PiJ_sym.max()),
        "N": int(n),
    }
    with open(out_dir / f"{tag}_stats.json", "w") as f:
        json.dump(norms, f, indent=2)
    return norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", type=Path, required=True, help="VMEC input file (.json or INDATA)"
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="output tag (files named <tag>_*.png/.npz)",
    )
    ap.add_argument(
        "--eps", type=float, default=1e-7, help="finite-difference step (default 1e-7)"
    )
    ap.add_argument(
        "--out-dir", type=Path, default=Path("demos/precond_diagnostic/jacobian")
    )
    ap.add_argument(
        "--mpol", type=int, default=None, help="override mpol (trims rbc/zbs envelope)"
    )
    ap.add_argument("--ntor", type=int, default=None, help="override ntor")
    ap.add_argument(
        "--ns", type=int, default=None, help="override ns_array to a single value"
    )
    args = ap.parse_args()

    print(f"[probe] loading {args.input} ...", flush=True)
    probe = build_probe(args.input, args.mpol, args.ntor, args.ns)
    n = probe.num_state_vars()
    print(
        f"[probe] converged; N={n}, mpol={probe.mpol()}, "
        f"ntor+1={probe.ntor_plus_one()}, num_basis={probe.num_basis()}, "
        f"ns_local={probe.num_full_surfaces()}, "
        f"axisym={probe.is_axisymmetric()}, asym={probe.is_asymmetric()}",
        flush=True,
    )

    meta = {
        "jF": np.asarray(probe.index_jF()),
        "m": np.asarray(probe.index_m()),
        "n": np.asarray(probe.index_n()),
        "basis": np.asarray(probe.index_basis()),
        "comp": np.asarray(probe.index_comp()),
        "basis_names": probe.get_basis_names(),
        "mpol": probe.mpol(),
        "ntor_plus_one": probe.ntor_plus_one(),
        "num_basis": probe.num_basis(),
        "num_full_surfaces": probe.num_full_surfaces(),
        "is_axisymmetric": probe.is_axisymmetric(),
        "is_asymmetric": probe.is_asymmetric(),
    }

    print("[probe] extracting J (unpreconditioned) ...", flush=True)
    J = extract_jacobian(probe, preconditioned=False, eps=args.eps)
    print("[probe] extracting P^-1 J (preconditioned) ...", flush=True)
    PiJ = extract_jacobian(probe, preconditioned=True, eps=args.eps)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / f"{args.out}.npz",
        J=J,
        PiJ=PiJ,
        **{k: v for k, v in meta.items() if not isinstance(v, list)},
    )
    print(f"[probe] saved {out_dir / (args.out + '.npz')}", flush=True)

    stats = plot_diagnostics(J, PiJ, out_dir, args.out)
    print(f"[probe] stats: {json.dumps(stats, indent=2)}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Plot total force-residual convergence (wout.fsqt) for every config, one subplot per
initial-guess method (default / zeno / map2disc).

QUASR boundaries are pre-flipped in theta when needed so the guess is actually
applied (otherwise VMEC++ discards it and the curves would be identical across
panels -- see FINDINGS.md).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vmecpp

WORK = Path("/tmp/quasr_init_study")
REPO = Path()
QUASR = "https://quasr.flatironinstitute.org/"
METHODS = ["default", "map2disc"]
NITER_CAP = 2000
FTOL = 1e-11


def flip_indata(v):
    """Pre-flip the boundary (per-m-parity) so VMEC++ does not internally re-flip,
    keeping boundary and hot-restart interior in a consistent orientation."""
    v = v.model_copy(deep=True)
    rbc = np.asarray(v.rbc, float).copy()
    zbs = np.asarray(v.zbs, float).copy()
    for m in range(1, rbc.shape[0]):
        par = 1 if m % 2 == 0 else -1
        rbc[m, :] *= par
        zbs[m, :] *= -par
    v.rbc, v.zbs = rbc, zbs
    return v


def needs_flip(vi) -> bool:
    import os

    inp = vi.model_copy(deep=True)
    inp.ns_array = np.array([13])
    inp.ftol_array = np.array([1e-6])
    inp.niter_array = np.array([1])
    inp.return_outputs_even_if_not_converged = True
    r, w = os.pipe()
    old = os.dup(1)
    os.dup2(w, 1)
    try:
        try:
            vmecpp.run(inp, verbose=True, max_threads=1)
        except Exception:
            pass
    finally:
        os.dup2(old, 1)
        os.close(w)
        out = os.read(r, 200000).decode("utf-8", "replace")
        os.close(r)
    return "flip theta" in out


def trace(vi, ns, method):
    """Return the total force-residual trace fsqt and ftolv for one run."""
    inp = vi.model_copy(deep=True)
    inp.ns_array = np.array([ns])
    inp.ftol_array = np.array([FTOL])
    inp.niter_array = np.array([NITER_CAP])
    inp.return_outputs_even_if_not_converged = True
    g = None
    try:
        if method == "zeno":
            g = vmecpp.zeno_guess(vi, ns=ns, lmax=3)
        elif method == "map2disc":
            g = vmecpp.map2disc_guess(vi, ns=ns)
    except Exception:
        return None
    if g is not None:
        g.input = g.input.model_copy(deep=True)
        g.input.ns_array = np.array([ns])
        g.input.ftol_array = np.array([FTOL])
        g.input.niter_array = np.array([NITER_CAP])
    try:
        o = vmecpp.run(inp, restart_from=g, verbose=False, max_threads=1)
    except Exception:
        return None
    return np.asarray(o.wout.fsqt), float(o.wout.ftolv)


def quasr_input(rid, n=7):
    i = f"{rid}".zfill(7)
    dst = WORK / f"input.{i}"
    if not dst.exists():
        req = urllib.request.Request(
            f"{QUASR}nml/{i[:4]}/input.{i}", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            dst.write_bytes(r.read())
    vi = vmecpp.VmecInput.from_file(dst)
    vi.rbc = vmecpp.VmecInput.resize_2d_coeff(np.asarray(vi.rbc), n, n)
    vi.zbs = vmecpp.VmecInput.resize_2d_coeff(np.asarray(vi.zbs), n, n)

    def fix(a):
        a = np.asarray(a, float)
        out = np.zeros(n + 1)
        out[: min(len(a), n + 1)] = a[: n + 1]
        return out

    vi.raxis_c = fix(vi.raxis_c)
    vi.zaxis_s = fix(vi.zaxis_s)
    vi.mpol = n
    vi.ntor = n
    vi.pmass_type = "power_series"
    vi.am = np.array([0.0])
    vi.pres_scale = 0.0
    vi.ncurr = 1
    vi.ac = np.array([0.0])
    vi.curtor = 0.0
    vi.gamma = 0.0
    return vi


def main():
    # (label, VmecInput, ns)
    configs = []
    checked = [
        ("cth_like", REPO / "examples/data/cth_like_fixed_bdy.json", None),
        ("solovev", REPO / "examples/data/solovev.json", 51),
        (
            "circular_tokamak",
            REPO / "src/vmecpp/cpp/vmecpp/test_data/circular_tokamak.json",
            None,
        ),
        ("cma", REPO / "src/vmecpp/cpp/vmecpp/test_data/cma.json", None),
        ("w7x", REPO / "examples/data/w7x.json", 25),
    ]
    for name, path, nso in checked:
        vi = vmecpp.VmecInput.from_file(path)
        ns = nso or int(np.asarray(vi.ns_array)[-1])
        if needs_flip(vi):
            vi = flip_indata(vi)
        configs.append((name, vi, ns))

    quasr_ids = [952, 953, 954, 957, 40376, 40379, 65527, 65528, 65529, 65530]
    for rid in quasr_ids:
        vi = quasr_input(rid)
        if needs_flip(vi):
            vi = flip_indata(vi)
        configs.append((f"QUASR-{rid}", vi, 25))

    # collect traces: results[method][label] = (fsqt, ftolv)
    results = {m: {} for m in METHODS}
    ftolv_seen = []
    for label, vi, ns in configs:
        for m in METHODS:
            t = trace(vi, ns, m)
            if t is not None:
                results[m][label] = t
                ftolv_seen.append(t[1])
        print(f"done {label}", flush=True)

    ftolv = min(ftolv_seen) if ftolv_seen else FTOL

    # consistent color per config across panels
    labels = [c[0] for c in configs]
    cmap = plt.get_cmap("tab20")
    colors = {lab: cmap(i % 20) for i, lab in enumerate(labels)}

    fig, axes = plt.subplots(
        1, len(METHODS), figsize=(6 * len(METHODS), 6), sharey=True
    )
    for ax, method in zip(axes, METHODS):
        for lab in labels:
            tr = results[method].get(lab)
            if tr is None:
                continue
            fsqt = tr[0]
            ax.plot(np.arange(len(fsqt)), fsqt, color=colors[lab], lw=1.0, label=lab)
        ax.axhline(ftolv, color="red", ls="--", lw=1, label="tolerance")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_title(f"{method} initial guess")
        ax.grid(True, which="both", alpha=0.2)
    axes[0].set_ylabel("Total force residual (fsqt)")
    # one shared legend (configs) to the right
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="center right", fontsize=8, framealpha=0.9)
    fig.suptitle(
        "Total force-residual convergence by initial-guess method "
        "(single multigrid step; QUASR theta-flipped so guess is applied)"
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.96))
    out = WORK / "force_convergence_by_method.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

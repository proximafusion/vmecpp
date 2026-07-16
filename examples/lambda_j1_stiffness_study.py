"""Measure the effective stiffness of the first-evolved-surface lambda modes.

Finding 12 (docs/convergence_study.md) left a mystery: the whole -23% w7x tail
gain of the lambda-preconditioner boost comes from the m <= 1 lambda modes on
the single first evolved surface j = 1, i.e. their effective stiffness is
~5-10x below what faclam (the flux-surface-averaged diagonal second variation)
estimates. Candidate mechanisms:

  (a) local operator defect: the true Hessian diagonal at j = 1 is genuinely
      smaller than faclam's estimate (half-grid metric quality on the innermost
      tiny surface, or the lambda(0) = 0 boundary condition entering the
      discrete operator asymmetrically at j = 1);
  (b) coupled-mode softness: the diagonal is fine, but the slow eigenvector of
      the preconditioned Hessian is an extended radial mode whose Rayleigh
      quotient is small -- then a diagonal fix is inherently heuristic.

This script discriminates them empirically at a converged-tail snapshot using
the pybind probe machinery:

  - VmecModel.hessian_vector_product(e) gives raw Hessian columns H e (central
    FD of the analytic force) in the decomposed internal basis -- exactly the
    coordinates the descent iterates in;
  - VmecModel.apply_preconditioner maps them through the native preconditioner
    sequence (m=1 gauge, radial R/Z block, lambda diagonal), so the recorded
    columns are columns of the effective iteration matrix B = M^{-1} H.

For each lambda (m, n) family we assemble the full ns x ns radial block of B
(one HVP per radial coordinate), and report:

  - the diagonal profile B_jj vs j     -> hypothesis (a) predicts B_11 ~ 5-10x
                                          below the mid-radius plateau;
  - the tridiagonal couplings           -> radial structure of the operator;
  - eigenvalues of the family block and the localization of the softest
    eigenvector                        -> hypothesis (b) predicts an extended
                                          soft mode with healthy diagonals;
  - off-family leakage of each column   -> how (in)valid the family-block
                                          restriction is.

An R-family block is recorded as a well-preconditioned reference, and the
whole measurement is repeated with the j<=1, m<=1 5x boost enabled as a
consistency check (its only effect must be to scale the corresponding rows of
B by 5).

State layout (FlattenActive): [span][js][m][n] with spans
[rcc, rss, zsc, zcs, lsc, lcs] for a symmetric lthreed case. Note m = 0
lambda content lives in lmncs (span 5, sin(n zeta)); m >= 1 in lmnsc (span 4).
"""

import argparse
import contextlib

import numpy as np

import vmecpp
from vmecpp._iteration import solve_equilibrium
from vmecpp.cpp import _vmecpp  # type: ignore[attr-defined]

CASES = {
    "cth": "src/vmecpp/cpp/vmecpp/test_data/cth_like_fixed_bdy.json",
    "cma": "src/vmecpp/cpp/vmecpp/test_data/cma.json",
    "w7x": "examples/data/w7x.json",
}
NS_OLD = 25
NS_NEW = 99


class _StopEarly(Exception):
    """Raised from a callback to end solve_equilibrium at the snapshot point."""


def make_model(path):
    vi = vmecpp.VmecInput.from_file(path)
    cpp = vi._to_cpp_vmecindata()
    cpp.ns_array = np.array([NS_OLD, NS_NEW], dtype=np.int64)
    cpp.ftol_array = np.array([1.0e-9, 1.0e-11])
    cpp.niter_array = np.array([6000, 6000], dtype=np.int64)
    return _vmecpp.VmecModel.create(cpp, NS_OLD)


def tail_snapshot(model, snapshot_fsq):
    r1 = solve_equilibrium(model)
    assert r1.converged, "coarse stage did not converge"
    model.refine_to(NS_NEW, interpolation=_vmecpp.MultigridInterpolationScheme.CUBIC)
    snap = {}

    def cb(s):
        if s.fsq_invariant < snapshot_fsq:
            snap["state"] = np.array(model.get_state())
            snap["iteration"] = s.iteration
            raise _StopEarly

    with contextlib.suppress(_StopEarly):
        solve_equilibrium(
            model, style="delt_recovery", delt_start_fraction=0.5, callback=cb
        )
    assert "state" in snap, "snapshot threshold not crossed"
    return snap


def family_block(model, shape, span, m, n, js_list):
    """Columns of B = M^{-1} H for the (span, m, n) radial family.

    Returns (block[js, js'], raw_diag[js], leak[js], leak_where[js]) where block[:, k]
    is the family-restricted preconditioned Hessian column for a unit perturbation at
    js_list[k], raw_diag the unpreconditioned Hessian diagonal, leak the largest |entry|
    of the column outside the family and leak_where its (span, js, m, n) index.
    """
    n_js = len(js_list)
    block = np.zeros((n_js, n_js))
    raw_diag = np.zeros(n_js)
    leak = np.zeros(n_js)
    leak_where = np.zeros((n_js, 4), dtype=int)
    size = int(np.prod(shape))
    for k, js in enumerate(js_list):
        e = np.zeros(shape)
        e[span, js, m, n] = 1.0
        h = np.asarray(model.hessian_vector_product(e.reshape(size)))
        p = np.asarray(model.apply_preconditioner(h))
        p4 = p.reshape(shape)
        raw_diag[k] = h.reshape(shape)[span, js, m, n]
        col = p4[span, :, m, n]
        block[:, k] = col[list(js_list)]
        off = np.array(p4)
        off[span, :, m, n] = 0.0
        flat = np.argmax(np.abs(off))
        leak[k] = np.abs(off).reshape(-1)[flat]
        leak_where[k] = np.unravel_index(flat, shape)
    return block, raw_diag, leak, leak_where


def describe_modes(js, w, v, order, label):
    print(f"  {label}:")
    for i in order:
        vec = np.real(v[:, i])
        avec = np.abs(vec)
        peak = int(js[np.argmax(avec)])
        share1 = float(avec[0] ** 2 / np.sum(avec**2))
        width = float(np.sum(avec**2) ** 2 / np.sum(avec**4))
        # sign alternation rate along the radial direction (1.0 = grid-scale
        # oscillation, 0.0 = smooth)
        sig = np.sign(vec[avec > 1e-3 * avec.max()])
        flips = float(np.mean(sig[1:] * sig[:-1] < 0)) if sig.size > 1 else 0.0
        print(
            f"    mu = {np.real(w[i]):9.5f}  peak js={peak:3d}  "
            f"|v(1)|^2 = {share1:6.3f}  width = {width:6.1f}  "
            f"alternation = {flips:4.2f}"
        )


def analyze_family(name, block, raw_diag, leak, leak_where, js_list):
    js = np.array(js_list)
    d = np.diag(block)
    # plateau reference: median effective diagonal over mid radius
    mid = d[(js >= 10) & (js <= len(js) - 10)]
    ref = np.median(mid) if mid.size else np.median(d)
    print(f"\n== family {name}: effective preconditioned diagonal B_jj ==")
    print("  js:  " + " ".join(f"{j:7d}" for j in js[:12]))
    print("  Bjj: " + " ".join(f"{v:7.3f}" for v in d[:12]))
    print(f"  mid-radius plateau median: {ref:.4f}")
    print(f"  B_11/plateau = {d[0] / ref:.3f}   (js={js[0]})")
    print(
        "  raw H diag (first 6 js): "
        + " ".join(f"{v:9.2e}" for v in raw_diag[:6])
        + f"   mid-radius median: {np.median(raw_diag[10:-10]):9.2e}"
    )
    sub = np.array([block[k + 1, k] for k in range(len(js) - 1)])
    sup = np.array([block[k, k + 1] for k in range(len(js) - 1)])
    print(
        "  couplings B_(j+1,j)/B_jj (first 6): "
        + " ".join(f"{sub[k] / d[k]:7.3f}" for k in range(6))
    )
    print(
        "  couplings B_(j,j+1)/B_jj (first 6): "
        + " ".join(f"{sup[k] / d[k]:7.3f}" for k in range(6))
    )
    kmax = int(np.argmax(leak))
    print(
        f"  max off-family leakage: {np.max(leak):.3e} "
        f"(column js={js[kmax]} -> span/js/m/n {tuple(leak_where[kmax])})"
    )
    w, v = np.linalg.eig(block)
    # eigenvalues are negative (B = -M^{-1} d2W/dx2); mu magnitude is the
    # descent stiffness. SOFT modes are the ones closest to zero.
    order_soft = np.argsort(-np.real(w))[:5]
    order_stiff = np.argsort(np.real(w))[:3]
    describe_modes(js, w, v, order_soft, "softest 5 modes (closest to zero)")
    describe_modes(js, w, v, order_stiff, "stiffest 3 modes")
    return d, ref


def cluster_probe(model, shape, js_set, m_max):
    """Cross-family probe: the low-m modes of ALL spans on the innermost
    surfaces form one strongly coupled cluster (the family-restricted analysis
    showed L columns at js=1 leaking 2-3x their diagonal into other lambda
    families at js=1). Assemble the preconditioned Hessian block over the full
    cluster, plus each cluster column's largest coupling OUTSIDE the cluster.

    Returns (coords, block, ext_leak) with coords a list of
    (span, js, m, n) tuples.
    """
    n_spans, _, _, nt1 = shape
    coords = [
        (sp, js, m, n)
        for sp in range(n_spans)
        for js in js_set
        for m in range(m_max + 1)
        for n in range(nt1)
    ]
    size = int(np.prod(shape))
    ncl = len(coords)
    block = np.zeros((ncl, ncl))
    ext_leak = np.zeros(ncl)
    cl_mask = np.zeros(shape, dtype=bool)
    for sp, js, m, n in coords:
        cl_mask[sp, js, m, n] = True
    for k, (sp, js, m, n) in enumerate(coords):
        e = np.zeros(shape)
        e[sp, js, m, n] = 1.0
        h = np.asarray(model.hessian_vector_product(e.reshape(size)))
        p4 = np.asarray(model.apply_preconditioner(h)).reshape(shape)
        block[:, k] = np.array([p4[c] for c in coords])
        ext = np.array(p4)
        ext[cl_mask] = 0.0
        ext_leak[k] = np.max(np.abs(ext))
    return coords, block, ext_leak


def coord_label(span_names, c):
    sp, js, m, n = c
    return f"{span_names[sp]}(js={js},m={m},n={n})"


def analyze_cluster(coords, block, ext_leak, span_names, g_pre=None, tag=""):
    # drop inactive coordinates (zero diagonal): fixed/undefined DOFs
    d = np.diag(block)
    active = np.abs(d) > 1e-8
    idx = np.where(active)[0]
    b = block[np.ix_(idx, idx)]
    print(
        f"\n== cluster analysis{tag}: {idx.size} active of {len(coords)} "
        f"coordinates, max external leakage {np.max(ext_leak):.3e} =="
    )
    w, v = np.linalg.eig(b)
    order_soft = np.argsort(-np.real(w))[:8]
    for i in order_soft:
        vec = np.abs(np.real(v[:, i]))
        top = np.argsort(-vec)[:4]
        desc = ", ".join(
            f"{coord_label(span_names, coords[idx[t]])}:{vec[t] / vec.max():.2f}"
            for t in top
        )
        overlap = ""
        if g_pre is not None:
            gv = g_pre[idx]
            gn = np.linalg.norm(gv)
            if gn > 0:
                overlap = f"  force overlap = {abs(np.real(v[:, i]) @ gv) / gn:.3f}"
        print(f"  mu = {np.real(w[i]):9.5f}{overlap}   [{desc}]")
    if g_pre is not None:
        gv = g_pre[idx]
        gn = np.linalg.norm(gv)
        top = np.argsort(-np.abs(gv))[:5]
        desc = ", ".join(
            f"{coord_label(span_names, coords[idx[t]])}:{abs(gv[t]) / gn:.2f}"
            for t in top
        )
        print(f"  preconditioned residual direction in cluster: [{desc}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="w7x", choices=sorted(CASES))
    ap.add_argument("--with-boost-check", action="store_true")
    ap.add_argument("--cluster", action="store_true", help="cross-family probe")
    ap.add_argument("--snapshot-fsq", type=float, default=1.0e-6)
    ap.add_argument("--out", default="", help="save the raw blocks to this .npz")
    args = ap.parse_args()

    model = make_model(CASES[args.case])
    snap = tail_snapshot(model, args.snapshot_fsq)
    print(f"# {args.case}: snapshot at fine-stage iteration {snap['iteration']}")

    model.set_state(snap["state"])
    model.save_backup()
    model.zero_velocity()
    # assemble the preconditioner at the snapshot state (radial + lambda)
    model.evaluate(2, 2, precondition=True)
    print(
        f"# snapshot fsq = {model.fsqr + model.fsqz + model.fsql:.3e}, "
        f"mhd_energy = {model.mhd_energy:.10e}"
    )

    mp, nt1 = model.mpol, model.ntor + 1
    n_spans = (2 if model.lthreed else 1) * 3
    x = np.array(model.get_state())
    shape = (n_spans, x.size // (mp * nt1 * n_spans), mp, nt1)
    ns = shape[1]
    assert ns == NS_NEW, (ns, NS_NEW)
    span_lsc = 4 if model.lthreed else 2
    span_lcs = 5  # m=0 lambda lives here (lthreed only)

    if args.cluster:
        span_names = (
            ["rcc", "rss", "zsc", "zcs", "lsc", "lcs"]
            if model.lthreed
            else ["rcc", "zsc", "lsc"]
        )
        # raw force at the snapshot, then its preconditioned direction (the
        # preconditioner arrays are already assembled by the evaluate above)
        model.evaluate(2, 2, precondition=False)
        f_raw = np.array(model.get_forces())
        g4 = np.asarray(model.apply_preconditioner(f_raw)).reshape(shape)
        js_set = (0, 1, 2)
        coords, block, ext_leak = cluster_probe(model, shape, js_set, 1)
        g_pre = np.array([g4[c] for c in coords])
        analyze_cluster(coords, block, ext_leak, span_names, g_pre)
        saved_cl = {
            "coords": np.array(coords),
            "block": block,
            "ext_leak": ext_leak,
            "g_pre": g_pre,
        }
        if args.with_boost_check:
            model.set_lambda_preconditioner_boost(scale=5.0, mmax=1, jmax=1)
            model.evaluate(2, 2, precondition=True)
            model.evaluate(2, 2, precondition=False)
            g4b = np.asarray(model.apply_preconditioner(f_raw)).reshape(shape)
            coords, block_b, ext_b = cluster_probe(model, shape, js_set, 1)
            g_pre_b = np.array([g4b[c] for c in coords])
            analyze_cluster(
                coords, block_b, ext_b, span_names, g_pre_b, tag=" [boosted]"
            )
            saved_cl["block_boosted"] = block_b
            saved_cl["g_pre_boosted"] = g_pre_b
            model.set_lambda_preconditioner_boost(scale=1.0)
        if args.out:
            np.savez(args.out, **saved_cl)  # pyright: ignore[reportArgumentType]
            print(f"\n# cluster blocks saved to {args.out}")
        return

    js_list = list(range(1, ns))  # j=0 lambda has no evolved DOF
    families = [
        ("L m=1 n=0 (lsc)", span_lsc, 1, 0),
        ("L m=1 n=1 (lsc)", span_lsc, 1, 1),
    ]
    if model.lthreed:
        families = [
            ("L m=0 n=1 (lcs)", span_lcs, 0, 1),
            ("L m=0 n=2 (lcs)", span_lcs, 0, 2),
            *families,
        ]
    # well-preconditioned reference channel
    families.append(("R m=1 n=0 (rcc)", 0, 1, 0))

    saved = {"js_list": np.array(js_list)}
    for name, span, m, n in families:
        block, raw_diag, leak, leak_where = family_block(
            model, shape, span, m, n, js_list
        )
        analyze_family(name, block, raw_diag, leak, leak_where, js_list)
        saved[f"{name}/block"] = block
        saved[f"{name}/raw_diag"] = raw_diag
        saved[f"{name}/leak"] = leak
        saved[f"{name}/leak_where"] = leak_where

    if args.with_boost_check:
        print("\n# same measurement with boost scale=5, mmax=1, jmax=1 enabled")
        model.set_lambda_preconditioner_boost(scale=5.0, mmax=1, jmax=1)
        model.evaluate(2, 2, precondition=True)  # preconditioner re-assembly
        for name, span, m, n in families:
            if m > 1 or name.startswith("R "):
                continue
            block, raw_diag, leak, leak_where = family_block(
                model, shape, span, m, n, js_list
            )
            analyze_family(
                f"{name} [boosted]", block, raw_diag, leak, leak_where, js_list
            )
            saved[f"{name}/block_boosted"] = block
        model.set_lambda_preconditioner_boost(scale=1.0)

    if args.out:
        np.savez(args.out, **saved)  # pyright: ignore[reportArgumentType]
        print(f"\n# raw blocks saved to {args.out}")


if __name__ == "__main__":
    main()

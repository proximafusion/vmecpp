"""How do the solve dynamics change if lambda is solved exactly at each step?

Motivated by Finding 14 (docs/convergence_study.md): the tail of strongly
shaped configurations is rate-limited by an axis-localized cluster of low-m
lambda modes whose softness comes from coupling (radial half-grid-averaging
chain within each (m,n) family, angular cross-(m,n) coupling on the innermost
surfaces) that the diagonal faclam preconditioner cannot represent. The
natural next lever is to stop RELAXING lambda and start SOLVING it: at fixed
R, Z the MHD energy is exactly quadratic in lambda, so the inner lambda
problem is one linear system -- per flux surface if the surface blocks
dominate, block-tridiagonal in radius (PR #616 infrastructure) if the
averaging chain matters. Before implementing that in C++, this study measures
two things at a converged-tail snapshot:

1. --dynamics: the "exact lambda" dynamics preview. Solve the lambda
   subspace to high precision by preconditioned CG (matvecs =
   hessian_vector_product embedded/restricted to the lambda spans; the
   preconditioner is the native one via apply_preconditioner), then continue
   the standard iteration and compare the tail against the unmodified
   baseline from the identical snapshot. If the tail rate is unchanged, the
   slow subspace re-enters through R/Z coupling and a lambda-only solve
   cannot beat the coupled cluster; if the tail collapses, an in-loop exact
   lambda solve (as a preconditioner modification) is worth the C++ work.
   Also reports the CG cost (force evaluations) of one exact solve, i.e. the
   per-step price of the naive approach.

2. --blocks: per-surface lambda Hessian blocks, WITH raw (unpreconditioned)
   columns, at a few radii: the surface-local block S_js over all (span, m,
   n) lambda coordinates of one surface, plus the radial coupling to js +/-
   1. Offline these let us simulate candidate preconditioners exactly on
   measured data: current diagonal, per-surface dense inverse (the "linear
   system for each flux surface"), per-family radial tridiagonal, and the
   full block-tridiagonal, and compare their spectra before writing any C++.

Protocol as in examples/lambda_j1_stiffness_study.py: converge the coarse
stage, refine (cubic) to ns=99, run delt_recovery to the snapshot residual,
then probe from the frozen snapshot.
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
FTOL_FINAL = 1.0e-11


class _StopEarly(Exception):
    """Raised from a callback to end solve_equilibrium at the snapshot point."""


def make_model(path):
    vi = vmecpp.VmecInput.from_file(path)
    cpp = vi._to_cpp_vmecindata()
    cpp.ns_array = np.array([NS_OLD, NS_NEW], dtype=np.int64)
    cpp.ftol_array = np.array([1.0e-9, FTOL_FINAL])
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


def state_shape(model):
    mp, nt1 = model.mpol, model.ntor + 1
    n_spans = (2 if model.lthreed else 1) * 3
    x = np.array(model.get_state())
    return (n_spans, x.size // (mp * nt1 * n_spans), mp, nt1)


def lambda_mask(shape, lthreed):
    """Boolean mask (flat) selecting the lambda spans of the state vector."""
    mask = np.zeros(shape, dtype=bool)
    if lthreed:
        mask[4] = True
        mask[5] = True
    else:
        mask[2] = True
    return mask.reshape(-1)


def restore(model, seed):
    model.set_state(seed)
    model.save_backup()
    model.zero_velocity()
    model.evaluate(2, 2, precondition=True)


def lambda_cg_solve(model, mask, tol, max_iter, zero_velocity=True):
    """Preconditioned CG on the lambda subspace at fixed R, Z.

    Minimizes the (exactly quadratic) energy over the lambda coordinates: solve H_ll dx
    = f_l with H_ll v = restrict(HVP(embed(v))) and the native preconditioner as the CG
    metric. Each matvec costs 2 force evaluations. Updates the model state in place;
    returns (n_matvecs, residual_history).
    """
    size = mask.size

    # Sign conventions (measured, Finding 14 probes): the raw force response
    # A = dF/dx restricted to the lambda block is symmetric with POSITIVE
    # diagonal, and the native preconditioner approximates -A^{-1} (its
    # lambda entries are negative). The Newton displacement that zeroes the
    # lambda force is delta = -A^{-1} F_l, so CG solves A y = F_l with -M^{-1}
    # as the SPD metric, and the state update subtracts y.
    def matvec(v_l):
        e = np.zeros(size)
        e[mask] = v_l
        return np.asarray(model.hessian_vector_product(e))[mask]

    def precon(r_l):
        e = np.zeros(size)
        e[mask] = r_l
        return -np.asarray(model.apply_preconditioner(e))[mask]

    x0 = np.array(model.get_state())
    model.evaluate(2, 2, precondition=False)
    r = np.asarray(model.get_forces())[mask]  # residual = force = -grad
    hist = [float(np.linalg.norm(r))]
    z = precon(r)
    p = z.copy()
    rz = float(r @ z)
    dx = np.zeros_like(r)
    n_mv = 0
    for _ in range(max_iter):
        hp = matvec(p)
        n_mv += 1
        alpha = rz / float(p @ hp)
        dx += alpha * p
        r = r - alpha * hp
        hist.append(float(np.linalg.norm(r)))
        if hist[-1] < tol * hist[0]:
            break
        z = precon(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    x1 = x0.copy()
    x1[mask] -= dx  # delta = -A^{-1} F_l (see sign note above)
    model.set_state(x1)
    model.save_backup()
    if zero_velocity:
        # cold restart of the momentum; the periodic mode keeps the R/Z
        # momentum alive across solves instead
        model.zero_velocity()
    model.evaluate(2, 2, precondition=True)
    return n_mv, hist


def run_tail(model, label):
    hist = []
    r = solve_equilibrium(
        model, style="delt_recovery", delt_start_fraction=1.0, callback=hist.append
    )
    print(
        f"  {label}: iters={r.num_iterations} restarts={r.restarts} "
        f"converged={r.converged} fsq={r.fsqr + r.fsqz + r.fsql:.2e} "
        f"energy={model.mhd_energy:.10e}"
    )
    return r


def run_tail_with_periodic_solves(model, mask, every, cg_iters, cg_tol, label):
    """Standard tail interleaved with a lambda CG solve every `every` outer iterations.

    Total cost is reported in force evaluations (1 per outer iteration + 2 per CG
    matvec) for a fair comparison.
    """
    total_mv = 0
    n_solves = 0
    done = {"flag": False}

    class _Stop(Exception):
        pass

    def cb(s):
        if s.iteration > 0 and s.iteration % every == 0:
            raise _Stop

    # run in segments of `every` outer iterations, solving lambda in between
    total_outer = 0
    r = None
    while True:
        hist = []

        def segment_cb(s, hist=hist):
            hist.append(s)
            cb(s)

        with contextlib.suppress(_Stop):
            r = solve_equilibrium(
                model,
                style="delt_recovery",
                delt_start_fraction=1.0,
                callback=segment_cb,
            )
            done["flag"] = True
        total_outer += len(hist)
        if done["flag"]:
            assert r is not None
            print(
                f"  {label}: outer={total_outer} solves={n_solves} "
                f"matvecs={total_mv} eval-equivalent="
                f"{total_outer + 2 * total_mv} converged={r.converged} "
                f"fsq={r.fsqr + r.fsqz + r.fsql:.2e} "
                f"energy={model.mhd_energy:.10e}"
            )
            return
        if total_outer > 30000:
            print(f"  {label}: no convergence within 30000 outer iterations")
            return
        n_mv, _ = lambda_cg_solve(
            model, mask, tol=cg_tol, max_iter=cg_iters, zero_velocity=False
        )
        total_mv += n_mv
        n_solves += 1


def dynamics_experiment(model, snap, mask, args):
    print("# dynamics: baseline tail from the snapshot")
    restore(model, snap["state"])
    run_tail(model, "baseline")

    print("# dynamics: exact-lambda tail (lambda CG solve, then standard tail)")
    restore(model, snap["state"])
    fsq0 = model.fsqr + model.fsqz + model.fsql
    fsql0 = model.fsql
    n_mv, hist = lambda_cg_solve(model, mask, tol=args.cg_tol, max_iter=args.cg_iters)
    print(
        f"  lambda CG: {n_mv} matvecs ({2 * n_mv} force evals), "
        f"|r_l| {hist[0]:.3e} -> {hist[-1]:.3e} "
        f"(fsq {fsq0:.2e}, fsql {fsql0:.2e} before)"
    )
    print(
        f"  after solve: fsq={model.fsqr + model.fsqz + model.fsql:.2e} "
        f"fsqr={model.fsqr:.2e} fsqz={model.fsqz:.2e} fsql={model.fsql:.2e}"
    )
    run_tail(model, "post-exact-lambda")

    for every, mv in ((50, 40), (200, 100)):
        print(f"# dynamics: periodic lambda solves (every {every}, cg<={mv})")
        restore(model, snap["state"])
        run_tail_with_periodic_solves(
            model, mask, every, mv, 1.0e-3, f"periodic-{every}"
        )


def blocks_experiment(model, snap, args):
    shape = state_shape(model)
    _, ns, mp, nt1 = shape
    lam_spans = (4, 5) if model.lthreed else (2,)
    js_probes = sorted(set(args.block_js))
    restore(model, snap["state"])
    size = int(np.prod(shape))
    coords = [(sp, m, n) for sp in lam_spans for m in range(mp) for n in range(nt1)]
    saved = {"js_probes": np.array(js_probes)}
    for js in js_probes:
        ncl = len(coords)
        raw_self = np.zeros((ncl, ncl))
        raw_dn = np.zeros((ncl, ncl))  # response at js-1
        raw_up = np.zeros((ncl, ncl))  # response at js+1
        pre_self = np.zeros((ncl, ncl))
        for k, (sp, m, n) in enumerate(coords):
            e = np.zeros(shape)
            e[sp, js, m, n] = 1.0
            h = np.asarray(model.hessian_vector_product(e.reshape(size)))
            p4 = np.asarray(model.apply_preconditioner(h)).reshape(shape)
            h4 = h.reshape(shape)
            raw_self[:, k] = [h4[sp2, js, m2, n2] for sp2, m2, n2 in coords]
            if js > 0:
                raw_dn[:, k] = [h4[sp2, js - 1, m2, n2] for sp2, m2, n2 in coords]
            if js < ns - 1:
                raw_up[:, k] = [h4[sp2, js + 1, m2, n2] for sp2, m2, n2 in coords]
            pre_self[:, k] = [p4[sp2, js, m2, n2] for sp2, m2, n2 in coords]
        saved[f"js{js}/raw_self"] = raw_self
        saved[f"js{js}/raw_dn"] = raw_dn
        saved[f"js{js}/raw_up"] = raw_up
        saved[f"js{js}/pre_self"] = pre_self
        d = np.diag(raw_self)
        act = np.abs(d) > 1e-12 * np.max(np.abs(d))
        s_act = raw_self[np.ix_(act, act)]
        off = s_act - np.diag(np.diag(s_act))
        print(
            f"# js={js}: {act.sum()} active lambda coords, "
            f"max|offdiag|/|diag| (surface) = "
            f"{np.max(np.abs(off)) / np.max(np.abs(np.diag(s_act))):.3f}, "
            f"max|coupling js+1|/|diag| = "
            f"{np.max(np.abs(raw_up[np.ix_(act, act)])) / np.max(np.abs(np.diag(s_act))):.3f}"
        )
    if args.out:
        np.savez(args.out, **saved)  # pyright: ignore[reportArgumentType]
        print(f"# per-surface blocks saved to {args.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="w7x", choices=sorted(CASES))
    ap.add_argument("--snapshot-fsq", type=float, default=1.0e-6)
    ap.add_argument("--dynamics", action="store_true")
    ap.add_argument("--cg-tol", type=float, default=1.0e-6)
    ap.add_argument("--cg-iters", type=int, default=400)
    ap.add_argument("--blocks", action="store_true")
    ap.add_argument("--block-js", type=int, nargs="+", default=[1, 2, 3, 50, 97])
    ap.add_argument("--out", default="", help="save per-surface blocks to this .npz")
    args = ap.parse_args()

    model = make_model(CASES[args.case])
    snap = tail_snapshot(model, args.snapshot_fsq)
    print(f"# {args.case}: snapshot at fine-stage iteration {snap['iteration']}")
    shape = state_shape(model)
    mask = lambda_mask(shape, model.lthreed)

    if args.dynamics:
        dynamics_experiment(model, snap, mask, args)
    if args.blocks:
        blocks_experiment(model, snap, args)


if __name__ == "__main__":
    main()

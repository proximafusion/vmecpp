"""Anderson acceleration of the VMEC++ tail-convergence regime.

The tail of the force iteration (invariant residual fsq from ~1e-6 down to
1e-11) is rate-limited by the width of the preconditioned-Hessian spectrum;
the slow subspace is low-dimensional (docs/convergence_study.md Finding 10:
mostly low-m lambda modes plus a few highest-m geometry modes). Anderson
acceleration (AA) builds a low-rank secant model of exactly that subspace, so
it should collapse the tail without touching the C++ core.

Method: type-II AA(m) on a fixed-point map built from the existing solver
step, two candidate maps:

  (a) richardson: G(x) = x + eta * P F(x). One force evaluation per outer
      iteration; get_forces() after evaluate(precondition=True) IS the
      preconditioned direction P F(x). eta starts at delt_snap^2 / 2 (the
      first-order stability limit implied by the second-order scheme's
      delt_crit) and is halved whenever the plain map proves unstable during
      a measured warmup.
  (b) native_k: G(x) = K native damped Garabedian steps (evaluate +
      perform_time_step with the reference damping recipe from
      vmecpp._iteration.solve_equilibrium) at the snapshot time step,
      velocity zeroed at each block entry so G is a function of x alone.

Safeguards: the accelerated iterate is evaluated before being trusted; if its
invariant residual exceeds REJECT_FACTOR x the last accepted one (or is
non-finite), the candidate is rejected, the iteration falls back to the plain
step from the last good state, and the AA history is cleared. The AA update
can be damped (AA_BETA). The AA history is also cleared whenever the radial
preconditioner refreshes (every 25 force iterations), because P -- and with
it the map G -- changes discontinuously there.

Benchmark protocol (mirrors examples/delt_tail_study.py): converge the
ns=25 coarse stage, refine (cubic) to ns=99, run the delt_recovery control
until fsq first drops below 1e-6, snapshot the state; then from that
identical snapshot measure iterations and force evaluations to reach
fsq < 1e-11 for (i) the baseline delt_recovery continuation and (ii) AA.

Findings (measured on cth [25, 99], snapshot at fsq = 1e-6, target 1e-11;
baseline delt_recovery continuation: 441 force evals):

  - The plain Richardson map is stable and smooth at eta = delt_snap^2/2
    (no eta halving ever triggered) and contracts monotonically, but slowly:
    fsq rate ~ 0.9984/iter, i.e. an effective condition number ~ 300 of the
    *preconditioned* operator. Consistent with the native momentum tail
    (fsq rate 0.974 ~ the sqrt speedup of second-order damping).
  - Type-II AA does NOT beat the native momentum iteration on this map.
    Every configuration measured at least 3x worse than baseline (cth,
    force evals to fsq < 1e-11 from the identical snapshot, baseline 441):
    m=8, rcond=1e-10, refresh+clear every 25, warmup 10: 1359 (best);
    m=8, Tikhonov reg=1e-8, warmup 50, refresh every 25: 1678;
    m=8, rcond=1e-5 + gamma cap, frozen P, refresh every 100: 3409;
    m=60, frozen P, rcond=1e-6: 4000-eval cap at fsq 4.6e-9 (raising the
    depth does NOT help); keep-history-across-refresh: cap at 2.0e-8.
    On w7x the m=8/rcond=1e-5 configuration hit the 4000-eval cap at
    fsq 7.5e-11 vs baseline 1590 evals.
  - Why: the LS diagnostics show ls_rel ~ 0.99 during AA at every depth --
    the residual is never in the span of the history differences. The tail
    spectrum acts as a near-continuum (radial families of slow modes), not
    a low-dimensional slow subspace, so cancelling it requires extrapolation
    coefficients gamma >= 1/(eta*mu_min) ~ 1e3; at that amplification the
    secant model is destroyed by the map's nonlinearity, and truncating the
    LS (rcond/gamma cap) reduces AA to a small perturbation of the plain
    map. Finding 10's "low-dimensional" tail refers to family/m shares of
    the residual, not to the eigenvalue count of the preconditioned Hessian.
  - Map (b) (composite of K=5 native damped steps, velocity zeroed at each
    block entry for determinism) loses the momentum ramp every block
    (kNDamp = 10 damping history) and underperforms both baseline and
    map (a): capped at 4000 evals, fsq 6.5e-11. The native velocity cannot
    be read back through the pybind API, so the composite cannot preserve
    momentum across blocks.
  - Conclusion: the velocity-free preconditioned-Richardson fixed-point map
    is not a productive substrate for Anderson acceleration of the VMEC++
    tail; the native second-order damping is already a near-optimal
    polynomial accelerator for this spectrum. A win would need either a
    spectrally deflated map (an actual low-rank slow subspace, e.g. from
    the block-tridiagonal preconditioner) or AA on the full (x, v) phase
    space, which requires exposing the velocity in the pybind API.
"""

import argparse
import contextlib
import math
from collections import deque

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
SNAPSHOT_FSQ = 1.0e-6  # snapshot the state at the first crossing of this
TARGET_FSQ = 1.0e-11  # tail target: fsqr + fsqz + fsql below this
MAX_ITERS = 4000  # cap on outer iterations per run
MAX_EVALS = 4000  # cap on force evaluations per run
COARSE_NITER = 6000
TAIL_NITER = 8000

# --- Anderson parameters ---
AA_DEPTH = 8  # history depth m
AA_BETA = 1.0  # mixing damping on the AA update
AA_RCOND = 1.0e-5  # SVD cutoff of the LS problem (np.linalg.lstsq rcond):
#                    singular directions of DR below rcond x the largest one
#                    are noise (measured: keeping them yields gamma ~ 1e7
#                    cancelling coefficients and destroys the secant model)
AA_GAMMA_MAX = 1.0e4  # cap on the mixing coefficients: the slow-mode
#                       extrapolation length is ~ 1/(eta*mu_min) ~ 1e3; any
#                       gamma far beyond that is noise amplification
AA_WARMUP = 10  # plain-map iterations before arming AA (also probes
#                 the plain map's stability for eta tuning)
REJECT_FACTOR = 10.0  # reject AA iterate if fsq grows past this x last good
PLAIN_BLOWUP = 100.0  # plain-map blowup leash (halves eta, map (a) only)
PRECOND_INTERVAL = 25  # FlowControl::kPreconditionerUpdateInterval
NATIVE_K = 5  # native steps per composite map (b) evaluation
NDAMP = 10  # Vmec::kNDamp


class _StopEarly(Exception):
    """Raised from a callback to end solve_equilibrium at a snapshot/target."""


def make_indata(path: str):
    vi = vmecpp.VmecInput.from_file(path)
    cpp = vi._to_cpp_vmecindata()
    cpp.ns_array = np.array([NS_OLD, NS_NEW], dtype=np.int64)
    # fine-stage ftol below the tail target so the reference loop cannot
    # converge-exit before the 1e-11 crossing is recorded by the callback.
    cpp.ftol_array = np.array([1.0e-9, 5.0e-13])
    cpp.niter_array = np.array([COARSE_NITER, TAIL_NITER], dtype=np.int64)
    return cpp


def collect_snapshot(model):
    """Run the fine stage (delt_recovery control) until the invariant residual first
    crosses SNAPSHOT_FSQ; return the state and the working time step."""
    snap = {}

    def cb(s):
        if s.fsq_invariant < SNAPSHOT_FSQ:
            snap["state"] = np.array(model.get_state())
            snap["iteration"] = s.iteration
            snap["delt"] = s.delt
            snap["fsq"] = s.fsq_invariant
            raise _StopEarly

    with contextlib.suppress(_StopEarly):
        solve_equilibrium(
            model, style="delt_recovery", delt_start_fraction=0.5, callback=cb
        )
    return snap or None


def reset_to(model, seed):
    model.set_state(seed)
    model.zero_velocity()
    model.save_backup()


def run_baseline(model, seed, delt_fraction):
    """Continue the native delt_recovery control from the snapshot until the invariant
    residual crosses TARGET_FSQ (or the iteration cap)."""
    reset_to(model, seed)
    model.reset_force_eval_count()
    hist = []

    def cb(s):
        hist.append(s.fsq_invariant)
        if s.fsq_invariant < TARGET_FSQ or len(hist) >= MAX_ITERS:
            raise _StopEarly

    with contextlib.suppress(_StopEarly):
        solve_equilibrium(
            model,
            style="delt_recovery",
            delt_start_fraction=delt_fraction,
            callback=cb,
        )
    fsq = hist[-1] if hist else float("nan")
    return {
        "iterations": len(hist),
        "force_evals": int(model.force_eval_count),
        "fsq": fsq,
        "rejects": 0,
        "reached": bool(hist) and fsq < TARGET_FSQ,
    }


class RichardsonMap:
    """Map (a): G(x) = x + eta * P F(x); one force evaluation per call.

    The preconditioner-refresh schedule keys on the (iter1, iter2) counters
    passed to evaluate ((iter2 - iter1) % 25 == 0), and iter2 == iter1
    additionally re-anchors the spectral-condensation reference geometry
    (rzConIntoVolume). Both change the map G discontinuously, which prevents
    the AA secant model from ever settling if they fire on the native
    25-iteration cadence. This map therefore freezes G between deliberate
    refreshes: the first call passes (1, 1) (anchor + refresh at the seed),
    every refresh_every-th call passes (1, 76) (refresh only), and all other
    calls cycle iter2 through 52..74 so no refresh triggers. The offset past
    50 keeps includeEdgeRZForces off, so the invariant-residual measure
    matches the baseline tail (where iter2 - iter1 is large).

    Assumes the model state is already x. Leaves the model state at x.
    Returns (g, fsq_at_x, preconditioner_refreshed).
    """

    cost = 1

    def __init__(self, model, eta, refresh_every):
        self.model = model
        self.eta = eta
        self.refresh_every = refresh_every
        self.it = 0

    def __call__(self, x):
        self.it += 1
        if self.it == 1:
            iter1, iter2 = 1, 1
            refreshed = True
        elif self.refresh_every > 0 and (self.it - 1) % self.refresh_every == 0:
            iter1, iter2 = 1, 1 + 3 * PRECOND_INTERVAL
            refreshed = True
        else:
            iter1 = 1
            iter2 = 2 + 2 * PRECOND_INTERVAL + ((self.it - 1) % (PRECOND_INTERVAL - 2))
            refreshed = False
        self.model.evaluate(iter1, iter2)
        fsq = self.model.fsqr + self.model.fsqz + self.model.fsql
        if not math.isfinite(fsq):
            return x, fsq, refreshed
        f = np.array(self.model.get_forces())
        return x + self.eta * f, fsq, refreshed


class NativeKMap:
    """Map (b): G(x) = K native damped Garabedian steps at a fixed time step.

    The damping recipe (inv_tau history -> otav -> dtau) is copied from
    vmecpp._iteration.solve_equilibrium; the velocity is zeroed at each block entry so
    the composite is a function of the state alone. Assumes the model state is already
    x; leaves the model state at G(x). Returns (g, fsq_at_x,
    preconditioner_refreshed_during_block).
    """

    def __init__(self, model, delt0r, k):
        self.model = model
        self.delt0r = delt0r
        self.k = k
        self.cost = k
        self.it = 0
        self.inv_tau = np.full(NDAMP, 0.15 / delt0r)
        self.fsq_prev = 1.0

    def __call__(self, x):
        m = self.model
        m.zero_velocity()
        fsq_at_x = float("nan")
        refreshed = False
        for j in range(self.k):
            self.it += 1
            refreshed = refreshed or (self.it - 1) % PRECOND_INTERVAL == 0
            m.evaluate(1, self.it)
            fsq = m.fsqr + m.fsqz + m.fsql
            if j == 0:
                fsq_at_x = fsq
            if not math.isfinite(fsq):
                return x, float("nan"), refreshed
            fsq1 = m.fsqr1 + m.fsqz1 + m.fsql1
            self.inv_tau[:-1] = self.inv_tau[1:]
            if self.it > 1 and fsq1 != 0.0:
                self.inv_tau[-1] = (
                    min(abs(math.log(fsq1 / self.fsq_prev)), 0.15) / self.delt0r
                )
            self.fsq_prev = fsq1
            otav = float(self.inv_tau.mean())
            dtau = self.delt0r * otav / 2.0
            m.perform_time_step(1.0 / (1.0 + dtau), 1.0 - dtau, self.delt0r)
        return np.array(m.get_state()), fsq_at_x, refreshed


def anderson_gamma(dr_mat, r, rcond, reg=0.0):
    """Type-II Anderson least squares gamma = argmin ||r - DR gamma||.

    reg == 0: solved by SVD (np.linalg.lstsq) with relative cutoff rcond.
    reg > 0: relative Tikhonov regularization on the normal equations (the
    variant of the first benchmark cycle; strong shrinkage of the mixing
    coefficients, which measured best among the AA configurations on cth).
    """
    if reg > 0.0:
        a = dr_mat.T @ dr_mat
        mk = a.shape[0]
        a[np.diag_indices(mk)] += reg * np.trace(a) / mk + 1e-300
        try:
            return np.linalg.solve(a, dr_mat.T @ r)
        except np.linalg.LinAlgError:
            return np.zeros(mk)
    gamma, _, _, _ = np.linalg.lstsq(dr_mat, r, rcond=rcond)
    return gamma


def run_anderson(
    model,
    seed,
    map_factory,
    *,
    depth=AA_DEPTH,
    beta=AA_BETA,
    rcond=AA_RCOND,
    reg=0.0,
    warmup=AA_WARMUP,
    tune_eta=False,
    clear_on_refresh=True,
    verbose=False,
):
    """Type-II Anderson acceleration on a fixed-point map from the snapshot.

    map_factory(scale) builds a fresh map object at a scaled eta (so eta retuning can
    restart the run cleanly). Safeguards: candidates whose residual is non-finite or
    more than REJECT_FACTOR above the last accepted residual are rejected in favor of
    the plain step from the last good state, and the history is cleared; a plain-map
    blowup (map (a) only, PLAIN_BLOWUP leash) halves eta and restarts from the seed.
    """
    model.reset_force_eval_count()
    eta_scale = 1.0
    g_map = map_factory(eta_scale)
    x = seed.copy()
    reset_to(model, x)
    d_x: deque = deque(maxlen=depth)
    d_r: deque = deque(maxlen=depth)
    prev_x = prev_r = None
    good = None  # (x, g, fsq) of the last accepted iterate
    last_was_aa = False
    rejects = 0
    it = 0
    fsq = float("nan")
    gamma_inf = ls_rel = float("nan")
    trace = []

    def clear_history():
        nonlocal prev_x, prev_r
        d_x.clear()
        d_r.clear()
        prev_x = prev_r = None

    while it < MAX_ITERS and model.force_eval_count < MAX_EVALS:
        it += 1
        g, fsq, refreshed = g_map(x)
        trace.append(fsq)

        if not math.isfinite(fsq):
            if good is None:
                break  # the seed itself is bad: give up
            rejects += 1
            clear_history()
            if last_was_aa or not tune_eta:
                x = good[1]  # plain step from the last good state
            else:
                # the plain map itself diverged: halve eta, restart from seed
                eta_scale *= 0.5
                g_map = map_factory(eta_scale)
                good = None
                x = seed.copy()
            last_was_aa = False
            reset_to(model, x)
            continue

        if fsq < TARGET_FSQ:
            break

        if good is not None and fsq > REJECT_FACTOR * good[2]:
            if last_was_aa:
                # reject the accelerated iterate: plain step from last good
                rejects += 1
                clear_history()
                x = good[1]
                last_was_aa = False
                reset_to(model, x)
                continue
            if tune_eta and fsq > PLAIN_BLOWUP * good[2]:
                # the plain map is unstable at this eta: halve and restart
                rejects += 1
                clear_history()
                eta_scale *= 0.5
                g_map = map_factory(eta_scale)
                good = None
                x = seed.copy()
                last_was_aa = False
                reset_to(model, x)
                continue

        # accept the iterate
        r = g - x
        if refreshed and clear_on_refresh:
            # P (and with it G) changed discontinuously: drop stale columns
            clear_history()
        if prev_x is not None:
            d_x.append(x - prev_x)
            d_r.append(r - prev_r)
        prev_x, prev_r = x, r
        good = (x, g, fsq)
        model.save_backup()

        take_plain = it <= warmup or not d_r
        x = g
        last_was_aa = False
        if d_r:
            # LS model quality diagnostic (computed on plain steps too): how
            # much of the current residual the history differences explain.
            dx_mat = np.column_stack(d_x)
            dr_mat = np.column_stack(d_r)
            gamma = anderson_gamma(dr_mat, r, rcond, reg)
            gamma_inf = float(np.max(np.abs(gamma)))
            if gamma_inf > AA_GAMMA_MAX:
                gamma *= AA_GAMMA_MAX / gamma_inf
            ls_rel = float(
                np.linalg.norm(r - dr_mat @ gamma) / (np.linalg.norm(r) or 1.0)
            )
            if not take_plain:
                x = prev_x + beta * r - (dx_mat + beta * dr_mat) @ gamma
                last_was_aa = True

        if verbose and it % 25 == 0:
            print(
                f"#   aa it={it:5d} fsq={fsq:.3e} rejects={rejects} "
                f"gamma_inf={gamma_inf:.1e} ls_rel={ls_rel:.2e}",
                flush=True,
            )
        model.set_state(x)

    return {
        "iterations": it,
        "force_evals": int(model.force_eval_count),
        "fsq": fsq,
        "rejects": rejects,
        "reached": math.isfinite(fsq) and fsq < TARGET_FSQ,
        "trace": trace,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="cth,cma,w7x")
    ap.add_argument("--maps", default="richardson,native_k")
    ap.add_argument("--depth", type=int, default=AA_DEPTH)
    ap.add_argument("--beta", type=float, default=AA_BETA)
    ap.add_argument("--rcond", type=float, default=AA_RCOND)
    ap.add_argument("--reg", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=AA_WARMUP)
    ap.add_argument("--eta-scale", type=float, default=1.0)
    ap.add_argument("--refresh-every", type=int, default=100)
    ap.add_argument("--keep-history-on-refresh", action="store_true")
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    clear_on_refresh = not args.keep_history_on_refresh

    print("case,method,iterations,force_evals,final_fsq,rejects,reached")
    for name in args.cases.split(","):
        path = CASES[name]
        cpp = make_indata(path)
        model = _vmecpp.VmecModel.create(cpp, NS_OLD)
        r1 = solve_equilibrium(model)
        if not r1.converged:
            print(f"# {name}: coarse stage did not converge, skipping")
            continue
        model.refine_to(
            NS_NEW, interpolation=_vmecpp.MultigridInterpolationScheme.CUBIC
        )
        snap = collect_snapshot(model)
        if snap is None:
            print(f"# {name}: snapshot threshold not crossed, skipping")
            continue
        delt_user = model.delt
        print(
            f"# {name}: snapshot at iter {snap['iteration']} "
            f"fsq={snap['fsq']:.2e} delt={snap['delt']:.3f} "
            f"(user delt {delt_user:.3f})",
            flush=True,
        )
        seed = snap["state"]

        if not args.skip_baseline:
            frac = min(1.0, snap["delt"] / delt_user)
            res = run_baseline(model, seed, frac)
            print(
                f"{name},baseline,{res['iterations']},{res['force_evals']},"
                f"{res['fsq']:.3e},{res['rejects']},{int(res['reached'])}",
                flush=True,
            )

        for map_name in args.maps.split(","):
            if map_name == "richardson":
                eta0 = args.eta_scale * snap["delt"] ** 2 / 2.0

                def richardson_factory(
                    scale=1.0, _m=model, _e=eta0, _r=args.refresh_every
                ):
                    return RichardsonMap(_m, _e * scale, _r)

                res = run_anderson(
                    model,
                    seed,
                    richardson_factory,
                    depth=args.depth,
                    beta=args.beta,
                    rcond=args.rcond,
                    reg=args.reg,
                    warmup=args.warmup,
                    tune_eta=True,
                    clear_on_refresh=clear_on_refresh,
                    verbose=args.verbose,
                )
                tag = f"aa_richardson(m={args.depth},eta={eta0:.3f})"
            elif map_name == "native_k":

                def native_k_factory(_scale=1.0, _m=model, _d=snap["delt"]):
                    return NativeKMap(_m, _d, NATIVE_K)

                res = run_anderson(
                    model,
                    seed,
                    native_k_factory,
                    depth=args.depth,
                    beta=args.beta,
                    rcond=args.rcond,
                    reg=args.reg,
                    warmup=max(2, args.warmup // NATIVE_K),
                    tune_eta=False,
                    clear_on_refresh=clear_on_refresh,
                    verbose=args.verbose,
                )
                tag = f"aa_native_k(m={args.depth},K={NATIVE_K})"
            else:
                print(f"# unknown map {map_name}, skipping")
                continue
            print(
                f"{name},{tag},{res['iterations']},{res['force_evals']},"
                f"{res['fsq']:.3e},{res['rejects']},{int(res['reached'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()

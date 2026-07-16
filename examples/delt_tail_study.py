"""Tail-regime time-step stability and residual anatomy of the fine multigrid stage.

Two questions about the slow late-convergence regime of strongly shaped
equilibria (the corner-filling phase, where the iteration is limited by the
width of the preconditioned-Hessian spectrum):

A. Is the time-step stability ceiling learned during the stage-entry transient
   stale by the time the iteration is in the tail? For each equilibrium,
   converge the coarse stage, refine (cubic), run the fine stage with the
   delt_recovery control, snapshot the state when the invariant residual first
   crosses 1e-6 / 1e-9 / 1e-12, and from each identical snapshot (state reset +
   zeroed velocity) probe the plain 8.52 control at delt fractions up to 3x the
   user delt for a fixed number of iterations. Stable probes report the decay
   rate per iteration (the payoff of a larger step); unstable probes report the
   implied stability boundary delt_crit = 2/sqrt(mu) extracted from the early
   residual growth factor (residual ~ amplitude^2; the undamped Garabedian
   runaway root satisfies rho + 1/rho = delt^2 mu - 2).

C. What is actually slow in the tail? At each snapshot, decompose the force
   residual by family (R / Z / lambda), by poloidal mode number m, and by
   radial region, for both the preconditioned search direction (what the
   momentum iteration actually steps along) and the raw gradient. If the tail
   is dominated by high-m geometry modes at mid/edge radius, the bottleneck is
   the shaping-induced mode coupling; if fsql dominates, it is the lambda /
   spectral-condensation channel and the lever is lambda preconditioning.
"""

import contextlib
import math

import numpy as np

import vmecpp
from vmecpp._iteration import solve_equilibrium
from vmecpp.cpp import _vmecpp  # type: ignore[attr-defined]

CASES = {
    "w7x": "examples/data/w7x.json",
    "cth": "src/vmecpp/cpp/vmecpp/test_data/cth_like_fixed_bdy.json",
    "cma": "src/vmecpp/cpp/vmecpp/test_data/cma.json",
    "li383": "src/vmecpp/cpp/vmecpp/test_data/li383_low_res.json",
}
NS_OLD = 25
NS_NEW = 99
SNAPSHOT_FSQ = (1.0e-6, 1.0e-9, 1.0e-12)
FRACTIONS = (0.4, 0.7, 1.0, 1.5, 2.0, 3.0)
PROBE_ITERS = 200
TAIL_NITER = 8000


class _StopEarly(Exception):
    """Raised from a callback to end solve_equilibrium at a probe/snapshot point."""


def make_indata(path: str):
    vi = vmecpp.VmecInput.from_file(path)
    cpp = vi._to_cpp_vmecindata()
    cpp.ns_array = np.array([NS_OLD, NS_NEW], dtype=np.int64)
    # ftol below the deepest snapshot so the tail run does not converge-exit
    # before the 1e-12 crossing.
    cpp.ftol_array = np.array([1.0e-9, 5.0e-13])
    cpp.niter_array = np.array([6000, TAIL_NITER], dtype=np.int64)
    return cpp


def collect_tail_snapshots(model):
    """Run the fine stage (delt_recovery control) and snapshot the state at the first
    crossing of each SNAPSHOT_FSQ threshold of the invariant residual."""
    snaps = []
    remaining = list(SNAPSHOT_FSQ)

    def cb(s):
        if remaining and s.fsq_invariant < remaining[0]:
            snaps.append(
                {
                    "target": remaining.pop(0),
                    "state": np.array(model.get_state()),
                    "iteration": s.iteration,
                    "delt": s.delt,
                    "ceiling": s.delt_ceiling,
                    "fsq": s.fsq_invariant,
                    "fam_inv": (s.fsqr, s.fsqz, s.fsql),
                    "fam_pre": (s.fsqr1, s.fsqz1, s.fsql1),
                }
            )
        if not remaining:
            raise _StopEarly

    with contextlib.suppress(_StopEarly):
        solve_equilibrium(
            model, style="delt_recovery", delt_start_fraction=0.5, callback=cb
        )
    return snaps


def probe(model, seed, frac):
    """PROBE_ITERS iterations of plain 8.52 at delt = frac * delt_user from the
    identical seed; returns stability, decay rate, and implied delt_crit data."""
    model.set_state(seed)
    model.save_backup()
    model.zero_velocity()
    history = []

    def cb(s):
        history.append(s)
        if len(history) >= PROBE_ITERS:
            raise _StopEarly

    converged = False
    try:
        r = solve_equilibrium(
            model, style="vmec_8_52", delt_start_fraction=frac, callback=cb
        )
        converged = r.converged
    except _StopEarly:
        pass
    if not history:
        return None
    fsq = np.array([s.fsq_invariant for s in history])
    fsq1 = np.array([s.fsq_preconditioned for s in history])
    restarted = np.array([s.restarted for s in history], dtype=bool)
    restarts = history[-1].n_restarts
    reguesses = history[-1].n_reguesses
    n = len(history)
    stable = restarts == 0 and np.isfinite(fsq[-1]) and fsq[-1] < fsq[0]
    rate = float("nan")
    if n > 1 and np.isfinite(fsq[-1]) and fsq[0] > 0 and fsq[-1] > 0:
        rate = (fsq[-1] / fsq[0]) ** (1.0 / (n - 1))

    # early preconditioned-residual growth factor, measured up to first revert
    first_restart = int(np.argmax(restarted)) if restarted.any() else fsq1.size
    k = min(first_restart, 8)
    growth = float("nan")
    if k >= 3 and np.isfinite(fsq1[1:k]).all() and (fsq1[1:k] > 0).all():
        growth = (fsq1[k - 1] / fsq1[1]) ** (1.0 / (k - 2))
    return {
        "restarts": restarts,
        "reguesses": reguesses,
        "converged": converged,
        "stable": stable,
        "fsq1_start": fsq1[0],
        "fsq1_peak": fsq1.max(),
        "fsq1_end": fsq1[-1],
        "rate": rate,
        "growth": growth,
        "iters": n,
    }


def implied_delt_crit(growth: float, delt: float) -> float:
    if not (math.isfinite(growth) and growth > 1.0):
        return float("nan")
    rho = math.sqrt(growth)
    mu = (rho + 1.0 / rho + 2.0) / delt**2
    return 2.0 / math.sqrt(mu)


def decompose_forces(model, precondition: bool):
    """Per-family / per-m / per-radial-third squared-force decomposition of the current
    state's force residual (preconditioned search direction when precondition=True, raw
    gradient when False).

    The flat force vector carries a different internal normalization per family (lambda
    is lamscale-scaled), so cross-family shares are taken from the fsq* members; the
    span decomposition is only used within a family, where the constant normalization
    cancels.
    """
    model.evaluate(1, 2, precondition)
    f = np.array(model.get_forces())
    fam_fsq = (
        {"R": model.fsqr1, "Z": model.fsqz1, "L": model.fsql1}
        if precondition
        else {"R": model.fsqr, "Z": model.fsqz, "L": model.fsql}
    )
    mpol, ntor = model.mpol, model.ntor
    mn = mpol * (ntor + 1)
    n_spans = (2 if model.lthreed else 1) * 3  # symmetric cases only
    assert not model.lasym, "decomposition below assumes lasym=False"
    assert f.size % n_spans == 0
    per = f.size // n_spans
    ns_span = per // mn
    assert ns_span * mn == per
    spans = f.reshape(n_spans, ns_span, mpol, ntor + 1)
    per_parity = 2 if model.lthreed else 1
    out = {}
    for i, fam in enumerate(("R", "Z", "L")):
        block = spans[i * per_parity : (i + 1) * per_parity]
        sq = np.sum(block**2, axis=0)  # (ns_span, mpol, ntor+1)
        out[fam] = {
            "fsq": fam_fsq[fam],
            "total": float(sq.sum()),
            "per_m": sq.sum(axis=(0, 2)),  # (mpol,)
            "per_mn": sq.sum(axis=0),  # (mpol, ntor+1)
            "radial_thirds": [
                float(sq[: ns_span // 3].sum()),
                float(sq[ns_span // 3 : 2 * ns_span // 3].sum()),
                float(sq[2 * ns_span // 3 :].sum()),
            ],
        }
    return out


def print_decomposition(tag, dec):
    fsq_total = sum(d["fsq"] for d in dec.values()) or 1.0
    fam_share = " ".join(f"{fam}={d['fsq'] / fsq_total:.2%}" for fam, d in dec.items())
    print(f"#   {tag}: family shares {fam_share}")
    for fam, d in dec.items():
        if d["fsq"] / fsq_total < 0.01:
            continue
        m_share = d["per_m"] / (d["total"] or 1.0)
        top_m = np.argsort(m_share)[::-1][:4]
        m_str = " ".join(f"m={m}:{m_share[m]:.1%}" for m in top_m)
        thirds = np.array(d["radial_thirds"]) / (d["total"] or 1.0)
        print(
            f"#     {fam}: {m_str} | radial core/mid/edge "
            f"{thirds[0]:.1%}/{thirds[1]:.1%}/{thirds[2]:.1%}"
        )
        flat = d["per_mn"].flatten()
        top = np.argsort(flat)[::-1][:3]
        ntor1 = d["per_mn"].shape[1]
        mn_str = " ".join(
            f"(m={i // ntor1},n={i % ntor1}):{flat[i] / (d['total'] or 1.0):.1%}"
            for i in top
        )
        print(f"#     {fam} top modes: {mn_str}")


print(
    "case,snap_fsq,tail_iter,tail_delt,tail_ceiling,frac,delt_eff,"
    "restarts,reguesses,stable,converged,fsq1_start,fsq1_peak,fsq1_end,"
    "rate_per_iter,growth,implied_delt_crit"
)
for name, path in CASES.items():
    cpp = make_indata(path)
    model = _vmecpp.VmecModel.create(cpp, NS_OLD)
    r1 = solve_equilibrium(model)
    if not r1.converged:
        print(f"# {name}: coarse stage did not converge, skipping")
        continue
    model.refine_to(NS_NEW, interpolation=_vmecpp.MultigridInterpolationScheme.CUBIC)
    snaps = collect_tail_snapshots(model)
    delt_user = model.delt
    if not snaps:
        print(f"# {name}: no snapshot thresholds crossed in {TAIL_NITER} iters")
        continue
    for snap in snaps:
        fam_i = snap["fam_inv"]
        fam_p = snap["fam_pre"]
        print(
            f"# {name} snapshot fsq<{snap['target']:.0e} at iter "
            f"{snap['iteration']}: delt={snap['delt']:.3f} "
            f"ceiling={snap['ceiling']:.3f} "
            f"fsqr/z/l={fam_i[0]:.1e}/{fam_i[1]:.1e}/{fam_i[2]:.1e} "
            f"fsqr1/z1/l1={fam_p[0]:.1e}/{fam_p[1]:.1e}/{fam_p[2]:.1e}"
        )
        model.set_state(snap["state"])
        print_decomposition(
            "preconditioned", decompose_forces(model, precondition=True)
        )
        model.set_state(snap["state"])
        print_decomposition("raw gradient", decompose_forces(model, precondition=False))
        for frac in FRACTIONS:
            p = probe(model, snap["state"], frac)
            if p is None:
                continue
            crit = implied_delt_crit(p["growth"], frac * delt_user)
            print(
                f"{name},{snap['target']:.0e},{snap['iteration']},"
                f"{snap['delt']:.3f},{snap['ceiling']:.3f},{frac},"
                f"{frac * delt_user:.3f},{p['restarts']},{p['reguesses']},"
                f"{int(p['stable'])},{int(p['converged'])},"
                f"{p['fsq1_start']:.2e},{p['fsq1_peak']:.2e},{p['fsq1_end']:.2e},"
                f"{p['rate']:.5f},{p['growth']:.3f},{crit:.3f}"
            )

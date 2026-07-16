"""Relationship between multigrid ns-jump size and the stable stage-entry delt.

For each (equilibrium, ns_old -> ns_new) transition:
- converge the coarse stage, interpolate (cubic) onto the fine grid,
- probe stage-entry stability for a range of delt fractions: run 120
  iterations of the plain 8.52 control at delt = frac * delt_user from the
  identical seed (state snapshot + zeroed velocity between probes), recording
  restarts and net residual decay,
- for unstable probes, extract the early residual growth factor g per
  iteration; the force residual is quadratic in the runaway mode amplitude,
  so the amplitude growth is rho = sqrt(g). For the (undamped) Garabedian
  stepper the runaway root satisfies rho + 1/rho = delt^2 mu - 2, giving the
  effective preconditioned-Hessian eigenvalue mu and the predicted stability
  boundary delt_crit = 2 / sqrt(mu) -- no autodiff or HVP involved, only the
  residual traces the loop records anyway.
"""

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
# main scan: fixed coarse grid, growing jump ratio
NS_OLD = 25
NS_NEW = (37, 49, 75, 99)
# w7x only: a coarser source grid at the same ratios, to separate the
# jump-ratio dependence from the source-grid-resolution dependence
NS_OLD_B = 13
NS_NEW_B = (19, 25, 37, 49)
FRACTIONS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3)
PROBE_ITERS = 120


def make_indata(path: str, ns_old: int, ns_new: int):
    vi = vmecpp.VmecInput.from_file(path)
    cpp = vi._to_cpp_vmecindata()
    cpp.ns_array = np.array([ns_old, ns_new], dtype=np.int64)
    cpp.ftol_array = np.array([1.0e-9, 1.0e-16])
    cpp.niter_array = np.array([6000, PROBE_ITERS], dtype=np.int64)
    return cpp


def probe(model, seed, frac):
    """120 iterations of plain 8.52 at delt = frac * delt_user from the seed."""
    model.set_state(seed)
    model.save_backup()
    model.zero_velocity()
    history = []
    r = solve_equilibrium(
        model,
        style="vmec_8_52",
        delt_start_fraction=frac,
        callback=history.append,
    )
    fsq1 = np.array([s.fsq_preconditioned for s in history])
    restarted = np.array([s.restarted for s in history], dtype=bool)
    start = fsq1[0] if fsq1.size else float("nan")
    end = fsq1[-1] if fsq1.size else float("nan")
    peak = fsq1.max() if fsq1.size else float("nan")
    stable = r.restarts == 0 and np.isfinite(end) and end < start

    # early growth factor per iteration, measured up to the first revert
    first_restart = int(np.argmax(restarted)) if restarted.any() else fsq1.size
    k = min(first_restart, 8)
    growth = float("nan")
    if k >= 3 and np.isfinite(fsq1[1:k]).all() and (fsq1[1:k] > 0).all():
        growth = (fsq1[k - 1] / fsq1[1]) ** (1.0 / (k - 2))
    return r, start, end, peak, stable, growth


def implied_delt_crit(growth: float, delt: float) -> float:
    """Predicted stability boundary from the measured residual growth factor.

    Residual ~ amplitude^2, so amplitude growth rho = sqrt(g). The undamped
    Garabedian runaway root satisfies rho + 1/rho = delt^2 mu - 2, hence
    mu = (rho + 1/rho + 2) / delt^2 and delt_crit = 2/sqrt(mu).
    """
    if not (math.isfinite(growth) and growth > 1.0):
        return float("nan")
    rho = math.sqrt(growth)
    mu = (rho + 1.0 / rho + 2.0) / delt**2
    return 2.0 / math.sqrt(mu)


def scan(name: str, path: str, ns_old: int, ns_new_list) -> None:
    for ns_new in ns_new_list:
        cpp = make_indata(path, ns_old, ns_new)
        model = _vmecpp.VmecModel.create(cpp, ns_old)
        r1 = solve_equilibrium(model)
        if not r1.converged:
            print(f"# {name} ns_old={ns_old}: coarse stage did not converge")
            continue
        model.refine_to(
            ns_new, interpolation=_vmecpp.MultigridInterpolationScheme.CUBIC
        )
        seed = np.array(model.get_state())
        ratio = ns_new / ns_old
        delt_user = model.delt
        for frac in FRACTIONS:
            r, start, end, peak, stable, growth = probe(model, seed, frac)
            crit = implied_delt_crit(growth, frac * delt_user)
            print(
                f"{name},{ns_old},{ns_new},{ratio:.2f},{frac},"
                f"{r.restarts},{r.axis_reguesses},"
                f"{start:.2e},{peak:.2e},{end:.2e},{int(stable)},"
                f"{growth:.3f},{crit:.3f}"
            )


print(
    "case,ns_old,ns_new,ratio,frac,restarts,reguesses,"
    "fsq_start,fsq_peak,fsq_end,stable,growth,implied_delt_crit"
)
for name, path in CASES.items():
    scan(name, path, NS_OLD, NS_NEW)
scan("w7x", CASES["w7x"], NS_OLD_B, NS_NEW_B)

# self-check: the snapshot-reuse probe must reproduce a fresh full pipeline
cpp = make_indata(CASES["w7x"], NS_OLD, 49)
model = _vmecpp.VmecModel.create(cpp, NS_OLD)
solve_equilibrium(model)
model.refine_to(49, interpolation=_vmecpp.MultigridInterpolationScheme.CUBIC)
fresh = solve_equilibrium(model, style="vmec_8_52", delt_start_fraction=0.5)
print(
    f"# self-check fresh w7x->49 frac=0.5: restarts={fresh.restarts} "
    f"fsqr={fresh.fsqr:.6e}"
)

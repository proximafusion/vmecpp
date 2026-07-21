"""Probe the lambda-preconditioner scaling as a tail-convergence lever.

Finding 10 (docs/convergence_study.md): through the fsq 1e-6 .. 1e-9 bulk of
the tail, 78-99% of the preconditioned residual is the lambda channel at low
m in the core, i.e. the slow eigenmodes of the preconditioned descent are
lambda-flavored -- the lambda preconditioner (faclam) under-relaxes them
relative to the R/Z blocks. This study measures how much of that is real,
recoverable headroom using VmecModel.set_lambda_preconditioner_boost
(multiplier on the lambda preconditioner elements, optionally restricted to
m <= mmax / jF <= jmax):

- converge the coarse stage, refine (cubic) to ns=99, run the delt_recovery
  control until fsq first crosses 1e-6, snapshot the state;
- from the identical snapshot (state reset + zeroed velocity), continue with
  the delt_recovery control to ftol = 1e-11 for a range of scale values
  (the boost takes effect at the next preconditioner update), recording
  iterations to convergence, restarts, and the per-family preconditioned
  residual shares at the end (did lambda stop dominating?).

The scale does not move the fixed point (a preconditioner rescaling changes
the search direction, not the force balance), so converged physics must
agree across probes; fsq_final and the MHD energy are printed as a check.
"""

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
SNAPSHOT_FSQ = 1.0e-6
FTOL_FINAL = 1.0e-11
# (scale, mmax) probe grid; mmax None means all m
PROBES = (
    (1.0, None),  # baseline
    (2.0, None),
    (5.0, None),
    (10.0, None),
    (0.5, None),  # counter-probe: if boosting helps, halving should hurt
    (5.0, 1),  # boost only the m<=1 modes the anatomy flagged
    (10.0, 1),
)
MAX_ITERS = 6000


class _StopEarly(Exception):
    """Raised from a callback to end solve_equilibrium at the snapshot point."""


def set_scale(model, scale, mmax):
    """Configure the boost through the VmecModel API (persists on the model, so every
    probe sets it explicitly; scale 1.0 restores the default)."""
    model.set_lambda_preconditioner_boost(
        scale=scale, mmax=-1 if mmax is None else mmax
    )


def make_model(path):
    vi = vmecpp.VmecInput.from_file(path)
    cpp = vi._to_cpp_vmecindata()
    cpp.ns_array = np.array([NS_OLD, NS_NEW], dtype=np.int64)
    cpp.ftol_array = np.array([1.0e-9, FTOL_FINAL])
    cpp.niter_array = np.array([6000, MAX_ITERS], dtype=np.int64)
    return _vmecpp.VmecModel.create(cpp, NS_OLD)


def tail_snapshot(model):
    """Run the seeded fine stage until fsq first crosses SNAPSHOT_FSQ."""
    snap = {}

    def cb(s):
        if s.fsq_invariant < SNAPSHOT_FSQ:
            snap["state"] = np.array(model.get_state())
            snap["iteration"] = s.iteration
            raise _StopEarly

    with contextlib.suppress(_StopEarly):
        solve_equilibrium(
            model, style="delt_recovery", delt_start_fraction=0.5, callback=cb
        )
    return snap


def probe(model, seed):
    """Continue from the seed to FTOL_FINAL under the current env scaling."""
    model.set_state(seed)
    model.save_backup()
    model.zero_velocity()
    history = []
    r = solve_equilibrium(
        model, style="delt_recovery", delt_start_fraction=1.0, callback=history.append
    )
    last = history[-1] if history else None
    fam = (
        (last.fsqr1, last.fsqz1, last.fsql1)
        if last
        else (model.fsqr1, model.fsqz1, model.fsql1)
    )
    fam_total = sum(fam) or 1.0
    return {
        "iters": r.num_iterations,
        "restarts": r.restarts,
        "converged": r.converged,
        "fsq": r.fsqr + r.fsqz + r.fsql,
        "l_share": fam[2] / fam_total,
        "energy": model.mhd_energy,
    }


print(
    "case,scale,mmax,iters_to_1e-11,restarts,converged,fsq_final,"
    "final_L_share,mhd_energy"
)
for name, path in CASES.items():
    model = make_model(path)
    r1 = solve_equilibrium(model)
    if not r1.converged:
        print(f"# {name}: coarse stage did not converge, skipping")
        continue
    model.refine_to(NS_NEW, interpolation=_vmecpp.MultigridInterpolationScheme.CUBIC)
    snap = tail_snapshot(model)
    if "state" not in snap:
        print(f"# {name}: snapshot threshold not crossed, skipping")
        continue
    print(f"# {name}: snapshot at fine-stage iteration {snap['iteration']}")
    for scale, mmax in PROBES:
        set_scale(model, scale, mmax)
        p = probe(model, snap["state"])
        print(
            f"{name},{scale},{mmax if mmax is not None else 'all'},"
            f"{p['iters']},{p['restarts']},{int(p['converged'])},"
            f"{p['fsq']:.2e},{p['l_share']:.2%},{p['energy']:.10e}"
        )
    set_scale(model, 1.0, None)

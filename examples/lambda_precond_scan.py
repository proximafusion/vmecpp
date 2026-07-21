"""Hyperparameter scan of the lambda-preconditioner boost, including high-mpol cases.

Extends examples/lambda_precond_study.py (Finding 11) in three directions:

1. Full grid over the lambda-preconditioner boost scale x mmax
   (VmecModel.set_lambda_preconditioner_boost) on the standard cases
   (cth/cma/w7x at ns [25,99]), to map where the w7x -22% optimum sits and
   whether an m-window beats the m<=1 restriction.

2. High-mpol variants (mpol = 18 via VmecINDATA._set_mpol_ntor zero-padding,
   ns [13,49]): the high-m lambda damping sqrt(s)^min((m/16)^2, 8) is nearly
   inert for mpol <= 12, so only such cases can show what it guards against.
   The scan includes boosts of the full m range (partially cancelling the
   guard) vs m-restricted boosts: if top-m modes destabilize specifically
   when boosted, the guard protects marginally-stiff high-m modes near the
   axis (whose true radial envelope ~ s^(m/2) the flux-surface-averaged
   faclam cannot see).

3. Physical truncation spectra: full vmecpp.run at mpol = default and 18,
   printing the per-m peak amplitudes of R/Z/lambda from the wout physical
   coefficients. This measures how much high-m content the solutions
   actually carry (the truncation level at the top of the spectrum), which
   is the other candidate reason for the guard.

Probes reuse the Finding 11 protocol: converge the coarse stage, refine
(cubic), run delt_recovery until the invariant residual crosses 1e-6,
snapshot, and continue to ftol from the identical snapshot under each env
setting (the knobs are re-read at every preconditioner update).
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
SNAPSHOT_FSQ = 1.0e-6
FTOL_FINAL = 1.0e-11
MAX_ITERS = 8000

# standard-resolution scan grids: (scale, mmax); mmax None = all m
GRID_SMALL = (  # cth, cma: known unresponsive, coarse grid
    (1.0, None),
    (2.0, None),
    (5.0, None),
    (10.0, None),
    (5.0, 1),
    (10.0, 1),
)
GRID_W7X = (  # w7x: the responsive case, full grid
    (1.0, None),
    (2.0, None),
    (5.0, None),
    (10.0, None),
    (2.0, 1),
    (5.0, 1),
    (10.0, 1),
    (2.0, 2),
    (5.0, 2),
    (10.0, 2),
    (5.0, 4),
    (10.0, 4),
)
# high-mpol scan: full-range boosts vs m-restricted, to expose the guard
GRID_HIGH_MPOL = (
    (1.0, None),
    (2.0, None),
    (5.0, None),
    (10.0, None),
    (5.0, 1),
    (5.0, 4),
    (5.0, 11),
)
HIGH_MPOL = 18


class _StopEarly(Exception):
    """Raised from a callback to end solve_equilibrium at the snapshot point."""


def set_scale(model, scale, mmax):
    """Configure the boost through the VmecModel API (persists on the model, so every
    probe sets it explicitly; scale 1.0 restores the default)."""
    model.set_lambda_preconditioner_boost(
        scale=scale, mmax=-1 if mmax is None else mmax
    )


def make_model(path, ns_pair, mpol=None):
    vi = vmecpp.VmecInput.from_file(path)
    cpp = vi._to_cpp_vmecindata()
    if mpol is not None:
        cpp._set_mpol_ntor(mpol, cpp.ntor)
        cpp.ntheta = 0  # let Sizes re-derive the Nyquist-satisfying grid
    cpp.ns_array = np.array(ns_pair, dtype=np.int64)
    cpp.ftol_array = np.array([1.0e-9, FTOL_FINAL])
    cpp.niter_array = np.array([8000, MAX_ITERS], dtype=np.int64)
    return _vmecpp.VmecModel.create(cpp, ns_pair[0])


def tail_snapshot(model, ns_new):
    r1 = solve_equilibrium(model)
    if not r1.converged:
        return None
    model.refine_to(ns_new, interpolation=_vmecpp.MultigridInterpolationScheme.CUBIC)
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
    return snap if "state" in snap else None


def probe(model, seed):
    model.set_state(seed)
    model.save_backup()
    model.zero_velocity()
    r = solve_equilibrium(model, style="delt_recovery", delt_start_fraction=1.0)
    return r


def scan(name, path, ns_pair, grid, mpol=None):
    model = make_model(path, ns_pair, mpol=mpol)
    snap = tail_snapshot(model, ns_pair[1])
    if snap is None:
        print(f"# {name}: no snapshot (coarse stage or threshold failed), skipping")
        return
    print(f"# {name}: snapshot at fine-stage iteration {snap['iteration']}")
    for scale, mmax in grid:
        set_scale(model, scale, mmax)
        r = probe(model, snap["state"])
        print(
            f"{name},{scale},{mmax if mmax is not None else 'all'},"
            f"{r.num_iterations},{r.restarts},{int(r.converged)},"
            f"{r.fsqr + r.fsqz + r.fsql:.2e},{model.mhd_energy:.10e}"
        )
    set_scale(model, 1.0, None)


def truncation_spectrum(name, path, mpol, ns):
    """Per-m peak amplitude of the physical wout coefficients."""
    vi = vmecpp.VmecInput.from_file(path)
    cpp = vi._to_cpp_vmecindata()
    if mpol is not None:
        cpp._set_mpol_ntor(mpol, cpp.ntor)
        cpp.ntheta = 0
    cpp.ns_array = np.array([13, ns], dtype=np.int64)
    cpp.ftol_array = np.array([1.0e-9, 1.0e-11])
    cpp.niter_array = np.array([8000, 8000], dtype=np.int64)
    model = _vmecpp.VmecModel.create(cpp, 13)
    r1 = solve_equilibrium(model)
    model.refine_to(ns, interpolation=_vmecpp.MultigridInterpolationScheme.CUBIC)
    r2 = solve_equilibrium(model, style="delt_recovery", delt_start_fraction=0.5)
    if not (r1.converged and r2.converged):
        print(f"# {name} mpol={model.mpol}: did not converge, spectra skipped")
        return
    # decomposed state layout: [span][ns][mpol][ntor+1]; per-m peak within a
    # parity class is comparable (the radial scaling is parity-uniform in m)
    x = np.array(model.get_state())
    mp, nt1 = model.mpol, model.ntor + 1
    n_spans = (2 if model.lthreed else 1) * 3
    spans = x.reshape(n_spans, -1, mp, nt1)
    pp = 2 if model.lthreed else 1
    for i, fam in enumerate(("R", "Z", "L")):
        block = spans[i * pp : (i + 1) * pp]
        per_m = np.sqrt(np.sum(block**2, axis=(0, 1, 3)))
        ref = per_m.max() or 1.0
        prof = " ".join(f"{np.log10(a / ref + 1e-300):.1f}" for a in per_m)
        print(f"# spectrum {name} mpol={mp} {fam}: log10(a_m/a_peak) = {prof}")
        print(
            f"# spectrum {name} mpol={mp} {fam}: top-m ratio "
            f"a[{mp - 1}]/a_peak = {per_m[-1] / ref:.2e}"
        )


print("case,scale,mmax,iters_to_ftol,restarts,converged,fsq_final,mhd_energy")
scan("cth", CASES["cth"], (25, 99), GRID_SMALL)
scan("cma", CASES["cma"], (25, 99), GRID_SMALL)
scan("cma18", CASES["cma"], (13, 49), GRID_HIGH_MPOL, mpol=HIGH_MPOL)
truncation_spectrum("cma", CASES["cma"], None, 49)
truncation_spectrum("cma", CASES["cma"], HIGH_MPOL, 49)
truncation_spectrum("w7x", CASES["w7x"], None, 49)
truncation_spectrum("w7x", CASES["w7x"], HIGH_MPOL, 49)
scan("w7x", CASES["w7x"], (25, 99), GRID_W7X)
scan("w7x18", CASES["w7x"], (13, 49), GRID_HIGH_MPOL, mpol=HIGH_MPOL)

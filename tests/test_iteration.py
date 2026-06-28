# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the Python force-balance iteration (``vmecpp._iteration``).

The iteration logic -- time-step damping, time-step control, restart decisions,
the bad-Jacobian axis reguess, and the convergence test -- is ported to Python
from the C++ ``Vmec::SolveEquilibrium`` and drives the forward model exposed as
``VmecModel``. For boundaries whose linear initial guess is well posed, the
Python loop reproduces the C++ inner solve step-for-step; the reguess case is
checked for convergence to force balance. The PARVMEC and robust time-step-
control styles are checked against the VMEC 8.52 baseline.
"""

from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp import RestartReason
from vmecpp.cpp import _vmecpp  # type: ignore

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
EXAMPLES_DATA = REPO_ROOT / "examples" / "data"

# Inputs whose single-resolution linear guess is well posed (no axis reguess) live
# in TEST_DATA as JSON; w7x ships as an INDATA file under examples.
_INPUT_PATHS = {
    "w7x": EXAMPLES_DATA / "input.w7x",
}


def _single_resolution_indata(name: str, ns: int, ftol: float, niter: int):
    """C++ VmecINDATA for ``name`` forced to a single radial resolution."""
    path = _INPUT_PATHS.get(name, TEST_DATA / f"{name}.json")
    vmec_input = vmecpp.VmecInput.from_file(path)
    cpp_indata = vmec_input._to_cpp_vmecindata()
    cpp_indata.ns_array = np.array([ns], dtype=np.int64)
    cpp_indata.ftol_array = np.array([ftol])
    cpp_indata.niter_array = np.array([niter], dtype=np.int64)
    return cpp_indata


@pytest.mark.parametrize(
    ("name", "ns", "ftol", "niter"),
    [
        ("solovev", 15, 1.0e-12, 3000),
        ("cth_like_fixed_bdy", 25, 1.0e-8, 3000),
    ],
)
def test_python_iteration_reproduces_cpp_inner_solve(
    name: str, ns: int, ftol: float, niter: int
):
    """The Python loop reproduces the C++ Vmec::SolveEquilibrium step-for-step.

    From the same initialized state, running the Python iteration and the C++
    inner solve gives the same converged decision vector and the same invariant
    force-residual trace (to a floating-point floor; the only non-bit-exact
    operation is the damping-average sum, numpy vs Eigen).
    """
    cpp_indata = _single_resolution_indata(name, ns, ftol, niter)

    reference = _vmecpp.VmecModel.create(cpp_indata, ns)
    reference.solve()

    model = _vmecpp.VmecModel.create(cpp_indata, ns)
    result = vmecpp.solve_equilibrium(model)

    assert result.converged
    assert not result.failed
    assert result.axis_reguesses == 0

    cpp_state = np.asarray(reference.get_state())
    py_state = np.asarray(model.get_state())
    np.testing.assert_allclose(py_state, cpp_state, rtol=0, atol=1.0e-9)

    np.testing.assert_allclose(
        reference.force_residual_r, result.force_residual_r, rtol=1.0e-10, atol=1e-15
    )
    np.testing.assert_allclose(
        reference.force_residual_z, result.force_residual_z, rtol=1.0e-10, atol=1e-15
    )
    np.testing.assert_allclose(
        reference.force_residual_lambda,
        result.force_residual_lambda,
        rtol=1.0e-10,
        atol=1e-15,
    )


def test_python_iteration_recovers_from_bad_initial_jacobian():
    """Cma's linear guess overlaps at a single resolution (a bad initial Jacobian); the
    loop must recompute the magnetic axis and still converge to force balance."""
    cpp_indata = _single_resolution_indata("cma", 15, 1.0e-10, 8000)
    model = _vmecpp.VmecModel.create(cpp_indata, 15)
    result = vmecpp.solve_equilibrium(model)

    assert result.converged
    assert not result.failed
    assert result.axis_reguesses >= 1
    assert result.fsqr <= 1.0e-10
    assert result.fsqz <= 1.0e-10
    assert result.fsql <= 1.0e-10


def test_styles_agree_when_no_restart():
    """When no restart fires, the three styles take the identical path.

    The styles differ only in how a growing residual is handled, so on a case that needs
    no restart they advance the geometry through exactly the same states (the bit-
    identical result seen across the test_data equilibria). niter is capped low: the
    early descent already triggers no restart for any style, so the truncated runs are
    enough to show the three paths coincide.
    """
    cpp_indata = _single_resolution_indata("cth_like_fixed_bdy", 25, 1.0e-10, 300)
    states = {}
    for style in ("vmec_8_52", "parvmec", "robust"):
        model = _vmecpp.VmecModel.create(cpp_indata, 25)
        result = vmecpp.solve_equilibrium(model, style=style)
        assert result.restarts == 0, style
        states[style] = np.asarray(model.get_state())
    assert np.array_equal(states["vmec_8_52"], states["parvmec"])
    assert np.array_equal(states["vmec_8_52"], states["robust"])


@pytest.mark.parametrize(
    ("name", "ns", "niter", "reasons"),
    [
        # cma's cold guess has a bad Jacobian -> axis reguess + huge initial forces
        ("cma", 72, 200, {RestartReason.HUGE_INITIAL_FORCES}),
        # W7-X drives a descent with mid-run bad-Jacobian time-step reverts
        ("w7x", 15, 300, {RestartReason.BAD_JACOBIAN}),
        # li383's cold guess also needs the axis reguess; the descent itself is
        # clean, which makes it a sharp probe of the post-reguess state (the
        # lambda scaling after reinitialize).
        ("li383_low_res", 31, 250, set()),
    ],
)
def test_python_iteration_matches_cpp_restart_path(
    name: str, ns: int, niter: int, reasons: set
):
    """Hard single-resolution cases reproduce the C++ inner solve, restarts and all.

    cma's linear guess has a bad Jacobian (forcing a cold-start axis reguess and huge
    initial forces); W7-X triggers mid-run bad-Jacobian time-step reverts. Together they
    exercise the restart machinery the well-posed cases never touch. niter is capped low
    -- the test compares the Python loop and the C++ inner solve step for step over the
    same (artificially truncated) run, so it does not need to reach convergence; the
    truncation just keeps it fast. The control flow is checked on the *restart-reason*
    trace (NO_RESTART / BAD_JACOBIAN / BAD_PROGRESS / HUGE_INITIAL_FORCES per recorded
    iteration, exposed in the wout) -- it must match the C++ reference exactly -- and
    the residual trace must agree until the numpy-vs-Eigen floating-point floor of the
    damping average accumulates through the chaotic early transient.
    """
    cpp_indata = _single_resolution_indata(name, ns, ftol=1.0e-16, niter=niter)

    reference = _vmecpp.VmecModel.create(cpp_indata, ns)
    reference.solve()

    model = _vmecpp.VmecModel.create(cpp_indata, ns)
    result = vmecpp.solve_equilibrium(model)

    assert not result.failed

    cpp_rr = np.asarray(reference.restart_reasons)
    py_rr = np.asarray(result.restart_reasons)
    # Identical recorded length and the identical restart-reason path, step for
    # step -- this is what the original port got wrong on these cases.
    assert len(py_rr) == len(cpp_rr)
    np.testing.assert_array_equal(py_rr, cpp_rr)
    # The truncated run actually exercises the expected restart machinery (a sticky
    # restart reason or a dropped reguess would change which reasons appear).
    assert reasons.issubset(set(py_rr.tolist()))

    # The residual traces track tightly; the only non-bit-exact operation left
    # is the order of additions in the damping average, whose last-bit noise
    # can grow through long chaotic transients.
    cpp_r = np.asarray(reference.force_residual_r)
    py_r = np.asarray(result.force_residual_r)
    np.testing.assert_allclose(py_r[:50], cpp_r[:50], rtol=1.0e-9, atol=1e-15)
    np.testing.assert_allclose(py_r, cpp_r, rtol=1.0e-3, atol=1e-15)


def test_alternative_styles_converge_cma():
    """All three time-step-control styles converge cma to the same tolerance.

    They differ only in how a growing residual is handled, so on a case that descends
    smoothly they take very similar paths (only a couple of reverts); the test guards
    that none of them thrashes or fails. (A coarse ns keeps it fast while still needing
    the cold-start axis reguess.)
    """
    cpp_indata = _single_resolution_indata("cma", 15, 1.0e-8, 8000)
    results = {}
    for style in ("vmec_8_52", "parvmec", "robust"):
        model = _vmecpp.VmecModel.create(cpp_indata, 15)
        results[style] = vmecpp.solve_equilibrium(model, style=style)

    for style, result in results.items():
        assert result.converged, style
        assert result.fsqr <= 1.0e-8, style
        assert result.fsqz <= 1.0e-8, style
        assert result.fsql <= 1.0e-8, style
        # No style should thrash now that the restart reason is cleared each
        # evaluation (a sticky HUGE_INITIAL_FORCES used to force ~20 reverts).
        assert result.restarts < 10, style
    assert results["parvmec"].restarts == 0


def test_callback_records_iteration_state():
    """The per-iteration callback fires once per recorded iteration with a consistent
    IterationState snapshot of the convergence / flow-control state.

    A coarse cma solve converges quickly but still needs the cold-start axis reguess, so
    ijacob escalates and the reguess bookkeeping is exercised.
    """
    cpp_indata = _single_resolution_indata("cma", 15, 1.0e-8, 8000)
    model = _vmecpp.VmecModel.create(cpp_indata, 15)

    states: list[vmecpp.IterationState] = []
    result = vmecpp.solve_equilibrium(model, style="vmec_8_52", callback=states.append)

    assert result.converged
    # One snapshot per recorded force iteration.
    assert len(states) == result.num_iterations
    # The iteration index is 1-based and monotonically increasing.
    assert [s.iteration for s in states] == list(range(1, len(states) + 1))
    # The flow-control counters are consistent with the result summary.
    assert states[-1].n_restarts == result.restarts
    assert sum(s.restarted for s in states) == result.restarts
    assert states[-1].n_reguesses == result.axis_reguesses
    # fsq_invariant is exactly the sum tested for convergence. (The final converged
    # iteration returns before the callback fires, so the last snapshot is the step
    # just before convergence, not result.fsqr -- num_iterations already covers the
    # count.)
    last = states[-1]
    assert last.fsq_invariant == pytest.approx(last.fsqr + last.fsqz + last.fsql)
    # cma's cold guess has a bad Jacobian, so the axis reguess fires (ijacob >= 1).
    assert states[-1].ijacob >= 1
    assert result.axis_reguesses >= 1


def test_reinitialize_preserves_lambda_scaling():
    """Reinitialize() must leave the lambda scaling (lamscale) unchanged.

    InitializeRadial accumulates rmsPhiP across calls (Vmec::run resets the
    constants before each call); without the matching reset in reinitialize(),
    a second initialization doubles rmsPhiP, rescales lamscale by sqrt(2), and
    the whole lambda sector (preconditioned residual fsql1 above all) silently
    diverges from the C++ on every post-reguess trajectory.
    """
    cpp_indata = _single_resolution_indata("solovev", 15, 1.0e-12, 3000)

    once = _vmecpp.VmecModel.create(cpp_indata, 15)
    once.reinitialize()
    once.evaluate(1, 1)

    twice = _vmecpp.VmecModel.create(cpp_indata, 15)
    twice.reinitialize()
    twice.reinitialize()
    twice.evaluate(1, 1)

    assert once.fsqr == twice.fsqr
    assert once.fsql == pytest.approx(twice.fsql, rel=1e-14)
    # fsql1 has no normalization that cancels a lamscale change; before the
    # constants reset the second reinitialize shifted it by a factor 1.5.
    assert once.fsql1 == pytest.approx(twice.fsql1, rel=1e-14)


def test_python_multigrid_matches_cpp_run():
    """solve_multigrid reproduces the C++ Vmec::run multi-grid ramp.

    cma's own input is a two-stage ramp (ns_array [25, 51]) whose coarse stage
    needs the cold-start axis reguess. The C++ reference is the full
    vmecpp.run; its wout records the force-residual and restart-reason traces
    concatenated across stages, which must match the concatenation of the
    Python per-stage traces step for step.
    """
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA / "cma.json")

    reference = vmecpp.run(vmec_input, max_threads=1, verbose=False)
    cpp_r = np.asarray(reference.wout.force_residual_r)
    cpp_rr = np.asarray(reference.wout.restart_reason_timetrace)

    model, results = vmecpp.solve_multigrid(vmec_input)
    assert len(results) == 2
    assert all(r.converged for r in results)
    assert model.ns == vmec_input.ns_array[-1]

    py_r = np.concatenate([np.asarray(r.force_residual_r) for r in results])
    py_rr = np.concatenate([np.asarray(r.restart_reasons) for r in results]).astype(
        np.int64
    )

    assert len(py_rr) == len(cpp_rr)
    np.testing.assert_array_equal(py_rr, cpp_rr)
    np.testing.assert_allclose(py_r[:50], cpp_r[:50], rtol=1.0e-9, atol=1e-15)
    np.testing.assert_allclose(py_r, cpp_r, rtol=1.0e-3, atol=1e-15)


def test_refine_to_multigrid_ramp_converges():
    """Driving the multi-grid ramp from Python (refine_to interpolates the converged
    coarse geometry onto the finer grid) converges, and seeding the finer grid from a
    coarse solution takes fewer steps -- and no reverts -- than a cold single-
    resolution solve at the same resolution.

    (cma 9 -> 15 keeps it fast while still exercising the radial interpolation and the
    cold reguess on the coarse stage.)
    """
    cpp_indata = _single_resolution_indata("cma", 15, 1.0e-10, 8000)

    model = _vmecpp.VmecModel.create(cpp_indata, 9)
    coarse = vmecpp.solve_equilibrium(model, style="vmec_8_52")
    assert coarse.converged
    assert model.ns == 9

    model.refine_to(15)
    assert model.ns == 15
    ramped = vmecpp.solve_equilibrium(model, style="vmec_8_52")
    assert ramped.converged
    assert ramped.fsqr <= 1.0e-10

    # A cold solve straight at ns=15 needs a revert and more steps; the ramped fine
    # solve, seeded by the interpolated coarse solution, needs none.
    cold = vmecpp.solve_equilibrium(
        _vmecpp.VmecModel.create(cpp_indata, 15), style="vmec_8_52"
    )
    assert cold.converged
    assert ramped.restarts == 0
    assert ramped.num_iterations < cold.num_iterations

    # refine_to only refines: refining to the current (or a coarser) ns is rejected.
    with pytest.raises(RuntimeError):
        model.refine_to(15)

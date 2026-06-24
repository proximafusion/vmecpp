# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Python port of the VMEC++ force-balance iteration.

The expensive forward model (flux-surface geometry -> MHD forces) and the
per-step Fourier-coefficient arithmetic stay in C++
(:class:`vmecpp.cpp._vmecpp.VmecModel`); this module owns the iteration *logic*
-- the time-step damping, time-step control, restart decisions, the bad-Jacobian
axis reguess, and the convergence test -- ported one-to-one from the C++
``Vmec::SolveEquilibrium`` / ``Vmec::SolveEquilibriumLoop`` / ``Vmec::Evolve``.

Three time-step-control styles share the same forward model, damping, axis
reguess, and ijacob 25/50/75 escalation; they differ only in how a growing
residual is handled:

* ``"vmec_8_52"`` -- the Fortran VMEC 8.52 control VMEC++ implements: revert the
  geometry and cut the step 10% (counting toward the give-up escalation) once the
  preconditioned residual grows past 100x its running minimum, plus a
  slow-progress branch. Reproduces the C++ inner solve step-for-step.
* ``"parvmec"`` -- the PARVMEC / VMEC2000 9.0 control: track the running minima of
  both the preconditioned and the invariant residual sums, store whenever both
  improve, and revert only when either grows past 1e4x its minimum, and then
  gently (step /1.03, not counted toward the give-up escalation). The 100x-more-
  permissive leash rides through transients that 8.52 reverts prematurely;
  conversely 8.52 converges some cases where PARVMEC's permissiveness wanders.
* ``"robust"`` -- a common-ground scheme (Python only): PARVMEC's dual-residual
  permissive, non-escalating revert at a moderate 1e3 leash, plus VMEC 8.52's
  slow-progress safeguard so the permissiveness cannot stall short of force
  balance. Built to converge for the equilibria each of the other two handles.

Owning this loop in Python is the foundation for developing further iteration
schemes (like ``"robust"``) without touching the C++ core.
"""

from __future__ import annotations

import enum
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from vmecpp.cpp import _vmecpp  # type: ignore


class RestartReason(enum.IntEnum):
    BAD_JACOBIAN = 2
    """Irst == 2, bad Jacobian, flux surfaces are overlapping."""

    BAD_PROGRESS = 3
    """Irst == 3, bad progress, residuals not decaying as expected."""

    HUGE_INITIAL_FORCES = 4
    """Irst == 4, huge initial forces, flux surfaces are too close to each other (but
    not overlapping yet)"""


# Vmec::kNDamp -- history length for the 1/tau time-step damping average.
_NDAMP = 10
# FlowControl::kPreconditionerUpdateInterval.
_PRECOND_INTERVAL = 25

# Iteration styles. "vmec_8_52" and "parvmec" are vmecpp::IterationStyle values
# (exposed as VmecModel.iteration_style); "robust" is a Python-only common-ground
# scheme (see solve_equilibrium) that runs on the same forward model.
_VMEC_8_52 = "vmec_8_52"
_PARVMEC = "parvmec"
_ROBUST = "robust"

# Residual-growth tolerance before reverting the geometry: VMEC 8.52 reverts at
# 100x the running minimum (TimeStepControl in Vmec::SolveEquilibriumLoop),
# PARVMEC at 1e4x (TimeStepControl in PARVMEC's evolve.f). The "robust" scheme
# uses a moderate leash between the two.
_VMEC_8_52_BLOWUP = 100.0
_PARVMEC_BLOWUP = 1.0e4
_ROBUST_BLOWUP = 1.0e3

_logger = logging.getLogger(__name__)


@dataclass
class IterationState:
    """Snapshot of all convergence and flow-control quantities at one force iteration,
    passed to the ``callback`` of :func:`solve_equilibrium`.

    Captured *after* the time-step / restart control has run for the iteration, so
    ``delt`` / ``restarted`` / ``saved_backup`` reflect the decision taken this
    step. Use it to trace why the convergence dynamics change qualitatively (e.g.
    the time step collapsing, the residual leash repeatedly reverting, the damping
    average saturating) over the course of a run.
    """

    # --- iteration counters ---
    iteration: int
    """Global force-iteration index (1-based, monotonically increasing)."""
    iter1: int
    """First iteration of the current time-step segment (reset on every restart)."""
    iter2: int
    """Effective iteration index used by the forward model (iteration - bad_resets)."""

    # --- invariant force residuals (the convergence test) ---
    fsqr: float
    fsqz: float
    fsql: float
    fsq_invariant: float
    """Fsqr + fsqz + fsql, the sum tested against ``ftolv`` for convergence."""

    # --- preconditioned force residuals (drive the damping / time-step control) ---
    fsqr1: float
    fsqz1: float
    fsql1: float
    fsq_preconditioned: float
    """Fsqr1 + fsqz1 + fsql1, the sum the time-step control watches for blow-up."""

    # --- running minima tracked by the time-step control ---
    res0: float
    """Running minimum of the preconditioned residual sum (revert leash baseline)."""
    res1: float
    """Running minimum of the invariant residual sum (parvmec/robust leash)."""

    # --- time-step / damping state ---
    delt: float
    """Current time step ``delt0r`` after this iteration's control decision."""
    otav: float
    """Mean inverse damping time 1/tau over the last ``_NDAMP`` steps."""
    dtau: float
    """Delt0r * otav / 2, the damping applied in the Garabedian time step."""

    # --- flow control / restart bookkeeping ---
    ijacob: int
    """Bad-Jacobian / blow-up escalation counter (give-up at >= 75)."""
    restart_reason: int
    """RestartReason returned by the forward model this iteration (0 if none)."""
    n_restarts: int
    """Cumulative time-step reverts so far."""
    n_reguesses: int
    """Cumulative magnetic-axis reguesses so far."""
    restarted: bool
    """True if the geometry was reverted (time step cut) this iteration."""
    saved_backup: bool
    """True if the geometry was checkpointed (both residuals improved) this step."""

    # --- physics ---
    mhd_energy: float
    """Total MHD energy W of the current geometry."""


@dataclass
class IterationResult:
    """Outcome of :func:`solve_equilibrium`."""

    converged: bool
    failed: bool
    num_iterations: int
    restarts: int
    axis_reguesses: int
    fsqr: float
    fsqz: float
    fsql: float
    iteration_style: str = _VMEC_8_52
    force_residual_r: list[float] = field(default_factory=list)
    force_residual_z: list[float] = field(default_factory=list)
    force_residual_lambda: list[float] = field(default_factory=list)
    restart_reasons: list[int] = field(default_factory=list)


def solve_equilibrium(
    model: _vmecpp.VmecModel,
    *,
    style: str | None = None,
    verbose: bool = False,
    callback: Callable[[IterationState], None] | None = None,
):
    """Drive an initialized ``VmecModel`` to force balance.

    Reproduces the C++ ``Vmec::SolveEquilibrium`` inner solve: an outer retry loop
    wraps the force iteration so a bad initial Jacobian triggers a magnetic axis
    reguess; the inner loop evaluates the forward model, tests convergence on the
    invariant residuals, advances the geometry with the Garabedian time step
    (damping from the preconditioned-residual history), and applies the time-step
    / restart control of the selected ``style``. ``style`` is one of
    ``"vmec_8_52"``, ``"parvmec"``, ``"robust"``; when ``None`` it defaults to
    ``model.iteration_style``. ``model`` must already be initialized at a single
    radial resolution via ``VmecModel.create``.

    If ``callback`` is given, it is invoked once per force iteration with an
    :class:`IterationState` snapshot of every convergence and flow-control quantity
    (after that iteration's time-step / restart decision), for tracing the
    iteration dynamics. It is not called on iterations that immediately break out
    for convergence, a bad-Jacobian axis reguess, or an ijacob escalation reset.
    """
    style = style if style is not None else model.iteration_style
    parvmec = style == _PARVMEC
    robust = style == _ROBUST

    ftolv = model.ftolv
    niterv = model.niterv
    delt_user = model.delt  # indata.delt, the reference time step

    delt0r = delt_user
    inv_tau = np.zeros(_NDAMP)
    fsq_prev = 1.0  # fc_.fsq is initialized to 1.0 in InitializeRadial
    res0 = -1.0  # running minimum of the preconditioned residual sum
    res1 = -1.0  # running minimum of the invariant residual sum (parvmec/robust)
    iter1 = 1
    iter2 = 1
    ijacob = 0

    trace_r: list[float] = []
    trace_z: list[float] = []
    trace_l: list[float] = []
    trace_rr: list[int] = []  # restart_reason per recorded iteration (C++ parity)
    fsqr = fsqz = fsql = float("nan")
    converged = False
    failed = False
    n_restarts = 0
    n_reguesses = 0

    # The reguess re-initialization is deferred to the top of the next outer
    # retry, exactly as the C++ MUST_RETRY path defers it to the next entry of
    # SolveEquilibriumLoop: the failed evaluation has already set restart_reason,
    # and lreset_internal records that the state must be reset from the
    # (re-guessed) boundary/axis before the next force iteration.
    pending_bad_jacobian_reinit = False
    lreset_internal = False

    # --- outer retry loop (Vmec::SolveEquilibrium) ---
    must_retry = True
    eqsolve_retries = 0
    while must_retry and eqsolve_retries < niterv and not converged and not failed:
        must_retry = False
        eqsolve_retries += 1

        # --- SolveEquilibriumLoop entry: re-init from a pending reguess ---
        # Mirrors vmec.cc lines 777-799: when a bad Jacobian (or the ijacob 25/50
        # escalation) left restart_reason == BAD_JACOBIAN, re-initialize the
        # radial state from the (recomputed) axis and back it up. iter1/iter2/
        # inv_tau are NOT reset here -- they carry across the retry via the
        # force_iteration counter, just as the C++ iter1_/iter2_/invTau_ persist
        # across SolveEqLoop calls. Reinitialize re-runs InitializeRadial so the
        # forward-model state (preconditioner / scratch) starts clean; the bare
        # setZero + interp the C++ does inside its single continuous parallel
        # region leaves stale forward-model state that a step-by-step driver
        # cannot tolerate (see VmecModel::Reinitialize).
        if pending_bad_jacobian_reinit:
            if lreset_internal:
                model.reinitialize()
                model.save_backup()
            pending_bad_jacobian_reinit = False
            lreset_internal = False

        # --- inner force-iteration loop (Vmec::SolveEquilibriumLoop) ---
        bad_resets = 0
        force_iteration = iter2
        while force_iteration <= niterv:
            iter2 = force_iteration - bad_resets

            # Evolve: forward model, then convergence / damping / time step.
            model.evaluate(iter1, iter2)
            fsqr, fsqz, fsql = model.fsqr, model.fsqz, model.fsql
            rr = model.restart_reason
            finite = math.isfinite(fsqr) and math.isfinite(fsqz) and math.isfinite(fsql)
            status_bad_jacobian = (
                iter2 == 1 and rr == RestartReason.BAD_JACOBIAN
            ) or not finite

            # converged? Evolve returns before recording / time-stepping once the
            # invariant residuals are below tolerance (vmec.cc lines 1194-1206).
            if (
                not status_bad_jacobian
                and fsqr <= ftolv
                and fsqz <= ftolv
                and (fsql <= ftolv)
            ):
                converged = True
                break

            # A bad initial Jacobian (status_bad_jacobian) makes Evolve return
            # *before* recording the trace or time-stepping; a finite huge-initial-
            # forces evaluation runs the full Evolve (trace + damping + time step)
            # and is only reguessed afterwards. So only time-step when the forward
            # model produced a usable geometry.
            fsq1 = model.fsqr1 + model.fsqz1 + model.fsql1
            otav = -1.0  # invalid value
            dtau = -1.0  # invalid value
            if not status_bad_jacobian:
                if iter2 == iter1:
                    inv_tau[:] = 0.15 / delt0r
                inv_tau[:-1] = inv_tau[1:]
                if iter2 > iter1 and fsq1 != 0.0:
                    inv_tau[-1] = min(abs(math.log(fsq1 / fsq_prev)), 0.15) / delt0r
                fsq_prev = fsq1

                trace_r.append(fsqr)
                trace_z.append(fsqz)
                trace_l.append(fsql)
                trace_rr.append(int(rr))

                # Sequential left-to-right sum, matching Eigen's scalar
                # VectorXd::sum() bit for bit. numpy's pairwise summation
                # associates differently; the last-bit difference in the
                # damping average amplifies through the early force
                # transient on cold-start cases and shifts the convergence
                # iteration against the C++ loop.
                otav = 0.0
                for _v in inv_tau:
                    otav += float(_v)
                otav /= _NDAMP
                dtau = delt0r * otav / 2.0
                model.perform_time_step(1.0 / (1.0 + dtau), 1.0 - dtau, delt0r)

            saved_backup = False
            restarted = False

            # bad initial Jacobian / huge initial forces: recompute the magnetic
            # axis once and restart the force iteration (vmec.cc lines 833-874).
            # Counters carry: the C++ does not reset iter1_/iter2_ here, so the
            # reguessed retry continues the same iteration numbering.
            if (
                ijacob == 0
                and (status_bad_jacobian or rr == RestartReason.HUGE_INITIAL_FORCES)
                and model.ns >= 3
            ):
                # the axis is recomputed by reinitialize() at the next outer-loop
                # entry (mirroring the C++ recompute + lreset re-init)
                ijacob = 1
                n_reguesses += 1
                pending_bad_jacobian_reinit = True
                lreset_internal = True
                must_retry = True
                break
            if status_bad_jacobian:
                failed = True  # ijacob already used: cannot recover
                break

            # repeated bad Jacobian: reset the time step (ijacob 25/50 escalation,
            # vmec.cc lines 906-942). RestartIteration reverts and bumps ijacob,
            # then delt0r is reset relative to the user step.
            if ijacob in (25, 50):
                model.restore_backup()
                ijacob += 1
                iter1 = iter2
                delt0r = (0.98 if ijacob < 50 else 0.96) * delt_user
                n_restarts += 1
                pending_bad_jacobian_reinit = True
                must_retry = True
                break
            if ijacob >= 75:
                failed = True
                break

            # Huge initial forces (forward model flags this only at iter2 == 1,
            # once the bad-Jacobian reguess has been used up so ijacob != 0): the
            # C++ TimeStepControl seeds res0 from this iteration, then the
            # terminal "restart_reason != NO_RESTART" check (vmec.cc line 1007)
            # runs RestartIteration -- which for HUGE_INITIAL_FORCES *saves* the
            # (already time-stepped) state rather than reverting it -- and bumps
            # bad_resets / iter1 so the next evaluation keeps iter2 == 1. Reproduce
            # that here so the post-reguess trajectory matches C++ step for step.
            if rr == RestartReason.HUGE_INITIAL_FORCES:
                if iter2 == iter1 or res0 == -1.0:
                    res0 = fsq1
                    res1 = fsqr + fsqz + fsql
                res0 = min(res0, fsq1)
                res1 = min(res1, fsqr + fsqz + fsql)
                model.save_backup()
                saved_backup = True
                iter1 = iter2
                bad_resets += 1
            # --- time-step / restart control (style-dependent) ---
            elif parvmec:
                # PARVMEC TimeStepControl: track the running minima of both the
                # preconditioned (res0) and invariant (res1) residual sums; store
                # whenever both improve; revert (gently, /1.03, without counting
                # toward the give-up escalation) only when either grows past 1e4x
                # its minimum after more than 10 steps since the last restart.
                fsq0 = fsqr + fsqz + fsql
                if iter2 == iter1 or res0 == -1.0:
                    res0 = fsq1
                    res1 = fsq0
                res0 = min(res0, fsq1)
                res1 = min(res1, fsq0)
                if fsq1 <= res0 and fsq0 <= res1:
                    model.save_backup()
                    saved_backup = True
                elif (iter2 - iter1) > 10 and (
                    fsq1 > _PARVMEC_BLOWUP * res0 or fsq0 > _PARVMEC_BLOWUP * res1
                ):
                    model.restore_backup()
                    delt0r /= 1.03
                    iter1 = iter2
                    bad_resets += 1
                    n_restarts += 1
                    restarted = True
            elif robust:
                # Common-ground control: PARVMEC's dual-residual permissive leash
                # (revert gently, without counting toward the give-up escalation)
                # at a moderate threshold, plus VMEC 8.52's slow-progress safeguard
                # so the permissiveness cannot stall short of convergence.
                fsq0 = fsqr + fsqz + fsql
                if iter2 == iter1 or res0 == -1.0:
                    res0 = fsq1
                    res1 = fsq0
                res0 = min(res0, fsq1)
                res1 = min(res1, fsq0)
                if fsq1 <= res0 and fsq0 <= res1:
                    model.save_backup()
                    saved_backup = True
                elif (
                    (iter2 - iter1) > 10
                    and (fsq1 > _ROBUST_BLOWUP * res0 or fsq0 > _ROBUST_BLOWUP * res1)
                ) or (
                    (iter2 - iter1) > _PRECOND_INTERVAL // 2
                    and iter2 > 2 * _PRECOND_INTERVAL
                    and (fsqr + fsqz) > 1.0e-2
                ):
                    model.restore_backup()
                    delt0r /= 1.03
                    iter1 = iter2
                    bad_resets += 1
                    n_restarts += 1
                    restarted = True
            else:
                # VMEC 8.52 TimeStepControl (reproduces the C++ inner solve,
                # vmec.cc lines 962-1016). The forward model may set
                # restart_reason == BAD_JACOBIAN mid-run (overlapping surfaces);
                # the C++ reverts on any non-NO_RESTART reason at the end of the
                # iteration, so a forward-model bad Jacobian reverts via the same
                # RestartIteration BAD_JACOBIAN branch as the 100x leash (delt0r
                # *= 0.9, ijacob++). The slow-progress branch reverts gently
                # (/1.03) without escalating ijacob.
                if iter2 == iter1 or res0 == -1.0:
                    res0 = fsq1
                res0 = min(res0, fsq1)
                blew_up = fsq1 > _VMEC_8_52_BLOWUP * res0 and iter2 > iter1
                if rr == RestartReason.BAD_JACOBIAN or blew_up:
                    model.restore_backup()
                    delt0r *= 0.9
                    ijacob += 1
                    iter1 = iter2
                    bad_resets += 1
                    n_restarts += 1
                    restarted = True
                elif fsq1 <= res0 and (iter2 - iter1) > 10:
                    model.save_backup()
                    saved_backup = True
                elif (
                    (iter2 - iter1) > _PRECOND_INTERVAL // 2
                    and iter2 > 2 * _PRECOND_INTERVAL
                    and (fsqr + fsqz) > 1.0e-2
                ):
                    model.restore_backup()
                    delt0r /= 1.03
                    iter1 = iter2
                    bad_resets += 1
                    n_restarts += 1
                    restarted = True

            if callback is not None:
                callback(
                    IterationState(
                        iteration=len(trace_r),
                        iter1=iter1,
                        iter2=iter2,
                        fsqr=fsqr,
                        fsqz=fsqz,
                        fsql=fsql,
                        fsq_invariant=fsqr + fsqz + fsql,
                        fsqr1=model.fsqr1,
                        fsqz1=model.fsqz1,
                        fsql1=model.fsql1,
                        fsq_preconditioned=fsq1,
                        res0=res0,
                        res1=res1,
                        delt=delt0r,
                        otav=otav,
                        dtau=dtau,
                        ijacob=ijacob,
                        restart_reason=int(rr),
                        n_restarts=n_restarts,
                        n_reguesses=n_reguesses,
                        restarted=restarted,
                        saved_backup=saved_backup,
                        mhd_energy=model.mhd_energy,
                    )
                )

            if verbose and iter2 % 200 == 0:
                _logger.info(
                    "it=%6d fsqr=%.3e fsqz=%.3e fsql=%.3e delt=%.4f",
                    iter2,
                    fsqr,
                    fsqz,
                    fsql,
                    delt0r,
                )

            force_iteration += 1

    return IterationResult(
        converged=converged,
        failed=failed,
        num_iterations=len(trace_r),
        restarts=n_restarts,
        axis_reguesses=n_reguesses,
        fsqr=fsqr,
        fsqz=fsqz,
        fsql=fsql,
        iteration_style=style,
        force_residual_r=trace_r,
        force_residual_z=trace_z,
        force_residual_lambda=trace_l,
        restart_reasons=trace_rr,
    )


def solve_multigrid(
    vmec_input,
    *,
    iteration_style: str | None = None,
    verbose: bool = False,
    callback: Callable[[IterationState], None] | None = None,
):
    """Drive the full coarse-to-fine multi-grid ramp of ``vmec_input`` from
    Python, mirroring the multi-grid loop of the C++ ``Vmec::run``.

    Walks ``ns_array`` in order: entries coarser than the finest resolution
    solved so far are skipped (the C++ ``ns_min`` rule), each remaining stage
    is solved to its ``ftol_array`` / ``niter_array`` tolerance with
    :func:`solve_equilibrium`, and the converged geometry is interpolated onto
    the next stage's radial grid with ``VmecModel.refine_to`` (the C++
    ``InterpolateToNextMultigridStep``). A stage that exhausts its iteration
    budget without converging proceeds to the next stage, exactly as the C++
    run does; a stage that hard-fails (unrecoverable bad Jacobian, the
    ijacob >= 75 give-up) stops the ramp.

    ``ftol_array`` / ``niter_array`` entries are matched to ``ns_array``
    entries by ns value (``VmecModel.create`` / ``refine_to`` semantics);
    repeated ns values in ``ns_array`` are not supported because the model
    only refines to strictly finer grids.

    Returns ``(model, results)``: the model holds the final stage's geometry
    and ``results`` is the per-stage list of :class:`IterationResult`.
    """
    cpp_indata = vmec_input._to_cpp_vmecindata()
    if iteration_style in (_VMEC_8_52, _PARVMEC):
        cpp_indata.iteration_style = getattr(
            _vmecpp.IterationStyle, iteration_style.upper()
        )

    ns_array = [int(ns) for ns in np.asarray(cpp_indata.ns_array)]
    if len(set(ns_array)) != len(ns_array):
        msg = (
            "solve_multigrid: repeated ns values in ns_array are not "
            f"supported (got {ns_array})"
        )
        raise ValueError(msg)

    model = None
    results: list[IterationResult] = []
    ns_min = 3  # FlowControl::ns_min -- no refinement downward
    for ns in ns_array:
        if ns < ns_min:
            # mirror the C++ run: skip entries coarser than already solved
            continue
        ns_min = ns
        if model is None:
            model = _vmecpp.VmecModel.create(cpp_indata, ns)
        else:
            model.refine_to(ns)
        result = solve_equilibrium(
            model, style=iteration_style, verbose=verbose, callback=callback
        )
        results.append(result)
        if result.failed:
            break
    if model is None:
        msg = f"solve_multigrid: no usable ns_array entries (got {ns_array})"
        raise ValueError(msg)
    return model, results


def iterate(
    vmec_input,
    ns: int | None = None,
    *,
    iteration_style: str | None = None,
    verbose: bool = False,
    callback: Callable[[IterationState], None] | None = None,
):
    """Create a single-resolution :class:`VmecModel` for ``vmec_input`` and run the
    Python force-balance iteration on it.

    ``iteration_style`` is one of ``"vmec_8_52"``, ``"parvmec"``, ``"robust"`` and
    overrides the input's own setting; when ``None`` the input's
    ``iteration_style`` is used (default ``"vmec_8_52"``). Returns
    ``(model, IterationResult)``; ``model`` holds the converged geometry (inspect
    it with ``get_state()`` / the residual properties). ``ns`` defaults to the
    last entry of the input's ``ns_array``.

    ``callback`` is forwarded to :func:`solve_equilibrium`: when given, it is
    called with an :class:`IterationState` snapshot once per force iteration, for
    tracing the convergence and flow-control dynamics.
    """
    if ns is None:
        ns = int(np.asarray(vmec_input.ns_array)[-1])
    cpp_indata = vmec_input._to_cpp_vmecindata()
    # vmec_8_52 and parvmec are C++ IterationStyle values; "robust" is a
    # Python-only common-ground scheme on the (style-agnostic) forward model, so
    # only the former two set the enum -- the style is always passed to the loop.
    if iteration_style in (_VMEC_8_52, _PARVMEC):
        cpp_indata.iteration_style = getattr(
            _vmecpp.IterationStyle, iteration_style.upper()
        )
    model = _vmecpp.VmecModel.create(cpp_indata, ns)
    result = solve_equilibrium(
        model, style=iteration_style, verbose=verbose, callback=callback
    )
    return model, result

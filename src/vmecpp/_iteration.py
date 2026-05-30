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

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from vmecpp.cpp import _vmecpp  # type: ignore

# Restart reasons, mirroring vmecpp::RestartReason.
_NO_RESTART = 1
_BAD_JACOBIAN = 2
_BAD_PROGRESS = 3
_HUGE_INITIAL_FORCES = 4

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


def solve_equilibrium(
    model: _vmecpp.VmecModel, *, style: str | None = None, verbose: bool = False
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
    fsqr = fsqz = fsql = float("nan")
    converged = False
    failed = False
    n_restarts = 0
    n_reguesses = 0

    # --- outer retry loop (Vmec::SolveEquilibrium) ---
    must_retry = True
    eqsolve_retries = 0
    while must_retry and eqsolve_retries < niterv and not converged and not failed:
        must_retry = False
        eqsolve_retries += 1

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
            status_bad_jacobian = (iter2 == 1 and rr == _BAD_JACOBIAN) or not finite

            # converged? (Evolve returns before time-stepping once converged)
            if (
                not status_bad_jacobian
                and fsqr <= ftolv
                and fsqz <= ftolv
                and (fsql <= ftolv)
            ):
                converged = True
                break

            # bad initial Jacobian / huge initial forces: recompute axis once
            if (
                ijacob == 0
                and (status_bad_jacobian or rr == _HUGE_INITIAL_FORCES)
                and model.ns >= 3
            ):
                # recompute the magnetic axis and re-initialize from it (with a
                # clean forward-model state), then restart the force iteration
                model.reinitialize()
                ijacob = 1
                n_reguesses += 1
                iter1 = 1
                iter2 = 1
                inv_tau[:] = 0.0
                fsq_prev = 1.0
                res0 = -1.0
                res1 = -1.0
                must_retry = True
                break
            if status_bad_jacobian:
                failed = True  # ijacob already used: cannot recover
                break

            # repeated bad Jacobian: reset the time step (ijacob escalation)
            if ijacob in (25, 50):
                model.restore_backup()
                ijacob += 1
                iter1 = iter2
                delt0r = (0.98 if ijacob < 50 else 0.96) * delt_user
                n_restarts += 1
                must_retry = True
                break
            if ijacob >= 75:
                failed = True
                break

            # --- Garabedian time step (shared; the damping is identical for all
            # styles, since VMEC++ has no 2D preconditioner so PARVMEC's bprec is
            # always 1) ---
            fsq1 = model.fsqr1 + model.fsqz1 + model.fsql1
            if iter2 == iter1:
                inv_tau[:] = 0.15 / delt0r
            inv_tau[:-1] = inv_tau[1:]
            if iter2 > iter1 and fsq1 != 0.0:
                inv_tau[-1] = min(abs(math.log(fsq1 / fsq_prev)), 0.15) / delt0r
            fsq_prev = fsq1

            trace_r.append(fsqr)
            trace_z.append(fsqz)
            trace_l.append(fsql)

            otav = inv_tau.sum() / _NDAMP
            dtau = delt0r * otav / 2.0
            model.perform_time_step(1.0 / (1.0 + dtau), 1.0 - dtau, delt0r)

            # --- time-step / restart control (style-dependent) ---
            if parvmec:
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
                elif (iter2 - iter1) > 10 and (
                    fsq1 > _PARVMEC_BLOWUP * res0 or fsq0 > _PARVMEC_BLOWUP * res1
                ):
                    model.restore_backup()
                    delt0r /= 1.03
                    iter1 = iter2
                    bad_resets += 1
                    n_restarts += 1
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
            else:
                # VMEC 8.52 TimeStepControl (reproduces the C++ inner solve): revert
                # and cut the step 10% (counting toward the give-up escalation) once
                # the preconditioned residual grows past 100x its minimum, plus a
                # slow-progress branch (/1.03).
                if iter2 == iter1 or res0 == -1.0:
                    res0 = fsq1
                res0 = min(res0, fsq1)
                if fsq1 <= res0 and (iter2 - iter1) > 10:
                    model.save_backup()
                elif fsq1 > _VMEC_8_52_BLOWUP * res0 and iter2 > iter1:
                    model.restore_backup()
                    delt0r *= 0.9
                    ijacob += 1
                    iter1 = iter2
                    bad_resets += 1
                    n_restarts += 1
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
    )


def iterate(
    vmec_input,
    ns: int | None = None,
    *,
    iteration_style: str | None = None,
    verbose: bool = False,
):
    """Create a single-resolution :class:`VmecModel` for ``vmec_input`` and run the
    Python force-balance iteration on it.

    ``iteration_style`` is one of ``"vmec_8_52"``, ``"parvmec"``, ``"robust"`` and
    overrides the input's own setting; when ``None`` the input's
    ``iteration_style`` is used (default ``"vmec_8_52"``). Returns
    ``(model, IterationResult)``; ``model`` holds the converged geometry (inspect
    it with ``get_state()`` / the residual properties). ``ns`` defaults to the
    last entry of the input's ``ns_array``.
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
    result = solve_equilibrium(model, style=iteration_style, verbose=verbose)
    return model, result

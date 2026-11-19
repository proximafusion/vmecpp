# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Optimize a fixed boundary and its VMEC state under the constraint of force balance.

Each Levenberg-Marquardt step solves::

    min |r + J_x dx + J_b db|^2 + damping |db|^2  subject to  F + H dx + F_b db = 0

over the state ``x`` and the boundary ``b``. ``r`` are the residuals, ``F`` is the force
on the interior of the state and ``H`` is its Jacobian. Eliminating the state gives
``dx = dn + T db`` with the normal step ``dn = -H^-1 F`` and the state tangents
``T = -H^-1 F_b``. The normal step is limited to a fraction of the major radius. The
merit of an iterate is ``|r + J_x dn|^2``. This is the objective with the force residual
removed to first order. VMEC's own descent then relaxes the state as far as the error of
the merit requires. A trial step whose ``|r|^2`` shows no clear drop takes ``dn`` from
GMRES on Hessian products at its own state. The factors of the last point precondition
GMRES. ``H`` is assembled only if that merit falls. The nested baseline converges the
state at every step instead.

A quantity ``c`` that has to stay between a lower and an upper bound joins the residuals
as ``weight (c + shift - z)``, with ``z`` the value within the bounds closest to
``c + shift``. Each step solves its least-squares problem for the boundary coefficients
within their own bounds and for every ``z`` within its bounds. The stages before the
last hold such a constraint with its weight alone. When the last stage has stalled,
each shift moves to ``c + shift - z``, which is the multiplier update of the augmented
Lagrangian, and the stage goes on until the shifts stop moving.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from vmecpp import _continuation, _iteration, _qs, autodiff, geometry
from vmecpp.cpp import _vmecpp  # type: ignore

NESTED_TOLERANCE = 1.0e-12
"""Force tolerance of the steps of the nested baseline and the tightest of any step."""

LOOSEST_TOLERANCE = 1.0e-5
"""Loosest force tolerance of a step."""

STAGE_DROP = 1.0e-2
"""Relative drop of the merit per step below which a stage has stalled."""

RELAXATION_ITERATIONS = 3000
"""Largest number of descent iterations of one relaxation between steps."""

CONSTRAINT_TOLERANCE = 1.0e-4
"""Move of a shift, relative to the size of the bounds of its constraint, below which
the constraints of a stage have converged."""

_FIRST_TOLERANCE = 1.0e-6
"""Force tolerance of the first state of a run."""

_NORMAL_RADIUS = 0.05
"""Largest normal step in one step as a fraction of the major radius."""

_SMALLEST_DROP = 1.0e-5
"""Predicted drop relative to the merit below which no trial step is taken."""

_FIRST_DAMPING = 1.0e-3
"""Damping at the start of a stage.

A damping above it limits the step.
"""

_CLEAR_DROP = 0.5
"""Drop of ``|r|^2`` relative to the predicted drop above which a trial step has its
force Jacobian assembled without a merit on the factors of the last point."""

_NORMAL_ERROR = 1.0e-2
"""Estimated relative error at which GMRES stops on a normal step."""

_NORMAL_ITERATIONS = 64
"""Largest number of GMRES iterations on a normal step."""

_TANGENT_CHUNK = 16
"""Residual tangents per call of the compiled derivative.

A fixed size compiles once.
"""

_SHIFT_UPDATES = 8
"""Largest number of updates of the shifts in one stage."""

_SLOW_SHIFT = 0.25
"""Distance of a constraint from its bounds, relative to that at the last update of its
shift, above which it takes ten times its weight."""

_STIFFNESS = 10.0
"""Curvature that a constraint outside its bounds adds along its gradient at the start
of a stage, relative to that of the residuals."""


def boundary_modes(max_mode: int, ntor: int) -> list[tuple[int, int, int]]:
    """Indices (row, m, ntor + n) of the free boundary coefficients.

    rbc(0, 0) is fixed.
    """
    return [
        (row, m, ntor + n)
        for row in (0, 1)
        for m in range(max_mode + 1)
        for n in range(-max_mode, max_mode + 1)
        if m > 0 or n > 0
    ]


def _resized(
    values: np.ndarray, ns: int, old: tuple[int, int], new: tuple[int, int]
) -> np.ndarray:
    """Blocks of (ns, mpol, ntor + 1) coefficients along the last axis on other modes.

    The modes that ``new`` adds to ``old`` are zero.
    """
    lead = values.shape[:-1]
    blocks = values.reshape(*lead, -1, ns, old[0], old[1] + 1)
    resized = np.zeros((*blocks.shape[:-2], new[0], new[1] + 1))
    mpol, columns = min(old[0], new[0]), min(old[1], new[1]) + 1
    resized[..., :mpol, :columns] = blocks[..., :mpol, :columns]
    return resized.reshape(*lead, -1)


def _resized_boundary(boundary: np.ndarray, mpol: int, ntor: int) -> np.ndarray:
    """Boundary coefficients of shape (..., mpol, 2 ntor + 1) on other modes."""
    boundary = np.asarray(boundary)
    resized = np.zeros((*boundary.shape[:-2], mpol, 2 * ntor + 1))
    rows = min(mpol, boundary.shape[-2])
    old_ntor = (boundary.shape[-1] - 1) // 2
    span = min(ntor, old_ntor)
    resized[..., :rows, ntor - span : ntor + span + 1] = boundary[
        ..., :rows, old_ntor - span : old_ntor + span + 1
    ]
    return resized


def _gmres(apply, precondition, rhs: np.ndarray) -> np.ndarray:
    """Right-preconditioned GMRES to _NORMAL_ERROR or to its last iterate.

    The preconditioned residual estimates the error. Along the soft modes of the
    operator the residual alone falls long before the error does.
    """
    norm = float(np.linalg.norm(rhs))
    basis = [rhs / norm]
    directions = []
    hessenberg = np.zeros((_NORMAL_ITERATIONS + 1, _NORMAL_ITERATIONS))
    weights = np.zeros(0)
    for k in range(_NORMAL_ITERATIONS):
        directions.append(precondition(basis[k]))
        image = apply(directions[k])
        if not np.all(np.isfinite(image)):
            message = "a Hessian product is not finite"
            raise ValueError(message)
        for i in range(k + 1):
            hessenberg[i, k] = image @ basis[i]
            image = image - hessenberg[i, k] * basis[i]
        hessenberg[k + 1, k] = np.linalg.norm(image)
        block = hessenberg[: k + 2, : k + 1]
        target = np.zeros(k + 2)
        target[0] = norm
        weights = np.linalg.lstsq(block, target, rcond=None)[0]
        if hessenberg[k + 1, k] == 0.0:
            break
        basis.append(image / hessenberg[k + 1, k])
        misfit = target - block @ weights
        if np.linalg.norm(misfit) <= _NORMAL_ERROR * norm:
            solution = np.stack(directions, axis=1) @ weights
            error = np.linalg.norm(precondition(np.stack(basis, axis=1) @ misfit))
            if error <= _NORMAL_ERROR * np.linalg.norm(solution):
                break
    return np.stack(directions, axis=1) @ weights


@dataclass
class Evaluation:
    """The residual vector at one state, its named parts and the values of the
    constraints.

    The residuals end with the weighted distance of each shifted constraint from its
    bounds.
    """

    residuals: np.ndarray
    parts: dict[str, float]
    constraints: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def merit(self) -> float:
        return float(self.residuals @ self.residuals)


@dataclass
class Costs:
    """Force evaluations, Hessian products of the normal steps, force Jacobians and wall
    time by kind of work."""

    force_evaluations: int = 0
    normal_products: int = 0
    assemblies: int = 0
    solve_seconds: float = 0.0
    assembly_seconds: float = 0.0
    tangent_seconds: float = 0.0
    objective_seconds: float = 0.0


@dataclass
class Shifts:
    """The shift and the weight of each constraint of a problem, and the distance of
    each from its bounds at the last update, relative to their size."""

    values: np.ndarray
    weights: np.ndarray
    violations: np.ndarray


@dataclass
class JointProblem:
    """A least-squares objective over a fixed boundary and its state.

    A subclass defines ``residuals``. JAX must be able to differentiate them. ``workers``
    is the number of threads for the force Jacobian and the residual derivatives. Its
    default is one per processor. ``exact_products`` takes the Hessian products of the
    force Jacobian from Enzyme in a build that has it. Otherwise they are central
    differences of the force. Both cost about the same. A multigrid input runs on its
    last resolution. ``parent`` is the problem whose compiled residuals a problem on
    fewer modes shares.

    A subclass that defines ``constraints`` keeps their values between ``lower`` and
    ``upper``, each without a bound by default. ``weights`` gives the weights that the
    constraints start with. Without it each stage starts every constraint at the weight
    that :meth:`balance` gives it. ``boundary_lower`` and ``boundary_upper`` bound the
    boundary coefficients, as rbc and then zbs. ``shifts`` holds the shifts and the
    weights of the constraints, which a run moves and a further run of the same problem
    starts from.
    """

    vmec_input: Any
    max_mode: int = 3
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)
    exact_products: bool = bool(_vmecpp.VMECPP_ENABLE_ENZYME)
    costs: Costs = field(default_factory=Costs)
    parent: Any = None
    lower: Any = None
    upper: Any = None
    weights: Any = None
    boundary_lower: Any = None
    boundary_upper: Any = None
    shifts: Any = None

    def __post_init__(self) -> None:
        if not jax.config.read("jax_enable_x64"):
            message = "vmecpp.joint needs jax_enable_x64 for its residuals"
            raise ValueError(message)
        if self.vmec_input.lfreeb or self.vmec_input.lasym:
            message = "vmecpp.joint needs a fixed-boundary, stellarator-symmetric input"
            raise ValueError(message)
        self.mpol, self.ntor = self.vmec_input.mpol, self.vmec_input.ntor
        if self.max_mode >= self.mpol or self.max_mode > self.ntor:
            message = "max_mode needs mpol above it and ntor at least as high"
            raise ValueError(message)
        given = [
            None if bound is None else np.atleast_1d(np.asarray(bound, dtype=float))
            for bound in (self.lower, self.upper)
        ]
        count = max((bound.size for bound in given if bound is not None), default=0)
        if any(bound is not None and bound.size not in (1, count) for bound in given):
            message = "the constraints need as many lower bounds as upper ones"
            raise ValueError(message)
        self.lower = np.full(count, -np.inf if given[0] is None else given[0])
        self.upper = np.full(count, np.inf if given[1] is None else given[1])
        if np.any(self.lower > self.upper):
            message = "the constraints need each lower bound at or below its upper one"
            raise ValueError(message)
        # the size of the bounds of each constraint, or one where they are zero
        sizes = [
            np.where(np.isfinite(b), np.abs(b), 0.0) for b in (self.lower, self.upper)
        ]
        self.scales = np.maximum(*sizes)
        self.scales[self.scales == 0.0] = 1.0
        if self.shifts is None:
            weights = np.ones(count) if self.weights is None else self.weights
            self.shifts = Shifts(
                np.zeros(count),
                np.broadcast_to(np.asarray(weights, dtype=float), count).copy(),
                np.full(count, np.inf),
            )
        self.template = self.vmec_input._to_cpp_vmecindata()
        self.template.always_fix_m1_gauge = True
        self.ns = int(np.asarray(self.vmec_input.ns_array)[-1])
        self.template.ns_array = np.asarray([self.ns])
        self.template.ftol_array = np.asarray(self.vmec_input.ftol_array)[-1:]
        self.template.niter_array = np.asarray(self.vmec_input.niter_array)[-1:]
        self.boundary0 = np.stack(
            [np.asarray(self.vmec_input.rbc), np.asarray(self.vmec_input.zbs)]
        )
        for name, open_bound in (
            ("boundary_lower", -np.inf),
            ("boundary_upper", np.inf),
        ):
            bound = open_bound if getattr(self, name) is None else getattr(self, name)
            setattr(
                self,
                name,
                np.broadcast_to(
                    np.asarray(bound, dtype=float), self.boundary0.shape
                ).copy(),
            )
        for key in boundary_modes(self.max_mode, self.ntor):
            if (
                not self.boundary_lower[key]
                <= self.boundary0[key]
                <= self.boundary_upper[key]
            ):
                message = "vmecpp.joint needs the free coefficients of the input within their bounds"
                raise ValueError(message)
        model = self.create_model(self.boundary0)
        model.evaluate(2, 2, True)
        if model.restart_reason == _iteration.RestartReason.BAD_JACOBIAN:
            # a first state whose surfaces cross has no force to probe
            model.reinitialize()
            model.evaluate(2, 2, True)
        interior, edge = autodiff._interior_and_boundary(model)
        gauge = autodiff._gauge_entries(model)
        self.prescribed = np.concatenate([edge, gauge])
        self.interior = autodiff._structural_nullfree_interior(
            model, np.setdiff1d(interior, gauge), exact=self.exact_products
        )
        base = np.array(self.create_model(self.boundary0).get_state())
        self.state_size = base.size
        # the boundary and gauge entries of the state are linear in the coefficients
        self.boundary_map = {}
        for key in boundary_modes(self.max_mode, self.ntor):
            shifted = self.boundary0.copy()
            shifted[key] += 1.0e-3
            column = np.zeros(base.size)
            column[self.prescribed] = (
                np.asarray(self.create_model(shifted).get_state()) - base
            )[self.prescribed] / 1.0e-3
            self.boundary_map[key] = column
        # each geometry coefficient is a multiple of state entries at its surface and mode
        span = self.ns * self.mpol * (self.ntor + 1)
        self.coefficient_map = np.stack(
            [
                np.asarray(
                    model.geometry_state_vjp(np.kron(np.eye(12)[block], np.ones(span)))
                ).reshape(-1, span)
                for block in range(12)
            ]
        )
        if self.parent is None:
            self._residuals = jax.jit(self._flat_residuals)
            self._linearized = jax.jit(self._linearized_residuals)
            self._along = jax.jit(self._residuals_along)
        else:
            # the residuals compile once, for the modes of the parent
            self._residuals = self.parent._residuals
            self._linearized = self.parent._linearized
            self._along = self.parent._along
        self._last: tuple[tuple[bytes, bytes], np.ndarray] | None = None

    def _at_resolution(self, mpol: int, ntor: int) -> JointProblem:
        """This problem on a state of ``mpol`` and ``ntor`` below its own.

        They hold every nonzero mode of the boundary. The angular grid is VMEC's default
        for them. The residuals are those of this problem, with zeros in the modes its
        state adds.
        """
        boundary = _resized_boundary(self.boundary0, mpol, ntor)
        reduced = _continuation._step_input(
            self.vmec_input.model_copy(update={"rbc": boundary[0], "zbs": boundary[1]}),
            self.ns,
            mpol,
            ntor,
            float(np.asarray(self.vmec_input.ftol_array)[-1]),
            int(np.asarray(self.vmec_input.niter_array)[-1]),
        )
        reduced.ntheta, reduced.nzeta = 0, 0
        return replace(
            self,
            vmec_input=reduced,
            max_mode=min(self.max_mode, mpol - 1, ntor),
            parent=self if self.parent is None else self.parent,
            boundary_lower=_resized_boundary(self.boundary_lower, mpol, ntor),
            boundary_upper=_resized_boundary(self.boundary_upper, mpol, ntor),
        )

    def _held_modes(self) -> tuple[int, int]:
        """The highest m and abs(n) that are nonzero in the input's boundary."""
        used = np.any(self.boundary0 != 0.0, axis=0)
        rows = np.flatnonzero(np.any(used, axis=1))
        columns = np.flatnonzero(np.any(used, axis=0))
        return (
            int(rows.max()) if rows.size else 0,
            int(np.abs(columns - self.ntor).max()) if columns.size else 0,
        )

    def _full_flat(self, flat: np.ndarray) -> np.ndarray:
        """Flat geometries along the last axis on the modes the residuals compile on."""
        if self.parent is None:
            return flat
        head = 2 * self.ns
        return np.concatenate(
            [
                flat[..., :head],
                _resized(
                    flat[..., head:],
                    self.ns,
                    (self.mpol, self.ntor),
                    (self.parent.mpol, self.parent.ntor),
                ),
            ],
            axis=-1,
        )

    def _full_boundary(self, boundary) -> np.ndarray:
        """Boundary coefficients on the modes of the compiled residuals."""
        if self.parent is None:
            return np.asarray(boundary)
        return _resized_boundary(boundary, self.parent.mpol, self.parent.ntor)

    def residuals(self, solved: geometry.Geometry, boundary: jax.Array) -> jax.Array:
        """The residual vector of a state.

        ``boundary`` holds rbc and then zbs.
        """
        raise NotImplementedError

    def constraints(self, solved: geometry.Geometry, boundary: jax.Array) -> jax.Array:  # noqa: ARG002
        """The quantities of a state that stay between ``lower`` and ``upper``."""
        return jnp.zeros(0)

    def parts(self, residuals: np.ndarray) -> dict[str, float]:  # noqa: ARG002
        """Named parts of the residual vector, without its constraints, for the log and
        the result."""
        return {}

    def _distance(self, constraints: np.ndarray) -> np.ndarray:
        """The distance of each shifted constraint from its bounds."""
        shifted = constraints + self.shifts.values
        return shifted - np.clip(shifted, self.lower, self.upper)

    def _penalized(self, raw: np.ndarray) -> np.ndarray:
        """Residuals followed by the values of the constraints, with each constraint as
        the weighted distance of its shifted value from its bounds."""
        residuals = np.array(raw, dtype=float)
        first = residuals.size - self.lower.size
        residuals[first:] = self.shifts.weights * self._distance(residuals[first:])
        return residuals

    def _outside(self, constraints: np.ndarray) -> np.ndarray:
        """The distance of each constraint from its bounds."""
        return constraints - np.clip(constraints, self.lower, self.upper)

    def violations(self, constraints: np.ndarray) -> np.ndarray:
        """The distance of each constraint from its bounds, relative to their size."""
        return np.abs(self._outside(constraints)) / self.scales

    def shift(self, constraints: np.ndarray, tolerance: float) -> np.ndarray | None:
        """Move the shift of each constraint to the distance of its shifted value from
        its bounds.

        The shifts stay as they are, and the result is None, when none would move by
        more than the tolerance times the size of the bounds of its constraint.
        Otherwise the result marks each constraint that lies further from its bounds
        than the tolerance and than _SLOW_SHIFT of its distance at the last update.
        """
        shifts = self.shifts
        distance = self._distance(constraints)
        if not np.any(np.abs(distance - shifts.values) > tolerance * self.scales):
            return None
        violations = self.violations(constraints)
        slow = (violations > tolerance) & (violations > _SLOW_SHIFT * shifts.violations)
        shifts.values[:] = distance
        shifts.violations[:] = violations
        return slow

    def stiffen(self, marked: np.ndarray) -> None:
        """Give the marked constraints ten times their weight at the same multiplier."""
        self.shifts.weights[marked] *= 10.0
        self.shifts.values[marked] /= 100.0

    def balance(self, jacobian: np.ndarray) -> bool:
        """Give each constraint the weight at which it adds _STIFFNESS times the
        curvature of the residuals along its own gradient, at the same multiplier, and
        tell whether any weight moved.

        ``jacobian`` holds the derivatives of the residuals and then of the
        constraints. The weights stay as they are when the problem was given them, and
        a constraint keeps its weight where it or the residuals do not depend on the
        coefficients.
        """
        if self.weights is not None or not self.lower.size:
            return False
        first = jacobian.shape[0] - self.lower.size
        normal = jacobian[:first].T @ jacobian[:first]
        if not normal.any():
            return False
        gradients = jacobian[first:]
        softness = np.einsum(
            "ij,ji->i",
            gradients,
            np.linalg.solve(
                normal + _FIRST_DAMPING * np.diag(_step_scale(normal)), gradients.T
            ),
        )
        moved = softness > 1.0e-12 * softness.max()
        weights = np.sqrt(_STIFFNESS / softness[moved])
        self.shifts.values[moved] *= (self.shifts.weights[moved] / weights) ** 2
        self.shifts.weights[moved] = weights
        return bool(moved.any())

    def multipliers(self) -> np.ndarray:
        """The Lagrange multiplier of each constraint for the sum of the squares of the
        residuals, as of the last update of its shift: positive at an upper bound and
        negative at a lower one."""
        return 2.0 * self.shifts.weights**2 * self.shifts.values

    def create_model(self, boundary, ftol=None, niter=None):
        indata = autodiff._make_indata(self.template, boundary)
        if ftol is not None:
            indata.ftol_array = np.asarray([float(ftol)])
        if niter is not None:
            indata.niter_array = np.asarray([int(niter)])
        return _vmecpp.VmecModel.create(indata, self.ns)

    def relax(self, boundary, state=None, ftol=None, niter=None):
        """VMEC's own descent from the state to the force tolerance."""
        start = time.perf_counter()
        model = self.create_model(boundary, ftol, niter)
        if state is not None:
            model.set_state(np.ascontiguousarray(state))
            # a rollback of the descent otherwise restores the cold initial guess
            model.save_backup()
        try:
            model.solve()
        finally:
            self.costs.force_evaluations += int(model.force_eval_count)
            self.costs.solve_seconds += time.perf_counter() - start
        return model

    def _models_at(self, model, state, *, precondition: bool) -> list:
        """The model and one more per further worker.

        All are evaluated at the state.
        """

        def at_state(index: int):
            worker = model if index == 0 else self.create_model(self.boundary0)
            worker.set_state(np.ascontiguousarray(state))
            worker.evaluate(2, 2, precondition)
            return worker

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(at_state, range(self.workers)))

    def _geometry(self, flat) -> geometry.Geometry:
        ns, modes = self.ns, self.mpol * (self.ntor + 1)
        coefficients = [
            flat[2 * ns + k * ns * modes : 2 * ns + (k + 1) * ns * modes].reshape(
                ns, self.mpol, self.ntor + 1
            )
            for k in range(12)
        ]
        return geometry.Geometry(
            flat[:ns], flat[ns : 2 * ns], *coefficients, nfp=self.vmec_input.nfp
        )

    def _flat_residuals(self, flat, boundary):
        """The residuals followed by the values of the constraints."""
        solved = self._geometry(flat)
        constraints = self.constraints(solved, boundary)
        if constraints.shape != self.lower.shape:
            message = (
                f"{constraints.shape[0]} constraints have {self.lower.size} bounds"
            )
            raise ValueError(message)
        return jnp.concatenate([self.residuals(solved, boundary), constraints])

    def _linearized_residuals(self, flat, boundary, directions, boundary_directions):
        _, linear = jax.linearize(self._flat_residuals, flat, boundary)
        return jax.vmap(linear)(directions, boundary_directions)

    def _residuals_along(self, flat, boundary, direction):
        return jax.jvp(
            lambda at: self._flat_residuals(at, boundary), (flat,), (direction,)
        )[1]

    def _flat_geometry(self, model) -> np.ndarray:
        return self._full_flat(
            autodiff._cpp_geometry_flat(
                model.get_geometry(), self.ns, self.mpol, self.ntor
            )
        )

    def _poloidal_flux_along(self, model, state, direction) -> np.ndarray:
        """Chi along a state direction by a central difference through the model."""
        step = 1.0e-7 / max(np.abs(direction).max(), 1.0e-3)
        sides = []
        for sign in (1.0, -1.0):
            model.set_state(np.ascontiguousarray(state + sign * step * direction))
            model.evaluate(2, 2, False)
            sides.append(np.asarray(model.get_geometry().poloidal_flux, dtype=float))
        return (sides[0] - sides[1]) / (2.0 * step)

    def _geometry_directions(self, models, state, directions) -> np.ndarray:
        """The flat geometry along each column of ``directions``, one row per column.

        The coefficients are linear in the state. With ncurr = 1, chi depends on the
        state and takes a difference through the models.
        """
        ns, count = self.ns, directions.shape[1]
        blocks, arrays, span = self.coefficient_map.shape
        parts = directions.reshape(arrays, span, count)
        coefficients = np.zeros((blocks, span, count))
        for block, array in np.argwhere(np.any(self.coefficient_map, axis=-1)):
            weights = self.coefficient_map[block, array, :, None]
            coefficients[block] += weights * parts[array]
        result = np.zeros((count, 2 * ns + blocks * span))
        result[:, 2 * ns :] = coefficients.transpose(2, 0, 1).reshape(count, -1)
        if self.vmec_input.ncurr != 0:
            result[:, ns : 2 * ns] = autodiff._map_on_models(
                lambda worker, i: self._poloidal_flux_along(
                    worker, state, directions[:, i]
                ),
                models,
                range(count),
            )
        return self._full_flat(result)

    def _evaluation(self, raw: np.ndarray) -> Evaluation:
        """The evaluation of residuals followed by the values of the constraints."""
        first = raw.size - self.lower.size
        parts = self.parts(raw[:first])
        if self.lower.size:
            parts["violation"] = float(self.violations(raw[first:]).max())
        return Evaluation(self._penalized(raw), parts, np.array(raw[first:]))

    def _values(self, flat: np.ndarray, boundary: np.ndarray) -> np.ndarray:
        """The residuals followed by the values of the constraints of a flat geometry.

        The last result is kept: the merits of a trial state and the derivatives at an
        accepted one take the residuals of the same state.
        """
        key = (flat.tobytes(), boundary.tobytes())
        if self._last is None or self._last[0] != key:
            values = np.asarray(self._residuals(flat, jnp.asarray(boundary)))
            self._last = (key, values)
        return self._last[1].copy()

    def evaluate(self, model, boundary) -> Evaluation:
        """The residuals of the state the model holds.

        The state need not be in force balance.
        """
        model.evaluate(2, 2, False)
        raw = self._values(self._flat_geometry(model), self._full_boundary(boundary))
        return self._evaluation(raw)

    def force_jacobian(self, model, state):
        """The factored H at the state from Hessian products on one model per worker."""
        start = time.perf_counter()
        models = self._models_at(model, state, precondition=True)
        matrix = autodiff._assemble_block_tridiagonal(
            model,
            self.interior,
            transposed=False,
            extra_models=models[1:],
            exact=self.exact_products,
        )
        matrix.factorize()
        self.costs.assembly_seconds += time.perf_counter() - start
        self.costs.assemblies += 1
        return matrix

    def state_tangents(self, model, state, keys, matrix=None) -> np.ndarray:
        """D state / D boundary at fixed force with one column per boundary mode."""
        if matrix is None:
            matrix = self.force_jacobian(model, state)
        start = time.perf_counter()
        models = self._models_at(model, state, precondition=True)
        tangents = np.stack([self.boundary_map[key] for key in keys], axis=1)
        forcing = autodiff._map_on_models(
            lambda worker, i: autodiff._force_product(
                worker, tangents[:, i], exact=self.exact_products
            )[self.interior],
            models,
            range(len(keys)),
        )
        tangents[self.interior, :] = matrix.solve(-np.stack(forcing, axis=1))
        self.costs.tangent_seconds += time.perf_counter() - start
        return tangents

    def residual_jacobian(self, model, state, boundary, keys, tangents):
        """The evaluation and the derivatives of the residuals, then of the constraints,
        along the state tangents."""
        start = time.perf_counter()
        state = np.ascontiguousarray(state)
        models = self._models_at(model, state, precondition=False)
        directions = self._geometry_directions(models, state, tangents)
        model.set_state(state)
        model.evaluate(2, 2, False)
        flat = self._flat_geometry(model)
        boundary = self._full_boundary(boundary)
        shift = (boundary.shape[-1] - 1) // 2 - self.ntor
        boundary_directions = np.zeros((len(keys), *boundary.shape))
        for i, (row, m, column) in enumerate(keys):
            boundary_directions[i, row, m, column + shift] = 1.0
        values = self._values(flat, boundary)

        def linearized(first: int) -> np.ndarray:
            chunk = range(first, min(first + _TANGENT_CHUNK, len(keys)))
            padding = _TANGENT_CHUNK - len(chunk)
            return np.asarray(
                self._linearized(
                    flat,
                    jnp.asarray(boundary),
                    jnp.asarray(_padded(directions[chunk], padding)),
                    jnp.asarray(_padded(boundary_directions[chunk], padding)),
                )
            )[: len(chunk)]

        firsts = range(0, len(keys), _TANGENT_CHUNK)
        with ThreadPoolExecutor(min(self.workers, len(firsts))) as pool:
            rows = list(pool.map(linearized, firsts))
        self.costs.objective_seconds += time.perf_counter() - start
        return self._evaluation(values), np.vstack(rows).T

    def _normal_step(self, model, state, force, matrix) -> np.ndarray:
        """``-H^-1 F`` on the interior by GMRES on products at the state."""
        model.evaluate(2, 2, True)

        def apply(vector: np.ndarray) -> np.ndarray:
            self.costs.normal_products += 1
            full = np.zeros(state.size)
            full[self.interior] = vector
            return autodiff._force_product(model, full, exact=self.exact_products)[
                self.interior
            ]

        return _gmres(apply, lambda vector: matrix.solve(vector[:, None])[:, 0], -force)

    def balanced_residuals(self, boundary, state, matrix):
        """The residuals ``r`` at the state, ``r + J_x dn`` and ``-H^-1 F``.

        ``r + J_x dn`` are the residuals after the normal step to first order with ``dn``
        within its radius. ``matrix`` holds the factors of this state or a nearby one.
        The values of the constraints follow the residuals in both.
        """
        start = time.perf_counter()
        state = np.ascontiguousarray(state)
        model = self.create_model(boundary)
        model.set_state(state)
        model.evaluate(2, 2, False)
        flat = self._flat_geometry(model)
        force = np.array(model.get_forces())[self.interior]
        full = self._full_boundary(boundary)
        wide = jnp.asarray(full)
        values = self._values(flat, full)
        normal = np.zeros(state.size)
        try:
            normal[self.interior] = self._normal_step(model, state, force, matrix)
        except ValueError:  # the force, the factors or a product are not finite
            normal[:] = np.nan
        change = np.full_like(values, np.nan)
        if np.all(np.isfinite(normal)):
            direction = self._geometry_directions([model], state, normal[:, None])[0]
            change = np.asarray(self._along(flat, wide, jnp.asarray(direction)))
        self.costs.objective_seconds += time.perf_counter() - start
        return values, values + _normal_fraction(boundary, normal) * change, normal


def _padded(rows: np.ndarray, padding: int) -> np.ndarray:
    return np.concatenate([rows, np.zeros((padding, *rows.shape[1:]))])


def _normal_fraction(boundary: np.ndarray, normal: np.ndarray) -> float:
    """The fraction of a normal step within its radius."""
    radius = _NORMAL_RADIUS * np.abs(boundary[0]).max()
    return min(1.0, radius / max(float(np.abs(normal).max()), 1.0e-300))


@dataclass(kw_only=True)
class QuasisymmetryProblem(JointProblem):
    """SIMSOPT's QuasisymmetryRatioResidual with aspect-ratio and mean-iota targets as
    residuals of unit weight."""

    surfaces: tuple[float, ...] = tuple(np.arange(1, 11) / 10.0)
    helicity_m: int = 1
    helicity_n: int = 0
    aspect: float = 6.0
    mean_iota: float = 0.42
    ntheta: int = 32
    nphi: int = 32

    def _aspect(self, boundary):
        """The aspect ratio of the boundary by VMEC's definition."""
        theta = 2.0 * jnp.pi * jnp.arange(64) / 64
        zeta = 2.0 * jnp.pi * jnp.arange(64) / (64 * self.vmec_input.nfp)
        m = jnp.arange(self.mpol, dtype=jnp.float64)
        n = jnp.arange(-self.ntor, self.ntor + 1, dtype=jnp.float64)
        angle = (
            m[:, None, None, None] * theta[None, None, :, None]
            - n[None, :, None, None] * self.vmec_input.nfp * zeta[None, None, None, :]
        )
        cosine = jnp.cos(angle)
        r = jnp.einsum("mn,mntz->tz", boundary[0], cosine)
        z_theta = jnp.einsum("mn,mntz->tz", boundary[1] * m[:, None], cosine)
        area = jnp.abs(2.0 * jnp.pi * jnp.mean(r * z_theta))
        volume = jnp.abs(4.0 * jnp.pi**2 * jnp.mean(0.5 * r**2 * z_theta))
        minor_radius = jnp.sqrt(area / jnp.pi)
        return volume / (2.0 * jnp.pi**2 * minor_radius**3)

    def residuals(self, solved: geometry.Geometry, boundary: jax.Array) -> jax.Array:
        """Aspect ratio and mean iota against their targets followed by the residuals of
        SIMSOPT's QuasisymmetryRatioResidual."""
        quasisymmetry = _qs.quasisymmetry_residuals(
            solved,
            self.surfaces,
            helicity_m=self.helicity_m,
            helicity_n=self.helicity_n,
            ntheta=self.ntheta,
            nphi=self.nphi,
        )
        mean_iota = (solved.poloidal_flux[-1] - solved.poloidal_flux[0]) / (
            solved.toroidal_flux[-1] - solved.toroidal_flux[0]
        )
        targets = jnp.stack(
            [self._aspect(boundary) - self.aspect, mean_iota - self.mean_iota]
        )
        return jnp.concatenate([targets, quasisymmetry])

    def parts(self, residuals: np.ndarray) -> dict[str, float]:
        return {
            "aspect": float(residuals[0]) + self.aspect,
            "mean_iota": float(residuals[1]) + self.mean_iota,
            "quasisymmetry": float(residuals[2:] @ residuals[2:]),
        }


@dataclass
class Step:
    """One accepted step of the optimization.

    ``evaluation`` holds its merit.
    """

    max_mode: int
    evaluation: Evaluation
    force_residual: float
    force_evaluations: int
    seconds: float


@dataclass
class Stage:
    """How one stage of boundary modes ended.

    ``stopped`` is "stalled" after three steps in a row below STAGE_DROP that the damping
    did not limit. It is "no_step" when no trial step predicts a drop or when none holds
    even from the state in force balance. It is "max_steps" at the step limit.
    ``rejected_steps`` counts the trial steps refused for their merit or force residual.
    ``failed_solves`` counts those on which VMEC raised. ``tightened`` counts the
    relaxations further for the accuracy of a merit. ``shift_updates`` counts the
    updates of the shifts of the constraints. ``mpol`` and ``ntor`` are those of the
    state the stage ran on.
    """

    max_mode: int
    stopped: str = "max_steps"
    rejected_steps: int = 0
    failed_solves: int = 0
    tightened: int = 0
    mpol: int = 0
    ntor: int = 0
    shift_updates: int = 0


@dataclass
class Result:
    """The final boundary, its state relaxed to NESTED_TOLERANCE with its residuals and
    force residual, every accepted step, how each stage ended and the Lagrange
    multipliers of the constraints."""

    boundary: np.ndarray
    state: np.ndarray
    evaluation: Evaluation
    force_residual: float
    steps: list[Step]
    stages: list[Stage]
    multipliers: np.ndarray


@dataclass
class _Point:
    """A boundary and state, the residuals there and their value in force balance.

    ``matrix`` is the factored force Jacobian of this state or of the last accepted one.
    The nested baseline takes the residuals as they are. It factors the Jacobian only for
    an accepted step. ``relaxable`` is false once a further relaxation of the point has
    failed. ``constraints`` holds the values of the constraints in force balance,
    ``plain`` those at the state as it is and ``outside`` the weighted distance of each
    from its bounds in force balance.
    """

    boundary: np.ndarray
    state: np.ndarray
    residuals: np.ndarray
    balanced: np.ndarray
    normal: np.ndarray | None = None
    matrix: Any = None
    relaxable: bool = True
    constraints: np.ndarray = field(default_factory=lambda: np.zeros(0))
    plain: np.ndarray = field(default_factory=lambda: np.zeros(0))
    outside: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def merit(self) -> float:
        return float(self.balanced @ self.balanced)

    @property
    def size(self) -> float:
        """The merit without the shifts of the constraints, which is the scale of a drop
        of the merit.

        The merit itself holds the square of each shift at a bound.
        """
        first = self.balanced.size - self.constraints.size
        return float(
            self.balanced[:first] @ self.balanced[:first] + self.outside @ self.outside
        )

    @property
    def raw(self) -> np.ndarray:
        """The residuals in force balance followed by the values of the constraints."""
        first = self.balanced.size - self.constraints.size
        return np.concatenate([self.balanced[:first], self.constraints])

    @property
    def normal_size(self) -> float:
        return 0.0 if self.normal is None else float(np.abs(self.normal).max())

    def normal_step(self) -> np.ndarray:
        """The normal step within its radius."""
        if self.normal is None or not np.isfinite(self.normal_size):
            return np.zeros_like(self.state)
        return _normal_fraction(self.boundary, self.normal) * self.normal

    @property
    def merit_error(self) -> float:
        """Estimated error ``2 |r_b| |J_x dn| |dn| / R`` of the merit for the whole of
        ``dn = -H^-1 F``."""
        if self.normal is None:
            return 0.0
        change = np.linalg.norm(self.balanced - self.residuals) / _normal_fraction(
            self.boundary, self.normal
        )
        return float(
            2.0
            * np.linalg.norm(self.balanced)
            * change
            * self.normal_size
            / np.abs(self.boundary[0]).max()
        )


@dataclass
class _Iterate:
    """The current point, the force tolerance of the relaxations, its ceiling and the
    damping."""

    point: _Point
    tolerance: float
    ceiling: float = LOOSEST_TOLERANCE
    damping: float = _FIRST_DAMPING


@dataclass
class _Trial:
    """An accepted trial step."""

    point: _Point
    force_residual: float
    gain: float
    drop: float
    allowed_error: float


def _penalized(problem: JointProblem, values, balanced) -> dict[str, Any]:
    """The residuals and the constraints of a point as they are and in force balance,
    from the residuals of each followed by the values of the constraints."""
    first = balanced.size - problem.lower.size
    return {
        "residuals": problem._penalized(values),
        "balanced": problem._penalized(balanced),
        "constraints": np.array(balanced[first:]),
        "plain": np.array(values[first:]),
        "outside": problem.shifts.weights * problem._outside(balanced[first:]),
    }


def _on_shifts(problem: JointProblem, point: _Point) -> _Point:
    """The point with its residuals on the shifts and the weights that the constraints
    have now."""
    first = point.balanced.size - problem.lower.size
    values = np.concatenate([point.residuals[:first], point.plain])
    return replace(point, **_penalized(problem, values, point.raw))


def _joint_point(problem: JointProblem, boundary, state, matrix=None) -> _Point:
    """The point with its merit on the given factors or on its own."""
    state = np.ascontiguousarray(state)
    if matrix is None:
        matrix = problem.force_jacobian(problem.create_model(boundary), state)
    values, balanced, normal = problem.balanced_residuals(boundary, state, matrix)
    return _Point(
        boundary,
        state,
        normal=normal,
        matrix=matrix,
        **_penalized(problem, values, balanced),
    )


def _nested_point(problem: JointProblem, boundary, model) -> _Point:
    state = np.ascontiguousarray(np.array(model.get_state()))
    evaluation = problem.evaluate(model, boundary)
    return _Point(
        boundary,
        state,
        evaluation.residuals,
        evaluation.residuals,
        constraints=evaluation.constraints,
        plain=evaluation.constraints,
        outside=problem.shifts.weights * problem._outside(evaluation.constraints),
    )


def _relaxed_further(
    problem: JointProblem, point: _Point, tolerance: float
) -> _Point | None:
    """The point relaxed to a tighter tolerance on its factored Jacobian or None.

    The result is None where VMEC raises or the tighter state has no finite merit. A
    surface whose Jacobian changes sign has no finite merit.
    """
    try:
        model = problem.relax(
            point.boundary, point.state, tolerance, RELAXATION_ITERATIONS
        )
    except RuntimeError:
        return None
    state = np.ascontiguousarray(np.array(model.get_state()))
    if point.matrix is None:
        further = _nested_point(problem, point.boundary, model)
    else:
        values, balanced, normal = problem.balanced_residuals(
            point.boundary, state, point.matrix
        )
        further = replace(
            point, state=state, normal=normal, **_penalized(problem, values, balanced)
        )
    return further if np.isfinite(further.merit) else None


def _decade_tighter(
    problem: JointProblem, point: _Point, tolerance: float, end: Stage
) -> tuple[_Point, float]:
    """The point relaxed one decade further and its tolerance or the point marked as not
    relaxable."""
    tighter = max(tolerance / 10.0, NESTED_TOLERANCE)
    further = _relaxed_further(problem, point, tighter)
    if further is None:
        return replace(point, relaxable=False), tolerance
    end.tightened += 1
    return further, tighter


def _tightened(
    problem: JointProblem, point: _Point, tolerance: float, allowed: float, end: Stage
) -> tuple[_Point, float]:
    """The point and its tolerance after relaxing until the error of its merit is within
    the allowance."""
    while (
        point.relaxable and point.merit_error > allowed and tolerance > NESTED_TOLERANCE
    ):
        point, tolerance = _decade_tighter(problem, point, tolerance, end)
    return point, tolerance


def _restored(problem: JointProblem, point: _Point, end: Stage) -> _Point:
    """The point in force balance with its Jacobian factored there."""
    try:
        model = problem.relax(
            point.boundary,
            point.state + point.normal_step(),
            NESTED_TOLERANCE,
            RELAXATION_ITERATIONS,
        )
    except RuntimeError:
        return point
    end.tightened += 1
    restored = _joint_point(problem, point.boundary, model.get_state())
    return restored if np.isfinite(restored.merit) else point


def _carried(
    old: JointProblem,
    new: JointProblem,
    point: _Point,
    tolerance: float,
    *,
    nested: bool,
) -> _Point:
    """The point on the modes of another problem, relaxed there to the tolerance."""
    boundary = _resized_boundary(point.boundary, new.mpol, new.ntor)
    sizes = (old.mpol, old.ntor), (new.mpol, new.ntor)
    niter = None if nested or tolerance <= NESTED_TOLERANCE else RELAXATION_ITERATIONS
    model = None
    for state in (point.state + point.normal_step(), point.state):
        try:
            model = new.relax(
                boundary, _resized(state, new.ns, *sizes), tolerance, niter
            )
            break
        except RuntimeError:  # the state crosses flux surfaces on the new modes
            continue
    if model is None:
        model = new.relax(boundary, ftol=tolerance, niter=niter)
    if nested:
        return _nested_point(new, boundary, model)
    return _joint_point(new, boundary, model.get_state())


def _step_scale(normal: np.ndarray) -> np.ndarray:
    """The diagonal that the damping of a step multiplies.

    Its floor bounds the step in weakly coupled modes when residuals are few.
    """
    return np.maximum(np.diag(normal), max(1.0e-4 * np.diag(normal).max(), 1.0e-10))


def _free_step(point: _Point, jacobian, damping: float) -> tuple[np.ndarray, float]:
    """The damped step of the boundary coefficients of a problem without constraints or
    bounds and the drop of the merit that it predicts."""
    normal = jacobian.T @ jacobian
    delta = -np.linalg.solve(
        normal + damping * np.diag(_step_scale(normal)), jacobian.T @ point.balanced
    )
    return delta, point.merit - float(np.sum((point.balanced + jacobian @ delta) ** 2))


def _ridge(
    problem: JointProblem, point: _Point, jacobian, damping: float
) -> np.ndarray:
    """The damping of a step by boundary coefficient, from the residuals and the
    constraints that lie outside their bounds."""
    first = jacobian.shape[0] - problem.lower.size
    slopes = problem.shifts.weights[:, None] * jacobian[first:]
    outside = problem._distance(point.constraints) != 0.0
    return damping * _step_scale(
        jacobian[:first].T @ jacobian[:first] + slopes[outside].T @ slopes[outside]
    )


def _triangle(problem: JointProblem, point: _Point, jacobian) -> np.ndarray:
    """The residuals of a point and their derivatives as a triangle of the same sum of
    squares."""
    first = jacobian.shape[0] - problem.lower.size
    return np.asarray(
        np.linalg.qr(
            np.column_stack([jacobian[:first], point.balanced[:first]]), mode="r"
        )
    )


def _bounded_step(
    problem: JointProblem,
    point: _Point,
    keys,
    jacobian,
    damping: float,
    triangle: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """The damped step of the boundary coefficients within their bounds and the drop of
    the merit that it predicts.

    Its least-squares problem holds each constraint with two different bounds at a value
    between them, which it solves for with the step. A constraint that the step leaves
    between its bounds adds nothing, so the problem holds those at or outside their
    bounds and is solved again with each that its step moves outside, until none is.
    ``triangle`` is that of :func:`_triangle` for the point, when it is at hand.
    """
    size = len(keys)
    first = jacobian.shape[0] - problem.lower.size
    weights, slopes = problem.shifts.weights, jacobian[first:]
    if triangle is None:
        triangle = _triangle(problem, point, jacobian)
    rows = triangle.shape[0]
    held = rows + size
    top = np.zeros((held, size))
    top[:rows] = triangle[:, :size]
    top[rows:] = np.diag(np.sqrt(_ridge(problem, point, jacobian, damping)))
    shifted = point.constraints + problem.shifts.values
    at = np.asarray([point.boundary[key] for key in keys])
    lowest = np.asarray([problem.boundary_lower[key] for key in keys]) - at
    highest = np.asarray([problem.boundary_upper[key] for key in keys]) - at
    kept = (shifted <= problem.lower) | (shifted >= problem.upper)
    while True:
        index = np.flatnonzero(kept)
        lower, upper = problem.lower[index], problem.upper[index]
        between = lower < upper
        matrix = np.zeros((held + index.size, size + int(between.sum())))
        target = np.zeros(held + index.size)
        matrix[:held, :size] = top
        target[:rows] = -triangle[:, size]
        matrix[held:, :size] = weights[index, None] * slopes[index]
        matrix[
            held + np.flatnonzero(between), size + np.arange(between.sum())
        ] = -weights[index][between]
        target[held:] = -weights[index] * (
            shifted[index] - np.where(between, 0.0, lower)
        )
        solution = scipy.optimize.lsq_linear(
            matrix,
            target,
            bounds=(
                np.concatenate([lowest, lower[between]]),
                np.concatenate([highest, upper[between]]),
            ),
            method="bvls",
        ).x
        moved = shifted + slopes @ solution[:size]
        outside = ~kept & ((moved < problem.lower) | (moved > problem.upper))
        if not outside.any():
            break
        kept |= outside
    misfit = matrix @ solution - target
    misfit[rows:held] = 0.0
    return solution[:size], point.merit - float(misfit @ misfit)


def _first_accepted_step(
    problem: JointProblem,
    iterate: _Iterate,
    keys,
    tangents,
    jacobian,
    end: Stage,
    *,
    nested: bool,
    error_fraction: float,
) -> _Trial | None:
    """The first trial step that holds or None after twelve."""
    bounded = any(
        np.isfinite(problem.boundary_lower[key])
        or np.isfinite(problem.boundary_upper[key])
        for key in keys
    )
    constrained = bounded or problem.lower.size > 0
    # the triangle of the last point, which the trials of one point share
    shared: list[Any] = [None, None]

    def proposal():
        point = iterate.point
        if constrained:
            if shared[0] is not point:
                shared[:] = point, _triangle(problem, point, jacobian)
            delta, predicted = _bounded_step(
                problem, point, keys, jacobian, iterate.damping, shared[1]
            )
        else:
            delta, predicted = _free_step(point, jacobian, iterate.damping)
        return (
            delta,
            predicted,
            error_fraction * max(predicted, STAGE_DROP * point.size),
        )

    take_normal = not nested
    for _ in range(12):
        delta, predicted, allowed = proposal()
        if predicted < _SMALLEST_DROP * iterate.point.size:
            return None
        if not nested and iterate.point.merit_error > allowed:
            iterate.point, iterate.tolerance = _tightened(
                problem, iterate.point, iterate.tolerance, allowed, end
            )
            delta, predicted, allowed = proposal()
        point = iterate.point
        trial = point.boundary.copy()
        for key, value in zip(keys, delta, strict=True):
            trial[key] += value
            if bounded:
                # rounding can leave a coefficient just outside the bound it reached
                trial[key] = np.clip(
                    trial[key], problem.boundary_lower[key], problem.boundary_upper[key]
                )
        moved = point.state + tangents @ delta
        if take_normal:
            moved = moved + point.normal_step()
        try:
            relaxed = problem.relax(
                trial,
                moved,
                iterate.tolerance,
                None if nested else RELAXATION_ITERATIONS,
            )
        except RuntimeError:  # the step crossed flux surfaces
            end.failed_solves += 1
            iterate.damping *= 4.0
            take_normal = False
            continue
        force_residual = relaxed.fsqr + relaxed.fsqz + relaxed.fsql
        candidate = _nested_point(problem, trial, relaxed)
        # the descent stayed near force balance
        holds = force_residual <= max(30.0 * iterate.tolerance, 1.0e-4)
        if holds and not nested:
            plain = float(point.residuals @ point.residuals)
            holds = candidate.merit <= (
                plain + abs(plain - point.merit) + 0.25 * predicted
            )
            if holds:
                # a clear drop of |r|^2 skips the merit on the last point's factors
                clear = point.merit - candidate.merit > _CLEAR_DROP * predicted
                candidate = _joint_point(
                    problem, trial, candidate.state, None if clear else point.matrix
                )
                candidate, iterate.tolerance = _tightened(
                    problem, candidate, iterate.tolerance, allowed, end
                )
        if holds:
            gain = (point.merit - candidate.merit) / max(predicted, 1.0e-300)
            if not nested and gain > 1.0e-3 and candidate.matrix is point.matrix:
                # the merit on the trial state's own factors decides
                factored = _joint_point(problem, trial, candidate.state)
                if np.isfinite(factored.merit):
                    candidate = factored
                    gain = (point.merit - candidate.merit) / max(predicted, 1.0e-300)
            if gain > 1.0e-3:
                drop = (point.merit - candidate.merit) / max(point.size, 1.0e-300)
                return _Trial(candidate, force_residual, gain, drop, allowed)
            if not nested and point.relaxable and iterate.tolerance > NESTED_TOLERANCE:
                # the merit at this tolerance need not resolve the drop
                iterate.point, iterate.tolerance = _decade_tighter(
                    problem, point, iterate.tolerance, end
                )
                if iterate.point.relaxable:
                    iterate.ceiling = iterate.tolerance
        end.rejected_steps += 1
        iterate.damping *= 4.0
        take_normal = take_normal and holds
    return None


def _moved_shifts(
    problem: JointProblem,
    iterate: _Iterate,
    end: Stage,
    tolerance: float,
) -> bool:
    """Whether the shifts of the constraints moved at the point of a stage that has
    stalled.

    A constraint further from its bounds than the tolerance takes ten times its weight
    when the last update of its shift has not brought it to _SLOW_SHIFT of its distance,
    or when its weighted distance is smaller than the residuals, so that a step of the
    stage does not resolve it.
    """
    point = iterate.point
    if end.shift_updates >= _SHIFT_UPDATES:
        return False
    slow = problem.shift(point.constraints, tolerance)
    if slow is None:
        return False
    end.shift_updates += 1
    first = point.balanced.size - point.constraints.size
    faint = point.outside**2 < point.balanced[:first] @ point.balanced[:first]
    far = problem.violations(point.constraints) > tolerance
    problem.stiffen(slow | (far & faint))
    iterate.point = _on_shifts(problem, point)
    return True


@autodiff._BLAS.wrap(limits=autodiff._BLAS_THREADS, user_api="blas")
def optimize(
    problem: JointProblem,
    stages=(1, 2, 3),
    *,
    nested: bool = False,
    error_fraction: float = 0.01,
    max_steps: int = 120,
    margin: int | None = None,
    constraint_tolerance: float = CONSTRAINT_TOLERANCE,
    log=None,
) -> Result:
    """Levenberg-Marquardt on the boundary and the state with boundary modes added stage
    by stage.

    Each trial step relaxes the state until the estimated error of the merit is
    ``error_fraction`` of the larger of the drop the step predicts and STAGE_DROP times
    the merit. The force tolerance stays between NESTED_TOLERANCE and LOOSEST_TOLERANCE.
    The last stage starts in force balance and relaxes every step to NESTED_TOLERANCE.
    The final state is relaxed to NESTED_TOLERANCE. ``nested`` instead converges every
    step to NESTED_TOLERANCE and takes the residuals as they are. With ``margin``, every
    stage but the last runs on a state whose ``mpol`` and ``ntor`` are that many above
    the highest mode of its boundary, up to the input's, and the last on the input's.
    The last stage, once it has stalled, moves the shifts of the constraints of the
    problem and goes on, until none moves by more than ``constraint_tolerance`` times
    the size of the bounds of its constraint. BLAS runs on one thread during the run.
    Residuals that are not finite at the first state raise a ValueError.
    """
    if max(stages) > problem.max_mode:
        message = f"stage {max(stages)} is above max_mode {problem.max_mode}"
        raise ValueError(message)
    if margin is not None and margin < 1:
        message = f"margin needs to be at least 1, but is {margin}"
        raise ValueError(message)
    start = time.perf_counter()
    reduced: dict[tuple[int, int], JointProblem] = {}

    def staged(index: int) -> JointProblem:
        """The problem on the modes of a stage."""
        if margin is None or index == len(stages) - 1:
            return problem
        held_m, held_n = problem._held_modes()
        mpol = min(problem.mpol, max(stages[index], held_m) + margin)
        ntor = min(problem.ntor, max(stages[index], held_n) + margin)
        if (mpol, ntor) == (problem.mpol, problem.ntor):
            return problem
        if (mpol, ntor) not in reduced:
            reduced[mpol, ntor] = problem._at_resolution(mpol, ntor)
        return reduced[mpol, ntor]

    active = staged(0)
    boundary = active.boundary0.copy()
    model = active.relax(boundary, ftol=_FIRST_TOLERANCE)
    if nested:
        first = _nested_point(active, boundary, model)
        if np.isfinite(first.merit):
            first = _relaxed_further(active, first, NESTED_TOLERANCE) or first
        iterate = _Iterate(first, NESTED_TOLERANCE)
    else:
        first = _joint_point(active, boundary, model.get_state())
        iterate = _Iterate(first, _FIRST_TOLERANCE)
    if not np.isfinite(iterate.point.merit):
        message = "the residuals are not finite at the first state"
        raise ValueError(message)
    steps: list[Step] = []
    ends: list[Stage] = []
    for index, max_mode in enumerate(stages):
        last = index == len(stages) - 1
        following = staged(index)
        end = Stage(max_mode, mpol=following.mpol, ntor=following.ntor)
        ends.append(end)
        iterate.damping = _FIRST_DAMPING
        iterate.ceiling = LOOSEST_TOLERANCE
        if following is not active:
            iterate.point = _carried(
                active,
                following,
                iterate.point,
                NESTED_TOLERANCE if nested or last else iterate.tolerance,
                nested=nested,
            )
            active = following
        elif not nested and last:
            # the last stage takes its derivatives in force balance
            iterate.point = _restored(active, iterate.point, end)
        if not nested and last:
            iterate.tolerance = iterate.ceiling = NESTED_TOLERANCE
        keys = boundary_modes(max_mode, active.ntor)
        slow = 0
        restored = False
        weighed = False
        for _ in range(max_steps):
            point = iterate.point
            model = active.create_model(point.boundary)
            tangents = active.state_tangents(model, point.state, keys, point.matrix)
            _, jacobian = active.residual_jacobian(
                model, point.state, point.boundary, keys, tangents
            )
            if not weighed:
                # the modes of a stage decide how stiff a weight makes a constraint
                if active.balance(jacobian):
                    iterate.point = _on_shifts(active, point)
                weighed = True
            refused = end.rejected_steps + end.failed_solves
            trial = _first_accepted_step(
                active,
                iterate,
                keys,
                tangents,
                jacobian,
                end,
                nested=nested,
                error_fraction=error_fraction,
            )
            if trial is None:
                if (
                    nested
                    or restored
                    or refused == (end.rejected_steps + end.failed_solves)
                ):
                    if last and _moved_shifts(
                        active, iterate, end, constraint_tolerance
                    ):
                        restored = False
                        slow = 0
                        iterate.damping = _FIRST_DAMPING
                        continue
                    end.stopped = "no_step"
                    break
                # the merit of a loose state can hide a drop smaller than its error
                iterate.point = _restored(active, iterate.point, end)
                iterate.tolerance = NESTED_TOLERANCE
                iterate.damping = _FIRST_DAMPING
                restored = True
                continue
            restored = False
            limited = iterate.damping > _FIRST_DAMPING and trial.gain > 0.5
            if trial.gain > 0.5:
                iterate.damping = max(
                    iterate.damping / (10.0 if limited else 3.0), 1.0e-9
                )
            elif trial.gain < 0.25:
                iterate.damping *= 2.0
            iterate.point = trial.point
            if not nested:
                # the error scales with the force tolerance in the stiff modes
                iterate.tolerance = float(
                    np.clip(
                        0.5
                        * iterate.tolerance
                        * trial.allowed_error
                        / max(trial.point.merit_error, 1.0e-300),
                        iterate.tolerance,
                        min(10.0 * iterate.tolerance, iterate.ceiling),
                    )
                )
            evaluation = active._evaluation(trial.point.raw)
            steps.append(
                Step(
                    max_mode,
                    evaluation,
                    trial.force_residual,
                    problem.costs.force_evaluations,
                    time.perf_counter() - start,
                )
            )
            if log is not None:
                named = "".join(
                    f" {name} {value:.5g}" for name, value in evaluation.parts.items()
                )
                log(
                    f"modes {max_mode} step {len(steps):3d}"
                    f" merit {evaluation.merit:.4e}{named}"
                    f" force {trial.force_residual:.1e}"
                    f" evals {problem.costs.force_evaluations}"
                )
            # a small drop from a step the damping limits is no stall
            if trial.drop >= STAGE_DROP:
                slow = 0
            elif not limited:
                slow += 1
            if slow >= 3:
                if last and _moved_shifts(active, iterate, end, constraint_tolerance):
                    slow = 0
                    continue
                end.stopped = "stalled"
                break
        if log is not None:
            shifts = f", {end.shift_updates} shift updates" if active.lower.size else ""
            log(
                f"modes {max_mode} stopped: {end.stopped},"
                f" {end.rejected_steps} rejected steps,"
                f" {end.failed_solves} failed solves,"
                f" {end.tightened} tightened{shifts}"
            )
    point = iterate.point
    try:
        model = active.relax(
            point.boundary, point.state + point.normal_step(), NESTED_TOLERANCE
        )
    except RuntimeError:  # the normal step crossed flux surfaces
        model = active.relax(point.boundary, point.state, NESTED_TOLERANCE)
    force_residual = model.fsqr + model.fsqz + model.fsql
    return Result(
        point.boundary,
        np.array(model.get_state()),
        active.evaluate(model, point.boundary),
        force_residual,
        steps,
        ends,
        active.multipliers(),
    )


def converged_input(problem: JointProblem, boundary, *, pin_gauge: bool, ns: int = 51):
    """The boundary as a fully converged multigrid run from its own cold initial
    guess."""
    return problem.vmec_input.model_copy(
        update={
            "rbc": boundary[0],
            "zbs": boundary[1],
            "always_fix_m1_gauge": pin_gauge,
            "ns_array": np.asarray([16, ns]),
            "ftol_array": np.asarray([1.0e-8, 1.0e-13]),
            "niter_array": np.asarray([5000, 50000]),
        }
    )


__all__ = [
    "CONSTRAINT_TOLERANCE",
    "LOOSEST_TOLERANCE",
    "NESTED_TOLERANCE",
    "RELAXATION_ITERATIONS",
    "STAGE_DROP",
    "Costs",
    "Evaluation",
    "JointProblem",
    "QuasisymmetryProblem",
    "Result",
    "Shifts",
    "Stage",
    "Step",
    "boundary_modes",
    "converged_input",
    "optimize",
]

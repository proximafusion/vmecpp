# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Optimize a fixed boundary and its VMEC state under the constraint of force balance.

Each Levenberg-Marquardt step solves::

    min |r + J_x dx + J_b db|^2 + damping |db|^2  subject to  F + H dx + F_b db = 0

over the state ``x`` and the boundary ``b``, with the residuals ``r``, the force ``F`` on
the interior of the state and its Jacobian ``H``. Eliminating the state gives
``dx = dn + T db`` with the normal step ``dn = -H^-1 F``, limited to a fraction of the
major radius, and the state tangents ``T = -H^-1 F_b``. The merit of an iterate is
``|r + J_x dn|^2``. This is the objective with the force residual removed to first order.
VMEC's own descent then relaxes the state as far as the error of the merit requires. A
trial step whose ``|r|^2`` shows no clear drop takes ``dn`` from GMRES on Hessian products
at its own state, preconditioned by the factors of the last point, and ``H`` is assembled
only if that merit falls. The nested baseline converges the state at every step instead.
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

from vmecpp import _qs, autodiff, geometry
from vmecpp.cpp import _vmecpp  # type: ignore

NESTED_TOLERANCE = 1.0e-12
"""Force tolerance of the steps of the nested baseline, and the tightest of any step."""

LOOSEST_TOLERANCE = 1.0e-5
"""Loosest force tolerance of a step."""

STAGE_DROP = 1.0e-2
"""Relative drop of the merit per step below which a stage has stalled."""

RELAXATION_ITERATIONS = 3000
"""Largest number of descent iterations of one relaxation between steps."""

_FIRST_TOLERANCE = 1.0e-6
"""Force tolerance of the first state of a run."""

_NORMAL_RADIUS = 0.05
"""Largest normal step in one step, as a fraction of the major radius."""

_SMALLEST_DROP = 1.0e-5
"""Predicted drop, relative to the merit, below which no trial step is taken."""

_FIRST_DAMPING = 1.0e-3
"""Damping at the start of a stage, above which the damping limits the step."""

_CLEAR_DROP = 0.5
"""Drop of ``|r|^2`` relative to the predicted drop above which a trial step has its
force Jacobian assembled without a merit on the factors of the last point."""

_NORMAL_ERROR = 1.0e-2
"""Estimated relative error at which GMRES stops on a normal step."""

_NORMAL_ITERATIONS = 64
"""Largest number of GMRES iterations on a normal step."""

_TANGENT_CHUNK = 16
"""Residual tangents per call of the compiled derivative, a size that compiles once."""


def boundary_modes(max_mode: int, ntor: int) -> list[tuple[int, int, int]]:
    """Indices (row, m, ntor + n) of the free boundary coefficients; rbc(0, 0) is
    fixed."""
    return [
        (row, m, ntor + n)
        for row in (0, 1)
        for m in range(max_mode + 1)
        for n in range(-max_mode, max_mode + 1)
        if m > 0 or n > 0
    ]


def _gmres(apply, precondition, rhs: np.ndarray) -> np.ndarray:
    """Right-preconditioned GMRES, to _NORMAL_ERROR or its last iterate.

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
    """The residual vector at one state and its named parts."""

    residuals: np.ndarray
    parts: dict[str, float]

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
class JointProblem:
    """A least-squares objective over a fixed boundary and its state.

    A subclass defines ``residuals``, which JAX must be able to differentiate.
    ``workers`` is the number of threads for the force Jacobian and the residual
    derivatives, by default one per processor. ``exact_products`` takes the Hessian
    products of the force Jacobian from Enzyme, in a build that has it. Otherwise they
    are central differences of the force, at about the same cost. A multigrid input runs
    on its last resolution.
    """

    vmec_input: Any
    max_mode: int = 3
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)
    exact_products: bool = bool(_vmecpp.VMECPP_ENABLE_ENZYME)
    costs: Costs = field(default_factory=Costs)

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
        self.template = self.vmec_input._to_cpp_vmecindata()
        self.template.always_fix_m1_gauge = True
        self.ns = int(np.asarray(self.vmec_input.ns_array)[-1])
        self.template.ns_array = np.asarray([self.ns])
        self.template.ftol_array = np.asarray(self.vmec_input.ftol_array)[-1:]
        self.template.niter_array = np.asarray(self.vmec_input.niter_array)[-1:]
        self.boundary0 = np.stack(
            [np.asarray(self.vmec_input.rbc), np.asarray(self.vmec_input.zbs)]
        )
        model = self.create_model(self.boundary0)
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
        self._residuals = jax.jit(self._flat_residuals)
        self._linearized = jax.jit(self._linearized_residuals)
        self._along = jax.jit(self._residuals_along)

    def residuals(self, solved: geometry.Geometry, boundary: jax.Array) -> jax.Array:
        """The residual vector of a state; ``boundary`` holds rbc, then zbs."""
        raise NotImplementedError

    def parts(self, residuals: np.ndarray) -> dict[str, float]:  # noqa: ARG002
        """Named parts of the residual vector, for the log and the result."""
        return {}

    def create_model(self, boundary, ftol=None, niter=None):
        indata = autodiff._make_indata(self.template, boundary)
        if ftol is not None:
            indata.ftol_array = np.asarray([float(ftol)])
        if niter is not None:
            indata.niter_array = np.asarray([int(niter)])
        return _vmecpp.VmecModel.create(indata, self.ns)

    def relax(self, boundary, state=None, ftol=None, niter=None):
        """VMEC's own descent from the state, to the force tolerance."""
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
        """The model and one more per further worker, all evaluated at the state."""

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
        return self.residuals(self._geometry(flat), boundary)

    def _linearized_residuals(self, flat, boundary, directions, boundary_directions):
        _, linear = jax.linearize(self._flat_residuals, flat, boundary)
        return jax.vmap(linear)(directions, boundary_directions)

    def _residuals_along(self, flat, boundary, direction):
        return jax.jvp(
            lambda at: self._flat_residuals(at, boundary), (flat,), (direction,)
        )[1]

    def _flat_geometry(self, model) -> np.ndarray:
        return autodiff._cpp_geometry_flat(
            model.get_geometry(), self.ns, self.mpol, self.ntor
        )

    def _geometry_along(self, model, state, direction) -> np.ndarray:
        """The flat geometry along a state direction, differenced through the model.

        Chi depends on the state when ncurr = 1.
        """
        step = 1.0e-7 / max(np.abs(direction).max(), 1.0e-3)
        sides = []
        for sign in (1.0, -1.0):
            model.set_state(np.ascontiguousarray(state + sign * step * direction))
            model.evaluate(2, 2, False)
            sides.append(self._flat_geometry(model))
        return (sides[0] - sides[1]) / (2.0 * step)

    def _evaluation(self, residuals: np.ndarray) -> Evaluation:
        return Evaluation(residuals, self.parts(residuals))

    def evaluate(self, model, boundary) -> Evaluation:
        """The residuals of the state the model holds, in force balance or not."""
        model.evaluate(2, 2, False)
        return self._evaluation(
            np.asarray(
                self._residuals(self._flat_geometry(model), jnp.asarray(boundary))
            )
        )

    def force_jacobian(self, model, state):
        """H at the state, factored, from Hessian products on one model per worker."""
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
        """D state / D boundary at fixed force, one column per boundary mode."""
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
        """The evaluation and the residual derivatives along the state tangents."""
        start = time.perf_counter()
        state = np.ascontiguousarray(state)
        models = self._models_at(model, state, precondition=False)
        directions = np.stack(
            autodiff._map_on_models(
                lambda worker, i: self._geometry_along(worker, state, tangents[:, i]),
                models,
                range(len(keys)),
            )
        )
        model.set_state(state)
        model.evaluate(2, 2, False)
        flat = self._flat_geometry(model)
        boundary_directions = np.zeros((len(keys), *np.shape(boundary)))
        for i, key in enumerate(keys):
            boundary_directions[(i, *key)] = 1.0
        values = np.asarray(self._residuals(flat, jnp.asarray(boundary)))
        rows = []
        for first in range(0, len(keys), _TANGENT_CHUNK):
            chunk = range(first, min(first + _TANGENT_CHUNK, len(keys)))
            padding = _TANGENT_CHUNK - len(chunk)
            rows.append(
                np.asarray(
                    self._linearized(
                        flat,
                        jnp.asarray(boundary),
                        jnp.asarray(_padded(directions[chunk], padding)),
                        jnp.asarray(_padded(boundary_directions[chunk], padding)),
                    )
                )[: len(chunk)]
            )
        self.costs.objective_seconds += time.perf_counter() - start
        return self._evaluation(values), np.vstack(rows).T

    def _normal_step(self, model, state, force, matrix) -> np.ndarray:
        """``-H^-1 F`` on the interior, by GMRES on products at the state."""
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
        """The residuals at the state, the same residuals after the normal step to first
        order, ``r + J_x dn`` with ``dn`` within its radius, and ``-H^-1 F``.

        ``matrix`` holds the factors of this state or a nearby one.
        """
        start = time.perf_counter()
        state = np.ascontiguousarray(state)
        model = self.create_model(boundary)
        model.set_state(state)
        model.evaluate(2, 2, False)
        flat = self._flat_geometry(model)
        force = np.array(model.get_forces())[self.interior]
        values = np.asarray(self._residuals(flat, jnp.asarray(boundary)))
        normal = np.zeros(state.size)
        try:
            normal[self.interior] = self._normal_step(model, state, force, matrix)
        except ValueError:  # the force, the factors or a product are not finite
            normal[:] = np.nan
        change = np.full_like(values, np.nan)
        if np.all(np.isfinite(normal)):
            change = np.asarray(
                self._along(
                    flat,
                    jnp.asarray(boundary),
                    jnp.asarray(self._geometry_along(model, state, normal)),
                )
            )
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
    """SIMSOPT's QuasisymmetryRatioResidual with aspect-ratio and mean-iota targets, as
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
        """Aspect ratio and mean iota against their targets, then the residuals of
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
    """One accepted step of the optimization, with its merit in ``evaluation``."""

    max_mode: int
    evaluation: Evaluation
    force_residual: float
    force_evaluations: int
    seconds: float


@dataclass
class Stage:
    """How one stage of boundary modes ended.

    ``stopped`` is "stalled" after three steps in a row below STAGE_DROP that the damping
    did not limit, "no_step" when no trial step predicts a drop or none holds, also from
    the state in force balance, and "max_steps" at the step limit. ``rejected_steps`` counts the trial steps refused
    for their merit or force residual, ``failed_solves`` those on which VMEC raised, and
    ``tightened`` the relaxations further for the accuracy of a merit.
    """

    max_mode: int
    stopped: str = "max_steps"
    rejected_steps: int = 0
    failed_solves: int = 0
    tightened: int = 0


@dataclass
class Result:
    """The final boundary, its state relaxed to NESTED_TOLERANCE with its residuals and
    force residual, every accepted step and how each stage ended."""

    boundary: np.ndarray
    state: np.ndarray
    evaluation: Evaluation
    force_residual: float
    steps: list[Step]
    stages: list[Stage]


@dataclass
class _Point:
    """A boundary and state, the residuals there and their value in force balance.

    ``matrix`` is the factored force Jacobian, of this state or of the last accepted one.
    The nested baseline takes the residuals as they are, and factors the Jacobian only for
    an accepted step. ``relaxable`` is false once a further relaxation of the point has
    failed.
    """

    boundary: np.ndarray
    state: np.ndarray
    residuals: np.ndarray
    balanced: np.ndarray
    normal: np.ndarray | None = None
    matrix: Any = None
    relaxable: bool = True

    @property
    def merit(self) -> float:
        return float(self.balanced @ self.balanced)

    @property
    def normal_size(self) -> float:
        return 0.0 if self.normal is None else float(np.abs(self.normal).max())

    def normal_step(self) -> np.ndarray:
        """The normal step, within its radius."""
        if self.normal is None or not np.isfinite(self.normal_size):
            return np.zeros_like(self.state)
        return _normal_fraction(self.boundary, self.normal) * self.normal

    @property
    def merit_error(self) -> float:
        """Estimated error of the merit, ``2 |r_b| |J_x dn| |dn| / R``, for the whole of
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


def _joint_point(problem: JointProblem, boundary, state, matrix=None) -> _Point:
    """The point with its merit on the given factors, or on its own."""
    state = np.ascontiguousarray(state)
    if matrix is None:
        matrix = problem.force_jacobian(problem.create_model(boundary), state)
    values, balanced, normal = problem.balanced_residuals(boundary, state, matrix)
    return _Point(boundary, state, values, balanced, normal, matrix)


def _nested_point(problem: JointProblem, boundary, model) -> _Point:
    state = np.ascontiguousarray(np.array(model.get_state()))
    values = problem.evaluate(model, boundary).residuals
    return _Point(boundary, state, values, values)


def _relaxed_further(
    problem: JointProblem, point: _Point, tolerance: float
) -> _Point | None:
    """The point relaxed to a tighter tolerance on its factored Jacobian, or None where
    VMEC raises or the tighter state has no finite merit.

    A surface whose Jacobian changes sign has no finite merit.
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
            point, state=state, residuals=values, balanced=balanced, normal=normal
        )
    return further if np.isfinite(further.merit) else None


def _decade_tighter(
    problem: JointProblem, point: _Point, tolerance: float, end: Stage
) -> tuple[_Point, float]:
    """The point relaxed one decade further and its tolerance, or the point marked as
    not relaxable."""
    tighter = max(tolerance / 10.0, NESTED_TOLERANCE)
    further = _relaxed_further(problem, point, tighter)
    if further is None:
        return replace(point, relaxable=False), tolerance
    end.tightened += 1
    return further, tighter


def _tightened(
    problem: JointProblem, point: _Point, tolerance: float, allowed: float, end: Stage
) -> tuple[_Point, float]:
    """The point relaxed until the error of its merit is within the allowance, and the
    tolerance this took."""
    while (
        point.relaxable and point.merit_error > allowed and tolerance > NESTED_TOLERANCE
    ):
        point, tolerance = _decade_tighter(problem, point, tolerance, end)
    return point, tolerance


def _restored(problem: JointProblem, point: _Point, end: Stage) -> _Point:
    """The point in force balance, with its Jacobian factored there."""
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
    """The first trial step that holds, or None after twelve."""
    normal = jacobian.T @ jacobian
    # the floor bounds the step in weakly coupled modes when residuals are few
    scale = np.maximum(np.diag(normal), max(1.0e-4 * np.diag(normal).max(), 1.0e-10))

    def proposal():
        balanced = iterate.point.balanced
        delta = -np.linalg.solve(
            normal + iterate.damping * np.diag(scale), jacobian.T @ balanced
        )
        predicted = iterate.point.merit - float(
            np.sum((balanced + jacobian @ delta) ** 2)
        )
        return (
            delta,
            predicted,
            error_fraction * max(predicted, STAGE_DROP * iterate.point.merit),
        )

    take_normal = not nested
    for _ in range(12):
        delta, predicted, allowed = proposal()
        if predicted < _SMALLEST_DROP * iterate.point.merit:
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
                drop = (point.merit - candidate.merit) / point.merit
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


def optimize(
    problem: JointProblem,
    stages=(1, 2, 3),
    *,
    nested: bool = False,
    error_fraction: float = 0.01,
    max_steps: int = 120,
    log=None,
) -> Result:
    """Levenberg-Marquardt on the boundary and the state, adding boundary modes stage by
    stage.

    Each trial step relaxes the state until the estimated error of the merit is
    ``error_fraction`` of the drop the step predicts, and of STAGE_DROP times the merit
    at least, between NESTED_TOLERANCE and LOOSEST_TOLERANCE. The last stage starts in
    force balance and relaxes every step to NESTED_TOLERANCE, as does the last state.
    ``nested`` instead converges every step to NESTED_TOLERANCE and
    takes the residuals as they are. Residuals that are not finite at the first state
    raise a ValueError.
    """
    if max(stages) > problem.max_mode:
        message = f"stage {max(stages)} is above max_mode {problem.max_mode}"
        raise ValueError(message)
    start = time.perf_counter()
    boundary = problem.boundary0.copy()
    model = problem.relax(boundary, ftol=_FIRST_TOLERANCE)
    if nested:
        first = _nested_point(problem, boundary, model)
        if np.isfinite(first.merit):
            first = _relaxed_further(problem, first, NESTED_TOLERANCE) or first
        iterate = _Iterate(first, NESTED_TOLERANCE)
    else:
        first = _joint_point(problem, boundary, model.get_state())
        iterate = _Iterate(first, _FIRST_TOLERANCE)
    if not np.isfinite(iterate.point.merit):
        message = "the residuals are not finite at the first state"
        raise ValueError(message)
    steps: list[Step] = []
    ends: list[Stage] = []
    for index, max_mode in enumerate(stages):
        keys = boundary_modes(max_mode, problem.ntor)
        end = Stage(max_mode)
        ends.append(end)
        iterate.damping = _FIRST_DAMPING
        iterate.ceiling = LOOSEST_TOLERANCE
        if not nested and index == len(stages) - 1:
            # the last stage takes its derivatives in force balance
            iterate.point = _restored(problem, iterate.point, end)
            iterate.tolerance = iterate.ceiling = NESTED_TOLERANCE
        slow = 0
        restored = False
        for _ in range(max_steps):
            point = iterate.point
            model = problem.create_model(point.boundary)
            tangents = problem.state_tangents(model, point.state, keys, point.matrix)
            _, jacobian = problem.residual_jacobian(
                model, point.state, point.boundary, keys, tangents
            )
            refused = end.rejected_steps + end.failed_solves
            trial = _first_accepted_step(
                problem,
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
                    end.stopped = "no_step"
                    break
                # the merit of a loose state can hide a drop smaller than its error
                iterate.point = _restored(problem, iterate.point, end)
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
            evaluation = problem._evaluation(trial.point.balanced)
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
                end.stopped = "stalled"
                break
        if log is not None:
            log(
                f"modes {max_mode} stopped: {end.stopped},"
                f" {end.rejected_steps} rejected steps,"
                f" {end.failed_solves} failed solves,"
                f" {end.tightened} tightened"
            )
    point = iterate.point
    try:
        model = problem.relax(
            point.boundary, point.state + point.normal_step(), NESTED_TOLERANCE
        )
    except RuntimeError:  # the normal step crossed flux surfaces
        model = problem.relax(point.boundary, point.state, NESTED_TOLERANCE)
    force_residual = model.fsqr + model.fsqz + model.fsql
    return Result(
        point.boundary,
        np.array(model.get_state()),
        problem.evaluate(model, point.boundary),
        force_residual,
        steps,
        ends,
    )


def converged_input(problem: JointProblem, boundary, *, pin_gauge: bool, ns: int = 51):
    """The boundary as a fully converged multigrid run, cold from its own initial
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
    "LOOSEST_TOLERANCE",
    "NESTED_TOLERANCE",
    "RELAXATION_ITERATIONS",
    "STAGE_DROP",
    "Costs",
    "Evaluation",
    "JointProblem",
    "QuasisymmetryProblem",
    "Result",
    "Stage",
    "Step",
    "boundary_modes",
    "converged_input",
    "optimize",
]

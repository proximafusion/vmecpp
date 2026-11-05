# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Optimize a fixed boundary without converging VMEC between steps.

Each Levenberg-Marquardt step on the boundary moves the state along its tangent, and
VMEC's own descent then relaxes the force residual only as far as the step needs: the
error of the merit, which grows with the square root of the force residual, stays a
fixed fraction of the drop the step predicts. A fraction of zero converges every step,
the nested baseline.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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

MERIT_TOLERANCE = 1.0e-3
"""Largest force tolerance of a step, as a fraction of the merit."""

STAGE_DROP = 1.0e-2
"""Relative drop of the merit per step below which a stage has stalled."""

_TANGENT_CHUNK = 16
"""Residual tangents per call of the compiled derivative, which then compiles once."""


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
    """Force evaluations and wall time by kind of work."""

    force_evaluations: int = 0
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
    products of the force Jacobian from Enzyme, in a build that has it; otherwise they
    are central differences of the force, which cost about the same.
    """

    vmec_input: Any
    max_mode: int = 3
    workers: int = field(default_factory=lambda: os.cpu_count() or 1)
    exact_products: bool = bool(_vmecpp.VMECPP_ENABLE_ENZYME)
    costs: Costs = field(default_factory=Costs)

    def __post_init__(self) -> None:
        if not jax.config.jax_enable_x64:
            message = "vmecpp.joint needs jax_enable_x64 for its residuals"
            raise ValueError(message)
        self.template = self.vmec_input._to_cpp_vmecindata()
        self.template.always_fix_m1_gauge = True
        self.ns = int(np.asarray(self.vmec_input.ns_array)[-1])
        self.mpol, self.ntor = self.vmec_input.mpol, self.vmec_input.ntor
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
        model.solve()
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

    def _flat_geometry(self, model) -> np.ndarray:
        return autodiff._cpp_geometry_flat(
            model.get_geometry(), self.ns, self.mpol, self.ntor
        )

    def _evaluation(self, residuals: np.ndarray) -> Evaluation:
        return Evaluation(residuals, self.parts(residuals))

    def evaluate(self, model, boundary) -> Evaluation:
        """The residuals of the state the model holds, which need not be in force
        balance."""
        model.evaluate(2, 2, False)
        return self._evaluation(
            np.asarray(
                self._residuals(self._flat_geometry(model), jnp.asarray(boundary))
            )
        )

    def state_tangents(self, model, state, keys) -> np.ndarray:
        """D state / D boundary at fixed force, one column per boundary mode."""
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
        start = time.perf_counter()
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

        # differenced through the model, whose chi depends on the state when ncurr = 1
        def direction(worker, i: int) -> np.ndarray:
            step = 1.0e-7 / max(np.abs(tangents[:, i]).max(), 1.0e-3)
            sides = []
            for sign in (1.0, -1.0):
                worker.set_state(
                    np.ascontiguousarray(state + sign * step * tangents[:, i])
                )
                worker.evaluate(2, 2, False)
                sides.append(self._flat_geometry(worker))
            return (sides[0] - sides[1]) / (2.0 * step)

        directions = np.stack(
            autodiff._map_on_models(direction, models, range(len(keys)))
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


def _padded(rows: np.ndarray, padding: int) -> np.ndarray:
    return np.concatenate([rows, np.zeros((padding, *rows.shape[1:]))])


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
    """One accepted step of the optimization."""

    max_mode: int
    evaluation: Evaluation
    force_residual: float
    force_evaluations: int
    seconds: float


@dataclass
class Stage:
    """How one stage of boundary modes ended.

    ``stopped`` is "stalled" after three steps in a row below STAGE_DROP, "no_step" when
    twelve trial steps in a row fail, and "max_steps" at the step limit.
    ``rejected_steps`` counts the trial steps refused for their merit or force residual,
    ``failed_solves`` those on which VMEC raised, and ``tightened`` the times the state
    was relaxed further before a step.
    """

    max_mode: int
    stopped: str = "max_steps"
    rejected_steps: int = 0
    failed_solves: int = 0
    tightened: int = 0


@dataclass
class Result:
    """The final boundary and state, every accepted step and how each stage ended."""

    boundary: np.ndarray
    state: np.ndarray
    steps: list[Step]
    stages: list[Stage]


@dataclass
class _Iterate:
    """The boundary and state of a run, the force tolerance the state was relaxed to,
    and the measured change of the merit per square root of force residual."""

    boundary: np.ndarray
    state: np.ndarray
    tolerance: float
    noise: float


@dataclass
class _Trial:
    """An accepted trial step and the merit it improved on."""

    boundary: np.ndarray
    model: Any
    evaluation: Evaluation
    force_residual: float
    tolerance: float
    gain: float
    merit_before: float


def _relax_further(
    problem: JointProblem, iterate: _Iterate, tolerance: float, merit: float
) -> Evaluation | None:
    """Relax the state to a tighter tolerance, and measure the noise on the way.

    The state stays as it is where the tighter one has no finite merit, which can be a
    surface whose Jacobian changes sign.
    """
    model = problem.relax(iterate.boundary, iterate.state, tolerance)
    evaluation = problem.evaluate(model, iterate.boundary)
    if not np.isfinite(evaluation.merit):
        return None
    change = abs(evaluation.merit - merit) / np.sqrt(iterate.tolerance)
    iterate.noise = max(change, 0.7 * iterate.noise, 1.0e-6)
    iterate.state, iterate.tolerance = np.array(model.get_state()), tolerance
    return evaluation


def _first_accepted_step(
    problem: JointProblem,
    iterate: _Iterate,
    keys,
    tangents,
    current: Evaluation,
    jacobian,
    noise_fraction: float,
    damping: float,
    end: Stage,
) -> tuple[_Trial | None, float]:
    """The first trial step that holds, or None after twelve, and the damping it
    took."""
    normal = jacobian.T @ jacobian
    # the floor bounds the step in weakly coupled modes when residuals are few
    scale = np.maximum(np.diag(normal), max(1.0e-4 * np.diag(normal).max(), 1.0e-10))
    residuals, merit = current.residuals, current.merit

    def proposal():
        delta = -np.linalg.solve(
            normal + damping * np.diag(scale), jacobian.T @ residuals
        )
        predicted = merit - float(np.sum((residuals + jacobian @ delta) ** 2))
        error = noise_fraction * predicted / iterate.noise
        loosest = min(LOOSEST_TOLERANCE, MERIT_TOLERANCE * merit)
        return (
            delta,
            predicted,
            float(np.clip(error**2, NESTED_TOLERANCE, max(loosest, NESTED_TOLERANCE))),
        )

    for _ in range(12):
        delta, predicted, tolerance = proposal()
        # both sides of the comparison of merits need the accuracy of the step
        if tolerance < 0.5 * iterate.tolerance:
            further = _relax_further(problem, iterate, tolerance, merit)
            if further is not None:
                end.tightened += 1
                residuals, merit = further.residuals, further.merit
                delta, predicted, tolerance = proposal()
        trial = iterate.boundary.copy()
        for key, value in zip(keys, delta, strict=True):
            trial[key] += value
        try:
            relaxed = problem.relax(trial, iterate.state + tangents @ delta, tolerance)
        except RuntimeError:  # the step crossed flux surfaces
            end.failed_solves += 1
            damping *= 4.0
            continue
        force_residual = relaxed.fsqr + relaxed.fsqz + relaxed.fsql
        evaluation = problem.evaluate(relaxed, trial)
        gain = (merit - evaluation.merit) / max(predicted, 1.0e-300)
        # the descent stayed near force balance and the merit fell
        if force_residual <= max(30.0 * tolerance, 1.0e-4) and gain > 1.0e-3:
            accepted = _Trial(
                trial, relaxed, evaluation, force_residual, tolerance, gain, merit
            )
            return accepted, damping
        end.rejected_steps += 1
        damping *= 4.0
    return None, damping


def optimize(
    problem: JointProblem,
    stages=(1, 2, 3),
    noise_fraction: float = 0.3,
    max_steps: int = 120,
    log=None,
) -> Result:
    """Levenberg-Marquardt on the boundary, adding boundary modes stage by stage.

    Each trial step relaxes the state until the error of the merit is about
    ``noise_fraction`` of the drop the step predicts, between NESTED_TOLERANCE and the
    smaller of LOOSEST_TOLERANCE and MERIT_TOLERANCE times the merit. After each step the
    state is relaxed ten times further, which measures the noise and starts the next
    step. A fraction of zero converges every step to NESTED_TOLERANCE.
    Residuals that are not finite raise a ValueError.
    """
    boundary = problem.boundary0.copy()
    model = problem.relax(boundary, ftol=1.0e-6)
    merit = problem.evaluate(model, boundary).merit
    iterate = _Iterate(boundary, np.array(model.get_state()), 1.0e-6, 1.0e-6)
    _relax_further(problem, iterate, 1.0e-7, merit)
    steps: list[Step] = []
    ends: list[Stage] = []
    start = time.perf_counter()
    for max_mode in stages:
        keys = boundary_modes(max_mode, problem.ntor)
        end = Stage(max_mode)
        ends.append(end)
        damping = 1.0e-3
        slow = 0
        for _ in range(max_steps):
            model = problem.create_model(iterate.boundary)
            tangents = problem.state_tangents(model, iterate.state, keys)
            current, jacobian = problem.residual_jacobian(
                model, iterate.state, iterate.boundary, keys, tangents
            )
            if not np.isfinite(current.merit):
                message = f"the residuals are not finite after {len(steps)} steps"
                raise ValueError(message)
            trial, damping = _first_accepted_step(
                problem,
                iterate,
                keys,
                tangents,
                current,
                jacobian,
                noise_fraction,
                damping,
                end,
            )
            if trial is None:
                end.stopped = "no_step"
                break
            if trial.gain > 0.5:
                damping = max(damping / 3.0, 1.0e-9)
            elif trial.gain < 0.25:
                damping *= 2.0
            merit = trial.evaluation.merit
            drop = (trial.merit_before - merit) / trial.merit_before
            iterate.boundary = trial.boundary
            iterate.state = np.array(trial.model.get_state())
            iterate.tolerance = trial.tolerance
            # keeps the measured noise current
            if trial.tolerance >= 10.0 * NESTED_TOLERANCE:
                _relax_further(problem, iterate, trial.tolerance / 10.0, merit)
            steps.append(
                Step(
                    max_mode,
                    trial.evaluation,
                    trial.force_residual,
                    problem.costs.force_evaluations,
                    time.perf_counter() - start,
                )
            )
            if log is not None:
                named = "".join(
                    f" {name} {value:.5g}"
                    for name, value in trial.evaluation.parts.items()
                )
                log(
                    f"modes {max_mode} step {len(steps):3d}"
                    f" merit {merit:.4e}{named}"
                    f" force {trial.force_residual:.1e}"
                    f" evals {problem.costs.force_evaluations}"
                )
            slow = slow + 1 if drop < STAGE_DROP else 0
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
    return Result(iterate.boundary, iterate.state, steps, ends)


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
    "MERIT_TOLERANCE",
    "NESTED_TOLERANCE",
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

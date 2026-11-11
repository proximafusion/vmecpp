# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Coil currents of a free-boundary equilibrium for a target aspect ratio and mean iota:
the currents, the boundary and the VMEC state move together under force balance and
pressure balance.

VMEC++'s own free-boundary run checks each result.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import vmecpp
from vmecpp import autodiff_wout
from vmecpp.joint import QuasisymmetryProblem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from free_boundary_root import MU0, Matching  # type: ignore

jax.config.update("jax_enable_x64", True)

TEST_DATA = (
    Path(__file__)
    .resolve()
    .parents[1]
    .joinpath("src", "vmecpp", "cpp", "vmecpp", "test_data")
)

MAX_MOVE = 5.0e-3  # largest move of a shape mode in one step, in m


def cth_like(ns: int):
    """The CTH-like free-boundary input on one radial grid and the response table of its
    two coil groups."""
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA / "cth_like_free_bdy.json")
    vmec_input.ns_array = np.array([ns])
    vmec_input.ftol_array = np.array([1.0e-12])
    vmec_input.niter_array = np.array([20000])
    vmec_input.mgrid_file = "NONE"
    vmec_input.return_outputs_even_if_not_converged = True
    response = vmecpp.MagneticFieldResponseTable.from_coils_file(
        TEST_DATA / "coils.cth_like",
        vmecpp.MakegridParameters.from_file(
            TEST_DATA / "makegrid_parameters_cth_like.json"
        ),
    )
    return vmec_input, response


def shape_modes(m_max: int, n_max: int) -> tuple:
    """One copy of each stellarator-symmetric shape mode (m, n, 0)."""
    return tuple(
        (m, n, 0)
        for m in range(m_max + 1)
        for n in range(-n_max, n_max + 1)
        if m > 0 or n >= 0
    )


@dataclass(kw_only=True)
class FreeBoundaryProblem(QuasisymmetryProblem):
    """The targets of the base class, and its quasisymmetry on the surfaces given, on a
    boundary that moves along its normal in shape modes, in the field of coils.

    The residuals of the base class are followed by the total pressure of the plasma on
    the boundary in the shape modes, in units of ``scale``: ``B^2 / 2`` at the
    volume-averaged field inside the input's boundary. The vacuum side of the jump
    comes from one free-boundary iteration of VMEC++ on the state. It depends on the
    boundary, on the coil currents and, through the field of a net toroidal current, on
    the magnetic axis.
    """

    response: Any
    modes: tuple
    surfaces: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.matching = Matching(self.vmec_input, self.modes)
        # the wout that the states of the vacuum side fill
        self._wout = vmecpp.run(
            self.vmec_input.model_copy(update={"always_fix_m1_gauge": True}),
            max_threads=1,
            verbose=False,
        ).wout
        self.scale = 0.5 * float(self._wout.volavgB) ** 2
        zeta, theta = self.matching.grid
        angle = (
            self._wout.xm_nyq[:, None, None] * theta[None, None, :]
            - self._wout.xn_nyq[:, None, None] * zeta[None, :, None]
        )
        self._cosines = jnp.asarray(np.cos(angle).reshape(angle.shape[0], -1))
        self._projection = jnp.asarray(self.matching.projection)
        self._surfaces = jax.jit(self._state_surfaces)
        count = len(self.modes)
        self.reference = self._fitted(np.zeros(count))
        # the boundary is affine in the shape modes
        self.shape_directions = np.stack(
            [self._fitted(np.eye(count)[k]) - self.reference for k in range(count)]
        )
        base = np.array(self.create_model(self.reference).get_state())
        self.shape_states = np.zeros((base.size, count))
        step = 1.0e-3
        for k in range(count):
            moved = self.reference + step * self.shape_directions[k]
            change = np.array(self.create_model(moved).get_state()) - base
            self.shape_states[self.prescribed, k] = change[self.prescribed] / step
        self.linearizations = 0
        self.largest_force = 0.0

    def _fitted(self, shape) -> np.ndarray:
        fitted = self.matching.boundary(np.asarray(shape, dtype=float))
        return np.stack([fitted["rbc"], fitted["zbs"]])

    def boundary_of(self, shape) -> np.ndarray:
        """The boundary coefficients of a shape."""
        return self.reference + np.tensordot(shape, self.shape_directions, axes=1)

    def carried(self, state, change, tangents=None) -> np.ndarray:
        """A state on the boundary of a changed shape.

        The boundary and gauge entries follow the shape. The interior follows the
        tangents when given.
        """
        along = self.shape_states if tangents is None else tangents
        return np.ascontiguousarray(state + along @ np.asarray(change, dtype=float))

    def root(self, boundary, state=None, tolerance=1.0e-11, steps=6):
        """The state at the root of the force inside a boundary and the force Jacobian
        factored at the last iterate.

        VMEC's descent starts from no state. Newton steps on the force follow: the
        descent stops at its tolerance short of the root along the soft modes, and the
        pressure on the boundary is sensitive to them.
        """
        if state is None:
            state = np.array(self.relax(boundary, None, 1.0e-10).get_state())
        state = np.ascontiguousarray(state).copy()
        matrix = None
        for _ in range(steps):
            force = self._force(boundary, state)
            if matrix is not None and np.abs(force).max() < tolerance:
                break
            matrix = self.force_jacobian(self.create_model(boundary), state)
            state[self.interior] += matrix.solve(-force)
        self.largest_force = float(np.abs(force).max())
        return state, matrix

    def _force(self, boundary, state) -> np.ndarray:
        model = self.create_model(boundary)
        model.set_state(state)
        model.evaluate(2, 2, False)
        return np.array(model.get_forces())[self.interior]

    def _edge_modes(self, solved):
        """The total pressure of the plasma on the boundary in the shape modes,
        extrapolated linearly in s from the two outermost half-grid surfaces."""
        wout = autodiff_wout.wout_quantities(solved, self.vmec_input)
        field = wout["bmnc"][:, -2:].T @ self._cosines
        outer = 0.5 * field**2 + MU0 * wout["pres"][-2:, None]
        return self._projection @ ((1.5 * outer[1] - 0.5 * outer[0]) / self.scale)

    def residuals(self, solved, boundary):
        return jnp.concatenate(
            [super().residuals(solved, boundary), self._edge_modes(solved)]
        )

    def split(self, values):
        """The objective residuals and the plasma side of the jump."""
        count = len(self.modes)
        return values[..., :-count], values[..., -count:]

    def parts(self, residuals):
        return super().parts(self.split(np.asarray(residuals))[0])

    def _state_surfaces(self, solved):
        wout = autodiff_wout.wout_quantities(solved, self.vmec_input)
        return {name: wout[name] for name in ("rmnc", "zmns", "lmns_full")}

    def vacuum_modes(self, state, currents) -> np.ndarray:
        """The magnetic pressure of the vacuum on the boundary of a state in the shape
        modes, from one free-boundary iteration restarted from the state."""
        model = self.create_model(self.reference)
        model.set_state(np.ascontiguousarray(state))
        surfaces = self._surfaces(self.solved(model))
        free = vmecpp.run(
            self.vmec_input.model_copy(
                update={
                    "lfreeb": True,
                    "mgrid_file": "NONE",
                    "extcur": np.asarray(currents, dtype=float),
                    "nvacskip": 1,
                    "niter_array": np.array([1]),
                    # a pinned gauge would be reset to that of the input's boundary
                    "always_fix_m1_gauge": False,
                    "return_vacuum_field": True,
                    "return_outputs_even_if_not_converged": True,
                }
            ),
            self.response,
            max_threads=1,
            verbose=False,
            restart_from=self._wout.model_copy(
                update={name: np.asarray(value) for name, value in surfaces.items()}
            ),
        )
        vacuum = np.asarray(free.threed1_free_boundary.bsqvaci)
        return self.matching.projection @ (vacuum.ravel() / self.scale)

    def vacuum_side(self, state, currents, tangents, step=1.0e-5, relative=1.0e-4):
        """The vacuum modes of a state, their derivatives along the columns of the state
        tangents and with respect to each current, by central differences."""
        currents = np.asarray(currents, dtype=float)
        points = [(state, currents)]
        for k in range(tangents.shape[1]):
            points += [
                (state + sign * step * tangents[:, k], currents) for sign in (1.0, -1.0)
            ]
        for i in range(currents.size):
            for sign in (1.0, -1.0):
                moved = currents.copy()
                moved[i] *= 1.0 + sign * relative
                points.append((state, moved))
        with ThreadPoolExecutor(self.workers) as pool:
            values = np.asarray(
                list(pool.map(lambda point: self.vacuum_modes(*point), points))
            )
        count = 2 * tangents.shape[1]
        along_tangents = (values[1 : count + 1 : 2] - values[2 : count + 1 : 2]).T
        rest = values[count + 1 :]
        along_currents = (rest[0::2] - rest[1::2]).T / (2.0 * relative * currents)
        return values[0], along_tangents / (2.0 * step), along_currents

    def linearize(self, shape, currents, state, *, at_root: bool) -> dict:
        """The objective residuals and the jump at a shape, currents and state, to first
        order in force balance, with their derivatives.

        ``at_root`` first takes the state to the root of the force. Otherwise the
        residuals and the jump take the normal step ``-H^-1 F`` to first order. The
        restoring move is the Newton step of the shape on the jump, limited to
        MAX_MOVE, and ``restorable`` tells whether the limit left it whole. The
        sensitivity is the derivative of the shape with respect to the logarithm of
        each current at zero jump. The merit is the squared norm of the objective
        residuals after the restoring move, to first order.
        """
        self.linearizations += 1
        count = len(self.modes)
        boundary = self.boundary_of(shape)
        if at_root:
            state, matrix = self.root(boundary, state)
            normal = np.zeros(state.size)
        else:
            state = np.ascontiguousarray(state)
            force = self._force(boundary, state)
            self.largest_force = float(np.abs(force).max())
            matrix = self.force_jacobian(self.create_model(boundary), state)
            normal = np.zeros(state.size)
            normal[self.interior] = matrix.solve(-force)
        tangents = self.tangents_along(
            self.create_model(boundary), state, self.shape_states, matrix
        )
        values, jacobian = self.residual_derivatives(
            self.create_model(boundary),
            state,
            boundary,
            self.shape_directions,
            tangents,
        )
        if not at_root:
            values = values + self.residuals_along(
                self.create_model(boundary), state, boundary, normal
            )
        residuals, plasma = self.split(values)
        vacuum, vacuum_along_shape, vacuum_along_currents = self.vacuum_side(
            state + normal, currents, tangents
        )
        jump = plasma - vacuum
        along_shape = jacobian[:-count]
        jump_along_shape = jacobian[-count:] - vacuum_along_shape
        jump_along_currents = -vacuum_along_currents * np.asarray(currents)
        restore = -np.linalg.solve(jump_along_shape, jump)
        largest = max(float(np.abs(restore).max()), 1.0e-300)
        restore *= min(1.0, MAX_MOVE / largest)
        sensitivity = -np.linalg.solve(jump_along_shape, jump_along_currents)
        balanced = residuals + along_shape @ restore
        return {
            "shape": np.array(shape, dtype=float),
            "currents": np.array(currents, dtype=float),
            "state": state,
            "normal": normal,
            "tangents": tangents,
            "residuals": residuals,
            "jump": jump,
            "jump_along_shape": jump_along_shape,
            "restore": restore,
            "restorable": largest <= MAX_MOVE,
            "sensitivity": sensitivity,
            "balanced": balanced,
            "reduced": along_shape @ sensitivity,
            "merit": float(balanced @ balanced),
        }


def converge(problem, shape, currents, state, tolerance=1.0e-9) -> dict:
    """Newton on the shape with the interior at the root of the force, until every jump
    mode is below the tolerance.

    A step that does not lower the jump, or on which VMEC or the factorization fails, is
    halved. Thirty steps without convergence raise a RuntimeError.
    """
    point = problem.linearize(shape, currents, state, at_root=True)
    steps = 0
    while np.abs(point["jump"]).max() >= tolerance:
        if steps == 30:
            message = "Newton on the jump did not converge"
            raise RuntimeError(message)
        steps += 1
        move = point["restore"]
        for _ in range(8):
            try:
                trial = problem.linearize(
                    point["shape"] + move,
                    currents,
                    problem.carried(point["state"], move, point["tangents"]),
                    at_root=True,
                )
                if np.linalg.norm(trial["jump"]) < np.linalg.norm(point["jump"]):
                    break
            except (ValueError, RuntimeError):
                pass
            move = 0.5 * move
        else:
            message = "the jump does not fall along the Newton direction"
            raise RuntimeError(message)
        point = trial
    return point


def stepped(problem, point, move, *, nested: bool) -> dict:
    """The point after a step of the logarithms of the currents: the shape moves along
    its sensitivity and its restoring move, and the state along its tangents and its
    normal step.

    ``nested`` then converges the free boundary. Otherwise the point is linearized as
    it is.
    """
    change = point["restore"] + point["sensitivity"] @ move
    state = point["state"] + point["normal"] + point["tangents"] @ change
    shape, currents = point["shape"] + change, point["currents"] * np.exp(move)
    if nested:
        return converge(problem, shape, currents, state)
    return problem.linearize(shape, currents, state, at_root=False)


def optimize(
    problem,
    point,
    *,
    nested: bool,
    max_steps: int = 30,
    tolerance: float = 1.0e-12,
    log=None,
) -> dict:
    """Levenberg-Marquardt on the logarithms of the coil currents from a converged free
    boundary.

    The move of a shape mode along the sensitivity stays within a radius of at most
    MAX_MOVE. Without ``nested``, a trial step whose restoring move exceeds MAX_MOVE is
    refused, and the first refusal at a point takes the restoring move and the normal
    step alone. Any other refusal halves the radius. A run ends at a merit below
    ``tolerance``, when no trial step predicts a drop, or after a drop below 1e-4.
    """
    damping, radius = 1.0e-3, MAX_MOVE
    restored = True
    for step in range(1, max_steps + 1):
        if point["merit"] < tolerance:
            break
        accepted = None
        for _ in range(8):
            normal = point["reduced"].T @ point["reduced"]
            move = -np.linalg.solve(
                normal + damping * np.diag(np.diag(normal)),
                point["reduced"].T @ point["balanced"],
            )
            along = np.abs(point["sensitivity"] @ move).max()
            move = move * min(1.0, radius / max(along, 1.0e-300))
            after = point["balanced"] + point["reduced"] @ move
            predicted = point["merit"] - float(after @ after)
            if predicted < 1.0e-6 * point["merit"]:
                break
            gain = 0.0
            try:
                trial = stepped(problem, point, move, nested=nested)
                if trial["restorable"]:
                    gain = (point["merit"] - trial["merit"]) / predicted
            except (ValueError, RuntimeError):  # the step left the equilibria
                pass
            if gain > 1.0e-3:
                accepted = trial
                if gain > 0.5:
                    damping = max(damping / 3.0, 1.0e-9)
                    radius = min(2.0 * radius, MAX_MOVE)
                elif gain < 0.25:
                    damping *= 2.0
                break
            if not restored:
                # the merit of a point off pressure balance can hide a drop
                restored = True
                try:
                    point = stepped(problem, point, np.zeros(move.size), nested=False)
                    continue
                except (ValueError, RuntimeError):
                    pass
            damping *= 4.0
            radius *= 0.5
        if accepted is None:
            break
        restored = nested
        drop = (point["merit"] - accepted["merit"]) / point["merit"]
        point = accepted
        if log is not None:
            named = "".join(
                f" {name} {value:.5g}"
                for name, value in problem.parts(
                    np.concatenate([point["residuals"], point["jump"]])
                ).items()
            )
            log(
                f"step {step:2d} merit {point['merit']:.4e}{named} jump"
                f" {np.abs(point['jump']).max():.1e} force {problem.largest_force:.1e}"
                f" linearizations {problem.linearizations}"
            )
        if drop < 1.0e-4:
            break
    return point


def summary(problem, point) -> str:
    """The currents, the aspect ratio and the mean iota of a point."""
    parts = problem.parts(np.concatenate([point["residuals"], point["jump"]]))
    return (
        f"currents {np.round(point['currents'], 2).tolist()}, aspect"
        f" {parts['aspect']:.5f}, mean iota {parts['mean_iota']:.5f}"
    )


def check(problem, vmec_input, currents) -> None:
    """Print the aspect ratio and the mean iota of VMEC++'s own free-boundary run at the
    currents."""
    wout = vmecpp.run(
        vmec_input.model_copy(update={"extcur": np.asarray(currents)}),
        problem.response,
        verbose=False,
        max_threads=1,
    ).wout
    print(
        f"  VMEC++'s free-boundary run: force residual"
        f" {max(wout.fsqr, wout.fsqz, wout.fsql):.1e}, aspect {wout.aspect:.5f}, mean"
        f" iota {np.mean(np.asarray(wout.iotas)[1:]):.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, default=15)
    parser.add_argument("--aspect", type=float, default=5.5)
    parser.add_argument("--iota", type=float, default=1.15)
    args = parser.parse_args()
    vmec_input, response = cth_like(args.ns)
    currents = np.asarray(vmec_input.extcur, dtype=float)
    problem = FreeBoundaryProblem(
        vmec_input.model_copy(update={"lfreeb": False}),
        aspect=args.aspect,
        mean_iota=args.iota,
        response=response,
        modes=shape_modes(vmec_input.mpol - 1, vmec_input.ntor),
    )
    start = converge(problem, np.zeros(len(problem.modes)), currents, None)
    print(f"start: {summary(problem, start)}")
    # compile the derivative along a state before the timed runs
    problem.linearize(start["shape"], currents, start["state"], at_root=False)
    for name, nested in (("nested", True), ("joint", False)):
        problem.linearizations = 0
        jacobians = problem.costs.assemblies
        begin = time.perf_counter()
        result = optimize(problem, start, nested=nested, log=print)
        seconds = time.perf_counter() - begin
        used = problem.linearizations
        jacobians = problem.costs.assemblies - jacobians
        final = converge(problem, result["shape"], result["currents"], result["state"])
        print(
            f"{name}: {used} linearizations, {jacobians} force Jacobians,"
            f" {seconds:.1f} s, on its converged free boundary"
            f" {summary(problem, final)}"
        )
        check(problem, vmec_input, final["currents"])


if __name__ == "__main__":
    main()

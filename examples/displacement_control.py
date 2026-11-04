# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Equilibria of either stability by displacement control on the line between two
minima, followed under an error field to the fold of the saddle."""

from __future__ import annotations

import argparse
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, cg

import vmecpp
from vmecpp._iteration import RestartReason
from vmecpp.cpp import _vmecpp  # type: ignore[import]

MAX_STEPS = 60000  # of one relaxation
FIRST_EPS = 1.0e-5  # first rung of a ladder, which measures how fast the saddle moves


class Case(NamedTuple):
    """A tokamak of one field period and the parameters of the search on it."""

    boundary: tuple  # R by m, Z by m and R of the axis guess
    resolution: tuple  # mpol, ntor and ns
    seed: float  # helical displacement of the axis guess that reaches a helical core
    heavy_ball: tuple  # step and momentum
    inputs: dict  # the other fields of the input

    def make(self, pressure, eps=0.0, seed=0.0) -> vmecpp.VmecInput:
        """The input under a helical shift eps of the boundary, with the axis guess
        displaced helically by seed."""
        (r_mn, z_mn, r_axis), (mpol, ntor, ns) = self.boundary, self.resolution
        rbc, zbs = np.zeros((2, mpol, 2 * ntor + 1))
        rbc[: len(r_mn), ntor], zbs[: len(z_mn), ntor] = r_mn, z_mn
        rbc[0, ntor + 1], zbs[0, ntor + 1] = eps, -eps
        raxis_c, zaxis_s = np.zeros((2, ntor + 1))
        raxis_c[:2], zaxis_s[1] = (r_axis, seed), -seed
        return vmecpp.VmecInput(
            nfp=1,
            mpol=mpol,
            ntor=ntor,
            ns_array=np.array([ns]),
            ncurr=0,
            pmass_type="power_series",
            pres_scale=pressure,
            gamma=0.0,
            always_fix_m1_gauge=True,
            rbc=rbc,
            zbs=zbs,
            raxis_c=raxis_c,
            zaxis_s=zaxis_s,
            **self.inputs,
        )


SPHERICAL_TOKAMAK = Case(
    ((1.5, 1.0), (0.0, 1.6), 1.5),
    (6, 1, 25),
    0.3,
    (0.8, 0.97),
    {
        "ftol_array": np.array([1.0e-20]),
        "niter_array": np.array([200000]),
        "phiedge": 5.0,
        "ai": np.array([1.0, 0.0, 0.0, 0.0, -2.0 / 3.0]),  # q = 1 on the axis
        "am": np.array([1.0, -4.0, 6.0, -4.0, 1.0]),  # (1 - s)^4
    },
)

_S = np.linspace(0.0, 1.0, 41)
_Q_INNER = 1.04 + 0.02 * ((0.575 - _S) / 0.575) ** 2
_Q_OUTER = 1.04 + 2.96 * ((_S - 0.575) / 0.425) ** 2

# R = 6.2, a = 2 at 5.3 T, with the pressure on the axis over the magnetic pressure
REACTOR = Case(
    ((6.2, 2.0, -0.33), (0.0, 3.4), 6.23),
    (8, 2, 33),
    0.2,
    (0.5, 0.99),
    {
        "ftol_array": np.array([1.0e-18]),
        "niter_array": np.array([150000]),
        "delt": 0.25,
        "phiedge": np.pi * 2.0 * 3.4 * 5.3,
        "piota_type": "cubic_spline",
        "ai_aux_s": _S,
        "ai_aux_f": 1.0 / np.where(_S < 0.575, _Q_INNER, _Q_OUTER),
        "am": np.array([1.0, -2.0, 1.0]) * 5.3**2 / (8.0e-7 * np.pi),  # (1 - s)^2
    },
)


def create_model(vmec_input):
    ns = int(vmec_input.ns_array[-1])
    return _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), ns, None)


def solve(vmec_input, x_start=None):
    """The equilibrium the solver reaches from the input's guess or from x_start, as a
    VmecModel evaluated there and its state."""
    model = create_model(vmec_input)
    if x_start is not None:
        model.set_state(x_start)
        model.save_backup()
    model.solve()
    model.evaluate(2, 2, False)
    return model, np.asarray(model.get_state()).copy()


def residual(model) -> float:
    return max(model.fsqr, model.fsqz, model.fsql)


def free_coefficients(model) -> np.ndarray:
    """The coefficients the solver evolves, from a state without symmetry."""
    model.evaluate(2, 2, True)
    return np.flatnonzero(np.asarray(model.get_forces()) != 0.0)


def mode_indices(model, indices):
    """Block, surface, m and n of state coefficients."""
    size = model.ns * model.mpol * (model.ntor + 1)
    blocks = np.asarray(model.get_state()).size // size
    return np.unravel_index(indices, (blocks, model.ns, model.mpol, model.ntor + 1))


def row_scale(model, free) -> np.ndarray:
    """The raw force over (ns - 1) dW/dx, smaller by sqrt(2) in R_ss at m = 1."""
    block, _, m, _ = mode_indices(model, free)
    return np.where((block == 1) & (m == 1), np.sqrt(0.5), 1.0)


def precondition(model, free, v) -> np.ndarray:
    """M^-1 v on the free coefficients."""
    full = np.zeros_like(model.get_state())
    full[free] = v
    return -np.asarray(model.apply_preconditioner(full))[free]


def spectrum(model, free, state, coordinates=None) -> np.ndarray:
    """The eigenvalues, ascending, of the force Jacobian at a state, in coordinates
    where the preconditioner is the identity."""
    rows = row_scale(model, free)
    metric = np.column_stack(
        [precondition(model, free, rows * e) for e in np.eye(free.size)]
    )
    factor = np.linalg.cholesky(0.5 * (metric + metric.T))
    model.set_state(state)
    model.evaluate(2, 2, False)
    coordinates = np.arange(free.size) if coordinates is None else coordinates
    matrix = np.empty((coordinates.size, coordinates.size))
    for column, k in enumerate(coordinates):
        v = np.zeros_like(state)
        v[free] = factor[:, k]
        force = np.asarray(model.hessian_vector_product(v))[free] / rows
        matrix[:, column] = (factor.T @ force)[coordinates]
    return np.sort(np.linalg.eigvals(matrix).real)


class _Converged(Exception):
    """The state on the line has reached ftol."""


class Saddle:
    """The saddle at the error field eps on the line from the minimum x_from to x_to.

    Its amplitude tau is c . x, with x the free coefficients measured from x_from and c
    the direction of the line in the metric of the preconditioner.
    """

    def __init__(self, case, eps, model, x_from, x_to, free):
        self.case, self.eps, self.model, self.free = case, eps, model, free
        self.x0 = x_from.copy()
        model.set_state(x_from)
        model.evaluate(2, 2, True)
        self.energy_minimum = model.mhd_energy
        self.rows = row_scale(model, free)
        direction = (x_to - x_from)[free]
        metric = LinearOperator(
            (free.size, free.size),
            matvec=lambda v: precondition(model, free, self.rows * v),
        )
        solved, _ = cg(metric, direction, rtol=1.0e-12)
        self.length = np.sqrt(direction @ solved)
        self.c, self.along = solved / self.length, direction / self.length
        self.x = np.zeros_like(direction)

    def state(self, x) -> np.ndarray:
        full = self.x0.copy()
        full[self.free] += x
        return full

    def relax(self, x, tau, across=0.02):
        """Descend at fixed tau until the force across the line is below across |mu|.

        Returns x and the force mu along the line.
        """
        x = x + (tau - self.c @ x) * self.along
        step, momentum = self.case.heavy_ball
        velocity, previous = np.zeros_like(x), x
        for _ in range(MAX_STEPS):
            self.model.set_state(self.state(x))
            self.model.evaluate(2, 2, False)
            if self.model.restart_reason == RestartReason.BAD_JACOBIAN:
                x, velocity, step = previous, np.zeros_like(x), 0.5 * step
                continue
            previous = x
            force = np.asarray(self.model.get_forces())[self.free]
            descent = precondition(self.model, self.free, force)
            mu = self.c @ descent
            descent = descent - mu * self.along
            gradient = force / self.rows - mu * self.c
            size = np.sqrt(abs(gradient @ descent))
            if residual(self.model) < self.model.ftolv or size < across * abs(mu):
                return x, mu
            velocity = momentum * velocity - step * descent
            if velocity @ gradient > 0.0:
                velocity[:] = 0.0
            x = x + velocity
        msg = f"the relaxation at tau = {tau} did not converge in {MAX_STEPS} steps"
        raise RuntimeError(msg)

    def find(self, fraction=0.5):
        """Find the zero of mu nearest a fraction of the line.

        Returns self, or None if steps along the line reach an end of it first.
        """
        known, path = {}, [(-1.0, self.x), (0.0, self.x)]

        def mu_at(tau):
            if tau not in known:
                (tau_a, x_a), (tau_b, x_b) = path[-2:]
                guess = x_b + (tau - tau_b) / (tau_b - tau_a) * (x_b - x_a)
                self.x, known[tau] = self.relax(guess, tau)
                path.append((tau, self.x))
                if residual(self.model) < self.model.ftolv:
                    raise _Converged
            return known[tau]

        tau = fraction * self.length
        try:
            ratio = 1.3 if mu_at(tau) > 0.0 else 1.0 / 1.3
            while True:
                last, tau = tau, min(ratio * tau, 0.97 * self.length)
                if mu_at(last) * mu_at(tau) <= 0.0:
                    break
                if tau >= 0.97 * self.length or tau < 1.0e-4 * self.length:
                    return None
            brentq(mu_at, last, tau, xtol=1.0e-14)
        except _Converged:
            pass
        self.tau = self.c @ self.x
        self.residual, self.energy = residual(self.model), self.model.mhd_energy
        return self

    @property
    def fraction(self) -> float:
        return self.tau / self.length

    def barrier(self) -> float:
        """The work against the force from the minimum to the saddle, in the units of
        W."""
        nodes, weights = np.polynomial.legendre.leggauss(6)
        work = sum(
            weight * self.relax(s * self.x, s * self.tau, 0.01)[1]
            for s, weight in zip(0.5 * (nodes + 1.0), weights, strict=True)
        )
        return 0.5 * self.tau * work / (self.model.ns - 1)

    def lowest_eigenvalues(self) -> np.ndarray:
        return spectrum(self.model, self.free, self.state(self.x))[:2]


def mirror_minima(case, pressure):
    """The helical cores at eps = 0, as the model at the first, the two states and the
    free coefficients."""
    model, x = solve(case.make(pressure, 0.0, case.seed))
    _, x_mirror = solve(case.make(pressure, 0.0, -case.seed))
    free = free_coefficients(create_model(case.make(pressure, 0.0, case.seed)))
    return model, x, x_mirror, free


def ladder(case, pressure, stop_fraction=0.05):
    """The saddles at eps rising from zero to the fold, each from the nearer minimum."""

    def guess(eps, seed):
        return np.asarray(create_model(case.make(pressure, eps, seed)).get_state())

    _, x_plus, x_minus, free = mirror_minima(case, pressure)
    states = {case.seed: x_plus, -case.seed: x_minus}
    near = case.seed
    rows, eps, step = [], 0.0, FIRST_EPS
    while step > 1.0e-4 * eps and len(rows) < 16:
        trial = eps + step if rows else 0.0
        solved = {
            seed: solve(
                case.make(pressure, trial, seed),
                x + guess(trial, seed) - guess(eps, seed),
            )
            for seed, x in states.items()
        }
        (model_a, x_a), (model_b, x_b) = solved[near], solved[-near]
        saddle = None
        if max(residual(model_a), residual(model_b)) < 1.0e-11 and np.linalg.norm(
            (x_a - x_b)[free]
        ) > 1.0e-3 * np.linalg.norm(x_b[free]):
            fraction = 0.7 * rows[-1].fraction if rows else 0.5
            saddle = Saddle(case, trial, model_a, x_a, x_b, free).find(fraction)
        if saddle is not None and saddle.fraction > 0.51:
            near = -near
            saddle = Saddle(case, trial, model_b, x_b, x_a, free).find(
                1.0 - saddle.fraction
            )
        if saddle is None:
            step *= 0.5
            continue
        rows.append(saddle)
        eps, states = trial, {seed: solved[seed][1] for seed in states}
        if saddle.fraction < stop_fraction:
            break
        if len(rows) >= 2:
            # the squared distance to the nearer minimum falls linearly to the fold
            a, b = (min(row.tau, row.length - row.tau) ** 2 for row in rows[-2:])
            slope = (b - a) / (rows[-1].eps - rows[-2].eps)
            step = 0.6 * b / -slope if slope < 0.0 else 2.0 * step
    return rows


def near_fold(rows, count=4):
    """The last rows of a ladder, nearer the fold than the middle of the line."""
    return [row for row in rows if row.fraction < 0.45][-count:]


def fold_position(rows, values) -> float:
    """The eps at which a quantity on the rows falls linearly to zero."""
    slope, offset = np.polyfit([row.eps for row in rows], values, 1)
    return -offset / slope


def stability_limit(case, pressures) -> float:
    """The pressure at which the critical eigenvalue of the axisymmetric equilibrium,
    extrapolated linearly from two pressures, reaches zero."""
    probe = create_model(case.make(pressures[0], 0.0, case.seed))
    free = free_coefficients(probe)
    # the critical mode has odd n
    odd = np.flatnonzero(mode_indices(probe, free)[3] % 2 == 1)
    values = []
    for pressure in pressures:
        model, x = solve(case.make(pressure))
        model.evaluate(2, 2, True)
        values.append(spectrum(model, free, x, odd)[0])
    (p_a, p_b), (l_a, l_b) = pressures, values
    return p_b - l_b * (p_b - p_a) / (l_b - l_a)


def exponent(x, y) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactor", action="store_true")
    parser.add_argument("--pressures", type=float, nargs="+")
    parser.add_argument("--resolution", type=int, nargs=3, help="mpol, ntor and ns")
    args = parser.parse_args()
    case = REACTOR if args.reactor else SPHERICAL_TOKAMAK
    if args.resolution:
        case = case._replace(resolution=tuple(args.resolution))
    pressures = args.pressures or (
        (0.054, 0.056, 0.06) if args.reactor else (2.65e4, 2.7e4, 2.8e4)
    )

    p_c = stability_limit(case, sorted(pressures)[:2])
    print(f"p_c = {p_c:.6g}")
    print(
        "p, a0, barrier, W barrier, eps_c, eps_c / a0^3, saddle, fsq, two eigenvalues"
    )
    table = []
    for pressure in pressures:
        rows = ladder(case, pressure)
        a0, barrier = rows[0].length, rows[0].barrier()
        near = near_fold(rows)
        eps_c = fold_position(near, [row.tau**2 for row in near])
        table.append((pressure, a0, barrier, eps_c))
        lowest, second = rows[-1].lowest_eigenvalues()
        print(
            f"{pressure:.4g} {a0:.4f} {barrier:.3e} "
            f"{rows[0].energy - rows[0].energy_minimum:.3e} {eps_c:.4e} "
            f"{eps_c / a0**3:.4f} {rows[-1].fraction:.3f} {rows[-1].residual:.1e} "
            f"{lowest:.2e} {second:.2e}"
        )
    pressure, a0, barrier, eps_c = (np.array(c) for c in zip(*table, strict=True))
    print(f"a0^2 ~ (p - p_c)^{exponent(pressure - p_c, a0**2):.3f}")
    print(f"barrier ~ (p - p_c)^{exponent(pressure - p_c, barrier):.3f}")
    print(f"eps_c ~ (p - p_c)^{exponent(pressure - p_c, eps_c):.3f}")
    print(f"eps_c ~ a0^{exponent(a0, eps_c):.3f}")


if __name__ == "__main__":
    main()

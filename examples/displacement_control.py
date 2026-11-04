# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Equilibria of either stability by displacement control, and the barrier between them.

The descent of VMEC++ converges to minima of its functional. It reaches an unstable
equilibrium only from an initial guess with a symmetry that protects it. Displacement
control works on the line from one equilibrium to another, in coordinates y with
x = x0 + L y, where L L^T is VMEC's preconditioner. It holds the amplitude tau = y . e
along the unit vector e of the line, relaxes every other direction by a heavy-ball
descent, and reads the force left along the line, mu(tau) = g . e. A zero of mu is a
state with no force in any direction, whatever its stability. A bracket and the Illinois
variant of regula falsi find it with forces alone.

W is not the potential of this force, and it is not stationary at an equilibrium. The
lambda rows are the hybrid force on the full radial grid and vanish where dW/dlambda does
not, and the R and Z rows carry the constraint. Without the constraint, the R and Z rows
are (ns - 1) dW/dx, and the rows of R_ss with m = 1 are smaller by sqrt(2), since that
coefficient carries Z_cs with it. With those rows rescaled, the asymmetry of the Jacobian
of the force is mostly in the lambda rows, and the integral of mu along the relaxed line
over ns - 1 is a barrier in the units of W that obeys the laws of a gradient system.

The configurations are a spherical tokamak and a hybrid-scenario tokamak at reactor beta.
Their axisymmetric equilibria lose stability to the m = 1, n = 1 internal kink at a
threshold p_c, and above it the helical core comes as a pair of mirror images. A helical
shift eps of the boundary, in the m = 0, n = 1 coefficients of R and Z, breaks that
symmetry. The amplitude a of the kink then sees the potential of an imperfect pitchfork,

    V(a) = -r a^2 / 2 + g a^4 / 4 - h a,   r ~ p - p_c,   h ~ eps.

At eps = 0 this gives a0^2 ~ p - p_c and a barrier ~ (p - p_c)^2. With eps the higher
minimum and the saddle approach each other and merge at a fold, with
eps_c ~ (p - p_c)^(3/2) ~ a0^3. Near the fold, the squared distance between the two and
the squared lowest eigenvalue fall linearly in eps_c - eps, and the barrier falls as
(eps_c - eps)^(3/2). No symmetry protects this saddle, so the solver cannot reach it.

These states are those of the stellarator-symmetric space of the input. With lasym = True
the core can also turn toroidally: the lower state stays a minimum, the higher one gains
an unstable direction and the saddle a second one.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy import linalg, sparse

import vmecpp
from vmecpp._iteration import RestartReason
from vmecpp.cpp import _vmecpp  # type: ignore[import]

MU_0 = 4.0e-7 * np.pi
MAX_STEPS = 60000
WORK_NODES = 6
# growth of the amplitude per step of the bracket
BRACKET_RATIO = 1.3
# fraction of the distance to the estimated fold covered per step in eps
APPROACH = 0.6
# the laws of the fold are fitted to saddles nearer than this fraction of the line
NEAR_FOLD = 0.45


def tokamak_input(r_mn, z_mn, r_axis, eps, seed, mpol, ntor, ns, **others):
    """A stellarator-symmetric input with one field period, the axisymmetric boundary of
    coefficients r_mn and z_mn by m, and the helical shift eps."""
    rbc, zbs = np.zeros((mpol, 2 * ntor + 1)), np.zeros((mpol, 2 * ntor + 1))
    rbc[: len(r_mn), ntor], zbs[: len(z_mn), ntor] = r_mn, z_mn
    rbc[0, ntor + 1], zbs[0, ntor + 1] = eps, -eps
    raxis_c, zaxis_s = np.zeros(ntor + 1), np.zeros(ntor + 1)
    raxis_c[:2], zaxis_s[1] = (r_axis, seed), -seed
    return vmecpp.VmecInput(
        lasym=False,
        nfp=1,
        mpol=mpol,
        ntor=ntor,
        ns_array=np.array([ns]),
        ncurr=0,
        pmass_type="power_series",
        gamma=0.0,
        always_fix_m1_gauge=True,
        rbc=rbc,
        zbs=zbs,
        raxis_c=raxis_c,
        zaxis_s=zaxis_s,
        **others,
    )


def spherical_tokamak_input(pres_scale, eps=0.0, seed=0.0, mpol=6, ntor=1, ns=25):
    """Major radius 1.5, minor radius 1 and elongation 1.6, with q = 1 on the axis and
    low shear out to s = 0.5."""
    shape = (1.5, 1.0), (0.0, 1.6), 1.5
    return tokamak_input(
        *shape,
        eps,
        seed,
        mpol,
        ntor,
        ns,
        ftol_array=np.array([1.0e-20]),
        niter_array=np.array([200000]),
        phiedge=5.0,
        # iota = 1 - (2/3) s^4: q = 1 on the axis and 3 at the boundary
        ai=np.array([1.0, 0.0, 0.0, 0.0, -2.0 / 3.0]),
        am=np.array([1.0, -4.0, 6.0, -4.0, 1.0]),  # p = p0 (1 - s)^4
        pres_scale=pres_scale,
    )


def reactor_input(beta0, eps=0.0, seed=0.0, mpol=8, ntor=2, ns=33):
    """Major radius 6.2, minor radius 2, elongation 1.7 and triangularity 0.33 at 5.3 T,
    with q = 1.06 on the axis, 1.04 at s = 0.575 and 4 at the boundary.

    beta0 is the pressure on the axis over the magnetic pressure.
    """
    s = np.linspace(0.0, 1.0, 41)
    q_inner = 1.04 + 0.02 * ((0.575 - s) / 0.575) ** 2
    q_outer = 1.04 + 2.96 * ((s - 0.575) / 0.425) ** 2
    shape = (6.2, 2.0, -0.33), (0.0, 3.4), 6.23
    return tokamak_input(
        *shape,
        eps,
        seed,
        mpol,
        ntor,
        ns,
        ntheta=2 * mpol + 8,
        nzeta=2 * ntor + 8,
        ftol_array=np.array([1.0e-18]),
        niter_array=np.array([150000]),
        delt=0.25,
        phiedge=np.pi * 2.0 * 3.4 * 5.3,
        piota_type="cubic_spline",
        ai_aux_s=s,
        ai_aux_f=1.0 / np.where(s < 0.575, q_inner, q_outer),
        am=np.array([1.0, -2.0, 1.0]),  # p = p0 (1 - s)^2
        pres_scale=beta0 * 5.3**2 / (2.0 * MU_0),
    )


@dataclass(frozen=True)
class Case:
    """An input, the helical seed that reaches its helical core, and the heavy-ball
    parameters that are stable on its force."""

    make: Callable[..., vmecpp.VmecInput]
    seed: float
    step: float
    momentum: float


SPHERICAL_TOKAMAK = Case(spherical_tokamak_input, 0.3, 0.8, 0.97)
REACTOR = Case(reactor_input, 0.2, 0.5, 0.99)


def create_model(vmec_input: vmecpp.VmecInput):
    return _vmecpp.VmecModel.create(
        vmec_input._to_cpp_vmecindata(), int(vmec_input.ns_array[-1]), None
    )


def solve(vmec_input: vmecpp.VmecInput, x_start: np.ndarray | None = None):
    """The equilibrium the native solver reaches from the initial guess of the input or
    from the state x_start, as a VmecModel and its state."""
    model = create_model(vmec_input)
    if x_start is not None:
        model.set_state(x_start)
        model.save_backup()
    model.solve()
    model.evaluate(2, 2, False)
    return model, np.asarray(model.get_state()).copy()


def follow(case: Case, pressure, seed, eps, x_previous, eps_previous):
    """The minimum at eps, hot-started from the one at the previous eps with the
    boundary moved."""

    def guess(value):
        return np.asarray(create_model(case.make(pressure, value, seed)).get_state())

    return solve(
        case.make(pressure, eps, seed), x_previous + guess(eps) - guess(eps_previous)
    )


def residual(model) -> float:
    """VMEC's convergence measure at the last evaluated state."""
    return max(model.fsqr, model.fsqz, model.fsql)


def free_coefficients(model) -> np.ndarray:
    """Indices of the state coefficients the solver evolves: those with a nonzero
    preconditioned force at a state without symmetry."""
    model.evaluate(2, 2, True)
    return np.flatnonzero(np.asarray(model.get_forces()) != 0.0)


def mode_indices(model, indices: np.ndarray):
    """The block, surface, m and n of state coefficients.

    The blocks are R cos cos, R sin sin, Z sin cos, Z cos sin, lambda sin cos and lambda
    cos sin, with the blocks without stellarator symmetry between them if there are any.
    """
    per_block = model.ns * model.mpol * (model.ntor + 1)
    blocks = np.asarray(model.get_state()).size // per_block
    return np.unravel_index(indices, (blocks, model.ns, model.mpol, model.ntor + 1))


class Landscape:
    """The force on the free coefficients of a VmecModel as a gradient in coordinates y.

    The state at y is x0 + L y, where L L^T is the preconditioner at x0 with the rows of
    R_ss with m = 1 rescaled. L has one block per Fourier mode and is stored sparse.
    """

    def __init__(self, model, x0: np.ndarray, free: np.ndarray):
        self.model, self.x0, self.free = model, x0.copy(), free
        block, _, m, _ = mode_indices(model, free)
        self.rows = np.where((block == 1) & (m == 1), 1.0 / np.sqrt(2.0), 1.0)
        model.set_state(x0)
        model.evaluate(2, 2, True)
        inverse_metric = np.empty((free.size, free.size))
        for column, i in enumerate(free):
            unit = np.zeros_like(x0)
            unit[i] = self.rows[column]
            # apply_preconditioner returns -M^-1 v
            inverse_metric[:, column] = -np.asarray(model.apply_preconditioner(unit))[
                free
            ]
        self.factor = linalg.cholesky(
            0.5 * (inverse_metric + inverse_metric.T), lower=True
        )
        self.L = sparse.csr_matrix(self.factor)
        self.L_transpose = self.L.T.tocsr()
        self.force_evaluations = 0

    def state(self, y: np.ndarray) -> np.ndarray:
        x = self.x0.copy()
        x[self.free] += self.L @ y
        return x

    def coordinates(self, x: np.ndarray) -> np.ndarray:
        return linalg.solve_triangular(
            self.factor, (x - self.x0)[self.free], lower=True
        )

    def gradient(self, y: np.ndarray) -> np.ndarray:
        """The rescaled force at y in the coordinates y.

        Leaves the model evaluated at y.
        """
        self.model.set_state(self.state(y))
        self.model.evaluate(2, 2, False)
        self.force_evaluations += 1
        return self.L_transpose @ (
            np.asarray(self.model.get_forces())[self.free] / self.rows
        )

    def jacobian(self, y: np.ndarray, coordinates=None) -> np.ndarray:
        """The derivative of the gradient at y on the given coordinates, by default
        all."""
        x = self.state(y)
        self.model.set_state(x)
        self.model.evaluate(2, 2, False)
        coordinates = np.arange(self.free.size) if coordinates is None else coordinates
        matrix = np.empty((coordinates.size, coordinates.size))
        for column, k in enumerate(coordinates):
            v = np.zeros_like(x)
            v[self.free] = self.factor[:, k]
            force = np.asarray(self.model.hessian_vector_product(v))[self.free]
            matrix[:, column] = (self.L_transpose @ (force / self.rows))[coordinates]
        return matrix

    def lowest_eigenvalues(self, y: np.ndarray, count: int = 2) -> np.ndarray:
        """The lowest eigenvalues of the force Jacobian at y."""
        return np.sort(np.linalg.eigvals(self.jacobian(y)).real)[:count]


class Saddle:
    """The straight line from the minimum x_from to the minimum x_to, with y = 0 at
    x_from, and the saddle on it at the error field eps.

    The model is evaluated at x_from.
    """

    def __init__(self, case: Case, eps: float, model, x_from, x_to, free):
        self.case, self.eps, self.energy_minimum = case, eps, model.mhd_energy
        self.landscape = Landscape(model, x_from, free)
        direction = self.landscape.coordinates(x_to)
        self.length = np.linalg.norm(direction)
        self.e = direction / self.length
        self.ftol = float(case.make(1.0).ftol_array[-1])
        self.tau, self.y, self.x = 0.0, np.zeros_like(direction), x_from
        self.residual, self.energy = np.inf, model.mhd_energy

    def relax(self, y: np.ndarray, tau: float, across: float = 0.02):
        """Descend with y . e = tau held, until the force across the line is below
        ``across`` |mu| or the state has converged.

        Returns y and mu at y.
        """
        y = y + (tau - y @ self.e) * self.e
        velocity, step, previous = np.zeros_like(y), self.case.step, y
        for _ in range(MAX_STEPS):
            g = self.landscape.gradient(y)
            if self.landscape.model.restart_reason == RestartReason.BAD_JACOBIAN:
                y, velocity, step = previous, np.zeros_like(y), 0.5 * step
                continue
            previous = y
            mu = g @ self.e
            g_across = g - mu * self.e
            if residual(self.landscape.model) < self.ftol or np.linalg.norm(
                g_across
            ) < across * abs(mu):
                return y, mu
            velocity = self.case.momentum * velocity - step * g_across
            if velocity @ g_across > 0.0:
                velocity[:] = 0.0
            y = y + velocity
            y = y + (tau - y @ self.e) * self.e
        msg = f"the relaxation at tau = {tau} did not converge in {MAX_STEPS} steps"
        raise RuntimeError(msg)

    def bracket(self, fraction: float):
        """Two relaxed states (tau, y, mu), with mu falling through zero between them,
        by steps from the fraction of the line given.

        None if the steps reach an end of the line first.
        """
        tau = fraction * self.length
        y, mu = self.relax(np.zeros_like(self.e), tau)
        ratio = BRACKET_RATIO if mu > 0.0 else 1.0 / BRACKET_RATIO
        while True:
            previous = (tau, y, mu)
            tau = min(ratio * tau, 0.97 * self.length)
            y, mu = self.relax(y, tau)
            if mu * previous[2] <= 0.0:
                return (
                    (previous, (tau, y, mu)) if mu <= 0.0 else ((tau, y, mu), previous)
                )
            if tau >= 0.97 * self.length or tau < 1.0e-4 * self.length:
                return None

    def find(self, fraction: float = 0.5) -> bool:
        """Find the zero of mu nearest the fraction of the line given, by the Illinois
        variant of regula falsi.

        False if there is no bracket.
        """
        bracket = self.bracket(fraction)
        if bracket is None:
            return False
        (tau_a, y_a, f_a), (tau_b, y_b, f_b) = bracket
        side = 0
        for _ in range(60):
            self.tau = (tau_a * f_b - tau_b * f_a) / (f_b - f_a)
            guess = y_a + (self.tau - tau_a) / (tau_b - tau_a) * (y_b - y_a)
            self.y, mu = self.relax(guess, self.tau)
            model = self.landscape.model
            if residual(model) < self.ftol:
                self.x = self.landscape.state(self.y)
                self.residual, self.energy = residual(model), model.mhd_energy
                return True
            if mu * f_b > 0.0:
                tau_b, y_b, f_b = self.tau, self.y, mu
                f_a = 0.5 * f_a if side == -1 else f_a
                side = -1
            else:
                tau_a, y_a, f_a = self.tau, self.y, mu
                f_b = 0.5 * f_b if side == 1 else f_b
                side = 1
        msg = "the root of the force along the line did not converge"
        raise RuntimeError(msg)

    @property
    def fraction(self) -> float:
        return self.tau / self.length

    def barrier(self) -> float:
        """The work against the force from y = 0 to the saddle, by Gauss-Legendre
        quadrature, in the units of W."""
        nodes, weights = np.polynomial.legendre.leggauss(WORK_NODES)
        at = 0.5 * (nodes + 1.0)
        work = sum(
            0.5 * self.tau * weight * self.relax(s * self.y, s * self.tau, 0.01)[1]
            for s, weight in zip(at, weights, strict=True)
        )
        return work / (self.landscape.model.ns - 1)

    def lowest_eigenvalues(self) -> np.ndarray:
        return self.landscape.lowest_eigenvalues(self.y)

    def lowest_at_minimum(self) -> float:
        return self.landscape.lowest_eigenvalues(np.zeros_like(self.y), 1)[0]


def saddle_between(case, eps, model, x_from, x_to, free, fraction=0.5):
    """The Saddle from the minimum x_from to the minimum x_to, None if there is no
    bracket from the fraction of the line given."""
    saddle = Saddle(case, eps, model, x_from, x_to, free)
    return saddle if saddle.find(fraction) else None


def lowest_without_symmetry(vmec_input: vmecpp.VmecInput, x: np.ndarray, count=3):
    """The lowest eigenvalues of the force Jacobian at the stellarator-symmetric state x
    in the larger space of lasym = True."""
    nothing, seed = np.zeros_like(vmec_input.rbc), np.zeros_like(vmec_input.raxis_c)
    seed[1] = 0.1
    model = create_model(
        vmec_input.model_copy(
            update={
                "lasym": True,
                "rbs": nothing,
                "zbc": nothing,
                "raxis_s": seed,
                "zaxis_c": seed,
            }
        )
    )
    free = free_coefficients(model)
    state = np.zeros_like(model.get_state())
    per_block = x.size // 6
    # the blocks of the symmetric state among the twelve of the larger one
    for k, block in enumerate((0, 1, 4, 5, 8, 9)):
        state[block * per_block : (block + 1) * per_block] = x[
            k * per_block : (k + 1) * per_block
        ]
    return Landscape(model, state, free).lowest_eigenvalues(np.zeros(free.size), count)


def mirror_minima(case: Case, pressure: float):
    """The two helical cores at eps = 0, as the model at the first, the two states and
    the free coefficients."""
    model, x = solve(case.make(pressure, 0.0, case.seed))
    _, x_mirror = solve(case.make(pressure, 0.0, -case.seed))
    free = free_coefficients(create_model(case.make(pressure, 0.0, case.seed)))
    model.set_state(x)
    model.evaluate(2, 2, False)
    return model, x, x_mirror, free


def odd_amplitude(model, x: np.ndarray) -> float:
    """The largest coefficient of the state with odd toroidal mode number."""
    n = mode_indices(model, np.arange(x.size))[3]
    return float(np.abs(x[n % 2 == 1]).max())


def ladder(case: Case, pressure: float, first: float, stop_fraction: float = 0.05):
    """The saddle at eps rising from zero up to the fold, as a list of Saddle.

    Each is measured from the nearer minimum, the one that folds. W does not tell which:
    its difference between the minima can have the opposite sign.
    """
    _, x_plus, x_minus, free = mirror_minima(case, pressure)
    states = {case.seed: x_plus, -case.seed: x_minus}
    near = case.seed
    rows, eps, step = [], 0.0, first
    while step > first / 64.0 and len(rows) < 16:
        trial = eps + step if rows else 0.0
        solved = {
            seed: follow(case, pressure, seed, trial, x, eps)
            for seed, x in states.items()
        }
        (model_a, x_a), (model_b, x_b) = solved[near], solved[-near]
        distance = np.linalg.norm((x_a - x_b)[free]) / np.linalg.norm(x_b[free])
        saddle = None
        if max(residual(model_a), residual(model_b)) < 1.0e-11 and distance > 1.0e-3:
            fraction = 0.7 * rows[-1].fraction if rows else 0.5
            saddle = saddle_between(case, trial, model_a, x_a, x_b, free, fraction)
        if saddle is not None and saddle.fraction > 0.51:
            near = -near
            saddle = saddle_between(
                case, trial, model_b, x_b, x_a, free, 1.0 - saddle.fraction
            )
        if saddle is None:
            step *= 0.5
            continue
        rows.append(saddle)
        eps, states = trial, {seed: solved[seed][1] for seed in states}
        if saddle.fraction < stop_fraction:
            break
        if len(rows) >= 2:
            # tau^2 falls linearly to zero at the fold
            slope = (saddle.tau**2 - rows[-2].tau ** 2) / (saddle.eps - rows[-2].eps)
            reach = APPROACH * saddle.tau**2 / -slope if slope < 0.0 else np.inf
            step = min(reach, 2.0 * step)
    return rows


def fold_position(eps, values) -> float:
    """The zero of the line through a quantity that falls linearly to zero at the
    fold."""
    slope, offset = np.polyfit(eps, values, 1)
    return -offset / slope


def fold_of(rows) -> float:
    """The eps of the fold from tau^2 at the nearest rows."""
    near = [row for row in rows if row.fraction < NEAR_FOLD][-4:]
    return fold_position([row.eps for row in near], [row.tau**2 for row in near])


def stability_limit(case: Case, pressures) -> float:
    """The pressure at which the critical eigenvalue of the unbroken equilibrium reaches
    zero, extrapolated linearly from two pressures, the lower one last.

    The critical mode lies in the block of coefficients with odd toroidal mode number.
    """
    probe = create_model(case.make(pressures[0], 0.0, case.seed))
    free = free_coefficients(probe)
    odd = np.flatnonzero(mode_indices(probe, free)[3] % 2 == 1)
    values = []
    for pressure in pressures:
        model, x = solve(case.make(pressure))
        matrix = Landscape(model, x, free).jacobian(np.zeros(free.size), odd)
        values.append(np.linalg.eigvals(matrix).real.min())
    (p_a, p_b), (l_a, l_b) = pressures, values
    return p_b - l_b * (p_b - p_a) / (l_b - l_a)


def exponent(x, y) -> float:
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    parser = argparse.ArgumentParser(description="Displacement control on a pitchfork.")
    parser.add_argument("--reactor", action="store_true")
    parser.add_argument("--pressures", type=float, nargs="+")
    parser.add_argument("--limit", type=float, nargs=2)
    for name in ("mpol", "ntor", "ns"):
        parser.add_argument(f"--{name}", type=int)
    args = parser.parse_args()
    base = REACTOR if args.reactor else SPHERICAL_TOKAMAK
    resolution = {
        k: getattr(args, k) for k in ("mpol", "ntor", "ns") if getattr(args, k)
    }
    case = Case(partial(base.make, **resolution), base.seed, base.step, base.momentum)
    defaults = {
        "pressures": ((0.054, 0.056, 0.06), (2.65e4, 2.7e4, 2.8e4)),
        "limit": ((0.054, 0.052), (2.7e4, 2.6e4)),
    }
    pressures = args.pressures or defaults["pressures"][0 if args.reactor else 1]
    limit = args.limit or defaults["limit"][0 if args.reactor else 1]
    # the first step in eps over the cube of the amplitude: about a sixth of the fold
    first = 0.11 if args.reactor else 4.0

    p_c = stability_limit(case, list(limit))
    print(f"the unbroken equilibrium loses its stability at p_c = {p_c:.6g}")
    print("p, a0, barrier, W barrier, eps_c, eps_c / a0^3, and nearest the fold the")
    print("saddle's fraction of the line, fsq, lowest two eigenvalues and evaluations")
    table = []
    for pressure in pressures:
        model, x, _, _ = mirror_minima(case, pressure)
        amplitude = odd_amplitude(model, x)
        rows = ladder(case, pressure, first * amplitude**3)
        barrier, eps_c = rows[0].barrier(), fold_of(rows)
        table.append((pressure, amplitude, barrier, eps_c))
        lowest, second = rows[-1].lowest_eigenvalues()
        print(
            f"{pressure:.4g} {amplitude:.4f} {barrier:.3e} "
            f"{rows[0].energy - rows[0].energy_minimum:.3e} {eps_c:.4e} "
            f"{eps_c / amplitude**3:.3f} {rows[-1].fraction:.3f} "
            f"{rows[-1].residual:.1e} {lowest:.2e} {second:.2e} "
            f"{sum(row.landscape.force_evaluations for row in rows) // len(rows)}"
        )
    pressure, amplitude, barrier, eps_c = (
        np.array(c) for c in zip(*table, strict=True)
    )
    distance = pressure - p_c
    print(f"barrier ~ (p - p_c)^{exponent(distance, barrier):.3f}")
    print(f"eps_c ~ (p - p_c)^{exponent(distance, eps_c):.3f}")
    print(f"eps_c ~ a0^{exponent(amplitude, eps_c):.3f}")


if __name__ == "__main__":
    main()

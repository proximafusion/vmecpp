# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A transition state between two equilibria, found by the dimer method.

Equilibria are stationary points of VMEC's augmented functional, the MHD energy plus
the spectral-condensation constraint. A transition state is a stationary point with a
single direction of negative curvature, the highest point on the lowest path between
two minima. The dimer method (Henkelman and Jonsson, J. Chem. Phys. 111, 7010 (1999))
finds it from a minimum with forces alone. The forces at two states a small distance
apart along a unit vector N give the curvature along N. Rotating N towards lower
curvature makes it the direction of lowest curvature, and moving the pair with the
force component along N reversed climbs along N while it descends in every other
direction.

The configuration is a spherical tokamak of major radius 1.5, minor radius 1 and
elongation 1.6, with q = 1 on the axis and low shear out to s = 0.5. Its axisymmetric
equilibrium is unstable to the m = 1, n = 1 internal kink. With the axisymmetric
boundary and the stellarator symmetry of the input, the helical-core equilibrium the
kink leads to comes as a pair of mirror images, half a toroidal turn apart. Starting
from one of them, the dimer climbs to the saddle between them. Here that saddle is the
axisymmetric equilibrium, which the solver also reaches directly from an axisymmetric
initial guess, because the n = 1 coefficients then start at zero and stay there.

The search runs on the free coefficients of the decomposed state of a ``VmecModel``,
in coordinates y with x = x0 + L y, where L L^T is VMEC's preconditioner. The dimer
first climbs along the helical part of the starting equilibrium with N held fixed, and
rotates N once the curvature along it is negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore[import]

NS = 25
MPOL = 6
NTOR = 1
FTOL = 1.0e-13


def helical_core_input(helical_seed: float = 0.0) -> vmecpp.VmecInput:
    """The spherical tokamak, with the magnetic axis guess displaced helically by
    ``helical_seed``; a seed of zero keeps the initial guess axisymmetric."""
    vmec_input = vmecpp.VmecInput(
        lasym=False,
        nfp=1,
        mpol=MPOL,
        ntor=NTOR,
        ns_array=np.array([NS]),
        ftol_array=np.array([FTOL]),
        niter_array=np.array([30000]),
        phiedge=5.0,
        ncurr=0,
        # iota = 1 - (2/3) s^4: q = 1 on the axis and 3 at the boundary
        ai=np.array([1.0, 0.0, 0.0, 0.0, -2.0 / 3.0]),
        pmass_type="power_series",
        am=np.array([1.0, -4.0, 6.0, -4.0, 1.0]),  # p = p0 (1 - s)^4
        pres_scale=1.2e5,
        gamma=0.0,
        always_fix_m1_gauge=True,
        raxis_c=np.array([1.5, helical_seed]),
        zaxis_s=np.array([0.0, -helical_seed]),
    )
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[1.5], [1.0]]), MPOL, NTOR
    )
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[0.0], [1.6]]), MPOL, NTOR
    )
    return vmec_input


def create_model(vmec_input: vmecpp.VmecInput):
    return _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), NS, None)


def solve(vmec_input: vmecpp.VmecInput):
    """The equilibrium the native solver converges to, as a VmecModel at that state."""
    model = create_model(vmec_input)
    model.solve()
    model.evaluate(2, 2, False)
    return model


def free_coefficients(model) -> np.ndarray:
    """Indices of the state coefficients the solver evolves: those with a nonzero
    preconditioned force at a state without symmetry."""
    model.evaluate(2, 2, True)
    return np.flatnonzero(np.asarray(model.get_forces()) != 0.0)


def odd_n(model) -> np.ndarray:
    """Mask of the state coefficients with odd toroidal mode number n.

    The state holds R cos cos, R sin sin, Z sin cos, Z cos sin, lambda sin cos and
    lambda cos sin, each ordered by surface, then m, then n.
    """
    n = np.indices((6, model.ns, model.mpol, model.ntor + 1))[3].ravel()
    return n % 2 == 1


class Landscape:
    """VMEC's augmented functional on the free coefficients, in coordinates y.

    The state at y is x0 + L y, where L L^T is the preconditioner at x0.
    """

    def __init__(self, model, x0: np.ndarray, free: np.ndarray):
        self.model = model
        self.x0 = x0.copy()
        self.free = free
        model.set_state(x0)
        model.evaluate(2, 2, True)
        columns = []
        for i in free:
            unit = np.zeros_like(x0)
            unit[i] = 1.0
            # apply_preconditioner returns -M^-1 v
            columns.append(-np.asarray(model.apply_preconditioner(unit))[free])
        inverse_metric = np.array(columns).T
        self.L = np.linalg.cholesky(0.5 * (inverse_metric + inverse_metric.T))
        self.force_evaluations = 0

    def state(self, y: np.ndarray) -> np.ndarray:
        x = self.x0.copy()
        x[self.free] += self.L @ y
        return x

    def coordinates(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self.L, (x - self.x0)[self.free])

    def gradient(self, y: np.ndarray) -> np.ndarray:
        """The raw force at y, in the coordinates y.

        It is the gradient of the functional up to a positive row scaling. Leaves the
        model evaluated at y.
        """
        self.model.set_state(self.state(y))
        self.model.evaluate(2, 2, False)
        self.force_evaluations += 1
        return self.L.T @ np.asarray(self.model.get_forces())[self.free]

    def residual(self) -> float:
        """VMEC's convergence measure at the last evaluated state."""
        return max(self.model.fsqr, self.model.fsqz, self.model.fsql)


def rotate(landscape, y, direction, g0, separation, max_rotations=4):
    """Turn the dimer at y towards the direction of lowest curvature.

    Each rotation fits the curvature as a function of the rotation angle from one trial
    rotation and turns to the minimum of the fit (Kaestner and Sherwood, J. Chem. Phys.
    128, 014106 (2008)). Returns the new direction and its curvature.
    """
    g1 = landscape.gradient(y + separation * direction)
    curvature = (g1 - g0) @ direction / separation
    trial_angle = np.pi / 16.0
    for _ in range(max_rotations):
        rotational_force = -(g1 - g0) / separation + curvature * direction
        size = np.linalg.norm(rotational_force)
        if size == 0.0:
            break
        theta = rotational_force / size
        trial = direction * np.cos(trial_angle) + theta * np.sin(trial_angle)
        g1_trial = landscape.gradient(y + separation * trial)
        curvature_trial = (g1_trial - g0) @ trial / separation
        b1 = -size
        a1 = (curvature - curvature_trial + b1 * np.sin(2.0 * trial_angle)) / (
            1.0 - np.cos(2.0 * trial_angle)
        )
        angle = 0.5 * np.arctan2(b1, a1)
        if (a1 * np.cos(2.0 * angle) + b1 * np.sin(2.0 * angle)) > a1:
            angle += 0.5 * np.pi
        # end-point force at the new angle, interpolated from the two evaluated ones
        g1 = (
            np.sin(trial_angle - angle) / np.sin(trial_angle) * g1
            + np.sin(angle) / np.sin(trial_angle) * g1_trial
            + (1.0 - np.cos(angle) - np.sin(angle) * np.tan(0.5 * trial_angle)) * g0
        )
        direction = direction * np.cos(angle) + theta * np.sin(angle)
        direction /= np.linalg.norm(direction)
        curvature = (g1 - g0) @ direction / separation
        if abs(angle) < 1.0e-3:
            break
    return direction, curvature


def dimer_search(
    landscape,
    y_start,
    direction,
    separation,
    climb_rate,
    *,
    step=1.0,
    momentum=0.97,
    ftol=FTOL,
    max_iterations=20000,
):
    """Move a dimer from y_start, oriented along direction, to a saddle point.

    The dimer takes the force at y and at y + separation * direction. While the
    curvature along the dimer is positive, it climbs along its fixed direction, pushed
    by climb_rate per step, and relaxes the other directions. Once the curvature is
    negative it rotates every step and moves with the force component along the dimer
    reversed. Steps follow a heavy-ball iteration whose velocity is reset whenever it
    points against the force. Returns the final y, direction, curvature and iteration
    count.
    """
    direction = direction / np.linalg.norm(direction)
    y = y_start.copy()
    velocity = np.zeros_like(y)
    rotating = False
    curvature = np.inf
    for iteration in range(max_iterations):
        g0 = landscape.gradient(y)
        if rotating and landscape.residual() < ftol:
            return y, direction, curvature, iteration
        if rotating:
            direction, curvature = rotate(landscape, y, direction, g0, separation)
        elif iteration % 10 == 0:
            g1 = landscape.gradient(y + separation * direction)
            curvature = (g1 - g0) @ direction / separation
            rotating = curvature < 0.0
        parallel = (g0 @ direction) * direction
        if rotating:
            force = -g0 + 2.0 * parallel
        else:
            force = -(g0 - parallel) + climb_rate * direction
        velocity = momentum * velocity + step * force
        if velocity @ force < 0.0:
            velocity[:] = 0.0
        y = y + velocity
    msg = f"the dimer search did not converge in {max_iterations} iterations"
    raise RuntimeError(msg)


def negative_curvatures(landscape, y) -> int:
    """Number of directions of negative curvature at y, from the eigenvalues of the
    Hessian in y, assembled column by column from Hessian-vector products."""
    model = landscape.model
    x = landscape.state(y)
    model.set_state(x)
    model.evaluate(2, 2, False)
    columns = []
    for k in range(landscape.free.size):
        v = np.zeros_like(x)
        v[landscape.free] = landscape.L[:, k]
        columns.append(np.asarray(model.hessian_vector_product(v))[landscape.free])
    hessian = landscape.L.T @ np.array(columns).T
    eigenvalues = np.linalg.eigvals(hessian).real
    return int(np.sum(eigenvalues < -1.0e-6 * np.abs(eigenvalues).max()))


@dataclass
class TransitionState:
    """The saddle point a dimer search reached, and the minimum it started from."""

    landscape: Landscape
    y: np.ndarray
    curvature: float
    iterations: int
    energy: float
    x_min: np.ndarray
    energy_min: float
    odd: np.ndarray


def climb_from_helical_minimum() -> TransitionState:
    """Solve for one helical-core equilibrium and climb from it with a dimer that starts
    along the helical part of its state."""
    minimum = solve(helical_core_input(helical_seed=0.3))
    x_min = np.asarray(minimum.get_state()).copy()
    energy_min = minimum.mhd_energy
    free = free_coefficients(create_model(helical_core_input(helical_seed=0.3)))
    odd = odd_n(minimum)

    landscape = Landscape(minimum, x_min, free)
    # towards smaller helical amplitude: along minus the odd-n part of x_min
    axisymmetric_part = landscape.coordinates(np.where(odd, 0.0, x_min))
    distance = np.linalg.norm(axisymmetric_part)
    y, _, curvature, iterations = dimer_search(
        landscape,
        0.1 * axisymmetric_part,
        axisymmetric_part,
        separation=1.0e-4 * distance,
        climb_rate=1.0e-3 * distance,
    )
    landscape.gradient(y)
    return TransitionState(
        landscape,
        y,
        curvature,
        iterations,
        landscape.model.mhd_energy,
        x_min,
        energy_min,
        odd,
    )


def main():
    saddle = climb_from_helical_minimum()
    x_saddle = saddle.landscape.state(saddle.y)
    axisymmetric = solve(helical_core_input())

    print(
        f"dimer iterations: {saddle.iterations}, force evaluations: "
        f"{saddle.landscape.force_evaluations}"
    )
    print(f"curvature along the dimer at the saddle: {saddle.curvature:.3e}")
    print(
        "directions of negative curvature: "
        f"{negative_curvatures(saddle.landscape, saddle.y)}"
    )
    print(
        "largest odd-n coefficient: helical minimum "
        f"{np.abs(saddle.x_min[saddle.odd]).max():.3e}, saddle "
        f"{np.abs(x_saddle[saddle.odd]).max():.1e}"
    )
    print(f"barrier W_saddle - W_min: {saddle.energy - saddle.energy_min:.6e}")
    print(
        "axisymmetric equilibrium W - W_min: "
        f"{axisymmetric.mhd_energy - saddle.energy_min:.6e}"
    )


if __name__ == "__main__":
    main()

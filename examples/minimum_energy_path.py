# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The minimum-energy path between two equilibria, found by the string method.

Equilibria are stationary points of VMEC's augmented functional, the MHD energy plus
the spectral-condensation constraint. The minimum-energy path between two minima is
the curve along which the force is everywhere tangent to the curve; its highest point
is the transition state, and its height above a minimum is the energy barrier between
the two equilibria. The string method (E, Ren and Vanden-Eijnden, J. Chem. Phys. 126,
164103 (2007)) finds it from a chain of states between the minima: every state moves
along the force, and after every step the chain is spread back to equal spacing along
its length, so that only the force across the chain acts. The highest state moves with
the force component along the chain reversed and climbs to the transition state
(climbing image; Henkelman, Uberuaga and Jonsson, J. Chem. Phys. 113, 9901 (2000)).

The configuration is a spherical tokamak of major radius 1.5, minor radius 1 and
elongation 1.6, with q = 1 on the axis and low shear out to s = 0.5. Its axisymmetric
equilibrium is unstable to the m = 1, n = 1 internal kink. With the axisymmetric
boundary and the stellarator symmetry of the input, the helical-core equilibrium the
kink leads to comes as a pair of mirror images, half a toroidal turn apart; in the
state, the second follows from the first by flipping the sign of every coefficient with
odd toroidal mode number n. The path between the two passes through the axisymmetric
equilibrium, which the solver also reaches directly from an axisymmetric initial guess,
because the n = 1 coefficients then start at zero and stay there.

The chain lives on the free coefficients of the decomposed state of a ``VmecModel``,
in coordinates y with x = x0 + L y, where L L^T is VMEC's preconditioner at the
midpoint of the two minima.
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


def equal_spacing(chain: np.ndarray) -> np.ndarray:
    """The chain moved along its piecewise-linear curve to equal spacing."""
    lengths = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    position = np.concatenate([[0.0], np.cumsum(lengths)]) / lengths.sum()
    spaced = np.empty_like(chain)
    for k, target in enumerate(np.linspace(0.0, 1.0, len(chain))):
        i = int(
            np.clip(
                np.searchsorted(position, target, side="right") - 1, 0, len(chain) - 2
            )
        )
        weight = (target - position[i]) / (position[i + 1] - position[i])
        spaced[k] = (1.0 - weight) * chain[i] + weight * chain[i + 1]
    return spaced


def string_method(
    landscape,
    y_start,
    y_end,
    *,
    images=9,
    step=1.0,
    momentum=0.95,
    climb_after=100,
    ftol=FTOL,
    max_iterations=10000,
):
    """The minimum-energy path from y_start to y_end, both minima, as a chain of
    ``images`` states, and the energy of each state.

    Interior states take heavy-ball steps along the force, with the velocity reset
    whenever it points against the force, and the chain is respaced after every step.
    After climb_after steps the highest interior state climbs instead: its force
    component along the chain is reversed, and the respacing leaves it in place. The
    search stops when the climbing state meets VMEC's convergence criterion and no
    state moves by more than 1e-9 of the length of the chain in a step.
    """
    chain = np.array(
        [y_start + (y_end - y_start) * k / (images - 1) for k in range(images)]
    )
    velocity = np.zeros_like(chain)
    energies = np.empty(images)
    for k in (0, images - 1):
        landscape.gradient(chain[k])
        energies[k] = landscape.model.mhd_energy
    for iteration in range(max_iterations):
        forces = np.zeros_like(chain)
        residuals = np.zeros(images)
        for k in range(1, images - 1):
            forces[k] = -landscape.gradient(chain[k])
            energies[k] = landscape.model.mhd_energy
            residuals[k] = landscape.residual()
        top = 1 + int(np.argmax(energies[1:-1]))
        climbing = iteration >= climb_after
        if climbing:
            tangent = chain[top + 1] - chain[top - 1]
            tangent /= np.linalg.norm(tangent)
            forces[top] -= 2.0 * (forces[top] @ tangent) * tangent
        velocity = momentum * velocity + step * forces
        for k in range(1, images - 1):
            if velocity[k] @ forces[k] < 0.0:
                velocity[k] = 0.0
        moved = chain + velocity
        if climbing:
            left = equal_spacing(moved[: top + 1])
            right = equal_spacing(moved[top:])
            moved = np.concatenate([left, right[1:]])
        else:
            moved = equal_spacing(moved)
        displacement = np.linalg.norm(moved - chain, axis=1).max()
        length = np.linalg.norm(np.diff(chain, axis=0), axis=1).sum()
        if climbing and residuals[top] < ftol and displacement < 1.0e-9 * length:
            return chain, energies, top, iteration
        chain = moved
    msg = f"the string method did not converge in {max_iterations} iterations"
    raise RuntimeError(msg)


@dataclass
class MinimumEnergyPath:
    """The converged chain between the two minima, and the energy of each state."""

    landscape: Landscape
    chain: np.ndarray
    energies: np.ndarray
    top: int
    iterations: int
    odd: np.ndarray


def path_between_mirror_images() -> MinimumEnergyPath:
    """Solve for one helical-core equilibrium, mirror it, and run the string method
    between the two."""
    minimum = solve(helical_core_input(helical_seed=0.3))
    odd = odd_n(minimum)
    x_start = np.asarray(minimum.get_state()).copy()
    x_end = np.where(odd, -x_start, x_start)
    free = free_coefficients(create_model(helical_core_input(helical_seed=0.3)))

    landscape = Landscape(minimum, np.where(odd, 0.0, x_start), free)
    chain, energies, top, iterations = string_method(
        landscape, landscape.coordinates(x_start), landscape.coordinates(x_end)
    )
    return MinimumEnergyPath(landscape, chain, energies, top, iterations, odd)


def main():
    path = path_between_mirror_images()
    energies = path.energies
    barrier = energies[path.top] - energies[0]
    axisymmetric = solve(helical_core_input())

    print(
        f"string iterations: {path.iterations}, force evaluations: "
        f"{path.landscape.force_evaluations}"
    )
    print(" image   (W - W_min) / barrier   largest odd-n coefficient")
    for k, y in enumerate(path.chain):
        x = path.landscape.state(y)
        print(
            f"{k:6d}   {(energies[k] - energies[0]) / barrier:22.4f}"
            f"   {np.abs(x[path.odd]).max():25.3e}"
        )
    print(f"barrier from the climbing image: {barrier:.6e}")
    print(
        "axisymmetric equilibrium W - W_min: "
        f"{axisymmetric.mhd_energy - energies[0]:.6e}"
    )


if __name__ == "__main__":
    main()

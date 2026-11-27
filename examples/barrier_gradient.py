# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The gradient of the energy barrier between two equilibria, from one adjoint solve per
equilibrium and four force evaluations per parameter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from vmecpp_adjoint import mhd_energy, objective_state_gradient

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore[import]

# the spherical tokamak with q = 1 on the axis, above the threshold of its kink
SPHERICAL_TOKAMAK = vmecpp.VmecInput.from_file(
    Path(__file__).parent / "data" / "spherical_tokamak.json"
)
NTOR = SPHERICAL_TOKAMAK.ntor_max
FTOL = 1.0e-18
BASE_PARAMETERS = {
    "pres_scale": SPHERICAL_TOKAMAK.pres_scale,
    "elongation": float(SPHERICAL_TOKAMAK.zbs[1, NTOR]),
}


def helical_core_input(
    pres_scale: float = BASE_PARAMETERS["pres_scale"],
    elongation: float = BASE_PARAMETERS["elongation"],
    helical_seed: float = 0.0,
) -> vmecpp.VmecInput:
    """The spherical tokamak, with the magnetic axis guess displaced helically by
    ``helical_seed``."""
    vmec_input = SPHERICAL_TOKAMAK.model_copy(deep=True)
    vmec_input.ftol_array = np.array([FTOL])
    vmec_input.pres_scale = pres_scale
    vmec_input.zbs[1, NTOR] = elongation
    vmec_input.raxis_c[1] = helical_seed
    vmec_input.zaxis_s[1] = -helical_seed
    return vmec_input


def create_model(vmec_input: vmecpp.VmecInput):
    ns = int(vmec_input.ns_array[-1])
    return _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), ns, None)


def solve(vmec_input: vmecpp.VmecInput):
    """The equilibrium the native solver converges to: its state and MHD energy."""
    model = create_model(vmec_input)
    model.solve()
    model.evaluate(2, 2, False)
    return np.asarray(model.get_state()).copy(), model.mhd_energy


def free_coefficients(model) -> np.ndarray:
    """The coefficients the solver evolves, from a state without symmetry."""
    model.evaluate(2, 2, True)
    return np.flatnonzero(np.asarray(model.get_forces()) != 0.0)


def energy_and_force(vmec_input: vmecpp.VmecInput, x: np.ndarray, free: np.ndarray):
    """The MHD energy and the raw force of ``vmec_input`` at the free coefficients of
    x."""
    model = create_model(vmec_input)
    state = np.asarray(model.get_state()).copy()
    state[free] = x[free]
    model.set_state(state)
    model.evaluate(2, 2, False)
    return model.mhd_energy, np.asarray(model.get_forces())[free]


def adjoint(parameters: dict, x: np.ndarray, free: np.ndarray, step: float = 1.0e-6):
    """The adjoint lambda of J^T lambda = dW/dx at the equilibrium x."""
    model = create_model(helical_core_input(**parameters))
    energy_gradient = objective_state_gradient(model, x, mhd_energy, step)[free]
    model.set_state(x)
    jacobian = np.empty((free.size, free.size))
    for column, i in enumerate(free):
        unit = np.zeros_like(x)
        unit[i] = 1.0
        jacobian[:, column] = np.asarray(model.hessian_vector_product(unit))[free]
    return np.linalg.solve(jacobian.T, energy_gradient)


def branch_derivative(name, step, parameters, x, free, lam):
    """The derivative dW/dq along the branch through the equilibrium x, and its envelope
    term."""
    plus = {**parameters, name: parameters[name] + step}
    minus = {**parameters, name: parameters[name] - step}
    energy_plus, force_plus = energy_and_force(helical_core_input(**plus), x, free)
    energy_minus, force_minus = energy_and_force(helical_core_input(**minus), x, free)
    envelope = (energy_plus - energy_minus) / (2.0 * step)
    return envelope - lam @ (force_plus - force_minus) / (2.0 * step), envelope


def barrier(parameters: dict) -> float:
    """The barrier between the helical-core equilibria, from two equilibrium solves."""
    _, energy_saddle = solve(helical_core_input(**parameters))
    _, energy_min = solve(helical_core_input(**parameters, helical_seed=0.3))
    return energy_saddle - energy_min


def barrier_gradients(parameters: dict, explicit_steps: dict) -> dict:
    """The derivative of the barrier with respect to each parameter named in
    explicit_steps, and its envelope term, as {name: (gradient, envelope term)}."""
    free = free_coefficients(create_model(helical_core_input(helical_seed=0.3)))
    x_saddle, _ = solve(helical_core_input(**parameters))
    x_min, _ = solve(helical_core_input(**parameters, helical_seed=0.3))
    lam_saddle = adjoint(parameters, x_saddle, free)
    lam_min = adjoint(parameters, x_min, free)
    gradients = {}
    for name, step in explicit_steps.items():
        gradient_saddle, envelope_saddle = branch_derivative(
            name, step, parameters, x_saddle, free, lam_saddle
        )
        gradient_min, envelope_min = branch_derivative(
            name, step, parameters, x_min, free, lam_min
        )
        gradients[name] = (
            gradient_saddle - gradient_min,
            envelope_saddle - envelope_min,
        )
    return gradients


def re_solved_barrier_gradient(parameters: dict, name: str, step: float) -> float:
    """The derivative of the barrier from a central difference of re-solved
    equilibria."""
    plus = {**parameters, name: parameters[name] + step}
    minus = {**parameters, name: parameters[name] - step}
    return (barrier(plus) - barrier(minus)) / (2.0 * step)


def main():
    parameters = BASE_PARAMETERS
    gradients = barrier_gradients(
        parameters, {"pres_scale": 1.0e2, "elongation": 1.0e-5}
    )
    print(f"barrier W_saddle - W_min: {barrier(parameters):.6e}")
    print(
        f"{'parameter':>10s} {'envelope term':>14s} {'gradient':>14s} "
        f"{'re-solved':>14s}"
    )
    for name, step in (("pres_scale", 1.0e2), ("elongation", 2.5e-4)):
        gradient, envelope = gradients[name]
        re_solved = re_solved_barrier_gradient(parameters, name, step)
        print(f"{name:>10s} {envelope:14.6e} {gradient:14.6e} {re_solved:14.6e}")


if __name__ == "__main__":
    main()

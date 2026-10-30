# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The gradient of the energy barrier between two equilibria.

The barrier between two equilibria is the energy of the transition state between
them minus the energy of the lower one. Its derivative with respect to a parameter q
of the input, a boundary coefficient or a profile parameter, is the difference of the
derivatives of the MHD energy W along the two equilibrium branches. Along a branch
x(q), defined by the force balance F(x(q), q) = 0, the implicit function theorem gives

    dW/dq = dW/dq|_x - lambda^T dF/dq|_x,   J^T lambda = dW/dx,   J = dF/dx.

The first term alone, the envelope theorem, is the derivative where W is stationary.
VMEC's equilibria are stationary points of its augmented functional, the MHD energy
plus the spectral-condensation constraint, and dW/dx does not vanish there; the
example prints the first term beside the derivative. One solve for lambda per
equilibrium serves every parameter; each parameter then costs four force evaluations.

The force is a row-scaled gradient, so J is not symmetric and the adjoint system
needs J^T. J and dW/dx are assembled densely on the free coefficients of the
decomposed state of a ``VmecModel``, from Hessian-vector products and central
differences of the energy.

The configuration is a spherical tokamak of major radius 1.5, minor radius 1 and
elongation 1.6, with q = 1 on the axis and low shear out to s = 0.5. Its axisymmetric
equilibrium is unstable to the m = 1, n = 1 internal kink, which leads to a
helical-core equilibrium. With the axisymmetric boundary and the stellarator symmetry
of the input, that equilibrium comes as a pair of mirror images, half a toroidal turn
apart, and the transition state between them is the axisymmetric equilibrium. The
solver reaches each directly: the helical one from a helically displaced axis guess,
the axisymmetric one from an axisymmetric guess, in which the n = 1 coefficients start
at zero and stay there.
"""

from __future__ import annotations

import numpy as np
from vmecpp_adjoint import mhd_energy, objective_state_gradient

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore[import]

NS = 25
MPOL = 6
NTOR = 1
FTOL = 1.0e-14
BASE_PARAMETERS = {"pres_scale": 1.2e5, "elongation": 1.6}


def helical_core_input(
    pres_scale: float = 1.2e5, elongation: float = 1.6, helical_seed: float = 0.0
) -> vmecpp.VmecInput:
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
        pres_scale=pres_scale,
        gamma=0.0,
        always_fix_m1_gauge=True,
        raxis_c=np.array([1.5, helical_seed]),
        zaxis_s=np.array([0.0, -helical_seed]),
    )
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[1.5], [1.0]]), MPOL, NTOR
    )
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[0.0], [elongation]]), MPOL, NTOR
    )
    return vmec_input


def create_model(vmec_input: vmecpp.VmecInput):
    return _vmecpp.VmecModel.create(vmec_input._to_cpp_vmecindata(), NS, None)


def solve(vmec_input: vmecpp.VmecInput):
    """The equilibrium the native solver converges to: its state and MHD energy."""
    model = create_model(vmec_input)
    model.solve()
    model.evaluate(2, 2, False)
    return np.asarray(model.get_state()).copy(), model.mhd_energy


def free_coefficients(model) -> np.ndarray:
    """Indices of the state coefficients the solver evolves: those with a nonzero
    preconditioned force at a state without symmetry."""
    model.evaluate(2, 2, True)
    return np.flatnonzero(np.asarray(model.get_forces()) != 0.0)


def energy_and_force(vmec_input: vmecpp.VmecInput, x: np.ndarray, free: np.ndarray):
    """The MHD energy and the raw force of ``vmec_input`` at the free coefficients of x,
    with the fixed coefficients, among them the boundary, taken from the input."""
    model = create_model(vmec_input)
    state = np.asarray(model.get_state()).copy()
    state[free] = x[free]
    model.set_state(state)
    model.evaluate(2, 2, False)
    return model.mhd_energy, np.asarray(model.get_forces())[free]


def adjoint(parameters: dict, x: np.ndarray, free: np.ndarray, step: float = 1.0e-6):
    """The adjoint lambda of J^T lambda = dW/dx at the equilibrium x.

    Both sides live on the free coefficients. The energy gradient dW/dx comes from
    central differences of the energy, and J column by column from Hessian-vector
    products, the directional derivatives of the force that VmecModel computes.
    """
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
    explicit_steps, and its envelope term, as {name: (gradient, envelope term)}.

    The steps are those of the central differences that give the explicit derivatives at
    fixed state.
    """
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
    for name, step in (("pres_scale", 1.0e3), ("elongation", 2.0e-3)):
        gradient, envelope = gradients[name]
        re_solved = re_solved_barrier_gradient(parameters, name, step)
        print(f"{name:>10s} {envelope:14.6e} {gradient:14.6e} {re_solved:14.6e}")


if __name__ == "__main__":
    main()

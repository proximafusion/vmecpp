# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A real SIMSOPT gradient-based optimization loop driven by VMEC++.

Builds the VmecEnergy Optimizable (MHD energy of the converged equilibrium, with
the analytic adjoint boundary gradient from the exact autodiff Hessian-vector
product), wraps it in a SIMSOPT least-squares problem, and runs
least_squares_serial_solve to drive the plasma boundary toward a target energy.
Demonstrates that VMEC++ is usable as a gradient-providing component in a real
SIMSOPT optimization loop. Requires SIMSOPT and an Enzyme-enabled VMEC++ build.

Run: python examples/simsopt_optimization_loop.py
"""

import time

from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve
from simsopt_vmec_gradient import VmecBoundaryProblem, make_simsopt_optimizable

problem = VmecBoundaryProblem(ns=11)
energy = make_simsopt_optimizable(problem)

j0 = energy.J()
target = 1.03 * j0
print(f"initial MHD energy = {j0:.6e}, target = {target:.6e}")

least_squares = LeastSquaresProblem.from_tuples([(energy.J, target, 1.0)])

problem.model.reset_force_eval_count()
t0 = time.perf_counter()
least_squares_serial_solve(least_squares, grad=True)  # analytic adjoint gradient
seconds = time.perf_counter() - t0

j_final = energy.J()
print(f"final MHD energy   = {j_final:.6e}")
print(f"|J - target| / target = {abs(j_final - target) / target:.2e}")
print(f"VMEC++ force evals = {problem.model.force_eval_count}, time = {seconds:.2f}s")

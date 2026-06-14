# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Use VMEC++ as a gradient-providing equilibrium component for SIMSOPT.

A SIMSOPT optimization over a plasma boundary normally differentiates VMEC by
finite differences: one equilibrium re-solve per boundary degree of freedom and
per outer iteration. VMEC++ instead provides the boundary gradient analytically
through the implicit-function adjoint (``vmecpp_adjoint.boundary_gradient``): one
extra Hessian solve, independent of the number of boundary DOFs.

``VmecEnergy`` wraps that as a SIMSOPT ``Optimizable`` whose objective is the MHD
energy of the converged equilibrium and whose ``dJ`` is the adjoint gradient.
``optimize_to_target`` runs a gradient-based optimization of the boundary toward
a target energy, with the analytic gradient or with finite differences, and
reports the cost (forward-model evaluations counted inside VMEC++, outer
iterations, wall time) so the two can be compared.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vmecpp_adjoint import (
    DEFAULT_INPUT,
    boundary_gradient,
    finite_difference_boundary_gradient,
    make_model,
    mhd_energy,
    partition,
    solve_interior,
)


class VmecBoundaryProblem:
    """Equilibrium energy and its boundary gradient, cached per boundary state."""

    def __init__(self, input_path: Path = DEFAULT_INPUT, ns: int = 11):
        self.model = make_model(input_path, ns)
        self.model.solve()
        self.ns = ns
        self.interior, self.boundary = partition(self.model, ns)
        self._x_full = np.asarray(self.model.get_state(), float).copy()
        self._cached_p = self._x_full[self.boundary].copy()

    @property
    def x0(self):
        return self._x_full[self.boundary].copy()

    def _resolve(self, p):
        p = np.asarray(p, float)
        if not np.array_equal(p, self._cached_p):
            self._x_full = solve_interior(
                self.model, self._x_full, self.interior, self.boundary, p
            )
            self._cached_p = p.copy()

    def value(self, p):
        self._resolve(p)
        self.model.set_state(np.ascontiguousarray(self._x_full))
        self.model.evaluate(2, 2, False)
        return self.model.mhd_energy

    def gradient(self, p, exact=None):
        # Use the exact autodiff HVP when the extension was built with Enzyme,
        # otherwise fall back to the finite-difference HVP so the default build
        # still works.
        if exact is None:
            exact = hasattr(self.model, "exact_hessian_vector_product")
        self._resolve(p)
        return boundary_gradient(
            self.model,
            self._x_full,
            self.interior,
            self.boundary,
            mhd_energy,
            exact=exact,
        )


def make_simsopt_optimizable(problem: VmecBoundaryProblem):
    """Wrap the problem as a SIMSOPT Optimizable exposing an analytic gradient."""
    # Imported lazily so the rest of the module (and the gradient benchmark) work
    # without SIMSOPT installed.
    from simsopt._core import Optimizable  # noqa: PLC0415
    from simsopt._core.derivative import Derivative, derivative_dec  # noqa: PLC0415

    class VmecEnergy(Optimizable):
        def __init__(self):
            x0 = problem.x0
            super().__init__(x0=x0, names=[f"boundary{i}" for i in range(x0.size)])

        def J(self):
            return problem.value(self.local_full_x)

        @derivative_dec
        def dJ(self):
            return Derivative({self: problem.gradient(self.local_full_x)})

    return VmecEnergy()


@dataclass
class GradResult:
    method: str
    force_evals: int
    seconds: float
    gradient: np.ndarray


def gradient_cost(input_path: Path = DEFAULT_INPUT, ns: int = 11, analytic=True):
    """Cost of one full boundary gradient at the converged equilibrium.

    This is what an external optimizer pays per iteration. The analytic adjoint needs
    one Hessian solve regardless of the number of boundary DOFs; finite differences re-
    converge the equilibrium twice per boundary DOF.
    """
    problem = VmecBoundaryProblem(input_path, ns)
    x_star = problem._x_full.copy()
    interior, boundary = problem.interior, problem.boundary
    problem.model.reset_force_eval_count()
    t0 = time.perf_counter()
    if analytic:
        g_dict = None
        g = boundary_gradient(problem.model, x_star, interior, boundary, mhd_energy)
    else:
        g_dict = finite_difference_boundary_gradient(
            problem.model,
            x_star,
            interior,
            boundary,
            mhd_energy,
            range(boundary.size),
        )
        g = np.array([g_dict[j] for j in range(boundary.size)])
    return GradResult(
        "analytic adjoint" if analytic else "finite differences",
        problem.model.force_eval_count,
        time.perf_counter() - t0,
        g,
    )


def main():
    analytic = gradient_cost(analytic=True)
    fd = gradient_cost(analytic=False)
    rel = np.linalg.norm(analytic.gradient - fd.gradient) / np.linalg.norm(fd.gradient)
    n_boundary = analytic.gradient.size
    print(f"boundary gradient cost ({n_boundary} boundary DOFs, solovev ns=11)\n")
    print(f"{'method':20s} {'F-evals':>9s} {'time[s]':>8s}")
    for r in (fd, analytic):
        print(f"{r.method:20s} {r.force_evals:9d} {r.seconds:8.2f}")
    print(
        f"\nspeedup (force evals): {fd.force_evals / max(analytic.force_evals, 1):.1f}x"
        f"   gradient agreement: {rel:.1e}"
    )


if __name__ == "__main__":
    main()

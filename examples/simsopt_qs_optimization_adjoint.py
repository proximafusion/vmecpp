# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Gradient-based quasi-helical optimization with the VMEC++ adjoint and SIMSOPT.

Same problem as ``simsopt_qh_fixed_resolution.py``: reduce the quasisymmetry error
of the nfp=4 QH warm start at fixed aspect ratio by varying the boundary Fourier
modes. There SIMSOPT differentiates by finite differences, which costs one
equilibrium solve per boundary mode and iteration. Here the gradient comes from
``vmecpp.autodiff``: JAX differentiates the objective, and VMEC++ supplies the
implicit VJP of the equilibrium solve, so one gradient costs about one forward
solve plus one adjoint solve regardless of the number of modes.

SIMSOPT owns the degrees of freedom (``Optimizable``); SciPy's L-BFGS-B consumes
the objective and its analytic gradient.
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, OptimizableDefaultDict, derivative_dec

import vmecpp
from vmecpp import autodiff, autodiff_wout, qs

jax.config.update("jax_enable_x64", True)

BOX_HALF_WIDTH = 0.01
FAILED_SOLVE_OBJECTIVE = 1e6
DEFAULT_INPUT = Path(__file__).parent / "data" / "input.nfp4_QH_warm_start"


class QuasisymmetryObjective(Optimizable):
    """Quasisymmetry error plus an aspect-ratio penalty, over the boundary modes."""

    def __init__(
        self,
        vmec_input: vmecpp.VmecInput,
        max_mode: int = 1,
        surfaces: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9),
        helicity_m: int = 1,
        helicity_n: int = -1,
        aspect_target: float = 7.0,
    ):
        self.solver = autodiff.make_solver(vmec_input)
        self.vmec_input = vmec_input
        self.surfaces = surfaces
        self.helicity = (helicity_m, helicity_n)
        self.aspect_target = aspect_target
        self.boundary0 = jnp.stack(
            [jnp.asarray(vmec_input.rbc), jnp.asarray(vmec_input.zbs)]
        )

        # Rows are rbc and zbs, columns are n = -ntor..ntor. The m=0 modes with
        # n<=0 are redundant (cos(-n phi) = cos(n phi)), and rbc(0,0) fixes the
        # major radius.
        _, mpol, ncols = self.boundary0.shape
        ntor = ncols // 2
        m, n = np.meshgrid(np.arange(mpol), np.arange(-ntor, ntor + 1), indexing="ij")
        free = (m <= max_mode) & (abs(n) <= max_mode) & ((m > 0) | (n > 0))
        self.free = jnp.asarray(np.stack([free, free]))
        names = [
            f"{name}({mm},{nn})"
            for name in ("rbc", "zbs")
            for mm, nn in zip(m[free], n[free], strict=True)
        ]
        super().__init__(x0=np.array(self.boundary0[self.free]), names=names)

        self._value_and_grad = jax.jit(jax.value_and_grad(self._objective))
        self._cache: tuple[np.ndarray, float, np.ndarray] | None = None

    def _boundary(self, x):
        return self.boundary0.at[self.free].set(x)

    def _objective(self, x):
        geometry = self.solver(self._boundary(x))
        quasisymmetry = qs.quasisymmetry_total(
            geometry,
            self.surfaces,
            helicity_m=self.helicity[0],
            helicity_n=self.helicity[1],
        )
        aspect = autodiff_wout.wout_quantities(geometry, self.vmec_input)["aspect"]
        return quasisymmetry + (aspect - self.aspect_target) ** 2

    def _evaluate(self) -> tuple[float, np.ndarray]:
        x = self.local_full_x
        if self._cache is None or not np.array_equal(self._cache[0], x):
            value, gradient = self._value_and_grad(jnp.asarray(x))
            self._cache = (x.copy(), float(value), np.asarray(gradient))
        return self._cache[1], self._cache[2]

    def J(self) -> float:
        return self._evaluate()[0]

    @derivative_dec
    def dJ(self) -> Derivative:
        return Derivative(OptimizableDefaultDict({self: self._evaluate()[1]}))


def _value_and_gradient(x: np.ndarray, objective: QuasisymmetryObjective):
    objective.x = x
    try:
        return objective.J(), objective.dJ()
    except RuntimeError:
        # The line search may step to a boundary VMEC++ cannot converge on; a
        # large value makes it backtrack.
        return FAILED_SOLVE_OBJECTIVE, np.zeros_like(x)


def main() -> None:
    with vmecpp.ensure_vmecpp_input(DEFAULT_INPUT) as json_path:
        vmec_input = vmecpp.VmecInput.from_file(json_path)
    objective = QuasisymmetryObjective(vmec_input)
    print(f"{objective.dof_size} boundary degrees of freedom")
    initial = objective.J()

    start = time.perf_counter()
    result = minimize(
        _value_and_gradient,
        objective.x,
        args=(objective,),
        jac=True,
        method="L-BFGS-B",
        # Outside a box around the warm start VMEC++ may not converge.
        bounds=[(x - BOX_HALF_WIDTH, x + BOX_HALF_WIDTH) for x in objective.x],
        options={"maxiter": 15},
    )
    print(
        f"objective {initial:.4e} -> {result.fun:.4e} after {result.nfev} "
        f"gradient evaluations, {time.perf_counter() - start:.0f} s"
    )


if __name__ == "__main__":
    main()

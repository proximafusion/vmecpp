# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Couple boundary-shape optimization to the Python force-balance iteration.

``vmecpp._optimization.optimize_boundary`` runs the ported Python iteration
(``vmecpp._iteration``) as the equilibrium solver inside a derivative-free
optimization over the plasma boundary: each candidate boundary is solved to force
balance with the Python loop, then a scalar objective is evaluated on the
converged equilibrium. This is the nested structure stellarator-optimization
codes (SIMSOPT, STELLOPT) are built on; owning the iteration in Python is what
makes it expressible without touching the C++ core.

Here we vary one boundary shaping coefficient of a fixed-boundary stellarator to
drive the stored MHD energy to a target value, solving the equilibrium with the
common-ground ``"robust"`` style at each optimization step.
"""

from pathlib import Path

import vmecpp
from vmecpp._optimization import optimize_boundary

vmec_input = vmecpp.VmecInput.from_file(
    Path("examples") / "data" / "cth_like_fixed_bdy.json"
)
# Single radial resolution keeps each equilibrium solve quick for the demo.
NS = 25

# Optimization variable: an additive change to one boundary shaping harmonic
# (poloidal m=1, toroidal n=0). rbc/zbs are indexed [m, ntor + n].
M, N = 1, 0
COL = vmec_input.ntor + N
base_coeff = float(vmec_input.rbc[M, COL])


def set_params(base, x):
    """Map the optimization vector to a configuration with a modified boundary."""
    modified = base.model_copy(deep=True)
    modified.rbc[M, COL] = base_coeff + float(x[0])
    return modified


def objective(model, _result):
    """Stored MHD energy of the converged equilibrium."""
    return model.mhd_energy


# Pick a target: the stored energy of a known, slightly perturbed boundary.
x_true = 0.03
target_model, _ = vmecpp.iterate(set_params(vmec_input, [x_true]), ns=NS)
energy_target = target_model.mhd_energy


def energy_miss(model, result):
    return (objective(model, result) - energy_target) ** 2


result = optimize_boundary(
    vmec_input,
    set_params,
    energy_miss,
    x0=[0.0],
    ns=NS,
    iteration_style="robust",
    method="Nelder-Mead",
    options={"xatol": 1.0e-3, "fatol": 1.0e-18},
)

print(f"target energy   : {energy_target:.6e}")
print(f"boundary delta  : x_opt={result.x[0]:+.4f}  (target x={x_true:+.4f})")
print(f"objective       : {result.objective:.3e}")
print(f"equilibrium solves: {result.num_evaluations}")
print(f"converged       : {result.converged}")

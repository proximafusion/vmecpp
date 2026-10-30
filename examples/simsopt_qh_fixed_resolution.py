"""
This example reproduces https://github.com/hiddenSymmetries/simsopt/blob/master/examples/2_Intermediate/QH_fixed_resolution.py
using VMEC++ instead of VMEC2000.

It shows how to optimize a VMEC equilibrium for quasi-helical symmetry (M=1, N=1) throughout the volume.
"""

from pathlib import Path

import numpy as np
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

from vmecpp.simsopt_compat import Vmec

# This problem has 24 degrees of freedom, so we can use 24 + 1 = 25
# concurrent function evaluations for 1-sided finite difference
# gradients.
proc0_print("Running 2_Intermediate/QH_fixed_resolution.py")
proc0_print("=============================================")

mpi = MpiPartition(25)

filename = Path(__file__).parent / "data" / "input.nfp4_QH_warm_start"
vmec = Vmec(str(filename))

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
max_mode = 2
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

proc0_print("Parameter space:", surf.dof_names)

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(
    vmec,
    np.arange(0, 1.01, 0.1),  # Radii to target
    helicity_m=1,
    helicity_n=-1,
)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7, 1), (qs.residuals, 0, 1)])

# Make sure all procs participate in computing the objective:
prob.objective()

proc0_print("Quasisymmetry objective before optimization:", qs.total())
proc0_print("Total objective before optimization:", prob.objective())

# To keep this example fast, we stop after the first function
# evaluation. For a "real" optimization, remove the max_nfev
# parameter.
least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8, max_nfev=1)

# Make sure all procs participate in computing the objective:
prob.objective()

proc0_print("Final aspect ratio:", vmec.aspect())
proc0_print("Quasisymmetry objective after optimization:", qs.total())
proc0_print("Total objective after optimization:", prob.objective())

proc0_print("End of 2_Intermediate/QH_fixed_resolution.py")
proc0_print("============================================")

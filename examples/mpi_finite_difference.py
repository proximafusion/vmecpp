# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Example of how to utilize the hot restart functionality, to evaluate a finite
difference estimate of the local Jacobian efficiently.

We parallelize the evaluation using MPI and distribute the workload across a number of
ranks. In this example, we compute the derivatives of the volume in respect to all
Fourier components of the geometry.
"""

from pathlib import Path

import mpi4py
import numpy as np

import vmecpp

# Notice that the OpenMP parallelism in VMEC++ allows us to use a simple MPI
# communicator, without the need to split into sub-groups.
comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

# In a real application, only root would probably read the input file and broadcast it.
filename = Path(__file__).parent / "data" / "cth_like_fixed_bdy.json"
initial_input = vmecpp.VmecInput.from_file(filename)

# All Fourier components of the geometry R, Z are degrees of freedom
n_dofs = np.prod(initial_input.rbc.shape) + np.prod(initial_input.zbs.shape)
m_outputs = 1  # Only interested in the volume
# One process per DOF. Root can also do a finite difference evaluation
assert n_dofs % comm.Get_size() == 0, f"Number of degrees of freedom: {n_dofs}"
n_dofs_per_proc = n_dofs // comm.Get_size()

# Base evaluation
initial_output = None
if rank == 0:
    initial_output = vmecpp.run(initial_input)
initial_output = comm.bcast(initial_output)

# ...and fix up the multigrid steps: hot-restarted runs only allow a single step
initial_input.ns_array = initial_input.ns_array[-1:]
initial_input.ftol_array = initial_input.ftol_array[-1:]
initial_input.niter_array = initial_input.niter_array[-1:]

eps = 1e-8
my_jacobian = np.zeros((n_dofs_per_proc, m_outputs))
# Start the finite difference evaluation
for i in range(n_dofs_per_proc):
    perturbed_input = initial_input.model_copy(deep=True)
    dof_idx = i + rank * n_dofs_per_proc
    if dof_idx < n_dofs // 2:
        perturbed_input.rbc.flat[dof_idx] += eps
    else:
        perturbed_input.zbs.flat[dof_idx - n_dofs // 2] += eps

    # We can now run a finite difference evaluation with hot restart:
    hot_restarted_output = vmecpp.run(
        perturbed_input, restart_from=initial_output, max_threads=1
    )
    dVdx = (hot_restarted_output.wout.volume - initial_output.wout.volume) / eps
    print(f"{dof_idx:3d} dVdx: {dVdx}")
    my_jacobian[i, :] = dVdx

# Gather Jacobians on root process
jacobian = comm.gather(my_jacobian, root=0)
if rank == 0:
    jacobian = np.vstack(jacobian)
    print("Final Jacobian matrix:\n", jacobian)

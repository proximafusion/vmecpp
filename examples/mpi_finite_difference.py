from pathlib import Path

import mpi4py
import numpy as np

import vmecpp

comm = mpi4py.MPI.COMM_WORLD
rank = comm.Get_rank()

# In a real application, only root would probably read the input file and broadcast it.
filename = Path(__file__).parent / "data" / "w7x.json"
input = vmecpp.VmecInput.from_file(filename)

n_dofs = np.prod(input.rbc.shape) + np.prod(input.zbs.shape)
m_outputs = 1  # Only interested in the volume
# One process per DOF. Root can also do a finite difference evaluation
assert n_dofs % comm.Get_size() == 0, f"Number of degrees of freedom: {n_dofs}"
n_dofs_per_proc = n_dofs // comm.Get_size()

# Base evaluation
output = None
if rank == 0:
    output = vmecpp.run(input)
output = comm.bcast(output)

# ...and fix up the multigrid steps: hot-restarted runs only allow a single step
input.ns_array = input.ns_array[-1:]
input.ftol_array = input.ftol_array[-1:]
input.niter_array = input.niter_array[-1:]

eps = 1e-8
my_jacobian = np.zeros((n_dofs_per_proc, m_outputs))
# Start the finite difference evaluation
for i in range(n_dofs_per_proc):
    dof_idx = i + rank * n_dofs_per_proc
    if dof_idx < n_dofs // 2:
        input.rbc.flat[dof_idx] += eps
    else:
        input.zbs.flat[dof_idx - n_dofs // 2] += eps

    # We can now run a finite difference evaluation with hot restart:
    hot_restarted_output = vmecpp.run(input, restart_from=output, verbose=False)
    dVdx = (hot_restarted_output.wout.volume - output.wout.volume) / eps
    print(f"{dof_idx:3d} dVdx: {dVdx}")
    my_jacobian[i, :] = dVdx

# Gather Jacobians on root process
jacobian = comm.gather(my_jacobian, root=0)
if rank == 0:
    jacobian = np.vstack(jacobian)
    print("Final Jacobian matrix:\n", jacobian)

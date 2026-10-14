# Examples

The `examples/` directory contains short scripts demonstrating typical tasks:
- [`python_api.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/python_api.py): run VMEC++ from Python using a provided input file.
- [`python_api_input.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/python_api_input.py): build a `VmecInput` object directly in Python.
- [`normal_run_and_hot_restart.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/normal_run_and_hot_restart.py): perform a standard run and then hot restart.
- [`freeboundary_run_and_hot_restart.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/freeboundary_run_and_hot_restart.py): demonstrate free-boundary runs with hot restart.
- [`hot_restart_scaling.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/hot_restart_scaling.py): illustrate how boundary perturbations affect hot-restart iteration counts.
- [`compare_vmecpp_to_parvmec.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/compare_vmecpp_to_parvmec.py): compare VMEC++ results to a PARVMEC reference solution.
- [`plot_plasma_boundary.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/plot_plasma_boundary.py): plot the outer plasma boundary with `matplotlib`.
- [`visualize_magnetic_field.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/visualize_magnetic_field.py): calculate the magnetic field and current density, and visualize with `pyvista`.
- [`mpi_finite_difference.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/mpi_finite_difference.py): compute finite-difference derivatives in parallel using MPI.
- [`simsopt_integration.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/simsopt_integration.py): minimal integration with SIMSOPT.
- [`simsopt_qh_fixed_resolution.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/simsopt_qh_fixed_resolution.py): reproduce SIMSOPT's QH fixed resolution example.
- [`sample_hot_restarts_with_random_perturbations.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/sample_hot_restarts_with_random_perturbations.py): explore restart sensitivity to small boundary perturbations.
- [`force_residual_convergence.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/force_residual_convergence.py): plot convergence of force residuals during VMEC++ runs.
- [`fourier_resolution_increase.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/fourier_resolution_increase.py): reach a high Fourier resolution by continuation, in fewer iterations than a fixed-resolution solve.
- [`watch_solve.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/watch_solve.py): draw the flux surfaces and the force residuals while VMEC++ converges, through the iteration callback of `vmecpp.run`.
- [`self_consistent_bootstrap_current.py`](https://github.com/proximafusion/vmecpp/blob/main/examples/self_consistent_bootstrap_current.py): make the enclosed current equal to SIMSOPT's Redl bootstrap current during the solve, by changing `state.curr_h` in the iteration callback of `vmecpp.run`.

# Examples

The `examples/` directory contains short scripts demonstrating typical tasks:
- [`python_api.py`](../examples/python_api.py): run VMEC++ from Python using a provided input file.
- [`python_api_input.py`](../examples/python_api_input.py): build a `VmecInput` object directly in Python.
- [`normal_run_and_hot_restart.py`](../examples/normal_run_and_hot_restart.py): perform a standard run and then hot restart.
- [`freeboundary_run_and_hot_restart.py`](../examples/freeboundary_run_and_hot_restart.py): demonstrate free-boundary runs with hot restart.
- [`hot_restart_scaling.py`](../examples/hot_restart_scaling.py): illustrate how boundary perturbations affect hot-restart iteration counts.
- [`compare_vmecpp_to_parvmec.py`](../examples/compare_vmecpp_to_parvmec.py): compare VMEC++ results to a PARVMEC reference solution.
- [`plot_plasma_boundary.py`](../examples/plot_plasma_boundary.py): plot the outer plasma boundary with `matplotlib`.
- [`mpi_finite_difference.py`](../examples/mpi_finite_difference.py): compute finite-difference derivatives in parallel using MPI.
- [`simsopt_integration.py`](../examples/simsopt_integration.py): minimal integration with SIMSOPT.
- [`simsopt_qh_fixed_resolution.py`](../examples/simsopt_qh_fixed_resolution.py): reproduce SIMSOPT's QH fixed resolution example.
- [`sample_hot_restarts_with_random_perturbations.py`](../examples/sample_hot_restarts_with_random_perturbations.py): explore restart sensitivity to small boundary perturbations.
- [`force_residual_convergence.py`](../examples/force_residual_convergence.py): plot convergence of force residuals during VMEC++ runs.

# Examples

The `examples/` directory contains short scripts demonstrating typical tasks:
- [`python_api.py`](../examples/python_api.py): run VMEC++ from Python using a provided input file.
- [`python_api_input.py`](../examples/python_api_input.py): build a `VmecInput` object directly in Python.
- [`normal_run_and_hot_restart.py`](../examples/normal_run_and_hot_restart.py): perform a standard run and then hot restart.
- [`freeboundary_run_and_hot_restart.py`](../examples/freeboundary_run_and_hot_restart.py): demonstrate free-boundary runs with hot restart.
- [`hot_restart_scaling.py`](../examples/hot_restart_scaling.py): illustrate how boundary perturbations affect hot-restart iteration counts.
- [`compare_vmecpp_to_parvmec.py`](../examples/compare_vmecpp_to_parvmec.py): compare VMEC++ results to a PARVMEC reference solution.
- [`plot_plasma_boundary.py`](../examples/plot_plasma_boundary.py): plot the outer plasma boundary with `matplotlib`.
- [`visualize_magnetic_field.py`](../examples/visualize_magnetic_field.py): calculate the magnetic field and current density, and visualize with `pyvista`.
- [`mpi_finite_difference.py`](../examples/mpi_finite_difference.py): compute finite-difference derivatives in parallel using MPI.
- [`simsopt_integration.py`](../examples/simsopt_integration.py): minimal integration with SIMSOPT.
- [`simsopt_qh_fixed_resolution.py`](../examples/simsopt_qh_fixed_resolution.py): reproduce SIMSOPT's QH fixed resolution example.
- [`sample_hot_restarts_with_random_perturbations.py`](../examples/sample_hot_restarts_with_random_perturbations.py): explore restart sensitivity to small boundary perturbations.
- [`force_residual_convergence.py`](../examples/force_residual_convergence.py): plot convergence of force residuals during VMEC++ runs.
- [`fourier_resolution_increase.py`](../examples/fourier_resolution_increase.py): reach a high Fourier resolution by continuation, in fewer iterations than a fixed-resolution solve.
- [`watch_solve.py`](../examples/watch_solve.py): draw the flux surfaces and the force residuals while VMEC++ converges, through the iteration callback of `vmecpp.run`.
- [`self_consistent_bootstrap_current.py`](../examples/self_consistent_bootstrap_current.py): make the enclosed current equal to SIMSOPT's Redl bootstrap current during the solve, by changing `state.curr_h` in the iteration callback of `vmecpp.run`.
- [`w7x_lbsubs_current_density.py`](../examples/w7x_lbsubs_current_density.py): compare the toroidal current density of W7-X with `lbsubs` off and on.
- [`force_balance_spectrum.py`](../examples/force_balance_spectrum.py): visualize force-residual Fourier coefficients and their real-space contributions.
- [`island_interior.py`](../examples/island_interior.py): solve the interior of an island of the W7-X 5/5, 5/4 or 5/6 chain as an equilibrium whose magnetic axis is the island's O-point line, the last two over the 4 or 6 toroidal transits after which their single island closes, compare it with field-line tracing, and move the island by the field of a finite-beta core plasma.
- [`knotted_stellarator.py`](../examples/knotted_stellarator.py): solve stellarators whose magnetic axis is a torus knot over the toroidal transits after which the knot closes, the trefoil knotatron of Hudson, Startsev and Feibush, a trefoil of turning elliptical coils, the trefoil as the (3, 2) torus knot over three transits and the cinquefoil, compare them with field-line tracing, and add pressure.

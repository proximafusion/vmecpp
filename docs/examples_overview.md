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

## Resolution continuation

Hard equilibria converge more reliably when they are approached through a
sequence of increasing resolutions: the classic radial multi-grid (`ns_array`),
and Fourier continuation in the poloidal and toroidal resolution (`mpol` /
`ntor`). VMEC++ provides two functions for this, both operating on a converged
`VmecOutput`:

- `vmecpp.interpolate_solution(source, target_input)` interpolates a converged
  solution onto the resolution of `target_input` -- radial interpolation in
  `sqrt(s)` (with the odd-`m` axis behaviour the solver uses internally) and
  Fourier zero-padding or truncation -- producing a `restart_from` guess for the
  next step.
- `vmecpp.run_continuation(input, ns_array=..., mpol_array=..., ntor_array=...)`
  runs a whole schedule in one call, solving each resolution and hot-restarting
  from the previous step. With only `ns_array` it performs the radial
  continuation; `mpol_array` and `ntor_array` add Fourier continuation.

```python
output = vmecpp.run_continuation(
    vmec_input,
    ns_array=[15, 31, 31],
    mpol_array=[5, 9, 13],
    ntor_array=[4, 4, 4],
)
```

[`fourier_resolution_increase.py`](../examples/fourier_resolution_increase.py)
ramps `mpol` from 5 to 13 on the `cth_like_fixed_bdy` case. It reaches the same
equilibrium as a direct `mpol=13` solve (geometry agreeing to about 3e-6) in
roughly 78% of the iterations, and the advantage grows with the target
resolution.

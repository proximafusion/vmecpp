# WP1: force-definition handoff (partial)

## Force differentiated by H

`get_forces()` returns the latest decomposed force buffer contents, not an
unconditionally raw residual accessor. `evaluate(2, 2, True)` applies force
preconditioning. `evaluate(2, 2, False, True)` stops at the invariant-residual
checkpoint, before force preconditioning, and supplies the raw decomposed force
in the exact HVP's output basis. No new C++ accessor is needed.

The equal iteration counters refresh the radial preconditioner and `tcon`.
The exact force composition also recomputes `tcon` from the geometry. This is
not the lagged-`tcon` residual of an arbitrary native solver iteration.

Evidence:

- `src/vmecpp/cpp/vmecpp/vmec/pybind11/pybind_vmec.cc`:
  `VmecModel::Evaluate`, `GetForces`, `ExactHessianVectorProduct`, and
  `ExactHessianVectorProductTranspose`.
- `src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc`:
  `IdealMhdModel::update`, its `INVARIANT_RESIDUALS` checkpoint, and
  `shouldUpdateRadialPreconditioner`.
- Existing regression:
  `tests/test_autodiff.py::test_exact_hessian_vector_product_is_the_derivative_of_the_force`.

Both exact products use/overwrite force scratch buffers. WP5 and WP7 must copy
F before assembling H or applying HVPs. `autodiff_diagnostics.raw_force` performs
the refresh and copy, requires the pinned gauge, and rejects a bad Jacobian.
It does not replace a missing exact Jacobian with finite differences.

## Verification status

- Three raw-force API/buffer tests pass.
- The small CTH case has raw full-state force norm `2.322798823e-8`, versus
  preconditioned norm `5.355550172e-8`. The relative change from preconditioning
  is `2.617580166`; the buffers cannot be interchanged.
- Across nine FD steps from `1e-3` to `1e-7`, minimum relative H/FD errors are
  `1.175329755e-9` (full direction), `8.315951489e-10` (interior direction),
  and `7.944398721e-10` (boundary direction). The error decreases quadratically
  before reaching roundoff. These measurements use the seeded shared small
  fixture and the pinned gauge.
- Reproduction: run `scripts/plot_force_definition.py --output <directory>`.
  The PNG and NPZ are in `results/plot_00_force_definition_small.*`.
- Full record population, JAX exposure, warning thresholds, objective-class-4
  warnings, and the less-than-5-percent overhead acceptance are not implemented.
- WP4's generator overlap remains unavailable. No projection recommendation
  is inferred from this force-definition work.

## Contract clarification

`src/vmecpp/_adjoint_contracts.py` now specifies the evaluation sequence and
force-buffer lifetime. Signatures are unchanged. The assumption that the
current `get_forces()` buffer after a full evaluation is the residual
differentiated by H is false.

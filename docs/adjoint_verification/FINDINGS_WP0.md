# WP0: shared harness and contracts

## Deliverables

- The seven existing JSON inputs, an explicit prescribed-current CTH variant,
  and the CTH small case (`ns=7`, `mpol=4`, `ntor=2`) are shared through
  `tests/adjoint_verification/conftest.py`.
- The resolution factory covers all nine combinations of
  `mpol={4,8,12}` and `ns={11,51,101}`.
- All inputs are independent copies, stellarator symmetric, fixed boundary,
  and use `always_fix_m1_gauge=True`. Fourier resizing uses `VmecInput.resize`.
- Reusable objectives cover volume, aspect, iota at fixed normalized flux,
  Mercier `DWell` at fixed flux, QS residual squared, boundary R/Z,
  interior lambda_sc squared, interior m=2 R squared, and seeded linear
  functionals of every Geometry leaf.
- `dense_jacobian` uses full-state exact unit HVPs, without symmetrization or
  deflation. Default limits are 2048 state entries and 64 MiB. Cache keys include
  the input, state, extension content, harness content, and evaluation settings.
  Cache hits require a checksum and two fresh full-state HVP probes.
- `src/vmecpp/_adjoint_contracts.py` defines the records and four explicit
  `NotImplementedError` stubs.
- `scripts/plot_style.py` defines shared styling, mode colors, filename rules,
  and named PNG/NPZ pairs without object arrays.

Standalone scripts import `adjoint_verification.harness` after adding the
repository's `tests` directory to the import path. The module does not depend
on pytest.

## Acceptance and verification

- Initial non-Enzyme harness validation: 55 passed, 3 Enzyme tests skipped.
- Five contract tests and five plot-convention tests pass.
- Final owned harness validation: 75 passed, 0 skipped on the Enzyme build.
  This includes nine fixture solve/exact-HVP smoke tests, prescribed-current
  and prescribed-iota VJPs, full-state dense assembly, and cache checks.
- The initial extension reported `VMECPP_ENABLE_ENZYME=False`. Validation used
  Enzyme v0.0.264 built against Clang/LLVM 18 and an extension configured with
  `VMECPP_ENABLE_ENZYME=ON`. The runtime flag and exact-Jacobian capability both
  report true.
- This package establishes fixtures, not N1-N10 or a physics-gradient reference.
  Full resolution-sweep solves and invariant-objective relabeling checks belong
  to the subsequent work packages.

## Interface choices and assumptions

The state classes partition interior indices; LCFS R/Z entries remain separately
prescribed. `NewtonResult` includes the active-index mapping and a convergence
flag, and requires factors at the returned state. Corrections return an additive
value correction, with sign `-lambda^T F`. The force refresh/buffer-lifetime
clarification from WP1 is recorded in the contract before its consumers use it.

`well` denotes the native Mercier `DWell` diagnostic, not an alternative magnetic
well definition. No numerical convergence or gauge-invariance assumption has
been established by fixture construction alone.

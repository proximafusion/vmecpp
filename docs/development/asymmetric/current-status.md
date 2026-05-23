# Current Status

## Scope Of This Fork

This fork is carrying a substantial asymmetric and tokamak-focused branch on top of upstream `vmecpp`. The branch contains both real implementation work and a large investigation trail. This cleanup keeps that history while separating:

- active implementation code
- supported tests and benchmark-facing interfaces
- archived investigation material

## High-Signal Source Changes Relative To Upstream

The core implementation delta is concentrated in a small set of areas:

- `src/vmecpp/__init__.py`
  Changes in the Python-facing behavior used by the benchmark workflows.
- `src/vmecpp/cpp/vmecpp/common/vmec_indata/vmec_indata.cc`
  Input validation and asymmetric array sizing changes.
- `src/vmecpp/cpp/vmecpp/common/vmec_indata/boundary_from_json.cc`
  Boundary parsing updates.
- `src/vmecpp/cpp/vmecpp/vmec/boundaries/boundaries.cc`
  Boundary and M=1 / tokamak-related handling.
- `src/vmecpp/cpp/vmecpp/vmec/boundaries/guess_magnetic_axis.cc`
  Axis-domain and Jacobian-recovery related work.
- `src/vmecpp/cpp/vmecpp/vmec/fourier_asymmetric/`
  New asymmetric transform implementation plus associated test coverage and investigation targets.
- `src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.cc`
  Solver integration for asymmetric transforms and constraint handling.
- `src/vmecpp/cpp/vmecpp/vmec/output_quantities/output_quantities.cc`
  Output handling touched by the asymmetric branch.
- `src/vmecpp/cpp/vmecpp/vmec/radial_profiles/radial_profiles.cc`
  Supporting numerical behavior touched during the branch.

## Stable vs Historical Material

Stable enough to keep visible:

- implementation code in the paths above
- benchmark-relevant example input under `examples/data/`
- real test data under `src/vmecpp/cpp/vmecpp/test_data/`
- official Python tests under `tests/`

Historical but worth preserving:

- line-by-line jVMEC comparison notes
- missing-feature analyses
- phased fix plans and convergence summaries
- one-off scripts that produced those findings

Those historical materials now live in:

- `docs/development/asymmetric/archive/`
- `tools/investigation/`

## Remaining Cleanup Work

The largest remaining source of churn is the number of exploratory test targets under `src/vmecpp/cpp/vmecpp/vmec/fourier_asymmetric/`. They were useful during diagnosis and should not be dropped casually. The next pass should classify them into:

- keep as supported regression coverage
- move into an explicit investigation-only section or subdirectory
- delete only if fully superseded by tighter tests

That reduction should be done with targeted test execution, not as a blind file sweep.

# Fourier Asymmetric Work Area

This directory mixes two kinds of material:

- the actual asymmetric transform implementation used by the solver
- a large number of exploratory tests and probes created while matching `jVMEC`

## Core Files

- `fourier_asymmetric.cc`
- `fourier_asymmetric.h`
- `BUILD.bazel`
- `CMakeLists.txt`

These are the files that define and expose the asymmetric transform code.

## Tests To Treat As The Main Entry Points

Start with the focused targets before touching the larger investigation set:

- `fourier_asymmetric_test`
- `fourier_transform_unit_test`
- `stellarator_asymmetric_test`
- `small_asymmetric_tokamak_test`

Additional solver-integration coverage also lives nearby in:

- `../ideal_mhd_model/ideal_mhd_model_asymmetric_test.cc`
- `../ideal_mhd_model/dealias_constraint_force_asymmetric_test.cc`

## Investigation Targets

Many of the remaining `debug_*` and `test_*` files in this directory were created to isolate specific failures while aligning behavior with `jVMEC`. They are valuable, but they are not equally important. Before deleting or renaming them, classify them by whether they still cover a unique bug or physics invariant.

## Next Cleanup Rule

Do not remove these exploratory targets in bulk. Reduce them only after confirming that their coverage has been replaced by smaller, named regression tests.

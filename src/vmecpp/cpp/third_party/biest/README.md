# BIEST (fetched dependency)

BIEST (https://github.com/dmalhotra/BIEST, Apache-2.0) and its SCTL
dependency (https://github.com/dmalhotra/SCTL, Apache-2.0) are fetched as
pinned archives in `//third_party:non_module_deps.bzl` (Bazel) and via
`FetchContent` (CMake); only the build overlays, local patches, and our
tests live in this repository.

Local patch (`warm_start.patch`): warm-start the iterative solve in
`ExtVacuumField::ComputeBplasma` from the previous sigma solution (SCTL's
GMRES natively accepts a non-empty initial guess; upstream never passes
one). This cuts the solve cost when the boundary changes little between
calls, as in the free-boundary VMEC iteration.

The FFT provider is Intel MKL's FFTW3 interface (see
`//third_party/mkl`); FFTW itself is GPL and cannot be linked into the
MIT-licensed VMEC++.

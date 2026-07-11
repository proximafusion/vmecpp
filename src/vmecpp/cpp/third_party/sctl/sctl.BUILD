# Build overlay for the SCTL archive fetched in
# //third_party:non_module_deps.bzl (header-only).
#
# SCTL_HAVE_LAPACK and SCTL_HAVE_FFTW are required in practice: without an
# FFT provider, SCTL's fallback FFT builds dense DFT matrices and runs out
# of memory at realistic resolutions. The FFTW3 API is provided by Intel
# MKL's FFTW interface (@mkl_fftw) -- FFTW itself is GPL and cannot be
# linked into the MIT-licensed VMEC++. LAPACKE comes from the BSD netlib
# packages (liblapacke-dev). SCTL_HAVE_BLAS is not set because
# cblas.h/cblas_f77.h are not commonly packaged together; SCTL's internal
# GEMM fallback is used instead.
cc_library(
    name = "sctl",
    hdrs = glob(["include/**"]),
    includes = ["include"],
    defines = [
        "SCTL_GLOBAL_MEM_BUFF=0",
        "SCTL_QUAD_T=__float128",
        "SCTL_HAVE_LAPACK",
        "SCTL_HAVE_FFTW",
    ],
    linkopts = [
        "-llapacke",
        "-llapack",
    ],
    visibility = ["//visibility:public"],
    deps = ["@mkl_fftw//:mkl_fftw"],
)

# Intel MKL's FFTW3 interface (the FFTW3 C API is built into the MKL core
# libraries). Used as the FFT provider for SCTL/BIEST and the Vac2 solver;
# FFTW itself is GPL and cannot be linked into the MIT-licensed VMEC++.
# The repository points at the system MKL include directory (Ubuntu:
# /usr/include/mkl from libmkl-dev).
cc_library(
    name = "mkl_fftw",
    hdrs = glob(["fftw/*.h"]),
    includes = ["fftw"],
    # Explicit GNU-threading link chain: mkl_rt's runtime layer detection
    # fails under libgomp-based processes (undefined symbol
    # mkl_sparse_optimize_bsr_trsm_i8 when loading libmkl_def.so).
    linkopts = [
        "-lmkl_gf_lp64",
        "-lmkl_gnu_thread",
        "-lmkl_core",
    ],
    visibility = ["//visibility:public"],
)

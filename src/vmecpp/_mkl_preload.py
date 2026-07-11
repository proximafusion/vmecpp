# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
# <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Make Intel MKL's internal kernel loading work under Python.

The C++ core links MKL's FFTW3 interface (the FFT provider for the BIEST
and Vac2 free-boundary solvers). MKL dispatches its computational kernels
at runtime by dlopen-ing e.g. libmkl_def.so, which must resolve symbols
from the MKL core/threading libraries. Python loads extension modules with
RTLD_LOCAL, so that resolution fails ("undefined symbol: mkl_sparse_...")
unless the MKL libraries are made globally visible first.
"""

import ctypes
import os

_MKL_LIBRARIES = (
    "libmkl_core.so",
    "libmkl_gnu_thread.so",
    "libmkl_gf_lp64.so",
)


def preload_mkl() -> None:
    """Load the MKL libraries with RTLD_GLOBAL | RTLD_LAZY (no-op without MKL)."""
    for lib in _MKL_LIBRARIES:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL | os.RTLD_LAZY)
        except OSError:
            # MKL not present as shared libraries (e.g. statically linked
            # builds); nothing to do.
            return


# run at import time: this module is imported (first) by vmecpp/__init__.py,
# before the C++ extension is loaded
preload_mkl()

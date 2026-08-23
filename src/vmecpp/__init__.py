# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++: a modern C++ reimplementation of the VMEC MHD equilibrium solver.

This module is the public interface of the package; it only re-exports the names
defined in the core modules:

- :mod:`vmecpp._types`: shared pydantic field types, the enums and array dimensions
- :mod:`vmecpp._indata`: :class:`VmecInput`, the profile helpers and the
  INDATA / VMEC++ JSON input-file handling
- :mod:`vmecpp._wout`: the output data models (wout, jxbout, mercier, threed1)
- :mod:`vmecpp._output`: :class:`VmecOutput`
- :mod:`vmecpp._run`: :func:`run`, the main entry point
- :mod:`vmecpp._continuation`: resolution interpolation for continuation schedules
- :mod:`vmecpp._iteration`: the Python-side force-balance iteration
- :mod:`vmecpp._free_boundary`: makegrid parameters and magnetic field response tables

Everything that used to live in this module is still importable from here, both the
documented API listed in ``__all__`` and the names below that are not part of it.
"""

from __future__ import annotations

from vmecpp import _util  # noqa: F401
from vmecpp._continuation import interpolate_solution
from vmecpp._free_boundary import (
    MagneticFieldResponseTable,
    MakegridParameters,
)
from vmecpp._indata import (
    VmecInput,
    ensure_vmec2000_input,  # noqa: F401
    ensure_vmecpp_input,  # noqa: F401
    is_vmec2000_input,  # noqa: F401
    populate_raw_profile,  # noqa: F401
    set_profile,
)
from vmecpp._iteration import (
    IterationResult,
    IterationState,
    RestartReason,  # noqa: F401
    iterate,
    solve_equilibrium,
    solve_multigrid,
)
from vmecpp._output import VmecOutput
from vmecpp._pydantic_numpy import BaseModelWithNumpy  # noqa: F401
from vmecpp._run import run
from vmecpp._types import (
    AuxFType,  # noqa: F401
    AuxSType,  # noqa: F401
    FreeBoundaryMethod,
    IterationStyle,
    MgridModeType,  # noqa: F401
    MpolNtorField,  # noqa: F401
    OutputMode,  # noqa: F401
    ProfileType,  # noqa: F401
    SerializableSparseCoefficientArray,  # noqa: F401
    SerializeIntAsFloat,  # noqa: F401
    ndfmax,  # noqa: F401
    preset,  # noqa: F401
)
from vmecpp._wout import (
    JxBOut,
    Mercier,
    Threed1AxisGeometry,  # noqa: F401
    Threed1Betas,  # noqa: F401
    Threed1FirstTable,  # noqa: F401
    Threed1GeometricAndMagneticQuantities,  # noqa: F401
    Threed1ShafranovIntegrals,  # noqa: F401
    Threed1Volumetrics,
    VmecWOut,
)
from vmecpp.cpp import _vmecpp  # type: ignore # noqa: F401  bindings to the C++ core

# Ordered this way to ensure run, VmecInput, and VmecOutput are the first three
# items in the generated documentation.
__all__ = [  # noqa: RUF022
    "run",
    "interpolate_solution",
    "VmecInput",
    "VmecOutput",
    "VmecWOut",
    "JxBOut",
    "Mercier",
    "Threed1Volumetrics",
    "MakegridParameters",
    "MagneticFieldResponseTable",
    "FreeBoundaryMethod",
    "IterationStyle",
    "set_profile",
    "iterate",
    "solve_equilibrium",
    "solve_multigrid",
    "IterationResult",
    "IterationState",
]

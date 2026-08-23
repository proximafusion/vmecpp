# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""VMEC++: a modern C++ reimplementation of the VMEC MHD equilibrium solver.

This module is the public interface of the package: it only re-exports what the
core modules define -- see their own docstrings for what each one owns.
"""

from __future__ import annotations

from vmecpp._continuation import interpolate_solution
from vmecpp._free_boundary import (
    MagneticFieldResponseTable,
    MakegridParameters,
)
from vmecpp._indata import (
    VmecInput,
    ensure_vmec2000_input,
    ensure_vmecpp_input,
    is_vmec2000_input,
    populate_raw_profile,
    set_profile,
)
from vmecpp._iteration import (
    IterationResult,
    IterationState,
    iterate,
    solve_equilibrium,
    solve_multigrid,
)
from vmecpp._output import VmecOutput
from vmecpp._pydantic_numpy import BaseModelWithNumpy
from vmecpp._run import run
from vmecpp._types import (
    AuxFType,
    AuxSType,
    FreeBoundaryMethod,
    IterationStyle,
    MgridModeType,
    MpolNtorField,
    OutputMode,
    ProfileType,
    RestartReason,
    SerializableSparseCoefficientArray,
    SerializeIntAsFloat,
    ndfmax,
    preset,
)
from vmecpp._wout import (
    JxBOut,
    Mercier,
    Threed1AxisGeometry,
    Threed1Betas,
    Threed1FirstTable,
    Threed1GeometricAndMagneticQuantities,
    Threed1ShafranovIntegrals,
    Threed1Volumetrics,
    VmecWOut,
)
from vmecpp.cpp import _vmecpp  # type: ignore # bindings to the C++ core

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

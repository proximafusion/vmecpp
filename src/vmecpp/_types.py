# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Shared pydantic/NumPy field types, enums and fixed array dimensions.

The annotated types defined here are used by both the input model
(:mod:`vmecpp._indata`) and the output models (:mod:`vmecpp._wout`); the enums are
part of the public API and are re-exported from the ``vmecpp`` package.
"""

from __future__ import annotations

import enum
import typing

import jaxtyping as jt
import numpy as np
import pydantic

from vmecpp import _util
from vmecpp.cpp import _vmecpp  # type: ignore # bindings to the C++ core

# Fixed dimension of the profile inputs (i.e. pressure, iota, current)
preset = 21
# Fixed dimension of the auxiliary profile quantities (i.e. am_aux_f)
ndfmax = 101

_ArrayType = typing.TypeVar("_ArrayType")


def _wrap_dense_to_sparse(
    value: typing.Any,
    handler: pydantic.SerializerFunctionWrapHandler,
    _: pydantic.FieldSerializationInfo,
) -> list[dict[str, float | int]]:
    # Handle ndarray directly, and also lists (which arise when the base class
    # _serialize_field wrap serializer converts ndarray -> list before this runs).
    if isinstance(value, (np.ndarray, list)):
        return _util.dense_to_sparse_coefficients(np.asarray(value))
    return handler(value)


SerializableSparseCoefficientArray: typing.TypeAlias = typing.Annotated[
    _ArrayType,
    pydantic.WrapSerializer(_wrap_dense_to_sparse, when_used="unless-none"),  # pyright: ignore[reportArgumentType]
    pydantic.BeforeValidator(_util.sparse_to_dense_coefficients_implicit),
]


def _wrap_int_as_float(
    value: typing.Any,
    handler: pydantic.SerializerFunctionWrapHandler,
    _: pydantic.FieldSerializationInfo,
) -> list[float]:
    if isinstance(value, (np.ndarray, list)):
        return np.array(value).astype(np.float64).tolist()
    return handler(value)


SerializeIntAsFloat: typing.TypeAlias = typing.Annotated[
    _ArrayType,
    pydantic.WrapSerializer(_wrap_int_as_float),  # pyright: ignore[reportArgumentType]
    pydantic.BeforeValidator(lambda x: np.array(x).astype(np.int64)),
]


def _coerce_mpol_ntor(value: typing.Any) -> int | np.ndarray:
    """Normalizes an ``mpol``/``ntor`` field value.

    A length-1 sequence is equivalent to a scalar and is collapsed to one; anything
    longer is kept as an int array representing a per-``ns_array``-step Fourier
    resolution continuation schedule.
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    array = np.atleast_1d(np.asarray(value, dtype=np.int64))
    return int(array[0]) if array.size == 1 else array


MpolNtorField: typing.TypeAlias = typing.Annotated[
    int | jt.Int[np.ndarray, "num_fourier_steps"],
    pydantic.BeforeValidator(_coerce_mpol_ntor),
]


def _final_resolution(value: int | np.ndarray) -> int:
    """The target Fourier resolution: itself if scalar, else the schedule's last
    (finest) entry."""
    return value if isinstance(value, int) else int(value[-1])


AuxFType = typing.Annotated[
    _ArrayType,
    pydantic.BeforeValidator(lambda x: _util.right_pad(x, ndfmax, 0.0)),
]
AuxSType = typing.Annotated[
    _ArrayType,
    pydantic.BeforeValidator(lambda x: _util.right_pad(x, ndfmax, -1.0)),
]

MgridModeType: typing.TypeAlias = typing.Annotated[
    typing.Literal["R", "S", ""], pydantic.Field(max_length=1)
]
"""[Scaled, Raw, Unset]"""

ProfileType = typing.Annotated[str, pydantic.Field(max_length=20)]


class FreeBoundaryMethod(str, enum.Enum):
    """Method for handling free-boundary conditions."""

    NESTOR = "nestor"
    """NEumann Solver for TORoidal systems."""

    ONLY_COILS = "only_coils"
    """Use just the coils field for the free-boundary force contribution.

    This can be particularly useful for verification calculations.

    Warning: This is only valid for vacuum calculations and will ignore
    the plasma current contribution!
    """

    BIEST = "biest"
    """Boundary Integral Equation Solver for Toroidal systems."""


class IterationStyle(str, enum.Enum):
    """Time-step / restart control scheme for the equilibrium iteration."""

    VMEC_8_52 = "vmec_8_52"
    """The Fortran VMEC 8.52 control (the default)."""

    PARVMEC = "parvmec"
    """The PARVMEC / VMEC2000 9.0 control."""


class OutputMode(enum.Enum):
    """Controls the output format of iteration logging.."""

    SILENT = _vmecpp.OutputMode.SILENT  # 0
    """No output."""

    LEGACY = _vmecpp.OutputMode.LEGACY  # 1
    """Traditional table output (original VMEC++ format)"""

    PROGRESS = _vmecpp.OutputMode.PROGRESS  # 2
    """Multi-line progress bars with ANSI cursor movement (TTY)"""

    PROGRESS_NON_TTY = _vmecpp.OutputMode.PROGRESS_NON_TTY  # 3
    """Single-line progress with carriage return (Jupyter, etc.)"""


def _validate_free_boundary_method(
    value: _vmecpp.FreeBoundaryMethod | str | FreeBoundaryMethod,
) -> FreeBoundaryMethod:
    """Convert various representations to FreeBoundaryMethod."""
    if isinstance(value, _vmecpp.FreeBoundaryMethod):
        return FreeBoundaryMethod(value.name.lower())  # pyright: ignore[reportAttributeAccessIssue]
    return FreeBoundaryMethod(str(value))


def _validate_iteration_style(
    value: _vmecpp.IterationStyle | str | IterationStyle,
) -> IterationStyle:
    """Convert various representations to IterationStyle."""
    if isinstance(value, _vmecpp.IterationStyle):
        return IterationStyle(value.name.lower())  # pyright: ignore[reportAttributeAccessIssue]
    return IterationStyle(str(value))

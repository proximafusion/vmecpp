# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""SIMSOPT compatibility layer for VMEC++."""

import logging
from pathlib import Path
from typing import Optional, cast

import jaxtyping as jt
import numpy as np
from simsopt._core.optimizable import Optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.util.mpi import MpiPartition

import vmecpp
from vmecpp import (  # noqa: F401
    # Re-export specific functions from vmecpp for backwards compatibility
    ensure_vmec2000_input,
    ensure_vmecpp_input,
    is_vmec2000_input,
)

logger = logging.getLogger(__name__)

# NOTE: this will be needed to set Vmec.mpi.
# VMEC++ does not use MPI, but Vmec.mpi must be set anyways to make tools like Boozer
# happy: they expect to be able to extract the mpi controller from the Vmec object,
# e.g. here:
# https://github.com/hiddenSymmetries/simsopt/blob/d95a479257c3e7373c82ba2bc1613e1ee3e0a42f/src/simsopt/mhd/boozer.py#L80
# starfinder/mhd/vmec_decorator.py also expects a non-null self.mpi:
# for example it unconditionally accesses self.mpi.group.
#
# Creating an MpiPartition hogs memory until process exit, so we do it here once at
# module scope rather than every time Vmec.__init__ is called.
try:
    from mpi4py import MPI  # pyright: ignore

    MPI_PARTITION = MpiPartition(ngroups=1)
except ImportError:
    MPI = None



import contextlib
import enum
import json
import os
import sys
import tempfile
import types
import typing
from collections.abc import Generator
import jaxtyping as jt
import netCDF4
import pydantic
from vmecpp import _util
from vmecpp.cpp import _vmecpp
from vmecpp._pydantic_numpy import BaseModelWithNumpy
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


# This is a pure Python equivalent of VmecINDATAPyWrapper.
# In the future VmecINDATAPyWrapper and the C++ VmecINDATA will merge into one type,
# and this will become a Python wrapper around the one C++ VmecINDATA type.
# This pure Python type could _also_ disappear if we can get proper autocompletion,
# docstring peeking etc. for the one C++ VmecINDATA type bound via pybind11.
class VmecInput(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    """The input to a VMEC++ run. Contains settings as well as the definition of the
    plasma boundary.

    Python equivalent of a VMEC++ JSON input file or a classic INDATA file (e.g.
    "input.best").

    Deserialize from JSON and serialize to JSON using the usual pydantic methods:
    ``model_validate_json`` and ``model_dump_json``.
    """

    model_config = pydantic.ConfigDict(
        # serialize NaN and infinite floats as strings in JSON output.
        ser_json_inf_nan="strings",
    )

    lasym: bool = False
    """Flag to indicate non-stellarator-symmetry.

    - False, assumes stellarator symmetry (only cosine/sine coefficients used).
    - True, (currently unsupported) allows for non-stellarator-symmetric terms.
    """

    nfp: int = 1
    """Number of toroidal field periods (=1 for Tokamak)"""

    mpol: MpolNtorField = 6
    """Number of poloidal Fourier harmonics; m = 0, 1, ..., (mpol-1).

    May also be a sequence of ints, with one entry per ``ns_array`` step (a scalar
    broadcasts to every step), to request continuation in Fourier resolution:
    ``vmecpp.run()`` then solves each step in turn, hot-restarting from the
    previous step's solution interpolated to the new resolution (see
    :func:`interpolate_solution`). The boundary coefficients (``rbc``, ``zbs``, ...)
    are always defined at the final (largest-index) entry's resolution.
    """

    ntor: MpolNtorField = 0
    """Number of toroidal Fourier harmonics; n = -ntor, -ntor+1, ..., -1, 0, 1, ...,
    ntor-1, ntor.

    May be a sequence of ints, analogous to :attr:`mpol`; see its docstring.
    """

    mpol_geometry: int = -1
    """Optional reduced poloidal resolution for the geometry (R, Z).

    If in [1, mpol), R/Z modes with m >= mpol_geometry are held fixed while lambda keeps
    the full mpol. < 0 (default) means geometry uses mpol.
    """

    ntor_geometry: int = -1
    """Optional reduced toroidal resolution for the geometry (R, Z).

    If in [0, ntor), R/Z modes with n > ntor_geometry are held fixed while lambda keeps
    the full ntor. < 0 (default) means geometry uses ntor.
    """

    ntheta: int = 0
    """Number of poloidal grid points (ntheta >= 0).

    Controls the poloidal resolution in real space. If 0, chosen automatically as
    minimally allowed. Must be at least 2*mpol + 6.
    """

    nzeta: int = 0
    """Number of toroidal grid points (nzeta >= 0).

    Controls the toroidal resolution in real space. If 0, chosen automatically as
    minimally allowed. Must be at least 2*ntor + 4. We typically use use phi as the
    convention for the toroidal angle, the name nzeta is due to beckwards compatibility.
    """

    ns_array: jt.Int[np.ndarray, "num_grids"] = pydantic.Field(
        default_factory=lambda: np.array([31], dtype=np.int64)
    )
    """Number of flux surfaces per multigrid step.

    Each entry >= 3 and >= previous entry.
    """

    ftol_array: jt.Float[np.ndarray, "num_grids"] = pydantic.Field(
        default_factory=lambda: np.array([1.0e-10])
    )
    """Requested force tolerance for convergence per multigrid step."""

    niter_array: jt.Int[np.ndarray, "num_grids"] = pydantic.Field(
        default_factory=lambda: np.array([100], dtype=np.int64)
    )
    """Maximum number of iterations per multigrid step."""

    phiedge: float = 1.0
    """Total enclosed toroidal magnetic flux in Vs == Wb.

    - In fixed-boundary, this determines the magnetic field strength.
    - In free-boundary, the magnetic field strength is given externally,
      so this determines cross-section area and volume of the plasma.
    """

    ncurr: typing.Literal[0, 1] = typing.cast(typing.Literal[0, 1], 0)
    """Select constraint on iota or enclosed toroidal current profiles.

    - 0: constrained-iota (rotational transform profile specified)
    - 1: constrained-current (toroidal current profile specified)
    """

    pmass_type: ProfileType = "power_series"
    """Parametrization of mass/pressure profile."""

    am: jt.Float[np.ndarray, "am_len"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Mass/pressure profile coefficients.

    Units: Pascals for pressure.
    """

    am_aux_s: jt.Float[np.ndarray, "am_aux_len"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Spline mass/pressure profile: knot locations in s"""

    am_aux_f: jt.Float[np.ndarray, "am_aux_len"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Spline mass/pressure profile: values at knots"""

    pres_scale: float = 1.0
    """Global scaling factor for mass/pressure profile."""

    gamma: float = 0.0
    r"""Adiabatic index :math:`\gamma` (ratio of specific heats).

    Specifying 0 implies that the pressure profile is specified. For all other values,
    the mass profile is specified.
    """

    spres_ped: float = 1.0
    """Location of pressure pedestal in s.

    Outside this radial location, pressure is constant.
    """

    piota_type: ProfileType = "power_series"
    """Parametrization of iota (rotational transform) profile."""

    ai: jt.Float[np.ndarray, "ai_len"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Iota profile coefficients."""

    ai_aux_s: jt.Float[np.ndarray, "ai_aux_len"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Spline iota profile: knot locations in s"""

    ai_aux_f: jt.Float[np.ndarray, "ai_aux_len"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Spline iota profile: values at knots"""

    pcurr_type: ProfileType = "power_series"
    """Parametrization of toroidal current profile."""

    ac: jt.Float[np.ndarray, "ac_len"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Enclosed toroidal current profile coefficients."""

    ac_aux_s: jt.Float[np.ndarray, "ac_aux_len"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Spline toroidal current profile: knot locations in s"""

    ac_aux_f: jt.Float[np.ndarray, "ac_aux_len"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Spline toroidal current profile: values at knots"""

    curtor: float = 0.0
    """Net toroidal current in A.

    The toroidal current profile is scaled to yield this total.
    """

    bloat: float = 1.0
    """Bloating factor (for constrained toroidal current)"""

    lfreeb: bool = False
    """Flag to indicate free-boundary.

    If True, run in free-boundary mode; if False, fixed-boundary.
    """

    mgrid_file: typing.Annotated[str, pydantic.Field(max_length=200)] = "NONE"
    """Full path for vacuum Green's function data.

    NetCDF MGRID file with magnetic field response factors for external coils.
    """

    extcur: jt.Float[np.ndarray, "ext_current"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Coil currents in A."""

    nvacskip: int = 1
    """Number of iterations between full vacuum calculations."""

    free_boundary_method: typing.Annotated[
        FreeBoundaryMethod,
        pydantic.BeforeValidator(_validate_free_boundary_method),
        pydantic.Field(),
    ] = FreeBoundaryMethod.NESTOR
    """Method for handling free-boundary conditions."""

    iteration_style: typing.Annotated[
        IterationStyle,
        pydantic.BeforeValidator(_validate_iteration_style),
        pydantic.Field(),
    ] = IterationStyle.VMEC_8_52
    """Time-step / restart control scheme for the equilibrium iteration (``"vmec_8_52"``
    or ``"parvmec"``)."""

    nstep: int = 10
    """Printout interval at which convergence progress is logged."""

    aphi: jt.Float[np.ndarray, "aphi_len"] = pydantic.Field(
        default_factory=lambda: np.array([1.0])
    )
    """Radial flux zoning profile coefficients."""

    delt: float = 1.0
    """Initial value for artificial time step in iterative solver."""

    tcon0: float = 1.0
    """Constraint force scaling factor for ns --> 0."""

    lforbal: bool = False
    """Hack: directly compute innermost flux surface geometry from radial force balance"""

    return_outputs_even_if_not_converged: bool = False
    """If true, return a wout even if VMEC++ did not converge, instead of raising a
    RuntimeError.

    This is intended for debugging purposes (e.g. inspecting how far the geometry
    got, or where the force residuals blew up) since the returned quantities are
    computed from whatever internal state the solver was in when it gave up, and
    can be arbitrarily unphysical. Always check `wout.ier_flag` / the accompanying
    log warning to see why the run did not converge before interpreting any
    physical quantity in the output.
    """

    raxis_c: jt.Float[np.ndarray, "ntor_plus_1"] = pydantic.Field(
        default_factory=lambda: np.array([0.0])
    )
    """Magnetic axis coefficients for R ~ cos(n*v); stellarator-symmetric.

    At least 1 value required, up to n=ntor considered.
    """

    zaxis_s: jt.Float[np.ndarray, "ntor_plus_1"] = pydantic.Field(
        default_factory=lambda: np.array([0.0])
    )
    """Magnetic axis coefficients for Z ~ sin(n*v); stellarator-symmetric.

    Up to n=ntor considered; first entry (n=0) is ignored.
    """

    raxis_s: jt.Float[np.ndarray, "ntor_plus_1"] | None = None
    """Magnetic axis coefficients for R ~ sin(n*v); non-stellarator-symmetric.

    Up to n=ntor considered; first entry (n=0) is ignored. Only used if lasym=True.
    """

    zaxis_c: jt.Float[np.ndarray, "ntor_plus_1"] | None = None
    """Magnetic axis coefficients for Z ~ cos(n*v); non-stellarator-symmetric.

    Only used if lasym=True.
    """

    rbc: SerializableSparseCoefficientArray[
        jt.Float[np.ndarray, "mpol two_ntor_plus_one"]
    ] = pydantic.Field(default_factory=lambda: np.zeros((6, 1)))
    """Boundary coefficients for R ~ cos(m*u - n*v); stellarator-symmetric"""

    zbs: SerializableSparseCoefficientArray[
        jt.Float[np.ndarray, "mpol two_ntor_plus_one"]
    ] = pydantic.Field(default_factory=lambda: np.zeros((6, 1)))
    """Boundary coefficients for Z ~ sin(m*u - n*v); stellarator-symmetric"""

    rbs: (
        SerializableSparseCoefficientArray[
            jt.Float[np.ndarray, "mpol two_ntor_plus_one"]
        ]
        | None
    ) = None
    """Boundary coefficients for R ~ sin(m*u - n*v); non-stellarator-symmetric.

    Only used if lasym=True.
    """

    zbc: (
        SerializableSparseCoefficientArray[
            jt.Float[np.ndarray, "mpol two_ntor_plus_one"]
        ]
        | None
    ) = None
    """Boundary coefficients for Z ~ cos(m*u - n*v); non-stellarator-symmetric.

    Only used if lasym=True.
    """

    @pydantic.model_validator(mode="after")
    def _validate_fourier_coefficients_shapes(self) -> VmecInput:
        """All geometry coefficients need to have the shape (mpol, 2*ntor+1), wit 'rbs',
        'zbc' only populated for non-stellarator symmetric configurations."""
        mpol_two_ntor_plus_one_fields = ["rbc", "zbs"]
        if self.lasym:
            mpol_two_ntor_plus_one_fields.extend(["rbs", "zbc"])

        mpol_final = _final_resolution(self.mpol)
        ntor_final = _final_resolution(self.ntor)
        expected_shape = (mpol_final, 2 * ntor_final + 1)
        for field in mpol_two_ntor_plus_one_fields:
            current_value = getattr(self, field)

            if current_value is None:
                current_value = np.zeros(expected_shape)
                setattr(self, field, current_value)

            shape = np.shape(current_value)
            if shape != expected_shape:
                setattr(
                    self,
                    field,
                    VmecInput.resize_2d_coeff(
                        current_value,
                        mpol_new=mpol_final,
                        ntor_new=ntor_final,
                    ),
                )

        # The 1D magnetic-axis arrays must have length ntor+1. Shorter arrays
        # simply omit trailing (zero) coefficients and are silently zero-padded;
        # longer arrays are rejected rather than silently truncated.
        ntor_plus_one_fields = ["raxis_c", "zaxis_s"]
        if self.lasym:
            ntor_plus_one_fields.extend(["raxis_s", "zaxis_c"])
        expected_axis_len = ntor_final + 1
        for field in ntor_plus_one_fields:
            current_value = getattr(self, field)
            if current_value is None:
                continue
            if np.size(current_value) != expected_axis_len:
                setattr(
                    self,
                    field,
                    VmecInput.resize_1d_axis_coeff(current_value, ntor_new=ntor_final),
                )
        return self

    @pydantic.model_validator(mode="after")
    def _validate_stellarator_asymmetric_fields(self) -> VmecInput:
        """Check if all fields that break stellarator symmetry match the lasym flag."""
        ASYMMETRIC_FIELDS = ["rbs", "zbc", "zaxis_c", "raxis_s"]
        is_stellarator_symmetric = not self.lasym
        if is_stellarator_symmetric:
            for key in ASYMMETRIC_FIELDS:
                value = getattr(self, key)
                # Then all asymmetric fields should be None
                if value is not None:
                    msg = (
                        "The input is for a stellarator symmetric configuration (lasym=False), "
                        f"but the symmetry-breaking field '{key}' is populated with \n{value}"
                    )
                    raise ValueError(msg)
        return self

    @staticmethod
    def resize_1d_axis_coeff(
        coeff: jt.Float[np.ndarray, "ntor_plus_1"],
        ntor_new: int,
    ) -> jt.Float[np.ndarray, "ntor_new_plus_1"]:
        """Resizes a 1D magnetic-axis Fourier coefficient array to length ntor_new+1.

        Arrays shorter than ntor_new+1 are zero-padded (the omitted trailing
        coefficients are implicitly zero). Arrays longer than ntor_new+1 are
        rejected to avoid silently truncating user-data.

        Args:
            coeff: A 1D NumPy array of axis coefficients (length ntor+1).
            ntor_new: The new number of toroidal modes.

        Examples:
            >>> VmecInput.resize_1d_axis_coeff(np.array([1.0, 2.0]), ntor_new=3)
            array([1., 2., 0., 0.])
        """
        assert ntor_new >= 0
        coeff = np.asarray(coeff, dtype=float).ravel()
        new_len = ntor_new + 1
        if coeff.size > new_len:
            msg = (
                f"length of axis coefficient array ({coeff.size}) exceeds ntor+1 ({new_len}). "
                f"Please truncate r_axis_c and zaxis_s to a size consistent with ntor={ntor_new}."
            )
            raise ValueError(msg)
        resized_coeff = np.zeros(new_len)
        resized_coeff[: coeff.size] = coeff
        return resized_coeff

    @staticmethod
    def resize_2d_coeff(
        coeff: jt.Float[np.ndarray, "mpol two_ntor_plus_one"],
        mpol_new: int,
        ntor_new: int,
    ) -> jt.Float[np.ndarray, "mpol_new two_ntor_new_plus_one"]:
        """Resizes a 2D NumPy array representing Fourier coefficients, padding with
        zeros or truncating as needed.

        Args:
            coeff: A NumPy array of shape (mpol, 2 * ntor + 1).
            mpol_new: The new number of poloidal modes.
            ntor_new: The new number of toroidal modes.

        Examples:
            >>> coeff = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            >>> VmecInput.resize_2d_coeff(coeff, 3, 3)
            array([[ 0.,  1.,  2.,  3.,  4.,  5.,  0.],
                   [ 0.,  6.,  7.,  8.,  9., 10.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]])

            >>> VmecInput.resize_2d_coeff(coeff, 1, 1)
            array([[2., 3., 4.]])

            >>> VmecInput.resize_2d_coeff(coeff, 4, 1)
            array([[2., 3., 4.],
                   [7., 8., 9.],
                   [0., 0., 0.],
                   [0., 0., 0.]])
        """

        assert mpol_new >= 0
        assert ntor_new >= 0
        coeff = np.array(coeff)
        mpol, nmax = coeff.shape
        ntor = (nmax - 1) // 2
        assert nmax == 2 * ntor + 1

        new_nmax = 2 * ntor_new + 1
        resized_coeff = np.zeros((mpol_new, new_nmax))

        smaller_ntor = min(ntor, ntor_new)
        smaller_mpol = min(mpol, mpol_new)
        if mpol_new < mpol or ntor_new < ntor:
            logger.warning(
                f"Discarding coefficients because mpol={mpol} or ntor={ntor} "
                f"are smaller than mpol_new={mpol_new} or ntor_new={ntor_new}"
            )

        for m in range(smaller_mpol):
            for n in range(-smaller_ntor, smaller_ntor + 1):
                resized_coeff[m, n + ntor_new] = coeff[m, n + ntor]

        return resized_coeff

    @staticmethod
    def from_file(input_file: str | Path) -> VmecInput:
        """Build a VmecInput from either a VMEC++ JSON input file or a classic INDATA
        file."""
        absolute_input_path = Path(input_file).resolve()

        # we call this in a temporary directory because it produces the file in the current working directory
        with (  # noqa: SIM117
            tempfile.TemporaryDirectory() as tmpdir,
            _util.change_working_directory_to(Path(tmpdir)),
        ):
            with ensure_vmecpp_input(absolute_input_path) as vmecpp_input_file:
                # `VmecINDATA` populates missing fields with default values, while `VmecInput` doesn't.
                # Therefore we use `VmecINDATA` here to read the user input, before validating the model
                vmecpp_indata = _vmecpp.VmecINDATA.from_file(vmecpp_input_file)
        # At this point all required fields are populated with user defined or default values.
        # Passing missing or extra fields to `VmecInput.model_validate` will otherwise raise an error.
        return VmecInput._from_cpp_vmecindata(vmecpp_indata)

    @staticmethod
    def _from_cpp_vmecindata(
        vmecindata: _vmecpp.VmecINDATA,
    ) -> VmecInput:
        # The VmecInput.model_validate() is strict in its data model, all fields need to be present and valid.
        # VmecInput does _not_ have any default values.
        vmec_input_dict = {
            attr_name: getattr(vmecindata, attr_name)
            for attr_name in VmecInput.model_fields
        }
        vmec_input_dict["ns_array"] = vmec_input_dict["ns_array"].astype(np.int64)
        vmec_input_dict["niter_array"] = vmec_input_dict["niter_array"].astype(np.int64)

        return VmecInput.model_validate(vmec_input_dict)

    @staticmethod
    def default():
        """Return a ``VmecInput`` with VMEC++ default values."""
        return VmecInput()

    def _to_cpp_vmecindata(self) -> _vmecpp.VmecINDATA:
        cpp_indata = _vmecpp.VmecINDATA()

        # these are read-only in VmecINDATA to
        # guarantee consistency with mpol and ntor:
        # we can't set the attributes directly but we
        # can set their elements after calling _set_mpol_ntor.
        readonly_attrs = {
            "mpol",
            "ntor",
            "raxis_c",
            "zaxis_s",
            "raxis_s",
            "zaxis_c",
            "rbc",
            "zbs",
            "rbs",
            "zbc",
        }

        for attr in VmecInput.model_fields:
            if attr in readonly_attrs or attr in (
                "free_boundary_method",
                "iteration_style",
            ):
                continue  # these must be set separately
            setattr(cpp_indata, attr, getattr(self, attr))

        # Convert Python enum to C++ enum
        cpp_indata.free_boundary_method = getattr(
            _vmecpp.FreeBoundaryMethod, self.free_boundary_method.upper()
        )
        cpp_indata.iteration_style = getattr(
            _vmecpp.IterationStyle, self.iteration_style.upper()
        )

        # this also resizes the readonly_attrs
        cpp_indata._set_mpol_ntor(
            _final_resolution(self.mpol), _final_resolution(self.ntor)
        )
        for attr in readonly_attrs - {"mpol", "ntor"}:
            # now we can set the elements of the readonly_attrs
            value = getattr(self, attr)

            # Asymmetric fields are only populated when lasym==True
            # so we need to skip them for itemwise assignment
            if value is None:
                assert attr in {"rbs", "zbc", "zaxis_c", "raxis_s"}
                # All asymmetric fields should be initialized when lasym=True
                if cpp_indata.lasym:
                    msg = f"Field {attr} should not be None when lasym=True"
                    raise ValueError(msg)
                # Skip None values (don't try to assign them)
            else:
                # Check if non-None asymmetric fields are being set when lasym=False
                if (
                    attr in {"rbs", "zbc", "zaxis_c", "raxis_s"}
                    and not cpp_indata.lasym
                ):
                    msg = (
                        f"Cannot set asymmetric field '{attr}' when lasym=False. "
                        "Either set lasym=True or remove the asymmetric field."
                    )
                    raise ValueError(msg)
                getattr(cpp_indata, attr)[:] = value

        return cpp_indata

    # By default we want to write everything to JSON, so the the file is a
    # single source of truth without an implicit dependence on defaults.
    def to_json(self, **kwargs) -> str:
        """Serialize the object to JSON.

        Keyword Args:
            **kwargs: Additional keyword arguments to forward to the model_dump_json method.
        """

        return self.model_dump_json(**kwargs)

    def save(self, output_path: str | Path, **kwargs) -> None:
        json_serialized = self.to_json(**kwargs)
        output_path = Path(output_path)
        output_path.write_text(json_serialized)


# Fixed dimension of the profile inputs (i.e. pressure, iota, current)
preset = 21
# Fixed dimension of the auxiliary profile quantities (i.e. am_aux_f)
ndfmax = 101


# NOTE: in the future we want to change the C++ WOutFileContents layout so that it
# matches the classic Fortran one, so most of the compatibility layer here could
# disappear.
class VmecWOut(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    """Python equivalent of a VMEC "wout file".

    VmecWOut exposes the layout that SIMSOPT expects.
    The ``save`` method produces a NetCDF file compatible with SIMSOPT/Fortran VMEC ``wout.nc``.
    """

    # We use alias names to map to the wout keys, when they differ from the variable
    # names in Python (e.g. lasym__logical__ instead of lasym). By default, we want
    # to use the nicer Python names and explicitly opt in to use the wout names.
    model_config = pydantic.ConfigDict(
        validate_by_alias=False,
        validate_by_name=True,
        serialize_by_alias=False,
        # Allow for variables in the wout file even if VMEC++ doesn't use them.
        extra="allow",
    )

    _CPP_WOUT_SPECIAL_HANDLING: typing.ClassVar[list[str]] = [
        # Asymmetric-only fields (None when lasym=False)
        "raxis_cs",
        "zaxis_cc",
        "rmns",
        "zmnc",
        "lmnc_full",
        "bsubsmnc",
        # Asymmetric half-grid 2D arrays (None when lasym=False)
        "lmnc",
        "gmns",
        "bmns",
        "bsubumns",
        "bsubvmns",
        "bsupumns",
        "bsupvmns",
        "currumns",
        "currvmns",
    ]
    """If quantities are not exactly the same in C++ WoutFileContents and this class,
    add them to this list and implement the conversion logic in _to_cpp_wout and
    _from_cpp_wout (e.g. different naming, storage order).

    TODO(jurasic) homogenize the two so this list can disappear.
    """

    input_extension: typing.Annotated[str, pydantic.Field(max_length=100)] = ""
    """File extension of the input file."""

    ier_flag: int
    """Status code indicating success or problems during the VMEC++ run.

    See the ``reason`` property for a human-readable description.
    """

    @property
    def reason(self) -> str:
        return {
            0: "no fatal error but convergence was not reached",
            1: "initially bad Jacobian",
            3: "NCURR_NE_1_BLOAT_NE_1",
            4: "Jacobian reset 75 times, the geometry isn't well defined",
            5: "input parsing error",
            8: "NS array must not be all zeroes",
            9: "miscellaneous error, can happen in mgrid_mod",
            10: "vacuum VMEC and ITOR mismatch",
            11: "ftolv termination condition satisfied",
        }.get(self.ier_flag, "unknown error")

    nfp: int
    """Number of toroidal field periods."""

    ns: int
    """Number of radial grid points (=number of flux surfaces)."""

    mpol: int
    """Number of poloidal Fourier modes."""

    ntor: int
    """Number of toroidal Fourier modes."""

    mnmax: int
    """Number of Fourier coefficients for the state vector."""

    mnmax_nyq: int
    """Number of Fourier coefficients for the Nyquist-quantities."""

    # Serialized as int in the wout file under a different name
    lasym: typing.Annotated[
        bool,
        pydantic.PlainSerializer(int),
        pydantic.BeforeValidator(bool),
        pydantic.Field(alias="lasym__logical__"),
    ]
    """Flag indicating non-stellarator-symmetry.

    Non-stellarator symmetric fields are only populated if this is True.
    """

    lfreeb: typing.Annotated[
        bool,
        pydantic.PlainSerializer(int),
        pydantic.BeforeValidator(bool),
        pydantic.Field(alias="lfreeb__logical__"),
    ]
    """Flag indicating free-boundary computation."""

    lrfp: typing.Annotated[
        bool,
        pydantic.PlainSerializer(int),
        pydantic.BeforeValidator(bool),
        pydantic.Field(alias="lrfp__logical__", default=False),
    ]
    """Flag indicating reversed-field pinch configuration."""

    wb: float
    """Magnetic energy: volume integral of `|B|^2/(2 mu0)`."""

    wp: float
    """Kinetic energy: volume integral of `p`."""

    rmax_surf: float
    """Maximum ``R`` on the plasma boundary over all grid points."""

    rmin_surf: float
    """Minimum ``R`` on the plasma boundary over all grid points."""

    zmax_surf: float
    """Maximum ``Z`` on the plasma boundary over all grid points."""

    aspect: float
    """Aspect ratio (major radius over minor radius) of the plasma boundary."""
    betapol: float
    r"""Poloidal plasma beta.

    The ratio of the total thermal energy of the plasma to the total poloidal magnetic
    energy. :math:`\beta = W_{th} / W_{B_\theta} = \int p\, dV / \left( \int B_\theta^2
    / (2 \mu_0)\, dV \right )`
    """

    betator: float
    r"""Toroidal plasma beta.

    The ratio of the total thermal energy of the plasma to the total toroidal magnetic
    energy. :math:`\beta = W_{th} / W_{B_\phi} = \int p\, dV / \left( \int B_\phi^2 / (2
    \mu_0)\, dV \right )`
    """

    betaxis: float
    """Plasma beta on the magnetic axis."""

    b0: float
    """Toroidal magnetic flux density from poloidal current and magnetic axis position
    at ``phi=0``."""

    rbtor0: float
    """Poloidal ribbon current at the axis."""

    rbtor: float
    """Poloidal ribbon current at the plasma boundary."""

    IonLarmor: float
    """Larmor radius of plasma ions."""

    ctor: float
    """Net toroidal plasma current."""

    Aminor_p: float
    """Minor radius of the plasma."""

    Rmajor_p: float
    """Major radius of the plasma."""

    volume: typing.Annotated[float, pydantic.Field(alias="volume_p")]
    """Plasma volume."""

    fsqr: float
    """Invariant force residual of the force on ``R`` at end of the run."""

    fsqz: float
    """Invariant force residual of the force on ``Z`` at end of the run."""

    fsql: float
    """Invariant force residual of the force on ``lambda`` at end of the run."""

    ftolv: float
    """Force tolerance value used to determine convergence."""

    # Default initialized so reading stays backwards compatible pre v0.4.0
    itfsq: int = 0
    """Number of force-balance iterations after which the run terminated."""

    phipf: jt.Float[np.ndarray, "n_surfaces"]
    """Radial derivative of enclosed toroidal magnetic flux ``phi'`` on the full-
    grid."""

    # Defaulted for backwards compatibility with old wout files
    chipf: jt.Float[np.ndarray, "n_surfaces"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Radial derivative of enclosed poloidal magnetic flux ``chi'`` on the full-
    grid."""

    jcuru: jt.Float[np.ndarray, "n_surfaces"]
    """Radial derivative of enclosed poloidal current on full-grid."""

    jcurv: jt.Float[np.ndarray, "n_surfaces"]
    """Radial derivative of enclosed toroidal current on full-grid."""

    # Default initialized so reading stays backwards compatible pre v0.4.0
    fsqt: jt.Float[np.ndarray, "time"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Evolution of the total force residual along the run.

    This is the sum of ``force_residual_r``, ``force_residual_z``, and ``force_residual_lambda``.
    """

    force_residual_r: jt.Float[np.ndarray, "time"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Evolution of the r radial force residual along the run."""

    force_residual_z: jt.Float[np.ndarray, "time"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Evolution of the z vertical force residual along the run."""

    force_residual_lambda: jt.Float[np.ndarray, "time"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Evolution of the lambda force residual along the run."""

    delbsq: jt.Float[np.ndarray, "time"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Evolution of the force residual at the vacuum boundary along the run."""

    restart_reason_timetrace: typing.Annotated[
        jt.Int[np.ndarray, "time"],
        pydantic.Field(alias="restart_reasons"),
        pydantic.BeforeValidator(lambda x: np.array(x).astype(np.int64)),
    ] = pydantic.Field(default_factory=lambda: np.array([], dtype=np.int64))
    """Internal restart reasons at each step along the run.  (debugging quantity).

    Use the ``restart_reasons`` field to access a more readable enum version of this
    instead of integer status codes.
    """

    wdot: jt.Float[np.ndarray, "time"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    """Evolution of the MHD energy decay along the run."""

    jdotb: jt.Float[np.ndarray, "n_surfaces"]
    r"""Flux-surface-averaged :math:`\langle j \cdot B \rangle` on full-grid."""

    bdotb: jt.Float[np.ndarray, "n_surfaces"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    r"""Flux-surface-averaged :math:`\langle B \cdot B \rangle` on full-grid."""

    bdotgradv: jt.Float[np.ndarray, "n_surfaces"]
    r"""Flux-surface-averaged toroidal magnetic field component :math:`B \cdot \nabla v`
    on full-grid."""

    DMerc: jt.Float[np.ndarray, "n_surfaces"]
    """Full Mercier stability criterion on the full-grid."""

    equif: jt.Float[np.ndarray, "n_surfaces"]
    """Radial force balance residual on full-grid."""

    # In wout these are stored as float64, although they only take integer values.
    xm: SerializeIntAsFloat[jt.Int[np.ndarray, "mn_mode"]]
    """Poloidal mode numbers ``m`` for the Fourier coefficients in the state vector."""

    xn: SerializeIntAsFloat[jt.Int[np.ndarray, "mn_mode"]]
    """Toroidal mode numbers times number of toroidal field periods ``n * nfp`` for the
    Fourier coefficients in the state vector."""

    xm_nyq: SerializeIntAsFloat[jt.Int[np.ndarray, "mn_mode_nyq"]]
    """Poloidal mode numbers ``m`` for the Fourier coefficients in the Nyquist-
    quantities."""

    xn_nyq: SerializeIntAsFloat[jt.Int[np.ndarray, "mn_mode_nyq"]]
    """Toroidal mode numbers times number of toroidal field periods ``n * nfp`` for the
    Fourier coefficients in the Nyquist-quantities."""

    mass: jt.Float[np.ndarray, "n_surfaces"]
    """Plasma mass profile ``m`` on half-grid."""

    buco: jt.Float[np.ndarray, "n_surfaces"]
    """Profile of enclosed toroidal current ``I`` on half-grid."""

    bvco: jt.Float[np.ndarray, "n_surfaces"]
    """Profile of enclosed poloidal ribbon current ``G`` on half-grid."""

    phips: jt.Float[np.ndarray, "n_surfaces"]
    """Radial derivative of enclosed toroidal magnetic flux ``phi'`` on the half-
    grid."""

    bmnc: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"]
    """Fourier coefficients (cos) of the magnetic field strength ``|B|`` on the half-
    grid."""

    gmnc: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"]
    r"""Fourier coefficients (cos) of the Jacobian :math:`\sqrt{g}` on the half-grid."""

    bsubumnc: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"]
    r"""Fourier coefficients (cos) of the covariant magnetic field component
    :math:`B_{\theta}` on the half-grid."""

    bsubvmnc: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"]
    r"""Fourier coefficients (cos) of the covariant magnetic field component
    :math:`B_{\phi}` on the half-grid."""

    bsubsmns: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"]
    """Fourier coefficients (sin) of the covariant magnetic field component
    :math:`B_{s}` on the full- grid."""

    bsupumnc: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"]
    r"""Fourier coefficients (cos) of the contravariant magnetic field component
    :math:`B^{\theta}` on the half-grid."""

    bsupvmnc: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"]
    r"""Fourier coefficients (cos) of the contravariant magnetic field component
    :math:`B^{\phi}` on the half-grid."""

    # Defaulted for backwards compatibility with old wout files
    currumnc: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    r"""Fourier coefficients (cos) of :math:`\sqrt{g} J^{\theta}` on the full-grid."""

    # Defaulted for backwards compatibility with old wout files
    currvmnc: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    r"""Fourier coefficients (cos) of :math:`\sqrt{g} J^{\zeta}` on the full-grid."""

    rmnc: jt.Float[np.ndarray, "mn_mode n_surfaces"]
    """Fourier coefficients (cos) for ``R`` of the geometry of the flux surfaces on the
    full- grid."""

    zmns: jt.Float[np.ndarray, "mn_mode n_surfaces"]
    """Fourier coefficients (sin) for ``Z`` of the geometry of the flux surfaces on the
    full- grid."""

    lmns: jt.Float[np.ndarray, "mn_mode n_surfaces"]
    """Fourier coefficients (sin) for ``lambda`` stream function on the half-grid."""

    lmns_full: jt.Float[np.ndarray, "mn_mode n_surfaces"]
    """Fourier coefficients (sin) for ``lambda`` stream function on the full-grid.

    This quantity is VMEC++ specific and required for hot-restart to work properly. We
    store it with the Fortran convention for the order of the dimensions for consistency
    with lmns.
    """

    rmns: jt.Float[np.ndarray, "mn_mode n_surfaces"] | None = None
    """Fourier coefficients (sin) for `R` of the geometry of the flux surfaces on the
    full-grid; non-stellarator-symmetric."""

    zmnc: jt.Float[np.ndarray, "mn_mode n_surfaces"] | None = None
    """Fourier coefficients (cos) for `Z` of the geometry of the flux surfaces on the
    full-grid; non-stellarator-symmetric."""

    lmnc: jt.Float[np.ndarray, "mn_mode n_surfaces"] | None = None
    """Fourier coefficients (cos) for `lambda` stream function on the half-grid; non-
    stellarator-symmetric."""

    lmnc_full: jt.Float[np.ndarray, "mn_mode n_surfaces"] | None = None
    """Fourier coefficients (cos) for `lambda` stream function on the full-grid; non-
    stellarator-symmetric.

    This quantity is VMEC++ specific and required for hot-restart to work properly. We
    store it with the Fortran convention for the order of the dimensions for consistency
    with lmnc. Only populated when lasym=True (non-stellarator-symmetric
    configurations).
    """

    gmns: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] | None = None
    r"""Fourier coefficients (sin) of the Jacobian :math:`\sqrt{g}` on the half-grid;
    non-stellarator-symmetric."""

    bmns: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] | None = None
    """Fourier coefficients (sin) of the magnetic field strength ``|B|`` on the half-
    grid; non-stellarator-symmetric."""

    bsubumns: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] | None = None
    r"""Fourier coefficients (sin) of the covariant magnetic field component
    :math:`B_{\theta}` on the half-grid; non-stellarator-symmetric."""

    bsubvmns: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] | None = None
    r"""Fourier coefficients (sin) of the covariant magnetic field component
    :math:`B_{\phi}` on the half-grid; non-stellarator-symmetric."""

    bsubsmnc: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] | None = None
    """Fourier coefficients (cos) of the covariant magnetic field component
    :math:`B_{s}` on the full- grid; non-stellarator-symmetric."""

    bsupumns: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] | None = None
    r"""Fourier coefficients (sin) of the contravariant magnetic field component
    :math:`B^{\theta}` on the half-grid; non-stellarator-symmetric."""

    bsupvmns: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] | None = None
    r"""Fourier coefficients (sin) of the contravariant magnetic field component
    :math:`B^{\phi}` on the half-grid; non-stellarator-symmetric."""

    currumns: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] | None = None
    r"""Fourier coefficients (sin) of :math:`\sqrt{g} J^{\theta}` on the full-grid; non-
    stellarator-symmetric."""

    currvmns: jt.Float[np.ndarray, "mn_mode_nyq n_surfaces"] | None = None
    r"""Fourier coefficients (sin) of :math:`\sqrt{g} J^{\zeta}` on the full-grid; non-
    stellarator-symmetric."""

    pcurr_type: ProfileType
    """Parametrization of toroidal current profile (copied from input)."""

    pmass_type: ProfileType
    """Parametrization of mass/pressure profile (copied from input)."""

    piota_type: ProfileType
    """Parametrization of iota profile (copied from input)."""

    am: jt.Float[np.ndarray, "_preset"]
    """Mass/pressure profile coefficients (copied from input)."""

    ac: jt.Float[np.ndarray, "_preset"]
    """Enclosed toroidal current profile coefficients (copied from input)."""

    ai: jt.Float[np.ndarray, "_preset"]
    """Iota profile coefficients (copied from input)."""

    am_aux_s: AuxSType[jt.Float[np.ndarray, "_ndfmax"]]
    """Spline mass/pressure profile: knot locations in ``s`` (copied from input)."""

    am_aux_f: AuxFType[jt.Float[np.ndarray, "_ndfmax"]]
    """Spline mass/pressure profile: values at knots (copied from input)."""

    ac_aux_s: AuxSType[jt.Float[np.ndarray, "_ndfmax"]]
    """Spline toroidal current profile: knot locations in ``s`` (copied from input)."""

    ac_aux_f: AuxFType[jt.Float[np.ndarray, "_ndfmax"]]
    """Spline toroidal current profile: values at knots (copied from input)."""

    ai_aux_s: AuxSType[jt.Float[np.ndarray, "_ndfmax"]]
    """Spline iota profile: knot locations in ``s`` (copied from input)."""

    ai_aux_f: AuxFType[jt.Float[np.ndarray, "_ndfmax"]]
    """Spline iota profile: values at knots (copied from input)."""

    gamma: float
    r"""Adiabatic index :math:`\gamma` (copied from input)."""

    mgrid_file: typing.Annotated[str, pydantic.Field(max_length=200)]
    """Full path for vacuum Green's function data (copied from input)."""

    nextcur: int = 0
    """Number of external coil currents."""

    extcur: typing.Annotated[
        jt.Float[np.ndarray, "ext_current"],
        pydantic.BeforeValidator(lambda x: x if np.shape(x) != () else np.array([])),
        pydantic.WrapSerializer(
            lambda x, handler, _: (
                netCDF4.default_fillvals["f8"]
                if isinstance(x, (np.ndarray, list)) and np.shape(x) == (0,)
                else handler(x)
            )
        ),
    ]
    """Coil currents in A.

    for free-boundary runs, ``extcur`` has shape `(nextcur,)`
    for fixed-boundary it is a scalar float `extcur=nan`
    """

    mgrid_mode: MgridModeType
    """Indicates if the mgrid file was normalized to unit currents ("S") or not
    ("R")."""

    iotas: jt.Float[np.ndarray, "n_surfaces"]
    r"""Rotational transform :math:`\iota` on the half-grid."""

    iotaf: jt.Float[np.ndarray, "n_surfaces"]
    r"""Rotational transform :math:`\iota` on the full-grid."""

    betatotal: float
    r"""Total plasma beta.

    The ratio of the total thermal energy of the plasma to the total magnetic energy.

    :math:`\beta = W_{th} / W_B = \int p\, dV / \left( \int B^2 / (2 \mu_0)\, dV \right
    )`
    """

    raxis_cc: jt.Float[np.ndarray, "ntor_plus_1"]
    """Fourier coefficients of :math:`R(phi)` of the magnetic axis geometry."""

    zaxis_cs: jt.Float[np.ndarray, "ntor_plus_1"]
    """Fourier coefficients of :math:`Z(phi)` of the magnetic axis geometry."""

    raxis_cs: jt.Float[np.ndarray, "ntor_plus_1"] | None = None
    """Fourier coefficients of :math:`R(phi)` of the magnetic axis geometry; non-
    stellarator-symmetric."""

    zaxis_cc: jt.Float[np.ndarray, "ntor_plus_1"] | None = None
    """Fourier coefficients of :math:`Z(phi)` of the magnetic axis geometry; non-
    stellarator-symmetric."""

    vp: jt.Float[np.ndarray, "n_surfaces"]
    r"""Differential volume :math:`V' = \frac{\partial V}{\partial s}` on half-grid."""

    presf: jt.Float[np.ndarray, "n_surfaces"]
    """Kinetic pressure ``p`` on the full-grid."""

    pres: jt.Float[np.ndarray, "n_surfaces"]
    """Kinetic pressure ``p`` on the half-grid."""

    phi: jt.Float[np.ndarray, "n_surfaces"]
    r"""Enclosed toroidal magnetic flux :math:`\phi` on the full-grid."""

    signgs: int
    """Sign of the Jacobian of the coordinate transform between flux coordinates and
    cylindrical coordinates."""

    volavgB: float
    """Volume-averaged magnetic field strength."""

    # Defaulted for backwards compatibility with old wout files.
    q_factor: jt.Float[np.ndarray, "n_surfaces"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    r"""Safety factor :math:`q = 1/\iota` on the full-grid."""

    # Defaulted for backwards compatibility with old wout files.
    chi: jt.Float[np.ndarray, "n_surfaces"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    r"""Enclosed poloidal magnetic flux :math:`\chi` on the full-grid."""

    specw: jt.Float[np.ndarray, "n_surfaces"]
    """Spectral width ``M`` on the full-grid."""

    over_r: jt.Float[np.ndarray, "n_surfaces"]
    r"""``<\tau / R> / V'`` on half-grid.

    :math:`\left\langle \frac{\tau}{R} \right\rangle / V'`
    """

    DShear: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier stability criterion contribution due to magnetic shear."""

    DWell: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier stability criterion contribution due to magnetic well."""

    DCurr: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier stability criterion contribution due to plasma currents."""

    DGeod: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier stability criterion contribution due to geodesic curvature."""

    niter: int
    """Number of force-balance iterations taken to converge."""

    beta_vol: jt.Float[np.ndarray, "n_surfaces"]
    """Flux-surface averaged plasma beta on half-grid."""

    version_: float
    """Version number of VMEC, that this VMEC++ wout file is compatible with.

    Some codes change how they interpret values in the wout file depending on this
    number. (E.g. COBRAVMEC checks if >6 or not)
    """

    @property
    def volume_p(self):
        """The attribute is called volume_p in the Fortran wout file, while
        simsopt.mhd.Vmec.wout uses volume.

        We expose both.
        """
        return self.volume

    @property
    def lasym__logical__(self):
        """This is how the attribute is called in the Fortran wout file."""
        return self.lasym

    @property
    def lfreeb__logical__(self):
        """This is how the attribute is called in the Fortran wout file."""
        return self.lfreeb

    @property
    def lrfp__logical__(self):
        """This is how the attribute is called in the Fortran wout file."""
        return self.lrfp

    @property
    def restart_reasons(self) -> list[tuple[int, RestartReason]]:
        """Get the restart reasons as a list of tuples.

        Each tuple contains the iteration number and the reason for the restart.
        """
        return [
            (i, RestartReason(reason))
            for i, reason in enumerate(self.restart_reason_timetrace)
            if reason != 1  # skip the "no restart" reason
        ]

    def save(self, out_path: str | Path) -> None:
        """Save contents in NetCDF3 format, e.g. ``wout.nc``.

        This is the format used by Fortran VMEC implementations and the one expected by
        SIMSOPT.
        """
        out_path = Path(out_path)
        # protect against possible confusion between the C++ WOutFileContents::Save
        # and this method
        if out_path.suffix == ".h5":
            msg = (
                "You called `save` on a VmecWOut object: this produces a NetCDF3 "
                "file, but you specified an output file name ending in '.h5', which "
                "suggests an HDF5 output was expected. Please change output filename "
                "suffix."
            )
            raise ValueError(msg)

        # Write to a temporary file in the target directory and atomically move
        # it into place at the end, so that a failed save never leaves a
        # partial wout file behind at out_path.
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=out_path.parent, prefix=out_path.name + ".", suffix=".tmp"
        )
        os.close(tmp_fd)
        try:
            self._save_to_netcdf3(tmp_name)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        Path(tmp_name).replace(out_path)

    def _save_to_netcdf3(self, out_path: str | Path) -> None:
        """Write the NetCDF3 wout representation of this object to out_path."""
        with netCDF4.Dataset(out_path, "w", format="NETCDF3_CLASSIC") as fnc:
            # create dimensions (in the same order as VMEC2000)
            # Dimensions that are not in use yet, written for compatibility
            fnc.createDimension("mn_mode_pot", 100)
            fnc.createDimension("current_label", 30)
            fnc.createDimension("dim_00006", 6)

            # For some dimension names, we chose a different naming convention,
            # which we consider clearer. Here we translate them to the standard
            # wout equivalents, for compatibility.
            map_dimension_names = {
                "ntor_plus_1": "n_tor",
                "n_surfaces": "radius",
            }

            # Convert VmeWOut to its NetCDF3 compatible representation
            # (wout compatible names and datatypes)
            dumped_fields = self.model_dump(by_alias=True)

            # Make a dictionary of alias names to field info, from
            # model_fields (dictionary of non-alias names)
            alias_field_infos = {
                (
                    field_info.alias if field_info.alias is not None else field
                ): field_info
                for field, field_info in VmecWOut.model_fields.items()
            }
            # jaxtyping does not expose a stable public API for dimension marker
            # types. Older versions expose `_AnonymousDim`, while newer versions
            # expose `_anonymous_dim` (instance). Resolve both for compatibility.
            array_types = getattr(jt, "_array_types", None)
            named_dim_type = (
                getattr(array_types, "_NamedDim", None)
                if array_types is not None
                else None
            )
            anonymous_dim_type = (
                getattr(array_types, "_AnonymousDim", None)
                if array_types is not None
                else None
            )
            if anonymous_dim_type is None and array_types is not None:
                anonymous_dim_instance = getattr(array_types, "_anonymous_dim", None)
                if anonymous_dim_instance is not None:
                    anonymous_dim_type = type(anonymous_dim_instance)

            # Operates under the assumption that the order of the fields in
            # model_fields and model_dump are the same.
            for field, value in dumped_fields.items():
                field_type = type(value)
                # None for extra fields
                field_info = alias_field_infos.get(field)

                if field_type is int:
                    fnc.createVariable(field, np.int32)
                    fnc[field][:] = value
                elif field_type is float:
                    fnc.createVariable(field, np.float64)
                    fnc[field][:] = value
                elif field_type is str:
                    if field_info and len(field_info.metadata) > 0:
                        # Find the max_length metadata for the dimension annotation
                        # TODO(jurasic) this assumes that the first metadata is the
                        # max_length, could be generalized
                        max_len = field_info.metadata[0].max_length
                    else:
                        # No max_length metadata, dynamic length
                        max_len = len(value)
                    dim_name = f"dim_{max_len:05d}"
                    # Create the dimension if it doesn't exist yet
                    if dim_name not in fnc.dimensions:
                        fnc.createDimension(dim_name, (max_len))

                    string_variable = fnc.createVariable(field, "S1", (dim_name,))

                    # Put the string in the format netCDF3 requires. Don't know what to say.
                    padded_value_as_array = np.array(
                        value.encode(encoding="ascii").ljust(max_len)
                    )
                    padded_value_as_netcdf3_compatible_chararray = np.frombuffer(
                        padded_value_as_array, dtype="S1"
                    )
                    string_variable[:] = padded_value_as_netcdf3_compatible_chararray
                elif value is None:
                    # Skip None values (e.g., asymmetric arrays when lasym=False)
                    continue
                elif field_type is np.ndarray or field_type is list:
                    value_array = np.array(value)
                    # Fallback to default dimension names like dim_00001, dim_00002, etc.
                    shape_string = tuple(
                        [f"dim_{dim:05d}" for dim in value_array.shape]
                    )
                    # Asymmetric arrays are annotated as `<array type> | None`;
                    # unwrap such unions to recover the jaxtyping array
                    # annotation that carries the dimension names.
                    annotation = (
                        field_info.annotation if field_info is not None else None
                    )
                    if typing.get_origin(annotation) in (
                        typing.Union,
                        types.UnionType,
                    ):
                        non_none_args = [
                            arg
                            for arg in typing.get_args(annotation)
                            if arg is not type(None)
                        ]
                        annotation = (
                            non_none_args[0] if len(non_none_args) == 1 else None
                        )
                    if (
                        annotation is not None
                        and isinstance(annotation, type)
                        and issubclass(annotation, jt.AbstractArray)
                    ):
                        # Extract the dimension names used for NetCDF wout when available
                        annotation_dim_names = annotation.dim_str.split()
                        inferred_shape: list[str] = []
                        for dim, dim_default_name, annotation_dim_name in zip(
                            annotation.dims,
                            shape_string,
                            annotation_dim_names,
                            strict=True,
                        ):
                            dim_name: str | None = None
                            if named_dim_type is not None and isinstance(
                                dim, named_dim_type
                            ):
                                dim_name = str(dim.name).lstrip("_")
                            elif (
                                anonymous_dim_type is not None
                                and isinstance(dim, anonymous_dim_type)
                                and annotation_dim_name.startswith("_")
                            ):
                                dim_name = annotation_dim_name.lstrip("_")
                            inferred_shape.append(
                                map_dimension_names.get(dim_name, dim_name)
                                if dim_name is not None
                                else dim_default_name
                            )
                        shape_string = tuple(inferred_shape)

                    for dim_name, dim_size in zip(
                        shape_string, value_array.shape, strict=True
                    ):
                        if dim_name not in fnc.dimensions:
                            fnc.createDimension(dim_name, dim_size)

                    dtype = value_array.dtype
                    if np.issubdtype(dtype, np.integer):
                        # wout format uses 32 bit integers, Python uses 64 bit by default
                        dtype = np.int32

                    if len(shape_string) == 0:
                        # Scalar value, no dimensions
                        fnc.createVariable(field, dtype)
                        fnc[field][:] = value_array
                    elif len(shape_string) == 1:
                        fnc.createVariable(field, dtype, shape_string)
                        # Slice arrays that are padded in wout and unpadded in VMEC++
                        fnc[field][: len(value_array)] = value_array
                    elif len(shape_string) == 2:
                        # 2D arrays are transposed in Fortran, also reverse the dimension order
                        fnc.createVariable(field, dtype, shape_string[::-1])
                        fnc[field][:] = value_array.T
                    else:
                        msg = f"Field {field} has an unsupported shape: {shape_string}"
                        raise ValueError(msg)
                else:
                    msg = (
                        f"Field {field} has an unsupported type: {field_type}. "
                        "Please report this to the developers."
                    )
                    raise ValueError(msg)

    @staticmethod
    def _from_cpp_wout(cpp_wout: _vmecpp.VmecppWOut) -> VmecWOut:
        attrs = {}

        # These attributes are the same in VMEC++ and in Fortran VMEC
        for field in VmecWOut.model_fields:
            if field not in VmecWOut._CPP_WOUT_SPECIAL_HANDLING:
                attrs[field] = getattr(cpp_wout, field)

        # Asymmetric attributes are only populated when lasym=True
        # All of them are defaulted to None when lasym=False
        if cpp_wout.lasym:
            attrs["raxis_cs"] = cpp_wout.raxis_cs
            attrs["zaxis_cc"] = cpp_wout.zaxis_cc

            # Full-grid asymmetric 2D arrays
            attrs["rmns"] = cpp_wout.rmns
            attrs["zmnc"] = cpp_wout.zmnc
            attrs["lmnc_full"] = cpp_wout.lmnc_full
            attrs["bsubsmnc"] = cpp_wout.bsubsmnc

            # Half-grid asymmetric 2D arrays
            attrs["lmnc"] = cpp_wout.lmnc
            attrs["bmns"] = cpp_wout.bmns
            attrs["bsubumns"] = cpp_wout.bsubumns
            attrs["bsubvmns"] = cpp_wout.bsubvmns
            attrs["bsupumns"] = cpp_wout.bsupumns
            attrs["bsupvmns"] = cpp_wout.bsupvmns
            attrs["currumns"] = cpp_wout.currumns
            attrs["currvmns"] = cpp_wout.currvmns
            attrs["gmns"] = cpp_wout.gmns

        return VmecWOut(**attrs)

    def _to_cpp_wout(self) -> _vmecpp.WOutFileContents:
        cpp_wout = _vmecpp.WOutFileContents()

        # These attributes are the same in VMEC++ and in Fortran VMEC
        for field in VmecWOut.model_fields:
            if field not in VmecWOut._CPP_WOUT_SPECIAL_HANDLING:
                setattr(cpp_wout, field, getattr(self, field))

        # Asymmetric fields (only set when lasym=True)
        if self.lasym:
            cpp_wout.raxis_cs = self.raxis_cs
            cpp_wout.zaxis_cc = self.zaxis_cc

            for field in [
                "rmns",
                "zmnc",
                "lmnc_full",
                "bsubsmnc",
                "lmnc",
                "gmns",
                "bmns",
                "bsubumns",
                "bsubvmns",
                "bsupumns",
                "bsupvmns",
                "currumns",
                "currvmns",
            ]:
                value = getattr(self, field)
                if value is not None:
                    setattr(cpp_wout, field, value)

        return cpp_wout

    @staticmethod
    def from_wout_file(wout_filename: str | Path) -> VmecWOut:
        """Load wout contents in NetCDF format.

        This is the format used by Fortran VMEC implementations and the one expected by
        SIMSOPT. We allow for additional attributes to be present in the file, for
        compatibility with wouf files from other VMEC versions, but require at least the
        fields produced by VMEC++.
        """
        with netCDF4.Dataset(wout_filename, "r") as fnc:
            fnc.set_auto_mask(False)
            attrs = {}
            for var_name, variable in fnc.variables.items():
                if variable.dtype is str or variable.dtype == "S1":
                    raw_bytes = fnc[var_name][()].tobytes()
                    try:
                        # Remove both zero-padding and whitespaces.
                        attrs[var_name] = (
                            raw_bytes.decode("ascii").strip("\x00").strip()
                        )
                    except UnicodeDecodeError:
                        logger.warning(
                            "Could not decode variable '%s' as ascii text; "
                            "replacing it with an empty string.",
                            var_name,
                        )
                        attrs[var_name] = ""
                elif variable.ndim == 2:
                    # We transpose the 2D arrays to map from
                    # Column-major convention (Fortran) to Row-major (Python, C++)
                    attrs[var_name] = np.transpose(fnc[var_name][()])
                else:
                    attrs[var_name] = fnc[var_name][()]

        # Special handling for variables only present in VMEC++
        # For now, only special case for lambda coefficients: lambda = 0 is a physically meaningful fall-back value
        mnmax = attrs["mnmax"]
        ns = attrs["ns"]
        attrs.setdefault("lmns_full", np.zeros([mnmax, ns]))
        if attrs["lasym__logical__"]:
            attrs.setdefault("lmnc_full", np.zeros([mnmax, ns]))

        # Backwards compatibility: lrfp flag may not exist in older wout files
        attrs.setdefault("lrfp__logical__", 0)

        # Backwards compatibility for very old wout files
        if attrs["version_"] <= 8.0:
            attrs.setdefault("fsqr", np.nan)
            attrs.setdefault("fsqz", np.nan)
            attrs.setdefault("fsql", np.nan)
            attrs.setdefault("ftolv", np.nan)
            attrs.setdefault("pcurr_type", "UNKNOWN")
            attrs.setdefault("pmass_type", "UNKNOWN")
            attrs.setdefault("piota_type", "UNKNOWN")
            attrs.setdefault("am", np.array([]))
            attrs.setdefault("ac", np.array([]))
            attrs.setdefault("ai", np.array([]))
            attrs.setdefault("am_aux_s", np.array([]))
            attrs.setdefault("am_aux_f", np.array([]))
            attrs.setdefault("ac_aux_s", np.array([]))
            attrs.setdefault("ac_aux_f", np.array([]))
            attrs.setdefault("ai_aux_s", np.array([]))
            attrs.setdefault("ai_aux_f", np.array([]))
        return VmecWOut.model_validate(attrs, by_alias=True)


class Threed1Volumetrics(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    model_config = pydantic.ConfigDict(extra="forbid")

    int_p: float
    """Total plasma pressure integrated over the plasma volume."""

    avg_p: float
    """Volume-averaged plasma pressure."""

    int_bpol: float
    """Total poloidal magnetic field energy `B_phi^2/(2 mu0)` integrated over the plasma
    volume."""

    avg_bpol: float
    """Volume-averaged poloidal magnetic field energy."""

    int_btor: float
    """Total toroidal magnetic field energy integrated over the plasma volume."""

    avg_btor: float
    """Volume-averaged toroidal magnetic field energy."""

    int_modb: float
    """Total `|B|` integrated over the plasma volume."""

    avg_modb: float
    """Volume-averaged `|B|`."""

    int_ekin: float
    """Total kinetic energy integrated over the plasma volume."""

    avg_ekin: float
    """Volume-averaged kinetic energy."""

    @staticmethod
    def _from_cpp_threed1volumetrics(
        cpp_threed1volumetrics: _vmecpp.Threed1Volumetrics,
    ) -> Threed1Volumetrics:
        threed1volumetrics = Threed1Volumetrics(
            **{
                attr: getattr(cpp_threed1volumetrics, attr)
                for attr in Threed1Volumetrics.model_fields
            }
        )

        return threed1volumetrics


class Threed1FirstTable(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    """Python equivalent of the first table in VMEC's "threed1" file.

    Radial profiles on the full grid: flux-surface label, radial force balance,
    currents, pressure, and parallel-current diagnostics.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    s: jt.Float[np.ndarray, "num_full"]
    """Normalized toroidal flux `s` on the full grid."""

    radial_force: jt.Float[np.ndarray, "num_full"]
    """Radial force-balance residual on the full grid."""

    toroidal_flux: jt.Float[np.ndarray, "num_full"]
    """Toroidal flux profile on the full grid."""

    iota: jt.Float[np.ndarray, "num_full"]
    """Rotational transform profile on the full grid."""

    avg_jsupu: jt.Float[np.ndarray, "num_full"]
    """Surface-averaged poloidal current density `<JSUPU>` on the full grid."""

    avg_jsupv: jt.Float[np.ndarray, "num_full"]
    """Surface-averaged toroidal current density `<JSUPV>` on the full grid."""

    d_volume_d_phi: jt.Float[np.ndarray, "num_full"]
    """Differential volume `d(VOL)/d(PHI)` on the full grid."""

    d_pressure_d_phi: jt.Float[np.ndarray, "num_full"]
    """Radial derivative of pressure `d(PRES)/d(PHI)` on the full grid."""

    spectral_width: jt.Float[np.ndarray, "num_full"]
    """Surface-averaged spectral width `<M>` on the full grid."""

    pressure: jt.Float[np.ndarray, "num_full"]
    """Pressure `PRESF` on the full grid in Pa (without mu_0)."""

    buco_full: jt.Float[np.ndarray, "num_full"]
    """Enclosed toroidal current `<BSUBU>` on the full grid."""

    bvco_full: jt.Float[np.ndarray, "num_full"]
    """Enclosed poloidal current `<BSUBV>` on the full grid."""

    j_dot_b: jt.Float[np.ndarray, "num_full"]
    """Parallel current density `<J.B>` on the full grid."""

    b_dot_b: jt.Float[np.ndarray, "num_full"]
    """`<|B|^2>` on the full grid."""

    @staticmethod
    def _from_cpp_threed1_first_table(
        cpp_threed1_first_table: _vmecpp.Threed1FirstTable,
    ) -> Threed1FirstTable:
        return Threed1FirstTable(
            **{
                attr: getattr(cpp_threed1_first_table, attr)
                for attr in Threed1FirstTable.model_fields
            }
        )


class Threed1GeometricAndMagneticQuantities(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    """Python equivalent of the geometric and magnetic quantities in VMEC's "threed1"
    file.

    Global geometry, magnetic-field limits, aspect ratio, beta values, currents, and
    radial geometric profiles.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    toroidal_flux: float
    """Total enclosed toroidal flux."""

    circum_p: float
    """Poloidal circumference of the boundary cross-section."""

    surf_area_p: float
    """Plasma surface area."""

    cross_area_p: float
    """Plasma cross-sectional area."""

    volume_p: float
    """Plasma volume."""

    Rmajor_p: float
    """Major radius."""

    Aminor_p: float
    """Minor radius."""

    aspect: float
    """Aspect ratio."""

    kappa_p: float
    """Elongation."""

    rcen: float
    """Geometric center major radius."""

    aminr1: float
    """Volume-averaged minor radius."""

    pavg: float
    """Volume-averaged pressure."""

    factor: float
    """Normalization factor used in the threed1 computation."""

    b0: float
    """Magnetic field magnitude on the magnetic axis."""

    rmax_surf: float
    """Maximum major radius on the boundary."""

    rmin_surf: float
    """Minimum major radius on the boundary."""

    zmax_surf: float
    """Maximum height on the boundary."""

    bmin: jt.Float[np.ndarray, "num_half nThetaReduced"]
    """Minimum `|B|` per half-grid surface and poloidal angle."""

    bmax: jt.Float[np.ndarray, "num_half nThetaReduced"]
    """Maximum `|B|` per half-grid surface and poloidal angle."""

    waist: jt.Float[np.ndarray, "n_symmetry_planes"]
    """Plasma waist thickness in the phi = 0, pi symmetry planes."""

    height: jt.Float[np.ndarray, "n_symmetry_planes"]
    """Plasma height in the phi = 0, pi symmetry planes."""

    betapol: float
    """Poloidal beta."""

    betatot: float
    """Total beta."""

    betator: float
    """Toroidal beta."""

    VolAvgB: float
    """Volume-averaged magnetic field magnitude."""

    IonLarmor: float
    """Ion Larmor radius estimate."""

    jpar_perp: float
    """Volume-integrated ratio of parallel to perpendicular current."""

    jparPS_perp: float
    """Volume-integrated Pfirsch-Schlueter parallel/perpendicular current ratio."""

    toroidal_current: float
    """Net toroidal current in A."""

    rbtor: float
    """`R * Btor` at the boundary."""

    rbtor0: float
    """`R * Btor` on the magnetic axis."""

    psi: jt.Float[np.ndarray, "num_full"]
    """Poloidal magnetic flux on the full grid."""

    ygeo: jt.Float[np.ndarray, "num_full"]
    """Geometric minor radius profile."""

    yinden: jt.Float[np.ndarray, "num_full"]
    """Geometric indentation profile."""

    yellip: jt.Float[np.ndarray, "num_full"]
    """Geometric ellipticity profile."""

    ytrian: jt.Float[np.ndarray, "num_full"]
    """Geometric triangularity profile."""

    yshift: jt.Float[np.ndarray, "num_full"]
    """Geometric shift measured from the magnetic axis."""

    loc_jpar_perp: jt.Float[np.ndarray, "num_full"]
    """Local parallel/perpendicular current ratio profile."""

    loc_jparPS_perp: jt.Float[np.ndarray, "num_full"]
    """Local Pfirsch-Schlueter parallel/perpendicular current ratio profile."""

    @staticmethod
    def _from_cpp_threed1_geometric_and_magnetic_quantities(
        cpp_threed1_geometric_and_magnetic: _vmecpp.Threed1GeometricAndMagneticQuantities,
    ) -> Threed1GeometricAndMagneticQuantities:
        return Threed1GeometricAndMagneticQuantities(
            **{
                attr: getattr(cpp_threed1_geometric_and_magnetic, attr)
                for attr in Threed1GeometricAndMagneticQuantities.model_fields
            }
        )


class Threed1AxisGeometry(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    """Python equivalent of the magnetic-axis geometry in VMEC's "threed1" file:
    Fourier coefficients of the converged magnetic axis.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    raxis_symm: jt.Float[np.ndarray, "ntor_plus_1"]
    """Stellarator-symmetric axis coefficients `R * cos(n * zeta)`."""

    zaxis_symm: jt.Float[np.ndarray, "ntor_plus_1"]
    """Stellarator-symmetric axis coefficients `Z * sin(n * zeta)`."""

    raxis_asym: jt.Float[np.ndarray, "ntor_plus_1"]
    """Non-stellarator-symmetric axis coefficients `R * sin(n * zeta)`."""

    zaxis_asym: jt.Float[np.ndarray, "ntor_plus_1"]
    """Non-stellarator-symmetric axis coefficients `Z * cos(n * zeta)`."""

    @staticmethod
    def _from_cpp_threed1_axis_geometry(
        cpp_threed1_axis: _vmecpp.Threed1AxisGeometry,
    ) -> Threed1AxisGeometry:
        return Threed1AxisGeometry(
            **{
                attr: getattr(cpp_threed1_axis, attr)
                for attr in Threed1AxisGeometry.model_fields
            }
        )


class Threed1Betas(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    """Python equivalent of the beta values in VMEC's "threed1" file."""

    model_config = pydantic.ConfigDict(extra="forbid")

    betatot: float
    """Total beta."""

    betapol: float
    """Poloidal beta."""

    betator: float
    """Toroidal beta."""

    rbtor: float
    """`R * Btor` (vacuum)."""

    betaxis: float
    """Peak beta on the magnetic axis."""

    betstr: float
    """Beta-star."""

    @staticmethod
    def _from_cpp_threed1_betas(
        cpp_threed1_betas: _vmecpp.Threed1Betas,
    ) -> Threed1Betas:
        return Threed1Betas(
            **{
                attr: getattr(cpp_threed1_betas, attr)
                for attr in Threed1Betas.model_fields
            }
        )


class Threed1ShafranovIntegrals(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    """Python equivalent of the Shafranov surface integrals in VMEC's "threed1" file.

    Ref: S. P. Hirshman, Phys. Fluids B, 5, (1993) 3119.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    scaling_ratio: float
    """Scaling ratio applied to the surface integrals."""

    r_lao: float
    """Lao major radius."""

    f_lao: float
    """Lao form factor."""

    f_geo: float
    """Geometric form factor."""

    smaleli: float
    """Normalized internal inductance `li`."""

    betai: float
    """Internal beta."""

    musubi: float
    """Ratio of volume poloidal-field energy to the surface-integral estimate."""

    # `lambda` is a Python keyword, so the field is named `lambda_`; it maps to
    # the C++ `lambda` member in `_from_cpp_threed1_shafranov_integrals`.
    lambda_: float
    """Shafranov lambda parameter."""

    s11: float
    """Shafranov surface integral S1/2 (Hirshman definition)."""

    s12: float
    """Shafranov surface integral S2/2 (Hirshman definition)."""

    s13: float
    """Shafranov surface integral S3/2 (Lao definition)."""

    s2: float
    """Shafranov integral s2."""

    s3: float
    """Shafranov integral s3."""

    delta1: float
    """Shafranov shift parameter delta1."""

    delta2: float
    """Shafranov shift parameter delta2."""

    delta3: float
    """Shafranov shift parameter delta3."""

    @staticmethod
    def _from_cpp_threed1_shafranov_integrals(
        cpp_threed1_shafranov_integrals: _vmecpp.Threed1ShafranovIntegrals,
    ) -> Threed1ShafranovIntegrals:
        # `lambda_` maps to the C++ member `lambda` (a Python keyword).
        values = {}
        for field_name in Threed1ShafranovIntegrals.model_fields:
            cpp_name = "lambda" if field_name == "lambda_" else field_name
            values[field_name] = getattr(cpp_threed1_shafranov_integrals, cpp_name)
        return Threed1ShafranovIntegrals(**values)


class Mercier(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    model_config = pydantic.ConfigDict(extra="forbid")

    s: jt.Float[np.ndarray, "n_surfaces"]
    """Normalized toroidal flux coordinate `s`."""

    toroidal_flux: jt.Float[np.ndarray, "n_surfaces"]
    """Enclosed toroidal magnetic flux `phi`."""

    iota: jt.Float[np.ndarray, "n_surfaces"]
    """Rotational transform `iota`."""

    shear: jt.Float[np.ndarray, "n_surfaces"]
    """Magnetic shear profile."""

    d_volume_d_s: jt.Float[np.ndarray, "n_surfaces"]
    """Radial derivative of plasma volume with respect to `s`."""

    well: jt.Float[np.ndarray, "n_surfaces"]
    """Magnetic well profile."""

    toroidal_current: jt.Float[np.ndarray, "n_surfaces"]
    """Enclosed toroidal current profile."""

    d_toroidal_current_d_s: jt.Float[np.ndarray, "n_surfaces"]
    """Radial derivative of enclosed toroidal current."""

    pressure: jt.Float[np.ndarray, "n_surfaces"]
    """Pressure profile `p`."""

    d_pressure_d_s: jt.Float[np.ndarray, "n_surfaces"]
    """Radial derivative of pressure profile."""

    DMerc: jt.Float[np.ndarray, "n_surfaces"]
    """Full Mercier stability criterion."""

    Dshear: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier criterion contribution due to magnetic shear."""

    Dwell: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier criterion contribution due to magnetic well."""

    Dcurr: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier criterion contribution due to plasma currents."""

    Dgeod: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier criterion contribution due to geodesic curvature."""

    @staticmethod
    def _from_cpp_mercier(cpp_mercier: _vmecpp.Mercier) -> Mercier:
        mercier = Mercier(
            **{attr: getattr(cpp_mercier, attr) for attr in Mercier.model_fields}
        )

        return mercier


class JxBOut(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    model_config = pydantic.ConfigDict(extra="forbid")

    itheta: jt.Float[np.ndarray, "num_half nZnT"]
    r"""Poloidal surface current.

    :math:`itheta = (\frac{\partial B_s}{\partial \Phi} - \frac{\partial B_\phi}{\partial s}) / \mu_0`
    """

    izeta: jt.Float[np.ndarray, "num_half nZnT"]
    r"""Toroidal surface current.

    :math:`izeta = (-\frac{\partial B_s}{\partial \Theta} + \frac{\partial
    B_\theta}{\partial s}) / \mu_0`
    """

    bdotk: jt.Float[np.ndarray, "num_full nZnT"]

    amaxfor: jt.Float[np.ndarray, "n_surfaces"]
    """100 times the maximum value of the real space force residual on each radial
    surface."""

    aminfor: jt.Float[np.ndarray, "n_surfaces"]
    """100 times the minimum value of the real space force residual on each radial
    surface."""

    avforce: jt.Float[np.ndarray, "n_surfaces"]
    """Average force residual on each radial surface."""

    pprim: jt.Float[np.ndarray, "n_surfaces"]
    """Radial derivative of the pressure profile."""

    jdotb: jt.Float[np.ndarray, "n_surfaces"]
    r"""Flux-surface-averaged :math:`\langle j \cdot B \rangle` on full-grid."""

    bdotb: jt.Float[np.ndarray, "n_surfaces"]
    r"""Flux-surface-averaged :math:`\langle B \cdot B \rangle` on full-grid."""

    bdotgradv: jt.Float[np.ndarray, "n_surfaces"]

    jpar2: jt.Float[np.ndarray, "n_surfaces"]
    r"""Flux-surface-averaged squared parallel current density :math:`\langle j_{||}^2
    \rangle`."""

    jperp2: jt.Float[np.ndarray, "n_surfaces"]
    r"""Flux-surface-averaged squared perpendicular current density :math:`\langle
    j_{\perp}^2 \rangle`."""

    phin: jt.Float[np.ndarray, "n_surfaces"]
    """Normalized, enclosed toroidal flux at each radial surface.

    `phin = toroidal_flux/toroidal_flux[-1]`
    """

    jsupu3: jt.Float[np.ndarray, "num_full nZnT"]
    """Contravariant current density component `j^u` on the full grid.

    :math:`j^u = itheta/V'`
    """

    jsupv3: jt.Float[np.ndarray, "num_full nZnT"]
    """Contravariant current density component `j^v` on the full grid.

    :math:`j^u = izeta/V'`
    """

    jsups3: jt.Float[np.ndarray, "num_half nZnT"]
    r"""Contravariant current density component :math:`j^s` on the half grid.

    :math:`j^s = \frac{\partial B_\theta}{\partial \Phi} - \frac{\partial B_\phi}{\partial \Theta}{\mu_0 V'}`
    """

    bsupu3: jt.Float[np.ndarray, "num_full nZnT"]
    bsupv3: jt.Float[np.ndarray, "num_full nZnT"]
    jcrossb: jt.Float[np.ndarray, "num_full nZnT"]
    r"""Magnitude of :math:`j \times B` at each grid point."""

    jxb_gradp: jt.Float[np.ndarray, "num_full nZnT"]
    r"""Dot product of :math:`j \times B` and :math:`\nabla p` at each grid point."""

    jdotb_sqrtg: jt.Float[np.ndarray, "num_full nZnT"]
    r"""Product of :math:`j \cdot B` and :math:`\sqrt{g}` at each grid point."""

    sqrtg3: jt.Float[np.ndarray, "num_full nZnT"]
    r"""Jacobian determinant :math:`\sqrt{g}` at each grid point."""

    bsubu3: jt.Float[np.ndarray, "num_half nZnT"]
    bsubv3: jt.Float[np.ndarray, "num_half nZnT"]
    bsubs3: jt.Float[np.ndarray, "num_full nZnT"]

    @staticmethod
    def _from_cpp_jxbout(cpp_jxbout: _vmecpp.JxBOutFileContents) -> JxBOut:
        jxbout = JxBOut(
            **{attr: getattr(cpp_jxbout, attr) for attr in JxBOut.model_fields}
        )

        return jxbout


class VmecOutput(BaseModelWithNumpy):
_ArrayType = typing.TypeVar("_ArrayType")
    """Container for the full output of a VMEC run."""

    input: VmecInput
    """The input to the VMEC run that produced this output."""

    jxbout: JxBOut
    """Python equivalent of VMEC's "jxbout" file."""

    mercier: Mercier
    """Python equivalent of VMEC's "mercier" file.

    Contains radial profiles and stability criteria relevant for Mercier stability
    analysis, including the Mercier criterion and its decomposition into shear, well,
    current, and geodesic contributions. Also includes profiles of rotational transform,
    toroidal flux, pressure, and their derivatives.
    """

    threed1_volumetrics: Threed1Volumetrics
    """Python equivalent of VMEC's volumetrics section in the "threed1" file.

    Contains global and flux-surface-averaged quantities such as total and average
    pressure, poloidal and toroidal magnetic field energies, kinetic energy, and related
    integrals. Useful for postprocessing and global equilibrium characterization.
    """

    threed1_first_table: Threed1FirstTable
    """Python equivalent of the first table in VMEC's "threed1" file: radial
    profiles of s, iota, currents, pressure, and force balance."""

    threed1_geometric_magnetic: Threed1GeometricAndMagneticQuantities
    """Python equivalent of the geometric and magnetic quantities in VMEC's
    "threed1" file: global geometry, beta values, currents, and geometric profiles."""

    threed1_axis: Threed1AxisGeometry
    """Python equivalent of the magnetic-axis geometry in VMEC's "threed1" file."""

    threed1_betas: Threed1Betas
    """Python equivalent of the beta values in VMEC's "threed1" file."""

    threed1_shafranov_integrals: Threed1ShafranovIntegrals
    """Python equivalent of the Shafranov surface integrals in VMEC's "threed1" file."""

    wout: VmecWOut
    """Python equivalent of VMEC's "wout" file."""


_progress_tip_shown = False




class Vmec(Optimizable):
    """A SIMSOPT-compatible Python wrapper for VMEC++.

    Based on the original SIMSOPT wrapper for VMEC, see
    https://github.com/hiddenSymmetries/simsopt/blob/master/src/simsopt/mhd/vmec.py.
    """

    _boundary: SurfaceRZFourier
    # Corresponds to the keep_all_files flag passed to __init__:
    # if True, WOutFileContents are saved as a NetCDF3 file compatible
    # with Fortran VMEC.
    _should_save_outputs: bool
    n_pressure: int
    n_current: int
    n_iota: int
    iter: int
    free_boundary: bool
    indata: vmecpp.VmecInput | None
    # non-null if Vmec was initialized from an input file
    input_file: str | None
    # non-null if Vmec was initialized from an output file
    output_file: str | None
    # These are filled:
    # - by __init__ if Vmec is initialized with an output file
    # - by a call to run() and are None before otherwise
    s_full_grid: jt.Float[np.ndarray, "ns"] | None
    ds: float | None
    s_half_grid: jt.Float[np.ndarray, "nshalf"] | None

    # The loaded run results, either of the previous run or when constructing Vmec() from an output file
    wout: vmecpp.VmecWOut | None
    # Whether `run()` is available for this object:
    # depends on whether it has been initialized with an input configuration
    # or an output file.
    runnable: bool
    # False when the currently cached results are valid, True if we need to `run()`
    need_to_run_code: bool
    # Cannot use | None for type annotation, because the @SimsoptRequires makes MpiPartition a function object
    mpi: Optional[MpiPartition]  # pyright: ignore  # noqa: UP045
    verbose: bool

    def __init__(
        self,
        filename: str | Path,
        verbose: bool = True,
        ntheta: int = 50,
        nphi: int = 50,
        range_surface: str = "full torus",
        mpi: Optional[MpiPartition] = None,  # pyright: ignore  # noqa: UP045
        keep_all_files: bool = False,
    ):
        self.verbose = verbose

        if mpi is not None:
            logger.warning(
                "self.mpi is not None: note however that it is unused, "
                "only kept for compatibility with VMEC2000."
            )

        if mpi is None and MPI is not None:
            self.mpi = MPI_PARTITION
        else:
            self.mpi = mpi

        self._should_save_outputs = keep_all_files

        # default values from original SIMSOPT wrapper
        self.n_pressure = 10
        self.n_current = 10
        self.n_iota = 10
        self.wout = None
        self.s_full_grid = None
        self.ds = None
        self.s_half_grid = None

        # NOTE: this behavior is for compatibility with SIMSOPT's VMEC wrapper,
        # which supports initialization from an input.* file or from a wout.*file
        # and sets `self.runnable` depending on this.
        basename = Path(filename).name

        # Original VMEC follows the convention that all input files start with `input`,
        # but VMEC++ does not (see e.g. the contents of vmecpp/test_data).
        if basename.startswith("input") or basename.endswith(".json"):
            with vmecpp.ensure_vmecpp_input(Path(filename)) as vmecpp_filename:
                logger.debug(
                    f"Initializing a VMEC object from input file: {vmecpp_filename}"
                )
                self.indata = vmecpp.VmecInput.from_file(vmecpp_filename)
            assert self.indata is not None  # for pyright

            self.runnable = True
            self.need_to_run_code = True
            # intentionally using the original `filename` and not the potentially
            # different `vmecpp_filename` here: we want to behave as if the input
            # was `filename`, even if internally we converted it.
            self.input_file = str(filename)
            self.iter = -1

            # A vmec object has mpol and ntor attributes independent of
            # the boundary. The boundary surface object is initialized
            # with mpol and ntor values that match those of the vmec
            # object, but the mpol/ntor values of either the vmec object
            # or the boundary surface object can be changed independently
            # by the user.
            mpol, ntor = self._last_mpol_ntor(self.indata)
            mpol_for_surfacerzfourier, ntor_for_surfacerzfourier = (
                self._surface_rzfourier_resolution(mpol, ntor)
            )
            self._boundary = SurfaceRZFourier.from_nphi_ntheta(
                nfp=self.indata.nfp,
                stellsym=not self.indata.lasym,
                mpol=mpol_for_surfacerzfourier,
                ntor=ntor_for_surfacerzfourier,
                ntheta=ntheta,
                nphi=nphi,
                range=range_surface,
            )
            self.free_boundary = bool(self.indata.lfreeb)

            # Transfer boundary shape data from indata to _boundary:
            vi = self.indata
            for m in range(mpol):
                for n in range(2 * ntor + 1):
                    self._boundary.rc[m, n] = vi.rbc[m, n]
                    self._boundary.zs[m, n] = vi.zbs[m, n]
                    if vi.lasym:
                        assert vi.rbs is not None
                        assert vi.zbc is not None
                        self._boundary.rs[m, n] = vi.rbs[m, n]
                        self._boundary.zc[m, n] = vi.zbc[m, n]
            self._boundary.local_full_x = self._boundary.get_dofs()

        elif basename.startswith("wout"):  # from output results
            logger.debug(f"Initializing a VMEC object from wout file: {filename}")
            self.runnable = False
            self._boundary = SurfaceRZFourier.from_wout(
                str(filename), nphi=nphi, ntheta=ntheta, range=range_surface
            )
            self.output_file = str(filename)
            self.load_wout_from_outfile()

        else:  # bad input filename
            msg = (
                f'Invalid filename: "{basename}": '
                'Filename must start with "wout" or "input" or end in "json"'
            )
            raise ValueError(msg)

        # Handle a few variables that are not Parameters:
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ["delt", "tcon0", "phiedge", "curtor", "gamma"]
        Optimizable.__init__(
            self,
            x0=x0,
            fixed=fixed,
            names=names,
            depends_on=[self._boundary],
            external_dof_setter=Vmec.set_dofs,
        )

        if not self.runnable:
            # This next line must come after Optimizable.__init__
            # since that calls recompute_bell()
            self.need_to_run_code = False

    def recompute_bell(self, parent=None) -> None:  # noqa: ARG002
        self.need_to_run_code = True

    def run(
        self,
        restart_from: vmecpp.VmecOutput | None = None,
        max_threads: int | None = 1,
    ) -> None:
        """Run VMEC if ``need_to_run_code`` is ``True``.

        The max_threads argument is not present in SIMSOPT's original implementation as
        it is specific to VMEC++, which will spawn the corresponding number of OpenMP
        threads to parallelize execution. By default max_threads=1, so VMEC++ runs on a
        single thread. To automatically enable all threads, pass max_threads=None

        Most optimization frameworks use multi-process parallelization for finite
        differencing, and we do not want to end up overcommitting the machine with NCPU
        processes running NCPU threads each -- especially when OpenMP is involved, as
        OpenMP threads are generally bad at resource-sharing.
        """
        if not self.need_to_run_code:
            logger.debug("run() called but no need to re-run VMEC.")
            return

        if not self.runnable:
            msg = "Cannot run a Vmec object that was initialized from a wout file."
            raise RuntimeError(msg)

        self.iter += 1
        self.set_indata()  # update self.indata if needed

        assert self.indata is not None  # for pyright

        indata = self.indata
        if restart_from is not None:
            # we are going to perform a hot restart, so we are only going to
            # run the last of the multi-grid steps: adapt indata accordingly
            indata = self.indata.model_copy(deep=True)
            indata.ns_array = indata.ns_array[-1:]  # type: ignore
            indata.ftol_array = indata.ftol_array[-1:]  # type: ignore
            indata.niter_array = indata.niter_array[-1:]  # type: ignore

        logger.debug("Running VMEC++")

        try:
            self.output_quantities = vmecpp.run(
                indata,
                max_threads=max_threads,
                verbose=self.verbose,
                restart_from=restart_from,
            )
            self.wout = self.output_quantities.wout
        except RuntimeError as e:
            msg = f"Error while running VMEC++: {e}"
            raise ObjectiveFailure(msg) from e

        if self._should_save_outputs:
            assert self.input_file is not None
            wout_fname = _make_wout_filename(self.input_file)
            self.wout.save(Path(wout_fname))
            self.output_file = str(wout_fname)

        logger.debug("VMEC++ run complete. Now loading output.")
        self._set_grid()

        logger.debug("Done loading VMEC++ output.")
        self.need_to_run_code = False

    def load_wout_from_outfile(self) -> None:
        """Load data from self.output_file into self.wout."""
        logger.debug(f"Attempting to read file {self.output_file}")
        assert self.output_file is not None
        self.wout = vmecpp.VmecWOut.from_wout_file(self.output_file)

        self._set_grid()

    def _set_grid(self) -> None:
        assert self.wout is not None
        self.s_full_grid = np.linspace(0, 1, self.wout.ns)
        ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * ds
        self.ds = ds

    def aspect(self) -> float:
        """Return the plasma aspect ratio."""
        self.run()
        assert self.wout is not None
        return self.wout.aspect

    def volume(self) -> float:
        """Return the volume inside the VMEC last closed flux surface."""
        self.run()
        assert self.wout is not None
        return self.wout.volume_p

    def iota_axis(self) -> float:
        """Return the rotational transform on axis."""
        self.run()
        assert self.wout is not None
        return self.wout.iotaf[0]

    def iota_edge(self) -> float:
        """Return the rotational transform at the boundary."""
        self.run()
        assert self.wout is not None
        return self.wout.iotaf[-1]

    def mean_iota(self) -> float:
        """Return the mean rotational transform.

        The average is taken over the normalized toroidal flux s.
        """
        self.run()
        assert self.wout is not None
        return cast(float, np.mean(self.wout.iotas[1:]))

    def mean_shear(self) -> float:
        """Return an average magnetic shear, d(iota)/ds, where s is the normalized
        toroidal flux.

        This is computed by fitting the rotational transform to a linear (plus constant)
        function in s. The slope of this fit function is returned.
        """
        self.run()
        assert self.wout is not None
        iota_half = self.wout.iotas[1:]

        # This is set both when running VMEC or when reading a wout file
        assert isinstance(self.s_half_grid, np.ndarray)
        # Fit a linear polynomial:
        poly = np.polynomial.Polynomial.fit(self.s_half_grid, iota_half, deg=1)
        # Return the slope:
        return float(poly.deriv()(0))

    def get_dofs(self) -> np.ndarray:
        if not self.runnable:
            # Use default values from vmec_input (copied from SIMSOPT)
            return np.array([1, 1, 1, 0, 0])
        assert self.indata is not None
        return np.array(
            [
                self.indata.delt,
                self.indata.tcon0,
                self.indata.phiedge,
                self.indata.curtor,
                self.indata.gamma,
            ]
        )

    def set_dofs(self, x: list[float]) -> None:
        if self.runnable:
            assert self.indata is not None
            self.need_to_run_code = True
            self.indata.delt = x[0]
            self.indata.tcon0 = x[1]
            self.indata.phiedge = x[2]
            self.indata.curtor = x[3]
            self.indata.gamma = x[4]

    def vacuum_well(self) -> float:
        """Compute a single number W that summarizes the vacuum magnetic well, given by
        the formula.

        W = (dV/ds(s=0) - dV/ds(s=1)) / (dV/ds(s=0)

        where dVds is the derivative of the flux surface volume with
        respect to the radial coordinate s. Positive values of W are
        favorable for stability to interchange modes. This formula for
        W is motivated by the fact that

        d^2 V / d s^2 < 0

        is favorable for stability. Integrating over s from 0 to 1
        and normalizing gives the above formula for W. Notice that W
        is dimensionless, and it scales as the square of the minor
        radius. To compute dV/ds, we use

        dV/ds = 4 * pi**2 * abs(sqrt(g)_{0,0})

        where sqrt(g) is the Jacobian of (s, theta, phi) coordinates,
        computed by VMEC in the gmnc array, and _{0,0} indicates the
        m=n=0 Fourier component. Since gmnc is reported by VMEC on the
        half mesh, we extrapolate by half of a radial grid point to s
        = 0 and 1.
        """
        self.run()
        assert self.wout is not None

        # gmnc is on the half mesh, so drop the 0th radial entry:
        dVds = 4 * np.pi * np.pi * np.abs(self.wout.gmnc[0, 1:])

        # To get from the half grid to s=0 and s=1, we must
        # extrapolate by 1/2 of a radial grid point:
        dVds_s0 = 1.5 * dVds[0] - 0.5 * dVds[1]
        dVds_s1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]

        well = (dVds_s0 - dVds_s1) / dVds_s0
        return well

    def external_current(self) -> float:
        """Return the total electric current associated with external currents, i.e. the
        current through the "doughnut hole". This number is useful for coil
        optimization, to know what the sum of the coil currents must be.

        Returns:
            float with the total external electric current in Amperes.
        """
        self.run()
        assert self.wout is not None
        bvco = self.wout.bvco[-1] * 1.5 - self.wout.bvco[-2] * 0.5
        mu0 = 4 * np.pi * (1.0e-7)
        # The formula in the next line follows from Ampere's law:
        # \int \vec{B} dot (d\vec{r} / d phi) d phi = mu_0 I.
        return 2 * np.pi * bvco / mu0

    @property
    def boundary(self) -> SurfaceRZFourier:
        return self._boundary

    @boundary.setter
    def boundary(self, boundary: SurfaceRZFourier) -> None:
        if boundary is not self._boundary:
            logging.debug("Replacing surface in boundary setter")
            self.remove_parent(self._boundary)
            self._boundary = boundary
            self.append_parent(boundary)
            self.need_to_run_code = True

    @staticmethod
    def _last_mpol_ntor(indata: vmecpp.VmecInput) -> tuple[int, int]:
        """(indata.mpol, indata.ntor) as plain ints.

        VmecInput.mpol/.ntor may also be a Fourier-resolution continuation schedule (a
        sequence); SIMSOPT's SurfaceRZFourier only has a single resolution, so the last
        (finest, target) entry of the schedule is used.
        """
        mpol = indata.mpol if isinstance(indata.mpol, int) else int(indata.mpol[-1])
        ntor = indata.ntor if isinstance(indata.ntor, int) else int(indata.ntor[-1])
        return mpol, ntor

    @staticmethod
    def _surface_rzfourier_resolution(mpol: int, ntor: int) -> tuple[int, int]:
        # SurfaceRZFourier uses m up to mpol inclusive, unlike VMEC++.
        return mpol - 1, ntor

    @staticmethod
    def _resize_surface_rzfourier(
        surface: SurfaceRZFourier, mpol: int, ntor: int
    ) -> SurfaceRZFourier:
        if surface.mpol == mpol and surface.ntor == ntor:
            return surface

        updated_surface = surface.change_resolution(mpol, ntor)
        return surface if updated_surface is None else updated_surface

    def set_indata(self) -> None:
        """Transfer data from simsopt objects to vmec.indata.

        Presently, this function sets the boundary shape and magnetic axis shape.  In
        the future, the input profiles will be set here as well. This data transfer is
        performed before writing a Vmec input file or running Vmec.
        """
        if not self.runnable:
            msg = "Cannot access indata for a Vmec object that was initialized from a wout file."
            raise RuntimeError(msg)
        assert self.indata is not None
        vi = self.indata  # Shorthand
        mpol, ntor = self._last_mpol_ntor(self.indata)
        target_mpol, target_ntor = self._surface_rzfourier_resolution(mpol, ntor)
        boundary_RZFourier = self._resize_surface_rzfourier(
            self.boundary.to_RZFourier().copy(),
            target_mpol,
            target_ntor,
        )
        vi.rbc.fill(0.0)
        vi.zbs.fill(0.0)
        rbs = None
        zbc = None
        if vi.lasym:
            assert vi.rbs is not None
            assert vi.zbc is not None
            rbs = vi.rbs
            zbc = vi.zbc
            rbs.fill(0.0)
            zbc.fill(0.0)

        # Transfer boundary shape data from the surface object to VMEC:
        for m in range(mpol):
            for n in range(2 * ntor + 1):
                vi.rbc[m, n] = boundary_RZFourier.get_rc(m, n - ntor)
                vi.zbs[m, n] = boundary_RZFourier.get_zs(m, n - ntor)
                if rbs is not None and zbc is not None:
                    rbs[m, n] = boundary_RZFourier.get_rs(m, n - ntor)
                    zbc[m, n] = boundary_RZFourier.get_zc(m, n - ntor)

        # NOTE: The following comment is from VMEC2000.
        # Set axis shape to something that is obviously wrong (R=0) to
        # trigger vmec's internal guess_axis.f to run. Otherwise the
        # initial axis shape for run N will be the final axis shape
        # from run N-1, which makes VMEC results depend slightly on
        # the history of previous evaluations, confusing the finite
        # differencing.
        vi.raxis_c.fill(0.0)
        vi.zaxis_s.fill(0.0)

        if vi.lasym:
            assert vi.raxis_s is not None
            assert vi.zaxis_c is not None
            vi.raxis_s.fill(0.0)
            vi.zaxis_c.fill(0.0)

        # TODO(eguiraud): Starfinder does not use profiles yet
        # Set profiles, if they are not None
        # self.set_profile("pressure", "mass", "m")
        # self.set_profile("current", "curr", "c")
        # self.set_profile("iota", "iota", "i")
        # if self.pressure_profile is not None:
        #     vi.pres_scale = 1.0
        # if self.current_profile is not None:
        #     integral, _ = quad(self.current_profile, 0, 1)
        #     vi.curtor = integral

    def get_input(self) -> str:
        """Generate a VMEC++ input file (in JSON format).

        The JSON data will be returned as a string. To save a file, see
        the ``write_input()`` function.
        """
        self.set_indata()
        assert self.indata is not None
        return self.indata.model_dump_json()

    def write_input(self, filename: str | Path) -> None:
        """Write a VMEC++ input file (in JSON format).

        To just get the result as a string without saving a file, see
        the ``get_input()`` function.
        """
        indata_json = self.get_input()
        filename = Path(filename)
        filename.write_text(indata_json)

    def set_mpol_ntor(self, new_mpol: int, new_ntor: int):
        assert self.indata is not None
        # Converting to and back is a bit unfortunate, but avoids
        # having the resize method both in C++ and Python
        indata_wrapper = self.indata._to_cpp_vmecindata()
        indata_wrapper._set_mpol_ntor(new_mpol, new_ntor)
        self.indata = vmecpp.VmecInput._from_cpp_vmecindata(indata_wrapper)

        mpol_for_surfacerzfourier, ntor_for_surfacerzfourier = (
            self._surface_rzfourier_resolution(new_mpol, new_ntor)
        )
        updated_boundary = self._resize_surface_rzfourier(
            self.boundary, mpol_for_surfacerzfourier, ntor_for_surfacerzfourier
        )
        if updated_boundary is not self.boundary:
            old_boundary = self.boundary
            # Transfer children (other than self) from old boundary to new boundary
            # to preserve the optimization dependency chain.
            # self (Vmec) is excluded here; the boundary setter handles it below.
            old_children = [c() for c in old_boundary._children if c() is not None]
            for child in old_children:
                if child is not self:
                    child.remove_parent(old_boundary)
                    child.append_parent(updated_boundary)
            self.boundary = updated_boundary
        self.recompute_bell()


def _make_wout_filename(input_file: str | Path) -> str:
    # - input.foo -> wout_foo.nc
    # - input.json -> wout_input.nc
    # - foo.json -> wout_foo.nc
    # - input.foo.json -> wout_foo.nc
    # - input.foo.bar.json -> wout_foo.bar.nc
    input_file_basename = Path(input_file).name
    if input_file_basename.startswith("input.") and input_file_basename.endswith(
        ".json"
    ):
        out = ".".join(input_file_basename.split(".")[1:-1])
    elif input_file_basename.endswith(".json"):
        out = input_file_basename.removesuffix(".json")
    elif input_file_basename.startswith("input."):
        out = input_file_basename.removeprefix("input.")
    else:
        msg = f"Input file name {input_file} cannot be converted to output file name"
        raise RuntimeError(msg)

    return f"wout_{out}.nc"

# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import contextlib
import enum
import json
import logging
import os
import tempfile
import typing
from collections.abc import Generator
from pathlib import Path

import jaxtyping as jt
import netCDF4
import numpy as np
import pydantic

from vmecpp import _util
from vmecpp._free_boundary import (
    MagneticFieldResponseTable,
    MakegridParameters,
)
from vmecpp._pydantic_numpy import BaseModelWithNumpy
from vmecpp.cpp import _vmecpp  # type: ignore # bindings to the C++ core

logger = logging.getLogger(__name__)


_ArrayType = typing.TypeVar("_ArrayType")


SerializableSparseCoefficientArray: typing.TypeAlias = typing.Annotated[
    _ArrayType,
    pydantic.PlainSerializer(
        _util.dense_to_sparse_coefficients, when_used="unless-none"
    ),
    pydantic.BeforeValidator(_util.sparse_to_dense_coefficients_implicit),
]
SerializeIntAsFloat: typing.TypeAlias = typing.Annotated[
    _ArrayType,
    pydantic.PlainSerializer(lambda x: np.array(x).astype(np.float64).tolist()),
    pydantic.BeforeValidator(lambda x: np.array(x).astype(np.int64)),
]

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


class RestartReason(enum.Enum):
    BAD_JACOBIAN = 2
    """Irst == 2, bad Jacobian, flux surfaces are overlapping."""

    BAD_PROGRESS = 3
    """Irst == 3, bad progress, residuals not decaying as expected."""

    HUGE_INITIAL_FORCES = 4
    """Irst == 4, huge initial forces, flux surfaces are too close to each other (but
    not overlapping yet)"""


# This is a pure Python equivalent of VmecINDATAPyWrapper.
# In the future VmecINDATAPyWrapper and the C++ VmecINDATA will merge into one type,
# and this will become a Python wrapper around the one C++ VmecINDATA type.
# This pure Python type could _also_ disappear if we can get proper autocompletion,
# docstring peeking etc. for the one C++ VmecINDATA type bound via pybind11.
class VmecInput(BaseModelWithNumpy):
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

    mpol: int = 6
    """Number of poloidal Fourier harmonics; m = 0, 1, ..., (mpol-1)"""

    ntor: int = 0
    """Number of toroidal Fourier harmonics; n = -ntor, -ntor+1, ..., -1, 0, 1, ...,
    ntor-1, ntor."""

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
    """If true, return the outputs even if VMEC++ did not converge.

    Otherwise a RuntimeError will be raised.
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

        expected_shape = (self.mpol, 2 * self.ntor + 1)
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
                        mpol_new=self.mpol,
                        ntor_new=self.ntor,
                    ),
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
                # `VmecINDATAPyWrapper` populates missing fields with default values, while `VmecInput` doesn't.
                # Therefore we use `VmecINDATAPyWrapper` here to read the user input, before validating the model
                vmecpp_indata_pywrapper = _vmecpp.VmecINDATAPyWrapper.from_file(
                    vmecpp_input_file
                )
        # At this point all required fields are populated with user defined or default values.
        # Passing missing or extra fields to `VmecInput.model_validate` will otherwise raise an error.
        return VmecInput._from_cpp_vmecindatapywrapper(vmecpp_indata_pywrapper)

    @staticmethod
    def _from_cpp_vmecindatapywrapper(
        vmecindatapywrapper: _vmecpp.VmecINDATAPyWrapper,
    ) -> VmecInput:
        # The VmecInput.model_validate() is strict in its data model, all fields need to be present and valid.
        # VmecInput does _not_ have any default values.
        vmec_input_dict = {
            attr_name: getattr(vmecindatapywrapper, attr_name)
            for attr_name in VmecInput.model_fields
        }
        vmec_input_dict["ns_array"] = vmec_input_dict["ns_array"].astype(np.int64)
        vmec_input_dict["niter_array"] = vmec_input_dict["niter_array"].astype(np.int64)

        return VmecInput.model_validate(vmec_input_dict)

    @staticmethod
    def default():
        """Return a ``VmecInput`` with VMEC++ default values."""
        return VmecInput()

    def _to_cpp_vmecindatapywrapper(self) -> _vmecpp.VmecINDATAPyWrapper:
        cpp_indata = _vmecpp.VmecINDATAPyWrapper()

        # these are read-only in VmecINDATAPyWrapper to
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
            if attr in readonly_attrs:
                continue  # these must be set separately
            setattr(cpp_indata, attr, getattr(self, attr))

        # this also resizes the readonly_attrs
        cpp_indata._set_mpol_ntor(self.mpol, self.ntor)
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
        "niter",
        "signgs",
        "betatotal",
        "volavgB",
        "iotaf",
        "q_factor",
        "presf",
        "phi",
        "chi",
        "beta_vol",
        "specw",
        "DShear",
        "DWell",
        "DCurr",
        "DGeod",
        "raxis_cc",
        "zaxis_cs",
        "raxis_cs",
        "zaxis_cc",
        "version_",
        "bvco",
        "buco",
        "vp",
        "volume",
        "pres",
        "mass",
        "phips",
        "over_r",
        "iotas",
        "rmnc",
        "zmns",
        "rmns",
        "zmnc",
        "lmnc",
        "lmnc_full",
        "bsubsmns",
        "lmns_full",
        "lmns",
        "bmnc",
        "bsubumnc",
        "bsubvmnc",
        "bsupumnc",
        "bsupvmnc",
        "gmnc",
        "rmns",
        "zmnc",
        "gmns",
        "bmns",
        "bsubumns",
        "bsubvmns",
        "bsubsmnc",
        "bsupumns",
        "bsupvmns",
        "restart_reason_timetrace",
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
        pydantic.PlainSerializer(
            lambda x: int(x),
        ),
        pydantic.BeforeValidator(
            lambda x: bool(x),
        ),
        pydantic.Field(alias="lasym__logical__"),
    ]
    """Flag indicating non-stellarator-symmetry.

    Non-stellarator symmetric fields are only populated if this is True.
    """

    lfreeb: typing.Annotated[
        bool,
        pydantic.PlainSerializer(
            lambda x: int(x),
        ),
        pydantic.BeforeValidator(
            lambda x: bool(x),
        ),
        pydantic.Field(alias="lfreeb__logical__"),
    ]
    """Flag indicating free-boundary computation."""

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

    pcurr_type: ProfileType
    """Parametrization of toroidal current profile (copied from input)."""

    pmass_type: ProfileType
    """Parametrization of mass/pressure profile (copied from input)."""

    piota_type: ProfileType
    """Parametrization of iota profile (copied from input)."""

    am: jt.Float[np.ndarray, "preset"]
    """Mass/pressure profile coefficients (copied from input)."""

    ac: jt.Float[np.ndarray, "preset"]
    """Enclosed toroidal current profile coefficients (copied from input)."""

    ai: jt.Float[np.ndarray, "preset"]
    """Iota profile coefficients (copied from input)."""

    am_aux_s: AuxSType[jt.Float[np.ndarray, "ndfmax"]]
    """Spline mass/pressure profile: knot locations in ``s`` (copied from input)."""

    am_aux_f: AuxFType[jt.Float[np.ndarray, "ndfmax"]]
    """Spline mass/pressure profile: values at knots (copied from input)."""

    ac_aux_s: AuxSType[jt.Float[np.ndarray, "ndfmax"]]
    """Spline toroidal current profile: knot locations in ``s`` (copied from input)."""

    ac_aux_f: AuxFType[jt.Float[np.ndarray, "ndfmax"]]
    """Spline toroidal current profile: values at knots (copied from input)."""

    ai_aux_s: AuxSType[jt.Float[np.ndarray, "ndfmax"]]
    """Spline iota profile: knot locations in ``s`` (copied from input)."""

    ai_aux_f: AuxFType[jt.Float[np.ndarray, "ndfmax"]]
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
        pydantic.PlainSerializer(
            lambda x: x if np.shape(x) != (0,) else netCDF4.default_fillvals["f8"]
        ),
    ]
    """Coil currents in A.

    for free-boundary runs, ``extcur`` has shape `(nextcur,)`
    for fixed-boundary it is a scalar float `extcur=nan`
    """

    mgrid_mode: MgridModeType
    """Indicates if the mgrid file was normalized to unit currents ("S") or not
    ("R")."""

    # In the C++ WOutFileContents this is called iota_half.
    iotas: jt.Float[np.ndarray, "n_surfaces"]
    r"""Rotational transform :math:`\iota` on the half-grid."""

    # In the C++ WOutFileContents this is called iota_full.
    iotaf: jt.Float[np.ndarray, "n_surfaces"]
    r"""Rotational transform :math:`\iota` on the full-grid."""

    # In the C++ WOutFileContents this is called betatot.
    betatotal: float
    r"""Total plasma beta.

    The ratio of the total thermal energy of the plasma to the total magnetic energy.

    :math:`\beta = W_{th} / W_B = \int p\, dV / \left( \int B^2 / (2 \mu_0)\, dV \right
    )`
    """

    # In the C++ WOutFileContents this is called raxis_c.
    raxis_cc: jt.Float[np.ndarray, "ntor_plus_1"]
    """Fourier coefficients of :math:`R(phi)` of the magnetic axis geometry."""

    # In the C++ WOutFileContents this is called zaxis_s.
    zaxis_cs: jt.Float[np.ndarray, "ntor_plus_1"]
    """Fourier coefficients of :math:`Z(phi)` of the magnetic axis geometry."""

    # In the C++ WOutFileContents this is called raxis_s.
    raxis_cs: jt.Float[np.ndarray, "ntor_plus_1"] | None = None
    """Fourier coefficients of :math:`R(phi)` of the magnetic axis geometry; non-
    stellarator-symmetric."""

    # In the C++ WOutFileContents this is called zaxis_c.
    zaxis_cc: jt.Float[np.ndarray, "ntor_plus_1"] | None = None
    """Fourier coefficients of :math:`Z(phi)` of the magnetic axis geometry; non-
    stellarator-symmetric."""

    # In the C++ WOutFileContents this is called dVds.
    vp: jt.Float[np.ndarray, "n_surfaces"]
    r"""Differential volume :math:`V' = \frac{\partial V}{\partial s}` on half-grid.

    Note: called ``dVds`` in cpp
    """

    # In the C++ WOutFileContents this is called pressure_full.
    presf: jt.Float[np.ndarray, "n_surfaces"]
    """Kinetic pressure ``p`` on the full-grid."""

    # In the C++ WOutFileContents this is called pressure_half.
    pres: jt.Float[np.ndarray, "n_surfaces"]
    """Kinetic pressure ``p`` on the half-grid."""

    # In the C++ WOutFileContents this is called toroidal_flux.
    phi: jt.Float[np.ndarray, "n_surfaces"]
    r"""Enclosed toroidal magnetic flux :math:`\phi` on the full-grid."""

    # In the C++ WOutFileContents this is called sign_of_jacobian.
    signgs: int
    """Sign of the Jacobian of the coordinate transform between flux coordinates and
    cylindrical coordinates."""

    # In the C++ WOutFileContents this is called VolAvgB.
    volavgB: float
    """Volume-averaged magnetic field strength."""

    # In the C++ WOutFileContents this is called safety_factor.
    # Defaulted for backwards compatibility with old wout files.
    q_factor: jt.Float[np.ndarray, "n_surfaces"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    r"""Safety factor :math:`q = 1/\iota` on the full-grid."""

    # In the C++ WOutFileContents this is called poloidal_flux.
    # Defaulted for backwards compatibility with old wout files.
    chi: jt.Float[np.ndarray, "n_surfaces"] = pydantic.Field(
        default_factory=lambda: np.array([])
    )
    r"""Enclosed poloidal magnetic flux :math:`\chi` on the full-grid."""

    # In the C++ WOutFileContents this is called spectral_width.
    specw: jt.Float[np.ndarray, "n_surfaces"]
    """Spectral width ``M`` on the full-grid."""

    # In the C++ WOutFileContents this is called overr.
    over_r: jt.Float[np.ndarray, "n_surfaces"]
    r"""``<\tau / R> / V'`` on half-grid.

    :math:`\left\langle \frac{\tau}{R} \right\rangle / V'`
    """

    # In the C++ WOutFileContents this is called Dshear.
    DShear: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier stability criterion contribution due to magnetic shear."""

    # In the C++ WOutFileContents this is called Dwell.
    DWell: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier stability criterion contribution due to magnetic well."""

    # In the C++ WOutFileContents this is called Dcurr.
    DCurr: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier stability criterion contribution due to plasma currents."""

    # In the C++ WOutFileContents this is called Dgeod.
    DGeod: jt.Float[np.ndarray, "n_surfaces"]
    """Mercier stability criterion contribution due to geodesic curvature."""

    # In the C++ WOutFileContents this is called maximum_iterations.
    niter: int
    """Maximum number of force-balance iterations allowed."""

    # In the C++ WOutFileContents this is called beta.
    beta_vol: jt.Float[np.ndarray, "n_surfaces"]
    """Flux-surface averaged plasma beta on half-grid."""

    # In the C++ WOutFileContents this is called 'version' and it is a string.
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
                    padded_value_as_netcdf3_compatible_chararray = netCDF4.stringtochar(
                        padded_value_as_array
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
                    if (
                        field_info is not None  # is a model field
                        and field_info.annotation is not None  # has an annotation
                        and issubclass(
                            field_info.annotation,
                            jt.AbstractArray,
                        )
                    ):
                        # Extract the dimension names used for NetCDF wout when available
                        shape_string = tuple(
                            [
                                map_dimension_names.get(dim.name, str(dim.name))
                                if isinstance(dim, jt._array_types._NamedDim)
                                else dim_default_name
                                for dim, dim_default_name in zip(
                                    field_info.annotation.dims,
                                    shape_string,
                                    strict=True,
                                )
                            ]
                        )

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

        attrs["volume"] = cpp_wout.volume_p

        # These attributes are called differently
        attrs["niter"] = cpp_wout.maximum_iterations
        attrs["signgs"] = cpp_wout.sign_of_jacobian
        attrs["betatotal"] = cpp_wout.betatot
        attrs["volavgB"] = cpp_wout.VolAvgB
        attrs["iotaf"] = cpp_wout.iota_full
        attrs["q_factor"] = cpp_wout.safety_factor
        attrs["presf"] = cpp_wout.pressure_full
        attrs["phi"] = cpp_wout.toroidal_flux
        attrs["chi"] = cpp_wout.poloidal_flux
        attrs["beta_vol"] = cpp_wout.beta
        attrs["specw"] = cpp_wout.spectral_width
        attrs["DShear"] = cpp_wout.Dshear
        attrs["DWell"] = cpp_wout.Dwell
        attrs["DCurr"] = cpp_wout.Dcurr
        attrs["DGeod"] = cpp_wout.Dgeod
        attrs["raxis_cc"] = cpp_wout.raxis_c
        attrs["zaxis_cs"] = cpp_wout.zaxis_s

        # These attributes have one element more in VMEC2000
        # (i.e. they have size ns instead of ns - 1).
        # VMEC2000 then indexes them as with [1:], so we pad VMEC++'s.
        # And they might be called differently.
        attrs["bvco"] = np.concatenate(([0.0], cpp_wout.bvco))
        attrs["buco"] = np.concatenate(([0.0], cpp_wout.buco))
        attrs["vp"] = np.concatenate(([0.0], cpp_wout.dVds))
        attrs["pres"] = np.concatenate(([0.0], cpp_wout.pressure_half))
        attrs["mass"] = np.concatenate(([0.0], cpp_wout.mass))
        attrs["beta_vol"] = np.concatenate(([0.0], cpp_wout.beta))
        attrs["phips"] = np.concatenate(([0.0], cpp_wout.phips))
        attrs["over_r"] = np.concatenate(([0.0], cpp_wout.overr))
        attrs["iotas"] = np.concatenate(([0.0], cpp_wout.iota_half))

        # These attributes are transposed in SIMSOPT/Fortran VMEC
        attrs["rmnc"] = cpp_wout.rmnc.T
        attrs["zmns"] = cpp_wout.zmns.T
        attrs["bsubsmns"] = cpp_wout.bsubsmns.T

        # This is a VMEC++-only quantity but it's transposed when
        # stored in a wout file for consistency with lmns.
        attrs["lmns_full"] = cpp_wout.lmns_full.T

        # Asymmetric attributes are transposed and only populated when lasym=True
        # All of them are defaulted to None when lasym=False
        if cpp_wout.lasym:
            attrs["raxis_cs"] = cpp_wout.raxis_s
            attrs["zaxis_cc"] = cpp_wout.zaxis_c

            attrs["bsubsmnc"] = cpp_wout.bsubsmnc.T
            attrs["rmns"] = cpp_wout.rmns.T
            attrs["zmnc"] = cpp_wout.zmnc.T
            attrs["lmnc_full"] = cpp_wout.lmnc_full.T

            attrs["lmnc"] = _pad_and_transpose(cpp_wout.lmnc, attrs["mnmax"])

            attrs["bmns"] = _pad_and_transpose(cpp_wout.bmns, attrs["mnmax_nyq"])
            attrs["bsubumns"] = _pad_and_transpose(
                cpp_wout.bsubumns, attrs["mnmax_nyq"]
            )
            attrs["bsubvmns"] = _pad_and_transpose(
                cpp_wout.bsubvmns, attrs["mnmax_nyq"]
            )
            attrs["bsupumns"] = _pad_and_transpose(
                cpp_wout.bsupumns, attrs["mnmax_nyq"]
            )
            attrs["bsupvmns"] = _pad_and_transpose(
                cpp_wout.bsupvmns, attrs["mnmax_nyq"]
            )
            attrs["gmns"] = _pad_and_transpose(cpp_wout.gmns, attrs["mnmax_nyq"])

        # These attributes have one column less and their elements are transposed
        # in VMEC++ with respect to SIMSOPT/VMEC2000
        attrs["lmns"] = _pad_and_transpose(cpp_wout.lmns, attrs["mnmax"])

        attrs["bmnc"] = _pad_and_transpose(cpp_wout.bmnc, attrs["mnmax_nyq"])
        attrs["bsubumnc"] = _pad_and_transpose(cpp_wout.bsubumnc, attrs["mnmax_nyq"])
        attrs["bsubvmnc"] = _pad_and_transpose(cpp_wout.bsubvmnc, attrs["mnmax_nyq"])
        attrs["bsupumnc"] = _pad_and_transpose(cpp_wout.bsupumnc, attrs["mnmax_nyq"])
        attrs["bsupvmnc"] = _pad_and_transpose(cpp_wout.bsupvmnc, attrs["mnmax_nyq"])
        attrs["gmnc"] = _pad_and_transpose(cpp_wout.gmnc, attrs["mnmax_nyq"])

        # These attributes have zero-padding at the end up to a fixed length
        attrs["am"] = _util.right_pad(cpp_wout.am, preset)
        attrs["ac"] = _util.right_pad(cpp_wout.ac, preset)
        attrs["ai"] = _util.right_pad(cpp_wout.ai, preset)
        attrs["am_aux_s"] = _util.right_pad(cpp_wout.am_aux_s, ndfmax, -1.0)
        attrs["am_aux_f"] = _util.right_pad(cpp_wout.am_aux_f, ndfmax)
        attrs["ac_aux_s"] = _util.right_pad(cpp_wout.ac_aux_s, ndfmax, -1.0)
        attrs["ac_aux_f"] = _util.right_pad(cpp_wout.ac_aux_f, ndfmax)
        attrs["ai_aux_s"] = _util.right_pad(cpp_wout.ai_aux_s, ndfmax, -1.0)
        attrs["ai_aux_f"] = _util.right_pad(cpp_wout.ai_aux_f, ndfmax)

        attrs["restart_reason_timetrace"] = cpp_wout.restart_reasons

        attrs["version_"] = float(cpp_wout.version)

        return VmecWOut(**attrs)

    def _to_cpp_wout(self) -> _vmecpp.WOutFileContents:
        cpp_wout = _vmecpp.WOutFileContents()

        # These attributes are the same in VMEC++ and in Fortran VMEC
        for field in VmecWOut.model_fields:
            if field not in VmecWOut._CPP_WOUT_SPECIAL_HANDLING:
                setattr(cpp_wout, field, getattr(self, field))

        # These attributes are called differently
        cpp_wout.volume_p = self.volume
        cpp_wout.maximum_iterations = self.niter
        cpp_wout.sign_of_jacobian = self.signgs
        cpp_wout.betatot = self.betatotal
        cpp_wout.VolAvgB = self.volavgB
        cpp_wout.iota_full = self.iotaf
        cpp_wout.safety_factor = self.q_factor
        cpp_wout.pressure_full = self.presf
        cpp_wout.toroidal_flux = self.phi
        cpp_wout.poloidal_flux = self.chi
        cpp_wout.beta = self.beta_vol
        cpp_wout.spectral_width = self.specw
        cpp_wout.Dshear = self.DShear
        cpp_wout.Dwell = self.DWell
        cpp_wout.Dcurr = self.DCurr
        cpp_wout.Dgeod = self.DGeod
        cpp_wout.raxis_c = self.raxis_cc
        cpp_wout.zaxis_s = self.zaxis_cs
        cpp_wout.version = str(self.version_)  # also needs a float -> str conversion

        # These attributes have one element more in VMEC2000
        # (i.e. they have size ns instead of ns - 1).
        # VMEC2000 then indexes them as with [1:], so we pad VMEC++'s.
        # And they might be called differently.
        cpp_wout.bvco = self.bvco[1:]
        cpp_wout.buco = self.buco[1:]
        cpp_wout.dVds = self.vp[1:]
        cpp_wout.pressure_half = self.pres[1:]
        cpp_wout.mass = self.mass[1:]
        cpp_wout.beta = self.beta_vol[1:]
        cpp_wout.phips = self.phips[1:]
        cpp_wout.overr = self.over_r[1:]
        cpp_wout.iota_half = self.iotas[1:]

        # These attributes are transposed in SIMSOPT
        cpp_wout.rmnc = self.rmnc.T
        cpp_wout.zmns = self.zmns.T
        cpp_wout.bsubsmns = self.bsubsmns.T

        # This is a VMEC++-only quantity but it's transposed when
        # stored in a wout file for consistency with lmns.
        cpp_wout.lmns_full = self.lmns_full.T

        if self.lasym:
            cpp_wout.raxis_s = self.raxis_cs
            cpp_wout.zaxis_c = self.zaxis_cc
        # Asymmetric attributes are transposed and only set when lasym=True
        for field in [
            "bsubsmnc",
            "rmns",
            "zmnc",
            "lmnc_full",
            "lmnc",
            "bmns",
        ]:
            value = getattr(self, field)
            if value is not None:
                setattr(cpp_wout, field, value.T)

        # This is a VMEC++ only quantity
        cpp_wout.restart_reasons = self.restart_reason_timetrace

        # coefficients on half-grid
        # These attributes have one column less and their elements are transposed
        # in VMEC++ with respect to SIMSOPT/VMEC2000
        for field in [
            "lmns",
            "gmnc",
            "bmnc",
            "bsubumnc",
            "bsubvmnc",
            "bsupumnc",
            "bsupvmnc",
            # Asymmetric coefficients (only when lasym=True)
            "lmnc",
            "gmns",
            "bmns",
            "bsubumns",
            "bsubvmns",
            "bsupumns",
            "bsupvmns",
        ]:
            value = getattr(self, field)
            # Asymmetric coefficients may be None when lasym=False
            if value is not None:
                setattr(cpp_wout, field, value.T[1:, :])

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
                    # Remove both zero-padding and whitespaces.
                    attrs[var_name] = (
                        fnc[var_name][()]
                        .tobytes()
                        .decode("ascii")
                        .strip("\x00")
                        .strip()
                    )
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


class Mercier(BaseModelWithNumpy):
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

    wout: VmecWOut
    """Python equivalent of VMEC's "wout" file."""


def run(
    input: VmecInput,
    magnetic_field: MagneticFieldResponseTable | None = None,
    *,
    max_threads: int | None = None,
    verbose: bool = True,
    restart_from: VmecOutput | None = None,
) -> VmecOutput:
    """Run VMEC++ using the provided input. This is the main entrypoint for both fixed-
    and free-boundary calculations.

    Args:
        input: a VmecInput instance, corresponding to the contents of a classic VMEC input file
        magnetic_field: if present, VMEC++ will pass the magnetic field object in memory instead of reading
            it from an mgrid file (only relevant in free-boundary runs).
        max_threads: maximum number of threads that VMEC++ should spawn. The actual number might still
            be lower that this in case there are too few flux surfaces to keep these many threads
            busy. If None, a number of threads equal to the number of logical cores is used.
        verbose: if True, VMEC++ logs its progress to standard output.
        restart_from: if present, VMEC++ is initialized using the converged equilibrium from the
            provided VmecOutput. This can dramatically decrease the number of iterations to
            convergence when running VMEC++ on a configuration that is very similar to the `restart_from` equilibrium.

    Example:
        >>> import vmecpp
        >>> path = "examples/data/solovev.json"
        >>> vmec_input = vmecpp.VmecInput.from_file(path)
        >>> output = vmecpp.run(vmec_input, verbose=False, max_threads=1)
        >>> round(output.wout.b0, 14) # Exact value may differ by C library
        0.20333137113443
    """
    input = VmecInput.model_validate(input)
    cpp_indata = input._to_cpp_vmecindatapywrapper()

    if restart_from is None:
        initial_state = None
    else:
        initial_state = _vmecpp.HotRestartState(
            wout=restart_from.wout._to_cpp_wout(),
            indata=restart_from.input._to_cpp_vmecindatapywrapper(),
        )

    if max_threads is not None and max_threads <= 0:
        msg = (
            "The number of threads must be >=1. To automatically use all "
            "available threads, pass max_threads=None"
        )
        raise RuntimeError(msg)

    if magnetic_field is None:
        cpp_output_quantities = _vmecpp.run(
            cpp_indata,
            initial_state=initial_state,
            max_threads=max_threads,
            verbose=verbose,
        )
    else:
        # magnetic_response_table takes precedence anyway, but let's be explicit, to ensure
        # we don't silently use the mgrid file in input, instead of the magnetic_response_table object.
        cpp_indata.mgrid_file = "NONE"
        cpp_output_quantities = _vmecpp.run(
            cpp_indata,
            magnetic_response_table=magnetic_field._to_cpp_magnetic_field_response_table(),
            initial_state=initial_state,
            max_threads=max_threads,
            verbose=verbose,
        )

    cpp_wout = cpp_output_quantities.wout
    wout = VmecWOut._from_cpp_wout(cpp_wout)
    jxbout = JxBOut._from_cpp_jxbout(cpp_output_quantities.jxbout)
    mercier = Mercier._from_cpp_mercier(cpp_output_quantities.mercier)
    threed1_volumetrics = Threed1Volumetrics._from_cpp_threed1volumetrics(
        cpp_output_quantities.threed1_volumetrics
    )
    return VmecOutput(
        input=input,
        wout=wout,
        jxbout=jxbout,
        mercier=mercier,
        threed1_volumetrics=threed1_volumetrics,
    )


def is_vmec2000_input(input_file: Path) -> bool:
    """Returns true if the input file looks like a Fortran VMEC/VMEC2000 INDATA file."""
    # we peek at the first few non-blank, non-comment lines in the file:
    # if one of them is "&INDATA", then this is an INDATA file
    with open(input_file) as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("!"):
                continue
            return stripped_line == "&INDATA"
    return False


@contextlib.contextmanager
def ensure_vmecpp_input(input_path: Path) -> Generator[Path, None, None]:
    """If input_path looks like a Fortran INDATA file, convert it to a VMEC++ JSON input
    and return the path to this new JSON file.

    Otherwise assume it is a VMEC++ json input: simply return the input_path unchanged.
    """
    if is_vmec2000_input(input_path):
        logger.debug(
            f"VMEC++ is being run with input file '{input_path}', which looks like "
            "a Fortran INDATA file. It will be converted to a VMEC++ JSON input "
            "on the fly. Please consider permanently converting the input to a "
            " VMEC++ input JSON using the //third_party/indata2json tool."
        )

        # We also add the PID to the output file to ensure that the output file
        # is different for multiple processes that run indata_to_json
        # concurrently on the same input, as it happens e.g. when the SIMSOPT
        # wrapper is run under `mpirun`.
        configuration_name = _util.get_vmec_configuration_name(input_path)
        output_file = input_path.with_name(f"{configuration_name}.{os.getpid()}.json")

        vmecpp_input_path = _util.indata_to_json(
            input_path, output_override=output_file
        )
        assert vmecpp_input_path == output_file.resolve()
        try:
            yield vmecpp_input_path
        finally:
            os.remove(vmecpp_input_path)
    else:
        # if the file is not a VMEC2000 indata file, we assume
        # it is a VMEC++ JSON input file
        yield input_path


@contextlib.contextmanager
def ensure_vmec2000_input(input_path: Path) -> Generator[Path, None, None]:
    """If input_path does not look like a VMEC2000 INDATA file, assume it is a VMEC++
    JSON input file, convert it to VMEC2000's format and return the path to the
    converted file.

    Otherwise simply return the input_path unchanged.

    Given a VMEC++ JSON input file with path 'path/to/[input.]NAME[.json]' the converted
    INDATA file will have path 'some/tmp/dir/input.NAME'.
    A temporary directory is used in order to avoid race conditions when calling this
    function multiple times on the same input concurrently; the `NAME` section of the
    file name is preserved as it is common to have logic that extracts it and re-uses
    it e.g. to decide how related files should be called.
    """

    if is_vmec2000_input(input_path):
        # nothing to do: must yield result on first generator call,
        # then exit (via a return)
        yield input_path
        return

    vmecpp_input_basename = input_path.name.removesuffix(".json").removeprefix("input.")
    indata_file = f"input.{vmecpp_input_basename}"

    with open(input_path) as vmecpp_json_f:
        vmecpp_json_dict = json.load(vmecpp_json_f)

    indata_contents = _util.vmecpp_json_to_indata(vmecpp_json_dict)

    # Otherwise we actually need to perform the JSON -> INDATA conversion.
    # We need the try/finally in order to correctly clean up after
    # ourselves even in case of errors raised from the body of the `with`
    # in user code.
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / indata_file
        with open(out_path, "w") as out_f:
            out_f.write(indata_contents)
        yield out_path


def _pad_and_transpose(
    arr: jt.Float[np.ndarray, "ns_minus_one mn"] | None, mnsize: int
) -> jt.Float[np.ndarray, "mn ns"] | None:
    if arr is None:
        return None
    stacked = np.vstack((np.zeros(mnsize), arr)).T
    assert stacked.shape[1] == arr.shape[0] + 1
    assert stacked.shape[0] == arr.shape[1]
    return stacked


def populate_raw_profile(
    vmec_input: VmecInput,
    field: typing.Literal["pressure", "iota", "current"],
    f: typing.Callable[[np.ndarray], np.ndarray],
) -> None:
    """Populate a line segment profile using callable ``f``.

    The callable is evaluated on all unique ``s`` values required for the
    multi-grid steps (full and half grids). The resulting knots and values are
    stored in the auxiliary arrays for the chosen profile.
    """
    s_values: set[float] = set()
    for ns in vmec_input.ns_array:
        full_grid = np.linspace(0.0, 1.0, ns)
        half_grid = full_grid - 0.5 * (full_grid[1] - full_grid[0])
        s_values.update(full_grid)
        s_values.update(half_grid)
    knots = np.array(np.sort(np.array(list(s_values))))
    values = np.array(f(knots))

    if field == "pressure":
        vmec_input.pmass_type = "line_segment"
        vmec_input.am_aux_s = knots
        vmec_input.am_aux_f = values
        vmec_input.am = np.array([])
    elif field == "iota":
        vmec_input.piota_type = "line_segment"
        vmec_input.ai_aux_s = knots
        vmec_input.ai_aux_f = values
        vmec_input.ai = np.array([])
    elif field == "current":
        vmec_input.pcurr_type = "line_segment_i"
        vmec_input.ac_aux_s = knots
        vmec_input.ac_aux_f = values
        vmec_input.ac = np.array([])
    else:
        msg = "field must be one of 'pressure', 'iota', 'current'"
        raise ValueError(msg)


# Ordered this way to ensure run, VmecInput, and VmecOutput are the first three
# items in the generated documentation.
__all__ = [
    "run",
    "VmecInput",
    "VmecOutput",
    "VmecWOut",
    "JxBOut",
    "Mercier",
    "Threed1Volumetrics",
    "MakegridParameters",
    "MagneticFieldResponseTable",
    "populate_raw_profile",
]

# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import contextlib
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
from vmecpp._pydantic_numpy import BaseModelWithNumpy
from vmecpp.cpp import _vmecpp  # type: ignore # bindings to the C++ core
from vmecpp.free_boundary import (
    MagneticFieldResponseTable,
    MakegridParameters,
)

logger = logging.getLogger(__name__)

SerializableSparseCoefficientArray = typing.Annotated[
    jt.Float[np.ndarray, "mpol two_ntor_plus_one"],
    pydantic.PlainSerializer(
        _util.dense_to_sparse_coefficients, when_used="unless-none"
    ),
    pydantic.BeforeValidator(_util.sparse_to_dense_coefficients_implicit),
]

MgridModeType: typing.TypeAlias = typing.Literal["R", "S", ""]
"""[Scaled, Raw, Unset]"""


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
    `model_validate_json` and `model_dump_json`.
    """

    model_config = pydantic.ConfigDict(
        # serialize NaN and infinite floats as strings in JSON output.
        ser_json_inf_nan="strings",
    )

    lasym: bool
    """Flag to indicate non-stellarator-symmetry.

    Note: this flag is False if stellarator symmetry is present, True if not.
    """

    nfp: int
    """Number of toroidal field periods (=1 for Tokamak)"""

    mpol: int
    """Number of poloidal Fourier harmonics; m = 0, 1, ..., (mpol-1)"""

    ntor: int
    """Number of toroidal Fourier harmonics; n = -ntor, -ntor+1, ..., -1, 0, 1, ...,
    ntor-1, ntor."""

    ntheta: int
    """Number of poloidal grid points; if odd: is rounded to next smaller even
    number."""

    nzeta: int
    """Number of toroidal grid points; must match nzeta of mgrid file if using free-
    boundary."""

    ns_array: jt.Int[np.ndarray, "num_grids"]
    """Number of flux surfaces per multigrid step."""

    ftol_array: jt.Float[np.ndarray, "num_grids"]
    """Requested force tolerance for convergence per multigrid step."""

    niter_array: jt.Int[np.ndarray, "num_grids"]
    """Maximum number of iterations per multigrid step."""

    phiedge: float
    """Total enclosed toroidal magnetic flux in Vs == Wb."""

    ncurr: typing.Literal[0, 1]
    """Select constraint on iota or enclosed toroidal current profiles 0: constrained-iota; 1: constrained-current"""

    pmass_type: str
    """Parametrization of mass/pressure profile."""

    am: jt.Float[np.ndarray, "am_len"]
    """Mass/pressure profile coefficients."""

    am_aux_s: jt.Float[np.ndarray, "am_aux_len"]
    """Spline mass/pressure profile: knot locations in s"""

    am_aux_f: jt.Float[np.ndarray, "am_aux_len"]
    """Spline mass/pressure profile: values at knots"""

    pres_scale: float
    """Global scaling factor for mass/pressure profile."""

    gamma: float
    """Adiabatic index."""

    spres_ped: float
    """Location of pressure pedestal in s."""

    piota_type: str
    """Parametrization of iota profile."""

    ai: jt.Float[np.ndarray, "ai_len"]
    """Iota profile coefficients."""

    ai_aux_s: jt.Float[np.ndarray, "ai_aux_len"]
    """Spline iota profile: knot locations in s"""

    ai_aux_f: jt.Float[np.ndarray, "ai_aux_len"]
    """Spline iota profile: values at knots"""

    pcurr_type: str
    """Parametrization of toroidal current profile."""

    ac: jt.Float[np.ndarray, "ac_len"]
    """Enclosed toroidal current profile coefficients."""

    ac_aux_s: jt.Float[np.ndarray, "ac_aux_len"]
    """Spline toroidal current profile: knot locations in s"""

    ac_aux_f: jt.Float[np.ndarray, "ac_aux_len"]
    """Spline toroidal current profile: values at knots"""

    curtor: float
    """Toroidal current in A."""

    bloat: float
    """Bloating factor (for constrained toroidal current)"""

    lfreeb: bool
    """Flag to indicate free-boundary."""

    mgrid_file: str
    """Full path for vacuum Green's function data."""

    extcur: jt.Float[np.ndarray, "extcur_len"]
    """Coil currents in A."""

    nvacskip: int
    """Number of iterations between full vacuum calculations."""

    nstep: int
    """Printout interval."""

    aphi: jt.Float[np.ndarray, "aphi_len"]
    """Radial flux zoning profile coefficients."""

    delt: float
    """Initial value for artificial time step in iterative solver."""

    tcon0: float
    """Constraint force scaling factor for ns --> 0."""

    lforbal: bool
    """Hack: directly compute innermost flux surface geometry from radial force balance"""

    return_outputs_even_if_not_converged: bool = False
    """If true, return the outputs even if VMEC++ did not converge.

    Otherwise a RuntimeError will be raised.
    """

    raxis_c: jt.Float[np.ndarray, "ntor_plus_1"]
    """Magnetic axis coefficients for R ~ cos(n*v); stellarator-symmetric."""

    zaxis_s: jt.Float[np.ndarray, "ntor_plus_1"]
    """Magnetic axis coefficients for Z ~ sin(n*v); stellarator-symmetric."""

    raxis_s: jt.Float[np.ndarray, "ntor_plus_1"] | None = None
    """Magnetic axis coefficients for R ~ sin(n*v); non-stellarator-symmetric."""

    zaxis_c: jt.Float[np.ndarray, "ntor_plus_1"] | None = None
    """Magnetic axis coefficients for Z ~ cos(n*v); non-stellarator-symmetric."""

    rbc: SerializableSparseCoefficientArray  # [mpol, 2 * ntor + 1]
    """Boundary coefficients for R ~ cos(m*u - n*v); stellarator-symmetric"""

    zbs: SerializableSparseCoefficientArray  # [mpol, 2 * ntor + 1]
    """Boundary coefficients for Z ~ sin(m*u - n*v); stellarator-symmetric"""

    rbs: SerializableSparseCoefficientArray | None = None  # [mpol, 2 * ntor + 1]
    """Boundary coefficients for R ~ sin(m*u - n*v); non-stellarator-symmetric"""

    zbc: SerializableSparseCoefficientArray | None = None  # [mpol, 2 * ntor + 1]
    """Boundary coefficients for Z ~ cos(m*u - n*v); non-stellarator-symmetric"""

    @pydantic.model_validator(mode="after")
    def _validate_fourier_coefficients_shapes(self) -> VmecInput:
        """All geometry coefficients need to have the shape (mpol, 2*ntor+1), wit 'rbs',
        'zbc' only populated for non-stellarator symmetric configurations."""
        mpol_two_ntor_plus_one_fields = ["rbc", "zbs"]
        mpol_two_ntor_plus_one_fields_lasym = ["rbs", "zbc"]
        if self.lasym:
            mpol_two_ntor_plus_one_fields.extend(mpol_two_ntor_plus_one_fields_lasym)

        expected_shape = (self.mpol, 2 * self.ntor + 1)
        for field in mpol_two_ntor_plus_one_fields:
            shape = np.shape(getattr(self, field))
            if shape != expected_shape:
                msg = f"{field} has shape {shape} instead of the expected {expected_shape}."
                "Please resize your coefficient array using vmecpp.VmecInput.resize_2d_coeff"
                f"({field}, {self.mpol}, {self.ntor}) to match mpol={self.mpol} or ntor={self.ntor}"
                raise ValueError(msg)
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
        """Construct a VmecInput with the same default settings as VMEC2000."""
        cpp_defaults = _vmecpp.VmecINDATAPyWrapper()
        return VmecInput._from_cpp_vmecindatapywrapper(cpp_defaults)

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
            if value is None:  # must be a symmetric field and lasym == False
                assert attr in {"rbs", "zbc", "zaxis_c", "raxis_s"}
                assert not cpp_indata.lasym
            else:
                getattr(cpp_indata, attr)[:] = value

        return cpp_indata

    def to_json(self, **kwargs) -> str:
        return self.model_dump_json(**kwargs)

    def save(self, output_path: str | Path) -> None:
        json_serialized = self.model_dump_json()
        output_path = Path(output_path)
        output_path.write_text(json_serialized)


# Fixed dimension of the profile inputs (i.e. pressure, iota, current)
_PRESET_DIM = 21
# Fixed dimension of the auxiliary profile quantities (i.e. am_aux_f)
_NDF_MAX_DIM = 101


# NOTE: in the future we want to change the C++ WOutFileContents layout so that it
# matches the classic Fortran one, so most of the compatibility layer here could
# disappear.
class VmecWOut(BaseModelWithNumpy):
    """Python equivalent of a VMEC "wout file".

    VmecWOut exposes the layout that SIMSOPT expects.
    The `save` method produces a NetCDF file compatible with SIMSOPT/Fortran VMEC.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    _MISSING_FORTRAN_VARIABLES: typing.ClassVar[list[str]] = [
        "lrecon__logical__",
        "lrfp__logical__",
        "lmove_axis__logical__",
        "mnyq",
        "nnyq",
        "currumnc",
        "currvmnc",
        "curlabel",
        "potvac",
        "nobser",
        "nobd",
        "nbsets",
    ]
    """The complete list of variables that can be found in Fortran VMEC wout files but
    not in wout files produced by VMEC++."""

    input_extension: str
    ier_flag: int
    nfp: int
    ns: int
    mpol: int
    ntor: int
    mnmax: int
    mnmax_nyq: int
    lasym: bool
    lfreeb: bool
    wb: float
    wp: float
    rmax_surf: float
    rmin_surf: float
    zmax_surf: float
    aspect: float
    betapol: float
    betator: float
    betaxis: float
    b0: float
    rbtor0: float
    rbtor: float
    IonLarmor: float
    ctor: float
    Aminor_p: float
    Rmajor_p: float
    volume: float
    fsqr: float
    fsqz: float
    fsql: float
    ftolv: float
    # Default initialized so reading stays backwards compatible pre v0.3.6
    itfsq: int = 0
    # NOTE: here, usage of the same dim1 or dim2 does NOT mean
    # they must have the same value across different attributes.
    phipf: jt.Float[np.ndarray, "dim1"]
    chipf: jt.Float[np.ndarray, "dim1"]
    jcuru: jt.Float[np.ndarray, "dim1"]
    jcurv: jt.Float[np.ndarray, "dim1"]
    # Default initialized so reading stays backwards compatible pre v0.3.6
    fsqt: jt.Float[np.ndarray, "time"] = np.array([])
    wdot: jt.Float[np.ndarray, "time"] = np.array([])
    jdotb: jt.Float[np.ndarray, "dim1"]
    bdotb: jt.Float[np.ndarray, "dim1"]
    bdotgradv: jt.Float[np.ndarray, "dim1"]
    DMerc: jt.Float[np.ndarray, "dim1"]
    equif: jt.Float[np.ndarray, "dim1"]
    xm: jt.Int[np.ndarray, "dim1"]
    xn: jt.Int[np.ndarray, "dim1"]
    xm_nyq: jt.Int[np.ndarray, "dim1"]
    xn_nyq: jt.Int[np.ndarray, "dim1"]
    mass: jt.Float[np.ndarray, "dim1"]
    buco: jt.Float[np.ndarray, "dim1"]
    bvco: jt.Float[np.ndarray, "dim1"]
    phips: jt.Float[np.ndarray, "dim1"]
    bmnc: jt.Float[np.ndarray, "dim1 dim2"]
    gmnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubumnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubvmnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubsmns: jt.Float[np.ndarray, "dim1 dim2"]
    bsupumnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsupvmnc: jt.Float[np.ndarray, "dim1 dim2"]
    rmnc: jt.Float[np.ndarray, "dim1 dim2"]
    zmns: jt.Float[np.ndarray, "dim1 dim2"]
    lmns: jt.Float[np.ndarray, "mnmax n_surfaces"]
    # lmns_full is not present in a typical Fortran wout file,
    # but we need to save it for fixed-boundary hot restart
    # to work properly. We store it with the Fortran convention
    # for the order of the dimensions for consistency with lmns.
    lmns_full: jt.Float[np.ndarray, "mnmax n_surfaces"]
    pcurr_type: str
    pmass_type: str
    piota_type: str
    am: jt.Float[np.ndarray, "_PRESET_DIM"]
    ac: jt.Float[np.ndarray, "_PRESET_DIM"]
    ai: jt.Float[np.ndarray, "_PRESET_DIM"]
    am_aux_s: jt.Float[np.ndarray, "_NDF_MAX_DIM"]
    am_aux_f: jt.Float[np.ndarray, "_NDF_MAX_DIM"]
    ac_aux_s: jt.Float[np.ndarray, "_NDF_MAX_DIM"]
    ac_aux_f: jt.Float[np.ndarray, "_NDF_MAX_DIM"]
    ai_aux_s: jt.Float[np.ndarray, "_NDF_MAX_DIM"]
    ai_aux_f: jt.Float[np.ndarray, "_NDF_MAX_DIM"]
    gamma: float
    mgrid_file: str
    nextcur: int
    extcur: jt.Float[np.ndarray, "extcur_len"] | float
    """Coil currents in A.

    for free-boundary runs, `extcur` has shape `(nextcur,)`
    for fixed-boundary it is a scalar float `extcur=nan`
    """

    mgrid_mode: MgridModeType

    # In the C++ WOutFileContents this is called iota_half.
    iotas: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called iota_full.
    iotaf: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called betatot.
    betatotal: float

    # In the C++ WOutFileContents this is called raxis_c.
    raxis_cc: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called zaxis_s.
    zaxis_cs: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called dVds.
    vp: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called pressure_full.
    presf: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called pressure_half.
    pres: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called toroidal_flux.
    phi: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called sign_of_jacobian.
    signgs: int

    # In the C++ WOutFileContents this is called VolAvgB.
    volavgB: float

    # In the C++ WOutFileContents this is called safety_factor.
    q_factor: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called poloidal_flux.
    chi: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called spectral_width.
    specw: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called overr.
    over_r: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called Dshear.
    DShear: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called Dwell.
    DWell: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called Dcurr.
    DCurr: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called Dgeod.
    DGeod: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called maximum_iterations.
    niter: int

    # In the C++ WOutFileContents this is called beta.
    beta_vol: jt.Float[np.ndarray, "dim1"]

    # In the C++ WOutFileContents this is called 'version' and it is a string.
    version_: float

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

    def save(self, out_path: str | Path) -> None:
        """Save contents in NetCDF3 format.

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
            # scalar ints
            for varname in [
                "nfp",
                "ns",
                "mpol",
                "ntor",
                "mnmax",
                "mnmax_nyq",
                "niter",
                "ier_flag",
                "signgs",
                "nextcur",
            ]:
                fnc.createVariable(varname, np.int32)
                fnc[varname][:] = getattr(self, varname)
            fnc.createVariable("lasym__logical__", np.int32)
            fnc["lasym__logical__"][:] = self.lasym
            fnc.createVariable("lfreeb__logical__", np.int32)
            fnc["lfreeb__logical__"][:] = self.lfreeb

            # scalar floats
            for varname in [
                "wb",
                "wp",
                "gamma",
                "rmax_surf",
                "rmin_surf",
                "zmax_surf",
                "aspect",
                "betatotal",
                "betapol",
                "betator",
                "betaxis",
                "b0",
                "rbtor0",
                "rbtor",
                "IonLarmor",
                "volavgB",
                "ctor",
                "Aminor_p",
                "Rmajor_p",
                "volume_p",
                "ftolv",
                "fsql",
                "fsqr",
                "fsqz",
                "itfsq",
            ]:
                fnc.createVariable(varname, np.float64)
                fnc[varname][:] = getattr(self, varname)

            # create dimensions (in the same order as VMEC2000)
            # For the input extension
            fnc.createDimension("dim_00100", 100)
            # dim_00200 = mgrid_file_max_string_length
            fnc.createDimension("dim_00200", 200)
            # dim_00020 = profile_strings_max_len
            fnc.createDimension("dim_00020", 20)
            # dimension of extcur and curlabel is not written in fixed-boundary wout
            if self.lfreeb:
                assert self.nextcur > 0
                fnc.createDimension("ext_current", self.nextcur)
            # and a single character for mgrid_mode
            fnc.createDimension("dim_00001", 1)
            fnc.createDimension("mn_mode", self.mnmax)
            fnc.createDimension("mn_mode_nyq", self.mnmax_nyq)
            fnc.createDimension("n_tor", self.ntor + 1)  # Fortran quirk
            fnc.createDimension("preset", _PRESET_DIM)
            fnc.createDimension("ndfmax", _NDF_MAX_DIM)
            fnc.createDimension("radius", self.ns)

            # Dimensions that are not in use yet
            _TIME = (
                self.itfsq
            )  # VMEC2000 has a fixed value of 100, we store all steps instead
            _MN_MAX_POT = 100  # TODO(jurasic) self.mnmaxpot
            fnc.createDimension("time", _TIME)
            fnc.createDimension("mn_mode_pot", _MN_MAX_POT)
            fnc.createDimension("current_label", 30)
            fnc.createDimension("dim_00006", 6)

            # radial profiles
            for varname in [
                "iotaf",
                "q_factor",
                "presf",
                "phi",
                "phipf",
                "chi",
                "chipf",
                "jcuru",
                "jcurv",
                "iotas",
                "mass",
                "pres",
                "beta_vol",
                "buco",
                "bvco",
                "vp",
                "specw",
                "phips",
                "over_r",
                "jdotb",
                "bdotb",
                "bdotgradv",
                "DMerc",
                "DShear",
                "DWell",
                "DCurr",
                "DGeod",
                "equif",
            ]:
                fnc.createVariable(varname, np.float64, ("radius",))
                fnc[varname][:] = getattr(self, varname)[:]

            for varname in ["am", "ai", "ac"]:
                fnc.createVariable(varname, np.float64, ("preset",))
                unpadded_array = getattr(self, varname)[:]
                fnc[varname][: len(unpadded_array)] = unpadded_array
                for aux_suffix, default_value in [("_aux_f", 0.0), ("_aux_s", -1.0)]:
                    auxname = varname + aux_suffix
                    fnc.createVariable(auxname, np.float64, ("ndfmax",))
                    # am_aux_f in C++ return a length 1 array with default values in
                    # NonEmptyVectorOr, but Fortran VMEC uses a fixed ndfmax=1001 elements.
                    # Repeat the default value as needed.
                    unpadded_array = getattr(self, auxname)[:]
                    fnc[auxname][:] = np.pad(
                        unpadded_array,
                        (0, _NDF_MAX_DIM - len(unpadded_array)),
                        mode="constant",
                        constant_values=default_value,
                    )

            for varname in ["raxis_cc", "zaxis_cs"]:
                fnc.createVariable(varname, np.float64, ("n_tor",))
                fnc[varname][:] = getattr(self, varname)[:]

            for varname in ["xm", "xn"]:
                fnc.createVariable(varname, np.float64, ("mn_mode",))
                fnc[varname][:] = getattr(self, varname)[:]

            for varname in ["xm_nyq", "xn_nyq"]:
                fnc.createVariable(varname, np.float64, ("mn_mode_nyq",))
                fnc[varname][:] = getattr(self, varname)[:]

            for varname in [
                "gmnc",
                "bmnc",
                "bsubumnc",
                "bsubvmnc",
                "bsubsmns",
                "bsupumnc",
                "bsupvmnc",
            ]:
                fnc.createVariable(varname, np.float64, ("radius", "mn_mode_nyq"))
                fnc[varname][:] = getattr(self, varname).T[:]

            # fourier coefficients
            for varname in ["rmnc", "zmns", "lmns"]:
                fnc.createVariable(varname, np.float64, ("radius", "mn_mode"))
                fnc[varname][:] = getattr(self, varname).T[:]

            # Convergence time trace information
            for varname in ["fsqt", "wdot"]:
                fnc.createVariable(varname, np.float64, ("time",))
                fnc[varname][:] = getattr(self, varname).T[:]

            fnc.createVariable("lmns_full", np.float64, ("radius", "mn_mode"))
            fnc["lmns_full"][:] = self.lmns_full.T[:]

            if self.nextcur == 0:
                # extcur is a scalar in fixed-boundary runs
                assert isinstance(self.extcur, float)
                assert self.extcur == netCDF4.default_fillvals["f8"]
                fnc.createVariable("extcur", np.float64)
            else:
                # extcur is the array of coil currents in free-boundary runs
                assert np.shape(self.extcur)[0] == self.nextcur
                fnc.createVariable("extcur", np.float64, ("ext_current",))
            fnc["extcur"][:] = self.extcur

            # version_ is required to make COBRAVMEC work correctly:
            # it changes its behavior depending on the VMEC version (>6 or not)
            fnc.createVariable("version_", np.float64)
            fnc["version_"][:] = self.version_

            # strings
            def create_string_variable(
                varname: str, dimension_name: str
            ) -> netCDF4.Variable:
                max_string_length = fnc.dimensions[dimension_name].size
                string_variable = fnc.createVariable(varname, "S1", (dimension_name,))

                # Put the string in the format netCDF3 requires. Don't know what to say.
                value = getattr(self, varname)
                padded_value_as_array = np.array(
                    value.encode(encoding="ascii").ljust(max_string_length)
                )
                padded_value_as_netcdf3_compatible_chararray = netCDF4.stringtochar(
                    padded_value_as_array
                )
                string_variable[:] = padded_value_as_netcdf3_compatible_chararray
                return string_variable

            for varname in ["pcurr_type", "pmass_type", "piota_type"]:
                create_string_variable(varname, "dim_00020")

            create_string_variable("mgrid_file", "dim_00200")
            create_string_variable("mgrid_mode", "dim_00001")
            create_string_variable("input_extension", "dim_00100")

    @staticmethod
    def _from_cpp_wout(cpp_wout: _vmecpp.VmecppWOut) -> VmecWOut:
        attrs = {}

        # These attributes are the same in VMEC++ and in Fortran VMEC
        attrs["ier_flag"] = cpp_wout.ier_flag
        attrs["nfp"] = cpp_wout.nfp
        attrs["ns"] = cpp_wout.ns
        attrs["mpol"] = cpp_wout.mpol
        attrs["ntor"] = cpp_wout.ntor
        attrs["mnmax"] = cpp_wout.mnmax
        attrs["mnmax_nyq"] = cpp_wout.mnmax_nyq
        attrs["lasym"] = cpp_wout.lasym
        attrs["lfreeb"] = cpp_wout.lfreeb
        attrs["wb"] = cpp_wout.wb
        attrs["wp"] = cpp_wout.wp
        attrs["rmax_surf"] = cpp_wout.rmax_surf
        attrs["rmin_surf"] = cpp_wout.rmin_surf
        attrs["zmax_surf"] = cpp_wout.zmax_surf
        attrs["aspect"] = cpp_wout.aspect
        attrs["betapol"] = cpp_wout.betapol
        attrs["betator"] = cpp_wout.betator
        attrs["betaxis"] = cpp_wout.betaxis
        attrs["b0"] = cpp_wout.b0
        attrs["rbtor0"] = cpp_wout.rbtor0
        attrs["rbtor"] = cpp_wout.rbtor
        attrs["IonLarmor"] = cpp_wout.IonLarmor
        attrs["ctor"] = cpp_wout.ctor
        attrs["Aminor_p"] = cpp_wout.Aminor_p
        attrs["Rmajor_p"] = cpp_wout.Rmajor_p
        attrs["volume"] = cpp_wout.volume_p
        attrs["fsqr"] = cpp_wout.fsqr
        attrs["fsqz"] = cpp_wout.fsqz
        attrs["fsql"] = cpp_wout.fsql
        attrs["itfsq"] = cpp_wout.itfsq
        attrs["phipf"] = cpp_wout.phipf
        attrs["chipf"] = cpp_wout.chipf
        attrs["jcuru"] = cpp_wout.jcuru
        attrs["jcurv"] = cpp_wout.jcurv
        attrs["fsqt"] = cpp_wout.fsqt
        attrs["wdot"] = cpp_wout.wdot
        attrs["jdotb"] = cpp_wout.jdotb
        attrs["bdotb"] = cpp_wout.bdotb
        attrs["bdotgradv"] = cpp_wout.bdotgradv
        attrs["DMerc"] = cpp_wout.DMerc
        attrs["equif"] = cpp_wout.equif
        attrs["xm"] = cpp_wout.xm.astype(np.int64)
        attrs["xn"] = cpp_wout.xn.astype(np.int64)
        attrs["xm_nyq"] = cpp_wout.xm_nyq.astype(np.int64)
        attrs["xn_nyq"] = cpp_wout.xn_nyq.astype(np.int64)
        attrs["ftolv"] = cpp_wout.ftolv
        attrs["pcurr_type"] = cpp_wout.pcurr_type
        attrs["pmass_type"] = cpp_wout.pmass_type
        attrs["piota_type"] = cpp_wout.piota_type
        attrs["gamma"] = cpp_wout.gamma
        attrs["mgrid_file"] = cpp_wout.mgrid_file
        attrs["mgrid_mode"] = cpp_wout.mgrid_mode
        attrs["nextcur"] = cpp_wout.nextcur
        # extcur needs special treatment for fixed-boundary cases:
        # in VMEC++ it will be an empty array, but Fortran VMEC uses
        # a default fill value (masked out in the netcdf file)
        if cpp_wout.nextcur > 0:
            attrs["extcur"] = cpp_wout.extcur
        else:
            attrs["extcur"] = netCDF4.default_fillvals["f8"]

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
        attrs["am"] = np.pad(cpp_wout.am, (0, _PRESET_DIM - len(cpp_wout.am)))
        attrs["ac"] = np.pad(cpp_wout.ac, (0, _PRESET_DIM - len(cpp_wout.ac)))
        attrs["ai"] = np.pad(cpp_wout.ai, (0, _PRESET_DIM - len(cpp_wout.ai)))
        attrs["am_aux_s"] = np.pad(
            cpp_wout.am_aux_s,
            (0, _NDF_MAX_DIM - len(cpp_wout.am_aux_s)),
            mode="constant",
            constant_values=-1.0,
        )
        attrs["am_aux_f"] = np.pad(
            cpp_wout.am_aux_f,
            (0, _NDF_MAX_DIM - len(cpp_wout.am_aux_f)),
            mode="constant",
            constant_values=0.0,
        )
        attrs["ac_aux_s"] = np.pad(
            cpp_wout.ac_aux_s,
            (0, _NDF_MAX_DIM - len(cpp_wout.ac_aux_s)),
            mode="constant",
            constant_values=-1.0,
        )
        attrs["ac_aux_f"] = np.pad(
            cpp_wout.ac_aux_f,
            (0, _NDF_MAX_DIM - len(cpp_wout.ac_aux_f)),
            mode="constant",
            constant_values=0.0,
        )
        attrs["ai_aux_s"] = np.pad(
            cpp_wout.ai_aux_s,
            (0, _NDF_MAX_DIM - len(cpp_wout.ai_aux_s)),
            mode="constant",
            constant_values=-1.0,
        )
        attrs["ai_aux_f"] = np.pad(
            cpp_wout.ai_aux_f,
            (0, _NDF_MAX_DIM - len(cpp_wout.ai_aux_f)),
            mode="constant",
            constant_values=0.0,
        )

        attrs["version_"] = float(cpp_wout.version)

        attrs["input_extension"] = cpp_wout.input_extension

        # The Pydantic model raises an error if there are missing keys,
        # or extra keys that weren't included in MISSING_FORTRAN_VARIABLES
        return VmecWOut(**attrs)

    def _to_cpp_wout(self) -> _vmecpp.WOutFileContents:
        cpp_wout = _vmecpp.WOutFileContents()

        # These attributes are the same in VMEC++ and in Fortran VMEC
        cpp_wout.ier_flag = self.ier_flag
        cpp_wout.nfp = self.nfp
        cpp_wout.ns = self.ns
        cpp_wout.mpol = self.mpol
        cpp_wout.ntor = self.ntor
        cpp_wout.mnmax = self.mnmax
        cpp_wout.mnmax_nyq = self.mnmax_nyq
        cpp_wout.lasym = self.lasym
        cpp_wout.lfreeb = self.lfreeb
        cpp_wout.wb = self.wb
        cpp_wout.wp = self.wp
        cpp_wout.rmax_surf = self.rmax_surf
        cpp_wout.rmin_surf = self.rmin_surf
        cpp_wout.zmax_surf = self.zmax_surf
        cpp_wout.aspect = self.aspect
        cpp_wout.betapol = self.betapol
        cpp_wout.betator = self.betator
        cpp_wout.betaxis = self.betaxis
        cpp_wout.b0 = self.b0
        cpp_wout.rbtor0 = self.rbtor0
        cpp_wout.rbtor = self.rbtor
        cpp_wout.IonLarmor = self.IonLarmor
        cpp_wout.ctor = self.ctor
        cpp_wout.Aminor_p = self.Aminor_p
        cpp_wout.Rmajor_p = self.Rmajor_p
        cpp_wout.volume_p = self.volume_p
        cpp_wout.fsqr = self.fsqr
        cpp_wout.fsqz = self.fsqz
        cpp_wout.fsql = self.fsql
        cpp_wout.itfsq = self.itfsq
        cpp_wout.phipf = self.phipf
        cpp_wout.chipf = self.chipf
        cpp_wout.jcuru = self.jcuru
        cpp_wout.jcurv = self.jcurv
        cpp_wout.fsqt = self.fsqt
        cpp_wout.wdot = self.wdot
        cpp_wout.jdotb = self.jdotb
        cpp_wout.bdotb = self.bdotb
        cpp_wout.bdotgradv = self.bdotgradv
        cpp_wout.DMerc = self.DMerc
        cpp_wout.equif = self.equif
        cpp_wout.xm = self.xm
        cpp_wout.xn = self.xn
        cpp_wout.xm_nyq = self.xm_nyq
        cpp_wout.xn_nyq = self.xn_nyq
        cpp_wout.ftolv = self.ftolv
        cpp_wout.pcurr_type = self.pcurr_type
        cpp_wout.pmass_type = self.pmass_type
        cpp_wout.piota_type = self.piota_type
        cpp_wout.gamma = self.gamma
        cpp_wout.mgrid_file = self.mgrid_file
        cpp_wout.nextcur = self.nextcur
        if self.nextcur > 0:
            cpp_wout.extcur = self.extcur
        else:
            cpp_wout.extcur = np.array([])
        cpp_wout.mgrid_mode = self.mgrid_mode

        # These attributes are called differently
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
        cpp_wout.input_extension = self.input_extension

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

        # These attributes have one column less and their elements are transposed
        # in VMEC++ with respect to SIMSOPT/VMEC2000
        cpp_wout.lmns = self.lmns.T[1:, :]
        cpp_wout.bmnc = self.bmnc.T[1:, :]
        cpp_wout.bsubumnc = self.bsubumnc.T[1:, :]
        cpp_wout.bsubvmnc = self.bsubvmnc.T[1:, :]
        cpp_wout.bsupumnc = self.bsupumnc.T[1:, :]
        cpp_wout.bsupvmnc = self.bsupvmnc.T[1:, :]
        cpp_wout.gmnc = self.gmnc.T[1:, :]

        return cpp_wout

    @staticmethod
    def from_wout_file(wout_filename: str | Path) -> VmecWOut:
        """Load wout contents in NetCDF format.

        This is the format used by Fortran VMEC implementations and the one expected by
        SIMSOPT.
        """
        with netCDF4.Dataset(wout_filename, "r") as fnc:
            fnc.set_auto_mask(False)
            attrs = {}
            for var_name in fnc.variables:
                if var_name in VmecWOut._MISSING_FORTRAN_VARIABLES:
                    continue
                if var_name in ["lasym__logical__", "lfreeb__logical__"]:
                    key = var_name.removesuffix("__logical__")
                    attrs[key] = fnc[var_name][()] != 0
                elif var_name == "volume_p":
                    key = var_name.removesuffix("_p")
                    attrs[key] = fnc[var_name][()]
                elif var_name in ["xm", "xn", "xm_nyq", "xn_nyq"]:
                    attrs[var_name] = np.array(fnc[var_name][()], dtype=np.int32)
                elif var_name in [
                    "pmass_type",
                    "piota_type",
                    "pcurr_type",
                    "mgrid_file",
                    "mgrid_mode",
                ]:
                    # Remove both zero-padding and whitespaces.
                    attrs[var_name] = (
                        fnc[var_name][()]
                        .tobytes()
                        .decode("ascii")
                        .strip("\x00")
                        .strip()
                    )
                elif var_name in [
                    "bmnc",
                    "gmnc",
                    "bsubumnc",
                    "bsubvmnc",
                    "bsubsmns",
                    "bsupumnc",
                    "bsupvmnc",
                    "rmnc",
                    "zmns",
                    "lmns",
                    "lmns_full",
                ]:
                    attrs[var_name] = np.transpose(fnc[var_name][()])
                else:
                    attrs[var_name] = fnc[var_name][()]

        # Special handling for variables only present in VMEC++
        # For now, only special case for lambda coefficients: lambda = 0 is a physically meaningful fall-back value
        if "lmns_full" not in attrs:
            mnmax = attrs["mnmax"]
            ns = attrs["ns"]
            attrs["lmns_full"] = np.zeros([mnmax, ns])

        # Optional handling for backwards compatibility with wout files produced before v0.3.3
        # Handle extcur
        if "extcur" not in attrs:
            assert (
                attrs["nextcur"] == 0
            ), "extcur must be present for free-boundary wout files"
            attrs["extcur"] = netCDF4.default_fillvals["f8"]
        # Handle input_extension
        if "input_extension" in attrs:
            attrs["input_extension"] = (
                attrs["input_extension"].tobytes().decode("ascii").strip()
            )
        else:
            attrs["input_extension"] = ""

        return VmecWOut(**attrs)


class Threed1Volumetrics(BaseModelWithNumpy):
    model_config = pydantic.ConfigDict(extra="forbid")

    int_p: float
    avg_p: float

    int_bpol: float
    avg_bpol: float

    int_btor: float
    avg_btor: float

    int_modb: float
    avg_modb: float

    int_ekin: float
    avg_ekin: float

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

    s: jt.Float[np.ndarray, "dim1"]
    toroidal_flux: jt.Float[np.ndarray, "dim1"]
    iota: jt.Float[np.ndarray, "dim1"]
    shear: jt.Float[np.ndarray, "dim1"]
    d_volume_d_s: jt.Float[np.ndarray, "dim1"]
    well: jt.Float[np.ndarray, "dim1"]
    toroidal_current: jt.Float[np.ndarray, "dim1"]
    d_toroidal_current_d_s: jt.Float[np.ndarray, "dim1"]
    pressure: jt.Float[np.ndarray, "dim1"]
    d_pressure_d_s: jt.Float[np.ndarray, "dim1"]
    DMerc: jt.Float[np.ndarray, "dim1"]
    Dshear: jt.Float[np.ndarray, "dim1"]
    Dwell: jt.Float[np.ndarray, "dim1"]
    Dcurr: jt.Float[np.ndarray, "dim1"]
    Dgeod: jt.Float[np.ndarray, "dim1"]

    @staticmethod
    def _from_cpp_mercier(cpp_mercier: _vmecpp.Mercier) -> Mercier:
        mercier = Mercier(
            **{attr: getattr(cpp_mercier, attr) for attr in Mercier.model_fields}
        )

        return mercier


class JxBOut(BaseModelWithNumpy):
    model_config = pydantic.ConfigDict(extra="forbid")

    itheta: jt.Float[np.ndarray, "num_full nZnT"]
    izeta: jt.Float[np.ndarray, "num_full nZnT"]
    bdotk: jt.Float[np.ndarray, "num_full nZnT"]

    amaxfor: jt.Float[np.ndarray, "dim1"]
    aminfor: jt.Float[np.ndarray, "dim1"]
    avforce: jt.Float[np.ndarray, "dim1"]
    pprim: jt.Float[np.ndarray, "dim1"]
    jdotb: jt.Float[np.ndarray, "dim1"]
    bdotb: jt.Float[np.ndarray, "dim1"]
    bdotgradv: jt.Float[np.ndarray, "dim1"]
    jpar2: jt.Float[np.ndarray, "dim1"]
    jperp2: jt.Float[np.ndarray, "dim1"]
    phin: jt.Float[np.ndarray, "dim1"]

    jsupu3: jt.Float[np.ndarray, "num_full nZnT"]
    jsupv3: jt.Float[np.ndarray, "num_full nZnT"]
    jsups3: jt.Float[np.ndarray, "num_half nZnT"]

    bsupu3: jt.Float[np.ndarray, "num_full nZnT"]
    bsupv3: jt.Float[np.ndarray, "num_full nZnT"]
    jcrossb: jt.Float[np.ndarray, "num_full nZnT"]
    jxb_gradp: jt.Float[np.ndarray, "num_full nZnT"]
    jdotb_sqrtg: jt.Float[np.ndarray, "num_full nZnT"]
    sqrtg3: jt.Float[np.ndarray, "num_full nZnT"]

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
    """Python equivalent of VMEC's "mercier" file."""

    threed1_volumetrics: Threed1Volumetrics
    """Python equivalent of VMEC's volumetrics section in the "threed1" file."""

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

    Keyword Args:
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
            magnetic_response_table=magnetic_field,
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
    arr: jt.Float[np.ndarray, "ns_minus_one mn"], mnsize: int
) -> jt.Float[np.ndarray, "mn ns"]:
    stacked = np.vstack((np.zeros(mnsize), arr)).T
    assert stacked.shape[1] == arr.shape[0] + 1
    assert stacked.shape[0] == arr.shape[1]
    return stacked


__all__ = [
    "JxBOut",
    "MagneticFieldResponseTable",
    "MakegridParameters",
    "Mercier",
    "Threed1Volumetrics",
    "VmecInput",
    "VmecOutput",
    "VmecWOut",
    "run",
]

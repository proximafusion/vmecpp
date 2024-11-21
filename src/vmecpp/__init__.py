# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import tempfile
from pathlib import Path

import jaxtyping as jt
import netCDF4
import numpy as np
import pydantic
from beartype import beartype

from vmecpp import _util
from vmecpp.cpp import _vmecpp  # bindings to the C++ core
from vmecpp.cpp.vmecpp import simsopt_compat


# This is a pure Python equivalent of VmecINDATAPyWrapper.
# In the future VmecINDATAPyWrapper and the C++ VmecINDATA will merge into one type,
# and this will become a Python wrapper around the one C++ VmecINDATA type.
# This pure Python type could _also_ disappear if we can get proper autocompletion,
# docstring peeking etc. for the one C++ VmecINDATA type bound via pybind11.
class VmecInput(pydantic.BaseModel):
    """The input to a VMEC++ run. Contains settings as well as the definition of the
    plasma boundary.

    Python equivalent of a VMEC++ JSON input file or a classic INDATA file (e.g.
    "input.best").

    Deserialize from JSON and serialize to JSON using the usual pydantic methods:
    `model_validate_json` and `model_dump_json`.
    """

    model_config = pydantic.ConfigDict(
        # allow numpy arrays as attributes -- although this breaks pydantic's automatic JSON (de)serialization
        arbitrary_types_allowed=True,
        # serialize NaN and infinite floats as strings in JSON output.
        ser_json_inf_nan="strings",
    )

    lasym: bool
    """Flag to indicate non-stellarator-symmetry."""

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

    ns_array: jt.Int[np.ndarray, " num_grids"]
    """Number of flux surfaces per multigrid step."""

    ftol_array: jt.Float[np.ndarray, " num_grids"]
    """Requested force tolerance for convergence per multigrid step."""

    niter_array: jt.Int[np.ndarray, " num_grids"]
    """Maximum number of iterations per multigrid step."""

    phiedge: float
    """Total enclosed toroidal magnetic flux in Vs == Wb."""

    ncurr: int
    """Select constraint on iota or enclosed toroidal current profiles 0: constrained-iota; 1: constrained-current"""

    pmass_type: str
    """Parametrization of mass/pressure profile."""

    am: jt.Float[np.ndarray, " am_len"]
    """Mass/pressure profile coefficients."""

    am_aux_s: jt.Float[np.ndarray, " am_aux_len"]
    """Spline mass/pressure profile: knot locations in s"""

    am_aux_f: jt.Float[np.ndarray, " am_aux_len"]
    """Spline mass/pressure profile: values at knots"""

    pres_scale: float
    """Global scaling factor for mass/pressure profile."""

    gamma: float
    """Adiabatic index."""

    spres_ped: float
    """Location of pressure pedestal in s."""

    piota_type: str
    """Parametrization of iota profile."""

    ai: jt.Float[np.ndarray, " ai_len"]
    """Iota profile coefficients."""

    ai_aux_s: jt.Float[np.ndarray, " ai_aux_len"]
    """Spline iota profile: knot locations in s"""

    ai_aux_f: jt.Float[np.ndarray, " ai_aux_len"]
    """Spline iota profile: values at knots"""

    pcurr_type: str
    """Parametrization of toroidal current profile."""

    ac: jt.Float[np.ndarray, " ac_len"]
    """Enclosed toroidal current profile coefficients."""

    ac_aux_s: jt.Float[np.ndarray, " ac_aux_len"]
    """Spline toroidal current profile: knot locations in s"""

    ac_aux_f: jt.Float[np.ndarray, " ac_aux_len"]
    """Spline toroidal current profile: values at knots"""

    curtor: float
    """Toroidal current in A."""

    bloat: float
    """Bloating factor (for constrained toroidal current)"""

    lfreeb: bool
    """Flag to indicate free-boundary."""

    mgrid_file: str
    """Full path for vacuum Green's function data."""

    extcur: jt.Float[np.ndarray, " extcur_len"]
    """Coil currents in A."""

    nvacskip: int
    """Number of iterations between full vacuum calculations."""

    nstep: int
    """Printout interval."""

    aphi: jt.Float[np.ndarray, " aphi_len"]
    """Radial flux zoning profile coefficients."""

    delt: float
    """Initial value for artificial time step in iterative solver."""

    tcon0: float
    """Constraint force scaling factor for ns --> 0."""

    lforbal: bool
    """Hack: directly compute innermost flux surface geometry from radial force balance"""

    raxis_c: jt.Float[np.ndarray, " ntor_plus_1"]
    """Magnetic axis coefficients for R ~ cos(n*v); stellarator-symmetric."""

    zaxis_s: jt.Float[np.ndarray, " ntor_plus_1"]
    """Magnetic axis coefficients for Z ~ sin(n*v); stellarator-symmetric."""

    raxis_s: jt.Float[np.ndarray, " ntor_plus_1"]
    """Magnetic axis coefficients for R ~ sin(n*v); non-stellarator-symmetric."""

    zaxis_c: jt.Float[np.ndarray, " ntor_plus_1"]
    """Magnetic axis coefficients for Z ~ cos(n*v); non-stellarator-symmetric."""

    rbc: jt.Float[np.ndarray, " mpol two_ntor_plus_one"]
    """Boundary coefficients for R ~ cos(m*u - n*v); stellarator-symmetric"""

    zbs: jt.Float[np.ndarray, " mpol two_ntor_plus_one"]
    """Boundary coefficients for Z ~ sin(m*u - n*v); stellarator-symmetric"""

    rbs: jt.Float[np.ndarray, " mpol two_ntor_plus_one"]
    """Boundary coefficients for R ~ sin(m*u - n*v); non-stellarator-symmetric"""

    zbc: jt.Float[np.ndarray, " mpol two_ntor_plus_one"]
    """Boundary coefficients for Z ~ cos(m*u - n*v); non-stellarator-symmetric"""

    @staticmethod
    def from_file(input_file: str | Path) -> VmecInput:
        """Build a VmecInput from either a VMEC++ JSON input file or a classic INDATA
        file."""
        absolute_input_path = Path(input_file).resolve()

        with tempfile.TemporaryDirectory() as tmpdir, _util.change_working_directory_to(  # noqa: SIM117
            Path(tmpdir)
        ):  # we call this in a temporary directory because it produces the file in the current working directory
            with simsopt_compat.ensure_vmecpp_input(
                absolute_input_path
            ) as vmecpp_input_file:
                vmecpp_indata_pywrapper = _vmecpp.VmecINDATAPyWrapper.from_file(
                    vmecpp_input_file
                )

        return VmecInput._from_cpp_vmecindatapywrapper(vmecpp_indata_pywrapper)

    @staticmethod
    def _from_cpp_vmecindatapywrapper(
        vmecindatapywrapper: _vmecpp.VmecINDATAPyWrapper,
    ) -> VmecInput:
        return VmecInput.model_validate(
            {
                attr_name: getattr(vmecindatapywrapper, attr_name)
                for attr_name in VmecInput.model_fields
            }
        )

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
            getattr(cpp_indata, attr)[:] = getattr(self, attr)

        return cpp_indata

    # TODO(eguiraud): implement a save function. we can either teach pydantic
    # how we want the fourier coefficients to be formatted, or convert
    # to VmecINDATAPyWrapper and leverage its to_json function.
    # But then model_dump_json and model_validate_json will not work properly.


# NOTE: in the future we want to change the C++ WOutFileContents layout so that it
# matches the classic Fortran one, so most of the compatibility layer here could
# disappear.
class VmecWout(pydantic.BaseModel):
    """Python equivalent of a VMEC "wout file".

    VmecWout exposes the layout that SIMSOPT expects.
    The `save` method produces a NetCDF file compatible with SIMSOPT/Fortran VMEC.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra="forbid")

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
    # NOTE: here, usage of the same dim1 or dim2 does NOT mean
    # they must have the same value across different attributes.
    phipf: jt.Float[np.ndarray, " dim1"]
    chipf: jt.Float[np.ndarray, " dim1"]
    jcuru: jt.Float[np.ndarray, " dim1"]
    jcurv: jt.Float[np.ndarray, " dim1"]
    jdotb: jt.Float[np.ndarray, " dim1"]
    bdotgradv: jt.Float[np.ndarray, " dim1"]
    DMerc: jt.Float[np.ndarray, " dim1"]
    equif: jt.Float[np.ndarray, " dim1"]
    xm: jt.Int[np.ndarray, " dim1"]
    xn: jt.Int[np.ndarray, " dim1"]
    xm_nyq: jt.Int[np.ndarray, " dim1"]
    xn_nyq: jt.Int[np.ndarray, " dim1"]
    mass: jt.Float[np.ndarray, " dim1"]
    buco: jt.Float[np.ndarray, " dim1"]
    bvco: jt.Float[np.ndarray, " dim1"]
    phips: jt.Float[np.ndarray, " dim1"]
    bmnc: jt.Float[np.ndarray, "dim1 dim2"]
    gmnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubumnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubvmnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubsmns: jt.Float[np.ndarray, "dim1 dim2"]
    bsupumnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsupvmnc: jt.Float[np.ndarray, "dim1 dim2"]
    rmnc: jt.Float[np.ndarray, "dim1 dim2"]
    zmns: jt.Float[np.ndarray, "dim1 dim2"]
    lmns: jt.Float[np.ndarray, "dim1 dim2"]
    pcurr_type: str
    pmass_type: str
    piota_type: str
    gamma: float
    mgrid_file: str

    iotas: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called iota_half."""

    iotaf: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called iota_full."""

    betatotal: float
    """In VMEC++ this is called betatot."""

    raxis_cc: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called raxis_c."""

    zaxis_cs: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called zaxis_s."""

    vp: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called dVds."""

    presf: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called pressure_full."""

    pres: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called pressure_half."""

    phi: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called toroidal_flux."""

    signgs: int
    """In VMEC++ this is called sign_of_jacobian."""

    volavgB: float
    """In VMEC++ this is called VolAvgB."""

    q_factor: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called safety_factor."""

    chi: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called poloidal_flux."""

    specw: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called spectral_width."""

    over_r: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called overr."""

    DShear: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called Dshear."""

    DWell: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called Dwell."""

    DCurr: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called Dcurr."""

    DGeod: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called Dgeod."""

    niter: int
    """In VMEC++ this is called maximum_iterations."""

    beta_vol: jt.Float[np.ndarray, "..."]
    """In VMEC++ this is called beta."""

    version_: float
    """In VMEC++ this is called 'version' and it is a string."""

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

    def save(self, out_path: Path) -> None:
        """Save contents in NetCDF3 format.

        This is the format used by Fortran VMEC implementations and the one expected by
        SIMSOPT.
        """
        with netCDF4.Dataset(out_path, "w", format="NETCDF3_CLASSIC") as fnc:
            # scalar ints
            for varname in [
                "ier_flag",
                "niter",
                "nfp",
                "ns",
                "mpol",
                "ntor",
                "mnmax",
                "mnmax_nyq",
                "signgs",
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
                "fsqr",
                "fsqz",
                "fsql",
                "ftolv",
                "gamma",
            ]:
                fnc.createVariable(varname, np.float64)
                fnc[varname][:] = getattr(self, varname)

            # create dimensions
            fnc.createDimension("mn_mode", self.mnmax)
            fnc.createDimension("radius", self.ns)
            fnc.createDimension("n_tor", self.ntor + 1)  # Fortran quirk
            fnc.createDimension("mn_mode_nyq", self.mnmax_nyq)

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
                "bmnc",
                "gmnc",
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

            # version_ is required to make COBRAVMEC work correctly:
            # it changes its behavior depending on the VMEC version (>6 or not)
            fnc.createVariable("version_", np.float64)
            fnc["version_"][:] = self.version_

            # strings
            # maximum length of the string, copied from wout_cma.nc
            max_string_length = 20
            fnc.createDimension("profile_strings_max_len", max_string_length)
            for varname in ["pcurr_type", "pmass_type", "piota_type"]:
                string_variable = fnc.createVariable(
                    varname, "S1", ("profile_strings_max_len",)
                )

                # Put the string in the format netCDF3 requires. Don't know what to say.
                value = getattr(self, varname)
                padded_value_as_array = np.array(
                    value.encode(encoding="ascii").ljust(max_string_length)
                )
                padded_value_as_netcdf3_compatible_chararray = netCDF4.stringtochar(
                    padded_value_as_array
                )  # pyright: ignore

                string_variable[:] = padded_value_as_netcdf3_compatible_chararray

            # now mgrid_file
            varname = "mgrid_file"
            max_string_length = 200  # value copied from wout_cma.nc
            fnc.createDimension("mgrid_file_max_string_length", max_string_length)
            string_variable = fnc.createVariable(
                varname, "S1", ("mgrid_file_max_string_length",)
            )
            value = getattr(self, varname)
            padded_value_as_array = np.array(
                value.encode(encoding="ascii").ljust(max_string_length)
            )
            padded_value_as_netcdf3_compatible_chararray = netCDF4.stringtochar(
                padded_value_as_array
            )  # pyright: ignore

            string_variable[:] = padded_value_as_netcdf3_compatible_chararray

    @staticmethod
    def _from_cpp_wout(cpp_wout: _vmecpp.VmecppWOut) -> VmecWout:
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
        attrs["phipf"] = cpp_wout.phipf
        attrs["chipf"] = cpp_wout.chipf
        attrs["jcuru"] = cpp_wout.jcuru
        attrs["jcurv"] = cpp_wout.jcurv
        attrs["jdotb"] = cpp_wout.jdotb
        attrs["bdotgradv"] = cpp_wout.bdotgradv
        attrs["DMerc"] = cpp_wout.DMerc
        attrs["equif"] = cpp_wout.equif
        attrs["xm"] = cpp_wout.xm
        attrs["xn"] = cpp_wout.xn
        attrs["xm_nyq"] = cpp_wout.xm_nyq
        attrs["xn_nyq"] = cpp_wout.xn_nyq
        attrs["ftolv"] = cpp_wout.ftolv
        attrs["pcurr_type"] = cpp_wout.pcurr_type
        attrs["pmass_type"] = cpp_wout.pmass_type
        attrs["piota_type"] = cpp_wout.piota_type
        attrs["gamma"] = cpp_wout.gamma
        attrs["mgrid_file"] = cpp_wout.mgrid_file

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

        # These attributes are transposed in SIMSOPT
        attrs["rmnc"] = cpp_wout.rmnc.T
        attrs["zmns"] = cpp_wout.zmns.T
        attrs["bsubsmns"] = cpp_wout.bsubsmns.T

        # These attributes have one column less and their elements are transposed
        # in VMEC++ with respect to SIMSOPT/VMEC2000
        attrs["lmns"] = _pad_and_transpose(cpp_wout.lmns, attrs["mnmax"])
        attrs["bmnc"] = _pad_and_transpose(cpp_wout.bmnc, attrs["mnmax_nyq"])
        attrs["bsubumnc"] = _pad_and_transpose(cpp_wout.bsubumnc, attrs["mnmax_nyq"])
        attrs["bsubvmnc"] = _pad_and_transpose(cpp_wout.bsubvmnc, attrs["mnmax_nyq"])
        attrs["bsupumnc"] = _pad_and_transpose(cpp_wout.bsupumnc, attrs["mnmax_nyq"])
        attrs["bsupvmnc"] = _pad_and_transpose(cpp_wout.bsupvmnc, attrs["mnmax_nyq"])
        attrs["gmnc"] = _pad_and_transpose(cpp_wout.gmnc, attrs["mnmax_nyq"])

        attrs["version_"] = float(cpp_wout.version)

        return VmecWout(**attrs)

    # TODO(eguiraud): implement from_wout_file


class VmecOutput(pydantic.BaseModel):
    """Container for the full output of a VMEC run."""

    wout: VmecWout
    """Python equivalent of VMEC's "wout file"."""

    input: VmecInput
    """The input to the VMEC run that produced this output."""


def run(input: VmecInput, max_threads=None) -> VmecOutput:
    """Run VMEC++ using the provided input."""
    cpp_indata = input._to_cpp_vmecindatapywrapper()
    cpp_output_quantities = _vmecpp.run(cpp_indata, max_threads)
    cpp_wout = cpp_output_quantities.wout
    wout = VmecWout._from_cpp_wout(cpp_wout)
    return VmecOutput(wout=wout, input=input)


@jt.jaxtyped(typechecker=beartype)
def _pad_and_transpose(
    arr: jt.Float[np.ndarray, "ns_minus_one mn"], mnsize: int
) -> jt.Float[np.ndarray, "mn ns_minus_one+1"]:
    return np.vstack((np.zeros(mnsize), arr)).T


__all__ = ["VmecInput", "VmecOutput", "VmecWout", "run"]

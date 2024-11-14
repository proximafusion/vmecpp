from __future__ import annotations

from pathlib import Path
from typing import Protocol

import jaxtyping as jt
import netCDF4
import numpy as np
import pydantic
from util.types import runtime_check_array_sizes

VARIABLES_MISSING_FROM_FORTRAN_WOUT_ADAPTER = [
    "input_extension",
    "nextcur",
    "extcur",
    "mgrid_mode",
    "am",
    "ac",
    "ai",
    "am_aux_s",
    "am_aux_f",
    "ai_aux_s",
    "ai_aux_f",
    "ac_aux_s",
    "ac_aux_f",
    "itfsq",
    "lrecon__logical__",
    "lrfp__logical__",
    "bdotb",
    "fsqt",
    "wdot",
    "currumnc",
    "currvmnc",
]
"""The complete list of variables that can be found in Fortran VMEC wout files that are
not exposed by FortranWOutAdapter."""


@runtime_check_array_sizes
def pad_and_transpose(
    arr: jt.Float[np.ndarray, "ns_minus_one mn"], mnsize: int
) -> jt.Float[np.ndarray, "mn ns_minus_one+1"]:
    return np.vstack((np.zeros(mnsize), arr)).T


class _VmecppWOutLike(Protocol):
    """A Python protocol describing a type that has the attributes of a VMEC++
    WOutFileContents object.

    There is a dapper type with same layout, and our FortranWOutAdapter below can only
    operate on types with this layout.
    """

    version: str
    sign_of_jacobian: int
    gamma: float
    pcurr_type: str
    pmass_type: str
    piota_type: str
    # NOTE: the same dim1 does NOT indicate all these arrays have the same dimensions.
    # TODO(eguiraud): give different names to each separate size
    am: jt.Float[np.ndarray, " dim1"]
    ac: jt.Float[np.ndarray, " dim1"]
    ai: jt.Float[np.ndarray, " dim1"]
    am_aux_s: jt.Float[np.ndarray, " dim1"]
    am_aux_f: jt.Float[np.ndarray, " dim1"]
    ac_aux_s: jt.Float[np.ndarray, " dim1"]
    ac_aux_f: jt.Float[np.ndarray, " dim1"]
    ai_aux_s: jt.Float[np.ndarray, " dim1"]
    ai_aux_f: jt.Float[np.ndarray, " dim1"]
    nfp: int
    mpol: int
    ntor: int
    lasym: bool
    ns: int
    ftolv: float
    maximum_iterations: int
    lfreeb: bool
    mgrid_file: str
    extcur: jt.Float[np.ndarray, " dim1"]
    mgrid_mode: str
    wb: float
    wp: float
    rmax_surf: float
    rmin_surf: float
    zmax_surf: float
    mnmax: int
    mnmax_nyq: int
    ier_flag: int
    aspect: float
    betatot: float
    betapol: float
    betator: float
    betaxis: float
    b0: float
    rbtor0: float
    rbtor: float
    IonLarmor: float
    VolAvgB: float
    ctor: float
    Aminor_p: float
    Rmajor_p: float
    volume_p: float
    fsqr: float
    fsqz: float
    fsql: float
    iota_full: jt.Float[np.ndarray, " dim1"]
    safety_factor: jt.Float[np.ndarray, " dim1"]
    pressure_full: jt.Float[np.ndarray, " dim1"]
    toroidal_flux: jt.Float[np.ndarray, " dim1"]
    phipf: jt.Float[np.ndarray, " dim1"]
    poloidal_flux: jt.Float[np.ndarray, " dim1"]
    chipf: jt.Float[np.ndarray, " dim1"]
    jcuru: jt.Float[np.ndarray, " dim1"]
    jcurv: jt.Float[np.ndarray, " dim1"]
    iota_half: jt.Float[np.ndarray, " dim1"]
    mass: jt.Float[np.ndarray, " dim1"]
    pressure_half: jt.Float[np.ndarray, " dim1"]
    beta: jt.Float[np.ndarray, " dim1"]
    buco: jt.Float[np.ndarray, " dim1"]
    bvco: jt.Float[np.ndarray, " dim1"]
    dVds: jt.Float[np.ndarray, " dim1"]
    spectral_width: jt.Float[np.ndarray, " dim1"]
    phips: jt.Float[np.ndarray, " dim1"]
    overr: jt.Float[np.ndarray, " dim1"]
    jdotb: jt.Float[np.ndarray, " dim1"]
    bdotgradv: jt.Float[np.ndarray, " dim1"]
    DMerc: jt.Float[np.ndarray, " dim1"]
    Dshear: jt.Float[np.ndarray, " dim1"]
    Dwell: jt.Float[np.ndarray, " dim1"]
    Dcurr: jt.Float[np.ndarray, " dim1"]
    Dgeod: jt.Float[np.ndarray, " dim1"]
    equif: jt.Float[np.ndarray, " dim1"]
    curlabel: list[str]
    xm: jt.Int[np.ndarray, " dim1"]
    xn: jt.Int[np.ndarray, " dim1"]
    xm_nyq: jt.Int[np.ndarray, " dim1"]
    xn_nyq: jt.Int[np.ndarray, " dim1"]
    raxis_c: jt.Float[np.ndarray, " dim1"]
    zaxis_s: jt.Float[np.ndarray, " dim1"]
    rmnc: jt.Float[np.ndarray, "dim1 dim2"]
    zmns: jt.Float[np.ndarray, "dim1 dim2"]
    lmns_full: jt.Float[np.ndarray, "dim1 dim2"]
    lmns: jt.Float[np.ndarray, "dim1 dim2"]
    gmnc: jt.Float[np.ndarray, "dim1 dim2"]
    bmnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubumnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubvmnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubsmns: jt.Float[np.ndarray, "dim1 dim2"]
    bsubsmns_full: jt.Float[np.ndarray, "dim1 dim2"]
    bsupumnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsupvmnc: jt.Float[np.ndarray, "dim1 dim2"]
    raxis_s: jt.Float[np.ndarray, " dim1"]
    zaxis_c: jt.Float[np.ndarray, " dim1"]
    rmns: jt.Float[np.ndarray, "dim1 dim2"]
    zmnc: jt.Float[np.ndarray, "dim1 dim2"]
    lmnc_full: jt.Float[np.ndarray, "dim1 dim2"]
    lmnc: jt.Float[np.ndarray, "dim1 dim2"]
    gmns: jt.Float[np.ndarray, "dim1 dim2"]
    bmns: jt.Float[np.ndarray, "dim1 dim2"]
    bsubumns: jt.Float[np.ndarray, "dim1 dim2"]
    bsubvmns: jt.Float[np.ndarray, "dim1 dim2"]
    bsubsmnc: jt.Float[np.ndarray, "dim1 dim2"]
    bsubsmnc_full: jt.Float[np.ndarray, "dim1 dim2"]
    bsupumns: jt.Float[np.ndarray, "dim1 dim2"]
    bsupvmns: jt.Float[np.ndarray, "dim1 dim2"]


class FortranWOutAdapter(pydantic.BaseModel):
    """An adapter that makes VMEC++'s WOutFileContents look like Fortran VMEC's wout.

    It can be constructed form any type that looks like a VMEC++ WOutFileContents class,
    i.e. that satisfies the _VmecppWOutLike protocol.

    FortranWOutAdapter exposes the layout that SIMSOPT expects.
    The `save` method produces a NetCDF3 file compatible with SIMSOPT/Fortran VMEC.
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

    @staticmethod
    def from_vmecpp_wout(vmecpp_wout: _VmecppWOutLike) -> FortranWOutAdapter:
        attrs = {}

        # These attributes are the same in VMEC++ and in Fortran VMEC
        attrs["ier_flag"] = vmecpp_wout.ier_flag
        attrs["nfp"] = vmecpp_wout.nfp
        attrs["ns"] = vmecpp_wout.ns
        attrs["mpol"] = vmecpp_wout.mpol
        attrs["ntor"] = vmecpp_wout.ntor
        attrs["mnmax"] = vmecpp_wout.mnmax
        attrs["mnmax_nyq"] = vmecpp_wout.mnmax_nyq
        attrs["lasym"] = vmecpp_wout.lasym
        attrs["lfreeb"] = vmecpp_wout.lfreeb
        attrs["wb"] = vmecpp_wout.wb
        attrs["wp"] = vmecpp_wout.wp
        attrs["rmax_surf"] = vmecpp_wout.rmax_surf
        attrs["rmin_surf"] = vmecpp_wout.rmin_surf
        attrs["zmax_surf"] = vmecpp_wout.zmax_surf
        attrs["aspect"] = vmecpp_wout.aspect
        attrs["betapol"] = vmecpp_wout.betapol
        attrs["betator"] = vmecpp_wout.betator
        attrs["betaxis"] = vmecpp_wout.betaxis
        attrs["b0"] = vmecpp_wout.b0
        attrs["rbtor0"] = vmecpp_wout.rbtor0
        attrs["rbtor"] = vmecpp_wout.rbtor
        attrs["IonLarmor"] = vmecpp_wout.IonLarmor
        attrs["ctor"] = vmecpp_wout.ctor
        attrs["Aminor_p"] = vmecpp_wout.Aminor_p
        attrs["Rmajor_p"] = vmecpp_wout.Rmajor_p
        attrs["volume"] = vmecpp_wout.volume_p
        attrs["fsqr"] = vmecpp_wout.fsqr
        attrs["fsqz"] = vmecpp_wout.fsqz
        attrs["fsql"] = vmecpp_wout.fsql
        attrs["phipf"] = vmecpp_wout.phipf
        attrs["chipf"] = vmecpp_wout.chipf
        attrs["jcuru"] = vmecpp_wout.jcuru
        attrs["jcurv"] = vmecpp_wout.jcurv
        attrs["jdotb"] = vmecpp_wout.jdotb
        attrs["bdotgradv"] = vmecpp_wout.bdotgradv
        attrs["DMerc"] = vmecpp_wout.DMerc
        attrs["equif"] = vmecpp_wout.equif
        attrs["xm"] = vmecpp_wout.xm
        attrs["xn"] = vmecpp_wout.xn
        attrs["xm_nyq"] = vmecpp_wout.xm_nyq
        attrs["xn_nyq"] = vmecpp_wout.xn_nyq
        attrs["ftolv"] = vmecpp_wout.ftolv
        attrs["pcurr_type"] = vmecpp_wout.pcurr_type
        attrs["pmass_type"] = vmecpp_wout.pmass_type
        attrs["piota_type"] = vmecpp_wout.piota_type
        attrs["gamma"] = vmecpp_wout.gamma
        attrs["mgrid_file"] = vmecpp_wout.mgrid_file

        # These attributes are called differently
        attrs["niter"] = vmecpp_wout.maximum_iterations
        attrs["signgs"] = vmecpp_wout.sign_of_jacobian
        attrs["betatotal"] = vmecpp_wout.betatot
        attrs["volavgB"] = vmecpp_wout.VolAvgB
        attrs["iotaf"] = vmecpp_wout.iota_full
        attrs["q_factor"] = vmecpp_wout.safety_factor
        attrs["presf"] = vmecpp_wout.pressure_full
        attrs["phi"] = vmecpp_wout.toroidal_flux
        attrs["chi"] = vmecpp_wout.poloidal_flux
        attrs["beta_vol"] = vmecpp_wout.beta
        attrs["specw"] = vmecpp_wout.spectral_width
        attrs["DShear"] = vmecpp_wout.Dshear
        attrs["DWell"] = vmecpp_wout.Dwell
        attrs["DCurr"] = vmecpp_wout.Dcurr
        attrs["DGeod"] = vmecpp_wout.Dgeod
        attrs["raxis_cc"] = vmecpp_wout.raxis_c
        attrs["zaxis_cs"] = vmecpp_wout.zaxis_s

        # These attributes have one element more in VMEC2000
        # (i.e. they have size ns instead of ns - 1).
        # VMEC2000 then indexes them as with [1:], so we pad VMEC++'s.
        # And they might be called differently.
        attrs["bvco"] = np.concatenate(([0.0], vmecpp_wout.bvco))
        attrs["buco"] = np.concatenate(([0.0], vmecpp_wout.buco))
        attrs["vp"] = np.concatenate(([0.0], vmecpp_wout.dVds))
        attrs["pres"] = np.concatenate(([0.0], vmecpp_wout.pressure_half))
        attrs["mass"] = np.concatenate(([0.0], vmecpp_wout.mass))
        attrs["beta_vol"] = np.concatenate(([0.0], vmecpp_wout.beta))
        attrs["phips"] = np.concatenate(([0.0], vmecpp_wout.phips))
        attrs["over_r"] = np.concatenate(([0.0], vmecpp_wout.overr))
        attrs["iotas"] = np.concatenate(([0.0], vmecpp_wout.iota_half))

        # These attributes are transposed in SIMSOPT
        attrs["rmnc"] = vmecpp_wout.rmnc.T
        attrs["zmns"] = vmecpp_wout.zmns.T
        attrs["bsubsmns"] = vmecpp_wout.bsubsmns.T

        # These attributes have one column less and their elements are transposed
        # in VMEC++ with respect to SIMSOPT/VMEC2000
        attrs["lmns"] = pad_and_transpose(vmecpp_wout.lmns, attrs["mnmax"])
        attrs["bmnc"] = pad_and_transpose(vmecpp_wout.bmnc, attrs["mnmax_nyq"])
        attrs["bsubumnc"] = pad_and_transpose(vmecpp_wout.bsubumnc, attrs["mnmax_nyq"])
        attrs["bsubvmnc"] = pad_and_transpose(vmecpp_wout.bsubvmnc, attrs["mnmax_nyq"])
        attrs["bsupumnc"] = pad_and_transpose(vmecpp_wout.bsupumnc, attrs["mnmax_nyq"])
        attrs["bsupvmnc"] = pad_and_transpose(vmecpp_wout.bsupvmnc, attrs["mnmax_nyq"])
        attrs["gmnc"] = pad_and_transpose(vmecpp_wout.gmnc, attrs["mnmax_nyq"])

        attrs["version_"] = float(vmecpp_wout.version)

        return FortranWOutAdapter(**attrs)

    def save(self, out_path: Path) -> None:
        """Save contents in NetCDF3 format.

        This is the format used by Fortran VMEC implementations and the one expected by
        SIMSOPT.
        """
        # protect against possible confusion between the C++ WOutFileContents::Save
        # and this method
        if out_path.suffix == ".h5":
            msg = (
                "You called `save` on a FortranWOutAdapter: this produces a NetCDF3 "
                "file, but you specified an output file name ending in '.h5', which "
                "suggests an HDF5 output was expected. Please change output filename "
                "suffix."
            )
            raise ValueError(msg)

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

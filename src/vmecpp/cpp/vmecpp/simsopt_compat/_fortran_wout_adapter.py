# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import netCDF4
import numpy as np
import numpydantic as npyd
import pydantic

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


def pad_and_transpose(
    arr: npyd.NDArray[npyd.Shape["* ns_minus_one, * mn"], float], mnsize: int
) -> npyd.NDArray[npyd.Shape["* mn, * ns"], float]:
    stacked = np.vstack((np.zeros(mnsize), arr)).T
    assert stacked.shape[1] == arr.shape[0] + 1
    assert stacked.shape[0] == arr.shape[1]
    return stacked


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
    am: npyd.NDArray[npyd.Shape["* dim1"], float]
    ac: npyd.NDArray[npyd.Shape["* dim1"], float]
    ai: npyd.NDArray[npyd.Shape["* dim1"], float]
    am_aux_s: npyd.NDArray[npyd.Shape["* dim1"], float]
    am_aux_f: npyd.NDArray[npyd.Shape["* dim1"], float]
    ac_aux_s: npyd.NDArray[npyd.Shape["* dim1"], float]
    ac_aux_f: npyd.NDArray[npyd.Shape["* dim1"], float]
    ai_aux_s: npyd.NDArray[npyd.Shape["* dim1"], float]
    ai_aux_f: npyd.NDArray[npyd.Shape["* dim1"], float]
    nfp: int
    mpol: int
    ntor: int
    lasym: bool
    ns: int
    ftolv: float
    maximum_iterations: int
    lfreeb: bool
    mgrid_file: str
    extcur: npyd.NDArray[npyd.Shape["* dim1"], float]
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
    iota_full: npyd.NDArray[npyd.Shape["* dim1"], float]
    safety_factor: npyd.NDArray[npyd.Shape["* dim1"], float]
    pressure_full: npyd.NDArray[npyd.Shape["* dim1"], float]
    toroidal_flux: npyd.NDArray[npyd.Shape["* dim1"], float]
    phipf: npyd.NDArray[npyd.Shape["* dim1"], float]
    poloidal_flux: npyd.NDArray[npyd.Shape["* dim1"], float]
    chipf: npyd.NDArray[npyd.Shape["* dim1"], float]
    jcuru: npyd.NDArray[npyd.Shape["* dim1"], float]
    jcurv: npyd.NDArray[npyd.Shape["* dim1"], float]
    iota_half: npyd.NDArray[npyd.Shape["* dim1"], float]
    mass: npyd.NDArray[npyd.Shape["* dim1"], float]
    pressure_half: npyd.NDArray[npyd.Shape["* dim1"], float]
    beta: npyd.NDArray[npyd.Shape["* dim1"], float]
    buco: npyd.NDArray[npyd.Shape["* dim1"], float]
    bvco: npyd.NDArray[npyd.Shape["* dim1"], float]
    dVds: npyd.NDArray[npyd.Shape["* dim1"], float]
    spectral_width: npyd.NDArray[npyd.Shape["* dim1"], float]
    phips: npyd.NDArray[npyd.Shape["* dim1"], float]
    overr: npyd.NDArray[npyd.Shape["* dim1"], float]
    jdotb: npyd.NDArray[npyd.Shape["* dim1"], float]
    bdotgradv: npyd.NDArray[npyd.Shape["* dim1"], float]
    DMerc: npyd.NDArray[npyd.Shape["* dim1"], float]
    Dshear: npyd.NDArray[npyd.Shape["* dim1"], float]
    Dwell: npyd.NDArray[npyd.Shape["* dim1"], float]
    Dcurr: npyd.NDArray[npyd.Shape["* dim1"], float]
    Dgeod: npyd.NDArray[npyd.Shape["* dim1"], float]
    equif: npyd.NDArray[npyd.Shape["* dim1"], float]
    curlabel: list[str]
    xm: npyd.NDArray[npyd.Shape["* dim1"], int]
    xn: npyd.NDArray[npyd.Shape["* dim1"], int]
    xm_nyq: npyd.NDArray[npyd.Shape["* dim1"], int]
    xn_nyq: npyd.NDArray[npyd.Shape["* dim1"], int]
    raxis_c: npyd.NDArray[npyd.Shape["* dim1"], float]
    zaxis_s: npyd.NDArray[npyd.Shape["* dim1"], float]
    rmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    zmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    lmns_full: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    lmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    gmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubumnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubvmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubsmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubsmns_full: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsupumnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsupvmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    raxis_s: npyd.NDArray[npyd.Shape["* dim1"], float]
    zaxis_c: npyd.NDArray[npyd.Shape["* dim1"], float]
    rmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    zmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    lmnc_full: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    lmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    gmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubumns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubvmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubsmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubsmnc_full: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsupumns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsupvmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]


class FortranWOutAdapter(pydantic.BaseModel):
    """An adapter that makes VMEC++'s WOutFileContents look like Fortran VMEC's wout.

    It can be constructed form any type that looks like a VMEC++ WOutFileContents class,
    i.e. that satisfies the _VmecppWOutLike protocol.

    FortranWOutAdapter exposes the layout that SIMSOPT expects.
    The `save` method produces a NetCDF3 file compatible with SIMSOPT/Fortran VMEC.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

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
    phipf: npyd.NDArray[npyd.Shape["* dim1"], float]
    chipf: npyd.NDArray[npyd.Shape["* dim1"], float]
    jcuru: npyd.NDArray[npyd.Shape["* dim1"], float]
    jcurv: npyd.NDArray[npyd.Shape["* dim1"], float]
    jdotb: npyd.NDArray[npyd.Shape["* dim1"], float]
    bdotgradv: npyd.NDArray[npyd.Shape["* dim1"], float]
    DMerc: npyd.NDArray[npyd.Shape["* dim1"], float]
    equif: npyd.NDArray[npyd.Shape["* dim1"], float]
    xm: npyd.NDArray[npyd.Shape["* dim1"], int]
    xn: npyd.NDArray[npyd.Shape["* dim1"], int]
    xm_nyq: npyd.NDArray[npyd.Shape["* dim1"], int]
    xn_nyq: npyd.NDArray[npyd.Shape["* dim1"], int]
    mass: npyd.NDArray[npyd.Shape["* dim1"], float]
    buco: npyd.NDArray[npyd.Shape["* dim1"], float]
    bvco: npyd.NDArray[npyd.Shape["* dim1"], float]
    phips: npyd.NDArray[npyd.Shape["* dim1"], float]
    bmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    gmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubumnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubvmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsubsmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsupumnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    bsupvmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    rmnc: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    zmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    lmns: npyd.NDArray[npyd.Shape["* dim1, * dim2"], float]
    pcurr_type: str
    pmass_type: str
    piota_type: str
    gamma: float
    mgrid_file: str

    iotas: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called iota_half."""

    iotaf: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called iota_full."""

    betatotal: float
    """In VMEC++ this is called betatot."""

    raxis_cc: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called raxis_c."""

    zaxis_cs: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called zaxis_s."""

    vp: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called dVds."""

    presf: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called pressure_full."""

    pres: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called pressure_half."""

    phi: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called toroidal_flux."""

    signgs: int
    """In VMEC++ this is called sign_of_jacobian."""

    volavgB: float
    """In VMEC++ this is called VolAvgB."""

    q_factor: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called safety_factor."""

    chi: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called poloidal_flux."""

    specw: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called spectral_width."""

    over_r: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called overr."""

    DShear: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called Dshear."""

    DWell: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called Dwell."""

    DCurr: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called Dcurr."""

    DGeod: npyd.NDArray[npyd.Shape["*"], float]
    """In VMEC++ this is called Dgeod."""

    niter: int
    """In VMEC++ this is called maximum_iterations."""

    beta_vol: npyd.NDArray[npyd.Shape["*"], float]
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

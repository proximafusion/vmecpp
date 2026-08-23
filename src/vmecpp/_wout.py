# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The VMEC++ output data models: wout, jxbout, mercier and the threed1 tables."""

from __future__ import annotations

import logging
import os
import tempfile
import types
import typing
from pathlib import Path

import jaxtyping as jt
import netCDF4
import numpy as np
import pydantic

from vmecpp._pydantic_numpy import BaseModelWithNumpy
from vmecpp._types import (
    AuxFType,
    AuxSType,
    MgridModeType,
    ProfileType,
    RestartReason,
    SerializeIntAsFloat,
)
from vmecpp.cpp import _vmecpp  # type: ignore # bindings to the C++ core

logger = logging.getLogger(__name__)


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

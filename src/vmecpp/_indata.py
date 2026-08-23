# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
""":class:`VmecInput`, the Python equivalent of a classic VMEC INDATA file."""

from __future__ import annotations

import logging
import tempfile
import typing
from pathlib import Path

import jaxtyping as jt
import numpy as np
import pydantic

from vmecpp import _util
from vmecpp._input_files import ensure_vmecpp_input
from vmecpp._pydantic_numpy import BaseModelWithNumpy
from vmecpp._types import (
    FreeBoundaryMethod,
    IterationStyle,
    MpolNtorField,
    ProfileType,
    SerializableSparseCoefficientArray,
    _final_resolution,
    _validate_free_boundary_method,
    _validate_iteration_style,
)
from vmecpp.cpp import _vmecpp  # type: ignore # bindings to the C++ core

logger = logging.getLogger(__name__)


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


def set_profile(
    vmec_input: VmecInput,
    field: typing.Literal["pressure", "iota", "current"],
    f: typing.Callable[[np.ndarray], np.ndarray],
) -> VmecInput:
    """Populate a line segment profile using callable ``f``.

    This allows users to set a precise pressure/iota/curent profile on flux
    surfaces without any precision loss by fitting the profile to e.g.
    experimental data first.

    The callable is evaluated on all unique ``s`` values required for the
    multi-grid steps (full and half grids). The resulting knots and values are
    stored in the auxiliary arrays for the chosen profile. Therefore you should
    populate the multigrid ``ns_array`` resolutions before calling this function.

    Args:
        vmec_input: The vmec input to be modified.
        field: The profile quantity to set.
        f: A callable taking an array of flux coordinates ``s`` and returning
            the value of the selected quantity (pressure/iota/current)
    Returns:
        A modified copy of the given ``VmecInput``.
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
        return vmec_input.model_copy(
            update={
                "pmass_type": "line_segment",
                "am_aux_s": knots,
                "am_aux_f": values,
                "am": np.array([]),
            }
        )
    if field == "iota":
        return vmec_input.model_copy(
            update={
                "piota_type": "line_segment",
                "ai_aux_s": knots,
                "ai_aux_f": values,
                "ai": np.array([]),
            }
        )
    if field == "current":
        return vmec_input.model_copy(
            update={
                "pcurr_type": "line_segment_i",
                "ac_aux_s": knots,
                "ac_aux_f": values,
                "ac": np.array([]),
            }
        )
    msg = "field must be one of 'pressure', 'iota', 'current'"
    raise ValueError(msg)


# Backwards compatible name
populate_raw_profile = set_profile

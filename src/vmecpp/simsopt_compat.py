# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""SIMSOPT compatibility layer for VMEC++."""

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Generator, Optional, Union, cast

import numpy as np
from jaxtyping import Bool, Float
from mpi4py import MPI
from numpy.typing import NDArray
from scipy.io import netcdf_file
from simsopt._core.optimizable import Optimizable
from simsopt._core.util import ObjectiveFailure, Struct
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.util.mpi import MpiPartition

from vmecpp import _util
from vmecpp.cpp import _vmecpp as vmec
from vmecpp.cpp.vmecpp.simsopt_compat import FortranWOutAdapter

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
MPI_PARTITION = MpiPartition(ngroups=1)


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
    indata: Optional[vmec.VmecINDATAPyWrapper]
    # non-null if Vmec was initialized from an input file
    input_file: Optional[str]
    # non-null if Vmec was initialized from an output file
    output_file: Optional[str]
    # These are filled:
    # - by __init__ if Vmec is initialized with an output file
    # - by a call to run() and are None before otherwise
    s_full_grid = Optional[Float[np.ndarray, " ns"]]
    ds = Optional[Float]
    s_half_grid = Optional[Float[np.ndarray, " nshalf"]]

    # The loaded run results (or None if no results are present yet):
    # - a SIMSOPT Struct when reading an output file
    # - a FortranWOutAdapter when reading the results of a VMEC++ run
    wout: Optional[Union[Struct, FortranWOutAdapter]]
    # Whether `run()` is available for this object:
    # depends on whether it has been initialized with an input configuration
    # or an output file.
    runnable: bool
    # False when the currently cached results are valid, True if we need to `run()`
    need_to_run_code: bool
    mpi: Optional[MpiPartition]  # pyright: ignore
    verbose: bool

    def __init__(
        self,
        filename: str | Path,
        verbose: bool = True,
        ntheta: int = 50,
        nphi: int = 50,
        range_surface: str = "full torus",
        mpi: Optional[MpiPartition] = None,  # pyright: ignore
        keep_all_files: bool = False,
    ):
        self.verbose = verbose

        if mpi is not None:
            logging.warning(
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
            with ensure_vmecpp_input(Path(filename)) as vmecpp_filename:
                logger.debug(
                    f"Initializing a VMEC object from input file: {vmecpp_filename}"
                )
                self.indata = vmec.VmecINDATAPyWrapper.from_file(vmecpp_filename)
            assert self.indata is not None  # for pyright

            self.runnable = True
            self.need_to_run_code = True
            # intentionally using the original `filename` and not the potentially
            # different `vmecpp_filename` here: we want to behave as if the input
            # was `filename`, even if internally we converted it.
            self.input_file = str(filename)
            self.iter = -1

            # NOTE: SurfaceRZFourier uses m up to mpol _inclusive_,
            # differently from VMEC++, so have to manually reduce the range by one.
            mpol_for_surfacerzfourier = self.indata.mpol - 1

            # A vmec object has mpol and ntor attributes independent of
            # the boundary. The boundary surface object is initialized
            # with mpol and ntor values that match those of the vmec
            # object, but the mpol/ntor values of either the vmec object
            # or the boundary surface object can be changed independently
            # by the user.
            self._boundary = SurfaceRZFourier.from_nphi_ntheta(
                nfp=self.indata.nfp,
                stellsym=not self.indata.lasym,
                mpol=mpol_for_surfacerzfourier,
                ntor=self.indata.ntor,
                ntheta=ntheta,
                nphi=nphi,
                range=range_surface,
            )
            self.free_boundary = bool(self.indata.lfreeb)

            # Transfer boundary shape data from indata to _boundary:
            vi = self.indata
            for m in range(vi.mpol):
                for n in range(2 * vi.ntor + 1):
                    self._boundary.rc[m, n] = vi.rbc[m, n]
                    self._boundary.zs[m, n] = vi.zbs[m, n]
                    if vi.lasym:
                        self._boundary.rs[m, n] = vi.rbs[m, n]
                        self._boundary.zc[m, n] = vi.zbc[m, n]
            self._boundary.local_full_x = cast(
                Float[np.ndarray, "..."], self._boundary.get_dofs()
            )

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
        x0 = cast(Float[np.ndarray, "..."], self.get_dofs())
        fixed = cast(Bool[np.ndarray, "..."], np.full(len(x0), True))
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

    def run(self, initial_state=None, max_threads=None) -> None:
        """Run VMEC if ``need_to_run_code`` is ``True``.

        The max_threads argument is not present in SIMSOPT's original implementation as
        it is specific to VMEC++, which will spawn the corresponding number of OpenMP
        threads to parallelize execution. If max_threads is None (the default), VMEC++
        runs on a single thread.
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
        if initial_state is not None:
            # we are going to perform a hot restart, so we are only going to
            # run the last of the multi-grid steps: adapt indata accordingly
            indata = self.indata.copy()
            indata.ns_array = indata.ns_array[-1:]
            indata.ftol_array = indata.ftol_array[-1:]
            indata.niter_array = indata.niter_array[-1:]

        logger.debug("Running VMEC++")

        if max_threads is None:
            # Starfinder does its own multi-process parallelization, and we do not want
            # to end up overcommitting the machine with NCPU processes running NCPU
            # threads each -- especially when OpenMP is involved, as OpenMP threads are
            # generally bad at resource-sharing. So we set max_threads=1 by default.
            max_threads = 1
        try:
            output_quantities = vmec.run(
                indata,
                initial_state=initial_state,
                max_threads=max_threads,
                verbose=self.verbose,
            )
        except RuntimeError as e:
            msg = f"Error while running VMEC++: {e}"
            raise ObjectiveFailure(msg) from e
        self.output_quantities = output_quantities
        self.wout = FortranWOutAdapter.from_vmecpp_wout(output_quantities.wout)

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

        self.wout = Struct()
        # to make pyright happy (not sure why the previous statement is not enough)
        assert self.wout is not None
        with netcdf_file(str(self.output_file), mmap=False) as f:
            for key, val in f.variables.items():
                # 2D arrays need to be transposed.
                val2 = val[()]  # Convert to numpy array
                val3 = val2.T if len(val2.shape) == 2 else val2
                setattr(self.wout, key, val3)

            if self.wout.ier_flag != 0:
                logger.warning("VMEC did not succeed!")
                msg = "VMEC did not succeed"
                raise ObjectiveFailure(msg)

            # Shorthands for long variable names:
            self.wout.lasym = f.variables["lasym__logical__"][()]
            self.wout.volume = self.wout.volume_p

        self._set_grid()

    def _set_grid(self) -> None:
        assert self.wout is not None
        self.s_full_grid = np.linspace(0, 1, self.wout.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds

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

    def get_dofs(self) -> NDArray:
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

    def set_indata(self) -> None:
        """Transfer data from simsopt objects to vmec.indata.

        Presently, this function sets the boundary shape and magnetic
        axis shape.  In the future, the input profiles will be set here
        as well. This data transfer is performed before writing a Vmec
        input file or running Vmec. The boundary surface object
        converted to ``SurfaceRZFourier`` is returned.
        """
        if not self.runnable:
            msg = "Cannot access indata for a Vmec object that was initialized from a wout file."
            raise RuntimeError(msg)
        assert self.indata is not None
        vi = self.indata  # Shorthand
        # Convert boundary to RZFourier if needed:
        boundary_RZFourier = self.boundary.to_RZFourier()
        vi.rbc.fill(0.0)
        vi.zbs.fill(0.0)

        # Transfer boundary shape data from the surface object to VMEC:
        ntor = self.indata.ntor
        for m in range(self.indata.mpol):
            for n in range(2 * ntor + 1):
                vi.rbc[m, n] = boundary_RZFourier.get_rc(m, n - ntor)
                vi.zbs[m, n] = boundary_RZFourier.get_zs(m, n - ntor)

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
        return self.indata.to_json()

    def write_input(self, filename: str) -> None:
        """Write a VMEC++ input file (in JSON format).

        To just get the result as a string without saving a file, see
        the ``get_input()`` function.
        """
        indata_json = self.get_input()
        with open(filename, "w") as f:
            f.write(indata_json)

    def set_mpol_ntor(self, new_mpol: int, new_ntor: int):
        assert self.indata is not None
        self.indata._set_mpol_ntor(new_mpol, new_ntor)
        # NOTE: SurfaceRZFourier uses m up to mpol _inclusive_,
        # differently from VMEC++, so have to manually reduce the range by one.
        mpol_for_surfacerzfourier = new_mpol - 1
        self.boundary.change_resolution(mpol_for_surfacerzfourier, new_ntor)
        self.recompute_bell()


def is_vmec2000_input(input_file: Path) -> bool:
    """Returns true if the input file looks like a Fortran VMEC/VMEC2000 INDATA file."""
    # we peek at the first few characters in the file: if they are "&INDATA",
    # this is an INDATA file
    with open(input_file) as f:
        first_line = f.readline().strip()
    return first_line == "&INDATA"


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
        assert vmecpp_input_path == output_file
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

    indata_contents = _vmecpp_json_to_indata(vmecpp_json_dict)

    # Otherwise we actually need to perform the JSON -> INDATA conversion.
    # We need the try/finally in order to correctly clean up after
    # ourselves even in case of errors raised from the body of the `with`
    # in user code.
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / indata_file
        with open(out_path, "w") as out_f:
            out_f.write(indata_contents)
        yield out_path


# adapted from https://github.com/jonathanschilling/indata2json/blob/4274976/json2indata
def _vmecpp_json_to_indata(vmecpp_json: dict[str, Any]) -> str:
    """Convert a dictionary with the contents of a VMEC++ JSON input file to the
    corresponding conents of a VMEC2000 INDATA file."""

    indata: str = "&INDATA\n"

    indata += "\n  ! numerical resolution, symmetry assumption\n"
    indata += _bool_to_namelist("lasym", vmecpp_json)
    for varname in ("nfp", "mpol", "ntor", "ntheta", "nzeta"):
        indata += _int_to_namelist(varname, vmecpp_json)

    indata += "\n  ! multi-grid steps\n"
    indata += _int_array_to_namelist("ns_array", vmecpp_json)
    indata += _float_array_to_namelist("ftol_array", vmecpp_json)
    indata += _int_array_to_namelist("niter_array", vmecpp_json)

    indata += "\n  ! solution method tweaking parameters\n"
    indata += _float_to_namelist("delt", vmecpp_json)
    indata += _float_to_namelist("tcon0", vmecpp_json)
    indata += _float_array_to_namelist("aphi", vmecpp_json)
    indata += _bool_to_namelist("lforbal", vmecpp_json)

    indata += "\n  ! printout interval\n"
    indata += _int_to_namelist("nstep", vmecpp_json)

    indata += "\n  ! total enclosed toroidal magnetic flux\n"
    indata += _float_to_namelist("phiedge", vmecpp_json)

    indata += "\n  ! mass / pressure profile\n"
    indata += _string_to_namelist("pmass_type", vmecpp_json)
    indata += _float_array_to_namelist("am", vmecpp_json)
    indata += _float_array_to_namelist("am_aux_s", vmecpp_json)
    indata += _float_array_to_namelist("am_aux_f", vmecpp_json)
    indata += _float_to_namelist("pres_scale", vmecpp_json)
    indata += _float_to_namelist("gamma", vmecpp_json)
    indata += _float_to_namelist("spres_ped", vmecpp_json)

    indata += "\n  ! select constraint on iota or enclosed toroidal current profiles\n"
    indata += _int_to_namelist("ncurr", vmecpp_json)

    indata += "\n  ! (initial guess for) iota profile\n"
    indata += _string_to_namelist("piota_type", vmecpp_json)
    indata += _float_array_to_namelist("ai", vmecpp_json)
    indata += _float_array_to_namelist("ai_aux_s", vmecpp_json)
    indata += _float_array_to_namelist("ai_aux_f", vmecpp_json)

    indata += "\n  ! enclosed toroidal current profile\n"
    indata += _string_to_namelist("pcurr_type", vmecpp_json)
    indata += _float_array_to_namelist("ac", vmecpp_json)
    indata += _float_array_to_namelist("ac_aux_s", vmecpp_json)
    indata += _float_array_to_namelist("ac_aux_f", vmecpp_json)
    indata += _float_to_namelist("curtor", vmecpp_json)
    indata += _float_to_namelist("bloat", vmecpp_json)

    indata += "\n  ! free-boundary parameters\n"
    indata += _bool_to_namelist("lfreeb", vmecpp_json)
    indata += _string_to_namelist("mgrid_file", vmecpp_json)
    indata += _float_array_to_namelist("extcur", vmecpp_json)
    indata += _int_to_namelist("nvacskip", vmecpp_json)

    indata += "\n  ! initial guess for magnetic axis\n"
    indata += _float_array_to_namelist("raxis_cc", vmecpp_json)
    indata += _float_array_to_namelist("zaxis_cs", vmecpp_json)
    indata += _float_array_to_namelist("raxis_cs", vmecpp_json)
    indata += _float_array_to_namelist("zaxis_cc", vmecpp_json)

    indata += "\n  ! (initial guess for) boundary shape\n"
    indata += _fourier_coefficients_to_namelist("rbc", vmecpp_json)
    indata += _fourier_coefficients_to_namelist("zbs", vmecpp_json)
    indata += _fourier_coefficients_to_namelist("rbs", vmecpp_json)
    indata += _fourier_coefficients_to_namelist("zbc", vmecpp_json)

    indata += "\n/\n&END\n"

    return indata


def _bool_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname not in vmecpp_json:
        return ""

    return f"  {varname} = {'.true.' if vmecpp_json[varname] else '.false.'}\n"


def _string_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname not in vmecpp_json:
        return ""

    return f"  {varname} = '{vmecpp_json[varname]}'\n"


def _int_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname not in vmecpp_json:
        return ""

    return f"  {varname} = {vmecpp_json[varname]}\n"


def _float_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname not in vmecpp_json:
        return ""

    return f"  {varname} = {vmecpp_json[varname]:.20e}\n"


def _int_array_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname in vmecpp_json and len(vmecpp_json[varname]) > 0:
        elements = ", ".join(map(str, vmecpp_json[varname]))
        return f"  {varname} = {elements}\n"
    return ""


def _float_array_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname in vmecpp_json and len(vmecpp_json[varname]) > 0:
        elements = ", ".join([f"{x:.20e}" for x in vmecpp_json[varname]])
        return f"  {varname} = {elements}\n"
    return ""


def _fourier_coefficients_to_namelist(varname: str, vmecpp_json: dict[str, Any]) -> str:
    if varname in vmecpp_json and len(vmecpp_json[varname]) > 0:
        out = ""
        for coefficient in vmecpp_json[varname]:
            m = coefficient["m"]
            n = coefficient["n"]
            value = coefficient["value"]
            out += f"  {varname}({n}, {m}) = {value:.20e}\n"
        return out
    return ""


def _make_wout_filename(input_file: str) -> str:
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

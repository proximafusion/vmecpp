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
    mpi: Optional[MpiPartition]  # pyright: ignore # noqa: UP007
    verbose: bool

    def __init__(
        self,
        filename: str | Path,
        verbose: bool = True,
        ntheta: int = 50,
        nphi: int = 50,
        range_surface: str = "full torus",
        mpi: Optional[MpiPartition] = None,  # pyright: ignore  # noqa: UP007
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
        boundary_RZFourier.change_resolution(self.indata.mpol, self.indata.ntor)
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
        indata_wrapper = self.indata._to_cpp_vmecindatapywrapper()
        indata_wrapper._set_mpol_ntor(new_mpol, new_ntor)
        self.indata = vmecpp.VmecInput._from_cpp_vmecindatapywrapper(indata_wrapper)

        # NOTE: SurfaceRZFourier uses m up to mpol _inclusive_,
        # differently from VMEC++, so have to manually reduce the range by one.
        mpol_for_surfacerzfourier = new_mpol - 1
        self.boundary.change_resolution(mpol_for_surfacerzfourier, new_ntor)
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

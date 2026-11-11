# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Coil shapes and currents of a quasi-axisymmetric stellarator for quasisymmetry on its
free boundary: the coils and the boundary move together under pressure balance, with
the VMEC state in force balance inside every boundary.

The coils are those of Giuliani et al. (J. Plasma Phys. 88, 2022) that ship with
SIMSOPT. They are moved at random and optimized from there, and VMEC++'s own
free-boundary run in the Biot-Savart field of the coils checks each result.
"""

import argparse
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from simsopt import configs
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import QfmSurface, SurfaceRZFourier, ToroidalFlux, Volume
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual

import vmecpp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_free_boundary import (  # type: ignore
    FreeBoundaryProblem,
    converge,
    optimize,
    shape_modes,
)

TUBE = 0.12  # radius of the tube around the axis that the surface starts from, in m


class CoilShapes:
    """The Biot-Savart field of coils made from base curves by the symmetries of a
    stellarator.

    The controls are the currents of the base curves, followed by the Fourier
    coefficients of each curve up to ``order``.
    """

    def __init__(self, curves, currents, nfp: int, order: int) -> None:
        self.curves = curves
        self.currents = [Current(float(value)) for value in currents]
        self.coils = coils_via_symmetries(curves, self.currents, nfp, True)
        self.biot_savart = BiotSavart(self.coils)
        self.free = [
            np.array(
                [
                    index
                    for index, name in enumerate(curve.local_dof_names)
                    if int(name[name.index("(") + 1 : -1]) <= order
                ]
            )
            for curve in curves
        ]
        self.lock = threading.Lock()
        self.applied = self.controls()

    def controls(self) -> np.ndarray:
        """The controls of the coils as they are."""
        return np.concatenate(
            [
                [float(current.get_value()) for current in self.currents],
                *[
                    curve.x[free]
                    for curve, free in zip(self.curves, self.free, strict=True)
                ],
            ]
        )

    def scales(self, controls, fraction: float, length: float) -> np.ndarray:
        """A fraction of each current of the controls and a length for every
        coefficient."""
        count = len(self.currents)
        return np.concatenate(
            [
                fraction * np.abs(controls[:count]),
                np.full(controls.size - count, length),
            ]
        )

    def _apply(self, controls) -> None:
        if np.array_equal(controls, self.applied):
            return
        count = len(self.currents)
        for current, value in zip(self.currents, controls[:count], strict=True):
            current.x = np.array([value])
        offset = count
        for curve, free in zip(self.curves, self.free, strict=True):
            coefficients = curve.x.copy()
            coefficients[free] = controls[offset : offset + free.size]
            curve.x = coefficients
            offset += free.size
        self.applied = np.array(controls, dtype=float)

    def __call__(self, controls):
        """The magnetic field and the coil currents that vmecpp.run takes."""
        controls = np.array(controls, dtype=float)

        def field(r, phi, z):
            r, phi, z = (np.asarray(value, dtype=float) for value in (r, phi, z))
            cosine, sine = np.cos(phi), np.sin(phi)
            points = np.stack([r * cosine, r * sine, z], axis=-1).reshape(-1, 3)
            # the coils are shared by the threads of the vacuum side
            with self.lock:
                self._apply(controls)
                self.biot_savart.set_points(np.ascontiguousarray(points))
                b = np.array(self.biot_savart.B()).reshape(*r.shape, 3)
            return (
                b[..., 0] * cosine + b[..., 1] * sine,
                -b[..., 0] * sine + b[..., 1] * cosine,
                b[..., 2],
            )

        return field, np.ones(1)


def giuliani(mpol: int, ntor: int, ns: int, order: int):
    """The field of the coils of Giuliani et al. and a fixed-boundary vacuum input
    inside a flux surface of that field.

    The surface minimizes the quadratic flux of the coil field, from a tube around its
    magnetic axis.
    """
    curves, currents, axis, nfp = configs.get_data("giuliani")[:4]
    field = CoilShapes(
        curves, [current.get_value() for current in currents], nfp, order
    )
    surface = SurfaceRZFourier(
        mpol=3,
        ntor=3,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 11, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 11, endpoint=False),
    )
    surface.fit_to_curve(axis, TUBE, flip_theta=True)
    volume = Volume(surface)
    QfmSurface(
        field.biot_savart, surface, volume, volume.J()
    ).minimize_qfm_penalty_constraints_LBFGS(
        tol=1.0e-12, maxiter=100, constraint_weight=1.0e-3
    )
    vmec_input = vmecpp.VmecInput()
    vmec_input.lasym = False
    vmec_input.nfp = nfp
    vmec_input.mpol, vmec_input.ntor = mpol, ntor
    vmec_input.ntheta, vmec_input.nzeta = 0, 4 * ntor
    vmec_input.ns_array = np.array([ns])
    vmec_input.ftol_array = np.array([1.0e-12])
    vmec_input.niter_array = np.array([20000])
    vmec_input.phiedge = ToroidalFlux(surface, BiotSavart(field.coils)).J()
    vmec_input.gamma = 0.0
    vmec_input.pmass_type = "power_series"
    vmec_input.am = np.zeros(1)
    vmec_input.pres_scale = 0.0
    vmec_input.ncurr = 1
    vmec_input.pcurr_type = "power_series"
    vmec_input.ac = np.zeros(1)
    vmec_input.curtor = 0.0
    vmec_input.lfreeb = False
    vmec_input.mgrid_file = "NONE"
    vmec_input.extcur = np.ones(1)
    vmec_input.nvacskip = 6
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(surface.rc, mpol, ntor)
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(surface.zs, mpol, ntor)
    vmec_input.raxis_c = np.zeros(ntor + 1)
    vmec_input.zaxis_s = np.zeros(ntor + 1)
    vmec_input.raxis_c[0] = surface.get_rc(0, 0)
    vmec_input.return_outputs_even_if_not_converged = True
    return vmec_input, field


def moved_to(problem, point, controls, stages: int) -> dict:
    """The converged free boundary at other controls, reached in stages from a converged
    point with its sensitivity as the predictor."""
    first = point["controls"]
    for stage in range(1, stages + 1):
        target = first + (controls - first) * stage / stages
        move = point["sensitivity"] @ ((target - point["controls"]) / problem.scales)
        point = converge(
            problem,
            point["shape"] + move,
            target,
            problem.carried(point["state"], move, point["tangents"]),
        )
    return point


def summary(problem, point) -> str:
    """The quasisymmetry error, the aspect ratio and the mean iota of a point."""
    parts = problem.parts(np.concatenate([point["residuals"], point["jump"]]))
    return (
        f"QS {parts['quasisymmetry']:.4e}, aspect {parts['aspect']:.5f}, mean iota"
        f" {parts['mean_iota']:.5f}"
    )


def check(problem, point) -> None:
    """Print the quasisymmetry error by SIMSOPT, the aspect ratio and the mean iota of
    VMEC++'s own free-boundary run from the boundary of a point, in the field of its
    controls."""
    boundary = problem.boundary_of(point["shape"])
    magnetic_field, currents = problem.field(point["controls"])
    wout = vmecpp.run(
        problem.vmec_input.model_copy(
            update={
                "lfreeb": True,
                "rbc": boundary[0],
                "zbs": boundary[1],
                "extcur": currents,
            }
        ),
        magnetic_field,
        verbose=False,
        max_threads=1,
    ).wout
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "wout_check.nc"
        wout.save(path)
        quasisymmetry = QuasisymmetryRatioResidual(
            Vmec(str(path)),
            np.asarray(problem.surfaces),
            helicity_m=problem.helicity_m,
            helicity_n=problem.helicity_n,
        ).total()
    print(
        f"  VMEC++'s free-boundary run: force residual"
        f" {max(wout.fsqr, wout.fsqz, wout.fsql):.1e}, QS {quasisymmetry:.4e}, aspect"
        f" {wout.aspect:.5f}, mean iota {np.mean(np.asarray(wout.iotas)[1:]):.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, default=15)
    parser.add_argument("--mpol", type=int, default=5, help="mpol and ntor")
    parser.add_argument(
        "--order", type=int, default=1, help="highest free harmonic of each coil"
    )
    parser.add_argument(
        "--shift", type=float, default=3.0, help="largest move of a coefficient, in mm"
    )
    parser.add_argument(
        "--percent", type=float, default=1.0, help="largest move of a current, in %%"
    )
    parser.add_argument(
        "--tether", type=float, default=1.0e-3, help="weight of the moves of the coils"
    )
    args = parser.parse_args()
    vmec_input, field = giuliani(args.mpol, args.mpol, args.ns, args.order)
    wout = vmecpp.run(vmec_input, max_threads=1, verbose=False).wout
    design = field.controls()
    problem = FreeBoundaryProblem(
        vmec_input,
        aspect=float(wout.aspect),
        mean_iota=float(np.mean(np.asarray(wout.iotas)[1:])),
        ntheta=24,
        nphi=24,
        field=field,
        scales=field.scales(design, 0.05, 5.0e-3),
        modes=shape_modes(args.mpol - 1, args.mpol),
        surfaces=tuple(np.arange(1, 11) / 10.0),
    )
    own = converge(problem, np.zeros(len(problem.modes)), design, None)
    print(f"coils as they are: {summary(problem, own)}")
    check(problem, own)
    generator = np.random.default_rng(0)
    count = len(field.currents)
    moved = design.copy()
    moved[:count] *= 1.0 + 0.01 * args.percent * generator.uniform(-1.0, 1.0, count)
    moved[count:] += (
        1.0e-3 * args.shift * generator.uniform(-1.0, 1.0, moved.size - count)
    )
    start = moved_to(problem, own, moved, 4)
    print(
        f"coils moved by up to {args.shift} mm and {args.percent}%:"
        f" {summary(problem, start)}"
    )
    check(problem, start)
    problem.anchor, problem.tether = moved, args.tether
    start = problem.linearize(start["shape"], moved, start["state"])
    for name, nested in (("nested", True), ("joint", False)):
        problem.linearizations = 0
        begin = time.perf_counter()
        result = optimize(problem, start, nested=nested, log=print)
        seconds = time.perf_counter() - begin
        used = problem.linearizations
        final = converge(problem, result["shape"], result["controls"], result["state"])
        change = np.abs(final["controls"] - moved)
        print(
            f"{name}: {used} linearizations, {seconds:.1f} s, coils"
            f" {change[count:].max() * 1e3:.2f} mm and"
            f" {(change[:count] / np.abs(moved[:count])).max() * 100:.2f}% from the"
            f" moved ones, on its converged free boundary {summary(problem, final)}"
        )
        check(problem, final)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Quasi-axisymmetry from a rotating ellipse by vmecpp.joint and by the nested baseline.

A cold, fully converged run read by SIMSOPT checks each result.
"""

import argparse
import tempfile
from pathlib import Path

import jax
import numpy as np
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual

import vmecpp
from vmecpp.joint import QuasisymmetryProblem, converged_input, optimize

jax.config.update("jax_enable_x64", True)

CTH_LIKE = Path(__file__).parent / "data" / "cth_like_fixed_bdy.json"

CONSTRAINT_TOLERANCE = 1.0e-3
"""Largest miss of the aspect ratio or the mean iota that counts as met."""


def rotating_ellipse(mpol: int, ntor: int, ns: int, nfp: int = 2) -> vmecpp.VmecInput:
    """A vacuum-like input with no pressure or current inside a rotating ellipse."""
    vmec_input = vmecpp.VmecInput.from_file(CTH_LIKE)
    vmec_input.nfp, vmec_input.mpol, vmec_input.ntor = nfp, mpol, ntor
    vmec_input.ntheta, vmec_input.nzeta = 0, 0
    rbc, zbs = np.zeros((mpol, 2 * ntor + 1)), np.zeros((mpol, 2 * ntor + 1))
    rbc[0, ntor] = 1.0
    rbc[1, ntor], zbs[1, ntor] = 0.16, 0.16
    rbc[1, ntor + 1], zbs[1, ntor + 1] = 0.05, -0.05
    vmec_input.rbc, vmec_input.zbs = rbc, zbs
    vmec_input.raxis_c = np.eye(1, ntor + 1)[0]
    vmec_input.zaxis_s = np.zeros(ntor + 1)
    vmec_input.ncurr, vmec_input.curtor = 1, 0.0
    vmec_input.pres_scale, vmec_input.phiedge = 0.0, 0.08
    vmec_input.pmass_type = "power_series"
    vmec_input.am, vmec_input.ac, vmec_input.ai = (np.zeros(3) for _ in range(3))
    vmec_input.ns_array = np.asarray([ns])
    vmec_input.ftol_array = np.asarray([1.0e-11])
    vmec_input.niter_array = np.asarray([30000])
    vmec_input.always_fix_m1_gauge = True
    vmec_input.return_outputs_even_if_not_converged = True
    return vmec_input


def check(problem: QuasisymmetryProblem, boundary: np.ndarray) -> None:
    """Print the QS error, the targets and any unmet constraint of a cold run."""
    wout = vmecpp.run(
        converged_input(problem, boundary, pin_gauge=True),
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
    mean_iota = float(np.mean(np.asarray(wout.iotas)[1:]))
    misses = {
        "aspect": wout.aspect - problem.aspect,
        "mean iota": mean_iota - problem.mean_iota,
    }
    unmet = [name for name, miss in misses.items() if abs(miss) > CONSTRAINT_TOLERANCE]
    print(
        f"  converged run: force residual {max(wout.fsqr, wout.fsqz, wout.fsql):.1e},"
        f" QS {quasisymmetry:.3e}, aspect {wout.aspect:.5f},"
        f" mean iota {mean_iota:.5f}, unmet constraints: {', '.join(unmet) or 'none'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, default=25)
    parser.add_argument("--mpol", type=int, default=5)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--helicity-n", type=int, default=0)
    parser.add_argument("--aspect", type=float, default=6.0)
    parser.add_argument("--iota", type=float, default=0.42)
    parser.add_argument("--pressure", type=float, default=0.0, help="on axis")
    args = parser.parse_args()
    for name, nested in (("nested", True), ("joint", False)):
        vmec_input = rotating_ellipse(args.mpol, args.mpol, args.ns, args.nfp)
        vmec_input.pres_scale = args.pressure
        vmec_input.am = np.asarray([1.0, -1.0, 0.0])
        problem = QuasisymmetryProblem(
            vmec_input,
            helicity_n=args.helicity_n,
            aspect=args.aspect,
            mean_iota=args.iota,
            max_mode=max(args.stages),
        )
        result = optimize(problem, tuple(args.stages), nested=nested, log=print)
        costs = problem.costs
        print(
            f"{name}: {len(result.steps)} steps, {costs.force_evaluations} force"
            f" evaluations, {result.steps[-1].seconds:.1f} s, of which solves"
            f" {costs.solve_seconds:.1f} s and {costs.assemblies} assemblies"
            f" {costs.assembly_seconds:.1f} s"
        )
        check(problem, result.boundary)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Quasi-axisymmetry at a pressure within the Mercier criterion by vmecpp.joint and by
the nested baseline.

The rotating ellipse of examples/joint_optimization.py carries a pressure that falls
linearly to the edge. Its quasisymmetry run without constraints ends unstable by the
Mercier criterion on every surface. Here ``DMerc`` of VMEC stays above a bound on the
surfaces between two radii as constraints. A cold, fully converged run on more
surfaces, read by SIMSOPT, checks each result.
"""

import argparse
import dataclasses
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual

import vmecpp
from vmecpp import autodiff_wout
from vmecpp.joint import QuasisymmetryProblem, converged_input, optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_optimization import CONSTRAINT_TOLERANCE, rotating_ellipse  # type: ignore

jax.config.update("jax_enable_x64", True)


@dataclasses.dataclass(kw_only=True)
class MercierStableProblem(QuasisymmetryProblem):
    """The targets and the quasisymmetry residuals of the base class, with ``DMerc``
    held above ``mercier`` on every surface of the full grid between the two radii
    ``stable``, in normalized toroidal flux.

    None leaves ``DMerc`` free.
    """

    mercier: float | None = 0.0
    stable: tuple[float, float] = (0.1, 0.9)

    def __post_init__(self) -> None:
        full = np.linspace(0.0, 1.0, int(np.asarray(self.vmec_input.ns_array)[-1]))
        self._stable = np.flatnonzero(
            (full >= self.stable[0]) & (full <= self.stable[1])
        )
        if self.mercier is not None:
            self.lower = np.full(self._stable.size, self.mercier)
        super().__post_init__()

    def constraints(self, solved, boundary):  # noqa: ARG002
        """``DMerc`` of each surface between the radii."""
        if self.mercier is None:
            return jnp.zeros(0)
        wout = autodiff_wout.wout_quantities(solved, self.vmec_input)
        return wout["DMerc"][self._stable]


def stability(problem: MercierStableProblem, wout) -> dict[str, float]:
    """The smallest and the largest ``DMerc`` of a wout on its surfaces between the
    radii of the problem, with the number of them below 0."""
    full = np.linspace(0.0, 1.0, wout.ns)
    inside = (full >= problem.stable[0]) & (full <= problem.stable[1])
    mercier = np.asarray(wout.DMerc)[inside]
    return {
        "lowest": float(mercier.min()),
        "highest": float(mercier.max()),
        "unstable": int((mercier < 0.0).sum()),
        "surfaces": int(mercier.size),
    }


def check(problem: MercierStableProblem, boundary: np.ndarray) -> None:
    """Print the QS error, the targets and the stability of a cold run."""
    cold = converged_input(problem, boundary, pin_gauge=True).model_copy(
        update={"return_outputs_even_if_not_converged": False}
    )
    try:
        wout = vmecpp.run(cold, verbose=False, max_threads=1).wout
    except RuntimeError:
        print("  converged run: VMEC++ did not converge")
        return
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
    found = stability(problem, wout)
    print(
        f"  converged run: force residual {max(wout.fsqr, wout.fsqz, wout.fsql):.1e},"
        f" QS {quasisymmetry:.3e}, aspect {wout.aspect:.5f}, mean iota"
        f" {mean_iota:.5f}, beta {wout.betatotal:.5f}, DMerc from"
        f" {found['lowest']:+.3e} to {found['highest']:+.3e} with"
        f" {found['unstable']} of {found['surfaces']} surfaces below 0, unmet"
        f" targets: {', '.join(unmet) or 'none'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, default=25)
    parser.add_argument("--mpol", type=int, default=5)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--aspect", type=float, default=6.0)
    parser.add_argument("--iota", type=float, default=0.42)
    parser.add_argument("--pressure", type=float, default=6000.0, help="on axis")
    parser.add_argument("--mercier", type=float, default=0.0, help="lowest DMerc")
    parser.add_argument(
        "--stable",
        type=float,
        nargs=2,
        default=[0.1, 0.9],
        help="radii between which DMerc is held, in normalized toroidal flux",
    )
    args = parser.parse_args()
    for name, nested, mercier in (
        ("joint without the constraints", False, None),
        ("nested", True, args.mercier),
        ("joint", False, args.mercier),
    ):
        vmec_input = rotating_ellipse(args.mpol, args.mpol, args.ns)
        vmec_input.pres_scale = args.pressure
        vmec_input.am = np.asarray([1.0, -1.0, 0.0])
        problem = MercierStableProblem(
            vmec_input,
            aspect=args.aspect,
            mean_iota=args.iota,
            max_mode=max(args.stages),
            mercier=mercier,
            stable=tuple(args.stable),
        )
        result = optimize(problem, tuple(args.stages), nested=nested, log=print)
        costs = problem.costs
        print(
            f"{name}: {len(result.steps)} steps, {costs.force_evaluations} force"
            f" evaluations, {result.steps[-1].seconds:.1f} s, quasisymmetry"
            f" {result.evaluation.parts['quasisymmetry']:.3e} on these modes"
        )
        check(problem, result.boundary)


if __name__ == "__main__":
    main()

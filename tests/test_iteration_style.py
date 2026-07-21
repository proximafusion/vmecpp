# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Compare iteration_style="vmecpp" against the vmec_8_52 baseline.

The "vmecpp" style bundles the convergence improvements from
docs/convergence_study.md (cubic multigrid transfer, reduced-delt stage entry
with time-step recovery and seed preservation) behind the single
iteration_style input switch.
These tests change ONLY iteration_style and check, on fixed-boundary and
finite-beta free-boundary cases:

- the converged physics is the same (none of the scheme's components moves
  the force balance: interpolation only changes stage seeds, time-step
  control only changes the trajectory, and a preconditioner rescaling cannot
  move the fixed point);
- convergence quality does not degrade (converged within the same iteration
  budget, final residuals at tolerance);
- the iteration count does not regress materially.
"""

from pathlib import Path

import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"

# (filename, free_boundary): the free-boundary cases carry finite beta
# (cth_like: pres_scale 432 with net toroidal current; solovev: am pressure
# profile), so they exercise the pressure-consistent vacuum matching under
# the new scheme. cth_like_free_bdy_multigrid additionally crosses a
# multigrid transition, engaging the cubic transfer + reduced-delt entry +
# recovery machinery; the single-stage cases (delt <= 1) reduce to the 8.52
# control and must be bit-identical.
CASES = [
    ("cma.json", False),
    ("cth_like_free_bdy.json", True),
    ("cth_like_free_bdy_multigrid.json", True),
    ("solovev_free_bdy.json", True),
]


def _run_with_style(filename: str, free_boundary: bool, style: str):
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / filename)
    vmec_input.iteration_style = vmecpp.IterationStyle(style)
    if free_boundary:
        # the checked-in mgrid paths are relative to the bazel workspace root
        vmec_input.mgrid_file = str(TEST_DATA_DIR / Path(vmec_input.mgrid_file).name)
    if filename == "cma.json":
        # cma ships with ftol 1e-6, too loose for a physics comparison: two
        # different descent trajectories stop ~sqrt-of-nothing apart in energy
        # but O(fsq/mu) apart in state (mu ~ 1e-3 soft modes), which is ~1%
        # in first-order quantities like the volume. Converge properly.
        vmec_input.ftol_array = np.array([1.0e-9, 1.0e-11])
    return vmecpp.run(vmec_input, verbose=False)


@pytest.mark.parametrize(("filename", "free_boundary"), CASES)
def test_vmecpp_style_same_physics_no_iteration_regression(
    filename: str, free_boundary: bool
):
    base = _run_with_style(filename, free_boundary, "vmec_8_52")
    new = _run_with_style(filename, free_boundary, "vmecpp")

    # both runs converged (ier_flag 0 = successful termination)
    assert base.wout.ier_flag == 0
    assert new.wout.ier_flag == 0

    # identical converged physics, up to the scatter between two different
    # descent trajectories into the same force-balance minimum. The MHD
    # energy is variational (second order in the state error at the minimum)
    # and agrees tightly (the residual ~1e-9 absolute scatter comes from the
    # near-null lambda checkerboard subspace, Finding 14); first-order
    # quantities (volume, aspect, b0, beta) inherit the O(fsq/mu) state
    # scatter of the softest modes and get a correspondingly looser tolerance.
    assert new.wout.wb == pytest.approx(base.wout.wb, rel=5e-6)
    assert new.wout.volume == pytest.approx(base.wout.volume, rel=3e-5)
    assert new.wout.aspect == pytest.approx(base.wout.aspect, rel=3e-5)
    assert new.wout.b0 == pytest.approx(base.wout.b0, rel=3e-5)
    # abs floor for the zero-pressure cases where betatotal ~ 0
    assert new.wout.betatotal == pytest.approx(base.wout.betatotal, rel=1e-3, abs=1e-12)

    # no material iteration-count regression (the gains show on multigrid
    # transitions and strongly shaped tails; benign single-stage cases just
    # must not get slower)
    print(f"{filename}: itfsq vmec_8_52 = {base.wout.itfsq}, vmecpp = {new.wout.itfsq}")
    assert new.wout.itfsq <= 1.10 * base.wout.itfsq

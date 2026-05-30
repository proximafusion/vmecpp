# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the Python force-balance iteration (``vmecpp._iteration``).

The iteration logic -- time-step damping, time-step control, restart decisions,
the bad-Jacobian axis reguess, and the convergence test -- is ported to Python
from the C++ ``Vmec::SolveEquilibrium`` and drives the forward model exposed as
``VmecModel``. For boundaries whose linear initial guess is well posed, the
Python loop reproduces the C++ inner solve step-for-step; the reguess case is
checked for convergence to force balance. The PARVMEC and robust time-step-
control styles are checked against the VMEC 8.52 baseline.
"""

from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp.cpp import _vmecpp  # type: ignore

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def _single_resolution_indata(name: str, ns: int, ftol: float, niter: int):
    """C++ VmecINDATA for ``name`` forced to a single radial resolution."""
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA / f"{name}.json")
    cpp_indata = vmec_input._to_cpp_vmecindata()
    cpp_indata.ns_array = np.array([ns], dtype=np.int64)
    cpp_indata.ftol_array = np.array([ftol])
    cpp_indata.niter_array = np.array([niter], dtype=np.int64)
    return cpp_indata


@pytest.mark.parametrize(
    ("name", "ns", "ftol", "niter"),
    [
        ("solovev", 15, 1.0e-12, 3000),
        ("cth_like_fixed_bdy", 25, 1.0e-10, 3000),
    ],
)
def test_python_iteration_reproduces_cpp_inner_solve(
    name: str, ns: int, ftol: float, niter: int
):
    """The Python loop reproduces the C++ Vmec::SolveEquilibrium step-for-step.

    From the same initialized state, running the Python iteration and the C++
    inner solve gives the same converged decision vector and the same invariant
    force-residual trace (to a floating-point floor; the only non-bit-exact
    operation is the damping-average sum, numpy vs Eigen).
    """
    cpp_indata = _single_resolution_indata(name, ns, ftol, niter)

    reference = _vmecpp.VmecModel.create(cpp_indata, ns)
    reference.solve()

    model = _vmecpp.VmecModel.create(cpp_indata, ns)
    result = vmecpp.solve_equilibrium(model)

    assert result.converged
    assert not result.failed
    assert result.axis_reguesses == 0

    cpp_state = np.asarray(reference.get_state())
    py_state = np.asarray(model.get_state())
    assert cpp_state.shape == py_state.shape
    assert np.max(np.abs(cpp_state - py_state)) < 1.0e-9

    cpp_trace = np.asarray(reference.force_residual_r)
    py_trace = np.asarray(result.force_residual_r)
    assert len(cpp_trace) == len(py_trace)
    assert np.max(np.abs(cpp_trace - py_trace)) < 1.0e-10


def test_python_iteration_recovers_from_bad_initial_jacobian():
    """Cma's linear guess overlaps at a single resolution (a bad initial Jacobian); the
    loop must recompute the magnetic axis and still converge to force balance."""
    cpp_indata = _single_resolution_indata("cma", 15, 1.0e-10, 8000)
    model = _vmecpp.VmecModel.create(cpp_indata, 15)
    result = vmecpp.solve_equilibrium(model)

    assert result.converged
    assert not result.failed
    assert result.axis_reguesses >= 1
    assert result.fsqr <= 1.0e-10
    assert result.fsqz <= 1.0e-10
    assert result.fsql <= 1.0e-10


def test_styles_agree_when_no_restart():
    """When no restart fires, the three styles take the identical path.

    The styles differ only in how a growing residual is handled, so on a case that needs
    no restart they advance the geometry through exactly the same states (the bit-
    identical result seen across the test_data equilibria).
    """
    cpp_indata = _single_resolution_indata("cth_like_fixed_bdy", 25, 1.0e-10, 3000)
    states = {}
    for style in ("vmec_8_52", "parvmec", "robust"):
        model = _vmecpp.VmecModel.create(cpp_indata, 25)
        result = vmecpp.solve_equilibrium(model, style=style)
        assert result.converged, style
        assert result.restarts == 0, style
        states[style] = np.asarray(model.get_state())
    assert np.array_equal(states["vmec_8_52"], states["parvmec"])
    assert np.array_equal(states["vmec_8_52"], states["robust"])


def test_alternative_styles_avoid_vmec_8_52_thrashing_on_cma():
    """Cma at ns=72 trips VMEC 8.52's 100x leash ~20 times; PARVMEC and robust avoid it.

    All three styles converge cma to the same equilibrium, but the single- resolution
    solve at ns=72 makes 8.52's tight 100x residual leash revert (and bump ijacob toward
    the give-up escalation) about twenty times, while PARVMEC's 1e4 leash never reverts
    and the robust scheme's moderate leash reverts only a couple of times.
    """
    cpp_indata = _single_resolution_indata("cma", 72, 1.0e-11, 20000)
    results = {}
    for style in ("vmec_8_52", "parvmec", "robust"):
        model = _vmecpp.VmecModel.create(cpp_indata, 72)
        results[style] = vmecpp.solve_equilibrium(model, style=style)

    for style, result in results.items():
        assert result.converged, style
        assert result.fsqr <= 1.0e-11, style
        assert result.fsqz <= 1.0e-11, style
        assert result.fsql <= 1.0e-11, style

    assert results["vmec_8_52"].restarts >= 10
    assert results["parvmec"].restarts == 0
    assert results["robust"].restarts < results["vmec_8_52"].restarts

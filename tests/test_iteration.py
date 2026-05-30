# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the Python force-balance iteration (``vmecpp._iteration``).

The iteration logic -- time-step damping, time-step control, restart decisions,
the bad-Jacobian axis reguess, and the convergence test -- is ported to Python
from the C++ ``Vmec::SolveEquilibrium`` and drives the forward model exposed as
``VmecModel``. For boundaries whose linear initial guess is well posed, the
Python loop reproduces the C++ inner solve step-for-step; the reguess case is
checked for convergence to force balance.
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

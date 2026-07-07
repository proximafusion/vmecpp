# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for an independent lambda Fourier resolution.

``mpol_geometry`` / ``ntor_geometry`` cap the geometry (R, Z) resolution below
the working ``mpol`` / ``ntor`` while lambda keeps the full resolution. The
tests drive cma with the geometry capped below its resolution and check that the
capped modes are frozen at zero, that lambda is still resolved above the cap, and
that a cap equal to the working resolution is a no-op.
"""

from pathlib import Path

import numpy as np

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def _modes_first(array: np.ndarray, mnmax: int) -> np.ndarray:
    """Return ``array`` with the Fourier-mode axis (length ``mnmax``) first."""
    a = np.asarray(array, dtype=float)
    return a if a.shape[0] == mnmax else a.T


def _cma(mpol_geometry: int = -1, ntor_geometry: int = -1) -> vmecpp.VmecInput:
    """Cma capped at (mpol_geometry, ntor_geometry), run for a few iterations."""
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cma.json").model_copy(
        update={
            "mpol_geometry": mpol_geometry,
            "ntor_geometry": ntor_geometry,
            "ns_array": np.array([25], dtype=np.int64),
            "ftol_array": np.array([1e-12]),
            "niter_array": np.array([80], dtype=np.int64),
            "return_outputs_even_if_not_converged": True,
        }
    )


def test_geometry_cap_at_full_resolution_is_a_noop():
    """A geometry cap equal to the working resolution must not change the run."""
    baseline = vmecpp.run(_cma(), max_threads=1).wout  # mpol_geometry unset (-1)
    full = _cma()
    capped = vmecpp.run(
        _cma(mpol_geometry=full.mpol, ntor_geometry=full.ntor), max_threads=1
    ).wout

    for name in ("rmnc", "zmns", "lmns"):
        np.testing.assert_array_equal(
            np.asarray(getattr(capped, name)), np.asarray(getattr(baseline, name))
        )
    assert capped.niter == baseline.niter


def test_geometry_frozen_while_lambda_is_resolved():
    """Above the geometry cap, R/Z stay at zero while lambda keeps content."""
    mpol_geometry, ntor_geometry = 3, 4
    inp = _cma(mpol_geometry=mpol_geometry, ntor_geometry=ntor_geometry)
    wout = vmecpp.run(inp, max_threads=1).wout

    xm = np.asarray(wout.xm)
    xn = np.asarray(wout.xn)
    mnmax = len(xm)
    # Modes above the geometry cap: poloidal m >= cap or toroidal |n| > cap.
    above_cap = (xm >= mpol_geometry) | (np.abs(xn) // inp.nfp > ntor_geometry)
    assert above_cap.any()

    rmnc = _modes_first(wout.rmnc, mnmax)[above_cap]
    zmns = _modes_first(wout.zmns, mnmax)[above_cap]
    lmns = _modes_first(wout.lmns, mnmax)[above_cap]

    # Geometry above the cap is held at exactly zero.
    assert np.max(np.abs(rmnc)) == 0.0
    assert np.max(np.abs(zmns)) == 0.0
    # Lambda above the cap is resolved and carries non-trivial content.
    assert np.max(np.abs(lmns)) > 1e-10

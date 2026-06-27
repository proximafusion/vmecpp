# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for keeping explicit extra geometry modes above the resolution cap.

``extra_geometry_m`` / ``extra_geometry_n`` list (m, n) geometry modes that stay
free above the ``mpol_geometry`` / ``ntor_geometry`` cap. This lets a
reduced-resolution base equilibrium carry a few explicit high-frequency modes
(an isolated perturbation) without raising the whole geometry resolution. The
tests drive solovev with an added m=4 boundary perturbation and check that the
perturbation is frozen to zero under a cap, kept when listed as an extra mode,
and that only the listed modes are kept.
"""

from pathlib import Path

import numpy as np

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"

PERTURBATION_M = 4
PERTURBATION_AMPLITUDE = 0.02


def _modes_first(array: np.ndarray, mnmax: int) -> np.ndarray:
    """Return ``array`` with the Fourier-mode axis (length ``mnmax``) first."""
    a = np.asarray(array, dtype=float)
    return a if a.shape[0] == mnmax else a.T


def _solovev_with_perturbation(
    mpol_geometry: int = -1,
    extra_geometry_m: list[int] | None = None,
    extra_geometry_n: list[int] | None = None,
    sparse_lambda: bool = False,
) -> vmecpp.VmecInput:
    """Solovev with an m=4 boundary perturbation, an optional geometry cap, and optional
    extra modes kept free above the cap."""
    inp = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    rbc = np.asarray(inp.rbc).copy()
    zbs = np.asarray(inp.zbs).copy()
    # ntor == 0, so the n=0 column is column 0.
    rbc[PERTURBATION_M, 0] = PERTURBATION_AMPLITUDE
    zbs[PERTURBATION_M, 0] = PERTURBATION_AMPLITUDE
    update = {
        "rbc": rbc,
        "zbs": zbs,
        "mpol_geometry": mpol_geometry,
        "sparse_lambda": sparse_lambda,
        "ns_array": np.array([15], dtype=np.int64),
        "ftol_array": np.array([1e-11]),
        "niter_array": np.array([500], dtype=np.int64),
        "return_outputs_even_if_not_converged": True,
    }
    if extra_geometry_m is not None:
        update["extra_geometry_m"] = np.array(extra_geometry_m, dtype=np.int64)
        update["extra_geometry_n"] = np.array(extra_geometry_n, dtype=np.int64)
    return inp.model_copy(update=update)


def _boundary_coefficient(wout, array_name: str, m: int, n: int) -> float:
    xm = np.asarray(wout.xm).astype(int)
    xn = np.asarray(wout.xn).astype(int)
    idx = np.where((xm == m) & (xn == n))[0]
    assert idx.size == 1
    return _modes_first(getattr(wout, array_name), len(xm))[idx[0], -1]


def test_extra_mode_kept_above_geometry_cap():
    """Listing m=4 as an extra mode keeps it free where the cap alone freezes it."""
    # Cap geometry at m < 3, no extras: the m=4 perturbation is frozen to zero.
    capped = vmecpp.run(_solovev_with_perturbation(mpol_geometry=3), max_threads=1).wout
    assert _boundary_coefficient(capped, "rmnc", PERTURBATION_M, 0) == 0.0
    assert _boundary_coefficient(capped, "zmns", PERTURBATION_M, 0) == 0.0

    # Same cap, but keep m=4 free: it matches the full-resolution run.
    dense = vmecpp.run(_solovev_with_perturbation(), max_threads=1).wout
    kept = vmecpp.run(
        _solovev_with_perturbation(
            mpol_geometry=3, extra_geometry_m=[PERTURBATION_M], extra_geometry_n=[0]
        ),
        max_threads=1,
    ).wout
    kept_m4 = _boundary_coefficient(kept, "rmnc", PERTURBATION_M, 0)
    assert abs(kept_m4) > 1e-6
    np.testing.assert_allclose(
        kept_m4,
        _boundary_coefficient(dense, "rmnc", PERTURBATION_M, 0),
        rtol=1e-6,
        atol=1e-9,
    )


def test_only_listed_extra_modes_are_kept():
    """Modes above the cap that are not listed as extras stay frozen at zero."""
    kept = vmecpp.run(
        _solovev_with_perturbation(
            mpol_geometry=3, extra_geometry_m=[PERTURBATION_M], extra_geometry_n=[0]
        ),
        max_threads=1,
    ).wout

    # The listed extra mode (m=4) is kept.
    assert abs(_boundary_coefficient(kept, "rmnc", PERTURBATION_M, 0)) > 1e-6

    # Other modes above the cap (m=3, m=5) that were not listed stay at zero.
    xm = np.asarray(kept.xm).astype(int)
    rmnc = _modes_first(kept.rmnc, len(xm))
    zmns = _modes_first(kept.zmns, len(xm))
    for m in (3, 5):
        idx = np.where(xm == m)[0]
        assert idx.size > 0
        assert np.max(np.abs(rmnc[idx])) == 0.0
        assert np.max(np.abs(zmns[idx])) == 0.0


def test_sparse_lambda_freezes_lambda_above_cap():
    """sparse_lambda restricts lambda to the active modes; without it lambda is full."""
    full = vmecpp.run(_solovev_with_perturbation(mpol_geometry=3), max_threads=1).wout
    sparse = vmecpp.run(
        _solovev_with_perturbation(mpol_geometry=3, sparse_lambda=True), max_threads=1
    ).wout

    xm = np.asarray(full.xm).astype(int)
    above_cap = xm >= 3
    assert above_cap.any()
    lmns_full = _modes_first(full.lmns, len(xm))[above_cap]
    lmns_sparse = _modes_first(sparse.lmns, len(xm))[above_cap]

    # Without sparse_lambda, lambda keeps content above the geometry cap ...
    assert np.max(np.abs(lmns_full)) > 1e-10
    # ... with sparse_lambda those lambda modes are frozen at exactly zero.
    assert np.max(np.abs(lmns_sparse)) == 0.0


def test_non_nfp_global_perturbation_runs_sparsely():
    """A global n=1 mode (only representable at nfp=1) can be explored sparsely."""
    inp0 = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")
    mpol = inp0.mpol
    ntor = 2
    base_rbc = np.asarray(inp0.rbc)[:, 0]
    base_zbs = np.asarray(inp0.zbs)[:, 0]
    rbc = np.zeros((mpol, 2 * ntor + 1))
    zbs = np.zeros((mpol, 2 * ntor + 1))
    rbc[:, ntor] = base_rbc  # n=0 base
    zbs[:, ntor] = base_zbs
    rbc[1, ntor + 1] = 0.02  # global (m=1, n=1) helical mode
    zbs[1, ntor + 1] = 0.02

    inp = inp0.model_copy(
        update={
            "nfp": 1,
            "ntor": ntor,
            "rbc": rbc,
            "zbs": zbs,
            "raxis_c": np.array([4.0, 0.0, 0.0]),
            "zaxis_s": np.array([0.0, 0.0, 0.0]),
            # keep the axisymmetric base (n=0) plus the global (m=1, n=1) mode
            "ntor_geometry": 0,
            "extra_geometry_m": np.array([1], dtype=np.int64),
            "extra_geometry_n": np.array([1], dtype=np.int64),
            "sparse_lambda": True,
            "ns_array": np.array([25], dtype=np.int64),
            "ftol_array": np.array([1e-11]),
            "niter_array": np.array([2000], dtype=np.int64),
            "return_outputs_even_if_not_converged": True,
        }
    )
    wout = vmecpp.run(inp, max_threads=1).wout

    # The global (m=1, n=1) mode is kept ...
    assert abs(_boundary_coefficient(wout, "rmnc", 1, 1)) > 1e-6
    # ... while frozen toroidal modes (n=2) stay at exactly zero.
    xn = np.asarray(wout.xn).astype(int)
    rmnc = _modes_first(wout.rmnc, len(xn))
    idx_n2 = np.where(xn == 2)[0]
    assert idx_n2.size > 0
    assert np.max(np.abs(rmnc[idx_n2])) == 0.0

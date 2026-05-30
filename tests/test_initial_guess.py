# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the alternative initial-guess constructions (Zeno, map2disc).

A construction is correct if the interior coordinate map it produces is valid (non-
overlapping), which we verify operationally: VMEC++ hot-restarted from the guess
converges to the same equilibrium as the default linear guess. An overlapping guess
(negative Jacobian) would instead make VMEC++ fail in the first iterations.
"""

from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp._initial_guess import map2disc_guess, zeno_guess

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


@pytest.fixture(scope="module")
def cth_like() -> vmecpp.VmecInput:
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")


@pytest.fixture(scope="module")
def cth_like_volume(cth_like: vmecpp.VmecInput) -> float:
    """Plasma volume from the default (linear) initial guess, for comparison."""
    return float(vmecpp.run(cth_like, verbose=False, max_threads=1).wout.volume)


def _cross_section_areas(guess: vmecpp.VmecOutput) -> np.ndarray:
    """Enclosed area of each flux surface at the zeta=0 cross-section, by Green's
    theorem.

    Monotonically increasing areas mean the surfaces are nested.
    """
    wout = guess.wout
    xm = np.asarray(wout.xm)
    rmnc = np.asarray(wout.rmnc)
    zmns = np.asarray(wout.zmns)
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    # zeta = 0 cross-section: cos(m theta - n zeta) = cos(m theta).
    cos_mt = np.cos(xm[:, None] * theta[None, :])
    sin_mt = np.sin(xm[:, None] * theta[None, :])
    big_r = rmnc.T @ cos_mt  # [ns, n_theta]
    big_z = zmns.T @ sin_mt
    r_t = rmnc.T @ (-xm[:, None] * sin_mt)
    z_t = zmns.T @ (xm[:, None] * cos_mt)
    dtheta = float(theta[1] - theta[0])
    # signed area enclosed by each closed (R(theta), Z(theta)) curve
    return 0.5 * np.sum(big_r * z_t - big_z * r_t, axis=1) * dtheta


def test_zeno_guess_reproduces_equilibrium(
    cth_like: vmecpp.VmecInput, cth_like_volume: float
):
    """zeno_guess builds a valid interior; VMEC++ restarted from it converges to the
    same equilibrium as the default guess."""
    guess = zeno_guess(cth_like)
    assert isinstance(guess, vmecpp.VmecOutput)
    ns = int(np.asarray(cth_like.ns_array)[-1])
    assert np.asarray(guess.wout.rmnc).shape[1] == ns
    # the guess carries geometry only; the lambda stream starts from zero
    assert np.allclose(np.asarray(guess.wout.lmns_full), 0.0)
    out = vmecpp.run(cth_like, restart_from=guess, verbose=False, max_threads=1)
    assert out.wout.volume == pytest.approx(cth_like_volume, rel=1e-6)


def test_map2disc_guess_surfaces_are_nested(cth_like: vmecpp.VmecInput):
    """Map2disc is a diffeomorphism, so its surfaces are nested: the enclosed cross-
    section area grows monotonically from axis to boundary."""
    pytest.importorskip("map2disc")
    area = np.abs(_cross_section_areas(map2disc_guess(cth_like)))
    assert np.all(np.diff(area) > 0.0)


def test_map2disc_guess_reproduces_equilibrium(
    cth_like: vmecpp.VmecInput, cth_like_volume: float
):
    """map2disc_guess (optional dependency) likewise yields a valid interior."""
    pytest.importorskip("map2disc")
    guess = map2disc_guess(cth_like)
    out = vmecpp.run(cth_like, restart_from=guess, verbose=False, max_threads=1)
    assert out.wout.volume == pytest.approx(cth_like_volume, rel=1e-6)

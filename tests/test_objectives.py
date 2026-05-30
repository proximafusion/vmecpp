# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the stellarator-optimization objectives (``vmecpp._objectives``).

On a converged equilibrium, the aspect ratio, the iota profile, the magnetic
well, and the mirror-ratio ``|B|`` field match SIMSOPT to numerical precision; the
Boozer transform is validated by self-consistency (it reproduces ``|B|`` from its
own Boozer spectrum, converging to machine precision as the mode count grows); and
the quasi-symmetry residual is cross-checked against simsopt's
``QuasisymmetryRatioResidual``.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.mhd import Vmec as SimsoptVmec
from simsopt.mhd.vmec_diagnostics import vmec_compute_geometry

import vmecpp
from vmecpp import _objectives as obj
from vmecpp.simsopt_compat import Vmec

REPO_ROOT = Path(__file__).parent.parent
CASE = REPO_ROOT / "examples" / "data" / "cth_like_fixed_bdy.json"


@pytest.fixture(scope="module")
def converged() -> Vmec:
    """A converged cth_like fixed-boundary equilibrium, as a SIMSOPT Vmec wrapper."""
    vmec = Vmec(str(CASE), verbose=False)
    vmec.run()
    return vmec


@pytest.fixture(scope="module")
def wout(converged: Vmec) -> vmecpp.VmecWOut:
    """The converged wout (run() guarantees it is populated)."""
    result = converged.wout
    assert result is not None
    return result


def test_aspect_ratio_matches_simsopt(converged: Vmec, wout: vmecpp.VmecWOut):
    assert abs(obj.aspect_ratio(wout) - converged.aspect()) < 1.0e-10


def test_iota_profile_matches_simsopt(converged: Vmec, wout: vmecpp.VmecWOut):
    _, iota = obj.iota_profile(wout)
    assert abs(float(iota[0]) - converged.iota_axis()) < 1.0e-10
    assert abs(float(iota[-1]) - converged.iota_edge()) < 1.0e-10
    mean_iota = float(np.mean(np.asarray(wout.iotas)[1:]))
    assert abs(mean_iota - converged.mean_iota()) < 1.0e-10


def test_iota_profile_residual_against_itself_is_zero(wout: vmecpp.VmecWOut):
    _, iota = obj.iota_profile(wout)
    assert obj.iota_profile_residual(wout, iota) < 1.0e-28


def test_magnetic_well_matches_simsopt(converged: Vmec, wout: vmecpp.VmecWOut):
    assert abs(obj.magnetic_well(wout) - converged.vacuum_well()) < 1.0e-10


def test_mirror_ratio_matches_simsopt_field(wout: vmecpp.VmecWOut):
    ns = int(wout.ns)
    wout_path = str(Path(tempfile.gettempdir()) / "wout_objtest.nc")
    wout.save(wout_path)
    reference = SimsoptVmec(wout_path, verbose=False)

    s_edge = 1.0 - 0.5 / (ns - 1)
    n_theta = n_zeta = 48
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_zeta, endpoint=False)
    mod_b_ref = np.asarray(
        vmec_compute_geometry(reference, np.array([s_edge]), theta, phi).modB
    ).squeeze()
    mod_b_mine = obj.field_strength_on_surface(
        wout, surface=-1, n_theta=n_theta, n_zeta=n_zeta
    )
    assert np.max(np.abs(mod_b_mine - mod_b_ref)) < 1.0e-8

    mirror_ref = (mod_b_ref.max() - mod_b_ref.min()) / (
        mod_b_ref.max() + mod_b_ref.min()
    )
    mirror_mine = obj.mirror_ratio(wout, surface=-1, n_theta=n_theta, n_zeta=n_zeta)
    assert abs(mirror_mine - mirror_ref) < 1.0e-8


def test_boozer_transform_self_consistent(wout: vmecpp.VmecWOut):
    """|B| reconstructs from its own Boozer spectrum to near machine precision."""
    residual = obj.boozer_roundtrip_residual(wout, surface=-1, m_boozer=48, n_boozer=20)
    assert residual < 1.0e-8


def test_boozer_roundtrip_converges(wout: vmecpp.VmecWOut):
    """The self-consistency residual decreases as the Boozer mode count grows."""
    coarse = obj.boozer_roundtrip_residual(wout, surface=-1, m_boozer=12, n_boozer=8)
    fine = obj.boozer_roundtrip_residual(wout, surface=-1, m_boozer=36, n_boozer=16)
    assert fine < coarse
    assert fine < 1.0e-7


def test_quasisymmetry_residual_cross_checks_simsopt(
    converged: Vmec, wout: vmecpp.VmecWOut
):
    s_edge = 1.0 - 0.5 / (int(wout.ns) - 1)
    mine = obj.quasisymmetry_residual(wout, helicity_m=1, helicity_n=0)
    landreman = QuasisymmetryRatioResidual(
        converged, s_edge, helicity_m=1, helicity_n=0
    ).total()
    # cth_like is not quasi-axisymmetric: both metrics report a finite residual,
    # and the Boozer-harmonic fraction is between 0 and 1.
    assert 0.0 < mine < 1.0
    assert landreman > 0.0

# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Ports of (some of) SIMSOPT's VMEC tests, taken from:

https://github.com/hiddenSymmetries/simsopt/blob/50f41d6b020d09d9a13f179a42752487ef0ed911/tests/mhd/test_vmec.py
"""

import logging
import math
from pathlib import Path

import numpy as np
import pytest
from simsopt.geo.surfacerzfourier import SurfaceRZFourier

from vmecpp import _util
from vmecpp.cpp.vmecpp.vmec.pybind11.simsopt_vmecpp import Vmec

logger = logging.getLogger(__name__)


TEST_DATA_DIR = Path(_util.package_root(), "cpp", "vmecpp", "test_data")

########################################################################
# Tests for VMEC initialized from an output file
########################################################################


def test_vacuum_well():
    """Test the calculation of magnetic well.

    This is done by comparison to a high-aspect-ratio configuration, in which case the
    magnetic well can be computed analytically by the method in Landreman & Jorge, J
    Plasma Phys 86, 905860510 (2020). The specific configuration considered is the one
    of section 5.4 in Landreman & Sengupta, J Plasma Phys 85, 815850601 (2019), also
    considered in the 2020 paper. We increase the mean field B0 to 2T in that
    configuration to make sure all the factors of B0 are correct.
    """
    filename = (
        TEST_DATA_DIR / "wout_LandremanSengupta2019_section5.4_B2_A80_reference.nc"
    )
    vmec = Vmec(str(filename))
    assert vmec.wout is not None
    well_vmec = vmec.vacuum_well()
    # Let psi be the toroidal flux divided by (2 pi)
    abs_psi_a = np.abs(vmec.wout.phi[-1]) / (2 * np.pi)
    # Data for this configuration from the near-axis construction code qsc:
    # https://github.com/landreman/pyQSC
    # or
    # https://github.com/landreman/qsc
    B0 = 2.0
    G0 = 2.401752071286676
    d2_volume_d_psi2 = 25.3041656424299
    # See also "20210504-01 Computing magnetic well from VMEC.docx" by MJL
    well_analytic = (
        -abs_psi_a * B0 * B0 * d2_volume_d_psi2 / (4 * np.pi * np.pi * np.abs(G0))
    )
    logger.info(f"well_vmec: {well_vmec}  well_analytic: {well_analytic}")
    assert np.allclose(well_vmec, well_analytic, rtol=2e-2, atol=0)


def test_iota():
    """Test the functions related to iota."""
    filename = (
        TEST_DATA_DIR / "wout_LandremanSengupta2019_section5.4_B2_A80_reference.nc"
    )
    vmec = Vmec(str(filename))

    iota_axis = vmec.iota_axis()
    iota_edge = vmec.iota_edge()
    mean_iota = vmec.mean_iota()
    mean_shear = vmec.mean_shear()
    # These next 2 lines are different ways the mean iota and
    # shear could be defined. They are not mathematically
    # identical to mean_iota() and mean_shear(), but they should
    # be close.
    mean_iota_alt = (iota_axis + iota_edge) * 0.5
    mean_shear_alt = iota_edge - iota_axis
    logger.info(f"iota_axis: {iota_axis}, iota_edge: {iota_edge}")
    logger.info(f"    mean_iota: {mean_iota},     mean_shear: {mean_shear}")
    logger.info(f"mean_iota_alt: {mean_iota_alt}, mean_shear_alt: {mean_shear_alt}")

    assert math.isclose(mean_iota, mean_iota_alt, abs_tol=1e-4)
    assert math.isclose(mean_shear, mean_shear_alt, abs_tol=1e-4)


# TODO(eguiraud): move this test to standalone vmecpp-tests repo once we have it
@pytest.mark.skip(
    reason="test file is too large to be included in standalone VMEC++ repo"
)
def test_external_current():
    """Test the external_current() function."""
    filename = (
        TEST_DATA_DIR
        / "wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc"
    )
    vmec = Vmec(str(filename))
    assert vmec.wout is not None
    bsubvmnc = 1.5 * vmec.wout.bsubvmnc[0, -1] - 0.5 * vmec.wout.bsubvmnc[0, -2]
    mu0 = 4 * np.pi * (1.0e-7)
    external_current = 2 * np.pi * bsubvmnc / mu0
    assert np.allclose(external_current, vmec.external_current())


# TODO(eguiraud): This test is present in SIMSOPT's master branch but not in the
# version that we currently use (v0.18.1): it has been introduced by
# https://github.com/hiddenSymmetries/simsopt/pull/341, and with v0.18.1
# the test fails because it's missing the corresponding fix to the free-boundary logic.
# To be uncommented once we update the SIMSOPT version we use.
# def test_curve_orientation_sign_for_free_boundary(self):
#         """
#         For free-boundary equilibrium calculations to work following stage-2
#         optimization, the sign of the toroidal field created by the coils must
#         match the sign of the toroidal field in the original target equilibrium.
#         This is a particular issue for the finite-beta case, in which the
#         Btarget from virtual casing has a definite sign. Associated with this
#         issue, in the ``stage_two_optimization_finite_beta.py`` example, there is a
#         sign
#         associated with ``vmec.external_current()`` that must be consistent with the
#         direction of the coils created by ``create_equally_spaced_curves()``.
#         """
#         # The code that follows is extracted from
#         # stage_two_optimization_finite_beta.py
#         n_coils_per_half_period = 5
#         R0 = 5.5
#         R1 = 1.25
#         order = 6
#         filename = 'wout_W7-X_without_coil_ripple_beta0p05_d23p4_tm_reference.nc'
#         vmec_file = TEST_DIR / filename
#         vmec = Vmec(vmec_file)
#         s = vmec.boundary
#         total_current = vmec.external_current() / (2 * s.nfp)
#         base_curves = create_equally_spaced_curves(
#            n_coils_per_half_period,
#            s.nfp,
#            stellsym=True,
#            R0=R0,
#            R1=R1,
#            order=order,
#            numquadpoints=128)
#         base_currents = [
#              Current(total_current
#              / n_coils_per_half_period * 1e-5)
#              * 1e5 for _ in range(n_coils_per_half_period-1)
#         ]
#         total_current = Current(total_current)
#         total_current.fix_all()
#         base_currents += [total_current - sum(base_currents)]
#
#         coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
#         bs = BiotSavart(coils)
#         bs.set_points(np.array([[vmec.wout.Rmajor_p, 0, 0]]))
#         B = bs.B()
#         print("B:", B)
#         self.assertGreater(B[0, 1], 0)


def test_error_on_rerun():
    """If a vmec object is initialized from a wout file, and if the dofs are then
    changed, vmec output functions should raise an exception."""
    filename = TEST_DATA_DIR / "wout_li383_low_res_reference.nc"
    vmec = Vmec(str(filename))
    _ = vmec.mean_iota()
    vmec.boundary.set_rc(1, 0, 2.0)
    with pytest.raises(RuntimeError):
        vmec.mean_iota()


# #########################################################################
# # Tests for VMEC initialized from an input file
# ########################################################################


def test_init_from_file():
    """Try creating a Vmec instance from a specified input file."""

    filename = TEST_DATA_DIR / "li383_low_res.json"
    v = Vmec(str(filename))
    assert v.indata is not None
    assert v.indata.nfp == 3
    assert v.indata.mpol == 4
    assert v.indata.ntor == 3
    # NOTE: SurfaceRZFourier and VMEC++ have different conventions:
    # for the former, m goes up to mpol _included_.
    assert v.boundary.mpol == v.indata.mpol - 1
    assert v.boundary.ntor == 3

    # n = 0, m = 0:
    assert math.isclose(v.boundary.get_rc(0, 0), 1.3782)

    # n = 0, m = 1:
    assert math.isclose(v.boundary.get_zs(1, 0), 4.6465e-01)

    # n = 1, m = 1:
    assert math.isclose(v.boundary.get_zs(1, 1), 1.6516e-01)

    assert v.indata.ncurr == 1
    assert not v.free_boundary
    assert v.need_to_run_code


def test_surface_4_ways():
    """If we initialize a Vmec object, the boundary surface object should be (almost)
    the same as if we initialize a SurfaceRZFourier using SurfaceRZFourier.from_wout()
    or SurfaceRZFourier.from_vmec_input().

    The possible difference is that mpol, ntor, and the quadrature points may be
    different.
    """

    def compare_surfaces_sym(s1, s2):
        logger.debug("compare_surfaces_sym called")
        mpol = min(s1.mpol, s2.mpol)
        ntor = min(s1.ntor, s2.ntor)
        # NOTE: the original SIMSOPT test was checking the larger range
        # m in range(mpol + 1), but VMEC++ only uses range(mpol)
        for m in range(mpol):
            nmin = 0 if m == 0 else -ntor
            for n in range(nmin, ntor + 1):
                assert math.isclose(s1.get_rc(m, n), s2.get_rc(m, n))
                assert math.isclose(s1.get_zs(m, n), s2.get_zs(m, n))

    def compare_surfaces_asym(s1, s2):
        logger.debug("compare_surfaces_asym called")
        assert math.isclose(np.abs(s1.volume()), np.abs(s2.volume()))
        assert math.isclose(s1.area(), s2.area())
        mpol = min(s1.mpol, s2.mpol)
        ntor = min(s1.ntor, s2.ntor)
        # NOTE: the original SIMSOPT test was checking the larger range
        # m in range(mpol + 1), but VMEC++ only uses range(mpol)
        for m in range(mpol):
            nmin = 0 if m == 0 else -ntor
            for n in range(nmin, ntor + 1):
                assert math.isclose(s1.get_rc(m, n), s2.get_rc(m, n))
                assert math.isclose(s1.get_zs(m, n), s2.get_zs(m, n))
                assert math.isclose(s1.get_rs(m, n), s2.get_rs(m, n))
                assert math.isclose(s1.get_zc(m, n), s2.get_zc(m, n))

    # First try a stellarator-symmetric example:
    filename1 = TEST_DATA_DIR / "li383_low_res.json"
    filename2 = TEST_DATA_DIR / "wout_li383_low_res_reference.nc"
    v = Vmec(str(filename1))
    s1 = v.boundary
    # Compare initializing a Vmec object from an input file vs from a wout file:
    v2 = Vmec(str(filename2))
    s2 = v2.boundary
    compare_surfaces_sym(s1, s2)
    # Compare to initializing a surface using from_wout()
    s2 = SurfaceRZFourier.from_wout(str(filename2))
    compare_surfaces_sym(s1, s2)
    # Now try from_vmec_input() instead of from_wout():
    # NOTE: here we use the Fortran input from which VMEC++'s JSON input was derived
    s2 = SurfaceRZFourier.from_vmec_input(str(TEST_DATA_DIR / "input.li383_low_res"))
    compare_surfaces_sym(s1, s2)

    # TODO(eguiraud): the input file for the following test,
    # input.LandremanSenguptaPlunk_section5p3, cannot be converted to JSON by
    # indata2json. It results in this error:
    #
    # $ indata2json/indata2json vmecpp/test_data/input.LandremanSenguptaPlunk_section5p3
    #    In VMEC, indata NAMELIST error: iostat = 5010
    #    Invalid line in INDATA namelist:   PT_TYPE = 'power_series'
    #
    # So for now we leave out this non-stellarator-symmetric test.
    #
    # # Now try a non-stellarator-symmetric example.
    # # For non-stellarator-symmetric cases, we must be careful when
    # # directly comparing the rc/zs/rs/zc coefficients, because
    # # VMEC shifts the poloidal angle in readin.f upon loading the
    # # file. Moreover, in versions of VMEC other than the
    # # hiddenSymmetries python module, the shift to theta may have
    # # a bug so the angle shift is not by the claimed value. The
    # # specific input file used here has a boundary that should not
    # # be shifted by the hiddenSymmetries VMEC2000 module.  For any
    # # input file and version of VMEC, we can compare
    # # coordinate-independent properties like the volume.
    # filename1 = os.path.join(test_dir, "input.LandremanSenguptaPlunk_section5p3")
    # filename2 = os.path.join(
    #     test_dir, "wout_LandremanSenguptaPlunk_section5p3_reference.nc"
    # )
    # v = Vmec(filename1)
    # s1 = v.boundary

    # # Compare initializing a Vmec object from an input file vs from a wout file:
    # v2 = Vmec(filename2)
    # s2 = v2.boundary
    # compare_surfaces_asym(s1, s2)

    # s2 = SurfaceRZFourier.from_wout(filename2)
    # compare_surfaces_asym(s1, s2)

    # s2 = SurfaceRZFourier.from_vmec_input(filename1)
    # compare_surfaces_asym(s1, s2)


# TODO(eguiraud): this test fails because running with "li383_low_res.json" as input
# does not converge. SIMSOPT seems to work just fine with this input though?
# def test_2_init_methods(test_dir):
#     """
#     If we initialize a Vmec object from an input file, or initialize a
#     Vmec object from the corresponding wout file, physics quantities
#     should be the same.
#     """
#     # TODO(eguiraud): non-stellarator-symmetric scenarios are not supported
#     # at the moment (see previous TODO).
#     # for jsym in range(2):
#         # if jsym == 0:
#         #     # Try a stellarator-symmetric scenario:
#         #     filename1 = os.path.join(test_dir, "wout_li383_low_res_reference.nc")
#         #     filename2 = os.path.join(test_dir, "li383_low_res.json")
#         # else:
#         #     # Try a non-stellarator-symmetric scenario:
#         #     filename1 = os.path.join(
#         #         test_dir, "wout_LandremanSenguptaPlunk_section5p3_reference.nc"
#         #     )
#         #     filename2 = os.path.join(
#         #         test_dir, "input.LandremanSenguptaPlunk_section5p3"
#         #     )
#     filename1 = os.path.join(test_dir, "wout_li383_low_res_reference.nc")
#     filename2 = os.path.join(test_dir, "li383_low_res.json")

#     vmec1 = Vmec(filename1)
#     iota1 = vmec1.wout.iotaf
#     bmnc1 = vmec1.wout.bmnc

#     vmec2 = Vmec(filename2)
#     vmec2.run()
#     iota2 = vmec2.wout.iotaf
#     bmnc2 = vmec2.wout.bmnc

#     assert np.allclose(iota1, iota2, atol=1e-10)
#     assert np.allclose(bmnc1, bmnc2, atol=1e-10)


# TODO(eguiraud): VMEC++ does not have a verbose flag, it prints the state of the
# optimization unconditionally.
# def test_verbose(test_dir):
#     """
#     I'm not sure how to confirm that nothing is printed if ``verbose``
#     is set to ``False``, but we can at least make sure the code
#     doesn't crash in this case.
#     """
#     for verbose in [True, False]:
#         filename = os.path.join(test_dir, "li383_low_res.json")
#         Vmec(filename, verbose=verbose).run()


def test_write_input():
    """Check that working input files can be written."""
    # TODO(eguiraud): the original SIMSOPT test was testing two
    # more configurations that are currently not supported by VMEC++.
    configs = [
        "circular_tokamak",
    ]
    for config in configs:
        infilename = TEST_DATA_DIR / f"{config}.json"
        outfilename = TEST_DATA_DIR / f"wout_{config}_reference.nc"
        vmec1 = Vmec(str(infilename))
        newfile = "input.test"
        vmec1.write_input(newfile)
        # Now read in the newly created input file and run:
        vmec2 = Vmec(newfile)
        vmec2.run()
        assert vmec2.wout is not None
        # Read in reference values and compare:
        vmec3 = Vmec(str(outfilename))
        assert vmec3.wout is not None
        np.testing.assert_allclose(vmec2.wout.rmnc, vmec3.wout.rmnc, atol=1e-10)

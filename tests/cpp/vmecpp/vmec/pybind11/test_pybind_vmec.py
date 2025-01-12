# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for VMEC++ pybind11 Python bindings."""

from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from vmecpp.cpp import _vmecpp as vmec

# We don't want to install tests and test data as part of the package,
# but scikit-build-core + hatchling does not support editable installs,
# so the tests live in the sources but the vmecpp module lives in site_packages.
# Therefore, in order to find the test data we use the relative path to this file.
# I'm very open to alternative solutions :)
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


def is_close_ra(actual, expected, tolerance, context=""):
    all_good = True

    actual_shape = np.shape(actual)
    expected_shape = np.shape(expected)
    if len(actual_shape) != len(expected_shape):
        print(
            "rank mismatch: actual is rank-%d, expected is rank-%d"
            % (len(actual_shape), len(expected_shape))
        )
        return False

    if len(actual_shape) > 0:
        if not np.array_equal(actual_shape, expected_shape):
            print(f"size mismatch: actual: {actual_shape}, expected :{expected_shape}")
            return False

        for i, a in enumerate(actual):
            e = expected[i]
            all_good &= is_close_ra(a, e, tolerance, " at %d" % (i,))
    else:
        # This is the actual test, on scalars.
        ra_err = (actual - expected) / (1.0 + abs(expected))
        if abs(ra_err) > tolerance:
            print(
                (
                    "mismatch%s:\n"
                    + "    actual = % .12e\n"
                    + "  expected = % .12e\n"
                    + "     error = % .12e\n"
                )
                % (context, actual, expected, ra_err),
                flush=True,
            )
            all_good = False

    return all_good


def test_indata_readwrite():
    """Test that we can read in an VmecINDATA object and then modify its contents."""
    indata = vmec.VmecINDATAPyWrapper.from_file(TEST_DATA_DIR / "solovev.json")
    assert indata.ntor == 0
    assert indata.mpol == 6
    assert len(indata.ns_array) == 3
    assert len(indata.rbc) == 6  # mpol * (2 * ntor + 1)

    # Read/write whole arrays
    # Only permitted on some of indata's arrays (see below).
    indata.ns_array = np.array((1, 2, 3))
    assert np.all(indata.ns_array == (1, 2, 3))

    # Read/write array elements
    # These checks might seem trivial, but depending on how the pybind11 bindings
    # are done, assignments like these can break silently.
    indata.ns_array[0] = 6
    assert indata.ns_array[0] == 6
    indata.raxis_c[0] = 8.0
    assert indata.raxis_c[0] == 8.0

    # Read/write matrix elements
    indata.rbc[2, 0] = 42.0
    assert indata.rbc[2, 0] == 42.0

    # Read/write enum
    indata.free_boundary_method = vmec.FreeBoundaryMethod.BIEST
    assert indata.free_boundary_method == vmec.FreeBoundaryMethod.BIEST

    # Assignments like these would break VmecINDATA's invariants:
    # make sure they are not allowed
    with pytest.raises(AttributeError):
        indata.mpol = 0

    with pytest.raises(AttributeError):
        indata.raxis_c = np.array([])

    with pytest.raises(AttributeError):
        indata.rbc = np.array([])


def test_output_quantities():
    case_name = "cma"

    indata = vmec.VmecINDATAPyWrapper.from_file(TEST_DATA_DIR / f"{case_name}.json")
    output_quantities = vmec.run(indata)

    # jxbout
    jxbout = Dataset(TEST_DATA_DIR / f"jxbout_{case_name}.nc", "r")
    assert is_close_ra(output_quantities.jxbout.phin, jxbout["phin"][()], 1.0e-12)
    assert is_close_ra(output_quantities.jxbout.avforce, jxbout["avforce"][()], 1.0e-6)
    assert is_close_ra(
        output_quantities.jxbout.jdotb, jxbout["surf_av_jdotb"][()], 1.0e-6
    )
    assert is_close_ra(
        output_quantities.jxbout.bdotgradv, jxbout["bdotgradv"][()], 1.0e-6
    )
    assert is_close_ra(output_quantities.jxbout.pprim, jxbout["pprime"][()], 1.0e-6)
    assert is_close_ra(output_quantities.jxbout.aminfor, jxbout["aminfor"][()], 1.0e-6)
    assert is_close_ra(output_quantities.jxbout.amaxfor, jxbout["amaxfor"][()], 1.0e-6)
    (n_theta, n_zeta, ns) = np.shape(jxbout["sqrt_g__bdotk"][()])
    jkl_shape = (ns, n_zeta, n_theta)
    assert is_close_ra(
        np.reshape(
            output_quantities.jxbout.jdotb_sqrtg,
            jkl_shape,
        ).T,
        jxbout["sqrt_g__bdotk"][()],
        1.0e-3,
    )
    assert is_close_ra(
        np.reshape(output_quantities.jxbout.sqrtg3, jkl_shape).T,
        jxbout["sqrt_g_"][()],
        1.0e-3,
    )
    assert is_close_ra(
        np.reshape(output_quantities.jxbout.jsupu3, jkl_shape).T,
        jxbout["jsupu"][()],
        1.0e-4,
    )
    assert is_close_ra(
        np.reshape(output_quantities.jxbout.jsupv3, jkl_shape).T,
        jxbout["jsupv"][()],
        1.0e-4,
    )
    assert is_close_ra(
        np.reshape(
            output_quantities.jxbout.jsups3,
            [ns - 1, n_zeta, n_theta],
        ).T,
        jxbout["jsups"][()][:, :, 1:],
        1.0e-3,
    )
    assert is_close_ra(
        np.reshape(output_quantities.jxbout.bsupu3, jkl_shape).T,
        jxbout["bsupu"][()],
        1.0e-4,
    )
    assert is_close_ra(
        np.reshape(output_quantities.jxbout.bsupv3, jkl_shape).T,
        jxbout["bsupv"][()],
        1.0e-4,
    )
    assert is_close_ra(
        np.reshape(output_quantities.jxbout.jcrossb, jkl_shape).T,
        jxbout["jcrossb"][()],
        1.0e-4,
    )
    assert is_close_ra(
        np.reshape(output_quantities.jxbout.jxb_gradp, jkl_shape).T,
        jxbout["jxb_gradp"][()],
        1.0e-4,
    )
    assert is_close_ra(
        np.reshape(
            output_quantities.jxbout.bsubu3,
            [ns - 1, n_zeta, n_theta],
        ).T,
        jxbout["bsubu"][()][:, :, 1:],
        1.0e-4,
    )
    assert is_close_ra(
        np.reshape(
            output_quantities.jxbout.bsubv3,
            [ns - 1, n_zeta, n_theta],
        ).T,
        jxbout["bsubv"][()][:, :, 1:],
        1.0e-4,
    )
    assert is_close_ra(
        np.reshape(output_quantities.jxbout.bsubs3, jkl_shape).T,
        jxbout["bsubs"][()],
        1.0e-4,
    )

    jxbout.close()

    # TODO(jons): implement tests against JSON debugging output
    # threed1_first_table
    # threed1_geometric_magnetic
    # threed1_volumetrics
    # threed1_axis
    # threed1_betas
    # threed1_shafranov_integrals

    wout = Dataset(TEST_DATA_DIR / f"wout_{case_name}.nc", "r")

    # mercier
    # The Mercier stability outputs are also written in the wout file,
    # so we take the reference from there
    # instead of trying to parse the text output file.
    #
    # only check Mercier outputs for now
    # if they are fine, the intermediate quantities will be as well...
    assert is_close_ra(output_quantities.mercier.DMerc, wout["DMerc"][()], 1.0e-8)
    assert is_close_ra(output_quantities.mercier.Dshear, wout["DShear"][()], 1.0e-8)
    assert is_close_ra(output_quantities.mercier.Dwell, wout["DWell"][()], 1.0e-9)
    assert is_close_ra(output_quantities.mercier.Dcurr, wout["DCurr"][()], 1.0e-8)
    assert is_close_ra(output_quantities.mercier.Dgeod, wout["DGeod"][()], 1.0e-8)

    # wout
    assert output_quantities.wout.sign_of_jacobian == wout["signgs"][()]
    assert output_quantities.wout.gamma == wout["gamma"][()]

    assert (
        output_quantities.wout.pcurr_type
        == wout["pcurr_type"][()].tobytes().decode().strip()
    )
    assert (
        output_quantities.wout.pmass_type
        == wout["pmass_type"][()].tobytes().decode().strip()
    )
    assert (
        output_quantities.wout.piota_type
        == wout["piota_type"][()].tobytes().decode().strip()
    )

    # mass profile: am, am_aux_s, am_aux_f
    last_zero = 0

    while wout["am"][()][last_zero] != 0.0:
        last_zero += 1
    assert is_close_ra(
        output_quantities.wout.am, wout["am"][()][: last_zero + 1], 1.0e-15
    )
    last_zero = 0
    while wout["am_aux_s"][()][last_zero] != -1.0:
        last_zero += 1
    assert is_close_ra(
        output_quantities.wout.am_aux_s,
        wout["am_aux_s"][()][: last_zero + 1],
        1.0e-15,
    )
    last_zero = 0
    while wout["am_aux_f"][()][last_zero] != 0.0:
        last_zero += 1
    assert is_close_ra(
        output_quantities.wout.am_aux_f,
        wout["am_aux_f"][()][: last_zero + 1],
        1.0e-15,
    )

    # current profile: ac, ac_aux_s, ac_aux_f
    last_zero = 0
    while wout["ac"][()][last_zero] != 0.0:
        last_zero += 1
    assert is_close_ra(
        output_quantities.wout.ac, wout["ac"][()][: last_zero + 1], 1.0e-15
    )
    last_zero = 0
    while wout["ac_aux_s"][()][last_zero] != -1.0:
        last_zero += 1
    assert is_close_ra(
        output_quantities.wout.ac_aux_s,
        wout["ac_aux_s"][()][: last_zero + 1],
        1.0e-15,
    )
    last_zero = 0
    while wout["ac_aux_f"][()][last_zero] != 0.0:
        last_zero += 1
    assert is_close_ra(
        output_quantities.wout.ac_aux_f,
        wout["ac_aux_f"][()][: last_zero + 1],
        1.0e-15,
    )

    # iota profile: ai, ai_aux_s, ai_aux_f
    last_zero = 0
    while wout["ai"][()][last_zero] != 0.0:
        last_zero += 1
    assert is_close_ra(
        output_quantities.wout.ai, wout["ai"][()][: last_zero + 1], 1.0e-15
    )
    last_zero = 0
    while wout["ai_aux_s"][()][last_zero] != -1.0:
        last_zero += 1
    assert is_close_ra(
        output_quantities.wout.ai_aux_s,
        wout["ai_aux_s"][()][: last_zero + 1],
        1.0e-15,
    )
    last_zero = 0
    while wout["ai_aux_f"][()][last_zero] != 0.0:
        last_zero += 1
    assert is_close_ra(
        output_quantities.wout.ai_aux_f,
        wout["ai_aux_f"][()][: last_zero + 1],
        1.0e-15,
    )

    assert output_quantities.wout.nfp == wout["nfp"][()]
    assert output_quantities.wout.mpol == wout["mpol"][()]
    assert output_quantities.wout.ntor == wout["ntor"][()]
    assert output_quantities.wout.lasym == (wout["lasym__logical__"][()] != 0)

    assert output_quantities.wout.ns == wout["ns"][()]
    assert output_quantities.wout.ftolv == wout["ftolv"][()]
    assert output_quantities.wout.maximum_iterations == wout["niter"][()]

    assert output_quantities.wout.lfreeb == (wout["lfreeb__logical__"][()] != 0)
    assert (
        output_quantities.wout.mgrid_file
        == wout["mgrid_file"][()].tobytes().decode().strip()
    )
    # extcur is scalar in a fixed-boundary VMEC wout file -> don't test here...
    assert output_quantities.wout.mgrid_file, (
        wout["mgrid_file"][()].tobytes().decode().strip()
    )

    # -------------------
    # scalar quantities

    assert is_close_ra(output_quantities.wout.wb, wout["wb"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.wp, wout["wp"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.rmax_surf, wout["rmax_surf"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.rmin_surf, wout["rmin_surf"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.zmax_surf, wout["zmax_surf"][()], 1.0e-11)

    mnmax = output_quantities.wout.mnmax
    assert mnmax == wout["mnmax"][()]

    mnmax_nyq = output_quantities.wout.mnmax_nyq
    assert mnmax_nyq == wout["mnmax_nyq"][()]

    assert output_quantities.wout.ier_flag == wout["ier_flag"][()]

    assert is_close_ra(output_quantities.wout.aspect, wout["aspect"][()], 1.0e-11)

    assert is_close_ra(output_quantities.wout.betatot, wout["betatotal"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.betapol, wout["betapol"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.betator, wout["betator"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.betaxis, wout["betaxis"][()], 1.0e-11)

    assert is_close_ra(output_quantities.wout.b0, wout["b0"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.rbtor0, wout["rbtor0"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.rbtor, wout["rbtor"][()], 1.0e-11)

    assert is_close_ra(output_quantities.wout.IonLarmor, wout["IonLarmor"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.VolAvgB, wout["volavgB"][()], 1.0e-11)

    assert is_close_ra(output_quantities.wout.ctor, wout["ctor"][()], 1.0e-6)

    assert is_close_ra(output_quantities.wout.Aminor_p, wout["Aminor_p"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.Rmajor_p, wout["Rmajor_p"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.volume_p, wout["volume_p"][()], 1.0e-11)

    assert is_close_ra(output_quantities.wout.fsqr, wout["fsqr"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.fsqz, wout["fsqz"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.fsql, wout["fsql"][()], 1.0e-11)

    # -------------------
    # one-dimensional array quantities

    assert is_close_ra(output_quantities.wout.iota_full, wout["iotaf"][()], 1.0e-6)
    assert is_close_ra(
        output_quantities.wout.safety_factor, wout["q_factor"][()], 1.0e-10
    )
    assert is_close_ra(output_quantities.wout.pressure_full, wout["presf"][()], 1.0e-8)
    assert is_close_ra(output_quantities.wout.toroidal_flux, wout["phi"][()], 1.0e-8)
    assert is_close_ra(output_quantities.wout.phipf, wout["phipf"][()], 1.0e-8)
    assert is_close_ra(output_quantities.wout.poloidal_flux, wout["chi"][()], 1.0e-8)
    assert is_close_ra(output_quantities.wout.chipf, wout["chipf"][()], 1.0e-8)
    assert is_close_ra(output_quantities.wout.jcuru, wout["jcuru"][()], 1.0e-6)
    assert is_close_ra(output_quantities.wout.jcurv, wout["jcurv"][()], 1.0e-6)

    assert is_close_ra(output_quantities.wout.iota_half, wout["iotas"][()][1:], 1.0e-8)
    assert is_close_ra(output_quantities.wout.mass, wout["mass"][()][1:], 1.0e-8)
    assert is_close_ra(
        output_quantities.wout.pressure_half, wout["pres"][()][1:], 1.0e-8
    )
    assert is_close_ra(output_quantities.wout.beta, wout["beta_vol"][()][1:], 1.0e-8)
    assert is_close_ra(output_quantities.wout.buco, wout["buco"][()][1:], 1.0e-8)
    assert is_close_ra(output_quantities.wout.bvco, wout["bvco"][()][1:], 1.0e-8)
    assert is_close_ra(output_quantities.wout.dVds, wout["vp"][()][1:], 1.0e-8)
    assert is_close_ra(output_quantities.wout.spectral_width, wout["specw"][()], 1.0e-8)
    assert is_close_ra(output_quantities.wout.phips, wout["phips"][()][1:], 1.0e-8)
    assert is_close_ra(output_quantities.wout.overr, wout["over_r"][()][1:], 1.0e-8)

    assert is_close_ra(output_quantities.wout.jdotb, wout["jdotb"][()], 1.0e-6)
    assert is_close_ra(output_quantities.wout.bdotgradv, wout["bdotgradv"][()], 1.0e-8)

    assert is_close_ra(output_quantities.wout.DMerc, wout["DMerc"][()], 1.0e-8)
    assert is_close_ra(output_quantities.wout.Dshear, wout["DShear"][()], 1.0e-8)
    assert is_close_ra(output_quantities.wout.Dwell, wout["DWell"][()], 1.0e-9)
    assert is_close_ra(output_quantities.wout.Dcurr, wout["DCurr"][()], 1.0e-8)
    assert is_close_ra(output_quantities.wout.Dgeod, wout["DGeod"][()], 1.0e-8)

    # Fortran VMEC defines equif over all flux surfaces,
    # but then skips the axis and boundary during writing.
    assert is_close_ra(output_quantities.wout.equif, wout["equif"][()], 1.0e-6)

    # TODO(jons): test curlabel, once implemented

    # TODO(jons): test potvac, once implemented

    # -------------------
    # mode numbers for Fourier coefficient arrays below

    assert is_close_ra(output_quantities.wout.xm, wout["xm"][()], 1.0e-15)
    assert is_close_ra(output_quantities.wout.xn, wout["xn"][()], 1.0e-15)
    assert is_close_ra(output_quantities.wout.xm_nyq, wout["xm_nyq"][()], 1.0e-15)
    assert is_close_ra(output_quantities.wout.xn_nyq, wout["xn_nyq"][()], 1.0e-15)

    # -------------------
    # stellarator-symmetric Fourier coefficients

    assert is_close_ra(output_quantities.wout.raxis_c, wout["raxis_cc"][()], 1.0e-11)
    assert is_close_ra(output_quantities.wout.zaxis_s, wout["zaxis_cs"][()], 1.0e-11)

    assert is_close_ra(
        np.reshape(output_quantities.wout.rmnc, [ns, mnmax], order="C"),
        wout["rmnc"][()],
        1.0e-11,
    )

    assert is_close_ra(
        np.reshape(output_quantities.wout.zmns, [ns, mnmax], order="C"),
        wout["zmns"][()],
        1.0e-11,
    )

    # NOTE: lmns_full is not available from Fortran VMEC

    assert is_close_ra(
        np.reshape(output_quantities.wout.lmns, [ns - 1, mnmax], order="C"),
        wout["lmns"][()][1:, :],
        1.0e-10,
    )

    assert is_close_ra(
        np.reshape(output_quantities.wout.gmnc, [ns - 1, mnmax_nyq], order="C"),
        wout["gmnc"][()][1:, :],
        1.0e-11,
    )

    assert is_close_ra(
        np.reshape(output_quantities.wout.bmnc, [ns - 1, mnmax_nyq], order="C"),
        wout["bmnc"][()][1:, :],
        1.0e-11,
    )

    assert is_close_ra(
        np.reshape(output_quantities.wout.bsubumnc, [ns - 1, mnmax_nyq], order="C"),
        wout["bsubumnc"][()][1:, :],
        1.0e-11,
    )

    assert is_close_ra(
        np.reshape(output_quantities.wout.bsubvmnc, [ns - 1, mnmax_nyq], order="C"),
        wout["bsubvmnc"][()][1:, :],
        1.0e-11,
    )

    assert is_close_ra(
        np.reshape(output_quantities.wout.bsubsmns, [ns, mnmax_nyq], order="C"),
        wout["bsubsmns"][()],
        1.0e-11,
    )

    assert is_close_ra(
        np.reshape(output_quantities.wout.bsupumnc, [ns - 1, mnmax_nyq], order="C"),
        wout["bsupumnc"][()][1:, :],
        1.0e-10,
    )

    assert is_close_ra(
        np.reshape(output_quantities.wout.bsupvmnc, [ns - 1, mnmax_nyq], order="C"),
        wout["bsupvmnc"][()][1:, :],
        1.0e-11,
    )

    # -------------------
    # non-stellarator-symmetric Fourier coefficients

    # TODO(jons): implement these once VMEC++ has
    # the non-stellarator-symmetric parts implemented

    wout.close()


def test_vmecpp_run_from_inmemory_mgrid():
    indata_fname = TEST_DATA_DIR / "cth_like_free_bdy.json"
    coils_fname = TEST_DATA_DIR / "coils.cth_like"
    makegrid_params_fname = TEST_DATA_DIR / "makegrid_parameters_cth_like.json"

    indata = vmec.VmecINDATAPyWrapper.from_file(indata_fname)
    indata.niter_array = np.array([1])  # to speed up the test

    mgrid_params = vmec.MakegridParameters.from_file(makegrid_params_fname)
    magnetic_configuration = vmec.MagneticConfiguration.from_file(coils_fname)
    magnetic_response_table = vmec.compute_magnetic_field_response_table(
        mgrid_params, magnetic_configuration
    )

    # we expect that with a single iteration VMEC++ complains about non-convergence
    with pytest.raises(RuntimeError) as vmecpp_error:
        vmec.run(indata, magnetic_response_table)
    assert "VMEC++ did not converge" in str(vmecpp_error.value)


def test_makegridparameters_constructor():
    makegrid_parameters = vmec.MakegridParameters(
        normalize_by_currents=True,
        assume_stellarator_symmetry=True,
        number_of_field_periods=1,
        r_grid_minimum=-1.0,
        r_grid_maximum=1.0,
        number_of_r_grid_points=100,
        z_grid_minimum=-1.0,
        z_grid_maximum=1.0,
        number_of_z_grid_points=100,
        number_of_phi_grid_points=10,
    )

    assert makegrid_parameters.normalize_by_currents
    assert makegrid_parameters.assume_stellarator_symmetry
    assert makegrid_parameters.number_of_field_periods == 1
    assert makegrid_parameters.r_grid_minimum == -1.0
    assert makegrid_parameters.r_grid_maximum == 1.0
    assert makegrid_parameters.number_of_r_grid_points == 100
    assert makegrid_parameters.z_grid_minimum == -1.0
    assert makegrid_parameters.z_grid_maximum == 1.0
    assert makegrid_parameters.number_of_z_grid_points == 100
    assert makegrid_parameters.number_of_phi_grid_points == 10


def test_magneticfieldresponsetable_constructor():
    makegrid_parameters = vmec.MakegridParameters(
        normalize_by_currents=True,
        assume_stellarator_symmetry=True,
        number_of_field_periods=1,
        r_grid_minimum=-1.0,
        r_grid_maximum=1.0,
        number_of_r_grid_points=10,
        z_grid_minimum=-1.0,
        z_grid_maximum=1.0,
        number_of_z_grid_points=10,
        number_of_phi_grid_points=10,
    )

    n_coil_groups = 10
    magnetic_field_shape = (
        n_coil_groups,
        makegrid_parameters.number_of_r_grid_points
        * makegrid_parameters.number_of_z_grid_points
        * makegrid_parameters.number_of_phi_grid_points,
    )
    magnetic_field_component = np.arange(
        magnetic_field_shape[0] * magnetic_field_shape[1], dtype=float
    ).reshape(10, -1)

    magnetic_field_response_table = vmec.MagneticFieldResponseTable(
        parameters=makegrid_parameters,
        b_r=magnetic_field_component,
        b_p=magnetic_field_component,
        b_z=magnetic_field_component,
    )

    np.testing.assert_allclose(
        magnetic_field_response_table.b_r, magnetic_field_component
    )
    np.testing.assert_allclose(
        magnetic_field_response_table.b_p, magnetic_field_component
    )
    np.testing.assert_allclose(
        magnetic_field_response_table.b_z, magnetic_field_component
    )

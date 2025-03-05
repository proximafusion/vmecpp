// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/output_quantities/output_quantities.h"

#include <netcdf.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/netcdf_io/netcdf_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/vmec/vmec.h"

using nlohmann::json;

using file_io::ReadFile;
using netcdf_io::NetcdfReadArray1D;
using netcdf_io::NetcdfReadArray2D;
using netcdf_io::NetcdfReadBool;
using netcdf_io::NetcdfReadChar;
using netcdf_io::NetcdfReadDouble;
using netcdf_io::NetcdfReadInt;
using netcdf_io::NetcdfReadString;
using testing::IsCloseRelAbs;

using ::testing::ElementsAreArray;
using ::testing::TestWithParam;
using ::testing::Values;

namespace fs = std::filesystem;

namespace vmecpp {

// used to specify case-specific tolerances
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
};

class WOutFileContentsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(WOutFileContentsTest, CheckWOutFileContents) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // run until convergence
  bool reached_checkpoint = vmec.run().value();
  ASSERT_FALSE(reached_checkpoint);

  const OutputQuantities& output_quantities = vmec.output_quantities_;
  const WOutFileContents& wout = output_quantities.wout;

  // Note that the actual `wout` file itself is taken as reference here.
  filename =
      absl::StrFormat("vmecpp/test_data/wout_%s.nc", data_source_.identifier);

  int ncid;
  ASSERT_EQ(nc_open(filename.c_str(), NC_NOWRITE, &ncid), NC_NOERR);

  EXPECT_EQ(wout.sign_of_jacobian, NetcdfReadInt(ncid, "signgs"));

  EXPECT_EQ(wout.gamma, NetcdfReadDouble(ncid, "gamma"));

  EXPECT_EQ(wout.pcurr_type, NetcdfReadString(ncid, "pcurr_type"));
  EXPECT_EQ(wout.pmass_type, NetcdfReadString(ncid, "pmass_type"));
  EXPECT_EQ(wout.piota_type, NetcdfReadString(ncid, "piota_type"));

  std::vector<double> reference_am = NetcdfReadArray1D(ncid, "am");
  // remove zero-padding at end
  reference_am.resize(wout.am.size());
  EXPECT_THAT(wout.am, ElementsAreArray(reference_am));
  // TODO(jons): check for spline profiles -> need to check am_aux_*

  if (vmec_indata->ncurr == 0) {
    // constrained-iota; ignore current profile coefficients
    // TODO(jons): check for spline profiles -> need to check ai_aux_*
    std::vector<double> reference_ai = NetcdfReadArray1D(ncid, "ai");
    // remove zero-padding at end
    reference_ai.resize(wout.ai.size());
    EXPECT_THAT(wout.ai, ElementsAreArray(reference_ai));
  } else {
    // constrained-current
    // TODO(jons): check for spline profiles -> need to check ac_aux_*
    std::vector<double> reference_ac = NetcdfReadArray1D(ncid, "ac");
    reference_ac.resize(wout.ac.size());
    EXPECT_THAT(wout.ac, ElementsAreArray(reference_ac));

    if (wout.ai.size() > 0) {
      // iota profile (if present) taken as initial guess for first iteration
      // TODO(jons): check for spline profiles -> need to check ai_aux_*
      std::vector<double> reference_ai = NetcdfReadArray1D(ncid, "ai");
      // remove zero-padding at end
      reference_ai.resize(wout.ai.size());
      EXPECT_THAT(wout.ai, ElementsAreArray(reference_ai));
    }
  }

  EXPECT_EQ(wout.nfp, NetcdfReadInt(ncid, "nfp"));
  EXPECT_EQ(wout.mpol, NetcdfReadInt(ncid, "mpol"));
  EXPECT_EQ(wout.ntor, NetcdfReadInt(ncid, "ntor"));
  EXPECT_EQ(wout.lasym, NetcdfReadBool(ncid, "lasym"));

  EXPECT_EQ(wout.ns, NetcdfReadInt(ncid, "ns"));
  EXPECT_EQ(wout.ftolv, NetcdfReadDouble(ncid, "ftolv"));
  EXPECT_EQ(wout.maximum_iterations, NetcdfReadInt(ncid, "niter"));

  EXPECT_EQ(wout.lfreeb, NetcdfReadBool(ncid, "lfreeb"));
  if (wout.lfreeb) {
    // The reference data is generated using educational_VMEC,
    // which is run from within //vmecpp/test_data.
    // This means that the mgrid file (in that folder) is in the current working
    // directory and no path prefix is required for the Fortran input files.
    // However, the VMEC++ tests are executed from the repository root,
    // which means that the mgrid file needs to be specified with a path prefix
    // "vmecpp/test_data/". This is taken care of in
    // //vmecpp/test_data/regenerate_test_data.sh.
    EXPECT_EQ(wout.mgrid_file,
              absl::StrCat("vmecpp/test_data/",
                           NetcdfReadString(ncid, "mgrid_file")));
    std::vector<double> reference_extcur = NetcdfReadArray1D(ncid, "extcur");
    EXPECT_THAT(wout.extcur, ElementsAreArray(reference_extcur));
  }
  EXPECT_EQ(wout.mgrid_mode, NetcdfReadString(ncid, "mgrid_mode"));

  // -------------------
  // scalar quantities

  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "wb"), wout.wb, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "wp"), wout.wp, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "rmax_surf"), wout.rmax_surf,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "rmin_surf"), wout.rmin_surf,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "zmax_surf"), wout.zmax_surf,
                            tolerance));

  EXPECT_EQ(wout.mnmax, NetcdfReadInt(ncid, "mnmax"));
  EXPECT_EQ(wout.mnmax_nyq, NetcdfReadInt(ncid, "mnmax_nyq"));

  EXPECT_EQ(wout.ier_flag, NetcdfReadInt(ncid, "ier_flag"));

  EXPECT_TRUE(
      IsCloseRelAbs(NetcdfReadDouble(ncid, "aspect"), wout.aspect, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "betatotal"), wout.betatot,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "betapol"), wout.betapol,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "betator"), wout.betator,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "betaxis"), wout.betaxis,
                            tolerance));

  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "b0"), wout.b0, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(NetcdfReadDouble(ncid, "rbtor0"), wout.rbtor0, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(NetcdfReadDouble(ncid, "rbtor"), wout.rbtor, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "IonLarmor"), wout.IonLarmor,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "volavgB"), wout.VolAvgB,
                            tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(NetcdfReadDouble(ncid, "ctor"), wout.ctor, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "Aminor_p"), wout.Aminor_p,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "Rmajor_p"), wout.Rmajor_p,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "volume_p"), wout.volume_p,
                            tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(NetcdfReadDouble(ncid, "fsqr"), wout.fsqr, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(NetcdfReadDouble(ncid, "fsqz"), wout.fsqz, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(NetcdfReadDouble(ncid, "fsql"), wout.fsql, tolerance));

  // -------------------
  // one-dimensional array quantities

  std::vector<double> reference_iota_full = NetcdfReadArray1D(ncid, "iotaf");
  std::vector<double> reference_safety_factor =
      NetcdfReadArray1D(ncid, "q_factor");
  std::vector<double> reference_pressure_full =
      NetcdfReadArray1D(ncid, "presf");
  std::vector<double> reference_toroidal_flux = NetcdfReadArray1D(ncid, "phi");
  std::vector<double> reference_poloidal_flux = NetcdfReadArray1D(ncid, "chi");
  std::vector<double> reference_phipf = NetcdfReadArray1D(ncid, "phipf");
  std::vector<double> reference_chipf = NetcdfReadArray1D(ncid, "chipf");
  std::vector<double> reference_jcuru = NetcdfReadArray1D(ncid, "jcuru");
  std::vector<double> reference_jcurv = NetcdfReadArray1D(ncid, "jcurv");
  std::vector<double> reference_spectral_width =
      NetcdfReadArray1D(ncid, "specw");
  for (int j_f = 0; j_f < fc.ns; ++j_f) {
    EXPECT_TRUE(
        IsCloseRelAbs(reference_iota_full[j_f], wout.iota_full[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_safety_factor[j_f],
                              wout.safety_factor[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_pressure_full[j_f],
                              wout.pressure_full[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_toroidal_flux[j_f],
                              wout.toroidal_flux[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_poloidal_flux[j_f],
                              wout.poloidal_flux[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_phipf[j_f], wout.phipf[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_chipf[j_f], wout.chipf[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_jcuru[j_f], wout.jcuru[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_jcurv[j_f], wout.jcurv[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_spectral_width[j_f],
                              wout.spectral_width[j_f], tolerance));
  }  // jF

  std::vector<double> reference_iota_half = NetcdfReadArray1D(ncid, "iotas");
  std::vector<double> reference_mass_half = NetcdfReadArray1D(ncid, "mass");
  std::vector<double> reference_pressure_half = NetcdfReadArray1D(ncid, "pres");
  std::vector<double> reference_beta = NetcdfReadArray1D(ncid, "beta_vol");
  std::vector<double> reference_buco = NetcdfReadArray1D(ncid, "buco");
  std::vector<double> reference_bvco = NetcdfReadArray1D(ncid, "bvco");
  std::vector<double> reference_d_vds = NetcdfReadArray1D(ncid, "vp");
  std::vector<double> reference_phips = NetcdfReadArray1D(ncid, "phips");
  std::vector<double> reference_overr = NetcdfReadArray1D(ncid, "over_r");
  for (int j_h = 0; j_h < fc.ns - 1; ++j_h) {
    EXPECT_TRUE(IsCloseRelAbs(reference_iota_half[j_h + 1], wout.iota_half[j_h],
                              tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_mass_half[j_h + 1], wout.mass[j_h], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_pressure_half[j_h + 1],
                              wout.pressure_half[j_h], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_beta[j_h + 1], wout.beta[j_h], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_buco[j_h + 1], wout.buco[j_h], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_bvco[j_h + 1], wout.bvco[j_h], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_d_vds[j_h + 1], wout.dVds[j_h], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_phips[j_h + 1], wout.phips[j_h], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_overr[j_h + 1], wout.overr[j_h], tolerance));
  }  // jH

  std::vector<double> reference_jdotb = NetcdfReadArray1D(ncid, "jdotb");
  std::vector<double> reference_bdotgradv =
      NetcdfReadArray1D(ncid, "bdotgradv");
  for (int j_f = 0; j_f < fc.ns; ++j_f) {
    EXPECT_TRUE(IsCloseRelAbs(reference_jdotb[j_f], wout.jdotb[j_f], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_bdotgradv[j_f], wout.bdotgradv[j_f], tolerance));
  }  // jF

  std::vector<double> reference_d_merc = NetcdfReadArray1D(ncid, "DMerc");
  std::vector<double> reference_dshear = NetcdfReadArray1D(ncid, "DShear");
  std::vector<double> reference_dwell = NetcdfReadArray1D(ncid, "DWell");
  std::vector<double> reference_dcurr = NetcdfReadArray1D(ncid, "DCurr");
  std::vector<double> reference_dgeod = NetcdfReadArray1D(ncid, "DGeod");
  for (int j_f = 0; j_f < fc.ns; ++j_f) {
    EXPECT_TRUE(IsCloseRelAbs(reference_d_merc[j_f], wout.DMerc[j_f], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_dshear[j_f], wout.Dshear[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_dwell[j_f], wout.Dwell[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_dcurr[j_f], wout.Dcurr[j_f], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_dgeod[j_f], wout.Dgeod[j_f], tolerance));
  }  // jF

  std::vector<double> reference_equif = NetcdfReadArray1D(ncid, "equif");
  for (int j_f = 0; j_f < fc.ns; ++j_f) {
    EXPECT_TRUE(IsCloseRelAbs(reference_equif[j_f], wout.equif[j_f], tolerance));
  }

  // TODO(jons): curlabel, potvac: once free-boundary works

  // -------------------
  // mode numbers for Fourier coefficient arrays below

  std::vector<double> reference_xm = NetcdfReadArray1D(ncid, "xm");
  std::vector<double> reference_xn = NetcdfReadArray1D(ncid, "xn");
  for (int mn = 0; mn < wout.mnmax; ++mn) {
    EXPECT_EQ(wout.xm[mn], reference_xm[mn]);
    EXPECT_EQ(wout.xn[mn], reference_xn[mn]);
  }  // mn

  std::vector<double> reference_xm_nyq = NetcdfReadArray1D(ncid, "xm_nyq");
  std::vector<double> reference_xn_nyq = NetcdfReadArray1D(ncid, "xn_nyq");
  for (int mn_nyq = 0; mn_nyq < wout.mnmax_nyq; ++mn_nyq) {
    EXPECT_EQ(wout.xm_nyq[mn_nyq], reference_xm_nyq[mn_nyq]);
    EXPECT_EQ(wout.xn_nyq[mn_nyq], reference_xn_nyq[mn_nyq]);
  }  // mn_nyq

  // -------------------
  // stellarator-symmetric Fourier coefficients

  std::vector<double> reference_raxis_cc = NetcdfReadArray1D(ncid, "raxis_cc");
  std::vector<double> reference_zaxis_cs = NetcdfReadArray1D(ncid, "zaxis_cs");
  for (int n = 0; n <= wout.ntor; ++n) {
    EXPECT_TRUE(
        IsCloseRelAbs(reference_raxis_cc[n], wout.raxis_c[n], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_zaxis_cs[n], wout.zaxis_s[n], tolerance));
  }  // n

  std::vector<std::vector<double>> reference_rmnc =
      NetcdfReadArray2D(ncid, "rmnc");
  std::vector<std::vector<double>> reference_zmns =
      NetcdfReadArray2D(ncid, "zmns");
  for (int j_f = 0; j_f < fc.ns; ++j_f) {
    for (int mn = 0; mn < s.mnmax; ++mn) {
      EXPECT_TRUE(IsCloseRelAbs(reference_rmnc[j_f][mn],
                                wout.rmnc(j_f * s.mnmax + mn), tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_zmns[j_f][mn],
                                wout.zmns(j_f * s.mnmax + mn), tolerance));
    }  // mn
  }    // jF

  std::vector<std::vector<double>> reference_lmns =
      NetcdfReadArray2D(ncid, "lmns");
  for (int j_h = 0; j_h < fc.ns - 1; ++j_h) {
    for (int mn = 0; mn < s.mnmax; ++mn) {
      EXPECT_TRUE(IsCloseRelAbs(reference_lmns[j_h + 1][mn],
                                wout.lmns(j_h * s.mnmax + mn), tolerance));
    }  // mn
  }    // jH

  std::vector<std::vector<double>> reference_gmnc =
      NetcdfReadArray2D(ncid, "gmnc");
  std::vector<std::vector<double>> reference_bmnc =
      NetcdfReadArray2D(ncid, "bmnc");
  std::vector<std::vector<double>> reference_bsubumnc =
      NetcdfReadArray2D(ncid, "bsubumnc");
  std::vector<std::vector<double>> reference_bsubvmnc =
      NetcdfReadArray2D(ncid, "bsubvmnc");
  std::vector<std::vector<double>> reference_bsubsmns =
      NetcdfReadArray2D(ncid, "bsubsmns");
  std::vector<std::vector<double>> reference_bsupumnc =
      NetcdfReadArray2D(ncid, "bsupumnc");
  std::vector<std::vector<double>> reference_bsupvmnc =
      NetcdfReadArray2D(ncid, "bsupvmnc");
  for (int j_h = 0; j_h < fc.ns - 1; ++j_h) {
    for (int mn_nyq = 0; mn_nyq < s.mnmax_nyq; ++mn_nyq) {
      EXPECT_TRUE(IsCloseRelAbs(reference_gmnc[j_h + 1][mn_nyq],
                                wout.gmnc(j_h * s.mnmax_nyq + mn_nyq),
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bmnc[j_h + 1][mn_nyq],
                                wout.bmnc(j_h * s.mnmax_nyq + mn_nyq),
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsubumnc[j_h + 1][mn_nyq],
                                wout.bsubumnc(j_h * s.mnmax_nyq + mn_nyq),
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsubvmnc[j_h + 1][mn_nyq],
                                wout.bsubvmnc(j_h * s.mnmax_nyq + mn_nyq),
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsubsmns[j_h + 1][mn_nyq],
                                wout.bsubsmns((j_h + 1) * s.mnmax_nyq + mn_nyq),
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsupumnc[j_h + 1][mn_nyq],
                                wout.bsupumnc(j_h * s.mnmax_nyq + mn_nyq),
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsupvmnc[j_h + 1][mn_nyq],
                                wout.bsupvmnc(j_h * s.mnmax_nyq + mn_nyq),
                                tolerance));
    }  // mn_nyq
  }    // jH

  // also test the wrong extrapolation of bsubsmns
  // beyond the magnetic axis for backward compatibility
  for (int mn_nyq = 0; mn_nyq < s.mnmax_nyq; ++mn_nyq) {
    EXPECT_TRUE(IsCloseRelAbs(reference_bsubsmns[0][mn_nyq],
                              wout.bsubsmns(0 * s.mnmax_nyq + mn_nyq),
                              tolerance));
  }  // mn_nyq

  // -------------------
  // non-stellarator-symmetric Fourier coefficients

  if (s.lasym) {
    std::vector<double> reference_raxis_cs =
        NetcdfReadArray1D(ncid, "raxis_cs");
    std::vector<double> reference_zaxis_cc =
        NetcdfReadArray1D(ncid, "zaxis_cc");
    for (int n = 0; n <= wout.ntor; ++n) {
      EXPECT_TRUE(
          IsCloseRelAbs(reference_raxis_cs[n], wout.raxis_s[n], tolerance));
      EXPECT_TRUE(
          IsCloseRelAbs(reference_zaxis_cc[n], wout.zaxis_c[n], tolerance));
    }  // n

    // TODO(jons): implement tests when first non-stellarator-symmetric test
    // case is ready
  }  // lasym

  ASSERT_EQ(nc_close(ncid), NC_NOERR);
}  // CheckWOutFileContents

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, WOutFileContentsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-7},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-7},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-6},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-6},
           DataSource{.identifier = "cma", .tolerance = 1.0e-6},
           DataSource{.identifier = "cth_like_free_bdy", .tolerance = 1.0e-6}));

}  // namespace vmecpp

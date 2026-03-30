// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/output_quantities/output_quantities.h"

#include <netcdf.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
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

  auto maybe_vmec = Vmec::FromIndata(*vmec_indata);
  ASSERT_TRUE(maybe_vmec.ok());
  Vmec& vmec = **maybe_vmec;

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

  EXPECT_EQ(wout.signgs, NetcdfReadInt(ncid, "signgs"));

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
  EXPECT_EQ(wout.niter, NetcdfReadInt(ncid, "niter"));

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
    EXPECT_EQ(wout.nextcur, static_cast<int>(reference_extcur.size()));
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

  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "betatotal"), wout.betatotal,
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
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "volavgB"), wout.volavgB,
                            tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(NetcdfReadDouble(ncid, "ctor"), wout.ctor, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "Aminor_p"), wout.Aminor_p,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "Rmajor_p"), wout.Rmajor_p,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(NetcdfReadDouble(ncid, "volume_p"), wout.volume,
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
  for (int jF = 0; jF < fc.ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(reference_iota_full[jF], wout.iotaf[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_safety_factor[jF], wout.q_factor[jF],
                              tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_pressure_full[jF], wout.presf[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_toroidal_flux[jF], wout.phi[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_poloidal_flux[jF], wout.chi[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_phipf[jF], wout.phipf[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_chipf[jF], wout.chipf[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_jcuru[jF], wout.jcuru[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_jcurv[jF], wout.jcurv[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_spectral_width[jF], wout.specw[jF], tolerance));
  }  // jF

  std::vector<double> reference_iota_half = NetcdfReadArray1D(ncid, "iotas");
  std::vector<double> reference_mass_half = NetcdfReadArray1D(ncid, "mass");
  std::vector<double> reference_pressure_half = NetcdfReadArray1D(ncid, "pres");
  std::vector<double> reference_beta = NetcdfReadArray1D(ncid, "beta_vol");
  std::vector<double> reference_buco = NetcdfReadArray1D(ncid, "buco");
  std::vector<double> reference_bvco = NetcdfReadArray1D(ncid, "bvco");
  std::vector<double> reference_dVds = NetcdfReadArray1D(ncid, "vp");
  std::vector<double> reference_phips = NetcdfReadArray1D(ncid, "phips");
  std::vector<double> reference_overr = NetcdfReadArray1D(ncid, "over_r");
  for (int jF = 0; jF < fc.ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(reference_iota_half[jF], wout.iotas[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_mass_half[jF], wout.mass[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_pressure_half[jF], wout.pres[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_beta[jF], wout.beta_vol[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_buco[jF], wout.buco[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_bvco[jF], wout.bvco[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_dVds[jF], wout.vp[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_phips[jF], wout.phips[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_overr[jF], wout.over_r[jF], tolerance));
  }  // jF

  std::vector<double> reference_jdotb = NetcdfReadArray1D(ncid, "jdotb");
  std::vector<double> reference_bdotb = NetcdfReadArray1D(ncid, "bdotb");
  std::vector<double> reference_bdotgradv =
      NetcdfReadArray1D(ncid, "bdotgradv");
  for (int jF = 0; jF < fc.ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(reference_jdotb[jF], wout.jdotb[jF], 10 * tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_bdotb[jF], wout.bdotb[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_bdotgradv[jF], wout.bdotgradv[jF], tolerance));
  }  // jF

  std::vector<double> reference_DMerc = NetcdfReadArray1D(ncid, "DMerc");
  std::vector<double> reference_Dshear = NetcdfReadArray1D(ncid, "DShear");
  std::vector<double> reference_Dwell = NetcdfReadArray1D(ncid, "DWell");
  std::vector<double> reference_Dcurr = NetcdfReadArray1D(ncid, "DCurr");
  std::vector<double> reference_Dgeod = NetcdfReadArray1D(ncid, "DGeod");
  for (int jF = 0; jF < fc.ns; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(reference_DMerc[jF], wout.DMerc[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_Dshear[jF], wout.DShear[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_Dwell[jF], wout.DWell[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_Dcurr[jF], wout.DCurr[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(reference_Dgeod[jF], wout.DGeod[jF], tolerance));
  }  // jF

  std::vector<double> reference_equif = NetcdfReadArray1D(ncid, "equif");
  for (int jF = 0; jF < fc.ns; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(reference_equif[jF], wout.equif[jF], tolerance));
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
        IsCloseRelAbs(reference_raxis_cc[n], wout.raxis_cc[n], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(reference_zaxis_cs[n], wout.zaxis_cs[n], tolerance));
  }  // n

  std::vector<std::vector<double>> reference_rmnc =
      NetcdfReadArray2D(ncid, "rmnc");
  std::vector<std::vector<double>> reference_zmns =
      NetcdfReadArray2D(ncid, "zmns");
  for (int jF = 0; jF < fc.ns; ++jF) {
    for (int mn = 0; mn < s.mnmax; ++mn) {
      EXPECT_TRUE(
          IsCloseRelAbs(reference_rmnc[jF][mn], wout.rmnc(mn, jF), tolerance));
      EXPECT_TRUE(
          IsCloseRelAbs(reference_zmns[jF][mn], wout.zmns(mn, jF), tolerance));
    }  // mn
  }  // jF

  std::vector<std::vector<double>> reference_lmns =
      NetcdfReadArray2D(ncid, "lmns");
  for (int jF = 0; jF < fc.ns; ++jF) {
    for (int mn = 0; mn < s.mnmax; ++mn) {
      EXPECT_TRUE(
          IsCloseRelAbs(reference_lmns[jF][mn], wout.lmns(mn, jF), tolerance));
    }  // mn
  }  // jF

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
  for (int jF = 0; jF < fc.ns; ++jF) {
    for (int mn_nyq = 0; mn_nyq < s.mnmax_nyq; ++mn_nyq) {
      EXPECT_TRUE(IsCloseRelAbs(reference_gmnc[jF][mn_nyq],
                                wout.gmnc(mn_nyq, jF), tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bmnc[jF][mn_nyq],
                                wout.bmnc(mn_nyq, jF), tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsubumnc[jF][mn_nyq],
                                wout.bsubumnc(mn_nyq, jF), tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsubvmnc[jF][mn_nyq],
                                wout.bsubvmnc(mn_nyq, jF), tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsubsmns[jF][mn_nyq],
                                wout.bsubsmns(mn_nyq, jF), tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsupumnc[jF][mn_nyq],
                                wout.bsupumnc(mn_nyq, jF), tolerance));
      EXPECT_TRUE(IsCloseRelAbs(reference_bsupvmnc[jF][mn_nyq],
                                wout.bsupvmnc(mn_nyq, jF), tolerance));
    }  // mn_nyq
  }  // jF

  // -------------------
  // non-stellarator-symmetric Fourier coefficients

  if (s.lasym) {
    std::vector<double> reference_raxis_cs =
        NetcdfReadArray1D(ncid, "raxis_cs");
    std::vector<double> reference_zaxis_cc =
        NetcdfReadArray1D(ncid, "zaxis_cc");
    for (int n = 0; n <= wout.ntor; ++n) {
      EXPECT_TRUE(
          IsCloseRelAbs(reference_raxis_cs[n], wout.raxis_cs[n], tolerance));
      EXPECT_TRUE(
          IsCloseRelAbs(reference_zaxis_cc[n], wout.zaxis_cc[n], tolerance));
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
           DataSource{.identifier = "solovev_free_bdy", .tolerance = 5.0e-7},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-6},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-6},
           DataSource{.identifier = "cma", .tolerance = 1.0e-6},
           DataSource{.identifier = "cth_like_free_bdy", .tolerance = 1.0e-6}));

}  // namespace vmecpp

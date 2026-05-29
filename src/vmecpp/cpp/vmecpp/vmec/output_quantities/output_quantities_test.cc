// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/output_quantities/output_quantities.h"

#include <netcdf.h>

#include <filesystem>
#include <fstream>
#include <iostream>
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

// bsubsmns_full holds the full-grid covariant B_s Fourier coefficients. It is
// the forward transform of the full-grid realspace B_s, which is the radial
// half->full interpolation (in PutBSubSOnFullGrid) of the half-grid B_s that
// underlies bsubsmns. Since that radial interpolation is linear and commutes
// with the angular DFT, on the interior full-grid surfaces (where the full-grid
// value is the average of the two neighboring half-grid values) the transforms
// must satisfy
//   bsubsmns_full(:, jF) == 0.5 * (bsubsmns(:, jF+1) + bsubsmns(:, jF)).
// The axis (jF=0) and edge (jF=ns-1) are extrapolated in realspace
// (ExtrapolateBSubS), so they are excluded. This also guards the regression
// where bsubsmns_full was never assigned and was emitted as an empty (0, 0)
// array.
TEST_P(WOutFileContentsTest, BSubSFullMatchesInterpolatedBSubSHalf) {
  const std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  const absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  auto maybe_vmec = Vmec::FromIndata(*vmec_indata);
  ASSERT_TRUE(maybe_vmec.ok());
  Vmec& vmec = **maybe_vmec;

  const bool reached_checkpoint = vmec.run().value();
  ASSERT_FALSE(reached_checkpoint);

  const WOutFileContents& wout = vmec.output_quantities_.wout;

  // Both arrays must be fully sized on the Nyquist mode set over all surfaces;
  // in particular bsubsmns_full must not be the empty matrix it used to default
  // to.
  ASSERT_EQ(wout.bsubsmns_full.rows(), wout.mnmax_nyq);
  ASSERT_EQ(wout.bsubsmns_full.cols(), wout.ns);
  ASSERT_EQ(wout.bsubsmns.rows(), wout.mnmax_nyq);
  ASSERT_EQ(wout.bsubsmns.cols(), wout.ns);

  // The identity is mathematical (up to round-off), so use a tight tolerance
  // independent of the case-specific reference tolerance.
  constexpr double kInterpolationTolerance = 1.0e-10;
  for (int jF = 1; jF < wout.ns - 1; ++jF) {
    for (int mn_nyq = 0; mn_nyq < wout.mnmax_nyq; ++mn_nyq) {
      const double interpolated =
          0.5 * (wout.bsubsmns(mn_nyq, jF + 1) + wout.bsubsmns(mn_nyq, jF));
      EXPECT_TRUE(IsCloseRelAbs(interpolated, wout.bsubsmns_full(mn_nyq, jF),
                                kInterpolationTolerance))
          << "jF = " << jF << ", mn_nyq = " << mn_nyq;
    }  // mn_nyq
  }  // jF
}  // BSubSFullMatchesInterpolatedBSubSHalf

INSTANTIATE_TEST_SUITE_P(
    TestOutputQuantities, WOutFileContentsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-7},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-7},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-6},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-6},
           DataSource{.identifier = "cma", .tolerance = 1.0e-6},
           DataSource{.identifier = "cth_like_free_bdy", .tolerance = 1.0e-6}));

// End-to-end exercise of the spline profile path through a full equilibrium.
// cth_like_fixed_bdy_spline_pressure.json is the cth_like_fixed_bdy case with
// its two_power pressure (1 - s^5)^10 re-expressed as a cubic_spline sampled at
// 201 uniform knots on [0, 1]; that spline reproduces the analytic profile to
// 5e-9 over the interval. Running it to convergence calls evalCubic at every
// half-grid surface on every iteration, and the converged wout is diffed
// against the educational_VMEC golden wout_cth_like_fixed_bdy.nc that the
// analytic two_power case is validated against, at the same 1e-6 tolerance.
// This is the seam the leaf and dispatch tests cannot reach: a spline profile
// driving a real solve to the Fortran-referenced equilibrium. Input-echo fields
// (pmass_type, am) legitimately differ for a spline input and are not compared.
TEST(SplineProfileEquilibrium, CthLikeCubicSplinePressureMatchesFortranGolden) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_fixed_bdy_spline_pressure.json");
  ASSERT_TRUE(indata_json.ok());
  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());
  ASSERT_EQ(vmec_indata->pmass_type, "cubic_spline");

  auto maybe_vmec = Vmec::FromIndata(*vmec_indata);
  ASSERT_TRUE(maybe_vmec.ok());
  Vmec& vmec = **maybe_vmec;
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  const bool reached_checkpoint = vmec.run().value();
  ASSERT_FALSE(reached_checkpoint);  // ran to convergence

  const WOutFileContents& wout = vmec.output_quantities_.wout;

  int ncid;
  ASSERT_EQ(
      nc_open("vmecpp/test_data/wout_cth_like_fixed_bdy.nc", NC_NOWRITE, &ncid),
      NC_NOERR);

  const double tolerance = 1.0e-6;
  double worst_abs = 0.0;
  // worst deviation normalized by each field's own peak magnitude, which avoids
  // the spurious "large relative error" on quantities like the pressure that
  // decay to zero at the plasma edge.
  double worst_norm = 0.0;
  std::string worst_norm_field;

  auto compare = [&](const std::string& name, const std::vector<double>& ref,
                     const std::vector<double>& val) {
    double peak = 1e-300;
    for (double r : ref) {
      peak = std::max(peak, std::abs(r));
    }
    for (size_t i = 0; i < ref.size(); ++i) {
      EXPECT_TRUE(IsCloseRelAbs(ref[i], val[i], tolerance))
          << name << "[" << i << "]: ref=" << ref[i] << " val=" << val[i];
      const double abs_dev = std::abs(ref[i] - val[i]);
      worst_abs = std::max(worst_abs, abs_dev);
      const double norm_dev = abs_dev / peak;
      if (norm_dev > worst_norm) {
        worst_norm = norm_dev;
        worst_norm_field = name;
      }
    }
  };
  auto scalar = [&](const std::string& name, double ref, double val) {
    compare(name, {ref}, {val});
  };
  auto flatten = [&](const std::vector<std::vector<double>>& ref2d, int rows,
                     int cols, auto getter) {
    std::vector<double> ref;
    std::vector<double> val;
    ref.reserve(static_cast<size_t>(rows) * cols);
    val.reserve(static_cast<size_t>(rows) * cols);
    for (int jF = 0; jF < rows; ++jF) {
      for (int mn = 0; mn < cols; ++mn) {
        ref.push_back(ref2d[jF][mn]);
        val.push_back(getter(mn, jF));
      }
    }
    return std::make_pair(ref, val);
  };

  // scalar physics quantities
  scalar("volume_p", NetcdfReadDouble(ncid, "volume_p"), wout.volume);
  scalar("betatotal", NetcdfReadDouble(ncid, "betatotal"), wout.betatotal);
  scalar("aspect", NetcdfReadDouble(ncid, "aspect"), wout.aspect);
  scalar("b0", NetcdfReadDouble(ncid, "b0"), wout.b0);
  scalar("wp", NetcdfReadDouble(ncid, "wp"), wout.wp);
  scalar("wb", NetcdfReadDouble(ncid, "wb"), wout.wb);
  scalar("rbtor", NetcdfReadDouble(ncid, "rbtor"), wout.rbtor);
  scalar("ctor", NetcdfReadDouble(ncid, "ctor"), wout.ctor);
  scalar("Aminor_p", NetcdfReadDouble(ncid, "Aminor_p"), wout.Aminor_p);
  scalar("Rmajor_p", NetcdfReadDouble(ncid, "Rmajor_p"), wout.Rmajor_p);

  // 1D radial profiles, including the spline-driven pressure
  std::vector<double> wpresf(fc.ns), wpres(fc.ns), wmass(fc.ns), wiotaf(fc.ns),
      wiotas(fc.ns), wjcurv(fc.ns);
  for (int jF = 0; jF < fc.ns; ++jF) {
    wpresf[jF] = wout.presf[jF];
    wpres[jF] = wout.pres[jF];
    wmass[jF] = wout.mass[jF];
    wiotaf[jF] = wout.iotaf[jF];
    wiotas[jF] = wout.iotas[jF];
    wjcurv[jF] = wout.jcurv[jF];
  }
  compare("presf", NetcdfReadArray1D(ncid, "presf"), wpresf);
  compare("pres", NetcdfReadArray1D(ncid, "pres"), wpres);
  compare("mass", NetcdfReadArray1D(ncid, "mass"), wmass);
  compare("iotaf", NetcdfReadArray1D(ncid, "iotaf"), wiotaf);
  compare("iotas", NetcdfReadArray1D(ncid, "iotas"), wiotas);
  compare("jcurv", NetcdfReadArray1D(ncid, "jcurv"), wjcurv);

  // flux-surface geometry
  auto [r_ref, r_val] =
      flatten(NetcdfReadArray2D(ncid, "rmnc"), fc.ns, s.mnmax,
              [&](int mn, int jF) { return wout.rmnc(mn, jF); });
  compare("rmnc", r_ref, r_val);
  auto [z_ref, z_val] =
      flatten(NetcdfReadArray2D(ncid, "zmns"), fc.ns, s.mnmax,
              [&](int mn, int jF) { return wout.zmns(mn, jF); });
  compare("zmns", z_ref, z_val);
  auto [l_ref, l_val] =
      flatten(NetcdfReadArray2D(ncid, "lmns"), fc.ns, s.mnmax,
              [&](int mn, int jF) { return wout.lmns(mn, jF); });
  compare("lmns", l_ref, l_val);

  // magnetic field on the Nyquist mode set
  auto [b_ref, b_val] =
      flatten(NetcdfReadArray2D(ncid, "bmnc"), fc.ns, s.mnmax_nyq,
              [&](int mn, int jF) { return wout.bmnc(mn, jF); });
  compare("bmnc", b_ref, b_val);
  auto [bu_ref, bu_val] =
      flatten(NetcdfReadArray2D(ncid, "bsubumnc"), fc.ns, s.mnmax_nyq,
              [&](int mn, int jF) { return wout.bsubumnc(mn, jF); });
  compare("bsubumnc", bu_ref, bu_val);
  auto [bv_ref, bv_val] =
      flatten(NetcdfReadArray2D(ncid, "bsubvmnc"), fc.ns, s.mnmax_nyq,
              [&](int mn, int jF) { return wout.bsubvmnc(mn, jF); });
  compare("bsubvmnc", bv_ref, bv_val);

  ASSERT_EQ(nc_close(ncid), NC_NOERR);

  std::cout << "[spline-vs-Fortran-golden] worst abs dev = " << worst_abs
            << ", worst dev normalized by field peak = " << worst_norm << " ("
            << worst_norm_field << ")" << std::endl;
}

// lforbal through the non-stellarator-symmetric (lasym) free-boundary path.
// With lforbal = true the flux-averaged m=1,n=0 force balance evolves those
// components, so the converged equilibrium differs from the variational one.
// educational_VMEC applies lforbal only to the symmetric (frcc/fzsc) m=1
// components, so this runs the asymmetric cth-like free-boundary case both ways
// and checks that the lforbal run converges and changes the equilibrium,
// exercising the full lasym code path (full-grid cos01/sin01 + asymmetric DFT).
TEST(CthLikeFreeBoundaryAsymLforbal, LforbalConvergesAndModifiesEquilibrium) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy_asym.json");
  ASSERT_TRUE(indata_json.ok());
  const absl::StatusOr<VmecINDATA> base = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(base.ok());
  ASSERT_TRUE(base->lasym);
  ASSERT_FALSE(base->lforbal);

  // Run the case and flatten the R,Z spectra so the two solves can be compared
  // after their Vmec objects go out of scope.
  auto solve = [](const VmecINDATA& indata) -> std::vector<double> {
    auto maybe_vmec = Vmec::FromIndata(indata);
    EXPECT_TRUE(maybe_vmec.ok());
    Vmec& vmec = **maybe_vmec;
    const bool reached_checkpoint = vmec.run().value();
    EXPECT_FALSE(reached_checkpoint);  // ran to convergence
    const WOutFileContents& wout = vmec.output_quantities_.wout;
    std::vector<double> rz;
    rz.reserve(static_cast<size_t>(vmec.fc_.ns) * vmec.s_.mnmax * 2);
    for (int jF = 0; jF < vmec.fc_.ns; ++jF) {
      for (int mn = 0; mn < vmec.s_.mnmax; ++mn) {
        rz.push_back(wout.rmnc(mn, jF));
        rz.push_back(wout.zmns(mn, jF));
      }
    }
    return rz;
  };

  const std::vector<double> variational = solve(*base);

  VmecINDATA with_lforbal = *base;
  with_lforbal.lforbal = true;
  const std::vector<double> lforbal = solve(with_lforbal);

  ASSERT_EQ(lforbal.size(), variational.size());
  double max_abs_diff = 0.0;
  for (size_t i = 0; i < lforbal.size(); ++i) {
    max_abs_diff =
        std::max(max_abs_diff, std::abs(lforbal[i] - variational[i]));
  }
  EXPECT_GT(max_abs_diff, 1e-9)
      << "lforbal did not change the converged lasym equilibrium";

  std::cout << "[lforbal-lasym] max |R,Z| change vs variational = "
            << max_abs_diff << std::endl;
}

}  // namespace vmecpp

#include <filesystem>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "gmock/gmock.h"  // ElementsAreArray
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/tools/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/tools/makegrid/makegrid_lib/makegrid_lib.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

using ::testing::ElementsAreArray;
using ::testing::TestWithParam;
using ::testing::Values;

using file_io::ReadFile;
using magnetics::ImportMagneticConfigurationFromCoilsFile;
using makegrid::ImportMakegridParametersFromFile;
using testing::IsCloseRelAbs;
using vmecpp::RadialPartitioning;
using vmecpp::Sizes;
using vmecpp::Vmec;
using vmecpp::VmecCheckpoint;
using vmecpp::VmecINDATA;
namespace fs = std::filesystem;

// used to specify case-specific tolerances
// and which iterations to test
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::vector<int> iter2_to_test = {1, 2};
};

class GeometryInitializationTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(GeometryInitializationTest, CheckGeometryInitialization) {
  const double tolerance = data_source_.tolerance;

  // SETUP
  const std::string filename =
      absl::StrCat("vmecpp/test_data/", data_source_.identifier, ".json");
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  const std::string out_filename =
      absl::StrCat("vmecpp/test_data/", data_source_.identifier, ".out.h5");
  const auto maybe_out = vmecpp::OutputQuantities::Load(out_filename);
  ASSERT_TRUE(maybe_out.ok());
  const auto& output_quantities = *maybe_out;

  // remove intermediate multi-grid steps for input of restarted run
  int last_multigrid_step = static_cast<int>(indata->ns_array.size()) - 1;
  indata->ns_array = {indata->ns_array[last_multigrid_step]};
  indata->ftol_array = {indata->ftol_array[last_multigrid_step]};
  indata->niter_array = {indata->niter_array[last_multigrid_step]};

  // ACTUAL CALL
  Vmec vmec(*indata);
  const auto checkpoint = VmecCheckpoint::SETUP_INITIAL_STATE;
  const absl::StatusOr<bool> checkpoint_reached = vmec.run(
      checkpoint, /*maximum_iterations=*/0,
      /*maximum_multi_grid_step=*/500, /*initial_state=*/output_quantities);
  ASSERT_TRUE(checkpoint_reached.ok());
  ASSERT_TRUE(*checkpoint_reached);

  // COMPARISON WITH REFERENCES
  // test fourier geometry matches what was in the wout file
  const int ns = vmec.fc_.ns;
  const Sizes& s = vmec.s_;

  for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
    const vmecpp::FourierGeometry& g = *vmec.decomposed_x_[thread_id];
    const RadialPartitioning& rp = *vmec.r_[thread_id];

    const int nsMinF1 = rp.nsMinF1;
    const int nsMaxF1 = rp.nsMaxF1;

    // use another FourierGeometry to represent the WOutFileContents
    vmecpp::FourierGeometry ref_fg(&s, &rp, ns);
    ref_fg.InitFromState(vmec.t_, output_quantities.wout.rmnc,
                         output_quantities.wout.zmns,
                         output_quantities.wout.lmns_full, *vmec.p_[thread_id],
                         vmec.constants_, &(vmec.b_));

    for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
      for (int m = 0; m < s.mpol; ++m) {
        for (int n = 0; n < s.ntor + 1; ++n) {
          const int jFForThisThread = jF - g.nsMin();
          const int idx_fc = (jFForThisThread * s.mpol + m) * (s.ntor + 1) + n;

          EXPECT_TRUE(
              IsCloseRelAbs(ref_fg.rmncc[idx_fc], g.rmncc[idx_fc], tolerance));
          EXPECT_TRUE(
              IsCloseRelAbs(ref_fg.zmnsc[idx_fc], g.zmnsc[idx_fc], tolerance));
          EXPECT_TRUE(
              IsCloseRelAbs(ref_fg.lmnsc[idx_fc], g.lmnsc[idx_fc], tolerance));

          if (s.lthreed) {
            EXPECT_TRUE(IsCloseRelAbs(ref_fg.rmnss[idx_fc], g.rmnss[idx_fc],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(ref_fg.zmncs[idx_fc], g.zmncs[idx_fc],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(ref_fg.lmncs[idx_fc], g.lmncs[idx_fc],
                                      tolerance));
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(HotRestart, GeometryInitializationTest,
                         Values(DataSource{.identifier = "solovev",
                                           .tolerance = 1.0e-15},
                                DataSource{.identifier = "cth_like_fixed_bdy",
                                           .tolerance = 1.0e-15}));

class JacobianTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(JacobianTest, CheckJacobian) {
  // SETUP
  const std::string filename =
      absl::StrCat("vmecpp/test_data/", data_source_.identifier, ".json");
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  const std::string out_filename =
      absl::StrCat("vmecpp/test_data/", data_source_.identifier, ".out.h5");
  const auto maybe_out = vmecpp::OutputQuantities::Load(out_filename);
  ASSERT_TRUE(maybe_out.ok());
  const auto& output_quantities = *maybe_out;

  // remove intermediate multi-grid steps for input of restarted run
  int last_multigrid_step = static_cast<int>(indata->ns_array.size()) - 1;
  indata->ns_array = {indata->ns_array[last_multigrid_step]};
  indata->ftol_array = {indata->ftol_array[last_multigrid_step]};
  indata->niter_array = {indata->niter_array[last_multigrid_step]};

  // ACTUAL Vmec::run CALL
  Vmec vmec(*indata);
  const auto checkpoint = VmecCheckpoint::JACOBIAN;
  const absl::StatusOr<bool> checkpoint_reached = vmec.run(
      checkpoint, /*maximum_iterations=*/0,
      /*maximum_multi_grid_step=*/500, /*initial_state=*/output_quantities);
  ASSERT_TRUE(checkpoint_reached.ok());
  ASSERT_TRUE(*checkpoint_reached);
  // expect that initial restart geometry has no bad jacobian
  EXPECT_EQ(vmec.fc_.restart_reason, vmecpp::RestartReason::NO_RESTART);
}

INSTANTIATE_TEST_SUITE_P(HotRestart, JacobianTest,
                         Values(DataSource{.identifier = "solovev",
                                           .tolerance = 1.0e-15},
                                DataSource{.identifier = "cth_like_fixed_bdy",
                                           .tolerance = 1.0e-15}));

class EnergiesTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(EnergiesTest, CheckEnergies) {
  const double tolerance = data_source_.tolerance;

  // SETUP
  const std::string filename =
      absl::StrCat("vmecpp/test_data/", data_source_.identifier, ".json");
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  const std::string out_filename =
      absl::StrCat("vmecpp/test_data/", data_source_.identifier, ".out.h5");
  const auto maybe_out = vmecpp::OutputQuantities::Load(out_filename);
  ASSERT_TRUE(maybe_out.ok());
  const auto& output_quantities = *maybe_out;

  // remove intermediate multi-grid steps for input of restarted run
  int last_multigrid_step = static_cast<int>(indata->ns_array.size()) - 1;
  indata->ns_array = {indata->ns_array[last_multigrid_step]};
  indata->ftol_array = {indata->ftol_array[last_multigrid_step]};
  indata->niter_array = {indata->niter_array[last_multigrid_step]};

  // ACTUAL Vmec::run CALL
  Vmec vmec(*indata);
  const auto checkpoint = VmecCheckpoint::ENERGY;
  const absl::StatusOr<bool> checkpoint_reached = vmec.run(
      checkpoint, /*maximum_iterations=*/0,
      /*maximum_multi_grid_step=*/500, /*initial_state=*/output_quantities);
  ASSERT_TRUE(checkpoint_reached.ok());
  ASSERT_TRUE(*checkpoint_reached);

  // expect that magnetic and thermal energies are the same as in the initial
  // state
  EXPECT_TRUE(IsCloseRelAbs(output_quantities.wout.wp, vmec.h_.thermalEnergy,
                            tolerance));
  EXPECT_TRUE(IsCloseRelAbs(output_quantities.wout.wb, vmec.h_.magneticEnergy,
                            tolerance));
}

INSTANTIATE_TEST_SUITE_P(HotRestart, EnergiesTest,
                         Values(DataSource{.identifier = "solovev",
                                           .tolerance = 1.0e-15},
                                DataSource{.identifier = "cth_like_fixed_bdy",
                                           .tolerance = 1.0e-15}));

class InvariantForceResidualsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(InvariantForceResidualsTest, CheckInvariantForceResiduals) {
  // SETUP
  const std::string filename =
      absl::StrCat("vmecpp/test_data/", data_source_.identifier, ".json");
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  const std::string out_filename =
      absl::StrCat("vmecpp/test_data/", data_source_.identifier, ".out.h5");
  const auto maybe_out = vmecpp::OutputQuantities::Load(out_filename);
  ASSERT_TRUE(maybe_out.ok());
  const auto& output_quantities = *maybe_out;

  // remove intermediate multi-grid steps for input of restarted run
  int last_multigrid_step = static_cast<int>(indata->ns_array.size()) - 1;
  indata->ns_array = {indata->ns_array[last_multigrid_step]};
  indata->ftol_array = {indata->ftol_array[last_multigrid_step]};
  indata->niter_array = {indata->niter_array[last_multigrid_step]};

  // ACTUAL Vmec::run CALL
  Vmec vmec(*indata);
  const auto checkpoint = VmecCheckpoint::INVARIANT_RESIDUALS;
  const absl::StatusOr<bool> checkpoint_reached = vmec.run(
      checkpoint, /*maximum_iterations=*/0,
      /*maximum_multi_grid_step=*/500, /*initial_state=*/output_quantities);
  ASSERT_TRUE(checkpoint_reached.ok());
  ASSERT_TRUE(*checkpoint_reached);

  // expect that force residuals are less than tolerance
  EXPECT_LE(vmec.fc_.fsqr, vmec.fc_.ftolv);
  EXPECT_LE(vmec.fc_.fsqz, vmec.fc_.ftolv);
  EXPECT_LE(vmec.fc_.fsql, vmec.fc_.ftolv);
}

INSTANTIATE_TEST_SUITE_P(HotRestart, InvariantForceResidualsTest,
                         Values(DataSource{.identifier = "solovev",
                                           .tolerance = 1.0e-15},
                                DataSource{.identifier = "cth_like_fixed_bdy",
                                           .tolerance = 1.0e-15}));

class HotRestartIntegration : public TestWithParam<DataSource> {};

TEST_P(HotRestartIntegration, CheckVaryingBoundary) {
  // test that vmec run results are in agreement between a run from scratch in
  // which we slightly perturb the fixed boundary, and a hot-restarted run with
  // a corresponding boundary.
  // The checks mirror those of CheckWOutFileContents in
  // output_quantities_test.cc.

  // SETUP
  const auto& ds = GetParam();

  const std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", ds.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  VmecINDATA& indata = maybe_indata.value();

  // the one with the slightly perturbed boundary
  VmecINDATA perturbed_indata = indata;
  perturbed_indata.rbc[0] *= 1.0 + 1.0e-8;

  // RUNS
  const auto perturbed_output = vmecpp::run(perturbed_indata);
  ASSERT_TRUE(perturbed_output.ok());

  const auto original_output = vmecpp::run(indata);
  ASSERT_TRUE(original_output.ok());

  // remove intermediate multi-grid steps for input of restarted run
  int last_multigrid_step =
      static_cast<int>(perturbed_indata.ns_array.size()) - 1;
  perturbed_indata.ns_array = {perturbed_indata.ns_array[last_multigrid_step]};
  perturbed_indata.ftol_array = {
      perturbed_indata.ftol_array[last_multigrid_step]};
  perturbed_indata.niter_array = {
      perturbed_indata.niter_array[last_multigrid_step]};

  const auto perturbed_restarted_output =
      vmecpp::run(perturbed_indata, *original_output);
  ASSERT_TRUE(perturbed_restarted_output.ok());

  // Check that contents of wout match
  // Logic copied fromWOutFileContentsTest in output_quantities_test.cc
  // (with free-boundary checks removed as well as several TODOs).
  const auto& expected_wout = perturbed_output->wout;
  const auto& test_wout = perturbed_restarted_output->wout;
  const auto tolerance = ds.tolerance;

  EXPECT_EQ(test_wout.sign_of_jacobian, expected_wout.sign_of_jacobian);
  EXPECT_EQ(test_wout.gamma, expected_wout.gamma);
  EXPECT_EQ(test_wout.pcurr_type, expected_wout.pcurr_type);
  EXPECT_EQ(test_wout.pmass_type, expected_wout.pmass_type);
  EXPECT_EQ(test_wout.piota_type, expected_wout.piota_type);

  EXPECT_THAT(test_wout.am, ElementsAreArray(expected_wout.am));

  if (indata.ncurr == 0) {
    // constrained-iota; ignore current profile coefficients
    EXPECT_THAT(test_wout.ai, ElementsAreArray(expected_wout.ai));
  } else {
    // constrained-current
    EXPECT_THAT(test_wout.ac, ElementsAreArray(expected_wout.ac));

    if (test_wout.ai.size() > 0) {
      // iota profile (if present) taken as initial guess for first iteration
      EXPECT_THAT(test_wout.ai, ElementsAreArray(expected_wout.ai));
    }
  }

  EXPECT_EQ(test_wout.nfp, expected_wout.nfp);
  EXPECT_EQ(test_wout.mpol, expected_wout.mpol);
  EXPECT_EQ(test_wout.ntor, expected_wout.ntor);
  EXPECT_EQ(test_wout.lasym, expected_wout.lasym);

  EXPECT_EQ(test_wout.ns, expected_wout.ns);
  EXPECT_EQ(test_wout.ftolv, expected_wout.ftolv);
  EXPECT_LT(test_wout.maximum_iterations, expected_wout.maximum_iterations);

  EXPECT_EQ(test_wout.lfreeb, expected_wout.lfreeb);
  EXPECT_EQ(test_wout.mgrid_mode, expected_wout.mgrid_mode);

  // -------------------
  // scalar quantities

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.wb, test_wout.wb, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(expected_wout.wp, test_wout.wp, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.rmax_surf, test_wout.rmax_surf, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.rmin_surf, test_wout.rmin_surf, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.zmax_surf, test_wout.zmax_surf, tolerance));

  EXPECT_EQ(test_wout.mnmax, expected_wout.mnmax);
  EXPECT_EQ(test_wout.mnmax_nyq, expected_wout.mnmax_nyq);

  EXPECT_EQ(test_wout.ier_flag, expected_wout.ier_flag);

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.aspect, test_wout.aspect, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.betatot, test_wout.betatot, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.betapol, test_wout.betapol, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.betator, test_wout.betator, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.betaxis, test_wout.betaxis, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.b0, test_wout.b0, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.rbtor0, test_wout.rbtor0, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(expected_wout.rbtor, test_wout.rbtor, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.IonLarmor, test_wout.IonLarmor, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.VolAvgB, test_wout.VolAvgB, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.ctor, test_wout.ctor, tolerance));

  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.Aminor_p, test_wout.Aminor_p, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.Rmajor_p, test_wout.Rmajor_p, tolerance));
  EXPECT_TRUE(
      IsCloseRelAbs(expected_wout.volume_p, test_wout.volume_p, tolerance));

  EXPECT_TRUE(IsCloseRelAbs(expected_wout.fsqr, test_wout.fsqr, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(expected_wout.fsqz, test_wout.fsqz, tolerance));
  EXPECT_TRUE(IsCloseRelAbs(expected_wout.fsql, test_wout.fsql, tolerance));

  // -------------------
  // one-dimensional array quantities

  const int ns = static_cast<int>(expected_wout.iota_full.size());
  for (int jF = 0; jF < ns; ++jF) {
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.iota_full[jF],
                              test_wout.iota_full[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.safety_factor[jF],
                              test_wout.safety_factor[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.pressure_full[jF],
                              test_wout.pressure_full[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.toroidal_flux[jF],
                              test_wout.toroidal_flux[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.poloidal_flux[jF],
                              test_wout.poloidal_flux[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.phipf[jF], test_wout.phipf[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.chipf[jF], test_wout.chipf[jF], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.jcuru[jF], test_wout.jcuru[jF], tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.jcurv[jF], test_wout.jcurv[jF], tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.spectral_width[jF],
                              test_wout.spectral_width[jF], tolerance));
  }  // jF

  for (int jH = 0; jH < ns - 1; ++jH) {
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.iota_half[jH],
                              test_wout.iota_half[jH], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.mass[jH], test_wout.mass[jH], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.pressure_half[jH],
                              test_wout.pressure_half[jH], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.beta[jH], test_wout.beta[jH], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.buco[jH], test_wout.buco[jH], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.bvco[jH], test_wout.bvco[jH], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.dVds[jH], test_wout.dVds[jH], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.phips[jH], test_wout.phips[jH], tolerance));
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.overr[jH], test_wout.overr[jH], tolerance));
  }  // jH

  for (int jF = 0; jF < ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.jdotb[jF], test_wout.jdotb[jF], tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.bdotgradv[jF],
                              test_wout.bdotgradv[jF], tolerance));
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.DMerc[jF], test_wout.DMerc[jF], tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.Dshear[jF], test_wout.Dshear[jF],
                              tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.Dwell[jF], test_wout.Dwell[jF], tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.Dcurr[jF], test_wout.Dcurr[jF], tolerance))
        << "jF = " << jF;
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.Dgeod[jF], test_wout.Dgeod[jF], tolerance))
        << "jF = " << jF;
  }  // jF

  for (int jF = 0; jF < ns; ++jF) {
    EXPECT_TRUE(
        IsCloseRelAbs(expected_wout.equif[jF], test_wout.equif[jF], tolerance))
        << "jF = " << jF;
  }

  // // -------------------
  // // mode numbers for Fourier coefficient arrays below

  for (int mn = 0; mn < test_wout.mnmax; ++mn) {
    EXPECT_EQ(test_wout.xm[mn], expected_wout.xm[mn]);
    EXPECT_EQ(test_wout.xn[mn], expected_wout.xn[mn]);
  }  // mn

  for (int mn_nyq = 0; mn_nyq < test_wout.mnmax_nyq; ++mn_nyq) {
    EXPECT_EQ(test_wout.xm_nyq[mn_nyq], expected_wout.xm_nyq[mn_nyq]);
    EXPECT_EQ(test_wout.xn_nyq[mn_nyq], expected_wout.xn_nyq[mn_nyq]);
  }  // mn_nyq

  // // -------------------
  // // stellarator-symmetric Fourier coefficients

  for (int n = 0; n <= test_wout.ntor; ++n) {
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.raxis_c[n], test_wout.raxis_c[n],
                              tolerance));
    EXPECT_TRUE(IsCloseRelAbs(expected_wout.zaxis_s[n], test_wout.zaxis_s[n],
                              tolerance));
  }  // n

  for (int jF = 0; jF < ns; ++jF) {
    for (int mn = 0; mn < test_wout.mnmax; ++mn) {
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.rmnc(jF * test_wout.mnmax + mn),
                                test_wout.rmnc(jF * test_wout.mnmax + mn),
                                tolerance))
          << "jF = " << jF << " mn = " << mn;
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.zmns(jF * test_wout.mnmax + mn),
                                test_wout.zmns(jF * test_wout.mnmax + mn),
                                tolerance))
          << "jF = " << jF << " mn = " << mn;
    }  // mn
  }    // jF

  for (int jH = 0; jH < ns - 1; ++jH) {
    for (int mn = 0; mn < test_wout.mnmax; ++mn) {
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.lmns(jH * test_wout.mnmax + mn),
                                test_wout.lmns(jH * test_wout.mnmax + mn),
                                tolerance))
          << "jH = " << jH << " mn = " << mn;
    }  // mn
  }    // jH

  for (int jH = 0; jH < ns - 1; ++jH) {
    for (int mn_nyq = 0; mn_nyq < test_wout.mnmax_nyq; ++mn_nyq) {
      EXPECT_TRUE(IsCloseRelAbs(
          expected_wout.gmnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.gmnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(
          expected_wout.bmnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bmnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(
          expected_wout.bsubumnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsubumnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(
          expected_wout.bsubvmnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsubvmnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      // FIXME(jons): something is still weird here with bsubsmns...
      EXPECT_TRUE(IsCloseRelAbs(
          expected_wout.bsubsmns(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsubsmns(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(
          expected_wout.bsupumnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsupumnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
      EXPECT_TRUE(IsCloseRelAbs(
          expected_wout.bsupvmnc(jH * test_wout.mnmax_nyq + mn_nyq),
          test_wout.bsupvmnc(jH * test_wout.mnmax_nyq + mn_nyq), tolerance))
          << "jH = " << jH << " mn_nyq = " << mn_nyq;
    }  // mn_nyq
  }    // jH

  // also test the wrong extrapolation of bsubsmns
  // beyond the magnetic axis for backward compatibility
  for (int mn_nyq = 0; mn_nyq < test_wout.mnmax_nyq; ++mn_nyq) {
    EXPECT_TRUE(IsCloseRelAbs(
        expected_wout.bsubsmns(0 * test_wout.mnmax_nyq + mn_nyq),
        test_wout.bsubsmns(0 * test_wout.mnmax_nyq + mn_nyq), tolerance));
  }  // mn_nyq

  // -------------------
  // non-stellarator-symmetric Fourier coefficients

  if (test_wout.lasym) {
    for (int n = 0; n <= test_wout.ntor; ++n) {
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.raxis_s[n], test_wout.raxis_s[n],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(expected_wout.zaxis_c[n], test_wout.zaxis_c[n],
                                tolerance));
    }  // n
  }
}

INSTANTIATE_TEST_SUITE_P(
    TestHotRestart, HotRestartIntegration,
    // using the same levels of tolerance as CheckWOutFileContents in
    // output_quantities_test.cc
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-4},
           DataSource{.identifier = "cma", .tolerance = 1.0e-4},
           DataSource{.identifier = "cth_like_fixed_bdy",
                      .tolerance = 1.0e-4}));

TEST(HotRestartIntegration, FreeBoundary) {
  // LOAD INDATA FILE
  const std::string filename = "vmecpp/test_data/cth_like_free_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  VmecINDATA& indata = maybe_indata.value();

  // LOAD COILS FILE
  const std::string coils_filename = "vmecpp/test_data/coils.cth_like";
  const auto maybe_magnetic_configuration =
      ImportMagneticConfigurationFromCoilsFile(coils_filename);
  ASSERT_TRUE(maybe_magnetic_configuration.ok());
  const auto& magnetic_configuration = *maybe_magnetic_configuration;

  // load makegrid params
  const auto maybe_makegrid_params = ImportMakegridParametersFromFile(
      "vmecpp/test_data/makegrid_parameters_cth_like.json");
  ASSERT_TRUE(maybe_makegrid_params.ok());
  const auto& makegrid_params = *maybe_makegrid_params;

  // compute magnetic field response tables
  const auto maybe_magnetic_response_table =
      makegrid::ComputeMagneticFieldResponseTable(makegrid_params,
                                                  magnetic_configuration);
  ASSERT_TRUE(maybe_magnetic_response_table.ok());
  const auto& magnetic_response_table = *maybe_magnetic_response_table;

  // RUNS
  const auto original_output = vmecpp::run(indata, magnetic_response_table);
  ASSERT_TRUE(original_output.ok());

  // MODIFY COILS
  const double radial_coil_displacement = 1e-3;
  makegrid::MagneticConfiguration displaced_magnetic_configuration =
      magnetic_configuration;
  const auto s = magnetics::MoveRadially(radial_coil_displacement,
                                         displaced_magnetic_configuration);
  CHECK_OK(s);

  // RECOMPUTE MGRID FILE
  const auto maybe_displaced_magnetic_response_table =
      makegrid::ComputeMagneticFieldResponseTable(
          makegrid_params, displaced_magnetic_configuration);
  ASSERT_TRUE(maybe_displaced_magnetic_response_table.ok());
  const auto& displaced_magnetic_response_table =
      *maybe_displaced_magnetic_response_table;

  // FREE-BOUNDARY HOT RESTART RUN (SAME INDATA, RECOMPUTED MGRID FILE)
  const auto displaced_hotrestarted_output =
      vmecpp::run(indata, displaced_magnetic_response_table, *original_output);
  ASSERT_TRUE(displaced_hotrestarted_output.ok());

  // FROM SCRATCH RUN (SAME INDATA, RECOMPUTED MGRID FILE)
  const auto displaced_fromscratch_output =
      vmecpp::run(indata, displaced_magnetic_response_table);
  ASSERT_TRUE(displaced_fromscratch_output.ok());

  // COMPARE RUN FROM SCRATCH AND HOT-RESTARTED RUN
  // FIXME(jons): How realistic are these tolerances?
  const double tolerance = 0.1;
  const bool check_equal_maximum_iterations = false;
  vmecpp::CompareWOut(displaced_hotrestarted_output->wout,
                      displaced_fromscratch_output->wout, tolerance,
                      check_equal_maximum_iterations);

  // TODO(eguiraud): we'd like to use these to test that the displaced output
  // _is_ different, but the current CompareWOut implementation simply aborts in
  // that case. vmecpp::CompareWOut(displaced_fromscratch_output->wout,
  //                     original_output->wout, tolerance);

  // vmecpp::CompareWOut(displaced_hotrestarted_output->wout,
  //                     original_output->wout, tolerance);
}

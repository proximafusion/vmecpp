#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"

#include <fstream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/boundaries/boundaries.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

namespace {
using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;
}  // namespace

// used to specify case-specific tolerances
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
};

class InterpolateFromInitialGuessTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(InterpolateFromInitialGuessTest, CheckInterpolateFromInitialGuess) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const FlowControl& fc = vmec.fc_;
  const Sizes& s = vmec.s_;

  bool reached_checkpoint =
      vmec.run(VmecCheckpoint::SETUP_INITIAL_STATE, 1).value();
  ASSERT_TRUE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp/test_data/%s/profil3d/profil3d_%05d_000001_%02d.%s.json",
      data_source_.identifier, fc.ns, vmec.get_num_eqsolve_retries(),
      data_source_.identifier);
  std::ifstream ifs_profil3d(filename);
  ASSERT_TRUE(ifs_profil3d.is_open());
  json profil3d = json::parse(ifs_profil3d);

  // perform testing outside of multi-threaded region to avoid overlapping error
  // messages
  for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
    const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

    int nsMinF1 = radial_partitioning.nsMinF1;
    int nsMaxF1 = radial_partitioning.nsMaxF1;

    for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
      for (int m = 0; m < s.mpol; ++m) {
        for (int n = 0; n < s.ntor + 1; ++n) {
          int idx_fc = ((jF - nsMinF1) * s.mpol + m) * (s.ntor + 1) + n;

          int basis_index = 0;

          EXPECT_TRUE(IsCloseRelAbs(
              profil3d["rmn"][basis_index][jF][n][m],
              vmec.decomposed_x_[thread_id]->rmncc[idx_fc], tolerance));
          EXPECT_TRUE(IsCloseRelAbs(
              profil3d["zmn"][basis_index][jF][n][m],
              vmec.decomposed_x_[thread_id]->zmnsc[idx_fc], tolerance));
          basis_index++;
          if (s.lthreed) {
            EXPECT_TRUE(IsCloseRelAbs(
                profil3d["rmn"][basis_index][jF][n][m],
                vmec.decomposed_x_[thread_id]->rmnss[idx_fc], tolerance));
            EXPECT_TRUE(IsCloseRelAbs(
                profil3d["zmn"][basis_index][jF][n][m],
                vmec.decomposed_x_[thread_id]->zmncs[idx_fc], tolerance));
            basis_index++;
          }
          if (s.lasym) {
            EXPECT_TRUE(IsCloseRelAbs(
                profil3d["rmn"][basis_index][jF][n][m],
                vmec.decomposed_x_[thread_id]->rmnsc[idx_fc], tolerance));
            EXPECT_TRUE(IsCloseRelAbs(
                profil3d["zmn"][basis_index][jF][n][m],
                vmec.decomposed_x_[thread_id]->zmncc[idx_fc], tolerance));
            basis_index++;
            if (s.lthreed) {
              EXPECT_TRUE(IsCloseRelAbs(
                  profil3d["rmn"][basis_index][jF][n][m],
                  vmec.decomposed_x_[thread_id]->rmncs[idx_fc], tolerance));
              EXPECT_TRUE(IsCloseRelAbs(
                  profil3d["zmn"][basis_index][jF][n][m],
                  vmec.decomposed_x_[thread_id]->zmnss[idx_fc], tolerance));
              basis_index++;
            }
          }

          ASSERT_EQ(basis_index, s.num_basis);
        }  // m
      }    // n
    }      // jF
  }        // thread_id
}  // CheckInterpolateFromInitialGuess

INSTANTIATE_TEST_SUITE_P(
    TestInterpolateFromInitialGuess, InterpolateFromInitialGuessTest,
    Values(DataSource{.identifier = "solovev", .tolerance = DBL_EPSILON},
           DataSource{.identifier = "solovev_no_axis",
                      .tolerance = DBL_EPSILON},
           DataSource{.identifier = "cth_like_fixed_bdy",
                      .tolerance = DBL_EPSILON},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = DBL_EPSILON},
           DataSource{.identifier = "cma", .tolerance = DBL_EPSILON},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = DBL_EPSILON}));

TEST(HotRestart, InitializeFromExistingState) {
  // Test that FourierGeometry can be initialized from rmnc, zmns, lmns_full and
  // a Boundary that sets the LCFS As a reference we use the internal flux
  // surface geometry (including lambda) from a wout file produced by VMEC++.
  // The geometry of the LCFS is taken from the corresponding indata input file.

  // SETUP
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  const int num_threads = 1;
  const int thread_id = 0;

  Sizes s(*indata);
  FourierBasisFastPoloidal fb(&s);

  const bool lfreeb = false;
  const double delt = 1.0;
  const int num_grids = 1;
  FlowControl fc(lfreeb, delt, num_grids + 1, num_threads);
  fc.ns = indata->ns_array.back();
  fc.deltaS = 1.0 / (fc.ns - 1.0);

  RadialPartitioning rp;
  rp.adjustRadialPartitioning(num_threads, thread_id, fc.ns, indata->lfreeb,
                              /*printout=*/false);

  Boundaries b(&s, &fb, -1);
  b.setupFromIndata(*indata);

  HandoverStorage h(&s);

  const int signOfJacobian = -1;
  const double pDamp = 0.05;
  RadialProfiles p(&rp, &h, &(*indata), &fc, signOfJacobian, pDamp);

  // update profile parameterizations based on p****_type strings
  p.setupInputProfiles();

  VmecConstants constants;
  constants.reset();

  p.evalRadialProfiles(fc.haveToFlipTheta, thread_id, constants);

  // Now that all contributions to lamscale have been accumulated in
  // VmecConstants::rmsPhiP, can update lamscale.
  constants.lamscale = sqrt(constants.rmsPhiP * fc.deltaS);

  FourierGeometry g(&s, &rp, fc.ns);

  const std::string wout_filename =
      "vmecpp/test_data/cth_like_fixed_bdy.out.h5";
  const auto maybe_out = OutputQuantities::Load(wout_filename);
  ASSERT_TRUE(maybe_out.ok());
  const auto& wout = maybe_out->wout;
  const double tol = 1e-15;

  // THIS IS THE CALL UNDER TEST
  g.InitFromState(fb, wout.rmnc, wout.zmns, wout.lmns_full, p, constants, &b);

  // COMPARISON WITH REFERENCES
  for (int jF = 0; jF < fc.ns; ++jF) {
    const auto& rmnc_row = wout.rmnc.row(jF);
    const auto& ref_rmnc =
        std::vector<double>(rmnc_row.data(), rmnc_row.data() + rmnc_row.size());
    std::vector<double> ref_rmncc(s.mpol * (s.ntor + 1));
    std::vector<double> ref_rmnss(s.mpol * (s.ntor + 1));
    fb.cos_to_cc_ss(ref_rmnc, ref_rmncc, ref_rmnss, s.ntor, s.mpol);

    const auto& zmns_row = wout.zmns.row(jF);
    const auto& ref_zmns =
        std::vector<double>(zmns_row.data(), zmns_row.data() + zmns_row.size());
    std::vector<double> ref_zmnsc(s.mpol * (s.ntor + 1));
    std::vector<double> ref_zmncs(s.mpol * (s.ntor + 1));
    fb.sin_to_sc_cs(ref_zmns, ref_zmnsc, ref_zmncs, s.ntor, s.mpol);

    // The m=1 constraint needs to be activated among reference data
    // before doing the comparison.
    const int m1 = 1;
    for (int n = 0; n < s.ntor + 1; ++n) {
      const int idx_mn = m1 * (s.ntor + 1) + n;
      if (s.lthreed) {
        const double old_rss = ref_rmnss[idx_mn];
        ref_rmnss[idx_mn] = (old_rss + ref_zmncs[idx_mn]) * 0.5;
        ref_zmncs[idx_mn] = (old_rss - ref_zmncs[idx_mn]) * 0.5;
      }
      // TODO(jons): activate this when lasym=true branch is implemented
      // if (s.lasym) {
      //   const double old_rsc = ref_rmnsc[idx_mn];
      //   ref_rmnsc[idx_mn] = (old_rsc + ref_zmncc[idx_mn]) * 0.5;
      //   ref_zmncc[idx_mn] = (old_rsc - ref_zmncc[idx_mn]) * 0.5;
      // }
    }

    const auto& lmns_row = wout.lmns_full.row(jF);
    const auto& ref_lmns =
        std::vector<double>(lmns_row.data(), lmns_row.data() + lmns_row.size());
    std::vector<double> ref_lmnsc(s.mpol * (s.ntor + 1));
    std::vector<double> ref_lmncs(s.mpol * (s.ntor + 1));
    fb.sin_to_sc_cs(ref_lmns, ref_lmnsc, ref_lmncs, s.ntor, s.mpol);

    // lambda is normally zero at the axis, but the wout file contents
    // contain extrapolated data at the axis in lambda,
    // which we need to remove for this comparison.
    if (jF == 0) {
      absl::c_fill(ref_lmnsc, 0.0);
      absl::c_fill(ref_lmncs, 0.0);
    }

    for (int m = 0; m < s.mpol; ++m) {
      for (int n = 0; n < s.ntor + 1; ++n) {
        const int idx_mn = m * (s.ntor + 1) + n;
        const int idx_fc = (jF * s.mpol + m) * (s.ntor + 1) + n;

        if (jF < fc.ns - 1) {
          // last surface is tested separately
          EXPECT_TRUE(IsCloseRelAbs(ref_rmncc[idx_mn], g.rmncc[idx_fc], tol))
              << "rmncc @ j = " << jF << " m = " << m << " n = " << n;
          EXPECT_TRUE(IsCloseRelAbs(ref_rmnss[idx_mn], g.rmnss[idx_fc], tol))
              << "rmnss @ j = " << jF << " m = " << m << " n = " << n;

          EXPECT_TRUE(IsCloseRelAbs(ref_zmnsc[idx_mn], g.zmnsc[idx_fc], tol))
              << "zmnsc @ j = " << jF << " m = " << m << " n = " << n;
          EXPECT_TRUE(IsCloseRelAbs(ref_zmncs[idx_mn], g.zmncs[idx_fc], tol))
              << "zmncs @ j = " << jF << " m = " << m << " n = " << n;
        }

        EXPECT_TRUE(IsCloseRelAbs(ref_lmnsc[idx_mn], g.lmnsc[idx_fc], tol))
            << "lmnsc @ j = " << jF << " m = " << m << " n = " << n;
        EXPECT_TRUE(IsCloseRelAbs(ref_lmncs[idx_mn], g.lmncs[idx_fc], tol))
            << "lmncs @ j = " << jF << " m = " << m << " n = " << n;
      }
    }
  }

  // now the last surface
  const int jF = fc.ns - 1;
  for (int m = 0; m < s.mpol; ++m) {
    for (int n = 0; n < s.ntor + 1; ++n) {
      const int idx_mn = m * (s.ntor + 1) + n;
      const int idx_fc = (jF * s.mpol + m) * (s.ntor + 1) + n;

      // basis_norm is normally applied within
      // FourierGeometry::interpFromBoundaryAndAxis(), but
      // FourierGeometry::InitFromState() also does it, so we need to mimic this
      // for the Boundaries members.
      const double basis_norm = 1.0 / (fb.mscale[m] * fb.nscale[n]);

      EXPECT_TRUE(
          IsCloseRelAbs(b.rbcc[idx_mn] * basis_norm, g.rmncc[idx_fc], tol))
          << "rbcc @ m = " << m << " n = " << n;
      EXPECT_TRUE(
          IsCloseRelAbs(b.rbss[idx_mn] * basis_norm, g.rmnss[idx_fc], tol))
          << "rbss @ m = " << m << " n = " << n;

      EXPECT_TRUE(
          IsCloseRelAbs(b.zbsc[idx_mn] * basis_norm, g.zmnsc[idx_fc], tol))
          << "zbsc @ m = " << m << " n = " << n;
      EXPECT_TRUE(
          IsCloseRelAbs(b.zbcs[idx_mn] * basis_norm, g.zmncs[idx_fc], tol))
          << "zbcs @ m = " << m << " n = " << n;
    }
  }
}

}  // namespace vmecpp

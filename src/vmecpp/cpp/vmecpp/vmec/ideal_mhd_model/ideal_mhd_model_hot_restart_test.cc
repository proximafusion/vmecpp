#include <fstream>  // std::ifstream
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

using nlohmann::json;
using ::testing::DoubleNear;
using ::testing::ElementsAreArray;
using ::testing::Pointwise;
using ::testing::TestWithParam;
using ::testing::Values;

using file_io::ReadFile;
using testing::IsCloseRelAbs;
using vmecpp::Vmec;

namespace vmecpp {

// used to specify case-specific tolerances
// and which iterations to test
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::vector<int> iter2_to_test = {1, 2};
};

class FourierGeometryToStartWithFromWOutTest
    : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(FourierGeometryToStartWithFromWOutTest,
       CheckFourierGeometryToStartWithFromWOut) {
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
  const auto checkpoint = VmecCheckpoint::FOURIER_GEOMETRY_TO_START_WITH;
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

INSTANTIATE_TEST_SUITE_P(HotRestart, FourierGeometryToStartWithFromWOutTest,
                         Values(DataSource{.identifier = "solovev",
                                           .tolerance = 1.0e-15},
                                DataSource{.identifier = "cth_like_fixed_bdy",
                                           .tolerance = 1.0e-15}));

class FourierGeometryToStartWithFromDebugOutTest
    : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(FourierGeometryToStartWithFromDebugOutTest,
       CheckFourierGeometryToStartWithFromDebugOut) {
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
  const auto checkpoint = VmecCheckpoint::FOURIER_GEOMETRY_TO_START_WITH;
  const absl::StatusOr<bool> checkpoint_reached = vmec.run(
      checkpoint, /*iterations_before_checkpointing=*/0,
      /*maximum_multi_grid_step=*/500, /*initial_state=*/output_quantities);
  ASSERT_TRUE(checkpoint_reached.ok());
  ASSERT_TRUE(*checkpoint_reached);

  const int niter = output_quantities.wout.maximum_iterations - 1;
  const std::string ref_filename = absl::StrFormat(
      "vmecpp/test_data/%s/totzsp_input/"
      "totzsp_input_%05d_%06d_01.%s.json",
      data_source_.identifier, indata->ns_array[0], niter,
      data_source_.identifier);

  std::ifstream ifs_totzsp_input(ref_filename);
  ASSERT_TRUE(ifs_totzsp_input.is_open())
      << "failed to open reference file: " << ref_filename;
  json totzsp_input = json::parse(ifs_totzsp_input);

  const Sizes& s = vmec.s_;

  // perform testing outside of multi-threaded region to avoid overlapping
  // error messages
  for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
    const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];
    const FourierGeometry& physical_x = *(vmec.physical_x_[thread_id]);

    const int nsMinF = radial_partitioning.nsMinF;

    for (int rzl = 0; rzl < 3; ++rzl) {
      for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
        // time stepping is only done on those Fourier coefficients
        // for which the forces were computed in this thread!
        for (int jF = nsMinF; jF < radial_partitioning.nsMaxFIncludingLcfs;
             ++jF) {
          for (int n = 0; n < s.ntor + 1; ++n) {
            for (int m = 0; m < s.mpol; ++m) {
              EXPECT_TRUE(IsCloseRelAbs(
                  totzsp_input["gc"][rzl][idx_basis][jF][n][m],
                  physical_x.GetXcElement(rzl, idx_basis, jF, n, m), tolerance))
                  << rzl << " " << idx_basis << " " << jF << " " << n << " "
                  << m;
            }  // m
          }    // n
        }      // jF
      }        // idx_basis
    }          // rzl
  }            // thread_id
}

INSTANTIATE_TEST_SUITE_P(HotRestart, FourierGeometryToStartWithFromDebugOutTest,
                         Values(DataSource{.identifier = "solovev",
                                           .tolerance = 1.0e-11},
                                DataSource{.identifier = "cth_like_fixed_bdy",
                                           .tolerance = 1.0e-13}));

class InverseFourierTransformGeometryTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(InverseFourierTransformGeometryTest,
       CheckInverseFourierTransformGeometry) {
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
  const auto checkpoint = VmecCheckpoint::INV_DFT_GEOMETRY;
  const absl::StatusOr<bool> checkpoint_reached = vmec.run(
      checkpoint, /*iterations_before_checkpointing=*/0,
      /*maximum_multi_grid_step=*/500, /*initial_state=*/output_quantities);
  ASSERT_TRUE(checkpoint_reached.ok());
  ASSERT_TRUE(*checkpoint_reached);

  // COMPARISON WITH REFERENCES
  const int niter = output_quantities.wout.maximum_iterations - 1;
  const std::string ref_filename = absl::StrFormat(
      "vmecpp/test_data/%s/funct3d_geometry/"
      "funct3d_geometry_%05d_%06d_01.%s.json",
      data_source_.identifier, indata->ns_array[0], niter,
      data_source_.identifier);

  std::ifstream ifs_funct3d_geometry(ref_filename);
  ASSERT_TRUE(ifs_funct3d_geometry.is_open())
      << "failed to open reference file: " << ref_filename;
  json funct3d_geometry = json::parse(ifs_funct3d_geometry);

  const Sizes& s = vmec.s_;

  // perform testing outside of multi-threaded region to avoid overlapping
  // error messages
  for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
    const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];
    const RadialProfiles& radial_profiles = *vmec.p_[thread_id];

    const int nsMinF1 = radial_partitioning.nsMinF1;
    const int nsMaxF1 = radial_partitioning.nsMaxF1;

    const int nsMinF = radial_partitioning.nsMinF;

    const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

    for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
      for (int k = 0; k < s.nZeta; ++k) {
        for (int l = 0; l < s.nThetaEff; ++l) {
          int idx_kl = ((jF - nsMinF1) * s.nZeta + k) * s.nThetaEff + l;

          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["r1"][jF][m_evn][k][l],
                                    vmec.m_[thread_id]->r1_e[idx_kl],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["r1"][jF][m_odd][k][l],
                                    vmec.m_[thread_id]->r1_o[idx_kl],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["ru"][jF][m_evn][k][l],
                                    vmec.m_[thread_id]->ru_e[idx_kl],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["ru"][jF][m_odd][k][l],
                                    vmec.m_[thread_id]->ru_o[idx_kl],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["z1"][jF][m_evn][k][l],
                                    vmec.m_[thread_id]->z1_e[idx_kl],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["z1"][jF][m_odd][k][l],
                                    vmec.m_[thread_id]->z1_o[idx_kl],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["zu"][jF][m_evn][k][l],
                                    vmec.m_[thread_id]->zu_e[idx_kl],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["zu"][jF][m_odd][k][l],
                                    vmec.m_[thread_id]->zu_o[idx_kl],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["lu"][jF][m_evn][k][l],
                                    vmec.m_[thread_id]->lu_e[idx_kl],
                                    tolerance));
          EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["lu"][jF][m_odd][k][l],
                                    vmec.m_[thread_id]->lu_o[idx_kl],
                                    tolerance));
          if (s.lthreed) {
            EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["rv"][jF][m_evn][k][l],
                                      vmec.m_[thread_id]->rv_e[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["rv"][jF][m_odd][k][l],
                                      vmec.m_[thread_id]->rv_o[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["zv"][jF][m_evn][k][l],
                                      vmec.m_[thread_id]->zv_e[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["zv"][jF][m_odd][k][l],
                                      vmec.m_[thread_id]->zv_o[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["lv"][jF][m_evn][k][l],
                                      vmec.m_[thread_id]->lv_e[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(funct3d_geometry["lv"][jF][m_odd][k][l],
                                      vmec.m_[thread_id]->lv_o[idx_kl],
                                      tolerance));
          }  // lthreed

          if (nsMinF <= jF && jF < nsMaxFIncludingLcfs) {
            // spectral condensation is local per flux surface --> no need for
            // numFull1 This index is invalid outside of [r->nsMinF, jMaxCon[
            // !
            const int idx_con = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff + l;

            const double sqrtSF =
                radial_profiles.sqrtSF[jF - radial_partitioning.nsMinF1];

            // VMEC++ computes rCon, zCon directly combined from even-m and
            // odd-m contributions. This is different from educational_VMEC
            // (and the other Fortran implementations), where the combination
            // is computed after totzsp(s,a). Hence, need to combine the
            // even-m and odd-m contributions in order to compute the required
            // reference quantities.
            const double expected_rcon_even =
                funct3d_geometry["rcon"][jF][m_evn][k][l];
            const double expected_rcon_odd =
                funct3d_geometry["rcon"][jF][m_odd][k][l];
            const double expected_rcon =
                expected_rcon_even + sqrtSF * expected_rcon_odd;

            const double expected_zcon_even =
                funct3d_geometry["zcon"][jF][m_evn][k][l];
            const double expected_zcon_odd =
                funct3d_geometry["zcon"][jF][m_odd][k][l];
            const double expected_zcon =
                expected_zcon_even + sqrtSF * expected_zcon_odd;

            EXPECT_TRUE(IsCloseRelAbs(
                expected_rcon, vmec.m_[thread_id]->rCon[idx_con], tolerance));
            EXPECT_TRUE(IsCloseRelAbs(
                expected_zcon, vmec.m_[thread_id]->zCon[idx_con], tolerance));
          }  // within ...Con range: forces full-grid (numFull)
        }    // l
      }      // k
    }        // jF
  }          // thread_id
}

INSTANTIATE_TEST_SUITE_P(HotRestart, InverseFourierTransformGeometryTest,
                         Values(DataSource{.identifier = "solovev",
                                           .tolerance = 1.0e-10},
                                DataSource{.identifier = "cth_like_fixed_bdy",
                                           .tolerance = 1.0e-12}));

}  // namespace vmecpp

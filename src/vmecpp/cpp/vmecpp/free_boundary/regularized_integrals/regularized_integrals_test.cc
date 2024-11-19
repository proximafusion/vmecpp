#include "vmecpp/free_boundary/regularized_integrals/regularized_integrals.h"

#include <fstream>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;

// used to specify case-specific tolerances
// and which iterations to test
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::vector<int> iter2_to_test = {1, 2};
};

class TanuTanvTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(TanuTanvTest, CheckTanuTanv) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::VAC1_VACUUM, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_precal/vac1n_precal_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_precal(filename);
    ASSERT_TRUE(ifs_vac1n_precal.is_open())
        << "failed to open reference file: " << filename;
    json vac1n_precal = json::parse(ifs_vac1n_precal);

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);
      const RegularizedIntegrals& ri = n.GetRegularizedIntegrals();

      for (int l = 0; l < s.nThetaEven; ++l) {
        EXPECT_TRUE(
            IsCloseRelAbs(vac1n_precal["tanu_1d"][l], ri.tanu[l], tolerance));
      }  // l

      for (int k = 0; k < s.nZeta; ++k) {
        EXPECT_TRUE(
            IsCloseRelAbs(vac1n_precal["tanv_1d"][k], ri.tanv[k], tolerance));
      }  // k
    }    // thread_id
  }
}  // CheckTanuTanv

INSTANTIATE_TEST_SUITE_P(TestRegularizedIntegrals, TanuTanvTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-14,
                                           .iter2_to_test = {53}}));

class GreenFTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(GreenFTest, CheckGreenF) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::VAC1_GREENF, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_greenf/vac1n_greenf_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_full_update_nestor(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_greenf(filename);
    ASSERT_TRUE(ifs_vac1n_greenf.is_open())
        << "failed to open reference file: " << filename;
    json vac1n_greenf = json::parse(ifs_vac1n_greenf);

    // accumulate contributions from all threads
    // in order to compare against Fortran single-threaded data
    std::vector<double> gstore(s.nZeta * s.nThetaEven);
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);
      const RegularizedIntegrals& ri = n.GetRegularizedIntegrals();
      for (int kl = 0; kl < s.nZeta * s.nThetaEven; ++kl) {
        gstore[kl] += ri.gstore[kl];
      }  // kl
    }    // thread_id

    for (int kl = 0; kl < s.nZeta * s.nThetaEven; ++kl) {
      const int l = kl / s.nZeta;
      const int k = kl % s.nZeta;

      // bexni contains a factor of (2 pi)^2, which is not present in VMEC++
      // gstore get a factor of 2 pi / nfp in VMEC++, which is not present in
      // Fortran VMEC/Nestor.
      const double factor_to_match_fortran =
          4.0 * M_PI * M_PI / (2.0 * M_PI / s.nfp);

      EXPECT_TRUE(IsCloseRelAbs(vac1n_greenf["gstore"][k][l],
                                factor_to_match_fortran * gstore[kl],
                                tolerance));
    }

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);
      const TangentialPartitioning& tp = *vmec.tp_[thread_id];
      const RegularizedIntegrals& ri = n.GetRegularizedIntegrals();

      for (int klp = tp.ztMin; klp < tp.ztMax; ++klp) {
        const int lp = klp / s.nZeta;
        const int kp = klp % s.nZeta;

        const int klpRel = klp - tp.ztMin;

        for (int kl = 0; kl < s.nZeta * s.nThetaEven; ++kl) {
          const int l = kl / s.nZeta;
          const int k = kl % s.nZeta;

          int ip = klpRel * s.nThetaEven * s.nZeta + kl;

          // greenp get a factor of 2 pi / nfp in VMEC++, which is not present
          // in Fortran VMEC/Nestor.
          const double factor_to_match_fortran = 1.0 / (2.0 * M_PI / s.nfp);

          EXPECT_TRUE(IsCloseRelAbs(vac1n_greenf["greenp"][k][l][kp][lp],
                                    factor_to_match_fortran * ri.greenp[ip],
                                    tolerance));
        }  // kl
      }    // klp
    }      // thread_id
  }
}  // CheckGreenF

INSTANTIATE_TEST_SUITE_P(TestRegularizedIntegrals, GreenFTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 5.0e-10,
                                           .iter2_to_test = {53, 54}}));

}  // namespace vmecpp

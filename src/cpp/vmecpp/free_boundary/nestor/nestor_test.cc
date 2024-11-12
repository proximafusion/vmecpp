#include "vmecpp/free_boundary/nestor/nestor.h"

#include <fstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

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

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

// used to specify case-specific tolerances
// and which iterations to test
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::vector<int> iter2_to_test = {1, 2};
};

class InputsToNestorCallTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(InputsToNestorCallTest, CheckInputsToNestorCall) {
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
    const FourierBasisFastToroidal fourier_basis(&s);
    const FlowControl& fc = vmec.fc_;
    const HandoverStorage& h = vmec.h_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::VAC1_VACUUM, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_vacuum/vac1n_vacuum_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_vacuum(filename);
    ASSERT_TRUE(ifs_vac1n_vacuum.is_open());
    json vac1n_vacuum = json::parse(ifs_vac1n_vacuum);

    // The following relies on the parameters passed in the call to
    // Nestor::update in IdealMHDModel.

    // Convert rmnc, ... from Fortran VMEC/Nestor into the rCC, ...-style arrays
    // convert into 2D Fourier coefficient arrays
    std::vector<double> rCC(s.mnsize);
    std::vector<double> rSS(s.mnsize);
    std::vector<double> rSC;
    std::vector<double> rCS;
    fourier_basis.cos_to_cc_ss(vac1n_vacuum["rmnc"], rCC, rSS, s.ntor, s.mpol);
    if (s.lasym) {
      rSC.resize(s.mnsize);
      rCS.resize(s.mnsize);
      fourier_basis.sin_to_sc_cs(vac1n_vacuum["rmns"], rSC, rCS, s.ntor,
                                 s.mpol);
    }

    std::vector<double> zSC(s.mnsize);
    std::vector<double> zCS(s.mnsize);
    std::vector<double> zCC;
    std::vector<double> zSS;
    fourier_basis.sin_to_sc_cs(vac1n_vacuum["zmns"], zSC, zCS, s.ntor, s.mpol);
    if (s.lasym) {
      zCC.resize(s.mnsize);
      zSS.resize(s.mnsize);
      fourier_basis.cos_to_cc_ss(vac1n_vacuum["zmnc"], zCC, zSS, s.ntor,
                                 s.mpol);
    }

    for (int mn = 0; mn < s.mnsize; ++mn) {
      EXPECT_TRUE(IsCloseRelAbs(rCC[mn], h.rCC_LCFS[mn], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(rSS[mn], h.rSS_LCFS[mn], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(zSC[mn], h.zSC_LCFS[mn], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(zCS[mn], h.zCS_LCFS[mn], tolerance));
    }  // mn
    if (s.lasym) {
      for (int mn = 0; mn < s.mnsize; ++mn) {
        EXPECT_TRUE(IsCloseRelAbs(rSC[mn], h.rSC_LCFS[mn], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(rCS[mn], h.rCS_LCFS[mn], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(zCC[mn], h.zCC_LCFS[mn], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(zSS[mn], h.zSS_LCFS[mn], tolerance));
      }  // mn
    }

    EXPECT_TRUE(IsCloseRelAbs(vac1n_vacuum["plascur"], h.cTor, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(vac1n_vacuum["rbtor"], h.rBtor, tolerance));

    for (int k = 0; k < s.nZeta; ++k) {
      EXPECT_TRUE(IsCloseRelAbs(vac1n_vacuum["raxis_nestor"][k], h.rAxis[k],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(vac1n_vacuum["zaxis_nestor"][k], h.zAxis[k],
                                tolerance));
    }  // k
  }
}  // CheckInputsToNestorCall

INSTANTIATE_TEST_SUITE_P(TestNestor, InputsToNestorCallTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-12,
                                           .iter2_to_test = {53, 54}}));

class BsqVacTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(BsqVacTest, CheckBsqVac) {
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
    const HandoverStorage& h = vmec.h_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::VAC1_BSQVAC, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_bsqvac/vac1n_bsqvac_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_bsqvac(filename);
    ASSERT_TRUE(ifs_vac1n_bsqvac.is_open());
    json vac1n_bsqvac = json::parse(ifs_vac1n_bsqvac);

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);
      const TangentialPartitioning& tp = *vmec.tp_[thread_id];

      for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
        const int l = kl / s.nZeta;
        const int k = kl % s.nZeta;

        EXPECT_TRUE(IsCloseRelAbs(vac1n_bsqvac["potu"][k][l],
                                  n.potU[kl - tp.ztMin], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bsqvac["potv"][k][l],
                                  n.potV[kl - tp.ztMin], tolerance));

        EXPECT_TRUE(IsCloseRelAbs(vac1n_bsqvac["bsubu"][k][l],
                                  n.bSubU[kl - tp.ztMin], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_bsqvac["bsubv"][k][l],
                                  n.bSubV[kl - tp.ztMin], tolerance));
      }  // kl
    }    // thread_id

    // test bsqvac already in full-surface target array
    for (int kl = 0; kl < s.nZnT; ++kl) {
      const int l = kl / s.nZeta;
      const int k = kl % s.nZeta;

      EXPECT_TRUE(IsCloseRelAbs(vac1n_bsqvac["bsqvac"][k][l],
                                h.vacuum_magnetic_pressure[kl], tolerance));

      EXPECT_TRUE(IsCloseRelAbs(vac1n_bsqvac["brv"][k][l], h.vacuum_b_r[kl],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(vac1n_bsqvac["bphiv"][k][l], h.vacuum_b_phi[kl],
                                tolerance));
      EXPECT_TRUE(IsCloseRelAbs(vac1n_bsqvac["bzv"][k][l], h.vacuum_b_z[kl],
                                tolerance));
    }  // kl
  }
}  // CheckBsqVac

INSTANTIATE_TEST_SUITE_P(TestNestor, BsqVacTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-10,
                                           .iter2_to_test = {53, 54}}));

}  // namespace vmecpp

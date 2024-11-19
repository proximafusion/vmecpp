#include "vmecpp/free_boundary/laplace_solver/laplace_solver.h"

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/status/status.h"
#include "absl/status/statusor.h"
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

class FourPTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(FourPTest, CheckFourP) {
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
        vmec.run(VmecCheckpoint::VAC1_FOURP, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_fourp/vac1n_fourp_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_full_update_nestor(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_fourp(filename);
    ASSERT_TRUE(ifs_vac1n_fourp.is_open());
    json vac1n_fourp = json::parse(ifs_vac1n_fourp);

    const int nf = s.ntor;
    const int mf = s.mpol + 1;

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);

      const TangentialPartitioning& tp = *vmec.tp_[thread_id];
      const int numLocal = tp.ztMax - tp.ztMin;

      const SingularIntegrals& si = n.GetSingularIntegrals();
      const LaplaceSolver& ls = n.GetLaplaceSolver();

      if (vmec.m_[0]->get_ivacskip() == 0) {
        // grpmn can be tested here already, as there is no reduction over
        // threads involved
        for (int n = 0; n < nf + 1; ++n) {
          for (int m = 0; m < mf + 1; ++m) {
            const int idx_m_posn = (nf + n) * (mf + 1) + m;
            const int idx_m_negn = (nf - n) * (mf + 1) + m;

            for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
              const int l = kl / s.nZeta;
              const int k = kl % s.nZeta;

              const int klRel = kl - tp.ztMin;

              // cmns in Fortran has alp (= 2 pi / nfp) in it; VMEC++ does not
              const double scale_to_match_fortran_singular = 2.0 * M_PI / s.nfp;

              // These are tested already in singular_integrals_test:AnalytTest
              const double grpmn_sin_singular_posn =
                  scale_to_match_fortran_singular *
                  si.grpmn_sin[idx_m_posn * numLocal + klRel];
              const double grpmn_sin_singular_negn =
                  scale_to_match_fortran_singular *
                  si.grpmn_sin[idx_m_negn * numLocal + klRel];

              // cosn, sinn have 1/nfp in them; VMEC++ does not have that
              // --> [(2 pi)/nfp] / [1/nfp] == 2 pi
              const double scale_to_match_fortran_regular = 2.0 * M_PI;

              // These are the ones actually under test here; computed in
              // LaplaceSolver::transformGreensDerivative
              const double grpmn_sin_regular_posn =
                  scale_to_match_fortran_regular *
                  ls.grpmn_sin[idx_m_posn * numLocal + klRel];
              const double grpmn_sin_regular_negn =
                  scale_to_match_fortran_regular *
                  ls.grpmn_sin[idx_m_negn * numLocal + klRel];

              // compute stand-alone reference values for
              // transformGreensDerivative output
              const double grpmn_sin_reference_posn =
                  static_cast<double>(vac1n_fourp["grpmn"][m][nf + n][k][l]) -
                  grpmn_sin_singular_posn;
              const double grpmn_sin_reference_negn =
                  static_cast<double>(vac1n_fourp["grpmn"][m][nf - n][k][l]) -
                  grpmn_sin_singular_negn;

              // TODO(jons): for lasym, need cos-part of grpmn from
              // educational_VMEC
              EXPECT_TRUE(IsCloseRelAbs(grpmn_sin_reference_posn,
                                        grpmn_sin_regular_posn, tolerance));
              EXPECT_TRUE(IsCloseRelAbs(grpmn_sin_reference_negn,
                                        grpmn_sin_regular_negn, tolerance));
            }  // kl
          }    // m
        }      // n
      }        // fullUpdate
    }          // thread_id
  }
}  // CheckFourP

INSTANTIATE_TEST_SUITE_P(TestLaplaceSolver, FourPTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-9,
                                           .iter2_to_test = {53, 54}}));

class FourISymmTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(FourISymmTest, CheckFourISymm) {
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
        vmec.run(VmecCheckpoint::VAC1_FOURI_SYMM, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_fouri/vac1n_fouri_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_full_update_nestor(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_fouri(filename);
    ASSERT_TRUE(ifs_vac1n_fouri.is_open());
    json vac1n_fouri = json::parse(ifs_vac1n_fouri);

    // accumulate contributions from all threads
    // in order to compare against Fortran single-threaded data
    std::vector<double> gstore_symm(s.nThetaReduced * s.nZeta);

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);

      const LaplaceSolver& ls = n.GetLaplaceSolver();

      if (vmec.m_[0]->get_ivacskip() == 0) {
        for (int kl = 0; kl < s.nThetaReduced * s.nZeta; ++kl) {
          gstore_symm[kl] += ls.gstore_symm[kl];
        }  // kl
      }    // fullUpdate
    }      // thread_id

    if (vmec.m_[0]->get_ivacskip() == 0) {
      for (int l = 0; l < s.nThetaReduced; ++l) {
        for (int k = 0; k < s.nZeta; ++k) {
          const int kl = l * s.nZeta + k;

          // gstore gets (2 pi) / nfp in RegularizedIntegrals in VMEC++; not in
          // Fortran VMEC/Nestor. source gets 1/nfp in fouri; not in VMEC++
          // --> [(2 pi)/nfp] / [1/nfp] == 2 pi
          const double factor_to_match_fortran = 2.0 * M_PI;

          EXPECT_TRUE(IsCloseRelAbs(vac1n_fouri["source"][k][l],
                                    factor_to_match_fortran * gstore_symm[kl],
                                    tolerance));
        }  // k
      }    // l
    }      // fullUpdate
  }
}  // CheckFourISymm

INSTANTIATE_TEST_SUITE_P(TestLaplaceSolver, FourISymmTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-9,
                                           .iter2_to_test = {53, 54}}));

class FourIAccumulateGrpmnTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(FourIAccumulateGrpmnTest, CheckFourIAccumulateGrpmn) {
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
        vmec.run(VmecCheckpoint::VAC1_FOURI_KV_DFT, number_of_iterations)
            .value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_fourp/vac1n_fourp_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_full_update_nestor(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_fourp(filename);
    ASSERT_TRUE(ifs_vac1n_fourp.is_open());
    json vac1n_fourp = json::parse(ifs_vac1n_fourp);

    const int nf = s.ntor;
    const int mf = s.mpol + 1;

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);

      const TangentialPartitioning& tp = *vmec.tp_[thread_id];
      const int numLocal = tp.ztMax - tp.ztMin;

      const LaplaceSolver& ls = n.GetLaplaceSolver();

      if (vmec.m_[0]->get_ivacskip() == 0) {
        // grpmn can be tested here already, as there is no reduction over
        // threads involved
        for (int n = 0; n < nf + 1; ++n) {
          for (int m = 0; m < mf + 1; ++m) {
            const int idx_m_posn = (nf + n) * (mf + 1) + m;
            const int idx_m_negn = (nf - n) * (mf + 1) + m;

            for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
              const int l = kl / s.nZeta;
              const int k = kl % s.nZeta;

              const int klRel = kl - tp.ztMin;

              // cosn, sinn have 1/nfp in them; VMEC++ does not have that
              // --> [(2 pi)/nfp] / [1/nfp] == 2 pi
              const double scale_to_match_fortran_regular = 2.0 * M_PI;

              // These are the ones actually under test here; computed in
              // LaplaceSolver::transformGreensDerivative
              const double grpmn_sin_regular_posn =
                  scale_to_match_fortran_regular *
                  ls.grpmn_sin[idx_m_posn * numLocal + klRel];
              const double grpmn_sin_regular_negn =
                  scale_to_match_fortran_regular *
                  ls.grpmn_sin[idx_m_negn * numLocal + klRel];

              // TODO(jons): for lasym, need cos-part of grpmn from
              // educational_VMEC
              EXPECT_TRUE(IsCloseRelAbs(vac1n_fourp["grpmn"][m][nf + n][k][l],
                                        grpmn_sin_regular_posn, tolerance));
              EXPECT_TRUE(IsCloseRelAbs(vac1n_fourp["grpmn"][m][nf - n][k][l],
                                        grpmn_sin_regular_negn, tolerance));
            }  // kl
          }    // m
        }      // n
      }        // fullUpdate
    }          // thread_id
  }
}  // CheckFourIAccumulateGrpmn

INSTANTIATE_TEST_SUITE_P(TestLaplaceSolver, FourIAccumulateGrpmnTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-9,
                                           .iter2_to_test = {53, 54}}));

class FourIKvDftTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(FourIKvDftTest, CheckFourIKvDft) {
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
        vmec.run(VmecCheckpoint::VAC1_FOURI_KV_DFT, number_of_iterations)
            .value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_fouri/vac1n_fouri_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_full_update_nestor(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_fouri(filename);
    ASSERT_TRUE(ifs_vac1n_fouri.is_open());
    json vac1n_fouri = json::parse(ifs_vac1n_fouri);

    const int mf = s.mpol + 1;
    const int nf = s.ntor;
    const int mnpd = (mf + 1) * (2 * nf + 1);

    // Accumulate contributions from all threads
    // to complete the toroidal Fourier integrals.
    std::vector<double> bcos_full((2 * nf + 1) * s.nThetaReduced);
    std::vector<double> bsin_full((2 * nf + 1) * s.nThetaReduced);
    const int size_a_temp = mnpd * (2 * nf + 1) * s.nThetaEff;
    std::vector<double> actemp_full(size_a_temp, 0.0);
    std::vector<double> astemp_full(size_a_temp);
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);

      const LaplaceSolver& ls = n.GetLaplaceSolver();

      if (vmec.m_[0]->get_ivacskip() == 0) {
        for (int all_n = 0; all_n < 2 * nf + 1; ++all_n) {
          for (int l = 0; l < s.nThetaReduced; ++l) {
            const int idx_nl = all_n * s.nThetaReduced + l;
            bcos_full[idx_nl] += ls.bcos[idx_nl];
            bsin_full[idx_nl] += ls.bsin[idx_nl];
          }  // l
        }    // n

        for (int mn = 0; mn < mnpd; ++mn) {
          for (int n = 0; n < nf + 1; ++n) {
            for (int l = 0; l < s.nThetaEff; ++l) {
              const int idx_a_posn =
                  (mn * (2 * nf + 1) + (nf + n)) * s.nThetaEff + l;
              actemp_full[idx_a_posn] += ls.actemp[idx_a_posn];
              astemp_full[idx_a_posn] += ls.astemp[idx_a_posn];
              if (n > 0) {
                const int idx_a_negn =
                    (mn * (2 * nf + 1) + (nf - n)) * s.nThetaEff + l;
                actemp_full[idx_a_negn] += ls.actemp[idx_a_negn];
                astemp_full[idx_a_negn] += ls.astemp[idx_a_negn];
              }
            }  // kl'
          }    // n
        }      // mn
      }        // fullUpdate
    }          // thread_id

    if (vmec.m_[0]->get_ivacskip() == 0) {
      for (int n = 0; n < nf + 1; ++n) {
        for (int l = 0; l < s.nThetaReduced; ++l) {
          const int idx_l_posn = (nf + n) * s.nThetaReduced + l;
          const int idx_l_negn = (nf - n) * s.nThetaReduced + l;

          // gstore gets (2 pi) / nfp in RegularizedIntegrals in VMEC++; not in
          // Fortran VMEC/Nestor. source gets 1/nfp in fouri; not in VMEC++
          // --> [(2 pi)/nfp] / [1/nfp] == 2 pi
          const double factor_to_match_fortran = 2.0 * M_PI;

          EXPECT_TRUE(IsCloseRelAbs(
              vac1n_fouri["bcos"][l][nf + n],
              factor_to_match_fortran * bcos_full[idx_l_posn], tolerance));
          EXPECT_TRUE(IsCloseRelAbs(
              vac1n_fouri["bcos"][l][nf - n],
              factor_to_match_fortran * bcos_full[idx_l_negn], tolerance));

          EXPECT_TRUE(IsCloseRelAbs(
              vac1n_fouri["bsin"][l][nf + n],
              factor_to_match_fortran * bsin_full[idx_l_posn], tolerance));
          EXPECT_TRUE(IsCloseRelAbs(
              vac1n_fouri["bsin"][l][nf - n],
              factor_to_match_fortran * bsin_full[idx_l_negn], tolerance));
        }  // l
      }    // n

      for (int mn = 0; mn < mnpd; ++mn) {
        // linear index representing -nf:nf
        const int np = mn / (mf + 1);
        const int mp = mn % (mf + 1);
        for (int n = 0; n < nf + 1; ++n) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            // same as for grpmn_sin after AccumulateFullGrpmn
            const double scale_to_match_fortran = 2.0 * M_PI;

            const int idx_a_posn =
                (mn * (2 * nf + 1) + (nf + n)) * s.nThetaEff + l;
            EXPECT_TRUE(IsCloseRelAbs(
                vac1n_fouri["actemp"][mp][np][nf + n][l],
                scale_to_match_fortran * actemp_full[idx_a_posn], tolerance));
            EXPECT_TRUE(IsCloseRelAbs(
                vac1n_fouri["astemp"][mp][np][nf + n][l],
                scale_to_match_fortran * astemp_full[idx_a_posn], tolerance));
            if (n > 0) {
              const int idx_a_negn =
                  (mn * (2 * nf + 1) + (nf - n)) * s.nThetaEff + l;
              EXPECT_TRUE(IsCloseRelAbs(
                  vac1n_fouri["actemp"][mp][np][nf - n][l],
                  scale_to_match_fortran * actemp_full[idx_a_negn], tolerance));
              EXPECT_TRUE(IsCloseRelAbs(
                  vac1n_fouri["astemp"][mp][np][nf - n][l],
                  scale_to_match_fortran * astemp_full[idx_a_negn], tolerance));
            }
          }  // kl'
        }    // n
      }      // mn
    }        // fullUpdate
  }
}  // CheckFourIKvDft

INSTANTIATE_TEST_SUITE_P(TestLaplaceSolver, FourIKvDftTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-9,
                                           .iter2_to_test = {53, 54}}));

class FourIKuDftTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(FourIKuDftTest, CheckFourIKuDft) {
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
        vmec.run(VmecCheckpoint::VAC1_FOURI_KU_DFT, number_of_iterations)
            .value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_fouri/vac1n_fouri_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_full_update_nestor(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_fouri(filename);
    ASSERT_TRUE(ifs_vac1n_fouri.is_open());
    json vac1n_fouri = json::parse(ifs_vac1n_fouri);

    const int mf = s.mpol + 1;
    const int nf = s.ntor;
    const int mnpd = (mf + 1) * (2 * nf + 1);

    // bvec is only accumulated in SolveForPotential,
    // there must construct it here as well for testing
    std::vector<double> bvec_sin(mnpd);

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);

      const LaplaceSolver& ls = n.GetLaplaceSolver();
      const SingularIntegrals& si = n.GetSingularIntegrals();

      if (vmec.m_[0]->get_ivacskip() == 0) {
        for (int mn = 0; mn < mnpd; ++mn) {
          bvec_sin[mn] += ls.bvec_sin[mn] + si.bvec_sin[mn] / s.nfp;
        }  // mn
      }    // fullUpdate
    }      // thread_id

    // also need to set some elements to 0, as done in SolveForPotential:
    // set n = [-nf, ..., -1], m=0 elements to zero
    // --> only have unique non-zero Fourier coefficients in linear system!
    for (int all_n = 0; all_n < nf; ++all_n) {
      const int m = 0;
      bvec_sin[all_n * (mf + 1) + m] = 0.0;
    }

    // gstore gets (2 pi) / nfp in RegularizedIntegrals in VMEC++; not in
    // Fortran VMEC/Nestor. source gets 1/nfp in fouri; not in VMEC++
    // --> [(2 pi)/nfp] / [1/nfp] == 2 pi
    // (2 pi)^2 comes in Fortran VMEC into the game in fouri when using cosui,
    // sinui, which have alu, alv in them
    const double scale_to_match_fortran = 2.0 * M_PI * 4.0 * M_PI * M_PI;

    // perform the actual test in a single thread,
    // where the data from all threads has been accumulated already
    if (vmec.m_[0]->get_ivacskip() == 0) {
      for (int mn = 0; mn < mnpd; ++mn) {
        // linear index representing -nf:nf
        const int np = mn / (mf + 1);
        const int mp = mn % (mf + 1);

        EXPECT_TRUE(IsCloseRelAbs(vac1n_fouri["bvec"][mp][np],
                                  scale_to_match_fortran * bvec_sin[mn],
                                  tolerance));
      }  // mn

      for (int mn = 0; mn < mnpd; ++mn) {
        // linear index representing -nf:nf
        const int n = mn / (mf + 1);
        const int m = mn % (mf + 1);

        for (int mnp = 0; mnp < mnpd; ++mnp) {
          // linear index representing -nf:nf
          const int np = mnp / (mf + 1);
          const int mp = mnp % (mf + 1);

          EXPECT_TRUE(IsCloseRelAbs(
              vac1n_fouri["amatrix"][m][n][mp][np],
              scale_to_match_fortran * vmec.matrixShare[mnp * mnpd + mn],
              tolerance));
        }  // mn'
      }    // mn
    }      // fullUpdate
  }
}  // CheckFourIKuDft

INSTANTIATE_TEST_SUITE_P(TestLaplaceSolver, FourIKuDftTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-9,
                                           .iter2_to_test = {53, 54}}));

class SolverInputsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(SolverInputsTest, CheckSolverInputs) {
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
        vmec.run(VmecCheckpoint::VAC1_FOURI_KU_DFT, number_of_iterations)
            .value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_solver/vac1n_solver_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_full_update_nestor(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_solver(filename);
    ASSERT_TRUE(ifs_vac1n_solver.is_open());
    json vac1n_solver = json::parse(ifs_vac1n_solver);

    const int mf = s.mpol + 1;
    const int nf = s.ntor;
    const int mnpd = (mf + 1) * (2 * nf + 1);

    // bvec is only accumulated in SolveForPotential,
    // there must construct it here as well for testing
    std::vector<double> bvec_sin(mnpd);

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);

      const LaplaceSolver& ls = n.GetLaplaceSolver();
      const SingularIntegrals& si = n.GetSingularIntegrals();

      if (vmec.m_[0]->get_ivacskip() == 0) {
        for (int mn = 0; mn < mnpd; ++mn) {
          bvec_sin[mn] += ls.bvec_sin[mn] + si.bvec_sin[mn] / s.nfp;
        }  // mn
      }    // fullUpdate
    }      // thread_id

    // also need to set some elements to 0, as done in SolveForPotential:
    // set n = [-nf, ..., -1], m=0 elements to zero
    // --> only have unique non-zero Fourier coefficients in linear system!
    for (int all_n = 0; all_n < nf; ++all_n) {
      const int m = 0;
      bvec_sin[all_n * (mf + 1) + m] = 0.0;
    }

    // gstore gets (2 pi) / nfp in RegularizedIntegrals in VMEC++; not in
    // Fortran VMEC/Nestor. source gets 1/nfp in fouri; not in VMEC++
    // --> [(2 pi)/nfp] / [1/nfp] == 2 pi
    // (2 pi)^2 comes in Fortran VMEC into the game in fouri when using cosui,
    // sinui, which have alu, alv in them
    const double scale_to_match_fortran = 2.0 * M_PI * 4.0 * M_PI * M_PI;

    // perform the actual test in a single thread,
    // where the data from all threads has been accumulated already
    if (vmec.m_[0]->get_ivacskip() == 0) {
      for (int mn = 0; mn < mnpd; ++mn) {
        EXPECT_TRUE(IsCloseRelAbs(vac1n_solver["potvac_in"][mn],
                                  scale_to_match_fortran * bvec_sin[mn],
                                  tolerance));
      }  // mn

      for (int mn = 0; mn < mnpd; ++mn) {
        for (int mnp = 0; mnp < mnpd; ++mnp) {
          EXPECT_TRUE(IsCloseRelAbs(
              vac1n_solver["amatrix"][mn][mnp],
              scale_to_match_fortran * vmec.matrixShare[mnp * mnpd + mn],
              tolerance));
        }  // mn'
      }    // mn
    }      // fullUpdate
  }
}  // CheckSolverInputs

INSTANTIATE_TEST_SUITE_P(TestLaplaceSolver, SolverInputsTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-9,
                                           .iter2_to_test = {53, 54}}));

class LinearSolverTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(LinearSolverTest, CheckLinearSolver) {
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
        vmec.run(VmecCheckpoint::VAC1_SOLVER, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_solver/vac1n_solver_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_solver(filename);
    ASSERT_TRUE(ifs_vac1n_solver.is_open());
    json vac1n_solver = json::parse(ifs_vac1n_solver);

    const int mf = s.mpol + 1;
    const int nf = s.ntor;
    const int mnpd = (mf + 1) * (2 * nf + 1);

    // perform the actual test in a single thread,
    // where the data from all threads has been accumulated already
    if (vmec.m_[0]->get_ivacskip() == 0) {
      for (int mn = 0; mn < mnpd; ++mn) {
        EXPECT_TRUE(IsCloseRelAbs(vac1n_solver["potvac_out"][mn],
                                  vmec.bvecShare[mn], tolerance));
      }  // mn
    }    // fullUpdate
  }
}  // CheckLinearSolver

INSTANTIATE_TEST_SUITE_P(TestLaplaceSolver, LinearSolverTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-9,
                                           .iter2_to_test = {53, 54}}));

}  // namespace vmecpp

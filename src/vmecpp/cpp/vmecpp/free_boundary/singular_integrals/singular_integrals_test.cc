#include "vmecpp/free_boundary/singular_integrals/singular_integrals.h"

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

using ::testing::TestWithParam;
using ::testing::Values;

// used to specify case-specific tolerances
// and which iterations to test
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::vector<int> iter2_to_test = {1, 2};
};

TEST(TestSingularIntegrals, CheckConstants) {
  static constexpr double kTolerance = 1.0e-12;

  const bool lasym = false;
  const int nfp = 5;
  const int mpol = 6;
  const int ntor = 6;
  // will be auto-adjusted by Sizes
  const int ntheta = 0;
  const int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);
  SurfaceGeometry sg(&s, &fb, &tp);

  const int nf = ntor;
  const int mf = mpol + 1;
  SingularIntegrals si(&s, &fb, &tp, &sg, nf, mf);

  for (int n = 0; n < nf + 1; ++n) {
    for (int m = 0; m < mf + 1; ++m) {
      for (int l = std::abs(m - n); l <= m + n; l += 2) {
        const int lnm = (l * (nf + 1) + n) * (mf + 1) + m;

        const int sign = ((l - m + n) / 2) % 2 == 0 ? 1 : -1;

        // need to compute n! / m!
        // Note: n! = gamma(n + 1)
        // Note: lgamma(n + 1) == log(n!)
        // exp(lgamma(n +1) - lgamma(m+1)) == n! / m!
        const double numFac = std::lgamma((m + n + l) / 2 + 1);
        const double denFac1 = std::lgamma((m + n - l) / 2 + 1);
        const double denFac2 = std::lgamma((l + std::abs(m - n)) / 2 + 1);
        const double denFac3 = std::lgamma((l - std::abs(m - n)) / 2 + 1);

        const double cmnRef =
            sign * std::exp(numFac - denFac1 - denFac2 - denFac3);

        EXPECT_TRUE(IsCloseRelAbs(cmnRef, si.cmn[lnm], kTolerance));
      }  // l
    }    // m
  }      // n
}  // CheckConstants

class CmnsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(CmnsTest, CheckCmns) {
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
    ASSERT_TRUE(ifs_vac1n_precal.is_open());
    json vac1n_precal = json::parse(ifs_vac1n_precal);

    // In Fortran VMEC/Nestor, a factor of 2 pi / nfp (called `alp` there) is
    // included in cmns that must be accounted for in this test.
    const double alp = 2.0 * M_PI / s.nfp;

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);
      const SingularIntegrals& si = n.GetSingularIntegrals();

      const int nf = s.ntor;
      const int mf = s.mpol + 1;
      for (int n = 0; n < nf + 1; ++n) {
        for (int m = 0; m < mf + 1; ++m) {
          for (int l = std::abs(m - n); l <= m + n; l += 2) {
            int lnm = (l * (nf + 1) + n) * (mf + 1) + m;

            EXPECT_TRUE(IsCloseRelAbs(vac1n_precal["cmns"][l][m][n],
                                      alp * si.cmns[lnm], tolerance));
          }  // l
        }    // m
      }      // n
    }        // thread_id
  }
}  // CheckCmns

INSTANTIATE_TEST_SUITE_P(TestSingularIntegrals, CmnsTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-14,
                                           .iter2_to_test = {53}}));

class AnalytTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(AnalytTest, CheckAnalyt) {
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
        vmec.run(VmecCheckpoint::VAC1_ANALYT, number_of_iterations).value();
    ASSERT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp/test_data/%s/vac1n_analyt/vac1n_analyt_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_analyt(filename);
    ASSERT_TRUE(ifs_vac1n_analyt.is_open());
    json vac1n_analyt = json::parse(ifs_vac1n_analyt);

    const int nf = s.ntor;
    const int mf = s.mpol + 1;
    const int mnfull = (2 * nf + 1) * (mf + 1);
    std::vector<double> bvec_sin(mnfull, 0.0);

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_[thread_id]);

      const TangentialPartitioning& tp = *vmec.tp_[thread_id];
      const int numLocal = tp.ztMax - tp.ztMin;

      const SingularIntegrals& si = n.GetSingularIntegrals();

      for (int fl = 0; fl < mf + nf + 1; ++fl) {
        for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
          const int l = kl / s.nZeta;
          const int k = kl % s.nZeta;

          const int klRel = kl - tp.ztMin;

          EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["all_tlp"][fl][k][l],
                                    si.Tlp[fl][klRel], tolerance));
          EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["all_tlm"][fl][k][l],
                                    si.Tlm[fl][klRel], tolerance));
        }  // kl
      }    // fl

      // bvec needs to be accumulated over all threads to be compared
      // --> accumulate contributions to Fourier transform from all threads
      for (int mn = 0; mn < mnfull; ++mn) {
        bvec_sin[mn] += si.bvec_sin[mn];
      }

      if (vmec.m_[0]->get_ivacskip() == 0) {
        for (int fl = 0; fl < mf + nf + 1; ++fl) {
          for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
            const int l = kl / s.nZeta;
            const int k = kl % s.nZeta;

            const int klRel = kl - tp.ztMin;

            EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["all_slp"][fl][k][l],
                                      si.Slp[fl][klRel], tolerance));
            EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["all_slm"][fl][k][l],
                                      si.Slm[fl][klRel], tolerance));
          }  // kl
        }    // fl

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
              const double scale_to_match_fortran = 2.0 * M_PI / s.nfp;

              // TODO(jons): for lasym, need cos-part of grpmn from
              // educational_VMEC
              EXPECT_TRUE(
                  IsCloseRelAbs(vac1n_analyt["grpmn"][m][nf + n][k][l],
                                scale_to_match_fortran *
                                    si.grpmn_sin[idx_m_posn * numLocal + klRel],
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(vac1n_analyt["grpmn"][m][nf - n][k][l],
                                scale_to_match_fortran *
                                    si.grpmn_sin[idx_m_negn * numLocal + klRel],
                                tolerance));
            }  // kl
          }    // m
        }      // n
      }        // fullUpdate
    }          // thread_id

    for (int n = 0; n < nf + 1; ++n) {
      for (int m = 0; m < mf + 1; ++m) {
        const int idx_m_posn = (nf + n) * (mf + 1) + m;
        const int idx_m_negn = (nf - n) * (mf + 1) + m;

        // 1. cmns in Fortran has alp (= 2 pi / nfp) in it; VMEC++ does not
        // 2. bexni in Fortran has (2 pi)^2 in it; VMEC++ does not
        const double scale_to_match_fortran =
            2.0 * M_PI / s.nfp * 4.0 * M_PI * M_PI;

        // TODO(jons): for lasym, need cos-part of bvec from educational_VMEC

        // Fortran order along n in bvec: -nf, -nf+1, ..., -1, 0, 1, ..., nf-1,
        // nf
        EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["bvec"][m][nf + n],
                                  scale_to_match_fortran * bvec_sin[idx_m_posn],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["bvec"][m][nf - n],
                                  scale_to_match_fortran * bvec_sin[idx_m_negn],
                                  tolerance));
      }  // m
    }    // n
  }
}  // CheckAnalyt

INSTANTIATE_TEST_SUITE_P(TestSingularIntegrals, AnalytTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-9,
                                           .iter2_to_test = {53, 54}}));

}  // namespace vmecpp

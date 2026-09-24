// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/singular_integrals/singular_integrals.h"

#include <algorithm>
#include <cmath>
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

// Value at the monomial t^l of a linear functional given on the Chebyshev
// polynomials T_0, ..., T_l:
//   t^l = 2^{1-l} sum_{j=0}^{floor(l/2)} binom(l, j) T_{l-2j},
// with the T_0 term halved. The reference data of educational_VMEC is in
// powers of t (cmns, T_l, S_l); VMEC++ holds the same quantities in the
// Chebyshev basis.
static double AtMonomial(int l, const std::vector<double>& at_chebyshev) {
  double sum = 0.0;
  double binomial = 1.0;
  for (int j = 0; 2 * j <= l; ++j) {
    const double weight = (l - 2 * j == 0) ? 0.5 : 1.0;
    sum += weight * binomial * at_chebyshev[l - 2 * j];
    binomial *= static_cast<double>(l - j) / (j + 1);
  }
  return sum * std::ldexp(1.0, 1 - l);
}

static double ChebyshevT(int k, double t) {
  return std::cos(k * std::acos(std::clamp(t, -1.0, 1.0)));
}

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
        "vmecpp_large_cpp_tests/test_data/%s/vac1n_precal/"
        "vac1n_precal_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_precal(filename);
    ASSERT_TRUE(ifs_vac1n_precal.is_open());
    json vac1n_precal = json::parse(ifs_vac1n_precal);

    // In Fortran VMEC/Nestor, a factor of 2 pi / nfp (called `alp` there) is
    // included in cmns that must be accounted for in this test.
    const double alp = 2.0 * M_PI / s.nfp;

    for (int thread_id = 0; thread_id < vmec.vac_num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_vac_[thread_id]);
      const SingularIntegrals& si = n.GetSingularIntegrals();

      const int nf = s.ntor;
      const int mf = s.mpol + 1;
      // cmns are the monomial coefficients of the add-back polynomials, whose
      // Chebyshev coefficients VMEC++ holds; the two are compared as values.
      for (int n = 0; n < nf + 1; ++n) {
        for (int m = 0; m < mf + 1; ++m) {
          for (const double t : {-1.0, -0.6, -0.2, 0.1, 0.5, 0.9, 1.0}) {
            double expected = 0.0;
            for (int l = 0; l <= m + n; ++l) {
              const double cmns = vac1n_precal["cmns"][l][m][n];
              expected += cmns * std::pow(t, l);
            }  // l
            double actual = 0.0;
            for (int k = 0; k <= mf + nf; ++k) {
              const int knm = (k * (nf + 1) + n) * (mf + 1) + m;
              actual += si.chebyshev_coefficients[knm] * ChebyshevT(k, t);
            }  // k
            EXPECT_TRUE(IsCloseRelAbs(expected, alp * actual, tolerance))
                << "(m, n) = (" << m << ", " << n << ") at t = " << t;
          }  // t
        }  // m
      }  // n
    }  // thread_id
  }
}  // CheckCmns

INSTANTIATE_TEST_SUITE_P(TestSingularIntegrals, CmnsTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-12,
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
        "vmecpp_large_cpp_tests/test_data/%s/vac1n_analyt/"
        "vac1n_analyt_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_vac1n_analyt(filename);
    ASSERT_TRUE(ifs_vac1n_analyt.is_open());
    json vac1n_analyt = json::parse(ifs_vac1n_analyt);

    const int nf = s.ntor;
    const int mf = s.mpol + 1;
    const int mnfull = (2 * nf + 1) * (mf + 1);
    std::vector<double> bvec_sin(mnfull, 0.0);
    std::vector<double> bvec_cos(mnfull, 0.0);

    for (int thread_id = 0; thread_id < vmec.vac_num_threads_; ++thread_id) {
      const Nestor& n = static_cast<const Nestor&>(*vmec.fb_vac_[thread_id]);

      const TangentialPartitioning& tp = *vmec.tp_vac_[thread_id];
      const int numLocal = tp.ztMax - tp.ztMin;

      const SingularIntegrals& si = n.GetSingularIntegrals();

      // T_l and S_l of the reference are the monomial counterparts of the
      // Chebyshev moments
      std::vector<double> at_chebyshev_p(mf + nf + 1);
      std::vector<double> at_chebyshev_m(mf + nf + 1);

      for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
        const int l = kl / s.nZeta;
        const int k = kl % s.nZeta;

        const int klRel = kl - tp.ztMin;

        for (int order = 0; order < mf + nf + 1; ++order) {
          at_chebyshev_p[order] = si.chebyshev_moments_p[order][klRel];
          at_chebyshev_m[order] = si.chebyshev_moments_m[order][klRel];
        }
        for (int fl = 0; fl < mf + nf + 1; ++fl) {
          EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["all_tlp"][fl][k][l],
                                    AtMonomial(fl, at_chebyshev_p), tolerance));
          EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["all_tlm"][fl][k][l],
                                    AtMonomial(fl, at_chebyshev_m), tolerance));
        }  // fl
      }  // kl

      // bvec needs to be accumulated over all threads to be compared
      // --> accumulate contributions to Fourier transform from all threads
      for (int mn = 0; mn < mnfull; ++mn) {
        bvec_sin[mn] += si.bvec_sin[mn];
        if (s.lasym) {
          bvec_cos[mn] += si.bvec_cos[mn];
        }
      }

      if (vmec.m_[0]->get_ivacskip() == 0) {
        for (int kl = tp.ztMin; kl < tp.ztMax; ++kl) {
          const int l = kl / s.nZeta;
          const int k = kl % s.nZeta;

          const int klRel = kl - tp.ztMin;

          for (int order = 0; order < mf + nf + 1; ++order) {
            at_chebyshev_p[order] = si.chebyshev_s_moments_p[order][klRel];
            at_chebyshev_m[order] = si.chebyshev_s_moments_m[order][klRel];
          }
          for (int fl = 0; fl < mf + nf + 1; ++fl) {
            EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["all_slp"][fl][k][l],
                                      AtMonomial(fl, at_chebyshev_p),
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["all_slm"][fl][k][l],
                                      AtMonomial(fl, at_chebyshev_m),
                                      tolerance));
          }  // fl
        }  // kl

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

              if (s.lasym) {
                EXPECT_TRUE(IsCloseRelAbs(
                    vac1n_analyt["grpmn_cos"][m][nf + n][k][l],
                    scale_to_match_fortran *
                        si.grpmn_cos[idx_m_posn * numLocal + klRel],
                    tolerance))
                    << "m = " << m << ", n = " << n << ", kl = " << kl;
                EXPECT_TRUE(IsCloseRelAbs(
                    vac1n_analyt["grpmn_cos"][m][nf - n][k][l],
                    scale_to_match_fortran *
                        si.grpmn_cos[idx_m_negn * numLocal + klRel],
                    tolerance))
                    << "m = " << m << ", n = " << n << ", kl = " << kl;
              }
            }  // kl
          }  // m
        }  // n
      }  // fullUpdate
    }  // thread_id

    for (int n = 0; n < nf + 1; ++n) {
      for (int m = 0; m < mf + 1; ++m) {
        const int idx_m_posn = (nf + n) * (mf + 1) + m;
        const int idx_m_negn = (nf - n) * (mf + 1) + m;

        // 1. cmns in Fortran has alp (= 2 pi / nfp) in it; VMEC++ does not
        // 2. bexni in Fortran has (2 pi)^2 in it; VMEC++ does not
        const double scale_to_match_fortran =
            2.0 * M_PI / s.nfp * 4.0 * M_PI * M_PI;

        // Fortran order along n in bvec: -nf, -nf+1, ..., -1, 0, 1, ..., nf-1,
        // nf
        EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["bvec"][m][nf + n],
                                  scale_to_match_fortran * bvec_sin[idx_m_posn],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(vac1n_analyt["bvec"][m][nf - n],
                                  scale_to_match_fortran * bvec_sin[idx_m_negn],
                                  tolerance));

        if (s.lasym) {
          EXPECT_TRUE(IsCloseRelAbs(
              vac1n_analyt["bvec_cos"][m][nf + n],
              scale_to_match_fortran * bvec_cos[idx_m_posn], tolerance))
              << "m = " << m << ", n = " << n;
          EXPECT_TRUE(IsCloseRelAbs(
              vac1n_analyt["bvec_cos"][m][nf - n],
              scale_to_match_fortran * bvec_cos[idx_m_negn], tolerance))
              << "m = " << m << ", n = " << n;
        }
      }  // m
    }  // n
  }
}  // CheckAnalyt

INSTANTIATE_TEST_SUITE_P(TestSingularIntegrals, AnalytTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-9,
                                           .iter2_to_test = {53, 54}},
                                DataSource{
                                    .identifier = "cth_like_free_bdy_asym",
                                    .tolerance = 1.0e-9,
                                    .iter2_to_test = {53}}));

}  // namespace vmecpp

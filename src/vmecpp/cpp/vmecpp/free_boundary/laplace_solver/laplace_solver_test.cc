// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/laplace_solver/laplace_solver.h"

#include <cmath>
#include <numbers>
#include <vector>

#include "gtest/gtest.h"
#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {
namespace {

// Method of manufactured solutions for TransformGreensFunctionDerivative.
//
// We inject a greenp that is exactly one Fourier mode and verify that the
// output grpmn lands in the right coefficient slot with the right value.
//
// Grid angles:  theta_l = 2*pi*l / nThetaEven,  l = 0 ... nThetaReduced-1
//               phi_k   = 2*pi*k / nZeta,        k = 0 ... nZeta-1
//
// Parameterised over lasym so the same fixture exercises both the
// stellarator-symmetric path (grpmn_sin from kernel_odd) and the
// non-stellarator-symmetric path (grpmn_cos from kernel_even).
struct TransformParam {
  bool lasym;
  int m0;  // poloidal mode to inject
  int n0;  // toroidal mode to inject (non-negative, <= nf)
};

class TransformGreensFunctionDerivativeTest
    : public ::testing::TestWithParam<TransformParam> {};

TEST_P(TransformGreensFunctionDerivativeTest, SingleModeRoundTrip) {
  static constexpr double kTol = 1.0e-12;

  const TransformParam& p = GetParam();

  const int nfp = 5;
  const int mpol = 4;
  const int ntor = 4;
  const int ntheta = 0;              // auto-adjusted by Sizes
  const int nzeta = 4 * (ntor + 1);  // must satisfy nzeta >= 2*ntor+1

  Sizes s(p.lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);

  const int nf = ntor;
  const int mf = mpol + 1;
  const int mnpd = (2 * nf + 1) * (mf + 1);
  const int numLocal = tp.ztMax - tp.ztMin;

  // Dummy shared arrays (single-threaded, no cross-thread accumulation needed)
  std::vector<double> matrixShare(mnpd * mnpd, 0.0);
  std::vector<int> iPiv(mnpd, 0);
  std::vector<double> bvecShare(mnpd, 0.0);

  LaplaceSolver ls(&s, &fb, &tp, nf, mf, std::span<double>(matrixShare),
                   std::span<int>(iPiv), std::span<double>(bvecShare));

  // Build greenp[klpRel * nThetaEven * nZeta + l * nZeta + k].
  // For each klp, inject exactly one mode:
  //   stellarator-symmetric:     greenp = sin(m0*theta - n0*phi)
  //   non-stellarator-symmetric: greenp = cos(m0*theta - n0*phi)
  //
  // sin(m0*theta - n0*phi) is odd under (theta,phi) -> (-theta,-phi), so it
  // contributes only to kernel_odd and thus only to grpmn_sin.
  //
  // cos(m0*theta - n0*phi) is even under the same reversal, so it contributes
  // only to kernel_even and thus only to grpmn_cos.
  const int nThetaEven = s.nThetaEven;
  const int nZeta = s.nZeta;

  std::vector<double> greenp(numLocal * nThetaEven * nZeta, 0.0);

  for (int klpRel = 0; klpRel < numLocal; ++klpRel) {
    for (int l = 0; l < nThetaEven; ++l) {
      const double theta = 2.0 * std::numbers::pi * l / nThetaEven;
      for (int k = 0; k < nZeta; ++k) {
        const double phi = 2.0 * std::numbers::pi * k / nZeta;
        const int idx = klpRel * nThetaEven * nZeta + l * nZeta + k;
        if (p.lasym) {
          greenp[idx] = std::cos(p.m0 * theta - p.n0 * phi);
        } else {
          greenp[idx] = std::sin(p.m0 * theta - p.n0 * phi);
        }
      }
    }
  }

  ls.TransformGreensFunctionDerivative(greenp);

  // Expected value: the Fourier transform of sin/cos(m0*theta - n0*phi) over
  // the (l, k) grid, using the integration-weighted scaled basis.
  //
  // The basis functions are normalized so that the Fourier orthogonality sum
  // yields 0.5 per mode (not 1.0). This comes from mscale[m>0] = sqrt(2) and
  // nscale[n>0] = sqrt(2): each divides out of cosmui_scaled/cosnv_scaled, so
  // the trapezoidal DFT of sin^2 or cos^2 gives:
  //   sum_l sinmui_scaled(l,m)*sin(m*theta_l) = (1/mscale) * 0.5 * mscale = 0.5
  //
  // Therefore grpmn_sin/cos at the injected (m0, +n0) slot equals 0.5 per
  // klpRel.

  const int idx_posn = (nf + p.n0) * (mf + 1) + p.m0;
  static constexpr double kExpected = 0.5;

  for (int klpRel = 0; klpRel < numLocal; ++klpRel) {
    if (!p.lasym) {
      EXPECT_NEAR(ls.grpmn_sin[idx_posn * numLocal + klpRel], kExpected, kTol)
          << "grpmn_sin mismatch at klpRel=" << klpRel << " m0=" << p.m0
          << " n0=" << p.n0;
    } else {
      // cos(m*theta - n*phi) is even under (theta,phi)->(-theta,-phi),
      // so it contributes only to grpmn_cos, not grpmn_sin.
      EXPECT_NEAR(ls.grpmn_cos[idx_posn * numLocal + klpRel], kExpected, kTol)
          << "grpmn_cos mismatch at klpRel=" << klpRel << " m0=" << p.m0
          << " n0=" << p.n0;
      EXPECT_NEAR(ls.grpmn_sin[idx_posn * numLocal + klpRel], 0.0, kTol)
          << "grpmn_sin should be zero for pure even mode at klpRel=" << klpRel;
    }
  }

  // For n0 > 0, the transform also fills the negative-n mirror slot
  // (nf - n0) because the code explicitly symmetrises n -> -n.
  // Both slots should have magnitude kExpected; the sign differs:
  //   grpmn_sin[negn] = +kExpected (same sign as posn)
  //   grpmn_cos[negn] = -kExpected (opposite sign)
  const int idx_negn = (nf - p.n0) * (mf + 1) + p.m0;

  // Verify cross-contamination is zero: no energy in other modes.
  for (int mn = 0; mn < mnpd; ++mn) {
    if (mn == idx_posn) continue;
    if (p.n0 > 0 && mn == idx_negn) continue;  // mirror slot is expected
    for (int klpRel = 0; klpRel < numLocal; ++klpRel) {
      EXPECT_NEAR(ls.grpmn_sin[mn * numLocal + klpRel], 0.0, kTol)
          << "unexpected grpmn_sin energy at mn=" << mn;
      if (p.lasym) {
        EXPECT_NEAR(ls.grpmn_cos[mn * numLocal + klpRel], 0.0, kTol)
            << "unexpected grpmn_cos energy at mn=" << mn;
      }
    }
  }

  // Verify the negative-n mirror slot.
  // The negn value depends on m0:
  //   m0 == 0: sin/cos(0 - n0*phi) = -sin/+cos(n0*phi), which has no poloidal
  //            variation; the negn integral == -kExpected (sin) or +kExpected
  //            (cos)
  //   m0 >  0: the cross-term sum_l cos(2*m0*theta_l) vanishes by
  //   orthogonality,
  //            leaving grpmn[negn] == 0.
  if (p.n0 > 0) {
    const double expected_negn_sin = (p.m0 == 0) ? -kExpected : 0.0;
    const double expected_negn_cos = (p.m0 == 0) ? kExpected : 0.0;
    for (int klpRel = 0; klpRel < numLocal; ++klpRel) {
      if (!p.lasym) {
        EXPECT_NEAR(ls.grpmn_sin[idx_negn * numLocal + klpRel],
                    expected_negn_sin, kTol)
            << "grpmn_sin negn mismatch at klpRel=" << klpRel;
      } else {
        EXPECT_NEAR(ls.grpmn_cos[idx_negn * numLocal + klpRel],
                    expected_negn_cos, kTol)
            << "grpmn_cos negn mismatch at klpRel=" << klpRel;
        EXPECT_NEAR(ls.grpmn_sin[idx_negn * numLocal + klpRel], 0.0, kTol)
            << "grpmn_sin at negn should be zero for even mode at klpRel="
            << klpRel;
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    MmsTransformTest, TransformGreensFunctionDerivativeTest,
    ::testing::Values(
        // Stellarator-symmetric: sin(m*theta - n*phi) -> grpmn_sin
        TransformParam{false, 1, 0},  // m=1, n=0 (purely poloidal)
        TransformParam{false, 0, 1},  // m=0, n=1 (purely toroidal)
        TransformParam{false, 2, 1},  // m=2, n=1 (mixed mode)
        TransformParam{false, 3, 2},  // m=3, n=2 (higher mode)
        // Non-stellarator-symmetric: cos(m*theta - n*phi) -> grpmn_cos
        TransformParam{true, 1, 0}, TransformParam{true, 0, 1},
        TransformParam{true, 2, 1}, TransformParam{true, 3, 2}),
    [](const ::testing::TestParamInfo<TransformParam>& info) {
      const auto& p = info.param;
      return std::string(p.lasym ? "asym" : "symm") + "_m" +
             std::to_string(p.m0) + "_n" + std::to_string(p.n0);
    });

// Verify that with zero greenp and zero gstore but a non-zero singular RHS,
// the solve gives exactly Phi = 2 * bvec_sin_singular / nfp.
//
// With no kernel contributions, BuildMatrix gives A = 0.5*I, so the system is
//   0.5 * Phi = b_singular / nfp
// => Phi = 2 * b_singular / nfp
TEST(LaplaceSolverTest, ZeroKernelSolveGivesHalfInverse) {
  static constexpr double kTol = 1.0e-12;

  const bool lasym = false;
  const int nfp = 5;
  const int mpol = 4;
  const int ntor = 4;
  const int ntheta = 0;
  const int nzeta = 4 * (ntor + 1);

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);
  TangentialPartitioning tp(s.nZnT);

  const int nf = ntor;
  const int mf = mpol + 1;
  const int mnpd = (2 * nf + 1) * (mf + 1);
  const int numLocal = tp.ztMax - tp.ztMin;

  std::vector<double> matrixShare(mnpd * mnpd, 0.0);
  std::vector<int> iPiv(mnpd, 0);
  std::vector<double> bvecShare(mnpd, 0.0);

  LaplaceSolver ls(&s, &fb, &tp, nf, mf, std::span<double>(matrixShare),
                   std::span<int>(iPiv), std::span<double>(bvecShare));

  // Zero greenp: no double-layer kernel contribution.
  std::vector<double> greenp(numLocal * s.nThetaEven * s.nZeta, 0.0);
  ls.TransformGreensFunctionDerivative(greenp);

  std::vector<double> grpmn_sin_singular(mnpd * numLocal, 0.0);
  std::vector<double> grpmn_cos_singular(mnpd * numLocal, 0.0);
  ls.AccumulateFullGrpmn(grpmn_sin_singular, grpmn_cos_singular);

  // Zero gstore: no source term from regularized integrals.
  std::vector<double> gstore(s.nThetaEven * s.nZeta, 0.0);
  ls.SymmetriseSourceTerm(gstore);
  ls.PerformToroidalFourierTransforms();
  ls.PerformPoloidalFourierTransforms();
  ls.BuildMatrix();
  ls.DecomposeMatrix();

  // Non-zero singular RHS: inject 1.0 into one mode.
  const int m0 = 2;
  const int n0 = 1;
  const int idx_posn = (nf + n0) * (mf + 1) + m0;
  std::vector<double> bvec_sin_singular(mnpd, 0.0);
  bvec_sin_singular[idx_posn] = 1.0;

  ls.SolveForPotential(bvec_sin_singular);

  // Expected: Phi[idx_posn] = 2 * 1.0 / nfp = 2/nfp
  // All other modes should be zero (A = 0.5*I, diagonal system).
  const double expected = 2.0 / nfp;
  for (int mn = 0; mn < mnpd; ++mn) {
    if (mn == idx_posn) {
      EXPECT_NEAR(bvecShare[mn], expected, kTol)
          << "solution mismatch at mn=" << mn;
    } else {
      EXPECT_NEAR(bvecShare[mn], 0.0, kTol)
          << "unexpected non-zero solution at mn=" << mn;
    }
  }
}

}  // namespace
}  // namespace vmecpp

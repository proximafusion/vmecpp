// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"

#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

namespace {
using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;
}  // namespace

TEST(TestFourierBasisFastToroidal, CheckMNIdx) {
  bool lasym = false;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
  // will be auto-adjusted by Sizes
  int ntheta = 0;
  int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  for (int mn = 0; mn < s.mnmax; ++mn) {
    int m = fb.xm[mn];
    int n = fb.xn[mn] / s.nfp;

    EXPECT_EQ(fb.mnIdx(m, n), mn) << "at m=" << m << " n=" << n;
  }
}  // CheckMNIdx

TEST(TestFourierBasisFastToroidal, CheckCos2CCSS) {
  static constexpr double kTolerance = 1.0e-13;

  bool lasym = false;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
  // will be auto-adjusted by Sizes
  int ntheta = 0;
  int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  // linearly-indexed FCs: cos(xm[mn] theta - xn[mn] zeta), mn = 0, 1, ...,
  // (mnmax-1)
  std::vector<double> fcCosIn(s.mnmax);
  std::vector<double> fcCosOut(s.mnmax);

  // 2D Fourier coefficients
  // cos(m theta) * cos(n zeta)
  std::vector<double> fcCC(s.mpol * (s.ntor + 1));
  // sin(m theta) * sin(n zeta)
  std::vector<double> fcSS(s.mpol * (s.ntor + 1));

  // random test data...
  // standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  for (int mn = 0; mn < s.mnmax; ++mn) {
    fcCosIn[mn] = dist(rng);
  }

  // convert to 2D arrays
  fb.cos_to_cc_ss(fcCosIn, fcCC, fcSS, ntor, mpol);

  // check fcCC and fcSS
  for (int m = 0; m < s.mpol; ++m) {
    // test n = 0 separately: only one contrib from fcCC
    {
      int n = 0;

      double basisNorm = fb.mscale[m] * fb.nscale[n];

      int mnPos = fb.mnIdx(m, n);
      double toTest = fcCC[n * s.mpol + m] * basisNorm;
      EXPECT_TRUE(IsCloseRelAbs(fcCosIn[mnPos], toTest, kTolerance))
          << absl::StrFormat("m=%d n=%d", m, n);
    }

    for (int n = 1; n < s.ntor + 1; ++n) {
      double basisNorm = fb.mscale[m] * fb.nscale[n];

      {
        // test +n
        int mnPos = fb.mnIdx(m, n);
        double toTest = fcCC[n * s.mpol + m] * basisNorm;
        if (m > 0) {
          toTest += fcSS[n * s.mpol + m] * basisNorm;
          toTest /= 2.0;
        }
        EXPECT_TRUE(IsCloseRelAbs(fcCosIn[mnPos], toTest, kTolerance))
            << absl::StrFormat("m=%d n=%d", m, n);
      }

      if (m > 0) {
        // only m=1, 2, ... has negative-n coefficients
        // test -n
        int mnNeg = fb.mnIdx(m, -n);
        double toTest =
            0.5 * (fcCC[n * s.mpol + m] - fcSS[n * s.mpol + m]) * basisNorm;
        EXPECT_TRUE(IsCloseRelAbs(fcCosIn[mnNeg], toTest, kTolerance))
            << absl::StrFormat("m=%d n=%d", m, n);
      }
    }  // n
  }  // m
}  // CheckCos2CCSS

TEST(TestFourierBasisFastToroidal, CheckSin2SCCS) {
  static constexpr double kTolerance = 1.0e-13;

  bool lasym = false;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
  // will be auto-adjusted by Sizes
  int ntheta = 0;
  int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  // linearly-indexed FCs: sin(xm[mn] theta - xn[mn] zeta) for
  // mn = 0, 1, ..., (mnmax-1)
  // --> first one (m=0, n=0) is fixed at 0
  std::vector<double> fcSinIn(s.mnmax);
  std::vector<double> fcSinOut(s.mnmax);

  // 2D Fourier coefficients
  // sin(m theta) * cos(n zeta)
  std::vector<double> fcSC(s.mpol * (s.ntor + 1));
  // cos(m theta) * sin(n zeta)
  std::vector<double> fcCS(s.mpol * (s.ntor + 1));

  // random test data...
  // standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  for (int mn = 0; mn < s.mnmax; ++mn) {
    if (mn > 0) {
      fcSinIn[mn] = dist(rng);
    }
  }
  // sin-(0,0) mode HAS to be zero; just to be sure...
  fcSinIn[0] = 0.0;

  // back ...
  fb.sin_to_sc_cs(fcSinIn, fcSC, fcCS, ntor, mpol);

  // check fcSC and fcCS
  for (int m = 0; m < s.mpol; ++m) {
    int n = 0;
    if (m == 0) {
      // m = 0, n = 0 should be zero
      EXPECT_TRUE(IsCloseRelAbs(0.0, fcSC[n * s.mpol + m], kTolerance))
          << absl::StrFormat("m=%d n=%d", m, n);
    } else {
      // test m > 0, n = 0 separately: only one contrib from fcSC
      double basisNorm = fb.mscale[m] * fb.nscale[n];

      int mnPos = fb.mnIdx(m, n);
      double toTest = fcSC[n * s.mpol + m] * basisNorm;
      EXPECT_TRUE(IsCloseRelAbs(fcSinIn[mnPos], toTest, kTolerance))
          << absl::StrFormat("m=%d n=%d", m, n);
    }

    for (n = 1; n < s.ntor + 1; ++n) {
      double basisNorm = fb.mscale[m] * fb.nscale[n];

      if (m == 0) {
        // test m = 0 separately: only one contrib from fcCS
        {
          // test +n
          int mnPos = fb.mnIdx(m, n);
          double toTest = -fcCS[n * s.mpol + m] * basisNorm;
          EXPECT_TRUE(IsCloseRelAbs(fcSinIn[mnPos], toTest, kTolerance))
              << absl::StrFormat("m=%d n=%d", m, n);
        }
      } else {
        {
          // test +n
          int mnPos = fb.mnIdx(m, n);
          double toTest =
              0.5 * (fcSC[n * s.mpol + m] - fcCS[n * s.mpol + m]) * basisNorm;
          EXPECT_TRUE(IsCloseRelAbs(fcSinIn[mnPos], toTest, kTolerance))
              << absl::StrFormat("m=%d n=%d", m, n);
        }

        {
          // test -n
          int mnNeg = fb.mnIdx(m, -n);
          double toTest =
              0.5 * (fcSC[n * s.mpol + m] + fcCS[n * s.mpol + m]) * basisNorm;
          EXPECT_TRUE(IsCloseRelAbs(fcSinIn[mnNeg], toTest, kTolerance))
              << absl::StrFormat("m=%d n=%d", m, n);
        }
      }
    }  // n
  }  // m
}  // CheckSin2SCCS

TEST(TestFourierBasisFastToroidal, CheckCCSS2Cos) {
  static constexpr double kTolerance = 1.0e-13;

  bool lasym = false;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
  // will be auto-adjusted by Sizes
  int ntheta = 0;
  int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  // linearly-indexed FCs: cos(xm[mn] theta - xn[mn] zeta),
  // mn = 0, 1, ..., (mnmax-1)
  std::vector<double> fcCosIn(s.mnmax);
  std::vector<double> fcCosOut(s.mnmax);

  // 2D Fourier coefficients
  // cos(m theta) * cos(n zeta)
  std::vector<double> fcCC(s.mpol * (s.ntor + 1));
  // sin(m theta) * sin(n zeta)
  std::vector<double> fcSS(s.mpol * (s.ntor + 1));

  // random test data...
  // standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  for (int mn = 0; mn < s.mnmax; ++mn) {
    fcCosIn[mn] = dist(rng);
  }

  // back ...
  fb.cos_to_cc_ss(fcCosIn, fcCC, fcSS, ntor, mpol);

  // ... and forth
  fb.cc_ss_to_cos(fcCC, fcSS, fcCosOut, ntor, mpol);

  // now check for equality
  for (int mn = 0; mn < s.mnmax; ++mn) {
    EXPECT_TRUE(IsCloseRelAbs(fcCosIn[mn], fcCosOut[mn], kTolerance))
        << absl::StrFormat("m=%d n=%d", fb.xm[mn], fb.xn[mn] / s.nfp);
  }
}  // CheckCCSS2Cos

TEST(TestFourierBasisFastToroidal, CheckSCCS2Sin) {
  static constexpr double kTolerance = 1.0e-13;

  bool lasym = false;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
  // will be auto-adjusted by Sizes
  int ntheta = 0;
  int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  // linearly-indexed FCs: // sin(xm[mn] theta - xn[mn] zeta) for
  // mn = 0, 1, ..., (mnmax-1)
  // --> first one (m=0, n=0) is fixed at 0
  std::vector<double> fcSinIn(s.mnmax);
  std::vector<double> fcSinOut(s.mnmax);

  // 2D Fourier coefficients
  // sin(m theta) * cos(n zeta)
  std::vector<double> fcSC(s.mpol * (s.ntor + 1));
  // cos(m theta) * sin(n zeta)
  std::vector<double> fcCS(s.mpol * (s.ntor + 1));

  // random test data...
  // standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  for (int mn = 0; mn < s.mnmax; ++mn) {
    if (mn > 0) {
      fcSinIn[mn] = dist(rng);
    }
  }
  // sin-(0,0) mode HAS to be zero; just to be sure...
  fcSinIn[0] = 0.0;

  // back ...
  fb.sin_to_sc_cs(fcSinIn, fcSC, fcCS, ntor, mpol);

  // ... and forth
  fb.sc_cs_to_sin(fcSC, fcCS, fcSinOut, ntor, mpol);

  // now check for equality
  for (int mn = 0; mn < s.mnmax; ++mn) {
    EXPECT_TRUE(IsCloseRelAbs(fcSinIn[mn], fcSinOut[mn], kTolerance))
        << absl::StrFormat("m=%d n=%d", fb.xm[mn], fb.xn[mn] / s.nfp);
  }
}  // CheckSCCS2Sin

TEST(TestFourierBasisFastToroidal, CheckInvDFTEvn) {
  static constexpr double kTolerance = 1.0e-13;

  bool lasym = false;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
  // will be auto-adjusted by Sizes
  int ntheta = 0;
  int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  // originals: linearly-indexed FCs
  // cos(m theta - n zeta)
  std::vector<double> fcCosIn(s.mnmax);

  // 2D Fourier coefficients
  // cos(m theta) * cos(n zeta)
  std::vector<double> fcCC(s.mpol * (s.ntor + 1));
  // sin(m theta) * sin(n zeta)
  std::vector<double> fcSS(s.mpol * (s.ntor + 1));

  // reduced poloidal range
  std::vector<double> realspace_evn(s.nThetaReduced * s.nZeta);

  // random test data...
  // standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  for (int mn = 0; mn < s.mnmax; ++mn) {
    fcCosIn[mn] = dist(rng);
  }

  // convert to 2D arrays
  fb.cos_to_cc_ss(fcCosIn, fcCC, fcSS, s.ntor, s.mpol);

  // perform inv-DFT from 2D arrays to realspace
  // now perform inv-DFT again and check against original data
  absl::c_fill_n(realspace_evn, s.nThetaReduced * s.nZeta, 0);
  for (int n = 0; n < s.ntor + 1; ++n) {
    for (int l = 0; l < s.nThetaReduced; ++l) {
      double rnkcc = 0.0;
      double rnkss = 0.0;

      for (int m = 0; m < s.mpol; ++m) {
        int idx_mn = n * s.mpol + m;

        double cosmu = fb.cosmu[l * (s.mnyq2 + 1) + m];
        double sinmu = fb.sinmu[l * (s.mnyq2 + 1) + m];

        rnkcc += fcCC[idx_mn] * cosmu;
        rnkss += fcSS[idx_mn] * sinmu;
      }  // m

      for (int k = 0; k < s.nZeta; ++k) {
        int idx_kl = l * s.nZeta + k;

        double cosnv = fb.cosnv[n * s.nZeta + k];
        double sinnv = fb.sinnv[n * s.nZeta + k];

        realspace_evn[idx_kl] += rnkcc * cosnv + rnkss * sinnv;
      }  // k
    }  // l
  }  // n

  double omega_theta = 2.0 * M_PI / s.nThetaEven;
  double omega_zeta = 2.0 * M_PI / (s.nfp * s.nZeta);
  for (int l = 0; l < s.nThetaReduced; ++l) {
    double theta = omega_theta * l;
    for (int k = 0; k < s.nZeta; ++k) {
      double zeta = omega_zeta * k;

      int idx_kl = l * s.nZeta + k;

      double ref_evn = 0.0;
      for (int mn = 0; mn < s.mnmax; ++mn) {
        double kernel = fb.xm[mn] * theta - fb.xn[mn] * zeta;
        ref_evn += fcCosIn[mn] * cos(kernel);
      }

      EXPECT_TRUE(IsCloseRelAbs(ref_evn, realspace_evn[idx_kl], kTolerance))
          << absl::StrFormat("k=%d l=%d", k, l);
    }  // k
  }  // l
}  // CheckInvDFTEvn

TEST(TestFourierBasisFastToroidal, CheckInvDFTOdd) {
  static constexpr double kTolerance = 1.0e-13;

  bool lasym = false;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
  // will be auto-adjusted by Sizes
  int ntheta = 0;
  int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  // originals: linearly-indexed FCs
  // sin(m theta - n zeta) --> first one is fixed at 0
  std::vector<double> fcSinIn(s.mnmax);

  // 2D Fourier coefficients
  // sin(m theta) * cos(n zeta)
  std::vector<double> fcSC(s.mpol * (s.ntor + 1));
  // cos(m theta) * sin(n zeta)
  std::vector<double> fcCS(s.mpol * (s.ntor + 1));

  // reduced poloidal range
  std::vector<double> realspace_odd(s.nThetaReduced * s.nZeta);

  // random test data...
  // standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  for (int mn = 0; mn < s.mnmax; ++mn) {
    if (mn > 0) {
      fcSinIn[mn] = dist(rng);
    }
  }
  // sin-(0,0) mode HAS to be zero; just to be sure...
  fcSinIn[0] = 0.0;

  // convert to 2D arrays
  fb.sin_to_sc_cs(fcSinIn, fcSC, fcCS, ntor, mpol);

  // perform inv-DFT from 2D arrays to realspace
  // now perform inv-DFT again and check against original data
  absl::c_fill_n(realspace_odd, s.nThetaReduced * s.nZeta, 0);
  for (int n = 0; n < s.ntor + 1; ++n) {
    for (int l = 0; l < s.nThetaReduced; ++l) {
      double rnksc = 0.0;
      double rnkcs = 0.0;

      for (int m = 0; m < s.mpol; ++m) {
        int idx_mn = n * s.mpol + m;

        double cosmu = fb.cosmu[l * (s.mnyq2 + 1) + m];
        double sinmu = fb.sinmu[l * (s.mnyq2 + 1) + m];

        rnksc += fcSC[idx_mn] * sinmu;
        rnkcs += fcCS[idx_mn] * cosmu;
      }  // m

      for (int k = 0; k < s.nZeta; ++k) {
        int idx_kl = l * s.nZeta + k;

        double cosnv = fb.cosnv[n * s.nZeta + k];
        double sinnv = fb.sinnv[n * s.nZeta + k];

        realspace_odd[idx_kl] += rnksc * cosnv + rnkcs * sinnv;
      }  // k
    }  // l
  }  // n

  double omega_theta = 2.0 * M_PI / s.nThetaEven;
  double omega_zeta = 2.0 * M_PI / (s.nfp * s.nZeta);
  for (int l = 0; l < s.nThetaReduced; ++l) {
    double theta = omega_theta * l;
    for (int k = 0; k < s.nZeta; ++k) {
      double zeta = omega_zeta * k;

      int idx_kl = l * s.nZeta + k;

      double ref_odd = 0.0;
      for (int mn = 0; mn < s.mnmax; ++mn) {
        double kernel = fb.xm[mn] * theta - fb.xn[mn] * zeta;
        ref_odd += fcSinIn[mn] * sin(kernel);
      }

      EXPECT_TRUE(IsCloseRelAbs(ref_odd, realspace_odd[idx_kl], kTolerance))
          << absl::StrFormat("k=%d l=%d", k, l);
    }
  }
}  // CheckInvDFTOdd

TEST(TestFourierBasisFastToroidal, CheckInvDFTCombined) {
  static constexpr double kTolerance = 1.0e-13;

  bool lasym = false;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
  // will be auto-adjusted by Sizes
  int ntheta = 0;
  int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  // originals: linearly-indexed FCs
  // cos(m theta - n zeta)
  std::vector<double> fcCosIn(s.mnmax);
  // sin(m theta - n zeta) --> first one is fixed at 0
  std::vector<double> fcSinIn(s.mnmax);

  // 2D Fourier coefficients
  // cos(m theta) * cos(n zeta)
  std::vector<double> fcCC(s.mpol * (s.ntor + 1));
  // sin(m theta) * sin(n zeta)
  std::vector<double> fcSS(s.mpol * (s.ntor + 1));
  // sin(m theta) * cos(n zeta)
  std::vector<double> fcSC(s.mpol * (s.ntor + 1));
  // cos(m theta) * sin(n zeta)
  std::vector<double> fcCS(s.mpol * (s.ntor + 1));

  // reduced poloidal range
  std::vector<double> realspace_evn(s.nThetaReduced * s.nZeta);
  std::vector<double> realspace_odd(s.nThetaReduced * s.nZeta);

  // full poloidal range
  std::vector<double> realspace(s.nThetaEven * s.nZeta);

  // random test data...
  // standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  for (int mn = 0; mn < s.mnmax; ++mn) {
    fcCosIn[mn] = dist(rng);
    if (mn > 0) {
      fcSinIn[mn] = dist(rng);
    }
  }
  // sin-(0,0) mode HAS to be zero; just to be sure...
  fcSinIn[0] = 0.0;

  // convert to 2D arrays
  fb.cos_to_cc_ss(fcCosIn, fcCC, fcSS, ntor, mpol);
  fb.sin_to_sc_cs(fcSinIn, fcSC, fcCS, ntor, mpol);

  // perform inv-DFT from 2D arrays to realspace
  // now perform inv-DFT again and check against original data
  absl::c_fill_n(realspace_evn, s.nThetaReduced * s.nZeta, 0);
  absl::c_fill_n(realspace_odd, s.nThetaReduced * s.nZeta, 0);
  for (int n = 0; n < s.ntor + 1; ++n) {
    for (int l = 0; l < s.nThetaReduced; ++l) {
      double rnkcc = 0.0;
      double rnkss = 0.0;
      double rnksc = 0.0;
      double rnkcs = 0.0;

      for (int m = 0; m < s.mpol; ++m) {
        int idx_mn = n * s.mpol + m;

        double cosmu = fb.cosmu[l * (s.mnyq2 + 1) + m];
        double sinmu = fb.sinmu[l * (s.mnyq2 + 1) + m];

        rnkcc += fcCC[idx_mn] * cosmu;
        rnkss += fcSS[idx_mn] * sinmu;
        rnksc += fcSC[idx_mn] * sinmu;
        rnkcs += fcCS[idx_mn] * cosmu;
      }  // m

      for (int k = 0; k < s.nZeta; ++k) {
        int idx_kl = l * s.nZeta + k;

        double cosnv = fb.cosnv[n * s.nZeta + k];
        double sinnv = fb.sinnv[n * s.nZeta + k];

        realspace_evn[idx_kl] += rnkcc * cosnv + rnkss * sinnv;
        realspace_odd[idx_kl] += rnksc * cosnv + rnkcs * sinnv;
      }  // k
    }  // l
  }  // n

  // compose full realspace from even- and odd-parity contributions
  absl::c_fill_n(realspace, s.nThetaEven * s.nZeta, 0);
  for (int l = 0; l < s.nThetaReduced; ++l) {
    int lReversed = (s.nThetaEven - l) % s.nThetaEven;
    for (int k = 0; k < s.nZeta; ++k) {
      int kReversed = (s.nZeta - k) % s.nZeta;

      int kl = l * s.nZeta + k;
      int klReversed = lReversed * s.nZeta + kReversed;

      realspace[kl] = realspace_evn[kl] + realspace_odd[kl];
      if (kl != klReversed) {
        realspace[klReversed] = realspace_evn[kl] - realspace_odd[kl];
      }
    }  // l
  }  // k

  double omega_theta = 2.0 * M_PI / s.nThetaEven;
  double omega_zeta = 2.0 * M_PI / (s.nfp * s.nZeta);
  for (int l = 0; l < s.nThetaEven; ++l) {
    double theta = omega_theta * l;
    for (int k = 0; k < s.nZeta; ++k) {
      double zeta = omega_zeta * k;

      int idx_kl = l * s.nZeta + k;

      double ref = 0.0;
      for (int mn = 0; mn < s.mnmax; ++mn) {
        double kernel = fb.xm[mn] * theta - fb.xn[mn] * zeta;
        ref += fcCosIn[mn] * cos(kernel) + fcSinIn[mn] * sin(kernel);
      }

      EXPECT_TRUE(IsCloseRelAbs(ref, realspace[idx_kl], kTolerance))
          << absl::StrFormat("k=%d l=%d", k, l);
    }
  }
}  // CheckInvDFTCombined

TEST(TestFourierBasisFastToroidal, CheckOrthogonality) {
  static constexpr double kTolerance = 1.0e-13;

  bool lasym = false;
  int nfp = 5;
  int mpol = 12;
  int ntor = 12;
  // will be auto-adjusted by Sizes
  int ntheta = 0;
  int nzeta = 36;

  Sizes s(lasym, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastToroidal fb(&s);

  // originals: linearly-indexed FCs
  // cos(m theta - n zeta)
  std::vector<double> fcCosIn(s.mnmax);
  // sin(m theta - n zeta) --> first one is fixed at 0
  std::vector<double> fcSinIn(s.mnmax);

  // reproductions: linearly-indexed FCs
  // cos(m theta - n zeta)
  std::vector<double> fcCosOut(s.mnmax);
  // sin(m theta - n zeta) --> first one is fixed at 0
  std::vector<double> fcSinOut(s.mnmax);

  // 2D Fourier coefficients
  // cos(m theta) * cos(n zeta)
  std::vector<double> fcCC(s.mpol * (s.ntor + 1));
  // sin(m theta) * sin(n zeta)
  std::vector<double> fcSS(s.mpol * (s.ntor + 1));
  // sin(m theta) * cos(n zeta)
  std::vector<double> fcSC(s.mpol * (s.ntor + 1));
  // cos(m theta) * sin(n zeta)
  std::vector<double> fcCS(s.mpol * (s.ntor + 1));

  // reduced poloidal range
  std::vector<double> realspace_evn(s.nThetaReduced * s.nZeta);
  std::vector<double> realspace_odd(s.nThetaReduced * s.nZeta);

  // full poloidal range
  std::vector<double> realspace(s.nThetaEven * s.nZeta);

  // random test data...
  // standard mersenne_twister_engine
  std::mt19937 rng(42);
  std::uniform_real_distribution<> dist(-1., 1.);

  for (int mn = 0; mn < s.mnmax; ++mn) {
    fcCosIn[mn] = dist(rng);
    if (mn > 0) {
      fcSinIn[mn] = dist(rng);
    }
  }
  // sin-(0,0) mode HAS to be zero; just to be sure...
  fcSinIn[0] = 0.0;

  // convert to 2D arrays
  fb.cos_to_cc_ss(fcCosIn, fcCC, fcSS, ntor, mpol);
  fb.sin_to_sc_cs(fcSinIn, fcSC, fcCS, ntor, mpol);

  // perform inv-DFT from 2D arrays to realspace
  // now perform inv-DFT again and check against original data
  absl::c_fill_n(realspace_evn, s.nThetaReduced * s.nZeta, 0);
  absl::c_fill_n(realspace_odd, s.nThetaReduced * s.nZeta, 0);
  for (int n = 0; n < s.ntor + 1; ++n) {
    for (int l = 0; l < s.nThetaReduced; ++l) {
      double rnkcc = 0.0;
      double rnkss = 0.0;
      double rnksc = 0.0;
      double rnkcs = 0.0;

      for (int m = 0; m < s.mpol; ++m) {
        int idx_mn = n * s.mpol + m;

        double cosmu = fb.cosmu[l * (s.mnyq2 + 1) + m];
        double sinmu = fb.sinmu[l * (s.mnyq2 + 1) + m];

        rnkcc += fcCC[idx_mn] * cosmu;
        rnkss += fcSS[idx_mn] * sinmu;
        rnksc += fcSC[idx_mn] * sinmu;
        rnkcs += fcCS[idx_mn] * cosmu;
      }  // m

      for (int k = 0; k < s.nZeta; ++k) {
        int idx_kl = l * s.nZeta + k;

        double cosnv = fb.cosnv[n * s.nZeta + k];
        double sinnv = fb.sinnv[n * s.nZeta + k];

        realspace_evn[idx_kl] += rnkcc * cosnv + rnkss * sinnv;
        realspace_odd[idx_kl] += rnksc * cosnv + rnkcs * sinnv;
      }  // k
    }  // l
  }  // n

  // compose full realspace from even- and odd-parity contributions
  absl::c_fill_n(realspace, s.nThetaEven * s.nZeta, 0);
  for (int l = 0; l < s.nThetaReduced; ++l) {
    int lReversed = (s.nThetaEven - l) % s.nThetaEven;
    for (int k = 0; k < s.nZeta; ++k) {
      int kReversed = (s.nZeta - k) % s.nZeta;

      int kl = l * s.nZeta + k;
      int klReversed = lReversed * s.nZeta + kReversed;

      realspace[kl] = realspace_evn[kl] + realspace_odd[kl];
      if (kl != klReversed) {
        realspace[klReversed] = realspace_evn[kl] - realspace_odd[kl];
      }
    }  // l
  }  // k

  // decompose back into even- and odd-parity contributions
  for (int l = 0; l < s.nThetaReduced; ++l) {
    int lReversed = (s.nThetaEven - l) % s.nThetaEven;
    for (int k = 0; k < s.nZeta; ++k) {
      int kReversed = (s.nZeta - k) % s.nZeta;

      int kl = l * s.nZeta + k;
      int klReversed = lReversed * s.nZeta + kReversed;

      realspace_evn[kl] = 0.5 * (realspace[kl] + realspace[klReversed]);
      realspace_odd[kl] = 0.5 * (realspace[kl] - realspace[klReversed]);
    }
  }

  // perform fwd-DFT separately for even- and odd-parity contributions
  absl::c_fill_n(fcCC, s.mpol * (s.ntor + 1), 0);
  absl::c_fill_n(fcSS, s.mpol * (s.ntor + 1), 0);
  absl::c_fill_n(fcSC, s.mpol * (s.ntor + 1), 0);
  absl::c_fill_n(fcCS, s.mpol * (s.ntor + 1), 0);
  for (int n = 0; n < s.ntor + 1; ++n) {
    for (int l = 0; l < s.nThetaReduced; ++l) {
      double rnkcc = 0.0;
      double rnkss = 0.0;
      double rnksc = 0.0;
      double rnkcs = 0.0;

      for (int k = 0; k < s.nZeta; ++k) {
        int idx_kl = l * s.nZeta + k;

        double cosnv = fb.cosnv[n * s.nZeta + k];
        double sinnv = fb.sinnv[n * s.nZeta + k];

        rnkcc += realspace_evn[idx_kl] * cosnv;
        rnkss += realspace_evn[idx_kl] * sinnv;
        rnksc += realspace_odd[idx_kl] * cosnv;
        rnkcs += realspace_odd[idx_kl] * sinnv;
      }  // k

      for (int m = 0; m < s.mpol; ++m) {
        int idx_mn = n * s.mpol + m;

        double cosmui = fb.cosmui[l * (s.mnyq2 + 1) + m];
        double sinmui = fb.sinmui[l * (s.mnyq2 + 1) + m];

        fcCC[idx_mn] += rnkcc * cosmui;
        fcSS[idx_mn] += rnkss * sinmui;
        fcSC[idx_mn] += rnksc * sinmui;
        fcCS[idx_mn] += rnkcs * cosmui;
      }  // m
    }  // l
  }  // n

  // combine back into linear-indexed Fourier coefficient arrays
  fb.cc_ss_to_cos(fcCC, fcSS, fcCosOut, ntor, mpol);
  fb.sc_cs_to_sin(fcSC, fcCS, fcSinOut, ntor, mpol);

  // check for equality with initial data
  for (int mn = 0; mn < s.mnmax; ++mn) {
    EXPECT_TRUE(IsCloseRelAbs(fcCosIn[mn], fcCosOut[mn], kTolerance))
        << absl::StrFormat("m=%d n=%d", fb.xm[mn], fb.xn[mn] / s.nfp);
    EXPECT_TRUE(IsCloseRelAbs(fcSinIn[mn], fcSinOut[mn], kTolerance))
        << absl::StrFormat("m=%d n=%d", fb.xm[mn], fb.xn[mn] / s.nfp);
  }
}  // CheckOrthogonality

// Test that the functionality within FourierBasisFastToroidal is consistent
// with a straight-forward implementation done here separately.
// Note that this does not compare against Fortran yet!
TEST(TestFourierBasisFastToroidal, CheckInternally) {
  static constexpr double kTwoPi = 2.0 * M_PI;

  static constexpr double kTolerance = 1.0e-30;

  bool lasym = false;
  int nfp = 1;
  int mpol = 12;
  int ntor = 12;
  // Sizes-internal value will be automatically chosen based on mpol
  int ntheta = 0;
  // Sizes-internal value will be automatically chosen based on ntor
  int nzeta = 0;
  Sizes sizes(lasym, nfp, mpol, ntor, ntheta, nzeta);

  FourierBasisFastToroidal fourier_basis(&sizes);

  // norm for surface integrals (wInt in Sizes)
  const double d_theta_zeta = 1.0 / (sizes.nZeta * (sizes.nThetaReduced - 1));

  {  // test poloidal Fourier basis
    EXPECT_EQ(fourier_basis.mscale[0], 1.0);
    for (int m = 1; m < sizes.mnyq2 + 1; ++m) {
      EXPECT_EQ(fourier_basis.mscale[m], std::sqrt(2.0));
    }

    // cosmu = cos(m * theta)
    // sinmu = sin(m * theta)
    for (int l = 0; l < sizes.nThetaReduced; ++l) {
      const double theta = kTwoPi * l / sizes.nThetaEven;
      for (int m = 0; m < sizes.mnyq2 + 1; ++m) {
        const int idx_lm = l * (sizes.mnyq2 + 1) + m;

        const double arg = m * theta;

        const double reference_cosmu = std::cos(arg) * fourier_basis.mscale[m];
        const double reference_sinmu = std::sin(arg) * fourier_basis.mscale[m];

        double reference_cosmui = reference_cosmu * d_theta_zeta;
        const double reference_sinmui = reference_sinmu * d_theta_zeta;

        if (l == 0 || l == sizes.nThetaReduced - 1) {
          reference_cosmui /= 2.0;
        }

        {  // non-derivative transforms
          EXPECT_TRUE(IsCloseRelAbs(reference_cosmu,
                                    fourier_basis.cosmu[idx_lm], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(reference_sinmu,
                                    fourier_basis.sinmu[idx_lm], kTolerance));

          EXPECT_TRUE(IsCloseRelAbs(reference_cosmui,
                                    fourier_basis.cosmui[idx_lm], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(reference_sinmui,
                                    fourier_basis.sinmui[idx_lm], kTolerance));
        }

        {  // poloidal derivatives
          EXPECT_TRUE(IsCloseRelAbs(m * reference_cosmu,
                                    fourier_basis.cosmum[idx_lm], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(-m * reference_sinmu,
                                    fourier_basis.sinmum[idx_lm], kTolerance));

          EXPECT_TRUE(IsCloseRelAbs(m * reference_cosmui,
                                    fourier_basis.cosmumi[idx_lm], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(-m * reference_sinmui,
                                    fourier_basis.sinmumi[idx_lm], kTolerance));
        }
      }  // l
    }  // m
  }

  {  // test toroidal Fourier basis
    EXPECT_EQ(fourier_basis.nscale[0], 1.0);
    for (int n = 1; n < sizes.nnyq2 + 1; ++n) {
      EXPECT_EQ(fourier_basis.nscale[n], std::sqrt(2.0));
    }

    for (int k = 0; k < sizes.nZeta; ++k) {
      const double zeta = kTwoPi * k / sizes.nZeta;
      for (int n = 0; n < sizes.nnyq2 + 1; ++n) {
        const int idx_nk = n * sizes.nZeta + k;

        const double arg = n * zeta;

        const double reference_cosnv = std::cos(arg) * fourier_basis.nscale[n];
        const double reference_sinnv = std::sin(arg) * fourier_basis.nscale[n];

        {  // non-derivative transforms
          EXPECT_TRUE(IsCloseRelAbs(reference_cosnv,
                                    fourier_basis.cosnv[idx_nk], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(reference_sinnv,
                                    fourier_basis.sinnv[idx_nk], kTolerance));
        }

        {  // toroidal derivatives
          EXPECT_TRUE(IsCloseRelAbs(n * reference_cosnv,
                                    fourier_basis.cosnvn[idx_nk], kTolerance));
          EXPECT_TRUE(IsCloseRelAbs(-n * reference_sinnv,
                                    fourier_basis.sinnvn[idx_nk], kTolerance));
        }
      }  // k
    }  // n
  }

  {  // test basis conversion indices
    int mn = 0;

    {  // m = 0; n = 0, 1, ..., ntor
      int m = 0;
      for (int n = 0; n <= sizes.ntor; ++n) {
        EXPECT_EQ(fourier_basis.xm[mn], m);
        EXPECT_EQ(fourier_basis.xn[mn], n * sizes.nfp);
        mn++;
      }
    }

    {  // m = 1, 2, ..., (mpol-1); n = -ntor, ..., -1, 0, 1, ..., ntor
      for (int m = 1; m < sizes.mpol; ++m) {
        for (int n = -sizes.ntor; n <= sizes.ntor; ++n) {
          EXPECT_EQ(fourier_basis.xm[mn], m);
          EXPECT_EQ(fourier_basis.xn[mn], n * sizes.nfp);
          mn++;
        }  // n
      }  // m
    }

    EXPECT_EQ(mn, sizes.mnmax);
  }

  {  // test basis conversion indices for Nyquist-sized arrays
    int mn_nyq = 0;

    {  // m = 0; n = 0, 1, ..., nnyq
      int m = 0;
      for (int n = 0; n < sizes.nnyq + 1; ++n) {
        EXPECT_EQ(fourier_basis.xm_nyq[mn_nyq], m);
        EXPECT_EQ(fourier_basis.xn_nyq[mn_nyq], n * sizes.nfp);
        mn_nyq++;
      }
    }

    {  // m = 1, 2, ..., mnyq; n = -nnyq, ..., -1, 0, 1, ..., nnyq
      for (int m = 1; m < sizes.mnyq + 1; ++m) {
        for (int n = -sizes.nnyq; n <= sizes.nnyq; ++n) {
          EXPECT_EQ(fourier_basis.xm_nyq[mn_nyq], m);
          EXPECT_EQ(fourier_basis.xn_nyq[mn_nyq], n * sizes.nfp);
          mn_nyq++;
        }  // n
      }  // m
    }

    EXPECT_EQ(mn_nyq, sizes.mnmax_nyq);
  }
}  // CheckInternally

}  // namespace vmecpp

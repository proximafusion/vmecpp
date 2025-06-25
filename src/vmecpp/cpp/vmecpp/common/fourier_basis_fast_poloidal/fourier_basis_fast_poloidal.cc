// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"

#include <cmath>
#include <numbers>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "vmecpp/common/util/util.h"

namespace vmecpp {

FourierBasisFastPoloidal::FourierBasisFastPoloidal(const Sizes* s) : s_(*s) {
  mscale.resize(s_.mnyq2 + 1);
  nscale.resize(s_.nnyq2 + 1);

  cosmu.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
  sinmu.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
  cosmum.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
  sinmum.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
  cosmui.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
  sinmui.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
  cosmumi.resize(s_.nThetaReduced * (s_.mnyq2 + 1));
  sinmumi.resize(s_.nThetaReduced * (s_.mnyq2 + 1));

  cosnv.resize((s_.nnyq2 + 1) * s_.nZeta);
  sinnv.resize((s_.nnyq2 + 1) * s_.nZeta);
  cosnvn.resize((s_.nnyq2 + 1) * s_.nZeta);
  sinnvn.resize((s_.nnyq2 + 1) * s_.nZeta);

  computeFourierBasisFastPoloidal(s_.nfp);

  // -----------------

  xm.resize(s_.mnmax, 0);
  xn.resize(s_.mnmax, 0);

  computeConversionIndices(/*m_xm=*/xm, /*m_xn=*/xn, s_.ntor, s_.mpol, s_.nfp);

  xm_nyq.resize(s_.mnmax_nyq, 0);
  xn_nyq.resize(s_.mnmax_nyq, 0);

  computeConversionIndices(/*m_xm=*/xm_nyq, /*m_xn=*/xn_nyq, s_.nnyq,
                           s_.mnyq + 1, s_.nfp);
}

void FourierBasisFastPoloidal::computeFourierBasisFastPoloidal(int nfp) {
  static constexpr double kTwoPi = 2.0 * M_PI;

  // Fourier transforms are always computed in VMEC
  // over the reduced theta interval from [0, pi].
  // Thus, need a fixed normalization factor (cannot use dnorm3 or wInt in
  // Sizes) here.
  const double intNorm = 1.0 / (s_.nZeta * (s_.nThetaReduced - 1));

  // poloidal
  for (int m = 0; m < s_.mnyq2 + 1; ++m) {
    // DFTs for m>0 need 1/pi==2/(2pi) normalization factor
    // vs. 1/(2pi) for the cos(m=0)-mode.
    // --> introduce one sqrt(2) in fwd-DFT (geometry-into-realspace)
    //     and one sqrt(2) into inv-DFT (forces-into-Fourier) via mscale
    if (m == 0) {
      mscale[m] = 1.0;
    } else {
      mscale[m] = std::numbers::sqrt2;
    }
  }  // m

  for (int m = 0; m < s_.mnyq2 + 1; ++m) {
    for (int l = 0; l < s_.nThetaReduced; ++l) {
      // need to compute theta grid using _full_ number of theta points!
      const double theta = kTwoPi * l / s_.nThetaEven;
      const int idx_ml = m * s_.nThetaReduced + l;

      const double arg = m * theta;

      // poloidal Fourier basis
      cosmu[idx_ml] = std::cos(arg) * mscale[m];
      sinmu[idx_ml] = std::sin(arg) * mscale[m];

      // integration
      cosmui[idx_ml] = cosmu[idx_ml] * intNorm;
      sinmui[idx_ml] = sinmu[idx_ml] * intNorm;

      if (l == 0 || l == s_.nThetaReduced - 1) {
        cosmui[idx_ml] /= 2.0;
      }

      // poloidal derivatives
      cosmum[idx_ml] = m * cosmu[idx_ml];
      sinmum[idx_ml] = -m * sinmu[idx_ml];

      cosmumi[idx_ml] = m * cosmui[idx_ml];
      sinmumi[idx_ml] = -m * sinmui[idx_ml];
    }  // l
  }  // m

  // toroidal
  for (int n = 0; n < s_.nnyq2 + 1; ++n) {
    // DFTs for m>0 need 1/pi==2/(2pi) normalization factor
    // vs. 1/(2pi) for the cos(m=0)-mode.
    // --> introduce one sqrt(2) in fwd-DFT (geometry-into-realspace)
    //     and one sqrt(2) into inv-DFT (forces-into-Fourier) via nscale
    if (n == 0) {
      nscale[n] = 1.0;
    } else {
      nscale[n] = std::numbers::sqrt2;
    }
  }  // n

  for (int k = 0; k < s_.nZeta; ++k) {
    const double zeta = kTwoPi * k / s_.nZeta;
    for (int n = 0; n < s_.nnyq2 + 1; ++n) {
      const int idx_kn = k * (s_.nnyq2 + 1) + n;

      const double arg = n * zeta;

      // toroidal Fourier basis
      cosnv[idx_kn] = std::cos(arg) * nscale[n];
      sinnv[idx_kn] = std::sin(arg) * nscale[n];

      // toroidal derivatives
      cosnvn[idx_kn] = n * nfp * cosnv[idx_kn];
      sinnvn[idx_kn] = -n * nfp * sinnv[idx_kn];
    }  // n
  }  // k
}

// convert cos(xm[mn] theta - xn[mn] zeta) into 2D FC array form
int FourierBasisFastPoloidal::cos_to_cc_ss(const std::span<const double> fcCos,
                                           std::span<double> m_fcCC,
                                           std::span<double> m_fcSS, int n_size,
                                           int m_size) const {
  // m = 0: n =  0, 1, ..., ntor --> ntor + 1
  // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
  int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

  absl::c_fill_n(m_fcCC, m_size * (n_size + 1), 0);
  absl::c_fill_n(m_fcSS, m_size * (n_size + 1), 0);

  int mn = 0;

  int m = 0;
  for (int n = 0; n < n_size + 1; ++n) {
    int abs_n = abs(n);

    double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

    double normedFC = basis_norm * fcCos[mn];

    m_fcCC[m * (n_size + 1) + abs_n] += normedFC;
    // no contribution to fcSS where (m == 0 || n == 0)

    mn++;
  }

  for (m = 1; m < m_size; ++m) {
    for (int n = -n_size; n < n_size + 1; ++n) {
      int abs_n = abs(n);
      int sgn_n = signum(n);

      double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

      double normedFC = basis_norm * fcCos[mn];

      m_fcCC[m * (n_size + 1) + abs_n] += normedFC;
      if (abs_n > 0) {
        m_fcSS[m * (n_size + 1) + abs_n] += sgn_n * normedFC;
      }

      mn++;
    }  // n
  }  // m

  CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax
                      << " in cos_to_cc_ss";

  return mnmax;
}

int FourierBasisFastPoloidal::sin_to_sc_cs(const std::span<const double> fcSin,
                                           std::span<double> m_fcSC,
                                           std::span<double> m_fcCS, int n_size,
                                           int m_size) const {
  // m = 0: n =  0, 1, ..., ntor --> ntor + 1
  // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
  int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

  absl::c_fill_n(m_fcSC, m_size * (n_size + 1), 0);
  absl::c_fill_n(m_fcCS, m_size * (n_size + 1), 0);

  int mn = 1;

  int m = 0;
  for (int n = 1; n < n_size + 1; ++n) {
    int abs_n = abs(n);
    int sgn_n = signum(n);

    double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

    double normedFC = basis_norm * fcSin[mn];

    // no contribution to fcSC where m == 0
    // check for n > 0 is redundant when starting loop at n=1
    m_fcCS[m * (n_size + 1) + abs_n] = -sgn_n * normedFC;

    mn++;
  }

  for (m = 1; m < m_size; ++m) {
    for (int n = -n_size; n < n_size + 1; ++n) {
      int abs_n = abs(n);
      int sgn_n = signum(n);

      double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

      double normedFC = basis_norm * fcSin[mn];

      m_fcSC[m * (n_size + 1) + abs_n] += normedFC;
      if (abs_n > 0) {
        m_fcCS[m * (n_size + 1) + abs_n] += -sgn_n * normedFC;
      }

      mn++;
    }  // n
  }  // m

  CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax
                      << " in sin_to_sc_cs";

  return mnmax;
}

int FourierBasisFastPoloidal::cc_ss_to_cos(const std::span<const double> fcCC,
                                           const std::span<const double> fcSS,
                                           std::span<double> m_fcCos,
                                           int n_size, int m_size) const {
  // m = 0: n =  0, 1, ..., ntor --> ntor + 1
  // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
  int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

  absl::c_fill_n(m_fcCos, mnmax, 0);

  int mn = 0;

  int m = 0;
  for (int n = 0; n < n_size + 1; ++n) {
    double basis_norm = 1.0 / (mscale[m] * nscale[n]);

    m_fcCos[mn] = fcCC[n] / basis_norm;

    mn++;
  }  // n

  for (m = 1; m < m_size; ++m) {
    for (int n = -n_size; n < n_size + 1; ++n) {
      int abs_n = abs(n);
      int sgn_n = signum(n);

      double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

      if (abs_n == 0) {
        m_fcCos[mn] = fcCC[m * (n_size + 1) + abs_n] / basis_norm;
      } else {
        double raw_cc = fcCC[m * (n_size + 1) + abs_n];
        double raw_ss = fcSS[m * (n_size + 1) + abs_n];
        m_fcCos[mn] = 0.5 * (raw_cc + sgn_n * raw_ss) / basis_norm;
      }

      mn++;
    }  // n
  }  // m

  CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax
                      << " in cc_ss_to_cos";

  return mnmax;
}

int FourierBasisFastPoloidal::sc_cs_to_sin(const std::span<const double> fcSC,
                                           const std::span<const double> fcCS,
                                           std::span<double> m_fcSin,
                                           int n_size, int m_size) const {
  // m = 0: n =  0, 1, ..., ntor --> ntor + 1
  // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
  int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

  absl::c_fill_n(m_fcSin, mnmax, 0);

  int mn = 1;

  int m = 0;
  for (int n = 1; n < n_size + 1; ++n) {
    double basis_norm = 1.0 / (mscale[m] * nscale[n]);

    m_fcSin[mn] = -fcCS[n] / basis_norm;

    mn++;
  }  // n

  for (m = 1; m < m_size; ++m) {
    for (int n = -n_size; n < n_size + 1; ++n) {
      int abs_n = abs(n);
      int sgn_n = signum(n);

      double basis_norm = 1.0 / (mscale[m] * nscale[abs_n]);

      if (abs_n == 0) {
        m_fcSin[mn] = fcSC[m * (n_size + 1) + abs_n] / basis_norm;
      } else {
        double raw_sc = fcSC[m * (n_size + 1) + abs_n];
        double raw_cs = fcCS[m * (n_size + 1) + abs_n];
        m_fcSin[mn] = 0.5 * (raw_sc - sgn_n * raw_cs) / basis_norm;
      }

      mn++;
    }  // n
  }  // m

  CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax
                      << " in sc_cs_to_sin";

  return mnmax;
}

int FourierBasisFastPoloidal::mnIdx(int m, int n) const {
  if (m == 0) {
    CHECK_GE(n, 0) << "no mn index available for n < 0";
    return n;
  } else {
    return (s_.ntor + 1) + (m - 1) * (2 * s_.ntor + 1) + (n + s_.ntor);
  }
}

// number of unique Fourier coefficients for
// m = 0, 1, ..., m_size - 1
// n = -n_size, -(n_size-1), ..., -1, 0, 1, ..., (n_size-1), n_size
int FourierBasisFastPoloidal::mnMax(int m_size, int n_size) const {
  // m = 0: n =  0, 1, ..., ntor --> ntor + 1
  // m > 0: n = -ntor, ..., ntor --> (mpol - 1) * (2 * ntor + 1)
  int mnmax = (n_size + 1) + (m_size - 1) * (2 * n_size + 1);

  return mnmax;
}

void FourierBasisFastPoloidal::computeConversionIndices(std::vector<int>& m_xm,
                                                        std::vector<int>& m_xn,
                                                        int n_size, int m_size,
                                                        int nfp) const {
  const int mnmax = mnMax(m_size, n_size);
  int mn = 0;

  int m = 0;
  for (int n = 0; n < n_size + 1; ++n) {
    m_xm[mn] = m;
    m_xn[mn] = n * nfp;
    mn++;
  }

  for (m = 1; m < m_size; ++m) {
    for (int n = -n_size; n < n_size + 1; ++n) {
      m_xm[mn] = m;
      m_xn[mn] = n * nfp;
      mn++;
    }
  }

  CHECK_EQ(mn, mnmax) << "counting error: mn=" << mn << " should be " << mnmax;
}

}  // namespace vmecpp

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_FOURIER_BASIS_FOURIER_BASIS_H_
#define VMECPP_COMMON_FOURIER_BASIS_FOURIER_BASIS_H_

#include <Eigen/Dense>
#include <span>

#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

// The two data layouts differ only in which flat index a (mode, grid-point)
// pair maps to. VMEC++ iterates its basis arrays with the poloidal (theta)
// coordinate as the fast axis, while Nestor iterates with the toroidal (zeta)
// coordinate fastest, so the two store the identical basis values in transposed
// memory order. Everything else (the trigonometric arithmetic, the scaling
// factors, the combined <-> product basis conversions, the mode-number
// bookkeeping) is layout independent. The layout is therefore a compile-time
// policy supplying only the three flat index formulas; FourierBasis below is
// written once against it.
//
// FastPoloidal (m-major): the poloidal (theta) coordinate is the fast index of
// the poloidal basis arrays, and the poloidal mode m is the slow index of the
// product-basis coefficient arrays. Used by the VMEC++ core solver.
struct FourierBasisFastPoloidalLayout {
  // Poloidal basis arrays, logical shape [num_m][num_l] over (mode m, theta l).
  static int PoloidalBasisIndex(int m, int l, int num_m, int num_l) {
    (void)num_m;
    return m * num_l + l;
  }
  // Toroidal basis arrays, logical shape [num_k][num_n] over (zeta k, mode n).
  static int ToroidalBasisIndex(int n, int k, int num_n, int num_k) {
    (void)num_k;
    return k * num_n + n;
  }
  // Product-basis coefficient arrays, logical shape [m_size][n_size + 1].
  static int ProductIndex(int m, int n, int m_size, int n_size) {
    (void)m_size;
    return m * (n_size + 1) + n;
  }
};

// FastToroidal (n-major): the toroidal (zeta) coordinate is the fast index of
// the toroidal basis arrays, and the toroidal mode n is the slow index of the
// product-basis coefficient arrays. Used by Nestor / the free-boundary code.
struct FourierBasisFastToroidalLayout {
  static int PoloidalBasisIndex(int m, int l, int num_m, int num_l) {
    (void)num_l;
    return l * num_m + m;
  }
  static int ToroidalBasisIndex(int n, int k, int num_n, int num_k) {
    (void)num_n;
    return n * num_k + k;
  }
  static int ProductIndex(int m, int n, int m_size, int n_size) {
    (void)n_size;
    return n * m_size + m;
  }
};

// Fourier basis representation for VMEC++ spectral computations.
//
// This class provides the fundamental spectral basis for VMEC++ computations,
// representing 3D plasma quantities using Fourier decomposition in flux
// coordinates (s, \theta, \zeta) where:
//   s     = normalized toroidal flux (radial coordinate)
//   \theta = poloidal angle
//   \zeta  = toroidal angle = nfp * \phi (field period toroidal angle)
//
// Physical quantities are expanded as:
//   f(s,\theta,\zeta) = \sum_{m,n} f_{mn}(s) * basis_function(m*\theta,
//   n*\zeta)
//
// The Layout template policy fixes the flat memory order of the basis and
// coefficient arrays (see FourierBasisFastPoloidalLayout /
// FourierBasisFastToroidalLayout). The concrete classes are the two type
// aliases at the bottom of this header:
//   FourierBasisFastPoloidal  (theta fast; VMEC++ core solver)
//   FourierBasisFastToroidal  (zeta fast;  Nestor / free boundary)
template <class Layout>
class FourierBasis {
 public:
  explicit FourierBasis(const Sizes* s);

  // ============================================================================
  // FOURIER BASIS SCALING FACTORS
  // ============================================================================

  // [mnyq2+1] Poloidal mode scaling factors: sqrt(2) for m>0, 1.0 for m=0
  // Applied to cos(m*\theta) and sin(m*\theta) basis functions for DFT
  // normalization Enables proper normalization: 1/\pi for m>0 modes, 1/(2\pi)
  // for m=0 mode
  Eigen::VectorXd mscale;

  // [nnyq2+1] Toroidal mode scaling factors: sqrt(2) for n>0, 1.0 for n=0
  // Applied to cos(n*\zeta) and sin(n*\zeta) basis functions for DFT
  // normalization Enables proper normalization: 1/\pi for n>0 modes, 1/(2\pi)
  // for n=0 mode
  Eigen::VectorXd nscale;

  // ============================================================================
  // POLOIDAL BASIS FUNCTIONS
  // ============================================================================
  // Flat index: Layout::PoloidalBasisIndex(m, l, mnyq2+1, nThetaReduced).
  // \theta[l] = 2*\pi*l/nThetaEven for l=0...nThetaReduced-1 (reduced [0,\pi]
  // interval).

  // [nThetaReduced * (mnyq2+1)] Pre-scaled poloidal cosine basis
  // cosmu[idx(m,l)] = cos(m*\theta[l]) * mscale[m]
  Eigen::VectorXd cosmu;

  // [nThetaReduced * (mnyq2+1)] Pre-scaled poloidal sine basis
  // sinmu[idx(m,l)] = sin(m*\theta[l]) * mscale[m]
  Eigen::VectorXd sinmu;

  // [nThetaReduced * (mnyq2+1)] Pre-scaled poloidal cosine derivative
  // cosmum[idx(m,l)] = m * cos(m*\theta[l]) * mscale[m]
  // Used for computing \partial/\partial\theta derivatives in force
  // calculations
  Eigen::VectorXd cosmum;

  // [nThetaReduced * (mnyq2+1)] Pre-scaled poloidal sine derivative
  // sinmum[idx(m,l)] = -m * sin(m*\theta[l]) * mscale[m]
  // Used for computing \partial/\partial\theta derivatives in force
  // calculations
  Eigen::VectorXd sinmum;

  // ============================================================================
  // POLOIDAL BASIS WITH INTEGRATION WEIGHTS
  // ============================================================================

  // [nThetaReduced * (mnyq2+1)] Integration-weighted poloidal cosine basis
  // cosmui[idx(m,l)] = cosmu[idx(m,l)] * intNorm
  // intNorm = 1/(nZeta*(nThetaReduced-1)), with boundary point factor 1/2
  Eigen::VectorXd cosmui;

  // [nThetaReduced * (mnyq2+1)] Integration-weighted poloidal sine basis
  // sinmui[idx(m,l)] = sinmu[idx(m,l)] * intNorm
  Eigen::VectorXd sinmui;

  // [nThetaReduced * (mnyq2+1)] Integration-weighted poloidal cosine derivative
  // cosmumi[idx(m,l)] = cosmum[idx(m,l)] * intNorm
  Eigen::VectorXd cosmumi;

  // [nThetaReduced * (mnyq2+1)] Integration-weighted poloidal sine derivative
  // sinmumi[idx(m,l)] = sinmum[idx(m,l)] * intNorm
  Eigen::VectorXd sinmumi;

  // ============================================================================
  // TOROIDAL BASIS FUNCTIONS
  // ============================================================================
  // Flat index: Layout::ToroidalBasisIndex(n, k, nnyq2+1, nZeta).
  // \zeta[k] = 2*\pi*k/nZeta for k=0...nZeta-1 (full [0,2\pi] interval).

  // [(nnyq2+1) * nZeta] Pre-scaled toroidal cosine basis
  // cosnv[idx(n,k)] = cos(n*\zeta[k]) * nscale[n]
  Eigen::VectorXd cosnv;

  // [(nnyq2+1) * nZeta] Pre-scaled toroidal sine basis
  // sinnv[idx(n,k)] = sin(n*\zeta[k]) * nscale[n]
  Eigen::VectorXd sinnv;

  // [(nnyq2+1) * nZeta] Pre-scaled toroidal cosine derivative with nfp factor
  // cosnvn[idx(n,k)] = n*nfp * cos(n*\zeta[k]) * nscale[n]
  // Factor nfp converts \partial/\partial\zeta to \partial/\partial\phi
  // derivatives
  Eigen::VectorXd cosnvn;

  // [(nnyq2+1) * nZeta] Pre-scaled toroidal sine derivative with nfp factor
  // sinnvn[idx(n,k)] = -n*nfp * sin(n*\zeta[k]) * nscale[n]
  // Factor nfp converts \partial/\partial\zeta to \partial/\partial\phi
  // derivatives
  Eigen::VectorXd sinnvn;

  // ============================================================================
  // FOURIER BASIS CONVERSION FUNCTIONS
  // ============================================================================
  //
  // These functions convert between VMEC++'s two Fourier basis representations
  // using trigonometric identities and pre-computed scaling factors.
  // See docs/fourier_basis_implementation.md for complete mathematical details.
  //
  // Two Fourier basis types:
  // 1. COMBINED BASIS (External): cos(m*\theta - n*\zeta), sin(m*\theta -
  // n*\zeta)
  //    - Used in: wout files, Python API, traditional VMEC format
  //    - Storage: Linear arrays indexed by mode number mn
  //
  // 2. PRODUCT BASIS (Internal): cos(m*\theta)*cos(n*\zeta),
  // sin(m*\theta)*sin(n*\zeta), etc.
  //    - Used in: Internal computations with separable DFT operations
  //    - Storage: 2D arrays indexed by (m,n) via Layout::ProductIndex
  //
  // Mathematical basis function identity:
  // cos(m*\theta - n*\zeta) = cos(m*\theta)*cos(n*\zeta) +
  // sin(m*\theta)*sin(n*\zeta)

  /**
   * Convert coefficients from combined cosine basis to separable product basis.
   *
   * Basis function identity:
   * cos(m*\theta - n*\zeta) = cos(m*\theta)*cos(n*\zeta) +
   * sin(m*\theta)*sin(n*\zeta)
   *
   * This function transforms coefficients for cos(m*\theta - n*\zeta) basis
   * functions into coefficients for the separable product basis
   * cos(m*\theta)*cos(n*\zeta) and sin(m*\theta)*sin(n*\zeta). The
   * transformation accounts for VMEC symmetry where only n >= 0 coefficients
   * are stored.
   *
   * Implementation uses pre-computed scaling factors (mscale, nscale) and
   * handles positive/negative toroidal mode symmetry. Standalone function.
   *
   * Physics context: Converts external coefficient format (wout files) to
   * internal product basis coefficients that enable separable DFT operations.
   *
   * @param fcCos [input] Coefficients for cos(m*\theta - n*\zeta) basis, size
   * mnmax
   * @param m_fcCC [output] Coefficients for cos(m*\theta)*cos(n*\zeta) basis,
   * size m_size*(n_size+1)
   * @param m_fcSS [output] Coefficients for sin(m*\theta)*sin(n*\zeta) basis,
   * size m_size*(n_size+1)
   * @param n_size Toroidal mode range: n in [-n_size, n_size]
   * @param m_size Poloidal mode range: m in [0, m_size-1]
   * @return Total number of modes processed (mnmax)
   */
  int cos_to_cc_ss(const std::span<const double> fcCos,
                   std::span<double> m_fcCC, std::span<double> m_fcSS,
                   int n_size, int m_size) const;

  /**
   * Convert coefficients from combined sine basis to separable product basis.
   *
   * Basis function identity:
   * sin(m*\theta - n*\zeta) = sin(m*\theta)*cos(n*\zeta) -
   * cos(m*\theta)*sin(n*\zeta)
   *
   * This function transforms coefficients for sin(m*\theta - n*\zeta) basis
   * functions into coefficients for the separable product basis
   * sin(m*\theta)*cos(n*\zeta) and cos(m*\theta)*sin(n*\zeta). Enforces
   * sin(0*\theta - 0*\zeta) = 0 constraint.
   *
   * Physics context: Handles sine-parity quantities like Z coordinates (zmns)
   * and \lambda angle functions (lmns coefficients).
   *
   * @param fcSin [input] Coefficients for sin(m*\theta - n*\zeta) basis, size
   * mnmax
   * @param m_fcSC [output] Coefficients for sin(m*\theta)*cos(n*\zeta) basis,
   * size m_size*(n_size+1)
   * @param m_fcCS [output] Coefficients for cos(m*\theta)*sin(n*\zeta) basis,
   * size m_size*(n_size+1)
   * @param n_size Toroidal mode range: n in [-n_size, n_size]
   * @param m_size Poloidal mode range: m in [0, m_size-1]
   * @return Total number of modes processed (mnmax)
   */
  int sin_to_sc_cs(const std::span<const double> fcSin,
                   std::span<double> m_fcSC, std::span<double> m_fcCS,
                   int n_size, int m_size) const;

  /**
   * Convert coefficients from separable product basis back to combined cosine
   * basis.
   *
   * Inverse transformation using basis function identity:
   * cos(m*\theta - n*\zeta) = cos(m*\theta)*cos(n*\zeta) +
   * sin(m*\theta)*sin(n*\zeta)
   *
   * This function reconstructs coefficients for cos(m*\theta - n*\zeta) basis
   * from coefficients of the separable product basis. Handles positive/negative
   * toroidal mode reconstruction and applies inverse scaling factors.
   *
   * Physics context: Converts internal computational results back to external
   * coefficient format for wout files, Python API, and traditional VMEC output.
   *
   * @param fcCC [input] Coefficients for cos(m*\theta)*cos(n*\zeta) basis, size
   * m_size*(n_size+1)
   * @param fcSS [input] Coefficients for sin(m*\theta)*sin(n*\zeta) basis, size
   * m_size*(n_size+1)
   * @param m_fcCos [output] Coefficients for cos(m*\theta - n*\zeta) basis,
   * size mnmax
   * @param n_size Toroidal mode range: n in [-n_size, n_size]
   * @param m_size Poloidal mode range: m in [0, m_size-1]
   * @return Total number of modes processed (mnmax)
   */
  int cc_ss_to_cos(const std::span<const double> fcCC,
                   const std::span<const double> fcSS,
                   std::span<double> m_fcCos, int n_size, int m_size) const;

  /**
   * Convert coefficients from separable product basis back to combined sine
   * basis.
   *
   * Inverse transformation using basis function identity:
   * sin(m*\theta - n*\zeta) = sin(m*\theta)*cos(n*\zeta) -
   * cos(m*\theta)*sin(n*\zeta)
   *
   * This function reconstructs coefficients for sin(m*\theta - n*\zeta) basis
   * from coefficients of the separable product basis. Enforces sin(0*\theta -
   * 0*\zeta) = 0.
   *
   * Physics context: Converts internal results for sine-parity quantities
   * back to external coefficient format for output and analysis.
   *
   * @param fcSC [input] Coefficients for sin(m*\theta)*cos(n*\zeta) basis, size
   * m_size*(n_size+1)
   * @param fcCS [input] Coefficients for cos(m*\theta)*sin(n*\zeta) basis, size
   * m_size*(n_size+1)
   * @param m_fcSin [output] Coefficients for sin(m*\theta - n*\zeta) basis,
   * size mnmax
   * @param n_size Toroidal mode range: n in [-n_size, n_size]
   * @param m_size Poloidal mode range: m in [0, m_size-1]
   * @return Total number of modes processed (mnmax)
   */
  int sc_cs_to_sin(const std::span<const double> fcSC,
                   const std::span<const double> fcCS,
                   std::span<double> m_fcSin, int n_size, int m_size) const;

  int mnIdx(int m, int n) const;
  int mnMax(int m_size, int n_size) const;
  void computeConversionIndices(Eigen::VectorXi& m_xm, Eigen::VectorXi& m_xn,
                                int n_size, int m_size, int nfp) const;

  // ============================================================================
  // MODE NUMBER MAPPING ARRAYS
  // ============================================================================

  // [mnmax] Poloidal mode numbers for standard resolution Fourier coefficients
  // Layout: xm[mn] = poloidal mode number m for the mn-th coefficient
  // Maps linear coefficient index mn to 2D mode (m,n) for spectral operations
  Eigen::VectorXi xm;

  // [mnmax] Toroidal mode numbers for standard resolution Fourier coefficients
  // Layout: xn[mn] = toroidal mode number n*nfp for the mn-th coefficient
  // Factor nfp included to convert from field periods to geometric toroidal
  // modes
  Eigen::VectorXi xn;

  // [mnmax_nyq] Poloidal mode numbers for Nyquist-extended Fourier coefficients
  // Layout: xm_nyq[mn] = poloidal mode number m for the mn-th Nyquist
  // coefficient Extended resolution to avoid aliasing in nonlinear force
  // calculations
  Eigen::VectorXi xm_nyq;

  // [mnmax_nyq] Toroidal mode numbers for Nyquist-extended Fourier coefficients
  // Layout: xn_nyq[mn] = toroidal mode number n*nfp for the mn-th Nyquist
  // coefficient Extended resolution to avoid aliasing in nonlinear force
  // calculations
  Eigen::VectorXi xn_nyq;

 private:
  const Sizes& s_;

  void computeFourierBasis(int nfp);
};

using FourierBasisFastPoloidal = FourierBasis<FourierBasisFastPoloidalLayout>;
using FourierBasisFastToroidal = FourierBasis<FourierBasisFastToroidalLayout>;

}  // namespace vmecpp

#endif  // VMECPP_COMMON_FOURIER_BASIS_FOURIER_BASIS_H_

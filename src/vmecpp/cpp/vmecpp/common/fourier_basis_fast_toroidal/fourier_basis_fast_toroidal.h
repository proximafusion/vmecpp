// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_FOURIER_BASIS_FAST_TOROIDAL_FOURIER_BASIS_FAST_TOROIDAL_H_
#define VMECPP_COMMON_FOURIER_BASIS_FAST_TOROIDAL_FOURIER_BASIS_FAST_TOROIDAL_H_

#include <span>
#include <vector>

#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

// Fourier basis representation optimized for toroidal coordinate operations.
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
// The "FastToroidal" layout stores data with toroidal (\zeta) coordinate as
// the fast (innermost) loop index, optimizing for operations that iterate
// over toroidal modes. This differs from FastPoloidal layout.
//
// NOTE: Nestor has its own implementation of this class because we want to be
// able to use different data layouts between VMEC++ and Nestor.
// TODO(eguiraud) reduce overall code duplication
class FourierBasisFastToroidal {
 public:
  explicit FourierBasisFastToroidal(const Sizes* s);

  // ============================================================================
  // FOURIER BASIS SCALING FACTORS
  // ============================================================================

  // [mnyq2+1] Poloidal mode scaling factors: sqrt(2) for m>0, 1.0 for m=0
  // Applied to cos(m*\theta) and sin(m*\theta) basis functions for DFT
  // normalization Enables proper normalization: 1/\pi for m>0 modes, 1/(2\pi)
  // for m=0 mode
  std::vector<double> mscale;

  // [nnyq2+1] Toroidal mode scaling factors: sqrt(2) for n>0, 1.0 for n=0
  // Applied to cos(n*\zeta) and sin(n*\zeta) basis functions for DFT
  // normalization Enables proper normalization: 1/\pi for n>0 modes, 1/(2\pi)
  // for n=0 mode
  std::vector<double> nscale;

  // ============================================================================
  // POLOIDAL BASIS FUNCTIONS (l-major layout: [l][m])
  // ============================================================================

  // [nThetaReduced * (mnyq2+1)] Pre-scaled poloidal cosine basis
  // Layout: cosmu[l*(mnyq2+1) + m] = cos(m*\theta[l]) * mscale[m]
  // \theta[l] = 2*\pi*l/nThetaEven for l=0...nThetaReduced-1 (reduced [0,\pi]
  // interval)
  std::vector<double> cosmu;

  // [nThetaReduced * (mnyq2+1)] Pre-scaled poloidal sine basis
  // Layout: sinmu[l*(mnyq2+1) + m] = sin(m*\theta[l]) * mscale[m]
  std::vector<double> sinmu;

  // [nThetaReduced * (mnyq2+1)] Pre-scaled poloidal cosine derivative
  // Layout: cosmum[l*(mnyq2+1) + m] = m * cos(m*\theta[l]) * mscale[m]
  // Used for computing \partial/\partial\theta derivatives in force
  // calculations
  std::vector<double> cosmum;

  // [nThetaReduced * (mnyq2+1)] Pre-scaled poloidal sine derivative
  // Layout: sinmum[l*(mnyq2+1) + m] = -m * sin(m*\theta[l]) * mscale[m]
  // Used for computing \partial/\partial\theta derivatives in force
  // calculations
  std::vector<double> sinmum;

  // ============================================================================
  // POLOIDAL BASIS WITH INTEGRATION WEIGHTS
  // ============================================================================

  // [nThetaReduced * (mnyq2+1)] Integration-weighted poloidal cosine basis
  // Layout: cosmui[l*(mnyq2+1) + m] = cosmu[l*(mnyq2+1) + m] * intNorm
  // intNorm = 1/(nZeta*(nThetaReduced-1)), with boundary point factor 1/2
  std::vector<double> cosmui;

  // [nThetaReduced * (mnyq2+1)] Integration-weighted poloidal sine basis
  // Layout: sinmui[l*(mnyq2+1) + m] = sinmu[l*(mnyq2+1) + m] * intNorm
  std::vector<double> sinmui;

  // [nThetaReduced * (mnyq2+1)] Integration-weighted poloidal cosine derivative
  // Layout: cosmumi[l*(mnyq2+1) + m] = cosmum[l*(mnyq2+1) + m] * intNorm
  std::vector<double> cosmumi;

  // [nThetaReduced * (mnyq2+1)] Integration-weighted poloidal sine derivative
  // Layout: sinmumi[l*(mnyq2+1) + m] = sinmum[l*(mnyq2+1) + m] * intNorm
  std::vector<double> sinmumi;

  // ============================================================================
  // TOROIDAL BASIS FUNCTIONS (n-major layout: [n][k])
  // ============================================================================

  // [(nnyq2+1) * nZeta] Pre-scaled toroidal cosine basis
  // Layout: cosnv[n*nZeta + k] = cos(n*\zeta[k]) * nscale[n]
  // \zeta[k] = 2*\pi*k/nZeta for k=0...nZeta-1 (full [0,2\pi] interval)
  std::vector<double> cosnv;

  // [(nnyq2+1) * nZeta] Pre-scaled toroidal sine basis
  // Layout: sinnv[n*nZeta + k] = sin(n*\zeta[k]) * nscale[n]
  std::vector<double> sinnv;

  // [(nnyq2+1) * nZeta] Pre-scaled toroidal cosine derivative with nfp factor
  // Layout: cosnvn[n*nZeta + k] = n*nfp * cos(n*\zeta[k]) * nscale[n]
  // Factor nfp converts \partial/\partial\zeta to \partial/\partial\phi
  // derivatives
  std::vector<double> cosnvn;

  // [(nnyq2+1) * nZeta] Pre-scaled toroidal sine derivative with nfp factor
  // Layout: sinnvn[n*nZeta + k] = -n*nfp * sin(n*\zeta[k]) * nscale[n]
  // Factor nfp converts \partial/\partial\zeta to \partial/\partial\phi
  // derivatives
  std::vector<double> sinnvn;

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
  //    - Storage: 2D arrays indexed by (m,n) separately
  //    - Layout: fcCC[n*m_size + m] (n-major ordering for toroidal class)
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
  void computeConversionIndices(std::vector<int>& m_xm, std::vector<int>& m_xn,
                                int n_size, int m_size, int nfp) const;

  // ============================================================================
  // MODE NUMBER MAPPING ARRAYS
  // ============================================================================

  // [mnmax] Poloidal mode numbers for standard resolution Fourier coefficients
  // Layout: xm[mn] = poloidal mode number m for the mn-th coefficient
  // Maps linear coefficient index mn to 2D mode (m,n) for spectral operations
  std::vector<int> xm;

  // [mnmax] Toroidal mode numbers for standard resolution Fourier coefficients
  // Layout: xn[mn] = toroidal mode number n*nfp for the mn-th coefficient
  // Factor nfp included to convert from field periods to geometric toroidal
  // modes
  std::vector<int> xn;

  // [mnmax_nyq] Poloidal mode numbers for Nyquist-extended Fourier coefficients
  // Layout: xm_nyq[mn] = poloidal mode number m for the mn-th Nyquist
  // coefficient Extended resolution to avoid aliasing in nonlinear force
  // calculations
  std::vector<int> xm_nyq;

  // [mnmax_nyq] Toroidal mode numbers for Nyquist-extended Fourier coefficients
  // Layout: xn_nyq[mn] = toroidal mode number n*nfp for the mn-th Nyquist
  // coefficient Extended resolution to avoid aliasing in nonlinear force
  // calculations
  std::vector<int> xn_nyq;

 private:
  const Sizes& s_;

  void computeFourierBasisFastToroidal(int nfp);
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_FOURIER_BASIS_FAST_TOROIDAL_FOURIER_BASIS_FAST_TOROIDAL_H_

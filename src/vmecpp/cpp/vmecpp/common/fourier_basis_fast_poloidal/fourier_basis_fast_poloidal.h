// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_FOURIER_BASIS_FAST_POLOIDAL_FOURIER_BASIS_FAST_POLOIDAL_H_
#define VMECPP_COMMON_FOURIER_BASIS_FAST_POLOIDAL_FOURIER_BASIS_FAST_POLOIDAL_H_

#include <span>
#include <vector>

#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

// An implementation of FourierBasis that stores its data so that
// the poloidal coordinate is the fast loop index.
// NOTE: Nestor has its own implementation of this class because we want to be
// able to use different data layouts between VMEC++ and Nestor.
class FourierBasisFastPoloidal {
 public:
  explicit FourierBasisFastPoloidal(const Sizes* s);

  std::vector<double> mscale;
  std::vector<double> nscale;

  std::vector<double> cosmu;
  std::vector<double> sinmu;
  std::vector<double> cosmum;
  std::vector<double> sinmum;

  std::vector<double> cosmui;
  std::vector<double> sinmui;
  std::vector<double> cosmumi;
  std::vector<double> sinmumi;

  std::vector<double> cosnv;
  std::vector<double> sinnv;
  std::vector<double> cosnvn;
  std::vector<double> sinnvn;

  // ---------------

  int cos_to_cc_ss(const std::span<const double> fcCos,
                   std::span<double> m_fcCC, std::span<double> m_fcSS,
                   int n_size, int m_size) const;
  int sin_to_sc_cs(const std::span<const double> fcSin,
                   std::span<double> m_fcSC, std::span<double> m_fcCS,
                   int n_size, int m_size) const;

  int cc_ss_to_cos(const std::span<const double> fcCC,
                   const std::span<const double> fcSS,
                   std::span<double> m_fcCos, int n_size, int m_size) const;
  int sc_cs_to_sin(const std::span<const double> fcSC,
                   const std::span<const double> fcCS,
                   std::span<double> m_fcSin, int n_size, int m_size) const;

  int mnIdx(int m, int n) const;
  int mnMax(int m_size, int n_size) const;
  void computeConversionIndices(std::vector<int>& m_xm, std::vector<int>& m_xn,
                                int n_size, int m_size, int nfp) const;

  std::vector<int> xm;
  std::vector<int> xn;

  std::vector<int> xm_nyq;
  std::vector<int> xn_nyq;

 private:
  const Sizes& s_;

  void computeFourierBasisFastPoloidal(int nfp);
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_FOURIER_BASIS_FAST_POLOIDAL_FOURIER_BASIS_FAST_POLOIDAL_H_

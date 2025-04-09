// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_FOURIER_BASIS_FAST_TOROIDAL_FOURIER_BASIS_FAST_TOROIDAL_H_
#define VMECPP_COMMON_FOURIER_BASIS_FAST_TOROIDAL_FOURIER_BASIS_FAST_TOROIDAL_H_

#include <vector>

#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

// An implementation of FourierBasis that stores its data so that the toroidal
// coordinate is the fast loop index.
// NOTE: Nestor has its own implementation of this class because we want to be
// able to use different data layouts between VMEC++ and Nestor.
// TODO(eguiraud) reduce overall code duplication
class FourierBasisFastToroidal {
 public:
  explicit FourierBasisFastToroidal(const Sizes* s);

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

  int cos_to_cc_ss(const std::vector<double>& fcCos,
                   std::vector<double>& m_fcCC, std::vector<double>& m_fcSS,
                   int n_size, int m_size) const;
  int sin_to_sc_cs(const std::vector<double>& fcSin,
                   std::vector<double>& m_fcSC, std::vector<double>& m_fcCS,
                   int n_size, int m_size) const;

  int cc_ss_to_cos(const std::vector<double>& fcCC,
                   const std::vector<double>& fcSS,
                   std::vector<double>& m_fcCos, int n_size, int m_size) const;
  int sc_cs_to_sin(const std::vector<double>& fcSC,
                   const std::vector<double>& fcCS,
                   std::vector<double>& m_fcSin, int n_size, int m_size) const;

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

  void computeFourierBasisFastToroidal(int nfp);
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_FOURIER_BASIS_FAST_TOROIDAL_FOURIER_BASIS_FAST_TOROIDAL_H_

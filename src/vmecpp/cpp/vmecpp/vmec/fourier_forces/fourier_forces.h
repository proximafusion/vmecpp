// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_
#define VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_

#include <span>
#include <vector>

#include "vmecpp/vmec/fourier_coefficients/fourier_coefficients.h"

namespace vmecpp {

class FourierForces : public FourierCoeffs {
 public:
  FourierForces(const Sizes* s, const RadialPartitioning* r, int ns);

  void zeroZForceForM1();
  void residuals(std::vector<double>& fRes, bool includeEdgeRZ) const;

  // appropriately-named variables for the data in FourierCoeffs
  std::span<double> frcc;
  std::span<double> frss;
  std::span<double> frsc;
  std::span<double> frcs;

  std::span<double> fzsc;
  std::span<double> fzcs;
  std::span<double> fzcc;
  std::span<double> fzss;

  std::span<double> flsc;
  std::span<double> flcs;
  std::span<double> flcc;
  std::span<double> flss;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_

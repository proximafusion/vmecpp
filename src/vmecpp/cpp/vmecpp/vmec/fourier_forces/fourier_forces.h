// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_
#define VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_

#include <Eigen/Dense>
#include <span>

#include "vmecpp/common/util/real_type.h"
#include "vmecpp/vmec/fourier_coefficients/fourier_coefficients.h"

namespace vmecpp {

class FourierForces : public FourierCoeffs {
 public:
  FourierForces(const Sizes* s, const RadialPartitioning* r, int ns);
  FourierForces(const FourierForces& other);
  FourierForces& operator=(const FourierForces& other);
  FourierForces(FourierForces&& other) noexcept;
  FourierForces& operator=(FourierForces&& other) noexcept;

  void zeroZForceForM1();
  void residuals(Eigen::Matrix<real_t, Eigen::Dynamic, 1>& fRes,
                 bool includeEdgeRZ) const;

  // appropriately-named variables for the data in FourierCoeffs
  std::span<real_t> frcc;
  std::span<real_t> frss;
  std::span<real_t> frsc;
  std::span<real_t> frcs;

  std::span<real_t> fzsc;
  std::span<real_t> fzcs;
  std::span<real_t> fzcc;
  std::span<real_t> fzss;

  std::span<real_t> flsc;
  std::span<real_t> flcs;
  std::span<real_t> flcc;
  std::span<real_t> flss;

 private:
  void BindSpans();
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_

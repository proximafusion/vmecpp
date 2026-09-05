// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_
#define VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_

#include <Eigen/Dense>
#include <span>

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
  // Writes each surface's squared force residuals into row jF of
  // `m_surface_rows` (`row_stride` doubles per row): R, Z and lambda in
  // columns 0, 1 and 2.
  void residuals(double* m_surface_rows, int row_stride,
                 bool includeEdgeRZ) const;

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

 private:
  void BindSpans();
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_FORCES_FOURIER_FORCES_H_

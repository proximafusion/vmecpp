// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_FOURIER_VELOCITY_FOURIER_VELOCITY_H_
#define VMECPP_VMEC_FOURIER_VELOCITY_FOURIER_VELOCITY_H_

#include <span>

#include "vmecpp/vmec/fourier_coefficients/fourier_coefficients.h"

namespace vmecpp {

class FourierVelocity : public FourierCoeffs {
 public:
  FourierVelocity(const Sizes* s, const RadialPartitioning* r, int ns);
  FourierVelocity(const FourierVelocity& other);
  FourierVelocity& operator=(const FourierVelocity& other);
  FourierVelocity(FourierVelocity&& other) noexcept;
  FourierVelocity& operator=(FourierVelocity&& other) noexcept;

  // appropriately-named variables for the data in FourierCoeffs
  std::span<double> vrcc;
  std::span<double> vrss;
  std::span<double> vrsc;
  std::span<double> vrcs;

  std::span<double> vzsc;
  std::span<double> vzcs;
  std::span<double> vzcc;
  std::span<double> vzss;

  std::span<double> vlsc;
  std::span<double> vlcs;
  std::span<double> vlcc;
  std::span<double> vlss;

 private:
  void BindSpans();
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_VELOCITY_FOURIER_VELOCITY_H_

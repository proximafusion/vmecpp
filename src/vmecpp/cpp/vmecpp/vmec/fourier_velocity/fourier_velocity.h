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
  std::span<real_t> vrcc;
  std::span<real_t> vrss;
  std::span<real_t> vrsc;
  std::span<real_t> vrcs;

  std::span<real_t> vzsc;
  std::span<real_t> vzcs;
  std::span<real_t> vzcc;
  std::span<real_t> vzss;

  std::span<real_t> vlsc;
  std::span<real_t> vlcs;
  std::span<real_t> vlcc;
  std::span<real_t> vlss;

 private:
  void BindSpans();
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_VELOCITY_FOURIER_VELOCITY_H_

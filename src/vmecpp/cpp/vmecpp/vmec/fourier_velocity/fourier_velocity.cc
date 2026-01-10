// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_velocity/fourier_velocity.h"

#include <utility>

namespace vmecpp {

FourierVelocity::FourierVelocity(const Sizes* s, const RadialPartitioning* r,
                                 int ns)
    : FourierCoeffs(s, r, r->nsMinF, r->nsMaxF, ns) {
  BindSpans();
}

FourierVelocity::FourierVelocity(const FourierVelocity& other)
    : FourierCoeffs(other) {
  BindSpans();
}

FourierVelocity& FourierVelocity::operator=(const FourierVelocity& other) {
  if (this != &other) {
    FourierCoeffs::operator=(other);
    BindSpans();
  }
  return *this;
}

FourierVelocity::FourierVelocity(FourierVelocity&& other) noexcept
    : FourierCoeffs(std::move(other)) {
  BindSpans();
}

FourierVelocity& FourierVelocity::operator=(FourierVelocity&& other) noexcept {
  if (this != &other) {
    FourierCoeffs::operator=(std::move(other));
    BindSpans();
  }
  return *this;
}

void FourierVelocity::BindSpans() {
  vrcc = std::span<double>(rcc);
  vrss = std::span<double>(rss);
  vrsc = std::span<double>(rsc);
  vrcs = std::span<double>(rcs);
  vzsc = std::span<double>(zsc);
  vzcs = std::span<double>(zcs);
  vzcc = std::span<double>(zcc);
  vzss = std::span<double>(zss);
  vlsc = std::span<double>(lsc);
  vlcs = std::span<double>(lcs);
  vlcc = std::span<double>(lcc);
  vlss = std::span<double>(lss);
}

}  // namespace vmecpp

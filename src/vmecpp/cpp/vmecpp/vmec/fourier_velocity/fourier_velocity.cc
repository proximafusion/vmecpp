// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_velocity/fourier_velocity.h"

#include <utility>

namespace vmecpp {

FourierVelocity::FourierVelocity(const Sizes* s, const RadialPartitioning* r,
                                 int ns)
    : FourierCoeffs(s, r, r->nsMinF, r->nsMaxF, ns),
      vrcc(rcc),
      vrss(rss),
      vrsc(rsc),
      vrcs(rcs),
      vzsc(zsc),
      vzcs(zcs),
      vzcc(zcc),
      vzss(zss),
      vlsc(lsc),
      vlcs(lcs),
      vlcc(lcc),
      vlss(lss) {}

FourierVelocity::FourierVelocity(const FourierVelocity& other)
    : FourierCoeffs(other),
      vrcc(rcc),
      vrss(rss),
      vrsc(rsc),
      vrcs(rcs),
      vzsc(zsc),
      vzcs(zcs),
      vzcc(zcc),
      vzss(zss),
      vlsc(lsc),
      vlcs(lcs),
      vlcc(lcc),
      vlss(lss) {}

void FourierVelocity::BindSpans() {
  vrcc = rcc;
  vrss = rss;
  vrsc = rsc;
  vrcs = rcs;
  vzsc = zsc;
  vzcs = zcs;
  vzcc = zcc;
  vzss = zss;
  vlsc = lsc;
  vlcs = lcs;
  vlcc = lcc;
  vlss = lss;
}

FourierVelocity& FourierVelocity::operator=(const FourierVelocity& other) {
  if (this != &other) {
    FourierCoeffs::operator=(other);
    BindSpans();
  }
  return *this;
}

FourierVelocity::FourierVelocity(FourierVelocity&& other) noexcept
    : FourierCoeffs(std::move(other)),
      vrcc(rcc),
      vrss(rss),
      vrsc(rsc),
      vrcs(rcs),
      vzsc(zsc),
      vzcs(zcs),
      vzcc(zcc),
      vzss(zss),
      vlsc(lsc),
      vlcs(lcs),
      vlcc(lcc),
      vlss(lss) {}

FourierVelocity& FourierVelocity::operator=(FourierVelocity&& other) noexcept {
  if (this != &other) {
    FourierCoeffs::operator=(std::move(other));
    BindSpans();
  }
  return *this;
}

}  // namespace vmecpp

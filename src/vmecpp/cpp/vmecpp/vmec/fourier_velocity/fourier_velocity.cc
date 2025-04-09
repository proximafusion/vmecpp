// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_velocity/fourier_velocity.h"

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

}  // namespace vmecpp

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_velocity/fourier_velocity.h"

namespace vmecpp {

FourierVelocity::FourierVelocity(const Sizes* s, const RadialPartitioning* r,
                                 int ns)
    : FourierCoeffs(s, r, r->nsMinF, r->nsMaxF, ns),
      vrcc(rcc.data(), rcc.size()),
      vrss(rss.data(), rss.size()),
      vrsc(rsc.data(), rsc.size()),
      vrcs(rcs.data(), rcs.size()),
      vzsc(zsc.data(), zsc.size()),
      vzcs(zcs.data(), zcs.size()),
      vzcc(zcc.data(), zcc.size()),
      vzss(zss.data(), zss.size()),
      vlsc(lsc.data(), lsc.size()),
      vlcs(lcs.data(), lcs.size()),
      vlcc(lcc.data(), lcc.size()),
      vlss(lss.data(), lss.size()) {}

}  // namespace vmecpp

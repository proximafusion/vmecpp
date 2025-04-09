// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

vmecpp::VmecConstants::VmecConstants() { reset(); }

void vmecpp::VmecConstants::reset() {
  rmsPhiP = 0.;
  lamscale = 0.;
}

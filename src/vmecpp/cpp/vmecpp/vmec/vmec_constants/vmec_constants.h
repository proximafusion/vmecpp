// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_VMEC_CONSTANTS_VMEC_CONSTANTS_H_
#define VMECPP_VMEC_VMEC_CONSTANTS_VMEC_CONSTANTS_H_

namespace vmecpp {

// Quantities that are set during the application of the forward model.
// Can be set e.g. at VMEC initialization.
struct VmecConstants {
  // for RadialProfiles
  double rmsPhiP;

  // RMS(phi') for normalizing lambda
  double lamscale;

  VmecConstants();

  void reset();
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_VMEC_CONSTANTS_VMEC_CONSTANTS_H_

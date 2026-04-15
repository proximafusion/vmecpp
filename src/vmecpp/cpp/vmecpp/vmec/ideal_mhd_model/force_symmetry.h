// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_FORCE_SYMMETRY_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_FORCE_SYMMETRY_H_

#include <span>

#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

enum class ReflectionParity {
  kStandard,
  kReversed,
};

void DecomposeForceComponent(const Sizes& sizes, std::span<const double> full,
                             ReflectionParity parity, std::span<double> sym,
                             std::span<double> asym);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_FORCE_SYMMETRY_H_

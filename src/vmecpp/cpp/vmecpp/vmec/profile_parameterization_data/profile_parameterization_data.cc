// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/profile_parameterization_data/profile_parameterization_data.h"

#include <string>

namespace vmecpp {

ProfileParameterizationData::ProfileParameterizationData(
    const std::string& name, bool allowedForPres, bool allowedForCurr,
    bool allowedForIota, bool needsSplineData)
    : name_(name),
      needsSplineData_(needsSplineData),
      allowedFor_({.pres = allowedForPres,
                   .curr = allowedForCurr,
                   .iota = allowedForIota}) {}

const std::string& ProfileParameterizationData::Name() { return name_; }

bool ProfileParameterizationData::NeedsSplineData() const {
  return needsSplineData_;
}

AllowedFor ProfileParameterizationData::IsAllowedFor() { return allowedFor_; }

}  // namespace vmecpp

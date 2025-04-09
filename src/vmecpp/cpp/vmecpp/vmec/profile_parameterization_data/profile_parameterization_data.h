// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_PROFILE_PARAMETERIZATION_DATA_PROFILE_PARAMETERIZATION_DATA_H_
#define VMECPP_VMEC_PROFILE_PARAMETERIZATION_DATA_PROFILE_PARAMETERIZATION_DATA_H_

#include <cstdint>
#include <cstdlib>
#include <string>

namespace vmecpp {

// number of profile parameterizations
#define NUM_PARAM 23

enum class ProfileType : std::uint8_t { PRESSURE, CURRENT, IOTA };

struct AllowedFor {
  bool pres;
  bool curr;
  bool iota;
};

class ProfileParameterizationData {
 public:
  ProfileParameterizationData(const std::string& name, bool allowedForPres,
                              bool allowedForCurr, bool allowedForIota,
                              bool needsSplineData);

  const std::string& Name();
  bool NeedsSplineData() const;
  AllowedFor IsAllowedFor();

 private:
  const std::string name_;
  bool needsSplineData_;
  AllowedFor allowedFor_;
};

enum class ProfileParameterization : std::uint8_t {
  INVALID_PARAM = 0,
  POWER_SERIES = 1,
  POWER_SERIES_I = 2,
  GAUSS_TRUNC = 3,
  SUM_ATAN = 4,
  TWO_LORENTZ = 5,
  TWO_POWER = 6,
  TWO_POWER_GS = 7,
  AKIMA_SPLINE = 8,
  AKIMA_SPLINE_I = 9,
  AKIMA_SPLINE_IP = 10,
  CUBIC_SPLINE = 11,
  CUBIC_SPLINE_I = 12,
  CUBIC_SPLINE_IP = 13,
  PEDESTAL = 14,
  RATIONAL = 15,
  LINE_SEGMENT = 16,
  LINE_SEGMENT_I = 17,
  LINE_SEGMENT_IP = 18,
  NICE_QUADRATIC = 19,
  SUM_COSSQ_S = 20,
  SUM_COSSQ_SQRTS = 21,
  SUM_COSSQ_S_FREE = 22
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_PROFILE_PARAMETERIZATION_DATA_PROFILE_PARAMETERIZATION_DATA_H_

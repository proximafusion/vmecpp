// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/profile_parameterization_data/profile_parameterization_data.h"

#include <string>
#include <vector>

namespace vmecpp {
namespace {

std::vector<ProfileParameterizationData> BuildProfileParameterizations() {
  // Entries are in ProfileParameterization order; findParameterization relies
  // on the index matching the enumerator.
  //
  //                       current | iota | pressure
  //                       --------+------+---------
  // POWER_SERIES          I-prime |   X  |     X
  // POWER_SERIES_I        I       |      |
  // GAUSS_TRUNC           I-prime |      |     X
  // SUM_ATAN              I       |   X  |
  // TWO_LORENTZ                   |      |     X
  // TWO_POWER             I-prime |      |     X
  // TWO_POWER_GS          I-prime |      |     X
  // AKIMA_SPLINE                  |   X  |     X
  // AKIMA_SPLINE_I        I       |      |
  // AKIMA_SPLINE_IP       I-prime |      |
  // CUBIC_SPLINE                  |   X  |     X
  // CUBIC_SPLINE_I        I       |      |
  // CUBIC_SPLINE_IP       I-prime |      |
  // PEDESTAL              I       |      |     X
  // RATIONAL              I       |   X  |     X
  // LINE_SEGMENT                  |   X  |     X
  // LINE_SEGMENT_I        I       |      |
  // LINE_SEGMENT_IP       I-prime |      |
  // NICE_QUADRATIC                |   X  |
  std::vector<ProfileParameterizationData> all;
  all.reserve(NUM_PARAM);
  all.emplace_back("---invalid---", /*allowedForPres=*/false,
                   /*allowedForCurr*/ false, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  all.emplace_back("power_series", /*allowedForPres=*/true,
                   /*allowedForCurr*/ true, /*allowedForIota*/ true,
                   /*needsSplineData*/ false);
  all.emplace_back("power_series_i", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  all.emplace_back("gauss_trunc", /*allowedForPres=*/true,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  all.emplace_back("sum_atan", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ true,
                   /*needsSplineData*/ false);
  all.emplace_back("two_lorentz", /*allowedForPres=*/true,
                   /*allowedForCurr*/ false, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  all.emplace_back("two_power", /*allowedForPres=*/true,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  all.emplace_back("two_power_gs", /*allowedForPres=*/true,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  all.emplace_back("akima_spline", /*allowedForPres=*/true,
                   /*allowedForCurr*/ false, /*allowedForIota*/ true,
                   /*needsSplineData*/ true);
  all.emplace_back("akima_spline_i", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ true);
  all.emplace_back("akima_spline_ip", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ true);
  all.emplace_back("cubic_spline", /*allowedForPres=*/true,
                   /*allowedForCurr*/ false, /*allowedForIota*/ true,
                   /*needsSplineData*/ true);
  all.emplace_back("cubic_spline_i", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ true);
  all.emplace_back("cubic_spline_ip", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ true);
  all.emplace_back("pedestal", /*allowedForPres=*/true,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  all.emplace_back("rational", /*allowedForPres=*/true,
                   /*allowedForCurr*/ true, /*allowedForIota*/ true,
                   /*needsSplineData*/ false);
  all.emplace_back("line_segment", /*allowedForPres=*/true,
                   /*allowedForCurr*/ false, /*allowedForIota*/ true,
                   /*needsSplineData*/ true);
  all.emplace_back("line_segment_i", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ true);
  all.emplace_back("line_segment_ip", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ true);
  all.emplace_back("nice_quadratic", /*allowedForPres=*/false,
                   /*allowedForCurr*/ false, /*allowedForIota*/ true,
                   /*needsSplineData*/ false);
  all.emplace_back("sum_cossq_s", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  all.emplace_back("sum_cossq_sqrts", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  all.emplace_back("sum_cossq_s_free", /*allowedForPres=*/false,
                   /*allowedForCurr*/ true, /*allowedForIota*/ false,
                   /*needsSplineData*/ false);
  return all;
}

}  // namespace

ProfileParameterizationData::ProfileParameterizationData(
    const std::string& name, bool allowedForPres, bool allowedForCurr,
    bool allowedForIota, bool needsSplineData)
    : name_(name),
      needsSplineData_(needsSplineData),
      allowedFor_({.pres = allowedForPres,
                   .curr = allowedForCurr,
                   .iota = allowedForIota}) {}

const std::string& ProfileParameterizationData::Name() const { return name_; }

bool ProfileParameterizationData::NeedsSplineData() const {
  return needsSplineData_;
}

AllowedFor ProfileParameterizationData::IsAllowedFor() const {
  return allowedFor_;
}

const std::vector<ProfileParameterizationData>& AllProfileParameterizations() {
  static const std::vector<ProfileParameterizationData>* const all =
      new std::vector<ProfileParameterizationData>(
          BuildProfileParameterizations());
  return *all;
}

const ProfileParameterizationData* FindProfileParameterization(
    const std::string& name) {
  for (const ProfileParameterizationData& entry :
       AllProfileParameterizations()) {
    if (entry.Name() == name) {
      return &entry;
    }
  }
  return nullptr;
}

bool IsProfileParameterizationAllowedFor(const std::string& name,
                                         ProfileType type) {
  const ProfileParameterizationData* const entry =
      FindProfileParameterization(name);
  if (entry == nullptr) {
    return false;
  }
  const AllowedFor allowed = entry->IsAllowedFor();
  switch (type) {
    case ProfileType::PRESSURE:
      return allowed.pres;
    case ProfileType::CURRENT:
      return allowed.curr;
    case ProfileType::IOTA:
      return allowed.iota;
  }
  return false;
}

}  // namespace vmecpp

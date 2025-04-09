// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/vmec_indata/boundary_from_json.h"

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "nlohmann/json.hpp"

namespace vmecpp {

using nlohmann::json;

using json_io::JsonReadBool;
using json_io::JsonReadDouble;
using json_io::JsonReadInt;
using json_io::JsonReadString;
using json_io::JsonReadVectorDouble;
using json_io::JsonReadVectorInt;

absl::StatusOr<std::optional<std::vector<BoundaryCoefficient> > >
BoundaryCoefficient::FromJson(const json& j, const std::string& name) {
  if (!j.contains(name)) {
    // not present --> skip
    return std::nullopt;
  }

  if (!j[name].is_array()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("JSON element '%s' is not an array", name));
  }

  std::vector<BoundaryCoefficient> entries;
  int i = 0;
  for (const json& entry : j[name]) {
    if (!entry.is_object()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("JSON entry '%s'[%d] is not an object", name, i));
    }

    auto m = JsonReadInt(entry, "m");
    if (!m.ok()) {
      return m.status();
    }
    if (!m->has_value()) {
      // skip entries where "m" is not specified
      continue;
    }

    auto n = JsonReadInt(entry, "n");
    if (!n.ok()) {
      return n.status();
    }
    if (!n->has_value()) {
      // skip entries where "n" is not specified
      continue;
    }

    auto value = JsonReadDouble(entry, "value");
    if (!value.ok()) {
      return value.status();
    }
    if (!value->has_value()) {
      // skip entries where "value" is not specified
      continue;
    }

    BoundaryCoefficient boundary_coefficient = {
        /*m=*/m->value(), /*n=*/n->value(), /*value=*/value->value()};
    entries.push_back(boundary_coefficient);

    i++;
  }

  return entries;
}  // JsonReadBoundary

std::vector<double> BoundaryFromJson(const nlohmann::json& json,
                                     const std::string& key, int mpol,
                                     int ntor) {
  std::vector<double> coeffs(mpol * (2 * ntor + 1));

  const auto maybe_entries = vmecpp::BoundaryCoefficient::FromJson(json, key);
  CHECK_OK(maybe_entries);
  CHECK(maybe_entries->has_value());

  const std::vector<vmecpp::BoundaryCoefficient> entries =
      maybe_entries->value();
  for (const vmecpp::BoundaryCoefficient& entry : entries) {
    // Fortran order along n: -ntor, -ntor+1, ..., -1, 0, 1, ..., ntor-1, ntor
    if (entry.m < 0 || entry.m >= mpol || entry.n < -ntor || entry.n > ntor) {
      // invalid indices for boundary coefficients in the json input are ignore
      continue;
    }

    const int index_along_n = ntor + entry.n;
    const int flat_index = entry.m * (2 * ntor + 1) + index_along_n;
    coeffs[flat_index] = entry.value;
  }

  return coeffs;
}

}  // namespace vmecpp

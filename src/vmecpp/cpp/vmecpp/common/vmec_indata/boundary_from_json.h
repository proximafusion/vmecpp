// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_VMEC_INDATA_BOUNDARY_FROM_JSON_H_
#define VMECPP_COMMON_VMEC_INDATA_BOUNDARY_FROM_JSON_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "util/json_io/json_io.h"

namespace vmecpp {

// A single Fourier coefficient describing, e.g., the plasma boundary in a VMEC
// input file. This is expected in JSON as {"n": 4, "m": 5, "value": 1.23}.
struct BoundaryCoefficient {
  // poloidal mode number
  int m;

  // toroidal mode number
  int n;

  // Fourier coefficient of boundary
  double value;

  // Try to read a VMEC boundary description from the given JSON data.
  // If the variable is not present in the given JSON object, status will be ok
  // and optional will be not populated. If the variable was found, but was not
  // the correct type, the status will be not ok.
  static absl::StatusOr<std::optional<std::vector<BoundaryCoefficient> > >
  FromJson(const nlohmann::json& j, const std::string& name);
};  // BoundaryCoefficient

// Read the desired boundary coefficient array (`key` = `rbc`,
// `zbs`, `rbs`, `zbc`)
// from the given JSON obect and put the populated coefficients
// in fast-toroidal order into a vector.
// The indexing is `m * (2 * ntor + 1) + (ntor + n)`
// and the size of the returned vector is `mpol * (2 * ntor + 1)`.
std::vector<double> BoundaryFromJson(const nlohmann::json& json,
                                     const std::string& key, int mpol,
                                     int ntor);

}  // namespace vmecpp

#endif  // VMECPP_COMMON_VMEC_INDATA_BOUNDARY_FROM_JSON_H_

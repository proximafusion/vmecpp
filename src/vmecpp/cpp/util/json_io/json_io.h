// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef UTIL_JSON_IO_JSON_IO_H_
#define UTIL_JSON_IO_JSON_IO_H_

#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "nlohmann/json.hpp"

namespace json_io {

// Try to read a bool from the given JSON data.
// If the variable is not present in the given JSON object, status will be ok
// and optional will be not populated. If the variable was found, but was not
// the correct type, the status will be not ok.
absl::StatusOr<std::optional<bool> > JsonReadBool(const nlohmann::json& j,
                                                  const std::string& name);

// Try to read an int from the given JSON data.
// If the variable is not present in the given JSON object, status will be ok
// and optional will be not populated. If the variable was found, but was not
// the correct type, the status will be not ok.
absl::StatusOr<std::optional<int> > JsonReadInt(const nlohmann::json& j,
                                                const std::string& name);

// Try to read a double from the given JSON data.
// If the variable is not present in the given JSON object, status will be ok
// and optional will be not populated. If the variable was found, but was not
// the correct type, the status will be not ok.
absl::StatusOr<std::optional<double> > JsonReadDouble(const nlohmann::json& j,
                                                      const std::string& name);

// Try to read a string from the given JSON data.
// If the variable is not present in the given JSON object, status will be ok
// and optional will be not populated. If the variable was found, but was not
// the correct type, the status will be not ok.
absl::StatusOr<std::optional<std::string> > JsonReadString(
    const nlohmann::json& j, const std::string& name);

// Try to read a vector of integers from the given JSON data.
// If the variable is not present in the given JSON object, status will be ok
// and optional will be not populated. If the variable was found, but was not
// the correct type, the status will be not ok.
absl::StatusOr<std::optional<std::vector<int> > > JsonReadVectorInt(
    const nlohmann::json& j, const std::string& name);

// Try to read a vector of doubles from the given JSON data.
// If the variable is not present in the given JSON object, status will be ok
// and optional will be not populated. If the variable was found, but was not
// the correct type, the status will be not ok.
absl::StatusOr<std::optional<std::vector<double> > > JsonReadVectorDouble(
    const nlohmann::json& j, const std::string& name);

}  // namespace json_io

#endif  // UTIL_JSON_IO_JSON_IO_H_

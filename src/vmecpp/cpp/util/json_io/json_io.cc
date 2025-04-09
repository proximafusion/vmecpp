// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/json_io/json_io.h"

#include <string>
#include <vector>

#include "absl/strings/str_format.h"

namespace {
using nlohmann::json;
}

namespace json_io {

absl::StatusOr<std::optional<bool> > JsonReadBool(const json& j,
                                                  const std::string& name) {
  if (!j.contains(name)) {
    // not present --> skip
    return std::nullopt;
  }

  if (!j[name].is_boolean()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("JSON element '%s' is not a boolean", name));
  }

  return j[name];
}  // JsonReadBool

absl::StatusOr<std::optional<int> > JsonReadInt(const json& j,
                                                const std::string& name) {
  if (!j.contains(name)) {
    // not present --> skip
    return std::nullopt;
  }

  if (!j[name].is_number_integer()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("JSON element '%s' is not an integer", name));
  }

  return j[name];
}  // JsonReadInt

absl::StatusOr<std::optional<double> > JsonReadDouble(const json& j,
                                                      const std::string& name) {
  if (!j.contains(name)) {
    // not present --> skip
    return std::nullopt;
  }

  if (!j[name].is_number()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("JSON element '%s' is not a number", name));
  }

  return j[name];
}  // JsonReadDouble

absl::StatusOr<std::optional<std::string> > JsonReadString(
    const json& j, const std::string& name) {
  if (!j.contains(name)) {
    // not present --> skip
    return std::nullopt;
  }

  if (!j[name].is_string()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("JSON element '%s' is not a string", name));
  }

  return j[name];
}  // JsonReadString

absl::StatusOr<std::optional<std::vector<int> > > JsonReadVectorInt(
    const json& j, const std::string& name) {
  if (!j.contains(name)) {
    // not present --> skip
    return std::nullopt;
  }

  if (!j[name].is_array()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("JSON element '%s' is not an array", name));
  }

  std::vector<int> entries;
  int i = 0;
  for (const auto& entry : j[name]) {
    if (entry.is_number_integer()) {
      entries.push_back(entry);
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("JSON entry '%s'[%d] is not an integer", name, i));
    }
    i++;
  }

  return entries;
}  // JsonReadVectorInt

absl::StatusOr<std::optional<std::vector<double> > > JsonReadVectorDouble(
    const json& j, const std::string& name) {
  if (!j.contains(name)) {
    // not present --> skip
    return std::nullopt;
  }

  if (!j[name].is_array()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("JSON element '%s' is not an array", name));
  }

  std::vector<double> entries;
  int i = 0;
  for (const auto& entry : j[name]) {
    if (entry.is_number()) {
      entries.push_back(entry);
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("JSON entry '%s'[%d] is not a number", name, i));
    }
    i++;
  }

  return entries;
}  // JsonReadVectorDouble

}  // namespace json_io

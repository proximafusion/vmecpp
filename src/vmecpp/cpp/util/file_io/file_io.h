// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef UTIL_FILE_IO_FILE_IO_H_
#define UTIL_FILE_IO_FILE_IO_H_

#include <filesystem>
#include <string>

#include "absl/status/statusor.h"

namespace file_io {

// read a file into a string
absl::StatusOr<std::string> ReadFile(const std::filesystem::path& filename);

// write a (text) file from a string
absl::Status WriteFile(const std::filesystem::path& filename,
                       const std::string& contents);

}  // namespace file_io

#endif  // UTIL_FILE_IO_FILE_IO_H_

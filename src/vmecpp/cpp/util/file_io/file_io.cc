// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "util/file_io/file_io.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace fs = std::filesystem;

absl::StatusOr<std::string> file_io::ReadFile(const fs::path& filename) {
  if (!fs::exists(filename)) {
    return absl::NotFoundError("File " + filename.string() + " not found.");
  }

  std::ifstream ifs(filename.c_str());
  if (!ifs.is_open()) {
    return absl::PermissionDeniedError("File " + filename.string() +
                                       " could not be opened for reading.");
  }

  std::string contents(std::istreambuf_iterator<char>{ifs}, {});
  return contents;
}

absl::Status file_io::WriteFile(const std::filesystem::path& filename,
                                const std::string& contents) {
  std::ofstream file_stream(filename.string());
  if (!file_stream.is_open()) {
    return absl::PermissionDeniedError("File " + filename.string() +
                                       " could not be opened for writing.");
  }

  file_stream << contents;

  file_stream.flush();
  file_stream.close();

  return absl::OkStatus();
}

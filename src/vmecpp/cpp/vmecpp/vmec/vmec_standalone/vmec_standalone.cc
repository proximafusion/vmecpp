// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <iostream>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/strip.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

using vmecpp::OutputQuantities;
using vmecpp::VmecINDATA;

using file_io::ReadFile;

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: " << argv[0] << " input_file.json [n_max_threads]\n";
    return 1;
  }

  // read input file provided on command line
  absl::StatusOr<std::string> indata_json = ReadFile(argv[1]);
  CHECK_OK(indata_json) << "Could not read input file '" << argv[1]
                        << "': " << indata_json.status();

  absl::StatusOr<VmecINDATA> vmec_indata = VmecINDATA::FromJson(*indata_json);
  CHECK_OK(vmec_indata) << "Could not parse input file '" << argv[1]
                        << "' into VmecINDATA: " << vmec_indata.status();

  std::optional<int> max_threads = std::nullopt;
  if (argc == 3) {
    max_threads = std::atoi(argv[2]);
  }

  const absl::StatusOr<OutputQuantities> out =
      vmecpp::run(*vmec_indata, /*initial_state=*/std::nullopt,
                  /*max_threads=*/max_threads);

  CHECK_OK(out) << "Error encountered during the VMEC++ run: " << out.status();

  const std::string out_path =
      absl::StrCat(absl::StripSuffix(argv[1], ".json"), ".out.h5");
  const absl::Status status = out->Save(out_path);
  CHECK_OK(status) << "Error encountered writing the output file '" << out_path
                   << "': " << status;

  return 0;
}

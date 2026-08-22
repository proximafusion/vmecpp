// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <filesystem>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "gmock/gmock.h"  // ElementsAreArray
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

using ::testing::ElementsAreArray;
using ::testing::TestWithParam;
using ::testing::Values;

using file_io::ReadFile;
using magnetics::ImportMagneticConfigurationFromCoilsFile;
using makegrid::ImportMakegridParametersFromFile;
using vmecpp::RadialPartitioning;
using vmecpp::Sizes;
using vmecpp::Vmec;
using vmecpp::VmecCheckpoint;
using vmecpp::VmecINDATA;
namespace fs = std::filesystem;

TEST(TestVmec, CheckInMemoryMgrid) {
  // test the constructor that takes an in-memory mgrid

  // LOAD INDATA FILE
  const std::string filename = "vmecpp/test_data/cth_like_free_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  VmecINDATA& indata = maybe_indata.value();

  // LOAD COILS FILE
  const std::string coils_filename = "vmecpp/test_data/coils.cth_like";
  const auto maybe_magnetic_configuration =
      magnetics::ImportMagneticConfigurationFromCoilsFile(coils_filename);
  ASSERT_TRUE(maybe_magnetic_configuration.ok());
  const auto& magnetic_configuration = *maybe_magnetic_configuration;

  // load makegrid params
  const auto maybe_makegrid_params = makegrid::ImportMakegridParametersFromFile(
      "vmecpp/test_data/makegrid_parameters_cth_like.json");
  ASSERT_TRUE(maybe_makegrid_params.ok());
  const auto& makegrid_params = *maybe_makegrid_params;

  // compute magnetic field response tables
  const auto maybe_magnetic_response_table =
      makegrid::ComputeMagneticFieldResponseTable(makegrid_params,
                                                  magnetic_configuration);
  ASSERT_TRUE(maybe_magnetic_response_table.ok());
  const auto& magnetic_response_table = *maybe_magnetic_response_table;

  // RUNS
  // using the mgrid file on disk
  // NOTE: we assume the mgrid file was produced with our C++ version of
  // makegrid. If it's re-generated using a different makegrid implementation,
  // this test might fail.
  const auto original_output = vmecpp::run(indata);
  ASSERT_TRUE(original_output.ok());

  // using the in-memory mgrid
  const auto output_with_inmemory_mgrid =
      vmecpp::run(indata, magnetic_response_table);
  ASSERT_TRUE(output_with_inmemory_mgrid.ok());

  // compare wout contents. jcuru/jcurv are curl(B) currents whose two solve
  // paths diverge by ~1.03e-7 across optimized/vectorized builds; keep every
  // other quantity at 1e-7 and compare those two at 2e-7.
  vmecpp::CompareWOut(output_with_inmemory_mgrid->wout, original_output->wout,
                      /*tolerance=*/1e-7, /*check_equal_niter=*/true,
                      /*current_density_tolerance=*/2e-7);
}

// Axisymmetric (ntor = 0, nzeta = 1) free-boundary tokamak (solovev_free_bdy).
// The committed-mgrid run is validated field-by-field against the
// educational_VMEC golden in WOutFileContentsTest (output_quantities_test).
// This test additionally requires the in-memory makegrid path, built from the
// coils file for a single toroidal plane, to reproduce the committed-mgrid run
// across the whole wout.
TEST(TestVmec, SolovevFreeBoundaryAxisymmetric) {
  const std::string filename = "vmecpp/test_data/solovev_free_bdy.json";
  const absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  VmecINDATA& indata = maybe_indata.value();

  // Run with the committed on-disk mgrid referenced by the input file.
  const auto disk_output = vmecpp::run(indata);
  ASSERT_TRUE(disk_output.ok());

  // Build the field response table in memory from the coils file and run again.
  const auto maybe_magnetic_configuration =
      magnetics::ImportMagneticConfigurationFromCoilsFile(
          "vmecpp/test_data/coils.solovev");
  ASSERT_TRUE(maybe_magnetic_configuration.ok());

  const auto maybe_makegrid_params = makegrid::ImportMakegridParametersFromFile(
      "vmecpp/test_data/makegrid_parameters_solovev.json");
  ASSERT_TRUE(maybe_makegrid_params.ok());

  const auto maybe_magnetic_response_table =
      makegrid::ComputeMagneticFieldResponseTable(
          *maybe_makegrid_params, *maybe_magnetic_configuration);
  ASSERT_TRUE(maybe_magnetic_response_table.ok());

  indata.mgrid_file = "";  // use the in-memory response table instead of disk
  const auto inmemory_output =
      vmecpp::run(indata, *maybe_magnetic_response_table);
  ASSERT_TRUE(inmemory_output.ok());

  // The in-memory makegrid path must reproduce the committed-mgrid run.
  // jcuru/jcurv are curl(B) currents whose two solve paths diverge by ~1.03e-7
  // across optimized/vectorized builds; keep every other quantity at 1e-7 and
  // compare those two at 2e-7.
  vmecpp::CompareWOut(inmemory_output->wout, disk_output->wout,
                      /*tolerance=*/1e-7, /*check_equal_niter=*/true,
                      /*current_density_tolerance=*/2e-7);
}

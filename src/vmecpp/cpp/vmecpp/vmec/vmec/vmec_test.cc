// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/vmec/vmec.h"

#include <fstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/util/util.h"

using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::DoubleNear;
using ::testing::ElementsAreArray;
using ::testing::Pointwise;
using ::testing::TestWithParam;
using ::testing::Values;

using vmecpp::FlowControl;
using vmecpp::HandoverStorage;
using vmecpp::RadialPartitioning;
using vmecpp::Sizes;
using vmecpp::Vmec;
using vmecpp::VmecCheckpoint;
using vmecpp::VmecINDATA;

namespace fs = std::filesystem;

// used to specify case-specific tolerances
// and which iterations to test
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::vector<int> iter2_to_test = {1, 2};
};

TEST(TestVmec, CheckErrorOnNonConvergence) {
  // make sure VMEC++ reports an error if the run couldn't converge
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  // allow only 1 iteration - not enough to let VMEC converge
  indata->niter_array[0] = 1;

  auto maybe_vmec = Vmec::FromIndata(*indata);
  ASSERT_TRUE(maybe_vmec.ok());
  Vmec& vmec = **maybe_vmec;

  const absl::StatusOr<bool> status = vmec.run();

  CHECK(!status.ok());
  CHECK(
      absl::StrContains(status.status().message(), "VMEC++ did not converge"));
}  // CheckErrorOnNonConvergence

TEST(TestVmec, CheckNoErrorOnNonConvergenceIfDesired) {
  // make sure VMEC++ returns the outputs without an error
  // if explicitly instructed to do so
  const std::string filename = "vmecpp/test_data/cth_like_fixed_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  // allow only 1 iteration - not enough to let VMEC converge
  indata->niter_array[0] = 1;

  // instruct VMEC++ to return its outputs, even if it did not converge
  indata->return_outputs_even_if_not_converged = true;

  auto maybe_vmec = Vmec::FromIndata(*indata);
  ASSERT_TRUE(maybe_vmec.ok());
  Vmec& vmec = **maybe_vmec;

  const absl::StatusOr<bool> status = vmec.run();

  CHECK(status.ok());
}  // CheckNoErrorOnNonConvergenceIfDesired

TEST(TestVmec, CheckFromIndataReturnsErrorForInvalidMgridPath) {
  // Verify that FromIndata returns an error status (rather than throwing)
  // when a free-boundary run specifies a non-existent mgrid file.
  const std::string filename = "vmecpp/test_data/cth_like_free_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  VmecINDATA& indata = maybe_indata.value();

  indata.mgrid_file = "/does/not/exist/mgrid.nc";

  auto maybe_vmec = Vmec::FromIndata(indata);
  EXPECT_FALSE(maybe_vmec.ok());
}  // CheckFromIndataReturnsErrorForInvalidMgridPath

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

  // compare wout contents
  vmecpp::CompareWOut(output_with_inmemory_mgrid->wout, original_output->wout,
                      /*tolerance=*/1e-7);
}  // CheckInMemoryMgrid

// A multi-grid free-boundary equilibrium (cth_like_free_bdy with an added grid
// step) must converge. The free-boundary (Nestor) solver, together with its
// accumulated vacuum response matrix and right-hand side, is kept in memory
// across the multi-grid steps (reproducing Fortran VMEC's persistent vacuum
// state), so this also exercises that the reused solver state stays valid
// across a grid-size change. The step-by-step agreement against the Fortran
// reference is exercised in vmecpp_large_cpp_tests.
TEST(TestVmec, MultiGridFreeBoundary) {
  const std::string filename =
      "vmecpp/test_data/cth_like_free_bdy_multigrid.json";
  const absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());
  ASSERT_EQ(indata->ns_array.size(), 2u);

  const auto output = vmecpp::run(*indata, std::nullopt, 1);
  ASSERT_TRUE(output.ok());

  // Regression guard for issue #330/#640 and other changes to the multigrid
  // convergence path
  EXPECT_EQ(output->wout.niter, 344);
}  // MultiGridFreeBoundary

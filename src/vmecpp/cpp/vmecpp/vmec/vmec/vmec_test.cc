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
  CHECK_EQ(status.status().message(), "VMEC++ did not converge");
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

// A free-boundary run whose ntor is raised above the highest populated
// toroidal mode triggers the sparse-toroidal layout (the active set is derived
// from the non-zero boundary/axis coefficients). The extra toroidal columns
// are exactly zero, so the sparse run must converge to the same equilibrium as
// the dense run at the original ntor. This exercises the sparse -> dense
// scatter in HandOverBoundaryGeometry and confirms the Nestor vacuum coupling
// is unaffected by the sparse internal layout.
//
// Note: unlike the fixed-boundary case, this is NOT bit-identical. Nestor's
// vacuum solve uses nf = ntor toroidal modes, so raising ntor changes the
// toroidal resolution of the vacuum field (and the constraint de-aliasing
// band) even though the plasma boundary is unchanged. The two converged
// equilibria therefore agree only to a physical tolerance (~1e-5 relative),
// not to round-off. The point of the test is that the sparse free-boundary
// path runs and lands on the same equilibrium, not that it reproduces the
// dense discretization exactly.
TEST(TestVmec, CheckSparseToroidalFreeBoundaryMatchesDense) {
  const std::string filename = "vmecpp/test_data/cth_like_free_bdy.json";
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());

  // Dense reference run at the native resolution (all |n| <= ntor populated).
  const VmecINDATA dense_indata = maybe_indata.value();
  const auto dense_output = vmecpp::run(dense_indata);
  ASSERT_TRUE(dense_output.ok()) << dense_output.status();

  // Sparse run: raise ntor well above the populated modes so the high-n columns
  // are zero-padded; the active toroidal set is then a strict subset of
  // {0, ..., ntor} and the sparse-toroidal path is taken.
  VmecINDATA sparse_indata = maybe_indata.value();
  const int dense_ntor = dense_indata.ntor;
  sparse_indata.SetMpolNtor(sparse_indata.mpol, /*new_ntor=*/dense_ntor + 6);
  const auto sparse_output = vmecpp::run(sparse_indata);
  ASSERT_TRUE(sparse_output.ok()) << sparse_output.status();

  // The external wout stays dense in both runs; the sparse run simply carries
  // zeros in the high-n columns. Restricting attention to the shared modes, the
  // two equilibria must agree to solver tolerance.
  EXPECT_EQ(dense_output->wout.ntor, dense_ntor);
  EXPECT_EQ(sparse_output->wout.ntor, dense_ntor + 6);
  EXPECT_NEAR(sparse_output->wout.volume, dense_output->wout.volume,
              1e-5 * std::abs(dense_output->wout.volume));
  EXPECT_NEAR(sparse_output->wout.betatotal, dense_output->wout.betatotal,
              1e-4 * std::abs(dense_output->wout.betatotal) + 1e-12);
  ASSERT_EQ(sparse_output->wout.iotaf.size(), dense_output->wout.iotaf.size());
  for (int j = 0; j < dense_output->wout.iotaf.size(); ++j) {
    EXPECT_NEAR(sparse_output->wout.iotaf[j], dense_output->wout.iotaf[j],
                1e-5 * std::abs(dense_output->wout.iotaf[j]) + 1e-7)
        << "iotaf mismatch at flux surface " << j;
  }
}  // CheckSparseToroidalFreeBoundaryMatchesDense

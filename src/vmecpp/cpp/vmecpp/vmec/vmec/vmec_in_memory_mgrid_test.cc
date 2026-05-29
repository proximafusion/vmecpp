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
using testing::IsCloseRelAbs;
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

  // compare wout contents
  vmecpp::CompareWOut(output_with_inmemory_mgrid->wout, original_output->wout,
                      /*tolerance=*/1e-7);
}

// Axisymmetric (ntor = 0, nzeta = 1) free-boundary tokamak equilibrium
// (solovev_free_bdy). The coil-field response table is generated in memory from
// the coils file, so this also exercises the in-memory makegrid path for a
// single toroidal plane.
TEST(TestVmec, SolovevFreeBoundaryAxisymmetric) {
  const std::string filename = "vmecpp/test_data/solovev_free_bdy.json";
  const absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  VmecINDATA& indata = maybe_indata.value();
  // The on-disk mgrid is not committed; the field comes from the in-memory
  // response table built below.
  indata.mgrid_file = "";

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

  const auto output = vmecpp::run(indata, *maybe_magnetic_response_table);
  ASSERT_TRUE(output.ok());

  // Validate the converged equilibrium against an educational_VMEC
  // free-boundary reference for the same coils and boundary (NS_ARRAY =
  // [16, 32], FTOL = 1e-14), generated from coils.solovev with VMEC++'s
  // makegrid. This pins the axisymmetric free-boundary solver (Nestor with the
  // nzeta = 1 vacuum integration) to the Fortran implementation.
  const auto& wout = output->wout;
  constexpr double kTol = 1.0e-4;
  EXPECT_TRUE(IsCloseRelAbs(6.2302973088e-02, wout.wb, kTol))
      << "wb=" << wout.wb;
  EXPECT_TRUE(IsCloseRelAbs(4.1817225953e-06, wout.betatotal, kTol))
      << "betatotal=" << wout.betatotal;
  EXPECT_TRUE(IsCloseRelAbs(3.1663788208e+00, wout.aspect, kTol))
      << "aspect=" << wout.aspect;
  EXPECT_TRUE(IsCloseRelAbs(1.9152722492e-01, wout.b0, kTol))
      << "b0=" << wout.b0;
  EXPECT_TRUE(IsCloseRelAbs(1.2883768174e+02, wout.volume, kTol))
      << "volume=" << wout.volume;
  EXPECT_TRUE(IsCloseRelAbs(1.2726728016e+00, wout.Aminor_p, kTol))
      << "Aminor_p=" << wout.Aminor_p;
  EXPECT_TRUE(IsCloseRelAbs(4.0297642049e+00, wout.Rmajor_p, kTol))
      << "Rmajor_p=" << wout.Rmajor_p;
  EXPECT_TRUE(IsCloseRelAbs(-9.4386273040e+04, wout.ctor, kTol))
      << "ctor=" << wout.ctor;
  EXPECT_TRUE(IsCloseRelAbs(1.9540145857e-01, wout.volavgB, kTol))
      << "volavgB=" << wout.volavgB;
}

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <filesystem>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/magnetic_field_provider/magnetic_field_provider_lib.h"

using composed_types::CurveRZFourier;
using composed_types::CurveRZFourierFromCsv;
using magnetics::MagneticConfiguration;
using testing::IsCloseRelAbs;

TEST(TestLinkingCurrent, CheckForW7X) {
  static constexpr double kTolerance = 5.0e-3;

  std::filesystem::path makegrid_coils_filename(
      "vmecpp_large_cpp_tests/test_data/coils.w7x");
  std::filesystem::path axis_fourier_geometry_filename(
      "vmecpp_large_cpp_tests/test_data/axis_coefficients_w7x.csv");

  // read coil geometry from file
  absl::StatusOr<MagneticConfiguration> maybe_magnetic_configuration =
      magnetics::ImportMagneticConfigurationFromCoilsFile(
          makegrid_coils_filename);
  CHECK_OK(maybe_magnetic_configuration);
  MagneticConfiguration magnetic_configuration =
      maybe_magnetic_configuration.value();

  // W7-X standard magnetic configuration
  // 13.5kA in non-planar coils
  // no current in planar coils
  Eigen::VectorXd circuit_currents(7);
  circuit_currents << 13.5e3, 13.5e3, 13.5e3, 13.5e3, 13.5e3, 0.0, 0.0;

  // set circuit currents in MagneticConfiguration
  CHECK_OK(magnetics::SetCircuitCurrents(
      circuit_currents,
      /*m_magnetic_configuration=*/magnetic_configuration));

  // read axis geometry coefficients
  absl::StatusOr<std::string> maybe_axis_coefficients_csv =
      file_io::ReadFile(axis_fourier_geometry_filename);
  CHECK_OK(maybe_axis_coefficients_csv);
  const std::string& axis_coefficients_csv =
      maybe_axis_coefficients_csv.value();

  absl::StatusOr<CurveRZFourier> maybe_axis_coefficients =
      CurveRZFourierFromCsv(axis_coefficients_csv);
  CHECK_OK(maybe_axis_coefficients);
  const CurveRZFourier& axis_coefficients = maybe_axis_coefficients.value();

  absl::StatusOr<double> maybe_linking_current =
      LinkingCurrent(magnetic_configuration, axis_coefficients);
  CHECK_OK(maybe_linking_current);
  const double linking_current = maybe_linking_current.value();

  EXPECT_TRUE(IsCloseRelAbs(72.9e6, linking_current, kTolerance));
}  // TestLinkingCurrent:CheckForW7X

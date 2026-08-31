// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include <netcdf.h>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/netcdf_io/netcdf_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/magnetic_field_provider/magnetic_field_provider_lib.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"

namespace {
using nlohmann::json;

using file_io::ReadFile;
using netcdf_io::NetcdfReadArray3D;
using netcdf_io::NetcdfReadBool;
using netcdf_io::NetcdfReadDouble;
using netcdf_io::NetcdfReadInt;
using testing::IsCloseRelAbs;

using magnetics::ImportMagneticConfigurationFromMakegrid;
using magnetics::MagneticConfiguration;
using magnetics::MagneticField;

using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::TestWithParam;
using ::testing::Values;
}  // namespace

namespace vmecpp {

// used to specify case-specific tolerances
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::string coils_file = "";
};

class LoadMGridTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(LoadMGridTest, CheckLoadMGrid) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // This test is only meaningful in case of a free-boundary run
  ASSERT_TRUE(vmec_indata->lfreeb);

  // make sure the mgrid file is available
  std::ifstream mgrid_file(vmec_indata->mgrid_file);
  ASSERT_TRUE(mgrid_file.is_open());
  mgrid_file.close();

  MGridProvider mgrid;
  absl::Status load_status =
      mgrid.LoadFile(vmec_indata->mgrid_file, vmec_indata->extcur);
  ASSERT_TRUE(load_status.ok()) << load_status;

  // The reference calculation for comparison is done using
  // //magnetics/magnetic_field_provider (which internally uses ABSCAB).
  std::filesystem::path makegrid_coils_file =
      "vmecpp/test_data/" + data_source_.coils_file;

  // load MagneticConfiguration from coils file
  absl::StatusOr<MagneticConfiguration> magnetic_configuration =
      magnetics::ImportMagneticConfigurationFromCoilsFile(makegrid_coils_file);
  ASSERT_TRUE(magnetic_configuration.ok());

  // get coil currents in A from INDATA and put them into the
  // MagneticConfiguration
  absl::Status status_set_currents = magnetics::SetCircuitCurrents(
      vmec_indata->extcur, *magnetic_configuration);
  ASSERT_TRUE(status_set_currents.ok());

  // get dimensions of mgrid file
  int ncid = 0;
  ASSERT_EQ(nc_open(vmec_indata->mgrid_file.c_str(), NC_NOWRITE, &ncid),
            NC_NOERR);

  const int number_of_field_periods = NetcdfReadInt(ncid, "nfp").value();

  const int number_of_r_grid_points = NetcdfReadInt(ncid, "ir").value();
  const double r_grid_minimum = NetcdfReadDouble(ncid, "rmin").value();
  const double r_grid_maximum = NetcdfReadDouble(ncid, "rmax").value();
  const double r_grid_increment =
      (r_grid_maximum - r_grid_minimum) / (number_of_r_grid_points - 1.0);

  const int number_of_z_grid_points = NetcdfReadInt(ncid, "jz").value();
  const double z_grid_minimum = NetcdfReadDouble(ncid, "zmin").value();
  const double z_grid_maximum = NetcdfReadDouble(ncid, "zmax").value();
  const double z_grid_increment =
      (z_grid_maximum - z_grid_minimum) / (number_of_z_grid_points - 1.0);

  const int number_of_phi_grid_points = NetcdfReadInt(ncid, "kp").value();
  const double phi_grid_increment =
      2.0 * M_PI / (number_of_phi_grid_points * number_of_field_periods);

  ASSERT_EQ(nc_close(ncid), NC_NOERR);

  // TODO(jons): A flag if stellarator symmetry was used in computing a given
  // mgrid file is not stored in the mgrid file. For now, hard-code this to
  // `true`, since all our test cases assume stellarator symmetry. To be revised
  // when a) we use non-stellarator-symmetric coil sets _and_ b) we have
  // transitioned to only using our own `makegrid`, in which we can define new
  // output variables and have the MakegridParameters at hand anyways.
  bool assume_stellarator_symmetry = true;

  // NOTE: The coil geometry in `coils.cth_like` was found to not be perfectly
  // stellarator-symmetric. Therefore, the resulting magnetic field is also not
  // perfectly stellarator symmetric. We ignore this issue for now and assume
  // both in `makegrid` and here the field to be perfectly
  // stellarator-symmetric. Therefore, we also only check the first
  // half-field-period for a stellarator-symmetric case as `cth_like`.
  int num_phi_effective = number_of_phi_grid_points;
  if (assume_stellarator_symmetry) {
    num_phi_effective = number_of_phi_grid_points / 2 + 1;
  }

  // Build the cylindrical grid based on mgrid dimensions.
  // The loop setup is re-used to also allocate the magnetic_field vectors.
  const int number_of_grid_points =
      number_of_r_grid_points * number_of_z_grid_points * num_phi_effective;
  std::vector<std::vector<double> > evaluation_locations(number_of_grid_points);
  std::vector<std::vector<double> > magnetic_field(number_of_grid_points);
  for (int index_phi = 0; index_phi < num_phi_effective; ++index_phi) {
    const double phi = index_phi * phi_grid_increment;
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);
    for (int index_z = 0; index_z < number_of_z_grid_points; ++index_z) {
      const double z = z_grid_minimum + index_z * z_grid_increment;
      for (int index_r = 0; index_r < number_of_r_grid_points; ++index_r) {
        const double r = r_grid_minimum + index_r * r_grid_increment;

        const double x = r * cos_phi;
        const double y = r * sin_phi;

        const int linear_index =
            (index_phi * number_of_z_grid_points + index_z) *
                number_of_r_grid_points +
            index_r;
        evaluation_locations[linear_index].resize(3);
        magnetic_field[linear_index].resize(3);

        evaluation_locations[linear_index][0] = x;
        evaluation_locations[linear_index][1] = y;
        evaluation_locations[linear_index][2] = z;
      }  // index_r
    }  // index_z
  }  // index_phi

  // evaluate magnetic field on grid
  absl::Status status = MagneticField(*magnetic_configuration,
                                      evaluation_locations, magnetic_field);
  ASSERT_TRUE(status.ok());

  // compare magnetic field point-wise
  for (int index_phi = 0; index_phi < num_phi_effective; ++index_phi) {
    const double phi = index_phi * phi_grid_increment;
    const double cos_phi = std::cos(phi);
    const double sin_phi = std::sin(phi);
    for (int index_z = 0; index_z < number_of_z_grid_points; ++index_z) {
      for (int index_r = 0; index_r < number_of_r_grid_points; ++index_r) {
        const int linear_index =
            (index_phi * number_of_z_grid_points + index_z) *
                number_of_r_grid_points +
            index_r;

        // ABSCAB computes the Cartesian components of the magnetic field,
        // so we need to convert the x and y componets into r and phi
        // (cylindrical) components for comparison against the cylindrical
        // components in the mgrid file.
        const double b_x = magnetic_field[linear_index][0];
        const double b_y = magnetic_field[linear_index][1];
        const double b_z = magnetic_field[linear_index][2];

        const double b_r = b_x * cos_phi + b_y * sin_phi;
        const double b_p = b_y * cos_phi - b_x * sin_phi;

        EXPECT_TRUE(IsCloseRelAbs(b_r, mgrid.bR[linear_index], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(b_p, mgrid.bP[linear_index], tolerance));
        EXPECT_TRUE(IsCloseRelAbs(b_z, mgrid.bZ[linear_index], tolerance));
      }  // index_r
    }  // index_z
  }  // index_phi
}  // CheckLoadMGrid

INSTANTIATE_TEST_SUITE_P(TestVmec, LoadMGridTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-12,
                                           .coils_file = "coils.cth_like"}));

// Number of tangential grid points used by the interpolation tests below.
static constexpr int kNumTangentialPoints = 8;

class MGridInterpolationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const absl::StatusOr<std::string> indata_json =
        ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
    ASSERT_TRUE(indata_json.ok()) << indata_json.status();

    const absl::StatusOr<VmecINDATA> vmec_indata =
        VmecINDATA::FromJson(*indata_json);
    ASSERT_TRUE(vmec_indata.ok()) << vmec_indata.status();

    const absl::Status load_status =
        mgrid_.LoadFile(vmec_indata->mgrid_file, vmec_indata->extcur);
    ASSERT_TRUE(load_status.ok()) << load_status;
  }

  // A closed contour that stays well inside the vacuum field grid.
  void FillInsideGrid(int num_points, Eigen::VectorXd& m_r,
                      Eigen::VectorXd& m_z) const {
    const double r_center = 0.5 * (mgrid_.minR + mgrid_.maxR);
    const double z_center = 0.5 * (mgrid_.minZ + mgrid_.maxZ);
    const double r_amplitude = 0.25 * (mgrid_.maxR - mgrid_.minR);
    const double z_amplitude = 0.25 * (mgrid_.maxZ - mgrid_.minZ);

    m_r.resize(num_points);
    m_z.resize(num_points);
    for (int index = 0; index < num_points; ++index) {
      const double theta = 2.0 * M_PI * index / num_points;
      m_r[index] = r_center + r_amplitude * std::cos(theta);
      m_z[index] = z_center + z_amplitude * std::sin(theta);
    }
  }

  // How far outside the grid the deliberately out-of-bounds points are put.
  double OutsideR() const {
    return mgrid_.maxR + 0.1 * (mgrid_.maxR - mgrid_.minR);
  }
  double OutsideZ() const {
    return mgrid_.minZ - 0.1 * (mgrid_.maxZ - mgrid_.minZ);
  }

  MGridProvider mgrid_;
};

TEST_F(MGridInterpolationTest, BoundaryInsideGridIsAccepted) {
  Eigen::VectorXd r;
  Eigen::VectorXd z;
  FillInsideGrid(kNumTangentialPoints, r, z);

  Eigen::VectorXd b_r(kNumTangentialPoints);
  Eigen::VectorXd b_p(kNumTangentialPoints);
  Eigen::VectorXd b_z(kNumTangentialPoints);
  const absl::Status status =
      mgrid_.interpolate(0, kNumTangentialPoints, mgrid_.numPhi,
                         kNumTangentialPoints, r, z, b_r, b_p, b_z);

  EXPECT_TRUE(status.ok()) << status;
  for (int index = 0; index < kNumTangentialPoints; ++index) {
    EXPECT_TRUE(std::isfinite(b_r[index]));
    EXPECT_TRUE(std::isfinite(b_p[index]));
    EXPECT_TRUE(std::isfinite(b_z[index]));
  }
}

// The reported extents are the whole boundary's, not the reporting slice's:
// this call covers the second slice only, which does not contain the minima.
TEST_F(MGridInterpolationTest, BoundaryOutsideGridIsAnError) {
  const int num_points = 2 * kNumTangentialPoints;
  Eigen::VectorXd r;
  Eigen::VectorXd z;
  FillInsideGrid(num_points, r, z);
  r[num_points - 1] = OutsideR();
  z[num_points - 2] = OutsideZ();

  Eigen::VectorXd b_r(kNumTangentialPoints);
  Eigen::VectorXd b_p(kNumTangentialPoints);
  Eigen::VectorXd b_z(kNumTangentialPoints);
  const absl::Status status =
      mgrid_.interpolate(kNumTangentialPoints, num_points, mgrid_.numPhi,
                         num_points, r, z, b_r, b_p, b_z);

  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);
  const std::string message(status.message());
  EXPECT_THAT(message, HasSubstr(absl::StrFormat("% .6e", r.minCoeff())));
  EXPECT_THAT(message, HasSubstr(absl::StrFormat("% .6e", r.maxCoeff())));
  EXPECT_THAT(message, HasSubstr(absl::StrFormat("% .6e", z.minCoeff())));
  EXPECT_THAT(message, HasSubstr(absl::StrFormat("% .6e", z.maxCoeff())));
}

#ifdef _OPENMP
// Hangs if the barrier is ever made conditional on the slice being in grid.
TEST_F(MGridInterpolationTest, MixedInAndOutOfGridSlicesDoNotDeadlock) {
  constexpr int kNumThreads = 4;
  const int num_points = kNumThreads * kNumTangentialPoints;

  Eigen::VectorXd r;
  Eigen::VectorXd z;
  FillInsideGrid(num_points, r, z);
  // Only the slice owned by the last thread leaves the grid.
  r[num_points - 1] = OutsideR();

  std::vector<absl::Status> per_thread_status(kNumThreads);
  int team_size = 0;

#pragma omp parallel num_threads(kNumThreads)
  {
    const int thread_id = omp_get_thread_num();
#pragma omp single
    {
      team_size = omp_get_num_threads();
    }

    const int zt_min = thread_id * kNumTangentialPoints;
    const int zt_max = zt_min + kNumTangentialPoints;

    Eigen::VectorXd b_r(kNumTangentialPoints);
    Eigen::VectorXd b_p(kNumTangentialPoints);
    Eigen::VectorXd b_z(kNumTangentialPoints);
    per_thread_status[thread_id] = mgrid_.interpolate(
        zt_min, zt_max, mgrid_.numPhi, num_points, r, z, b_r, b_p, b_z);
  }

  ASSERT_EQ(team_size, kNumThreads);
  for (int thread_id = 0; thread_id < kNumThreads - 1; ++thread_id) {
    EXPECT_TRUE(per_thread_status[thread_id].ok())
        << per_thread_status[thread_id];
  }
  EXPECT_EQ(per_thread_status[kNumThreads - 1].code(),
            absl::StatusCode::kFailedPrecondition);
}
#endif  // _OPENMP

// A field that is exactly bilinear in R and Z on each phi plane is reproduced
// exactly by bilinear interpolation, so an error in the corner weights, in the
// linear index arithmetic, or in which of R and Z a weight belongs to shows up
// as a mismatch at points inside the grid.
class MGridBilinearInterpolationTest : public ::testing::Test {
 protected:
  // The two resolutions differ so that a transposed linear index is caught.
  static constexpr int kNumR = 5;
  static constexpr int kNumZ = 7;
  static constexpr int kNumPhi = 3;

  static constexpr double kMinR = 0.8;
  static constexpr double kMaxR = 1.6;
  static constexpr double kMinZ = -0.5;
  static constexpr double kMaxZ = 0.7;

  static constexpr double kTolerance = 1.0e-12;

  // The three components differ from one another, and every phi plane differs
  // from every other, so a component or a plane taken from the wrong place does
  // not go unnoticed.
  static double ReferenceBr(double r, double z, int index_phi) {
    return 1.0 + 2.0 * r - 3.0 * z + 4.0 * r * z + 10.0 * index_phi;
  }
  static double ReferenceBp(double r, double z, int index_phi) {
    return -0.5 + 1.5 * r + 2.5 * z - 3.5 * r * z + 20.0 * index_phi;
  }
  static double ReferenceBz(double r, double z, int index_phi) {
    return 7.0 - 6.0 * r + 5.0 * z + 0.25 * r * z + 30.0 * index_phi;
  }

  void SetUp() override {
    makegrid::MagneticFieldResponseTable response_table;
    response_table.parameters = {.normalize_by_currents = false,
                                 .assume_stellarator_symmetry = false,
                                 .number_of_field_periods = 1,
                                 .r_grid_minimum = kMinR,
                                 .r_grid_maximum = kMaxR,
                                 .number_of_r_grid_points = kNumR,
                                 .z_grid_minimum = kMinZ,
                                 .z_grid_maximum = kMaxZ,
                                 .number_of_z_grid_points = kNumZ,
                                 .number_of_phi_grid_points = kNumPhi};

    const int num_grid_points = kNumPhi * kNumZ * kNumR;
    response_table.b_r.resize(1, num_grid_points);
    response_table.b_p.resize(1, num_grid_points);
    response_table.b_z.resize(1, num_grid_points);

    const double delta_r = (kMaxR - kMinR) / (kNumR - 1.0);
    const double delta_z = (kMaxZ - kMinZ) / (kNumZ - 1.0);
    for (int index_phi = 0; index_phi < kNumPhi; ++index_phi) {
      for (int index_z = 0; index_z < kNumZ; ++index_z) {
        for (int index_r = 0; index_r < kNumR; ++index_r) {
          const int linear_index =
              (index_phi * kNumZ + index_z) * kNumR + index_r;
          const double r = kMinR + index_r * delta_r;
          const double z = kMinZ + index_z * delta_z;
          response_table.b_r(0, linear_index) = ReferenceBr(r, z, index_phi);
          response_table.b_p(0, linear_index) = ReferenceBp(r, z, index_phi);
          response_table.b_z(0, linear_index) = ReferenceBz(r, z, index_phi);
        }  // index_r
      }  // index_z
    }  // index_phi

    Eigen::VectorXd coil_currents(1);
    coil_currents[0] = 1.0;
    const absl::Status status =
        mgrid_.LoadFields(response_table, coil_currents);
    ASSERT_TRUE(status.ok()) << status;
  }

  // Lays the given (R, Z) samples out so that every sample is evaluated on
  // every phi plane: interpolate() takes the plane of point kl to be
  // kl % nZeta.
  static void SpreadOverPhiPlanes(
      const std::vector<std::pair<double, double> >& samples,
      Eigen::VectorXd& m_r, Eigen::VectorXd& m_z) {
    const int num_points = static_cast<int>(samples.size()) * kNumPhi;
    m_r.resize(num_points);
    m_z.resize(num_points);
    for (int kl = 0; kl < num_points; ++kl) {
      m_r[kl] = samples[kl / kNumPhi].first;
      m_z[kl] = samples[kl / kNumPhi].second;
    }  // kl
  }

  MGridProvider mgrid_;
};

TEST_F(MGridBilinearInterpolationTest, ReproducesABilinearFieldExactly) {
  // Grid nodes, including both extreme corners, cell centers, and points at no
  // particular position within a cell.
  const std::vector<std::pair<double, double> > samples = {
      {kMinR, kMinZ},  // lower left corner node
      {kMaxR, kMaxZ},  // upper right corner node
      {kMaxR, kMinZ},  // lower right corner node
      {1.2, -0.1},     // interior node
      {0.9, -0.4},     // center of the first cell
      {1.5, 0.6},      // center of the last cell
      {1.13, 0.02},    // no particular position
      {0.85, 0.65},    // against the inner R edge, high Z
  };

  Eigen::VectorXd r;
  Eigen::VectorXd z;
  SpreadOverPhiPlanes(samples, r, z);
  const int num_points = static_cast<int>(r.size());

  Eigen::VectorXd b_r(num_points);
  Eigen::VectorXd b_p(num_points);
  Eigen::VectorXd b_z(num_points);
  const absl::Status status = mgrid_.interpolate(
      0, num_points, kNumPhi, num_points, r, z, b_r, b_p, b_z);
  ASSERT_TRUE(status.ok()) << status;

  for (int kl = 0; kl < num_points; ++kl) {
    const int index_phi = kl % kNumPhi;
    EXPECT_NEAR(b_r[kl], ReferenceBr(r[kl], z[kl], index_phi), kTolerance)
        << "b_r at point " << kl;
    EXPECT_NEAR(b_p[kl], ReferenceBp(r[kl], z[kl], index_phi), kTolerance)
        << "b_p at point " << kl;
    EXPECT_NEAR(b_z[kl], ReferenceBz(r[kl], z[kl], index_phi), kTolerance)
        << "b_z at point " << kl;
  }  // kl
}

TEST_F(MGridBilinearInterpolationTest, OutOfGridPointsAreClampedToTheEdge) {
  // The status is covered above; what is checked here is the value returned
  // alongside it, so that a point past a corner picks up that corner rather
  // than an extrapolation.
  const std::vector<std::pair<double, double> > samples = {
      {kMinR - 0.3, 0.1},          // outside in R only
      {1.1, kMaxZ + 0.4},          // outside in Z only
      {kMaxR + 0.2, kMinZ - 0.2},  // outside in both, past a corner
  };

  Eigen::VectorXd r;
  Eigen::VectorXd z;
  SpreadOverPhiPlanes(samples, r, z);
  const int num_points = static_cast<int>(r.size());

  Eigen::VectorXd b_r(num_points);
  Eigen::VectorXd b_p(num_points);
  Eigen::VectorXd b_z(num_points);
  const absl::Status status = mgrid_.interpolate(
      0, num_points, kNumPhi, num_points, r, z, b_r, b_p, b_z);
  EXPECT_EQ(status.code(), absl::StatusCode::kFailedPrecondition);

  for (int kl = 0; kl < num_points; ++kl) {
    const int index_phi = kl % kNumPhi;
    const double clamped_r = std::max(kMinR, std::min(r[kl], kMaxR));
    const double clamped_z = std::max(kMinZ, std::min(z[kl], kMaxZ));
    EXPECT_NEAR(b_r[kl], ReferenceBr(clamped_r, clamped_z, index_phi),
                kTolerance)
        << "b_r at point " << kl;
    EXPECT_NEAR(b_p[kl], ReferenceBp(clamped_r, clamped_z, index_phi),
                kTolerance)
        << "b_p at point " << kl;
    EXPECT_NEAR(b_z[kl], ReferenceBz(clamped_r, clamped_z, index_phi),
                kTolerance)
        << "b_z at point " << kl;
  }  // kl
}

TEST_F(MGridBilinearInterpolationTest, AFixedFieldIsCopiedAndTakesPrecedence) {
  // SetFixedMagneticField overrides the response table loaded in SetUp, and
  // hands back the requested slice verbatim instead of interpolating it.
  constexpr int kNumPoints = 6;
  constexpr int kZtMin = 2;
  constexpr int kZtMax = 5;

  Eigen::VectorXd fixed_b_r(kNumPoints);
  Eigen::VectorXd fixed_b_p(kNumPoints);
  Eigen::VectorXd fixed_b_z(kNumPoints);
  for (int kl = 0; kl < kNumPoints; ++kl) {
    fixed_b_r[kl] = 1.0 + kl;
    fixed_b_p[kl] = 10.0 + kl;
    fixed_b_z[kl] = 100.0 + kl;
  }  // kl
  mgrid_.SetFixedMagneticField(fixed_b_r, fixed_b_p, fixed_b_z);

  // Far outside the grid, so that an interpolated result would be an error.
  Eigen::VectorXd r = Eigen::VectorXd::Constant(kNumPoints, 100.0);
  Eigen::VectorXd z = Eigen::VectorXd::Constant(kNumPoints, -100.0);

  Eigen::VectorXd b_r(kZtMax - kZtMin);
  Eigen::VectorXd b_p(kZtMax - kZtMin);
  Eigen::VectorXd b_z(kZtMax - kZtMin);
  const absl::Status status = mgrid_.interpolate(
      kZtMin, kZtMax, kNumPhi, kNumPoints, r, z, b_r, b_p, b_z);
  ASSERT_TRUE(status.ok()) << status;

  for (int kl = kZtMin; kl < kZtMax; ++kl) {
    EXPECT_EQ(b_r[kl - kZtMin], fixed_b_r[kl]);
    EXPECT_EQ(b_p[kl - kZtMin], fixed_b_p[kl]);
    EXPECT_EQ(b_z[kl - kZtMin], fixed_b_z[kl]);
  }  // kl
}

}  // namespace vmecpp

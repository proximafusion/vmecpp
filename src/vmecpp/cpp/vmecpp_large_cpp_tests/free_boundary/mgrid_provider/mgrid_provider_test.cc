// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"

#include <cmath>
#include <fstream>
#include <string>
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

// LoadFile and LoadFields each reject a coil-current count that disagrees with
// the number of response tables they hold. VmecINDATA::IsConsistent cannot make
// this check, because it never sees the mgrid.
TEST(MGridProviderValidation, LoadFileRejectsWrongNumberOfCurrents) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
  ASSERT_TRUE(indata_json.ok()) << indata_json.status();
  const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok()) << indata.status();

  MGridProvider mgrid;

  // one too few
  Eigen::VectorXd too_few = indata->extcur.head(indata->extcur.size() - 1);
  const absl::Status short_status = mgrid.LoadFile(indata->mgrid_file, too_few);
  EXPECT_EQ(short_status.code(), absl::StatusCode::kInvalidArgument);

  // one too many
  Eigen::VectorXd too_many(indata->extcur.size() + 1);
  too_many.head(indata->extcur.size()) = indata->extcur;
  too_many[indata->extcur.size()] = 1.0;
  const absl::Status long_status = mgrid.LoadFile(indata->mgrid_file, too_many);
  EXPECT_EQ(long_status.code(), absl::StatusCode::kInvalidArgument);

  // the matching count still loads
  EXPECT_TRUE(mgrid.LoadFile(indata->mgrid_file, indata->extcur).ok());
}

TEST(MGridProviderValidation, LoadFieldsRejectsWrongNumberOfCurrents) {
  // A response table with two circuits on a small grid; only the shapes matter
  // here, not the field values.
  makegrid::MagneticFieldResponseTable response_table;
  response_table.parameters = {.normalize_by_currents = false,
                               .assume_stellarator_symmetry = false,
                               .number_of_field_periods = 1,
                               .r_grid_minimum = 1.0,
                               .r_grid_maximum = 2.0,
                               .number_of_r_grid_points = 3,
                               .z_grid_minimum = -1.0,
                               .z_grid_maximum = 1.0,
                               .number_of_z_grid_points = 3,
                               .number_of_phi_grid_points = 2};
  const int num_grid_points = 2 * 3 * 3;
  response_table.b_r = RowMatrixXd::Zero(2, num_grid_points);
  response_table.b_p = RowMatrixXd::Zero(2, num_grid_points);
  response_table.b_z = RowMatrixXd::Zero(2, num_grid_points);

  MGridProvider mgrid;
  EXPECT_EQ(mgrid.LoadFields(response_table, Eigen::VectorXd::Ones(1)).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(mgrid.LoadFields(response_table, Eigen::VectorXd::Ones(3)).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(mgrid.LoadFields(response_table, Eigen::VectorXd::Ones(2)).ok());
}

// The coil group names MAKEGRID writes into the mgrid file are what wout
// reports as curlabel. mgrid_cth_like.nc carries two of them.
TEST(MGridProviderValidation, LoadFileReadsCoilGroupNames) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
  ASSERT_TRUE(indata_json.ok()) << indata_json.status();
  const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok()) << indata.status();

  MGridProvider mgrid;
  ASSERT_TRUE(mgrid.LoadFile(indata->mgrid_file, indata->extcur).ok());

  ASSERT_EQ(static_cast<int>(mgrid.coil_group_names.size()), mgrid.nextcur);
  for (const std::string& name : mgrid.coil_group_names) {
    EXPECT_FALSE(name.empty());
    EXPECT_EQ(name.find_last_not_of(' '), name.size() - 1)
        << "name '" << name << "' still carries padding";
  }
}

}  // namespace vmecpp

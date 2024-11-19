#include "vmecpp/free_boundary/mgrid_provider/mgrid_provider.h"

#include <fstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include <netcdf.h>

#include "absl/strings/str_format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/netcdf_io/netcdf_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/tools/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/tools/magnetic_field_provider/magnetic_field_provider_lib.h"

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
  mgrid.loadFromMGrid(vmec_indata->mgrid_file, vmec_indata->extcur);

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

  const int number_of_field_periods = NetcdfReadInt(ncid, "nfp");

  const int number_of_r_grid_points = NetcdfReadInt(ncid, "ir");
  const double r_grid_minimum = NetcdfReadDouble(ncid, "rmin");
  const double r_grid_maximum = NetcdfReadDouble(ncid, "rmax");
  const double r_grid_increment =
      (r_grid_maximum - r_grid_minimum) / (number_of_r_grid_points - 1.0);

  const int number_of_z_grid_points = NetcdfReadInt(ncid, "jz");
  const double z_grid_minimum = NetcdfReadDouble(ncid, "zmin");
  const double z_grid_maximum = NetcdfReadDouble(ncid, "zmax");
  const double z_grid_increment =
      (z_grid_maximum - z_grid_minimum) / (number_of_z_grid_points - 1.0);

  const int number_of_phi_grid_points = NetcdfReadInt(ncid, "kp");
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
    }    // index_z
  }      // index_phi

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
    }    // index_z
  }      // index_phi
}  // CheckLoadMGrid

INSTANTIATE_TEST_SUITE_P(TestVmec, LoadMGridTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-12,
                                           .coils_file = "coils.cth_like"}));

}  // namespace vmecpp

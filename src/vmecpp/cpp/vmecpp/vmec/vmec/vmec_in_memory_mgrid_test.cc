// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <netcdf.h>

#include <filesystem>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "gmock/gmock.h"  // ElementsAreArray
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "util/netcdf_io/netcdf_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/magnetic_configuration_lib/magnetic_configuration_lib.h"
#include "vmecpp/common/makegrid_lib/makegrid_lib.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/output_quantities/test_helpers.h"
#include "vmecpp/vmec/vmec/vmec.h"

using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
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

// The toroidal resolution of the vacuum field and of the solver have to agree.
// VmecINDATA::IsConsistent cannot check this, because it never sees the mgrid,
// so Vmec::run does it once the provider is loaded.
TEST(TestVmec, InMemoryMgridWithMismatchedNzetaIsRejected) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
  ASSERT_TRUE(indata_json.ok());
  absl::StatusOr<VmecINDATA> maybe_indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  const VmecINDATA& indata = maybe_indata.value();

  const auto maybe_magnetic_configuration =
      magnetics::ImportMagneticConfigurationFromCoilsFile(
          "vmecpp/test_data/coils.cth_like");
  ASSERT_TRUE(maybe_magnetic_configuration.ok());

  auto maybe_makegrid_params = makegrid::ImportMakegridParametersFromFile(
      "vmecpp/test_data/makegrid_parameters_cth_like.json");
  ASSERT_TRUE(maybe_makegrid_params.ok());
  makegrid::MakegridParameters makegrid_params = *maybe_makegrid_params;

  // Half the toroidal resolution the input asks for, on a coarse R-Z grid so
  // that building the table stays cheap. Still even, which the symmetric grid
  // requires.
  ASSERT_EQ(makegrid_params.number_of_phi_grid_points, indata.nzeta);
  makegrid_params.number_of_phi_grid_points = indata.nzeta / 2;
  makegrid_params.number_of_r_grid_points = 5;
  makegrid_params.number_of_z_grid_points = 5;

  const auto maybe_response_table = makegrid::ComputeMagneticFieldResponseTable(
      makegrid_params, *maybe_magnetic_configuration);
  ASSERT_TRUE(maybe_response_table.ok());

  const auto output = vmecpp::run(indata, *maybe_response_table);
  ASSERT_FALSE(output.ok());
  EXPECT_EQ(output.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(output.status().message()),
              ::testing::HasSubstr("phi grid points"));
}

// The stellarator-symmetry operation maps toroidal plane k onto (kp - k) % kp
// and Z onto -Z, negates B_R and leaves B_phi and B_Z unchanged; the
// stellarator-symmetric mgrid_cth_like.nc satisfies that relation to 2e-15.
// Applied to the non-stellarator-symmetric mgrid_cth_like_asym.nc it gives the
// mirror image of that coil field, and the equilibrium it produces has to be
// the mirror image of the original: every gauge-invariant scalar unchanged,
// every antisymmetric Fourier array negated. That holds the sin/cos coupling
// blocks of the vacuum solver to account.
TEST(TestVmec, LasymFreeBoundaryIsMirrorCovariant) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy_asym.json");
  ASSERT_TRUE(indata_json.ok());
  const absl::StatusOr<VmecINDATA> maybe_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  const VmecINDATA& indata = *maybe_indata;
  ASSERT_TRUE(indata.lasym);

  // Read the response table of the asymmetric mgrid file.
  int ncid = 0;
  ASSERT_EQ(
      nc_open("vmecpp/test_data/mgrid_cth_like_asym.nc", NC_NOWRITE, &ncid),
      NC_NOERR);
  const auto read_int = [&](const char* name) {
    const absl::StatusOr<int> value = netcdf_io::NetcdfReadInt(ncid, name);
    CHECK_OK(value);
    return *value;
  };
  const auto read_double = [&](const char* name) {
    const absl::StatusOr<double> value =
        netcdf_io::NetcdfReadDouble(ncid, name);
    CHECK_OK(value);
    return *value;
  };
  const int num_r = read_int("ir");
  const int num_z = read_int("jz");
  const int num_phi = read_int("kp");
  const int nextcur = read_int("nextcur");
  ASSERT_EQ(nextcur, static_cast<int>(indata.extcur.size()));

  makegrid::MagneticFieldResponseTable table;
  table.parameters = {.normalize_by_currents = false,
                      .assume_stellarator_symmetry = false,
                      .number_of_field_periods = read_int("nfp"),
                      .r_grid_minimum = read_double("rmin"),
                      .r_grid_maximum = read_double("rmax"),
                      .number_of_r_grid_points = num_r,
                      .z_grid_minimum = read_double("zmin"),
                      .z_grid_maximum = read_double("zmax"),
                      .number_of_z_grid_points = num_z,
                      .number_of_phi_grid_points = num_phi};
  const int num_grid_points = num_phi * num_z * num_r;
  table.b_r = vmecpp::RowMatrixXd::Zero(nextcur, num_grid_points);
  table.b_p = vmecpp::RowMatrixXd::Zero(nextcur, num_grid_points);
  table.b_z = vmecpp::RowMatrixXd::Zero(nextcur, num_grid_points);
  for (int i = 0; i < nextcur; ++i) {
    const auto b_r =
        netcdf_io::NetcdfReadArray3D(ncid, absl::StrFormat("br_%03d", i + 1));
    const auto b_p =
        netcdf_io::NetcdfReadArray3D(ncid, absl::StrFormat("bp_%03d", i + 1));
    const auto b_z =
        netcdf_io::NetcdfReadArray3D(ncid, absl::StrFormat("bz_%03d", i + 1));
    ASSERT_TRUE(b_r.ok() && b_p.ok() && b_z.ok());
    for (int k = 0; k < num_phi; ++k) {
      for (int z = 0; z < num_z; ++z) {
        for (int r = 0; r < num_r; ++r) {
          const int index = (k * num_z + z) * num_r + r;
          table.b_r(i, index) = (*b_r)[k][z][r];
          table.b_p(i, index) = (*b_p)[k][z][r];
          table.b_z(i, index) = (*b_z)[k][z][r];
        }
      }
    }
  }
  ASSERT_EQ(nc_close(ncid), NC_NOERR);

  // Its mirror image.
  makegrid::MagneticFieldResponseTable mirror = table;
  for (int i = 0; i < nextcur; ++i) {
    for (int k = 0; k < num_phi; ++k) {
      const int k_source = (num_phi - k) % num_phi;
      for (int z = 0; z < num_z; ++z) {
        const int z_source = num_z - 1 - z;
        for (int r = 0; r < num_r; ++r) {
          const int index = (k * num_z + z) * num_r + r;
          const int source = (k_source * num_z + z_source) * num_r + r;
          mirror.b_r(i, index) = -table.b_r(i, source);
          mirror.b_p(i, index) = table.b_p(i, source);
          mirror.b_z(i, index) = table.b_z(i, source);
        }
      }
    }
  }

  const auto original = vmecpp::run(indata, table);
  ASSERT_TRUE(original.ok()) << original.status();
  const auto mirrored = vmecpp::run(indata, mirror);
  ASSERT_TRUE(mirrored.ok()) << mirrored.status();
  const vmecpp::WOutFileContents& a = original->wout;
  const vmecpp::WOutFileContents& b = mirrored->wout;

  // The coil field is asymmetric enough to give the equilibrium a measurable
  // antisymmetric part; otherwise the sign checks below would be empty.
  ASSERT_GT(a.rmns.cwiseAbs().maxCoeff(), 5.0e-4);
  ASSERT_GT(a.zmnc.cwiseAbs().maxCoeff(), 5.0e-4);

  const double scalar_tolerance = 1.0e-10;
  EXPECT_TRUE(IsCloseRelAbs(a.volume, b.volume, scalar_tolerance));
  EXPECT_TRUE(IsCloseRelAbs(a.aspect, b.aspect, scalar_tolerance));
  EXPECT_TRUE(IsCloseRelAbs(a.wb, b.wb, scalar_tolerance));
  EXPECT_TRUE(IsCloseRelAbs(a.rmax_surf, b.rmax_surf, scalar_tolerance));
  EXPECT_TRUE(IsCloseRelAbs(a.rmin_surf, b.rmin_surf, scalar_tolerance));
  EXPECT_TRUE(IsCloseRelAbs(a.zmax_surf, b.zmax_surf, scalar_tolerance));

  // Symmetric arrays are unchanged, antisymmetric arrays change sign.
  const double array_tolerance = 1.0e-9;
  const auto same = [&](const vmecpp::RowMatrixXd& x,
                        const vmecpp::RowMatrixXd& y, const char* name) {
    EXPECT_LE((x - y).cwiseAbs().maxCoeff(),
              array_tolerance * x.cwiseAbs().maxCoeff())
        << name;
  };
  const auto negated = [&](const vmecpp::RowMatrixXd& x,
                           const vmecpp::RowMatrixXd& y, const char* name) {
    EXPECT_LE((x + y).cwiseAbs().maxCoeff(),
              array_tolerance * x.cwiseAbs().maxCoeff())
        << name;
  };
  same(a.rmnc, b.rmnc, "rmnc");
  same(a.zmns, b.zmns, "zmns");
  same(a.lmns, b.lmns, "lmns");
  same(a.bmnc, b.bmnc, "bmnc");
  negated(a.rmns, b.rmns, "rmns");
  negated(a.zmnc, b.zmnc, "zmnc");
  negated(a.lmnc, b.lmnc, "lmnc");
  negated(a.bmns, b.bmns, "bmns");
}

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
  CompareWOut(output_with_inmemory_mgrid->wout, original_output->wout,
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
  CompareWOut(inmemory_output->wout, disk_output->wout,
              /*tolerance=*/1e-7, /*check_equal_niter=*/true,
              /*current_density_tolerance=*/2e-7);
}

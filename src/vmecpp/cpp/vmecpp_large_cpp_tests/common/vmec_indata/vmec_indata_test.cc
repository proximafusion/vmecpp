// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/common/vmec_indata/vmec_indata.h"

#include <H5File.h>

#include <filesystem>
#include <string>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"

namespace fs = std::filesystem;

namespace vmecpp {

using ::file_io::ReadFile;

using composed_types::CoefficientsRCos;
using composed_types::CoefficientsRSin;
using composed_types::CoefficientsZCos;
using composed_types::CoefficientsZSin;
using composed_types::CurveRZFourier;
using composed_types::CurveRZFourierFromCsv;
using composed_types::SurfaceRZFourier;
using composed_types::SurfaceRZFourierFromCsv;

using nlohmann::json;

using testing::ElementsAre;
using testing::ElementsAreArray;

// The tests below are setup to check that the input file contents were
// correctly parsed. The reference values come from manual parsing, i.e.,
// looking at the Fortran input file and copying over the values by hand.
// The purpose of these tests is to make sure that for all input files,
// the whole chain of (Fortran input file) -> (indata2json) -> (VMEC++ JSON
// input file)
// -> (JSON parsing) -> (VmecINDATA setup from JSON) works.
// As long as these tests work, we can be sure that inputs that the Reference
// Fortran VMEC saw and thus the corresponding Fortran reference data is
// actually what VMEC++ is to be compared against if given the JSON input file
// under test here.

TEST(TestVmecINDATA, CheckParsingSolovev) {
  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/solovev.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // numerical resolution, symmetry assumption
  EXPECT_EQ(vmec_indata->lasym, false);
  EXPECT_EQ(vmec_indata->nfp, 1);
  EXPECT_EQ(vmec_indata->mpol, 6);
  EXPECT_EQ(vmec_indata->ntor, 0);
  EXPECT_EQ(vmec_indata->ntheta, 0);
  EXPECT_EQ(vmec_indata->nzeta, 0);

  // multi-grid steps
  EXPECT_THAT(vmec_indata->ns_array, ElementsAre(5, 11, 55));
  EXPECT_THAT(vmec_indata->ftol_array, ElementsAre(1.0e-12, 1.0e-12, 1.0e-12));
  EXPECT_THAT(vmec_indata->niter_array, ElementsAre(1000, 2000, 2000));

  // global physics parameters
  EXPECT_EQ(vmec_indata->phiedge, 1.0);
  EXPECT_EQ(vmec_indata->ncurr, 0);

  // mass / pressure profile
  EXPECT_EQ(vmec_indata->pmass_type, "power_series");
  EXPECT_THAT(vmec_indata->am, ElementsAre(0.125, -0.125));
  EXPECT_EQ(vmec_indata->am_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->am_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->pres_scale, 1.0);
  EXPECT_EQ(vmec_indata->gamma, 0.0);
  EXPECT_EQ(vmec_indata->spres_ped, 1.0);

  // (initial guess for) iota profile
  EXPECT_EQ(vmec_indata->piota_type, "power_series");
  EXPECT_THAT(vmec_indata->ai, ElementsAre(1.0));
  EXPECT_EQ(vmec_indata->ai_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_f.size(), 0);

  // enclosed toroidal current profile
  EXPECT_EQ(vmec_indata->pcurr_type, "power_series");
  EXPECT_EQ(vmec_indata->ac.size(), 0);
  EXPECT_EQ(vmec_indata->ac_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ac_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->curtor, 0.0);
  EXPECT_EQ(vmec_indata->bloat, 1.0);

  // free-boundary parameters
  EXPECT_EQ(vmec_indata->lfreeb, false);
  EXPECT_EQ(vmec_indata->mgrid_file, "NONE");
  EXPECT_EQ(vmec_indata->extcur.size(), 0);
  EXPECT_EQ(vmec_indata->nvacskip, 1);

  // tweaking parameters
  EXPECT_EQ(vmec_indata->nstep, 250);
  EXPECT_THAT(vmec_indata->aphi, ElementsAre(1.0));
  EXPECT_EQ(vmec_indata->delt, 0.9);
  EXPECT_EQ(vmec_indata->tcon0, 1.0);
  EXPECT_EQ(vmec_indata->lforbal, false);

  // initial guess for magnetic axis
  absl::StatusOr<std::string> axis_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/axis_coefficients_solovev.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());

  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(*vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(*vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->raxis_s.has_value());
    EXPECT_FALSE(vmec_indata->zaxis_c.has_value());
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/boundary_coefficients_solovev.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc.transpose().reshaped(),
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs.transpose().reshaped(),
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs->transpose().reshaped(),
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc->transpose().reshaped(),
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->rbs.has_value());
    EXPECT_FALSE(vmec_indata->zbc.has_value());
  }
}  // CheckParsingSolovev

TEST(TestVmecINDATA, CheckParsingSolovevAnalytical) {
  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/solovev_analytical.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // numerical resolution, symmetry assumption
  EXPECT_EQ(vmec_indata->lasym, false);
  EXPECT_EQ(vmec_indata->nfp, 1);
  EXPECT_EQ(vmec_indata->mpol, 13);
  EXPECT_EQ(vmec_indata->ntor, 0);
  EXPECT_EQ(vmec_indata->ntheta, 0);
  EXPECT_EQ(vmec_indata->nzeta, 0);

  // multi-grid steps
  EXPECT_THAT(vmec_indata->ns_array, ElementsAre(31));
  EXPECT_THAT(vmec_indata->ftol_array, ElementsAre(1.0e-16));
  EXPECT_THAT(vmec_indata->niter_array, ElementsAre(2000));

  // global physics parameters
  EXPECT_EQ(vmec_indata->phiedge, 3.141592653590);
  EXPECT_EQ(vmec_indata->ncurr, 1);

  // mass / pressure profile
  EXPECT_EQ(vmec_indata->pmass_type, "power_series");
  EXPECT_THAT(vmec_indata->am, ElementsAre(1.0, -1.0));
  EXPECT_EQ(vmec_indata->am_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->am_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->pres_scale, 9.947183943243e+04);
  EXPECT_EQ(vmec_indata->gamma, 0.0);
  EXPECT_EQ(vmec_indata->spres_ped, 1.0);

  // (initial guess for) iota profile
  EXPECT_EQ(vmec_indata->piota_type, "power_series");
  EXPECT_EQ(vmec_indata->ai.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_f.size(), 0);

  // enclosed toroidal current profile
  EXPECT_EQ(vmec_indata->pcurr_type, "power_series");
  EXPECT_THAT(
      vmec_indata->ac,
      ElementsAre(9.798989768026e-01, 3.499639202867e-02, 6.561823505375e-03,
                  1.367046563620e-03, 2.990414357918e-04, 6.728432305316e-05,
                  1.541932403302e-05, 3.579485936236e-06, 8.389420163053e-07,
                  1.980835316276e-07, 4.704483876156e-08, 1.122660924992e-08,
                  2.689708466126e-09, 6.465645351265e-10));
  EXPECT_EQ(vmec_indata->ac_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ac_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->curtor, 2.823753282890e+06);
  EXPECT_EQ(vmec_indata->bloat, 1.0);

  // free-boundary parameters
  EXPECT_EQ(vmec_indata->lfreeb, false);
  EXPECT_EQ(vmec_indata->mgrid_file, "NONE");
  EXPECT_EQ(vmec_indata->extcur.size(), 0);
  EXPECT_EQ(vmec_indata->nvacskip, 1);

  // tweaking parameters
  EXPECT_EQ(vmec_indata->nstep, 100);
  EXPECT_THAT(vmec_indata->aphi, ElementsAre(1.0));
  EXPECT_EQ(vmec_indata->delt, 1.0);
  EXPECT_EQ(vmec_indata->tcon0, 1.0);
  EXPECT_EQ(vmec_indata->lforbal, false);

  // initial guess for magnetic axis
  absl::StatusOr<std::string> axis_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/"
      "axis_coefficients_solovev_analytical.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok())
      << axis_coefficients_csv.status().message();

  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(*vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(*vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->raxis_s.has_value());
    EXPECT_FALSE(vmec_indata->zaxis_c.has_value());
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/"
      "boundary_coefficients_solovev_analytical.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc.transpose().reshaped(),
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs.transpose().reshaped(),
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs->transpose().reshaped(),
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc->transpose().reshaped(),
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->rbs.has_value());
    EXPECT_FALSE(vmec_indata->zbc.has_value());
  }
}  // CheckParsingSolovevAnalytical

TEST(TestVmecINDATA, CheckParsingSolovevNoAxis) {
  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/solovev_no_axis.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // numerical resolution, symmetry assumption
  EXPECT_EQ(vmec_indata->lasym, false);
  EXPECT_EQ(vmec_indata->nfp, 1);
  EXPECT_EQ(vmec_indata->mpol, 6);
  EXPECT_EQ(vmec_indata->ntor, 0);
  EXPECT_EQ(vmec_indata->ntheta, 0);
  EXPECT_EQ(vmec_indata->nzeta, 0);

  // multi-grid steps
  EXPECT_THAT(vmec_indata->ns_array, ElementsAre(5, 11));
  EXPECT_THAT(vmec_indata->ftol_array, ElementsAre(1.0e-12, 1.0e-12));
  EXPECT_THAT(vmec_indata->niter_array, ElementsAre(1000, 2000));

  // global physics parameters
  EXPECT_EQ(vmec_indata->phiedge, 1.0);
  EXPECT_EQ(vmec_indata->ncurr, 0);

  // mass / pressure profile
  EXPECT_EQ(vmec_indata->pmass_type, "power_series");
  EXPECT_THAT(vmec_indata->am, ElementsAre(0.125, -0.125));
  EXPECT_EQ(vmec_indata->am_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->am_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->pres_scale, 1.0);
  EXPECT_EQ(vmec_indata->gamma, 0.0);
  EXPECT_EQ(vmec_indata->spres_ped, 1.0);

  // (initial guess for) iota profile
  EXPECT_EQ(vmec_indata->piota_type, "power_series");
  EXPECT_THAT(vmec_indata->ai, ElementsAre(1.0));
  EXPECT_EQ(vmec_indata->ai_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_f.size(), 0);

  // enclosed toroidal current profile
  EXPECT_EQ(vmec_indata->pcurr_type, "power_series");
  EXPECT_EQ(vmec_indata->ac.size(), 0);
  EXPECT_EQ(vmec_indata->ac_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ac_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->curtor, 0.0);
  EXPECT_EQ(vmec_indata->bloat, 1.0);

  // free-boundary parameters
  EXPECT_EQ(vmec_indata->lfreeb, false);
  EXPECT_EQ(vmec_indata->mgrid_file, "NONE");
  EXPECT_EQ(vmec_indata->extcur.size(), 0);
  EXPECT_EQ(vmec_indata->nvacskip, 1);

  // tweaking parameters
  EXPECT_EQ(vmec_indata->nstep, 250);
  EXPECT_THAT(vmec_indata->aphi, ElementsAre(1.0));
  EXPECT_EQ(vmec_indata->delt, 0.9);
  EXPECT_EQ(vmec_indata->tcon0, 1.0);
  EXPECT_EQ(vmec_indata->lforbal, false);

  // initial guess for magnetic axis
  absl::StatusOr<std::string> axis_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/axis_coefficients_solovev_no_axis.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(*vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(*vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->raxis_s.has_value());
    EXPECT_FALSE(vmec_indata->zaxis_c.has_value());
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/"
      "boundary_coefficients_solovev_no_axis.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc.transpose().reshaped(),
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs.transpose().reshaped(),
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs->transpose().reshaped(),
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc->transpose().reshaped(),
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->rbs.has_value());
    EXPECT_FALSE(vmec_indata->zbc.has_value());
  }
}  // CheckParsingSolovevNoAxis

TEST(TestVmecINDATA, CheckParsingCthLikeFixedBoundary) {
  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_fixed_bdy.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // numerical resolution, symmetry assumption
  EXPECT_EQ(vmec_indata->lasym, false);
  EXPECT_EQ(vmec_indata->nfp, 5);
  EXPECT_EQ(vmec_indata->mpol, 5);
  EXPECT_EQ(vmec_indata->ntor, 4);
  EXPECT_EQ(vmec_indata->ntheta, 0);
  EXPECT_EQ(vmec_indata->nzeta, 36);

  // multi-grid steps
  EXPECT_THAT(vmec_indata->ns_array, ElementsAre(25));
  EXPECT_THAT(vmec_indata->ftol_array, ElementsAre(1.0e-6));
  EXPECT_THAT(vmec_indata->niter_array, ElementsAre(25000));

  // global physics parameters
  EXPECT_EQ(vmec_indata->phiedge, -0.035);
  EXPECT_EQ(vmec_indata->ncurr, 1);

  // mass / pressure profile
  EXPECT_EQ(vmec_indata->pmass_type, "two_power");
  EXPECT_THAT(vmec_indata->am, ElementsAre(1.0, 5.0, 10.0));
  EXPECT_EQ(vmec_indata->am_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->am_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->pres_scale, 432.29080924603676);
  EXPECT_EQ(vmec_indata->gamma, 0.0);
  EXPECT_EQ(vmec_indata->spres_ped, 1.0);

  // (initial guess for) iota profile
  EXPECT_EQ(vmec_indata->piota_type, "power_series");
  EXPECT_EQ(vmec_indata->ai.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_f.size(), 0);

  // enclosed toroidal current profile
  EXPECT_EQ(vmec_indata->pcurr_type, "two_power");
  EXPECT_THAT(vmec_indata->ac, ElementsAre(1.0, 5.0, 10.0));
  EXPECT_EQ(vmec_indata->ac_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ac_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->curtor, 43229.08092460368);
  EXPECT_EQ(vmec_indata->bloat, 1.0);

  // free-boundary parameters
  EXPECT_EQ(vmec_indata->lfreeb, false);
  EXPECT_EQ(vmec_indata->mgrid_file, "NONE");
  EXPECT_EQ(vmec_indata->extcur.size(), 0);
  EXPECT_EQ(vmec_indata->nvacskip, 1);

  // tweaking parameters
  EXPECT_EQ(vmec_indata->nstep, 200);
  EXPECT_THAT(vmec_indata->aphi, ElementsAre(1.0));
  EXPECT_EQ(vmec_indata->delt, 0.7);
  EXPECT_EQ(vmec_indata->tcon0, 1.0);
  EXPECT_EQ(vmec_indata->lforbal, false);

  // initial guess for magnetic axis
  absl::StatusOr<std::string> axis_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/"
      "axis_coefficients_cth_like_fixed_bdy.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(*vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(*vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->raxis_s.has_value());
    EXPECT_FALSE(vmec_indata->zaxis_c.has_value());
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/"
      "boundary_coefficients_cth_like_fixed_bdy.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc.transpose().reshaped(),
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs.transpose().reshaped(),
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs->transpose().reshaped(),
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc->transpose().reshaped(),
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->rbs.has_value());
    EXPECT_FALSE(vmec_indata->zbc.has_value());
  }
}  // CheckParsingCthLikeFixedBoundary

TEST(TestVmecINDATA, CheckParsingCthLikeFixedBoundaryNZeta37) {
  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_fixed_bdy_nzeta_37.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // numerical resolution, symmetry assumption
  EXPECT_EQ(vmec_indata->lasym, false);
  EXPECT_EQ(vmec_indata->nfp, 5);
  EXPECT_EQ(vmec_indata->mpol, 5);
  EXPECT_EQ(vmec_indata->ntor, 4);
  EXPECT_EQ(vmec_indata->ntheta, 0);
  EXPECT_EQ(vmec_indata->nzeta, 37);

  // multi-grid steps
  EXPECT_THAT(vmec_indata->ns_array, ElementsAre(25));
  EXPECT_THAT(vmec_indata->ftol_array, ElementsAre(1.0e-6));
  EXPECT_THAT(vmec_indata->niter_array, ElementsAre(25000));

  // global physics parameters
  EXPECT_EQ(vmec_indata->phiedge, -0.035);
  EXPECT_EQ(vmec_indata->ncurr, 1);

  // mass / pressure profile
  EXPECT_EQ(vmec_indata->pmass_type, "two_power");
  EXPECT_THAT(vmec_indata->am, ElementsAre(1.0, 5.0, 10.0));
  EXPECT_EQ(vmec_indata->am_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->am_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->pres_scale, 432.29080924603676);
  EXPECT_EQ(vmec_indata->gamma, 0.0);
  EXPECT_EQ(vmec_indata->spres_ped, 1.0);

  // (initial guess for) iota profile
  EXPECT_EQ(vmec_indata->piota_type, "power_series");
  EXPECT_EQ(vmec_indata->ai.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_f.size(), 0);

  // enclosed toroidal current profile
  EXPECT_EQ(vmec_indata->pcurr_type, "two_power");
  EXPECT_THAT(vmec_indata->ac, ElementsAre(1.0, 5.0, 10.0));
  EXPECT_EQ(vmec_indata->ac_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ac_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->curtor, 43229.08092460368);
  EXPECT_EQ(vmec_indata->bloat, 1.0);

  // free-boundary parameters
  EXPECT_EQ(vmec_indata->lfreeb, false);
  EXPECT_EQ(vmec_indata->mgrid_file, "NONE");
  EXPECT_EQ(vmec_indata->extcur.size(), 0);
  EXPECT_EQ(vmec_indata->nvacskip, 1);

  // tweaking parameters
  EXPECT_EQ(vmec_indata->nstep, 200);
  EXPECT_THAT(vmec_indata->aphi, ElementsAre(1.0));
  EXPECT_EQ(vmec_indata->delt, 0.7);
  EXPECT_EQ(vmec_indata->tcon0, 1.0);
  EXPECT_EQ(vmec_indata->lforbal, false);

  // initial guess for magnetic axis
  absl::StatusOr<std::string> axis_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/"
      "axis_coefficients_cth_like_fixed_bdy_nzeta_37.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(*vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(*vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->raxis_s.has_value());
    EXPECT_FALSE(vmec_indata->zaxis_c.has_value());
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/"
      "boundary_coefficients_cth_like_fixed_bdy_nzeta_37.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc.transpose().reshaped(),
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs.transpose().reshaped(),
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs->transpose().reshaped(),
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc->transpose().reshaped(),
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->rbs.has_value());
    EXPECT_FALSE(vmec_indata->zbc.has_value());
  }
}  // CheckParsingCthLikeFixedBoundaryNZeta37

TEST(TestVmecINDATA, CheckParsingCma) {
  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cma.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // numerical resolution, symmetry assumption
  EXPECT_EQ(vmec_indata->lasym, false);
  EXPECT_EQ(vmec_indata->nfp, 2);
  EXPECT_EQ(vmec_indata->mpol, 5);
  EXPECT_EQ(vmec_indata->ntor, 6);
  EXPECT_EQ(vmec_indata->ntheta, 0);
  EXPECT_EQ(vmec_indata->nzeta, 0);

  // multi-grid steps
  EXPECT_THAT(vmec_indata->ns_array, ElementsAre(25, 51));
  EXPECT_THAT(vmec_indata->ftol_array, ElementsAre(1.0e-6, 1.0e-6));
  EXPECT_THAT(vmec_indata->niter_array, ElementsAre(1000, 60000));

  // global physics parameters
  EXPECT_EQ(vmec_indata->phiedge, 0.03);
  EXPECT_EQ(vmec_indata->ncurr, 1);

  // mass / pressure profile
  EXPECT_EQ(vmec_indata->pmass_type, "power_series");
  EXPECT_THAT(vmec_indata->am, ElementsAre(0.0));
  EXPECT_EQ(vmec_indata->am_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->am_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->pres_scale, 1.0);
  EXPECT_EQ(vmec_indata->gamma, 0.0);
  EXPECT_EQ(vmec_indata->spres_ped, 1.0);

  // (initial guess for) iota profile
  EXPECT_EQ(vmec_indata->piota_type, "power_series");
  EXPECT_EQ(vmec_indata->ai.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_f.size(), 0);

  // enclosed toroidal current profile
  EXPECT_EQ(vmec_indata->pcurr_type, "power_series");
  EXPECT_THAT(vmec_indata->ac, ElementsAre(0.0));
  EXPECT_EQ(vmec_indata->ac_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ac_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->curtor, 0.0);
  EXPECT_EQ(vmec_indata->bloat, 1.0);

  // free-boundary parameters
  EXPECT_EQ(vmec_indata->lfreeb, false);
  EXPECT_EQ(vmec_indata->mgrid_file, "NONE");
  EXPECT_EQ(vmec_indata->extcur.size(), 0);
  EXPECT_EQ(vmec_indata->nvacskip, 1);

  // tweaking parameters
  EXPECT_EQ(vmec_indata->nstep, 200);
  EXPECT_THAT(vmec_indata->aphi, ElementsAre(1.0));
  EXPECT_EQ(vmec_indata->delt, 0.5);
  EXPECT_EQ(vmec_indata->tcon0, 1.0);
  EXPECT_EQ(vmec_indata->lforbal, false);

  // initial guess for magnetic axis
  absl::StatusOr<std::string> axis_coefficients_csv =
      ReadFile("vmecpp_large_cpp_tests/test_data/axis_coefficients_cma.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(*vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(*vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->raxis_s.has_value());
    EXPECT_FALSE(vmec_indata->zaxis_c.has_value());
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/boundary_coefficients_cma.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc.transpose().reshaped(),
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs.transpose().reshaped(),
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs->transpose().reshaped(),
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc->transpose().reshaped(),
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->rbs.has_value());
    EXPECT_FALSE(vmec_indata->zbc.has_value());
  }
}  // CheckParsingCma

TEST(TestVmecINDATA, CheckParsingCthLikeFreeBoundary) {
  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  // numerical resolution, symmetry assumption
  EXPECT_EQ(vmec_indata->lasym, false);
  EXPECT_EQ(vmec_indata->nfp, 5);
  EXPECT_EQ(vmec_indata->mpol, 5);
  EXPECT_EQ(vmec_indata->ntor, 4);
  EXPECT_EQ(vmec_indata->ntheta, 0);
  EXPECT_EQ(vmec_indata->nzeta, 36);

  // multi-grid steps
  EXPECT_THAT(vmec_indata->ns_array, ElementsAre(15));
  EXPECT_THAT(vmec_indata->ftol_array, ElementsAre(1.0e-10));
  EXPECT_THAT(vmec_indata->niter_array, ElementsAre(2500));

  // global physics parameters
  EXPECT_EQ(vmec_indata->phiedge, -0.035);
  EXPECT_EQ(vmec_indata->ncurr, 1);

  // mass / pressure profile
  EXPECT_EQ(vmec_indata->pmass_type, "two_power");
  EXPECT_THAT(vmec_indata->am, ElementsAre(1.0, 5.0, 10.0));
  EXPECT_EQ(vmec_indata->am_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->am_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->pres_scale, 432.29080924603676);
  EXPECT_EQ(vmec_indata->gamma, 0.0);
  EXPECT_EQ(vmec_indata->spres_ped, 1.0);

  // (initial guess for) iota profile
  EXPECT_EQ(vmec_indata->piota_type, "power_series");
  EXPECT_EQ(vmec_indata->ai.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ai_aux_f.size(), 0);

  // enclosed toroidal current profile
  EXPECT_EQ(vmec_indata->pcurr_type, "two_power");
  EXPECT_THAT(vmec_indata->ac, ElementsAre(1.0, 5.0, 10.0));
  EXPECT_EQ(vmec_indata->ac_aux_s.size(), 0);
  EXPECT_EQ(vmec_indata->ac_aux_f.size(), 0);
  EXPECT_EQ(vmec_indata->curtor, 43229.08092460368);
  EXPECT_EQ(vmec_indata->bloat, 1.0);

  // free-boundary parameters
  EXPECT_EQ(vmec_indata->lfreeb, true);
  EXPECT_EQ(vmec_indata->mgrid_file, "vmecpp/test_data/mgrid_cth_like.nc");
  EXPECT_THAT(vmec_indata->extcur, ElementsAre(4700.0, 1000.0));
  EXPECT_EQ(vmec_indata->nvacskip, 9);

  // tweaking parameters
  EXPECT_EQ(vmec_indata->nstep, 100);
  EXPECT_THAT(vmec_indata->aphi, ElementsAre(1.0));
  EXPECT_EQ(vmec_indata->delt, 0.7);
  EXPECT_EQ(vmec_indata->tcon0, 1.0);
  EXPECT_EQ(vmec_indata->lforbal, false);

  // initial guess for magnetic axis
  absl::StatusOr<std::string> axis_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/"
      "axis_coefficients_cth_like_free_bdy.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(*vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(*vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->raxis_s.has_value());
    EXPECT_FALSE(vmec_indata->zaxis_c.has_value());
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv = ReadFile(
      "vmecpp_large_cpp_tests/test_data/"
      "boundary_coefficients_cth_like_free_bdy.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc.transpose().reshaped(),
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs.transpose().reshaped(),
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs->transpose().reshaped(),
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc->transpose().reshaped(),
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_FALSE(vmec_indata->rbs.has_value());
    EXPECT_FALSE(vmec_indata->zbc.has_value());
  }
}  // CheckParsingCthLikeFreeBoundary

}  // namespace vmecpp

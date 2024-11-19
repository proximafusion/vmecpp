#include "vmecpp/common/vmec_indata/vmec_indata.h"

#include <H5File.h>

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/vmec_indata/boundary_from_json.h"
#include "vmecpp/tools/composed_types_lib/composed_types_lib.h"

namespace fs = std::filesystem;

namespace {
std::vector<double> BoundaryFromJson(const nlohmann::json& json,
                                     const std::string& key, int mpol,
                                     int ntor) {
  std::vector<double> coeffs(mpol * (2 * ntor + 1));

  const auto maybe_entries = vmecpp::BoundaryCoefficient::FromJson(json, key);
  EXPECT_TRUE(maybe_entries.ok());
  EXPECT_TRUE(maybe_entries->has_value());

  const std::vector<vmecpp::BoundaryCoefficient> entries =
      maybe_entries->value();
  for (const vmecpp::BoundaryCoefficient& entry : entries) {
    // Fortran order along n: -ntor, -ntor+1, ..., -1, 0, 1, ..., ntor-1, ntor
    if (entry.m < 0 || entry.m >= mpol || entry.n < -ntor || entry.n > ntor) {
      // invalid indices for boundary coefficients in the json input are ignore
      continue;
    }

    const int index_along_n = ntor + entry.n;
    const int flat_index = entry.m * (2 * ntor + 1) + index_along_n;
    coeffs[flat_index] = entry.value;
  }

  return coeffs;
}
}  // namespace

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

using ::nlohmann::json;

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(TestVmecINDATA, CheckParseJsonBoundary) {
  json j =
      R"({"rbc":[{"m":0,"n":0,"value":3.999},{"m":1,"n":0,"value":1.026},{"m":2,"n":0,"value":-0.068}],"string_variable":"test string"})"_json;

  // NOTE: Before we enforce C++20, the order of assignment needs to be
  // consistent with the parsing in BoundaryCoefficient::FromJson. Therefore,
  // this is a little brittle...
  std::vector<BoundaryCoefficient> expected_coefficients = {
      {/*m=*/0, /*n=*/0, /*value=*/3.999},
      {/*m=*/1, /*n=*/0, /*value=*/1.026},
      {/*m=*/2, /*n=*/0, /*value=*/-0.068}};

  // test check for correct type
  auto read_string_as_boundary =
      BoundaryCoefficient::FromJson(j, "string_variable");
  ASSERT_FALSE(read_string_as_boundary.ok());

  // test check for presence
  auto read_non_existent = BoundaryCoefficient::FromJson(j, "i_dont_exist");
  ASSERT_TRUE(read_non_existent.ok());
  ASSERT_FALSE(read_non_existent->has_value());

  // check reading a set of BoundaryCoefficients
  auto boundary = BoundaryCoefficient::FromJson(j, "rbc");
  ASSERT_TRUE(boundary.ok());
  ASSERT_TRUE(boundary->has_value());
  // NOTE: testing::ElementsAreArray does not seems to work for a vector of
  // structs, so resort to testing element-by-element here
  ASSERT_EQ(boundary->value().size(), expected_coefficients.size());
  for (size_t i = 0; i < expected_coefficients.size(); ++i) {
    const BoundaryCoefficient& coefficient = boundary->value()[i];
    EXPECT_EQ(coefficient.n, expected_coefficients[i].n);
    EXPECT_EQ(coefficient.m, expected_coefficients[i].m);
    EXPECT_EQ(coefficient.value, expected_coefficients[i].value);
  }
}  // CheckParseJsonBoundary

// check that all options stay present
TEST(TestVmecINDATA, CheckFreeBoundaryMethodCases) {
  FreeBoundaryMethod free_boundary_method = FreeBoundaryMethod::NESTOR;
  EXPECT_EQ(free_boundary_method, FreeBoundaryMethod::NESTOR);
}

TEST(TestVmecINDATA, CheckFreeBoundaryMethodFromString) {
  absl::StatusOr<FreeBoundaryMethod> status_or_free_boundary_method =
      FreeBoundaryMethodFromString("nestor");
  ASSERT_TRUE(status_or_free_boundary_method.ok());
  EXPECT_EQ(*status_or_free_boundary_method, FreeBoundaryMethod::NESTOR);

  status_or_free_boundary_method = FreeBoundaryMethodFromString("blablubb");
  EXPECT_FALSE(status_or_free_boundary_method.ok());
}

TEST(TestVmecINDATA, CheckFreeBoundaryMethodToString) {
  EXPECT_EQ(ToString(FreeBoundaryMethod::NESTOR), "nestor");
}

TEST(TestVmecINDATA, CheckDefaults) {
  VmecINDATA indata;

  // numerical resolution, symmetry assumption
  EXPECT_EQ(indata.lasym, false);
  EXPECT_EQ(indata.nfp, 1);
  EXPECT_EQ(indata.mpol, 6);
  EXPECT_EQ(indata.ntor, 0);
  EXPECT_EQ(indata.ntheta, 0);
  EXPECT_EQ(indata.nzeta, 0);

  // multi-grid steps
  EXPECT_THAT(indata.ns_array, ElementsAre(31));
  EXPECT_THAT(indata.ftol_array, ElementsAre(1.0e-10));
  EXPECT_THAT(indata.niter_array, ElementsAre(100));

  // global physics parameters
  EXPECT_EQ(indata.phiedge, 1.0);
  EXPECT_EQ(indata.ncurr, 0);

  // mass / pressure profile
  EXPECT_EQ(indata.pmass_type, "power_series");
  EXPECT_EQ(indata.am.size(), 0);
  EXPECT_EQ(indata.am_aux_s.size(), 0);
  EXPECT_EQ(indata.am_aux_f.size(), 0);
  EXPECT_EQ(indata.pres_scale, 1.0);
  EXPECT_EQ(indata.gamma, 0.0);
  EXPECT_EQ(indata.spres_ped, 1.0);

  // (initial guess for) iota profile
  EXPECT_EQ(indata.piota_type, "power_series");
  EXPECT_EQ(indata.ai.size(), 0);
  EXPECT_EQ(indata.ai_aux_s.size(), 0);
  EXPECT_EQ(indata.ai_aux_f.size(), 0);

  // enclosed toroidal current profile
  EXPECT_EQ(indata.pcurr_type, "power_series");
  EXPECT_EQ(indata.ac.size(), 0);
  EXPECT_EQ(indata.ac_aux_s.size(), 0);
  EXPECT_EQ(indata.ac_aux_f.size(), 0);
  EXPECT_EQ(indata.curtor, 0.0);
  EXPECT_EQ(indata.bloat, 1.0);

  // free-boundary parameters
  EXPECT_EQ(indata.lfreeb, false);
  EXPECT_EQ(indata.mgrid_file, "NONE");
  EXPECT_EQ(indata.extcur.size(), 0);
  EXPECT_EQ(indata.nvacskip, 1);

  // tweaking parameters
  EXPECT_EQ(indata.nstep, 10);
  EXPECT_THAT(indata.aphi, ElementsAre(1.0));
  EXPECT_EQ(indata.delt, 1.0);
  EXPECT_EQ(indata.tcon0, 1.0);
  EXPECT_EQ(indata.lforbal, false);

  // initial guess for magnetic axis
  EXPECT_EQ(indata.raxis_c.size(), indata.ntor + 1);
  EXPECT_EQ(indata.zaxis_s.size(), indata.ntor + 1);
  EXPECT_EQ(indata.raxis_s.size(), indata.lasym ? indata.ntor + 1 : 0);
  EXPECT_EQ(indata.zaxis_c.size(), indata.lasym ? indata.ntor + 1 : 0);

  // (initial guess for) boundary shape
  const int bdy_size = indata.mpol * (2 * indata.ntor + 1);
  EXPECT_EQ(indata.rbc.size(), bdy_size);
  EXPECT_EQ(indata.zbs.size(), bdy_size);
  EXPECT_EQ(indata.rbs.size(), indata.lasym ? bdy_size : 0);
  EXPECT_EQ(indata.zbc.size(), indata.lasym ? bdy_size : 0);
}  // CheckDefaults

TEST(TestVmecINDATA, ToJson) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
  ASSERT_TRUE(indata_json.ok());

  absl::StatusOr<VmecINDATA> indata_ = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata_.ok());
  auto& indata = indata_.value();
  ASSERT_TRUE(IsConsistent(indata, /*enable_info_messages=*/false).ok());

  const absl::StatusOr<std::string> indata_as_json = indata.ToJson();
  ASSERT_TRUE(indata_as_json.ok());

  const auto indata_as_json_object = json::parse(*indata_as_json);
  const auto original_as_json_object = json::parse(*indata_json);
  const auto default_indata_as_json_object =
      json::parse(VmecINDATA().ToJson().value());

  for (auto& element : indata_as_json_object.items()) {
    const std::string& key = element.key();
    if (original_as_json_object.contains(key)) {
      if (key == "rbc" || key == "zbs") {
        const std::vector<double> original_bdy = BoundaryFromJson(
            original_as_json_object, key, indata.mpol, indata.ntor);
        const std::vector<double> out_bdy = BoundaryFromJson(
            original_as_json_object, key, indata.mpol, indata.ntor);
        EXPECT_THAT(out_bdy, ElementsAreArray(original_bdy));
      } else {
        EXPECT_EQ(element.value(), original_as_json_object.at(key));
      }
    } else {
      // this is a key from the new json that was not contained in the original
      // one: we expect the default value
      EXPECT_EQ(element.value(), default_indata_as_json_object.at(key));
    }
  }
}  // ToJson

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
  absl::StatusOr<std::string> axis_coefficients_csv =
      ReadFile("vmecpp/test_data/axis_coefficients_solovev.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());

  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->raxis_s.size(), 0);
    EXPECT_EQ(vmec_indata->zaxis_c.size(), 0);
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv =
      ReadFile("vmecpp/test_data/boundary_coefficients_solovev.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc,
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs,
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs,
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc,
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->rbs.size(), 0);
    EXPECT_EQ(vmec_indata->zbc.size(), 0);
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
  absl::StatusOr<std::string> axis_coefficients_csv =
      ReadFile("vmecpp/test_data/axis_coefficients_solovev_analytical.csv");
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
    EXPECT_THAT(vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->raxis_s.size(), 0);
    EXPECT_EQ(vmec_indata->zaxis_c.size(), 0);
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv =
      ReadFile("vmecpp/test_data/boundary_coefficients_solovev_analytical.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc,
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs,
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs,
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc,
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->rbs.size(), 0);
    EXPECT_EQ(vmec_indata->zbc.size(), 0);
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
  absl::StatusOr<std::string> axis_coefficients_csv =
      ReadFile("vmecpp/test_data/axis_coefficients_solovev_no_axis.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->raxis_s.size(), 0);
    EXPECT_EQ(vmec_indata->zaxis_c.size(), 0);
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv =
      ReadFile("vmecpp/test_data/boundary_coefficients_solovev_no_axis.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc,
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs,
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs,
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc,
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->rbs.size(), 0);
    EXPECT_EQ(vmec_indata->zbc.size(), 0);
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
  absl::StatusOr<std::string> axis_coefficients_csv =
      ReadFile("vmecpp/test_data/axis_coefficients_cth_like_fixed_bdy.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->raxis_s.size(), 0);
    EXPECT_EQ(vmec_indata->zaxis_c.size(), 0);
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv =
      ReadFile("vmecpp/test_data/boundary_coefficients_cth_like_fixed_bdy.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc,
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs,
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs,
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc,
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->rbs.size(), 0);
    EXPECT_EQ(vmec_indata->zbc.size(), 0);
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
      "vmecpp/test_data/axis_coefficients_cth_like_fixed_bdy_nzeta_37.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->raxis_s.size(), 0);
    EXPECT_EQ(vmec_indata->zaxis_c.size(), 0);
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv = ReadFile(
      "vmecpp/test_data/boundary_coefficients_cth_like_fixed_bdy_nzeta_37.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc,
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs,
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs,
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc,
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->rbs.size(), 0);
    EXPECT_EQ(vmec_indata->zbc.size(), 0);
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
      ReadFile("vmecpp/test_data/axis_coefficients_cma.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->raxis_s.size(), 0);
    EXPECT_EQ(vmec_indata->zaxis_c.size(), 0);
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv =
      ReadFile("vmecpp/test_data/boundary_coefficients_cma.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc,
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs,
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs,
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc,
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->rbs.size(), 0);
    EXPECT_EQ(vmec_indata->zbc.size(), 0);
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
  absl::StatusOr<std::string> axis_coefficients_csv =
      ReadFile("vmecpp/test_data/axis_coefficients_cth_like_free_bdy.csv");
  ASSERT_TRUE(axis_coefficients_csv.ok());
  absl::StatusOr<CurveRZFourier> axis_coefficients =
      CurveRZFourierFromCsv(*axis_coefficients_csv);
  ASSERT_TRUE(axis_coefficients.ok());
  EXPECT_THAT(vmec_indata->raxis_c,
              ElementsAreArray(*CoefficientsRCos(*axis_coefficients)));
  EXPECT_THAT(vmec_indata->zaxis_s,
              ElementsAreArray(*CoefficientsZSin(*axis_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->raxis_s,
                ElementsAreArray(*CoefficientsRSin(*axis_coefficients)));
    EXPECT_THAT(vmec_indata->zaxis_c,
                ElementsAreArray(*CoefficientsZCos(*axis_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->raxis_s.size(), 0);
    EXPECT_EQ(vmec_indata->zaxis_c.size(), 0);
  }

  // (initial guess for) boundary shape
  absl::StatusOr<std::string> boundary_coefficients_csv =
      ReadFile("vmecpp/test_data/boundary_coefficients_cth_like_free_bdy.csv");
  ASSERT_TRUE(boundary_coefficients_csv.ok());
  absl::StatusOr<SurfaceRZFourier> boundary_coefficients =
      SurfaceRZFourierFromCsv(*boundary_coefficients_csv);
  ASSERT_TRUE(boundary_coefficients.ok());
  EXPECT_THAT(vmec_indata->rbc,
              ElementsAreArray(*CoefficientsRCos(*boundary_coefficients)));
  EXPECT_THAT(vmec_indata->zbs,
              ElementsAreArray(*CoefficientsZSin(*boundary_coefficients)));
  if (vmec_indata->lasym) {
    EXPECT_THAT(vmec_indata->rbs,
                ElementsAreArray(*CoefficientsRSin(*boundary_coefficients)));
    EXPECT_THAT(vmec_indata->zbc,
                ElementsAreArray(*CoefficientsZCos(*boundary_coefficients)));
  } else {
    EXPECT_EQ(vmec_indata->rbs.size(), 0);
    EXPECT_EQ(vmec_indata->zbc.size(), 0);
  }
}  // CheckParsingCthLikeFreeBoundary

TEST(TestVmecINDATA, HDF5IO) {
  // setup
  absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> maybe_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(maybe_indata.ok());
  const auto& indata = maybe_indata.value();

  const fs::path test_dir = std::filesystem::temp_directory_path() /
                            ("vmecpp_tests_" + std::to_string(getpid()));
  std::error_code err;
  fs::create_directory(test_dir, err);
  ASSERT_FALSE(err) << "Could not create test directory " << test_dir
                    << ", error was: " << err.message();

  // write out...
  const fs::path fname = test_dir / "wout_filecontents_test.h5";
  H5::H5File file(fname, H5F_ACC_TRUNC);
  absl::Status s = indata.WriteTo(file);
  ASSERT_TRUE(s.ok()) << s;

  // ...and read back
  VmecINDATA indata_from_file;
  s = VmecINDATA::LoadInto(indata_from_file, file);
  ASSERT_TRUE(s.ok()) << s;

  EXPECT_EQ(indata.lasym, indata_from_file.lasym);
  EXPECT_EQ(indata.nfp, indata_from_file.nfp);
  EXPECT_EQ(indata.mpol, indata_from_file.mpol);
  EXPECT_EQ(indata.ntor, indata_from_file.ntor);
  EXPECT_EQ(indata.ntheta, indata_from_file.ntheta);
  EXPECT_EQ(indata.nzeta, indata_from_file.nzeta);
  EXPECT_EQ(indata.ns_array, indata_from_file.ns_array);
  EXPECT_EQ(indata.ftol_array, indata_from_file.ftol_array);
  EXPECT_EQ(indata.niter_array, indata_from_file.niter_array);
  EXPECT_EQ(indata.phiedge, indata_from_file.phiedge);
  EXPECT_EQ(indata.ncurr, indata_from_file.ncurr);
  EXPECT_EQ(indata.pmass_type, indata_from_file.pmass_type);
  EXPECT_EQ(indata.am, indata_from_file.am);
  EXPECT_EQ(indata.am_aux_s, indata_from_file.am_aux_s);
  EXPECT_EQ(indata.am_aux_f, indata_from_file.am_aux_f);
  EXPECT_EQ(indata.pres_scale, indata_from_file.pres_scale);
  EXPECT_EQ(indata.gamma, indata_from_file.gamma);
  EXPECT_EQ(indata.spres_ped, indata_from_file.spres_ped);
  EXPECT_EQ(indata.piota_type, indata_from_file.piota_type);
  EXPECT_EQ(indata.ai, indata_from_file.ai);
  EXPECT_EQ(indata.ai_aux_s, indata_from_file.ai_aux_s);
  EXPECT_EQ(indata.ai_aux_f, indata_from_file.ai_aux_f);
  EXPECT_EQ(indata.pcurr_type, indata_from_file.pcurr_type);
  EXPECT_EQ(indata.ac, indata_from_file.ac);
  EXPECT_EQ(indata.ac_aux_s, indata_from_file.ac_aux_s);
  EXPECT_EQ(indata.ac_aux_f, indata_from_file.ac_aux_f);
  EXPECT_EQ(indata.curtor, indata_from_file.curtor);
  EXPECT_EQ(indata.bloat, indata_from_file.bloat);
  EXPECT_EQ(indata.lfreeb, indata_from_file.lfreeb);
  EXPECT_EQ(indata.mgrid_file, indata_from_file.mgrid_file);
  EXPECT_EQ(indata.extcur, indata_from_file.extcur);
  EXPECT_EQ(indata.nvacskip, indata_from_file.nvacskip);
  EXPECT_EQ(indata.free_boundary_method, indata_from_file.free_boundary_method);
  EXPECT_EQ(indata.nstep, indata_from_file.nstep);
  EXPECT_EQ(indata.aphi, indata_from_file.aphi);
  EXPECT_EQ(indata.delt, indata_from_file.delt);
  EXPECT_EQ(indata.tcon0, indata_from_file.tcon0);
  EXPECT_EQ(indata.lforbal, indata_from_file.lforbal);
  EXPECT_EQ(indata.raxis_c, indata_from_file.raxis_c);
  EXPECT_EQ(indata.zaxis_s, indata_from_file.zaxis_s);
  EXPECT_EQ(indata.raxis_s, indata_from_file.raxis_s);
  EXPECT_EQ(indata.zaxis_c, indata_from_file.zaxis_c);
  EXPECT_EQ(indata.rbc, indata_from_file.rbc);
  EXPECT_EQ(indata.zbs, indata_from_file.zbs);
  EXPECT_EQ(indata.rbs, indata_from_file.rbs);
  EXPECT_EQ(indata.zbc, indata_from_file.zbc);
}

}  // namespace vmecpp

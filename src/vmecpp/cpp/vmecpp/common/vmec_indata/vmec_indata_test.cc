// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
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
#include "vmecpp/common/composed_types_lib/composed_types_lib.h"
#include "vmecpp/common/vmec_indata/boundary_from_json.h"

namespace fs = std::filesystem;

namespace vmecpp {

using file_io::ReadFile;

using composed_types::CoefficientsRCos;
using composed_types::CoefficientsRSin;
using composed_types::CoefficientsZCos;
using composed_types::CoefficientsZSin;
using composed_types::CurveRZFourier;
using composed_types::CurveRZFourierFromCsv;
using composed_types::SurfaceRZFourier;
using composed_types::SurfaceRZFourierFromCsv;

using nlohmann::json;

using testing::DoubleEq;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::Pointwise;

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

  free_boundary_method = FreeBoundaryMethod::BIEST;
  EXPECT_EQ(free_boundary_method, FreeBoundaryMethod::BIEST);
}  // CheckFreeBoundaryMethodCases

TEST(TestVmecINDATA, CheckFreeBoundaryMethodFromString) {
  absl::StatusOr<FreeBoundaryMethod> status_or_free_boundary_method =
      FreeBoundaryMethodFromString("nestor");
  ASSERT_TRUE(status_or_free_boundary_method.ok());
  EXPECT_EQ(*status_or_free_boundary_method, FreeBoundaryMethod::NESTOR);

  status_or_free_boundary_method = FreeBoundaryMethodFromString("biest");
  ASSERT_TRUE(status_or_free_boundary_method.ok());
  EXPECT_EQ(*status_or_free_boundary_method, FreeBoundaryMethod::BIEST);

  status_or_free_boundary_method = FreeBoundaryMethodFromString("blablubb");
  EXPECT_FALSE(status_or_free_boundary_method.ok());
}  // CheckFreeBoundaryMethodFromString

TEST(TestVmecINDATA, CheckFreeBoundaryMethodToString) {
  EXPECT_EQ(ToString(FreeBoundaryMethod::NESTOR), "nestor");
  EXPECT_EQ(ToString(FreeBoundaryMethod::BIEST), "biest");
}  // CheckFreeBoundaryMethodToString

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
  if (indata.lasym) {
    EXPECT_EQ(indata.raxis_s->size(), indata.ntor + 1);
    EXPECT_EQ(indata.zaxis_c->size(), indata.ntor + 1);
  } else {
    EXPECT_FALSE(indata.raxis_s.has_value());
    EXPECT_FALSE(indata.zaxis_c.has_value());
  }

  // (initial guess for) boundary shape
  const int bdy_size = indata.mpol * (2 * indata.ntor + 1);
  EXPECT_EQ(indata.rbc.size(), bdy_size);
  EXPECT_EQ(indata.zbs.size(), bdy_size);
  if (indata.lasym) {
    EXPECT_EQ(indata.rbs->size(), indata.lasym ? bdy_size : 0);
    EXPECT_EQ(indata.zbc->size(), indata.lasym ? bdy_size : 0);
  } else {
    EXPECT_FALSE(indata.rbs.has_value());
    EXPECT_FALSE(indata.zbc.has_value());
  }
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
}  // HDF5IO

TEST(TestVmecINDATA, SetMpolNtor) {
  VmecINDATA indata =
      VmecINDATA::FromFile("vmecpp/test_data/cth_like_free_bdy.json");

  const VmecINDATA old_indata = indata;
  const int old_mpol = indata.mpol;
  const int old_ntor = indata.ntor;

  // test expanding mpol and ntor
  indata.SetMpolNtor(indata.mpol + 1, indata.ntor + 1);
  ASSERT_TRUE(
      IsConsistent(VmecINDATA(indata), /*enable_info_messages=*/false).ok());
  EXPECT_EQ(indata.ntor, old_ntor + 1);
  EXPECT_EQ(indata.mpol, old_mpol + 1);

  // expect same elements as before and plus one zero at the end
  Eigen::VectorXd expected(old_indata.raxis_c.size() + 1);
  expected.head(old_indata.raxis_c.size()) = old_indata.raxis_c;
  expected.tail(1).setZero();
  EXPECT_THAT(indata.raxis_c, Pointwise(DoubleEq(), expected));

  expected.head(old_indata.zaxis_s.size()) = old_indata.zaxis_s;
  EXPECT_THAT(indata.zaxis_s, Pointwise(DoubleEq(), expected));

  // check the 2D coefficients have been zero-padded properly, leaving
  // non-zero elements at the right positions
  EXPECT_EQ(indata.rbc.rows(), indata.mpol);
  EXPECT_EQ(indata.rbc.cols(), 2 * indata.ntor + 1);
  EXPECT_EQ(indata.zbs.rows(), indata.mpol);
  EXPECT_EQ(indata.zbs.cols(), 2 * indata.ntor + 1);
  for (int m = 0; m < old_mpol; ++m) {
    for (int n = -old_ntor; n <= old_ntor; ++n) {
      EXPECT_DOUBLE_EQ(old_indata.rbc(m, old_ntor + n),
                       indata.rbc(m, indata.ntor + n));
      EXPECT_DOUBLE_EQ(old_indata.zbs(m, old_ntor + n),
                       indata.zbs(m, indata.ntor + n));
    }
  }

  // test shrinking mpol and ntor (back to less than the original sizes in
  // old_indata)
  indata.SetMpolNtor(old_mpol - 1, old_ntor - 1);
  ASSERT_TRUE(
      IsConsistent(VmecINDATA(indata), /*enable_info_messages=*/false).ok());
  EXPECT_EQ(indata.ntor, old_ntor - 1);
  EXPECT_EQ(indata.mpol, old_mpol - 1);

  expected = old_indata.raxis_c.head(old_indata.raxis_c.size() - 1);
  EXPECT_THAT(indata.raxis_c, ElementsAreArray(expected));
  expected = old_indata.zaxis_s.head(old_indata.zaxis_s.size() - 1);
  EXPECT_THAT(indata.zaxis_s, ElementsAreArray(expected));

  // check the 2D coefficients have been truncated properly, leaving other
  // elements at the right positions
  EXPECT_EQ(indata.rbc.rows(), indata.mpol);
  EXPECT_EQ(indata.rbc.cols(), 2 * indata.ntor + 1);
  EXPECT_EQ(indata.zbs.rows(), indata.mpol);
  EXPECT_EQ(indata.zbs.cols(), 2 * indata.ntor + 1);
  for (int m = 0; m < indata.mpol; ++m) {
    for (int n = 0; n < indata.ntor; ++n) {
      EXPECT_DOUBLE_EQ(old_indata.rbc(m, old_ntor + n),
                       indata.rbc(m, indata.ntor + n));
      EXPECT_DOUBLE_EQ(old_indata.zbs(m, old_ntor + n),
                       indata.zbs(m, indata.ntor + n));
    }
  }
}  // SetMpolNtor

TEST(TestVmecINDATA, CopyMethod) {
  const VmecINDATA indata =
      VmecINDATA::FromFile("vmecpp/test_data/cth_like_free_bdy.json");

  const auto copy = indata.Copy();

  // make sure we performed a deep copy
  EXPECT_NE(&copy.lasym, &indata.lasym);
  EXPECT_NE(copy.rbc.data(), indata.rbc.data());

  // make sure the copy went well
  EXPECT_EQ(copy.lasym, indata.lasym);
  EXPECT_EQ(copy.nfp, indata.nfp);
  EXPECT_EQ(copy.mpol, indata.mpol);
  EXPECT_EQ(copy.ntor, indata.ntor);
  EXPECT_EQ(copy.ntheta, indata.ntheta);
  EXPECT_EQ(copy.nzeta, indata.nzeta);
  EXPECT_EQ(copy.ns_array, indata.ns_array);
  EXPECT_EQ(copy.ftol_array, indata.ftol_array);
  EXPECT_EQ(copy.niter_array, indata.niter_array);
  EXPECT_EQ(copy.phiedge, indata.phiedge);
  EXPECT_EQ(copy.ncurr, indata.ncurr);
  EXPECT_EQ(copy.pmass_type, indata.pmass_type);
  EXPECT_EQ(copy.am, indata.am);
  EXPECT_EQ(copy.am_aux_s, indata.am_aux_s);
  EXPECT_EQ(copy.am_aux_f, indata.am_aux_f);
  EXPECT_EQ(copy.pres_scale, indata.pres_scale);
  EXPECT_EQ(copy.gamma, indata.gamma);
  EXPECT_EQ(copy.spres_ped, indata.spres_ped);
  EXPECT_EQ(copy.piota_type, indata.piota_type);
  EXPECT_EQ(copy.ai, indata.ai);
  EXPECT_EQ(copy.ai_aux_s, indata.ai_aux_s);
  EXPECT_EQ(copy.ai_aux_f, indata.ai_aux_f);
  EXPECT_EQ(copy.pcurr_type, indata.pcurr_type);
  EXPECT_EQ(copy.ac, indata.ac);
  EXPECT_EQ(copy.ac_aux_s, indata.ac_aux_s);
  EXPECT_EQ(copy.ac_aux_f, indata.ac_aux_f);
  EXPECT_EQ(copy.curtor, indata.curtor);
  EXPECT_EQ(copy.bloat, indata.bloat);
  EXPECT_EQ(copy.lfreeb, indata.lfreeb);
  EXPECT_EQ(copy.mgrid_file, indata.mgrid_file);
  EXPECT_EQ(copy.extcur, indata.extcur);
  EXPECT_EQ(copy.nvacskip, indata.nvacskip);
  EXPECT_EQ(copy.free_boundary_method, indata.free_boundary_method);
  EXPECT_EQ(copy.nstep, indata.nstep);
  EXPECT_EQ(copy.aphi, indata.aphi);
  EXPECT_EQ(copy.delt, indata.delt);
  EXPECT_EQ(copy.tcon0, indata.tcon0);
  EXPECT_EQ(copy.lforbal, indata.lforbal);
  EXPECT_EQ(copy.iteration_style, indata.iteration_style);
  EXPECT_EQ(copy.return_outputs_even_if_not_converged,
            indata.return_outputs_even_if_not_converged);
  EXPECT_EQ(copy.raxis_c, indata.raxis_c);
  EXPECT_EQ(copy.zaxis_s, indata.zaxis_s);
  EXPECT_EQ(copy.raxis_s, indata.raxis_s);
  EXPECT_EQ(copy.zaxis_c, indata.zaxis_c);
  EXPECT_EQ(copy.rbc, indata.rbc);
  EXPECT_EQ(copy.zbs, indata.zbs);
  EXPECT_EQ(copy.rbs, indata.rbs);
  EXPECT_EQ(copy.zbc, indata.zbc);
}  // CopyMethod

}  // namespace vmecpp

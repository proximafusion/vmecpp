// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/pybind11/vmec_indata_pywrapper.h"

#include <Eigen/Dense>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"

using Eigen::VectorXd;
using file_io::ReadFile;
using testing::DoubleEq;
using testing::ElementsAreArray;
using testing::Pointwise;

namespace {
void CheckEquality(const vmecpp::VmecINDATAPyWrapper &wrapper,
                   const vmecpp::VmecINDATA &indata) {
  EXPECT_EQ(wrapper.lasym, indata.lasym);
  EXPECT_EQ(wrapper.nfp, indata.nfp);
  EXPECT_EQ(wrapper.mpol, indata.mpol);
  EXPECT_EQ(wrapper.ntor, indata.ntor);
  EXPECT_EQ(wrapper.ntheta, indata.ntheta);
  EXPECT_EQ(wrapper.nzeta, indata.nzeta);
  EXPECT_THAT(wrapper.ns_array, ElementsAreArray(indata.ns_array));
  EXPECT_THAT(wrapper.ftol_array, ElementsAreArray(indata.ftol_array));
  EXPECT_THAT(wrapper.niter_array, ElementsAreArray(indata.niter_array));
  EXPECT_EQ(wrapper.phiedge, indata.phiedge);
  EXPECT_EQ(wrapper.ncurr, indata.ncurr);
  EXPECT_EQ(wrapper.pmass_type, indata.pmass_type);
  EXPECT_THAT(wrapper.am, ElementsAreArray(indata.am));
  EXPECT_THAT(wrapper.am_aux_s, ElementsAreArray(indata.am_aux_s));
  EXPECT_THAT(wrapper.am_aux_f, ElementsAreArray(indata.am_aux_f));
  EXPECT_EQ(wrapper.pres_scale, indata.pres_scale);
  EXPECT_EQ(wrapper.gamma, indata.gamma);
  EXPECT_EQ(wrapper.spres_ped, indata.spres_ped);
  EXPECT_EQ(wrapper.piota_type, indata.piota_type);
  EXPECT_THAT(wrapper.ai, ElementsAreArray(indata.ai));
  EXPECT_THAT(wrapper.ai_aux_s, ElementsAreArray(indata.ai_aux_s));
  EXPECT_THAT(wrapper.ai_aux_f, ElementsAreArray(indata.ai_aux_f));
  EXPECT_EQ(wrapper.pcurr_type, indata.pcurr_type);
  EXPECT_THAT(wrapper.ac, ElementsAreArray(indata.ac));
  EXPECT_THAT(wrapper.ac_aux_s, ElementsAreArray(indata.ac_aux_s));
  EXPECT_THAT(wrapper.ac_aux_f, ElementsAreArray(indata.ac_aux_f));
  EXPECT_EQ(wrapper.curtor, indata.curtor);
  EXPECT_EQ(wrapper.bloat, indata.bloat);
  EXPECT_EQ(wrapper.lfreeb, indata.lfreeb);
  EXPECT_EQ(wrapper.mgrid_file, indata.mgrid_file);
  EXPECT_THAT(wrapper.extcur, ElementsAreArray(indata.extcur));
  EXPECT_EQ(wrapper.nvacskip, indata.nvacskip);
  EXPECT_EQ(wrapper.free_boundary_method, indata.free_boundary_method);
  EXPECT_EQ(wrapper.nstep, indata.nstep);
  EXPECT_THAT(wrapper.aphi, ElementsAreArray(indata.aphi));
  EXPECT_EQ(wrapper.delt, indata.delt);
  EXPECT_EQ(wrapper.tcon0, indata.tcon0);
  EXPECT_EQ(wrapper.lforbal, indata.lforbal);
  EXPECT_THAT(wrapper.raxis_c, ElementsAreArray(indata.raxis_c));
  EXPECT_THAT(wrapper.zaxis_s, ElementsAreArray(indata.zaxis_s));
  if (indata.lasym) {
    EXPECT_THAT(wrapper.raxis_s.value(), ElementsAreArray(indata.raxis_s));
    EXPECT_THAT(wrapper.zaxis_c.value(), ElementsAreArray(indata.zaxis_c));
  } else {
    // For now, when stellaratory symmetric, VmecINDATA sets the asymmetric
    // elements to an empty array and VmecINDATAPyWrapper uses std::nullopt.
    // TODO(jurasic): reconcile the two
    EXPECT_EQ(indata.raxis_s.size(), 0);
    EXPECT_TRUE(!wrapper.raxis_s.has_value());
    EXPECT_EQ(indata.zaxis_c.size(), 0);
    EXPECT_TRUE(!wrapper.zaxis_c.has_value());
  }
  const auto flat_rbc = wrapper.rbc.reshaped<Eigen::RowMajor>();
  EXPECT_THAT(flat_rbc, ElementsAreArray(indata.rbc));
  const auto flat_zbs = wrapper.zbs.reshaped<Eigen::RowMajor>();
  EXPECT_THAT(flat_zbs, ElementsAreArray(indata.zbs));

  if (indata.lasym) {
    const auto flat_rbs = wrapper.rbs.value().reshaped<Eigen::RowMajor>();
    EXPECT_THAT(flat_rbs, ElementsAreArray(indata.rbs));
    const auto flat_zbc = wrapper.zbc.value().reshaped<Eigen::RowMajor>();
    EXPECT_THAT(flat_zbc, ElementsAreArray(indata.zbc));
  } else {
    // For now, when stellaratory symmetric, VmecINDATA sets the asymmetric
    // elements to an empty array and VmecINDATAPyWrapper uses std::nullopt.
    // TODO(jurasic): reconcile the two
    EXPECT_EQ(indata.rbs.size(), 0);
    EXPECT_TRUE(!wrapper.rbs.has_value());
    EXPECT_EQ(indata.zbc.size(), 0);
    EXPECT_TRUE(!wrapper.zbc.has_value());
  }
}
}  // namespace

namespace vmecpp {

TEST(VmecINDATAPyWrapper, FromVmecINDATA) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  VmecINDATAPyWrapper wrapper(*indata);
  CheckEquality(wrapper, *indata);
}

TEST(VmecINDATAPyWrapper, FromJson) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/cth_like_free_bdy.json");
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata.ok());

  const auto wrapper =
      VmecINDATAPyWrapper::FromFile("vmecpp/test_data/cth_like_free_bdy.json");
  CheckEquality(wrapper, *indata);
}

TEST(VmecINDATAPyWrapper, ToVmecINDATA) {
  const auto wrapper =
      VmecINDATAPyWrapper::FromFile("vmecpp/test_data/cth_like_free_bdy.json");

  VmecINDATA indata_from_wrapper(wrapper);

  CheckEquality(wrapper, indata_from_wrapper);
}

TEST(VmecINDATAPyWrapper, ToJson) {
  const auto wrapper =
      VmecINDATAPyWrapper::FromFile("vmecpp/test_data/cth_like_free_bdy.json");

  const std::string wrapper_as_json = wrapper.ToJson();
  const auto indata_from_json = VmecINDATA::FromJson(wrapper_as_json);
  ASSERT_TRUE(indata_from_json.ok());

  CheckEquality(wrapper, *indata_from_json);
}

TEST(VmecINDATAPyWrapper, SetMpolNtor) {
  VmecINDATAPyWrapper indata =
      VmecINDATAPyWrapper::FromFile("vmecpp/test_data/cth_like_free_bdy.json");

  const VmecINDATAPyWrapper old_indata = indata;
  const int old_mpol = indata.mpol;
  const int old_ntor = indata.ntor;

  // test expanding mpol and ntor
  indata.SetMpolNtor(indata.mpol + 1, indata.ntor + 1);
  ASSERT_TRUE(
      IsConsistent(VmecINDATA(indata), /*enable_info_messages=*/false).ok());
  EXPECT_EQ(indata.ntor, old_ntor + 1);
  EXPECT_EQ(indata.mpol, old_mpol + 1);

  // expect same elements as before and plus one zero at the end
  VectorXd expected(old_indata.raxis_c.size() + 1);
  expected.head(old_indata.raxis_c.size()) = old_indata.raxis_c;
  expected.tail(1).setZero();
  EXPECT_THAT(indata.raxis_c, Pointwise(DoubleEq(), expected));

  expected.head(old_indata.zaxis_s.size()) = old_indata.zaxis_s;
  EXPECT_THAT(indata.zaxis_s, Pointwise(DoubleEq(), expected));

  // check the 2D coefficients have been zero-padded properly, leaving
  // non-zero elements at the right positions
  EXPECT_EQ(indata.rbc.size(), indata.mpol * (2 * indata.ntor + 1));
  EXPECT_EQ(indata.zbs.size(), indata.mpol * (2 * indata.ntor + 1));
  for (int m = 0; m < old_mpol; ++m) {
    for (int n = -old_ntor; n <= old_ntor; ++n) {
      const int old_idx = m * (2 * old_ntor + 1) + (n + old_ntor);
      const int new_idx = m * (2 * indata.ntor + 1) + (n + indata.ntor);
      EXPECT_DOUBLE_EQ(old_indata.rbc(old_idx), indata.rbc(new_idx));
      EXPECT_DOUBLE_EQ(old_indata.zbs(old_idx), indata.zbs(new_idx));
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
  EXPECT_EQ(indata.rbc.size(), indata.mpol * (2 * indata.ntor + 1));
  EXPECT_EQ(indata.zbs.size(), indata.mpol * (2 * indata.ntor + 1));
  for (int m = 0; m < indata.mpol; ++m) {
    for (int n = 0; n < indata.ntor; ++n) {
      const int old_idx = m * (2 * old_ntor + 1) + (n + old_ntor);
      const int new_idx = m * (2 * indata.ntor + 1) + (n + indata.ntor);
      EXPECT_DOUBLE_EQ(old_indata.rbc(old_idx), indata.rbc(new_idx));
      EXPECT_DOUBLE_EQ(old_indata.zbs(old_idx), indata.zbs(new_idx));
    }
  }
}  // SetMpolNtor

TEST(VmecINDATAPyWrapper, CopyMethod) {
  const VmecINDATAPyWrapper indata =
      VmecINDATAPyWrapper::FromFile("vmecpp/test_data/cth_like_free_bdy.json");

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
  EXPECT_EQ(copy.raxis_c, indata.raxis_c);
  EXPECT_EQ(copy.zaxis_s, indata.zaxis_s);
  EXPECT_EQ(copy.raxis_s, indata.raxis_s);
  EXPECT_EQ(copy.zaxis_c, indata.zaxis_c);
  EXPECT_EQ(copy.rbc, indata.rbc);
  EXPECT_EQ(copy.zbs, indata.zbs);
  EXPECT_EQ(copy.rbs, indata.rbs);
  EXPECT_EQ(copy.zbc, indata.zbc);
}

}  // namespace vmecpp

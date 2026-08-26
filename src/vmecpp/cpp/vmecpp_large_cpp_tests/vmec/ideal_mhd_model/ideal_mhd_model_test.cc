// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/vmec/vmec.h"
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

using vmecpp::vmec_algorithm_constants::kEvenParity;
using vmecpp::vmec_algorithm_constants::kOddParity;

namespace vmecpp {
namespace {
using nlohmann::json;

using file_io::ReadFile;
using testing::IsCloseRelAbs;

using ::testing::TestWithParam;
using ::testing::Values;

template <typename VectorA, typename VectorB>
double DotValues(const VectorA& a, const VectorB& b) {
  EXPECT_EQ(a.size(), b.size());
  double result = 0.0;
  const std::size_t size = static_cast<std::size_t>(a.size());
  for (std::size_t i = 0; i < size; ++i) {
    result += a[i] * b[i];
  }
  return result;
}

template <typename Vector>
void FillRandom(Vector& values, std::mt19937& generator) {
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);
  const std::size_t size = static_cast<std::size_t>(values.size());
  for (std::size_t i = 0; i < size; ++i) {
    values[i] = distribution(generator);
  }
}

double RelativeDotError(double lhs, double rhs) {
  return std::abs(lhs - rhs) / std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

double DotGeometry(const FourierGeometry& a, const FourierGeometry& b) {
  return DotValues(a.rmncc, b.rmncc) + DotValues(a.rmnss, b.rmnss) +
         DotValues(a.rmnsc, b.rmnsc) + DotValues(a.rmncs, b.rmncs) +
         DotValues(a.zmnsc, b.zmnsc) + DotValues(a.zmncs, b.zmncs) +
         DotValues(a.zmncc, b.zmncc) + DotValues(a.zmnss, b.zmnss) +
         DotValues(a.lmnsc, b.lmnsc) + DotValues(a.lmncs, b.lmncs) +
         DotValues(a.lmncc, b.lmncc) + DotValues(a.lmnss, b.lmnss);
}

void FillGeometry(FourierGeometry& geometry, std::mt19937& generator) {
  geometry.setZero();
  FillRandom(geometry.rmncc, generator);
  FillRandom(geometry.rmnss, generator);
  FillRandom(geometry.rmnsc, generator);
  FillRandom(geometry.rmncs, generator);
  FillRandom(geometry.zmnsc, generator);
  FillRandom(geometry.zmncs, generator);
  FillRandom(geometry.zmncc, generator);
  FillRandom(geometry.zmnss, generator);
  FillRandom(geometry.lmnsc, generator);
  FillRandom(geometry.lmncs, generator);
  FillRandom(geometry.lmncc, generator);
  FillRandom(geometry.lmnss, generator);
}

void FillSymmetricForce(IdealMhdModel& model, std::mt19937& generator) {
  FillRandom(model.armn_e, generator);
  FillRandom(model.armn_o, generator);
  FillRandom(model.brmn_e, generator);
  FillRandom(model.brmn_o, generator);
  FillRandom(model.azmn_e, generator);
  FillRandom(model.azmn_o, generator);
  FillRandom(model.bzmn_e, generator);
  FillRandom(model.bzmn_o, generator);
  FillRandom(model.blmn_e, generator);
  FillRandom(model.blmn_o, generator);
  FillRandom(model.frcon_e, generator);
  FillRandom(model.frcon_o, generator);
  FillRandom(model.fzcon_e, generator);
  FillRandom(model.fzcon_o, generator);
  FillRandom(model.crmn_e, generator);
  FillRandom(model.crmn_o, generator);
  FillRandom(model.czmn_e, generator);
  FillRandom(model.czmn_o, generator);
  FillRandom(model.clmn_e, generator);
  FillRandom(model.clmn_o, generator);
}

double CheckGeometryTranspose2d(Vmec& vmec, std::mt19937& generator) {
  IdealMhdModel& model = *vmec.m_[0];
  FourierGeometry input = *vmec.physical_x_[0];
  FillGeometry(input, generator);
  model.dft_FourierToReal_2d_symm(input);

  const Eigen::VectorXd r1_e_forward = model.r1_e;
  const Eigen::VectorXd r1_o_forward = model.r1_o;
  const Eigen::VectorXd ru_e_forward = model.ru_e;
  const Eigen::VectorXd ru_o_forward = model.ru_o;
  const Eigen::VectorXd z1_e_forward = model.z1_e;
  const Eigen::VectorXd z1_o_forward = model.z1_o;
  const Eigen::VectorXd zu_e_forward = model.zu_e;
  const Eigen::VectorXd zu_o_forward = model.zu_o;
  const Eigen::VectorXd lu_e_forward = model.lu_e;
  const Eigen::VectorXd lu_o_forward = model.lu_o;
  const Eigen::VectorXd rCon_forward = model.rCon;
  const Eigen::VectorXd zCon_forward = model.zCon;

  Eigen::VectorXd r1_e_bar = model.r1_e;
  Eigen::VectorXd r1_o_bar = model.r1_o;
  Eigen::VectorXd ru_e_bar = model.ru_e;
  Eigen::VectorXd ru_o_bar = model.ru_o;
  Eigen::VectorXd z1_e_bar = model.z1_e;
  Eigen::VectorXd z1_o_bar = model.z1_o;
  Eigen::VectorXd zu_e_bar = model.zu_e;
  Eigen::VectorXd zu_o_bar = model.zu_o;
  Eigen::VectorXd lu_e_bar = model.lu_e;
  Eigen::VectorXd lu_o_bar = model.lu_o;
  Eigen::VectorXd rCon_bar = model.rCon;
  Eigen::VectorXd zCon_bar = model.zCon;
  FillRandom(r1_e_bar, generator);
  FillRandom(r1_o_bar, generator);
  FillRandom(ru_e_bar, generator);
  FillRandom(ru_o_bar, generator);
  FillRandom(z1_e_bar, generator);
  FillRandom(z1_o_bar, generator);
  FillRandom(zu_e_bar, generator);
  FillRandom(zu_o_bar, generator);
  FillRandom(lu_e_bar, generator);
  FillRandom(lu_o_bar, generator);
  FillRandom(rCon_bar, generator);
  FillRandom(zCon_bar, generator);
  model.r1_e = r1_e_bar;
  model.r1_o = r1_o_bar;
  model.ru_e = ru_e_bar;
  model.ru_o = ru_o_bar;
  model.z1_e = z1_e_bar;
  model.z1_o = z1_o_bar;
  model.zu_e = zu_e_bar;
  model.zu_o = zu_o_bar;
  model.lu_e = lu_e_bar;
  model.lu_o = lu_o_bar;
  model.rCon = rCon_bar;
  model.zCon = zCon_bar;

  FourierGeometry transpose = *vmec.physical_x_[0];
  transpose.setZero();
  model.dft_FourierToRealTranspose_2d_symm(transpose);
  const double lhs =
      DotValues(r1_e_forward, r1_e_bar) + DotValues(r1_o_forward, r1_o_bar) +
      DotValues(ru_e_forward, ru_e_bar) + DotValues(ru_o_forward, ru_o_bar) +
      DotValues(z1_e_forward, z1_e_bar) + DotValues(z1_o_forward, z1_o_bar) +
      DotValues(zu_e_forward, zu_e_bar) + DotValues(zu_o_forward, zu_o_bar) +
      DotValues(lu_e_forward, lu_e_bar) + DotValues(lu_o_forward, lu_o_bar) +
      DotValues(rCon_forward, rCon_bar) + DotValues(zCon_forward, zCon_bar);
  const double rhs = DotValues(input.rmncc, transpose.rmncc) +
                     DotValues(input.zmnsc, transpose.zmnsc) +
                     DotValues(input.lmnsc, transpose.lmnsc);
  return RelativeDotError(lhs, rhs);
}

double CheckGeometryTranspose3d(Vmec& vmec, std::mt19937& generator) {
  IdealMhdModel& model = *vmec.m_[0];
  FourierGeometry input = *vmec.physical_x_[0];
  FillGeometry(input, generator);
  model.dft_FourierToReal_3d_symm(input);

  const Eigen::VectorXd r1_e_forward = model.r1_e;
  const Eigen::VectorXd r1_o_forward = model.r1_o;
  const Eigen::VectorXd ru_e_forward = model.ru_e;
  const Eigen::VectorXd ru_o_forward = model.ru_o;
  const Eigen::VectorXd rv_e_forward = model.rv_e;
  const Eigen::VectorXd rv_o_forward = model.rv_o;
  const Eigen::VectorXd z1_e_forward = model.z1_e;
  const Eigen::VectorXd z1_o_forward = model.z1_o;
  const Eigen::VectorXd zu_e_forward = model.zu_e;
  const Eigen::VectorXd zu_o_forward = model.zu_o;
  const Eigen::VectorXd zv_e_forward = model.zv_e;
  const Eigen::VectorXd zv_o_forward = model.zv_o;
  const Eigen::VectorXd lu_e_forward = model.lu_e;
  const Eigen::VectorXd lu_o_forward = model.lu_o;
  const Eigen::VectorXd lv_e_forward = model.lv_e;
  const Eigen::VectorXd lv_o_forward = model.lv_o;
  const Eigen::VectorXd rCon_forward = model.rCon;
  const Eigen::VectorXd zCon_forward = model.zCon;

  Eigen::VectorXd r1_e_bar = model.r1_e;
  Eigen::VectorXd r1_o_bar = model.r1_o;
  Eigen::VectorXd ru_e_bar = model.ru_e;
  Eigen::VectorXd ru_o_bar = model.ru_o;
  Eigen::VectorXd rv_e_bar = model.rv_e;
  Eigen::VectorXd rv_o_bar = model.rv_o;
  Eigen::VectorXd z1_e_bar = model.z1_e;
  Eigen::VectorXd z1_o_bar = model.z1_o;
  Eigen::VectorXd zu_e_bar = model.zu_e;
  Eigen::VectorXd zu_o_bar = model.zu_o;
  Eigen::VectorXd zv_e_bar = model.zv_e;
  Eigen::VectorXd zv_o_bar = model.zv_o;
  Eigen::VectorXd lu_e_bar = model.lu_e;
  Eigen::VectorXd lu_o_bar = model.lu_o;
  Eigen::VectorXd lv_e_bar = model.lv_e;
  Eigen::VectorXd lv_o_bar = model.lv_o;
  Eigen::VectorXd rCon_bar = model.rCon;
  Eigen::VectorXd zCon_bar = model.zCon;
  FillRandom(r1_e_bar, generator);
  FillRandom(r1_o_bar, generator);
  FillRandom(ru_e_bar, generator);
  FillRandom(ru_o_bar, generator);
  FillRandom(rv_e_bar, generator);
  FillRandom(rv_o_bar, generator);
  FillRandom(z1_e_bar, generator);
  FillRandom(z1_o_bar, generator);
  FillRandom(zu_e_bar, generator);
  FillRandom(zu_o_bar, generator);
  FillRandom(zv_e_bar, generator);
  FillRandom(zv_o_bar, generator);
  FillRandom(lu_e_bar, generator);
  FillRandom(lu_o_bar, generator);
  FillRandom(lv_e_bar, generator);
  FillRandom(lv_o_bar, generator);
  FillRandom(rCon_bar, generator);
  FillRandom(zCon_bar, generator);
  model.r1_e = r1_e_bar;
  model.r1_o = r1_o_bar;
  model.ru_e = ru_e_bar;
  model.ru_o = ru_o_bar;
  model.rv_e = rv_e_bar;
  model.rv_o = rv_o_bar;
  model.z1_e = z1_e_bar;
  model.z1_o = z1_o_bar;
  model.zu_e = zu_e_bar;
  model.zu_o = zu_o_bar;
  model.zv_e = zv_e_bar;
  model.zv_o = zv_o_bar;
  model.lu_e = lu_e_bar;
  model.lu_o = lu_o_bar;
  model.lv_e = lv_e_bar;
  model.lv_o = lv_o_bar;
  model.rCon = rCon_bar;
  model.zCon = zCon_bar;

  FourierGeometry transpose = *vmec.physical_x_[0];
  transpose.setZero();
  model.dft_FourierToRealTranspose_3d_symm(transpose);
  const double lhs =
      DotValues(r1_e_forward, r1_e_bar) + DotValues(r1_o_forward, r1_o_bar) +
      DotValues(ru_e_forward, ru_e_bar) + DotValues(ru_o_forward, ru_o_bar) +
      DotValues(rv_e_forward, rv_e_bar) + DotValues(rv_o_forward, rv_o_bar) +
      DotValues(z1_e_forward, z1_e_bar) + DotValues(z1_o_forward, z1_o_bar) +
      DotValues(zu_e_forward, zu_e_bar) + DotValues(zu_o_forward, zu_o_bar) +
      DotValues(zv_e_forward, zv_e_bar) + DotValues(zv_o_forward, zv_o_bar) +
      DotValues(lu_e_forward, lu_e_bar) + DotValues(lu_o_forward, lu_o_bar) +
      DotValues(lv_e_forward, lv_e_bar) + DotValues(lv_o_forward, lv_o_bar) +
      DotValues(rCon_forward, rCon_bar) + DotValues(zCon_forward, zCon_bar);
  const double rhs = DotValues(input.rmncc, transpose.rmncc) +
                     DotValues(input.rmnss, transpose.rmnss) +
                     DotValues(input.zmnsc, transpose.zmnsc) +
                     DotValues(input.zmncs, transpose.zmncs) +
                     DotValues(input.lmnsc, transpose.lmnsc) +
                     DotValues(input.lmncs, transpose.lmncs);
  return RelativeDotError(lhs, rhs);
}

double CheckForceTranspose2d(Vmec& vmec, std::mt19937& generator) {
  IdealMhdModel& model = *vmec.m_[0];
  FillSymmetricForce(model, generator);
  const Eigen::VectorXd armn_e = model.armn_e;
  const Eigen::VectorXd armn_o = model.armn_o;
  const Eigen::VectorXd brmn_e = model.brmn_e;
  const Eigen::VectorXd brmn_o = model.brmn_o;
  const Eigen::VectorXd azmn_e = model.azmn_e;
  const Eigen::VectorXd azmn_o = model.azmn_o;
  const Eigen::VectorXd bzmn_e = model.bzmn_e;
  const Eigen::VectorXd bzmn_o = model.bzmn_o;
  const Eigen::VectorXd blmn_e = model.blmn_e;
  const Eigen::VectorXd blmn_o = model.blmn_o;
  const Eigen::VectorXd frcon_e = model.frcon_e;
  const Eigen::VectorXd frcon_o = model.frcon_o;
  const Eigen::VectorXd fzcon_e = model.fzcon_e;
  const Eigen::VectorXd fzcon_o = model.fzcon_o;
  FourierForces forward = *vmec.physical_f_[0];
  model.dft_ForcesToFourier_2d_symm(forward);
  FourierForces force_bar = forward;
  force_bar.setZero();
  FillRandom(force_bar.frcc, generator);
  FillRandom(force_bar.fzsc, generator);
  FillRandom(force_bar.flsc, generator);
  const double lhs = DotValues(forward.frcc, force_bar.frcc) +
                     DotValues(forward.fzsc, force_bar.fzsc) +
                     DotValues(forward.flsc, force_bar.flsc);
  model.dft_ForcesToFourierTranspose_2d_symm(force_bar);
  const double rhs =
      DotValues(armn_e, model.armn_e) + DotValues(armn_o, model.armn_o) +
      DotValues(brmn_e, model.brmn_e) + DotValues(brmn_o, model.brmn_o) +
      DotValues(azmn_e, model.azmn_e) + DotValues(azmn_o, model.azmn_o) +
      DotValues(bzmn_e, model.bzmn_e) + DotValues(bzmn_o, model.bzmn_o) +
      DotValues(blmn_e, model.blmn_e) + DotValues(blmn_o, model.blmn_o) +
      DotValues(frcon_e, model.frcon_e) + DotValues(frcon_o, model.frcon_o) +
      DotValues(fzcon_e, model.fzcon_e) + DotValues(fzcon_o, model.fzcon_o);
  return RelativeDotError(lhs, rhs);
}

double CheckForceTranspose3d(Vmec& vmec, std::mt19937& generator) {
  IdealMhdModel& model = *vmec.m_[0];
  FillSymmetricForce(model, generator);
  const Eigen::VectorXd armn_e = model.armn_e;
  const Eigen::VectorXd armn_o = model.armn_o;
  const Eigen::VectorXd brmn_e = model.brmn_e;
  const Eigen::VectorXd brmn_o = model.brmn_o;
  const Eigen::VectorXd crmn_e = model.crmn_e;
  const Eigen::VectorXd crmn_o = model.crmn_o;
  const Eigen::VectorXd azmn_e = model.azmn_e;
  const Eigen::VectorXd azmn_o = model.azmn_o;
  const Eigen::VectorXd bzmn_e = model.bzmn_e;
  const Eigen::VectorXd bzmn_o = model.bzmn_o;
  const Eigen::VectorXd czmn_e = model.czmn_e;
  const Eigen::VectorXd czmn_o = model.czmn_o;
  const Eigen::VectorXd blmn_e = model.blmn_e;
  const Eigen::VectorXd blmn_o = model.blmn_o;
  const Eigen::VectorXd clmn_e = model.clmn_e;
  const Eigen::VectorXd clmn_o = model.clmn_o;
  const Eigen::VectorXd frcon_e = model.frcon_e;
  const Eigen::VectorXd frcon_o = model.frcon_o;
  const Eigen::VectorXd fzcon_e = model.fzcon_e;
  const Eigen::VectorXd fzcon_o = model.fzcon_o;
  FourierForces forward = *vmec.physical_f_[0];
  model.dft_ForcesToFourier_3d_symm(forward);
  FourierForces force_bar = forward;
  force_bar.setZero();
  FillRandom(force_bar.frcc, generator);
  FillRandom(force_bar.frss, generator);
  FillRandom(force_bar.fzsc, generator);
  FillRandom(force_bar.fzcs, generator);
  FillRandom(force_bar.flsc, generator);
  FillRandom(force_bar.flcs, generator);
  const double lhs = DotValues(forward.frcc, force_bar.frcc) +
                     DotValues(forward.frss, force_bar.frss) +
                     DotValues(forward.fzsc, force_bar.fzsc) +
                     DotValues(forward.fzcs, force_bar.fzcs) +
                     DotValues(forward.flsc, force_bar.flsc) +
                     DotValues(forward.flcs, force_bar.flcs);
  model.dft_ForcesToFourierTranspose_3d_symm(force_bar);
  const double rhs =
      DotValues(armn_e, model.armn_e) + DotValues(armn_o, model.armn_o) +
      DotValues(brmn_e, model.brmn_e) + DotValues(brmn_o, model.brmn_o) +
      DotValues(crmn_e, model.crmn_e) + DotValues(crmn_o, model.crmn_o) +
      DotValues(azmn_e, model.azmn_e) + DotValues(azmn_o, model.azmn_o) +
      DotValues(bzmn_e, model.bzmn_e) + DotValues(bzmn_o, model.bzmn_o) +
      DotValues(czmn_e, model.czmn_e) + DotValues(czmn_o, model.czmn_o) +
      DotValues(blmn_e, model.blmn_e) + DotValues(blmn_o, model.blmn_o) +
      DotValues(clmn_e, model.clmn_e) + DotValues(clmn_o, model.clmn_o) +
      DotValues(frcon_e, model.frcon_e) + DotValues(frcon_o, model.frcon_o) +
      DotValues(fzcon_e, model.fzcon_e) + DotValues(fzcon_o, model.fzcon_o);
  return RelativeDotError(lhs, rhs);
}

double CheckAxisExtrapolateTranspose(Vmec& vmec, std::mt19937& generator) {
  FourierGeometry input = *vmec.decomposed_x_[0];
  FillGeometry(input, generator);
  FourierGeometry forward = input;
  forward.extrapolateTowardsAxis();
  FourierGeometry cotangent = input;
  cotangent.setZero();
  FillGeometry(cotangent, generator);
  FourierGeometry cotangent_before = cotangent;
  cotangent.extrapolateTowardsAxisTranspose();
  return RelativeDotError(DotGeometry(forward, cotangent_before),
                          DotGeometry(input, cotangent));
}

void CheckTransposeIdentities(const std::string& identifier, bool lthreed) {
  const std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", identifier);
  const absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());
  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());
  Vmec vmec(*vmec_indata, 1, OutputMode::kSilent);
  ASSERT_EQ(vmec.s_.lthreed, lthreed);
  const absl::StatusOr<bool> initialized =
      vmec.run(VmecCheckpoint::FOURIER_GEOMETRY_TO_START_WITH, 1);
  ASSERT_TRUE(initialized.ok());
  ASSERT_TRUE(*initialized);
  std::mt19937 generator(1729);
  EXPECT_LT(CheckAxisExtrapolateTranspose(vmec, generator), 1.0e-12);
  if (lthreed) {
    EXPECT_LT(CheckGeometryTranspose3d(vmec, generator), 1.0e-12);
    EXPECT_LT(CheckForceTranspose3d(vmec, generator), 1.0e-12);
  } else {
    EXPECT_LT(CheckGeometryTranspose2d(vmec, generator), 1.0e-12);
    EXPECT_LT(CheckForceTranspose2d(vmec, generator), 1.0e-12);
  }
}

TEST(IdealMhdModelTransposeTest, DirectTwoDimensionalAdjointIdentities) {
  CheckTransposeIdentities("solovev", false);
}

TEST(IdealMhdModelTransposeTest, DirectThreeDimensionalAdjointIdentities) {
  CheckTransposeIdentities("cth_like_fixed_bdy", true);
}
}  // namespace

// used to specify case-specific tolerances
// and which iterations to test
struct DataSource {
  std::string identifier;
  double tolerance = 0.0;
  std::vector<int> iter2_to_test = {1, 2};
};

class SpectralConstraintTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(SpectralConstraintTest, CheckSpectralConstraint) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  Vmec vmec(*vmec_indata);
  const Sizes& s = vmec.s_;

  bool reached_checkpoint =
      vmec.run(VmecCheckpoint::SPECTRAL_CONSTRAINT, 1).value();
  EXPECT_TRUE(reached_checkpoint);

  filename = absl::StrFormat(
      "vmecpp_large_cpp_tests/test_data/%s/spectral_constraint/"
      "spectral_constraint_00000_000001_%02d.%s.json",
      data_source_.identifier, vmec.get_num_eqsolve_retries(),
      data_source_.identifier);
  std::ifstream ifs_spectral_constraint(filename);
  ASSERT_TRUE(ifs_spectral_constraint.is_open())
      << "failed to open reference file: " << filename;
  json spectral_constraint = json::parse(ifs_spectral_constraint);

  // perform testing outside of multi-threaded region to avoid overlapping error
  // messages
  for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
    for (int m = 0; m < s.mpol; ++m) {
      EXPECT_TRUE(IsCloseRelAbs(spectral_constraint["xmpq"][m][0],
                                vmec.m_[thread_id]->xmpq[m], tolerance));
      EXPECT_TRUE(IsCloseRelAbs(spectral_constraint["faccon"][m],
                                vmec.m_[thread_id]->faccon[m], tolerance));
    }  // m
  }  // thread_id
}  // CheckSpectralConstraint

INSTANTIATE_TEST_SUITE_P(
    TestIdealMhdModel, SpectralConstraintTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-30},
           DataSource{.identifier = "solovev_analytical", .tolerance = 1.0e-30},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-30},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-30},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-30},
           DataSource{.identifier = "cma", .tolerance = 1.0e-30},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-30}));

class FourierGeometryToStartWithTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(FourierGeometryToStartWithTest, CheckFourierGeometryToStartWith) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::FOURIER_GEOMETRY_TO_START_WITH,
                 number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/totzsp_input/"
        "totzsp_input_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_totzsp_input(filename);
    ASSERT_TRUE(ifs_totzsp_input.is_open())
        << "failed to open reference file: " << filename;
    json totzsp_input = json::parse(ifs_totzsp_input);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];
      const FourierGeometry& physical_x = *(vmec.physical_x_[thread_id]);

      const int nsMinF = radial_partitioning.nsMinF;

      for (int rzl = 0; rzl < 3; ++rzl) {
        for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
          // time stepping is only done on those Fourier coefficients
          // for which the forces were computed in this thread!
          for (int jF = nsMinF; jF < radial_partitioning.nsMaxFIncludingLcfs;
               ++jF) {
            for (int n = 0; n < s.ntor + 1; ++n) {
              for (int m = 0; m < s.mpol; ++m) {
                EXPECT_TRUE(IsCloseRelAbs(
                    totzsp_input["gc"][rzl][idx_basis][jF][n][m],
                    physical_x.GetXcElement(rzl, idx_basis, jF, n, m),
                    tolerance));
              }  // m
            }  // n
          }  // jF
        }  // idx_basis
      }  // rzl
    }  // thread_id
  }
}  // CheckFourierGeometryToStartWith

INSTANTIATE_TEST_SUITE_P(
    TestIdealMhdModel, FourierGeometryToStartWithTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-14},
           DataSource{
               .identifier = "cma", .tolerance = 2.0e-13, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 5.0e-13,
                      .iter2_to_test = {1, 2, 53, 54}}));

class InverseFourierTransformGeometryTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(InverseFourierTransformGeometryTest,
       CheckInverseFourierTransformGeometry) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::INV_DFT_GEOMETRY, number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/funct3d_geometry/"
        "funct3d_geometry_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_funct3d_geometry(filename);
    ASSERT_TRUE(ifs_funct3d_geometry.is_open())
        << "failed to open reference file: " << filename;
    json funct3d_geometry = json::parse(ifs_funct3d_geometry);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];
      const RadialProfiles& radial_profiles = *vmec.p_[thread_id];

      const int nsMinF1 = radial_partitioning.nsMinF1;
      const int nsMaxF1 = radial_partitioning.nsMaxF1;

      const int nsMinF = radial_partitioning.nsMinF;

      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      for (int jF = nsMinF1; jF < nsMaxF1; ++jF) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            int idx_kl = ((jF - nsMinF1) * s.nZeta + k) * s.nThetaEff + l;

            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["r1"][jF][kEvenParity][k][l],
                              vmec.m_[thread_id]->r1_e[idx_kl], tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["r1"][jF][kOddParity][k][l],
                              vmec.m_[thread_id]->r1_o[idx_kl], tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["ru"][jF][kEvenParity][k][l],
                              vmec.m_[thread_id]->ru_e[idx_kl], tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["ru"][jF][kOddParity][k][l],
                              vmec.m_[thread_id]->ru_o[idx_kl], tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["z1"][jF][kEvenParity][k][l],
                              vmec.m_[thread_id]->z1_e[idx_kl], tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["z1"][jF][kOddParity][k][l],
                              vmec.m_[thread_id]->z1_o[idx_kl], tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["zu"][jF][kEvenParity][k][l],
                              vmec.m_[thread_id]->zu_e[idx_kl], tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["zu"][jF][kOddParity][k][l],
                              vmec.m_[thread_id]->zu_o[idx_kl], tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["lu"][jF][kEvenParity][k][l],
                              vmec.m_[thread_id]->lu_e[idx_kl], tolerance));
            EXPECT_TRUE(
                IsCloseRelAbs(funct3d_geometry["lu"][jF][kOddParity][k][l],
                              vmec.m_[thread_id]->lu_o[idx_kl], tolerance));
            if (s.lthreed) {
              EXPECT_TRUE(
                  IsCloseRelAbs(funct3d_geometry["rv"][jF][kEvenParity][k][l],
                                vmec.m_[thread_id]->rv_e[idx_kl], tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(funct3d_geometry["rv"][jF][kOddParity][k][l],
                                vmec.m_[thread_id]->rv_o[idx_kl], tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(funct3d_geometry["zv"][jF][kEvenParity][k][l],
                                vmec.m_[thread_id]->zv_e[idx_kl], tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(funct3d_geometry["zv"][jF][kOddParity][k][l],
                                vmec.m_[thread_id]->zv_o[idx_kl], tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(funct3d_geometry["lv"][jF][kEvenParity][k][l],
                                vmec.m_[thread_id]->lv_e[idx_kl], tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(funct3d_geometry["lv"][jF][kOddParity][k][l],
                                vmec.m_[thread_id]->lv_o[idx_kl], tolerance));
            }  // lthreed

            if (nsMinF <= jF && jF < nsMaxFIncludingLcfs) {
              // spectral condensation is local per flux surface --> no need for
              // numFull1 This index is invalid outside of [r->nsMinF, jMaxCon[
              // !
              const int idx_con =
                  ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff + l;

              const double sqrtSF =
                  radial_profiles.sqrtSF[jF - radial_partitioning.nsMinF1];

              // VMEC++ computes rCon, zCon directly combined from even-m and
              // odd-m contributions. This is different from educational_VMEC
              // (and the other Fortran implementations), where the combination
              // is computed after totzsp(s,a). Hence, need to combine the
              // even-m and odd-m contributions in order to compute the required
              // reference quantities.
              const double expected_rcon_even =
                  funct3d_geometry["rcon"][jF][kEvenParity][k][l];
              const double expected_rcon_odd =
                  funct3d_geometry["rcon"][jF][kOddParity][k][l];
              const double expected_rcon =
                  expected_rcon_even + sqrtSF * expected_rcon_odd;

              const double expected_zcon_even =
                  funct3d_geometry["zcon"][jF][kEvenParity][k][l];
              const double expected_zcon_odd =
                  funct3d_geometry["zcon"][jF][kOddParity][k][l];
              const double expected_zcon =
                  expected_zcon_even + sqrtSF * expected_zcon_odd;

              EXPECT_TRUE(IsCloseRelAbs(
                  expected_rcon, vmec.m_[thread_id]->rCon[idx_con], tolerance));
              EXPECT_TRUE(IsCloseRelAbs(
                  expected_zcon, vmec.m_[thread_id]->zCon[idx_con], tolerance));
            }  // within ...Con range: forces full-grid (numFull)
          }  // l
        }  // k
      }  // jF
    }  // thread_id
  }
}  // CheckInverseFourierTransformGeometry

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, InverseFourierTransformGeometryTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 2.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 2.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 6.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 6.0e-14},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-12, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-12,
                      .iter2_to_test = {1, 2, 53, 54}}));

class JacobianTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(JacobianTest, CheckJacobian) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::JACOBIAN, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/jacobian/"
        "jacobian_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_jacobian(filename);
    ASSERT_TRUE(ifs_jacobian.is_open())
        << "failed to open reference file: " << filename;
    json jacobian = json::parse(ifs_jacobian);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            int idx_kl = ((jH - nsMinH) * s.nZeta + k) * s.nThetaEff + l;

            EXPECT_TRUE(IsCloseRelAbs(jacobian["r12"][jH + 1][k][l],
                                      vmec.m_[thread_id]->r12[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(jacobian["ru12"][jH + 1][k][l],
                                      vmec.m_[thread_id]->ru12[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(jacobian["zu12"][jH + 1][k][l],
                                      vmec.m_[thread_id]->zu12[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(jacobian["rs"][jH + 1][k][l],
                                      vmec.m_[thread_id]->rs[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(jacobian["zs"][jH + 1][k][l],
                                      vmec.m_[thread_id]->zs[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(jacobian["tau"][jH + 1][k][l],
                                      vmec.m_[thread_id]->tau[idx_kl],
                                      tolerance));
          }  // l
        }  // k
      }  // jH

      EXPECT_EQ(RestartReasonFromInt(jacobian["irst"]), fc.restart_reason);
    }  // thread_id
  }
}  // CheckJacobian

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, JacobianTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 2.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 2.0e-14},
           DataSource{
               .identifier = "cma", .tolerance = 2.0e-13, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 5.0e-13,
                      .iter2_to_test = {1, 2, 53, 54}}));

class MetricTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(MetricTest, CheckMetric) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::METRIC, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/metric/"
        "metric_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_metric(filename);
    ASSERT_TRUE(ifs_metric.is_open())
        << "failed to open reference file: " << filename;
    json metric = json::parse(ifs_metric);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            int idx_kl = ((jH - nsMinH) * s.nZeta + k) * s.nThetaEff + l;

            EXPECT_TRUE(IsCloseRelAbs(metric["gsqrt"][jH + 1][k][l],
                                      vmec.m_[thread_id]->gsqrt[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(metric["guu"][jH + 1][k][l],
                                      vmec.m_[thread_id]->guu[idx_kl],
                                      tolerance));
            // Do not test r12sq, since this is not needed independently in
            // VMEC++.
            EXPECT_TRUE(IsCloseRelAbs(metric["gvv"][jH + 1][k][l],
                                      vmec.m_[thread_id]->gvv[idx_kl],
                                      tolerance));
            if (s.lthreed) {
              EXPECT_TRUE(IsCloseRelAbs(metric["guv"][jH + 1][k][l],
                                        vmec.m_[thread_id]->guv[idx_kl],
                                        tolerance));
            }
          }  // l
        }  // k
      }  // jH
    }  // thread_id
  }
}  // CheckMetric

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, MetricTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-15},
           DataSource{
               .identifier = "cma", .tolerance = 2.0e-14, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 5.0e-13,
                      .iter2_to_test = {1, 2, 53, 54}}));

class VolumeTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(VolumeTest, CheckVolume) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const FlowControl& fc = vmec.fc_;
    const HandoverStorage& h = vmec.h_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::VOLUME, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/volume/"
        "volume_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_volume(filename);
    ASSERT_TRUE(ifs_volume.is_open())
        << "failed to open reference file: " << filename;
    json volume = json::parse(ifs_volume);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        EXPECT_TRUE(IsCloseRelAbs(volume["vp"][jH + 1],
                                  vmec.p_[thread_id]->dVdsH[jH - nsMinH],
                                  tolerance));
      }  // jH
    }  // thread_id

    EXPECT_TRUE(IsCloseRelAbs(volume["voli"], h.voli, tolerance));
  }
}  // CheckVolume

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, VolumeTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-16},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-16},
           DataSource{
               .identifier = "cma", .tolerance = 1.0e-15, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-13,
                      .iter2_to_test = {1, 2, 53, 54}}));

class ContravariantMagneticFieldTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(ContravariantMagneticFieldTest, CheckContravariantMagneticField) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::B_CONTRA, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/bcontrav/"
        "bcontrav_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_bcontrav(filename);
    ASSERT_TRUE(ifs_bcontrav.is_open())
        << "failed to open reference file: " << filename;
    json bcontrav = json::parse(ifs_bcontrav);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/add_fluxes/"
        "add_fluxes_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_add_fluxes(filename);
    ASSERT_TRUE(ifs_add_fluxes.is_open())
        << "failed to open reference file: " << filename;
    json add_fluxes = json::parse(ifs_add_fluxes);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            const int idx_kl = ((jH - nsMinH) * s.nZeta + k) * s.nThetaEff + l;

            EXPECT_TRUE(IsCloseRelAbs(bcontrav["bsupv"][jH + 1][k][l],
                                      vmec.m_[thread_id]->bsupv[idx_kl],
                                      tolerance));
          }  // l
        }  // k

        EXPECT_TRUE(IsCloseRelAbs(add_fluxes["chips"][jH],
                                  vmec.p_[thread_id]->chipH[jH - nsMinH],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(add_fluxes["iotas"][jH],
                                  vmec.p_[thread_id]->iotaH[jH - nsMinH],
                                  tolerance));
      }  // jH

      const int nsMinF1 = radial_partitioning.nsMinF1;
      const int nsMaxF1 = radial_partitioning.nsMaxF1;

      const int nsMinFi = radial_partitioning.nsMinFi;
      const int nsMaxFi = radial_partitioning.nsMaxFi;

      int jMaxIncludingBoundary = nsMaxFi;
      if (nsMaxF1 == fc.ns) {
        jMaxIncludingBoundary = nsMaxF1;
      }

      for (int jFi = nsMinFi; jFi < jMaxIncludingBoundary; ++jFi) {
        EXPECT_TRUE(IsCloseRelAbs(add_fluxes["chipf"][jFi],
                                  vmec.p_[thread_id]->chipF[jFi - nsMinF1],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(add_fluxes["iotaf"][jFi],
                                  vmec.p_[thread_id]->iotaF[jFi - nsMinF1],
                                  tolerance));
      }  // jF
    }  // thread_id

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            const int idx_kl = ((jH - nsMinH) * s.nZeta + k) * s.nThetaEff + l;

            // add_fluxes is integrated into computeBContra() in VMEC++ already,
            // so we need to test the final bsupu here (the one coming out of
            // add_fluxes).
            EXPECT_TRUE(IsCloseRelAbs(add_fluxes["bsupu"][jH + 1][k][l],
                                      vmec.m_[thread_id]->bsupu[idx_kl],
                                      tolerance));
          }  // l
        }  // k
      }  // jH
    }  // thread_id
  }
}  // CheckContravariantMagneticField

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, ContravariantMagneticFieldTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-16},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-16},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-13},
           DataSource{
               .identifier = "cma", .tolerance = 6.0e-12, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-11,
                      .iter2_to_test = {1, 2, 53, 54}}));

class CovariantMagneticFieldTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(CovariantMagneticFieldTest, CheckCovariantMagneticField) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::B_CO, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/bcov/bcov_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_bcov(filename);
    ASSERT_TRUE(ifs_bcov.is_open())
        << "failed to open reference file: " << filename;
    json bcov = json::parse(ifs_bcov);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            int idx_kl = ((jH - nsMinH) * s.nZeta + k) * s.nThetaEff + l;

            EXPECT_TRUE(IsCloseRelAbs(bcov["bsubuh"][jH + 1][k][l],
                                      vmec.m_[thread_id]->bsubu[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(bcov["bsubvh"][jH + 1][k][l],
                                      vmec.m_[thread_id]->bsubv[idx_kl],
                                      tolerance));
          }  // l
        }  // k
      }  // jH
    }  // thread_id
  }
}  // CheckCovariantMagneticField

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, CovariantMagneticFieldTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 2.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 2.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-13},
           DataSource{
               .identifier = "cma", .tolerance = 6.0e-12, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-11,
                      .iter2_to_test = {1, 2, 53, 54}}));

class TotalPressureAndEnergiesTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(TotalPressureAndEnergiesTest, CheckTotalPressureAndEnergies) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;
    const HandoverStorage& h = vmec.h_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::ENERGY, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/bcov/bcov_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_bcov(filename);
    ASSERT_TRUE(ifs_bcov.is_open())
        << "failed to open reference file: " << filename;
    json bcov = json::parse(ifs_bcov);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            const int idx_kl = ((jH - nsMinH) * s.nZeta + k) * s.nThetaEff + l;

            EXPECT_TRUE(IsCloseRelAbs(bcov["bsq"][jH + 1][k][l],
                                      vmec.m_[thread_id]->totalPressure[idx_kl],
                                      tolerance));
          }  // l
        }  // k

        EXPECT_TRUE(IsCloseRelAbs(bcov["pres"][jH],
                                  vmec.p_[thread_id]->presH[jH - nsMinH],
                                  tolerance));
      }  // jH
    }  // thread_id

    EXPECT_TRUE(IsCloseRelAbs(bcov["wp"], h.thermalEnergy, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(bcov["wb"], h.magneticEnergy, tolerance));
    // The MHD energy is not available for testing in the bcov reference file
    // from educational_VMEC.
  }
}  // CheckTotalPressureAndEnergies

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, TotalPressureAndEnergiesTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-16},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-16},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-13},
           DataSource{
               .identifier = "cma", .tolerance = 2.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 5.0e-12,
                      .iter2_to_test = {1, 2, 53, 54}}));

class RadialForceBalanceTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(RadialForceBalanceTest, CheckRadialForceBalance) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::RADIAL_FORCE_BALANCE, number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/calc_fbal/"
        "calc_fbal_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_calc_fbal(filename);
    ASSERT_TRUE(ifs_calc_fbal.is_open())
        << "failed to open reference file: " << filename;
    json calc_fbal = json::parse(ifs_calc_fbal);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        EXPECT_TRUE(IsCloseRelAbs(calc_fbal["buco"][jH],
                                  vmec.p_[thread_id]->bucoH[jH - nsMinH],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(calc_fbal["bvco"][jH],
                                  vmec.p_[thread_id]->bvcoH[jH - nsMinH],
                                  tolerance));
      }  // jH

      const int nsMinFi = radial_partitioning.nsMinFi;
      const int nsMaxFi = radial_partitioning.nsMaxFi;

      for (int jFi = nsMinFi; jFi < nsMaxFi; ++jFi) {
        EXPECT_TRUE(IsCloseRelAbs(calc_fbal["jcurv"][jFi],
                                  vmec.p_[thread_id]->jcurvF[jFi - nsMinFi],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(calc_fbal["jcuru"][jFi],
                                  vmec.p_[thread_id]->jcuruF[jFi - nsMinFi],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(calc_fbal["presgrad"][jFi],
                                  vmec.p_[thread_id]->presgradF[jFi - nsMinFi],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(calc_fbal["vpphi"][jFi],
                                  vmec.p_[thread_id]->dVdsF[jFi - nsMinFi],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(calc_fbal["equif"][jFi],
                                  vmec.p_[thread_id]->equiF[jFi - nsMinFi],
                                  tolerance));
      }  // jFi
    }  // thread_id
  }
}  // CheckRadialForceBalance

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, RadialForceBalanceTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-14},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-12},
           DataSource{
               .identifier = "cma", .tolerance = 2.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-12,
                      .iter2_to_test = {1, 2, 53, 54}}));

class HybridLambdaForceTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(HybridLambdaForceTest, CheckHybridLambdaForce) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;
    const HandoverStorage& h = vmec.h_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::HYBRID_LAMBDA_FORCE, number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/bcov_full/"
        "bcov_full_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_bcov_full(filename);
    ASSERT_TRUE(ifs_bcov_full.is_open())
        << "failed to open reference file: " << filename;
    json bcov_full = json::parse(ifs_bcov_full);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/lulv_comb/"
        "lulv_comb_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_lulv_comb(filename);
    ASSERT_TRUE(ifs_lulv_comb.is_open())
        << "failed to open reference file: " << filename;
    json lulv_comb = json::parse(ifs_lulv_comb);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    EXPECT_TRUE(IsCloseRelAbs(bcov_full["rbtor0"], h.rBtor0, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(bcov_full["rbtor"], h.rBtor, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(bcov_full["ctor"], h.cTor, tolerance));

    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            const int idx_kl = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff + l;

            if (jF == 0) {
              // PARVMEC does not scale the axis value of the lambda forces by
              // (-lamscale). This is the behaviour desired for VMEC++, since it
              // improves stability of the iterations. Hence, we need to
              // back-port the axis lambda forces here for the comparison
              // against educational_VMEC.
              const double neg_lamscale = -1 * vmec.constants_.lamscale;
              EXPECT_TRUE(IsCloseRelAbs(
                  lulv_comb["bsubv_e"][jF][k][l],
                  vmec.m_[thread_id]->blmn_e[idx_kl] * neg_lamscale,
                  tolerance));
              EXPECT_TRUE(IsCloseRelAbs(
                  lulv_comb["bsubv_o"][jF][k][l],
                  vmec.m_[thread_id]->blmn_o[idx_kl] * neg_lamscale,
                  tolerance));
              if (s.lthreed) {
                EXPECT_TRUE(IsCloseRelAbs(
                    lulv_comb["bsubu_e"][jF][k][l],
                    vmec.m_[thread_id]->clmn_e[idx_kl] * neg_lamscale,
                    tolerance));
                EXPECT_TRUE(IsCloseRelAbs(
                    lulv_comb["bsubu_e"][jF][k][l],
                    vmec.m_[thread_id]->clmn_e[idx_kl] * neg_lamscale,
                    tolerance));
              }
            } else {
              EXPECT_TRUE(IsCloseRelAbs(lulv_comb["bsubv_e"][jF][k][l],
                                        vmec.m_[thread_id]->blmn_e[idx_kl],
                                        tolerance));
              EXPECT_TRUE(IsCloseRelAbs(lulv_comb["bsubv_o"][jF][k][l],
                                        vmec.m_[thread_id]->blmn_o[idx_kl],
                                        tolerance));
              if (s.lthreed) {
                EXPECT_TRUE(IsCloseRelAbs(lulv_comb["bsubu_e"][jF][k][l],
                                          vmec.m_[thread_id]->clmn_e[idx_kl],
                                          tolerance));
                EXPECT_TRUE(IsCloseRelAbs(lulv_comb["bsubu_e"][jF][k][l],
                                          vmec.m_[thread_id]->clmn_e[idx_kl],
                                          tolerance));
              }
            }
          }  // l
        }  // k
      }  // jF
    }  // thread_id
  }
}  // CheckHybridLambdaForce

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, HybridLambdaForceTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-13},
           DataSource{
               .identifier = "cma", .tolerance = 2.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 2.0e-12,
                      .iter2_to_test = {1, 2, 53, 54}}));

class UpdateRadialPreconditionerTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(UpdateRadialPreconditionerTest, CheckUpdateRadialPreconditioner) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::UPDATE_RADIAL_PRECONDITIONER,
                 number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/lamcal/"
        "lamcal_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_preconditioner_update(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_lamcal(filename);
    ASSERT_TRUE(ifs_lamcal.is_open())
        << "failed to open reference file: " << filename;
    json lamcal = json::parse(ifs_lamcal);

    // only one file, since only updated every 25 iterations
    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/precondn/"
        "precondn_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_preconditioner_update(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_precondn(filename);
    ASSERT_TRUE(ifs_precondn.is_open())
        << "failed to open reference file: " << filename;
    json precondn = json::parse(ifs_precondn);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxF = radial_partitioning.nsMaxF;

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      // first check the lambda preconditioner
      for (int jF = nsMinF; jF < nsMaxF; ++jF) {
        EXPECT_TRUE(IsCloseRelAbs(lamcal["blam"][jF],
                                  vmec.m_[thread_id]->bLambda[jF - nsMinF],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(lamcal["clam"][jF],
                                  vmec.m_[thread_id]->cLambda[jF - nsMinF],
                                  tolerance));
        EXPECT_TRUE(IsCloseRelAbs(lamcal["dlam"][jF],
                                  vmec.m_[thread_id]->dLambda[jF - nsMinF],
                                  tolerance));
      }  // jF

      int j_min_f = 0;
      if (nsMinF == 0) {
        // skip axis, as there is no lambda on the axis
        j_min_f = 1;
      }

      for (int jF = std::max(j_min_f, nsMinF); jF < nsMaxF; ++jF) {
        for (int n = 0; n < s.ntor + 1; ++n) {
          for (int m = 0; m < s.mpol; ++m) {
            if (m == 0 && n == 0) {
              // skip the (0,0)-component, as this is not used for lambda
              continue;
            }

            const int idx_mn = ((jF - nsMinF) * s.mpol + m) * (s.ntor + 1) + n;

            EXPECT_TRUE(IsCloseRelAbs(
                lamcal["faclam"][jF][n][m],
                vmec.m_[thread_id]->lambdaPreconditioner[idx_mn], tolerance));
          }  // m
        }  // n
      }  // jF

      // now check the preconditioner for R and Z
      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["arm"][jH + 1][kEvenParity],
            vmec.m_[thread_id]->arm[(jH - nsMinH) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["arm"][jH + 1][kOddParity],
            vmec.m_[thread_id]->arm[(jH - nsMinH) * 2 + kOddParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["azm"][jH + 1][kEvenParity],
            vmec.m_[thread_id]->azm[(jH - nsMinH) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["azm"][jH + 1][kOddParity],
            vmec.m_[thread_id]->azm[(jH - nsMinH) * 2 + kOddParity],
            tolerance));

        EXPECT_TRUE(IsCloseRelAbs(
            precondn["brm"][jH + 1][kEvenParity],
            vmec.m_[thread_id]->brm[(jH - nsMinH) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["brm"][jH + 1][kOddParity],
            vmec.m_[thread_id]->brm[(jH - nsMinH) * 2 + kOddParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["bzm"][jH + 1][kEvenParity],
            vmec.m_[thread_id]->bzm[(jH - nsMinH) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["bzm"][jH + 1][kOddParity],
            vmec.m_[thread_id]->bzm[(jH - nsMinH) * 2 + kOddParity],
            tolerance));
      }  // jH

      for (int jF = nsMinF; jF < nsMaxF; ++jF) {
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["ard"][jF][kEvenParity],
            vmec.m_[thread_id]->ard[(jF - nsMinF) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["ard"][jF][kOddParity],
            vmec.m_[thread_id]->ard[(jF - nsMinF) * 2 + kOddParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["azd"][jF][kEvenParity],
            vmec.m_[thread_id]->azd[(jF - nsMinF) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["azd"][jF][kOddParity],
            vmec.m_[thread_id]->azd[(jF - nsMinF) * 2 + kOddParity],
            tolerance));

        EXPECT_TRUE(IsCloseRelAbs(
            precondn["brd"][jF][kEvenParity],
            vmec.m_[thread_id]->brd[(jF - nsMinF) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["brd"][jF][kOddParity],
            vmec.m_[thread_id]->brd[(jF - nsMinF) * 2 + kOddParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["bzd"][jF][kEvenParity],
            vmec.m_[thread_id]->bzd[(jF - nsMinF) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            precondn["bzd"][jF][kOddParity],
            vmec.m_[thread_id]->bzd[(jF - nsMinF) * 2 + kOddParity],
            tolerance));

        EXPECT_TRUE(IsCloseRelAbs(precondn["crd"][jF],
                                  vmec.m_[thread_id]->cxd[jF - nsMinF],
                                  tolerance));
      }  // jF
    }  // thread_id
  }
}  // CheckUpdateRadialPreconditioner

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, UpdateRadialPreconditionerTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-14},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-12},
           DataSource{
               .identifier = "cma", .tolerance = 2.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-12,
                      .iter2_to_test = {1, 2, 53, 54}}));

class ForceNormsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(ForceNormsTest, CheckForceNorms) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const HandoverStorage& h = vmec.h_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::UPDATE_FORCE_NORMS, number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    // only one file, since only updated every 25 iterations
    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/forceNorms_tcon/"
        "forceNorms_tcon_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_preconditioner_update(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_forceNorms_tcon(filename);
    ASSERT_TRUE(ifs_forceNorms_tcon.is_open())
        << "failed to open reference file: " << filename;
    json forceNorms_tcon = json::parse(ifs_forceNorms_tcon);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];
      const FourierGeometry& decomposed_x = *vmec.decomposed_x_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      // test xc as well for fnorm1
      // only possible at every preconditioner update interval,
      // because the xc written to the file will be outdated otherwise
      if (vmec.m_[0]->shouldUpdateRadialPreconditioner(vmec.get_iter1(),
                                                       vmec.get_iter2())) {
        for (int rzl = 0; rzl < 3; ++rzl) {
          for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
            for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
              for (int n = 0; n < s.ntor + 1; ++n) {
                for (int m = 0; m < s.mpol; ++m) {
                  EXPECT_TRUE(IsCloseRelAbs(
                      forceNorms_tcon["xc"][rzl][idx_basis][jF][n][m],
                      decomposed_x.GetXcElement(rzl, idx_basis, jF, n, m),
                      tolerance));
                }  // m
              }  // n
            }  // jF
          }  // idx_basis
        }  // rzl
      }
    }  // thread_id

    EXPECT_TRUE(IsCloseRelAbs(forceNorms_tcon["fnorm"], h.fNormRZ, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(forceNorms_tcon["fnormL"], h.fNormL, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(forceNorms_tcon["fnorm1"], h.fNorm1, tolerance));
  }
}  // CheckForceNorms

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, ForceNormsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-14},
           DataSource{
               .identifier = "cma", .tolerance = 1.0e-12, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 5.0e-14,
                      .iter2_to_test = {1, 2, 53, 54}}));

class ConstraintForceMultiplierTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(ConstraintForceMultiplierTest, CheckConstraintForceMultiplier) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::UPDATE_TCON, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    // only one file, since only updated every 25 iterations
    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/forceNorms_tcon/"
        "forceNorms_tcon_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, vmec.get_last_preconditioner_update(),
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_forceNorms_tcon(filename);
    ASSERT_TRUE(ifs_forceNorms_tcon.is_open())
        << "failed to open reference file: " << filename;
    json forceNorms_tcon = json::parse(ifs_forceNorms_tcon);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      // skip axis -> no constraint
      const int jMin = std::max(1, nsMinF);

      for (int jF = jMin; jF < nsMaxFIncludingLcfs; ++jF) {
        EXPECT_TRUE(IsCloseRelAbs(forceNorms_tcon["tcon"][jF - 1],
                                  vmec.m_[thread_id]->tcon[jF - nsMinF],
                                  tolerance));
      }  // jF
    }  // thread_id
  }
}  // CheckConstraintForceMultiplier

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, ConstraintForceMultiplierTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 2.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 2.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-13},
           DataSource{
               .identifier = "cma", .tolerance = 1.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-13,
                      .iter2_to_test = {1, 2, 53, 54}}));

// NOTE: Along the forward model evaluation, this is where Nestor
// (//vmecpp/nestor/...) is udpated. When debugging VMEC++ by going through the
// tests one by one along an iteration, run the Nestor tests here.

class RBsqTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(RBsqTest, CheckRBsq) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::RBSQ, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    // The filename of rbsq debugging output is created based on the number of
    // Nestor calls, and not the number of overall iterations.
    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/rbsq/rbsq_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_rbsq(filename);
    ASSERT_TRUE(ifs_rbsq.is_open())
        << "failed to open reference file: " << filename;
    json rbsq = json::parse(ifs_rbsq);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      if (nsMaxFIncludingLcfs == fc.ns) {
        // this thread has the LCFS in it

        for (int l = 0; l < s.nThetaEff; ++l) {
          for (int k = 0; k < s.nZeta; ++k) {
            const int kl = k * s.nThetaEff + l;
            EXPECT_TRUE(IsCloseRelAbs(rbsq["rbsq"][k][l],
                                      vmec.m_[thread_id]->rBSq[kl], tolerance));
          }  // k
        }  // l
      }  // check for LCFS
    }  // thread_id
  }
}  // CheckRBsq

INSTANTIATE_TEST_SUITE_P(TestIdealMHDModel, RBsqTest,
                         Values(DataSource{.identifier = "cth_like_free_bdy",
                                           .tolerance = 1.0e-10,
                                           .iter2_to_test = {53, 54}}));

class AliasTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(AliasTest, CheckAlias) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::ALIAS, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/constraint_force/"
        "constraint_force_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_constraint_force(filename);
    ASSERT_TRUE(ifs_constraint_force.is_open())
        << "failed to open reference file: " << filename;
    json constraint_force = json::parse(ifs_constraint_force);

    // step 1: check ingredients to input to effctiveConstraintForce()
    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      // skip axis, as there is no constraint force on it
      const int jMin = std::max(1, nsMinF);

      for (int jF = jMin; jF < nsMaxFIncludingLcfs; ++jF) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            const int idx_kl = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff + l;

            EXPECT_TRUE(IsCloseRelAbs(constraint_force["rcon"][jF][k][l],
                                      vmec.m_[thread_id]->rCon[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(constraint_force["rcon0"][jF][k][l],
                                      vmec.m_[thread_id]->rCon0[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(constraint_force["ru0"][jF][k][l],
                                      vmec.m_[thread_id]->ruFull[idx_kl],
                                      tolerance));

            EXPECT_TRUE(IsCloseRelAbs(constraint_force["zcon"][jF][k][l],
                                      vmec.m_[thread_id]->zCon[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(constraint_force["zcon0"][jF][k][l],
                                      vmec.m_[thread_id]->zCon0[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(constraint_force["zu0"][jF][k][l],
                                      vmec.m_[thread_id]->zuFull[idx_kl],
                                      tolerance));
          }  // l
        }  // k
      }  // jF
    }  // thread_id

    // step 2: check input to and output from deAliasConstraintForce()
    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      // skip axis, as there is no constraint force on it
      const int jMin = std::max(1, nsMinF);

      for (int jF = jMin; jF < nsMaxFIncludingLcfs; ++jF) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            const int idx_kl = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff + l;

            EXPECT_TRUE(IsCloseRelAbs(constraint_force["extra1"][jF][k][l],
                                      vmec.m_[thread_id]->gConEff[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(constraint_force["gcon"][jF][k][l],
                                      vmec.m_[thread_id]->gCon[idx_kl],
                                      tolerance));
          }  // l
        }  // k
      }  // jF
    }  // thread_id
  }
}  // CheckAlias

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, AliasTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 2.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 2.0e-14},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-13, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-11,
                      .iter2_to_test = {1, 2, 53, 54}}));

class RealspaceForcesTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(RealspaceForcesTest, CheckRealspaceForces) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::REALSPACE_FORCES, number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/forces/"
        "forces_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_forces(filename.c_str());
    ASSERT_TRUE(ifs_forces.is_open())
        << "failed to open reference file: " << filename;
    json forces = json::parse(ifs_forces);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxF = radial_partitioning.nsMaxF;

      for (int jF = nsMinF; jF < nsMaxF; ++jF) {
        for (int k = 0; k < s.nZeta; ++k) {
          for (int l = 0; l < s.nThetaEff; ++l) {
            int idx_kl = ((jF - nsMinF) * s.nZeta + k) * s.nThetaEff + l;

            EXPECT_TRUE(IsCloseRelAbs(forces["armn_e"][jF][k][l],
                                      vmec.m_[thread_id]->armn_e[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(forces["armn_o"][jF][k][l],
                                      vmec.m_[thread_id]->armn_o[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(forces["azmn_e"][jF][k][l],
                                      vmec.m_[thread_id]->azmn_e[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(forces["azmn_o"][jF][k][l],
                                      vmec.m_[thread_id]->azmn_o[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(forces["brmn_e"][jF][k][l],
                                      vmec.m_[thread_id]->brmn_e[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(forces["brmn_o"][jF][k][l],
                                      vmec.m_[thread_id]->brmn_o[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(forces["bzmn_e"][jF][k][l],
                                      vmec.m_[thread_id]->bzmn_e[idx_kl],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(forces["bzmn_o"][jF][k][l],
                                      vmec.m_[thread_id]->bzmn_o[idx_kl],
                                      tolerance));
            if (s.lthreed) {
              EXPECT_TRUE(IsCloseRelAbs(forces["crmn_e"][jF][k][l],
                                        vmec.m_[thread_id]->crmn_e[idx_kl],
                                        tolerance));
              EXPECT_TRUE(IsCloseRelAbs(forces["crmn_o"][jF][k][l],
                                        vmec.m_[thread_id]->crmn_o[idx_kl],
                                        tolerance));
              EXPECT_TRUE(IsCloseRelAbs(forces["czmn_e"][jF][k][l],
                                        vmec.m_[thread_id]->czmn_e[idx_kl],
                                        tolerance));
              EXPECT_TRUE(IsCloseRelAbs(forces["czmn_o"][jF][k][l],
                                        vmec.m_[thread_id]->czmn_o[idx_kl],
                                        tolerance));
            }
          }  // l
        }  // k
      }  // jF
    }  // thread_id
  }
}  // CheckRealspaceForces

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, RealspaceForcesTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-14},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-11},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-11},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 2.0e-11,
                      .iter2_to_test = {1, 2, 53, 54}}));

class ForwardTransformForcesTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(ForwardTransformForcesTest, CheckForwardTransformForces) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::FWD_DFT_FORCES, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/tomnsps/"
        "tomnsps_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_tomnsps(filename);
    ASSERT_TRUE(ifs_tomnsps.is_open())
        << "failed to open reference file: " << filename;
    json tomnsps = json::parse(ifs_tomnsps);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxF = radial_partitioning.nsMaxF;
      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      int jMaxRZ = nsMaxF;

      for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
        for (int m = 0; m < s.mpol; ++m) {
          for (int n = 0; n < s.ntor + 1; ++n) {
            const int idx_mn = ((jF - nsMinF) * s.mpol + m) * (s.ntor + 1) + n;

            if (jF < jMaxRZ) {
              EXPECT_TRUE(IsCloseRelAbs(
                  tomnsps["frcc"][jF][n][m],
                  vmec.physical_f_[thread_id]->frcc[idx_mn], tolerance));
              EXPECT_TRUE(IsCloseRelAbs(
                  tomnsps["fzsc"][jF][n][m],
                  vmec.physical_f_[thread_id]->fzsc[idx_mn], tolerance));
            }
            EXPECT_TRUE(IsCloseRelAbs(tomnsps["flsc"][jF][n][m],
                                      vmec.physical_f_[thread_id]->flsc[idx_mn],
                                      tolerance));
            if (s.lthreed) {
              if (jF < jMaxRZ) {
                EXPECT_TRUE(IsCloseRelAbs(
                    tomnsps["frss"][jF][n][m],
                    vmec.physical_f_[thread_id]->frss[idx_mn], tolerance));
                EXPECT_TRUE(IsCloseRelAbs(
                    tomnsps["fzcs"][jF][n][m],
                    vmec.physical_f_[thread_id]->fzcs[idx_mn], tolerance));
              }
              EXPECT_TRUE(IsCloseRelAbs(
                  tomnsps["flcs"][jF][n][m],
                  vmec.physical_f_[thread_id]->flcs[idx_mn], tolerance));
            }
            if (s.lasym) {
              if (jF < jMaxRZ) {
                EXPECT_TRUE(IsCloseRelAbs(
                    tomnsps["frsc"][jF][n][m],
                    vmec.physical_f_[thread_id]->frsc[idx_mn], tolerance));
                EXPECT_TRUE(IsCloseRelAbs(
                    tomnsps["fzcc"][jF][n][m],
                    vmec.physical_f_[thread_id]->fzcc[idx_mn], tolerance));
              }
              EXPECT_TRUE(IsCloseRelAbs(
                  tomnsps["flcc"][jF][n][m],
                  vmec.physical_f_[thread_id]->flcc[idx_mn], tolerance));
              if (s.lthreed) {
                if (jF < jMaxRZ) {
                  EXPECT_TRUE(IsCloseRelAbs(
                      tomnsps["frcs"][jF][n][m],
                      vmec.physical_f_[thread_id]->frcs[idx_mn], tolerance));
                  EXPECT_TRUE(IsCloseRelAbs(
                      tomnsps["fzss"][jF][n][m],
                      vmec.physical_f_[thread_id]->fzss[idx_mn], tolerance));
                }
                EXPECT_TRUE(IsCloseRelAbs(
                    tomnsps["flss"][jF][n][m],
                    vmec.physical_f_[thread_id]->flss[idx_mn], tolerance));
              }
            }
          }  // m
        }  // n
      }  // jF
    }  // thread_id
  }
}  // CheckForwardTransformForces

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, ForwardTransformForcesTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-13},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-11,
                      .iter2_to_test = {1, 2, 53, 54}}));

class PhysicalForcesTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(PhysicalForcesTest, CheckPhysicalForces) {
  const double tolerance = data_source_.tolerance;

  static constexpr int kR = 0;
  static constexpr int kZ = 1;
  static constexpr int kL = 2;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::PHYSICAL_FORCES, number_of_iterations).value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/phys_gc/"
        "phys_gc_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);

    std::ifstream ifs_phys_gc(filename);
    ASSERT_TRUE(ifs_phys_gc.is_open())
        << "failed to open reference file: " << filename;
    json phys_gc = json::parse(ifs_phys_gc);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxF = radial_partitioning.nsMaxF;

      for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
        for (int jF = nsMinF; jF < nsMaxF; ++jF) {
          for (int m = 0; m < s.mpol; ++m) {
            for (int n = 0; n < s.ntor + 1; ++n) {
              EXPECT_TRUE(
                  IsCloseRelAbs(phys_gc["gcr"][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kR, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(phys_gc["gcz"][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kZ, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(phys_gc["gcl"][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kL, idx_basis, jF, n, m),
                                tolerance));
            }  // n
          }  // m
        }  // jF
      }  // idx_basis
    }  // thread_id
  }
}  // CheckPhysicalForces

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, PhysicalForcesTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-13},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-11,
                      .iter2_to_test = {1, 2, 53, 54}}));

class InvariantResidualsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(InvariantResidualsTest, CheckInvariantResiduals) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::INVARIANT_RESIDUALS, number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/fsq/fsq_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_fsq(filename);
    ASSERT_TRUE(ifs_fsq.is_open())
        << "failed to open reference file: " << filename;
    json fsq = json::parse(ifs_fsq);

    // copied from Vmec class for now
    bool includeEdgeRZForces = (vmec.get_iter2() - vmec.get_iter1() < 50 &&
                                fc.fsqr + fc.fsqz < 1.0e-6);
    EXPECT_EQ(fsq["jedge"], (includeEdgeRZForces ? 1 : 0));

    EXPECT_TRUE(IsCloseRelAbs(fsq["fsqr"], fc.fsqr, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(fsq["fsqz"], fc.fsqz, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(fsq["fsql"], fc.fsql, tolerance));
  }
}  // CheckInvariantResiduals

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, InvariantResidualsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-16},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-16},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-14},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 5.0e-13,
                      .iter2_to_test = {1, 2, 53, 54}}));

class ApplyM1PreconditionerTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(ApplyM1PreconditionerTest, CheckApplyM1Preconditioner) {
  const double tolerance = data_source_.tolerance;

  static constexpr int kR = 0;
  static constexpr int kZ = 1;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::APPLY_M1_PRECONDITIONER, number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/scale_m1/"
        "scale_m1_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_scale_m1(filename);
    ASSERT_TRUE(ifs_scale_m1.is_open())
        << "failed to open reference file: " << filename;
    json scale_m1 = json::parse(ifs_scale_m1);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxF = radial_partitioning.nsMaxF;

      for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
        for (int jF = nsMinF; jF < nsMaxF; ++jF) {
          for (int m = 0; m < s.mpol; ++m) {
            for (int n = 0; n < s.ntor + 1; ++n) {
              EXPECT_TRUE(
                  IsCloseRelAbs(scale_m1["gcr"][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kR, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(scale_m1["gcz"][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kZ, idx_basis, jF, n, m),
                                tolerance));
              // lambda is not touched -> no need to re-test here
            }  // n
          }  // m
        }  // jF
      }  // idx_basis
    }  // thread_id
  }
}  // CheckApplyM1Preconditioner

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, ApplyM1PreconditionerTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 5.0e-15},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 5.0e-15},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-13},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-13},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-11,
                      .iter2_to_test = {1, 2, 53, 54}}));

class AssembleRZPreconditionerTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(AssembleRZPreconditionerTest, CheckAssembleRZPreconditioner) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::ASSEMBLE_RZ_PRECONDITIONER,
                 number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/scalfor_R/"
        "scalfor_R_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_scalfor_R(filename);
    ASSERT_TRUE(ifs_scalfor_R.is_open())
        << "failed to open reference file: " << filename;
    json scalfor_R = json::parse(ifs_scalfor_R);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/scalfor_Z/"
        "scalfor_Z_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_scalfor_Z(filename);
    ASSERT_TRUE(ifs_scalfor_Z.is_open())
        << "failed to open reference file: " << filename;
    json scalfor_Z = json::parse(ifs_scalfor_Z);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxF = radial_partitioning.nsMaxF;

      for (int jF = nsMinF; jF < nsMaxF; ++jF) {
        for (int m = 0; m < s.mpol; ++m) {
          for (int n = 0; n < s.ntor + 1; ++n) {
            const int idx_mn = ((jF - nsMinF) * s.mpol + m) * (s.ntor + 1) + n;

            EXPECT_TRUE(IsCloseRelAbs(scalfor_R["ax"][jF][n][m],
                                      vmec.m_[thread_id]->ar[idx_mn],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(scalfor_R["dx"][jF][n][m],
                                      vmec.m_[thread_id]->dr[idx_mn],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(scalfor_R["bx"][jF][n][m],
                                      vmec.m_[thread_id]->br[idx_mn],
                                      tolerance));

            EXPECT_TRUE(IsCloseRelAbs(scalfor_Z["ax"][jF][n][m],
                                      vmec.m_[thread_id]->az[idx_mn],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(scalfor_Z["dx"][jF][n][m],
                                      vmec.m_[thread_id]->dz[idx_mn],
                                      tolerance));
            EXPECT_TRUE(IsCloseRelAbs(scalfor_Z["bx"][jF][n][m],
                                      vmec.m_[thread_id]->bz[idx_mn],
                                      tolerance));
          }  // m
        }  // n
      }  // jF
    }  // thread_id
  }
}  // CheckAssembleRZPreconditioner

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, AssembleRZPreconditionerTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-14},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-12},
           DataSource{
               .identifier = "cma", .tolerance = 5.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-12,
                      .iter2_to_test = {1, 2, 53, 54}}));

class ApplyPreconditionerTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(ApplyPreconditionerTest, CheckApplyPreconditioner) {
  const double tolerance = data_source_.tolerance;

  static constexpr int kR = 0;
  static constexpr int kZ = 1;
  static constexpr int kL = 2;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const Sizes& s = vmec.s_;
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::APPLY_RADIAL_PRECONDITIONER,
                 number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/scalfor_out/"
        "scalfor_out_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_scalfor_out(filename);
    ASSERT_TRUE(ifs_scalfor_out.is_open())
        << "failed to open reference file: " << filename;
    json scalfor_out = json::parse(ifs_scalfor_out);

    // perform testing outside of multi-threaded region to avoid overlapping
    // error messages
    for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
      const RadialPartitioning& radial_partitioning = *vmec.r_[thread_id];

      const int nsMinF = radial_partitioning.nsMinF;
      const int nsMaxF = radial_partitioning.nsMaxF;
      const int nsMaxFIncludingLcfs = radial_partitioning.nsMaxFIncludingLcfs;

      const int nsMinH = radial_partitioning.nsMinH;
      const int nsMaxH = radial_partitioning.nsMaxH;

      // check preconditioner matrix elements again, just to be sure...
      for (int jH = nsMinH; jH < nsMaxH; ++jH) {
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["arm"][jH + 1][kEvenParity],
            vmec.m_[thread_id]->arm[(jH - nsMinH) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["arm"][jH + 1][kOddParity],
            vmec.m_[thread_id]->arm[(jH - nsMinH) * 2 + kOddParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["azm"][jH + 1][kEvenParity],
            vmec.m_[thread_id]->azm[(jH - nsMinH) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["azm"][jH + 1][kOddParity],
            vmec.m_[thread_id]->azm[(jH - nsMinH) * 2 + kOddParity],
            tolerance));

        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["brm"][jH + 1][kEvenParity],
            vmec.m_[thread_id]->brm[(jH - nsMinH) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["brm"][jH + 1][kOddParity],
            vmec.m_[thread_id]->brm[(jH - nsMinH) * 2 + kOddParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["bzm"][jH + 1][kEvenParity],
            vmec.m_[thread_id]->bzm[(jH - nsMinH) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["bzm"][jH + 1][kOddParity],
            vmec.m_[thread_id]->bzm[(jH - nsMinH) * 2 + kOddParity],
            tolerance));
      }  // jH

      for (int jF = nsMinF; jF < nsMaxF; ++jF) {
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["ard"][jF][kEvenParity],
            vmec.m_[thread_id]->ard[(jF - nsMinF) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["ard"][jF][kOddParity],
            vmec.m_[thread_id]->ard[(jF - nsMinF) * 2 + kOddParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["azd"][jF][kEvenParity],
            vmec.m_[thread_id]->azd[(jF - nsMinF) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["azd"][jF][kOddParity],
            vmec.m_[thread_id]->azd[(jF - nsMinF) * 2 + kOddParity],
            tolerance));

        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["brd"][jF][kEvenParity],
            vmec.m_[thread_id]->brd[(jF - nsMinF) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["brd"][jF][kOddParity],
            vmec.m_[thread_id]->brd[(jF - nsMinF) * 2 + kOddParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["bzd"][jF][kEvenParity],
            vmec.m_[thread_id]->bzd[(jF - nsMinF) * 2 + kEvenParity],
            tolerance));
        EXPECT_TRUE(IsCloseRelAbs(
            scalfor_out["bzd"][jF][kOddParity],
            vmec.m_[thread_id]->bzd[(jF - nsMinF) * 2 + kOddParity],
            tolerance));

        EXPECT_TRUE(IsCloseRelAbs(scalfor_out["crd"][jF],
                                  vmec.m_[thread_id]->cxd[jF - nsMinF],
                                  tolerance));
      }  // jF

      for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
        for (int jF = nsMinF; jF < nsMaxF; ++jF) {
          for (int m = 0; m < s.mpol; ++m) {
            for (int n = 0; n < s.ntor + 1; ++n) {
              EXPECT_TRUE(
                  IsCloseRelAbs(scalfor_out["gcr"][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kR, idx_basis, jF, n, m),
                                tolerance));
              EXPECT_TRUE(
                  IsCloseRelAbs(scalfor_out["gcz"][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kZ, idx_basis, jF, n, m),
                                tolerance));
            }  // n
          }  // m
        }  // jF
      }  // idx_basis

      for (int idx_basis = 0; idx_basis < s.num_basis; ++idx_basis) {
        for (int jF = nsMinF; jF < nsMaxFIncludingLcfs; ++jF) {
          for (int m = 0; m < s.mpol; ++m) {
            for (int n = 0; n < s.ntor + 1; ++n) {
              EXPECT_TRUE(
                  IsCloseRelAbs(scalfor_out["gcl"][idx_basis][jF][n][m],
                                vmec.decomposed_f_[thread_id]->GetXcElement(
                                    kL, idx_basis, jF, n, m),
                                tolerance));
            }  // n
          }  // m
        }  // jF
      }  // idx_basis
    }  // thread_id
  }
}  // CheckApplyPreconditioner

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, ApplyPreconditionerTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-14},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-14},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 1.0e-12},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 1.0e-12},
           DataSource{
               .identifier = "cma", .tolerance = 2.0e-11, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-12,
                      .iter2_to_test = {1, 2, 53, 54}}));

class PreconditionedResidualsTest : public TestWithParam<DataSource> {
 protected:
  void SetUp() override { data_source_ = GetParam(); }
  DataSource data_source_;
};

TEST_P(PreconditionedResidualsTest, CheckPreconditionedResiduals) {
  const double tolerance = data_source_.tolerance;

  std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", data_source_.identifier);
  absl::StatusOr<std::string> indata_json = ReadFile(filename);
  ASSERT_TRUE(indata_json.ok());

  const absl::StatusOr<VmecINDATA> vmec_indata =
      VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(vmec_indata.ok());

  for (int number_of_iterations : data_source_.iter2_to_test) {
    Vmec vmec(*vmec_indata);
    const FlowControl& fc = vmec.fc_;

    bool reached_checkpoint =
        vmec.run(VmecCheckpoint::PRECONDITIONED_RESIDUALS, number_of_iterations)
            .value();
    EXPECT_TRUE(reached_checkpoint);

    filename = absl::StrFormat(
        "vmecpp_large_cpp_tests/test_data/%s/fsq1/fsq1_%05d_%06d_%02d.%s.json",
        data_source_.identifier, fc.ns, number_of_iterations,
        vmec.get_num_eqsolve_retries(), data_source_.identifier);
    std::ifstream ifs_fsq1(filename);
    ASSERT_TRUE(ifs_fsq1.is_open())
        << "failed to open reference file: " << filename;
    json fsq1 = json::parse(ifs_fsq1);

    EXPECT_TRUE(IsCloseRelAbs(fsq1["fnorm1"], vmec.h_.fNorm1, tolerance));

    EXPECT_TRUE(IsCloseRelAbs(fsq1["fsqr1"], fc.fsqr1, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(fsq1["fsqz1"], fc.fsqz1, tolerance));
    EXPECT_TRUE(IsCloseRelAbs(fsq1["fsql1"], fc.fsql1, tolerance));
  }
}  // CheckPreconditionedResiduals

INSTANTIATE_TEST_SUITE_P(
    TestIdealMHDModel, PreconditionedResidualsTest,
    Values(DataSource{.identifier = "solovev", .tolerance = 1.0e-16},
           DataSource{.identifier = "solovev_no_axis", .tolerance = 1.0e-16},
           DataSource{.identifier = "cth_like_fixed_bdy", .tolerance = 5.0e-16},
           DataSource{.identifier = "cth_like_fixed_bdy_nzeta_37",
                      .tolerance = 5.0e-16},
           DataSource{
               .identifier = "cma", .tolerance = 1.1e-14, .iter2_to_test = {1}},
           DataSource{.identifier = "cth_like_free_bdy",
                      .tolerance = 1.0e-13,
                      .iter2_to_test = {1, 2, 53, 54}}));

}  // namespace vmecpp

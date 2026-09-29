// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// Two properties of the ideal-MHD stability quantities VMEC++ provides.
//
// The second variation of the MHD energy is a symmetric form. Where the
// unpreconditioned force is the gradient of the energy (axisymmetric
// equilibria at prescribed iota with the spectral-condensation weight zero),
// its derivative along a displacement is the energy's Hessian applied to it, so
// <u, H v> = <v, H u> for any two admissible displacements; a central
// difference of the force gives H v.
//
// The geodesic-curvature term of the Mercier criterion,
//
//   DGeod = <G B^2 / |grad s|^3>^2 - <G^2 B^2 / |grad s|^3> <B^2 / |grad s|^3>
//
// in the notation of VMEC's mercier.f90, with G the parallel current density
// over B^2, is a Cauchy-Schwarz defect: the square of an average of a product
// never exceeds the product of the averages of the squares, so DGeod <= 0 on
// every surface whatever the equilibrium.
#include <algorithm>
#include <climits>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>

#include "Eigen/Dense"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/iteration_logger/iteration_logger.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {
namespace {

constexpr int kNs = 11;

VmecINDATA ReadIndata(const std::string& name) {
  const absl::StatusOr<std::string> json =
      file_io::ReadFile("vmecpp/test_data/" + name);
  CHECK_OK(json);
  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*json);
  CHECK_OK(indata);
  return *std::move(indata);
}

// A single-threaded Vmec at kNs surfaces with tcon0 zero, set up as Vmec::run
// sets up one multigrid step.
std::unique_ptr<Vmec> MakeVmec(const std::string& name) {
  VmecINDATA indata = ReadIndata(name);
  CHECK_EQ(indata.ncurr, 0);
  CHECK(!indata.lasym);
  CHECK_EQ(indata.ntor, 0);
  indata.tcon0 = 0.0;
  absl::StatusOr<std::unique_ptr<Vmec>> vmec =
      Vmec::FromIndata(indata, /*magnetic_response_table=*/nullptr,
                       /*max_threads=*/1, OutputMode::kSilent);
  CHECK_OK(vmec);
  Vmec& v = **vmec;
  v.fc_.ns_old = 0;
  v.fc_.delt0r = v.indata_.delt;
  v.fc_.ns_min = 3;
  v.fc_.nsval = kNs;
  // the tolerance and iteration budget of the ns_array entry at kNs, else of
  // the last entry
  const Eigen::VectorXi& ns_array = v.indata_.ns_array;
  int step = static_cast<int>(ns_array.size()) - 1;
  for (int i = 0; i < ns_array.size(); ++i) {
    if (ns_array[i] == kNs) {
      step = i;
      break;
    }
  }
  v.fc_.ftolv = v.indata_.ftol_array[step];
  v.fc_.niterv = v.indata_.niter_array[step];
  double delt0 = v.indata_.delt;
  CHECK_OK(v.InitializeRadial(VmecCheckpoint::NONE, INT_MAX, kNs,
                              /*ns_old=*/0, delt0));
  return std::move(*vmec);
}

Eigen::VectorXd Flatten(std::initializer_list<std::span<double>> blocks) {
  Eigen::Index size = 0;
  for (const std::span<double> block : blocks) {
    size += static_cast<Eigen::Index>(block.size());
  }
  Eigen::VectorXd flat(size);
  Eigen::Index offset = 0;
  for (const std::span<double> block : blocks) {
    const auto n = static_cast<Eigen::Index>(block.size());
    flat.segment(offset, n) =
        Eigen::Map<const Eigen::VectorXd>(block.data(), n);
    offset += n;
  }
  return flat;
}

// The coefficients of R, Z and lambda, one after another, each surface-major.
Eigen::VectorXd State(const Vmec& vmec) {
  const FourierGeometry& x = *vmec.decomposed_x_[0];
  return Flatten({x.rmncc, x.zmnsc, x.lmnsc});
}

void SetState(const Eigen::VectorXd& state, Vmec& m_vmec) {
  FourierGeometry& x = *m_vmec.decomposed_x_[0];
  Eigen::Index offset = 0;
  for (const std::span<double> block : {x.rmncc, x.zmnsc, x.lmnsc}) {
    const auto n = static_cast<Eigen::Index>(block.size());
    Eigen::Map<Eigen::VectorXd>(block.data(), n) = state.segment(offset, n);
    offset += n;
  }
  CHECK_EQ(offset, state.size());
}

// The raw force at the state, in its layout: the forward model run to the
// INVARIANT_RESIDUALS checkpoint with the m = 1 gauge fixed on the state.
Eigen::VectorXd RawForce(const Eigen::VectorXd& state, Vmec& m_vmec) {
  SetState(state, m_vmec);
  bool need_restart = false;
  int last_preconditioner_update = 0;
  int last_full_update_nestor = 0;
  m_vmec.fc_.restart_reason = RestartReason::NO_RESTART;
  absl::StatusOr<bool> status;
#ifdef _OPENMP
#pragma omp parallel num_threads(1)
#endif
  {
    status = m_vmec.m_[0]->update(
        *m_vmec.decomposed_x_[0], *m_vmec.physical_x_[0],
        *m_vmec.decomposed_f_[0], *m_vmec.physical_f_[0], need_restart,
        last_preconditioner_update, last_full_update_nestor, m_vmec.fc_,
        /*iter1=*/2, /*iter2=*/2, VmecCheckpoint::INVARIANT_RESIDUALS,
        /*iterations_before_checkpointing=*/0, /*verbose=*/false,
        /*always_fix_m1_gauge=*/true);
  }
  CHECK_OK(status);
  CHECK(m_vmec.fc_.restart_reason == RestartReason::NO_RESTART);
  const FourierForces& f = *m_vmec.decomposed_f_[0];
  return Flatten({f.frcc, f.fzsc, f.flsc});
}

// A random displacement of the R and Z coefficients of the surfaces from the
// second interior one to the one below the boundary, m = 1 left out, each
// scaled to the coefficient of x it moves.
Eigen::VectorXd Admissible(const Eigen::VectorXd& x, const Sizes& s,
                           std::mt19937& m_rng) {
  const int modes_per_surface = s.mpol * (s.ntor + 1);
  CHECK_EQ(x.size(), 3 * kNs * modes_per_surface);
  std::normal_distribution<double> normal;
  Eigen::VectorXd xi = Eigen::VectorXd::Zero(x.size());
  // R and Z; lambda is the last block
  for (int block = 0; block < 2; ++block) {
    for (int j = 2; j < kNs - 1; ++j) {
      for (int k = 0; k < modes_per_surface; ++k) {
        if (k / (s.ntor + 1) == 1) {
          continue;
        }
        const int i = (block * kNs + j) * modes_per_surface + k;
        xi[i] = normal(m_rng) * (std::abs(x[i]) + 1e-3);
      }
    }
  }
  return xi;
}

class EnergyHessianTest : public ::testing::TestWithParam<std::string> {};

TEST_P(EnergyHessianTest, IsSymmetric) {
  std::unique_ptr<Vmec> vmec = MakeVmec(GetParam());
  ASSERT_TRUE(vmec->SolveEquilibrium(VmecCheckpoint::NONE, INT_MAX).ok());
  const Eigen::VectorXd x = State(*vmec);
  std::mt19937 rng(3);
  const double eps = 1e-6;
  for (int trial = 0; trial < 3; ++trial) {
    const Eigen::VectorXd u = Admissible(x, vmec->s_, rng);
    const Eigen::VectorXd v = Admissible(x, vmec->s_, rng);
    const Eigen::VectorXd hu =
        (RawForce(x + eps * u, *vmec) - RawForce(x - eps * u, *vmec)) /
        (2.0 * eps);
    const Eigen::VectorXd hv =
        (RawForce(x + eps * v, *vmec) - RawForce(x - eps * v, *vmec)) /
        (2.0 * eps);
    const double scale = std::max(std::abs(u.dot(hu)), std::abs(v.dot(hv)));
    EXPECT_LE(std::abs(u.dot(hv) - v.dot(hu)), 1e-6 * scale)
        << GetParam() << ", trial " << trial;
  }
}

INSTANTIATE_TEST_SUITE_P(Axisymmetric, EnergyHessianTest,
                         ::testing::Values("solovev.json",
                                           "circular_tokamak.json"));

class GeodesicCurvatureTest : public ::testing::TestWithParam<std::string> {};

TEST_P(GeodesicCurvatureTest, IsNeverPositive) {
  const absl::StatusOr<OutputQuantities> output =
      vmecpp::run(ReadIndata(GetParam()), /*initial_state=*/std::nullopt,
                  /*max_threads=*/std::nullopt, OutputMode::kSilent);
  ASSERT_TRUE(output.ok()) << output.status();
  const Eigen::VectorXd& dgeod = output->wout.DGeod;
  // the axis and the boundary carry no Mercier terms
  const Eigen::VectorXd interior = dgeod.segment(1, dgeod.size() - 2);
  const double largest = interior.cwiseAbs().maxCoeff();
  const double scale = largest > 0.0 ? largest : 1.0;
  EXPECT_LE(interior.maxCoeff(), 1e-12 * scale) << GetParam();
}

INSTANTIATE_TEST_SUITE_P(
    TestCases, GeodesicCurvatureTest,
    ::testing::Values("solovev.json", "circular_tokamak.json",
                      "cth_like_fixed_bdy.json", "cth_like_fixed_bdy_asym.json",
                      "cma.json", "li383_low_res.json", "up_down_asym.json"));

}  // namespace
}  // namespace vmecpp

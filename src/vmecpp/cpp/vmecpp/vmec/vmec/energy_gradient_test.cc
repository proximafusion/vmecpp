// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// The unpreconditioned force is the gradient of the MHD energy. A central
// finite difference of the energy along a displacement of the R and Z
// coefficients equals the raw force, the forward model stopped at the
// INVARIANT_RESIDUALS checkpoint, contracted with that displacement, times the
// radial integration step 1 / (ns - 1) the energy carries and the force does
// not. The equilibria prescribe iota, so the energy is taken at fixed flux
// functions, and the spectral-condensation weight tcon0 is zero, since the
// constraint force it adds is not a gradient of the energy. The displacement
// leaves the axis, the first interior surface and the boundary, which the force
// treats on their own, and the m = 1 coefficients, which the poloidal-origin
// gauge couples between R and Z.
#include <climits>
#include <cmath>
#include <initializer_list>
#include <memory>
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
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {
namespace {

constexpr int kNs = 11;

// A single-threaded Vmec at kNs surfaces with tcon0 zero, set up as Vmec::run
// sets up one multigrid step.
std::unique_ptr<Vmec> MakeVmec(const std::string& name) {
  const absl::StatusOr<std::string> json =
      file_io::ReadFile("vmecpp/test_data/" + name);
  CHECK_OK(json);
  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*json);
  CHECK_OK(indata);
  CHECK_EQ(indata->ncurr, 0);
  CHECK(!indata->lasym);
  CHECK_EQ(indata->ntor, 0);
  indata->tcon0 = 0.0;
  absl::StatusOr<std::unique_ptr<Vmec>> vmec =
      Vmec::FromIndata(*indata, /*magnetic_response_table=*/nullptr,
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

// The forces on R, Z and lambda, in the layout of State.
Eigen::VectorXd RawForce(const Vmec& vmec) {
  const FourierForces& f = *vmec.decomposed_f_[0];
  return Flatten({f.frcc, f.fzsc, f.flsc});
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

// Runs the forward model to the INVARIANT_RESIDUALS checkpoint, where
// decomposed_f_ holds the raw force and h_.mhdEnergy the energy, with the m = 1
// gauge fixed on the state.
void Evaluate(Vmec& m_vmec) {
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
  // a flipped Jacobian ends the evaluation before the energy is formed
  CHECK(m_vmec.fc_.restart_reason == RestartReason::NO_RESTART);
}

double Energy(const Eigen::VectorXd& state, Vmec& m_vmec) {
  SetState(state, m_vmec);
  Evaluate(m_vmec);
  return m_vmec.h_.mhdEnergy;
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

class EnergyGradientTest : public ::testing::TestWithParam<std::string> {};

TEST_P(EnergyGradientTest, EnergyDifferenceIsForce) {
  std::unique_ptr<Vmec> vmec = MakeVmec(GetParam());
  ASSERT_TRUE(vmec->SolveEquilibrium(VmecCheckpoint::NONE, INT_MAX).ok());
  const Eigen::VectorXd x_eq = State(*vmec);
  std::mt19937 rng(7);
  for (int trial = 0; trial < 3; ++trial) {
    // a state displaced from the equilibrium, where the force is not zero
    SetState(x_eq + 1e-3 * Admissible(x_eq, vmec->s_, rng), *vmec);
    Evaluate(*vmec);
    const Eigen::VectorXd x = State(*vmec);
    const Eigen::VectorXd force = RawForce(*vmec);
    const Eigen::VectorXd xi = Admissible(x, vmec->s_, rng);
    const double eps = 1e-5;
    const double difference =
        (Energy(x + eps * xi, *vmec) - Energy(x - eps * xi, *vmec)) /
        (2.0 * eps);
    const double contracted = force.dot(xi) / (kNs - 1);
    EXPECT_NEAR(difference, contracted, 1e-4 * std::abs(contracted))
        << GetParam() << ", trial " << trial;
  }
}

// At the equilibrium the first variation of the energy along an admissible
// displacement is below a thousandth of what it is a thousandth of the
// coefficients away along that displacement.
TEST_P(EnergyGradientTest, FirstVariationVanishesAtEquilibrium) {
  std::unique_ptr<Vmec> vmec = MakeVmec(GetParam());
  ASSERT_TRUE(vmec->SolveEquilibrium(VmecCheckpoint::NONE, INT_MAX).ok());
  const Eigen::VectorXd x = State(*vmec);
  std::mt19937 rng(11);
  const Eigen::VectorXd xi = Admissible(x, vmec->s_, rng);
  const double eps = 1e-5;
  const auto first_variation = [&](const Eigen::VectorXd& state) {
    return (Energy(state + eps * xi, *vmec) - Energy(state - eps * xi, *vmec)) /
           (2.0 * eps);
  };
  const double at_equilibrium = first_variation(x);
  const double displaced = first_variation(x + 1e-3 * xi);
  EXPECT_LT(std::abs(at_equilibrium), 1e-3 * std::abs(displaced)) << GetParam();
}

INSTANTIATE_TEST_SUITE_P(Axisymmetric, EnergyGradientTest,
                         ::testing::Values("solovev.json",
                                           "circular_tokamak.json"));

}  // namespace
}  // namespace vmecpp

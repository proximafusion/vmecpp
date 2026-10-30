// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Microbenchmark for deAliasConstraintForce, the spectral-condensation
// constraint-force de-aliasing step (a forward+inverse poloidal/toroidal
// transform pair) run every iteration inside IdealMhdModel::update().
//
// Sweeps the same four representative (nfp, mpol, ntor) resolutions used by
// fft_toroidal_bench.cc so the constraint-force cost can be compared directly
// against the geometry/force transforms at each resolution.

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "Eigen/Dense"
#include "benchmark/benchmark.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {
namespace {

// ns = 51 is representative of a medium-resolution VMEC run (matches the
// convention used in fft_toroidal_bench.cc).
constexpr int kNs = 51;

// (nfp, mpol, ntor) resolutions, matching fft_toroidal_bench.cc for direct
// comparability.
struct ResParams {
  int nfp;
  int mpol;
  int ntor;
  const char* label;
};

constexpr ResParams kResolutions[] = {
    {1, 4, 4, "4x4"},
    {1, 7, 1, "7x1"},
    {5, 12, 12, "12x12"},
    {5, 16, 18, "16x18"},
};

// ----------------------------------------------------------------------------
// Shared fixture data, built once per (nfp, mpol, ntor) combination.
//
// Mirrors the buffer sizes established in IdealMhdModel::allocate():
//   faccon:  s.mpol,       with faccon[m-1] = -0.25 / xmpq[m]^2 for m > 1
//   tcon:    nsMaxFIncludingLcfs - nsMinF
//   gConEff: nZnT * (nsMaxFIncludingLcfs - nsMinF)   (nrztIncludingBoundary)
//   gsc/gcs: ntor + 1
//   gCon:    nrztIncludingBoundary
// ----------------------------------------------------------------------------
struct BenchFixture {
  Sizes s;
  RadialPartitioning rp;
  FourierBasisFastPoloidal fb;

  Eigen::VectorXd faccon;
  Eigen::VectorXd tcon;
  Eigen::VectorXd gConEff;
  Eigen::VectorXd gsc;
  Eigen::VectorXd gcs;
  Eigen::VectorXd gCon;

  explicit BenchFixture(int nfp, int mpol, int ntor)
      : s(/*lasym=*/false, nfp, mpol, ntor, /*ntheta=*/0, /*nzeta=*/0), fb(&s) {
    rp.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0, kNs,
                                /*lfreeb=*/false, /*printout=*/false);

    const int nrztIncludingBoundary =
        s.nZnT * (rp.nsMaxFIncludingLcfs - rp.nsMinF);

    // faccon[m] = -0.25 / (m*(m-1))^2 for m > 1 (signOfJacobian = -1 gives the
    // same magnitude sign convention as the solver; the exact sign does not
    // affect the timing).
    faccon.setZero(s.mpol);
    for (int m = 2; m < s.mpol; ++m) {
      const double xmpq = m * (m - 1);
      faccon[m - 1] = -0.25 * (-1) / (xmpq * xmpq);
    }

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto rfill = [&](Eigen::VectorXd& v, int n) {
      v.resize(n);
      for (int i = 0; i < n; ++i) v[i] = dist(rng);
    };

    rfill(tcon, rp.nsMaxFIncludingLcfs - rp.nsMinF);
    rfill(gConEff, nrztIncludingBoundary);

    gsc.setZero(s.ntor + 1);
    gcs.setZero(s.ntor + 1);
    gCon.setZero(nrztIncludingBoundary);
  }
};

template <int kIdx>
void BM_DeAliasConstraintForce(benchmark::State& state) {
  static BenchFixture fx(kResolutions[kIdx].nfp, kResolutions[kIdx].mpol,
                         kResolutions[kIdx].ntor);
  for (auto _ : state) {
    deAliasConstraintForce(fx.rp, fx.fb, fx.s, fx.faccon, fx.tcon, fx.gConEff,
                           fx.gsc, fx.gcs, fx.gCon);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kResolutions[kIdx].label);
}

BENCHMARK_TEMPLATE(BM_DeAliasConstraintForce, 0)
    ->Name("DeAliasConstraintForce/4x4");
BENCHMARK_TEMPLATE(BM_DeAliasConstraintForce, 1)
    ->Name("DeAliasConstraintForce/7x1");
BENCHMARK_TEMPLATE(BM_DeAliasConstraintForce, 2)
    ->Name("DeAliasConstraintForce/12x12");
BENCHMARK_TEMPLATE(BM_DeAliasConstraintForce, 3)
    ->Name("DeAliasConstraintForce/16x18");

}  // namespace
}  // namespace vmecpp

BENCHMARK_MAIN();

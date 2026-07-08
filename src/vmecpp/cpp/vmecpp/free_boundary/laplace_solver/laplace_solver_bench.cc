// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Microbenchmarks for the free-boundary Laplace solve (NESTOR).
//
// The dense linear system assembled and factorized here is the dominant
// free-boundary cost when NESTOR performs a full update: an
// mnpd x mnpd  (mnpd = (mf+1)*(2*nf+1))  matrix is built, LU-factorized via
// LAPACK dgetrf, and back-substituted via dgetrs.  We benchmark:
//
//   * TransformGreensFunctionDerivative -- the largest of the Fourier
//     transforms feeding the source term (uses the manufactured single-mode
//     input from laplace_solver_test.cc).
//   * BuildMatrix + DecomposeMatrix + SolveForPotential -- the assemble +
//     factorize + solve chain, which is the actual "Laplace solve".
//
// Sizes are chosen to bracket real free-boundary runs (cth_like / solovev
// free-boundary configurations use ntor ~ 4-6, mpol ~ 5-8).

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>
#include <random>
#include <span>
#include <vector>

#include "Eigen/Dense"
#include "benchmark/benchmark.h"
#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/free_boundary/laplace_solver/laplace_solver.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {
namespace {

struct ResParams {
  int nfp;
  int mpol;
  int ntor;
  const char* label;
};

// (nfp, mpol, ntor) sizes bracketing real free-boundary runs.
constexpr ResParams kResolutions[] = {
    {5, 5, 4, "5x4"},
    {5, 8, 6, "8x6"},
    {5, 12, 8, "12x8"},
};

// ----------------------------------------------------------------------------
// Fixture matching laplace_solver_test.cc's minimal single-threaded setup:
// a non-mocked LaplaceSolver with flat matrixShare / iPiv / bvecShare backing
// storage.
// ----------------------------------------------------------------------------
struct BenchFixture {
  Sizes s;
  FourierBasisFastToroidal fb;
  TangentialPartitioning tp;

  int nf;
  int mf;
  int mnpd;

  std::vector<double> matrixShare;
  std::vector<int> iPiv;
  std::vector<double> bvecShare;

  std::unique_ptr<LaplaceSolver> ls;

  // Seeded, well-conditioned system state re-applied before each timed solve.
  Eigen::VectorXd amat_seed;           // diagonally-dominant [mnpd*mnpd]
  Eigen::VectorXd bvec_singular_seed;  // [mnpd]

  // Manufactured single-mode Green's-function-derivative input.
  Eigen::VectorXd greenp;

  explicit BenchFixture(int nfp, int mpol, int ntor)
      : s(/*lasym=*/false, nfp, mpol, ntor, /*ntheta=*/0,
          /*nzeta=*/4 * (ntor + 1)),
        fb(&s),
        tp(s.nZnT),
        nf(ntor),
        mf(mpol + 1),
        mnpd((mf + 1) * (2 * nf + 1)),
        matrixShare(mnpd * mnpd, 0.0),
        iPiv(mnpd, 0),
        bvecShare(mnpd, 0.0) {
    ls = std::make_unique<LaplaceSolver>(
        &s, &fb, &tp, nf, mf, std::span<double>(matrixShare),
        std::span<int>(iPiv), std::span<double>(bvecShare));

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Diagonally-dominant matrix so dgetrf never hits a singular pivot.
    amat_seed.resize(mnpd * mnpd);
    for (int col = 0; col < mnpd; ++col) {
      for (int row = 0; row < mnpd; ++row) {
        amat_seed[col * mnpd + row] = dist(rng);
      }
      // BuildMatrix() adds +0.5 on the diagonal; add a large diagonal boost
      // here too to guarantee well-conditioned LU.
      amat_seed[col * mnpd + col] += 2.0 * mnpd;
    }

    bvec_singular_seed.resize(mnpd);
    for (int i = 0; i < mnpd; ++i) bvec_singular_seed[i] = dist(rng);

    // Manufactured single-mode greenp, as in laplace_solver_test.cc.
    const int numLocal = tp.ztMax - tp.ztMin;
    const int nThetaEven = s.nThetaEven;
    const int nZeta = s.nZeta;
    greenp = Eigen::VectorXd::Zero(numLocal * nThetaEven * nZeta);
    const int m0 = std::min(2, mpol - 1);
    const int n0 = std::min(1, ntor);
    for (int klpRel = 0; klpRel < numLocal; ++klpRel) {
      for (int l = 0; l < nThetaEven; ++l) {
        const double theta = 2.0 * std::numbers::pi * l / nThetaEven;
        for (int k = 0; k < nZeta; ++k) {
          const double phi = 2.0 * std::numbers::pi * k / nZeta;
          const int idx = klpRel * nThetaEven * nZeta + l * nZeta + k;
          greenp[idx] = std::sin(m0 * theta - n0 * phi);
        }
      }
    }
  }

  // Re-seed the LaplaceSolver's system state to the well-conditioned baseline.
  // dgetrf destroys matrixShare in place, so this must run before each
  // decompose+solve iteration.
  void ResetSystem() {
    ls->amat_sin_sin = amat_seed;
    ls->bvec_sin = Eigen::VectorXd::Zero(mnpd);
  }
};

// The assemble + factorize + solve chain: this is the actual Laplace solve.
template <int kIdx>
void BM_LaplaceSolve(benchmark::State& state) {
  static BenchFixture fx(kResolutions[kIdx].nfp, kResolutions[kIdx].mpol,
                         kResolutions[kIdx].ntor);
  for (auto _ : state) {
    state.PauseTiming();
    fx.ResetSystem();
    state.ResumeTiming();

    fx.ls->BuildMatrix();
    fx.ls->DecomposeMatrix();
    fx.ls->SolveForPotential(fx.bvec_singular_seed);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kResolutions[kIdx].label);
}

// Just the LU factorization (dgetrf), the O(mnpd^3) hotspot.
template <int kIdx>
void BM_LaplaceDecompose(benchmark::State& state) {
  static BenchFixture fx(kResolutions[kIdx].nfp, kResolutions[kIdx].mpol,
                         kResolutions[kIdx].ntor);
  for (auto _ : state) {
    state.PauseTiming();
    fx.ResetSystem();
    fx.ls->BuildMatrix();
    state.ResumeTiming();

    fx.ls->DecomposeMatrix();
    benchmark::ClobberMemory();
  }
  state.SetLabel(kResolutions[kIdx].label);
}

// The largest Green's-function-derivative Fourier transform.
template <int kIdx>
void BM_TransformGreensFunctionDerivative(benchmark::State& state) {
  static BenchFixture fx(kResolutions[kIdx].nfp, kResolutions[kIdx].mpol,
                         kResolutions[kIdx].ntor);
  for (auto _ : state) {
    fx.ls->TransformGreensFunctionDerivative(fx.greenp);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kResolutions[kIdx].label);
}

BENCHMARK_TEMPLATE(BM_LaplaceSolve, 0)->Name("LaplaceSolve/5x4");
BENCHMARK_TEMPLATE(BM_LaplaceSolve, 1)->Name("LaplaceSolve/8x6");
BENCHMARK_TEMPLATE(BM_LaplaceSolve, 2)->Name("LaplaceSolve/12x8");

BENCHMARK_TEMPLATE(BM_LaplaceDecompose, 0)->Name("LaplaceDecompose/5x4");
BENCHMARK_TEMPLATE(BM_LaplaceDecompose, 1)->Name("LaplaceDecompose/8x6");
BENCHMARK_TEMPLATE(BM_LaplaceDecompose, 2)->Name("LaplaceDecompose/12x8");

BENCHMARK_TEMPLATE(BM_TransformGreensFunctionDerivative, 0)
    ->Name("TransformGreensFunctionDerivative/5x4");
BENCHMARK_TEMPLATE(BM_TransformGreensFunctionDerivative, 1)
    ->Name("TransformGreensFunctionDerivative/8x6");
BENCHMARK_TEMPLATE(BM_TransformGreensFunctionDerivative, 2)
    ->Name("TransformGreensFunctionDerivative/12x8");

}  // namespace
}  // namespace vmecpp

BENCHMARK_MAIN();

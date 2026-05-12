// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Microbenchmarks for the toroidal FFT hot loop.
// Covers both directions (spectral->real, real->spectral) at four
// representative resolutions used in typical VMEC runs.
//
// The parallel benchmark (BM_FourierToReal_Parallel) matches the actual VMEC
// call pattern: N threads simultaneously call
// FourierToReal3DSymmFastPoloidalFft on their own radial slice, sharing only
// the read-only ToroidalFftPlans.

#include <omp.h>

#include <atomic>
#include <random>
#include <span>
#include <vector>

#include "Eigen/Dense"
#include "benchmark/benchmark.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {
namespace {

// ns = 51 is representative of a medium-resolution VMEC run.
constexpr int kNs = 51;

// ----------------------------------------------------------------------------
// Shared fixture data, built once per (nfp, mpol, ntor) combination.
//
// RadialProfiles stores pointers/references to its constructor arguments, so
// all dependencies (indata, handover, fc) must outlive the RadialProfiles
// object.  We heap-allocate everything here.
// ----------------------------------------------------------------------------

struct BenchFixture {
  // Dependencies that RadialProfiles points into -- must be declared first.
  Sizes s;
  RadialPartitioning rp;
  FourierBasisFastPoloidal fb;
  ToroidalFftPlans plans;
  Eigen::VectorXd xmpq;
  std::unique_ptr<VmecINDATA> indata;
  std::unique_ptr<HandoverStorage> handover;
  std::unique_ptr<FlowControl> fc;
  std::unique_ptr<RadialProfiles> rprof;

  // Forward-transform inputs.
  std::unique_ptr<FourierGeometry> phys_x;

  // Forward-transform output storage.
  std::vector<double> r1_e, r1_o, ru_e, ru_o, rv_e, rv_o;
  std::vector<double> z1_e, z1_o, zu_e, zu_o, zv_e, zv_o;
  std::vector<double> lu_e, lu_o, lv_e, lv_o;
  std::vector<double> rCon, zCon;
  RealSpaceGeometry geom;

  // Inverse-transform inputs.
  std::vector<double> armn_e, armn_o, azmn_e, azmn_o;
  std::vector<double> blmn_e, blmn_o, brmn_e, brmn_o, bzmn_e, bzmn_o;
  std::vector<double> clmn_e, clmn_o, crmn_e, crmn_o, czmn_e, czmn_o;
  std::vector<double> frcon_e, frcon_o, fzcon_e, fzcon_o;
  RealSpaceForces forces;

  // Inverse-transform output.
  std::unique_ptr<FourierForces> ff;

  explicit BenchFixture(int nfp, int mpol, int ntor, int ntheta = 0,
                        int nzeta = 0)
      : s(/*lasym=*/false, nfp, mpol, ntor, ntheta, nzeta),
        fb(&s),
        plans(s.nZeta, s.nfp, s.mpol),
        indata(std::make_unique<VmecINDATA>()),
        handover(std::make_unique<HandoverStorage>(&s)),
        fc(std::make_unique<FlowControl>(/*lfreeb=*/false, /*delt=*/0.9,
                                         /*num_grids=*/1)) {
    rp.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0, kNs,
                                /*lfreeb=*/false, /*printout=*/false);

    xmpq.resize(s.mpol);
    for (int m = 0; m < s.mpol; ++m) xmpq[m] = m * (m - 1);

    fc->ns = kNs;

    rprof = std::make_unique<RadialProfiles>(&rp, handover.get(), indata.get(),
                                             fc.get(), /*signOfJacobian=*/-1,
                                             /*pDamp=*/0.05);
    const int nsurf = rp.nsMaxF1 - rp.nsMinF1;
    rprof->sqrtSF.resize(nsurf);
    for (int j = 0; j < nsurf; ++j) {
      rprof->sqrtSF[j] =
          std::sqrt(0.05 + 0.9 * j / (nsurf > 1 ? nsurf - 1 : 1));
    }

    phys_x = std::make_unique<FourierGeometry>(&s, &rp, kNs);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto rfill = [&](std::span<double> sp) {
      for (double& x : sp) x = dist(rng);
    };
    rfill(phys_x->rmncc);
    rfill(phys_x->rmnss);
    rfill(phys_x->zmnsc);
    rfill(phys_x->zmncs);
    rfill(phys_x->lmnsc);
    rfill(phys_x->lmncs);

    // Allocate forward-transform output.
    const int nrzt1 = s.nZnT * (rp.nsMaxF1 - rp.nsMinF1);
    const int nrzt_con = s.nZnT * (rp.nsMaxFIncludingLcfs - rp.nsMinF);
    auto alloc = [](int n) { return std::vector<double>(n, 0.0); };
    r1_e = alloc(nrzt1);
    r1_o = alloc(nrzt1);
    ru_e = alloc(nrzt1);
    ru_o = alloc(nrzt1);
    rv_e = alloc(nrzt1);
    rv_o = alloc(nrzt1);
    z1_e = alloc(nrzt1);
    z1_o = alloc(nrzt1);
    zu_e = alloc(nrzt1);
    zu_o = alloc(nrzt1);
    zv_e = alloc(nrzt1);
    zv_o = alloc(nrzt1);
    lu_e = alloc(nrzt1);
    lu_o = alloc(nrzt1);
    lv_e = alloc(nrzt1);
    lv_o = alloc(nrzt1);
    rCon = alloc(nrzt_con);
    zCon = alloc(nrzt_con);
    geom =
        RealSpaceGeometry{r1_e, r1_o, ru_e, ru_o, rv_e, rv_o, z1_e, z1_o, zu_e,
                          zu_o, zv_e, zv_o, lu_e, lu_o, lv_e, lv_o, rCon, zCon};

    // Allocate inverse-transform input.
    const int nrzt = s.nZnT * (rp.nsMaxF - rp.nsMinF);
    const int nrzt_lcfs = s.nZnT * (rp.nsMaxFIncludingLcfs - rp.nsMinF);
    auto rvec = [&](int n) {
      std::vector<double> v(n);
      for (double& x : v) x = dist(rng);
      return v;
    };
    armn_e = rvec(nrzt);
    armn_o = rvec(nrzt);
    azmn_e = rvec(nrzt);
    azmn_o = rvec(nrzt);
    blmn_e = rvec(nrzt_lcfs);
    blmn_o = rvec(nrzt_lcfs);
    brmn_e = rvec(nrzt);
    brmn_o = rvec(nrzt);
    bzmn_e = rvec(nrzt);
    bzmn_o = rvec(nrzt);
    clmn_e = rvec(nrzt_lcfs);
    clmn_o = rvec(nrzt_lcfs);
    crmn_e = rvec(nrzt);
    crmn_o = rvec(nrzt);
    czmn_e = rvec(nrzt);
    czmn_o = rvec(nrzt);
    frcon_e = rvec(nrzt);
    frcon_o = rvec(nrzt);
    fzcon_e = rvec(nrzt);
    fzcon_o = rvec(nrzt);
    forces = RealSpaceForces{armn_e, armn_o,  azmn_e,  azmn_o,  blmn_e,
                             blmn_o, brmn_e,  brmn_o,  bzmn_e,  bzmn_o,
                             clmn_e, clmn_o,  crmn_e,  crmn_o,  czmn_e,
                             czmn_o, frcon_e, frcon_o, fzcon_e, fzcon_o};

    ff = std::make_unique<FourierForces>(&s, &rp, kNs);
  }
};

// ----------------------------------------------------------------------------
// Benchmark helpers
// ----------------------------------------------------------------------------

// (nfp, mpol, ntor) pairs for the four benchmark resolutions.
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

// Templated benchmarks parameterised by a ResParams index so GBench can name
// them clearly.  The fixture is a function-local static so it is built exactly
// once per process (not once per benchmark iteration).

template <int kIdx>
void BM_FourierToReal(benchmark::State& state) {
  static BenchFixture fx(kResolutions[kIdx].nfp, kResolutions[kIdx].mpol,
                         kResolutions[kIdx].ntor);
  for (auto _ : state) {
    FourierToReal3DSymmFastPoloidalFft(*fx.phys_x, fx.xmpq, fx.rp, fx.s,
                                       *fx.rprof, fx.fb, fx.plans, fx.geom);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kResolutions[kIdx].label);
}

template <int kIdx>
void BM_ForcesToFourier(benchmark::State& state) {
  static BenchFixture fx(kResolutions[kIdx].nfp, kResolutions[kIdx].mpol,
                         kResolutions[kIdx].ntor);
  for (auto _ : state) {
    ForcesToFourier3DSymmFastPoloidalFft(fx.forces, fx.xmpq, fx.rp, *fx.fc,
                                         fx.s, fx.fb, fx.plans,
                                         VacuumPressureState::kOff, *fx.ff);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kResolutions[kIdx].label);
}

template <int kIdx>
void BM_DftFourierToReal(benchmark::State& state) {
  static BenchFixture fx(kResolutions[kIdx].nfp, kResolutions[kIdx].mpol,
                         kResolutions[kIdx].ntor);
  for (auto _ : state) {
    FourierToReal3DSymmFastPoloidal(*fx.phys_x, fx.xmpq, fx.rp, fx.s, *fx.rprof,
                                    fx.fb, fx.geom);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kResolutions[kIdx].label);
}

BENCHMARK_TEMPLATE(BM_DftFourierToReal, 0)->Name("DftFourierToReal/4x4");
BENCHMARK_TEMPLATE(BM_FourierToReal, 0)->Name("FftFourierToReal/4x4");
BENCHMARK_TEMPLATE(BM_DftFourierToReal, 1)->Name("DftFourierToReal/7x1");
BENCHMARK_TEMPLATE(BM_FourierToReal, 1)->Name("FftFourierToReal/7x1");
BENCHMARK_TEMPLATE(BM_DftFourierToReal, 2)->Name("DftFourierToReal/12x12");
BENCHMARK_TEMPLATE(BM_FourierToReal, 2)->Name("FftFourierToReal/12x12");
BENCHMARK_TEMPLATE(BM_DftFourierToReal, 3)->Name("DftFourierToReal/16x18");
BENCHMARK_TEMPLATE(BM_FourierToReal, 3)->Name("FftFourierToReal/16x18");

// ----------------------------------------------------------------------------
// Real-space resolution sweep at fixed spectral resolution (mpol=12, ntor=12,
// nfp=5).  Default real-space grid is (ntheta=2*mpol+6=30, nzeta=2*ntor+4=28
// when nfp=1; for nfp=5 the toroidal grid extends accordingly).  We vary
// ntheta and nzeta independently to isolate poloidal-AXPY cost (ntheta) from
// toroidal-FFT cost (nzeta).
// ----------------------------------------------------------------------------

struct RealSpaceParams {
  int nfp;
  int mpol;
  int ntor;
  int ntheta;  // 0 = use Sizes default
  int nzeta;   // 0 = use Sizes default
  const char* label;
};

constexpr RealSpaceParams kRealSpaceSweep[] = {
    // Vary nzeta (toroidal real-space grid) at fixed ntheta default.
    {5, 12, 12, 0, 28, "12x12_ntheta-default_nzeta-28"},    // baseline default
    {5, 12, 12, 0, 56, "12x12_ntheta-default_nzeta-56"},    // 2x toroidal
    {5, 12, 12, 0, 84, "12x12_ntheta-default_nzeta-84"},    // 3x toroidal
    {5, 12, 12, 0, 112, "12x12_ntheta-default_nzeta-112"},  // 4x toroidal
    // Vary ntheta (poloidal real-space grid) at fixed nzeta default.
    {5, 12, 12, 30, 0, "12x12_ntheta-30_nzeta-default"},    // baseline default
    {5, 12, 12, 60, 0, "12x12_ntheta-60_nzeta-default"},    // 2x poloidal
    {5, 12, 12, 90, 0, "12x12_ntheta-90_nzeta-default"},    // 3x poloidal
    {5, 12, 12, 120, 0, "12x12_ntheta-120_nzeta-default"},  // 4x poloidal
};

template <int kIdx>
void BM_FftFourierToReal_RealSpace(benchmark::State& state) {
  static BenchFixture fx(kRealSpaceSweep[kIdx].nfp, kRealSpaceSweep[kIdx].mpol,
                         kRealSpaceSweep[kIdx].ntor,
                         kRealSpaceSweep[kIdx].ntheta,
                         kRealSpaceSweep[kIdx].nzeta);
  for (auto _ : state) {
    FourierToReal3DSymmFastPoloidalFft(*fx.phys_x, fx.xmpq, fx.rp, fx.s,
                                       *fx.rprof, fx.fb, fx.plans, fx.geom);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kRealSpaceSweep[kIdx].label);
}

template <int kIdx>
void BM_DftFourierToReal_RealSpace(benchmark::State& state) {
  static BenchFixture fx(kRealSpaceSweep[kIdx].nfp, kRealSpaceSweep[kIdx].mpol,
                         kRealSpaceSweep[kIdx].ntor,
                         kRealSpaceSweep[kIdx].ntheta,
                         kRealSpaceSweep[kIdx].nzeta);
  for (auto _ : state) {
    FourierToReal3DSymmFastPoloidal(*fx.phys_x, fx.xmpq, fx.rp, fx.s, *fx.rprof,
                                    fx.fb, fx.geom);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kRealSpaceSweep[kIdx].label);
}

#define REGISTER_RS(I, NAME)                                               \
  BENCHMARK_TEMPLATE(BM_DftFourierToReal_RealSpace, I)->Name("Dft/" NAME); \
  BENCHMARK_TEMPLATE(BM_FftFourierToReal_RealSpace, I)->Name("Fft/" NAME)

REGISTER_RS(0, "12x12_nzeta-28");
REGISTER_RS(1, "12x12_nzeta-56");
REGISTER_RS(2, "12x12_nzeta-84");
REGISTER_RS(3, "12x12_nzeta-112");
REGISTER_RS(4, "12x12_ntheta-30");
REGISTER_RS(5, "12x12_ntheta-60");
REGISTER_RS(6, "12x12_ntheta-90");
REGISTER_RS(7, "12x12_ntheta-120");

#undef REGISTER_RS

// ----------------------------------------------------------------------------
// Parallel fixture: one BenchFixture per thread, each covering its own radial
// slice, sharing the same ToroidalFftPlans (read-only during the hot loop).
// This matches the real VMEC calling pattern exactly:
//   #pragma omp parallel
//   { models[omp_get_thread_num()].geometryFromFourier(phys_x); }
// ----------------------------------------------------------------------------

struct ParallelBenchFixture {
  // Shared across threads (read-only during hot loop).
  Sizes s;
  FourierBasisFastPoloidal fb;
  ToroidalFftPlans plans;
  Eigen::VectorXd xmpq;

  // Per-thread state: each thread gets its own radial slice.
  struct ThreadSlice {
    RadialPartitioning rp;
    std::unique_ptr<VmecINDATA> indata;
    std::unique_ptr<HandoverStorage> handover;
    std::unique_ptr<FlowControl> fc;
    std::unique_ptr<RadialProfiles> rprof;
    std::unique_ptr<FourierGeometry> phys_x;
    // Output buffers (each thread writes only to its own slice).
    std::vector<double> r1_e, r1_o, ru_e, ru_o, rv_e, rv_o;
    std::vector<double> z1_e, z1_o, zu_e, zu_o, zv_e, zv_o;
    std::vector<double> lu_e, lu_o, lv_e, lv_o;
    std::vector<double> rCon, zCon;
    RealSpaceGeometry geom;
  };

  std::vector<ThreadSlice> threads;

  explicit ParallelBenchFixture(int nfp, int mpol, int ntor, int num_threads)
      : s(/*lasym=*/false, nfp, mpol, ntor, /*ntheta=*/0, /*nzeta=*/0),
        fb(&s),
        plans(s.nZeta, s.nfp, s.mpol),
        threads(num_threads) {
    xmpq.resize(s.mpol);
    for (int m = 0; m < s.mpol; ++m) xmpq[m] = m * (m - 1);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int t = 0; t < num_threads; ++t) {
      ThreadSlice& sl = threads[t];
      sl.rp.adjustRadialPartitioning(num_threads, t, kNs,
                                     /*lfreeb=*/false, /*printout=*/false);
      sl.indata = std::make_unique<VmecINDATA>();
      sl.handover = std::make_unique<HandoverStorage>(&s);
      sl.fc = std::make_unique<FlowControl>(/*lfreeb=*/false, /*delt=*/0.9,
                                            /*num_grids=*/1);
      sl.fc->ns = kNs;
      sl.rprof = std::make_unique<RadialProfiles>(
          &sl.rp, sl.handover.get(), sl.indata.get(), sl.fc.get(),
          /*signOfJacobian=*/-1, /*pDamp=*/0.05);
      const int nsurf = sl.rp.nsMaxF1 - sl.rp.nsMinF1;
      sl.rprof->sqrtSF.resize(nsurf);
      for (int j = 0; j < nsurf; ++j) {
        sl.rprof->sqrtSF[j] =
            std::sqrt(0.05 + 0.9 * j / (nsurf > 1 ? nsurf - 1 : 1));
      }

      sl.phys_x = std::make_unique<FourierGeometry>(&s, &sl.rp, kNs);
      auto rfill = [&](std::span<double> sp) {
        for (double& x : sp) x = dist(rng);
      };
      rfill(sl.phys_x->rmncc);
      rfill(sl.phys_x->rmnss);
      rfill(sl.phys_x->zmnsc);
      rfill(sl.phys_x->zmncs);
      rfill(sl.phys_x->lmnsc);
      rfill(sl.phys_x->lmncs);

      const int nrzt1 = s.nZnT * (sl.rp.nsMaxF1 - sl.rp.nsMinF1);
      const int nrzt_con = s.nZnT * (sl.rp.nsMaxFIncludingLcfs - sl.rp.nsMinF);
      auto alloc = [](int n) { return std::vector<double>(n, 0.0); };
      sl.r1_e = alloc(nrzt1);
      sl.r1_o = alloc(nrzt1);
      sl.ru_e = alloc(nrzt1);
      sl.ru_o = alloc(nrzt1);
      sl.rv_e = alloc(nrzt1);
      sl.rv_o = alloc(nrzt1);
      sl.z1_e = alloc(nrzt1);
      sl.z1_o = alloc(nrzt1);
      sl.zu_e = alloc(nrzt1);
      sl.zu_o = alloc(nrzt1);
      sl.zv_e = alloc(nrzt1);
      sl.zv_o = alloc(nrzt1);
      sl.lu_e = alloc(nrzt1);
      sl.lu_o = alloc(nrzt1);
      sl.lv_e = alloc(nrzt1);
      sl.lv_o = alloc(nrzt1);
      sl.rCon = alloc(nrzt_con);
      sl.zCon = alloc(nrzt_con);
      sl.geom = RealSpaceGeometry{sl.r1_e, sl.r1_o, sl.ru_e, sl.ru_o, sl.rv_e,
                                  sl.rv_o, sl.z1_e, sl.z1_o, sl.zu_e, sl.zu_o,
                                  sl.zv_e, sl.zv_o, sl.lu_e, sl.lu_o, sl.lv_e,
                                  sl.lv_o, sl.rCon, sl.zCon};
    }
  }
};

// w7x at 4 threads: matches the real VMEC calling pattern exactly.
//
// VMEC keeps a single #pragma omp parallel team alive for the entire solver
// run.  Each thread owns its own IdealMhdModel (with its own RadialPartitioning
// slice) and calls dft_FourierToReal_3d_symm directly from within that
// persistent team -- there is no nested fork/join per iteration, only
// #pragma omp barrier between phases.
//
// We replicate that by opening one persistent team and looping inside it.
// Thread 0 drives the Google Benchmark state machine; the others mirror it
// via a shared atomic flag.  Each call is bracketed by barriers so all threads
// start and finish together, and the wall-clock time reflects the slowest.
void BM_FourierToReal_Parallel_W7x_4t(benchmark::State& state) {
  constexpr int kNumThreads = 6;
  static ParallelBenchFixture fx(/*nfp=*/5, /*mpol=*/12, /*ntor=*/12,
                                 kNumThreads);

  std::atomic<bool> keep_going{true};

#pragma omp parallel num_threads(kNumThreads)
  {
    const int tid = omp_get_thread_num();
    ParallelBenchFixture::ThreadSlice& sl = fx.threads[tid];

    // Warmup: each thread faults in its own plan's twiddle pages and sizes
    // the thread_local scratch buffer before timing starts.
    FourierToReal3DSymmFastPoloidalFft(*sl.phys_x, fx.xmpq, sl.rp, fx.s,
                                       *sl.rprof, fx.fb, fx.plans, sl.geom);
#pragma omp barrier

    if (tid == 0) {
      for (auto _ : state) {
#pragma omp barrier
        FourierToReal3DSymmFastPoloidalFft(*sl.phys_x, fx.xmpq, sl.rp, fx.s,
                                           *sl.rprof, fx.fb, fx.plans, sl.geom);
#pragma omp barrier
      }
      keep_going.store(false, std::memory_order_release);
      // Final barrier so workers see the flag and exit cleanly.
#pragma omp barrier
    } else {
      while (true) {
#pragma omp barrier
        if (!keep_going.load(std::memory_order_acquire)) break;
        FourierToReal3DSymmFastPoloidalFft(*sl.phys_x, fx.xmpq, sl.rp, fx.s,
                                           *sl.rprof, fx.fb, fx.plans, sl.geom);
#pragma omp barrier
      }
    }
  }

  state.SetLabel("12x12 6-thread parallel fft");
}
BENCHMARK(BM_FourierToReal_Parallel_W7x_4t);

// DFT parallel: same structure, no FFT plans needed.
void BM_DftFourierToReal_Parallel_W7x_4t(benchmark::State& state) {
  constexpr int kNumThreads = 6;
  static ParallelBenchFixture fx(/*nfp=*/5, /*mpol=*/12, /*ntor=*/12,
                                 kNumThreads);

  std::atomic<bool> keep_going{true};

#pragma omp parallel num_threads(kNumThreads)
  {
    const int tid = omp_get_thread_num();
    ParallelBenchFixture::ThreadSlice& sl = fx.threads[tid];

    // Warmup: fault in thread_local memory / cache lines before timing.
    FourierToReal3DSymmFastPoloidal(*sl.phys_x, fx.xmpq, sl.rp, fx.s, *sl.rprof,
                                    fx.fb, sl.geom);
#pragma omp barrier

    if (tid == 0) {
      for (auto _ : state) {
#pragma omp barrier
        FourierToReal3DSymmFastPoloidal(*sl.phys_x, fx.xmpq, sl.rp, fx.s,
                                        *sl.rprof, fx.fb, sl.geom);
#pragma omp barrier
      }
      keep_going.store(false, std::memory_order_release);
#pragma omp barrier
    } else {
      while (true) {
#pragma omp barrier
        if (!keep_going.load(std::memory_order_acquire)) break;
        FourierToReal3DSymmFastPoloidal(*sl.phys_x, fx.xmpq, sl.rp, fx.s,
                                        *sl.rprof, fx.fb, sl.geom);
#pragma omp barrier
      }
    }
  }

  state.SetLabel("12x12 6-thread parallel dft");
}
BENCHMARK(BM_DftFourierToReal_Parallel_W7x_4t);

}  // namespace
}  // namespace vmecpp

BENCHMARK_MAIN();

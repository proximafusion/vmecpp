// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Microbenchmark for the toroidal transform hot loop, across a spread of
// resolutions covering both the FFT and DFT dispatch paths.
//
// This calls IdealMhdModel::dft_FourierToReal_3d_symm() /
// dft_ForcesToFourier_3d_symm() -- the same dispatcher the real solver calls
// every iteration -- rather than the underlying FFT/DFT kernels directly.
// That dispatcher internally picks the FFTX path when a precompiled codelet
// exists for the resolution's (nZeta, 12*mpol) shape, and falls back to the
// plain DFT otherwise (see kernels_available() in fft_toroidal.h). Measuring
// through the dispatcher means a single named series here transparently
// reflects whichever path is actually active for that size, instead of
// requiring separate Dft/Fft series that must be read together.

#include <random>
#include <span>

#include "Eigen/Dense"
#include "benchmark/benchmark.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"
#include "vmecpp/vmec/thread_local_storage/thread_local_storage.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

namespace vmecpp {
namespace {

constexpr int kNs = 51;

//   label       nfp mpol ntor  nZeta  batch  FFTX codelet?
//   4x4           1   4   4      12     48    no  (small DFT baseline)
//   cma_5x6       5   5   6      16     60    no  (real cma config, DFT)
//   6x8           5   8   6      16     96    yes (real cma_6x8 config, FFT)
//   w7x_12x12     5  12  12      28    144    yes (flagship W7-X, FFT)
//   12x13         5  12  13      30    144    no  (large DFT; same batch as
//                                                  w7x_12x12 but no codelet)
struct ResParams {
  int nfp;
  int mpol;
  int ntor;
  const char* label;
};

constexpr ResParams kResolutions[] = {
    {1, 4, 4, "4x4"},         {5, 5, 6, "cma_5x6"}, {5, 8, 6, "6x8"},
    {5, 12, 12, "w7x_12x12"}, {5, 12, 13, "12x13"},
};

// ----------------------------------------------------------------------------
// Fixture: builds a fixed-boundary IdealMhdModel and the FourierGeometry
// input it operates on. FreeBoundaryBase* is null: dft_FourierToReal_3d_symm
// and dft_ForcesToFourier_3d_symm never touch it (only referenced by the
// free-boundary vacuum path in update()/computeBContra(), which this
// benchmark never calls), and the constructor only requires a non-null
// FreeBoundaryBase when FlowControl::lfreeb is true.
// ----------------------------------------------------------------------------

struct BenchFixture {
  Sizes s;
  RadialPartitioning rp;
  FourierBasisFastPoloidal fb;
  VmecConstants constants;
  ThreadLocalStorage ls;
  std::unique_ptr<VmecINDATA> indata;
  std::unique_ptr<HandoverStorage> handover;
  std::unique_ptr<FlowControl> fc;
  std::unique_ptr<RadialProfiles> rprof;
  VacuumPressureState vacuum_pressure_state = VacuumPressureState::kOff;
  std::unique_ptr<IdealMhdModel> model;

  std::unique_ptr<FourierGeometry> phys_x;
  std::unique_ptr<FourierForces> phys_f;

  explicit BenchFixture(int nfp, int mpol, int ntor)
      : s(/*lasym=*/false, nfp, mpol, ntor, /*ntheta=*/0, /*nzeta=*/0),
        fb(&s),
        ls(&s),
        indata(std::make_unique<VmecINDATA>()),
        handover(std::make_unique<HandoverStorage>(&s)),
        fc(std::make_unique<FlowControl>(/*lfreeb=*/false, /*delt=*/0.9,
                                         /*num_grids=*/1)) {
    rp.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0, kNs,
                                /*lfreeb=*/false, /*printout=*/false);
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

    model = std::make_unique<IdealMhdModel>(
        fc.get(), &s, &fb, rprof.get(), &constants, &ls, handover.get(), &rp,
        /*m_fb_vac=*/nullptr, /*vac_num_threads=*/0, /*signOfJacobian=*/-1,
        /*nvacskip=*/0, &vacuum_pressure_state);

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

    phys_f = std::make_unique<FourierForces>(&s, &rp, kNs);
  }
};

template <int kIdx>
void BM_ToroidalTransform_FourierToReal(benchmark::State& state) {
  static BenchFixture fx(kResolutions[kIdx].nfp, kResolutions[kIdx].mpol,
                         kResolutions[kIdx].ntor);
  for (auto _ : state) {
    fx.model->dft_FourierToReal_3d_symm(*fx.phys_x);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kResolutions[kIdx].label);
}

template <int kIdx>
void BM_ToroidalTransform_ForcesToFourier(benchmark::State& state) {
  static BenchFixture fx(kResolutions[kIdx].nfp, kResolutions[kIdx].mpol,
                         kResolutions[kIdx].ntor);
  // dft_ForcesToFourier_3d_symm reads IdealMhdModel's private real-space
  // force members (armn_e, blmn_e, ...), which this fixture never populates
  // -- they stay zero-initialized. That's fine for a timing benchmark: the
  // transform cost doesn't depend on the input values, only its size.
  for (auto _ : state) {
    fx.model->dft_ForcesToFourier_3d_symm(*fx.phys_f);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kResolutions[kIdx].label);
}

#define REGISTER_RES(IDX, LABEL)                                \
  BENCHMARK_TEMPLATE(BM_ToroidalTransform_FourierToReal, IDX)   \
      ->Name("ToroidalFourierToReal/" LABEL);                   \
  BENCHMARK_TEMPLATE(BM_ToroidalTransform_ForcesToFourier, IDX) \
      ->Name("ToroidalForcesToFourier/" LABEL)

REGISTER_RES(0, "4x4");
REGISTER_RES(1, "6x8");
REGISTER_RES(2, "12x12");
REGISTER_RES(3, "12x13");

#undef REGISTER_RES

}  // namespace
}  // namespace vmecpp

BENCHMARK_MAIN();

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Validates that FourierToReal3DSymmFastPoloidalFft and
// ForcesToFourier3DSymmFastPoloidalFft produce results numerically identical
// to their DFT counterparts (FourierToReal3DSymmFastPoloidal and
// ForcesToFourier3DSymmFastPoloidal) for randomly-generated spectral data.

#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal.h"

#include <cmath>
#include <random>
#include <span>
#include <vector>

#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {
namespace {

// Absolute tolerance for comparing DFT vs FFT results.
// The two algorithms use different floating-point operation orderings, so small
// but non-zero differences are expected.
constexpr double kAbsTol = 1e-10;

// Create a RadialProfiles with sqrtSF populated for the given partitioning.
// Uses a default VmecINDATA and FlowControl; only sqrtSF is relied upon by
// the transform functions under test.
RadialProfiles MakeProfiles(const Sizes& s, const RadialPartitioning& rp,
                            int ns) {
  VmecINDATA indata;
  FlowControl fc(/*lfreeb=*/false, /*delt=*/0.9, /*num_grids=*/1);
  fc.ns = ns;
  HandoverStorage h(&s);
  RadialProfiles prof(&rp, &h, &indata, &fc, /*signOfJacobian=*/-1,
                      /*pDamp=*/0.05);
  const int n = rp.nsMaxF1 - rp.nsMinF1;
  prof.sqrtSF.resize(n);
  for (int j = 0; j < n; ++j) {
    prof.sqrtSF[j] = std::sqrt(0.05 + 0.9 * j / (n > 1 ? n - 1 : 1));
  }
  return prof;
}

// ============================================================================
// FourierToReal tests
// ============================================================================

struct FftTestParams {
  int nfp, mpol, ntor, nzeta, ns;
};

class FourierToRealFftTest : public ::testing::TestWithParam<FftTestParams> {};

TEST_P(FourierToRealFftTest, MatchesDft) {
  const auto& p = GetParam();
  const Sizes s(/*lasym=*/false, p.nfp, p.mpol, p.ntor,
                /*ntheta=*/0, p.nzeta);

  RadialPartitioning rp;
  rp.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0, p.ns,
                              /*lfreeb=*/false, /*printout=*/false);

  FourierBasisFastPoloidal fb(&s);
  ToroidalFftPlans plans(s.nZeta, s.nfp, s.mpol);

  // FourierGeometry with random spectral data.
  auto phys_x = std::make_unique<FourierGeometry>(&s, &rp, p.ns);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  auto rand_fill = [&](std::span<double> sp) {
    for (double& x : sp) x = dist(rng);
  };
  rand_fill(phys_x->rmncc);
  rand_fill(phys_x->rmnss);
  rand_fill(phys_x->zmnsc);
  rand_fill(phys_x->zmncs);
  rand_fill(phys_x->lmnsc);
  rand_fill(phys_x->lmncs);

  RadialProfiles rprof = MakeProfiles(s, rp, p.ns);

  Eigen::VectorXd xmpq(s.mpol);
  for (int m = 0; m < s.mpol; ++m) {
    xmpq[m] = m * (m - 1);
  }

  // Backing storage for both output geometries.
  const int nrzt1 = s.nZnT * (rp.nsMaxF1 - rp.nsMinF1);
  const int nrzt_con = s.nZnT * (rp.nsMaxFIncludingLcfs - rp.nsMinF);

  auto alloc = [](int n) { return std::vector<double>(n, 0.0); };

  // DFT output arrays.
  auto r1_e_d = alloc(nrzt1), r1_o_d = alloc(nrzt1);
  auto ru_e_d = alloc(nrzt1), ru_o_d = alloc(nrzt1);
  auto rv_e_d = alloc(nrzt1), rv_o_d = alloc(nrzt1);
  auto z1_e_d = alloc(nrzt1), z1_o_d = alloc(nrzt1);
  auto zu_e_d = alloc(nrzt1), zu_o_d = alloc(nrzt1);
  auto zv_e_d = alloc(nrzt1), zv_o_d = alloc(nrzt1);
  auto lu_e_d = alloc(nrzt1), lu_o_d = alloc(nrzt1);
  auto lv_e_d = alloc(nrzt1), lv_o_d = alloc(nrzt1);
  auto rCon_d = alloc(nrzt_con), zCon_d = alloc(nrzt_con);

  RealSpaceGeometry geom_dft{r1_e_d, r1_o_d, ru_e_d, ru_o_d, rv_e_d, rv_o_d,
                             z1_e_d, z1_o_d, zu_e_d, zu_o_d, zv_e_d, zv_o_d,
                             lu_e_d, lu_o_d, lv_e_d, lv_o_d, rCon_d, zCon_d};

  // FFT output arrays.
  auto r1_e_f = alloc(nrzt1), r1_o_f = alloc(nrzt1);
  auto ru_e_f = alloc(nrzt1), ru_o_f = alloc(nrzt1);
  auto rv_e_f = alloc(nrzt1), rv_o_f = alloc(nrzt1);
  auto z1_e_f = alloc(nrzt1), z1_o_f = alloc(nrzt1);
  auto zu_e_f = alloc(nrzt1), zu_o_f = alloc(nrzt1);
  auto zv_e_f = alloc(nrzt1), zv_o_f = alloc(nrzt1);
  auto lu_e_f = alloc(nrzt1), lu_o_f = alloc(nrzt1);
  auto lv_e_f = alloc(nrzt1), lv_o_f = alloc(nrzt1);
  auto rCon_f = alloc(nrzt_con), zCon_f = alloc(nrzt_con);

  RealSpaceGeometry geom_fft{r1_e_f, r1_o_f, ru_e_f, ru_o_f, rv_e_f, rv_o_f,
                             z1_e_f, z1_o_f, zu_e_f, zu_o_f, zv_e_f, zv_o_f,
                             lu_e_f, lu_o_f, lv_e_f, lv_o_f, rCon_f, zCon_f};

  // Run both transforms.
  FourierToReal3DSymmFastPoloidal(*phys_x, xmpq, rp, s, rprof, fb, geom_dft);
  FourierToReal3DSymmFastPoloidalFft(*phys_x, xmpq, rp, s, rprof, fb, plans,
                                     geom_fft);

  // Compare every output array element-wise.
  auto check = [&](const std::vector<double>& a, const std::vector<double>& b,
                   const char* name) {
    ASSERT_EQ(a.size(), b.size()) << "Size mismatch in " << name;
    for (size_t i = 0; i < a.size(); ++i) {
      EXPECT_NEAR(a[i], b[i], kAbsTol)
          << name << "[" << i << "]: DFT=" << a[i] << " FFT=" << b[i];
    }
  };

  check(r1_e_d, r1_e_f, "r1_e");
  check(r1_o_d, r1_o_f, "r1_o");
  check(ru_e_d, ru_e_f, "ru_e");
  check(ru_o_d, ru_o_f, "ru_o");
  check(rv_e_d, rv_e_f, "rv_e");
  check(rv_o_d, rv_o_f, "rv_o");
  check(z1_e_d, z1_e_f, "z1_e");
  check(z1_o_d, z1_o_f, "z1_o");
  check(zu_e_d, zu_e_f, "zu_e");
  check(zu_o_d, zu_o_f, "zu_o");
  check(zv_e_d, zv_e_f, "zv_e");
  check(zv_o_d, zv_o_f, "zv_o");
  check(lu_e_d, lu_e_f, "lu_e");
  check(lu_o_d, lu_o_f, "lu_o");
  check(lv_e_d, lv_e_f, "lv_e");
  check(lv_o_d, lv_o_f, "lv_o");
  check(rCon_d, rCon_f, "rCon");
  check(zCon_d, zCon_f, "zCon");
}

INSTANTIATE_TEST_SUITE_P(
    PhysicsParams, FourierToRealFftTest,
    ::testing::Values(FftTestParams{1, 4, 2, 12, 6},    // small tokamak-like
                      FftTestParams{5, 6, 3, 18, 10},   // medium stellarator
                      FftTestParams{5, 12, 6, 36, 20},  // realistic W7-X-like
                      FftTestParams{1, 2, 1, 8, 4}));   // minimal

// ============================================================================
// ForcesToFourier tests
// ============================================================================

class ForcesToFourierFftTest : public ::testing::TestWithParam<FftTestParams> {
};

TEST_P(ForcesToFourierFftTest, MatchesDft) {
  const auto& p = GetParam();
  const Sizes s(/*lasym=*/false, p.nfp, p.mpol, p.ntor,
                /*ntheta=*/0, p.nzeta);

  RadialPartitioning rp;
  rp.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0, p.ns,
                              /*lfreeb=*/false, /*printout=*/false);

  FourierBasisFastPoloidal fb(&s);
  ToroidalFftPlans plans(s.nZeta, s.nfp, s.mpol);

  // Real-space forces with random data.
  const int nrzt = s.nZnT * (rp.nsMaxF - rp.nsMinF);
  const int nrzt_lcfs = s.nZnT * (rp.nsMaxFIncludingLcfs - rp.nsMinF);

  std::mt19937 rng(137);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  auto rand_vec = [&](int n) {
    std::vector<double> v(n);
    for (double& x : v) x = dist(rng);
    return v;
  };

  auto armn_e = rand_vec(nrzt), armn_o = rand_vec(nrzt);
  auto azmn_e = rand_vec(nrzt), azmn_o = rand_vec(nrzt);
  auto blmn_e = rand_vec(nrzt_lcfs), blmn_o = rand_vec(nrzt_lcfs);
  auto brmn_e = rand_vec(nrzt), brmn_o = rand_vec(nrzt);
  auto bzmn_e = rand_vec(nrzt), bzmn_o = rand_vec(nrzt);
  auto clmn_e = rand_vec(nrzt_lcfs), clmn_o = rand_vec(nrzt_lcfs);
  auto crmn_e = rand_vec(nrzt), crmn_o = rand_vec(nrzt);
  auto czmn_e = rand_vec(nrzt), czmn_o = rand_vec(nrzt);
  auto frcon_e = rand_vec(nrzt), frcon_o = rand_vec(nrzt);
  auto fzcon_e = rand_vec(nrzt), fzcon_o = rand_vec(nrzt);

  const RealSpaceForces forces{armn_e, armn_o,  azmn_e,  azmn_o,  blmn_e,
                               blmn_o, brmn_e,  brmn_o,  bzmn_e,  bzmn_o,
                               clmn_e, clmn_o,  crmn_e,  crmn_o,  czmn_e,
                               czmn_o, frcon_e, frcon_o, fzcon_e, fzcon_o};

  Eigen::VectorXd xmpq(s.mpol);
  for (int m = 0; m < s.mpol; ++m) {
    xmpq[m] = m * (m - 1);
  }

  FlowControl fc(/*lfreeb=*/false, /*delt=*/0.9, /*num_grids=*/1);
  fc.ns = p.ns;

  auto ff_dft = std::make_unique<FourierForces>(&s, &rp, p.ns);
  auto ff_fft = std::make_unique<FourierForces>(&s, &rp, p.ns);

  ForcesToFourier3DSymmFastPoloidal(forces, xmpq, rp, fc, s, fb,
                                    VacuumPressureState::kOff, *ff_dft);
  ForcesToFourier3DSymmFastPoloidalFft(forces, xmpq, rp, fc, s, fb, plans,
                                       VacuumPressureState::kOff, *ff_fft);

  auto check = [&](std::span<const double> a, std::span<const double> b,
                   const char* name) {
    ASSERT_EQ(a.size(), b.size()) << "Size mismatch in " << name;
    for (size_t i = 0; i < a.size(); ++i) {
      EXPECT_NEAR(a[i], b[i], kAbsTol)
          << name << "[" << i << "]: DFT=" << a[i] << " FFT=" << b[i];
    }
  };

  check(ff_dft->frcc, ff_fft->frcc, "frcc");
  check(ff_dft->frss, ff_fft->frss, "frss");
  check(ff_dft->fzsc, ff_fft->fzsc, "fzsc");
  check(ff_dft->fzcs, ff_fft->fzcs, "fzcs");
  check(ff_dft->flsc, ff_fft->flsc, "flsc");
  check(ff_dft->flcs, ff_fft->flcs, "flcs");
}

INSTANTIATE_TEST_SUITE_P(PhysicsParams, ForcesToFourierFftTest,
                         ::testing::Values(FftTestParams{1, 4, 2, 12, 6},
                                           FftTestParams{5, 6, 3, 18, 10},
                                           FftTestParams{5, 12, 6, 36, 20},
                                           FftTestParams{1, 2, 1, 8, 4}));

}  // namespace
}  // namespace vmecpp

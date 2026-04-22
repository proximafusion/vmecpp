// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include <cmath>
#include <numbers>
#include <vector>

#include "gtest/gtest.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/ideal_mhd_model/ideal_mhd_model.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {
namespace {

// Helper to produce a single-surface real-space force field whose armn (the
// "bare" R-force component) is seeded with a single cos(m_seed * theta) *
// cos(n_seed * zeta) signal. All other real-space force components are zero.
// Since the armn term enters the forward DFT as armn.dot(cosmui) * cosnv,
// this produces exactly one (m, n) Fourier coefficient in the frcc channel.
struct ForceArrays {
  std::vector<double> armn_e, armn_o;
  std::vector<double> azmn_e, azmn_o;
  std::vector<double> blmn_e, blmn_o;
  std::vector<double> brmn_e, brmn_o;
  std::vector<double> bzmn_e, bzmn_o;
  std::vector<double> clmn_e, clmn_o;
  std::vector<double> crmn_e, crmn_o;
  std::vector<double> czmn_e, czmn_o;
  std::vector<double> frcon_e, frcon_o;
  std::vector<double> fzcon_e, fzcon_o;

  void allocate(int n) {
    armn_e.assign(n, 0.0);
    armn_o.assign(n, 0.0);
    azmn_e.assign(n, 0.0);
    azmn_o.assign(n, 0.0);
    blmn_e.assign(n, 0.0);
    blmn_o.assign(n, 0.0);
    brmn_e.assign(n, 0.0);
    brmn_o.assign(n, 0.0);
    bzmn_e.assign(n, 0.0);
    bzmn_o.assign(n, 0.0);
    clmn_e.assign(n, 0.0);
    clmn_o.assign(n, 0.0);
    crmn_e.assign(n, 0.0);
    crmn_o.assign(n, 0.0);
    czmn_e.assign(n, 0.0);
    czmn_o.assign(n, 0.0);
    frcon_e.assign(n, 0.0);
    frcon_o.assign(n, 0.0);
    fzcon_e.assign(n, 0.0);
    fzcon_o.assign(n, 0.0);
  }

  RealSpaceForces view() const {
    return RealSpaceForces{
        .armn_e = armn_e,
        .armn_o = armn_o,
        .azmn_e = azmn_e,
        .azmn_o = azmn_o,
        .blmn_e = blmn_e,
        .blmn_o = blmn_o,
        .brmn_e = brmn_e,
        .brmn_o = brmn_o,
        .bzmn_e = bzmn_e,
        .bzmn_o = bzmn_o,
        .clmn_e = clmn_e,
        .clmn_o = clmn_o,
        .crmn_e = crmn_e,
        .crmn_o = crmn_o,
        .czmn_e = czmn_e,
        .czmn_o = czmn_o,
        .frcon_e = frcon_e,
        .frcon_o = frcon_o,
        .fzcon_e = fzcon_e,
        .fzcon_o = fzcon_o,
    };
  }
};

// Compute the diagnostic for a Sizes(mpol, ntor, ntheta, nzeta) problem with
// a single radial surface, a single seeded mode in armn_e, and return the
// discard fraction for R. The test assumes lthreed, symmetric (lasym=false).
double ComputeRDiscardForSingleMode(int mpol, int ntor, int ntheta, int nzeta,
                                    int m_seed, int n_seed) {
  const int nfp = 1;
  Sizes sizes(/*lasym=*/false, nfp, mpol, ntor, ntheta, nzeta);
  FourierBasisFastPoloidal fb(&sizes);

  // Single radial surface partition: nsMinF=1, nsMaxF=2 (one interior
  // surface). Use ns=3 so that jMaxRZ = min(nsMaxF, ns-1) = 2.
  const int ns = 3;
  RadialPartitioning rp;
  rp.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0, ns,
                              /*lfreeb=*/false, /*printout=*/false);
  // Sanity: with ns=3 and 1 thread, the partition covers all interior
  // surfaces. We only care about the slice containing jF=1.

  FlowControl fc(/*lfreeb=*/false, /*delt=*/1.0, /*num_grids=*/1);
  fc.ns = ns;

  const int num_local = rp.nsMaxF - rp.nsMinF;
  const int per_surf = nzeta * sizes.nThetaEff;

  ForceArrays arrays;
  arrays.allocate(num_local * per_surf);

  // Seed a single cosine mode on the first local surface. Pick an offset
  // so that local_j=0 corresponds to jF=nsMinF; by construction nsMinF>=0.
  // For ns=3, typical partition gives nsMinF=0, nsMaxF=3. We seed on
  // local_j = max(1, nsMinF) - nsMinF = 1 to stay off-axis.
  const int local_j_seed = (rp.nsMinF == 0) ? 1 : 0;
  // VMEC separates real-space forces into even-m and odd-m parity arrays
  // (armn_e for m even, armn_o for m odd). Seed into the parity matching
  // m_seed so the diagnostic's parity-aware inner loop picks it up.
  const bool m_even = (m_seed % 2 == 0);
  std::vector<double>& target = m_even ? arrays.armn_e : arrays.armn_o;
  const int nThetaEff = sizes.nThetaEff;
  const double two_pi = 2.0 * std::numbers::pi;
  for (int k = 0; k < sizes.nZeta; ++k) {
    const double zeta = two_pi * k / sizes.nZeta;
    for (int l = 0; l < sizes.nThetaReduced; ++l) {
      const double theta = two_pi * l / sizes.nThetaEven;
      const int idx = (local_j_seed * sizes.nZeta + k) * nThetaEff + l;
      target[idx] = std::cos(m_seed * theta) * std::cos(n_seed * zeta);
    }
  }

  const auto report = ComputeForceSpectralTruncation(
      arrays.view(), rp, fc, sizes, fb, VacuumPressureState::kOff);

  return report.r_discarded_fraction[local_j_seed];
}

TEST(SpectralTruncationTest, ModeInsideBandHasZeroDiscard) {
  // mpol=6 means retained poloidal modes are m in [0, 5]; ntor=4 means
  // retained toroidal modes are n in [0, 4]. Seed a mode at (m=3, n=2),
  // well inside the band. Expect ~0 discard fraction.
  const double discard = ComputeRDiscardForSingleMode(
      /*mpol=*/6, /*ntor=*/4, /*ntheta=*/20, /*nzeta=*/20,
      /*m_seed=*/3, /*n_seed=*/2);
  EXPECT_NEAR(discard, 0.0, 1e-10);
}

TEST(SpectralTruncationTest, ModeOutsidePoloidalBandIsFullyDiscarded) {
  // Retained band: m < 4. Seed at m=6 (outside), n=0.
  // mnyq is derived from nThetaEven; with nTheta=20, nThetaEven=20,
  // mnyq0 = 10 so mnyq >= 6 is representable on the Nyquist grid.
  const double discard = ComputeRDiscardForSingleMode(
      /*mpol=*/4, /*ntor=*/3, /*ntheta=*/20, /*nzeta=*/20,
      /*m_seed=*/6, /*n_seed=*/0);
  EXPECT_NEAR(discard, 1.0, 1e-10);
}

TEST(SpectralTruncationTest, ModeOutsideToroidalBandIsFullyDiscarded) {
  // Retained band: n <= 2. Seed at m=1, n=5 (outside). Expect ~1.
  const double discard = ComputeRDiscardForSingleMode(
      /*mpol=*/5, /*ntor=*/2, /*ntheta=*/20, /*nzeta=*/20,
      /*m_seed=*/1, /*n_seed=*/5);
  EXPECT_NEAR(discard, 1.0, 1e-10);
}

}  // namespace
}  // namespace vmecpp

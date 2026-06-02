// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Validates the sparse-toroidal handling of the 3D stellarator-symmetric DFT
// kernels: a sparse run that carries only the active toroidal modes must
// produce results numerically identical to a dense run at the same ntor in
// which all inactive toroidal coefficients are exactly zero.

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
#include "vmecpp/vmec/ideal_mhd_model/dft_toroidal.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {
namespace {

// The sparse and dense paths do exactly the same arithmetic on the same active
// columns (only the inactive columns differ, and those are zero in the dense
// case), so the results should agree to round-off.
constexpr double kAbsTol = 1e-12;

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

struct SparseTestParams {
  int nfp, mpol, ntor, nzeta, ns;
  std::vector<int> active_n;  // must include 0, sorted, in [0, ntor]
};

// Copy the active toroidal columns of a dense per-(jF,m) FC array into the
// compact sparse layout.
void GatherToSparse(std::span<const double> dense, std::span<double> sparse,
                    const Sizes& s_dense, const Sizes& s_sparse, int n_jF) {
  for (int jF = 0; jF < n_jF; ++jF) {
    for (int m = 0; m < s_dense.mpol; ++m) {
      for (int c = 0; c < s_sparse.n_active; ++c) {
        const int n = s_sparse.active_n[c];
        const int dense_idx = (jF * s_dense.mpol + m) * (s_dense.ntor + 1) + n;
        const int sparse_idx = (jF * s_sparse.mpol + m) * s_sparse.n_active + c;
        sparse[sparse_idx] = dense[dense_idx];
      }
    }
  }
}

class SparseToroidalForwardTest
    : public ::testing::TestWithParam<SparseTestParams> {};

TEST_P(SparseToroidalForwardTest, MatchesDenseWithZeros) {
  const auto& p = GetParam();
  const Sizes s_dense(/*lasym=*/false, p.nfp, p.mpol, p.ntor,
                      /*ntheta=*/0, p.nzeta);
  const Sizes s_sparse(/*lasym=*/false, p.nfp, p.mpol, p.ntor,
                       /*ntheta=*/0, p.nzeta, p.active_n);

  ASSERT_TRUE(s_sparse.is_sparse_toroidal)
      << "test must use a genuinely sparse active set";

  RadialPartitioning rp;
  rp.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0, p.ns,
                              /*lfreeb=*/false, /*printout=*/false);

  FourierBasisFastPoloidal fb_dense(&s_dense);
  FourierBasisFastPoloidal fb_sparse(&s_sparse);

  auto phys_dense = std::make_unique<FourierGeometry>(&s_dense, &rp, p.ns);
  auto phys_sparse = std::make_unique<FourierGeometry>(&s_sparse, &rp, p.ns);

  // Fill the dense geometry with random data, but zero out all inactive
  // toroidal columns so that it represents exactly the same physical geometry
  // as the sparse one.
  std::mt19937 rng(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<bool> active(p.ntor + 1, false);
  for (int n : p.active_n) active[n] = true;

  const int n_jF = rp.nsMaxF1 - rp.nsMinF1;
  auto fill_dense = [&](std::span<double> arr) {
    for (int jF = 0; jF < n_jF; ++jF) {
      for (int m = 0; m < s_dense.mpol; ++m) {
        for (int n = 0; n <= s_dense.ntor; ++n) {
          const int idx = (jF * s_dense.mpol + m) * (s_dense.ntor + 1) + n;
          arr[idx] = active[n] ? dist(rng) : 0.0;
        }
      }
    }
  };
  fill_dense(phys_dense->rmncc);
  fill_dense(phys_dense->rmnss);
  fill_dense(phys_dense->zmnsc);
  fill_dense(phys_dense->zmncs);
  fill_dense(phys_dense->lmnsc);
  fill_dense(phys_dense->lmncs);

  // Gather the same values into the sparse compact layout.
  GatherToSparse(phys_dense->rmncc, phys_sparse->rmncc, s_dense, s_sparse,
                 n_jF);
  GatherToSparse(phys_dense->rmnss, phys_sparse->rmnss, s_dense, s_sparse,
                 n_jF);
  GatherToSparse(phys_dense->zmnsc, phys_sparse->zmnsc, s_dense, s_sparse,
                 n_jF);
  GatherToSparse(phys_dense->zmncs, phys_sparse->zmncs, s_dense, s_sparse,
                 n_jF);
  GatherToSparse(phys_dense->lmnsc, phys_sparse->lmnsc, s_dense, s_sparse,
                 n_jF);
  GatherToSparse(phys_dense->lmncs, phys_sparse->lmncs, s_dense, s_sparse,
                 n_jF);

  RadialProfiles prof_dense = MakeProfiles(s_dense, rp, p.ns);
  RadialProfiles prof_sparse = MakeProfiles(s_sparse, rp, p.ns);

  Eigen::VectorXd xmpq(p.mpol);
  for (int m = 0; m < p.mpol; ++m) xmpq[m] = m * (m - 1);

  const int nrzt1 = s_dense.nZnT * (rp.nsMaxF1 - rp.nsMinF1);
  const int nrzt_con = s_dense.nZnT * (rp.nsMaxFIncludingLcfs - rp.nsMinF);
  auto alloc = [](int n) { return std::vector<double>(n, 0.0); };

  auto run = [&](const FourierGeometry& phys, const Sizes& s,
                 const RadialProfiles& prof,
                 const FourierBasisFastPoloidal& fb) {
    auto r1_e = alloc(nrzt1), r1_o = alloc(nrzt1);
    auto ru_e = alloc(nrzt1), ru_o = alloc(nrzt1);
    auto rv_e = alloc(nrzt1), rv_o = alloc(nrzt1);
    auto z1_e = alloc(nrzt1), z1_o = alloc(nrzt1);
    auto zu_e = alloc(nrzt1), zu_o = alloc(nrzt1);
    auto zv_e = alloc(nrzt1), zv_o = alloc(nrzt1);
    auto lu_e = alloc(nrzt1), lu_o = alloc(nrzt1);
    auto lv_e = alloc(nrzt1), lv_o = alloc(nrzt1);
    auto rCon = alloc(nrzt_con), zCon = alloc(nrzt_con);
    // Keep storage alive by returning it together with the computed arrays.
    std::vector<std::vector<double>> out = {r1_e, r1_o, ru_e, ru_o, rv_e, rv_o,
                                            z1_e, z1_o, zu_e, zu_o, zv_e, zv_o,
                                            lu_e, lu_o, lv_e, lv_o, rCon, zCon};
    RealSpaceGeometry geom{out[0],  out[1],  out[2],  out[3],  out[4],
                           out[5],  out[6],  out[7],  out[8],  out[9],
                           out[10], out[11], out[12], out[13], out[14],
                           out[15], out[16], out[17]};
    FourierToReal3DSymmFastPoloidal(phys, xmpq, rp, s, prof, fb, geom);
    return out;
  };

  auto out_dense = run(*phys_dense, s_dense, prof_dense, fb_dense);
  auto out_sparse = run(*phys_sparse, s_sparse, prof_sparse, fb_sparse);

  const char* names[] = {"r1_e", "r1_o", "ru_e", "ru_o", "rv_e", "rv_o",
                         "z1_e", "z1_o", "zu_e", "zu_o", "zv_e", "zv_o",
                         "lu_e", "lu_o", "lv_e", "lv_o", "rCon", "zCon"};
  for (size_t a = 0; a < out_dense.size(); ++a) {
    ASSERT_EQ(out_dense[a].size(), out_sparse[a].size());
    for (size_t i = 0; i < out_dense[a].size(); ++i) {
      EXPECT_NEAR(out_dense[a][i], out_sparse[a][i], kAbsTol)
          << names[a] << "[" << i << "]";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    PhysicsParams, SparseToroidalForwardTest,
    ::testing::Values(
        // base on multiples of nfp implied via ntor; pick gaps in the n grid
        SparseTestParams{1, 6, 12, 28, 6, {0, 3, 10}},
        SparseTestParams{5, 8, 10, 24, 10, {0, 1, 7}},
        SparseTestParams{3, 10, 16, 36, 12, {0, 2, 5, 15}}));

}  // namespace
}  // namespace vmecpp

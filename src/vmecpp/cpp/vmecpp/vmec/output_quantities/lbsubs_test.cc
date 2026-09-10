// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// Tests for the lbsubs path of ComputeOutputQuantities: the radial-force-
// balance recompute of the full-grid covariant B_s that mirrors Fortran
// VMEC's jxbforce + getbsubs when lbsubs == true.
//
// lasym = false: run solovev with lbsubs both off and on. Three assertions
// cover (1) the half->full interpolation invariant when off, (2) a measurable
// change in bsubs_full when on, (3) the radial-force-balance residual on
// every interior surface when on.
//
// lasym = true: parse up_down_asym.json and construct the Vmec instance,
// then skip. RecomputeBSubSFromRadialForceBalance does not implement the
// non-stellarator-symmetric branch, and the equilibrium iteration does not
// handle lasym = true.
//
// JSON wiring: a one-line round-trip on lbsubs = true. Default-value parsing
// is covered in vmec_indata_test.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

namespace {

using file_io::ReadFile;

// Recompute the radial-force-balance RHS that
// RecomputeBSubSFromRadialForceBalance solves at the given interior surface
// jF. Returns brho_corrected (with the flux-surface-average subtraction
// applied) plus the surface-averaged sqrt(g) * B^{u,v} that the residual
// check needs. The formula is duplicated here on purpose so the test does
// not depend on internal scratch buffers.
struct ForceBalanceRhs {
  std::vector<double> bsupu_full;  // sqrt(g) * B^u averaged
  std::vector<double> bsupv_full;  // sqrt(g) * B^v averaged
  std::vector<double> brho;        // RHS after flux-surface-average subtraction
};

ForceBalanceRhs RebuildForceBalanceRhs(
    const Sizes& s, const FlowControl& fc,
    const VmecInternalResults& vmec_internal_results, int jF) {
  const int jHi = jF - 1;
  const int jHo = jF;
  const double ohs = 1.0 / fc.deltaS;
  const double sign_jacobian =
      static_cast<double>(vmec_internal_results.sign_of_jacobian);

  ForceBalanceRhs r;
  r.bsupu_full.assign(s.nZnT, 0.0);
  r.bsupv_full.assign(s.nZnT, 0.0);
  r.brho.assign(s.nZnT, 0.0);

  std::vector<double> sqrtg_full(s.nZnT, 0.0);

  for (int kl = 0; kl < s.nZnT; ++kl) {
    const int idx_o = jHo * s.nZnT + kl;
    const int idx_i = jHi * s.nZnT + kl;
    const double g_o = vmec_internal_results.gsqrt(idx_o);
    const double g_i = vmec_internal_results.gsqrt(idx_i);
    sqrtg_full[kl] = 0.5 * (g_o + g_i);
    r.bsupu_full[kl] = 0.5 * (vmec_internal_results.bsupu(idx_o) * g_o +
                              vmec_internal_results.bsupu(idx_i) * g_i);
    r.bsupv_full[kl] = 0.5 * (vmec_internal_results.bsupv(idx_o) * g_o +
                              vmec_internal_results.bsupv(idx_i) * g_i);
    const double dbsubu =
        vmec_internal_results.bsubu(idx_o) - vmec_internal_results.bsubu(idx_i);
    const double dbsubv =
        vmec_internal_results.bsubv(idx_o) - vmec_internal_results.bsubv(idx_i);
    r.brho[kl] =
        ohs * (r.bsupu_full[kl] * dbsubu + r.bsupv_full[kl] * dbsubv) +
        (vmec_internal_results.presH[jHo] - vmec_internal_results.presH[jHi]) *
            ohs * sqrtg_full[kl];
  }

  double brho00 = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    const int l = kl % s.nThetaEff;
    brho00 += r.brho[kl] * s.wInt[l];
  }
  const double vp_full = 0.5 * (vmec_internal_results.dVdsH[jHo] +
                                vmec_internal_results.dVdsH[jHi]);
  if (vp_full != 0.0) {
    for (int kl = 0; kl < s.nZnT; ++kl) {
      r.brho[kl] -= sign_jacobian * sqrtg_full[kl] * brho00 / vp_full;
    }
  }
  return r;
}

// Worst-case absolute residual of the radial force-balance equation on a
// single interior surface jF, using the recovered B_s derivatives.
double InteriorForceBalanceResidualMax(
    const Sizes& s, const FlowControl& fc,
    const VmecInternalResults& vmec_internal_results,
    const CovariantBDerivatives& covariant_b_derivatives, int jF) {
  const ForceBalanceRhs rhs =
      RebuildForceBalanceRhs(s, fc, vmec_internal_results, jF);
  double residual_max = 0.0;
  for (int kl = 0; kl < s.nZnT; ++kl) {
    const int idx = jF * s.nZnT + kl;
    const double residual =
        rhs.bsupu_full[kl] * covariant_b_derivatives.bsubsu(idx) +
        rhs.bsupv_full[kl] * covariant_b_derivatives.bsubsv(idx) - rhs.brho[kl];
    residual_max = std::max(residual_max, std::abs(residual));
  }
  return residual_max;
}

}  // namespace

// -------------------------------------------------------------------------
// lasym = false: solovev fixture, three assertions.
// -------------------------------------------------------------------------

class LbsubsSolovevTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const absl::StatusOr<std::string> indata_json =
        ReadFile("vmecpp/test_data/solovev.json");
    ASSERT_TRUE(indata_json.ok());
    const absl::StatusOr<VmecINDATA> parsed =
        VmecINDATA::FromJson(*indata_json);
    ASSERT_TRUE(parsed.ok());
    indata_ = *parsed;
    // sanity: the case is stellarator-symmetric, axisymmetric tokamak.
    ASSERT_FALSE(indata_.lasym);
    ASSERT_EQ(indata_.ntor, 0);
  }

  VmecINDATA indata_;
};

// lbsubs = false (default) must leave bsubs_full as the linear half->full
// interpolation of bsubs_half on every interior surface, since the new code
// is gated entirely on that flag.
TEST_F(LbsubsSolovevTest, FalseKeepsHalfToFullInterpolation) {
  indata_.lbsubs = false;
  auto maybe_vmec = Vmec::FromIndata(indata_);
  ASSERT_TRUE(maybe_vmec.ok());
  Vmec& vmec = **maybe_vmec;
  const absl::StatusOr<bool> reached_checkpoint = vmec.run();
  ASSERT_TRUE(reached_checkpoint.ok());
  ASSERT_FALSE(*reached_checkpoint) << "solovev did not converge";

  const auto& oq = vmec.output_quantities_;
  const auto& bsubs_half = oq.bsubs_half.bsubs_half;
  const auto& bsubs_full = oq.bsubs_full.bsubs_full;
  ASSERT_EQ(bsubs_full.rows(), oq.vmec_internal_results.num_full);
  ASSERT_EQ(bsubs_half.rows(), oq.vmec_internal_results.num_half);

  const int ns = oq.vmec_internal_results.num_full;
  const int nznt = bsubs_full.cols();
  double worst = 0.0;
  for (int jF = 1; jF < ns - 1; ++jF) {
    for (int kl = 0; kl < nznt; ++kl) {
      const double expected =
          0.5 * (bsubs_half(jF, kl) + bsubs_half(jF - 1, kl));
      worst = std::max(worst, std::abs(bsubs_full(jF, kl) - expected));
    }
  }
  EXPECT_LT(worst, 1.0e-12)
      << "lbsubs = false unexpectedly changed bsubs_full away from the "
         "half->full interpolation (worst absolute diff "
      << worst << ")";
}

// lbsubs = true must produce a measurably different bsubs_full on at least
// one interior point; a silent no-op would leave the two paths bit-identical
// and fail here.
TEST_F(LbsubsSolovevTest, TrueChangesBsubsFullVsFalseBaseline) {
  indata_.lbsubs = false;
  auto maybe_baseline = Vmec::FromIndata(indata_);
  ASSERT_TRUE(maybe_baseline.ok());
  Vmec& baseline_vmec = **maybe_baseline;
  const absl::StatusOr<bool> baseline_checkpoint = baseline_vmec.run();
  ASSERT_TRUE(baseline_checkpoint.ok());
  ASSERT_FALSE(*baseline_checkpoint) << "baseline solovev did not converge";
  const auto& baseline_bsubs_full =
      baseline_vmec.output_quantities_.bsubs_full.bsubs_full;

  indata_.lbsubs = true;
  auto maybe_corrected = Vmec::FromIndata(indata_);
  ASSERT_TRUE(maybe_corrected.ok());
  Vmec& corrected_vmec = **maybe_corrected;
  const absl::StatusOr<bool> corrected_checkpoint = corrected_vmec.run();
  ASSERT_TRUE(corrected_checkpoint.ok());
  ASSERT_FALSE(*corrected_checkpoint) << "corrected solovev did not converge";
  const auto& corrected_bsubs_full =
      corrected_vmec.output_quantities_.bsubs_full.bsubs_full;

  ASSERT_EQ(baseline_bsubs_full.rows(), corrected_bsubs_full.rows());
  ASSERT_EQ(baseline_bsubs_full.cols(), corrected_bsubs_full.cols());
  const double diff =
      (baseline_bsubs_full - corrected_bsubs_full).cwiseAbs().maxCoeff();
  // bsubs_half is O(0.1) for solovev; 1e-10 catches a true no-op while
  // leaving slack for cases where the half->full interpolation already
  // satisfies the local force balance.
  EXPECT_GT(diff, 1.0e-10)
      << "lbsubs = true left bsubs_full bit-identical to the lbsubs = false "
         "baseline (max absolute diff "
      << diff << ")";
}

// The recovered B_s and its tangential derivatives must satisfy
//   bsupu_full * d(B_s)/du + bsupv_full * d(B_s)/dv = brho_corrected
// at every collocation point on every interior surface, up to the linear
// solver's residual tolerance.
TEST_F(LbsubsSolovevTest, TrueSatisfiesRadialForceBalanceResidual) {
  indata_.lbsubs = true;
  auto maybe_vmec = Vmec::FromIndata(indata_);
  ASSERT_TRUE(maybe_vmec.ok());
  Vmec& vmec = **maybe_vmec;
  const absl::StatusOr<bool> reached_checkpoint = vmec.run();
  ASSERT_TRUE(reached_checkpoint.ok());
  ASSERT_FALSE(*reached_checkpoint) << "solovev did not converge";

  const auto& oq = vmec.output_quantities_;
  const Sizes& s = vmec.s_;
  const FlowControl& fc = vmec.fc_;

  // Sample brho across all interior surfaces to give the residual tolerance
  // an absolute scale in the same units as the RHS itself.
  double rhs_scale = 1.0;
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const ForceBalanceRhs rhs =
        RebuildForceBalanceRhs(s, fc, oq.vmec_internal_results, jF);
    for (double v : rhs.brho) {
      rhs_scale = std::max(rhs_scale, std::abs(v));
    }
  }
  // The implementation rejects any surface whose solve has a residual larger
  // than 1e-8 * |brhs| and falls back to the half->full interpolation; this
  // gate is one digit looser, so a single bad surface fails the test rather
  // than silently falling back.
  const double tol = 1.0e-7 * rhs_scale;

  double worst = 0.0;
  int worst_jF = -1;
  for (int jF = 1; jF < fc.ns - 1; ++jF) {
    const double r = InteriorForceBalanceResidualMax(
        s, fc, oq.vmec_internal_results, oq.covariant_b_derivatives, jF);
    if (r > worst) {
      worst = r;
      worst_jF = jF;
    }
  }
  EXPECT_LT(worst, tol) << "worst radial-force-balance residual " << worst
                        << " (at jF=" << worst_jF << ") exceeds tolerance "
                        << tol << " (rhs_scale=" << rhs_scale << ")";
}

// -------------------------------------------------------------------------
// lasym = true: parse up_down_asym.json and construct the Vmec instance to
// exercise input validation, then skip. RecomputeBSubSFromRadialForceBalance
// does not implement the non-stellarator-symmetric branch, and the
// equilibrium iteration does not handle lasym = true.
// -------------------------------------------------------------------------

TEST(LbsubsLasymTest, UpDownAsymPathIsDeferred) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/up_down_asym.json");
  ASSERT_TRUE(indata_json.ok());
  absl::StatusOr<VmecINDATA> parsed = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(parsed.ok());
  VmecINDATA indata = *parsed;
  ASSERT_TRUE(indata.lasym);
  indata.lbsubs = true;
  EXPECT_TRUE(indata.lbsubs);
  EXPECT_TRUE(indata.lasym);

  // No run() here; the equilibrium iteration does not handle lasym = true.
  const auto maybe_vmec = Vmec::FromIndata(indata);
  ASSERT_TRUE(maybe_vmec.ok())
      << "lasym Vmec::FromIndata failed: " << maybe_vmec.status().message();

  GTEST_SKIP() << "lbsubs lasym = true path is gated off; "
                  "see RecomputeBSubSFromRadialForceBalance.";
}

// -------------------------------------------------------------------------
// JSON wiring sanity check: a user-supplied lbsubs flag round-trips through
// the VmecINDATA serializer. The default-value path is covered in
// vmec_indata_test.
// -------------------------------------------------------------------------

TEST(LbsubsJsonTest, LbsubsTrueRoundTrips) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/solovev.json");
  ASSERT_TRUE(indata_json.ok());
  absl::StatusOr<VmecINDATA> parsed = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(parsed.ok());
  VmecINDATA indata = *parsed;
  EXPECT_FALSE(indata.lbsubs);  // not set in the file -> default false

  indata.lbsubs = true;
  const absl::StatusOr<std::string> reserialized = indata.ToJson();
  ASSERT_TRUE(reserialized.ok());
  const absl::StatusOr<VmecINDATA> reparsed =
      VmecINDATA::FromJson(*reserialized);
  ASSERT_TRUE(reparsed.ok());
  EXPECT_TRUE(reparsed->lbsubs);
}

}  // namespace vmecpp

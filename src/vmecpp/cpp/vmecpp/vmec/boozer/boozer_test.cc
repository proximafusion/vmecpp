// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/boozer/boozer.h"

#include <cmath>
#include <memory>
#include <string>

#include "absl/log/check.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "util/testing/numerical_comparison_lib.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

using file_io::ReadFile;
using testing::IsCloseRelAbs;
using vmecpp::BoozerCoordinates;
using vmecpp::BoozerTransform;
using vmecpp::Vmec;
using vmecpp::VmecINDATA;
using vmecpp::WOutFileContents;

namespace {

std::unique_ptr<Vmec> Solve(const std::string& case_name) {
  const absl::StatusOr<std::string> indata_json =
      ReadFile("vmecpp/test_data/" + case_name + ".json");
  CHECK(indata_json.ok()) << indata_json.status();
  const absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  CHECK(indata.ok()) << indata.status();
  absl::StatusOr<std::unique_ptr<Vmec>> vmec = Vmec::FromIndata(
      *indata, nullptr, std::nullopt, vmecpp::OutputMode::kSilent);
  CHECK(vmec.ok()) << vmec.status();
  const absl::StatusOr<bool> reached_checkpoint = (*vmec)->run();
  CHECK(reached_checkpoint.ok()) << reached_checkpoint.status();
  return std::move(*vmec);
}

// In Boozer coordinates sqrt(g_B) |B|^2 is a flux function, and the (0, 0)
// Boozer harmonic of sqrt(g_B) times that of |B|^2 has to agree with the
// direct surface integral of sqrt(g) |B|^2, which the angle map leaves
// invariant. The spread of sqrt(g_B) |B|^2 over the surface measures how well
// the transformation resolved the angle map.
TEST(BoozerTransform, JacobianTimesBSquaredIsAFluxFunction) {
  const std::unique_ptr<Vmec> vmec = Solve("cth_like_fixed_bdy");
  const WOutFileContents& wout = vmec->output_quantities_.wout;

  const absl::StatusOr<BoozerCoordinates> boozer =
      BoozerTransform(wout, /*mboz=*/8, /*nboz=*/8);
  ASSERT_TRUE(boozer.ok()) << boozer.status();
  const BoozerCoordinates& b = *boozer;

  EXPECT_EQ(b.surfaces.size(), wout.ns - 1);
  EXPECT_EQ(b.xm_b.size(), 9 + 7 * 17);
  for (int surface = 0; surface < b.surfaces.size(); ++surface) {
    // The Boozer condition holds only as exactly as the radial current
    // vanishes, which this case converges to ftol = 1e-6; the measured spread
    // is 1e-4 to 2e-4, and the bound sits five times above it.
    EXPECT_LT(b.jacobian_spread[surface], 1.0e-3)
        << "half-grid column " << b.surfaces[surface];
    // the Boozer currents are the (0, 0) covariant components of the wout
    const int js = b.surfaces[surface];
    EXPECT_EQ(b.g_b[surface], wout.bvco[js]);
    EXPECT_EQ(b.i_b[surface], wout.buco[js]);
    EXPECT_EQ(b.iota_b[surface], wout.iotas[js]);
  }
}

// The transformation is a relabeling of the angles, so the surface average
// of |B| weighted by the Jacobian is the same in both coordinate systems:
// sum over modes of gmnc * bmnc (with the factor 1/2 off the (0, 0) mode) is
// the flux-surface integral of sqrt(g) |B| / (2 pi)^2, in either basis, up to
// the flux normalization of the Boozer Jacobian. The Boozer side carries the
// Boozer relation sqrt(g_B) = (G + iota I) / |B|^2, which the equilibrium
// satisfies only as exactly as its force residual; the bound is five times
// the deviation measured at ftol = 1e-6.
TEST(BoozerTransform, JacobianWeightedFieldAverageIsInvariant) {
  const std::unique_ptr<Vmec> vmec = Solve("cth_like_fixed_bdy");
  const WOutFileContents& wout = vmec->output_quantities_.wout;
  const absl::StatusOr<BoozerCoordinates> boozer =
      BoozerTransform(wout, /*mboz=*/8, /*nboz=*/8);
  ASSERT_TRUE(boozer.ok()) << boozer.status();
  const BoozerCoordinates& b = *boozer;

  for (int surface = 0; surface < b.surfaces.size(); ++surface) {
    const int js = b.surfaces[surface];
    double vmec_average = 0.0;
    for (int mn = 0; mn < wout.mnmax_nyq; ++mn) {
      const double weight =
          (wout.xm_nyq[mn] == 0 && wout.xn_nyq[mn] == 0) ? 1.0 : 0.5;
      vmec_average += weight * wout.gmnc(mn, js) * wout.bmnc(mn, js);
    }
    double boozer_average = 0.0;
    for (int k = 0; k < b.xm_b.size(); ++k) {
      const double weight = (b.xm_b[k] == 0 && b.xn_b[k] == 0) ? 1.0 : 0.5;
      boozer_average += weight * b.gmnc_b(k, surface) * b.bmnc_b(k, surface);
    }
    boozer_average *= wout.phips[js];
    EXPECT_TRUE(IsCloseRelAbs(vmec_average, boozer_average, 1.0e-3))
        << "half-grid column " << js;
  }
}

// An axisymmetric equilibrium has Boozer angles that differ from the VMEC
// angles only poloidally, so every toroidal Boozer harmonic vanishes and the
// (0, 0) harmonic of |B| on axis-adjacent surfaces matches the VMEC one.
TEST(BoozerTransform, AxisymmetricSpectrumHasNoToroidalHarmonics) {
  const std::unique_ptr<Vmec> vmec = Solve("solovev");
  const WOutFileContents& wout = vmec->output_quantities_.wout;
  const absl::StatusOr<BoozerCoordinates> boozer =
      BoozerTransform(wout, /*mboz=*/6, /*nboz=*/2);
  ASSERT_TRUE(boozer.ok()) << boozer.status();
  const BoozerCoordinates& b = *boozer;

  for (int surface = 0; surface < b.surfaces.size(); ++surface) {
    for (int k = 0; k < b.xm_b.size(); ++k) {
      if (b.xn_b[k] != 0) {
        EXPECT_NEAR(b.bmnc_b(k, surface), 0.0, 1.0e-12);
        EXPECT_NEAR(b.rmnc_b(k, surface), 0.0, 1.0e-12);
        EXPECT_NEAR(b.zmns_b(k, surface), 0.0, 1.0e-12);
        EXPECT_NEAR(b.numns_b(k, surface), 0.0, 1.0e-12);
      }
    }
    // measured 1.2e-6 to 7.7e-6 at ftol = 1e-12; the bound is five times the
    // largest
    EXPECT_LT(b.jacobian_spread[surface], 5.0e-5);
  }
}

TEST(BoozerTransform, RejectsBadArguments) {
  const std::unique_ptr<Vmec> vmec = Solve("solovev");
  const WOutFileContents& wout = vmec->output_quantities_.wout;
  EXPECT_FALSE(BoozerTransform(wout, 0, 0).ok());
  EXPECT_FALSE(BoozerTransform(wout, 4, -1).ok());
  EXPECT_FALSE(BoozerTransform(wout, 4, 0, {0}).ok());
  EXPECT_FALSE(BoozerTransform(wout, 4, 0, {wout.ns}).ok());
  EXPECT_TRUE(BoozerTransform(wout, 4, 0, {1, wout.ns - 1}).ok());
}

}  // namespace

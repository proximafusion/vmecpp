// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/boundaries/boundaries.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

namespace vmecpp {

namespace {
double BoundarySpectralWidth(const VmecINDATA& indata) {
  const Sizes sizes(indata);
  const FourierBasisFastPoloidal fourier_basis(&sizes);
  Boundaries boundaries(&sizes, &fourier_basis,
                        vmec_algorithm_constants::kSignOfJacobian);
  boundaries.setupFromIndata(indata, /*verbose=*/false);
  return boundaries.ComputeSpectralWidth();
}
}  // namespace

TEST(TestBoundaries, SpectralWidthOfSmoothBoundaryIsLow) {
  const VmecINDATA indata =
      VmecINDATA::FromFile("vmecpp/test_data/solovev.json");

  // solovev has a smooth boundary (R at m=0,1,2; Z at m=1,2), so its spectral
  // width is close to 1 and well below the spectrally-dense warning threshold.
  const double spectral_width = BoundarySpectralWidth(indata);
  EXPECT_GE(spectral_width, 1.0);
  EXPECT_LT(spectral_width, kSpectrallyDenseBoundaryThreshold);
}

TEST(TestBoundaries, SpectralWidthOfDenseBoundaryExceedsThreshold) {
  VmecINDATA indata = VmecINDATA::FromFile("vmecpp/test_data/solovev.json");

  // Make room for high-m modes and inject a high-poloidal-mode ripple, which
  // drives the boundary spectral width above the warning threshold.
  indata.SetMpolNtor(/*new_mpol=*/12, /*new_ntor=*/0);
  for (int m = 8; m <= 10; ++m) {
    indata.rbc(m, 0) = 0.05;
    indata.zbs(m, 0) = 0.05;
  }
  EXPECT_GT(BoundarySpectralWidth(indata), kSpectrallyDenseBoundaryThreshold);

  // A spectrally dense boundary is still a valid input: it is warned about, not
  // rejected.
  EXPECT_TRUE(IsConsistent(indata, /*enable_info_messages=*/false).ok());
}

TEST(TestBoundaries, BundledInputsAreNotSpectrallyDense) {
  // The threshold has to sit above every input VMEC++ ships, or a routine run
  // would warn about a boundary that is known to be fine. cma is the densest of
  // them; cth_like_fixed_bdy is the densest three-dimensional one.
  const std::vector<std::string> input_files = {
      "vmecpp/test_data/solovev.json", "vmecpp/test_data/cma.json",
      "vmecpp/test_data/cth_like_fixed_bdy.json"};

  for (const std::string& input_file : input_files) {
    const VmecINDATA indata = VmecINDATA::FromFile(input_file);
    EXPECT_LT(BoundarySpectralWidth(indata), kSpectrallyDenseBoundaryThreshold)
        << "for " << input_file;
  }
}

namespace {
// theta -> pi - theta on an ntor = 0 boundary. The surface is unchanged as a
// point set and only the poloidal direction reverses, so the orientation check
// has to notice and flip it back.
VmecINDATA RelabelledByPiMinusTheta(const VmecINDATA& indata) {
  VmecINDATA out = indata;
  const int n0 = out.ntor;  // column of n = 0
  for (int m = 0; m < out.rbc.rows(); ++m) {
    const double s = (m % 2 == 0) ? 1.0 : -1.0;
    out.rbc(m, n0) *= s;
    out.zbs(m, n0) *= -s;
    if (out.rbs.has_value()) {
      (*out.rbs)(m, n0) *= -s;
    }
    if (out.zbc.has_value()) {
      (*out.zbc)(m, n0) *= s;
    }
  }
  return out;
}

// Whether the orientation check asks for the poloidal angle to be flipped.
bool AsksForThetaFlip(const VmecINDATA& indata) {
  const Sizes sizes(indata);
  const FourierBasisFastPoloidal fourier_basis(&sizes);
  Boundaries boundaries(&sizes, &fourier_basis,
                        vmec_algorithm_constants::kSignOfJacobian);
  return boundaries.setupFromIndata(indata, /*verbose=*/false);
}
}  // namespace

TEST(TestBoundaries, RelabelledAsymmetricBoundaryIsFlippedBack) {
  const VmecINDATA indata =
      VmecINDATA::FromFile("vmecpp/test_data/up_down_asym_current.json");
  ASSERT_TRUE(indata.lasym);
  ASSERT_EQ(indata.ntor, 0);

  // The boundary as given is already traversed the right way round.
  EXPECT_FALSE(AsksForThetaFlip(indata));

  // Reversing the poloidal direction of the same surface has to be caught.
  EXPECT_TRUE(AsksForThetaFlip(RelabelledByPiMinusTheta(indata)));
}

}  // namespace vmecpp

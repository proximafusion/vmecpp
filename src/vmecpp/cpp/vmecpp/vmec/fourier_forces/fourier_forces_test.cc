// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"

#include <utility>

#include "gtest/gtest.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {
namespace {

// Helper to create a FourierForces with known data in the frcc span.
// Uses a simple 3D, non-asymmetric configuration so that frcc, frss, fzsc,
// fzcs, flsc, flcs are all allocated (the lthreed=true spans).
class FourierForcesMoveTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 3D, non-asymmetric configuration
    sizes = std::make_unique<Sizes>(/*lasym=*/false, /*nfp=*/5, /*mpol=*/4,
                                    /*ntor=*/2, /*ntheta=*/0, /*nzeta=*/10);
    partitioning = std::make_unique<RadialPartitioning>();
    constexpr int kNs = 6;
    partitioning->adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0,
                                           kNs, /*lfreeb=*/false,
                                           /*printout=*/false);
    ns = kNs;
  }

  std::unique_ptr<FourierForces> MakeForces(double fill_value) {
    auto ff =
        std::make_unique<FourierForces>(sizes.get(), partitioning.get(), ns);
    // Fill the allocated spans with a known value.
    for (auto& v : ff->frcc) v = fill_value;
    for (auto& v : ff->fzsc) v = fill_value * 2.0;
    for (auto& v : ff->flsc) v = fill_value * 3.0;
    // lthreed spans
    for (auto& v : ff->frss) v = fill_value * 4.0;
    for (auto& v : ff->fzcs) v = fill_value * 5.0;
    for (auto& v : ff->flcs) v = fill_value * 6.0;
    return ff;
  }

  int ns = 0;
  std::unique_ptr<Sizes> sizes;
  std::unique_ptr<RadialPartitioning> partitioning;
};

TEST_F(FourierForcesMoveTest, MoveConstructorTransfersData) {
  auto original = MakeForces(1.5);
  const std::size_t expected_size = original->frcc.size();
  ASSERT_GT(expected_size, 0);

  // Move-construct a new object.
  FourierForces moved(std::move(*original));

  // The moved-to object should have the correct data and size.
  ASSERT_EQ(moved.frcc.size(), expected_size);
  for (const auto& v : moved.frcc) {
    EXPECT_DOUBLE_EQ(v, 1.5);
  }
  for (const auto& v : moved.fzsc) {
    EXPECT_DOUBLE_EQ(v, 3.0);
  }
  for (const auto& v : moved.flsc) {
    EXPECT_DOUBLE_EQ(v, 4.5);
  }
  for (const auto& v : moved.frss) {
    EXPECT_DOUBLE_EQ(v, 6.0);
  }
  for (const auto& v : moved.fzcs) {
    EXPECT_DOUBLE_EQ(v, 7.5);
  }
  for (const auto& v : moved.flcs) {
    EXPECT_DOUBLE_EQ(v, 9.0);
  }
}

TEST_F(FourierForcesMoveTest, MoveConstructorSpansPointToOwnData) {
  auto original = MakeForces(2.0);

  FourierForces moved(std::move(*original));

  // Verify spans point to the moved object's own data by mutating through the
  // span and checking the value is reflected.
  ASSERT_GT(moved.frcc.size(), 0);
  moved.frcc[0] = 42.0;
  EXPECT_DOUBLE_EQ(moved.frcc[0], 42.0);

  // Also verify that setZero (which writes through the base class vectors)
  // is reflected in the spans.
  moved.setZero();
  for (const auto& v : moved.frcc) {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
  for (const auto& v : moved.fzsc) {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
}

TEST_F(FourierForcesMoveTest, MoveAssignmentTransfersData) {
  auto source = MakeForces(3.0);
  auto target = MakeForces(0.0);
  const std::size_t expected_size = source->frcc.size();

  // Move-assign.
  *target = std::move(*source);

  ASSERT_EQ(target->frcc.size(), expected_size);
  for (const auto& v : target->frcc) {
    EXPECT_DOUBLE_EQ(v, 3.0);
  }
  for (const auto& v : target->fzsc) {
    EXPECT_DOUBLE_EQ(v, 6.0);
  }
  for (const auto& v : target->flsc) {
    EXPECT_DOUBLE_EQ(v, 9.0);
  }
}

TEST_F(FourierForcesMoveTest, MoveAssignmentSpansPointToOwnData) {
  auto source = MakeForces(5.0);
  auto target = MakeForces(0.0);

  *target = std::move(*source);

  // Mutate through span and verify consistency.
  ASSERT_GT(target->frcc.size(), 0);
  target->frcc[0] = 99.0;
  EXPECT_DOUBLE_EQ(target->frcc[0], 99.0);

  // setZero writes through the base class vectors; spans must reflect this.
  target->setZero();
  for (const auto& v : target->frcc) {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
  for (const auto& v : target->frss) {
    EXPECT_DOUBLE_EQ(v, 0.0);
  }
}

TEST_F(FourierForcesMoveTest, CopyConstructorCreatesIndependentCopy) {
  auto original = MakeForces(7.0);

  FourierForces copy(*original);

  // Copy should have the same data.
  ASSERT_EQ(copy.frcc.size(), original->frcc.size());
  for (std::size_t i = 0; i < copy.frcc.size(); ++i) {
    EXPECT_DOUBLE_EQ(copy.frcc[i], original->frcc[i]);
  }

  // Mutating the copy must not affect the original.
  copy.frcc[0] = 999.0;
  EXPECT_DOUBLE_EQ(original->frcc[0], 7.0);
}

TEST_F(FourierForcesMoveTest, CopyAssignmentCreatesIndependentCopy) {
  auto source = MakeForces(11.0);
  auto target = MakeForces(0.0);

  *target = *source;

  ASSERT_EQ(target->frcc.size(), source->frcc.size());
  for (std::size_t i = 0; i < target->frcc.size(); ++i) {
    EXPECT_DOUBLE_EQ(target->frcc[i], source->frcc[i]);
  }

  // Mutating the target must not affect the source.
  target->frcc[0] = 999.0;
  EXPECT_DOUBLE_EQ(source->frcc[0], 11.0);
}

}  // namespace
}  // namespace vmecpp

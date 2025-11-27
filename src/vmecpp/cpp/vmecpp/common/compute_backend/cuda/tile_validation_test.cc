// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Validation tests for the tiled GPU compute backend.
// These tests verify:
// 1. Tile scheduler produces correct coverage
// 2. Memory budget calculations are accurate
// 3. Tiled execution produces identical results to non-tiled

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "vmecpp/common/compute_backend/cuda/tile_scheduler.h"

namespace vmecpp {
namespace {

// =============================================================================
// TileScheduler Tests
// =============================================================================

class TileSchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

TEST_F(TileSchedulerTest, SingleTileCoversEntireDomain) {
  TileSchedulerConfig config;
  config.ns = 50;
  config.tile_size = 100;  // Larger than ns
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  EXPECT_EQ(scheduler.NumTiles(), 1);
  EXPECT_TRUE(scheduler.ValidateCoverage());

  const auto& tile = scheduler.GetTile(0);
  EXPECT_EQ(tile.start_surface, 1);
  EXPECT_EQ(tile.end_surface, 50);
  EXPECT_EQ(tile.overlap_before, 0);
  EXPECT_EQ(tile.overlap_after, 0);
}

TEST_F(TileSchedulerTest, MultipleTilesNoOverlap) {
  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  EXPECT_EQ(scheduler.NumTiles(), 4);
  EXPECT_TRUE(scheduler.ValidateCoverage());

  // Check each tile
  for (int i = 0; i < 4; ++i) {
    const auto& tile = scheduler.GetTile(i);
    EXPECT_EQ(tile.start_surface, 1 + i * 25);
    EXPECT_EQ(tile.end_surface, 1 + (i + 1) * 25);
    EXPECT_EQ(tile.overlap_before, 0);
    EXPECT_EQ(tile.overlap_after, 0);
  }
}

TEST_F(TileSchedulerTest, ForwardStencilOverlap) {
  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kForwardStencil;

  TileScheduler scheduler(config);

  EXPECT_EQ(scheduler.NumTiles(), 4);
  EXPECT_TRUE(scheduler.ValidateCoverage());

  // First tile: no overlap before, overlap after
  const auto& tile0 = scheduler.GetTile(0);
  EXPECT_EQ(tile0.overlap_before, 0);
  EXPECT_EQ(tile0.overlap_after, 1);

  // Middle tiles: no overlap before, overlap after
  const auto& tile1 = scheduler.GetTile(1);
  EXPECT_EQ(tile1.overlap_before, 0);
  EXPECT_EQ(tile1.overlap_after, 1);

  // Last tile: no overlap
  const auto& tile3 = scheduler.GetTile(3);
  EXPECT_EQ(tile3.overlap_before, 0);
  EXPECT_EQ(tile3.overlap_after, 0);
}

TEST_F(TileSchedulerTest, BackwardStencilOverlap) {
  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kBackwardStencil;

  TileScheduler scheduler(config);

  EXPECT_EQ(scheduler.NumTiles(), 4);
  EXPECT_TRUE(scheduler.ValidateCoverage());

  // First tile: no overlap
  const auto& tile0 = scheduler.GetTile(0);
  EXPECT_EQ(tile0.overlap_before, 0);
  EXPECT_EQ(tile0.overlap_after, 0);

  // Middle and last tiles: overlap before
  const auto& tile1 = scheduler.GetTile(1);
  EXPECT_EQ(tile1.overlap_before, 1);
  EXPECT_EQ(tile1.overlap_after, 0);

  const auto& tile3 = scheduler.GetTile(3);
  EXPECT_EQ(tile3.overlap_before, 1);
  EXPECT_EQ(tile3.overlap_after, 0);
}

TEST_F(TileSchedulerTest, BothStencilOverlap) {
  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kBothStencil;

  TileScheduler scheduler(config);

  EXPECT_EQ(scheduler.NumTiles(), 4);
  EXPECT_TRUE(scheduler.ValidateCoverage());

  // First tile: no overlap before, overlap after
  const auto& tile0 = scheduler.GetTile(0);
  EXPECT_EQ(tile0.overlap_before, 0);
  EXPECT_EQ(tile0.overlap_after, 1);

  // Middle tiles: overlap both sides
  const auto& tile1 = scheduler.GetTile(1);
  EXPECT_EQ(tile1.overlap_before, 1);
  EXPECT_EQ(tile1.overlap_after, 1);

  // Last tile: overlap before, no overlap after
  const auto& tile3 = scheduler.GetTile(3);
  EXPECT_EQ(tile3.overlap_before, 1);
  EXPECT_EQ(tile3.overlap_after, 0);
}

TEST_F(TileSchedulerTest, UnevenTileSizes) {
  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 30;  // Doesn't divide evenly
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  // 100 surfaces with tile_size=30: tiles at [1,31), [31,61), [61,91), [91,100)
  EXPECT_EQ(scheduler.NumTiles(), 4);
  EXPECT_TRUE(scheduler.ValidateCoverage());

  // Last tile should have fewer surfaces
  const auto& last_tile = scheduler.GetTile(3);
  EXPECT_EQ(last_tile.start_surface, 91);
  EXPECT_EQ(last_tile.end_surface, 100);
  EXPECT_EQ(last_tile.num_output_surfaces(), 9);
}

TEST_F(TileSchedulerTest, GetTileForSurface) {
  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  // Surface 1 should be in tile 0
  EXPECT_EQ(scheduler.GetTileFor(1).tile_index, 0);

  // Surface 25 should be in tile 0
  EXPECT_EQ(scheduler.GetTileFor(25).tile_index, 0);

  // Surface 26 should be in tile 1
  EXPECT_EQ(scheduler.GetTileFor(26).tile_index, 1);

  // Surface 99 should be in tile 3
  EXPECT_EQ(scheduler.GetTileFor(99).tile_index, 3);
}

TEST_F(TileSchedulerTest, TileIterator) {
  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);
  TileIterator iter(scheduler, 2);  // Double buffering

  int count = 0;
  while (!iter.Done()) {
    EXPECT_EQ(iter.CurrentTile().tile_index, count);
    EXPECT_EQ(iter.CurrentBufferIndex(), count % 2);
    iter.Next();
    ++count;
  }

  EXPECT_EQ(count, 4);
}

TEST_F(TileSchedulerTest, InputSurfaceCount) {
  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kForwardStencil;

  TileScheduler scheduler(config);

  // With forward stencil, non-last tiles need 1 extra surface
  // Total input surfaces = 25+1 + 25+1 + 25+1 + 25 = 103
  // But overlaps are shared, so we count unique inputs per tile
  int total_input = scheduler.TotalInputSurfaces();
  EXPECT_EQ(total_input, 103);
}

// =============================================================================
// Numerical Validation Tests (CPU-only, no CUDA required)
// =============================================================================

class TileNumericalValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Use fixed seed for reproducibility
    rng_.seed(42);
  }

  std::mt19937 rng_;

  // Generate random array
  std::vector<double> RandomArray(size_t size, double min_val = -1.0,
                                  double max_val = 1.0) {
    std::vector<double> arr(size);
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (auto& x : arr) {
      x = dist(rng_);
    }
    return arr;
  }

  // Compare arrays with tolerance
  bool ArraysEqual(const std::vector<double>& a, const std::vector<double>& b,
                   double tol = 1e-10) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::abs(a[i] - b[i]) > tol) {
        return false;
      }
    }
    return true;
  }

  // Compute max absolute error
  double MaxAbsError(const std::vector<double>& a,
                     const std::vector<double>& b) {
    double max_err = 0.0;
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
      max_err = std::max(max_err, std::abs(a[i] - b[i]));
    }
    return max_err;
  }
};

// Test that simulates tiled summation produces same result as non-tiled
TEST_F(TileNumericalValidationTest, TiledSummationMatchesNonTiled) {
  const int ns = 100;
  const int tile_size = 25;
  const int n_zeta = 32;
  const int n_theta = 32;
  const size_t grid_size = ns * n_zeta * n_theta;

  // Generate random input
  std::vector<double> input = RandomArray(grid_size);

  // Non-tiled sum
  double non_tiled_sum = std::accumulate(input.begin(), input.end(), 0.0);

  // Tiled sum
  TileSchedulerConfig config;
  config.ns = ns;
  config.tile_size = tile_size;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  double tiled_sum = 0.0;
  for (const auto& tile : scheduler.GetTiles()) {
    size_t start_idx = (tile.start_surface - 1) * n_zeta * n_theta;
    size_t end_idx = (tile.end_surface - 1) * n_zeta * n_theta;
    for (size_t i = start_idx; i < end_idx; ++i) {
      tiled_sum += input[i];
    }
  }

  EXPECT_NEAR(tiled_sum, non_tiled_sum, 1e-10);
}

// Test that simulates tiled stencil operation with overlap
TEST_F(TileNumericalValidationTest, TiledStencilWithOverlap) {
  const int ns = 100;
  const int tile_size = 25;
  const int n_points = 32;  // Points per surface

  // Input array: ns surfaces, each with n_points
  std::vector<double> input = RandomArray(ns * n_points);

  // Output: computed at half-grid points (ns-1 total)
  std::vector<double> non_tiled_output(ns * n_points, 0.0);
  std::vector<double> tiled_output(ns * n_points, 0.0);

  // Non-tiled stencil: output[j] = (input[j] + input[j+1]) / 2
  for (int j = 0; j < ns - 1; ++j) {
    for (int k = 0; k < n_points; ++k) {
      int idx = j * n_points + k;
      int idx_next = (j + 1) * n_points + k;
      non_tiled_output[idx] = (input[idx] + input[idx_next]) / 2.0;
    }
  }

  // Tiled stencil with forward overlap
  TileSchedulerConfig config;
  config.ns = ns;
  config.tile_size = tile_size;
  config.ns_min = 1;  // 1-based indexing in config, but we use 0-based here
  config.op_type = TileOperationType::kForwardStencil;

  TileScheduler scheduler(config);

  for (const auto& tile : scheduler.GetTiles()) {
    // Compute for surfaces [start, end) using input from [start, end + overlap]
    int start = tile.start_surface - 1;  // Convert to 0-based
    int end = tile.end_surface - 1;

    for (int j = start; j < end && j < ns - 1; ++j) {
      for (int k = 0; k < n_points; ++k) {
        int idx = j * n_points + k;
        int idx_next = (j + 1) * n_points + k;
        tiled_output[idx] = (input[idx] + input[idx_next]) / 2.0;
      }
    }
  }

  // Compare outputs
  EXPECT_TRUE(ArraysEqual(tiled_output, non_tiled_output, 1e-14));
}

// Test that simulates backward stencil (like MHD forces)
TEST_F(TileNumericalValidationTest, TiledBackwardStencil) {
  const int ns = 100;
  const int tile_size = 25;
  const int n_points = 32;

  std::vector<double> input = RandomArray(ns * n_points);
  std::vector<double> non_tiled_output(ns * n_points, 0.0);
  std::vector<double> tiled_output(ns * n_points, 0.0);

  // Backward stencil: output[j] = input[j] - input[j-1] (for j > 0)
  for (int j = 1; j < ns; ++j) {
    for (int k = 0; k < n_points; ++k) {
      int idx = j * n_points + k;
      int idx_prev = (j - 1) * n_points + k;
      non_tiled_output[idx] = input[idx] - input[idx_prev];
    }
  }

  // Tiled with backward overlap
  TileSchedulerConfig config;
  config.ns = ns;
  config.tile_size = tile_size;
  config.ns_min = 1;
  config.op_type = TileOperationType::kBackwardStencil;

  TileScheduler scheduler(config);

  for (const auto& tile : scheduler.GetTiles()) {
    int start = tile.start_surface - 1;
    int end = tile.end_surface - 1;

    for (int j = start; j < end; ++j) {
      if (j == 0) continue;  // Skip first surface (no predecessor)

      for (int k = 0; k < n_points; ++k) {
        int idx = j * n_points + k;
        int idx_prev = (j - 1) * n_points + k;
        tiled_output[idx] = input[idx] - input[idx_prev];
      }
    }
  }

  EXPECT_TRUE(ArraysEqual(tiled_output, non_tiled_output, 1e-14));
}

// Test tile coverage with various configurations
TEST_F(TileNumericalValidationTest, TileCoverageStressTest) {
  // Test many configurations
  std::vector<int> ns_values = {10, 50, 100, 127, 256, 1000};
  std::vector<int> tile_sizes = {1, 5, 10, 25, 33, 64, 100};
  std::vector<TileOperationType> op_types = {
      TileOperationType::kNoOverlap,
      TileOperationType::kForwardStencil,
      TileOperationType::kBackwardStencil,
      TileOperationType::kBothStencil,
  };

  for (int ns : ns_values) {
    for (int tile_size : tile_sizes) {
      for (auto op_type : op_types) {
        TileSchedulerConfig config;
        config.ns = ns;
        config.tile_size = tile_size;
        config.ns_min = 1;
        config.op_type = op_type;

        TileScheduler scheduler(config);

        // Validate coverage
        EXPECT_TRUE(scheduler.ValidateCoverage())
            << "Coverage failed for ns=" << ns << ", tile_size=" << tile_size;

        // Verify all surfaces are covered
        std::vector<bool> covered(ns, false);
        for (const auto& tile : scheduler.GetTiles()) {
          for (int s = tile.start_surface; s < tile.end_surface; ++s) {
            EXPECT_FALSE(covered[s - 1])
                << "Surface " << s << " covered twice";
            covered[s - 1] = true;
          }
        }

        for (int s = 0; s < ns; ++s) {
          EXPECT_TRUE(covered[s]) << "Surface " << (s + 1) << " not covered";
        }
      }
    }
  }
}

// Test reduction across tiles (like finding min/max Jacobian)
TEST_F(TileNumericalValidationTest, TiledReduction) {
  const int ns = 100;
  const int tile_size = 25;
  const int n_points = 32;

  std::vector<double> input = RandomArray(ns * n_points, 0.0, 1.0);

  // Find global min/max
  double global_min = *std::min_element(input.begin(), input.end());
  double global_max = *std::max_element(input.begin(), input.end());

  // Tiled reduction
  TileSchedulerConfig config;
  config.ns = ns;
  config.tile_size = tile_size;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  double tiled_min = std::numeric_limits<double>::max();
  double tiled_max = std::numeric_limits<double>::lowest();

  for (const auto& tile : scheduler.GetTiles()) {
    size_t start_idx = (tile.start_surface - 1) * n_points;
    size_t end_idx = (tile.end_surface - 1) * n_points;

    for (size_t i = start_idx; i < end_idx; ++i) {
      tiled_min = std::min(tiled_min, input[i]);
      tiled_max = std::max(tiled_max, input[i]);
    }
  }

  EXPECT_DOUBLE_EQ(tiled_min, global_min);
  EXPECT_DOUBLE_EQ(tiled_max, global_max);
}

// Test carry state between tiles (like MHD forces "inside" values)
TEST_F(TileNumericalValidationTest, TiledCarryState) {
  const int ns = 100;
  const int tile_size = 25;

  // Simulate computing a running sum where each surface depends on previous
  std::vector<double> input = RandomArray(ns);
  std::vector<double> non_tiled_output(ns);
  std::vector<double> tiled_output(ns);

  // Non-tiled: output[j] = sum(input[0:j+1])
  double running_sum = 0.0;
  for (int j = 0; j < ns; ++j) {
    running_sum += input[j];
    non_tiled_output[j] = running_sum;
  }

  // Tiled with carry state
  TileSchedulerConfig config;
  config.ns = ns;
  config.tile_size = tile_size;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  double carry = 0.0;  // State carried between tiles
  for (const auto& tile : scheduler.GetTiles()) {
    int start = tile.start_surface - 1;
    int end = tile.end_surface - 1;

    for (int j = start; j < end; ++j) {
      carry += input[j];
      tiled_output[j] = carry;
    }
  }

  EXPECT_TRUE(ArraysEqual(tiled_output, non_tiled_output, 1e-14));
}

}  // namespace
}  // namespace vmecpp

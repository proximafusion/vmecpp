// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Standalone validation for tile scheduler - no external dependencies.
// Compile with: g++ -std=c++20 -I src/vmecpp/cpp \
//   src/vmecpp/cpp/vmecpp/common/compute_backend/cuda/standalone_validation.cc \
//   src/vmecpp/cpp/vmecpp/common/compute_backend/cuda/tile_scheduler.cc \
//   -o standalone_validation

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "vmecpp/common/compute_backend/cuda/tile_scheduler.h"

namespace vmecpp {

// Simple test framework
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define EXPECT_TRUE(cond)                                              \
  do {                                                                 \
    tests_run++;                                                       \
    if (cond) {                                                        \
      tests_passed++;                                                  \
    } else {                                                           \
      tests_failed++;                                                  \
      std::cerr << "FAIL: " << #cond << " at line " << __LINE__ << "\n"; \
    }                                                                  \
  } while (0)

#define EXPECT_EQ(a, b)                                                \
  do {                                                                 \
    tests_run++;                                                       \
    if ((a) == (b)) {                                                  \
      tests_passed++;                                                  \
    } else {                                                           \
      tests_failed++;                                                  \
      std::cerr << "FAIL: " << #a << " == " << #b << " (got " << (a)   \
                << " vs " << (b) << ") at line " << __LINE__ << "\n";  \
    }                                                                  \
  } while (0)

#define EXPECT_NEAR(a, b, tol)                                         \
  do {                                                                 \
    tests_run++;                                                       \
    if (std::abs((a) - (b)) <= (tol)) {                                \
      tests_passed++;                                                  \
    } else {                                                           \
      tests_failed++;                                                  \
      std::cerr << "FAIL: |" << #a << " - " << #b << "| <= " << (tol)  \
                << " (got " << std::abs((a) - (b)) << ") at line "     \
                << __LINE__ << "\n";                                   \
    }                                                                  \
  } while (0)

// Test functions
void TestSingleTileCoversEntireDomain() {
  std::cout << "  Testing single tile covers entire domain... ";

  TileSchedulerConfig config;
  config.ns = 50;           // 50 surfaces total
  config.tile_size = 100;   // Larger than ns
  config.ns_min = 1;        // Starting from surface 1
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  EXPECT_EQ(scheduler.NumTiles(), 1);
  EXPECT_TRUE(scheduler.ValidateCoverage());

  const auto& tile = scheduler.GetTile(0);
  EXPECT_EQ(tile.start_surface, 1);
  // end_surface is exclusive: ns_min + ns = 1 + 50 = 51
  EXPECT_EQ(tile.end_surface, 51);
  EXPECT_EQ(tile.overlap_before, 0);
  EXPECT_EQ(tile.overlap_after, 0);
  EXPECT_EQ(tile.num_output_surfaces(), 50);

  std::cout << "done\n";
}

void TestMultipleTilesNoOverlap() {
  std::cout << "  Testing multiple tiles no overlap... ";

  TileSchedulerConfig config;
  config.ns = 100;        // 100 surfaces total
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  EXPECT_EQ(scheduler.NumTiles(), 4);
  EXPECT_TRUE(scheduler.ValidateCoverage());

  // With ns=100, ns_min=1, surfaces are [1, 101) in half-open notation
  // Tiles: [1,26), [26,51), [51,76), [76,101)
  for (int i = 0; i < 4; ++i) {
    const auto& tile = scheduler.GetTile(i);
    EXPECT_EQ(tile.start_surface, 1 + i * 25);
    EXPECT_EQ(tile.end_surface, 1 + (i + 1) * 25);
    EXPECT_EQ(tile.overlap_before, 0);
    EXPECT_EQ(tile.overlap_after, 0);
    EXPECT_EQ(tile.num_output_surfaces(), 25);
  }

  std::cout << "done\n";
}

void TestForwardStencilOverlap() {
  std::cout << "  Testing forward stencil overlap... ";

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

  std::cout << "done\n";
}

void TestBackwardStencilOverlap() {
  std::cout << "  Testing backward stencil overlap... ";

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

  std::cout << "done\n";
}

void TestBothStencilOverlap() {
  std::cout << "  Testing both stencil overlap... ";

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

  std::cout << "done\n";
}

void TestUnevenTileSizes() {
  std::cout << "  Testing uneven tile sizes... ";

  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 30;  // Doesn't divide evenly
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  // With ns=100, ns_min=1: surfaces are [1, 101)
  // Tiles: [1,31), [31,61), [61,91), [91,101)
  EXPECT_EQ(scheduler.NumTiles(), 4);
  EXPECT_TRUE(scheduler.ValidateCoverage());

  // Last tile should have fewer surfaces
  const auto& last_tile = scheduler.GetTile(3);
  EXPECT_EQ(last_tile.start_surface, 91);
  EXPECT_EQ(last_tile.end_surface, 101);  // exclusive end
  EXPECT_EQ(last_tile.num_output_surfaces(), 10);  // surfaces 91-100

  std::cout << "done\n";
}

void TestGetTileForSurface() {
  std::cout << "  Testing GetTileFor surface... ";

  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  // Tiles: [1,26), [26,51), [51,76), [76,101)
  EXPECT_EQ(scheduler.GetTileFor(1).tile_index, 0);
  EXPECT_EQ(scheduler.GetTileFor(25).tile_index, 0);   // last in tile 0
  EXPECT_EQ(scheduler.GetTileFor(26).tile_index, 1);   // first in tile 1
  EXPECT_EQ(scheduler.GetTileFor(100).tile_index, 3);  // last surface

  std::cout << "done\n";
}

void TestTileIterator() {
  std::cout << "  Testing tile iterator... ";

  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 25;
  config.ns_min = 1;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);
  TileIterator iter(scheduler, 2);

  int count = 0;
  while (!iter.Done()) {
    EXPECT_EQ(iter.CurrentTile().tile_index, count);
    EXPECT_EQ(iter.CurrentBufferIndex(), count % 2);
    iter.Next();
    ++count;
  }

  EXPECT_EQ(count, 4);

  std::cout << "done\n";
}

void TestTiledSummation() {
  std::cout << "  Testing tiled summation matches non-tiled... ";

  const int ns = 100;
  const int tile_size = 25;
  const int n_zeta = 32;
  const int n_theta = 32;
  const size_t grid_size = ns * n_zeta * n_theta;

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<double> input(grid_size);
  for (auto& x : input) {
    x = dist(rng);
  }

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
    // Convert surface numbers (1-based) to array indices (0-based)
    // tile covers surfaces [start_surface, end_surface)
    // which corresponds to array indices [start_surface-1, end_surface-1)
    size_t start_idx = (tile.start_surface - config.ns_min) *
                       static_cast<size_t>(n_zeta * n_theta);
    size_t end_idx = (tile.end_surface - config.ns_min) *
                     static_cast<size_t>(n_zeta * n_theta);
    for (size_t i = start_idx; i < end_idx; ++i) {
      tiled_sum += input[i];
    }
  }

  EXPECT_NEAR(tiled_sum, non_tiled_sum, 1e-10);

  std::cout << "done\n";
}

void TestTiledStencilWithOverlap() {
  std::cout << "  Testing tiled stencil with overlap... ";

  const int ns = 100;
  const int tile_size = 25;
  const int n_points = 32;
  const int ns_min = 1;

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<double> input(ns * n_points);
  for (auto& x : input) {
    x = dist(rng);
  }

  std::vector<double> non_tiled_output(ns * n_points, 0.0);
  std::vector<double> tiled_output(ns * n_points, 0.0);

  // Non-tiled stencil: output[j] = (input[j] + input[j+1]) / 2
  // j here is 0-based array index
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
  config.ns_min = ns_min;
  config.op_type = TileOperationType::kForwardStencil;

  TileScheduler scheduler(config);

  for (const auto& tile : scheduler.GetTiles()) {
    // Convert from 1-based surface numbers to 0-based array indices
    int start = tile.start_surface - ns_min;  // 0-based start
    int end = tile.end_surface - ns_min;      // 0-based exclusive end

    for (int j = start; j < end && j < ns - 1; ++j) {
      for (int k = 0; k < n_points; ++k) {
        int idx = j * n_points + k;
        int idx_next = (j + 1) * n_points + k;
        tiled_output[idx] = (input[idx] + input[idx_next]) / 2.0;
      }
    }
  }

  // Compare outputs
  double max_diff = 0.0;
  for (size_t i = 0; i < tiled_output.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(tiled_output[i] - non_tiled_output[i]));
  }
  EXPECT_NEAR(max_diff, 0.0, 1e-14);

  std::cout << "done\n";
}

void TestTiledBackwardStencil() {
  std::cout << "  Testing tiled backward stencil... ";

  const int ns = 100;
  const int tile_size = 25;
  const int n_points = 32;
  const int ns_min = 1;

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<double> input(ns * n_points);
  for (auto& x : input) {
    x = dist(rng);
  }

  std::vector<double> non_tiled_output(ns * n_points, 0.0);
  std::vector<double> tiled_output(ns * n_points, 0.0);

  // Backward stencil: output[j] = input[j] - input[j-1] (for j > 0)
  // j here is 0-based array index
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
  config.ns_min = ns_min;
  config.op_type = TileOperationType::kBackwardStencil;

  TileScheduler scheduler(config);

  for (const auto& tile : scheduler.GetTiles()) {
    // Convert from 1-based surface numbers to 0-based array indices
    int start = tile.start_surface - ns_min;
    int end = tile.end_surface - ns_min;

    for (int j = start; j < end; ++j) {
      if (j == 0) continue;  // Skip first surface (no predecessor)

      for (int k = 0; k < n_points; ++k) {
        int idx = j * n_points + k;
        int idx_prev = (j - 1) * n_points + k;
        tiled_output[idx] = input[idx] - input[idx_prev];
      }
    }
  }

  double max_diff = 0.0;
  for (size_t i = 0; i < tiled_output.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(tiled_output[i] - non_tiled_output[i]));
  }
  EXPECT_NEAR(max_diff, 0.0, 1e-14);

  std::cout << "done\n";
}

void TestCoverageStressTest() {
  std::cout << "  Testing coverage stress test... ";

  std::vector<int> ns_values = {10, 50, 100, 127, 256};
  std::vector<int> tile_sizes = {1, 5, 10, 25, 33, 64, 100};
  std::vector<TileOperationType> op_types = {
      TileOperationType::kNoOverlap,
      TileOperationType::kForwardStencil,
      TileOperationType::kBackwardStencil,
      TileOperationType::kBothStencil,
  };

  int configs_tested = 0;

  for (int ns : ns_values) {
    for (int tile_size : tile_sizes) {
      for (auto op_type : op_types) {
        TileSchedulerConfig config;
        config.ns = ns;
        config.tile_size = tile_size;
        config.ns_min = 1;
        config.op_type = op_type;

        TileScheduler scheduler(config);

        EXPECT_TRUE(scheduler.ValidateCoverage());

        // Verify all surfaces are covered exactly once
        std::vector<int> coverage(ns, 0);
        for (const auto& tile : scheduler.GetTiles()) {
          for (int s = tile.start_surface; s < tile.end_surface; ++s) {
            if (s >= 1 && s <= ns) {
              coverage[s - 1]++;
            }
          }
        }

        for (int s = 0; s < ns; ++s) {
          EXPECT_EQ(coverage[s], 1);
        }

        configs_tested++;
      }
    }
  }

  std::cout << configs_tested << " configurations tested, done\n";
}

void TestTiledReduction() {
  std::cout << "  Testing tiled reduction... ";

  const int ns = 100;
  const int tile_size = 25;
  const int n_points = 32;
  const int ns_min = 1;

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::vector<double> input(ns * n_points);
  for (auto& x : input) {
    x = dist(rng);
  }

  double global_min = *std::min_element(input.begin(), input.end());
  double global_max = *std::max_element(input.begin(), input.end());

  // Tiled reduction
  TileSchedulerConfig config;
  config.ns = ns;
  config.tile_size = tile_size;
  config.ns_min = ns_min;
  config.op_type = TileOperationType::kNoOverlap;

  TileScheduler scheduler(config);

  double tiled_min = std::numeric_limits<double>::max();
  double tiled_max = std::numeric_limits<double>::lowest();

  for (const auto& tile : scheduler.GetTiles()) {
    // Convert from 1-based surface numbers to 0-based array indices
    size_t start_idx = static_cast<size_t>(tile.start_surface - ns_min) *
                       n_points;
    size_t end_idx = static_cast<size_t>(tile.end_surface - ns_min) * n_points;

    for (size_t i = start_idx; i < end_idx; ++i) {
      tiled_min = std::min(tiled_min, input[i]);
      tiled_max = std::max(tiled_max, input[i]);
    }
  }

  EXPECT_NEAR(tiled_min, global_min, 1e-15);
  EXPECT_NEAR(tiled_max, global_max, 1e-15);

  std::cout << "done\n";
}

void TestTiledCarryState() {
  std::cout << "  Testing tiled carry state... ";

  const int ns = 100;
  const int tile_size = 25;

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  std::vector<double> input(ns);
  for (auto& x : input) {
    x = dist(rng);
  }

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

  double carry = 0.0;
  for (const auto& tile : scheduler.GetTiles()) {
    int start = tile.start_surface - 1;
    int end = tile.end_surface - 1;

    for (int j = start; j < end; ++j) {
      carry += input[j];
      tiled_output[j] = carry;
    }
  }

  double max_diff = 0.0;
  for (int i = 0; i < ns; ++i) {
    max_diff = std::max(max_diff, std::abs(tiled_output[i] - non_tiled_output[i]));
  }
  EXPECT_NEAR(max_diff, 0.0, 1e-14);

  std::cout << "done\n";
}

void TestTileReport() {
  std::cout << "  Testing tile report generation... ";

  TileSchedulerConfig config;
  config.ns = 100;
  config.tile_size = 30;
  config.ns_min = 1;
  config.op_type = TileOperationType::kForwardStencil;

  TileScheduler scheduler(config);
  std::string report = scheduler.GetTileReport();

  EXPECT_TRUE(report.find("Total surfaces: 100") != std::string::npos);
  EXPECT_TRUE(report.find("Tile size: 30") != std::string::npos);
  EXPECT_TRUE(report.find("PASSED") != std::string::npos);

  std::cout << "done\n";
}

}  // namespace vmecpp

int main() {
  std::cout << "==============================================\n";
  std::cout << "  VMEC++ Tiled GPU Backend Validation Tests\n";
  std::cout << "==============================================\n\n";

  std::cout << "Running TileScheduler tests:\n";
  vmecpp::TestSingleTileCoversEntireDomain();
  vmecpp::TestMultipleTilesNoOverlap();
  vmecpp::TestForwardStencilOverlap();
  vmecpp::TestBackwardStencilOverlap();
  vmecpp::TestBothStencilOverlap();
  vmecpp::TestUnevenTileSizes();
  vmecpp::TestGetTileForSurface();
  vmecpp::TestTileIterator();
  vmecpp::TestTileReport();

  std::cout << "\nRunning numerical validation tests:\n";
  vmecpp::TestTiledSummation();
  vmecpp::TestTiledStencilWithOverlap();
  vmecpp::TestTiledBackwardStencil();
  vmecpp::TestCoverageStressTest();
  vmecpp::TestTiledReduction();
  vmecpp::TestTiledCarryState();

  std::cout << "\n==============================================\n";
  std::cout << "  Results: " << vmecpp::tests_passed << "/" << vmecpp::tests_run
            << " tests passed";
  if (vmecpp::tests_failed > 0) {
    std::cout << " (" << vmecpp::tests_failed << " FAILED)";
  }
  std::cout << "\n==============================================\n";

  return vmecpp::tests_failed > 0 ? 1 : 0;
}

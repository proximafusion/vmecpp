// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#ifndef VMECPP_COMMON_COMPUTE_BACKEND_CUDA_TILE_SCHEDULER_H_
#define VMECPP_COMMON_COMPUTE_BACKEND_CUDA_TILE_SCHEDULER_H_

#include <string>
#include <vector>

namespace vmecpp {

// Represents a single radial tile with overlap information.
struct RadialTile {
  int tile_index;       // Index of this tile (0, 1, 2, ...)
  int start_surface;    // First surface in tile (inclusive)
  int end_surface;      // Last surface in tile (exclusive)
  int overlap_before;   // Extra surfaces needed before start
  int overlap_after;    // Extra surfaces needed after end

  // Input range including overlap (what we need to read)
  int input_start() const { return start_surface - overlap_before; }
  int input_end() const { return end_surface + overlap_after; }

  // Number of surfaces this tile computes output for
  int num_output_surfaces() const { return end_surface - start_surface; }

  // Number of input surfaces including overlap
  int num_input_surfaces() const { return input_end() - input_start(); }

  // Check if this is the first/last tile
  bool is_first() const { return overlap_before == 0; }
  bool is_last() const { return overlap_after == 0; }
};

// Operation types that determine overlap requirements.
enum class TileOperationType {
  kNoOverlap,       // FourierToReal, ForcesToFourier - no radial coupling
  kForwardStencil,  // ComputeJacobian, ComputeMetricElements - needs jF+1
  kBackwardStencil, // ComputeMHDForces - needs jH-1
  kBothStencil,     // Operations needing both directions
};

// Configuration for tile scheduling.
struct TileSchedulerConfig {
  int ns;                   // Total number of radial surfaces
  int tile_size;            // Target number of surfaces per tile
  int ns_min;               // Minimum surface index (usually 1)
  TileOperationType op_type; // Operation type for overlap calculation
};

// Schedules radial domain into tiles with appropriate overlap.
class TileScheduler {
 public:
  // Create scheduler with given configuration.
  explicit TileScheduler(const TileSchedulerConfig& config);

  // Get all tiles for the radial domain.
  const std::vector<RadialTile>& GetTiles() const { return tiles_; }

  // Get specific tile by index.
  const RadialTile& GetTile(int index) const { return tiles_[index]; }

  // Get tile containing a specific surface.
  const RadialTile& GetTileFor(int surface) const;

  // Number of tiles.
  int NumTiles() const { return static_cast<int>(tiles_.size()); }

  // Total surfaces across all tiles (for validation).
  int TotalInputSurfaces() const;

  // Configuration accessors.
  int TileSize() const { return config_.tile_size; }
  int TotalSurfaces() const { return config_.ns; }
  TileOperationType OperationType() const { return config_.op_type; }

  // Get human-readable tile report.
  std::string GetTileReport() const;

  // Validate tile coverage - ensure all surfaces are covered exactly once.
  bool ValidateCoverage() const;

  // Get overlap requirement for operation type.
  static int GetOverlapRequirement(TileOperationType op_type);

 private:
  void GenerateTiles();

  TileSchedulerConfig config_;
  std::vector<RadialTile> tiles_;
};

// Manages iteration over tiles with optional double-buffering.
class TileIterator {
 public:
  TileIterator(const TileScheduler& scheduler, int num_buffers = 2);

  // Get current tile.
  const RadialTile& CurrentTile() const;

  // Get buffer index for current tile (for double-buffering).
  int CurrentBufferIndex() const;

  // Move to next tile.
  void Next();

  // Check if iteration is complete.
  bool Done() const;

  // Reset to beginning.
  void Reset();

  // Get current tile index.
  int CurrentIndex() const { return current_index_; }

 private:
  const TileScheduler& scheduler_;
  int num_buffers_;
  int current_index_;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_COMPUTE_BACKEND_CUDA_TILE_SCHEDULER_H_

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/common/compute_backend/cuda/tile_scheduler.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace vmecpp {

int TileScheduler::GetOverlapRequirement(TileOperationType op_type) {
  switch (op_type) {
    case TileOperationType::kNoOverlap:
      return 0;
    case TileOperationType::kForwardStencil:
      return 1;  // Need surface j+1
    case TileOperationType::kBackwardStencil:
      return 1;  // Need surface j-1
    case TileOperationType::kBothStencil:
      return 1;  // Need both directions
    default:
      return 0;
  }
}

TileScheduler::TileScheduler(const TileSchedulerConfig& config)
    : config_(config) {
  if (config_.ns <= 0) {
    throw std::invalid_argument("TileScheduler: ns must be positive");
  }
  if (config_.tile_size <= 0) {
    throw std::invalid_argument("TileScheduler: tile_size must be positive");
  }
  GenerateTiles();
}

void TileScheduler::GenerateTiles() {
  tiles_.clear();

  const int overlap = GetOverlapRequirement(config_.op_type);
  const int ns_min = config_.ns_min;
  // ns is the total count, so surfaces are [ns_min, ns_min + ns)
  // which with ns_min=1 gives surfaces 1..ns (inclusive)
  const int ns_max_exclusive = ns_min + config_.ns;

  int tile_index = 0;
  for (int start = ns_min; start < ns_max_exclusive;
       start += config_.tile_size) {
    RadialTile tile;
    tile.tile_index = tile_index++;
    tile.start_surface = start;
    tile.end_surface = std::min(start + config_.tile_size, ns_max_exclusive);

    // Determine overlap requirements based on operation type
    switch (config_.op_type) {
      case TileOperationType::kNoOverlap:
        tile.overlap_before = 0;
        tile.overlap_after = 0;
        break;

      case TileOperationType::kForwardStencil:
        // Need surface j+1, so last tile doesn't need overlap after
        tile.overlap_before = 0;
        tile.overlap_after =
            (tile.end_surface < ns_max_exclusive) ? overlap : 0;
        break;

      case TileOperationType::kBackwardStencil:
        // Need surface j-1, so first tile doesn't need overlap before
        tile.overlap_before = (start > ns_min) ? overlap : 0;
        tile.overlap_after = 0;
        break;

      case TileOperationType::kBothStencil:
        // Need both directions
        tile.overlap_before = (start > ns_min) ? overlap : 0;
        tile.overlap_after =
            (tile.end_surface < ns_max_exclusive) ? overlap : 0;
        break;
    }

    tiles_.push_back(tile);
  }
}

const RadialTile& TileScheduler::GetTileFor(int surface) const {
  for (const auto& tile : tiles_) {
    if (surface >= tile.start_surface && surface < tile.end_surface) {
      return tile;
    }
  }
  throw std::out_of_range("Surface " + std::to_string(surface) +
                          " not found in any tile");
}

int TileScheduler::TotalInputSurfaces() const {
  int total = 0;
  for (const auto& tile : tiles_) {
    total += tile.num_input_surfaces();
  }
  return total;
}

bool TileScheduler::ValidateCoverage() const {
  if (tiles_.empty()) {
    return config_.ns == 0;
  }

  // ns is the total count of surfaces, starting from ns_min
  // Surfaces are [ns_min, ns_min + ns) in half-open notation
  const int ns_max_exclusive = config_.ns_min + config_.ns;

  // Check that tiles cover all surfaces exactly once
  std::vector<int> coverage(config_.ns, 0);

  for (const auto& tile : tiles_) {
    for (int s = tile.start_surface; s < tile.end_surface; ++s) {
      if (s >= config_.ns_min && s < ns_max_exclusive) {
        coverage[s - config_.ns_min]++;
      }
    }
  }

  // Each surface should be covered exactly once
  for (int i = 0; i < static_cast<int>(coverage.size()); ++i) {
    if (coverage[i] != 1) {
      return false;
    }
  }

  // Check tile ordering and continuity
  for (size_t i = 1; i < tiles_.size(); ++i) {
    if (tiles_[i].start_surface != tiles_[i - 1].end_surface) {
      return false;  // Gap or overlap in output ranges
    }
  }

  return true;
}

std::string TileScheduler::GetTileReport() const {
  std::ostringstream ss;

  ss << "=== Tile Schedule Report ===\n";
  ss << "Total surfaces: " << config_.ns << "\n";
  ss << "Tile size: " << config_.tile_size << "\n";
  ss << "Operation type: ";
  switch (config_.op_type) {
    case TileOperationType::kNoOverlap:
      ss << "No overlap (DFT)\n";
      break;
    case TileOperationType::kForwardStencil:
      ss << "Forward stencil (Jacobian, Metric)\n";
      break;
    case TileOperationType::kBackwardStencil:
      ss << "Backward stencil (MHD Forces)\n";
      break;
    case TileOperationType::kBothStencil:
      ss << "Both directions\n";
      break;
  }
  ss << "Number of tiles: " << tiles_.size() << "\n";
  ss << "\n";

  ss << "Tiles:\n";
  for (const auto& tile : tiles_) {
    ss << "  Tile " << tile.tile_index << ": ";
    ss << "output [" << tile.start_surface << ", " << tile.end_surface << ")";
    ss << " (" << tile.num_output_surfaces() << " surfaces)";
    if (tile.overlap_before > 0 || tile.overlap_after > 0) {
      ss << ", input [" << tile.input_start() << ", " << tile.input_end() << ")";
      ss << " (overlap: -" << tile.overlap_before << "/+"
         << tile.overlap_after << ")";
    }
    ss << "\n";
  }

  ss << "\nCoverage validation: "
     << (ValidateCoverage() ? "PASSED" : "FAILED") << "\n";

  return ss.str();
}

// TileIterator implementation

TileIterator::TileIterator(const TileScheduler& scheduler, int num_buffers)
    : scheduler_(scheduler), num_buffers_(num_buffers), current_index_(0) {}

const RadialTile& TileIterator::CurrentTile() const {
  return scheduler_.GetTile(current_index_);
}

int TileIterator::CurrentBufferIndex() const {
  return current_index_ % num_buffers_;
}

void TileIterator::Next() {
  ++current_index_;
}

bool TileIterator::Done() const {
  return current_index_ >= scheduler_.NumTiles();
}

void TileIterator::Reset() {
  current_index_ = 0;
}

}  // namespace vmecpp

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#ifndef VMECPP_COMMON_COMPUTE_BACKEND_CUDA_TILE_MEMORY_BUDGET_H_
#define VMECPP_COMMON_COMPUTE_BACKEND_CUDA_TILE_MEMORY_BUDGET_H_

#include <cstddef>

namespace vmecpp {

// Memory budget information for GPU tiling decisions.
struct MemoryBudget {
  size_t total_gpu_memory;       // Total GPU memory in bytes
  size_t free_gpu_memory;        // Free GPU memory in bytes
  size_t reserved_memory;        // Memory reserved for CUDA runtime
  size_t available_memory;       // Memory available for tiling
  size_t per_surface_memory;     // Memory required per radial surface
  int max_surfaces_per_tile;     // Maximum surfaces that fit in available memory
  bool tiling_required;          // True if problem exceeds available memory
};

// Grid size parameters needed for memory calculations.
struct GridSizeParams {
  int ns;           // Number of radial surfaces
  int mpol;         // Number of poloidal modes
  int ntor;         // Number of toroidal modes
  int n_zeta;       // Toroidal grid points
  int n_theta_eff;  // Effective poloidal grid points
};

// Calculates memory requirements and tile sizing for GPU execution.
class TileMemoryBudget {
 public:
  // Query GPU memory and calculate budget for given problem size.
  // device_id: CUDA device to query
  // params: Grid size parameters
  // memory_fraction: Fraction of free memory to use (default 0.8)
  static MemoryBudget Calculate(int device_id, const GridSizeParams& params,
                                double memory_fraction = 0.8);

  // Calculate memory required for one radial surface (in bytes).
  // This includes all arrays needed for DFT and physics operations.
  static size_t PerSurfaceMemory(const GridSizeParams& params);

  // Calculate memory for Fourier coefficient arrays per surface.
  static size_t FourierMemoryPerSurface(int mpol, int ntor);

  // Calculate memory for real-space geometry arrays per surface.
  static size_t GeometryMemoryPerSurface(int n_zeta, int n_theta_eff);

  // Calculate memory for real-space force arrays per surface.
  static size_t ForceMemoryPerSurface(int n_zeta, int n_theta_eff);

  // Calculate memory for half-grid physics arrays per surface.
  static size_t PhysicsMemoryPerSurface(int n_zeta, int n_theta_eff);

  // Recommend optimal tile size for given problem and GPU.
  // Returns 0 if even a single surface doesn't fit.
  static int RecommendTileSize(int device_id, const GridSizeParams& params,
                               double memory_fraction = 0.8,
                               int min_overlap = 1);

  // Check if tiling is required for the given problem size.
  static bool TilingRequired(int device_id, const GridSizeParams& params,
                             double memory_fraction = 0.8);

  // Get human-readable memory report string.
  static std::string GetMemoryReport(const MemoryBudget& budget,
                                     const GridSizeParams& params);

 private:
  // Constants for array counts in each category
  static constexpr int kNumFourierArrays = 12;      // rmncc, rmnss, etc.
  static constexpr int kNumGeometryArrays = 18;     // r1_e, r1_o, ru_e, etc.
  static constexpr int kNumForceArrays = 20;        // armn_e, armn_o, etc.
  static constexpr int kNumPhysicsArrays = 13;      // tau, gsqrt, guu, etc.
  static constexpr size_t kReservedMemoryBytes = 256 * 1024 * 1024;  // 256 MB
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_COMPUTE_BACKEND_CUDA_TILE_MEMORY_BUDGET_H_

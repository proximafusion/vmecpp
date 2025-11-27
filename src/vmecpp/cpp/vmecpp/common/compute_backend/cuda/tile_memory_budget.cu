// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/common/compute_backend/cuda/tile_memory_budget.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace vmecpp {

size_t TileMemoryBudget::FourierMemoryPerSurface(int mpol, int ntor) {
  // Fourier coefficients: 12 arrays (6 input + 6 output)
  // Each array has mpol * (ntor + 1) elements per surface
  const size_t coeffs_per_surface = static_cast<size_t>(mpol) * (ntor + 1);
  return kNumFourierArrays * coeffs_per_surface * sizeof(double);
}

size_t TileMemoryBudget::GeometryMemoryPerSurface(int n_zeta, int n_theta_eff) {
  // Real-space geometry: 18 arrays
  // r1_e, r1_o, ru_e, ru_o, rv_e, rv_o (6 for R)
  // z1_e, z1_o, zu_e, zu_o, zv_e, zv_o (6 for Z)
  // lu_e, lu_o, lv_e, lv_o (4 for lambda)
  // r_con, z_con (2 for constraints)
  const size_t grid_points = static_cast<size_t>(n_zeta) * n_theta_eff;
  return kNumGeometryArrays * grid_points * sizeof(double);
}

size_t TileMemoryBudget::ForceMemoryPerSurface(int n_zeta, int n_theta_eff) {
  // Real-space forces: 20 arrays
  // armn_e/o, azmn_e/o, brmn_e/o, bzmn_e/o, crmn_e/o, czmn_e/o (12)
  // blmn_e/o, clmn_e/o (4)
  // frcon_e/o, fzcon_e/o (4)
  const size_t grid_points = static_cast<size_t>(n_zeta) * n_theta_eff;
  return kNumForceArrays * grid_points * sizeof(double);
}

size_t TileMemoryBudget::PhysicsMemoryPerSurface(int n_zeta, int n_theta_eff) {
  // Half-grid physics arrays: 13 arrays
  // tau, r12, ru12, zu12, rs, zs (6 from Jacobian)
  // gsqrt, guu, guv, gvv (4 from metric)
  // bsupu, bsupv (2 from B-field)
  // totalPressure (1)
  const size_t grid_points = static_cast<size_t>(n_zeta) * n_theta_eff;
  return kNumPhysicsArrays * grid_points * sizeof(double);
}

size_t TileMemoryBudget::PerSurfaceMemory(const GridSizeParams& params) {
  size_t total = 0;
  total += FourierMemoryPerSurface(params.mpol, params.ntor);
  total += GeometryMemoryPerSurface(params.n_zeta, params.n_theta_eff);
  total += ForceMemoryPerSurface(params.n_zeta, params.n_theta_eff);
  total += PhysicsMemoryPerSurface(params.n_zeta, params.n_theta_eff);
  return total;
}

MemoryBudget TileMemoryBudget::Calculate(int device_id,
                                         const GridSizeParams& params,
                                         double memory_fraction) {
  MemoryBudget budget;

  // Query GPU memory
  cudaError_t err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to set CUDA device: ") +
                             cudaGetErrorString(err));
  }

  err = cudaMemGetInfo(&budget.free_gpu_memory, &budget.total_gpu_memory);
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to query GPU memory: ") +
                             cudaGetErrorString(err));
  }

  budget.reserved_memory = kReservedMemoryBytes;
  budget.per_surface_memory = PerSurfaceMemory(params);

  // Calculate available memory
  if (budget.free_gpu_memory > budget.reserved_memory) {
    size_t usable = budget.free_gpu_memory - budget.reserved_memory;
    budget.available_memory =
        static_cast<size_t>(usable * memory_fraction);
  } else {
    budget.available_memory = 0;
  }

  // Calculate max surfaces per tile
  if (budget.per_surface_memory > 0) {
    budget.max_surfaces_per_tile =
        static_cast<int>(budget.available_memory / budget.per_surface_memory);
  } else {
    budget.max_surfaces_per_tile = params.ns;
  }

  // Determine if tiling is required
  budget.tiling_required = (budget.max_surfaces_per_tile < params.ns);

  return budget;
}

int TileMemoryBudget::RecommendTileSize(int device_id,
                                        const GridSizeParams& params,
                                        double memory_fraction,
                                        int min_overlap) {
  MemoryBudget budget = Calculate(device_id, params, memory_fraction);

  // Need at least min_overlap + 1 surfaces for useful work
  if (budget.max_surfaces_per_tile <= min_overlap) {
    return 0;  // Cannot tile - insufficient memory
  }

  if (!budget.tiling_required) {
    return params.ns;  // No tiling needed - return full size
  }

  // Reserve space for overlap regions
  // We need overlap on both sides potentially, so reserve 2 * min_overlap
  int effective_tile_size = budget.max_surfaces_per_tile - 2 * min_overlap;

  // Ensure positive tile size
  return std::max(1, effective_tile_size);
}

bool TileMemoryBudget::TilingRequired(int device_id,
                                      const GridSizeParams& params,
                                      double memory_fraction) {
  MemoryBudget budget = Calculate(device_id, params, memory_fraction);
  return budget.tiling_required;
}

std::string TileMemoryBudget::GetMemoryReport(const MemoryBudget& budget,
                                              const GridSizeParams& params) {
  std::ostringstream ss;

  auto format_size = [](size_t bytes) -> std::string {
    if (bytes >= 1024 * 1024 * 1024) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
      return buf;
    } else if (bytes >= 1024 * 1024) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.2f MB", bytes / (1024.0 * 1024.0));
      return buf;
    } else if (bytes >= 1024) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%.2f KB", bytes / 1024.0);
      return buf;
    } else {
      return std::to_string(bytes) + " B";
    }
  };

  ss << "=== GPU Memory Budget Report ===\n";
  ss << "Problem size:\n";
  ss << "  Radial surfaces (ns): " << params.ns << "\n";
  ss << "  Poloidal modes (mpol): " << params.mpol << "\n";
  ss << "  Toroidal modes (ntor): " << params.ntor << "\n";
  ss << "  Grid size: " << params.n_zeta << " x " << params.n_theta_eff << "\n";
  ss << "\n";
  ss << "GPU memory:\n";
  ss << "  Total: " << format_size(budget.total_gpu_memory) << "\n";
  ss << "  Free: " << format_size(budget.free_gpu_memory) << "\n";
  ss << "  Reserved: " << format_size(budget.reserved_memory) << "\n";
  ss << "  Available for tiling: " << format_size(budget.available_memory)
     << "\n";
  ss << "\n";
  ss << "Memory per surface:\n";
  ss << "  Fourier coefficients: "
     << format_size(FourierMemoryPerSurface(params.mpol, params.ntor)) << "\n";
  ss << "  Real-space geometry: "
     << format_size(GeometryMemoryPerSurface(params.n_zeta, params.n_theta_eff))
     << "\n";
  ss << "  Real-space forces: "
     << format_size(ForceMemoryPerSurface(params.n_zeta, params.n_theta_eff))
     << "\n";
  ss << "  Physics (half-grid): "
     << format_size(PhysicsMemoryPerSurface(params.n_zeta, params.n_theta_eff))
     << "\n";
  ss << "  Total per surface: " << format_size(budget.per_surface_memory)
     << "\n";
  ss << "\n";
  ss << "Tiling:\n";
  ss << "  Max surfaces per tile: " << budget.max_surfaces_per_tile << "\n";
  ss << "  Tiling required: " << (budget.tiling_required ? "YES" : "NO")
     << "\n";

  if (budget.tiling_required) {
    int num_tiles =
        (params.ns + budget.max_surfaces_per_tile - 1) /
        budget.max_surfaces_per_tile;
    ss << "  Estimated tiles needed: " << num_tiles << "\n";
  }

  return ss.str();
}

}  // namespace vmecpp

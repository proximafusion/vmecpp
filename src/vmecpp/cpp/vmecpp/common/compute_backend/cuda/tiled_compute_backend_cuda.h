// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#ifndef VMECPP_COMMON_COMPUTE_BACKEND_CUDA_TILED_COMPUTE_BACKEND_CUDA_H_
#define VMECPP_COMMON_COMPUTE_BACKEND_CUDA_TILED_COMPUTE_BACKEND_CUDA_H_

#include <memory>
#include <string>

#include "vmecpp/common/compute_backend/compute_backend.h"
#include "vmecpp/common/compute_backend/cuda/tile_memory_budget.h"
#include "vmecpp/common/compute_backend/cuda/tile_scheduler.h"

namespace vmecpp {

// Configuration for tiled CUDA backend.
struct TiledBackendConfig {
  int device_id = 0;              // CUDA device to use
  double memory_fraction = 0.8;   // Fraction of GPU memory to use
  int num_streams = 2;            // Number of streams for pipelining
  int min_tile_size = 4;          // Minimum surfaces per tile
  bool force_tiling = false;      // Force tiling even if not needed
  bool enable_validation = false; // Enable numerical validation
  double validation_tol = 1e-10;  // Tolerance for validation
};

// Error codes for tiled backend operations.
enum class TiledBackendError {
  kSuccess = 0,
  kInsufficientMemory,    // Even minimum tile doesn't fit
  kTileSizeTooSmall,      // Tile smaller than overlap requirement
  kDeviceNotAvailable,    // CUDA device not found
  kValidationFailed,      // Numerical validation failed
  kInternalError,         // Unexpected internal error
};

// Validation result for comparing tiled vs non-tiled execution.
struct ValidationResult {
  bool passed = true;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  std::string error_location;
  std::string details;
};

// Forward declaration
class TiledComputeBackendCudaImpl;

// GPU compute backend with tiled execution for large problems.
// Automatically tiles the radial domain when problem size exceeds GPU memory.
class TiledComputeBackendCuda : public ComputeBackend {
 public:
  explicit TiledComputeBackendCuda(const TiledBackendConfig& config);
  ~TiledComputeBackendCuda() override;

  // Non-copyable
  TiledComputeBackendCuda(const TiledComputeBackendCuda&) = delete;
  TiledComputeBackendCuda& operator=(const TiledComputeBackendCuda&) = delete;

  // ComputeBackend interface
  void FourierToReal(const FourierGeometry& physical_x,
                     const std::vector<double>& xmpq,
                     const RadialPartitioning& rp, const Sizes& s,
                     const RadialProfiles& profiles,
                     const FourierBasisFastPoloidal& fb,
                     RealSpaceGeometry& m_geometry) override;

  void ForcesToFourier(const RealSpaceForces& forces,
                       const std::vector<double>& xmpq,
                       const RadialPartitioning& rp,
                       const ForceCoefficients& fc, const Sizes& s,
                       const FourierBasisFastPoloidal& fb,
                       const VacuumState vacuum_state,
                       ForceFourierCoefficients& m_forces) override;

  bool ComputeJacobian(const JacobianInput& input,
                       const RadialPartitioning& rp, const Sizes& s,
                       JacobianOutput& m_output) override;

  void ComputeMetricElements(const MetricInput& input,
                             const RadialPartitioning& rp, const Sizes& s,
                             MetricOutput& m_output) override;

  void ComputeBContra(const BContraInput& input, const RadialPartitioning& rp,
                      const Sizes& s, BContraOutput& m_output) override;

  void ComputeMHDForces(const MHDForcesInput& input,
                        const RadialPartitioning& rp, const Sizes& s,
                        MHDForcesOutput& m_output) override;

  // Tiling-specific methods

  // Check if tiling is required for given problem size.
  bool IsTilingRequired(int ns, int mpol, int ntor, int n_zeta,
                        int n_theta_eff) const;

  // Get memory budget for current configuration.
  MemoryBudget GetMemoryBudget(int ns, int mpol, int ntor, int n_zeta,
                               int n_theta_eff) const;

  // Get human-readable status report.
  std::string GetStatusReport() const;

  // Get last error (if any).
  TiledBackendError GetLastError() const { return last_error_; }

  // Get last validation result (if validation enabled).
  const ValidationResult& GetLastValidationResult() const {
    return last_validation_;
  }

  // Enable/disable validation at runtime.
  void SetValidationEnabled(bool enabled);

  // Configure tile size (0 = auto).
  void SetTileSize(int tile_size);

  // Get current tile size.
  int GetTileSize() const;

  // Get number of tiles for current problem.
  int GetNumTiles() const;

 private:
  std::unique_ptr<TiledComputeBackendCudaImpl> impl_;
  TiledBackendConfig config_;
  TiledBackendError last_error_ = TiledBackendError::kSuccess;
  ValidationResult last_validation_;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_COMPUTE_BACKEND_CUDA_TILED_COMPUTE_BACKEND_CUDA_H_

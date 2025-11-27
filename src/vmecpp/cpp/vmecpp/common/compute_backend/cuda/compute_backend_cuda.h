// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_COMPUTE_BACKEND_CUDA_COMPUTE_BACKEND_CUDA_H_
#define VMECPP_COMMON_COMPUTE_BACKEND_CUDA_COMPUTE_BACKEND_CUDA_H_

#include <memory>
#include <vector>

#include "vmecpp/common/compute_backend/compute_backend.h"

namespace vmecpp {

// Forward declaration for PIMPL pattern to hide CUDA details from headers.
class ComputeBackendCudaImpl;

// CUDA implementation of the compute backend interface.
//
// This implementation accelerates the DFT operations using NVIDIA GPUs.
// The primary hotspots (FourierToReal and ForcesToFourier) are implemented
// as CUDA kernels that exploit the parallel structure of the nested loops
// over radial surfaces, poloidal modes, and toroidal grid points.
//
// Memory management:
// - Device memory is allocated once and reused across iterations
// - Data transfers are minimized by keeping intermediate results on device
// - Async transfers can overlap with computation when multiple streams are used
//
// Thread mapping:
// - Each CUDA block handles one (jF, m) pair
// - Threads within a block handle different k (toroidal) grid points
// - Inner reductions over l (poloidal) and n (toroidal modes) use shared memory
class ComputeBackendCuda : public ComputeBackend {
 public:
  // Creates a CUDA backend using the specified GPU device.
  //
  // Parameters:
  //   device_id: CUDA device ID (0-indexed)
  //   num_streams: Number of CUDA streams for async operations
  explicit ComputeBackendCuda(int device_id = 0, int num_streams = 2);

  ~ComputeBackendCuda() override;

  // Non-copyable, movable.
  ComputeBackendCuda(const ComputeBackendCuda&) = delete;
  ComputeBackendCuda& operator=(const ComputeBackendCuda&) = delete;
  ComputeBackendCuda(ComputeBackendCuda&&) noexcept;
  ComputeBackendCuda& operator=(ComputeBackendCuda&&) noexcept;

  BackendType GetType() const override { return BackendType::kCuda; }

  std::string GetName() const override;

  void FourierToReal(const FourierGeometry& physical_x,
                     const std::vector<double>& xmpq,
                     const RadialPartitioning& rp, const Sizes& s,
                     const RadialProfiles& profiles,
                     const FourierBasisFastPoloidal& fb,
                     RealSpaceGeometry& m_geometry) override;

  void ForcesToFourier(const RealSpaceForces& forces,
                       const std::vector<double>& xmpq,
                       const RadialPartitioning& rp, const FlowControl& fc,
                       const Sizes& s, const FourierBasisFastPoloidal& fb,
                       VacuumPressureState vacuum_pressure_state,
                       FourierForces& m_physical_forces) override;

  void Synchronize() override;

  bool IsAvailable() const override;

  // Returns the CUDA device ID this backend is using.
  int GetDeviceId() const;

  // Returns the name of the CUDA device.
  std::string GetDeviceName() const;

  // Returns the compute capability of the CUDA device (e.g., "8.6").
  std::string GetComputeCapability() const;

  // Returns the amount of device memory available in bytes.
  size_t GetDeviceMemoryBytes() const;

 private:
  std::unique_ptr<ComputeBackendCudaImpl> impl_;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_COMPUTE_BACKEND_CUDA_COMPUTE_BACKEND_CUDA_H_

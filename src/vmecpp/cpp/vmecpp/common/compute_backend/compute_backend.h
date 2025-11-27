// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_H_
#define VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_H_

#include <memory>
#include <string>

#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {

// Enumeration of available compute backend types.
enum class BackendType {
  kCpu,   // CPU-based computation (default)
  kCuda,  // NVIDIA CUDA GPU acceleration
};

// Configuration options for compute backends.
struct BackendConfig {
  // The type of backend to use.
  BackendType type = BackendType::kCpu;

  // For CUDA backend: which GPU device to use (0-indexed).
  int cuda_device_id = 0;

  // For CUDA backend: number of CUDA streams for async operations.
  int cuda_num_streams = 2;

  // Enable verbose logging of backend operations.
  bool verbose = false;
};

// Abstract interface for compute backends.
//
// This interface abstracts the computational kernels that are candidates for
// GPU acceleration. The primary operations are the forward and inverse DFT
// (Discrete Fourier Transform) operations that dominate the computational
// cost of VMEC iterations.
//
// Implementations:
// - ComputeBackendCpu: Reference implementation using existing CPU code
// - ComputeBackendCuda: GPU-accelerated implementation using CUDA
class ComputeBackend {
 public:
  virtual ~ComputeBackend() = default;

  // Returns the backend type.
  virtual BackendType GetType() const = 0;

  // Returns a human-readable name for this backend.
  virtual std::string GetName() const = 0;

  // Performs inverse DFT: Fourier coefficients -> real-space geometry.
  //
  // This transforms the Fourier coefficients of the flux surface geometry
  // (R, Z, lambda) into their real-space representation on the computational
  // grid. This is one of the two main computational hotspots.
  //
  // Parameters:
  //   physical_x: Input Fourier coefficients of geometry
  //   xmpq: Spectral condensation factors
  //   rp: Radial partitioning information
  //   s: Grid sizes
  //   profiles: Radial profiles (pressure, iota, etc.)
  //   fb: Pre-computed Fourier basis functions
  //   m_geometry: Output real-space geometry (modified in place)
  virtual void FourierToReal(const FourierGeometry& physical_x,
                             const std::vector<double>& xmpq,
                             const RadialPartitioning& rp, const Sizes& s,
                             const RadialProfiles& profiles,
                             const FourierBasisFastPoloidal& fb,
                             RealSpaceGeometry& m_geometry) = 0;

  // Performs forward DFT: real-space forces -> Fourier coefficients.
  //
  // This transforms the real-space MHD forces into their Fourier coefficient
  // representation. This is one of the two main computational hotspots.
  //
  // Parameters:
  //   forces: Input real-space force arrays
  //   xmpq: Spectral condensation factors
  //   rp: Radial partitioning information
  //   fc: Flow control parameters
  //   s: Grid sizes
  //   fb: Pre-computed Fourier basis functions
  //   vacuum_pressure_state: Vacuum pressure constraint state
  //   m_physical_forces: Output Fourier coefficients (modified in place)
  virtual void ForcesToFourier(const RealSpaceForces& forces,
                               const std::vector<double>& xmpq,
                               const RadialPartitioning& rp,
                               const FlowControl& fc, const Sizes& s,
                               const FourierBasisFastPoloidal& fb,
                               VacuumPressureState vacuum_pressure_state,
                               FourierForces& m_physical_forces) = 0;

  // Synchronizes any pending asynchronous operations.
  //
  // For CPU backend, this is a no-op. For GPU backends, this ensures all
  // kernel launches and memory transfers have completed.
  virtual void Synchronize() = 0;

  // Returns true if this backend is available and functional.
  //
  // For CPU backend, always returns true. For CUDA backend, returns true
  // only if a compatible GPU is detected and CUDA runtime is initialized.
  virtual bool IsAvailable() const = 0;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_H_

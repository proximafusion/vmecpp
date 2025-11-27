// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_H_
#define VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_H_

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {

// Input data for Jacobian computation.
struct JacobianInput {
  std::span<const double> r1_e;
  std::span<const double> r1_o;
  std::span<const double> z1_e;
  std::span<const double> z1_o;
  std::span<const double> ru_e;
  std::span<const double> ru_o;
  std::span<const double> zu_e;
  std::span<const double> zu_o;
  std::span<const double> sqrtSH;
  double deltaS;
};

// Output data from Jacobian computation.
struct JacobianOutput {
  std::span<double> tau;
  std::span<double> r12;
  std::span<double> ru12;
  std::span<double> zu12;
  std::span<double> rs;
  std::span<double> zs;
};

// Input data for metric elements computation.
struct MetricInput {
  std::span<const double> r1_e;
  std::span<const double> r1_o;
  std::span<const double> z1_e;
  std::span<const double> z1_o;
  std::span<const double> ru_e;
  std::span<const double> ru_o;
  std::span<const double> zu_e;
  std::span<const double> zu_o;
  std::span<const double> rv_e;
  std::span<const double> rv_o;
  std::span<const double> zv_e;
  std::span<const double> zv_o;
  std::span<const double> tau;
  std::span<const double> r12;
  std::span<const double> sqrtSF;
  std::span<const double> sqrtSH;
  bool lthreed;
};

// Output data from metric elements computation.
struct MetricOutput {
  std::span<double> gsqrt;
  std::span<double> guu;
  std::span<double> guv;
  std::span<double> gvv;
};

// Input data for contravariant B field computation.
struct BContraInput {
  std::span<double> lu_e;  // Modified in place (unnormalized)
  std::span<double> lu_o;
  std::span<double> lv_e;
  std::span<double> lv_o;
  std::span<const double> gsqrt;
  std::span<const double> guu;
  std::span<const double> guv;
  std::span<const double> gvv;
  std::span<const double> phipF;
  std::span<const double> phipH;
  std::span<const double> currH;  // Used when ncurr==1
  std::span<const double> iotaH_in;  // Used when ncurr!=1
  std::span<const double> sqrtSH;
  std::span<const double> wInt;
  double lamscale;
  int ncurr;
  bool lthreed;
};

// Output data from contravariant B field computation.
struct BContraOutput {
  std::span<double> bsupu;
  std::span<double> bsupv;
  std::span<double> chipH;
  std::span<double> chipF;
  std::span<double> iotaH;
  std::span<double> iotaF;
};

// Input data for MHD forces computation.
struct MHDForcesInput {
  std::span<const double> r1_e;
  std::span<const double> r1_o;
  std::span<const double> z1_e;
  std::span<const double> z1_o;
  std::span<const double> ru_e;
  std::span<const double> ru_o;
  std::span<const double> zu_e;
  std::span<const double> zu_o;
  std::span<const double> rv_e;
  std::span<const double> rv_o;
  std::span<const double> zv_e;
  std::span<const double> zv_o;
  std::span<const double> r12;
  std::span<const double> ru12;
  std::span<const double> zu12;
  std::span<const double> rs;
  std::span<const double> zs;
  std::span<const double> tau;
  std::span<const double> gsqrt;
  std::span<const double> bsupu;
  std::span<const double> bsupv;
  std::span<const double> totalPressure;
  std::span<const double> sqrtSF;
  std::span<const double> sqrtSH;
  double deltaS;
  bool lfreeb;
  bool lthreed;
  int ns;
};

// Output data from MHD forces computation.
struct MHDForcesOutput {
  std::span<double> armn_e;
  std::span<double> armn_o;
  std::span<double> azmn_e;
  std::span<double> azmn_o;
  std::span<double> brmn_e;
  std::span<double> brmn_o;
  std::span<double> bzmn_e;
  std::span<double> bzmn_o;
  std::span<double> crmn_e;
  std::span<double> crmn_o;
  std::span<double> czmn_e;
  std::span<double> czmn_o;
};

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

  // =========================================================================
  // Additional compute-heavy operations (optional GPU acceleration)
  // =========================================================================

  // Computes the Jacobian sqrt(g)/R (tau) and related half-grid quantities.
  //
  // This computes the coordinate transformation Jacobian from the geometry
  // derivatives. It also detects bad Jacobians (sign changes) that require
  // iteration restarts.
  //
  // Parameters:
  //   input: Geometry data on full-grid (r1, z1, ru, zu)
  //   rp: Radial partitioning information
  //   s: Grid sizes
  //   m_output: Output half-grid quantities (tau, r12, ru12, zu12, rs, zs)
  //
  // Returns: true if a bad Jacobian (sign change in tau) was detected.
  virtual bool ComputeJacobian(const JacobianInput& input,
                               const RadialPartitioning& rp, const Sizes& s,
                               JacobianOutput& m_output) = 0;

  // Computes the metric tensor elements g_uu, g_uv, g_vv and Jacobian gsqrt.
  //
  // The metric tensor components are computed from the geometry derivatives
  // and averaged between adjacent radial surfaces.
  //
  // Parameters:
  //   input: Geometry data and tau from ComputeJacobian
  //   rp: Radial partitioning information
  //   s: Grid sizes
  //   m_output: Output metric elements (gsqrt, guu, guv, gvv)
  virtual void ComputeMetricElements(const MetricInput& input,
                                     const RadialPartitioning& rp,
                                     const Sizes& s, MetricOutput& m_output) = 0;

  // Computes the contravariant magnetic field components B^theta, B^zeta.
  //
  // This computes the magnetic field from the lambda field (covariant
  // potential) and applies either the iota constraint or the toroidal current
  // constraint depending on ncurr.
  //
  // Parameters:
  //   input: Lambda field, metric elements, profiles
  //   rp: Radial partitioning information
  //   s: Grid sizes
  //   m_output: Output magnetic field and updated profiles
  virtual void ComputeBContra(const BContraInput& input,
                              const RadialPartitioning& rp, const Sizes& s,
                              BContraOutput& m_output) = 0;

  // Computes the MHD (magnetohydrodynamic) forces in real space.
  //
  // This is a core computational routine that calculates pressure gradient
  // and magnetic pressure forces on the flux surfaces. The forces are split
  // into A, B, C contributions for both R and Z components.
  //
  // Parameters:
  //   input: Geometry, metric, magnetic field, pressure data
  //   rp: Radial partitioning information
  //   s: Grid sizes
  //   m_output: Output force arrays (armn, azmn, brmn, bzmn, crmn, czmn)
  virtual void ComputeMHDForces(const MHDForcesInput& input,
                                const RadialPartitioning& rp, const Sizes& s,
                                MHDForcesOutput& m_output) = 0;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_H_

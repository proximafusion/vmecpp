// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_CPU_H_
#define VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_CPU_H_

#include "vmecpp/common/compute_backend/compute_backend.h"

namespace vmecpp {

// CPU implementation of the compute backend interface.
//
// This implementation wraps the existing CPU-based DFT functions
// (FourierToReal3DSymmFastPoloidal and ForcesToFourier3DSymmFastPoloidal)
// to provide a consistent interface with other backend implementations.
//
// This is the reference implementation and default backend.
class ComputeBackendCpu : public ComputeBackend {
 public:
  ComputeBackendCpu() = default;
  ~ComputeBackendCpu() override = default;

  // Non-copyable, non-movable (stateless singleton pattern).
  ComputeBackendCpu(const ComputeBackendCpu&) = delete;
  ComputeBackendCpu& operator=(const ComputeBackendCpu&) = delete;
  ComputeBackendCpu(ComputeBackendCpu&&) = delete;
  ComputeBackendCpu& operator=(ComputeBackendCpu&&) = delete;

  BackendType GetType() const override { return BackendType::kCpu; }

  std::string GetName() const override { return "CPU"; }

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

  void Synchronize() override {
    // No-op for CPU backend - all operations are synchronous.
  }

  bool IsAvailable() const override {
    // CPU backend is always available.
    return true;
  }

  // Additional compute-heavy operations.
  bool ComputeJacobian(const JacobianInput& input, const RadialPartitioning& rp,
                       const Sizes& s, JacobianOutput& m_output) override;

  void ComputeMetricElements(const MetricInput& input,
                             const RadialPartitioning& rp, const Sizes& s,
                             MetricOutput& m_output) override;

  void ComputeBContra(const BContraInput& input, const RadialPartitioning& rp,
                      const Sizes& s, BContraOutput& m_output) override;

  void ComputeMHDForces(const MHDForcesInput& input,
                        const RadialPartitioning& rp, const Sizes& s,
                        MHDForcesOutput& m_output) override;
};

}  // namespace vmecpp

#endif  // VMECPP_COMMON_COMPUTE_BACKEND_COMPUTE_BACKEND_CPU_H_

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// VmecJacobianProbe: diagnostic helper to extract the preconditioner and
// Jacobian matrices of the VMEC++ force map via finite differences.
//
// Usage:
//   VmecJacobianProbe probe(indata);
//   probe.RunToConvergence();
//   probe.SnapshotState();
//   Eigen::VectorXd x0 = probe.GetStateVector();
//   Eigen::VectorXd f0 = probe.EvaluateForces(/*preconditioned=*/true);
//   // perturb, re-evaluate, etc.
//
// Thread model: forces single-thread execution; IdealMhdModel::update has
// OpenMP barriers that make multi-thread perturbation loops error-prone.
// This is a diagnostic tool, not a production path.
#ifndef VMECPP_VMEC_VMEC_JACOBIAN_PROBE_VMEC_JACOBIAN_PROBE_H_
#define VMECPP_VMEC_VMEC_JACOBIAN_PROBE_VMEC_JACOBIAN_PROBE_H_

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {

// Layout of a single entry in the flat state/force vector.
struct ModeIndex {
  int jF;        // radial full-grid index
  int m;         // poloidal mode number
  int n;         // toroidal mode number (0..ntor)
  int basis_id;  // 0..num_basis-1, see GetBasisNames()
  int comp_id;   // 0=R, 1=Z, 2=lambda
};

class VmecJacobianProbe {
 public:
  explicit VmecJacobianProbe(const VmecINDATA& indata);
  ~VmecJacobianProbe();

  VmecJacobianProbe(const VmecJacobianProbe&) = delete;
  VmecJacobianProbe& operator=(const VmecJacobianProbe&) = delete;

  // Run the full VMEC equilibrium solver to convergence.
  absl::Status RunToConvergence();

  // After convergence, snapshot decomposed_x and preconditioner state so
  // we can restore them between perturbation probes.
  void SnapshotState();

  // Restore decomposed_x from the snapshot.
  void RestoreState();

  // Total length of the state / force vector.
  int NumStateVars() const { return num_state_vars_; }

  // Index metadata: one entry per flat index.
  const std::vector<ModeIndex>& Index() const { return mode_index_; }

  // Names for the num_basis basis functions active in this configuration,
  // in the same order as basis_id in ModeIndex. E.g. {"cc","ss"} for 3D
  // symmetric.
  std::vector<std::string> GetBasisNames() const;

  // Pack the current decomposed_x[0] into a flat vector.
  Eigen::VectorXd GetStateVector() const;

  // Overwrite decomposed_x[0] from a flat vector. The caller is responsible
  // for restoring the state after probing.
  void SetStateVector(const Eigen::VectorXd& x);

  // Run one IdealMhdModel::update pass and read back forces.
  //
  // If preconditioned=true, reads forces after the full preconditioning
  // pipeline (M1 + RZ-tridiag + lambda-diag). If false, reads forces right
  // after forcesToFourier but before any preconditioner is applied.
  absl::StatusOr<Eigen::VectorXd> EvaluateForces(bool preconditioned);

  // Number of radial surfaces in the current (final) multigrid step.
  int NumFullSurfaces() const;

  int Mpol() const;
  int NtorPlusOne() const;
  int NumBasis() const;
  bool IsAxisymmetric() const;
  bool IsAsymmetric() const;

 private:
  // Fill mode_index_ from current sizes.
  void BuildModeIndex();

  // Pack/unpack helpers: copy between std::span<double> families and a
  // contiguous Eigen::VectorXd.
  Eigen::VectorXd PackForces(bool preconditioned) const;
  Eigen::VectorXd PackGeometry() const;
  void UnpackGeometry(const Eigen::VectorXd& x);

  VmecINDATA indata_;
  std::unique_ptr<Vmec> vmec_;
  bool converged_;
  int num_state_vars_;
  std::vector<ModeIndex> mode_index_;

  // Snapshot of state at convergence.
  Eigen::VectorXd state_snapshot_;
  bool have_snapshot_;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_VMEC_JACOBIAN_PROBE_VMEC_JACOBIAN_PROBE_H_

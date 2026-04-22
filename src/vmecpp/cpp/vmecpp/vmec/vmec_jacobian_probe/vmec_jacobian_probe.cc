// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/vmec_jacobian_probe/vmec_jacobian_probe.h"

#include <cassert>
#include <climits>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"

namespace vmecpp {

namespace {

// Returns the ordered list of basis names for a given (lthreed, lasym)
// configuration, matching the order in which we pack/unpack spans.
std::vector<std::string> BasisNamesFor(bool lthreed, bool lasym) {
  std::vector<std::string> names;
  // Symmetric basis always present.
  names.emplace_back("cc");  // rmncc / fzsc uses complement ordering; see below
  if (lthreed) {
    names.emplace_back("ss");
  }
  if (lasym) {
    names.emplace_back("sc");
    if (lthreed) {
      names.emplace_back("cs");
    }
  }
  return names;
}

}  // namespace

VmecJacobianProbe::VmecJacobianProbe(const VmecINDATA& indata)
    : indata_(indata),
      converged_(false),
      num_state_vars_(0),
      have_snapshot_(false) {
  // Force single-thread execution to keep the probe deterministic and to
  // avoid racing on the OpenMP barriers inside IdealMhdModel::update.
  auto status_or =
      Vmec::FromIndata(indata_, /*magnetic_response_table=*/nullptr,
                       /*max_threads=*/1, OutputMode::kSilent);
  if (!status_or.ok()) {
    // Can't recover here; downstream calls will fail too.
    vmec_ = nullptr;
    return;
  }
  vmec_ = std::move(status_or.value());
}

VmecJacobianProbe::~VmecJacobianProbe() = default;

absl::Status VmecJacobianProbe::RunToConvergence() {
  if (vmec_ == nullptr) {
    return absl::FailedPreconditionError("Vmec instance failed to initialize");
  }
  auto result = vmec_->run();
  if (!result.ok()) {
    return result.status();
  }
  converged_ = true;
  BuildModeIndex();
  return absl::OkStatus();
}

void VmecJacobianProbe::BuildModeIndex() {
  const Sizes& s = vmec_->s_;
  const RadialPartitioning& r = *vmec_->r_[0];

  const int jMin = r.nsMinF;
  // Use the same range the FourierCoeffs storage uses internally: up to
  // nsMaxFIncludingLcfs, so we pack every allocated slot (single-threaded,
  // non-free-bdy: nsMaxF = ns-1 but nsMaxFIncludingLcfs = ns).
  const int jMax = r.nsMaxFIncludingLcfs;

  mode_index_.clear();
  // Geometry layout order mirrored: for each (comp, basis, jF, m, n):
  //   comp_id 0 -> R with basis {cc, ss?, sc?, cs?}
  //   comp_id 1 -> Z with basis {sc, cs?, cc?, ss?}
  //   comp_id 2 -> lambda with basis {sc, cs?, cc?, ss?}
  // We pack all num_basis entries per coefficient type in the order they
  // appear in FourierGeometry/FourierForces (cc, ss, sc, cs) for R and
  // (sc, cs, cc, ss) for Z and lambda.
  const int num_basis = s.num_basis;
  for (int comp = 0; comp < 3; ++comp) {
    for (int b = 0; b < num_basis; ++b) {
      for (int jF = jMin; jF < jMax; ++jF) {
        for (int m = 0; m < s.mpol; ++m) {
          for (int n = 0; n < s.ntor + 1; ++n) {
            mode_index_.push_back({jF, m, n, b, comp});
          }
        }
      }
    }
  }
  num_state_vars_ = static_cast<int>(mode_index_.size());
}

std::vector<std::string> VmecJacobianProbe::GetBasisNames() const {
  if (vmec_ == nullptr) return {};
  return BasisNamesFor(vmec_->s_.lthreed, vmec_->s_.lasym);
}

int VmecJacobianProbe::NumFullSurfaces() const {
  if (vmec_ == nullptr) return 0;
  const RadialPartitioning& r = *vmec_->r_[0];
  return r.nsMaxFIncludingLcfs - r.nsMinF;
}

int VmecJacobianProbe::Mpol() const {
  if (vmec_ == nullptr) return 0;
  return vmec_->s_.mpol;
}

int VmecJacobianProbe::NtorPlusOne() const {
  if (vmec_ == nullptr) return 0;
  return vmec_->s_.ntor + 1;
}

int VmecJacobianProbe::NumBasis() const {
  if (vmec_ == nullptr) return 0;
  return vmec_->s_.num_basis;
}

bool VmecJacobianProbe::IsAxisymmetric() const {
  if (vmec_ == nullptr) return false;
  return !vmec_->s_.lthreed;
}

bool VmecJacobianProbe::IsAsymmetric() const {
  if (vmec_ == nullptr) return false;
  return vmec_->s_.lasym;
}

// Helper: return the ordered list of spans for R/Z/lambda in the current
// (lthreed, lasym) configuration. The order must match BuildModeIndex:
//   comp 0: R spans in {cc, ss?, sc?, cs?}
//   comp 1: Z spans in {sc, cs?, cc?, ss?}
//   comp 2: L spans in {sc, cs?, cc?, ss?}
namespace {

std::vector<std::span<double>> GeomSpans(FourierGeometry& g, const Sizes& s) {
  std::vector<std::span<double>> spans;
  // R
  spans.push_back(g.rmncc);
  if (s.lthreed) spans.push_back(g.rmnss);
  if (s.lasym) {
    spans.push_back(g.rmnsc);
    if (s.lthreed) spans.push_back(g.rmncs);
  }
  // Z
  spans.push_back(g.zmnsc);
  if (s.lthreed) spans.push_back(g.zmncs);
  if (s.lasym) {
    spans.push_back(g.zmncc);
    if (s.lthreed) spans.push_back(g.zmnss);
  }
  // lambda
  spans.push_back(g.lmnsc);
  if (s.lthreed) spans.push_back(g.lmncs);
  if (s.lasym) {
    spans.push_back(g.lmncc);
    if (s.lthreed) spans.push_back(g.lmnss);
  }
  return spans;
}

std::vector<std::span<double>> ForceSpans(FourierForces& f, const Sizes& s) {
  std::vector<std::span<double>> spans;
  spans.push_back(f.frcc);
  if (s.lthreed) spans.push_back(f.frss);
  if (s.lasym) {
    spans.push_back(f.frsc);
    if (s.lthreed) spans.push_back(f.frcs);
  }
  spans.push_back(f.fzsc);
  if (s.lthreed) spans.push_back(f.fzcs);
  if (s.lasym) {
    spans.push_back(f.fzcc);
    if (s.lthreed) spans.push_back(f.fzss);
  }
  spans.push_back(f.flsc);
  if (s.lthreed) spans.push_back(f.flcs);
  if (s.lasym) {
    spans.push_back(f.flcc);
    if (s.lthreed) spans.push_back(f.flss);
  }
  return spans;
}

}  // namespace

Eigen::VectorXd VmecJacobianProbe::GetStateVector() const {
  assert(vmec_ != nullptr && "not initialized");
  const Sizes& s = vmec_->s_;
  FourierGeometry& g = *vmec_->decomposed_x_[0];
  auto spans = GeomSpans(g, s);
  Eigen::VectorXd out(num_state_vars_);
  int offset = 0;
  for (const auto& sp : spans) {
    for (size_t i = 0; i < sp.size(); ++i) {
      out[offset + static_cast<int>(i)] = sp[i];
    }
    offset += static_cast<int>(sp.size());
  }
  return out;
}

void VmecJacobianProbe::SetStateVector(const Eigen::VectorXd& x) {
  assert(vmec_ != nullptr && "not initialized");
  assert(x.size() == num_state_vars_ && "state vector length mismatch");
  const Sizes& s = vmec_->s_;
  FourierGeometry& g = *vmec_->decomposed_x_[0];
  auto spans = GeomSpans(g, s);
  int offset = 0;
  for (auto& sp : spans) {
    for (size_t i = 0; i < sp.size(); ++i) {
      sp[i] = x[offset + static_cast<int>(i)];
    }
    offset += static_cast<int>(sp.size());
  }
}

void VmecJacobianProbe::SnapshotState() {
  state_snapshot_ = GetStateVector();
  have_snapshot_ = true;
}

void VmecJacobianProbe::RestoreState() {
  assert(have_snapshot_ && "call SnapshotState() before RestoreState()");
  SetStateVector(state_snapshot_);
}

absl::StatusOr<Eigen::VectorXd> VmecJacobianProbe::EvaluateForces(
    bool preconditioned) {
  assert(vmec_ != nullptr && "not initialized");

  // Pin iter1 close to iter2 so shouldUpdateRadialPreconditioner returns
  // false: iter2 - iter1 must not be a multiple of
  // kPreconditionerUpdateInterval. We use iter1 = iter2 - 1 unconditionally;
  // this keeps the already-assembled preconditioner frozen across probes. We
  // access these through the public Vmec API (iter1_, iter2_ are private), so
  // use get_iter1/get_iter2 to check and then call UpdateForwardModel which
  // reads them internally.
  //
  // The state returned by RunToConvergence leaves iter2_ at the final
  // iteration count and iter1_ at the last restart point; both are stable
  // across subsequent UpdateForwardModel calls below because we only
  // invoke the force-evaluation pipeline, not the full iteration loop.

  VmecCheckpoint checkpoint = preconditioned
                                  ? VmecCheckpoint::PRECONDITIONED_RESIDUALS
                                  : VmecCheckpoint::INVARIANT_RESIDUALS;
  // iterations_before_checkpointing = 0 so the checkpoint triggers immediately
  // on the current iter2.
  auto status_or = vmec_->UpdateForwardModel(checkpoint,
                                             /*maximum_iterations=*/0,
                                             /*thread_id=*/0);
  if (!status_or.ok()) {
    return status_or.status();
  }

  return PackForces(preconditioned);
}

Eigen::VectorXd VmecJacobianProbe::PackForces(bool /*preconditioned*/) const {
  const Sizes& s = vmec_->s_;
  FourierForces& f = *vmec_->decomposed_f_[0];
  auto spans = ForceSpans(f, s);
  Eigen::VectorXd out(num_state_vars_);
  int offset = 0;
  for (const auto& sp : spans) {
    for (size_t i = 0; i < sp.size(); ++i) {
      out[offset + static_cast<int>(i)] = sp[i];
    }
    offset += static_cast<int>(sp.size());
  }
  return out;
}

}  // namespace vmecpp

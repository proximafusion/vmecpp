// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// The additive spectral force source is given once for the whole radial grid,
// and every thread adds the slice its own surfaces own (IdealMhdModel::
// addForceSource, indexing from FourierForces::nsMin()). The radial force
// ranges tile the grid without a halo, so the change the source makes to the
// assembled force must not depend on how many threads that grid is split
// across; if a boundary row were counted twice or missed, it would.
#include <Eigen/Dense>
#include <climits>
#include <cmath>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {
namespace {

// The whole decomposed force, gathered from the per-thread slices into the flat
// layout SetForceSource takes: one block per active parity, each ns * mnsize.
Eigen::VectorXd AssembleForce(const Vmec& vmec) {
  const Sizes& s = vmec.s_;
  const int mnsize = s.mpol * (s.ntor + 1);
  const int ns = vmec.fc_.ns;
  const int block = ns * mnsize;
  Eigen::VectorXd out =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(s.num_basis) * 3 * block);
  for (int thread_id = 0; thread_id < vmec.num_threads_; ++thread_id) {
    FourierForces& f = *vmec.decomposed_f_[thread_id];
    const std::vector<std::span<double>> spans = f.ActiveSpans();
    for (size_t parity = 0; parity < spans.size(); ++parity) {
      const int rows = static_cast<int>(spans[parity].size()) / mnsize;
      for (int row = 0; row < rows; ++row) {
        const int global = (f.nsMin() + row) * mnsize;
        for (int i = 0; i < mnsize; ++i) {
          out[static_cast<Eigen::Index>(parity) * block + global + i] =
              spans[parity][row * mnsize + i];
        }
      }
    }
  }
  return out;
}

// One force evaluation of the initial state at the given thread count, with the
// given source installed. The INVARIANT_RESIDUALS checkpoint stops before the
// preconditioner, so the force is the raw gradient plus the source.
Eigen::VectorXd EvaluateOnce(const VmecINDATA& indata, int ns, int max_threads,
                             const Eigen::VectorXd& source,
                             int* threads_used = nullptr) {
  absl::StatusOr<std::unique_ptr<Vmec>> vmec_or =
      Vmec::FromIndata(indata, /*magnetic_response_table=*/nullptr, max_threads,
                       OutputMode::kSilent);
  CHECK_OK(vmec_or.status());
  Vmec& vmec = **vmec_or;
  vmec.fc_.ns_old = 0;
  vmec.fc_.delt0r = indata.delt;
  vmec.fc_.ns_min = 3;
  vmec.fc_.nsval = ns;
  vmec.fc_.ftolv = indata.ftol_array[0];
  vmec.fc_.niterv = indata.niter_array[0];
  double delt0 = indata.delt;
  vmec.InitializeRadial(VmecCheckpoint::NONE, INT_MAX, ns, /*ns_old=*/0, delt0,
                        std::nullopt);
  CHECK_OK(vmec.SetForceSource(source));
  const absl::StatusOr<bool> reached =
      vmec.SolveEquilibrium(VmecCheckpoint::INVARIANT_RESIDUALS, 1);
  CHECK_OK(reached.status());
  if (threads_used != nullptr) {
    *threads_used = vmec.num_threads_;
  }
  return AssembleForce(vmec);
}

VmecINDATA LoadIndata() {
  const absl::StatusOr<std::string> json =
      file_io::ReadFile("vmecpp/test_data/cth_like_fixed_bdy.json");
  CHECK_OK(json.status());
  absl::StatusOr<VmecINDATA> indata_or = VmecINDATA::FromJson(*json);
  CHECK_OK(indata_or.status());
  VmecINDATA indata = *indata_or;
  indata.enable_force_source = true;
  // one radial resolution, and never converge inside the single evaluation
  indata.ns_array.resize(1);
  indata.ns_array[0] = 31;
  indata.ftol_array.resize(1);
  indata.ftol_array[0] = 1.0e-30;
  indata.niter_array.resize(1);
  indata.niter_array[0] = 100000;
  return indata;
}

TEST(ForceSource, SlicingIsIndependentOfThreadCount) {
  const VmecINDATA indata = LoadIndata();
  const int ns = indata.ns_array[0];

  int threads_used = 0;
  const Eigen::VectorXd empty;
  const Eigen::VectorXd bare =
      EvaluateOnce(indata, ns, /*max_threads=*/1, empty, &threads_used);
  ASSERT_EQ(threads_used, 1);

  Eigen::VectorXd source(bare.size());
  for (Eigen::Index i = 0; i < source.size(); ++i) {
    source[i] = 1.0e-6 * std::sin(0.37 * static_cast<double>(i) + 0.11);
  }
  const Eigen::VectorXd single = EvaluateOnce(indata, ns, 1, source) - bare;

  // A source that changed nothing would make every comparison below vacuous.
  ASSERT_GT(single.cwiseAbs().maxCoeff(), 0.0);

  for (const int max_threads : {2, 4, 8}) {
    int threads = 0;
    const Eigen::VectorXd bare_n =
        EvaluateOnce(indata, ns, max_threads, empty, &threads);
    if (threads < 2) {
      GTEST_SKIP() << "the build or the machine gives only one radial thread";
    }
    const Eigen::VectorXd delta =
        EvaluateOnce(indata, ns, max_threads, source) - bare_n;
    // Everything the source passes through before this point is linear and
    // thread-local, so the difference is exact rather than close.
    EXPECT_EQ((delta - single).cwiseAbs().maxCoeff(), 0.0)
        << "the source lands differently on " << threads << " threads";
  }
}

}  // namespace
}  // namespace vmecpp

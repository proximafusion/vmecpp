// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "absl/log/log.h"
#include "vmecpp/vmec/vmec_constants/vmec_algorithm_constants.h"

namespace vmecpp {

namespace {

// Read a non-negative integer from `name` env var; fall back to `default_value`
// on absence or parse error.
int ReadIntEnv(const char* name, int default_value) {
  const char* v = std::getenv(name);
  if (v == nullptr) {
    return default_value;
  }
  char* endp = nullptr;
  long parsed = std::strtol(v, &endp, 10);
  if (endp == v || *endp != '\0' || parsed < 0 || parsed > 1000) {
    return default_value;
  }
  return static_cast<int>(parsed);
}

// Compute per-thread radial slice sizes.
//
// We model boundary thread overhead as if the axis-owning thread (tid 0)
// has `n_axis_off` invisible extra surfaces of work and the LCFS-owning
// thread (tid T-1) has `n_lcfs_off` invisible extras. The total amount of
// "effective work" to distribute is therefore
// `num_surfaces + n_axis_off + n_lcfs_off`, and we distribute that evenly
// across all threads (Hamilton's method). Each thread's actual surface
// count is then its assigned-work share minus the inherent boundary
// overhead it carries. This:
//
//   * Yields the makespan-optimal contiguous integer partition.
//   * Degrades gracefully at small T -- the offsets are a small fraction
//     of total work, so we converge naturally to the equal partition.
//   * Avoids the "interior thread becomes the new bottleneck" pathology
//     that plagues naive surplus-redistribution schemes.
//
// At very wide thread counts the assigned share for a boundary thread
// could fall to or below `n_*_off`; we then clamp the slice to at least
// one surface and rebalance the small residual against the heaviest
// interior. The `num_threads <= ns / 2` invariant (enforced upstream)
// makes this corner case rare in practice.
//
// Returns `true` when any non-zero shift was effectively applied.
bool ComputeSliceSizes(int num_surfaces, int num_threads, int n_axis_off,
                       int n_lcfs_off, std::vector<int>* sizes_out) {
  std::vector<int>& sizes = *sizes_out;
  sizes.assign(num_threads, 0);

  if (num_threads == 1) {
    sizes[0] = num_surfaces;
    return false;
  }

  if (n_axis_off <= 0 && n_lcfs_off <= 0) {
    // Standard equal-share partition.
    const int base = num_surfaces / num_threads;
    const int rem = num_surfaces % num_threads;
    for (int t = 0; t < num_threads; ++t) {
      sizes[t] = base + (t < rem ? 1 : 0);
    }
    return false;
  }

  // Effective-work distribution.
  const int total_work = num_surfaces + n_axis_off + n_lcfs_off;
  const int base_work = total_work / num_threads;
  const int rem_work = total_work % num_threads;
  for (int t = 0; t < num_threads; ++t) {
    sizes[t] = base_work + (t < rem_work ? 1 : 0);
  }

  // Convert assigned work back to surface counts.
  sizes[0] -= n_axis_off;
  sizes[num_threads - 1] -= n_lcfs_off;

  // Clamp boundary slices to at least one surface, then rebalance the
  // (typically zero) residual sum-deviation against the heaviest thread.
  sizes[0] = std::max(1, sizes[0]);
  sizes[num_threads - 1] = std::max(1, sizes[num_threads - 1]);
  int actual = 0;
  for (int s : sizes) {
    actual += s;
  }
  while (actual > num_surfaces) {
    int max_idx = 0;
    for (int t = 1; t < num_threads; ++t) {
      if (sizes[t] > sizes[max_idx]) {
        max_idx = t;
      }
    }
    if (sizes[max_idx] <= 1) {
      break;
    }
    --sizes[max_idx];
    --actual;
  }
  while (actual < num_surfaces) {
    int min_idx = 0;
    for (int t = 1; t < num_threads; ++t) {
      if (sizes[t] < sizes[min_idx]) {
        min_idx = t;
      }
    }
    ++sizes[min_idx];
    ++actual;
  }

  return true;
}

}  // namespace

RadialPartitioning::RadialPartitioning() {
  // defaults
  adjustRadialPartitioning(1, 0, kNsDefault, false, false);
}

void RadialPartitioning::adjustRadialPartitioning(int num_threads,
                                                  int thread_id, int ns_input,
                                                  bool lfreeb, bool printout) {
  ns_ = ns_input;
  int num_surfaces_to_distribute = ns_ - 1;
  if (lfreeb) {
    num_surfaces_to_distribute = ns_;
  }

  if (num_threads > ns_ / 2) {
    LOG(FATAL) << "Cannot make use of more than ns/2 (= " << (ns_ / 2)
               << ") threads, but tried to use " << num_threads << " threads.";
  }

  this->num_threads_ = num_threads;
  this->thread_id_ = thread_id;

  // ---------------------------------

  // full-grid range for inv-DFT, needed for Jacobian etc.
  nsMinF1 = 0;
  nsMaxF1 = ns_;

  // half-grid range for Jacobian etc., needed for forces
  nsMinH = 0;
  nsMaxH = ns_ - 1;

  // full-grid range for forces
  nsMinF = 0;
  if (lfreeb) {
    nsMaxF = ns_;
  } else {
    nsMaxF = ns_ - 1;
  }

  // interior full-grid range: radial force balance, Mercier stability
  nsMinFi = 1;
  nsMaxFi = ns_ - 1;

  // some things like the lambda force and the constraint force ingredients
  // are always needed up to the boundary
  nsMaxFIncludingLcfs = nsMaxF;
  if (nsMaxF1 == ns_) {
    nsMaxFIncludingLcfs = ns_;
  }

  // ---------------------------------

  // setup radial index ranges for multi-threading
  bool partition_shifted = false;
  if (num_threads > 1) {
    // Boundary threads (axis owner = tid 0; LCFS owner = tid num_threads-1)
    // pay constant per-call overhead beyond the per-surface cost. To balance
    // the makespan at team barriers, optionally shift surfaces off them.
    // Compile-time defaults from `vmec_algorithm_constants.h`; both
    // overridable at runtime via env vars for sweeping/A/B testing.
    const int n_axis_off = ReadIntEnv(
        "VMECPP_AXIS_SURFACES_OFF",
        vmec_algorithm_constants::kBoundarySurfacesOffAxis);
    const int n_lcfs_off = ReadIntEnv(
        "VMECPP_LCFS_SURFACES_OFF",
        vmec_algorithm_constants::kBoundarySurfacesOffLcfs);

    std::vector<int> sizes;
    partition_shifted = ComputeSliceSizes(num_surfaces_to_distribute,
                                           num_threads, n_axis_off,
                                           n_lcfs_off, &sizes);

    nsMinF = 0;
    for (int t = 0; t < thread_id; ++t) {
      nsMinF += sizes[t];
    }
    nsMaxF = nsMinF + sizes[thread_id];

    // --------------------------------

    // extended by +/- 1 flux surface: ingredients for half-grid points in this
    // rank

    nsMinF1 = std::max(0, nsMinF - 1);
    nsMaxF1 = std::min(ns_, nsMaxF + 1);

    // --------------------------------

    // half-grid points in this rank

    nsMinH = nsMinF1;
    nsMaxH = nsMaxF1 - 1;

    // --------------------------------

    // internal full-grid points in this rank
    // --> always exclude axis and LCFS
    // (mainly used for radial force balance and Mercier stability)

    nsMinFi = std::max(1, nsMinF);
    nsMaxFi = std::min(ns_ - 1, nsMaxF);

    // --------------------------------

    // some things like the lambda force and the constraint force ingredients
    // are always needed up to the boundary
    nsMaxFIncludingLcfs = nsMaxF;
    if (nsMaxF1 == ns_) {
      nsMaxFIncludingLcfs = ns_;
    }
  }  // num_threads > 1

  if (printout) {
    std::cout << absl::StrFormat(
        "thread %2d/%2d%s: "
        "{nsMinF=%2d nsMaxF=%2d numFull=%2d} "
        "{nsMinF1=%2d nsMaxF1=%2d numFull1=%2d} "
        "{nsMinH=%2d nsMaxH=%2d numHalf=%2d} "
        "{nsMinFi=%2d nsMaxFi=%2d numFullI=%2d}\n",
        thread_id + 1, num_threads,
        partition_shifted ? " [boundary-shifted]" : "",
        nsMinF, nsMaxF, nsMaxF - nsMinF, nsMinF1,
        nsMaxF1, nsMaxF1 - nsMinF1, nsMinH, nsMaxH, nsMaxH - nsMinH, nsMinFi,
        nsMaxFi, nsMaxFi - nsMinFi);
  }
}  // adjustRadialPartitioning

int RadialPartitioning::get_num_threads() const { return num_threads_; }

int RadialPartitioning::get_thread_id() const { return thread_id_; }

bool RadialPartitioning::has_boundary() const { return nsMaxF1 == ns_; }

}  // namespace vmecpp

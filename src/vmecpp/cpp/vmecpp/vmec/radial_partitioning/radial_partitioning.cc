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

// Compute the number of surfaces per-thread (work distribution).
//
// We model the work per thread as uniform + constant overhead for the two
// threads at the boundaries, as if the axis-owning thread (tid 0)
// has `n_axis_off` invisible extra surfaces of work and the LCFS-owning
// thread (tid T-1) has `n_lcfs_off` invisible extras.
// These "extra surfaces" model the extra work due to the boundary
// condition branches those two threads have to take.
// The total amount of "effective work" to distribute is therefore
// `num_surfaces + n_axis_off + n_lcfs_off`, and we distribute that evenly
// across all threads (Hamilton's method). Each thread's actual surface
// count is then its assigned-work share minus the inherent boundary
// overhead it carries.
// This leads to better work balance and therefore parallel scaling
std::vector<int> ComputeSliceSizes(int num_surfaces, int num_threads) {
  std::vector<int> sizes{num_threads};
  sizes.assign(num_threads, 0);
  static_assert(vmec_algorithm_constants::kBoundarySurfacesOffAxis >= 0);
  static_assert(vmec_algorithm_constants::kBoundarySurfacesOffLcfs >= 0);

  // Effective-work distribution.
  const int total_work = num_surfaces +
                         vmec_algorithm_constants::kBoundarySurfacesOffAxis +
                         vmec_algorithm_constants::kBoundarySurfacesOffLcfs;
  const int base_work = total_work / num_threads;
  const int rem_work = total_work % num_threads;
  for (int t = 0; t < num_threads; ++t) {
    sizes[t] = base_work + (t < rem_work ? 1 : 0);
  }

  // Convert assigned work back to surface counts.
  sizes[0] -= vmec_algorithm_constants::kBoundarySurfacesOffAxis;
  sizes[num_threads - 1] -= vmec_algorithm_constants::kBoundarySurfacesOffLcfs;

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

  return sizes;
}

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
    std::vector<int> sizes =
        ComputeSliceSizes(num_surfaces_to_distribute, num_threads);

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
        "thread %2d/%2d: "
        "{nsMinF=%2d nsMaxF=%2d numFull=%2d} "
        "{nsMinF1=%2d nsMaxF1=%2d numFull1=%2d} "
        "{nsMinH=%2d nsMaxH=%2d numHalf=%2d} "
        "{nsMinFi=%2d nsMaxFi=%2d numFullI=%2d}\n",
        thread_id + 1, num_threads, nsMinF, nsMaxF, nsMaxF - nsMinF, nsMinF1,
        nsMaxF1, nsMaxF1 - nsMinF1, nsMinH, nsMaxH, nsMaxH - nsMinH, nsMinFi,
        nsMaxFi, nsMaxFi - nsMinFi);
  }
}  // adjustRadialPartitioning

int RadialPartitioning::get_num_threads() const { return num_threads_; }

int RadialPartitioning::get_thread_id() const { return thread_id_; }

bool RadialPartitioning::has_boundary() const { return nsMaxF1 == ns_; }

}  // namespace vmecpp

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

#include <algorithm>
#include <iostream>

#include "absl/log/log.h"

namespace vmecpp {

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
  if (num_threads > 1) {
    int work_per_cpu = num_surfaces_to_distribute / num_threads;
    int work_remainder = num_surfaces_to_distribute % num_threads;

    nsMinF = thread_id * work_per_cpu;
    nsMaxF = (thread_id + 1) * work_per_cpu;
    if (thread_id < work_remainder) {
      nsMinF += thread_id;
      nsMaxF += thread_id + 1;
    } else {
      nsMinF += work_remainder;
      nsMaxF += work_remainder;
    }

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

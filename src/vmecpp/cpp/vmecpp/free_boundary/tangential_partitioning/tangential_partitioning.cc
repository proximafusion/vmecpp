// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

TangentialPartitioning::TangentialPartitioning(int nZnT, int num_threads,
                                               int thread_id)
    : num_threads_(num_threads), thread_id_(thread_id) {
  adjustPartitioning(nZnT);
}

void TangentialPartitioning::adjustPartitioning(int nZnT) {
  // NOTE:
  //  ztMin is inclusive (start)
  //  ztMax is exclusive (one past end)

  int work_per_CPU = nZnT / num_threads_;
  int work_remainder = nZnT % num_threads_;

  ztMin = thread_id_ * work_per_CPU;
  ztMax = (thread_id_ + 1) * work_per_CPU;
  if (thread_id_ < work_remainder) {
    ztMin += thread_id_;
    ztMax += thread_id_ + 1;
  } else {
    ztMin += work_remainder;
    ztMax += work_remainder;
  }
}

int TangentialPartitioning::get_thread_id() const { return thread_id_; }

}  // namespace vmecpp

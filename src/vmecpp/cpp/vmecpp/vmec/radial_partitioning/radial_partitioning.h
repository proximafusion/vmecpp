// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_RADIAL_PARTITIONING_RADIAL_PARTITIONING_H_
#define VMECPP_VMEC_RADIAL_PARTITIONING_RADIAL_PARTITIONING_H_

#include <mutex>
#include <vector>

#include "vmecpp/common/vmec_indata/vmec_indata.h"  // NS_DEFAULT

namespace vmecpp {

class RadialPartitioning {
 public:
  RadialPartitioning();

  // perform a collective call of this to figure out which CPUs are needed for
  // the current number of surfaces
  void adjustRadialPartitioning(int num_threads, int thread_id, int ns,
                                bool lfreeb, bool printout = true);

  // get the number of threads currently involved in the plasma calculation
  int get_num_threads() const;

  // get the identifier of the current thread
  int get_thread_id() const;

  // returns true if this thread has the plasma boundary
  bool has_boundary() const;

  // ---------------------------------

  // NOTE:
  //  nsMin* is inclusive (start)
  //  nsMax* is exclusive (one past end)

  // nsMinF-1 if possible
  int nsMinF1;

  // nsMaxF+1 if possible
  int nsMaxF1;

  // ---------------------------------

  // radial start index for half-grid arrays
  int nsMinH;

  // radial (end+1) index for half-grid arrays
  int nsMaxH;

  // ---------------------------------

  // radial start index for interior full-grid arrays
  int nsMinFi;

  // radial (end+1) index for interior full-grid arrays
  int nsMaxFi;

  // ---------------------------------

  // radial start index for full-grid arrays
  int nsMinF;

  // radial (end+1) index for full-grid arrays
  int nsMaxF;

  // ---------------------------------

  // radial (end+1) index for full-grid arrays,
  // always including the last closed flux surface (== boundary)
  int nsMaxFIncludingLcfs;

 private:
  int num_threads_;
  int thread_id_;

  int ns_;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_RADIAL_PARTITIONING_RADIAL_PARTITIONING_H_

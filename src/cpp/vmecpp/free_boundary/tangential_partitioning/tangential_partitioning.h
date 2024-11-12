#ifndef VMECPP_FREE_BOUNDARY_TANGENTIAL_PARTITIONING_TANGENTIAL_PARTITIONING_H_
#define VMECPP_FREE_BOUNDARY_TANGENTIAL_PARTITIONING_TANGENTIAL_PARTITIONING_H_

#include "vmecpp/common/sizes/sizes.h"

namespace vmecpp {

class TangentialPartitioning {
 public:
  explicit TangentialPartitioning(int nZnT, int num_threads = 1,
                                  int thread_id = 0);

  void adjustPartitioning(int nZnT);
  int get_thread_id();

  int ztMin;
  int ztMax;

 private:
  int num_threads_;
  int thread_id_;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_TANGENTIAL_PARTITIONING_TANGENTIAL_PARTITIONING_H_

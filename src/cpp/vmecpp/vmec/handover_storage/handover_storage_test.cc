#include "vmecpp/vmec/handover_storage/handover_storage.h"

#include <gtest/gtest.h>

TEST(TestHandoverStorage, CheckAllocateMultithreaded) {
  // TODO(jons): check that HandoverStorage::allocate() works in multithreaded
  // environments only if the omp barrier is in place after the omp single
  // allocations.
}  // CheckAllocateMultithreaded

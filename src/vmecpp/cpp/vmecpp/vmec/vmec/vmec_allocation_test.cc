// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// Guards that the VMEC++ iteration hot loop performs no heap allocations per
// steady-state iteration (issue #594). The whole malloc family is interposed
// (so Eigen's aligned allocations and operator new are both observed), the same
// fixed-boundary case is run to two different iteration counts, and the
// allocation delta over the extra iterations must be zero. The run is pinned to
// a single thread for determinism.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/vmec/vmec.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Allocation counting interposes the malloc family, which conflicts with the
// allocators that AddressSanitizer / ThreadSanitizer / MemorySanitizer install.
// In a sanitized build, drop the interposition and skip the test.
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define VMECPP_ALLOC_TEST_SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer)
#define VMECPP_ALLOC_TEST_SANITIZED 1
#endif
#endif

namespace {
std::atomic<bool> g_count_enabled{false};
std::atomic<uint64_t> g_alloc_count{0};
}  // namespace

#ifndef VMECPP_ALLOC_TEST_SANITIZED
namespace {
inline void NoteAllocation() {
  if (g_count_enabled.load(std::memory_order_relaxed)) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
}
}  // namespace

// Interpose the whole malloc family so both operator new and Eigen's aligned
// allocations are counted. Counting is gated by g_count_enabled, so allocations
// outside the measured window are passed straight through.
extern "C" {
void* __libc_malloc(size_t);
void* __libc_calloc(size_t, size_t);
void* __libc_realloc(void*, size_t);
void __libc_free(void*);
void* __libc_memalign(size_t, size_t);
void* malloc(size_t n) {
  NoteAllocation();
  return __libc_malloc(n);
}
void* calloc(size_t a, size_t b) {
  NoteAllocation();
  return __libc_calloc(a, b);
}
void* realloc(void* p, size_t n) {
  NoteAllocation();
  return __libc_realloc(p, n);
}
void free(void* p) { __libc_free(p); }
void* memalign(size_t al, size_t n) {
  NoteAllocation();
  return __libc_memalign(al, n);
}
void* aligned_alloc(size_t al, size_t n) {
  NoteAllocation();
  return __libc_memalign(al, n);
}
int posix_memalign(void** out, size_t al, size_t n) {
  NoteAllocation();
  void* p = __libc_memalign(al, n);
  if (p == nullptr) return 12;  // ENOMEM
  *out = p;
  return 0;
}
}  // extern "C"
#endif  // !VMECPP_ALLOC_TEST_SANITIZED

namespace vmecpp {
namespace {

// Builds a fresh Vmec and runs exactly `iterations` force evaluations, counting
// every heap allocation during the run (setup included). The EVOLVE checkpoint
// stops the solver after `iterations` evaluations.
uint64_t CountHotLoopAllocations(const VmecINDATA& base, int iterations) {
  VmecINDATA indata = base;
  absl::StatusOr<std::unique_ptr<Vmec>> maybe_vmec = Vmec::FromIndata(
      indata, /*magnetic_response_table=*/nullptr, /*max_threads=*/1);
  CHECK_OK(maybe_vmec.status());
  Vmec& vmec = **maybe_vmec;
  g_alloc_count.store(0);
  g_count_enabled.store(true);
  (void)vmec.run(VmecCheckpoint::EVOLVE, iterations);
  g_count_enabled.store(false);
  return g_alloc_count.load();
}

TEST(VmecHotLoop, IsAllocationFree) {
#ifdef VMECPP_ALLOC_TEST_SANITIZED
  GTEST_SKIP() << "allocation counting is incompatible with the sanitizer "
                  "allocator (asan/tsan/msan)";
#else
#ifdef _OPENMP
  // Pin to one thread: vmecpp is only run-to-run deterministic single-threaded,
  // and the per-thread scratch is warmed once below.
  omp_set_num_threads(1);
#endif

  const absl::StatusOr<std::string> indata_json =
      file_io::ReadFile("vmecpp/test_data/cth_like_fixed_bdy.json");
  ASSERT_TRUE(indata_json.ok());
  absl::StatusOr<VmecINDATA> indata_or = VmecINDATA::FromJson(*indata_json);
  ASSERT_TRUE(indata_or.ok());
  VmecINDATA indata = *indata_or;

  // Keep iterating through the whole measurement window: an unreachable force
  // tolerance prevents early convergence and a high iteration cap keeps the
  // first multigrid stage from transitioning, so each run performs exactly the
  // requested number of force evaluations within a single stage.
  indata.ftol_array.setConstant(1.0e-30);
  indata.niter_array.setConstant(100000);
  indata.return_outputs_even_if_not_converged = true;

  // Warm one-time per-thread scratch (e.g. thread_local DFT buffers) so the
  // differential isolates per-iteration allocations only.
  (void)CountHotLoopAllocations(indata, 30);

  constexpr int kFewerIters = 40;
  constexpr int kMoreIters = 80;
  const uint64_t allocs_fewer = CountHotLoopAllocations(indata, kFewerIters);
  const uint64_t allocs_more = CountHotLoopAllocations(indata, kMoreIters);

  // The two runs differ only by (kMoreIters - kFewerIters) extra steady-state
  // iterations; an allocation-free hot loop performs none.
  EXPECT_EQ(allocs_more, allocs_fewer)
      << "hot loop allocated "
      << (static_cast<long long>(allocs_more) - allocs_fewer)
      << " time(s) over " << (kMoreIters - kFewerIters) << " iterations";
#endif  // VMECPP_ALLOC_TEST_SANITIZED
}

// Positive control for the harness above: the interposed counter must observe a
// known number of heap allocations while enabled, and none while disabled.
// Without this, IsAllocationFree could pass trivially on a counter that never
// fires.
TEST(VmecHotLoop, AllocationCounterWorks) {
#ifdef VMECPP_ALLOC_TEST_SANITIZED
  GTEST_SKIP() << "allocation counting is incompatible with the sanitizer "
                  "allocator (asan/tsan/msan)";
#else
  constexpr int kAllocations = 64;

  // Reserve before enabling so only the make_unique allocations are counted.
  std::vector<std::unique_ptr<double>> kept;
  kept.reserve(kAllocations);

  g_alloc_count.store(0);
  g_count_enabled.store(true);
  for (int i = 0; i < kAllocations; ++i) {
    kept.push_back(std::make_unique<double>(static_cast<double>(i)));
  }
  g_count_enabled.store(false);
  const uint64_t counted_while_enabled = g_alloc_count.load();

  // Touch the data so the allocations cannot be optimized away.
  double sink = 0.0;
  for (const std::unique_ptr<double>& p : kept) {
    sink += *p;
  }
  ASSERT_EQ(sink, static_cast<double>(kAllocations * (kAllocations - 1) / 2));

  EXPECT_GE(counted_while_enabled, static_cast<uint64_t>(kAllocations))
      << "counter under-reported a known set of allocations";

  // With counting disabled, the same allocations are not observed.
  std::vector<std::unique_ptr<double>> kept2;
  kept2.reserve(kAllocations);
  g_alloc_count.store(0);
  for (int i = 0; i < kAllocations; ++i) {
    kept2.push_back(std::make_unique<double>(static_cast<double>(i)));
  }
  EXPECT_EQ(g_alloc_count.load(), 0u) << "counter incremented while disabled";
#endif  // VMECPP_ALLOC_TEST_SANITIZED
}

}  // namespace
}  // namespace vmecpp

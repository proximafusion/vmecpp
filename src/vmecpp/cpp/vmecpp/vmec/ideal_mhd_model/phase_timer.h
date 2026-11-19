// Phase-timer instrumentation for the VMEC++ iteration body. Maintains a
// thread-safe, process-global accumulator of elapsed wall time keyed by an
// arbitrary phase identifier string, with periodic and at-exit serialization
// to a file path determined by the VMECPP_PHASE_TIMING_PATH environment
// variable (a fixed default under /tmp is used when the variable is unset).
// Periodic serialization ensures that the accumulated statistics persist
// independently of the standard error stream lifetime, which under the
// embedded Python interpreter is closed prior to std::atexit handler
// execution and would otherwise lose the final-state report.
//
// The instrumentation is compiled only under VMECPP_USE_CUDA; otherwise the
// VMECPP_PHASE_TIMER macro expands to nothing.
#pragma once

#ifdef VMECPP_USE_CUDA

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace vmecpp {

// PhaseStats: process-global singleton accumulator for phase-elapsed wall
// time measurements. Each measurement is associated with a string identifier
// (the phase name) and the singleton maintains the total elapsed time and
// call count per identifier. Internal synchronization via a single mutex
// renders the accumulator safe for concurrent invocation from multiple
// OpenMP worker threads. Serialization to disk is performed by Dump and is
// also installed as an std::atexit handler during first construction.
class PhaseStats {
 public:
  // Returns a reference to the singleton instance. The instance is constructed
  // on the first invocation and is intentionally never destroyed: the
  // constructor registers Dump with std::atexit, which the C++ runtime runs
  // AFTER the destructor of a function-local static (handlers execute in
  // reverse registration order, and the destructor is registered after the
  // constructor body completes). A destroyed-then-dumped accumulator is
  // undefined behavior that manifests as an intermittent teardown segfault;
  // leaking the singleton keeps the object alive through every exit handler.
  static PhaseStats& instance() {
    static PhaseStats* s =  // NOLINT(cppcoreguidelines-owning-memory)
        new PhaseStats();
    return *s;
  }
  // Records an elapsed-time measurement of duration secs (in seconds) against
  // the phase identifier name. Both the cumulative total and the invocation
  // count for that identifier are advanced atomically under the singleton
  // mutex. The caller retains ownership of the name string and is responsible
  // for ensuring its lifetime exceeds the duration of statistical aggregation;
  // typical usage supplies a string literal.
  void add(const char* name, double secs) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& e = times_[name];
    e.first += secs;
    e.second += 1;
  }
  // Serializes the accumulated statistics to disk in a human-readable
  // formatted listing. Entries are sorted in descending order of cumulative
  // elapsed time so that the most expensive phases appear first. For each
  // phase identifier the report includes total elapsed seconds, invocation
  // count, mean per-invocation time in milliseconds, and the percentage of
  // the cumulative sum across all phases. A trailing line records the
  // cumulative sum itself; note that overlapping phases contribute their
  // intervals to this sum independently, so the cumulative figure can exceed
  // wall-clock time for the surrounding iteration body.
  void Dump() {
    std::lock_guard<std::mutex> lk(mu_);
    if (times_.empty()) return;
    std::vector<std::tuple<double, long long, std::string>> sorted;
    double total = 0;
    for (auto& kv : times_) {
      sorted.emplace_back(kv.second.first, kv.second.second, kv.first);
      total += kv.second.first;
    }
    std::sort(sorted.rbegin(), sorted.rend());
    // Serialization is directed to an on-disk file path rather than to the
    // standard error stream. The std::atexit handler ordering relative to the
    // host runtime's stream shutdown is implementation-defined, and under the
    // embedded Python interpreter the standard error stream is closed before
    // this handler executes; writing to a file path bypasses that constraint.
    const char* path = std::getenv("VMECPP_PHASE_TIMING_PATH");
    if (!path || !*path) path = "/tmp/vmecpp_phase_timing.txt";
    FILE* f =  // NOLINT(cppcoreguidelines-owning-memory)
        std::fopen(path, "w");
    if (!f) {
      times_.clear();
      return;
    }
    std::fprintf(f, "=== VMECPP PHASE TIMING ===\n");
    for (auto& t : sorted) {
      double secs;
      long long calls;
      std::string name;
      std::tie(secs, calls, name) = t;
      double avg_ms =
          (calls > 0) ? (secs / static_cast<double>(calls)) * 1000.0 : 0.0;
      double pct = (total > 0) ? (secs / total) * 100.0 : 0.0;
      std::fprintf(
          f, "  %-42s  total=%9.4fs  calls=%7lld  avg=%9.4fms  pct=%6.2f%%\n",
          name.c_str(), secs, calls, avg_ms, pct);
    }
    std::fprintf(f, "  ---\n  cumulative=%9.4fs (sum of overlapping phases)\n",
                 total);
    std::fclose(f);  // NOLINT(cppcoreguidelines-owning-memory)
    // The accumulator map is intentionally retained across invocations of
    // Dump so that periodic mid-execution serializations expose a
    // monotonically growing running tally. Clearing the map at each dump
    // would discard data accumulated prior to a periodic invocation and
    // produce inconsistent statistics between successive snapshots.
  }

 private:
  PhaseStats() {
    std::atexit([]() { PhaseStats::instance().Dump(); });
  }
  std::mutex mu_;
  std::map<std::string, std::pair<double, long long>> times_;
};

// PhaseTimer: RAII scope timer that captures elapsed wall time between
// construction and destruction and records the result against the supplied
// phase identifier in the process-global PhaseStats singleton. Typical use
// declares an instance at the start of a measured code region; the destructor
// fires automatically when the region exits, including via early return or
// exception propagation. Instances are intended to be cheap and short-lived.
class PhaseTimer {
 public:
  // Constructs the timer and captures the high-resolution clock value as the
  // start of the measured interval. The phase identifier name is stored by
  // pointer; the caller must guarantee its lifetime exceeds destruction of
  // this instance.
  explicit PhaseTimer(const char* name)
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
  // Computes the elapsed interval since construction and records it against
  // the stored phase identifier. Triggers an opportunistic serialization of
  // the global accumulator every fixed number of destructor invocations to
  // ensure partial data persists across abnormal process termination.
  ~PhaseTimer() {
    double secs = std::chrono::duration<double>(
                      std::chrono::high_resolution_clock::now() - start_)
                      .count();
    PhaseStats::instance().add(name_, secs);
    // Periodic serialization is triggered at fixed intervals of timer-instance
    // destruction (every 5000 invocations summed across all phase identifiers).
    // This decouples preservation of the accumulated statistics from the
    // std::atexit ordering with respect to host-runtime stream shutdown, so
    // partial timing data survives even when the process is terminated before
    // a clean exit handler executes.
    static std::atomic<long long> tick(0);
    long long v = tick.fetch_add(1, std::memory_order_relaxed);
    if ((v % 5000) == 0) {
      PhaseStats::instance().Dump();
    }
  }

 private:
  const char* name_;
  std::chrono::high_resolution_clock::time_point start_;
};

}  // namespace vmecpp

// Token-pasting helpers used by VMECPP_PHASE_TIMER to construct a unique
// local variable identifier per source line.
#define VMECPP_PHASE_CONCAT_(a, b) a##b
#define VMECPP_PHASE_CONCAT(a, b) VMECPP_PHASE_CONCAT_(a, b)

// VMECPP_PHASE_TIMER: declares a stack-allocated PhaseTimer at the macro's
// expansion point that measures the elapsed wall time from this point through
// the end of the enclosing scope. The identifier name is recorded against
// the global statistics accumulator at destruction. The local variable name
// is synthesized from the line number to permit nested or sequential usage
// within the same scope without identifier collisions.
#define VMECPP_PHASE_TIMER(name) \
  ::vmecpp::PhaseTimer VMECPP_PHASE_CONCAT(_vt_, __LINE__)(name)

#else  // !VMECPP_USE_CUDA

#define VMECPP_PHASE_TIMER(name) static_cast<void>(0)

#endif  // VMECPP_USE_CUDA

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Microbenchmark for ComputeOutputQuantities, the post-solve output
// computation.
//
// ComputeOutputQuantities() runs once after the equilibrium has converged and
// produces the entire wout contents, bsubs on both grids, jxbout, Mercier
// stability, and the other diagnostics.  None of this feeds back into the
// force-balance loop, so it is pure post-processing overhead -- this benchmark
// answers "how much time do we spend after the actual solve has finished".
//
// Setup (untimed): load a small fixed-boundary case, run the solver to
// convergence via Vmec::run().  Timed loop: re-invoke ComputeOutputQuantities
// on the converged state.  ComputeOutputQuantities takes all inputs by const
// reference and returns a fresh OutputQuantities by value, so repeated calls
// are idempotent.

#include <cstdlib>
#include <iostream>
#include <string>

#include "absl/strings/str_format.h"
#include "benchmark/benchmark.h"
#include "util/file_io/file_io.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/common/vmec_indata/vmec_indata.h"
#include "vmecpp/vmec/output_quantities/output_quantities.h"
#include "vmecpp/vmec/vmec/vmec.h"

namespace vmecpp {
namespace {

// Small fixed-boundary stellarator case; converges quickly during setup.
constexpr char kCase[] = "cma";

void BM_ComputeOutputQuantities(benchmark::State& state) {
  // ---- Untimed setup: drive the solver to convergence. ----
  const std::string filename =
      absl::StrFormat("vmecpp/test_data/%s.json", kCase);
  absl::StatusOr<std::string> indata_json = file_io::ReadFile(filename);
  if (!indata_json.ok()) {
    state.SkipWithError("failed to read input JSON");
    return;
  }
  absl::StatusOr<VmecINDATA> indata = VmecINDATA::FromJson(*indata_json);
  if (!indata.ok()) {
    state.SkipWithError("failed to parse INDATA");
    return;
  }

  absl::StatusOr<std::unique_ptr<Vmec>> maybe_vmec = Vmec::FromIndata(*indata);
  if (!maybe_vmec.ok()) {
    state.SkipWithError("Vmec::FromIndata failed");
    return;
  }
  Vmec& vmec = **maybe_vmec;

  absl::StatusOr<bool> ran = vmec.run();
  if (!ran.ok()) {
    state.SkipWithError("vmec.run() failed");
    return;
  }

  constexpr int kSignOfJacobian = -1;

  // ---- Timed loop: re-run only the post-solve output computation. ----
  for (auto _ : state) {
    OutputQuantities output_quantities = ComputeOutputQuantities(
        kSignOfJacobian, vmec.indata_, vmec.s_, vmec.fc_, vmec.constants_,
        vmec.t_, vmec.h_, vmec.mgrid_.mgrid_mode, vmec.r_, vmec.decomposed_x_,
        vmec.m_, vmec.p_, VmecCheckpoint::NONE,
        static_cast<VacuumPressureState>(vmec.get_ivac()), vmec.get_status(),
        vmec.get_iter2());
    benchmark::DoNotOptimize(output_quantities.wout.volume);
    benchmark::ClobberMemory();
  }
  state.SetLabel(kCase);
}

BENCHMARK(BM_ComputeOutputQuantities)->Name("ComputeOutputQuantities/cma");

}  // namespace
}  // namespace vmecpp

BENCHMARK_MAIN();

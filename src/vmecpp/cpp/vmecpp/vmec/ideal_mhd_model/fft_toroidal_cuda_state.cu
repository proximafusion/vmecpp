#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"
#include "vmecpp/common/util/os_compat.h"  // setenv on MSVC

namespace vmecpp {
// Definitions of the globals declared extern in the common header.
std::vector<double> g_residuals_invar_cache;
std::vector<double> g_residuals_precd_cache;
std::vector<double> g_fnorm_scalars_cache;
std::vector<double> g_jac_minmax_cache;
std::vector<double> g_pressure_scalars_cache;
std::vector<double> g_plasma_volume_cache;
std::vector<double> g_fnorm1_per_cfg_cache;
// Per-config normalized {fsqr,fsqz,fsql} snapshotted from the last iteration
// each configuration was active; see MutableFsqrPerCfgCache below.
std::vector<double> g_fsqr_per_cfg_cache;
int g_n_config_run = -1;
int g_batch_upscale_env = -1;
int g_batch_upscale_kernel_env = -1;
double g_ir_residual_sum = 1.0;
double g_ir_threshold    = 1e-5;
int    g_ir_staged       = -1;

// Prelude functions declared in the common header.
const std::vector<double>& GetResidualsPerCfgCacheInvar() {
  return g_residuals_invar_cache;
}
const std::vector<double>& GetResidualsPerCfgCachePrecd() {
  return g_residuals_precd_cache;
}
const std::vector<double>& GetFnormScalarsPerCfgCache() {
  return g_fnorm_scalars_cache;
}
const std::vector<double>& GetJacMinmaxPerCfgCache() {
  return g_jac_minmax_cache;
}
const std::vector<double>& GetPressureScalarsPerCfgCache() {
  return g_pressure_scalars_cache;
}
const std::vector<double>& GetPlasmaVolumePerCfgCache() {
  return g_plasma_volume_cache;
}
const std::vector<double>& GetFnorm1PerCfgCache() {
  return g_fnorm1_per_cfg_cache;
}
// Per-configuration normalized force residuals {fsqr, fsqz, fsql} from the last
// iteration each configuration was active. Unlike the live residual caches,
// masking a converged configuration does not zero its entry here, so the
// batched-output reconstruction can report each config's own converged
// residual instead of a post-mask zero. Layout: 3*n_cfg as
// [fsqr_c, fsqz_c, fsql_c, ...]. The iteration controller writes it through the
// mutable accessor at the per-cfg residual cadence.
const std::vector<double>& GetFsqrPerCfgCache() {
  return g_fsqr_per_cfg_cache;
}
std::vector<double>& MutableFsqrPerCfgCache() {
  return g_fsqr_per_cfg_cache;
}
// Raw upper estimate, in bytes, of the persistent device allocation at the
// given shape and configuration count, without the safety margin or the
// context cushion that the admission pre-flight adds on top. Shared by
// CudaVramBudgetCuda and by the per-Reshape bookkeeping that credits a
// follow-up run with the memory the next Reshape frees. The coefficient
// families: spectral-coefficient blocks (specs, force spectra, decomposed
// shadow, dealias intermediates, position/velocity/backup/final/prev
// state, lambda preconditioner, RZ tridiagonal rows), full-grid real-space
// arrays (outputs, forces, constraint terms, preconditioner inputs),
// half-grid arrays (jacobian, metric, fields, pressure), and the cuFFT
// scratch with its plan workspace.
long long CudaBudgetRawBytes(long long n_cfg, long long ns, long long mpol,
                             long long ntor, long long nZeta,
                             long long nThetaEff) {
  const long long mn = mpol * (ntor + 1);
  const long long spec = ns * mn;
  const long long nZnT = nZeta * nThetaEff;
  const long long full = ns * nZnT;
  const long long half = (ns - 1) * nZnT;
  const long long nhalf = nZeta / 2 + 1;
  const long long fft = 12 * ns * mpol * (2 * nhalf + nZeta);
  const long long doubles_per_cfg =
      61 * spec + 45 * full + 17 * half + (fft * 3) / 2 + fft;
  return n_cfg * doubles_per_cfg * (long long)sizeof(double);
}


// thread_local State to unblock multi-stream concurrency within a
// single process. Each thread gets its own CudaToroidalState (own stream,
// own buffers, own mutex). A subprocess-per-task execution pattern is
// unaffected (each subprocess has exactly one thread that ever calls into
// CUDA, so it sees its own thread_local). A future single-process multi-
// thread worker pattern would get true GPU concurrency for free.
//
// Caveat: there is no destructor on CudaToroidalState, so CUDA buffers
// allocated by a short-lived thread leak when that thread exits. For our
// long-lived worker patterns this is a non-issue (threads live until
// process exit). For experiments that spawn-and-die many threads, add an
// explicit cleanup or a dtor on Reshape's allocated buffers.
CudaToroidalState& State() {
  thread_local CudaToroidalState s;
  if (s.tk_env < 0) {
    const char* e = std::getenv("VMECPP_KERNEL_TIMING");
    s.tk_env = (e && std::atoi(e) > 0) ? 1 : 0;
    if (s.tk_env) {
      s.TKInit();
      // Auto-disable CUDA Graphs (seg-3, seg-4, fwd) when timing is on;
      // cudaEventRecord inside stream capture is illegal. _setenv_ ensures
      // the wrappers see graphs-off.
      setenv("VMECPP_UPDATE_GRAPH", "0", 1);  // disables seg-3
      setenv("VMECPP_SEG4_GRAPH",   "0", 1);  // disables seg-4
      setenv("VMECPP_SEG2_GRAPH",   "0", 1);  // disables seg-2
      setenv("VMECPP_FWD_GRAPH",    "0", 1);  // disables fwd
      std::fprintf(stderr, "[fft_toroidal_cuda] per-kernel cudaEvent timing "
                           "ENABLED (VMECPP_KERNEL_TIMING=1); graphs auto-"
                           "disabled to allow event capture\n");
      static bool atexit_installed = false;
      if (!atexit_installed) {
        atexit_installed = true;
        std::atexit([]() { State().TKDump(stderr); });
      }
    }
  }
  return s;
}


// Carson-Higham IR setters callable from ideal_mhd_model.cc. These bridge
// the per-iter residual sum from the host iteration controller into the
// file-scope globals above.
void SetIRResidualSum(double sum) {
  g_ir_residual_sum = sum;
  init_ir_env();
  if (g_ir_staged) {
    static int log_every = -1;
    if (log_every < 0) {
      const char* e = std::getenv("VMECPP_IR_LOG_EVERY");
      log_every = (e && std::atoi(e) > 0) ? std::atoi(e) : 0;
    }
    static long call_count = 0;
    ++call_count;
    if (log_every > 0 && (call_count == 1 || call_count % log_every == 0)) {
      int phase = (sum > g_ir_threshold) ? 1 : 0;
      std::fprintf(stderr, "[IR] iter=%ld fsq=%.3e phase=%s\n",
                   call_count, sum, phase ? "FP32" : "FP64");
    }
  }
}
int GetIRPhase() {
  init_ir_env();
  if (!g_ir_staged) return 0;
  return (g_ir_residual_sum > g_ir_threshold) ? 1 : 0;
}

// Resets the thread-local CudaToroidalState at the start of each
// Vmec::run so persistent device buffers carry nothing between runs
// in one process. Safe to invoke before the stream exists (no-op).
// True while the current Vmec::run solves a free-boundary input. The
// segment and whole-iteration CUDA graphs are disabled for the run: the
// vacuum block synchronizes the stream on every iteration and the edge
// force kernel toggles with the vacuum pressure state, both of which
// invalidate a captured kernel sequence.
bool g_free_boundary_run = false;

// Vacuum-edge force state for the current iteration. The segment-3
// graph contains the edge kernel, so a captured graph is only valid
// while this flag matches the value it was captured under.
bool g_vacuum_edge_run = false;

void SetVacuumEdgeCuda(int active) { g_vacuum_edge_run = (active != 0); }

// Run-level sync-elision flag (VMECPP_SYNC_ELIDE=K with K > 0), set by
// Vmec::Evolve when the run's K resolves. Distinct from the
// per-iteration sync_elide_iter flag: the free-boundary vacuum block
// consults this one to force full NESTOR updates at the window
// boundaries, where the partial-update cadence has no meaning.
bool g_sync_elide_run = false;

void SetSyncElideRunCuda(int active) { g_sync_elide_run = (active != 0); }

bool SyncElideRunActiveCuda() { return g_sync_elide_run; }

// Whole-iteration graph gate (VMECPP_ITER_GRAPH), run-scoped like the
// other gates: re-read at the start of every Vmec::run.
int g_iter_graph_env = -1;

// Residuals K-partition count, run-scoped: the auto default derives from
// the run's configuration count, and the partition count fixes the
// summation order of the residual reduction. A process-lifetime latch
// would carry the first run's partition geometry into later runs with a
// different configuration count, changing their residual rounding and,
// through the time-step damping, their trajectories.
int g_residuals_k_run = -1;

// int8-Ozaki limb count, run-scoped. VMECPP_SCATTER_I8_LIMBS=4 selects
// 28-bit operands (four 7-bit limbs, rel ~4e-9) at half the limb-plane
// traffic; the default 8 covers the FP64 mantissa. Under VMECPP_IR_STAGED
// the residual phase routes the choice per iteration: 4 limbs above the
// threshold (descent), 8 below. g_i8_limbs_last tracks the last
// dispatched width so a mid-run phase transition invalidates the
// whole-iteration graph, which otherwise replays the captured kernel.
int g_i8_limbs_env = -1;
int g_i8_limbs_last = 0;

void SetFreeBoundaryRunCuda(int enabled) {
  g_free_boundary_run = (enabled != 0);
}

void ResetCudaStateForNewVmecRun() {
  // Re-read the configuration count so a run can carry a different
  // VMECPP_N_CONFIG_MAX than its predecessor; the next Reshape resizes
  // the per-configuration buffers to the fresh value.
  {
    const char* env = std::getenv("VMECPP_N_CONFIG_MAX");
    g_n_config_run = (env != nullptr) ? std::max(1, std::atoi(env)) : 1;
  }
  // The multigrid-upscale, iteration-graph, residuals-partition, and
  // int8-limb gates re-read with the same per-run scope.
  g_batch_upscale_env = -1;
  g_batch_upscale_kernel_env = -1;
  g_iter_graph_env = -1;
  g_residuals_k_run = -1;
  g_i8_limbs_env = -1;
  g_i8_limbs_last = 0;
  g_ir_staged = -1;
  g_free_boundary_run = false;
  g_sync_elide_run = false;
  State().ResetForNewVmecRun();
}

// Segment-3 CUDA graph capture and replay coordinator.
//
// On successful replay of a previously captured graph the function
// returns true and the caller is expected to skip the wrapper
// invocations that the graph already executes; on a first capture
// pass or with graph capture disabled the function returns false and
// the caller proceeds to run the wrappers normally.
// Returns false if either:
//   (a) graphs are disabled (no env var): caller runs wrappers normally
//   (b) first call after Reshape: caller runs wrappers; their CUDA work
//       will be captured into the graph, then launched by EndUpdateSegment3.
// Whole-iteration graph. While its capture is open, the segment-graph
// Begin/End functions below run in passthrough: no replay (cudaGraphLaunch
// is illegal inside stream capture) and no nested capture; their raw
// kernel sequences feed the outer capture instead. Segment state machines
// do not advance during whole-iteration capture or replay.
bool g_iter_graph_capturing = false;

bool IterGraphEnabledCuda() {
  if (g_free_boundary_run) return false;
  if (g_iter_graph_env < 0) {
    const char* e = std::getenv("VMECPP_ITER_GRAPH");
    g_iter_graph_env = (e && std::atoi(e) > 0) ? 1 : 0;
    if (g_iter_graph_env) {
      std::fprintf(stderr,
                   "[fft_toroidal_cuda] whole-iteration CUDA Graph ENABLED "
                   "(VMECPP_ITER_GRAPH=1; replays sync-elided iterations)\n");
    }
  }
  return g_iter_graph_env != 0;
}

bool IterGraphCapturingCuda() { return g_iter_graph_capturing; }

bool IterGraphReplayCuda() {
  auto& S = State();
  if (!S.stream) return false;
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.iter_graph_captured) return false;
  cuda_check(cudaGraphLaunch(S.iter_graph_exec, S.stream),
             "iter graph launch (replay)");
  return true;
}

bool IterGraphBeginCaptureCuda() {
  auto& S = State();
  if (!S.stream) return false;
  std::lock_guard<std::mutex> lk(S.mu);
  if (S.iter_graph_captured || g_iter_graph_capturing) return false;
  if (S.iter_graph_warmups < CudaToroidalState::kIterGraphWarmups) {
    S.iter_graph_warmups += 1;
    return false;
  }
  cuda_check(cudaStreamBeginCapture(S.stream, cudaStreamCaptureModeGlobal),
             "begin capture iter graph");
  g_iter_graph_capturing = true;
  return true;
}

void IterGraphEndCaptureCuda() {
  auto& S = State();
  if (!g_iter_graph_capturing) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cuda_check(cudaStreamEndCapture(S.stream, &S.iter_graph),
             "end capture iter graph");
  cuda_check(cudaGraphInstantiate(&S.iter_graph_exec, S.iter_graph,
                                   nullptr, nullptr, 0),
             "instantiate iter graph");
  g_iter_graph_capturing = false;
  S.iter_graph_captured = true;
  // The captured iteration was enqueued, not executed; the first launch
  // runs it.
  cuda_check(cudaGraphLaunch(S.iter_graph_exec, S.stream),
             "iter graph launch (first)");
}

// Ends an open capture without instantiating and discards the graph. For
// early exits between the capture brackets; the iteration that was being
// captured re-runs uncaptured after the caller's normal control flow.
void AbortIterGraphCaptureCuda() {
  auto& S = State();
  if (!g_iter_graph_capturing) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaGraph_t scratch = nullptr;
  cudaStreamEndCapture(S.stream, &scratch);
  if (scratch) cudaGraphDestroy(scratch);
  g_iter_graph_capturing = false;
  S.iter_graph_warmups = 0;
}

void InvalidateIterationGraphCuda() {
  auto& S = State();
  std::lock_guard<std::mutex> lk(S.mu);
  if (S.iter_graph_exec) {
    cudaGraphExecDestroy(S.iter_graph_exec);
    S.iter_graph_exec = nullptr;
  }
  if (S.iter_graph) {
    cudaGraphDestroy(S.iter_graph);
    S.iter_graph = nullptr;
  }
  S.iter_graph_captured = false;
  S.iter_graph_warmups = 0;
}

// In-memory batch staging blocks (typed batch path). pybind
// run_batched_gpu hands the per-cfg spectral inputs and the per-cfg
// decomposed-position blocks here, both in the [sp][cfg][spec] layout the
// file pipeline uses; the staging loaders consume them ahead of the
// VMECPP_BATCH_INPUTS_FILE / VMECPP_BATCH_DEC_X_FILE fallback.
std::vector<double> g_batch_inputs_mem;
std::vector<double> g_batch_dec_x_mem;
int g_batch_mem_shape[4] = {0, 0, 0, 0};  // n_cfg, ns, mpol, ntor

// End-of-run converged spectra of a multi-configuration run, the same
// [sp][cfg][spec] block the VMECPP_BATCH_OUTPUTS_FILE dump writes. Filled
// by the end-of-run flush; read back by GetBatchOutputSpectraCuda.
std::vector<double> g_batch_outputs_mem;
int g_batch_outputs_shape[4] = {0, 0, 0, 0};

bool GetBatchOutputSpectraCuda(std::vector<double>* out, int* n_cfg, int* ns,
                               int* mpol, int* ntor) {
  if (g_batch_outputs_mem.empty()) return false;
  if (out) *out = g_batch_outputs_mem;
  if (n_cfg) *n_cfg = g_batch_outputs_shape[0];
  if (ns) *ns = g_batch_outputs_shape[1];
  if (mpol) *mpol = g_batch_outputs_shape[2];
  if (ntor) *ntor = g_batch_outputs_shape[3];
  return true;
}

// Per-config evaluated input profiles for distinct-mode batches: one set per
// configuration at the CURRENT multigrid level, flat [cfg][ns]. The host fills
// these from per-config RadialProfiles every level; the BContra / massH staging
// reads them to stage per-config instead of broadcasting the seed's profiles.
std::vector<double> g_batch_phipF, g_batch_phipH, g_batch_currH, g_batch_iotaH,
    g_batch_massH;
int g_batch_prof_ncfg = 0, g_batch_prof_nsh = 0, g_batch_prof_nsf = 0;

void SetBatchProfilesCuda(int n_cfg, int ns_h, int ns_f, const double* phipF,
                          const double* phipH, const double* currH,
                          const double* iotaH, const double* massH) {
  g_batch_phipF.assign(phipF, phipF + (size_t)n_cfg * ns_f);
  g_batch_phipH.assign(phipH, phipH + (size_t)n_cfg * ns_h);
  g_batch_currH.assign(currH, currH + (size_t)n_cfg * ns_h);
  g_batch_iotaH.assign(iotaH, iotaH + (size_t)n_cfg * ns_h);
  g_batch_massH.assign(massH, massH + (size_t)n_cfg * ns_h);
  g_batch_prof_ncfg = n_cfg;
  g_batch_prof_nsh = ns_h;
  g_batch_prof_nsf = ns_f;
}

void ClearBatchProfilesCuda() {
  g_batch_phipF.clear();
  g_batch_phipF.shrink_to_fit();
  g_batch_phipH.clear();
  g_batch_phipH.shrink_to_fit();
  g_batch_currH.clear();
  g_batch_currH.shrink_to_fit();
  g_batch_iotaH.clear();
  g_batch_iotaH.shrink_to_fit();
  g_batch_massH.clear();
  g_batch_massH.shrink_to_fit();
  g_batch_prof_ncfg = 0;
  g_batch_prof_nsh = 0;
  g_batch_prof_nsf = 0;
}

void SetBatchStagingCuda(const double* inputs, const double* dec_x,
                         int n_cfg, int ns, int mpol, int ntor) {
  const size_t total =
      (size_t)6 * n_cfg * ns * mpol * (size_t)(ntor + 1);
  g_batch_inputs_mem.assign(inputs, inputs + total);
  g_batch_dec_x_mem.assign(dec_x, dec_x + total);
  g_batch_mem_shape[0] = n_cfg;
  g_batch_mem_shape[1] = ns;
  g_batch_mem_shape[2] = mpol;
  g_batch_mem_shape[3] = ntor;
}

void ClearBatchStagingCuda() {
  // Drops the staged input blocks once the batched run that owns them
  // has finished, so a later run with a matching shape cannot consume
  // another batch's staging.
  g_batch_inputs_mem.clear();
  g_batch_inputs_mem.shrink_to_fit();
  g_batch_dec_x_mem.clear();
  g_batch_dec_x_mem.shrink_to_fit();
  g_batch_mem_shape[0] = 0;
  g_batch_mem_shape[1] = 0;
  g_batch_mem_shape[2] = 0;
  g_batch_mem_shape[3] = 0;
}

bool BeginUpdateSegment3GraphOrReplay() {
  auto& S0 = State();
  if (S0.seg3_graph_captured &&
      S0.seg3_vacuum_edge_at_capture != g_vacuum_edge_run) {
    // The vacuum pressure state changed since capture; the captured
    // kernel sequence no longer matches the iteration body.
    std::lock_guard<std::mutex> lk(S0.mu);
    if (S0.seg3_graph_exec) {
      cudaGraphExecDestroy(S0.seg3_graph_exec);
      S0.seg3_graph_exec = nullptr;
    }
    if (S0.seg3_graph) {
      cudaGraphDestroy(S0.seg3_graph);
      S0.seg3_graph = nullptr;
    }
    S0.seg3_graph_captured = false;
    S0.seg3_warmup_calls = 0;
  }
  static int env_enable = -1;
  if (env_enable < 0) {
    const char* e = std::getenv("VMECPP_UPDATE_GRAPH");
    // Default ON; set VMECPP_UPDATE_GRAPH=0 to disable. Validated bit-exact at
    // N=1 and N=16 against baseline; segment 3 is cuFFT-free so the forward-
    // graph regression (~6%) doesn't apply.
    env_enable = (e && std::atoi(e) == 0) ? 0 : 1;
    if (!env_enable) {
      std::fprintf(stderr, "[fft_toroidal_cuda] segment-3 CUDA Graph disabled "
                           "(VMECPP_UPDATE_GRAPH=0)\n");
    }
  }
  if (!env_enable) return false;
  if (g_iter_graph_capturing) return false;
  auto& S = State();
  if (!S.stream) return false;
  std::lock_guard<std::mutex> lk(S.mu);
  if (S.seg3_graph_captured) {
    // Replay path: launch the captured graph and tell caller to skip wrappers.
    cuda_check(cudaGraphLaunch(S.seg3_graph_exec, S.stream),
               "seg3 graph launch (replay)");
    return true;
  }
  // Warmup: skip capture on the first WARMUP_N calls so lazy cudaMalloc
  // inside Ensure*Buffers wrappers fire normally (cudaMalloc is forbidden
  // inside stream capture). After warmup, all buffers exist.
  constexpr int WARMUP_N = 2;
  if (S.seg3_warmup_calls < WARMUP_N) {
    S.seg3_warmup_calls += 1;
    return false;  // run wrappers without capture
  }
  // Capture path: begin capture, then return false so caller runs wrappers.
  cuda_check(cudaStreamBeginCapture(S.stream, cudaStreamCaptureModeGlobal),
             "begin capture seg3 graph");
  S.seg3_in_capture = true;
  return false;
}

// Called at end of segment 3. If we were capturing, end capture, instantiate,
// and launch the just-captured graph. If we already replayed (Begin returned
// true), this is a no-op.
void EndUpdateSegment3GraphOrLaunch() {
  static int env_enable = -1;
  if (env_enable < 0) {
    const char* e = std::getenv("VMECPP_UPDATE_GRAPH");
    env_enable = (e && std::atoi(e) == 0) ? 0 : 1;  // default ON
  }
  if (!env_enable) return;
  if (g_iter_graph_capturing) return;
  auto& S = State();
  if (!S.stream) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (S.seg3_in_capture) {
    cuda_check(cudaStreamEndCapture(S.stream, &S.seg3_graph),
               "end capture seg3 graph");
    cuda_check(cudaGraphInstantiate(&S.seg3_graph_exec, S.seg3_graph,
                                     nullptr, nullptr, 0),
               "graph instantiate seg3");
    S.seg3_graph_captured = true;
    S.seg3_vacuum_edge_at_capture = g_vacuum_edge_run;
    S.seg3_in_capture = false;
    // Launch the just-captured graph (during capture, kernels were recorded
    // but did not execute).
    cuda_check(cudaGraphLaunch(S.seg3_graph_exec, S.stream),
               "seg3 graph launch (first)");
  }
  // else: replay already done in Begin.
}

// Segment-4 CUDA graph capture and replay coordinator.
//
// The captured chain consists of ApplyM1PreconditionerCuda,
// AssembleRZPreconditionerCuda, ApplyRZPreconditionerCuda, and
// ApplyLambdaPreconditionerCuda. Each of these wrappers issues
// kernel launches only, performs no host synchronization or host
// memory access, and triggers no device allocation once the warmup
// pass has completed. The four wrappers execute consecutively
// between the stream synchronization at the end of the first
// ResidualsCuda invocation and the stream synchronization at the
// start of the second, satisfying the conditions for CUDA graph
// capture.
//
// The only kernel argument that varies across iterations is jMax,
// whose value depends on the free-boundary flag lfreeb and on the
// vacuum-pressure-state transitions. The most recent captured value
// of jMax is retained, and a mismatch with the current call triggers
// destruction and re-capture of the segment graph. In the canonical
// fixed-boundary benchmark jMax remains constant throughout the run,
// so a single captured graph services every iteration.
//
// Enablement is governed by the VMECPP_SEG4_GRAPH environment
// variable, which defaults to active when unset and is disabled when
// set to zero.
bool BeginUpdateSegment4GraphOrReplay(int jMax) {
  static int env_enable = -1;
  if (env_enable < 0) {
    const char* e = std::getenv("VMECPP_SEG4_GRAPH");
    env_enable = (e && std::atoi(e) == 0) ? 0 : 1;  // default ON
    if (!env_enable) {
      std::fprintf(stderr, "[fft_toroidal_cuda] segment-4 CUDA Graph disabled "
                           "(VMECPP_SEG4_GRAPH=0)\n");
    }
  }
  if (!env_enable) return false;
  if (g_iter_graph_capturing) return false;
  auto& S = State();
  if (!S.stream) return false;
  std::lock_guard<std::mutex> lk(S.mu);
  // Invalidate captured graph if jMax changed (rare; fires on vacuum
  // pressure state transition for free-boundary runs).
  if (S.seg4_graph_captured && jMax != S.seg4_last_jMax) {
    cudaGraphExecDestroy(S.seg4_graph_exec);
    cudaGraphDestroy(S.seg4_graph);
    S.seg4_graph_exec = nullptr;
    S.seg4_graph = nullptr;
    S.seg4_graph_captured = false;
    S.seg4_warmup_calls = 0;
    // The whole-iteration graph embeds the seg4 kernel sequence; a jMax
    // change invalidates it for the same reason.
    if (S.iter_graph_exec) {
      cudaGraphExecDestroy(S.iter_graph_exec);
      S.iter_graph_exec = nullptr;
    }
    if (S.iter_graph) {
      cudaGraphDestroy(S.iter_graph);
      S.iter_graph = nullptr;
    }
    S.iter_graph_captured = false;
    S.iter_graph_warmups = 0;
  }
  S.seg4_last_jMax = jMax;
  if (S.seg4_graph_captured) {
    cuda_check(cudaGraphLaunch(S.seg4_graph_exec, S.stream),
               "seg4 graph launch (replay)");
    return true;
  }
  constexpr int WARMUP_N = 2;
  if (S.seg4_warmup_calls < WARMUP_N) {
    S.seg4_warmup_calls += 1;
    return false;
  }
  cuda_check(cudaStreamBeginCapture(S.stream, cudaStreamCaptureModeGlobal),
             "begin capture seg4 graph");
  S.seg4_in_capture = true;
  return false;
}

void EndUpdateSegment4GraphOrLaunch() {
  static int env_enable = -1;
  if (env_enable < 0) {
    const char* e = std::getenv("VMECPP_SEG4_GRAPH");
    env_enable = (e && std::atoi(e) == 0) ? 0 : 1;
  }
  if (!env_enable) return;
  if (g_iter_graph_capturing) return;
  auto& S = State();
  if (!S.stream) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (S.seg4_in_capture) {
    cuda_check(cudaStreamEndCapture(S.stream, &S.seg4_graph),
               "end capture seg4 graph");
    cuda_check(cudaGraphInstantiate(&S.seg4_graph_exec, S.seg4_graph,
                                     nullptr, nullptr, 0),
               "graph instantiate seg4");
    S.seg4_graph_captured = true;
    S.seg4_in_capture = false;
    cuda_check(cudaGraphLaunch(S.seg4_graph_exec, S.stream),
               "seg4 graph launch (first)");
  }
}

// Segment-2 CUDA graph capture and replay coordinator.
//
// The captured chain comprises six kernel-only wrappers:
// ComputeMetricElementsCuda, UpdateDifferentialVolumeCuda,
// ComputeBContraCuda, ComputeBCoCuda, PressureAndEnergiesCuda, and
// RadialForceBalanceCuda. The chain executes between the stream
// synchronization at the end of ComputeJacobianCuda and the entry
// to the preconditioner-update block that precedes the segment-3
// chain. No host synchronization or host memory access occurs in
// this window once the iter-one initialization host-to-device
// transfers have completed.
//
// Enablement is governed by the VMECPP_SEG2_GRAPH environment
// variable, which defaults to active when unset and is disabled when
// set to zero.
//
// defer_capture: the caller's segment body performs a host synchronization
// on this iteration, which is illegal inside stream capture; run the
// wrappers uncaptured. An already-captured graph still replays.
bool BeginUpdateSegment2GraphOrReplay(bool defer_capture) {
  static int env_enable = -1;
  if (env_enable < 0) {
    const char* e = std::getenv("VMECPP_SEG2_GRAPH");
    env_enable = (e && std::atoi(e) == 0) ? 0 : 1;
    if (!env_enable) {
      std::fprintf(stderr, "[fft_toroidal_cuda] segment-2 CUDA Graph disabled "
                           "(VMECPP_SEG2_GRAPH=0)\n");
    }
  }
  if (!env_enable) return false;
  if (g_iter_graph_capturing) return false;
  auto& S = State();
  if (!S.stream) return false;
  std::lock_guard<std::mutex> lk(S.mu);
  if (S.seg2_graph_captured) {
    cuda_check(cudaGraphLaunch(S.seg2_graph_exec, S.stream),
               "seg2 graph launch (replay)");
    return true;
  }
  constexpr int WARMUP_N = 2;
  if (S.seg2_warmup_calls < WARMUP_N) {
    S.seg2_warmup_calls += 1;
    return false;
  }
  if (defer_capture) return false;
  cuda_check(cudaStreamBeginCapture(S.stream, cudaStreamCaptureModeGlobal),
             "begin capture seg2 graph");
  S.seg2_in_capture = true;
  return false;
}

void EndUpdateSegment2GraphOrLaunch() {
  static int env_enable = -1;
  if (env_enable < 0) {
    const char* e = std::getenv("VMECPP_SEG2_GRAPH");
    env_enable = (e && std::atoi(e) == 0) ? 0 : 1;
  }
  if (!env_enable) return;
  if (g_iter_graph_capturing) return;
  auto& S = State();
  if (!S.stream) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (S.seg2_in_capture) {
    cuda_check(cudaStreamEndCapture(S.stream, &S.seg2_graph),
               "end capture seg2 graph");
    cuda_check(cudaGraphInstantiate(&S.seg2_graph_exec, S.seg2_graph,
                                     nullptr, nullptr, 0),
               "graph instantiate seg2");
    S.seg2_graph_captured = true;
    S.seg2_in_capture = false;
    cuda_check(cudaGraphLaunch(S.seg2_graph_exec, S.stream),
               "seg2 graph launch (first)");
  }
}

void FlushFwdGeomScalarsToHost(double* r1_e, double* r1_o, double* z1_e) {
  auto& S = State();
  if (!S.fwd_geom_pending) return;
  int outer_idx = S.fwd_geom_outer_idx;
  int inner_idx = S.fwd_geom_inner_idx;
  r1_e[outer_idx] = S.h_geom_scalars[0];
  r1_o[outer_idx] = S.h_geom_scalars[1];
  r1_e[inner_idx] = S.h_geom_scalars[2];
  r1_o[inner_idx] = S.h_geom_scalars[3];
  r1_e[0]         = S.h_geom_scalars[4];
  z1_e[0]         = S.h_geom_scalars[5];
  S.fwd_geom_pending = false;
}

void DiagCfg01DiffCuda(const double* d_buf, int per_cfg_size,
                       const char* label) {
  static int trace_env = -1;
  if (trace_env < 0) {
    const char* e = std::getenv("VMECPP_TRACE_CFG_DIFF");
    trace_env = (e && std::atoi(e) > 0) ? 1 : 0;
  }
  if (!trace_env) return;
  auto& S = State();
  if (!S.stream || S.n_config_max < 2 || per_cfg_size <= 0) return;
  // No mu lock here: this is called from inside wrappers that already hold
  // S.mu (e.g. FourierToReal3DSymmFastPoloidalCuda, ComputeJacobianCuda).
  // std::mutex is non-recursive; re-locking would deadlock.
  // Reuse d_scalar (1 double) as the output target.
  S.EnsureScalarBuffer();
  k_cfg01_max_abs_diff<<<1, 256, 0, S.stream>>>(
      per_cfg_size, d_buf, S.d_scalar);
  cuda_check(cudaGetLastError(), "k_cfg01_max_abs_diff launch");
  double h_max = 0.0;
  cuda_check(cudaMemcpyAsync(&h_max, S.d_scalar, sizeof(double),
                              cudaMemcpyDeviceToHost, S.stream),
             "d2h diag cfg01");
  cuda_check(cudaStreamSynchronize(S.stream), "diag cfg01 sync");
  std::fprintf(stderr, "[diag-cfg01 %s] max|cfg0-cfg1| = %.15e\n",
               label, h_max);
}


}  // namespace vmecpp

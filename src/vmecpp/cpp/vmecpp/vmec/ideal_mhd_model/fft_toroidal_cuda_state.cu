#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"

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
int g_n_config_run = 1;
bool g_batch_distinct_run = false;

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
  return s;
}

// True while the current Vmec::run solves a free-boundary input, whose vacuum
// block reads bucoH on the host every iteration.
bool g_free_boundary_run = false;

// Vacuum-edge force state for the current iteration. The segment-3
// graph contains the edge kernel, so a captured graph is only valid
// while this flag matches the value it was captured under.
bool g_vacuum_edge_run = false;

void SetVacuumEdgeCuda(int active) { g_vacuum_edge_run = (active != 0); }

// Residuals K-partition count, run-scoped: the auto default derives from
// the run's configuration count, and the partition count fixes the
// summation order of the residual reduction. A process-lifetime latch
// would carry the first run's partition geometry into later runs with a
// different configuration count, changing their residual rounding and,
// through the time-step damping, their trajectories.
int g_residuals_k_run = -1;

void SetFreeBoundaryRunCuda(int enabled) {
  g_free_boundary_run = (enabled != 0);
}

// Resets the thread-local CudaToroidalState at the start of each Vmec::run so
// persistent device buffers carry nothing between runs in one process, and
// sets the run's configuration count and whether its configurations carry
// their own inputs. Safe to invoke before the stream exists.
void ResetCudaStateForNewVmecRun(int n_config, bool distinct) {
  g_n_config_run = std::max(1, n_config);
  g_batch_distinct_run = distinct;
  g_residuals_k_run = -1;
  g_free_boundary_run = false;
  State().ResetForNewVmecRun();
}

// In-memory batch staging blocks. pybind run_batched_gpu hands the per-cfg
// spectral inputs and the per-cfg decomposed-position blocks here, both in
// [sp][cfg][spec] layout.
std::vector<double> g_batch_inputs_mem;
std::vector<double> g_batch_dec_x_mem;
int g_batch_mem_shape[4] = {0, 0, 0, 0};  // n_cfg, ns, mpol, ntor

// End-of-run converged spectra of a multi-configuration run in
// [sp][cfg][spec] layout. Filled by the end-of-run flush; read back by
// GetBatchOutputSpectraCuda.
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

// Segment-3 CUDA graph: launches the captured graph and returns true, or
// returns false so the caller runs the wrappers, which are captured once the
// warmup calls are done; EndUpdateSegment3GraphOrLaunch closes the capture.
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
bool BeginUpdateSegment4GraphOrReplay(int jMax) {
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
// defer_capture: the caller's segment body performs a host synchronization
// on this iteration, which is illegal inside stream capture; run the
// wrappers uncaptured. An already-captured graph still replays.
bool BeginUpdateSegment2GraphOrReplay(bool defer_capture) {
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

}  // namespace vmecpp

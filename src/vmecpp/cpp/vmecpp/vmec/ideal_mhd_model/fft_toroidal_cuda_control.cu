#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"

namespace vmecpp {

// ============================================================================
// ResidualsCuda
//
// Mirror of FourierForces::residuals() against the device-resident decomposed
// shadow S.d_decomposed_frcc/frss/fzsc/fzcs/flsc/flcs. Single-block reduction
// kernel writes 3 doubles [fResR, fResZ, fResL] to S.d_residuals_partial; one
// small D2H + stream sync returns them.
//
// Honors the same jMaxRZ (with includeEdgeRZForces) and jMaxIncludeBoundary
// range logic as the CPU loop. Stellarator-symmetric only (lasym = false),
// matching the supported scope of the CUDA build.
// ============================================================================
void ResidualsCuda(const RadialPartitioning& r, const Sizes& s,
                    const FlowControl& fc, bool includeEdgeRZForces,
                    double& fResR_out, double& fResZ_out, double& fResL_out,
                    bool is_precd) {
  auto& S = State();
  int ns_dec_local =
      (r.nsMaxF1 == fc.ns) ? (fc.ns - r.nsMinF) : (r.nsMaxF - r.nsMinF);
  int mpol = s.mpol;
  int ntor = s.ntor;
  if (ns_dec_local <= 0) {
    fResR_out = 0.0; fResZ_out = 0.0; fResL_out = 0.0;
    return;
  }
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  S.EnsureResidualsBuffer();

  // FourierForces::residuals thresholds, shifted by nsMin_.
  int nsMin_ = r.nsMinF;
  int nsMax_ = r.nsMaxF;
  int ns = fc.ns;
  int jMaxRZ = std::min(nsMax_, ns - 1);
  if (includeEdgeRZForces && r.nsMaxF1 == ns) {
    jMaxRZ = ns;
  }
  int jMaxIncludeBoundary = nsMax_;
  if (r.nsMaxF1 == ns) {
    jMaxIncludeBoundary = ns;
  }
  int jLocal_max_rz = jMaxRZ - nsMin_;
  int jLocal_max_boundary = jMaxIncludeBoundary - nsMin_;
  if (jLocal_max_rz < 0) jLocal_max_rz = 0;
  if (jLocal_max_boundary < 0) jLocal_max_boundary = 0;
  if (jLocal_max_boundary > ns_dec_local) jLocal_max_boundary = ns_dec_local;
  if (jLocal_max_rz > ns_dec_local) jLocal_max_rz = ns_dec_local;

  // Batched execution: launch n_config_max blocks (one per config). Each writes 3
  // scalars at residuals_partial[config*3:].
  // K sub-blocks per configuration, K = max(1, 16 / n_config_max), so the
  // reduction spans about 16 blocks in total.
  if (g_residuals_k_run < 0) {
    const int auto_k = CudaToroidalState::kResidualsKPartitions / S.n_config_max;
    g_residuals_k_run = (auto_k < 1) ? 1 : auto_k;
  }
  const int residuals_k_env = g_residuals_k_run;
  if (residuals_k_env > 1) {
    // Multi-block partials path. Grid (K, n_config). Each block reduces
    // 1/K of the index space. Then finalize collapses K partials into
    // d_residuals_partial.
    dim3 partials_grid(residuals_k_env, S.n_config_max);
    k_residuals_par_K<<<partials_grid, 256, 0, st>>>(
        S.n_config_max, S.ns_local_cached,
        jLocal_max_rz, jLocal_max_boundary, mpol, ntor, s.lthreed,
        residuals_k_env,
        S.d_decomposed_frcc, S.d_decomposed_frss,
        S.d_decomposed_fzsc, S.d_decomposed_fzcs,
        S.d_decomposed_flsc, S.d_decomposed_flcs,
        S.d_residuals_partials_K,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_residuals_par_K launch");
    // Finalize: grid (n_config), TPB=32 (covers up to 32 partitions; we cap
    // K at 16 so half the lanes load zeros and the butterfly still gives
    // the correct sum).
    k_residuals_finalize_K<<<S.n_config_max, 32, 0, st>>>(
        S.n_config_max, residuals_k_env,
        S.d_residuals_partials_K,
        S.d_residuals_partial,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_residuals_finalize_K launch");
  } else {
    k_residuals_par<<<S.n_config_max, 256, 0, st>>>(
        S.n_config_max, S.ns_local_cached,
        jLocal_max_rz, jLocal_max_boundary, mpol, ntor, s.lthreed,
        S.d_decomposed_frcc, S.d_decomposed_frss,
        S.d_decomposed_fzsc, S.d_decomposed_fzcs,
        S.d_decomposed_flsc, S.d_decomposed_flcs,
        S.d_residuals_partial,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_residuals_par launch");
  }
  // Device-side convergence flag on normalized residuals (read through
  // GetConvergenceFlag). The normalization inputs are the persistent
  // per-cfg buffers (force-norm sums, energy scalars, plasma volumes)
  // plus the cached per-run lamscale; see k_check_convergence.
  if (!is_precd && S.d_conv_flag) {
    k_check_convergence<<<S.n_config_max, 1, 0, st>>>(
        S.n_config_max, S.d_residuals_partial,
        S.d_fnorm_scalars, S.d_pressure_scalars, S.d_scalar,
        S.lamscale_cached, fc.ftolv,
        S.d_conv_flag, S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_check_convergence launch");
    cuda_check(cudaMemcpyAsync(S.h_conv_flag_pinned, S.d_conv_flag,
                                sizeof(std::uint8_t) * S.n_config_max,
                                cudaMemcpyDeviceToHost, st),
               "conv_flag d2h async");
  }

  // Per-cfg cache: the 3-double transfer widens to 3*n_cfg doubles into
  // a static cache under the one synchronization this call already pays
  // (the transfer delta is ~100 ns at n_cfg = 64; the sync wait
  // dominates). Single-cfg behavior is preserved (fResR_out = cache[0]),
  // and the per-cfg consumers in evalFResInvar/Precd read the cache via
  // GetResidualsPerCfgCacheInvar / GetResidualsPerCfgCachePrecd with no
  // additional transfer or sync.
  int n_cfg = S.n_config_max;
  std::vector<double>& cache = is_precd ? g_residuals_precd_cache
                                          : g_residuals_invar_cache;
  if ((int)cache.size() != 3 * n_cfg) {
    cache.assign(3 * n_cfg, 0.0);
  }
  cuda_check(cudaMemcpyAsync(cache.data(), S.d_residuals_partial,
                              (size_t)3 * n_cfg * sizeof(double),
                              cudaMemcpyDeviceToHost, st),
             "d2h residuals (per-cfg cache)");
  cuda_check(cudaStreamSynchronize(st), "residuals stream sync");
  fResR_out = cache[0];
  fResZ_out = cache[1];
  fResL_out = cache[2];
}

// ============================================================================
// FlushDecomposedToHostCuda
//
// Flush the decomposed shadow S.d_decomposed_* back to host m_decomposed_f
// buffers. Consolidates the per-wrapper D2H+sync that
// DecomposeAndConstrainCuda and the ApplyM1/Lambda/RZ wrappers would
// otherwise pay individually. The iteration body does not call this per
// iteration (the device shadow is the authoritative state); the entry point
// serves the controller's explicit flush sites and diagnostics.
// ============================================================================

// (anonymous namespace hoisted across the split translation units)
// Declared ahead of its definition below so the d_pts_x initialization
// sites can arm the backup mirror as soon as the stage's initial state
// is resident on the device.
void EnsurePTSBackupBuffers(CudaToroidalState& S);
// (anonymous namespace hoisted across the split translation units)

// When set, PerformTimeStepCuda runs only its buffer-init section
// (EnsurePTSBuffers + the multigrid upscale / per-cfg dec_x load /
// broadcast fallback) and returns before the time-step kernel. Used by
// PrepareStagePtsXCuda.
bool g_pts_init_only = false;

void PerformTimeStepCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double velocity_scale, double conjugation_parameter, double time_step,
    double fnorm1, int iter_phase,
    double* m_dec_v_rcc, double* m_dec_v_rss,
    double* m_dec_v_zsc, double* m_dec_v_zcs,
    double* m_dec_v_lsc, double* m_dec_v_lcs,
    double* m_dec_x_rcc, double* m_dec_x_rss,
    double* m_dec_x_zsc, double* m_dec_x_zcs,
    double* m_dec_x_lsc, double* m_dec_x_lcs) {
  auto& S = State();
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  if (ns_con_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  S.EnsurePTSBuffers(ns_con_local, ns_local, mpol, ntor);

  size_t v_bytes_one = sizeof(double) * ns_con_local * mpol * (ntor + 1);
  size_t x_bytes_one = sizeof(double) * ns_local     * mpol * (ntor + 1);

  // One-shot init: v starts as zero, x starts from host m_decomposed_x.
  if (!S.pts_v_initialized) {
    // cudaMalloc generally returns zeroed memory but be explicit: zero ALL N
    // config slots so cfg 1..N-1 are clean at first call (in case the
    // broadcast convergence-fix is needed for v state, too).
    size_t v_bytes_all = sizeof(double) * (size_t)S.n_config_max * ns_con_local
                          * mpol * (ntor + 1);
    cuda_check(cudaMemsetAsync(S.d_pts_v_rcc, 0, v_bytes_all, st), "memset v_rcc");
    cuda_check(cudaMemsetAsync(S.d_pts_v_rss, 0, v_bytes_all, st), "memset v_rss");
    cuda_check(cudaMemsetAsync(S.d_pts_v_zsc, 0, v_bytes_all, st), "memset v_zsc");
    cuda_check(cudaMemsetAsync(S.d_pts_v_zcs, 0, v_bytes_all, st), "memset v_zcs");
    cuda_check(cudaMemsetAsync(S.d_pts_v_lsc, 0, v_bytes_all, st), "memset v_lsc");
    cuda_check(cudaMemsetAsync(S.d_pts_v_lcs, 0, v_bytes_all, st), "memset v_lcs");
    S.pts_v_initialized = true;
  }
  if (!S.pts_x_initialized) {
    bool initialized_by_upscale = false;
    // Multigrid-stage transition path: if EnsurePTSBuffers captured the
    // pre-Reshape d_pts_x into d_pts_x_prev because ns_local changed,
    // run the per-cfg radial-interp kernel to upscale the snapshot into
    // the freshly-allocated d_pts_x. This preserves per-cfg state across
    // multigrid stages in distinct mode (host m_decomposed_x is single-cfg
    // and would otherwise wipe cfg != 0 via the broadcast fallback below).
    if (g_batch_distinct_run && S.pts_x_prev_valid && S.pts_x_prev_ns > 0 &&
        S.pts_x_prev_size > 0 && S.d_scalxc) {
      int ns_old = S.pts_x_prev_ns;
      int ns_new = ns_local;
      // Per-cfg upscale on device, bit-identical to
      // Vmec::InterpolateToNextMultigridStep. Sequence per cfg: scale the
      // old stage by its scalxc, extrapolate the odd-m axis on the scaled
      // values, interpolate linearly in s dividing by the new stage's
      // scalxc, zero the odd-m axis rows.
      int scalxc_len_per_cfg = ns_new * 2;
      if (!S.scalxc_prev_valid || !S.d_scalxc_prev) {
        throw std::runtime_error(
            "CUDA multigrid upscale without the previous stage's scalxc");
      } else {
        // Device upscale, bit-identical to
        // Vmec::InterpolateToNextMultigridStep: scale the previous stage
        // by its scalxc (the caller-side decomposeInto pass), extrapolate
        // the odd-m axis on the scaled values, interpolate linearly in s
        // dividing by the new stage's scalxc, and zero the odd-m axis
        // rows. The snapshot is consumed in place; no host round trip.
        const int n_cfg = S.n_config_max;
        const int TPB = 32;
        dim3 tpb(TPB, 1, 1);
        dim3 sc_b((ntor + 1 + TPB - 1) / TPB, mpol, ns_old * n_cfg);
        k_scale_prev_by_scalxc<<<sc_b, tpb, 0, st>>>(
            n_cfg, ns_old, mpol, ntor, S.scalxc_prev_len,
            S.d_pts_x_prev_rcc, S.d_pts_x_prev_rss, S.d_pts_x_prev_zsc,
            S.d_pts_x_prev_zcs, S.d_pts_x_prev_lsc, S.d_pts_x_prev_lcs,
            S.d_scalxc_prev);
        cuda_check(cudaGetLastError(), "k_scale_prev_by_scalxc launch");
        dim3 ax_b((ntor + 1 + TPB - 1) / TPB, (mpol + 1) / 2, n_cfg);
        k_axis_extrapolate_odd_m_prev<<<ax_b, tpb, 0, st>>>(
            n_cfg, ns_old, mpol, ntor,
            S.d_pts_x_prev_rcc, S.d_pts_x_prev_rss, S.d_pts_x_prev_zsc,
            S.d_pts_x_prev_zcs, S.d_pts_x_prev_lsc, S.d_pts_x_prev_lcs);
        cuda_check(cudaGetLastError(),
                   "k_axis_extrapolate_odd_m_prev launch");
        dim3 in_b((ntor + 1 + TPB - 1) / TPB, mpol, ns_new * n_cfg);
        k_radial_interpolate_pts_x<<<in_b, tpb, 0, st>>>(
            n_cfg, ns_old, ns_new, mpol, ntor, scalxc_len_per_cfg,
            S.d_pts_x_prev_rcc, S.d_pts_x_prev_rss, S.d_pts_x_prev_zsc,
            S.d_pts_x_prev_zcs, S.d_pts_x_prev_lsc, S.d_pts_x_prev_lcs,
            S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
            S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs,
            S.d_scalxc);
        cuda_check(cudaGetLastError(), "k_radial_interpolate_pts_x launch");
      }
      // Host-shadow refresh: copy cfg 0's upscaled state into the host
      // m_decomposed_x arrays so host-side consumers (rzNorm / fNorm1,
      // restart backups) read the state the device evolves. The device
      // upscale is bit-identical to the host interpolation, so this is a
      // consistency guarantee rather than a correction; cfgs > 0 have no
      // host shadow.
      if (m_dec_x_rcc) {
        size_t up_spec_doubles = (size_t)ns_new * mpol * (ntor + 1);
        size_t up_bytes = up_spec_doubles * sizeof(double);
        cuda_check(cudaMemcpyAsync(m_dec_x_rcc, S.d_pts_x_rcc, up_bytes,
                                    cudaMemcpyDeviceToHost, st),
                   "d2h upscaled cfg0 rcc");
        cuda_check(cudaMemcpyAsync(m_dec_x_zsc, S.d_pts_x_zsc, up_bytes,
                                    cudaMemcpyDeviceToHost, st),
                   "d2h upscaled cfg0 zsc");
        cuda_check(cudaMemcpyAsync(m_dec_x_lsc, S.d_pts_x_lsc, up_bytes,
                                    cudaMemcpyDeviceToHost, st),
                   "d2h upscaled cfg0 lsc");
        // The rss/zcs/lcs (odd-parity) spectra are absent for axisymmetric
        // (ntor = 0) inputs: neither the host shadow nor the device slot is
        // allocated, so skip their transfer rather than issue an invalid copy.
        if (m_dec_x_rss && S.d_pts_x_rss)
          cuda_check(cudaMemcpyAsync(m_dec_x_rss, S.d_pts_x_rss, up_bytes,
                                      cudaMemcpyDeviceToHost, st),
                     "d2h upscaled cfg0 rss");
        if (m_dec_x_zcs && S.d_pts_x_zcs)
          cuda_check(cudaMemcpyAsync(m_dec_x_zcs, S.d_pts_x_zcs, up_bytes,
                                      cudaMemcpyDeviceToHost, st),
                     "d2h upscaled cfg0 zcs");
        if (m_dec_x_lcs && S.d_pts_x_lcs)
          cuda_check(cudaMemcpyAsync(m_dec_x_lcs, S.d_pts_x_lcs, up_bytes,
                                      cudaMemcpyDeviceToHost, st),
                     "d2h upscaled cfg0 lcs");
        cuda_check(cudaStreamSynchronize(st), "upscaled cfg0 flush sync");
      }
      auto free_if = [](double*& p) { if (p) { cudaFree(p); p = nullptr; } };
      free_if(S.d_pts_x_prev_rcc); free_if(S.d_pts_x_prev_rss);
      free_if(S.d_pts_x_prev_zsc); free_if(S.d_pts_x_prev_zcs);
      free_if(S.d_pts_x_prev_lsc); free_if(S.d_pts_x_prev_lcs);
      free_if(S.d_scalxc_prev);
      S.pts_x_prev_valid = false;
      S.scalxc_prev_valid = false;
      initialized_by_upscale = true;
    }
    if (!initialized_by_upscale) {
    // Distinct batches: load each configuration's decomposed position from
    // the run_batched_gpu staging block ([sp][cfg][spec] layout).
    // RecomposeToPhysicalCuda loads the same block into the same d_pts_x_*
    // destinations, so whichever runs first on iteration 1 populates them and
    // the other's init branch is skipped via pts_x_initialized.
    size_t one_spec_doubles = (size_t)ns_local * mpol * (ntor + 1);
    const bool dec_x_loaded =
        !g_batch_dec_x_mem.empty() && g_batch_mem_shape[0] == S.n_config_max &&
        g_batch_mem_shape[1] == ns_local && g_batch_mem_shape[2] == mpol &&
        g_batch_mem_shape[3] == ntor && S.n_config_max > 1;
    double* dst_x[6] = {S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
                        S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs};
    if (dec_x_loaded) {
      for (int sp = 0; sp < 6; ++sp) {
        for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
          const double* src =
              g_batch_dec_x_mem.data() +
              (size_t)sp * S.n_config_max * one_spec_doubles +
              (size_t)cfg * one_spec_doubles;
          cuda_check(cudaMemcpyAsync(
              dst_x[sp] + (size_t)cfg * one_spec_doubles,
              src, x_bytes_one, cudaMemcpyHostToDevice, st),
              "h2d per-cfg dec_x (PerformTimeStep)");
        }
      }
    } else {
      const double* src_x[6] = {m_dec_x_rcc, m_dec_x_rss,
                                m_dec_x_zsc, m_dec_x_zcs,
                                m_dec_x_lsc, m_dec_x_lcs};
      for (int i = 0; i < 6; ++i) {
        for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
          double* dst = dst_x[i] + (size_t)cfg * one_spec_doubles;
          if (src_x[i] == nullptr) {
            // lthreed-only spectrum absent for an axisymmetric (ntor=0) run;
            // the device slot stays zero.
            cuda_check(cudaMemsetAsync(dst, 0, x_bytes_one, st),
                       "memset pts x init (2d)");
          } else {
            cuda_check(cudaMemcpyAsync(dst, src_x[i], x_bytes_one,
                                        cudaMemcpyHostToDevice, st),
                       "h2d pts x init");
          }
        }
      }
    }
    }  // !initialized_by_upscale
    S.pts_x_initialized = true;
    // Arm the device backup with the stage's initial state so a
    // bad-Jacobian or bad-progress restore that fires before the first
    // periodic backup rewinds to a valid geometry. Mirrors the host
    // path, whose backup is synced from decomposed_x at stage start by
    // the RestartIteration call in InitializeRadial.
    EnsurePTSBackupBuffers(S);
  }
  (void)m_dec_v_rcc; (void)m_dec_v_rss; (void)m_dec_v_zsc;
  (void)m_dec_v_zcs; (void)m_dec_v_lsc; (void)m_dec_v_lcs;

  // Init-only invocation (PrepareStagePtsXCuda): the per-cfg device state
  // is now sized and populated for the current stage; skip the step.
  if (g_pts_init_only) return;

  int nsMinF_to_nsMinF1 = r.nsMinF - r.nsMinF1;

  // Batches take the per-configuration time-step controller: k_update_timestep
  // computes each configuration's (fac, b1) from its own residuals.
  const double* d_fac_b1_for_step = nullptr;
  if (S.n_config_max > 1) {
    S.EnsureTimestepBuffers(time_step);
    S.StageFnorm1(fnorm1);
    // Reset the ring on the same iterations the host controller does
    // (iter2 == iter1: stage starts and restarts), plus defensively after
    // a Reshape reallocated the ring buffers.
    int phase = (iter_phase == 0 || S.timestep_first_call_after_reset)
        ? 0 : 1;
    k_update_timestep<<<S.n_config_max, 32, 0, st>>>(
        S.n_config_max, phase, time_step, S.d_fnorm1, fc.deltaS,
        S.d_residuals_partial,
        S.d_inv_tau, S.d_prev_fsq, S.d_fac_b1,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(),
               "k_update_timestep launch (per-cfg ts)");
    S.timestep_first_call_after_reset = false;
    d_fac_b1_for_step = S.d_fac_b1;
  }

  // Launch k_perform_time_step.
  const int TPB = 16;
  dim3 b((ntor + 1 + TPB - 1) / TPB, mpol, ns_con_local * S.n_config_max);
  dim3 t(TPB, 1, 1);
  k_perform_time_step<<<b, t, 0, st>>>(
      S.n_config_max, ns_local, ns_con_local, mpol, ntor,
      nsMinF_to_nsMinF1, s.lthreed,
      velocity_scale, conjugation_parameter, time_step,
      S.d_decomposed_frcc, S.d_decomposed_frss,
      S.d_decomposed_fzsc, S.d_decomposed_fzcs,
      S.d_decomposed_flsc, S.d_decomposed_flcs,
      S.d_pts_v_rcc, S.d_pts_v_rss, S.d_pts_v_zsc,
      S.d_pts_v_zcs, S.d_pts_v_lsc, S.d_pts_v_lcs,
      S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
      S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs,
      d_fac_b1_for_step, S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_perform_time_step launch");
  if (!s.lthreed) {
    // Axisymmetric (ntor=0): the lthreed-only spectral slots carry no physical
    // contribution. The host keeps them null, so hold them (and their
    // velocities) at zero on the device too; otherwise the geometry inverse
    // picks up evolved garbage and the jacobian degrades.
    const size_t z_bytes =
        (size_t)S.n_config_max * (size_t)S.pts_x_size * sizeof(double);
    cuda_check(cudaMemsetAsync(S.d_pts_x_rss, 0, z_bytes, st), "zero 2d x_rss");
    cuda_check(cudaMemsetAsync(S.d_pts_x_zcs, 0, z_bytes, st), "zero 2d x_zcs");
    cuda_check(cudaMemsetAsync(S.d_pts_x_lcs, 0, z_bytes, st), "zero 2d x_lcs");
    cuda_check(cudaMemsetAsync(S.d_pts_v_rss, 0, z_bytes, st), "zero 2d v_rss");
    cuda_check(cudaMemsetAsync(S.d_pts_v_zcs, 0, z_bytes, st), "zero 2d v_zcs");
    cuda_check(cudaMemsetAsync(S.d_pts_v_lcs, 0, z_bytes, st), "zero 2d v_lcs");
  }

  // Sync deferral (every-K backup cadence):
  //   - per-iter D2H + sync of d_pts_x -> host m_decomposed_x ELIDED here.
  //   - Three on-demand flush sites call FlushDecomposedXToHostCuda():
  //       1. update() iter2<2 path before host decomposeInto (per stage)
  //       2. updateRadialPreconditioner before computeForceNorms (every 25)
  //       3. Vmec::Evolve end-of-run before ComputeOutputQuantities
  //   - Host backup save/restore in Vmec::RestartIteration moved device-side
  //     via BackupPtsXCuda / RestorePtsXFromBackupCuda (caller now gates the
  //     save to every-K iters to avoid net-zero per-iter device-D2D cost).
  (void)m_dec_x_rcc; (void)m_dec_x_rss; (void)m_dec_x_zsc;
  (void)m_dec_x_zcs; (void)m_dec_x_lsc; (void)m_dec_x_lcs;
}

// =============================================================================
// Per-config D2H entry points
// =============================================================================
// These expose the per-config arrays that the batched kernels write to
// device-side buffers, as explicit synchronizing transfers.
// Each function:
//   - Locks S.mu (consistent with the kernel wrappers it follows).
//   - cudaMemcpyAsync's the full n_cfg array to a pinned host stage.
//   - cudaStreamSynchronize before returning (caller treats as a sync point).
//   - Resizes + populates the caller's std::vector<double> out-arguments.
//
// They are intentionally separate from the single-cfg entries so the
// production iter loop is unchanged; the iter loop's per-cfg control logic
// consumes the per-cfg caches populated during its existing syncs instead,
// and these explicit entry points remain for diagnostics and external
// callers.

// H2D the host active_per_cfg vector to the device byte
// buffer. Caller invokes once per iter (or whenever the mask changes); kernels
// read d_active_per_cfg at blockIdx.z and early-return for inactive cfgs.
// Skipped when n_cfg <= 1 (single-cfg has nothing to mask). Compares against
// last-staged buffer and skips the H2D when unchanged (the mask only flips
// when a cfg converges, which happens rarely).
void SnapshotInactiveCfgCuda(int cfg) {
  auto& S = State();
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.pts_x_initialized || !S.d_pts_x_rcc || S.pts_x_size <= 0) return;
  if (cfg < 0 || cfg >= S.n_config_max) return;
  if (static_cast<int>(S.pts_x_final_taken.size()) != S.n_config_max) {
    S.pts_x_final_taken.assign(S.n_config_max,
                                static_cast<std::uint8_t>(0));
  }
  size_t bytes_all =
      sizeof(double) * (size_t)S.n_config_max * S.pts_x_size;
  if (!S.d_pts_x_final_rcc) {
    cuda_check(cudaMalloc(&S.d_pts_x_final_rcc, bytes_all), "alloc fin rcc");
    cuda_check(cudaMalloc(&S.d_pts_x_final_rss, bytes_all), "alloc fin rss");
    cuda_check(cudaMalloc(&S.d_pts_x_final_zsc, bytes_all), "alloc fin zsc");
    cuda_check(cudaMalloc(&S.d_pts_x_final_zcs, bytes_all), "alloc fin zcs");
    cuda_check(cudaMalloc(&S.d_pts_x_final_lsc, bytes_all), "alloc fin lsc");
    cuda_check(cudaMalloc(&S.d_pts_x_final_lcs, bytes_all), "alloc fin lcs");
  }
  size_t off = (size_t)cfg * S.pts_x_size;
  size_t bytes_one = sizeof(double) * (size_t)S.pts_x_size;
  const double* src[6] = {S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
                          S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs};
  double* dst[6] = {S.d_pts_x_final_rcc, S.d_pts_x_final_rss,
                    S.d_pts_x_final_zsc, S.d_pts_x_final_zcs,
                    S.d_pts_x_final_lsc, S.d_pts_x_final_lcs};
  for (int sp = 0; sp < 6; ++sp) {
    cuda_check(cudaMemcpyAsync(dst[sp] + off, src[sp] + off, bytes_one,
                                cudaMemcpyDeviceToDevice, S.stream),
               "snapshot inactive cfg");
  }
  S.pts_x_final_taken[cfg] = 1;
}

void StagePhaseDActiveCuda(const std::vector<std::uint8_t>& active_per_cfg) {
  auto& S = State();
  std::lock_guard<std::mutex> lk(S.mu);
  if (active_per_cfg.empty() || S.n_config_max <= 1 || !S.stream) return;
  S.EnsureActivePerCfgBuffer();
  const int n = std::min(static_cast<int>(active_per_cfg.size()),
                          S.n_config_max);
  // Check if anything changed from last staging.
  bool changed = false;
  if (static_cast<int>(S.h_active_staged.size()) != n) {
    S.h_active_staged.assign(n, 1);
    changed = true;
  }
  for (int c = 0; c < n; ++c) {
    if (S.h_active_staged[c] != active_per_cfg[c]) {
      S.h_active_staged[c] = active_per_cfg[c];
      changed = true;
    }
  }
  if (changed) {
    cuda_check(cudaMemcpyAsync(S.d_active_per_cfg, S.h_active_staged.data(),
                                sizeof(std::uint8_t) * n,
                                cudaMemcpyHostToDevice, S.stream),
               "h2d d_active_per_cfg (stage)");
  }
}

int GetNConfigMaxCuda() { return g_n_config_run; }

int CudaMaxRadialResolution() {
  // ns <= 1024 uses the PCR solver (one thread per radial row, bounded by the
  // 1024 threads-per-block limit); larger ns uses the block-Thomas solver,
  // which holds the elimination ratios in dynamic shared memory sized to ns.
  // The device's opt-in shared-memory capacity therefore sets the ceiling.
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 1024;
  int max_smem = 0;
  if (cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             dev) != cudaSuccess ||
      max_smem <= 0) {
    return 1024;
  }
  const int blk_limit = max_smem / static_cast<int>(sizeof(double));
  return blk_limit > 1024 ? blk_limit : 1024;
}

bool CudaVramBudgetCuda(long long n_cfg, long long ns, long long mpol,
                        long long ntor, long long nZeta, long long nThetaEff,
                        long long* needed_bytes, long long* free_bytes) {
  // Upper estimate of the persistent device allocation for one run at the
  // given shape and configuration count (CudaBudgetRawBytes), with an
  // eighth-part margin and a flat cushion that absorb the small profile
  // buffers, the pinned-host counterparts of lazy allocations, and
  // context overhead.
  long long needed = CudaBudgetRawBytes(n_cfg, ns, mpol, ntor, nZeta,
                                        nThetaEff);
  needed += needed / 8 + (256LL << 20);
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    // No queryable device; let the allocations themselves decide.
    if (needed_bytes) *needed_bytes = needed;
    if (free_bytes) *free_bytes = -1;
    return true;
  }
  size_t free_sz = 0;
  size_t total_sz = 0;
  if (cudaSetDevice(0) != cudaSuccess ||
      cudaMemGetInfo(&free_sz, &total_sz) != cudaSuccess) {
    if (needed_bytes) *needed_bytes = needed;
    if (free_bytes) *free_bytes = -1;
    return true;
  }
  // Credit the memory the next Reshape frees before this run's
  // allocations land: a prior run's persistent buffers are released when
  // the shape or the configuration count changes, and stay (already
  // counted inside the free query's deficit) when neither does.
  long long reclaimable = 0;
  {
    auto& S = State();
    std::lock_guard<std::mutex> lk(S.mu);
    if (S.stream) {
      reclaimable = S.reshape_budget_raw_bytes;
    }
  }
  if (needed_bytes) *needed_bytes = needed;
  if (free_bytes) *free_bytes = (long long)free_sz;
  return needed <= (long long)free_sz + reclaimable;
}

int GetConvergenceFlag(int cfg) {
  auto& S = State();
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.h_conv_flag_pinned) return -1;
  if (cfg < 0 || cfg >= S.n_config_max) return -1;
  return static_cast<int>(S.h_conv_flag_pinned[cfg]);
}

bool PtsXInitializedCuda() {
  auto& S = State();
  if (!S.stream) return false;
  std::lock_guard<std::mutex> lk(S.mu);
  return S.pts_x_initialized;
}

// True when the device-resident d_pts_x is initialized AND sized for the
// given stage geometry. Distinguishes "post-upscale, authoritative for the
// new stage" from "stale previous-stage buffer" at iteration 1 of a
// multigrid stage.
bool PtsXMatchesCuda(int ns_local, int mpol, int ntor) {
  auto& S = State();
  if (!S.stream) return false;
  std::lock_guard<std::mutex> lk(S.mu);
  return S.pts_x_initialized &&
         S.pts_x_size == ns_local * mpol * (ntor + 1);
}

}  // namespace vmecpp

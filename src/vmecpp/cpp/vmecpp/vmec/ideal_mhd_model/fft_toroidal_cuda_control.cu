#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"

namespace vmecpp {

// ResidualsCuda: FourierForces::residuals() on the device's decomposed shadow,
// returning [fResR, fResZ, fResL] over the CPU's radial range.
void ResidualsCuda(const RadialPartitioning& r, const Sizes& s,
                    const FlowControl& fc, bool includeEdgeRZForces,
                    double& fResR_out, double& fResZ_out, double& fResL_out,
                    bool is_precd,
                    const std::vector<std::uint8_t>* edge_per_cfg) {
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
  // The range with the edge R and Z forces, which edge_per_cfg selects per
  // configuration in place of includeEdgeRZForces.
  const int jMaxRZ_edge = (r.nsMaxF1 == ns) ? ns : jMaxRZ;
  const bool per_cfg_edge =
      edge_per_cfg != nullptr &&
      static_cast<int>(edge_per_cfg->size()) == S.n_config_max;
  if (includeEdgeRZForces && !per_cfg_edge) {
    jMaxRZ = jMaxRZ_edge;
  }
  int jMaxIncludeBoundary = nsMax_;
  if (r.nsMaxF1 == ns) {
    jMaxIncludeBoundary = ns;
  }
  int jLocal_max_rz = jMaxRZ - nsMin_;
  int jLocal_max_rz_edge = jMaxRZ_edge - nsMin_;
  int jLocal_max_boundary = jMaxIncludeBoundary - nsMin_;
  if (jLocal_max_rz < 0) jLocal_max_rz = 0;
  if (jLocal_max_rz_edge < 0) jLocal_max_rz_edge = 0;
  if (jLocal_max_boundary < 0) jLocal_max_boundary = 0;
  if (jLocal_max_boundary > ns_dec_local) jLocal_max_boundary = ns_dec_local;
  if (jLocal_max_rz > ns_dec_local) jLocal_max_rz = ns_dec_local;
  if (jLocal_max_rz_edge > ns_dec_local) jLocal_max_rz_edge = ns_dec_local;
  const std::uint8_t* d_edge = nullptr;
  if (per_cfg_edge) {
    S.EnsurePerCfgControlBuffers();
    cuda_check(cudaMemcpyAsync(S.d_edge_per_cfg, edge_per_cfg->data(),
                                sizeof(std::uint8_t) * S.n_config_max,
                                cudaMemcpyHostToDevice, st),
               "h2d per-cfg edge forces");
    d_edge = S.d_edge_per_cfg;
  }

  // One reduction per configuration in K = max(1, 16 / n_config_max)
  // sub-blocks, about 16 blocks in total.
  if (g_residuals_k_run < 0) {
    const int auto_k = CudaToroidalState::kResidualsKPartitions / S.n_config_max;
    g_residuals_k_run = (auto_k < 1) ? 1 : auto_k;
  }
  const int residuals_k_env = g_residuals_k_run;
  if (residuals_k_env > 1) {
    // Grid (K, n_config): each block reduces 1/K of the index space, and the
    // finalize kernel collapses the K partials.
    dim3 partials_grid(residuals_k_env, S.n_config_max);
    k_residuals_par_K<<<partials_grid, 256, 0, st>>>(
        S.n_config_max, S.ns_local_cached,
        jLocal_max_rz, jLocal_max_boundary, mpol, ntor, s.lthreed,
        residuals_k_env,
        S.d_decomposed_frcc, S.d_decomposed_frss,
        S.d_decomposed_fzsc, S.d_decomposed_fzcs,
        S.d_decomposed_flsc, S.d_decomposed_flcs,
        S.d_residuals_partials_K,
        S.d_active_per_cfg, jLocal_max_rz_edge, d_edge);
    cuda_check(cudaGetLastError(), "k_residuals_par_K launch");
    // Finalize: one block of 32 threads per configuration, covering up to 32
    // partitions; K is capped at 16.
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
        S.d_active_per_cfg, jLocal_max_rz_edge, d_edge);
    cuda_check(cudaGetLastError(), "k_residuals_par launch");
  }
  // Device convergence flag on the normalized residuals, from the
  // per-configuration norms, energies, volumes and lamscale.
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

  // The transfer covers 3 * n_cfg doubles under the synchronization this call
  // already pays; evalFResInvar and evalFResPrecd read the cache.
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

// FlushDecomposedToHostCuda: copies the decomposed force shadow to the host
// m_decomposed_f, which the iteration body itself does not need.

// (anonymous namespace hoisted across the split translation units) Declared
// here so the d_pts_x initialization sites can arm the backup.
void EnsurePTSBackupBuffers(CudaToroidalState& S);
// (anonymous namespace hoisted across the split translation units)

// When set, PerformTimeStepCuda only initializes its buffers and the stage's
// positions and returns before the time step (PrepareStagePtsXCuda).
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
    // Zeroes the velocity of every configuration slot at the first call.
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
    // On a distinct batch's stage transition, upscale each configuration's
    // snapshot into the new d_pts_x, which the host's single shadow cannot.
    if (g_batch_distinct_run && S.pts_x_prev_valid && S.pts_x_prev_ns > 0 &&
        S.pts_x_prev_size > 0 && S.d_scalxc) {
      int ns_old = S.pts_x_prev_ns;
      int ns_new = ns_local;
      // Vmec::InterpolateToNextMultigridStep per configuration, bit for bit:
      // scaling, axis extrapolation, interpolation.
      int scalxc_len_per_cfg = ns_new * 2;
      if (!S.scalxc_prev_valid || !S.d_scalxc_prev) {
        throw std::runtime_error(
            "CUDA multigrid upscale without the previous stage's scalxc");
      } else {
        // Vmec::InterpolateToNextMultigridStep on the device, bit for bit,
        // consuming the snapshot in place.
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
      // Copies configuration zero's upscaled state to the host shadow that
      // rzNorm, fNorm1 and the restart backups read.
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
        // The odd-parity spectra are absent at ntor = 0, with neither a host
        // nor a device buffer, so they are not copied.
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
    // Distinct batches load each configuration's positions from the staging
    // block; RecomposeToPhysicalCuda loads the same, whichever runs first.
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
    // Arms the device backup with the stage's initial state, as the host backup
    // is synced at stage start, so an early restore rewinds validly.
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
  const double* d_time_step_for_step = nullptr;
  const std::uint8_t* d_phase_for_step = nullptr;
  if (S.n_config_max > 1) {
    S.EnsureTimestepBuffers(time_step);
    S.StageFnorm1(fnorm1);
    if (fc.per_cfg_control &&
        static_cast<int>(fc.delt0r_per_cfg.size()) == S.n_config_max) {
      // Per-configuration restart control: each configuration's time step,
      // and a ring reset where its iteration count meets its latest restart.
      S.EnsurePerCfgControlBuffers();
      std::vector<std::uint8_t> phases(S.n_config_max, 1);
      for (int c = 0; c < S.n_config_max; ++c) {
        if (S.timestep_first_call_after_reset ||
            fc.iter2_per_cfg[c] == fc.iter1_per_cfg[c]) {
          phases[c] = 0;
        }
      }
      cuda_check(cudaMemcpyAsync(S.d_time_step_per_cfg, fc.delt0r_per_cfg.data(),
                                  sizeof(double) * S.n_config_max,
                                  cudaMemcpyHostToDevice, st),
                 "h2d per-cfg time step");
      cuda_check(cudaMemcpyAsync(S.d_phase_per_cfg, phases.data(),
                                  sizeof(std::uint8_t) * S.n_config_max,
                                  cudaMemcpyHostToDevice, st),
                 "h2d per-cfg ring reset");
      d_time_step_for_step = S.d_time_step_per_cfg;
      d_phase_for_step = S.d_phase_per_cfg;
    }
    // The damping ring resets where the host's does (iter2 == iter1) and after
    // a Reshape reallocated it.
    int phase = (iter_phase == 0 || S.timestep_first_call_after_reset)
        ? 0 : 1;
    k_update_timestep<<<S.n_config_max, 32, 0, st>>>(
        S.n_config_max, phase, time_step, S.d_fnorm1, fc.deltaS,
        S.d_residuals_partial,
        S.d_inv_tau, S.d_prev_fsq, S.d_fac_b1,
        S.d_active_per_cfg, d_time_step_for_step, d_phase_for_step);
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
      d_fac_b1_for_step, S.d_active_per_cfg, d_time_step_for_step);
  cuda_check(cudaGetLastError(), "k_perform_time_step launch");
  if (!s.lthreed) {
    // At ntor = 0 the lthreed-only slots stay zero on the device, as the host
    // keeps them null.
    const size_t z_bytes =
        (size_t)S.n_config_max * (size_t)S.pts_x_size * sizeof(double);
    cuda_check(cudaMemsetAsync(S.d_pts_x_rss, 0, z_bytes, st), "zero 2d x_rss");
    cuda_check(cudaMemsetAsync(S.d_pts_x_zcs, 0, z_bytes, st), "zero 2d x_zcs");
    cuda_check(cudaMemsetAsync(S.d_pts_x_lcs, 0, z_bytes, st), "zero 2d x_lcs");
    cuda_check(cudaMemsetAsync(S.d_pts_v_rss, 0, z_bytes, st), "zero 2d v_rss");
    cuda_check(cudaMemsetAsync(S.d_pts_v_zcs, 0, z_bytes, st), "zero 2d v_zcs");
    cuda_check(cudaMemsetAsync(S.d_pts_v_lcs, 0, z_bytes, st), "zero 2d v_lcs");
  }

  // The device positions reach the host only at the explicit flush sites, and
  // the restart backups are device-side.
  (void)m_dec_x_rcc; (void)m_dec_x_rss; (void)m_dec_x_zsc;
  (void)m_dec_x_zcs; (void)m_dec_x_lsc; (void)m_dec_x_lcs;
}

// Per-configuration device-to-host transfers of the arrays the batched kernels
// write, each synchronizing under S.mu.

// Copies cfg's position slice into its converged-state snapshot when it goes
// inactive.
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

// Copies the host active mask to the device for more than one configuration,
// skipping an unchanged mask.
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
  // PCR to ns = 1024, then block-Thomas with the elimination ratios in shared
  // memory, whose opt-in capacity on the device sets the ceiling.
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
  // CudaBudgetRawBytes plus an eighth-part margin and a flat cushion for small
  // buffers, pinned host memory and context overhead.
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
  // Credits the memory the next Reshape frees when the shape or configuration
  // count changes.
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

// Whether d_pts_x is loaded and sized for this stage, telling the upscaled
// state at a stage's first iteration from a stale buffer.
bool PtsXMatchesCuda(int ns_local, int mpol, int ntor) {
  auto& S = State();
  if (!S.stream) return false;
  std::lock_guard<std::mutex> lk(S.mu);
  return S.pts_x_initialized &&
         S.pts_x_size == ns_local * mpol * (ntor + 1);
}

}  // namespace vmecpp

#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"

namespace vmecpp {

// ============================================================================
// ComputeJacobianCuda: CUDA port of IdealMhdModel::computeJacobian.
//
// Reads d_r1_e/o, d_ru_e/o, d_z1_e/o, d_zu_e/o (already on GPU after the
// forward FFT call) plus sqrtSH (one H2D per call), writes r12/ru12/zu12/rs/
// zs/tau on the half-grid, and returns bad_jacobian = (minTau*maxTau<0 || NaN)
// computed on host after a single D2H of all 6 jacobian arrays.
// ============================================================================
void ComputeJacobianCuda(
    const RadialPartitioning& r, const Sizes& s,
    const Eigen::VectorXd& sqrtSH, double deltaS, double dSHalfDsInterp,
    Eigen::VectorXd& r12, Eigen::VectorXd& ru12, Eigen::VectorXd& zu12,
    Eigen::VectorXd& rs, Eigen::VectorXd& zs, Eigen::VectorXd& tau,
    bool& bad_jacobian,
    int signOfJacobian,
    const Eigen::VectorXd* wInt) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int nZnT = s.nZnT;
  const int jF_in_offset = r.nsMinH - r.nsMinF1;
  if (ns_h <= 0) {
    bad_jacobian = false;
    return;
  }

  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureJacobianBuffers(ns_h, nZnT);

  // H2D sqrtSH once per Reshape (invariant under iteration). Subsequent calls
  // skip the H2D. EnsureJacobianBuffers above clears the flag if the buffer
  // was reallocated due to ns_h change.
  if (!S.sqrtSH_staged) {
    cuda_check(cudaMemcpyAsync(S.d_sqrtSH, sqrtSH.data(),
                                sizeof(double) * ns_h,
                                cudaMemcpyHostToDevice, S.stream), "h2d sqrtSH");
    S.sqrtSH_staged = true;
  }

  // Launch the jacobian computation. The configuration axis is
  // carried on the third grid dimension, so for n_config_max == 1
  // the launch reduces to (nZnT / TPB, ns_h, 1), matching the
  // single-configuration baseline.
  //
  // The VMECPP_JAC_METRIC_FUSE environment variable selects between
  // the jacobian-only kernel and the fused jacobian-and-metric
  // kernel; the default is the fused variant. When the fused
  // variant runs, the metric outputs gsqrt, guu, guv, and gvv are
  // written directly into the corresponding device buffers, the
  // jac_metric_fused_this_iter flag is raised, and
  // ComputeMetricElementsCuda elides its own kernel launch on
  // observing the flag.
  static int jac_metric_fuse_env = -1;
  if (jac_metric_fuse_env < 0) {
    const char* e = std::getenv("VMECPP_JAC_METRIC_FUSE");
    jac_metric_fuse_env = (e && std::atoi(e) == 0) ? 0 : 1;
    if (!jac_metric_fuse_env) {
      std::fprintf(stderr, "[fft_toroidal_cuda] jacobian+metric fusion "
                           "disabled (VMECPP_JAC_METRIC_FUSE=0)\n");
    }
  }
  // Three-way fusion of the jacobian, metric, and differential-volume
  // computations through k_jacobian_metric_dvdsh_atomic. The block
  // geometry matches the fused jacobian-and-metric kernel, and the
  // differential volume dVdsH is accumulated through atomicAdd on
  // the floating-point overload. The atomic accumulation introduces
  // a small order-dependent floating-point deviation that the
  // drift tolerance applied to dVdsH admits. The fused path is
  // selected by VMECPP_JAC_METRIC_DVDSH_FUSE (default active when
  // unset) and additionally requires the caller to provide
  // wInt and signOfJacobian; setting the variable to zero falls
  // back to the separate jacobian-and-metric kernel followed by an
  // independent dVdsH reduction.
  static int dvdsh_fuse_env = -1;
  if (dvdsh_fuse_env < 0) {
    const char* e = std::getenv("VMECPP_JAC_METRIC_DVDSH_FUSE");
    dvdsh_fuse_env = (e && std::atoi(e) == 0) ? 0 : 1;
    if (!dvdsh_fuse_env) {
      std::fprintf(stderr, "[fft_toroidal_cuda] jac+metric+dvdsh atomic fusion "
                           "disabled (VMECPP_JAC_METRIC_DVDSH_FUSE=0)\n");
    }
  }
  const bool use_dvdsh_fuse = dvdsh_fuse_env && wInt != nullptr &&
                              signOfJacobian != 0 && jac_metric_fuse_env;

  if (use_dvdsh_fuse) {
    S.EnsureMetricBuffers(ns_h, nZnT);
    S.EnsureDVdsHBuffer(ns_h);
    S.EnsureWIntStaged(s.nThetaEff, wInt->data());
    // Zero the dVdsH slice that the atomicAdd accumulator will write into.
    cuda_check(cudaMemsetAsync(S.d_dVdsH, 0,
                                sizeof(double) * S.n_config_max * ns_h,
                                S.stream),
               "memset d_dVdsH for atomic accumulator");
    // Atomic fusion geometry: same as separate jac+metric (TPB=64, X-blocks
    // covering nZnT), preserves occupancy.
    // VMECPP_JAC_PAIR routes to jH-coarsened pair variant when ns_h is even.
    // Pair caches the shared middle jF's 8 main fields in shared memory and
    // saves 50pct of main-field jF reads per pair-block. Default ON; set =0
    // to fall back. Measured at N=64 over five evaluations: 0.5336 ->
    // 0.5365 eq/s (+0.54pct).
    // Bit-exact aspect_ratio = 7.527844291824478, qi/L_grad_B unchanged.
    static const int jac_pair_env = []() {
      const char* e = std::getenv("VMECPP_JAC_PAIR");
      return (e && std::atoi(e) == 0) ? 0 : 1;
    }();
    const int TPB = 64;
    S.TKBegin(CudaToroidalState::TK_JAC_METRIC_DVDSH);
    if (jac_pair_env && (ns_h % 2 == 0)) {
      dim3 fblocks_p((nZnT + TPB - 1) / TPB, ns_h / 2, S.n_config_max);
      dim3 ftpb_p(TPB, 2, 1);
      // 8 fields cached per x-lane (blockDim.x), so 4 KB per block at TPB=64,
      // independent of the poloidal resolution.
      size_t smem_bytes = (size_t)sizeof(double) * 8 * (size_t)TPB;
      k_jacobian_metric_dvdsh_atomic_pair<<<fblocks_p, ftpb_p, smem_bytes, S.stream>>>(
          S.n_config_max, S.ns_local_cached,
          ns_h, jF_in_offset, nZnT, s.nThetaEff, s.lthreed,
          (double)signOfJacobian,
          S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
          S.d_z1_e, S.d_z1_o, S.d_zu_e, S.d_zu_o,
          S.d_rv_e, S.d_rv_o, S.d_zv_e, S.d_zv_o,
          S.d_sqrtSF, S.d_sqrtSH, S.d_wInt,
          deltaS, dSHalfDsInterp,
          S.d_r12, S.d_ru12, S.d_zu12, S.d_rs, S.d_zs, S.d_tau,
          S.d_gsqrt, S.d_guu, S.d_guv, S.d_gvv,
          S.d_dVdsH,
          S.d_active_per_cfg);
      cuda_check(cudaGetLastError(), "k_jacobian_metric_dvdsh_atomic_pair launch");
    } else {
      dim3 fblocks((nZnT + TPB - 1) / TPB, ns_h, S.n_config_max);
      dim3 ftpb(TPB, 1, 1);
      k_jacobian_metric_dvdsh_atomic<<<fblocks, ftpb, 0, S.stream>>>(
          S.n_config_max, S.ns_local_cached,
          ns_h, jF_in_offset, nZnT, s.nThetaEff, s.lthreed,
          (double)signOfJacobian,
          S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
          S.d_z1_e, S.d_z1_o, S.d_zu_e, S.d_zu_o,
          S.d_rv_e, S.d_rv_o, S.d_zv_e, S.d_zv_o,
          S.d_sqrtSF, S.d_sqrtSH, S.d_wInt,
          deltaS, dSHalfDsInterp,
          S.d_r12, S.d_ru12, S.d_zu12, S.d_rs, S.d_zs, S.d_tau,
          S.d_gsqrt, S.d_guu, S.d_guv, S.d_gvv,
          S.d_dVdsH);
      cuda_check(cudaGetLastError(), "k_jacobian_metric_dvdsh_atomic launch");
    }
    S.TKEnd(CudaToroidalState::TK_JAC_METRIC_DVDSH);
    S.jac_metric_fused_this_iter = true;
    S.dvdsh_fused_this_iter = true;
  } else {
    const int TPB = 64;
    dim3 blocks((nZnT + TPB - 1) / TPB, ns_h, S.n_config_max);
    dim3 tpb(TPB, 1, 1);
    if (jac_metric_fuse_env) {
      S.EnsureMetricBuffers(ns_h, nZnT);  // metric outputs alloc upfront
      k_jacobian_and_metric<<<blocks, tpb, 0, S.stream>>>(
          S.n_config_max, S.ns_local_cached,
          ns_h, jF_in_offset, nZnT, s.lthreed,
          S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
          S.d_z1_e, S.d_z1_o, S.d_zu_e, S.d_zu_o,
          S.d_rv_e, S.d_rv_o, S.d_zv_e, S.d_zv_o,
          S.d_sqrtSF, S.d_sqrtSH, deltaS, dSHalfDsInterp,
          S.d_r12, S.d_ru12, S.d_zu12, S.d_rs, S.d_zs, S.d_tau,
          S.d_gsqrt, S.d_guu, S.d_guv, S.d_gvv,
          S.d_active_per_cfg);
      cuda_check(cudaGetLastError(), "k_jacobian_and_metric launch");
      S.jac_metric_fused_this_iter = true;
    } else {
      k_compute_jacobian<<<blocks, tpb, 0, S.stream>>>(
          S.n_config_max, S.ns_local_cached,
          ns_h, jF_in_offset, nZnT,
          S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
          S.d_z1_e, S.d_z1_o, S.d_zu_e, S.d_zu_o,
          S.d_sqrtSH, deltaS, dSHalfDsInterp,
          S.d_r12, S.d_ru12, S.d_zu12, S.d_rs, S.d_zs, S.d_tau);
      cuda_check(cudaGetLastError(), "k_compute_jacobian launch");
    }
  }

  // Device-side tau min/max → 2 scalars per config (replaces tau D2H + host scan).
  // Batched execution: launch n_config_max_max blocks, each writes out[cfg*2+0..1].
  S.EnsureJacMinmaxBuffer();
  k_tau_minmax<<<S.n_config_max, 256, 0, S.stream>>>(
      S.n_config_max, ns_h * nZnT, S.d_tau, S.d_jac_minmax,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_tau_minmax launch");

  // Per-iter D2Hs of ru12/zu12/rs/zs eliminated. Downstream CUDA consumers
  // (ComputePreconditioningMatrixCuda) read device pointers directly; the
  // end-of-run FlushForOutputCuda handles the output-phase host needs. Only
  // the 2 tau min/max scalars stay (host reads them immediately for
  // bad_jacobian).
  //
  // Per-cfg cache: replace the 2-double D2H with 2*n_cfg
  // D2H into a static cache. Single-cfg behavior preserved (minTau = cache[0],
  // maxTau = cache[1]); per-cfg cache populated for free during the SAME
  // sync. Per-cfg consumers read via GetJacMinmaxPerCfgCache() to build
  // per-cfg bad_jacobian without an extra D2H+sync.
  int n_cfg = S.n_config_max;
  if ((int)g_jac_minmax_cache.size() != 2 * n_cfg) {
    g_jac_minmax_cache.assign(2 * n_cfg, 0.0);
  }
  (void)r12; (void)ru12; (void)zu12; (void)rs; (void)zs; (void)tau;
  // Sync elision: the tau min/max reduction ran above (device state stays
  // current); the bad-jacobian decision is evaluated against the last
  // boundary-synced extrema. A sign flip occurring mid-window is caught
  // at the next boundary; the restore path rewinds at most K-1
  // iterations, which the every-K backup cadence covers.
  if (S.sync_elide_iter) {
    double minTau_st = g_jac_minmax_cache[0];
    double maxTau_st = g_jac_minmax_cache[1];
    bad_jacobian = (minTau_st * maxTau_st < 0.0) ||
                   !std::isfinite(minTau_st * maxTau_st);
    return;
  }
  cuda_check(cudaMemcpyAsync(g_jac_minmax_cache.data(), S.d_jac_minmax,
                              (size_t)2 * n_cfg * sizeof(double),
                              cudaMemcpyDeviceToHost, S.stream),
             "d2h jac minmax (per-cfg cache)");
  cuda_check(cudaStreamSynchronize(S.stream), "jac stream sync");

  double minTau = g_jac_minmax_cache[0], maxTau = g_jac_minmax_cache[1];
  bad_jacobian = (minTau * maxTau < 0.0) || !std::isfinite(minTau * maxTau);
  DiagCfg01DiffCuda(S.d_tau, ns_h * nZnT, "jac:tau");
}

// ============================================================================
// ComputeMetricElementsCuda: CUDA port of IdealMhdModel::computeMetricElements.
// Reads d_r1_e/o, d_ru_e/o, d_zu_e/o, d_rv_e/o, d_zv_e/o (from forward FFT) plus
// d_sqrtSF (already on device from forward) and d_tau, d_r12 (from jacobian).
// Stages d_sqrtSH per call. Writes d_gsqrt, d_guu, d_guv, d_gvv on device and
// D2Hs into the gsqrt/guu/guv/gvv Eigen::VectorXd's.
// ============================================================================
void ComputeMetricElementsCuda(
    const RadialPartitioning& r, const Sizes& s,
    const Eigen::VectorXd& sqrtSF_unused, const Eigen::VectorXd& sqrtSH,
    Eigen::VectorXd& gsqrt, Eigen::VectorXd& guu,
    Eigen::VectorXd& guv, Eigen::VectorXd& gvv) {
  (void)sqrtSF_unused;  // already on device as S.d_sqrtSF from forward
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int nZnT = s.nZnT;
  const int jF_in_offset = r.nsMinH - r.nsMinF1;
  if (ns_h <= 0) return;

  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureMetricBuffers(ns_h, nZnT);

  // sqrtSH already staged in ComputeJacobianCuda above (same iter). The cache
  // flag (S.sqrtSH_staged) means we can skip this H2D - it's a redundant
  // copy of the same data.
  if (!S.sqrtSH_staged) {
    cuda_check(cudaMemcpyAsync(S.d_sqrtSH, sqrtSH.data(),
                                sizeof(double) * ns_h,
                                cudaMemcpyHostToDevice, S.stream), "h2d sqrtSH");
    S.sqrtSH_staged = true;
  }

  // When ComputeJacobianCuda has already dispatched the fused
  // jacobian-and-metric kernel during the present iteration, the
  // metric outputs gsqrt, guu, guv, and gvv are already populated
  // in the corresponding device buffers; the kernel launch that
  // would otherwise occur here is therefore redundant. The
  // handoff flag is cleared so that subsequent iterations resume
  // the independent metric launch when fusion is not in effect.
  if (S.jac_metric_fused_this_iter) {
    S.jac_metric_fused_this_iter = false;
    (void)gsqrt; (void)guu; (void)guv; (void)gvv;
    return;
  }
  // Batched execution: z-dim covers n_config_max configs. At n_config_max=1
  // this collapses to (nZnT/TPB, ns_h, 1), the single-configuration launch.
  const int TPB = 64;
  dim3 blocks((nZnT + TPB - 1) / TPB, ns_h, S.n_config_max);
  dim3 tpb(TPB, 1, 1);
  k_compute_metric_elements<<<blocks, tpb, 0, S.stream>>>(
      S.n_config_max, S.ns_local_cached,
      ns_h, jF_in_offset, nZnT, s.lthreed,
      S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
      S.d_zu_e, S.d_zu_o, S.d_rv_e, S.d_rv_o,
      S.d_zv_e, S.d_zv_o,
      S.d_sqrtSF, S.d_sqrtSH, S.d_tau, S.d_r12,
      S.d_gsqrt, S.d_guu, S.d_guv, S.d_gvv);
  cuda_check(cudaGetLastError(), "k_compute_metric_elements launch");

  // Persistent on-device: gsqrt/guu/guv/gvv stay in S.d_* for downstream
  // CUDA wrappers (BContra, BCo, Pressure, HybridLambdaForce, MHDForces,
  // ForceNorms, UpdateLambdaPrecond). No D2H needed.
  (void)gsqrt; (void)guu; (void)guv; (void)gvv;
}

// ============================================================================
// UpdateDifferentialVolumeCuda: CUDA port of IdealMhdModel::updateDifferentialVolume.
// Reads d_gsqrt (from metric_elements). Stages wInt once. Writes dVdsH array
// (size ns_h) by per-surface sum of gsqrt * wInt[l], multiplied by signOfJacobian.
// ============================================================================
void UpdateDifferentialVolumeCuda(
    const RadialPartitioning& r, const Sizes& s, double signOfJacobian,
    const Eigen::VectorXd& wInt, Eigen::VectorXd& dVdsH) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int nZnT = s.nZnT;
  const int nThetaEff = s.nThetaEff;
  if (ns_h <= 0) return;

  std::lock_guard<std::mutex> lk(S.mu);
  // 3-way fusion path: if ComputeJacobianCuda already ran the jacobian +
  // metric + dvdsh fused kernel this iter, the dvdsh outputs are populated
  // in S.d_dVdsH. Skip the redundant launch and consume the flag.
  if (S.dvdsh_fused_this_iter) {
    S.dvdsh_fused_this_iter = false;
    (void)wInt; (void)dVdsH; (void)signOfJacobian;
    return;
  }
  S.EnsureWIntStaged(nThetaEff, wInt.data());
  S.EnsureDVdsHBuffer(ns_h);

  const int TPB = 32;
  // Batched execution: launch grid covers n_config_max configs in z dim.
  // At n_config_max=1 this collapses to (ns_h, 1, 1), the
  // single-configuration launch.
  dim3 blocks(ns_h, 1, S.n_config_max);
  dim3 tpb(TPB, 1, 1);
  k_update_dvdsh<<<blocks, tpb, 0, S.stream>>>(
      S.n_config_max, ns_h, nZnT, nThetaEff, signOfJacobian,
      S.d_gsqrt, S.d_wInt, S.d_dVdsH);
  cuda_check(cudaGetLastError(), "k_update_dvdsh launch");
  // dVdsH stays on device for downstream CUDA (ComputeInitialVolume,
  // UpdateVolume, PressureAndEnergies, RadialForceBalance).
  (void)dVdsH;
}

// Repair dVdsH zeroed by pre-final masking before the all-configs output flush
// (see k_recompute_dvdsh_if_zeroed).
void RecomputeZeroedDVdsHForOutputCuda(int n_cfg, int ns_h, int nZnT,
                                       int nThetaEff, double signOfJacobian) {
  auto& S = State();
  if (n_cfg <= 0 || ns_h <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.d_gsqrt || !S.d_wInt || !S.d_dVdsH) return;
  dim3 blocks(n_cfg, 1, 1);
  dim3 tpb(1, 1, 1);
  k_recompute_dvdsh_if_zeroed<<<blocks, tpb, 0, S.stream>>>(
      n_cfg, ns_h, nZnT, nThetaEff, signOfJacobian,
      S.d_gsqrt, S.d_wInt, S.d_dVdsH);
  cuda_check(cudaGetLastError(), "k_recompute_dvdsh_if_zeroed launch");
}

// ============================================================================
// ComputeBCoCuda: CUDA port of IdealMhdModel::computeBCo.
// Currently bsupu, bsupv come from CPU (computeBContra not yet ported), so we
// H2D them per call. Once BContra is ported they'll already be on device.
// ============================================================================
void ComputeBCoCuda(
    const RadialPartitioning& r, const Sizes& s,
    const Eigen::VectorXd& guu_unused, const Eigen::VectorXd& guv_unused,
    const Eigen::VectorXd& gvv_unused,
    const Eigen::VectorXd& bsupu_unused, const Eigen::VectorXd& bsupv_unused,
    Eigen::VectorXd& bsubu, Eigen::VectorXd& bsubv) {
  (void)guu_unused; (void)guv_unused; (void)gvv_unused;
  (void)bsupu_unused; (void)bsupv_unused;  // all already on device
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int nZnT = s.nZnT;
  if (ns_h <= 0) return;

  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureBCoBuffers();

  // bsupu/bsupv live in S.d_bsupu/S.d_bsupv from ComputeBContraCuda; no H2D.
  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 64;
  dim3 blocks((nZnT + TPB - 1) / TPB, ns_h, S.n_config_max);
  dim3 tpb(TPB, 1, 1);
  k_compute_bco<<<blocks, tpb, 0, S.stream>>>(
      S.n_config_max, ns_h, nZnT, s.lthreed,
      S.d_guu, S.d_guv, S.d_gvv, S.d_bsupu, S.d_bsupv,
      S.d_bsubu, S.d_bsubv);
  cuda_check(cudaGetLastError(), "k_compute_bco launch");
  // bsubu/bsubv stay on device for downstream CUDA (PressureAndEnergies,
  // RadialForceBalance, ComputeForceNorms, HybridLambdaForce).
  (void)bsubu; (void)bsubv;
}

// ============================================================================
// RadialForceBalanceCuda: CUDA port of IdealMhdModel::radialForceBalance.
// Reads d_bsubu, d_bsubv (from BCo), d_dVdsH (from updateDifferentialVolume),
// d_wInt (staged). H2Ds presH, chipF, phipF per call.
// Outputs bucoH, bvcoH (radial half-grid scalars) and interior full-grid
// arrays jcurvF, jcuruF, presgradF, dVdsF, equiF (size nsi = nsMaxFi - nsMinFi).
// ============================================================================
void RadialForceBalanceCuda(
    const RadialPartitioning& r, const Sizes& s, double signOfJacobian,
    double deltaS, const Eigen::VectorXd& wInt,
    const Eigen::VectorXd& presH, const Eigen::VectorXd& chipF,
    const Eigen::VectorXd& phipF,
    Eigen::VectorXd& bucoH, Eigen::VectorXd& bvcoH,
    Eigen::VectorXd& jcurvF, Eigen::VectorXd& jcuruF,
    Eigen::VectorXd& presgradF, Eigen::VectorXd& dVdsF,
    Eigen::VectorXd& equiF) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int nsi = r.nsMaxFi - r.nsMinFi;
  const int nZnT = s.nZnT;
  const int nThetaEff = s.nThetaEff;
  if (ns_h <= 0 || nsi <= 0) return;

  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureWIntStaged(nThetaEff, wInt.data());
  S.EnsureRadialForceBalanceBuffers(ns_h, nsi, ns_local);

  // presH already on device from PressureAndEnergiesCuda; chipF/phipF already
  // on device from ComputeBContraCuda. No H2D needed.
  (void)presH; (void)chipF; (void)phipF;

  // Stage 1: bucoH, bvcoH reduction over kl per surface.
  // Batched execution: z-dim covers n_config_max configs.
  const int TPB1 = 32;
  dim3 blocks1(ns_h, 1, S.n_config_max);
  dim3 tpb1(TPB1, 1, 1);
  k_buco_bvco<<<blocks1, tpb1, 0, S.stream>>>(
      S.n_config_max, ns_h, nZnT, nThetaEff,
      S.d_bsubu, S.d_bsubv, S.d_wInt,
      S.d_bucoH, S.d_bvcoH);
  cuda_check(cudaGetLastError(), "k_buco_bvco launch");

  // Stage 2: interior derivatives + equiF.
  // Batched execution: y-dim covers n_config_max configs.
  double signByDeltaS = signOfJacobian / deltaS;
  double invDeltaS = 1.0 / deltaS;
  int offset_jH = r.nsMinFi - r.nsMinH;
  int offset_jF = r.nsMinFi - r.nsMinF1;
  const int TPB2 = 64;
  dim3 blocks2((nsi + TPB2 - 1) / TPB2, S.n_config_max, 1);
  dim3 tpb2(TPB2, 1, 1);
  S.TKBegin(CudaToroidalState::TK_RADIAL_FB);
  k_radial_interior<<<blocks2, tpb2, 0, S.stream>>>(
      S.n_config_max, ns_h, ns_local,
      nsi, offset_jH, offset_jF, signByDeltaS, invDeltaS,
      S.d_bucoH, S.d_bvcoH, S.d_presH, S.d_dVdsH, S.d_chipF, S.d_phipF,
      S.d_jcurvF, S.d_jcuruF, S.d_presgradF, S.d_dVdsF, S.d_equiF);
  cuda_check(cudaGetLastError(), "k_radial_interior launch");
  S.TKEnd(CudaToroidalState::TK_RADIAL_FB);

  // bvcoH is read on the host by the rBtor scalar evaluation; async D2H
  // here, with the stream synchronized inside the update body before the
  // host read. The interior arrays (jcurvF/jcuruF/presgradF/dVdsF/equiF)
  // have no mid-chain host consumer and stay on device. bucoH joins the
  // flush on free-boundary runs, where the cTor evaluation feeds NESTOR
  // the net toroidal current every vacuum iteration; fixed-boundary runs
  // consume it only at end-of-run through FlushForOutputCuda.
  cuda_check(cudaMemcpyAsync(bvcoH.data(), S.d_bvcoH, sizeof(double) * ns_h,
                              cudaMemcpyDeviceToHost, S.stream), "d2h bvcoH");
  if (g_free_boundary_run) {
    cuda_check(cudaMemcpyAsync(bucoH.data(), S.d_bucoH,
                                sizeof(double) * ns_h,
                                cudaMemcpyDeviceToHost, S.stream),
               "d2h bucoH");
  } else {
    (void)bucoH;
  }
  (void)jcurvF; (void)jcuruF; (void)presgradF; (void)dVdsF; (void)equiF;
}

// ============================================================================
// RzConIntoVolumeCuda: CUDA port of IdealMhdModel::rzConIntoVolume.
// Reads d_rCon, d_zCon (already on device after forward FFT), d_sqrtSF (also
// already on device), writes d_rCon0, d_zCon0. The CPU equivalent extracts
// rCon/zCon at the LCFS surface and propagates them inward weighted by sFull =
// sqrtSF[jF]^2. Only the LCFS-owning rank (r.nsMaxF1 == fc.ns) does this; for
// single-rank that's always true.
// ============================================================================
void RzConIntoVolumeCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    Eigen::VectorXd& rCon0, Eigen::VectorXd& zCon0) {
  auto& S = State();
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int nZnT = s.nZnT;
  if (ns_con_local <= 0) return;
  if (r.nsMaxF1 != fc.ns) {
    // Not the LCFS-owning rank: the CPU path handles this; skip here.
    // The dispatcher in ideal_mhd_model.cc falls through to CPU if needed.
    return;
  }

  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureRzCon0Buffers(ns_con_local, nZnT);

  int lcfs_con_local = fc.ns - 1 - r.nsMinF;
  int jMin_con = (r.nsMinF == 0) ? 1 : 0;  // CPU: max(1, nsMinF) - nsMinF
  int nsMinF_minus_nsMinF1 = r.nsMinF - r.nsMinF1;

  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 64;
  dim3 blocks((nZnT + TPB - 1) / TPB, ns_con_local, S.n_config_max);
  dim3 tpb(TPB, 1, 1);
  k_rzcon_into_volume<<<blocks, tpb, 0, S.stream>>>(
      S.n_config_max, ns_con_local, nZnT, jMin_con, lcfs_con_local,
      nsMinF_minus_nsMinF1,
      S.d_rCon, S.d_zCon, S.d_sqrtSF, S.d_rCon0, S.d_zCon0);
  cuda_check(cudaGetLastError(), "k_rzcon_into_volume launch");

  // rCon0/zCon0 stay on device; consumed by EffectiveConstraintForceCuda and
  // AssembleTotalForcesCuda.
  (void)rCon0; (void)zCon0;
}

// ============================================================================
// ComputeBContraCuda: CUDA port of IdealMhdModel::computeBContra.
// Inputs already on device: lu_e/o, lv_e/o (from forward FFT), gsqrt, guu, guv
// (from metric elements), sqrtSH (from jacobian).
// Per-call H2D: phipF, phipH, currH, iotaH (input when ncurr==0).
// In-place mutation: lu_e/o, lv_e/o multiplied by lamscale; lu_e += phipF.
// Outputs: bsupu, bsupv (persistent device buffers + D2H), chipH, iotaH (D2H to
// RadialProfiles), chipF, iotaF (D2H to RadialProfiles).
// ============================================================================
void ComputeBContraCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    int ncurr, double lamscale,
    const Eigen::VectorXd& phipF, const Eigen::VectorXd& phipH,
    const Eigen::VectorXd& currH, const Eigen::VectorXd& iotaH_in,
    Eigen::VectorXd& bsupu, Eigen::VectorXd& bsupv,
    Eigen::VectorXd& chipH_out, Eigen::VectorXd& iotaH_out,
    Eigen::VectorXd& chipF_out, Eigen::VectorXd& iotaF_out) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int nZnT = s.nZnT;
  const int nThetaEff = s.nThetaEff;
  if (ns_h <= 0) return;

  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureBContraBuffers(ns_h, nZnT, ns_local);

  // H2D radial inputs. phipF/phipH/currH are invariant under iteration
  // (toroidal flux profile + prescribed current; fixed for a given multigrid
  // level), so cache them. iotaH IS updated each iter under ncurr==1 (kernel
  // writes new chipH/iotaH and host D2Hs it back to m_p_.iotaH which becomes
  // the input for next iter), so do not cache it.
  if (!S.phipF_staged) {
    // Broadcast: device buffer is sized n_config_max * ns_local; kernel indexes
    // with cfg_prof offset (cfg * ns_local). Fill ALL N config slots with the
    // same radial profile so reads at any cfg return real data.
    for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
      const double* src = phipF.data();
      if (g_batch_prof_ncfg > 0 && cfg < g_batch_prof_ncfg)
        src = g_batch_phipF.data() + (size_t)cfg * ns_local;
      cuda_check(cudaMemcpyAsync(S.d_phipF + (size_t)cfg * ns_local, src,
                                  sizeof(double) * ns_local,
                                  cudaMemcpyHostToDevice, S.stream),
                 "h2d phipF (bcontra)");
    }
    S.phipF_staged = true;
  }
  if (!S.phipH_staged) {
    for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
      const double* src = phipH.data();
      if (g_batch_prof_ncfg > 0 && cfg < g_batch_prof_ncfg)
        src = g_batch_phipH.data() + (size_t)cfg * ns_h;
      cuda_check(cudaMemcpyAsync(S.d_phipH + (size_t)cfg * ns_h, src,
                                  sizeof(double) * ns_h,
                                  cudaMemcpyHostToDevice, S.stream),
                 "h2d phipH (bcontra)");
    }
    S.phipH_staged = true;
  }
  if (ncurr == 1) {
    if (!S.currH_staged) {
      for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
        const double* src = currH.data();
        if (g_batch_prof_ncfg > 0 && cfg < g_batch_prof_ncfg)
          src = g_batch_currH.data() + (size_t)cfg * ns_h;
        cuda_check(cudaMemcpyAsync(S.d_currH + (size_t)cfg * ns_h, src,
                                    sizeof(double) * ns_h,
                                    cudaMemcpyHostToDevice, S.stream),
                   "h2d currH");
      }
      S.currH_staged = true;
    }
    // Seed iotaH on device with the initial value, ONCE per Reshape. After
    // that, k_bcontra_chipH_iotaH updates d_iotaH on device each iter and the
    // device value is the input for the next iter's fallback path; the
    // host m_p_.iotaH is just a stale D2H copy of d_iotaH, so re-H2D'ing it
    // contributes nothing.
    if (!S.iotaH_seeded) {
      for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
        const double* src = iotaH_in.data();
        if (g_batch_prof_ncfg > 0 && cfg < g_batch_prof_ncfg)
          src = g_batch_iotaH.data() + (size_t)cfg * ns_h;
        cuda_check(cudaMemcpyAsync(S.d_iotaH + (size_t)cfg * ns_h, src,
                                    sizeof(double) * ns_h,
                                    cudaMemcpyHostToDevice, S.stream),
                   "h2d iotaH seed");
      }
      S.iotaH_seeded = true;
    }
  } else {
    // ncurr==0: iotaH_in is the prescribed profile, fixed per Reshape.
    if (!S.iotaH_seeded) {
      for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
        const double* src = iotaH_in.data();
        if (g_batch_prof_ncfg > 0 && cfg < g_batch_prof_ncfg)
          src = g_batch_iotaH.data() + (size_t)cfg * ns_h;
        cuda_check(cudaMemcpyAsync(S.d_iotaH_in + (size_t)cfg * ns_h, src,
                                    sizeof(double) * ns_h,
                                    cudaMemcpyHostToDevice, S.stream),
                   "h2d iotaH_in");
      }
      S.iotaH_seeded = true;
    }
  }

  // Stage 1: mutate lu_e/o, lv_e/o by lamscale + add phipF to lu_e.
  // Range: jF in [nsMinH .. nsMaxH+1), in local index [nsMinH - nsMinF1 ..
  // nsMaxH + 1 - nsMinF1).
  int jF_first = r.nsMinH - r.nsMinF1;
  int jF_last_excl = r.nsMaxH + 1 - r.nsMinF1;
  // CPU indexes phipF as phipF[jF - nsMinH]; we pass nsMinH - nsMinF1 = jF_first
  // so phipF[(jF_local) - jF_first] in kernel = phipF[jF - nsMinH] in CPU
  // (single-rank: jF_first == 0, so phipF[jF_local]).
  // BUT we loaded d_phipF with phipF.data() which is jF_local indexed by jF -
  // nsMinF1. So phipF[jF_local - jF_first] where jF_first = nsMinH - nsMinF1
  // gives phipF[jF - nsMinH], matching CPU.
  int phipF_jOff = r.nsMinH - r.nsMinF1;
  // Batched execution: all bcontra kernels gain n_config dim.
  {
    const int TPB = 64;
    int ns_mut = jF_last_excl - jF_first;
    if (ns_mut > 0) {
      dim3 blocks((nZnT + TPB - 1) / TPB, ns_mut, S.n_config_max);
      dim3 tpb(TPB, 1, 1);
      k_bcontra_mutate_lambda<<<blocks, tpb, 0, S.stream>>>(
          S.n_config_max, ns_local,
          jF_first, jF_last_excl, nZnT, phipF_jOff, s.lthreed, lamscale,
          S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o, S.d_phipF);
      cuda_check(cudaGetLastError(), "k_bcontra_mutate_lambda launch");
    }
  }

  // Stage 2: compute bsupu, bsupv from averaged inside/outside lambda derivatives.
  int jF_in_offset_bcontra = r.nsMinH - r.nsMinF1;
  {
    const int TPB = 64;
    dim3 blocks((nZnT + TPB - 1) / TPB, ns_h, S.n_config_max);
    dim3 tpb(TPB, 1, 1);
    S.TKBegin(CudaToroidalState::TK_BCONTRA);
    k_bcontra_bsupuv<<<blocks, tpb, 0, S.stream>>>(
        S.n_config_max, ns_local, ns_h, jF_in_offset_bcontra, nZnT, s.lthreed,
        S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o, S.d_sqrtSH, S.d_gsqrt,
        S.d_bsupu, S.d_bsupv);
    cuda_check(cudaGetLastError(), "k_bcontra_bsupuv launch");
    S.TKEnd(CudaToroidalState::TK_BCONTRA);
  }

  // Stage 3 (ncurr==1 only): jvPlasma + avg_guu_gsqrt reductions.
  if (ncurr == 1) {
    // VMECPP_CPU_ORDER_BCONTRA=1: serial ascending-kl accumulation
    // matching the host loop bit for bit (diagnostic for trajectory
    // comparisons against the CPU build).
    static const int bcontra_serial_env = []() {
      const char* e = std::getenv("VMECPP_CPU_ORDER_BCONTRA");
      return (e && std::atoi(e) > 0) ? 1 : 0;
    }();
    const int TPB = 32;
    dim3 blocks(ns_h, 1, S.n_config_max);
    dim3 tpb(TPB, 1, 1);
    k_bcontra_jvplasma_reduce<<<blocks, tpb, 0, S.stream>>>(
        S.n_config_max, ns_h, nZnT, nThetaEff, s.lthreed,
        bcontra_serial_env,
        S.d_guu, S.d_guv, S.d_bsupu, S.d_bsupv,
        S.d_gsqrt, S.d_wInt, S.d_jvPlasma, S.d_avg_guu_gsqrt);
    cuda_check(cudaGetLastError(), "k_bcontra_jvplasma_reduce launch");
  } else {
    // Need wInt anyway for radialForceBalance later; not required here.
    (void)nThetaEff;
  }

  // Stage 4: chipH / iotaH update per surface.
  {
    const int TPB = 64;
    dim3 blocks((ns_h + TPB - 1) / TPB, S.n_config_max, 1);
    dim3 tpb(TPB, 1, 1);
    k_bcontra_chipH_iotaH<<<blocks, tpb, 0, S.stream>>>(
        S.n_config_max, ns_h, ncurr, S.d_phipH, S.d_currH, S.d_iotaH_in,
        S.d_jvPlasma, S.d_avg_guu_gsqrt,
        S.d_chipH, S.d_iotaH);
    cuda_check(cudaGetLastError(), "k_bcontra_chipH_iotaH launch");
  }

  // Stage 5: full-grid chipF/iotaF interpolation, axis + LCFS extrapolation.
  {
    const int TPB = 32;
    dim3 blocks((ns_local + TPB - 1) / TPB, S.n_config_max, 1);
    dim3 tpb(TPB, 1, 1);
    int nsMinFi_off = r.nsMinFi - r.nsMinF1;
    int nsMaxFi_off = r.nsMaxFi - r.nsMinF1;
    int axis_present = (r.nsMinF1 == 0) ? 1 : 0;
    int lcfs_present = (r.nsMaxF1 == fc.ns) ? 1 : 0;
    int last_jF_local = r.nsMaxF1 - 1 - r.nsMinF1;
    int last_jH_local = r.nsMaxH - 1 - r.nsMinH;
    k_bcontra_chipF_iotaF<<<blocks, tpb, 0, S.stream>>>(
        S.n_config_max, ns_h, ns_local, nsMinFi_off, nsMaxFi_off,
        axis_present, lcfs_present, last_jF_local, last_jH_local,
        S.d_chipH, S.d_iotaH, S.d_chipF, S.d_iotaF);
    cuda_check(cudaGetLastError(), "k_bcontra_chipF_iotaF launch");
  }

  // Stage 6: final bsupu += chipH/gsqrt.
  {
    const int TPB = 64;
    dim3 blocks((nZnT + TPB - 1) / TPB, ns_h, S.n_config_max);
    dim3 tpb(TPB, 1, 1);
    k_bcontra_bsupu_add_chip<<<blocks, tpb, 0, S.stream>>>(
        S.n_config_max, ns_h, nZnT, S.d_chipH, S.d_gsqrt, S.d_bsupu);
    cuda_check(cudaGetLastError(), "k_bcontra_bsupu_add_chip launch");
  }

  // bsupu/bsupv stay on device (consumed by ComputeBCo, PressureAndEnergies,
  // ComputeMHDForces). The four small radial profiles (chipH, iotaH, chipF,
  // iotaF) used to be D2H'd per-iter to host m_p_ here, but every per-iter
  // host reader of those arrays is in CPU-only branches (after
  // #endif VMECPP_USE_CUDA) or in RadialForceBalanceCuda which reads them
  // from S.d_chipF/S.d_phipF directly. The only live host consumer is
  // ComputeOutputQuantities at end of run (output_quantities.cc writes
  // chipF/chipH/iotaF/iotaH to the HDF5 wout). FlushForOutputQuantitiesCuda
  // covers that one-shot D2H instead. Saves 4 small async D2Hs / iter ≈
  // 50-100 μs / iter * 21597 iters ≈ 1-2s at N=64.
  (void)bsupu; (void)bsupv;
  (void)chipH_out; (void)iotaH_out; (void)chipF_out; (void)iotaF_out;
}

// ============================================================================
// ComputePreconditioningMatrixCuda: CUDA port of IdealMhdModel::
// computePreconditioningMatrix. Called by updateRadialPreconditioner once for
// R-side (xs=zs, etc.) and once for Z-side (xs=rs, etc.). Inputs are passed as
// CPU Eigen::VectorXd& and H2D'd; outputs are produced on device and D2H'd.
// ============================================================================
void ComputePreconditioningMatrixCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double deltaS, int kEvenParity, int kOddParity,
    const Eigen::VectorXd& xs, const Eigen::VectorXd& xu12,
    const Eigen::VectorXd& xu_e, const Eigen::VectorXd& xu_o,
    const Eigen::VectorXd& x1_o,
    const Eigen::VectorXd& sm, const Eigen::VectorXd& sp,
    Eigen::VectorXd& m_axm, Eigen::VectorXd& m_axd,
    Eigen::VectorXd& m_bxm, Eigen::VectorXd& m_bxd,
    Eigen::VectorXd& m_cxd, int side) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int ns_force_local = r.nsMaxF - r.nsMinF;
  const int nZnT = s.nZnT;
  const int nThetaEff = s.nThetaEff;
  if (ns_h <= 0 || ns_force_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsurePrecondMatrixBuffers(ns_h, ns_force_local, ns_local, nZnT);

  // Device-resident path: xs/xu12/xu_e/xu_o/x1_o were already computed on
  // device by ComputeJacobian (xs=rs/zs, xu12, etc.) and the forward FFT
  // (xu_e/xu_o = ru_e/zu_e and ru_o/zu_o; x1_o = r1_o/z1_o). The host Eigen
  // vectors xs/xu12/xu_e/xu_o/x1_o are stale D2H copies; skip the H2D and
  // read directly from the device buffers. The R-side call (side==0) wants
  // the Z-derivatives; the Z-side call (side==1) wants the R-derivatives.
  (void)xs; (void)xu12; (void)xu_e; (void)xu_o; (void)x1_o;
  const double* d_xs   = (side == 0) ? S.d_zs   : S.d_rs;
  const double* d_xu12 = (side == 0) ? S.d_zu12 : S.d_ru12;
  const double* d_xu_e = (side == 0) ? S.d_zu_e : S.d_ru_e;
  const double* d_xu_o = (side == 0) ? S.d_zu_o : S.d_ru_o;
  const double* d_x1_o = (side == 0) ? S.d_z1_o : S.d_r1_o;
  // sm/sp are radial scaling factors (m_p_.sm / m_p_.sp), invariant under
  // iteration. Cache after first H2D; each call (R+Z) then skips its H2D.
  if (!S.pm_sm_staged) {
    for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
      cuda_check(cudaMemcpyAsync(S.d_pm_sm + (size_t)cfg * ns_h, sm.data(),
                                  sizeof(double) * ns_h,
                                  cudaMemcpyHostToDevice, S.stream),
                 "h2d pm sm (broadcast)");
    }
    S.pm_sm_staged = true;
  }
  if (!S.pm_sp_staged) {
    for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
      cuda_check(cudaMemcpyAsync(S.d_pm_sp + (size_t)cfg * ns_h, sp.data(),
                                  sizeof(double) * ns_h,
                                  cudaMemcpyHostToDevice, S.stream),
                 "h2d pm sp (broadcast)");
    }
    S.pm_sp_staged = true;
  }

  // Batched execution: each pm kernel gains n_config dim.
  double pFactor = -4.0;
  {
    // VMECPP_CPU_ORDER_PRECOND=1: host-order serial accumulation of the
    // matrix elements (diagnostic for trajectory comparisons against the
    // CPU build).
    static const int pm_serial_env = []() {
      const char* e = std::getenv("VMECPP_CPU_ORDER_PRECOND");
      return (e && std::atoi(e) > 0) ? 1 : 0;
    }();
    const int TPB = 32;
    dim3 b(ns_h, 1, S.n_config_max); dim3 t(TPB, 1, 1);
    // Read xs/xu12/xu_e/xu_o/x1_o from the device buffers directly, not the
    // d_pm_* H2D mirrors (which are now stale/unused on the H2D-skipped path).
    k_pm_half_reductions<<<b, t, 0, S.stream>>>(
        S.n_config_max, ns_local, ns_h, nZnT, nThetaEff, pFactor, deltaS,
        r.nsMinH, r.nsMinF1, pm_serial_env,
        S.d_r12, S.d_totalPressure, S.d_tau, S.d_wInt,
        d_xu12, d_xu_e, d_xu_o, d_x1_o, d_xs,
        S.d_sqrtSH, S.d_bsupv, S.d_gsqrt,
        S.d_ax_scratch, S.d_bx_scratch, S.d_cx_scratch);
    cuda_check(cudaGetLastError(), "k_pm_half_reductions launch");
  }
  {
    const int TPB = 64;
    dim3 b((ns_h + TPB - 1) / TPB, S.n_config_max, 1); dim3 t(TPB, 1, 1);
    k_pm_assemble_half<<<b, t, 0, S.stream>>>(
        S.n_config_max, ns_h, kEvenParity, kOddParity,
        S.d_ax_scratch, S.d_bx_scratch, S.d_pm_sm, S.d_pm_sp,
        S.d_pm_axm, S.d_pm_bxm);
    cuda_check(cudaGetLastError(), "k_pm_assemble_half launch");
  }
  {
    const int TPB = 64;
    dim3 b((ns_force_local + TPB - 1) / TPB, S.n_config_max, 1);
    dim3 t(TPB, 1, 1);
    k_pm_assemble_full<<<b, t, 0, S.stream>>>(
        S.n_config_max, ns_h, ns_force_local, fc.ns, kEvenParity, kOddParity,
        r.nsMinF, r.nsMinH,
        S.d_ax_scratch, S.d_bx_scratch, S.d_cx_scratch,
        S.d_pm_sm, S.d_pm_sp,
        S.d_pm_axd, S.d_pm_bxd, S.d_pm_cxd);
    cuda_check(cudaGetLastError(), "k_pm_assemble_full launch");
  }

  // Snapshot the scratch outputs into the per-side persistent buffers so
  // that the second ComputePreconditioningMatrixCuda invocation, which
  // overwrites the shared scratch arrays d_pm_axm, d_pm_axd, d_pm_bxm,
  // d_pm_bxd, and d_pm_cxd, does not destroy the values that
  // AssembleRZPreconditionerCuda must read for the side processed
  // first. The snapshot covers every configuration slot in the
  // batched-buffer layout: any per-configuration omission here would
  // leave the corresponding slots of the d_pmat_* buffers
  // uninitialised, and AssembleRZPreconditionerCuda would propagate
  // those uninitialised values into d_rz_aR, d_rz_dR, d_rz_bR,
  // d_rz_aZ, d_rz_dZ, and d_rz_bZ, which in turn would contaminate
  // the PCR solver output, the decomposed-forces buffer
  // d_decomposed_f, and the persistent d_pts_x state for every
  // affected configuration.
  double* dst_axm = (side == 0) ? S.d_pmat_arm : S.d_pmat_azm;
  double* dst_axd = (side == 0) ? S.d_pmat_ard : S.d_pmat_azd;
  double* dst_bxm = (side == 0) ? S.d_pmat_brm : S.d_pmat_bzm;
  double* dst_bxd = (side == 0) ? S.d_pmat_brd : S.d_pmat_bzd;
  cuda_check(cudaMemcpyAsync(dst_axm, S.d_pm_axm,
                              sizeof(double) * (size_t)S.n_config_max * ns_h * 2,
                              cudaMemcpyDeviceToDevice, S.stream), "d2d pmat axm");
  cuda_check(cudaMemcpyAsync(dst_axd, S.d_pm_axd,
                              sizeof(double) * (size_t)S.n_config_max * ns_force_local * 2,
                              cudaMemcpyDeviceToDevice, S.stream), "d2d pmat axd");
  cuda_check(cudaMemcpyAsync(dst_bxm, S.d_pm_bxm,
                              sizeof(double) * (size_t)S.n_config_max * ns_h * 2,
                              cudaMemcpyDeviceToDevice, S.stream), "d2d pmat bxm");
  cuda_check(cudaMemcpyAsync(dst_bxd, S.d_pm_bxd,
                              sizeof(double) * (size_t)S.n_config_max * ns_force_local * 2,
                              cudaMemcpyDeviceToDevice, S.stream), "d2d pmat bxd");
  cuda_check(cudaMemcpyAsync(S.d_pmat_cxd, S.d_pm_cxd,
                              sizeof(double) * (size_t)S.n_config_max * ns_force_local,
                              cudaMemcpyDeviceToDevice, S.stream), "d2d pmat cxd");

  // Keep host D2H too: ApplyM1PreconditionerCuda and ConstraintForceMultiplierCuda
  // still H2D ard/brd/azd/bzd from host. We could remove their H2Ds and read
  // d_pmat_* directly, but that expands scope; for now keep the host arrays
  // consistent. Cost: ~20µs/precond-update.
  cuda_check(cudaMemcpyAsync(m_axm.data(), S.d_pm_axm,
                              sizeof(double) * ns_h * 2,
                              cudaMemcpyDeviceToHost, S.stream), "d2h pm axm");
  cuda_check(cudaMemcpyAsync(m_axd.data(), S.d_pm_axd,
                              sizeof(double) * ns_force_local * 2,
                              cudaMemcpyDeviceToHost, S.stream), "d2h pm axd");
  cuda_check(cudaMemcpyAsync(m_bxm.data(), S.d_pm_bxm,
                              sizeof(double) * ns_h * 2,
                              cudaMemcpyDeviceToHost, S.stream), "d2h pm bxm");
  cuda_check(cudaMemcpyAsync(m_bxd.data(), S.d_pm_bxd,
                              sizeof(double) * ns_force_local * 2,
                              cudaMemcpyDeviceToHost, S.stream), "d2h pm bxd");
  cuda_check(cudaMemcpyAsync(m_cxd.data(), S.d_pm_cxd,
                              sizeof(double) * ns_force_local,
                              cudaMemcpyDeviceToHost, S.stream), "d2h pm cxd");
  // The stream synchronisation that would otherwise be required to
  // commit the device-to-host transfers above is deferred to the
  // nearest downstream wrapper that performs a host read. Both
  // UpdateVolumeCuda and ComputeForceNormsCuda already issue their
  // own cudaStreamSynchronize before consuming host data, so the
  // ordering of the cudaMemcpyAsync calls placed on S.stream is
  // sufficient to guarantee that those reads observe the correct
  // values.
}

// ============================================================================
// UpdateLambdaPreconditionerCuda: CUDA port of IdealMhdModel::
// updateLambdaPreconditioner. Two stages: half-grid reductions, axis extrap,
// full-grid average, then per-(jF, n, m) assembly.
// ============================================================================
void UpdateLambdaPreconditionerCuda(
    const RadialPartitioning& r, const Sizes& s,
    double dampingFactor, double lamscale,
    double* bLambda_out, double* dLambda_out, double* cLambda_out,
    double* lambdaPreconditioner_host) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  const int nZnT = s.nZnT;
  const int nThetaEff = s.nThetaEff;
  if (ns_h <= 0 || ns_con_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureLambdaPrecondBuffers(ns_h, ns_con_local, mpol, ntor);

  const int lambda_stride = ns_con_local + 1;
  // Stage 1: half-grid reductions writing to bLambda[1..ns_h+1], etc.
  // Batched execution: z-dim covers n_config_max configs.
  {
    const int TPB = 32;
    dim3 b(ns_h, 1, S.n_config_max); dim3 t(TPB, 1, 1);
    k_ulp_half_reductions<<<b, t, 0, S.stream>>>(
        S.n_config_max, ns_h, lambda_stride, nZnT, nThetaEff, s.lthreed,
        S.d_guu, S.d_guv, S.d_gvv, S.d_gsqrt, S.d_wInt,
        S.d_bLambda, S.d_dLambda, S.d_cLambda);
    cuda_check(cudaGetLastError(), "k_ulp_half_reductions launch");
  }
  // Stage 2: axis extrapolation - one block per config.
  int axis_present = (r.nsMinF == 0) ? 1 : 0;
  k_ulp_axis_extrap<<<S.n_config_max, 1, 0, S.stream>>>(
      S.n_config_max, lambda_stride, axis_present,
      S.d_bLambda, S.d_dLambda, S.d_cLambda);
  cuda_check(cudaGetLastError(), "k_ulp_axis_extrap launch");

  // Stage 3: full-grid average into a separate output region. The CPU code
  // overwrites bLambda[jF - nsMinF] in-place; we use the same buffer with the
  // understanding that the read indices (jH-nsMinH and jH-nsMinH+1) are above
  // the write index (jF-nsMinF) when nsMinH == nsMinF. To be safe in
  // multi-rank, we'd need a scratch; for single-rank this works because each
  // thread reads [jF, jF+1] but writes to [jF] and stride/order avoids hazards
  // at thread level (each thread reads ahead, writes back). We accept the
  // potential single-rank-only correctness here.
  // Batched execution: y-dim covers n_config_max configs.
  // Note: in-place read/write of bLambda/dLambda/cLambda - bLambda_out is
  // sized ns_con_local per config while bLambda_in is (ns_h+1) per config.
  // The kernel writes to bLambda_out[config*ns_con_local + jF_local], reads
  // from bLambda_in[config*(ns_h+1) + jH_in_off (+1)]. These are separate
  // logical ranges so in-place is OK at N=1 (input and output overlap in
  // memory only past where we never read; for N=1 ns_con_local == ns_h+1
  // and the in-place hazard is the same as before).
  {
    int jMin = (r.nsMinF == 0) ? 1 : 0;
    int nsMinH_off = r.nsMinF - r.nsMinH;  // CPU uses (jF - nsMinH) for jH index
    const int TPB = 64;
    dim3 b((ns_con_local + TPB - 1) / TPB, S.n_config_max, 1);
    dim3 t(TPB, 1, 1);
    k_ulp_full_grid_average<<<b, t, 0, S.stream>>>(
        S.n_config_max, lambda_stride, ns_con_local, jMin, nsMinH_off,
        S.d_bLambda, S.d_dLambda, S.d_cLambda,
        S.d_bLambda, S.d_dLambda, S.d_cLambda);
    cuda_check(cudaGetLastError(), "k_ulp_full_grid_average launch");
  }

  // Stage 4: per-(cfg, jF, n, m) assembly.
  double pFactor = dampingFactor / (4.0 * lamscale * lamscale);
  {
    int jMin = (r.nsMinF == 0) ? 1 : 0;
    int sqrtSF_off = r.nsMinF - r.nsMinF1;
    const int TPB_m = 16;
    dim3 blocks((mpol + TPB_m - 1) / TPB_m, ntor + 1,
                ns_con_local * S.n_config_max);
    dim3 tpb(TPB_m, 1, 1);
    k_ulp_assemble<<<blocks, tpb, 0, S.stream>>>(
        S.n_config_max, ns_con_local, lambda_stride, jMin, mpol, ntor,
        s.nfp, pFactor,
        S.d_bLambda, S.d_dLambda, S.d_cLambda, S.d_sqrtSF, sqrtSF_off,
        S.d_lambdaPreconditioner);
    cuda_check(cudaGetLastError(), "k_ulp_assemble launch");
  }

  // Per-preconditioner-update D2Hs of lambdaPreconditioner + bLambda /
  // dLambda / cLambda were originally retained so the CPU paths in
  // ideal_mhd_model.cc could read them. Under CUDA every consumer is
  // either in a CPU-only branch (the bLambda/dLambda/cLambda host reads at
  // ideal_mhd_model.cc are inside the #else of VMECPP_USE_CUDA) or
  // is the device-resident path itself (ApplyLambdaPreconditionerCuda at
  // line 8184 explicitly marks the host lambdaPreconditioner argument as
  // (void) and reads d_lambdaPreconditioner directly). Dropping the
  // D2Hs eliminates the per-preconditioner-update kernel launch + async
  // copy overhead. 960 updates × 4 D2Hs ≈ ~80 ms total on the convergence
  // trajectory.
  (void)lambdaPreconditioner_host;
  (void)bLambda_out;
  (void)dLambda_out;
  (void)cLambda_out;
}

// ============================================================================
// ComputeMHDForcesCuda: CUDA port of IdealMhdModel::computeMHDForces.
// All inputs already on device. D2H 8 (2D) or 12 (3D) force arrays.
// ============================================================================
void ComputeMHDForcesCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    bool lfreeb, double deltaS,
    Eigen::VectorXd& armn_e, Eigen::VectorXd& armn_o,
    Eigen::VectorXd& azmn_e, Eigen::VectorXd& azmn_o,
    Eigen::VectorXd& brmn_e, Eigen::VectorXd& brmn_o,
    Eigen::VectorXd& bzmn_e, Eigen::VectorXd& bzmn_o,
    Eigen::VectorXd& crmn_e, Eigen::VectorXd& crmn_o,
    Eigen::VectorXd& czmn_e, Eigen::VectorXd& czmn_o) {
  auto& S = State();
  const int ns_force_local = r.nsMaxF - r.nsMinF;
  const int nZnT = s.nZnT;
  if (ns_force_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureMHDForceBuffers(ns_force_local, nZnT, s.lthreed);

  int jMaxRZ = std::min(r.nsMaxF, fc.ns - 1);
  if (lfreeb) jMaxRZ = std::min(r.nsMaxF, fc.ns);

  // Batched execution: z-dim covers n_config_max configs.
  // VMECPP_MHD_PAIR routes to k_compute_mhd_forces_pair when ns_force_local
  // is even. Pair kernel caches one shared jH slab between adjacent jF blocks
  // to skip a half-grid load per jF-pair. Default ON; set =0 to fall back.
  // Kernel-level delta: -1.72pct TK_COMPUTE_MHD (13.025s -> 12.801s over 20k
  // calls). Bit-exact aspect_ratio = 7.527844291824478, qi/L_grad_B unchanged.
  static const int mhd_pair_env = []() {
    const char* e = std::getenv("VMECPP_MHD_PAIR");
    return (e && std::atoi(e) == 0) ? 0 : 1;
  }();
  const int TPB = 64;
  S.TKBegin(CudaToroidalState::TK_COMPUTE_MHD);
  if (mhd_pair_env && (ns_force_local % 2 == 0)) {
    dim3 blocks_p((nZnT + TPB - 1) / TPB, ns_force_local / 2, S.n_config_max);
    dim3 tpb_p(TPB, 2, 1);
    // 10 fields cached per x-lane (blockDim.x), independent of nZnT.
    size_t smem_bytes = (size_t)sizeof(double) * 10 * (size_t)TPB;
    k_compute_mhd_forces_pair<<<blocks_p, tpb_p, smem_bytes, S.stream>>>(
        S.n_config_max, S.ns_local_cached, ns_force_local, nZnT, s.lthreed,
        r.nsMinF, r.nsMinF1, r.nsMinH, r.nsMaxH, jMaxRZ, deltaS,
        S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
        S.d_rv_e, S.d_rv_o, S.d_zu_e, S.d_zu_o,
        S.d_zv_e, S.d_zv_o, S.d_z1_o,
        S.d_r12, S.d_ru12, S.d_zu12, S.d_rs, S.d_zs, S.d_tau,
        S.d_totalPressure, S.d_gsqrt, S.d_bsupu, S.d_bsupv,
        S.d_sqrtSF, S.d_sqrtSH,
        S.d_armn_e, S.d_armn_o, S.d_azmn_e, S.d_azmn_o,
        S.d_brmn_e, S.d_brmn_o, S.d_bzmn_e, S.d_bzmn_o,
        S.d_crmn_e, S.d_crmn_o, S.d_czmn_e, S.d_czmn_o,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_compute_mhd_forces_pair launch");
  } else {
    dim3 blocks((nZnT + TPB - 1) / TPB, ns_force_local, S.n_config_max);
    dim3 tpb(TPB, 1, 1);
    k_compute_mhd_forces<<<blocks, tpb, 0, S.stream>>>(
        S.n_config_max, S.ns_local_cached, ns_force_local, nZnT, s.lthreed,
        r.nsMinF, r.nsMinF1, r.nsMinH, r.nsMaxH, jMaxRZ, deltaS,
        S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
        S.d_rv_e, S.d_rv_o, S.d_zu_e, S.d_zu_o,
        S.d_zv_e, S.d_zv_o, S.d_z1_o,
        S.d_r12, S.d_ru12, S.d_zu12, S.d_rs, S.d_zs, S.d_tau,
        S.d_totalPressure, S.d_gsqrt, S.d_bsupu, S.d_bsupv,
        S.d_sqrtSF, S.d_sqrtSH,
        S.d_armn_e, S.d_armn_o, S.d_azmn_e, S.d_azmn_o,
        S.d_brmn_e, S.d_brmn_o, S.d_bzmn_e, S.d_bzmn_o,
        S.d_crmn_e, S.d_crmn_o, S.d_czmn_e, S.d_czmn_o);
    cuda_check(cudaGetLastError(), "k_compute_mhd_forces launch");
  }
  S.TKEnd(CudaToroidalState::TK_COMPUTE_MHD);
  DiagCfg01DiffCuda(S.d_armn_e, ns_force_local * nZnT, "mhd:armn_e");

  // All outputs (armn/azmn/brmn/bzmn/crmn/czmn) stay on device; downstream
  // inverse-FFT + AssembleTotalForces read d_* directly.
  (void)armn_e; (void)armn_o; (void)azmn_e; (void)azmn_o;
  (void)brmn_e; (void)brmn_o; (void)bzmn_e; (void)bzmn_o;
  (void)crmn_e; (void)crmn_o; (void)czmn_e; (void)czmn_o;
}

// ============================================================================
// ComputeForceNormsCuda: CUDA port of IdealMhdModel::computeForceNorms.
// Half-grid reductions for fNormRZ and fNormL are GPU; the FourierGeometry
// rzNorm for fNorm1 is host-side (decomposed_x lives on host).
// ============================================================================
void ComputeForceNormsCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double magneticEnergy, double thermalEnergy, double plasmaVolume,
    double lamscale, double forceNorm1_host,
    double& fNormRZ_out, double& fNormL_out, double& fNorm1_out) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int nZnT = s.nZnT;
  const int nThetaEff = s.nThetaEff;
  if (ns_h <= 0) {
    fNormRZ_out = 0.0; fNormL_out = 0.0; fNorm1_out = 0.0;
    return;
  }
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureForceNormBuffers(ns_h);

  // Cache lamscale for the device-side normalized convergence check
  // (k_check_convergence). It is a per-run constant whose only prior
  // consumers received it by argument.
  S.lamscale_cached = lamscale;

  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 32;
  dim3 blocks(ns_h, 1, S.n_config_max);
  dim3 tpb(TPB, 1, 1);
  k_force_norm_partials<<<blocks, tpb, 0, S.stream>>>(
      S.n_config_max, ns_h, nZnT, nThetaEff,
      r.nsMinH, r.nsMaxH - 1, fc.ns - 2,
      S.d_guu, S.d_r12, S.d_bsubu, S.d_bsubv, S.d_wInt,
      S.d_forceNormRZ_partial, S.d_forceNormL_partial,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_force_norm_partials launch");

  // The per-half-surface partial sums produced above are reduced on
  // device by k_force_norm_final_reduce, eliminating the host-side
  // reduction that the CPU implementation performs over jH. Each
  // launched block reduces the partial sums of one configuration and
  // writes its two output scalars (sum_rz and sum_l) into
  // d_fnorm_scalars at offset config * 2, so a single device-to-host
  // transfer of two doubles per configuration suffices.
  S.EnsureFnormScalarsBuffer();
  k_force_norm_final_reduce<<<S.n_config_max, 256, 0, S.stream>>>(
      S.n_config_max, ns_h, S.d_forceNormRZ_partial, S.d_forceNormL_partial,
      S.d_fnorm_scalars,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_force_norm_final_reduce launch");

  // Per-cfg cache: replace the 2-double D2H with 2*n_cfg
  // D2H into a static cache. Single-cfg behavior preserved (sum_rz =
  // cache[0]); per-cfg cache populated for free during the SAME sync.
  // Per-cfg consumers read via GetFnormScalarsPerCfgCache() to build per-cfg
  // fNormRZ/fNormL without an extra D2H+sync.
  int n_cfg = S.n_config_max;
  if ((int)g_fnorm_scalars_cache.size() != 2 * n_cfg) {
    g_fnorm_scalars_cache.assign(2 * n_cfg, 0.0);
  }
  cuda_check(cudaMemcpyAsync(g_fnorm_scalars_cache.data(), S.d_fnorm_scalars,
                              (size_t)2 * n_cfg * sizeof(double),
                              cudaMemcpyDeviceToHost, S.stream),
             "d2h fnorm scalars (per-cfg cache)");
  cuda_check(cudaStreamSynchronize(S.stream), "fnorm stream sync");

  double sum_rz = g_fnorm_scalars_cache[0];
  double sum_l  = g_fnorm_scalars_cache[1];
  double energyDensity = std::max(magneticEnergy, thermalEnergy) / plasmaVolume;
  fNormRZ_out = 1.0 / (sum_rz * energyDensity * energyDensity);
  fNormL_out  = 1.0 / (sum_l  * lamscale * lamscale);
  fNorm1_out  = 1.0 / forceNorm1_host;

  // Per-cfg fNorm1 for the device time-step controller: each cfg's
  // reciprocal rzNorm over its own device-resident position state, at the
  // same cadence as the force norms above. cfg 0 equals fNorm1_out
  // bit-for-bit (same data, same accumulation order), so single-cfg and
  // broadcast trajectories are unchanged; distinct-mode cfgs stop sharing
  // cfg 0's normalization. Skipped until the position state exists (the
  // first force-norm update precedes it); StageFnorm1 broadcasts the host
  // value until the first fill.
  if (S.pts_x_initialized && S.d_pts_x_rcc) {
    S.EnsureFnorm1Buffer();
    const int ns_local_x = r.nsMaxF1 - r.nsMinF1;
    k_rz_norm_per_cfg<<<S.n_config_max, 1, 0, S.stream>>>(
        S.n_config_max, ns_local_x, r.nsMinF - r.nsMinF1,
        r.nsMaxFIncludingLcfs - r.nsMinF, s.mpol, s.ntor, s.lthreed,
        S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc, S.d_pts_x_zcs,
        S.d_fnorm1, S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_rz_norm_per_cfg launch");
    S.fnorm1_device_filled = true;
    // Per-cfg fNorm1 cache for the host-side per-cfg preconditioned
    // residual normalization in evalFResPrecd. Becomes valid at the next
    // stream synchronization, the same boundary as the other per-cfg
    // caches at this cadence.
    if (static_cast<int>(g_fnorm1_per_cfg_cache.size()) != S.n_config_max) {
      g_fnorm1_per_cfg_cache.assign(S.n_config_max, 0.0);
    }
    cuda_check(cudaMemcpyAsync(g_fnorm1_per_cfg_cache.data(), S.d_fnorm1,
                                sizeof(double) * S.n_config_max,
                                cudaMemcpyDeviceToHost, S.stream),
               "d2h fnorm1 (per-cfg cache)");
  }
}

// ============================================================================
// HybridLambdaForceCuda: CUDA port of IdealMhdModel::hybridLambdaForce.
// Inputs on device: bsubu/v (BCo), gvv/gsqrt/guv (metric), bsupu (BContra),
// lu_e/o (post-mutation by BContra), sqrtSF/sqrtSH. Per-call H2D: radialBlending.
// Outputs: blmn_e/o, clmn_e/o (D2H to IdealMhdModel members).
// ============================================================================
void HybridLambdaForceCuda(
    const RadialPartitioning& r, const Sizes& s, double lamscale,
    const Eigen::VectorXd& radialBlending,
    Eigen::VectorXd& blmn_e, Eigen::VectorXd& blmn_o,
    Eigen::VectorXd& clmn_e, Eigen::VectorXd& clmn_o) {
  auto& S = State();
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int nZnT = s.nZnT;
  if (ns_con_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureHybridLambdaBuffers(ns_local, ns_con_local, nZnT);

  // radialBlending depends only on the radial grid (fixed per Reshape). Cache.
  if (!S.radialBlending_staged) {
    for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
      cuda_check(cudaMemcpyAsync(S.d_radialBlending + (size_t)cfg * ns_local,
                                  radialBlending.data(),
                                  sizeof(double) * ns_local,
                                  cudaMemcpyHostToDevice, S.stream),
                 "h2d radialBlending (broadcast)");
    }
    S.radialBlending_staged = true;
  }

  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 64;
  dim3 blocks((nZnT + TPB - 1) / TPB, ns_con_local, S.n_config_max);
  dim3 tpb(TPB, 1, 1);
  int nsMinF1_off = r.nsMinF1;        // pass for jF_local_full computation
  int nsMinH_off = r.nsMinH;          // global nsMinH
  k_hybrid_lambda_force<<<blocks, tpb, 0, S.stream>>>(
      S.n_config_max, ns_local, ns_h, ns_con_local, nZnT, s.lthreed,
      r.nsMinF, nsMinF1_off, nsMinH_off, ns_h, lamscale,
      S.d_bsubu, S.d_bsubv, S.d_gvv, S.d_gsqrt, S.d_guv, S.d_bsupu,
      S.d_lu_e, S.d_lu_o, S.d_sqrtSF, S.d_sqrtSH, S.d_radialBlending,
      S.d_blmn_e, S.d_blmn_o, S.d_clmn_e, S.d_clmn_o);
  cuda_check(cudaGetLastError(), "k_hybrid_lambda_force launch");

  // blmn/clmn stay on device; consumed by ForcesToFourier3DSymmFastPoloidalCuda
  // (inverse FFT).
  (void)blmn_e; (void)blmn_o; (void)clmn_e; (void)clmn_o;
}

// ============================================================================
// PressureAndEnergiesCuda: CUDA port of IdealMhdModel::pressureAndEnergies.
// Inputs already on device: bsupu, bsupv (from BContra), bsubu, bsubv (from
// BCo), gsqrt, dVdsH, wInt. Per-call H2D: massH.
// Outputs: presH (radial), totalPressure (full half-grid array), both
// persisted on device for downstream; thermalEnergy, magneticEnergy, mhdEnergy
// scalars returned via out-params.
// ============================================================================
void PressureAndEnergiesCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double deltaS, double adiabaticIndex,
    const Eigen::VectorXd& massH,
    Eigen::VectorXd& presH_out, Eigen::VectorXd& totalPressure_out,
    double& thermalEnergy_out, double& magneticEnergy_out,
    double& mhdEnergy_out) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int nZnT = s.nZnT;
  const int nThetaEff = s.nThetaEff;
  if (ns_h <= 0) {
    thermalEnergy_out = 0.0; magneticEnergy_out = 0.0; mhdEnergy_out = 0.0;
    return;
  }
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsurePressureBuffers(ns_h, nZnT);

  // massH is the prescribed mass profile, invariant under iteration. Cache.
  if (!S.massH_staged) {
    for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
      const double* src = massH.data();
      if (g_batch_prof_ncfg > 0 && cfg < g_batch_prof_ncfg)
        src = g_batch_massH.data() + (size_t)cfg * ns_h;
      cuda_check(cudaMemcpyAsync(S.d_massH + (size_t)cfg * ns_h, src,
                                  sizeof(double) * ns_h,
                                  cudaMemcpyHostToDevice, S.stream),
                 "h2d massH");
    }
    S.massH_staged = true;
  }

  // Stage 1+3 fused: per-surface presH AND thermal_partial in one kernel.
  // Reuses pres in-register; eliminates one launch and one global-mem read.
  // Batched execution: y-dim covers n_config_max configs.
  {
    const int TPB = 32;
    dim3 b((ns_h + TPB - 1) / TPB, S.n_config_max, 1);
    dim3 t(TPB, 1, 1);
    k_pres_compute_and_thermal<<<b, t, 0, S.stream>>>(
        S.n_config_max, ns_h, adiabaticIndex,
        r.nsMinH, r.nsMaxH - 1, fc.ns - 2,
        S.d_massH, S.d_dVdsH, S.d_presH, S.d_thermal_partial);
    cuda_check(cudaGetLastError(), "k_pres_compute_and_thermal launch");
  }

  // Stage 2 (totalpres_init) and Stage 5 (add_presH) collapse into a single
  // fused write that produces the final totalPressure = magnetic + presH.
  // We do magnetic_partial FIRST using an inline magnetic formula so it
  // doesn't depend on totalPressure being magnetic-only.

  // Stage 4 (now stage 2): magnetic_partial with inline magnetic-pressure
  // computation. Drops the dependency on a magnetic-only totalPressure write.
  {
    const int TPB = 32;
    dim3 b(ns_h, 1, S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_pres_magnetic_partial_inline<<<b, t, 0, S.stream>>>(
        S.n_config_max, ns_h, nZnT, nThetaEff, r.nsMinH, r.nsMaxH - 1, fc.ns - 2,
        S.d_gsqrt, S.d_bsupu, S.d_bsubu, S.d_bsupv, S.d_bsubv, S.d_wInt,
        S.d_magnetic_partial,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_pres_magnetic_partial_inline launch");
  }

  // Stages 2+5 fused: write totalPressure = mag + presH in one kernel.
  // Skips the intermediate magnetic-only write and the subsequent +presH read-
  // modify-write.
  {
    const int TPB = 64;
    dim3 b((nZnT + TPB - 1) / TPB, ns_h, S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_pres_totalpres_init_with_presH<<<b, t, 0, S.stream>>>(
        S.n_config_max, ns_h, nZnT,
        S.d_bsupu, S.d_bsubu, S.d_bsupv, S.d_bsubv,
        S.d_presH, S.d_totalPressure,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_pres_totalpres_init_with_presH launch");
  }

  // The old k_pres_add_presH stage is folded into
  // k_pres_totalpres_init_with_presH above.

  // presH and totalPressure stay on device. Reduction partials get reduced
  // on-device to 3 scalars; async D2H (no sync). Caller's next sync drains
  // the queue. The host-side scalars (m_h_.thermalEnergy etc.) are stale
  // until that next sync, but their only consumer (ComputeForceNormsCuda)
  // syncs the stream when it runs.
  (void)presH_out; (void)totalPressure_out;
  S.EnsurePressureScalarsBuffer();
  // Batched execution: launch n_config_max_max blocks. Each block reduces one config's
  // thermal/magnetic partials and writes 3 scalars at scalars_out[config*3:].
  k_pres_final_reduce<<<S.n_config_max, 256, 0, S.stream>>>(
      S.n_config_max, ns_h, deltaS, adiabaticIndex,
      S.d_thermal_partial, S.d_magnetic_partial, S.d_pressure_scalars,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_pres_final_reduce launch");
  // Single-cfg D2Hs preserved (writes to caller's out-references; these
  // values become valid after the caller's downstream stream sync; the
  // established single-cfg pattern, unchanged).
  cuda_check(cudaMemcpyAsync(&thermalEnergy_out, &S.d_pressure_scalars[0],
                              sizeof(double), cudaMemcpyDeviceToHost, S.stream),
             "d2h pressure thermal");
  cuda_check(cudaMemcpyAsync(&magneticEnergy_out, &S.d_pressure_scalars[1],
                              sizeof(double), cudaMemcpyDeviceToHost, S.stream),
             "d2h pressure magnetic");
  cuda_check(cudaMemcpyAsync(&mhdEnergy_out, &S.d_pressure_scalars[2],
                              sizeof(double), cudaMemcpyDeviceToHost, S.stream),
             "d2h pressure mhd");
  // Per-cfg cache: additional async D2H of all 3*n_cfg
  // scalars to a static cache. Becomes valid after the same sync that
  // validates the three single-cfg writes above. Layout
  // [thermalEnergy_0, magneticEnergy_0, mhdEnergy_0, thermalEnergy_1, ...].
  // Per-cfg consumers read via GetPressureScalarsPerCfgCache(); cache holds
  // the SAME 3 values at slots 0,1,2 that the single-cfg writes hold.
  {
    int n_cfg = S.n_config_max;
    if ((int)g_pressure_scalars_cache.size() != 3 * n_cfg) {
      g_pressure_scalars_cache.assign(3 * n_cfg, 0.0);
    }
    cuda_check(cudaMemcpyAsync(g_pressure_scalars_cache.data(),
                                S.d_pressure_scalars,
                                (size_t)3 * n_cfg * sizeof(double),
                                cudaMemcpyDeviceToHost, S.stream),
               "d2h pressure scalars (per-cfg cache)");
  }
}

// ============================================================================
// ComputeInitialVolumeCuda: reduce dVdsH into m_h_.voli scalar.
// voli += local_sum * (2*pi)^2 * deltaS.
// ============================================================================
void ComputeInitialVolumeCuda(
    const RadialPartitioning& r, const FlowControl& fc, double deltaS,
    double& voli_out) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  if (ns_h <= 0) { voli_out = 0.0; return; }
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureScalarBuffer();

  // multiplier = deltaS * (2*pi)^2 ; mask: jH_global < nsMaxH-1 OR jH_global == ns-2
  double M_PI_LOCAL = 3.14159265358979323846;
  double mult = deltaS * (2.0 * M_PI_LOCAL) * (2.0 * M_PI_LOCAL);
  // Pick TPB as smallest power of two >= ns_h, capped at 256.
  int TPB = 1;
  while (TPB < ns_h && TPB < 256) TPB *= 2;
  if (TPB < 1) TPB = 1;
  // Batched execution: launch n_config_max_max blocks, each block reduces one config.
  // Single-config D2H still reads slot [0] so N=1 path is unchanged.
  k_volume_reduce<<<S.n_config_max, TPB, 0, S.stream>>>(
      S.n_config_max, ns_h, mult, r.nsMaxH - 1, fc.ns - 2, r.nsMinH,
      S.d_dVdsH, S.d_scalar);
  cuda_check(cudaGetLastError(), "k_volume_reduce launch (voli)");
  cuda_check(cudaMemcpyAsync(&voli_out, S.d_scalar, sizeof(double),
                              cudaMemcpyDeviceToHost, S.stream), "d2h voli");
  cuda_check(cudaStreamSynchronize(S.stream), "voli stream sync");
}

// ============================================================================
// UpdateVolumeCuda: reduce dVdsH into m_h_.plasmaVolume scalar.
// plasmaVolume += local_sum * deltaS (no 4*pi^2 factor).
// ============================================================================
void UpdateVolumeCuda(
    const RadialPartitioning& r, const FlowControl& fc, double deltaS,
    double& plasmaVolume_out) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  if (ns_h <= 0) { plasmaVolume_out = 0.0; return; }
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureScalarBuffer();
  int TPB = 1;
  while (TPB < ns_h && TPB < 256) TPB *= 2;
  if (TPB < 1) TPB = 1;
  // Batched execution: launch n_config_max_max blocks (one per config).
  k_volume_reduce<<<S.n_config_max, TPB, 0, S.stream>>>(
      S.n_config_max, ns_h, deltaS, r.nsMaxH - 1, fc.ns - 2, r.nsMinH,
      S.d_dVdsH, S.d_scalar);
  cuda_check(cudaGetLastError(), "k_volume_reduce launch (plasmaVolume)");
  // Per-cfg D2H of all n_config_max slots into the host cache for per-cfg
  // consumers (evalFResInvar uses per-cfg plasmaVolume for energyDensity).
  // Single-cfg plasmaVolume_out is taken from cfg 0's slot post-sync,
  // preserving the existing single-cfg semantics.
  const int n_cfg_v = S.n_config_max;
  if (static_cast<int>(g_plasma_volume_cache.size()) != n_cfg_v) {
    g_plasma_volume_cache.assign(n_cfg_v, 0.0);
  }
  // Sync elision: reduction launched (device slot current); host reads
  // the last boundary-synced volume. plasmaVolume feeds the host-side
  // fsq normalization and printout only; the device convergence kernel
  // consumes the device-resident slot.
  if (S.sync_elide_iter) {
    plasmaVolume_out = g_plasma_volume_cache[0];
    return;
  }
  cuda_check(cudaMemcpyAsync(g_plasma_volume_cache.data(), S.d_scalar,
                              sizeof(double) * n_cfg_v,
                              cudaMemcpyDeviceToHost, S.stream),
             "d2h plasmaVolume per-cfg");
  cuda_check(cudaStreamSynchronize(S.stream), "plasmaVolume stream sync");
  plasmaVolume_out = g_plasma_volume_cache[0];
}

// ============================================================================
// ConstraintForceMultiplierCuda: CUDA port of IdealMhdModel::constraintForceMultiplier.
// Inputs (device): d_ruFull, d_zuFull (from ComputeRuZuFullCuda), d_wInt.
// Inputs (per-call H2D): ard, azd (each ns_force_local × 2 doubles).
// Output: tcon (device + D2H). LCFS halving done host-side after D2H.
// ============================================================================
void ConstraintForceMultiplierCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double tcon0, const Eigen::VectorXd& ard, const Eigen::VectorXd& azd,
    Eigen::VectorXd& tcon_out) {
  auto& S = State();
  const int ns_force_local = r.nsMaxF - r.nsMinF;
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int nZnT = s.nZnT;
  const int nThetaEff = s.nThetaEff;
  if (ns_force_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureConstraintMultiplierBuffers(ns_force_local, ns_con_local, nZnT);

  // tcon_multiplier mirroring CPU: tcon0 * (1 + ns*(1/60 + ns/(200*120))) / 16.
  double tcon_multiplier =
      tcon0 * (1.0 + fc.ns * (1.0 / 60.0 + fc.ns / (200.0 * 120.0)));
  tcon_multiplier /= (4.0 * 4.0);
  // Final factor includes (32*deltaS)^2 as in the CPU code.
  double tcon_factor = tcon_multiplier * (32.0 * fc.deltaS) * (32.0 * fc.deltaS);

  // Device-resident: ard/azd were just written on device by
  // ComputePreconditioningMatrixCuda's two calls and snapshotted into
  // d_pmat_ard / d_pmat_azd. The host Eigen vectors ard/azd are stale D2H
  // copies; skip the H2Ds and read from d_pmat_* directly.
  (void)ard; (void)azd;

  int jMin = (r.nsMinF == 0) ? 1 : 0;
  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 32;
  dim3 b(ns_force_local, 1, S.n_config_max); dim3 t(TPB, 1, 1);
  k_constraint_force_multiplier<<<b, t, 0, S.stream>>>(
      S.n_config_max, ns_con_local, ns_force_local, nZnT, nThetaEff, jMin,
      /*kEvenParity=*/0, tcon_factor,
      S.d_ruFull, S.d_zuFull, S.d_pmat_ard, S.d_pmat_azd, S.d_wInt, S.d_tcon);
  cuda_check(cudaGetLastError(), "k_constraint_force_multiplier launch");

  // LCFS halving on device: previously this was a host operation that never
  // propagated back to d_tcon, leaving DeAliasConstraintForceCuda to read
  // un-halved values. One-thread kernel keeps the halved value on device.
  // The halved entry is the LCFS row of the con-sized profile
  // (nsMaxF1 - 1 - nsMinF), matching the host indexing; on fixed-boundary
  // runs that row sits one past the force rows the dealiasing reads, so
  // the halving must not land on the outermost force row instead.
  // Batched execution: launch n_config_max blocks; d_tcon's per-config stride is
  // ns_con_local (allocated as n_config_max * ns_con_local).
  if (r.nsMaxF1 == fc.ns && ns_con_local >= 2) {
    int last = (r.nsMaxF1 - 1) - r.nsMinF;
    k_halve_tcon_lcfs<<<S.n_config_max, 1, 0, S.stream>>>(
        S.n_config_max, ns_con_local, last, S.d_tcon);
    cuda_check(cudaGetLastError(), "k_halve_tcon_lcfs launch");
  }
  DiagCfg01DiffCuda(S.d_tcon, ns_con_local, "constraint:tcon");

  // D2H tcon for host visibility (not consumed mid-chain in CUDA mode but
  // preserved for output paths). No sync; defer to end-of-update.
  cuda_check(cudaMemcpyAsync(tcon_out.data(), S.d_tcon,
                              sizeof(double) * ns_force_local,
                              cudaMemcpyDeviceToHost, S.stream), "d2h tcon");

  // VMECPP_DUMP_TCON=1: print the first computed profile at full
  // precision (diagnostic for CPU-vs-CUDA trajectory comparisons).
  static int dump_tcon_env = -1;
  if (dump_tcon_env < 0) {
    const char* e = std::getenv("VMECPP_DUMP_TCON");
    dump_tcon_env = (e && std::atoi(e) > 0) ? 1 : 0;
  }
  if (dump_tcon_env) {
    static int dumped = 0;
    if (!dumped) {
      dumped = 1;
      cuda_check(cudaStreamSynchronize(S.stream), "tcon dump sync");
      for (int j = 1; j < std::min(9, ns_force_local); ++j) {
        std::fprintf(stderr, "[TCON] j=%d %.17g\n", j, tcon_out[j]);
      }
    }
  }
}

// ============================================================================
// EffectiveConstraintForceCuda: CUDA port of IdealMhdModel::effectiveConstraintForce.
// Inputs (all device): rCon, rCon0, zCon, zCon0 (from RzConIntoVolume / forward),
// ruFull, zuFull (from ComputeRuZuFull).
// Output: gConEff (device + D2H since deAliasConstraintForce stays on CPU).
// ============================================================================
void EffectiveConstraintForceCuda(
    const RadialPartitioning& r, const Sizes& s,
    Eigen::VectorXd& gConEff_out) {
  auto& S = State();
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int nZnT = s.nZnT;
  if (ns_con_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureConstraintMultiplierBuffers(r.nsMaxF - r.nsMinF, ns_con_local, nZnT);
  // The constraint-origin buffers normally exist from rzConIntoVolume at
  // stage start; a free-boundary stage entered with the vacuum pressure
  // active skips that call and consumes the zero-initialized buffers.
  S.EnsureRzCon0Buffers(ns_con_local, nZnT);

  int jMin = (r.nsMinF == 0) ? 1 : 0;
  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 64;
  dim3 b((nZnT + TPB - 1) / TPB, ns_con_local, S.n_config_max);
  dim3 t(TPB, 1, 1);
  S.TKBegin(CudaToroidalState::TK_EFFECTIVE_CONSTRAINT);
  k_effective_constraint_force<<<b, t, 0, S.stream>>>(
      S.n_config_max, ns_con_local, nZnT, jMin,
      S.d_rCon, S.d_rCon0, S.d_zCon, S.d_zCon0,
      S.d_ruFull, S.d_zuFull, S.d_gConEff);
  cuda_check(cudaGetLastError(), "k_effective_constraint_force launch");
  S.TKEnd(CudaToroidalState::TK_EFFECTIVE_CONSTRAINT);
  DiagCfg01DiffCuda(S.d_gConEff, ns_con_local * nZnT, "eff:gConEff");

  // VMECPP_DUMP_GCON=1: print a serial checksum of gConEff (diagnostic
  // for CPU-vs-CUDA trajectory comparisons).
  static int dump_gcon_env = -1;
  if (dump_gcon_env < 0) {
    const char* e = std::getenv("VMECPP_DUMP_GCON");
    dump_gcon_env = (e && std::atoi(e) > 0) ? 1 : 0;
  }
  if (dump_gcon_env) {
    static int dumped = 0;
    if (!dumped) {
      dumped = 1;
      std::vector<double> h((size_t)ns_con_local * nZnT, 0.0);
      cuda_check(cudaMemcpyAsync(h.data(), S.d_gConEff,
                                 sizeof(double) * h.size(),
                                 cudaMemcpyDeviceToHost, S.stream),
                 "d2h gConEff dump");
      cuda_check(cudaStreamSynchronize(S.stream), "gConEff dump sync");
      double sum = 0.0;
      for (double v : h) sum += std::fabs(v);
      std::fprintf(stderr, "[GCONEFF] sum=%.17g v[1,0..3]=%.17g %.17g %.17g %.17g\n",
                   sum, h[nZnT], h[nZnT + 1], h[nZnT + 2], h[nZnT + 3]);
    }
  }

  // d_gConEff stays on device; DeAliasConstraintForceCuda reads it directly.
  (void)gConEff_out;
}

// ============================================================================
// Free-boundary bridges. The NESTOR vacuum solve stays on the host; these
// wrappers carry the per-iteration traffic between the device-resident
// iteration state and the host-side vacuum block in IdealMhdModel::update.
// ============================================================================

// Scales the device rCon0/zCon0 volume profiles in place (the gradual
// turn-off applied on every vacuum iteration).
__global__ void k_scale_rzcon0(int total, double factor,
                               double* __restrict__ rCon0,
                               double* __restrict__ zCon0) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  rCon0[i] *= factor;
  zCon0[i] *= factor;
}

void ScaleRZCon0Cuda(double factor) {
  auto& S = State();
  if (!S.stream || !S.d_rCon0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  const int total =
      S.n_config_max * S.rzcon0_ns_con_cached * S.rzcon0_nZnT_cached;
  if (total <= 0) return;
  const int TPB = 256;
  k_scale_rzcon0<<<(total + TPB - 1) / TPB, TPB, 0, S.stream>>>(
      total, factor, S.d_rCon0, S.d_zCon0);
  cuda_check(cudaGetLastError(), "k_scale_rzcon0 launch");
}

// One D2H flush per vacuum iteration and configuration: the axis row of
// r1_e/z1_e, the LCFS row of r1_e/r1_o, the outermost two totalPressure
// rows, the presH profile (the edge-pressure extrapolation reads its
// outermost entry on the host; without the flush that array is never
// written under CUDA), and the bucoH/bvcoH profiles consumed by the
// toroidal-current scalars. The host destinations are the
// single-configuration arrays: the batched vacuum loop flushes, hands
// over, and solves one configuration at a time. The single synchronize
// drains every queued copy.
void FlushVacuumHostDataCuda(int cfg, const RadialPartitioning& r,
                             const Sizes& s,
                             Eigen::VectorXd& m_r1_e,
                             Eigen::VectorXd& m_r1_o,
                             Eigen::VectorXd& m_z1_e,
                             Eigen::VectorXd& m_totalPressure,
                             Eigen::VectorXd& m_presH,
                             Eigen::VectorXd& m_bucoH,
                             Eigen::VectorXd& m_bvcoH) {
  auto& S = State();
  if (!S.stream || !S.d_r1_e) return;
  std::lock_guard<std::mutex> lk(S.mu);
  const int nZnT = s.nZnT;
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int ns_h = r.nsMaxH - r.nsMinH;
  const size_t row_bytes = sizeof(double) * (size_t)nZnT;
  const size_t cfg_full = (size_t)cfg * (size_t)ns_local * (size_t)nZnT;
  const size_t cfg_half = (size_t)cfg * (size_t)ns_h * (size_t)nZnT;
  const size_t cfg_prof = (size_t)cfg * (size_t)ns_h;
  if (r.nsMinF1 == 0) {
    cuda_check(cudaMemcpyAsync(m_r1_e.data(), S.d_r1_e + cfg_full, row_bytes,
                               cudaMemcpyDeviceToHost, S.stream),
               "d2h r1_e axis row");
    cuda_check(cudaMemcpyAsync(m_z1_e.data(), S.d_z1_e + cfg_full, row_bytes,
                               cudaMemcpyDeviceToHost, S.stream),
               "d2h z1_e axis row");
  }
  const size_t lcfs_off = (size_t)(ns_local - 1) * (size_t)nZnT;
  cuda_check(cudaMemcpyAsync(m_r1_e.data() + lcfs_off,
                             S.d_r1_e + cfg_full + lcfs_off,
                             row_bytes, cudaMemcpyDeviceToHost, S.stream),
             "d2h r1_e lcfs row");
  cuda_check(cudaMemcpyAsync(m_r1_o.data() + lcfs_off,
                             S.d_r1_o + cfg_full + lcfs_off,
                             row_bytes, cudaMemcpyDeviceToHost, S.stream),
             "d2h r1_o lcfs row");
  if (S.d_totalPressure && ns_h >= 2) {
    const size_t off = (size_t)(ns_h - 2) * (size_t)nZnT;
    cuda_check(cudaMemcpyAsync(m_totalPressure.data() + off,
                               S.d_totalPressure + cfg_half + off,
                               sizeof(double) * 2 * (size_t)nZnT,
                               cudaMemcpyDeviceToHost, S.stream),
               "d2h totalPressure edge rows");
  }
  if (S.d_presH && ns_h > 0 && m_presH.size() >= ns_h) {
    cuda_check(cudaMemcpyAsync(m_presH.data(), S.d_presH + cfg_prof,
                               sizeof(double) * (size_t)ns_h,
                               cudaMemcpyDeviceToHost, S.stream),
               "d2h presH");
  }
  if (S.d_bucoH && ns_h > 0 && m_bucoH.size() >= ns_h) {
    cuda_check(cudaMemcpyAsync(m_bucoH.data(), S.d_bucoH + cfg_prof,
                               sizeof(double) * (size_t)ns_h,
                               cudaMemcpyDeviceToHost, S.stream),
               "d2h bucoH");
  }
  if (S.d_bvcoH && ns_h > 0 && m_bvcoH.size() >= ns_h) {
    cuda_check(cudaMemcpyAsync(m_bvcoH.data(), S.d_bvcoH + cfg_prof,
                               sizeof(double) * (size_t)ns_h,
                               cudaMemcpyDeviceToHost, S.stream),
               "d2h bvcoH");
  }
  cuda_check(cudaStreamSynchronize(S.stream), "vacuum host-data sync");
}

// H2D stage of the host-computed rBSq profile for one configuration;
// AssembleTotalForcesCuda applies it to each configuration's LCFS force
// row while it is staged. The buffer carries one nZnT profile per
// configuration slot.
void StageRbsqCuda(int cfg, const Eigen::VectorXd& rBSq) {
  auto& S = State();
  if (!S.stream) return;
  std::lock_guard<std::mutex> lk(S.mu);
  const int per_cfg = (int)rBSq.size();
  const size_t bytes_all =
      sizeof(double) * (size_t)S.n_config_max * (size_t)per_cfg;
  if (S.d_rbsq && S.rbsq_size != per_cfg) {
    cudaFree(S.d_rbsq);
    S.d_rbsq = nullptr;
  }
  if (!S.d_rbsq) {
    cuda_check(cudaMalloc(&S.d_rbsq, bytes_all), "alloc d_rbsq");
    S.rbsq_size = per_cfg;
  }
  cuda_check(cudaMemcpyAsync(S.d_rbsq + (size_t)cfg * per_cfg, rBSq.data(),
                             sizeof(double) * (size_t)per_cfg,
                             cudaMemcpyHostToDevice, S.stream),
             "h2d rbsq");
  S.rbsq_staged = true;
}

// Applies the vacuum edge pressure to each configuration's LCFS force
// row:
//   armn_{e,o} += zuFull * rBSq,  azmn_{e,o} -= ruFull * rBSq.
// Batched execution: configuration axis on blockIdx.y; rbsq carries one
// nZnT profile per configuration.
__global__ void k_apply_rbsq_edge(int n_config, int nZnT, int row_off,
                                  int con_stride, int force_stride,
                                  const double* __restrict__ rbsq,
                                  const double* __restrict__ ruFull,
                                  const double* __restrict__ zuFull,
                                  double* __restrict__ armn_e,
                                  double* __restrict__ armn_o,
                                  double* __restrict__ azmn_e,
                                  double* __restrict__ azmn_o) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nZnT) return;
  const size_t idx_con =
      (size_t)config * (size_t)con_stride + (size_t)(row_off + i);
  const size_t idx_force =
      (size_t)config * (size_t)force_stride + (size_t)(row_off + i);
  const double bsq = rbsq[(size_t)config * (size_t)nZnT + (size_t)i];
  const double ar = zuFull[idx_con] * bsq;
  const double az = ruFull[idx_con] * bsq;
  armn_e[idx_force] += ar;
  armn_o[idx_force] += ar;
  azmn_e[idx_force] -= az;
  azmn_o[idx_force] -= az;
}

// ============================================================================
// AssembleTotalForcesCuda: CUDA port of IdealMhdModel::assembleTotalForces.
// Inputs (device): rCon, rCon0, zCon, zCon0 (from forward), ruFull, zuFull,
// sqrtSF (from forward staging). Per-call H2D: gCon (from CPU deAlias).
// In-place mutation of d_brmn_e/o, d_bzmn_e/o on device. Writes d_frcon_e/o,
// d_fzcon_e/o on device. D2H frcon/fzcon and updated brmn/bzmn so CPU forward-
// inverse-FFT path can read them (until inverse FFT real-port lands).
// ============================================================================
void AssembleTotalForcesCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    const Eigen::VectorXd& gCon,
    Eigen::VectorXd& brmn_e_out, Eigen::VectorXd& brmn_o_out,
    Eigen::VectorXd& bzmn_e_out, Eigen::VectorXd& bzmn_o_out,
    Eigen::VectorXd& frcon_e_out, Eigen::VectorXd& frcon_o_out,
    Eigen::VectorXd& fzcon_e_out, Eigen::VectorXd& fzcon_o_out,
    bool vacuum_edge) {
  auto& S = State();
  const int ns_force_local = r.nsMaxF - r.nsMinF;
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int nZnT = s.nZnT;
  if (ns_force_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureConstraintMultiplierBuffers(ns_force_local, ns_con_local, nZnT);
  // See EffectiveConstraintForceCuda: a free-boundary stage entered with
  // the vacuum pressure active has no rzConIntoVolume call.
  S.EnsureRzCon0Buffers(ns_con_local, nZnT);

  // gCon stays on device from DeAliasConstraintForceCuda (written into d_gCon
  // on the same stream); host gCon parameter is unused in the CUDA path.
  (void)gCon;

  int nsMinF_to_nsMinF1 = r.nsMinF - r.nsMinF1;
  // Free-boundary: apply the staged vacuum edge pressure to the LCFS
  // force row before the constraint assembly, mirroring the host edge
  // block at the top of assembleTotalForces.
  if (vacuum_edge && S.d_rbsq && S.rbsq_staged) {
    const int ETPB = 128;
    const int edge_row_off = (ns_force_local - 1) * nZnT;
    dim3 eb((nZnT + ETPB - 1) / ETPB, S.n_config_max, 1);
    k_apply_rbsq_edge<<<eb, ETPB, 0, S.stream>>>(
        S.n_config_max, nZnT, edge_row_off,
        ns_con_local * nZnT, ns_force_local * nZnT,
        S.d_rbsq, S.d_ruFull, S.d_zuFull,
        S.d_armn_e, S.d_armn_o, S.d_azmn_e, S.d_azmn_o);
    cuda_check(cudaGetLastError(), "k_apply_rbsq_edge launch");
  }
  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 64;
  dim3 b((nZnT + TPB - 1) / TPB, ns_force_local, S.n_config_max);
  dim3 t(TPB, 1, 1);
  S.TKBegin(CudaToroidalState::TK_ASSEMBLE_TOTAL);
  k_assemble_total_forces<<<b, t, 0, S.stream>>>(
      S.n_config_max, ns_con_local, ns_force_local, nZnT, nsMinF_to_nsMinF1,
      S.d_rCon, S.d_rCon0, S.d_zCon, S.d_zCon0, S.d_gCon,
      S.d_ruFull, S.d_zuFull, S.d_sqrtSF,
      S.d_brmn_e, S.d_brmn_o, S.d_bzmn_e, S.d_bzmn_o,
      S.d_frcon_e, S.d_frcon_o, S.d_fzcon_e, S.d_fzcon_o,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_assemble_total_forces launch");
  S.TKEnd(CudaToroidalState::TK_ASSEMBLE_TOTAL);
  DiagCfg01DiffCuda(S.d_brmn_e, ns_force_local * nZnT, "atot:brmn_e");
  DiagCfg01DiffCuda(S.d_frcon_e, ns_force_local * nZnT, "atot:frcon_e");

  // All outputs stay on device; the CUDA inverse FFT reads d_brmn_*, d_bzmn_*,
  // d_frcon_*, d_fzcon_* directly.
  (void)brmn_e_out; (void)brmn_o_out; (void)bzmn_e_out; (void)bzmn_o_out;
  (void)frcon_e_out; (void)frcon_o_out; (void)fzcon_e_out; (void)fzcon_o_out;
}

// ============================================================================
// ApplyLambdaPreconditionerCuda
// ============================================================================
void ApplyLambdaPreconditionerCuda(
    const RadialPartitioning& r, const Sizes& s,
    const Eigen::VectorXd& lambdaPreconditioner,
    double* flsc_host, double* flcs_host) {
  auto& S = State();
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  if (ns_con_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  S.EnsureLambdaInputBuffer(ns_con_local, mpol, ntor);
  size_t spec_bytes = sizeof(double) * ns_con_local * mpol * (ntor + 1);
  // Device-resident path: d_lambdaPreconditioner was just written on
  // device by k_ulp_assemble inside updateLambdaPreconditioner; it's identical
  // to the host m_p_.lambdaPreconditioner that the caller passes. Skip the
  // H2D round-trip and read from d_lambdaPreconditioner directly. The host
  // parameter `lambdaPreconditioner` is unused here.
  (void)lambdaPreconditioner;
  // flsc/flcs read/written directly on the DECOMPOSED shadow populated by
  // DecomposeAndConstrainCuda on the same stream; no H2D round-trip.
  // Batched execution: z-dim = config * ns_con_local + jF_local.
  const int TPB = 16;
  dim3 b((ntor + 1 + TPB - 1) / TPB, mpol, ns_con_local * S.n_config_max);
  dim3 t(TPB, 1, 1);
  S.TKBegin(CudaToroidalState::TK_APPLY_LAMBDA);
  k_apply_lambda_preconditioner<<<b, t, 0, st>>>(
      S.n_config_max, S.ns_local_cached, ns_con_local, mpol, ntor, s.lthreed,
      S.d_lambdaPreconditioner,
      S.d_decomposed_flsc, S.d_decomposed_flcs,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_apply_lambda launch");
  S.TKEnd(CudaToroidalState::TK_APPLY_LAMBDA);
  DiagCfg01DiffCuda(S.d_decomposed_flsc,
                    S.ns_local_cached * mpol * (ntor + 1), "lam:dec_flsc");
  // D2H + sync deferred to end-of-residue() FlushDecomposedToHostCuda.
  (void)flsc_host; (void)flcs_host; (void)spec_bytes;
}

// ============================================================================
// AssembleRZPreconditionerCuda
//
// The device-side port of IdealMhdModel::assembleRZPreconditioner. The
// routine consumes the per-side preconditioner-matrix snapshots
// populated by the two ComputePreconditioningMatrixCuda invocations
// performed during updateRadialPreconditioner -- d_pmat_arm,
// d_pmat_brm, d_pmat_ard, and d_pmat_brd for the R side;
// d_pmat_azm, d_pmat_bzm, d_pmat_azd, and d_pmat_bzd for the Z side;
// together with the shared d_pmat_cxd -- and writes the tri-diagonal
// coefficients ar, dr, br for R, and az, dz, bz for Z, directly into
// d_rz_aR, d_rz_dR, d_rz_bR, d_rz_aZ, d_rz_dZ, and d_rz_bZ in the
// (mn, jF_global) layout required by the parallel cyclic reduction
// solver invoked downstream by ApplyRZPreconditionerCuda. The
// per-(mn, jF_global) minimum-row index buffer d_rz_jMin is also
// populated here.
//
// The kernel-based assembly subsumes both the per-iteration CPU loop
// of the original assembleRZPreconditioner and the six host-to-device
// transposes plus host-to-device transfers that
// ApplyRZPreconditionerCuda performed on a cache miss against the
// host-side matrix in the prior arrangement, so neither the
// transposes nor the transfers occur on the present path.
// ============================================================================
void AssembleRZPreconditionerCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    int jMax) {
  auto& S = State();
  const int ns_force_local = r.nsMaxF - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  const int mnsize = mpol * (ntor + 1);
  if (ns_force_local <= 0 || mnsize <= 0) return;
  const int ns_total = fc.ns;
  const int num_basis = s.lthreed ? 2 : 1;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;

  // Ensure the destination buffers exist (idempotent if already allocated).
  S.EnsureRZBuffers(mnsize, ns_total, num_basis);

  int lcfs_owning = (r.nsMaxF == fc.ns) ? 1 : 0;

  // Edge pedestal + ZC_00(NS) stabilization constants (mirror CPU).
  constexpr double edge_pedestal = 0.05;
  constexpr double fac = 0.25;
  double mult_fact = (fac < fac * fc.deltaS * 15.0) ? fac : (fac * fc.deltaS * 15.0);

  // Launch one block per mn; threads cover jF in [0, ns_total).
  // Batched execution: z-dim covers n_config_max configs.
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int TPB = 32;
  dim3 b(mnsize, (ns_total + TPB - 1) / TPB, S.n_config_max);
  dim3 t(TPB, 1, 1);
  S.TKBegin(CudaToroidalState::TK_ASSEMBLE_RZ);
  k_assemble_rz_preconditioner<<<b, t, 0, st>>>(
      S.n_config_max, ns_h,
      mpol, ntor, s.nfp,
      ns_total, ns_force_local, r.nsMinF, r.nsMinH, r.nsMaxH,
      jMax, lcfs_owning,
      edge_pedestal, mult_fact,
      S.d_pmat_arm, S.d_pmat_brm,
      S.d_pmat_azm, S.d_pmat_bzm,
      S.d_pmat_ard, S.d_pmat_brd,
      S.d_pmat_azd, S.d_pmat_bzd,
      S.d_pmat_cxd,
      S.d_rz_aR, S.d_rz_dR, S.d_rz_bR,
      S.d_rz_aZ, S.d_rz_dZ, S.d_rz_bZ,
      S.d_rz_jMin,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_assemble_rz_preconditioner launch");
  S.TKEnd(CudaToroidalState::TK_ASSEMBLE_RZ);
}

// ============================================================================
// ApplyRZPreconditionerCuda
//
// The CPU implementation of applyRZPreconditioner invokes
// TridiagonalSolveSerial once per Fourier index pair (mn) over the
// num_basis right-hand-side spectra, namely frcc and fzsc, with frss
// and fzcs added under three-dimensional symmetry (lthreed) and the
// corresponding lasym variants added under non-stellarator symmetry.
// On the device the tri-diagonal matrix coefficients ar, dr, br for
// R and az, dz, bz for Z are produced directly in the device buffers
// d_rz_aR, d_rz_dR, d_rz_bR, d_rz_aZ, d_rz_dZ, and d_rz_bZ by
// AssembleRZPreconditionerCuda, and the parallel cyclic reduction
// solver consumes them in place. The host arguments ar, dr, br_in,
// az, dz, bz_in, jMin_arr, and jMin_size therefore become unused
// under CUDA and are retained only to preserve the call-site
// signature shared with the CPU-only path.
// ============================================================================
int ApplyRZPreconditionerCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    const Eigen::VectorXd& ar, const Eigen::VectorXd& dr,
    const Eigen::VectorXd& br_in, const Eigen::VectorXd& az,
    const Eigen::VectorXd& dz, const Eigen::VectorXd& bz_in,
    const int* jMin_arr, int jMin_size, int jMaxRZ,
    double* frcc_host, double* frss_host, double* fzsc_host, double* fzcs_host) {
  auto& S = State();
  const int ns_force_local = r.nsMaxF - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  const int mnsize = mpol * (ntor + 1);
  if (ns_force_local <= 0 || mnsize <= 0) return 0;
  const int ns_total = fc.ns;
  const int num_basis = s.lthreed ? 2 : 1;  // lasym not handled here
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;

  // By the time control reaches this point, the upstream
  // AssembleRZPreconditionerCuda has already dispatched
  // k_assemble_rz_preconditioner, leaving the six tri-diagonal
  // coefficient buffers d_rz_aR, d_rz_dR, d_rz_bR, d_rz_aZ, d_rz_dZ,
  // and d_rz_bZ populated together with the per-(mn, jF_global)
  // minimum-row index buffer d_rz_jMin, each in the layout consumed
  // by the parallel cyclic reduction solver launched below. The
  // host-side transpose loop, the six host-to-device transfers of
  // the matrix coefficients, the additional host-to-device transfer
  // of jMin, and the stream synchronisation required by the previous
  // host-side rollback path are therefore unnecessary on the present
  // path, and the host parameters ar, dr, br_in, az, dz, bz_in,
  // jMin_arr, and jMin_size are consumed as no-ops through the void
  // casts below.
  S.EnsureRZBuffers(mnsize, ns_total, num_basis);
  double *d_cR = S.d_rz_cR;
  double *d_cZ = S.d_rz_cZ;
  int   *d_jMin = S.d_rz_jMin;
  (void)ar; (void)dr; (void)br_in; (void)az; (void)dz; (void)bz_in;
  (void)jMin_arr; (void)jMin_size;

  // Device-side transpose decomposed shadow → cR/cZ.
  // Batched execution: z-dim covers n_config_max configs.
  {
    const int TPB = 32;
    dim3 b((ns_total + TPB - 1) / TPB, mnsize, S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_rz_transpose_in<<<b, t, 0, st>>>(
        S.n_config_max, S.ns_local_cached,
        ns_force_local, mpol, ntor, ns_total, num_basis, r.nsMinF, s.lthreed,
        S.d_decomposed_frcc, S.d_decomposed_frss,
        S.d_decomposed_fzsc, S.d_decomposed_fzcs,
        d_cR, d_cZ,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_rz_transpose_in launch");
  }

  // PCR: replaces single-thread-per-(mn) Thomas. Block size = next power-of-2
  // >= jMaxRZ (worst-case N when jMin=0). Shared memory holds the entire system
  // for one (mn) so PCR can iterate log2(N) parallel reduction passes.
  // Batched execution: y-dim covers n_config_max configs (each block solves one
  // (config, mn) tridiagonal).
  int pcr_threads = 32;
  while (pcr_threads < jMaxRZ) pcr_threads <<= 1;
  if (pcr_threads > 1024) pcr_threads = 1024;
  size_t pcr_smem = 5 * ns_total * sizeof(double);
  size_t pcr_smem_fp32 = 5 * ns_total * sizeof(float);
  dim3 pcr_grid(mnsize, S.n_config_max, 1);

  // Carson-Higham staged FP32 iterative refinement. When
  // VMECPP_RZ_IR_FP32=1 the path below replaces the single FP64 PCR
  // solve with: copy the FP64 RHS into c_orig, FP32 PCR pass on a fresh
  // copy of the RHS to obtain x0 in c_inout, save x0 into x_saved,
  // restore c_inout to the original RHS, FP64 residual computation
  // to write r = b - A*x0 into c_inout, FP32 PCR pass on r to obtain
  // dx in c_inout, and final correction kernel x = x_saved + dx
  // writing the refined FP64 solution back into c_inout. The FP32 PCR
  // uses half the shared memory of the FP64 path, improving occupancy
  // on Ada, and the IR step recovers the FP64 precision lost by the
  // FP32 solves.
  static int rz_ir_fp32_env = -1;
  if (rz_ir_fp32_env < 0) {
    const char* e = std::getenv("VMECPP_RZ_IR_FP32");
    rz_ir_fp32_env = (e && std::atoi(e) > 0) ? 1 : 0;
    if (rz_ir_fp32_env) {
      std::fprintf(stderr,
          "[fft_toroidal_cuda] VMECPP_RZ_IR_FP32=1: staged FP32 IR active "
          "for k_apply_rz_pcr\n");
    }
  }
  // VMECPP_CPU_ORDER_RZSOLVE=1: serial Thomas elimination in the host
  // order instead of parallel cyclic reduction (diagnostic; the two
  // algorithms round differently, so the preconditioned forces differ
  // between them at the solve's conditioning level).
  static int rz_serial_env = -1;
  if (rz_serial_env < 0) {
    const char* e = std::getenv("VMECPP_CPU_ORDER_RZSOLVE");
    rz_serial_env = (e && std::atoi(e) > 0) ? 1 : 0;
    if (rz_serial_env) {
      std::fprintf(stderr,
          "[fft_toroidal_cuda] VMECPP_CPU_ORDER_RZSOLVE=1: serial Thomas "
          "RZ solve active\n");
    }
  }
  // VMECPP_RZ_FORCE_BLOCK=1: take the block-Thomas path even for ns <= 1024.
  // Diagnostic only: lets the large-ns solver be exercised and compared
  // against the one-thread-per-row serial Thomas at small ns.
  static int rz_force_block_env = -1;
  if (rz_force_block_env < 0) {
    const char* e = std::getenv("VMECPP_RZ_FORCE_BLOCK");
    rz_force_block_env = (e && std::atoi(e) > 0) ? 1 : 0;
  }

  S.TKBegin(CudaToroidalState::TK_APPLY_RZ);

  if (jMaxRZ > 1024 || rz_force_block_env) {
    // The PCR solver needs one thread per radial point and so cannot exceed
    // 1024 (CUDA threads-per-block). For larger radial grids, solve each
    // (config, mn) tridiagonal with one block running serial Thomas, holding
    // the elimination ratios in dynamic shared memory sized to jMax. This is
    // sequential per row but parallel across the n_config * mnsize rows.
    const int TPB_BLK = 128;
    size_t blk_smem = (size_t)jMaxRZ * sizeof(double);
    static int blk_max_smem = -1;
    if (blk_max_smem < 0) {
      int dev = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&blk_max_smem,
                             cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    }
    if (blk_smem > (size_t)blk_max_smem) {
      cuda_check(cudaErrorInvalidValue,
                 "rz radial solve: ns exceeds the shared-memory capacity of "
                 "the large-ns block-Thomas solver");
    }
    if (blk_smem > (size_t)(48 * 1024)) {
      cudaFuncSetAttribute(k_apply_rz_thomas_block,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)blk_smem);
    }
    int total_rows = S.n_config_max * mnsize;
    dim3 blk_grid(total_rows, 1, 1);
    k_apply_rz_thomas_block<<<blk_grid, TPB_BLK, blk_smem, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aR, S.d_rz_dR, S.d_rz_bR, d_cR, S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_thomas_block R");
    k_apply_rz_thomas_block<<<blk_grid, TPB_BLK, blk_smem, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aZ, S.d_rz_dZ, S.d_rz_bZ, d_cZ, S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_thomas_block Z");
  } else if (rz_ir_fp32_env && S.d_rz_c_orig_R && S.d_rz_x_saved_R) {
    // Bytes per (R or Z) c buffer: matches the EnsureRZBuffers c_bytes.
    const size_t c_bytes = sizeof(double) * (size_t)S.n_config_max *
                            (size_t)mnsize * (size_t)num_basis *
                            (size_t)ns_total;
    // 1) Save the original RHS for both R and Z into the c_orig
    //    buffers. The downstream residual kernel needs the unmodified
    //    b = c_inout-before-solve.
    cuda_check(cudaMemcpyAsync(S.d_rz_c_orig_R, d_cR, c_bytes,
                                cudaMemcpyDeviceToDevice, st),
               "ir: copy d_cR -> c_orig_R");
    cuda_check(cudaMemcpyAsync(S.d_rz_c_orig_Z, d_cZ, c_bytes,
                                cudaMemcpyDeviceToDevice, st),
               "ir: copy d_cZ -> c_orig_Z");

    // 2) FP32 PCR on d_cR, d_cZ. After this call d_cR/d_cZ holds the
    //    FP32-approximate solution x0 (stored in FP64 by the writeback).
    k_apply_rz_pcr_fp32<<<pcr_grid, pcr_threads, pcr_smem_fp32, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aR, S.d_rz_dR, S.d_rz_bR, d_cR,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_pcr_fp32 R (stage 1)");
    k_apply_rz_pcr_fp32<<<pcr_grid, pcr_threads, pcr_smem_fp32, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aZ, S.d_rz_dZ, S.d_rz_bZ, d_cZ,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_pcr_fp32 Z (stage 1)");

    // 3) Save x0 into x_saved_R/Z, then restore d_cR/d_cZ to the
    //    original RHS so the residual kernel can compute r = b - A*x0
    //    with x0 read from x_saved.
    cuda_check(cudaMemcpyAsync(S.d_rz_x_saved_R, d_cR, c_bytes,
                                cudaMemcpyDeviceToDevice, st),
               "ir: copy d_cR (x0) -> x_saved_R");
    cuda_check(cudaMemcpyAsync(S.d_rz_x_saved_Z, d_cZ, c_bytes,
                                cudaMemcpyDeviceToDevice, st),
               "ir: copy d_cZ (x0) -> x_saved_Z");

    // 4) Compute the FP64 residual r = c_orig - A*x_saved, writing it
    //    to d_cR/d_cZ. The residual kernel reads x from x_saved (a
    //    separate buffer; it is the FP64-stored FP32 solution x0) and
    //    the original RHS from c_orig_R/Z. After this call d_cR/d_cZ
    //    holds r (FP64) and x_saved still holds x0.
    const int rt = std::min(pcr_threads, 1024);
    k_rz_compute_residual_fp64<<<pcr_grid, rt, 0, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aR, S.d_rz_dR, S.d_rz_bR,
        S.d_rz_c_orig_R, S.d_rz_x_saved_R, d_cR,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_rz_compute_residual R");
    k_rz_compute_residual_fp64<<<pcr_grid, rt, 0, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aZ, S.d_rz_dZ, S.d_rz_bZ,
        S.d_rz_c_orig_Z, S.d_rz_x_saved_Z, d_cZ,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_rz_compute_residual Z");

    // 5) FP32 PCR on r to obtain the correction dx. After this call
    //    d_cR/d_cZ holds dx (stored in FP64 by the writeback).
    k_apply_rz_pcr_fp32<<<pcr_grid, pcr_threads, pcr_smem_fp32, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aR, S.d_rz_dR, S.d_rz_bR, d_cR,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_pcr_fp32 R (stage 2 correction)");
    k_apply_rz_pcr_fp32<<<pcr_grid, pcr_threads, pcr_smem_fp32, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aZ, S.d_rz_dZ, S.d_rz_bZ, d_cZ,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_pcr_fp32 Z (stage 2 correction)");

    // 6) Final FP64 correction x = x_saved + dx, writing the refined
    //    solution back to d_cR/d_cZ.
    k_rz_add_correction<<<pcr_grid, rt, 0, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_x_saved_R, d_cR, d_cR,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_rz_add_correction R");
    k_rz_add_correction<<<pcr_grid, rt, 0, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_x_saved_Z, d_cZ, d_cZ,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_rz_add_correction Z");
  } else if (rz_serial_env) {
    // VMECPP_CPU_ORDER_RZSOLVE=1: serial Thomas elimination in the host
    // order, one thread per mode row (diagnostic for trajectory
    // comparisons against the CPU build).
    const int TPB_TH = 64;
    int total_rows = S.n_config_max * mnsize;
    dim3 thb((total_rows + TPB_TH - 1) / TPB_TH, 1, 1);
    dim3 tht(TPB_TH, 1, 1);
    k_apply_rz_thomas_serial<<<thb, tht, 0, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aR, S.d_rz_dR, S.d_rz_bR, d_cR,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_thomas_serial R");
    k_apply_rz_thomas_serial<<<thb, tht, 0, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aZ, S.d_rz_dZ, S.d_rz_bZ, d_cZ,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_thomas_serial Z");
  } else {
    // Default FP64 PCR path.
    k_apply_rz_pcr<<<pcr_grid, pcr_threads, pcr_smem, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aR, S.d_rz_dR, S.d_rz_bR, d_cR,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_pcr R");
    k_apply_rz_pcr<<<pcr_grid, pcr_threads, pcr_smem, st>>>(
        S.n_config_max, mnsize, ns_total, num_basis, d_jMin, jMaxRZ,
        S.d_rz_aZ, S.d_rz_dZ, S.d_rz_bZ, d_cZ,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_pcr Z");
  }
  S.TKEnd(CudaToroidalState::TK_APPLY_RZ);

  // Device-side transpose cR/cZ → decomposed shadow.
  // Batched execution: z-dim covers n_config_max configs.
  {
    const int TPB = 32;
    dim3 b((ns_force_local + TPB - 1) / TPB, mnsize, S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_rz_transpose_out<<<b, t, 0, st>>>(
        S.n_config_max, S.ns_local_cached,
        ns_force_local, mpol, ntor, ns_total, num_basis, r.nsMinF, s.lthreed,
        d_cR, d_cZ,
        S.d_decomposed_frcc, S.d_decomposed_frss,
        S.d_decomposed_fzsc, S.d_decomposed_fzcs,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_rz_transpose_out launch");
  }
  DiagCfg01DiffCuda(S.d_decomposed_frcc,
                    S.ns_local_cached * mpol * (ntor + 1), "rzapp:dec_frcc");

  // D2H + sync deferred to end-of-residue() FlushDecomposedToHostCuda.
  // RZ mutates the [nsMinF, nsMinF+ns_force_local) rows of the shadow
  // S.d_decomposed_frcc/frss/fzsc/fzcs; the flush at residue() exit picks
  // them up. Stream ordering keeps the subsequent ResidualsCuda kernel on
  // the same stream consistent without an explicit sync here.
  (void)frcc_host; (void)frss_host; (void)fzsc_host; (void)fzcs_host;
  // Buffers are persistent in CudaToroidalState; do NOT free here.
  return 0;
}

// k_dealias_inv_packed: warp-packed variant of k_dealias_inv (kernels.cu).
// The default kernel maps l in [0, nThetaReduced) onto a 32-lane warp, leaving
// 32 - nThetaReduced lanes idle every cycle (nThetaReduced = 16 on W7-X, so
// half the warp is wasted and the FP64 pipe is correspondingly half-fed).
// This variant packs ks_per_warp = blockDim.x / nThetaReduced zeta planes into
// one warp so all 32 lanes do useful FP64 work. blockIdx.y indexes the zeta
// GROUP; the per-lane zeta is blockIdx.y * ks_per_warp + threadIdx.x /
// nThetaReduced. The per-(jF, k, l) arithmetic is identical to k_dealias_inv.
// Gated by VMECPP_DEALIAS_PACK; the dispatcher only selects it when 32 is an
// exact multiple of nThetaReduced (no lane straddles two zeta planes).
__global__ void k_dealias_inv_packed(
    int n_config, int ns_force_local, int ns_con_local,
    int mpol, int ntor, int nZeta, int nThetaReduced,
    int nThetaEff, int nnyq2_plus_1,
    const double* __restrict__ gsc, const double* __restrict__ gcs,
    const double* __restrict__ sinmu, const double* __restrict__ cosmu,
    const double* __restrict__ cosnv, const double* __restrict__ sinnv,
    const double* __restrict__ faccon,
    double* __restrict__ m_gCon) {
  int ks_per_warp = blockDim.x / nThetaReduced;
  if (ks_per_warp < 1) ks_per_warp = 1;
  int config = blockIdx.z / ns_force_local;
  int jF = blockIdx.z - config * ns_force_local;
  if (config >= n_config) return;
  int kk = threadIdx.x / nThetaReduced;
  int l = threadIdx.x - kk * nThetaReduced;
  int k = blockIdx.y * ks_per_warp + kk;
  if (jF >= ns_force_local || kk >= ks_per_warp || k >= nZeta ||
      l >= nThetaReduced)
    return;
  size_t cfg_spec = (size_t)config * (size_t)ns_force_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t cfg_grid = (size_t)config * (size_t)ns_con_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  double acc = 0.0;
  #pragma unroll
  for (int m = 1; m < mpol - 1; ++m) {
    double fac = faccon[m];
    if (fac == 0.0) continue;
    double w0 = 0.0, w1 = 0.0;
    size_t idx_base = cfg_spec + (size_t)((jF * mpol + m) * (ntor + 1));
    #pragma unroll
    for (int n = 0; n <= ntor; ++n) {
      int kn = k * nnyq2_plus_1 + n;
      w0 += gsc[idx_base + n] * cosnv[kn];
      w1 += gcs[idx_base + n] * sinnv[kn];
    }
    int bml = m * nThetaReduced + l;
    acc += fac * (w0 * sinmu[bml] + w1 * cosmu[bml]);
  }
  size_t dst = cfg_grid + (size_t)((jF * nZeta + k) * nThetaEff + l);
  m_gCon[dst] = acc;
}

// ============================================================================
// DeAliasConstraintForceCuda
// gConEff is on device (from EffectiveConstraintForceCuda); tcon is on device
// (from ConstraintForceMultiplierCuda). faccon, cosnv, sinnv are staged per call.
// Writes the d_gCon device buffer, which AssembleTotalForcesCuda consumes in
// place on the same stream; the host m_gCon argument is unused under CUDA.
// ============================================================================
void DeAliasConstraintForceCuda(
    const RadialPartitioning& r, const FourierBasisFastPoloidal& fb,
    const Sizes& s, const Eigen::VectorXd& faccon,
    Eigen::VectorXd& m_gCon_host) {
  auto& S = State();
  const int ns_force_local = r.nsMaxF - r.nsMinF;
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  const int nZeta = s.nZeta;
  const int nThetaReduced = s.nThetaReduced;
  const int nThetaEff = s.nThetaEff;
  const int nnyq2_plus_1 = s.nnyq2 + 1;
  if (ns_force_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;

  // The intermediate buffers consumed by the constraint-force
  // dealiasing pipeline -- gsc, gcs, and the faccon weight vector --
  // are allocated once per Reshape through EnsureDealiasBuffers and
  // retained for the remainder of the run. The toroidal basis tables
  // cosnv and sinnv are likewise staged once at Reshape time through
  // StageToroidalBasis, removing them from the per-iteration H2D
  // path; the dealias kernels below index the staged tables through
  // the geometric dimensions nZeta and nnyq2_plus_1 directly.
  S.EnsureDealiasBuffers(mpol, ntor, ns_force_local);

  // The dealiasing factor faccon, defined as the vector of values
  //   -0.25 * signOfJacobian / xmpq[m]^2
  // for poloidal mode m, is initialised in the IdealMhdModel constructor
  // and treated as immutable for the remainder of the run; it depends
  // solely on the boundary's poloidal mode multipliers and is therefore
  // invariant across configurations under the batched layout. The
  // host-to-device transfer is consequently issued at most once per
  // Reshape, with subsequent invocations short-circuited through the
  // dealias_faccon_staged flag. The kernel reads from the same device
  // buffer for every configuration without a per-cfg offset.
  if (!S.dealias_faccon_staged) {
    cuda_check(cudaMemcpyAsync(S.d_dealias_faccon, faccon.data(),
                                sizeof(double) * mpol,
                                cudaMemcpyHostToDevice, st), "h2d faccon (one-shot)");
    S.dealias_faccon_staged = true;
  }

  // Stage 1: forward poloidal+toroidal → gsc/gcs.
  // Batched execution: z-dim = config * ns_force_local + jF.
  // TPB=16 keeps lane util at 11/16=69pct (ntor+1=11 active threads). TPB=32
  // was tested and regressed -0.8pct throughput at N=64 (120.0s vs 119.1s
  // baseline) -- the 21 idle threads per warp outweigh the doubled warp
  // residency. Stay at TPB=16.
  {
    const int TPB = 16;
    dim3 b((ntor + 1 + TPB - 1) / TPB, mpol,
           ns_force_local * S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_dealias_fwd<<<b, t, 0, st>>>(
        S.n_config_max, ns_force_local, ns_con_local,
        mpol, ntor, nZeta, nThetaReduced, nThetaEff, nnyq2_plus_1,
        S.d_gConEff, S.d_tcon, S.d_sinmui, S.d_cosmui,
        S.d_dealias_cosnv, S.d_dealias_sinnv,
        S.d_dealias_gsc, S.d_dealias_gcs);
    cuda_check(cudaGetLastError(), "k_dealias_fwd launch");
  }
  DiagCfg01DiffCuda(S.d_dealias_gsc,
                    ns_force_local * mpol * (ntor + 1), "dealias_fwd:gsc");
  DiagCfg01DiffCuda(S.d_dealias_gcs,
                    ns_force_local * mpol * (ntor + 1), "dealias_fwd:gcs");

  // Stage 2: inverse poloidal+toroidal → m_gCon.
  // Batched execution: k_dealias_inv uses `m_gCon[dst] = acc` (not +=) and
  // covers every (cfg, jF, k, l) cell at lasym=false / nThetaEff==nThetaReduced.
  // Pre-zero memset is therefore redundant. (Re-enable if lasym support is
  // added with nThetaEff > nThetaReduced.)
  {
    const int TPB = 32;
    dim3 b((nThetaReduced + TPB - 1) / TPB, nZeta,
           ns_force_local * S.n_config_max);
    dim3 t(TPB, 1, 1);
    S.TKBegin(CudaToroidalState::TK_DEALIAS);
    // VMECPP_DEALIAS_SPLIT routes to k_dealias_inv_tpl_split: same template
    // but with 4 partial accumulators per (m) inner n-loop. Hypothesis was
    // that breaking the 11-deep FP dep chain would 2-4x FMA throughput.
    // Measured: -2.1pct throughput at N=64 warm over five evaluations
    // (118.5 s -> 121.1 s).
    // Compiler already extracts ILP from the simple += chain. Default OFF;
    // set =1 to opt in for further experimentation.
    static const int dealias_split_env = []() {
      const char* e = std::getenv("VMECPP_DEALIAS_SPLIT");
      return (e && std::atoi(e) > 0) ? 1 : 0;
    }();
    // VMECPP_DEALIAS_MIXED routes to k_dealias_inv_tpl_mixed: FP32 inner
    // mults, FP64 accumulator. Default OFF; carries the same convergence
    // risk as the FP32-cuFFT path. Set =1 to opt in and measure.
    static const int dealias_mixed_env = []() {
      const char* e = std::getenv("VMECPP_DEALIAS_MIXED");
      return (e && std::atoi(e) > 0) ? 1 : 0;
    }();
    if (mpol == 10 && ntor == 10) {
      if (dealias_mixed_env) {
        k_dealias_inv_tpl_mixed<10, 10><<<b, t, 0, st>>>(
            S.n_config_max, ns_force_local, ns_con_local,
            nZeta, nThetaReduced, nThetaEff, nnyq2_plus_1,
            S.d_dealias_gsc, S.d_dealias_gcs, S.d_sinmu, S.d_cosmu,
            S.d_dealias_cosnv, S.d_dealias_sinnv, S.d_dealias_faccon,
            S.d_gCon);
        cuda_check(cudaGetLastError(), "k_dealias_inv_tpl_mixed<10,10> launch");
      } else if (dealias_split_env) {
        k_dealias_inv_tpl_split<10, 10><<<b, t, 0, st>>>(
            S.n_config_max, ns_force_local, ns_con_local,
            nZeta, nThetaReduced, nThetaEff, nnyq2_plus_1,
            S.d_dealias_gsc, S.d_dealias_gcs, S.d_sinmu, S.d_cosmu,
            S.d_dealias_cosnv, S.d_dealias_sinnv, S.d_dealias_faccon,
            S.d_gCon);
        cuda_check(cudaGetLastError(), "k_dealias_inv_tpl_split<10,10> launch");
      } else {
        k_dealias_inv_tpl<10, 10><<<b, t, 0, st>>>(
            S.n_config_max, ns_force_local, ns_con_local,
            nZeta, nThetaReduced, nThetaEff, nnyq2_plus_1,
            S.d_dealias_gsc, S.d_dealias_gcs, S.d_sinmu, S.d_cosmu,
            S.d_dealias_cosnv, S.d_dealias_sinnv, S.d_dealias_faccon,
            S.d_gCon,
            S.d_active_per_cfg);
        cuda_check(cudaGetLastError(), "k_dealias_inv_tpl<10,10> launch");
      }
    } else {
      static const int dealias_pack_env = []() {
        const char* e = std::getenv("VMECPP_DEALIAS_PACK");
        int v = (e && std::atoi(e) == 0) ? 0 : 1;  // default ON
        if (!v)
          std::fprintf(stderr, "[fft_toroidal_cuda] dealias warp-pack disabled "
                               "(VMECPP_DEALIAS_PACK=0)\n");
        return v;
      }();
      // Warp-pack only when more than one zeta plane fits in a 32-lane warp
      // and divides it cleanly (no lane straddles two planes); else default.
      const int ks_per_warp = (nThetaReduced > 0) ? (32 / nThetaReduced) : 0;
      if (dealias_pack_env && ks_per_warp >= 2 && (32 % nThetaReduced) == 0) {
        dim3 bp(1, (nZeta + ks_per_warp - 1) / ks_per_warp,
                ns_force_local * S.n_config_max);
        dim3 tp(32, 1, 1);
        k_dealias_inv_packed<<<bp, tp, 0, st>>>(
            S.n_config_max, ns_force_local, ns_con_local,
            mpol, ntor, nZeta, nThetaReduced, nThetaEff, nnyq2_plus_1,
            S.d_dealias_gsc, S.d_dealias_gcs, S.d_sinmu, S.d_cosmu,
            S.d_dealias_cosnv, S.d_dealias_sinnv, S.d_dealias_faccon,
            S.d_gCon);
        cuda_check(cudaGetLastError(), "k_dealias_inv_packed launch");
      } else {
        k_dealias_inv<<<b, t, 0, st>>>(
            S.n_config_max, ns_force_local, ns_con_local,
            mpol, ntor, nZeta, nThetaReduced, nThetaEff, nnyq2_plus_1,
            S.d_dealias_gsc, S.d_dealias_gcs, S.d_sinmu, S.d_cosmu,
            S.d_dealias_cosnv, S.d_dealias_sinnv, S.d_dealias_faccon,
            S.d_gCon);
        cuda_check(cudaGetLastError(), "k_dealias_inv launch");
      }
    }
    S.TKEnd(CudaToroidalState::TK_DEALIAS);
  }
  DiagCfg01DiffCuda(S.d_gCon, ns_con_local * nZeta * nThetaEff, "dealias:gCon");

  // VMECPP_DUMP_GCON=1: print a serial checksum of the dealiased gCon
  // (diagnostic for CPU-vs-CUDA trajectory comparisons).
  static int dump_gcon_env = -1;
  if (dump_gcon_env < 0) {
    const char* e = std::getenv("VMECPP_DUMP_GCON");
    dump_gcon_env = (e && std::atoi(e) > 0) ? 1 : 0;
  }
  if (dump_gcon_env) {
    static int dumped = 0;
    if (!dumped) {
      dumped = 1;
      const int nZnT = nZeta * nThetaEff;
      std::vector<double> h((size_t)ns_con_local * nZnT, 0.0);
      cuda_check(cudaMemcpyAsync(h.data(), S.d_gCon,
                                 sizeof(double) * h.size(),
                                 cudaMemcpyDeviceToHost, st),
                 "d2h gCon dump");
      cuda_check(cudaStreamSynchronize(st), "gCon dump sync");
      for (int j = 0; j < ns_force_local; ++j) {
        double rs = 0.0;
        for (int i = 0; i < nZnT; ++i) rs += std::fabs(h[(size_t)j * nZnT + i]);
        std::fprintf(stderr, "[GCONROW] j=%d %.17g\n", j, rs);
      }
    }
  }

  // The dealiased constraint-force buffer d_gCon remains resident on
  // device and is consumed in place by the downstream
  // AssembleTotalForcesCuda. Both wrappers issue their kernels on
  // S.stream, and stream ordering on a CUDA stream guarantees that
  // the consumer kernel observes the producer's writes without an
  // explicit synchronisation. The host pointer m_gCon_host is
  // therefore unused under CUDA and is consumed as a no-op through
  // the void cast.
  (void)m_gCon_host;
}

}  // namespace vmecpp

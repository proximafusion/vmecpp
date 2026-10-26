#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"

namespace vmecpp {

// ComputeJacobianCuda: computeJacobian on the device, with bad_jacobian from
// the device min and max of tau.
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

  // sqrtSH is staged once per Reshape.
  if (!S.sqrtSH_staged) {
    cuda_check(cudaMemcpyAsync(S.d_sqrtSH, sqrtSH.data(),
                                sizeof(double) * ns_h,
                                cudaMemcpyHostToDevice, S.stream), "h2d sqrtSH");
    S.sqrtSH_staged = true;
  }

  // The configuration axis is the third grid dimension, so one configuration
  // launches (nZnT / TPB, ns_h, 1).
  const bool use_dvdsh_fuse = wInt != nullptr && signOfJacobian != 0;

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
    const int TPB = 64;
    if (ns_h % 2 == 0) {
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
    S.jac_metric_fused_this_iter = true;
    S.dvdsh_fused_this_iter = true;
  } else {
    const int TPB = 64;
    dim3 blocks((nZnT + TPB - 1) / TPB, ns_h, S.n_config_max);
    dim3 tpb(TPB, 1, 1);
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
  }

  // Device-side tau min/max -> 2 scalars per config (replaces tau D2H + host scan).
  // Batched execution: launch n_config_max_max blocks, each writes out[cfg*2+0..1].
  S.EnsureJacMinmaxBuffer();
  k_tau_minmax<<<S.n_config_max, 256, 0, S.stream>>>(
      S.n_config_max, ns_h * nZnT, S.d_tau, S.d_jac_minmax,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_tau_minmax launch");

  // Only the min and max of tau come to the host, for every configuration into
  // the cache, configuration zero's as the scalars.
  int n_cfg = S.n_config_max;
  if ((int)g_jac_minmax_cache.size() != 2 * n_cfg) {
    g_jac_minmax_cache.assign(2 * n_cfg, 0.0);
  }
  (void)r12; (void)ru12; (void)zu12; (void)rs; (void)zs; (void)tau;
  cuda_check(cudaMemcpyAsync(g_jac_minmax_cache.data(), S.d_jac_minmax,
                              (size_t)2 * n_cfg * sizeof(double),
                              cudaMemcpyDeviceToHost, S.stream),
             "d2h jac minmax (per-cfg cache)");
  cuda_check(cudaStreamSynchronize(S.stream), "jac stream sync");

  double minTau = g_jac_minmax_cache[0], maxTau = g_jac_minmax_cache[1];
  bad_jacobian = (minTau * maxTau < 0.0) || !std::isfinite(minTau * maxTau);
}

std::vector<std::uint8_t> BadJacobianPerCfgCuda(int n_cfg) {
  std::vector<std::uint8_t> bad;
  if (static_cast<int>(g_jac_minmax_cache.size()) != 2 * n_cfg) {
    return bad;
  }
  bad.resize(n_cfg);
  for (int c = 0; c < n_cfg; ++c) {
    const double prod = g_jac_minmax_cache[2 * c] * g_jac_minmax_cache[2 * c + 1];
    bad[c] = (prod < 0.0 || !std::isfinite(prod)) ? 1 : 0;
  }
  return bad;
}

// ComputeMetricElementsCuda: computeMetricElements on the device, gsqrt, guu,
// guv and gvv staying in device buffers.
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

  // sqrtSH was staged by ComputeJacobianCuda.
  if (!S.sqrtSH_staged) {
    cuda_check(cudaMemcpyAsync(S.d_sqrtSH, sqrtSH.data(),
                                sizeof(double) * ns_h,
                                cudaMemcpyHostToDevice, S.stream), "h2d sqrtSH");
    S.sqrtSH_staged = true;
  }

  // The fused Jacobian kernel produced the metric this iteration, so the launch
  // is skipped and the flag cleared.
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

  // gsqrt, guu, guv and gvv stay on the device for the wrappers downstream.
  (void)gsqrt; (void)guu; (void)guv; (void)gvv;
}

// UpdateDifferentialVolumeCuda: dVdsH = signOfJacobian * sum of gsqrt * wInt
// per half-grid surface on the device.
void UpdateDifferentialVolumeCuda(
    const RadialPartitioning& r, const Sizes& s, double signOfJacobian,
    const Eigen::VectorXd& wInt, Eigen::VectorXd& dVdsH) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int nZnT = s.nZnT;
  const int nThetaEff = s.nThetaEff;
  if (ns_h <= 0) return;

  std::lock_guard<std::mutex> lk(S.mu);
  // The fused Jacobian kernel produced dVdsH this iteration, so the launch is
  // skipped and the flag consumed.
  if (S.dvdsh_fused_this_iter) {
    S.dvdsh_fused_this_iter = false;
    (void)wInt; (void)dVdsH; (void)signOfJacobian;
    return;
  }
  S.EnsureWIntStaged(nThetaEff, wInt.data());
  S.EnsureDVdsHBuffer(ns_h);

  const int TPB = 32;
  // The z dimension covers the configurations; one configuration launches
  // (ns_h, 1, 1).
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

// ComputeBCoCuda: computeBCo on the device from the device bsupu and bsupv.
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

// RadialForceBalanceCuda: bucoH, bvcoH and the interior jcurvF, jcuruF,
// presgradF, dVdsF and equiF on the device.
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
  k_radial_interior<<<blocks2, tpb2, 0, S.stream>>>(
      S.n_config_max, ns_h, ns_local,
      nsi, offset_jH, offset_jF, signByDeltaS, invDeltaS,
      S.d_bucoH, S.d_bvcoH, S.d_presH, S.d_dVdsH, S.d_chipF, S.d_phipF,
      S.d_jcurvF, S.d_jcuruF, S.d_presgradF, S.d_dVdsF, S.d_equiF);
  cuda_check(cudaGetLastError(), "k_radial_interior launch");

  // bvcoH, and at free boundary bucoH, go to the host asynchronously for the
  // toroidal-current scalars; the interior arrays stay on the device.
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

// RzConIntoVolumeCuda: rCon0 and zCon0 from rCon and zCon at the LCFS weighted
// by sFull = sqrtSF^2, on the rank owning the LCFS.
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

void InvalidateStagedCurrHCuda() {
  auto& S = State();
  std::lock_guard<std::mutex> lk(S.mu);
  S.currH_staged = false;
}

// ComputeBContraCuda: computeBContra on the device; lu and lv are scaled by
// lamscale in place, and bsupu and bsupv stay on the device.
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

  // phipF, phipH and currH are fixed at a multigrid level and staged once;
  // iotaH changes each iteration under ncurr = 1.
  if (!S.phipF_staged) {
    // Fills every configuration slot with the radial profile.
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
    // iotaH is seeded once per Reshape; the device updates it afterwards.
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

  // Stage 1: scales lu and lv by lamscale and adds phipF to lu_e on [nsMinH,
  // nsMaxH + 1).
  int jF_first = r.nsMinH - r.nsMinF1;
  int jF_last_excl = r.nsMaxH + 1 - r.nsMinF1;
  // phipF[jF_local - (nsMinH - nsMinF1)] is the CPU's phipF[jF - nsMinH].
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
    k_bcontra_bsupuv<<<blocks, tpb, 0, S.stream>>>(
        S.n_config_max, ns_local, ns_h, jF_in_offset_bcontra, nZnT, s.lthreed,
        S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o, S.d_sqrtSH, S.d_gsqrt,
        S.d_bsupu, S.d_bsupv);
    cuda_check(cudaGetLastError(), "k_bcontra_bsupuv launch");
  }

  // Stage 3 (ncurr==1 only): jvPlasma + avg_guu_gsqrt reductions.
  if (ncurr == 1) {
    const int TPB = 32;
    dim3 blocks(ns_h, 1, S.n_config_max);
    dim3 tpb(TPB, 1, 1);
    k_bcontra_jvplasma_reduce<<<blocks, tpb, 0, S.stream>>>(
        S.n_config_max, ns_h, nZnT, nThetaEff, s.lthreed,
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

  // bsupu and bsupv stay on the device; the chip and iota profiles reach the
  // host only in the output flush.
  (void)bsupu; (void)bsupv;
  (void)chipH_out; (void)iotaH_out; (void)chipF_out; (void)iotaF_out;
}

// k_copy_active_cfg: copies the per-configuration slices of length per_cfg from
// src to dst for the active configurations.
__global__ void k_copy_active_cfg(int n_config, size_t per_cfg,
                                  const double* __restrict__ src,
                                  double* __restrict__ dst,
                                  const std::uint8_t* __restrict__ d_active_per_cfg) {
  const int config = blockIdx.y;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (config >= n_config || i >= per_cfg) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  const size_t off = (size_t)config * per_cfg + i;
  dst[off] = src[off];
}

// ComputePreconditioningMatrixCuda: computePreconditioningMatrix on the
// device, once for the R side and once for the Z side.
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

  // xs, xu12, xu_e, xu_o and x1_o are read from the device buffers of the
  // Jacobian and forward FFT: Z derivatives for side 0, R for side 1.
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
    const int TPB = 32;
    dim3 b(ns_h, 1, S.n_config_max); dim3 t(TPB, 1, 1);
    // Read xs/xu12/xu_e/xu_o/x1_o from the device buffers directly, not the
    // d_pm_* H2D mirrors (which are now stale/unused on the H2D-skipped path).
    k_pm_half_reductions<<<b, t, 0, S.stream>>>(
        S.n_config_max, ns_local, ns_h, nZnT, nThetaEff, pFactor, deltaS,
        r.nsMinH, r.nsMinF1,
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

  // Snapshots the scratch into the per-side buffers, which the second call
  // would overwrite; a configuration not due keeps its latest matrices.
  double* dst_axm = (side == 0) ? S.d_pmat_arm : S.d_pmat_azm;
  double* dst_axd = (side == 0) ? S.d_pmat_ard : S.d_pmat_azd;
  double* dst_bxm = (side == 0) ? S.d_pmat_brm : S.d_pmat_bzm;
  double* dst_bxd = (side == 0) ? S.d_pmat_brd : S.d_pmat_bzd;
  const auto snapshot = [&](double* dst, const double* src, size_t per_cfg,
                            const char* what) {
    if (S.n_config_max > 1 && S.d_active_per_cfg) {
      const int TPB = 256;
      dim3 grid((unsigned)((per_cfg + TPB - 1) / TPB), S.n_config_max, 1);
      k_copy_active_cfg<<<grid, TPB, 0, S.stream>>>(
          S.n_config_max, per_cfg, src, dst, S.d_active_per_cfg);
      cuda_check(cudaGetLastError(), what);
    } else {
      cuda_check(cudaMemcpyAsync(dst, src,
                                  sizeof(double) * (size_t)S.n_config_max * per_cfg,
                                  cudaMemcpyDeviceToDevice, S.stream), what);
    }
  };
  snapshot(dst_axm, S.d_pm_axm, (size_t)ns_h * 2, "d2d pmat axm");
  snapshot(dst_axd, S.d_pm_axd, (size_t)ns_force_local * 2, "d2d pmat axd");
  snapshot(dst_bxm, S.d_pm_bxm, (size_t)ns_h * 2, "d2d pmat bxm");
  snapshot(dst_bxd, S.d_pm_bxd, (size_t)ns_force_local * 2, "d2d pmat bxd");
  snapshot(S.d_pmat_cxd, S.d_pm_cxd, (size_t)ns_force_local, "d2d pmat cxd");

  // The matrices are also copied to the host arrays.
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
  // The host reads of these transfers follow the synchronizations in
  // UpdateVolumeCuda and ComputeForceNormsCuda.
}

// Copies each active configuration's half-grid bLambda into the m = 0 lambda
// preconditioner's profile, zero at the axis and beyond.
__global__ void k_lambda_b_half_snapshot(int n_config, int ns_h,
                                         int lambda_stride,
                                         const double* __restrict__ bLambda,
                                         double* __restrict__ b_half,
                                         const std::uint8_t* __restrict__ d_active_per_cfg) {
  const int config = blockIdx.y;
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (config >= n_config || k >= lambda_stride) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  const size_t i = (size_t)config * (size_t)lambda_stride + (size_t)k;
  b_half[i] = (k >= 1 && k <= ns_h) ? bLambda[i] : 0.0;
}

// UpdateLambdaPreconditionerCuda: half-grid reductions, axis extrapolation,
// full-grid average and the per-(jF, n, m) assembly.
void UpdateLambdaPreconditionerCuda(
    const RadialPartitioning& r, const Sizes& s,
    double lambdaPreconditionerScale, double lamscale,
    double checkerboard_terms, double* bLambda_out, double* dLambda_out,
    double* cLambda_out, double* lambdaPreconditioner_host) {
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
  // The half-grid <g_uu / sqrt g> of the tridiagonal m = 0 block, taken before
  // the axis extrapolation and the full-grid average overwrite bLambda.
  if (checkerboard_terms >= 0.0) {
    S.EnsureLambdaM0Buffers(lambda_stride, r.nsMaxFIncludingLcfs, ntor);
    const int TPB = 64;
    dim3 b((lambda_stride + TPB - 1) / TPB, S.n_config_max, 1);
    k_lambda_b_half_snapshot<<<b, TPB, 0, S.stream>>>(
        S.n_config_max, ns_h, lambda_stride, S.d_bLambda, S.d_lambda_b_half,
        S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_lambda_b_half_snapshot launch");
  }
  // Stage 2: axis extrapolation - one block per config.
  int axis_present = (r.nsMinF == 0) ? 1 : 0;
  k_ulp_axis_extrap<<<S.n_config_max, 1, 0, S.stream>>>(
      S.n_config_max, lambda_stride, axis_present,
      S.d_bLambda, S.d_dLambda, S.d_cLambda);
  cuda_check(cudaGetLastError(), "k_ulp_axis_extrap launch");

  // Stage 3: full-grid average in place, which reads [jF, jF + 1] and writes
  // [jF], safe on a single rank.
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
  double pFactor = lambdaPreconditionerScale / (lamscale * lamscale);
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
        S.d_lambdaPreconditioner, S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_ulp_assemble launch");
  }

  // lambdaPreconditioner and the bLambda, dLambda, cLambda profiles stay on the
  // device.
  (void)lambdaPreconditioner_host;
  (void)bLambda_out;
  (void)dLambda_out;
  (void)cLambda_out;
}

// ComputeMHDForcesCuda: computeMHDForces on the device inputs.
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

  // z covers the configurations; the pair kernel shares a half-grid slab
  // between two force surfaces and needs an even extent.
  const int TPB = 64;
  if (ns_force_local % 2 == 0) {
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

  // All outputs (armn/azmn/brmn/bzmn/crmn/czmn) stay on device; downstream
  // inverse-FFT + AssembleTotalForces read d_* directly.
  (void)armn_e; (void)armn_o; (void)azmn_e; (void)azmn_o;
  (void)brmn_e; (void)brmn_o; (void)bzmn_e; (void)bzmn_o;
  (void)crmn_e; (void)crmn_o; (void)czmn_e; (void)czmn_o;
}

// ComputeForceNormsCuda: the half-grid reductions of fNormRZ and fNormL on the
// device.
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

  // lamscale is cached for k_check_convergence.
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

  // k_force_norm_final_reduce sums each configuration's partials into [sum_rz,
  // sum_l], two doubles per configuration to copy.
  S.EnsureFnormScalarsBuffer();
  k_force_norm_final_reduce<<<S.n_config_max, 256, 0, S.stream>>>(
      S.n_config_max, ns_h, S.d_forceNormRZ_partial, S.d_forceNormL_partial,
      S.d_fnorm_scalars,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_force_norm_final_reduce launch");

  // The copy covers 2 * n_cfg doubles into the cache under the same
  // synchronization; configuration zero's are the scalars.
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

  // Each configuration's fNorm1 from its own device positions at this cadence,
  // configuration zero bit-identical to fNorm1_out.
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
    // The per-configuration fNorm1 cache for evalFResPrecd, valid at the next
    // synchronization.
    if (static_cast<int>(g_fnorm1_per_cfg_cache.size()) != S.n_config_max) {
      g_fnorm1_per_cfg_cache.assign(S.n_config_max, 0.0);
    }
    cuda_check(cudaMemcpyAsync(g_fnorm1_per_cfg_cache.data(), S.d_fnorm1,
                                sizeof(double) * S.n_config_max,
                                cudaMemcpyDeviceToHost, S.stream),
               "d2h fnorm1 (per-cfg cache)");
  }
}

// HybridLambdaForceCuda: hybridLambdaForce on the device, writing blmn and
// clmn.
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

// PressureAndEnergiesCuda: presH and totalPressure on the device and the
// thermal, magnetic and MHD energies through the out-parameters.
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

  // Per-surface presH and thermal_partial in one kernel, with pres kept in
  // registers. Batched execution: y-dim covers n_config_max configs.
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

  // magnetic_partial from the magnetic pressure computed inline, ahead of the
  // totalPressure write.
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

  // totalPressure = magnetic pressure + presH in one kernel.
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

  // presH and totalPressure stay on the device; the energies reach the host
  // asynchronously, valid at ComputeForceNormsCuda's synchronization.
  (void)presH_out; (void)totalPressure_out;
  S.EnsurePressureScalarsBuffer();
  // Batched execution: launch n_config_max_max blocks. Each block reduces one config's
  // thermal/magnetic partials and writes 3 scalars at scalars_out[config*3:].
  k_pres_final_reduce<<<S.n_config_max, 256, 0, S.stream>>>(
      S.n_config_max, ns_h, deltaS, adiabaticIndex,
      S.d_thermal_partial, S.d_magnetic_partial, S.d_pressure_scalars,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_pres_final_reduce launch");
  // Configuration zero's energies into the out-parameters, valid after the
  // caller's next synchronization.
  cuda_check(cudaMemcpyAsync(&thermalEnergy_out, &S.d_pressure_scalars[0],
                              sizeof(double), cudaMemcpyDeviceToHost, S.stream),
             "d2h pressure thermal");
  cuda_check(cudaMemcpyAsync(&magneticEnergy_out, &S.d_pressure_scalars[1],
                              sizeof(double), cudaMemcpyDeviceToHost, S.stream),
             "d2h pressure magnetic");
  cuda_check(cudaMemcpyAsync(&mhdEnergy_out, &S.d_pressure_scalars[2],
                              sizeof(double), cudaMemcpyDeviceToHost, S.stream),
             "d2h pressure mhd");
  // All 3 * n_cfg energies into the cache under the same synchronization.
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

// ComputeInitialVolumeCuda: voli += sum of dVdsH * (2 pi)^2 * deltaS.
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
      S.d_dVdsH, S.d_scalar, nullptr);
  cuda_check(cudaGetLastError(), "k_volume_reduce launch (voli)");
  cuda_check(cudaMemcpyAsync(&voli_out, S.d_scalar, sizeof(double),
                              cudaMemcpyDeviceToHost, S.stream), "d2h voli");
  cuda_check(cudaStreamSynchronize(S.stream), "voli stream sync");
}

// UpdateVolumeCuda: plasmaVolume += sum of dVdsH * deltaS.
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
      S.d_dVdsH, S.d_scalar, S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_volume_reduce launch (plasmaVolume)");
  // Every configuration's volume into the cache; configuration zero's is
  // plasmaVolume_out.
  const int n_cfg_v = S.n_config_max;
  if (static_cast<int>(g_plasma_volume_cache.size()) != n_cfg_v) {
    g_plasma_volume_cache.assign(n_cfg_v, 0.0);
  }
  cuda_check(cudaMemcpyAsync(g_plasma_volume_cache.data(), S.d_scalar,
                              sizeof(double) * n_cfg_v,
                              cudaMemcpyDeviceToHost, S.stream),
             "d2h plasmaVolume per-cfg");
  cuda_check(cudaStreamSynchronize(S.stream), "plasmaVolume stream sync");
  plasmaVolume_out = g_plasma_volume_cache[0];
}

// ConstraintForceMultiplierCuda: tcon on the device from ruFull, zuFull and the
// device ard and azd.
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

  // ard and azd come from the device snapshots of
  // ComputePreconditioningMatrixCuda; the host arguments are unused.
  (void)ard; (void)azd;

  int jMin = (r.nsMinF == 0) ? 1 : 0;
  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 32;
  dim3 b(ns_force_local, 1, S.n_config_max); dim3 t(TPB, 1, 1);
  k_constraint_force_multiplier<<<b, t, 0, S.stream>>>(
      S.n_config_max, ns_con_local, ns_force_local, nZnT, nThetaEff, jMin,
      /*kEvenParity=*/0, tcon_factor,
      S.d_ruFull, S.d_zuFull, S.d_pmat_ard, S.d_pmat_azd, S.d_wInt, S.d_tcon,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_constraint_force_multiplier launch");

  // Halves each configuration's LCFS row of tcon on the device, at the host's
  // index, before DeAliasConstraintForceCuda reads it.
  if (r.nsMaxF1 == fc.ns && ns_con_local >= 2) {
    int last = (r.nsMaxF1 - 1) - r.nsMinF;
    k_halve_tcon_lcfs<<<S.n_config_max, 1, 0, S.stream>>>(
        S.n_config_max, ns_con_local, last, S.d_tcon, S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_halve_tcon_lcfs launch");
  }

  // D2H tcon for host visibility (not consumed mid-chain in CUDA mode but
  // preserved for output paths). No sync; defer to end-of-update.
  cuda_check(cudaMemcpyAsync(tcon_out.data(), S.d_tcon,
                              sizeof(double) * ns_force_local,
                              cudaMemcpyDeviceToHost, S.stream), "d2h tcon");
}

// EffectiveConstraintForceCuda: gConEff on the device from rCon, rCon0, zCon,
// zCon0, ruFull and zuFull.
void EffectiveConstraintForceCuda(
    const RadialPartitioning& r, const Sizes& s,
    Eigen::VectorXd& gConEff_out) {
  auto& S = State();
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int nZnT = s.nZnT;
  if (ns_con_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureConstraintMultiplierBuffers(r.nsMaxF - r.nsMinF, ns_con_local, nZnT);
  // A free-boundary stage entered with the vacuum pressure active skips
  // rzConIntoVolume and uses the zeroed origins.
  S.EnsureRzCon0Buffers(ns_con_local, nZnT);

  int jMin = (r.nsMinF == 0) ? 1 : 0;
  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 64;
  dim3 b((nZnT + TPB - 1) / TPB, ns_con_local, S.n_config_max);
  dim3 t(TPB, 1, 1);
  k_effective_constraint_force<<<b, t, 0, S.stream>>>(
      S.n_config_max, ns_con_local, nZnT, jMin,
      S.d_rCon, S.d_rCon0, S.d_zCon, S.d_zCon0,
      S.d_ruFull, S.d_zuFull, S.d_gConEff);
  cuda_check(cudaGetLastError(), "k_effective_constraint_force launch");

  // d_gConEff stays on device; DeAliasConstraintForceCuda reads it directly.
  (void)gConEff_out;
}

// Free-boundary bridges between the device state and the host NESTOR solve in
// IdealMhdModel::update.

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

// One flush per vacuum iteration and configuration: the axis and LCFS rows,
// outer totalPressure rows and the presH, bucoH, bvcoH profiles.
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

// Stages one configuration's host rBSq profile, which AssembleTotalForcesCuda
// applies to its LCFS force row.
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

// Vacuum edge pressure on each configuration's LCFS row: armn += zuFull * rBSq,
// azmn -= ruFull * rBSq.
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

// AssembleTotalForcesCuda: assembleTotalForces on the device, adding the
// constraint force into brmn and bzmn and writing frcon and fzcon.
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
  // Free boundary: the staged vacuum edge pressure goes on the LCFS row before
  // the constraint assembly, as in assembleTotalForces.
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
  k_assemble_total_forces<<<b, t, 0, S.stream>>>(
      S.n_config_max, ns_con_local, ns_force_local, nZnT, nsMinF_to_nsMinF1,
      S.d_rCon, S.d_rCon0, S.d_zCon, S.d_zCon0, S.d_gCon,
      S.d_ruFull, S.d_zuFull, S.d_sqrtSF,
      S.d_brmn_e, S.d_brmn_o, S.d_bzmn_e, S.d_bzmn_o,
      S.d_frcon_e, S.d_frcon_o, S.d_fzcon_e, S.d_fzcon_o,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_assemble_total_forces launch");

  // All outputs stay on device; the CUDA inverse FFT reads d_brmn_*, d_bzmn_*,
  // d_frcon_*, d_fzcon_* directly.
  (void)brmn_e_out; (void)brmn_o_out; (void)bzmn_e_out; (void)bzmn_o_out;
  (void)frcon_e_out; (void)frcon_o_out; (void)fzcon_e_out; (void)fzcon_o_out;
}

// Copies the m = 0, n >= 1 lambda forces, before the diagonal scaling, into the
// right-hand sides of the tridiagonal m = 0 block.
__global__ void k_lambda_m0_gather(
    int n_config, int ns_local, int ns_con_local, int mpol, int ntor,
    int nsMinF, int ns_total, const double* __restrict__ flcs,
    double* __restrict__ m0_c,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  const int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  const int n = blockIdx.y + 1;
  const int jF_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_local >= ns_con_local || n > ntor) return;
  const size_t cfg_spec =
      (size_t)config * (size_t)ns_local * (size_t)mpol * (size_t)(ntor + 1);
  const size_t row = (size_t)config * (size_t)(ntor + 1) + (size_t)n;
  m0_c[row * (size_t)ns_total + (size_t)(jF_local + nsMinF)] =
      flcs[cfg_spec + (size_t)jF_local * (size_t)mpol * (size_t)(ntor + 1) +
           (size_t)n];
}

// The m = 0 lambda block of one (cfg, n) eliminated as TridiagonalSolveSerial
// does, uncontracted; step_scale times it replaces the forces.
__global__ void k_lambda_m0_solve(
    int n_config, int ns_local, int mpol, int ntor, int nfp, int nsMinF,
    int ns_total, int lambda_stride, double checkerboard_terms,
    double step_scale, const double* __restrict__ b_half,
    double* __restrict__ m0_c, double* __restrict__ m0_a,
    double* __restrict__ flcs,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  const int config = blockIdx.y;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  const int n = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (n > ntor) return;
  const int ns = ns_total;
  const double tnn = (double)(n * nfp * n * nfp);
  const double* lb = b_half + (size_t)config * (size_t)lambda_stride;
  const size_t row = (size_t)config * (size_t)(ntor + 1) + (size_t)n;
  double* c = m0_c + row * (size_t)ns;
  double* a = m0_a + row * (size_t)ns;
  const double diagonal_factor =
      __dadd_rn(0.25, __dmul_rn(0.5, checkerboard_terms));
  // Row j: a couples to j + 1, b to j - 1, d is the diagonal.
  auto coefficients = [&](int j, double& aj, double& dj, double& bj) {
    const double a_in = __dmul_rn(tnn, lb[j]);
    const double a_out = __dmul_rn(tnn, lb[j + 1]);
    bj = __dmul_rn(0.25, a_in);
    aj = __dmul_rn(0.25, a_out);
    dj = __dmul_rn(diagonal_factor, __dadd_rn(a_in, a_out));
    if (j == 1) {
      dj = __dadd_rn(dj, __dmul_rn(__dmul_rn(0.25, tnn), lb[1]));
    }
  };
  double aj = 0.0;
  double dj = 0.0;
  double bj = 0.0;
  coefficients(1, aj, dj, bj);
  const double d1 = dj;
  a[1] = __ddiv_rn(aj, d1);
  for (int j = 2; j < ns - 1; ++j) {
    coefficients(j, aj, dj, bj);
    a[j] = __ddiv_rn(aj, __dsub_rn(dj, __dmul_rn(a[j - 1], bj)));
  }
  c[1] = __ddiv_rn(c[1], d1);
  for (int j = 2; j < ns; ++j) {
    coefficients(j, aj, dj, bj);
    const double denominator = __dsub_rn(dj, __dmul_rn(a[j - 1], bj));
    c[j] = __ddiv_rn(__dsub_rn(c[j], __dmul_rn(c[j - 1], bj)), denominator);
  }
  for (int j = ns - 2; j > 0; --j) {
    c[j] = __dsub_rn(c[j], __dmul_rn(a[j], c[j + 1]));
  }
  const size_t cfg_spec =
      (size_t)config * (size_t)ns_local * (size_t)mpol * (size_t)(ntor + 1);
  for (int j = (nsMinF > 1 ? nsMinF : 1); j < ns; ++j) {
    flcs[cfg_spec + (size_t)(j - nsMinF) * (size_t)mpol * (size_t)(ntor + 1) +
         (size_t)n] = __dmul_rn(c[j], step_scale);
  }
}

// ApplyLambdaPreconditionerCuda
void ApplyLambdaPreconditionerCuda(
    const RadialPartitioning& r, const Sizes& s,
    const Eigen::VectorXd& lambdaPreconditioner, double checkerboard_terms,
    double step_scale, double* flsc_host, double* flcs_host) {
  auto& S = State();
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  if (ns_con_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  S.EnsureLambdaInputBuffer(ns_con_local, mpol, ntor);
  size_t spec_bytes = sizeof(double) * ns_con_local * mpol * (ntor + 1);
  // m = 0 lambda lives in lmncs, so the tridiagonal block applies in 3D only.
  // UpdateLambdaPreconditionerCuda has sized its buffers for this stage.
  const int ns_total = r.nsMaxFIncludingLcfs;
  const bool m0_tridiagonal =
      checkerboard_terms >= 0.0 && s.lthreed && ntor >= 1;
  if (m0_tridiagonal) {
    S.EnsureLambdaM0Buffers(ns_con_local + 1, ns_total, ntor);
    const int TPB = 32;
    dim3 b((ns_con_local + TPB - 1) / TPB, ntor, S.n_config_max);
    k_lambda_m0_gather<<<b, TPB, 0, st>>>(
        S.n_config_max, S.ns_local_cached, ns_con_local, mpol, ntor, r.nsMinF,
        ns_total, S.d_decomposed_flcs, S.d_lambda_m0_c, S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_lambda_m0_gather launch");
  }
  // d_lambdaPreconditioner from k_ulp_assemble equals the host argument, which
  // is unused here.
  (void)lambdaPreconditioner;
  // flsc and flcs are scaled in place on the decomposed shadow; z = config *
  // ns_con_local + jF_local.
  const int TPB = 16;
  dim3 b((ntor + 1 + TPB - 1) / TPB, mpol, ns_con_local * S.n_config_max);
  dim3 t(TPB, 1, 1);
  k_apply_lambda_preconditioner<<<b, t, 0, st>>>(
      S.n_config_max, S.ns_local_cached, ns_con_local, mpol, ntor, s.lthreed,
      S.d_lambdaPreconditioner,
      S.d_decomposed_flsc, S.d_decomposed_flcs,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_apply_lambda launch");
  if (m0_tridiagonal) {
    const int TPB = 32;
    dim3 b((ntor + TPB - 1) / TPB, S.n_config_max, 1);
    k_lambda_m0_solve<<<b, TPB, 0, st>>>(
        S.n_config_max, S.ns_local_cached, mpol, ntor, s.nfp, r.nsMinF,
        ns_total, ns_con_local + 1, checkerboard_terms, step_scale,
        S.d_lambda_b_half, S.d_lambda_m0_c, S.d_lambda_m0_a,
        S.d_decomposed_flcs, S.d_active_per_cfg);
    cuda_check(cudaGetLastError(), "k_lambda_m0_solve launch");
  }
  // D2H + sync deferred to end-of-residue() FlushDecomposedToHostCuda.
  (void)flsc_host; (void)flcs_host; (void)spec_bytes;
}

// AssembleRZPreconditionerCuda: assembleRZPreconditioner on the device into
// d_rz_* and d_rz_jMin in the (mn, jF_global) layout.
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
}

// ApplyRZPreconditionerCuda: the tridiagonal solves on the device coefficients;
// the host coefficient arguments are unused.
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

  // AssembleRZPreconditionerCuda has filled the coefficient buffers and
  // d_rz_jMin, so the host arguments go unused.
  S.EnsureRZBuffers(mnsize, ns_total, num_basis);
  double *d_cR = S.d_rz_cR;
  double *d_cZ = S.d_rz_cZ;
  int   *d_jMin = S.d_rz_jMin;
  (void)ar; (void)dr; (void)br_in; (void)az; (void)dz; (void)bz_in;
  (void)jMin_arr; (void)jMin_size;

  // Device-side transpose decomposed shadow -> cR/cZ.
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

  // PCR: one block per (config, mn), the next power of two at least jMaxRZ
  // threads, the whole system in shared memory.
  int pcr_threads = 32;
  while (pcr_threads < jMaxRZ) pcr_threads <<= 1;
  if (pcr_threads > 1024) pcr_threads = 1024;
  size_t pcr_smem = 5 * ns_total * sizeof(double);
  dim3 pcr_grid(mnsize, S.n_config_max, 1);

  if (jMaxRZ > 1024) {
    // Above 1024 radial points each (config, mn) row takes a serial Thomas
    // block with the ratios in dynamic shared memory.
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

  // Device-side transpose cR/cZ -> decomposed shadow.
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

  // The decomposed shadow reaches the host in residue()'s final flush; stream
  // order keeps ResidualsCuda consistent.
  (void)frcc_host; (void)frss_host; (void)fzsc_host; (void)fzcs_host;
  // Buffers are persistent in CudaToroidalState; do NOT free here.
  return 0;
}

// Warp-packed k_dealias_inv: blockDim.x / nThetaReduced zeta planes per warp,
// so all 32 lanes work.
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

// DeAliasConstraintForceCuda: gCon on the device from gConEff and tcon, which
// AssembleTotalForcesCuda reads; m_gCon is unused.
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

  // gsc, gcs and faccon are allocated once per Reshape, and cosnv and sinnv
  // staged once.
  S.EnsureDealiasBuffers(mpol, ntor, ns_force_local);

  // faccon = -0.25 * signOfJacobian / xmpq[m]^2 is fixed by the IdealMhdModel
  // constructor, so it is staged once per Reshape.
  if (!S.dealias_faccon_staged) {
    cuda_check(cudaMemcpyAsync(S.d_dealias_faccon, faccon.data(),
                                sizeof(double) * mpol,
                                cudaMemcpyHostToDevice, st), "h2d faccon (one-shot)");
    S.dealias_faccon_staged = true;
  }

  // Stage 1: forward poloidal and toroidal sums into gsc and gcs, z = config *
  // ns_force_local + jF; 16 threads per block.
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

  // Stage 2: inverse sums into gCon, which assigns every cell at lasym = false,
  // so no zero fill.
  {
    const int TPB = 32;
    dim3 b((nThetaReduced + TPB - 1) / TPB, nZeta,
           ns_force_local * S.n_config_max);
    dim3 t(TPB, 1, 1);
    if (mpol == 10 && ntor == 10) {
      k_dealias_inv_tpl<10, 10><<<b, t, 0, st>>>(
          S.n_config_max, ns_force_local, ns_con_local,
          nZeta, nThetaReduced, nThetaEff, nnyq2_plus_1,
          S.d_dealias_gsc, S.d_dealias_gcs, S.d_sinmu, S.d_cosmu,
          S.d_dealias_cosnv, S.d_dealias_sinnv, S.d_dealias_faccon,
          S.d_gCon,
          S.d_active_per_cfg);
      cuda_check(cudaGetLastError(), "k_dealias_inv_tpl<10,10> launch");
    } else {
      // Warp-pack only when more than one zeta plane fits in a 32-lane warp
      // and divides it cleanly (no lane straddles two planes); else default.
      const int ks_per_warp = (nThetaReduced > 0) ? (32 / nThetaReduced) : 0;
      if (ks_per_warp >= 2 && (32 % nThetaReduced) == 0) {
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
  }

  // d_gCon stays on the device for AssembleTotalForcesCuda on the same stream;
  // m_gCon_host is unused.
  (void)m_gCon_host;
}

}  // namespace vmecpp

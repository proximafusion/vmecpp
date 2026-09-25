#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"

namespace vmecpp {

void FlushDVdsHToHostCuda(int ns_h, double* dVdsH_host) {
  auto& S = State();
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.d_dVdsH || !S.stream || ns_h <= 0) return;
  // The device buffer's extent governs; the copy lands in a local staging
  // buffer so a partial fill never writes past the caller's array.
  if (S.ns_h_cached > 0 && S.ns_h_cached < ns_h) {
    ns_h = S.ns_h_cached;
  }
  std::vector<double> staged(ns_h, 0.0);
  cuda_check(cudaMemcpyAsync(staged.data(), S.d_dVdsH,
                              sizeof(double) * ns_h, cudaMemcpyDeviceToHost,
                              S.stream),
             "d2h dVdsH (printout)");
  cuda_check(cudaStreamSynchronize(S.stream), "dVdsH printout sync");
  if (dVdsH_host) {
    std::memcpy(dVdsH_host, staged.data(), sizeof(double) * ns_h);
  }
}

void FlushDecomposedToHostCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double* dec_frcc_host, double* dec_frss_host,
    double* dec_fzsc_host, double* dec_fzcs_host,
    double* dec_flsc_host, double* dec_flcs_host) {
  auto& S = State();
  int ns_dec_local =
      (r.nsMaxF1 == fc.ns) ? (fc.ns - r.nsMinF) : (r.nsMaxF - r.nsMinF);
  int mpol = s.mpol;
  int ntor = s.ntor;
  if (ns_dec_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  // Decompose-flush D2H + sync elided. The flush was originally required
  // for Vmec::performTimeStep, which predated the CUDA port of that step.
  // With PerformTimeStepCuda reading device d_decomposed_f directly,
  // no per-iter host consumer of m_decomposed_f remains: ResidualsCuda /
  // ApplyM1PreconditionerCuda / ApplyRZPreconditionerCuda /
  // ApplyLambdaPreconditionerCuda all operate on device buffers and explicitly
  // (void)cast their host args. The 6 D2Hs + sync per iter saved here is
  // measurable wall (~50-100us per iter * 10K iters = 0.5-1.0s = 0.4-0.8pct).
  (void)dec_frcc_host; (void)dec_frss_host;
  (void)dec_fzsc_host; (void)dec_fzcs_host;
  (void)dec_flsc_host; (void)dec_flcs_host;
  (void)st;
}

// ============================================================================
// FlushForOutputQuantitiesCuda
//
// Flush the device-resident half-grid scalar fields back to their host-side
// IdealMhdModel members so ComputeOutputQuantities → GatherDataFromThreads
// reads correct data. wout.bmnc and friends are filled from these arrays;
// without the flush they get the host buffers' uninitialized / denormal-noise
// content and the Boozer transform downstream sees ~0.
// ============================================================================
void FlushForOutputQuantitiesCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double* gsqrt_host, double* guu_host, double* guv_host, double* gvv_host,
    double* bsubu_host, double* bsubv_host,
    double* bsupu_host, double* bsupv_host,
    double* totalPressure_host,
    double* r12_host, double* ru12_host, double* zu12_host,
    double* rs_host, double* zs_host,
    double* r1_e_host, double* r1_o_host, double* z1_e_host, double* z1_o_host,
    double* ru_e_host, double* ru_o_host, double* zu_e_host, double* zu_o_host,
    double* rv_e_host, double* rv_o_host, double* zv_e_host, double* zv_o_host,
    double* ruFull_host, double* zuFull_host,
    double* blmn_e_host,
    double* presH_host, double* dVdsH_host, double* bvcoH_host,
    double* jcurvF_host, double* jcuruF_host, double* presgradF_host,
    double* dVdsF_host, double* equiF_host,
    double* chipH_host, double* iotaH_host,
    double* chipF_host, double* iotaF_host) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int ns_force_local = (r.nsMaxF1 == fc.ns) ? (fc.ns - r.nsMinF)
                                                  : (r.nsMaxF - r.nsMinF);
  const int nsi = r.nsMaxFi - r.nsMinFi;  // interior (axis-excluded) count;
                                          // d_jcurvF/etc. allocated this size
  const int nZnT = s.nZnT;
  if (ns_h <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  const size_t half_bytes     = sizeof(double) * ns_h * nZnT;
  const size_t full_bytes     = sizeof(double) * ns_local * nZnT;
  const size_t force_bytes    = sizeof(double) * ns_force_local * nZnT;
  const size_t presH_bytes    = sizeof(double) * ns_h;
  const size_t nsi_bytes      = sizeof(double) * nsi;

  auto d2h = [&](double* host, double* dev, size_t bytes, const char* name) {
    if (host && dev) {
      cuda_check(cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost, st),
                  name);
    }
  };

  // half-grid scalars (ns_h * nZnT each)
  d2h(gsqrt_host,          S.d_gsqrt,          half_bytes, "flush gsqrt");
  d2h(guu_host,            S.d_guu,            half_bytes, "flush guu");
  d2h(guv_host,            S.d_guv,            half_bytes, "flush guv");
  d2h(gvv_host,            S.d_gvv,            half_bytes, "flush gvv");
  d2h(bsubu_host,          S.d_bsubu,          half_bytes, "flush bsubu");
  d2h(bsubv_host,          S.d_bsubv,          half_bytes, "flush bsubv");
  d2h(bsupu_host,          S.d_bsupu,          half_bytes, "flush bsupu");
  d2h(bsupv_host,          S.d_bsupv,          half_bytes, "flush bsupv");
  d2h(totalPressure_host,  S.d_totalPressure,  half_bytes, "flush totalPressure");
  d2h(r12_host,            S.d_r12,            half_bytes, "flush r12");
  d2h(ru12_host,           S.d_ru12,           half_bytes, "flush ru12");
  d2h(zu12_host,           S.d_zu12,           half_bytes, "flush zu12");
  d2h(rs_host,             S.d_rs,             half_bytes, "flush rs");
  d2h(zs_host,             S.d_zs,             half_bytes, "flush zs");

  // full-grid R/Z and derivatives (ns_local * nZnT each)
  d2h(r1_e_host,           S.d_r1_e,           full_bytes, "flush r1_e");
  d2h(r1_o_host,           S.d_r1_o,           full_bytes, "flush r1_o");
  d2h(z1_e_host,           S.d_z1_e,           full_bytes, "flush z1_e");
  d2h(z1_o_host,           S.d_z1_o,           full_bytes, "flush z1_o");
  d2h(ru_e_host,           S.d_ru_e,           full_bytes, "flush ru_e");
  d2h(ru_o_host,           S.d_ru_o,           full_bytes, "flush ru_o");
  d2h(zu_e_host,           S.d_zu_e,           full_bytes, "flush zu_e");
  d2h(zu_o_host,           S.d_zu_o,           full_bytes, "flush zu_o");
  if (s.lthreed) {
    d2h(rv_e_host,         S.d_rv_e,           full_bytes, "flush rv_e");
    d2h(rv_o_host,         S.d_rv_o,           full_bytes, "flush rv_o");
    d2h(zv_e_host,         S.d_zv_e,           full_bytes, "flush zv_e");
    d2h(zv_o_host,         S.d_zv_o,           full_bytes, "flush zv_o");
  }
  d2h(ruFull_host,         S.d_ruFull,         force_bytes, "flush ruFull");
  d2h(zuFull_host,         S.d_zuFull,         force_bytes, "flush zuFull");

  // force-local
  d2h(blmn_e_host,         S.d_blmn_e,         force_bytes, "flush blmn_e");

  // radial half-grid (ns_h)
  d2h(presH_host,          S.d_presH,          presH_bytes, "flush presH");
  d2h(dVdsH_host,          S.d_dVdsH,          presH_bytes, "flush dVdsH");
  d2h(bvcoH_host,          S.d_bvcoH,          presH_bytes, "flush bvcoH");

  // chipH, iotaH (ns_h doubles each), chipF, iotaF (ns_local doubles each).
  // ComputeBContraCuda previously did these as per-iter async D2Hs even though
  // every host reader except output_quantities lives in a CPU-only branch.
  // Consolidated to this one-shot flush; per-iter 4 D2Hs eliminated.
  d2h(chipH_host,          S.d_chipH,          presH_bytes, "flush chipH");
  d2h(iotaH_host,          S.d_iotaH,          presH_bytes, "flush iotaH");
  const size_t chipFull_bytes = sizeof(double) * ns_local;
  d2h(chipF_host,          S.d_chipF,          chipFull_bytes, "flush chipF");
  d2h(iotaF_host,          S.d_iotaF,          chipFull_bytes, "flush iotaF");

  // radial interior (nsi = nsMaxFi - nsMinFi). d_jcurvF/d_jcuruF/d_presgradF/
  // d_dVdsF/d_equiF are allocated this size by EnsureRadialForceBalanceBuffers;
  // host m_p_.jcurvF is indexed by (jFi - nsMinFi), so destination base aligns
  // 1:1 with device base.
  if (nsi > 0) {
    d2h(jcurvF_host,       S.d_jcurvF,         nsi_bytes, "flush jcurvF");
    d2h(jcuruF_host,       S.d_jcuruF,         nsi_bytes, "flush jcuruF");
    d2h(presgradF_host,    S.d_presgradF,      nsi_bytes, "flush presgradF");
    d2h(dVdsF_host,        S.d_dVdsF,          nsi_bytes, "flush dVdsF");
    d2h(equiF_host,        S.d_equiF,          nsi_bytes, "flush equiF");
  }

  // (Per-config batched flush is provided by FlushAllConfigsForOutputCuda;
  //  see below.)

  // When the VMECPP_BATCH_OUTPUTS_FILE environment variable names a
  // destination path and the run carries more than one configuration,
  // the final per-configuration decomposed spectra are dumped to that
  // path. The six spectral components -- rmncc, rmnss, zmnsc, zmncs,
  // lmnsc, and lmncs -- are read from the device buffers d_pts_x_rcc,
  // d_pts_x_rss, d_pts_x_zsc, d_pts_x_zcs, d_pts_x_lsc, and
  // d_pts_x_lcs across all configurations and written to disk for
  // downstream per-equilibrium consumers (such as the per-cfg
  // aspect-ratio and metric reconstruction). The on-disk format
  // mirrors the input-side counterpart consumed by the file-based
  // batched-input pipeline: a four-element int32 header carrying
  // (N, ns_local, mpol, ntor), followed by
  // N * 6 * ns_local * mpol * (ntor + 1) double-precision values in
  // [spectral_component][configuration][spectra] order.
  const char* batch_out_path = std::getenv("VMECPP_BATCH_OUTPUTS_FILE");
  if (S.n_config_max > 1 && S.pts_x_initialized) {
    int N_out = S.n_config_max;
    int ns_out = S.ns_local_cached;
    int mpol_out = S.mpol_cached;
    int ntor_out = S.ntor_cached;
    size_t per_spec_doubles = (size_t)ns_out * mpol_out * (ntor_out + 1);
    size_t total_doubles = (size_t)N_out * 6 * per_spec_doubles;
    double* h_buf = nullptr;
    cuda_check(cudaMallocHost(&h_buf, sizeof(double) * total_doubles),
               "alloc batch_outputs_pinned");
    // D2H each spec array for all N cfgs. A cfg that went inactive during
    // the batch has a converged-state snapshot (d_pts_x_final_*); its live
    // slice kept being modified by mask-agnostic kernels afterward, so the
    // snapshot is the trustworthy source for that cfg.
    double* d_specs[6] = {S.d_pts_x_rcc, S.d_pts_x_rss,
                          S.d_pts_x_zsc, S.d_pts_x_zcs,
                          S.d_pts_x_lsc, S.d_pts_x_lcs};
    double* d_finals[6] = {S.d_pts_x_final_rcc, S.d_pts_x_final_rss,
                           S.d_pts_x_final_zsc, S.d_pts_x_final_zcs,
                           S.d_pts_x_final_lsc, S.d_pts_x_final_lcs};
    int n_snap = 0;
    for (int sp = 0; sp < 6; ++sp) {
      for (int cfg = 0; cfg < N_out; ++cfg) {
        const bool use_snap =
            S.d_pts_x_final_rcc &&
            cfg < static_cast<int>(S.pts_x_final_taken.size()) &&
            S.pts_x_final_taken[cfg];
        if (sp == 0 && use_snap) ++n_snap;
        const double* src =
            (use_snap ? d_finals[sp] : d_specs[sp]) +
            (size_t)cfg * per_spec_doubles;
        cuda_check(cudaMemcpyAsync(
            h_buf + ((size_t)sp * N_out + cfg) * per_spec_doubles, src,
            sizeof(double) * per_spec_doubles,
            cudaMemcpyDeviceToHost, st), "batch out D2H");
      }
    }
    if (n_snap > 0) {
      std::fprintf(stderr,
          "[fft_toroidal_cuda] batch outputs: %d/%d cfgs from "
          "converged-state snapshots\n", n_snap, N_out);
    }
    cuda_check(cudaStreamSynchronize(st), "batch out sync");
    g_batch_outputs_mem.assign(h_buf, h_buf + total_doubles);
    g_batch_outputs_shape[0] = N_out;
    g_batch_outputs_shape[1] = ns_out;
    g_batch_outputs_shape[2] = mpol_out;
    g_batch_outputs_shape[3] = ntor_out;
    if (batch_out_path && *batch_out_path) {
      FILE* f = std::fopen(batch_out_path, "wb");
      if (f) {
        int32_t header[4] = {N_out, ns_out, mpol_out, ntor_out};
        std::fwrite(header, sizeof(int32_t), 4, f);
        std::fwrite(h_buf, sizeof(double), total_doubles, f);
        std::fclose(f);
        std::fprintf(stderr,
            "[fft_toroidal_cuda] batch outputs written: N=%d ns=%d mpol=%d "
            "ntor=%d (%zu doubles to %s)\n",
            N_out, ns_out, mpol_out, ntor_out, total_doubles, batch_out_path);
      } else {
        std::fprintf(stderr,
            "[fft_toroidal_cuda] could not open %s for writing batch "
            "outputs\n",
            batch_out_path);
      }
    }
    cudaFreeHost(h_buf);
  }

  cuda_check(cudaStreamSynchronize(st), "flush output_quantities sync");
}

// ============================================================================
// FlushAllConfigsForOutputCuda
//
// The per-configuration variant of FlushForOutputQuantitiesCuda. The
// host-side destinations carry the same set of arrays, but each one is
// sized for all configurations in the batch and laid out in
// configuration-major order (the entries for configuration 0 followed
// by those for configuration 1, and so on). The corresponding device
// buffers are already strided per configuration through the
// batched-buffer layout, so each device-to-host transfer copies
// n_config_max times the single-configuration byte count in one
// operation. The routine is the device-side counterpart of the
// batched-output dump file, and together they enable Python-side
// post-processing of all converged equilibria emerging from a single
// batched VMEC run, completing the file-based batched-input/output
// pipeline that delivers per-configuration throughput to the
// Python batch driver.
//
// The caller pre-sizes every host buffer to n_cfg times the
// per-configuration byte count of the current Reshape; raw pointers
// carry no extent, so no size validation happens here.
// ============================================================================
void FlushAllConfigsForOutputCudaNs(
    int ns, const Sizes& s, int n_cfg,
    double* gsqrt_host, double* guu_host, double* guv_host, double* gvv_host,
    double* bsubu_host, double* bsubv_host,
    double* bsupu_host, double* bsupv_host,
    double* totalPressure_host,
    double* r12_host, double* ru12_host, double* zu12_host,
    double* rs_host, double* zs_host,
    double* r1_e_host, double* r1_o_host, double* z1_e_host, double* z1_o_host,
    double* ru_e_host, double* ru_o_host, double* zu_e_host, double* zu_o_host,
    double* rv_e_host, double* rv_o_host, double* zv_e_host, double* zv_o_host,
    double* ruFull_host, double* zuFull_host,
    double* blmn_e_host,
    double* presH_host, double* dVdsH_host, double* bvcoH_host,
    double* bucoH_host,
    double* chipH_host, double* iotaH_host,
    double* chipF_host, double* iotaF_host,
    double* pts_x_rcc_host, double* pts_x_rss_host,
    double* pts_x_zsc_host, double* pts_x_zcs_host,
    double* pts_x_lsc_host, double* pts_x_lcs_host) {
  // Single-rank extents derived from ns; forwards to the partition-based
  // variant via a minimal RadialPartitioning-free path. The interior
  // full-grid arrays (jcurvF and friends) are not flushed here.
  RadialPartitioning r;
  r.adjustRadialPartitioning(/*num_threads=*/1, /*thread_id=*/0, ns,
                             /*lfreeb=*/false, /*printout=*/false);
  FlowControl fc(/*lfreeb=*/false, /*delt=*/1.0, /*num_grids=*/1,
                 /*max_threads=*/1);
  fc.ns = ns;
  FlushAllConfigsForOutputCuda(
      r, s, fc, n_cfg, gsqrt_host, guu_host, guv_host, gvv_host, bsubu_host,
      bsubv_host, bsupu_host, bsupv_host, totalPressure_host, r12_host,
      ru12_host, zu12_host, rs_host, zs_host, r1_e_host, r1_o_host, z1_e_host,
      z1_o_host, ru_e_host, ru_o_host, zu_e_host, zu_o_host, rv_e_host,
      rv_o_host, zv_e_host, zv_o_host, ruFull_host, zuFull_host, blmn_e_host,
      presH_host, dVdsH_host, bvcoH_host, bucoH_host, nullptr, nullptr,
      nullptr, nullptr, nullptr, chipH_host, iotaH_host, chipF_host,
      iotaF_host, pts_x_rcc_host, pts_x_rss_host, pts_x_zsc_host,
      pts_x_zcs_host, pts_x_lsc_host, pts_x_lcs_host);
}

void FlushAllConfigsForOutputCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    int n_cfg,
    double* gsqrt_host, double* guu_host, double* guv_host, double* gvv_host,
    double* bsubu_host, double* bsubv_host,
    double* bsupu_host, double* bsupv_host,
    double* totalPressure_host,
    double* r12_host, double* ru12_host, double* zu12_host,
    double* rs_host, double* zs_host,
    double* r1_e_host, double* r1_o_host, double* z1_e_host, double* z1_o_host,
    double* ru_e_host, double* ru_o_host, double* zu_e_host, double* zu_o_host,
    double* rv_e_host, double* rv_o_host, double* zv_e_host, double* zv_o_host,
    double* ruFull_host, double* zuFull_host,
    double* blmn_e_host,
    double* presH_host, double* dVdsH_host, double* bvcoH_host,
    double* bucoH_host,
    double* jcurvF_host, double* jcuruF_host, double* presgradF_host,
    double* dVdsF_host, double* equiF_host,
    double* chipH_host, double* iotaH_host,
    double* chipF_host, double* iotaF_host,
    // pts_x spec arrays (rcc/rss/zsc/zcs/lsc/lcs) for ALL n_cfg configs:
    double* pts_x_rcc_host, double* pts_x_rss_host,
    double* pts_x_zsc_host, double* pts_x_zcs_host,
    double* pts_x_lsc_host, double* pts_x_lcs_host) {
  auto& S = State();
  const int ns_h = r.nsMaxH - r.nsMinH;
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int ns_force_local = (r.nsMaxF1 == fc.ns) ? (fc.ns - r.nsMinF)
                                                  : (r.nsMaxF - r.nsMinF);
  const int nsi = r.nsMaxFi - r.nsMinFi;
  const int nZnT = s.nZnT;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  if (ns_h <= 0 || n_cfg <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  const size_t per_cfg_doubles_half  = (size_t)ns_h         * (size_t)nZnT;
  const size_t per_cfg_doubles_full  = (size_t)ns_local     * (size_t)nZnT;
  const size_t per_cfg_doubles_force = (size_t)ns_force_local * (size_t)nZnT;
  const size_t per_cfg_doubles_presH = (size_t)ns_h;
  const size_t per_cfg_doubles_chipF = (size_t)ns_local;
  const size_t per_cfg_doubles_nsi   = (size_t)nsi;
  const size_t per_cfg_doubles_pts   = (size_t)ns_local * (size_t)mpol
                                       * (size_t)(ntor + 1);
  const size_t all_bytes_half  = sizeof(double) * (size_t)n_cfg * per_cfg_doubles_half;
  const size_t all_bytes_full  = sizeof(double) * (size_t)n_cfg * per_cfg_doubles_full;
  const size_t all_bytes_force = sizeof(double) * (size_t)n_cfg * per_cfg_doubles_force;
  const size_t all_bytes_presH = sizeof(double) * (size_t)n_cfg * per_cfg_doubles_presH;
  const size_t all_bytes_chipF = sizeof(double) * (size_t)n_cfg * per_cfg_doubles_chipF;
  const size_t all_bytes_nsi   = sizeof(double) * (size_t)n_cfg * per_cfg_doubles_nsi;
  const size_t all_bytes_pts   = sizeof(double) * (size_t)n_cfg * per_cfg_doubles_pts;

  auto d2h = [&](double* host, double* dev, size_t bytes, const char* name) {
    if (host && dev) {
      cuda_check(cudaMemcpyAsync(host, dev, bytes,
                                  cudaMemcpyDeviceToHost, st), name);
    }
  };

  // half-grid scalars
  d2h(gsqrt_host,         S.d_gsqrt,         all_bytes_half,  "flush all gsqrt");
  d2h(guu_host,           S.d_guu,           all_bytes_half,  "flush all guu");
  d2h(guv_host,           S.d_guv,           all_bytes_half,  "flush all guv");
  d2h(gvv_host,           S.d_gvv,           all_bytes_half,  "flush all gvv");
  d2h(bsubu_host,         S.d_bsubu,         all_bytes_half,  "flush all bsubu");
  d2h(bsubv_host,         S.d_bsubv,         all_bytes_half,  "flush all bsubv");
  d2h(bsupu_host,         S.d_bsupu,         all_bytes_half,  "flush all bsupu");
  d2h(bsupv_host,         S.d_bsupv,         all_bytes_half,  "flush all bsupv");
  d2h(totalPressure_host, S.d_totalPressure, all_bytes_half,  "flush all totalPressure");
  d2h(r12_host,           S.d_r12,           all_bytes_half,  "flush all r12");
  d2h(ru12_host,          S.d_ru12,          all_bytes_half,  "flush all ru12");
  d2h(zu12_host,          S.d_zu12,          all_bytes_half,  "flush all zu12");
  d2h(rs_host,            S.d_rs,            all_bytes_half,  "flush all rs");
  d2h(zs_host,            S.d_zs,            all_bytes_half,  "flush all zs");

  // full-grid R/Z and derivatives
  d2h(r1_e_host,          S.d_r1_e,          all_bytes_full,  "flush all r1_e");
  d2h(r1_o_host,          S.d_r1_o,          all_bytes_full,  "flush all r1_o");
  d2h(z1_e_host,          S.d_z1_e,          all_bytes_full,  "flush all z1_e");
  d2h(z1_o_host,          S.d_z1_o,          all_bytes_full,  "flush all z1_o");
  d2h(ru_e_host,          S.d_ru_e,          all_bytes_full,  "flush all ru_e");
  d2h(ru_o_host,          S.d_ru_o,          all_bytes_full,  "flush all ru_o");
  d2h(zu_e_host,          S.d_zu_e,          all_bytes_full,  "flush all zu_e");
  d2h(zu_o_host,          S.d_zu_o,          all_bytes_full,  "flush all zu_o");
  if (s.lthreed) {
    d2h(rv_e_host,        S.d_rv_e,          all_bytes_full,  "flush all rv_e");
    d2h(rv_o_host,        S.d_rv_o,          all_bytes_full,  "flush all rv_o");
    d2h(zv_e_host,        S.d_zv_e,          all_bytes_full,  "flush all zv_e");
    d2h(zv_o_host,        S.d_zv_o,          all_bytes_full,  "flush all zv_o");
  }
  d2h(ruFull_host,        S.d_ruFull,        all_bytes_force, "flush all ruFull");
  d2h(zuFull_host,        S.d_zuFull,        all_bytes_force, "flush all zuFull");

  // force-local
  d2h(blmn_e_host,        S.d_blmn_e,        all_bytes_force, "flush all blmn_e");

  // radial half-grid profiles
  d2h(presH_host,         S.d_presH,         all_bytes_presH, "flush all presH");
  d2h(dVdsH_host,         S.d_dVdsH,         all_bytes_presH, "flush all dVdsH");
  d2h(bvcoH_host,         S.d_bvcoH,         all_bytes_presH, "flush all bvcoH");
  d2h(bucoH_host,         S.d_bucoH,         all_bytes_presH, "flush all bucoH");
  d2h(chipH_host,         S.d_chipH,         all_bytes_presH, "flush all chipH");
  d2h(iotaH_host,         S.d_iotaH,         all_bytes_presH, "flush all iotaH");
  d2h(chipF_host,         S.d_chipF,         all_bytes_chipF, "flush all chipF");
  d2h(iotaF_host,         S.d_iotaF,         all_bytes_chipF, "flush all iotaF");

  if (nsi > 0) {
    d2h(jcurvF_host,      S.d_jcurvF,        all_bytes_nsi,   "flush all jcurvF");
    d2h(jcuruF_host,      S.d_jcuruF,        all_bytes_nsi,   "flush all jcuruF");
    d2h(presgradF_host,   S.d_presgradF,     all_bytes_nsi,   "flush all presgradF");
    d2h(dVdsF_host,       S.d_dVdsF,         all_bytes_nsi,   "flush all dVdsF");
    d2h(equiF_host,       S.d_equiF,         all_bytes_nsi,   "flush all equiF");
  }

  // Converged spectra (d_pts_x_*): the boundary-update results
  // VMEC produced for each cfg. Layout per cfg: ns_local × mpol × (ntor+1).
  d2h(pts_x_rcc_host,     S.d_pts_x_rcc,     all_bytes_pts,   "flush all pts_x rcc");
  d2h(pts_x_zsc_host,     S.d_pts_x_zsc,     all_bytes_pts,   "flush all pts_x zsc");
  d2h(pts_x_lsc_host,     S.d_pts_x_lsc,     all_bytes_pts,   "flush all pts_x lsc");
  if (s.lthreed) {
    d2h(pts_x_rss_host,   S.d_pts_x_rss,     all_bytes_pts,   "flush all pts_x rss");
    d2h(pts_x_zcs_host,   S.d_pts_x_zcs,     all_bytes_pts,   "flush all pts_x zcs");
    d2h(pts_x_lcs_host,   S.d_pts_x_lcs,     all_bytes_pts,   "flush all pts_x lcs");
  }

  cuda_check(cudaStreamSynchronize(st), "flush all configs sync");
}

// ============================================================================
// Device-side physical_x_backup mirror + device rzNorm. Together these let
// PerformTimeStepCuda drop its per-iter D2H of d_pts_x → host m_decomposed_x
// plus the trailing cudaStreamSynchronize. The host backup mechanism (in
// Vmec::RestartIteration) is replaced by device-to-device copies that keep
// the rollback semantics under CUDA. Bit-exact match to CPU rzNorm is
// preserved by per-jF partial accumulation in CPU's nested-loop order.
// ============================================================================

// (anonymous namespace hoisted across the split translation units)

// Lazy alloc of d_pts_x_backup_* (one-time per shape). Initializes the
// backup mirror to the current d_pts_x contents so the first NO_RESTART
// save is a no-op (mirrors the host pattern where physical_x_backup is
// implicitly synced to decomposed_x at multi-grid step transitions).
void EnsurePTSBackupBuffers(CudaToroidalState& S) {
  if (S.pts_x_backup_initialized) return;
  if (!S.d_pts_x_rcc || S.pts_x_size <= 0) return;
  size_t x_bytes_all = sizeof(double) * (size_t)S.n_config_max *
                        (size_t)S.pts_x_size;
  auto alloc = [&](double*& p, const char* name) {
    if (!p) cuda_check(cudaMalloc(&p, x_bytes_all), name);
  };
  alloc(S.d_pts_x_backup_rcc, "alloc d_pts_x_backup_rcc");
  alloc(S.d_pts_x_backup_rss, "alloc d_pts_x_backup_rss");
  alloc(S.d_pts_x_backup_zsc, "alloc d_pts_x_backup_zsc");
  alloc(S.d_pts_x_backup_zcs, "alloc d_pts_x_backup_zcs");
  alloc(S.d_pts_x_backup_lsc, "alloc d_pts_x_backup_lsc");
  alloc(S.d_pts_x_backup_lcs, "alloc d_pts_x_backup_lcs");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_backup_rcc, S.d_pts_x_rcc,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "init backup rcc");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_backup_rss, S.d_pts_x_rss,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "init backup rss");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_backup_zsc, S.d_pts_x_zsc,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "init backup zsc");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_backup_zcs, S.d_pts_x_zcs,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "init backup zcs");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_backup_lsc, S.d_pts_x_lsc,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "init backup lsc");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_backup_lcs, S.d_pts_x_lcs,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "init backup lcs");
  S.pts_x_backup_initialized = true;
}

// Per-cfg restart kernels: copy d_pts_x_backup_* → d_pts_x_* and zero
// d_pts_v_* only for cfgs whose mask byte is non-zero. Per (cfg, idx) thread.
__global__ void k_restore_pts_x_per_cfg(
    int n_cfg, int pts_x_size,
    const std::uint8_t* __restrict__ mask,
    double* __restrict__ x_rcc, double* __restrict__ x_rss,
    double* __restrict__ x_zsc, double* __restrict__ x_zcs,
    double* __restrict__ x_lsc, double* __restrict__ x_lcs,
    const double* __restrict__ bx_rcc, const double* __restrict__ bx_rss,
    const double* __restrict__ bx_zsc, const double* __restrict__ bx_zcs,
    const double* __restrict__ bx_lsc, const double* __restrict__ bx_lcs) {
  int cfg = blockIdx.y;
  if (cfg >= n_cfg || mask[cfg] == 0) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pts_x_size) return;
  size_t off = (size_t)cfg * (size_t)pts_x_size + (size_t)i;
  x_rcc[off] = bx_rcc[off];
  x_rss[off] = bx_rss[off];
  x_zsc[off] = bx_zsc[off];
  x_zcs[off] = bx_zcs[off];
  x_lsc[off] = bx_lsc[off];
  x_lcs[off] = bx_lcs[off];
}

__global__ void k_zero_pts_v_per_cfg(
    int n_cfg, int pts_v_size,
    const std::uint8_t* __restrict__ mask,
    double* __restrict__ v_rcc, double* __restrict__ v_rss,
    double* __restrict__ v_zsc, double* __restrict__ v_zcs,
    double* __restrict__ v_lsc, double* __restrict__ v_lcs) {
  int cfg = blockIdx.y;
  if (cfg >= n_cfg || mask[cfg] == 0) return;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pts_v_size) return;
  size_t off = (size_t)cfg * (size_t)pts_v_size + (size_t)i;
  v_rcc[off] = 0.0;
  v_rss[off] = 0.0;
  v_zsc[off] = 0.0;
  v_zcs[off] = 0.0;
  v_lsc[off] = 0.0;
  v_lcs[off] = 0.0;
}

// Fused backup copy: all six spectral components of every configuration
// slot in one launch, so the per-improving-iteration backup costs one
// kernel dispatch instead of six memcpy enqueues.
__global__ void k_backup_pts_x(
    int total,
    const double* __restrict__ x_rcc, const double* __restrict__ x_rss,
    const double* __restrict__ x_zsc, const double* __restrict__ x_zcs,
    const double* __restrict__ x_lsc, const double* __restrict__ x_lcs,
    double* __restrict__ bx_rcc, double* __restrict__ bx_rss,
    double* __restrict__ bx_zsc, double* __restrict__ bx_zcs,
    double* __restrict__ bx_lsc, double* __restrict__ bx_lcs) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  bx_rcc[i] = x_rcc[i];
  bx_rss[i] = x_rss[i];
  bx_zsc[i] = x_zsc[i];
  bx_zcs[i] = x_zcs[i];
  bx_lsc[i] = x_lsc[i];
  bx_lcs[i] = x_lcs[i];
}

// (anonymous namespace hoisted across the split translation units)

void BackupPtsXCuda() {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_rcc) return;
  std::lock_guard<std::mutex> lk(S.mu);
  // Nothing to back up until PerformTimeStepCuda has done its first init H2D.
  // RestartIteration is called from InitializeRadial and from the
  // SolveEqLoop BAD_JACOBIAN block BEFORE the first
  // PerformTimeStepCuda call, when d_pts_x still holds cudaMalloc zeros.
  // Skipping here keeps the backup buffer authoritative ("last good state").
  if (!S.pts_x_initialized) return;
  EnsurePTSBackupBuffers(S);
  const int total =
      static_cast<int>((size_t)S.n_config_max * (size_t)S.pts_x_size);
  const int TPB = 256;
  S.TKBegin(CudaToroidalState::TK_BACKUP_PTS);
  k_backup_pts_x<<<(total + TPB - 1) / TPB, TPB, 0, S.stream>>>(
      total,
      S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
      S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs,
      S.d_pts_x_backup_rcc, S.d_pts_x_backup_rss, S.d_pts_x_backup_zsc,
      S.d_pts_x_backup_zcs, S.d_pts_x_backup_lsc, S.d_pts_x_backup_lcs);
  S.TKEnd(CudaToroidalState::TK_BACKUP_PTS);
  cuda_check(cudaGetLastError(), "k_backup_pts_x launch");
}

void RestorePtsXFromBackupCuda() {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_rcc) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.pts_x_backup_initialized) return;
  size_t x_bytes_all = sizeof(double) * (size_t)S.n_config_max *
                        (size_t)S.pts_x_size;
  cuda_check(cudaMemcpyAsync(S.d_pts_x_rcc, S.d_pts_x_backup_rcc,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "restore rcc");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_rss, S.d_pts_x_backup_rss,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "restore rss");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_zsc, S.d_pts_x_backup_zsc,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "restore zsc");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_zcs, S.d_pts_x_backup_zcs,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "restore zcs");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_lsc, S.d_pts_x_backup_lsc,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "restore lsc");
  cuda_check(cudaMemcpyAsync(S.d_pts_x_lcs, S.d_pts_x_backup_lcs,
                              x_bytes_all, cudaMemcpyDeviceToDevice, S.stream),
             "restore lcs");
  // Mirror host decomposed_v.setZero(): zero d_pts_v across all cfgs.
  if (S.d_pts_v_rcc && S.pts_v_size > 0) {
    size_t v_bytes_all = sizeof(double) * (size_t)S.n_config_max *
                          (size_t)S.pts_v_size;
    cuda_check(cudaMemsetAsync(S.d_pts_v_rcc, 0, v_bytes_all, S.stream), "zero v rcc");
    cuda_check(cudaMemsetAsync(S.d_pts_v_rss, 0, v_bytes_all, S.stream), "zero v rss");
    cuda_check(cudaMemsetAsync(S.d_pts_v_zsc, 0, v_bytes_all, S.stream), "zero v zsc");
    cuda_check(cudaMemsetAsync(S.d_pts_v_zcs, 0, v_bytes_all, S.stream), "zero v zcs");
    cuda_check(cudaMemsetAsync(S.d_pts_v_lsc, 0, v_bytes_all, S.stream), "zero v lsc");
    cuda_check(cudaMemsetAsync(S.d_pts_v_lcs, 0, v_bytes_all, S.stream), "zero v lcs");
  }
}

// Per-cfg variant of RestorePtsXFromBackupCuda. The mask is a host
// std::vector<uint8_t> of size n_config_max; cfg c is restored iff mask[c]!=0.
// Used by vmec.cc::RestartIteration to avoid rolling back cfgs whose
// fc_.restart_reason_per_cfg is NO_RESTART. Whole-batch behavior is
// recovered by passing a mask of all 1's.
void RestorePtsXFromBackupPerCfgCuda(const std::vector<std::uint8_t>& mask) {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_rcc) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.pts_x_backup_initialized) return;
  if (static_cast<int>(mask.size()) != S.n_config_max) return;
  // Quick scan: if no cfg requests restore, skip the launch entirely.
  bool any_restore = false;
  for (std::uint8_t b : mask) { if (b) { any_restore = true; break; } }
  if (!any_restore) return;
  S.EnsureRestartMaskBuffer();
  cuda_check(cudaMemcpyAsync(S.d_restart_mask, mask.data(),
                              sizeof(std::uint8_t) * S.n_config_max,
                              cudaMemcpyHostToDevice, S.stream),
             "h2d restart mask");
  const int TPB = 256;
  dim3 grid_x((S.pts_x_size + TPB - 1) / TPB, S.n_config_max, 1);
  dim3 tpb(TPB, 1, 1);
  k_restore_pts_x_per_cfg<<<grid_x, tpb, 0, S.stream>>>(
      S.n_config_max, S.pts_x_size, S.d_restart_mask,
      S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
      S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs,
      S.d_pts_x_backup_rcc, S.d_pts_x_backup_rss, S.d_pts_x_backup_zsc,
      S.d_pts_x_backup_zcs, S.d_pts_x_backup_lsc, S.d_pts_x_backup_lcs);
  cuda_check(cudaGetLastError(), "k_restore_pts_x_per_cfg launch");
  if (S.d_pts_v_rcc && S.pts_v_size > 0) {
    dim3 grid_v((S.pts_v_size + TPB - 1) / TPB, S.n_config_max, 1);
    k_zero_pts_v_per_cfg<<<grid_v, tpb, 0, S.stream>>>(
        S.n_config_max, S.pts_v_size, S.d_restart_mask,
        S.d_pts_v_rcc, S.d_pts_v_rss, S.d_pts_v_zsc,
        S.d_pts_v_zcs, S.d_pts_v_lsc, S.d_pts_v_lcs);
    cuda_check(cudaGetLastError(), "k_zero_pts_v_per_cfg launch");
  }
}

// Invalidates the device-resident decomposed-position state so the next
// stage-preparation or time-step call re-stages it from the host
// m_decomposed_x. The iteration-1 recovery path re-interpolates the host
// state from the boundary and the recomputed magnetic axis; every
// device-side initialization is gated on pts_x_initialized, which the
// failed attempt left set, so without this call the retry replays the
// failed attempt's device copy. The velocity state and the restart
// backup are invalidated along with the position so the retry rebuilds
// all three from the fresh host state.
void InvalidatePtsXCuda() {
  auto& S = State();
  if (!S.stream) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.pts_x_initialized = false;
  S.pts_v_initialized = false;
  S.pts_x_backup_initialized = false;
}

void FlushDecomposedXToHostCuda(
    int cfg, int ns_local, int mpol, int ntor, bool lthreed,
    double* m_dec_x_rcc, double* m_dec_x_rss,
    double* m_dec_x_zsc, double* m_dec_x_zcs,
    double* m_dec_x_lsc, double* m_dec_x_lcs) {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_rcc) return;
  std::lock_guard<std::mutex> lk(S.mu);
  size_t x_bytes_one = sizeof(double) * (size_t)ns_local * (size_t)mpol *
                        (size_t)(ntor + 1);
  // Multigrid stage transition guard: when ns_local changes between stages
  // EnsurePTSBuffers has not yet been called for the new stage; S.pts_x_size
  // is still the OLD stage's per-config element count. Skip rather than risk
  // an out-of-bounds D2H. Host m_decomposed_x is freshly populated by
  // interpFromBoundaryAndAxis at that point anyway.
  if ((size_t)ns_local * (size_t)mpol * (size_t)(ntor + 1)
      != (size_t)S.pts_x_size) {
    return;
  }
  const size_t off = (size_t)cfg * (size_t)S.pts_x_size;
  cuda_check(cudaMemcpyAsync(m_dec_x_rcc, S.d_pts_x_rcc + off, x_bytes_one,
                              cudaMemcpyDeviceToHost, S.stream),
             "flush dec_x rcc");
  cuda_check(cudaMemcpyAsync(m_dec_x_zsc, S.d_pts_x_zsc + off, x_bytes_one,
                              cudaMemcpyDeviceToHost, S.stream),
             "flush dec_x zsc");
  cuda_check(cudaMemcpyAsync(m_dec_x_lsc, S.d_pts_x_lsc + off, x_bytes_one,
                              cudaMemcpyDeviceToHost, S.stream),
             "flush dec_x lsc");
  if (lthreed) {
    cuda_check(cudaMemcpyAsync(m_dec_x_rss, S.d_pts_x_rss + off, x_bytes_one,
                                cudaMemcpyDeviceToHost, S.stream),
               "flush dec_x rss");
    cuda_check(cudaMemcpyAsync(m_dec_x_zcs, S.d_pts_x_zcs + off, x_bytes_one,
                                cudaMemcpyDeviceToHost, S.stream),
               "flush dec_x zcs");
    cuda_check(cudaMemcpyAsync(m_dec_x_lcs, S.d_pts_x_lcs + off, x_bytes_one,
                                cudaMemcpyDeviceToHost, S.stream),
               "flush dec_x lcs");
  }
  cuda_check(cudaStreamSynchronize(S.stream), "flush dec_x sync");
}

// DumpPtsXAllCfgsCuda: write the full batched decomposed-x state (every
// configuration slot, six spectral components) to a raw binary file.
// Layout: 4 int64 header (n_config_max, pts_x_size, iter, n_components=6)
// followed by six contiguous arrays of n_config_max*pts_x_size doubles in
// the order rcc, rss, zsc, zcs, lsc, lcs (zeros for absent components).
// Diagnostic for cross-cfg contamination A/B runs; see the
// VMECPP_STATE_DUMP_ITERS hook in vmec.cc.
void DumpPtsXAllCfgsCuda(const char* path, long long iter) {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_rcc || S.pts_x_size <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  size_t n_per = (size_t)S.n_config_max * (size_t)S.pts_x_size;
  std::vector<double> h(n_per * 6, 0.0);
  const double* srcs[6] = {S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
                            S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs};
  for (int i = 0; i < 6; ++i) {
    if (srcs[i] == nullptr) continue;
    cuda_check(cudaMemcpyAsync(h.data() + (size_t)i * n_per, srcs[i],
                                sizeof(double) * n_per,
                                cudaMemcpyDeviceToHost, S.stream),
               "dump pts_x d2h");
  }
  cuda_check(cudaStreamSynchronize(S.stream), "dump pts_x sync");
  FILE* f = std::fopen(path, "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "[fft_toroidal_cuda] state dump: fopen failed: %s\n",
                 path);
    return;
  }
  long long hdr[4] = {(long long)S.n_config_max, (long long)S.pts_x_size,
                      iter, 6};
  std::fwrite(hdr, sizeof(long long), 4, f);
  std::fwrite(h.data(), sizeof(double), h.size(), f);
  std::fclose(f);
  std::fprintf(stderr,
      "[fft_toroidal_cuda] state dump: iter=%lld n_cfg=%d pts_x_size=%d -> %s\n",
      iter, S.n_config_max, S.pts_x_size, path);
}

// DumpDecomposedFAllCfgsCuda: same layout as DumpPtsXAllCfgsCuda but for
// the decomposed-forces buffers (frcc, frss, fzsc, fzcs, flsc, flcs) at
// per-cfg stride pts_v_size. Captures the forces of the most recent
// iteration; with the state dump it splits "physics inputs differ" from
// "controller decisions differ" in cross-cfg contamination A/B runs.
void DumpDecomposedFAllCfgsCuda(const char* path, long long iter) {
  auto& S = State();
  if (!S.stream || !S.d_decomposed_frcc || S.pts_v_size <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  size_t n_per = (size_t)S.n_config_max * (size_t)S.pts_v_size;
  std::vector<double> h(n_per * 6, 0.0);
  const double* srcs[6] = {S.d_decomposed_frcc, S.d_decomposed_frss,
                            S.d_decomposed_fzsc, S.d_decomposed_fzcs,
                            S.d_decomposed_flsc, S.d_decomposed_flcs};
  for (int i = 0; i < 6; ++i) {
    if (srcs[i] == nullptr) continue;
    cuda_check(cudaMemcpyAsync(h.data() + (size_t)i * n_per, srcs[i],
                                sizeof(double) * n_per,
                                cudaMemcpyDeviceToHost, S.stream),
               "dump dec_f d2h");
  }
  cuda_check(cudaStreamSynchronize(S.stream), "dump dec_f sync");
  FILE* f = std::fopen(path, "wb");
  if (f == nullptr) {
    std::fprintf(stderr, "[fft_toroidal_cuda] force dump: fopen failed: %s\n",
                 path);
    return;
  }
  long long hdr[4] = {(long long)S.n_config_max, (long long)S.pts_v_size,
                      iter, 6};
  std::fwrite(hdr, sizeof(long long), 4, f);
  std::fwrite(h.data(), sizeof(double), h.size(), f);
  std::fclose(f);
  std::fprintf(stderr,
      "[fft_toroidal_cuda] force dump: iter=%lld n_cfg=%d pts_v_size=%d -> %s\n",
      iter, S.n_config_max, S.pts_v_size, path);
}

// DumpBContraProfilesAllCfgsCuda: per-cfg half-grid radial profiles from
// the UpdateBContravariant chain (chipH, iotaH, jvPlasma, avg_guu_gsqrt),
// stride ns_h per cfg. Same header convention as the other dumpers with
// n_components=4.
void DumpBContraProfilesAllCfgsCuda(const char* path, long long iter,
                                    int ns_h) {
  auto& S = State();
  if (!S.stream || !S.d_chipH || ns_h <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  size_t n_per = (size_t)S.n_config_max * (size_t)ns_h;
  std::vector<double> h(n_per * 4, 0.0);
  const double* srcs[4] = {S.d_chipH, S.d_iotaH, S.d_jvPlasma,
                            S.d_avg_guu_gsqrt};
  for (int i = 0; i < 4; ++i) {
    if (srcs[i] == nullptr) continue;
    cuda_check(cudaMemcpyAsync(h.data() + (size_t)i * n_per, srcs[i],
                                sizeof(double) * n_per,
                                cudaMemcpyDeviceToHost, S.stream),
               "dump bcontra prof d2h");
  }
  cuda_check(cudaStreamSynchronize(S.stream), "dump bcontra prof sync");
  FILE* f = std::fopen(path, "wb");
  if (f == nullptr) return;
  long long hdr[4] = {(long long)S.n_config_max, (long long)ns_h, iter, 4};
  std::fwrite(hdr, sizeof(long long), 4, f);
  std::fwrite(h.data(), sizeof(double), h.size(), f);
  std::fclose(f);
  std::fprintf(stderr,
      "[fft_toroidal_cuda] bcontra prof dump: iter=%lld n_cfg=%d ns_h=%d -> %s\n",
      iter, S.n_config_max, ns_h, path);
}

}  // namespace vmecpp

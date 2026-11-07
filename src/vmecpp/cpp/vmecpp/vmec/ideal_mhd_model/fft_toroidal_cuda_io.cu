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

// FlushForOutputQuantitiesCuda: copies the device half-grid fields to the
// IdealMhdModel members that GatherDataFromThreads reads.
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
    double* bucoH_host,
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
  d2h(bucoH_host,          S.d_bucoH,          presH_bytes, "flush bucoH");

  // chipH, iotaH (ns_h doubles each), chipF, iotaF (ns_local doubles each);
  // under CUDA their only host reader is the output computation.
  d2h(chipH_host,          S.d_chipH,          presH_bytes, "flush chipH");
  d2h(iotaH_host,          S.d_iotaH,          presH_bytes, "flush iotaH");
  const size_t chipFull_bytes = sizeof(double) * ns_local;
  d2h(chipF_host,          S.d_chipF,          chipFull_bytes, "flush chipF");
  d2h(iotaF_host,          S.d_iotaF,          chipFull_bytes, "flush iotaF");

  // Radial interior nsi = nsMaxFi - nsMinFi, indexed alike on the device and
  // the host.
  if (nsi > 0) {
    d2h(jcurvF_host,       S.d_jcurvF,         nsi_bytes, "flush jcurvF");
    d2h(jcuruF_host,       S.d_jcuruF,         nsi_bytes, "flush jcuruF");
    d2h(presgradF_host,    S.d_presgradF,      nsi_bytes, "flush presgradF");
    d2h(dVdsF_host,        S.d_dVdsF,          nsi_bytes, "flush dVdsF");
    d2h(equiF_host,        S.d_equiF,          nsi_bytes, "flush equiF");
  }

  // (Per-config batched flush is provided by FlushAllConfigsForOutputCuda;
  //  see below.)

  // A multi-configuration run keeps each configuration's final spectra,
  // [sp][cfg][spec], for GetBatchOutputSpectraCuda.
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
    // Each spectrum of all configurations, from the converged-state snapshot of
    // a configuration that went inactive.
    double* d_specs[6] = {S.d_pts_x_rcc, S.d_pts_x_rss,
                          S.d_pts_x_zsc, S.d_pts_x_zcs,
                          S.d_pts_x_lsc, S.d_pts_x_lcs};
    double* d_finals[6] = {S.d_pts_x_final_rcc, S.d_pts_x_final_rss,
                           S.d_pts_x_final_zsc, S.d_pts_x_final_zcs,
                           S.d_pts_x_final_lsc, S.d_pts_x_final_lcs};
    for (int sp = 0; sp < 6; ++sp) {
      for (int cfg = 0; cfg < N_out; ++cfg) {
        const bool use_snap =
            S.d_pts_x_final_rcc &&
            cfg < static_cast<int>(S.pts_x_final_taken.size()) &&
            S.pts_x_final_taken[cfg];
        const double* src =
            (use_snap ? d_finals[sp] : d_specs[sp]) +
            (size_t)cfg * per_spec_doubles;
        cuda_check(cudaMemcpyAsync(
            h_buf + ((size_t)sp * N_out + cfg) * per_spec_doubles, src,
            sizeof(double) * per_spec_doubles,
            cudaMemcpyDeviceToHost, st), "batch out D2H");
      }
    }
    cuda_check(cudaStreamSynchronize(st), "batch out sync");
    g_batch_outputs_mem.assign(h_buf, h_buf + total_doubles);
    g_batch_outputs_shape[0] = N_out;
    g_batch_outputs_shape[1] = ns_out;
    g_batch_outputs_shape[2] = mpol_out;
    g_batch_outputs_shape[3] = ntor_out;
    cudaFreeHost(h_buf);
  }

  cuda_check(cudaStreamSynchronize(st), "flush output_quantities sync");
}

// FlushAllConfigsForOutputCuda: FlushForOutputQuantitiesCuda for all
// configurations, each host array configuration-major.
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
  // Single-rank extents from ns, without the interior full-grid arrays.
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
  // VMEC produced for each cfg. Layout per cfg: ns_local x mpol x (ntor+1).
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

// Device backup of the positions and device rzNorm, so the time step needs no
// per-iteration host copy; rzNorm matches the CPU bit for bit.

// (anonymous namespace hoisted across the split translation units)

// Allocates the backup once per shape, initialized to the current positions as
// the host backup is synced at stage transitions.
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

// Per-cfg restart kernels: copy d_pts_x_backup_* -> d_pts_x_* and zero
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

// Fused backup copy of the six components of every configuration slot in one
// launch.
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

void SnapshotEvaluatedPtsXCuda() {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_rcc) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.pts_x_initialized) return;
  size_t x_bytes_all = sizeof(double) * (size_t)S.n_config_max *
                       (size_t)S.pts_x_size;
  auto alloc = [&](double*& p, const char* name) {
    if (!p) cuda_check(cudaMalloc(&p, x_bytes_all), name);
  };
  alloc(S.d_pts_x_evaluated_rcc, "alloc d_pts_x_evaluated_rcc");
  alloc(S.d_pts_x_evaluated_rss, "alloc d_pts_x_evaluated_rss");
  alloc(S.d_pts_x_evaluated_zsc, "alloc d_pts_x_evaluated_zsc");
  alloc(S.d_pts_x_evaluated_zcs, "alloc d_pts_x_evaluated_zcs");
  alloc(S.d_pts_x_evaluated_lsc, "alloc d_pts_x_evaluated_lsc");
  alloc(S.d_pts_x_evaluated_lcs, "alloc d_pts_x_evaluated_lcs");
  const int total =
      static_cast<int>((size_t)S.n_config_max * (size_t)S.pts_x_size);
  const int TPB = 256;
  k_backup_pts_x<<<(total + TPB - 1) / TPB, TPB, 0, S.stream>>>(
      total,
      S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
      S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs,
      S.d_pts_x_evaluated_rcc, S.d_pts_x_evaluated_rss,
      S.d_pts_x_evaluated_zsc, S.d_pts_x_evaluated_zcs,
      S.d_pts_x_evaluated_lsc, S.d_pts_x_evaluated_lcs);
  cuda_check(cudaGetLastError(), "k_backup_pts_x (evaluated) launch");
  S.pts_x_evaluated_valid = true;
}

void SnapshotEvaluatedPtsXPerCfgCuda(const std::vector<std::uint8_t>& mask) {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_rcc) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.pts_x_initialized) return;
  if (static_cast<int>(mask.size()) != S.n_config_max) return;
  if (std::find(mask.begin(), mask.end(), 1) == mask.end()) return;
  size_t x_bytes_all = sizeof(double) * (size_t)S.n_config_max *
                       (size_t)S.pts_x_size;
  auto alloc = [&](double*& p, const char* name) {
    if (!p) cuda_check(cudaMalloc(&p, x_bytes_all), name);
  };
  alloc(S.d_pts_x_evaluated_rcc, "alloc d_pts_x_evaluated_rcc");
  alloc(S.d_pts_x_evaluated_rss, "alloc d_pts_x_evaluated_rss");
  alloc(S.d_pts_x_evaluated_zsc, "alloc d_pts_x_evaluated_zsc");
  alloc(S.d_pts_x_evaluated_zcs, "alloc d_pts_x_evaluated_zcs");
  alloc(S.d_pts_x_evaluated_lsc, "alloc d_pts_x_evaluated_lsc");
  alloc(S.d_pts_x_evaluated_lcs, "alloc d_pts_x_evaluated_lcs");
  S.EnsurePerCfgControlBuffers();
  cuda_check(cudaMemcpyAsync(S.d_copy_mask, mask.data(),
                              sizeof(std::uint8_t) * S.n_config_max,
                              cudaMemcpyHostToDevice, S.stream),
             "h2d snapshot mask");
  const int TPB = 256;
  dim3 grid((S.pts_x_size + TPB - 1) / TPB, S.n_config_max, 1);
  // k_restore_pts_x_per_cfg copies its second set of arrays into its first
  // for the masked configurations.
  k_restore_pts_x_per_cfg<<<grid, TPB, 0, S.stream>>>(
      S.n_config_max, S.pts_x_size, S.d_copy_mask,
      S.d_pts_x_evaluated_rcc, S.d_pts_x_evaluated_rss,
      S.d_pts_x_evaluated_zsc, S.d_pts_x_evaluated_zcs,
      S.d_pts_x_evaluated_lsc, S.d_pts_x_evaluated_lcs,
      S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
      S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs);
  cuda_check(cudaGetLastError(), "k_restore_pts_x_per_cfg (snapshot) launch");
  S.pts_x_evaluated_valid = true;
}

void BackupPtsXPerCfgCuda(const std::vector<std::uint8_t>& mask,
                          bool from_evaluated) {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_rcc) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.pts_x_initialized) return;
  if (static_cast<int>(mask.size()) != S.n_config_max) return;
  if (std::find(mask.begin(), mask.end(), 1) == mask.end()) return;
  EnsurePTSBackupBuffers(S);
  const bool evaluated = from_evaluated && S.pts_x_evaluated_valid;
  S.EnsurePerCfgControlBuffers();
  cuda_check(cudaMemcpyAsync(S.d_copy_mask, mask.data(),
                              sizeof(std::uint8_t) * S.n_config_max,
                              cudaMemcpyHostToDevice, S.stream),
             "h2d backup mask");
  const int TPB = 256;
  dim3 grid((S.pts_x_size + TPB - 1) / TPB, S.n_config_max, 1);
  k_restore_pts_x_per_cfg<<<grid, TPB, 0, S.stream>>>(
      S.n_config_max, S.pts_x_size, S.d_copy_mask,
      S.d_pts_x_backup_rcc, S.d_pts_x_backup_rss, S.d_pts_x_backup_zsc,
      S.d_pts_x_backup_zcs, S.d_pts_x_backup_lsc, S.d_pts_x_backup_lcs,
      evaluated ? S.d_pts_x_evaluated_rcc : S.d_pts_x_rcc,
      evaluated ? S.d_pts_x_evaluated_rss : S.d_pts_x_rss,
      evaluated ? S.d_pts_x_evaluated_zsc : S.d_pts_x_zsc,
      evaluated ? S.d_pts_x_evaluated_zcs : S.d_pts_x_zcs,
      evaluated ? S.d_pts_x_evaluated_lsc : S.d_pts_x_lsc,
      evaluated ? S.d_pts_x_evaluated_lcs : S.d_pts_x_lcs);
  cuda_check(cudaGetLastError(), "k_restore_pts_x_per_cfg (backup) launch");
}

void BackupPtsXCuda(bool from_evaluated) {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_rcc) return;
  std::lock_guard<std::mutex> lk(S.mu);
  // Nothing to back up before the first PerformTimeStepCuda loads d_pts_x.
  if (!S.pts_x_initialized) return;
  EnsurePTSBackupBuffers(S);
  // backup_evaluated_state: the state of the latest force evaluation, which
  // d_pts_x has already been advanced past.
  const bool evaluated = from_evaluated && S.pts_x_evaluated_valid;
  const int total =
      static_cast<int>((size_t)S.n_config_max * (size_t)S.pts_x_size);
  const int TPB = 256;
  k_backup_pts_x<<<(total + TPB - 1) / TPB, TPB, 0, S.stream>>>(
      total,
      evaluated ? S.d_pts_x_evaluated_rcc : S.d_pts_x_rcc,
      evaluated ? S.d_pts_x_evaluated_rss : S.d_pts_x_rss,
      evaluated ? S.d_pts_x_evaluated_zsc : S.d_pts_x_zsc,
      evaluated ? S.d_pts_x_evaluated_zcs : S.d_pts_x_zcs,
      evaluated ? S.d_pts_x_evaluated_lsc : S.d_pts_x_lsc,
      evaluated ? S.d_pts_x_evaluated_lcs : S.d_pts_x_lcs,
      S.d_pts_x_backup_rcc, S.d_pts_x_backup_rss, S.d_pts_x_backup_zsc,
      S.d_pts_x_backup_zcs, S.d_pts_x_backup_lsc, S.d_pts_x_backup_lcs);
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

// Restores the configurations whose mask byte is nonzero; a mask of ones
// restores the whole batch.
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

// Invalidates the device positions so the next call re-stages them from the
// host, as the axis-recovery retry needs.
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
  // Before this run's first time step the host copy is authoritative, as for a
  // hot restart that converges at its first evaluation.
  if (!S.pts_x_initialized) return;
  size_t x_bytes_one = sizeof(double) * (size_t)ns_local * (size_t)mpol *
                        (size_t)(ntor + 1);
  // At a stage transition before EnsurePTSBuffers, pts_x_size is the old
  // stage's, so the copy is skipped; the host is fresh then.
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

void FlushBackupXToHostCuda(int ns_local, int mpol, int ntor, bool lthreed,
                            double* rcc, double* rss, double* zsc, double* zcs,
                            double* lsc, double* lcs) {
  auto& S = State();
  if (!S.stream || !S.d_pts_x_backup_rcc) return;
  std::lock_guard<std::mutex> lk(S.mu);
  if (!S.pts_x_backup_initialized) return;
  if ((size_t)ns_local * (size_t)mpol * (size_t)(ntor + 1) !=
      (size_t)S.pts_x_size) {
    return;
  }
  const size_t bytes = sizeof(double) * (size_t)S.pts_x_size;
  auto d2h = [&](double* host, const double* dev, const char* name) {
    cuda_check(cudaMemcpyAsync(host, dev, bytes, cudaMemcpyDeviceToHost,
                                S.stream),
               name);
  };
  d2h(rcc, S.d_pts_x_backup_rcc, "flush backup rcc");
  d2h(zsc, S.d_pts_x_backup_zsc, "flush backup zsc");
  d2h(lsc, S.d_pts_x_backup_lsc, "flush backup lsc");
  if (lthreed) {
    d2h(rss, S.d_pts_x_backup_rss, "flush backup rss");
    d2h(zcs, S.d_pts_x_backup_zcs, "flush backup zcs");
    d2h(lcs, S.d_pts_x_backup_lcs, "flush backup lcs");
  }
  cuda_check(cudaStreamSynchronize(S.stream), "flush backup sync");
}

}  // namespace vmecpp

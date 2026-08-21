#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"

namespace vmecpp {

// Length-1 toroidal transforms (axisymmetric ntor=0, nZeta=1) are the identity:
// the inverse is the DC real part, the forward packs the real value as the DC
// with zero imaginary part. cuFFT's length-1 plans are bypassed with these.
namespace {
__global__ void k_z2d_identity_nzeta1(int n,
                                      const cufftDoubleComplex* __restrict__ X,
                                      double* __restrict__ Y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Y[i] = X[i].x;
}
__global__ void k_d2z_identity_nzeta1(int n, const double* __restrict__ Y,
                                      cufftDoubleComplex* __restrict__ X) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  X[i].x = Y[i];
  X[i].y = 0.0;
}
}  // namespace

// Warp-packed variant of k_scatter_main_and_con_v5 (kernels.cu). The v5 kernel
// assigns one warp to one (config, jF_local, k) tuple and uses lane = l for the
// poloidal axis, so 32 - nThetaReduced lanes are idle every cycle (16 of 32 on
// W7-X, half the FP64 throughput wasted). This variant packs ks_per_warp =
// blockDim.x / nThetaReduced zeta planes into each warp: lane splits into
// kk = lane / nThetaReduced (which zeta plane) and l = lane % nThetaReduced
// (poloidal point), so all 32 lanes do useful work. The shared Y tile is
// replicated ks_per_warp times per warp (one tile per packed zeta plane).
// Registers stay the occupancy limiter, so the extra shared memory is free.
// Gated by VMECPP_SCATTER_PACK; selected only when 32 is an exact multiple of
// nThetaReduced. Per-(jF, k, l) arithmetic is identical to v5.
__global__ __launch_bounds__(128, 5) void k_scatter_main_and_con_v5_packed(
    int n_config, int ns_local, int mpol, int nZeta, int nThetaReduced,
    int nThetaEff,
    const double* __restrict__ Y, const double* __restrict__ cosmu,
    const double* __restrict__ sinmu, const double* __restrict__ cosmum,
    const double* __restrict__ sinmum,
    const double* __restrict__ xmpq, const double* __restrict__ sqrtSF,
    double* __restrict__ r1_e, double* __restrict__ r1_o,
    double* __restrict__ ru_e, double* __restrict__ ru_o,
    double* __restrict__ rv_e, double* __restrict__ rv_o,
    double* __restrict__ z1_e, double* __restrict__ z1_o,
    double* __restrict__ zu_e, double* __restrict__ zu_o,
    double* __restrict__ zv_e, double* __restrict__ zv_o,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o,
    double* __restrict__ rCon, double* __restrict__ zCon) {
  int warp_id = threadIdx.y;
  int z_global = blockIdx.z * blockDim.y + warp_id;
  int config = z_global / ns_local;
  int jF_local = z_global - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int lane = threadIdx.x;
  int ks_per_warp = blockDim.x / nThetaReduced;  // 2 at nThetaReduced = 16

  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;

  extern __shared__ double s_Y_block[];  // [blockDim.y * ks_per_warp * tile]
  const int tile = mpol * kBatch;
  double* s_Y_warp = s_Y_block + (size_t)warp_id * (size_t)ks_per_warp * tile;

  // Cooperative load of all ks_per_warp tiles; 32 lanes stride over the slots.
  const int total_slots = ks_per_warp * tile;
  for (int t = lane; t < total_slots; t += 32) {
    int which_kk = t / tile;
    int within = t - which_kk * tile;
    int m_local = within / kBatch;
    int q_local = within - m_local * kBatch;
    int k_load = blockIdx.y * ks_per_warp + which_kk;
    double val = 0.0;
    if (k_load < nZeta) {
      size_t y_base_local = (size_t)((jF_local * mpol + m_local) * kBatch);
      val = Y[cfg_Y + (y_base_local + q_local) * nZeta + k_load];
    }
    s_Y_warp[t] = val;
  }
  __syncwarp();

  int kk = lane / nThetaReduced;
  int l = lane - kk * nThetaReduced;
  int k = blockIdx.y * ks_per_warp + kk;
  if (kk >= ks_per_warp || k >= nZeta || l >= nThetaReduced) return;
  const double* s_Y = s_Y_warp + (size_t)kk * tile;

  double r1e_acc = 0.0, r1o_acc = 0.0;
  double rue_acc = 0.0, ruo_acc = 0.0;
  double rve_acc = 0.0, rvo_acc = 0.0;
  double z1e_acc = 0.0, z1o_acc = 0.0;
  double zue_acc = 0.0, zuo_acc = 0.0;
  double zve_acc = 0.0, zvo_acc = 0.0;
  double lue_acc = 0.0, luo_acc = 0.0;
  double lve_acc = 0.0, lvo_acc = 0.0;
  double rcon_acc = 0.0, zcon_acc = 0.0;

  double sqrtSF_jF = sqrtSF[jF_local];

  #pragma unroll 10
  for (int m = 0; m < mpol; ++m) {
    double rmkcc  = s_Y[m * kBatch + kRmkcc];
    double rmkss  = s_Y[m * kBatch + kRmkss];
    double rmkccN = s_Y[m * kBatch + kRmkccN];
    double rmkssN = s_Y[m * kBatch + kRmkssN];
    double zmksc  = s_Y[m * kBatch + kZmksc];
    double zmkcs  = s_Y[m * kBatch + kZmkcs];
    double zmkscN = s_Y[m * kBatch + kZmkscN];
    double zmkcsN = s_Y[m * kBatch + kZmkcsN];
    double lmksc  = s_Y[m * kBatch + kLmksc];
    double lmkcs  = s_Y[m * kBatch + kLmkcs];
    double lmkscN = s_Y[m * kBatch + kLmkscN];
    double lmkcsN = s_Y[m * kBatch + kLmkcsN];

    int bml = m * nThetaReduced + l;
    double cmu  = cosmu[bml];
    double smu  = sinmu[bml];
    double cmum = cosmum[bml];
    double smum = sinmum[bml];
    bool m_even = ((m & 1) == 0);

    double r1_contrib = rmkcc * cmu + rmkss * smu;
    double ru_contrib = rmkcc * smum + rmkss * cmum;
    double rv_contrib = rmkccN * cmu + rmkssN * smu;
    double z1_contrib = zmksc * smu + zmkcs * cmu;
    double zu_contrib = zmksc * cmum + zmkcs * smum;
    double zv_contrib = zmkscN * smu + zmkcsN * cmu;
    double lu_contrib = lmksc * cmum + lmkcs * smum;
    double lv_contrib = -(lmkscN * smu + lmkcsN * cmu);
    if (m_even) {
      r1e_acc += r1_contrib; rue_acc += ru_contrib; rve_acc += rv_contrib;
      z1e_acc += z1_contrib; zue_acc += zu_contrib; zve_acc += zv_contrib;
      lue_acc += lu_contrib; lve_acc += lv_contrib;
    } else {
      r1o_acc += r1_contrib; ruo_acc += ru_contrib; rvo_acc += rv_contrib;
      z1o_acc += z1_contrib; zuo_acc += zu_contrib; zvo_acc += zv_contrib;
      luo_acc += lu_contrib; lvo_acc += lv_contrib;
    }
    double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF_jF;
    rcon_acc += r1_contrib * con_factor;
    zcon_acc += z1_contrib * con_factor;
  }

  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] = r1e_acc; r1_o[idx] = r1o_acc;
  ru_e[idx] = rue_acc; ru_o[idx] = ruo_acc;
  rv_e[idx] = rve_acc; rv_o[idx] = rvo_acc;
  z1_e[idx] = z1e_acc; z1_o[idx] = z1o_acc;
  zu_e[idx] = zue_acc; zu_o[idx] = zuo_acc;
  zv_e[idx] = zve_acc; zv_o[idx] = zvo_acc;
  lu_e[idx] = lue_acc; lu_o[idx] = luo_acc;
  lv_e[idx] = lve_acc; lv_o[idx] = lvo_acc;
  rCon[idx] = rcon_acc;
  zCon[idx] = zcon_acc;
}

void FourierToReal3DSymmFastPoloidalCuda(
    const FourierGeometry& physical_x, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb,
    RealSpaceGeometry& m_geometry) {
  auto& S = State();
  // Drop ToroidalFftPlans dependency: when VMECPP_USE_FFTX is off the type
  // does not exist. The CUDA path only needs (nZeta, nfp, mpol) which Sizes
  // already carries.
  S.OneTimeInit(s.nZeta, s.nfp, s.mpol);

  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  const int nhalf = s.nZeta / 2 + 1;
  const int nZeta = s.nZeta;
  const int nfp = s.nfp;
  const int nThetaReduced = s.nThetaReduced;
  const int nThetaEff = s.nThetaEff;

  // Diagnostic on first call only.
  static bool logged_shape = false;
  if (!logged_shape) {
    std::fprintf(stderr,
        "[fft_toroidal_cuda] FW shape: ns_local=%d ns_con_local=%d mpol=%d ntor=%d "
        "nhalf=%d nZeta=%d nThetaReduced=%d nThetaEff=%d nfp=%d nsMinF1=%d "
        "nsMinF=%d nsMaxF1=%d nsMaxFIncludingLcfs=%d xmpq.size=%d rp.sqrtSF.size=%d\n",
        ns_local, ns_con_local, mpol, ntor, nhalf, nZeta, nThetaReduced,
        nThetaEff, nfp, r.nsMinF1, r.nsMinF, r.nsMaxF1, r.nsMaxFIncludingLcfs,
        (int)xmpq.size(), (int)rp.sqrtSF.size());
    logged_shape = true;
  }

  if (ns_local <= 0) {
    // Nothing to do for this MPI rank's local range.
    return;
  }

  // Effective configuration count for this run, frozen at run start by
  // ResetCudaStateForNewVmecRun. Default 1 keeps the single-config path
  // bit-exact. At N>1, distinct inputs arrive through the run_batched_gpu
  // staging; otherwise the same input is broadcast to all N slots.
  const int n_cfg = GetNConfigMaxCuda();
  {
    static int logged_n_cfg = 1;
    if (n_cfg > 1 && n_cfg != logged_n_cfg) {
      logged_n_cfg = n_cfg;
      std::fprintf(stderr, "[fft_toroidal_cuda] batched mode active: n_config_max=%d "
                           "(broadcast input to all slots; only config 0 result "
                           "flows back to host)\n", n_cfg);
    }
  }

  // (Re)allocate device buffers if shape changed.
  std::lock_guard<std::mutex> lk(S.mu);
  if (S.ns_local_cached != ns_local || S.ns_con_local_cached != ns_con_local ||
      S.mpol_cached != mpol || S.ntor_cached != ntor ||
      S.nhalf_cached != nhalf || S.nZeta_cached != nZeta ||
      S.nThetaReduced_cached != nThetaReduced ||
      S.nThetaEff_cached != nThetaEff ||
      S.n_config_max != n_cfg) {
    S.Reshape(ns_local, ns_con_local, mpol, ntor, nhalf, nZeta, nThetaReduced,
              nThetaEff, n_cfg);
    S.StageBasis(nhalf, mpol, nThetaReduced, fb.nscale.data(), fb.cosmu.data(),
                 fb.sinmu.data(), fb.cosmum.data(), fb.sinmum.data());
    S.StageBasisI(mpol, nThetaReduced, fb.cosmui.data(), fb.sinmui.data(),
                    fb.cosmumi.data(), fb.sinmumi.data());
    S.StageToroidalBasis(nZeta, s.nnyq2 + 1, fb.cosnv.data(), fb.sinnv.data());
    S.StageDftBasis(ntor, nZeta, fb.nscale.data());
    S.StageInverseDftBasis(nhalf, nZeta);
    S.EnsureFourierForcesBuffers(ns_local, mpol, ntor);
  }

  cudaStream_t st = S.stream;

  // ----- Stage all specs + xmpq + sqrtSF into ONE pinned host buffer -----
  // At N=1 the layout is exactly as before (bit-exact). At N>1
  // each spectra block is N times bigger and we broadcast the same input to
  // all N slots within the block.
  size_t one_spec_bytes = sizeof(double) * ns_local * mpol * (ntor + 1);
  size_t one_spec_doubles = one_spec_bytes / sizeof(double);
  size_t block_doubles = (size_t)n_cfg * one_spec_doubles;
  double* h = S.h_specs_pinned;
  // The pinned host buffer h_specs_pinned holds six spectral-coefficient
  // blocks laid out contiguously, each block carrying all n_cfg
  // configurations end to end. When specs_populated_from_device is
  // asserted (set by RecomposeToPhysicalCuda upstream), the spec
  // sections of d_specs_block on the device have already been written
  // from the device-resident d_pts_x state, so the host-to-device
  // transfer of those sections that the conditional path below would
  // otherwise issue is unnecessary, and the host memcpy that would
  // populate the corresponding bytes of h_specs_pinned is similarly
  // unnecessary because the destination region is never read. The
  // host memcpy chain is therefore guarded by the negation of the
  // flag, and is bypassed once RecomposeToPhysicalCuda is the
  // authoritative producer of the spec sections, which is the case
  // from the second multigrid-stage iteration onward under CUDA.
  if (!S.specs_populated_from_device) {
    const double* src[6] = {physical_x.rmncc.data(), physical_x.rmnss.data(),
                            physical_x.zmnsc.data(), physical_x.zmncs.data(),
                            physical_x.lmnsc.data(), physical_x.lmncs.data()};
    // The VMECPP_BATCH_PERTURB environment variable provides an
    // opt-in perturbation knob for exercising the per-configuration
    // execution path with non-identical inputs. When the variable
    // names a non-zero scale, each configuration cfg's spectra are
    // scaled by the factor (1 + scale * cfg / n_cfg) before staging,
    // so the configurations drive slightly different equilibria
    // through the iteration chain. The kernel arithmetic is
    // independent of the input values, so the wall time under
    // perturbation is expected to match the broadcast baseline once
    // the configuration-zero-only write paths have been closed by the
    // per-configuration audit. The default state (variable unset or
    // zero) preserves pure broadcast.
    static double phase_d_perturb = -1.0;
    if (phase_d_perturb < 0.0) {
      const char* e = std::getenv("VMECPP_BATCH_PERTURB");
      phase_d_perturb = (e && *e) ? std::atof(e) : 0.0;
      if (phase_d_perturb != 0.0) {
        std::fprintf(stderr,
                     "[fft_toroidal_cuda] per-cfg input perturbation active "
                     "(VMECPP_BATCH_PERTURB=%.3e: per-cfg scale "
                     "1+%.3e*cfg/n_cfg)\n",
                     phase_d_perturb, phase_d_perturb);
      }
    }
    // Distinct-boundary spectral input pipeline. When the
    // VMECPP_BATCH_INPUTS_FILE environment variable identifies a
    // binary file, that file holds N_cfg * 6 * one_spec_doubles
    // double-precision values arranged in the layout
    // [sp][cfg][specs...], and the per-configuration spectral slots
    // are loaded from the file rather than being broadcast from the
    // host physical_x buffer of the seed Vmec instance. The file is
    // read exactly once per process and cached thereafter in pinned
    // host memory.
    // The staging cache lives in the State so ResetForNewVmecRun rearms
    // it; the consumed flag keeps the one-shot semantics within a run
    // (the iter-1 retry after the axis recompute must see the host
    // m_physical_x, not a reload of the pre-init staging).
    if (S.batch_inputs_loaded < 0 && !g_batch_inputs_mem.empty() &&
        g_batch_mem_shape[0] == n_cfg && g_batch_mem_shape[1] == ns_local &&
        g_batch_mem_shape[2] == mpol && g_batch_mem_shape[3] == ntor) {
      // In-memory block from SetBatchStagingCuda; same one-shot consume
      // semantics as the file path.
      const size_t total = g_batch_inputs_mem.size();
      cuda_check(cudaMallocHost(&S.batch_inputs_pinned,
                                sizeof(double) * total),
                 "alloc batch inputs pinned");
      std::memcpy(S.batch_inputs_pinned, g_batch_inputs_mem.data(),
                  sizeof(double) * total);
      S.batch_inputs_n_cfg = n_cfg;
      S.batch_inputs_one_spec_doubles =
          (size_t)ns_local * mpol * (ntor + 1);
      S.batch_inputs_loaded = 1;
      std::fprintf(stderr,
          "[fft_toroidal_cuda] batch inputs loaded: N=%d ns=%d "
          "mpol=%d ntor=%d (%zu doubles from memory)\n",
          n_cfg, ns_local, mpol, ntor, total);
    }
    if (S.batch_inputs_loaded < 0) {
      const char* path = std::getenv("VMECPP_BATCH_INPUTS_FILE");
      if (path && *path) {
        FILE* f = std::fopen(path, "rb");
        if (f) {
          // File header: int32 N, int32 ns_local, int32 mpol, int32 ntor.
          int32_t header[4] = {0, 0, 0, 0};
          size_t hread = std::fread(header, sizeof(int32_t), 4, f);
          int N_file = header[0];
          int ns_file = header[1];
          int mpol_file = header[2];
          int ntor_file = header[3];
          size_t expect_per_spec = (size_t)ns_file * mpol_file * (ntor_file + 1);
          size_t total_doubles = (size_t)N_file * 6 * expect_per_spec;
          if (hread == 4 && N_file == n_cfg && ns_file == ns_local &&
              mpol_file == mpol && ntor_file == ntor) {
            cuda_check(cudaMallocHost(&S.batch_inputs_pinned,
                                      sizeof(double) * total_doubles),
                       "alloc batch inputs pinned");
            size_t r = std::fread(S.batch_inputs_pinned, sizeof(double),
                                  total_doubles, f);
            if (r == total_doubles) {
              S.batch_inputs_n_cfg = N_file;
              S.batch_inputs_one_spec_doubles = expect_per_spec;
              S.batch_inputs_loaded = 1;
              std::fprintf(stderr,
                  "[fft_toroidal_cuda] batch inputs loaded: N=%d ns=%d "
                  "mpol=%d ntor=%d (%zu doubles from %s)\n",
                  N_file, ns_file, mpol_file, ntor_file, total_doubles, path);
            } else {
              std::fprintf(stderr,
                  "[fft_toroidal_cuda] batch inputs file truncated "
                  "(expected %zu doubles, got %zu); using broadcast\n",
                  total_doubles, r);
              S.batch_inputs_loaded = 0;
            }
          } else {
            std::fprintf(stderr,
                "[fft_toroidal_cuda] batch inputs file shape mismatch "
                "(file N=%d ns=%d mpol=%d ntor=%d vs run N=%d ns=%d mpol=%d "
                "ntor=%d); using broadcast\n",
                N_file, ns_file, mpol_file, ntor_file,
                n_cfg, ns_local, mpol, ntor);
            S.batch_inputs_loaded = 0;
          }
          std::fclose(f);
        } else {
          S.batch_inputs_loaded = 0;
        }
      } else {
        S.batch_inputs_loaded = 0;
      }
    }

    if (S.batch_inputs_loaded == 1 && !S.batch_inputs_consumed) {
      // Copy from bundle: layout [sp][cfg][specs...] into [sp_block][cfg].
      for (int sp = 0; sp < 6; ++sp) {
        for (int cfg = 0; cfg < n_cfg; ++cfg) {
          const double* src_cfg = S.batch_inputs_pinned +
              (size_t)sp * n_cfg * S.batch_inputs_one_spec_doubles +
              (size_t)cfg * S.batch_inputs_one_spec_doubles;
          std::memcpy(h + sp * block_doubles + cfg * one_spec_doubles,
                      src_cfg, one_spec_bytes);
        }
      }
      S.batch_inputs_consumed = true;
    } else if (phase_d_perturb == 0.0) {
      for (int sp = 0; sp < 6; ++sp) {
        // rmnss/zmncs/lmncs (odd sp) are the lthreed-only modes; for an
        // axisymmetric (ntor=0) run they are absent on the host, so stage
        // zeros rather than reading past the empty source array.
        const bool absent_2d = !s.lthreed && (sp == 1 || sp == 3 || sp == 5);
        for (int cfg = 0; cfg < n_cfg; ++cfg) {
          double* dst = h + sp * block_doubles + cfg * one_spec_doubles;
          if (absent_2d) {
            std::memset(dst, 0, one_spec_bytes);
          } else {
            std::memcpy(dst, src[sp], one_spec_bytes);
          }
        }
      }
    } else {
      for (int sp = 0; sp < 6; ++sp) {
        const bool absent_2d = !s.lthreed && (sp == 1 || sp == 3 || sp == 5);
        for (int cfg = 0; cfg < n_cfg; ++cfg) {
          double scale = 1.0 + phase_d_perturb *
                         (double)cfg / (double)std::max(1, n_cfg);
          double* dst = h + sp * block_doubles + cfg * one_spec_doubles;
          if (absent_2d) {
            std::memset(dst, 0, one_spec_bytes);
          } else {
            for (size_t i = 0; i < one_spec_doubles; ++i) {
              dst[i] = src[sp][i] * scale;
            }
          }
        }
      }
    }
  }
  // VMECPP_DUMP_SPECS=1: one-shot dump of h_specs_pinned cfg-0 slot for each
  // spec section, plus running sum and abs-sum. For distinct-vs-broadcast
  // bit-equivalence verification at iter 1.
  {
    static int dump_specs_env = -1;
    if (dump_specs_env < 0) {
      const char* e = std::getenv("VMECPP_DUMP_SPECS");
      dump_specs_env = (e && std::atoi(e) > 0) ? 1 : 0;
    }
    static int dump_count = 0;
    if (dump_specs_env && dump_count == 0) {
      dump_count = 1;
      const char* sp_names[6] = {"rmncc", "rmnss", "zmnsc", "zmncs",
                                  "lmnsc", "lmncs"};
      const char* env_batch = std::getenv("VMECPP_BATCH_INPUTS_FILE");
      const char* path_label = (env_batch && *env_batch) ? "FILE" : "BCAST";
      for (int sp = 0; sp < 6; ++sp) {
        double* cfg0 = h + sp * block_doubles + 0 * one_spec_doubles;
        double sum = 0.0, abs_sum = 0.0;
        for (size_t i = 0; i < one_spec_doubles; ++i) {
          sum += cfg0[i];
          abs_sum += std::abs(cfg0[i]);
        }
        std::fprintf(stderr,
                     "[DUMP_SPECS path=%s] %s cfg0 first5=%.16e %.16e %.16e %.16e %.16e "
                     "sum=%.16e abs_sum=%.16e\n",
                     path_label, sp_names[sp],
                     cfg0[0], cfg0[1], cfg0[2], cfg0[3], cfg0[4],
                     sum, abs_sum);
      }
    }
  }
  // xmpq: mpol doubles (shared across configs, single copy at the right offset).
  std::memcpy(h + 6 * block_doubles,
              xmpq.data(), sizeof(double) * mpol);
  // sqrtSF: ns_local doubles per config (broadcast same data to all N slots
  // since the radial grid is identical across configs).
  double* h_sqrtSF = h + 6 * block_doubles + mpol;
  for (int cfg = 0; cfg < n_cfg; ++cfg) {
    for (int jF_local = 0; jF_local < ns_local; ++jF_local) {
      h_sqrtSF[cfg * ns_local + jF_local] = rp.sqrtSF[jF_local];
    }
  }
  // When RecomposeToPhysicalCuda has populated the six spectral
  // sections of d_specs_block directly from device-resident sources,
  // the host-to-device transfer of those sections is unnecessary and
  // is elided here; the xmpq and sqrtSF tail of the staging buffer
  // is still transferred because RecomposeToPhysicalCuda does not
  // touch those regions. The producer flag is cleared after the
  // elision so that, should the next iteration not invoke
  // RecomposeToPhysicalCuda, the host-side staging path resumes the
  // full transfer defensively.
  if (S.specs_populated_from_device) {
    // Skip spec sections (6 * one_spec_bytes * n_cfg) but H2D the tail.
    size_t spec_total = (size_t)6 * one_spec_bytes * (size_t)n_cfg;
    size_t tail_bytes = sizeof(double) * mpol +
                        sizeof(double) * ns_local * (size_t)n_cfg;
    cuda_check(cudaMemcpyAsync(
        (char*)S.d_specs_block + spec_total,
        (char*)S.h_specs_pinned + spec_total,
        tail_bytes,
        cudaMemcpyHostToDevice, st), "h2d specs_block tail only");
    S.specs_populated_from_device = false;
  } else {
    cuda_check(cudaMemcpyAsync(S.d_specs_block, S.h_specs_pinned,
                               S.specs_block_bytes,
                               cudaMemcpyHostToDevice, st), "h2d specs_block");
  }

  // The per-iteration cudaMemsetAsync of d_outputs_block is omitted.
  // The active scatter kernel k_scatter_main_and_con writes the
  // sixteen even-parity and odd-parity main outputs and the two
  // constraint outputs with direct assignment at every
  // (cfg, jF_local, k, l) within the full output range, and is the
  // sole producer of those arrays between consecutive forward FFT
  // calls. Direct assignment is therefore sufficient and the
  // pre-launch zero-initialization is unnecessary. The disabled
  // fusion scaffolds k_forward_fft_fused, k_fwd_fused_R, k_fwd_fused_Z,
  // k_fwd_fused_L, and k_fwd_fused_warp accumulate with compound
  // addition and would require restoration of the memset if any of
  // them were re-enabled.

  // Launch fill kernel on stream.
  // Batched execution: z-dim is config * ns_local + jF_local. cuFFT batch dim
  // already covers n_config_max via the batched plan setup.
  const int FILL_TPB = 256;
  dim3 fill_blocks((kBatch * nhalf + FILL_TPB - 1) / FILL_TPB, mpol,
                   ns_local * S.n_config_max);
  dim3 fill_tpb(FILL_TPB, 1, 1);
  const int SCAT_TPB = 32;
  dim3 scat_blocks((nThetaReduced + SCAT_TPB - 1) / SCAT_TPB, nZeta,
                   ns_local * S.n_config_max);
  dim3 scat_tpb(SCAT_TPB, 1, 1);
  int nsMinF_offset_in_local = r.nsMinF - r.nsMinF1;

  // Hoist this declaration so gotos in the disabled-fusion blocks below
  // don't bypass it (CUDA compiler is strict about goto-bypass-init).
  bool can_fuse_main_con_cufft = (ns_con_local == ns_local) &&
                                  (nsMinF_offset_in_local == 0);

  // Forward-FFT CUDA graph enablement governed by the
  // VMECPP_FWD_GRAPH environment variable. The default is disabled
  // because the captured forward-FFT chain contains a cuFFT call,
  // and the cuFFT replay path on the current cuda-toolkit release
  // delivers no measurable improvement over the direct stream
  // execution at the canonical problem shape. The control is
  // retained as an opt-in for future toolkit releases whose
  // graph-mode cuFFT implementation may yield a positive delta.
  static int fwd_graph_env = -1;
  if (fwd_graph_env < 0) {
    const char* e = std::getenv("VMECPP_FWD_GRAPH");
    fwd_graph_env = (e && std::atoi(e) > 0) ? 1 : 0;
    if (fwd_graph_env) {
      std::fprintf(stderr, "[fft_toroidal_cuda] forward-FFT CUDA Graph "
                           "ENABLED (VMECPP_FWD_GRAPH=1; default is off, "
                           "+2.2pct regression at N=64)\n");
    }
  }
  const bool use_fwd_graph = (fwd_graph_env != 0) && can_fuse_main_con_cufft &&
                             !g_iter_graph_capturing;
  const int nZnT_local_pre = nZeta * nThetaEff;
  const int outer_idx_pre = (ns_local - 1) * nZnT_local_pre + 0;
  const int inner_idx_pre = (ns_local - 1) * nZnT_local_pre + (nThetaReduced - 1);
  bool replay_only = use_fwd_graph && S.fwd_graph_captured;
  bool capture_then_launch = use_fwd_graph && !S.fwd_graph_captured;

  // Disabled scaffold: warp-cooperative fusion through
  // k_fwd_fused_warp (one (cfg, jF_local, k) tuple per warp,
  // __shfl_xor_sync reductions, no d_X/d_Y intermediates). Its inner
  // transform is a direct-sum length-24 DFT, so the arithmetic exceeds
  // cufftExecZ2D's radix-8x3 by more than the saved memory traffic,
  // and aspect_ratio drifts ~2 ULP.
  if (false) {
    dim3 warp_blocks(1, nZeta, ns_local * S.n_config_max);
    dim3 warp_tpb(32, 1, 1);
    k_fwd_fused_warp<<<warp_blocks, warp_tpb, 0, st>>>(
        S.n_config_max, ns_local, mpol, ntor, nfp, nZeta, nThetaReduced,
        nThetaEff, r.nsMinF1,
        S.d_rmncc, S.d_rmnss, S.d_zmnsc, S.d_zmncs, S.d_lmnsc, S.d_lmncs,
        S.d_dft_cos, S.d_dft_sin,
        S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
        S.d_xmpq, S.d_sqrtSF,
        S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
        S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
        S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
        S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
        S.d_rCon, S.d_zCon);
    cuda_check(cudaGetLastError(), "k_fwd_fused_warp launch");
    goto scatter_done;
  }

  // Disabled scaffold: output-group-partitioned fusion through
  // k_fwd_fused_R, k_fwd_fused_Z, and k_fwd_fused_L. The three
  // launches keep per-thread register pressure within the available
  // file but retain the direct-sum inner DFT of the warp-cooperative
  // scaffold; the floating-point operation count remains the
  // governing constraint and the wall is correspondingly slower
  // than the production chain.
  if (false) {
    k_fwd_fused_R<<<scat_blocks, scat_tpb, 0, st>>>(
        S.n_config_max, ns_local, mpol, ntor, nfp, nZeta, nThetaReduced,
        nThetaEff, r.nsMinF1,
        S.d_rmncc, S.d_rmnss, S.d_dft_cos, S.d_dft_sin,
        S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
        S.d_xmpq, S.d_sqrtSF,
        S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
        S.d_rv_e, S.d_rv_o, S.d_rCon);
    cuda_check(cudaGetLastError(), "k_fwd_fused_R launch");
    k_fwd_fused_Z<<<scat_blocks, scat_tpb, 0, st>>>(
        S.n_config_max, ns_local, mpol, ntor, nfp, nZeta, nThetaReduced,
        nThetaEff, r.nsMinF1,
        S.d_zmnsc, S.d_zmncs, S.d_dft_cos, S.d_dft_sin,
        S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
        S.d_xmpq, S.d_sqrtSF,
        S.d_z1_e, S.d_z1_o, S.d_zu_e, S.d_zu_o,
        S.d_zv_e, S.d_zv_o, S.d_zCon);
    cuda_check(cudaGetLastError(), "k_fwd_fused_Z launch");
    k_fwd_fused_L<<<scat_blocks, scat_tpb, 0, st>>>(
        S.n_config_max, ns_local, mpol, ntor, nfp, nZeta, nThetaReduced,
        nThetaEff, r.nsMinF1,
        S.d_lmnsc, S.d_lmncs, S.d_dft_cos, S.d_dft_sin,
        S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
        S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o);
    cuda_check(cudaGetLastError(), "k_fwd_fused_L launch");
    goto scatter_done;
  }

  // Disabled scaffold: single-kernel full-pipeline fusion through
  // k_forward_fft_fused. The combined kernel carries the accumulator
  // doubles for all eighteen outputs in registers per thread and
  // spills to local memory on the target architecture, regressing
  // the wall measurably relative to the production chain.
  if (false) {
    k_forward_fft_fused<<<scat_blocks, scat_tpb, 0, st>>>(
        S.n_config_max, ns_local, mpol, ntor, nfp, nZeta, nThetaReduced,
        nThetaEff, r.nsMinF1,
        S.d_rmncc, S.d_rmnss, S.d_zmnsc, S.d_zmncs, S.d_lmnsc, S.d_lmncs,
        S.d_dft_cos, S.d_dft_sin,
        S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
        S.d_xmpq, S.d_sqrtSF,
        S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
        S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
        S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
        S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
        S.d_rCon, S.d_zCon);
    cuda_check(cudaGetLastError(), "k_forward_fft_fused launch");
    goto scatter_done;
  }

  // Forward-FFT CUDA graph dispatch. The graph state declared above
  // distinguishes three cases of the current invocation. When the
  // forward graph is enabled and a previous capture is already
  // available, the captured executable is launched and the remainder
  // of the chain is bypassed. When the forward graph is enabled but
  // no capture is available, stream capture is begun and the chain
  // body that follows is recorded; the capture is then ended,
  // instantiated, and launched at the bottom of the block. When the
  // forward graph is disabled, the chain body executes directly on
  // the stream with no capture in effect.
  if (replay_only) {
    cuda_check(cudaGraphLaunch(S.fwd_graph_exec, st), "graph launch fwd");
    goto fwd_chain_done;
  }

  if (capture_then_launch) {
    cuda_check(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal),
               "begin capture fwd graph");
  }

  k_fill_spectra<<<fill_blocks, fill_tpb, 0, st>>>(
      S.n_config_max, ns_local, mpol, ntor, nhalf, nfp, r.nsMinF1,
      S.d_rmncc, S.d_rmnss, S.d_zmnsc, S.d_zmncs, S.d_lmnsc, S.d_lmncs,
      S.d_nscale, S.d_X);
  cuda_check(cudaGetLastError(), "k_fill_spectra launch");

  // Mixed-precision Fourier transform branch. When the environment
  // variable VMECPP_FFT_FP32 is enabled, the double-precision
  // complex input d_X is narrowed to the single-precision buffer
  // d_X_fp32, cufftExecC2R produces the single-precision real
  // output d_Y_fp32, and that output is widened back to the
  // double-precision buffer d_Y consumed by the downstream scatter
  // kernels. The narrowed path delivers higher throughput at the
  // cost of reduced numerical fidelity; the resulting drift in
  // aspect_ratio places this branch outside the bit-exact contract,
  // so the branch is treated as an opt-in measurement scaffold
  // rather than a production path.
  static int fft_fp32_env = -1;
  if (fft_fp32_env < 0) {
    const char* e = std::getenv("VMECPP_FFT_FP32");
    fft_fp32_env = (e && std::atoi(e) > 0) ? 1 : 0;
    if (fft_fp32_env) {
      std::fprintf(stderr, "[fft_toroidal_cuda] mixed-precision FFT enabled "
                           "(VMECPP_FFT_FP32=1)\n");
    }
  }
  if (fft_fp32_env) {
    const int CAST_TPB = 256;
    int x_blocks = (int)((S.fft_x_elems + CAST_TPB - 1) / CAST_TPB);
    k_cast_complex_fp64_to_fp32<<<x_blocks, CAST_TPB, 0, st>>>(
        S.fft_x_elems, S.d_X, S.d_X_fp32);
    cuda_check(cudaGetLastError(), "k_cast_complex_fp64_to_fp32 launch");
    cufft_check(cufftExecC2R(S.cufft_plan_c2r_fp32, S.d_X_fp32, S.d_Y_fp32),
                "cufftExecC2R fp32");
    int y_blocks = (int)((S.fft_y_elems + CAST_TPB - 1) / CAST_TPB);
    k_cast_fp32_to_fp64<<<y_blocks, CAST_TPB, 0, st>>>(
        S.fft_y_elems, S.d_Y_fp32, S.d_Y);
    cuda_check(cudaGetLastError(), "k_cast_fp32_to_fp64 launch");
  } else {
    // Hand-coded radix-8x3 inverse Fourier transform as an opt-in
    // alternative to cuFFT's mixed-radix length-24 Z2D. The control
    // is governed by the VMECPP_FFT_RADIX environment variable and
    // defaults to disabled because the hand-coded path does not
    // match cuFFT's wall throughput at the canonical problem shape,
    // and its accumulation order yields a small drift in
    // aspect_ratio that falls outside the bit-exact contract. The
    // path is retained as an enabling control for the broader FFT
    // investigation. The factorization is specific to transform
    // length 24; other nZeta values stay on cuFFT.
    static int fft_radix_env = -1;
    if (fft_radix_env < 0) {
      const char* e = std::getenv("VMECPP_FFT_RADIX");
      fft_radix_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (fft_radix_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] hand-coded radix-8x3 FFT "
                             "enabled (VMECPP_FFT_RADIX=1)\n");
      }
    }
    if (fft_radix_env && nZeta != 24) {
      static bool radix_shape_warned = false;
      if (!radix_shape_warned) {
        radix_shape_warned = true;
        std::fprintf(stderr,
                     "[fft_toroidal_cuda] VMECPP_FFT_RADIX=1 requires "
                     "nZeta = 24 (this input has nZeta = %d); using cuFFT\n",
                     nZeta);
      }
    }
    if (fft_radix_env && nZeta == 24) {
      // 8 FFTs per block, 32 threads * 8 ffts = 256 threads/block.
      constexpr int FFTS_PER_BLOCK = 8;
      int total_batches = n_cfg * ns_local * mpol * kBatch;
      dim3 r_grid((total_batches + FFTS_PER_BLOCK - 1) / FFTS_PER_BLOCK, 1, 1);
      dim3 r_tpb(32, FFTS_PER_BLOCK, 1);
      // smem per FFT: 24 (X_re) + 24 (X_im) + 48 (T_re + T_im) = 96 doubles
      size_t smem = sizeof(double) * 96 * FFTS_PER_BLOCK;
      k_inverse_dft_24_radix83<<<r_grid, r_tpb, smem, st>>>(
          total_batches, nhalf, nZeta, S.d_X, S.d_Y);
      cuda_check(cudaGetLastError(), "k_inverse_dft_24_radix83 launch");
    } else if (nZeta == 1) {
      // Length-1 toroidal transform: the identity (DC real part). cuFFT's
      // length-1 plan is bypassed for the axisymmetric (ntor=0) case.
      int total_batches = n_cfg * ns_local * mpol * kBatch;
      int tpb = 256;
      k_z2d_identity_nzeta1<<<(total_batches + tpb - 1) / tpb, tpb, 0, st>>>(
          total_batches, S.d_X, S.d_Y);
      cuda_check(cudaGetLastError(), "k_z2d_identity_nzeta1");
    } else {
      S.TKBegin(CudaToroidalState::TK_CUFFT_INV);
      cufft_check(cufftExecZ2D(S.cufft_plan, S.d_X, S.d_Y), "cufftExecZ2D");
      S.TKEnd(CudaToroidalState::TK_CUFFT_INV);

      // One-shot cuFFT-vs-radix-8x3 dump.
      // Gated by VMECPP_FFT_DUMP=1. Captures the first call's complex input
      // X and the corresponding cuFFT Z2D output Y to disk, then re-runs the
      // hand-coded radix-8x3 on the same input X (writing to a scratch
      // buffer, leaving S.d_Y untouched) and dumps that too. The three
      // files at /tmp/vmecpp_fft_z2d_*.bin are the basis for the
      // ULP-by-ULP comparison and factorization analysis. Skipped for
      // transform lengths the radix kernel does not cover (nZeta != 24).
      static int fft_dump_env = -1;
      if (fft_dump_env < 0) {
        const char* e = std::getenv("VMECPP_FFT_DUMP");
        fft_dump_env = (e && std::atoi(e) > 0 && nZeta == 24) ? 1 : 0;
        if (e && std::atoi(e) > 0 && nZeta != 24) {
          std::fprintf(stderr,
                       "[fft_toroidal_cuda] VMECPP_FFT_DUMP=1 requires "
                       "nZeta = 24 (this input has nZeta = %d); skipped\n",
                       nZeta);
        }
      }
      static bool fft_dump_done = false;
      if (fft_dump_env && !fft_dump_done) {
        fft_dump_done = true;
        int total_batches = n_cfg * ns_local * mpol * kBatch;
        size_t X_bytes = (size_t)total_batches * nhalf * sizeof(cufftDoubleComplex);
        size_t Y_bytes = (size_t)total_batches * nZeta * sizeof(double);
        std::vector<cufftDoubleComplex> h_X(total_batches * nhalf);
        std::vector<double> h_Y_cufft(total_batches * nZeta);
        cuda_check(cudaMemcpyAsync(h_X.data(), S.d_X, X_bytes,
                                    cudaMemcpyDeviceToHost, st),
                   "d2h FFT_DUMP X");
        cuda_check(cudaMemcpyAsync(h_Y_cufft.data(), S.d_Y, Y_bytes,
                                    cudaMemcpyDeviceToHost, st),
                   "d2h FFT_DUMP Y cufft");
        // Run the radix-8x3 inverse on a scratch buffer to capture its
        // output without disturbing the production Y.
        double* d_Y_radix = nullptr;
        cuda_check(cudaMalloc(&d_Y_radix, Y_bytes),
                   "alloc FFT_DUMP d_Y_radix");
        constexpr int FFTS_PER_BLOCK = 8;
        dim3 r_grid((total_batches + FFTS_PER_BLOCK - 1) / FFTS_PER_BLOCK, 1, 1);
        dim3 r_tpb(32, FFTS_PER_BLOCK, 1);
        size_t smem = sizeof(double) * 96 * FFTS_PER_BLOCK;
        k_inverse_dft_24_radix83<<<r_grid, r_tpb, smem, st>>>(
            total_batches, nhalf, nZeta, S.d_X, d_Y_radix);
        cuda_check(cudaGetLastError(), "k_inverse_dft_24_radix83 (DUMP) launch");
        std::vector<double> h_Y_radix(total_batches * nZeta);
        cuda_check(cudaMemcpyAsync(h_Y_radix.data(), d_Y_radix, Y_bytes,
                                    cudaMemcpyDeviceToHost, st),
                   "d2h FFT_DUMP Y radix");
        cuda_check(cudaStreamSynchronize(st), "FFT_DUMP sync");
        cudaFree(d_Y_radix);
        // Header: 4 int32 (total_batches, nhalf, nZeta, padding).
        auto write_file = [&](const char* path, const void* buf, size_t bytes) {
          FILE* f = std::fopen(path, "wb");
          if (!f) return;
          int32_t hdr[4] = {total_batches, nhalf, nZeta, 0};
          std::fwrite(hdr, sizeof(int32_t), 4, f);
          std::fwrite(buf, 1, bytes, f);
          std::fclose(f);
        };
        write_file("/tmp/vmecpp_fft_z2d_in.bin",
                   h_X.data(), X_bytes);
        write_file("/tmp/vmecpp_fft_z2d_out_cufft.bin",
                   h_Y_cufft.data(), Y_bytes);
        write_file("/tmp/vmecpp_fft_z2d_out_radix83.bin",
                   h_Y_radix.data(), Y_bytes);
        std::fprintf(stderr,
            "[fft_toroidal_cuda] FFT_DUMP: dumped %d batches × (nhalf=%d "
            "complex in, nZeta=%d real out) cuFFT + radix-8x3 to "
            "/tmp/vmecpp_fft_z2d_*.bin\n",
            total_batches, nhalf, nZeta);
      }
    }
  }
  if (false) {
    int total_batches = n_cfg * ns_local * mpol * kBatch;
    dim3 tpb(32, 4);
    dim3 grid((nZeta + tpb.x - 1) / tpb.x,
              (total_batches + tpb.y - 1) / tpb.y);
    k_inverse_dft_24<<<grid, tpb, 0, st>>>(
        total_batches, nhalf, nZeta,
        S.d_X, S.d_idft_cos, S.d_idft_sin, S.d_Y);
    cuda_check(cudaGetLastError(), "k_inverse_dft_24 launch");
  }

  // Dispatch to the fused scatter that combines the main and
  // constraint outputs in a single pass over Y. The dispatch is
  // selected when the single-rank LCFS condition holds; under that
  // condition the k_scatter_main_and_con family of kernels produces
  // both output groups together. Earlier scaffolds k_scatter_main_and_con_v2
  // and k_scatter_main_and_con_v3 are retained in source as
  // disabled alternatives whose effective wall does not exceed
  // that of the current default.
  if (can_fuse_main_con_cufft) {
    // The v4 variant launches one block per (configuration,
    // jF_local), with the block split into four warps that each
    // process one (configuration, jF_local, k) triple. The
    // four-warp arrangement raises the warp count resident on a
    // single SM and thereby the instruction-issue concurrency of
    // the floating-point pipeline, without altering the per-warp
    // work assignment of the underlying scatter algorithm.
    constexpr int WARPS_PER_BLOCK = 4;
    int z_total = ns_local * S.n_config_max;
    int z_blocks = (z_total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 v4_blocks((nThetaReduced + 31) / 32, nZeta, z_blocks);
    dim3 v4_tpb(32, WARPS_PER_BLOCK, 1);
    S.TKBegin(CudaToroidalState::TK_SCATTER);
    // The v5 variant of the fused scatter caches the Y values
    // consumed during the inner toroidal-mode loop in a per-warp
    // shared-memory tile, removing the L1 broadcast that the v4
    // variant uses for the same loads. Selection is governed by
    // the VMECPP_SCATTER_V5 environment variable; the default is
    // active when unset and falls back to the v4 variant when the
    // variable is set to zero.
    static int scatter_v5_env = -1;
    if (scatter_v5_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_V5");
      scatter_v5_env = (e && std::atoi(e) == 0) ? 0 : 1;
      if (!scatter_v5_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] k_scatter v5 (shared-mem "
                             "Y cache) disabled (VMECPP_SCATTER_V5=0)\n");
      }
    }
    // FP32 Phase 2: DD-pair-accumulator FP32 scatter variant. Multiplications
    // run in native FP32, the 18 m-sum accumulators use DD pairs (~48-bit
    // mantissa). Gated by VMECPP_SCATTER_DD_FP32=1; default OFF. cuFFT stays
    // in FP64 regardless of this flag.
    static int scatter_dd_fp32_env = -1;
    if (scatter_dd_fp32_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_DD_FP32");
      scatter_dd_fp32_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_dd_fp32_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter DD-FP32 path enabled "
                             "(VMECPP_SCATTER_DD_FP32=1)\n");
      }
    }
    static int scatter_dd_fp64mul_env = -1;
    if (scatter_dd_fp64mul_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_DD_FP64MUL");
      scatter_dd_fp64mul_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_dd_fp64mul_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter Path-1 enabled "
                             "(VMECPP_SCATTER_DD_FP64MUL=1, FP64 mul + DD sum)\n");
      }
    }
    static int scatter_dd_fp32_ddmul_env = -1;
    if (scatter_dd_fp32_ddmul_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_DD_FP32_DDMUL");
      scatter_dd_fp32_ddmul_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_dd_fp32_ddmul_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter Path-2 enabled "
                             "(VMECPP_SCATTER_DD_FP32_DDMUL=1, DDxDD mul + DD sum)\n");
      }
    }
    static int scatter_ozaki_env = -1;
    if (scatter_ozaki_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_OZAKI_FP32");
      scatter_ozaki_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_ozaki_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter Path-3 enabled "
                             "(VMECPP_SCATTER_OZAKI_FP32=1, Ozaki 2-slice FP32 "
                             "mul + DD sum)\n");
      }
    }
    static int scatter_ozaki3_env = -1;
    if (scatter_ozaki3_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_OZAKI3_FP32");
      scatter_ozaki3_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_ozaki3_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter Path-3b enabled "
                             "(VMECPP_SCATTER_OZAKI3_FP32=1, Ozaki 3-slice "
                             "FP32 mul + DD sum, ~72-bit precision)\n");
      }
    }
    static int scatter_cublas_fp32_env = -1;
    if (scatter_cublas_fp32_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_CUBLAS_FP32");
      scatter_cublas_fp32_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_cublas_fp32_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter Path-4 enabled "
                             "(VMECPP_SCATTER_CUBLAS_FP32=1, cuBLAS GemmEx "
                             "FP32 + rcon/zcon FP64 trailing kernel)\n");
      }
    }
    static int scatter_cublas_ozaki_env = -1;
    if (scatter_cublas_ozaki_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_CUBLAS_OZAKI");
      scatter_cublas_ozaki_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_cublas_ozaki_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter Path-4b enabled "
                             "(VMECPP_SCATTER_CUBLAS_OZAKI=1, 4 GEMMs + DD "
                             "unpack)\n");
      }
    }
    static int scatter_custom_gemm_env = -1;
    if (scatter_custom_gemm_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_CUSTOM_GEMM");
      scatter_custom_gemm_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_custom_gemm_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter Path-5 enabled "
                             "(VMECPP_SCATTER_CUSTOM_GEMM=1, Custom "
                             "Veltkamp-Dekker Tile GEMM, shared-mem "
                             "cooperative loads + per-mul DD)\n");
      }
    }
    static int scatter_custom_gemm_wmma_env = -1;
    if (scatter_custom_gemm_wmma_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_CUSTOM_GEMM_WMMA");
      scatter_custom_gemm_wmma_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_custom_gemm_wmma_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter Path-5b enabled "
                             "(VMECPP_SCATTER_CUSTOM_GEMM_WMMA=1, TF32 "
                             "tensor-core dispatch via wmma::mma_sync with "
                             "3-slice Ozaki, 54 wmma calls per tile)\n");
      }
    }
    static int scatter_tf32_plain_env = -1;
    if (scatter_tf32_plain_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_TF32_PLAIN");
      scatter_tf32_plain_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_tf32_plain_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter TF32 plain output "
                             "ENABLED (VMECPP_SCATTER_TF32_PLAIN=1, skip "
                             "scalar VD correction, rel ~ 3e-6)\n");
      }
    }
    static int scatter_i8gemm_env = -1;
    if (scatter_i8gemm_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_I8GEMM");
      scatter_i8gemm_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_i8gemm_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter batched "
                             "int8-Ozaki GEMM ENABLED "
                             "(VMECPP_SCATTER_I8GEMM=1)\n");
      }
    }
    static int scatter_i8ozaki_env = -1;
    if (scatter_i8ozaki_env < 0) {
      const char* e = std::getenv("VMECPP_SCATTER_I8OZAKI");
      scatter_i8ozaki_env = (e && std::atoi(e) > 0) ? 1 : 0;
      if (scatter_i8ozaki_env) {
        std::fprintf(stderr, "[fft_toroidal_cuda] scatter int8-Ozaki "
                             "ENABLED (VMECPP_SCATTER_I8OZAKI=1, eight "
                             "7-bit limbs, exact s32 accumulation)\n");
      }
    }
    // The wmma tile geometry admits mpol up to 12 (4 * mpol <= K_PAD)
    // and nThetaReduced up to 16 (M_TILE); larger inputs stay on the
    // production scatter with a one-time notice.
    if ((scatter_custom_gemm_wmma_env || scatter_i8ozaki_env) &&
        (mpol > 12 || nThetaReduced > 16)) {
      static bool wmma_shape_warned = false;
      if (!wmma_shape_warned) {
        wmma_shape_warned = true;
        std::fprintf(stderr,
                     "[fft_toroidal_cuda] VMECPP_SCATTER_CUSTOM_GEMM_WMMA=1 "
                     "covers mpol <= 12 and nThetaReduced <= 16 (this input "
                     "has mpol = %d, nThetaReduced = %d); using the "
                     "production scatter\n",
                     mpol, nThetaReduced);
      }
      scatter_custom_gemm_wmma_env = 0;
      scatter_i8ozaki_env = 0;
    }
    // Limb width for the int8 paths: run-scoped override
    // (VMECPP_SCATTER_I8_LIMBS=4), per-iteration phase routing under IR
    // staging (4-limb descent above the residual threshold, 8 below). A
    // width change invalidates the whole-iteration graph, the only
    // capture that contains the forward scatter.
    int i8_limbs = 8;
    if (scatter_i8gemm_env || scatter_i8ozaki_env) {
      if (g_i8_limbs_env < 0) {
        const char* e = std::getenv("VMECPP_SCATTER_I8_LIMBS");
        g_i8_limbs_env = (e && std::atoi(e) == 4) ? 4 : 8;
        if (g_i8_limbs_env != 8) {
          std::fprintf(stderr, "[fft_toroidal_cuda] int8 scatter limb "
                               "width 4 (VMECPP_SCATTER_I8_LIMBS=4)\n");
        }
      }
      // Hysteresis around the IR threshold: 8 limbs below it, 4 limbs
      // only above a decade band, hold the current width inside the
      // band. Without the band the residual jitters across the
      // threshold and the width thrashes for several iterations per
      // crossing. Stage restarts push the residual far above the band,
      // so every multigrid stage re-enters its 4-limb descent.
      (void)GetIRPhase();  // ensures init_ir_env ran
      if (g_ir_staged == 1) {
        if (g_ir_residual_sum < g_ir_threshold) {
          i8_limbs = 8;
        } else if (g_ir_residual_sum > 100.0 * g_ir_threshold) {
          i8_limbs = 4;
        } else {
          i8_limbs = (g_i8_limbs_last != 0) ? g_i8_limbs_last : 4;
        }
      } else {
        i8_limbs = g_i8_limbs_env;
      }
      if (g_i8_limbs_last != 0 && g_i8_limbs_last != i8_limbs) {
        // Drop the whole-iteration graph in place: it captured the
        // previous width's kernel, and the state mutex is already held
        // in this scope (InvalidateIterationGraphCuda would self-lock).
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
        std::fprintf(stderr, "[fft_toroidal_cuda] int8 scatter limb "
                             "width %d -> %d\n",
                     g_i8_limbs_last, i8_limbs);
      }
      g_i8_limbs_last = i8_limbs;
    }
    // Dispatch order: explicit gate variants first, then v5 default.
    if (scatter_dd_fp32_env || scatter_dd_fp64mul_env ||
        scatter_dd_fp32_ddmul_env || scatter_ozaki_env ||
        scatter_ozaki3_env || scatter_cublas_fp32_env ||
        scatter_cublas_ozaki_env || scatter_custom_gemm_env ||
        scatter_custom_gemm_wmma_env ||
        scatter_i8ozaki_env || scatter_i8gemm_env) {
      // Grid: blockDim.x covers l in [0, nThetaReduced); blockIdx.y is k;
      // blockIdx.z is config*ns_local + jF_local.
      const int TPB_L = 32;
      dim3 dd_blocks((nThetaReduced + TPB_L - 1) / TPB_L,
                     nZeta, n_cfg * ns_local);
      dim3 dd_tpb(TPB_L, 1, 1);
      if (scatter_i8gemm_env) {
        const int K_g = 16 * mpol;
        const int N_g = 16 * nThetaReduced;
        const int B_g = n_cfg * ns_local * nZeta;
        const int B_pad = (B_g + 63) & ~63;
        if (!S.d_i8b_W) {
          cuda_check(cudaMalloc(&S.d_i8b_W,
                                sizeof(double) * (size_t)K_g * N_g),
                     "alloc i8b W");
          cuda_check(cudaMalloc(&S.d_i8b_Wl,
                                (size_t)8 * K_g * N_g), "alloc i8b Wl");
          cuda_check(cudaMalloc(&S.d_i8b_eW, sizeof(int) * N_g),
                     "alloc i8b eW");
        }
        if (S.i8b_B_pad < B_pad) {
          if (S.d_i8b_Yl) cudaFree(S.d_i8b_Yl);
          if (S.d_i8b_eY) cudaFree(S.d_i8b_eY);
          cuda_check(cudaMalloc(&S.d_i8b_Yl,
                                (size_t)8 * (size_t)B_pad * (size_t)K_g),
                     "alloc i8b Yl");
          cuda_check(cudaMalloc(&S.d_i8b_eY, sizeof(int) * B_pad),
                     "alloc i8b eY");
          S.i8b_B_pad = B_pad;
        }
        if (!S.i8b_w_built) {
          int wt = 256;
          k_i8b_build_w<<<(K_g * N_g + wt - 1) / wt, wt, 0, st>>>(
              mpol, nThetaReduced, S.d_cosmu, S.d_sinmu, S.d_cosmum,
              S.d_sinmum, S.d_i8b_W, K_g, N_g);
          cuda_check(cudaGetLastError(), "k_i8b_build_w launch");
          k_i8b_slice_w<<<(N_g + wt - 1) / wt, wt, 0, st>>>(
              S.d_i8b_W, K_g, N_g, S.d_i8b_Wl, S.d_i8b_eW);
          cuda_check(cudaGetLastError(), "k_i8b_slice_w launch");
          S.i8b_w_built = true;
        }
        {
          int wt = 256;
          k_i8b_row_exp<<<(B_g + wt - 1) / wt, wt, 0, st>>>(
              S.n_config_max, ns_local, mpol, nZeta, S.d_Y, S.d_i8b_eY);
          cuda_check(cudaGetLastError(), "k_i8b_row_exp launch");
          int total = B_pad * K_g;
          if (i8_limbs == 4) {
            k_i8b_slice_y<4><<<(total + wt - 1) / wt, wt, 0, st>>>(
                S.n_config_max, ns_local, mpol, nZeta, S.d_Y, S.d_i8b_eY,
                S.d_i8b_Yl, B_pad);
          } else {
            k_i8b_slice_y<8><<<(total + wt - 1) / wt, wt, 0, st>>>(
                S.n_config_max, ns_local, mpol, nZeta, S.d_Y, S.d_i8b_eY,
                S.d_i8b_Yl, B_pad);
          }
          cuda_check(cudaGetLastError(), "k_i8b_slice_y launch");
          dim3 gb(B_pad / 64, nThetaReduced, 1);
          size_t gs = (size_t)(i8_limbs * 64 * 16 + i8_limbs * 16 * 16) +
                      sizeof(int) * (size_t)i8_limbs * 64 * 16 + 32;
          if (i8_limbs == 4) {
            k_i8b_gemm<4><<<gb, 128, gs, st>>>(
                S.n_config_max, ns_local, mpol, nZeta, nThetaReduced,
                nThetaEff, B_pad, S.d_i8b_Yl, S.d_i8b_eY, S.d_i8b_Wl,
                S.d_i8b_eW,
                S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
                S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
                S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
                S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o);
          } else {
            k_i8b_gemm<8><<<gb, 256, gs, st>>>(
                S.n_config_max, ns_local, mpol, nZeta, nThetaReduced,
                nThetaEff, B_pad, S.d_i8b_Yl, S.d_i8b_eY, S.d_i8b_Wl,
                S.d_i8b_eW,
                S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
                S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
                S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
                S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o);
          }
          cuda_check(cudaGetLastError(), "k_i8b_gemm launch");
          const int TPB_RC_G = 32;
          dim3 rcg_blocks((nThetaReduced + TPB_RC_G - 1) / TPB_RC_G,
                          nZeta, n_cfg * ns_local);
          dim3 rcg_tpb(TPB_RC_G, 1, 1);
          k_scatter_rcon_zcon_fp64<<<rcg_blocks, rcg_tpb, 0, st>>>(
              S.n_config_max, ns_local, mpol, nZeta, nThetaReduced,
              nThetaEff, S.d_Y, S.d_cosmu, S.d_sinmu, S.d_xmpq, S.d_sqrtSF,
              S.d_rCon, S.d_zCon);
          cuda_check(cudaGetLastError(),
                     "k_scatter_rcon_zcon_fp64 launch (i8gemm path)");
        }
      } else if (scatter_i8ozaki_env) {
        // int8 tensor-core dispatch with exact s32 accumulation; the
        // FP64 output needs no scalar recovery pass. One warp per band.
        const int TPB_I8 = 32 * i8_limbs;
        dim3 i8_blocks(1, nZeta, n_cfg * ns_local);
        dim3 i8_tpb(TPB_I8, 1, 1);
        size_t i8_smem =
            sizeof(double) * ((size_t)kBatch * (size_t)mpol +
                              4 * (size_t)mpol * (size_t)nThetaReduced +
                              (size_t)mpol) +
            (size_t)(i8_limbs * 16 * 48 + i8_limbs * 48 * 16) +
            sizeof(int) * ((size_t)i8_limbs * 16 * 16 + 16 + 16) + 32;
        if (i8_limbs == 4) {
          k_scatter_main_and_con_i8ozaki<4>
              <<<i8_blocks, i8_tpb, i8_smem, st>>>(
              S.n_config_max, ns_local, mpol, nZeta, nThetaReduced,
              nThetaEff,
              S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
              S.d_xmpq, S.d_sqrtSF,
              S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
              S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
              S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
              S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o);
        } else {
          k_scatter_main_and_con_i8ozaki<8>
              <<<i8_blocks, i8_tpb, i8_smem, st>>>(
              S.n_config_max, ns_local, mpol, nZeta, nThetaReduced,
              nThetaEff,
              S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
              S.d_xmpq, S.d_sqrtSF,
              S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
              S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
              S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
              S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o);
        }
        cuda_check(cudaGetLastError(),
                   "k_scatter_main_and_con_i8ozaki launch");
        const int TPB_L_RC_I8 = 32;
        dim3 rci_blocks((nThetaReduced + TPB_L_RC_I8 - 1) / TPB_L_RC_I8,
                        nZeta, n_cfg * ns_local);
        dim3 rci_tpb(TPB_L_RC_I8, 1, 1);
        k_scatter_rcon_zcon_fp64<<<rci_blocks, rci_tpb, 0, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_xmpq, S.d_sqrtSF,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(),
                   "k_scatter_rcon_zcon_fp64 launch (i8ozaki path)");
      } else if (scatter_custom_gemm_wmma_env) {
        // TF32 tensor-core dispatch via wmma::mma_sync. One block per
        // (cfg*ns_local, k) tile; 256 threads (8 warps). 3-slice Ozaki
        // produces 9 cross-product wmma chains × 6 K-chunks = 54
        // wmma::mma_sync calls per tile, distributed across 8 warps.
        // Shared memory:
        //   doubles: kBatch*mpol + 4*mpol*nThetaReduced + mpol = 700
        //   floats : 3*16*48 + 3*48*16 + 9*16*16 = 6912
        //   total  : 5600 + 27648 = 33248 bytes per block
        const int TPB_W = 256;
        dim3 wm_blocks(1, nZeta, n_cfg * ns_local);
        dim3 wm_tpb(TPB_W, 1, 1);
        size_t wm_smem = sizeof(double) * (
                             (size_t)kBatch * (size_t)mpol +
                             4 * (size_t)mpol * (size_t)nThetaReduced +
                             (size_t)mpol)
                         + sizeof(float) * (3 * 16 * 48 + 3 * 48 * 16 +
                                            9 * 16 * 16);
        // IR phase override: if staged IR is on and residual is above
        // threshold, force plain_tf32=1 for the fast descent. Otherwise
        // respect the env var.
        int ir_phase = GetIRPhase();
        int plain_tf32_arg = ir_phase ? 1 : scatter_tf32_plain_env;
        k_scatter_main_and_con_wmma_tf32<<<wm_blocks, wm_tpb, wm_smem, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            plain_tf32_arg,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
            S.d_xmpq, S.d_sqrtSF,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o);
        cuda_check(cudaGetLastError(),
                   "k_scatter_main_and_con_wmma_tf32 launch");
        // rCon/zCon trailing kernel: FP64 mults from the produced r1/z1.
        const int TPB_L_RC_W = 32;
        dim3 rcw_blocks((nThetaReduced + TPB_L_RC_W - 1) / TPB_L_RC_W,
                        nZeta, n_cfg * ns_local);
        dim3 rcw_tpb(TPB_L_RC_W, 1, 1);
        k_scatter_rcon_zcon_fp64<<<rcw_blocks, rcw_tpb, 0, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_xmpq, S.d_sqrtSF,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(),
                   "k_scatter_rcon_zcon_fp64 launch (wmma path)");
      } else if (scatter_custom_gemm_env) {
        // Custom Veltkamp-Dekker Tile GEMM. One block per (cfg*ns_local,
        // k) tile; TPB_CG threads cover the l-axis within nThetaReduced.
        // Shared memory carries the per-tile Y values + basis + xmpq.
        const int TPB_CG = 64;
        dim3 cg_blocks((nThetaReduced + TPB_CG - 1) / TPB_CG, nZeta,
                       n_cfg * ns_local);
        dim3 cg_tpb(TPB_CG, 1, 1);
        size_t cg_smem = sizeof(double) * (
            (size_t)kBatch * (size_t)mpol +
            4 * (size_t)mpol * (size_t)nThetaReduced +
            (size_t)mpol);
        k_scatter_main_and_con_custom_gemm<<<cg_blocks, cg_tpb, cg_smem, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
            S.d_xmpq, S.d_sqrtSF,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(),
                   "k_scatter_main_and_con_custom_gemm launch");
      } else if (scatter_cublas_ozaki_env) {
        // 4-GEMM Ozaki at GEMM level. Each FP64 operand split into FP32
        // hi/lo; 4 cuBLAS calls produce the four cross-products; DD-pair
        // unpack reassembles ~48-bit precision per output.
        if (!S.cublas) {
          if (cublasCreate(&S.cublas) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[fft_toroidal_cuda] cublasCreate failed\n");
            std::abort();
          }
          cublasSetStream(S.cublas, st);
        }
        const int M_ozaki = mpol * kBatch;
        const int N_ozaki = nThetaReduced * 16;
        const size_t B_ozaki = (size_t)n_cfg * (size_t)ns_local * (size_t)nZeta;
        if (S.scatter_basis_M != (size_t)M_ozaki ||
            S.scatter_basis_N != (size_t)N_ozaki ||
            !S.d_scatter_basis_hi) {
          for (float** p : { &S.d_scatter_basis_hi, &S.d_scatter_basis_lo,
                              &S.d_scatter_Y_hi, &S.d_scatter_Y_lo,
                              &S.d_scatter_out_hh, &S.d_scatter_out_hl,
                              &S.d_scatter_out_lh, &S.d_scatter_out_ll }) {
            if (*p) { cudaFree(*p); *p = nullptr; }
          }
          cuda_check(cudaMalloc(&S.d_scatter_basis_hi,
                                 sizeof(float) * M_ozaki * N_ozaki),
                     "alloc basis_hi");
          cuda_check(cudaMalloc(&S.d_scatter_basis_lo,
                                 sizeof(float) * M_ozaki * N_ozaki),
                     "alloc basis_lo");
          cuda_check(cudaMemsetAsync(S.d_scatter_basis_hi, 0,
                                      sizeof(float) * M_ozaki * N_ozaki, st),
                     "zero basis_hi");
          cuda_check(cudaMemsetAsync(S.d_scatter_basis_lo, 0,
                                      sizeof(float) * M_ozaki * N_ozaki, st),
                     "zero basis_lo");
          dim3 wb(mpol, nThetaReduced, 1);
          dim3 wt(1, 1, 1);
          k_scatter_basis_init_split<<<wb, wt, 0, st>>>(
              mpol, nThetaReduced, kBatch,
              S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
              S.d_scatter_basis_hi, S.d_scatter_basis_lo);
          cuda_check(cudaGetLastError(), "k_scatter_basis_init_split launch");
          S.scatter_basis_M = M_ozaki;
          S.scatter_basis_N = N_ozaki;
        }
        if (!S.d_scatter_Y_hi) {
          cuda_check(cudaMalloc(&S.d_scatter_Y_hi,
                                 sizeof(float) * B_ozaki * M_ozaki), "alloc Y_hi");
          cuda_check(cudaMalloc(&S.d_scatter_Y_lo,
                                 sizeof(float) * B_ozaki * M_ozaki), "alloc Y_lo");
          cuda_check(cudaMalloc(&S.d_scatter_out_hh,
                                 sizeof(float) * B_ozaki * N_ozaki), "alloc out_hh");
          cuda_check(cudaMalloc(&S.d_scatter_out_hl,
                                 sizeof(float) * B_ozaki * N_ozaki), "alloc out_hl");
          cuda_check(cudaMalloc(&S.d_scatter_out_lh,
                                 sizeof(float) * B_ozaki * N_ozaki), "alloc out_lh");
          cuda_check(cudaMalloc(&S.d_scatter_out_ll,
                                 sizeof(float) * B_ozaki * N_ozaki), "alloc out_ll");
        }
        const int TPB_K2 = 32;
        dim3 pk_blocks((nZeta + TPB_K2 - 1) / TPB_K2, ns_local, n_cfg);
        dim3 pk_tpb(TPB_K2, 1, 1);
        k_scatter_pack_Y_fp32_split<<<pk_blocks, pk_tpb, 0, st>>>(
            n_cfg, ns_local, mpol, kBatch, nZeta, S.d_Y,
            S.d_scatter_Y_hi, S.d_scatter_Y_lo);
        cuda_check(cudaGetLastError(),
                   "k_scatter_pack_Y_fp32_split launch");
        const float alpha1 = 1.0f, beta0 = 0.0f;
        auto run_gemm = [&](const float* A, const float* B, float* C,
                            const char* tag) {
          cublasStatus_t cs = cublasGemmEx(
              S.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
              N_ozaki, (int)B_ozaki, M_ozaki,
              &alpha1, A, CUDA_R_32F, N_ozaki,
              B, CUDA_R_32F, M_ozaki,
              &beta0, C, CUDA_R_32F, N_ozaki,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
          if (cs != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[fft_toroidal_cuda] Ozaki GEMM %s "
                                 "failed: %d\n", tag, (int)cs);
            std::abort();
          }
        };
        run_gemm(S.d_scatter_basis_hi, S.d_scatter_Y_hi,
                 S.d_scatter_out_hh, "hh");
        run_gemm(S.d_scatter_basis_lo, S.d_scatter_Y_hi,
                 S.d_scatter_out_hl, "hl");
        run_gemm(S.d_scatter_basis_hi, S.d_scatter_Y_lo,
                 S.d_scatter_out_lh, "lh");
        run_gemm(S.d_scatter_basis_lo, S.d_scatter_Y_lo,
                 S.d_scatter_out_ll, "ll");
        dim3 un_blocks(((nZeta * nThetaReduced) + TPB_K2 - 1) / TPB_K2,
                       ns_local, n_cfg);
        dim3 un_tpb(TPB_K2, 1, 1);
        k_scatter_unpack_out_ozaki<<<un_blocks, un_tpb, 0, st>>>(
            n_cfg, ns_local, nZeta, nThetaReduced, nThetaEff,
            S.d_scatter_out_hh, S.d_scatter_out_hl,
            S.d_scatter_out_lh, S.d_scatter_out_ll,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o);
        cuda_check(cudaGetLastError(),
                   "k_scatter_unpack_out_ozaki launch");
        const int TPB_L_RC2 = 32;
        dim3 rc_blocks((nThetaReduced + TPB_L_RC2 - 1) / TPB_L_RC2,
                       nZeta, n_cfg * ns_local);
        dim3 rc_tpb(TPB_L_RC2, 1, 1);
        k_scatter_rcon_zcon_fp64<<<rc_blocks, rc_tpb, 0, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_xmpq, S.d_sqrtSF,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(),
                   "k_scatter_rcon_zcon_fp64 launch");
      } else if (scatter_cublas_fp32_env) {
        // Lazy cuBLAS init.
        if (!S.cublas) {
          if (cublasCreate(&S.cublas) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "[fft_toroidal_cuda] cublasCreate failed\n");
            std::abort();
          }
          cublasSetStream(S.cublas, st);
        }
        const int M = mpol * kBatch;
        const int N = nThetaReduced * 16;
        const size_t B = (size_t)n_cfg * (size_t)ns_local * (size_t)nZeta;
        // Lazy basis + scratch buffers.
        if (S.scatter_basis_M != (size_t)M || S.scatter_basis_N != (size_t)N) {
          if (S.d_scatter_basis_fp32) { cudaFree(S.d_scatter_basis_fp32);
            S.d_scatter_basis_fp32 = nullptr; }
          if (S.d_scatter_Y_fp32) { cudaFree(S.d_scatter_Y_fp32);
            S.d_scatter_Y_fp32 = nullptr; }
          if (S.d_scatter_out_fp32) { cudaFree(S.d_scatter_out_fp32);
            S.d_scatter_out_fp32 = nullptr; }
          cuda_check(cudaMalloc(&S.d_scatter_basis_fp32,
                                 sizeof(float) * M * N), "alloc scatter_basis_fp32");
          cuda_check(cudaMemsetAsync(S.d_scatter_basis_fp32, 0,
                                      sizeof(float) * M * N, st),
                     "zero scatter_basis_fp32");
          dim3 wb(mpol, nThetaReduced, 1);
          dim3 wt(1, 1, 1);
          k_scatter_basis_init<<<wb, wt, 0, st>>>(
              mpol, nThetaReduced, kBatch,
              S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
              S.d_scatter_basis_fp32);
          cuda_check(cudaGetLastError(), "k_scatter_basis_init launch");
          S.scatter_basis_M = M;
          S.scatter_basis_N = N;
        }
        if (!S.d_scatter_Y_fp32) {
          cuda_check(cudaMalloc(&S.d_scatter_Y_fp32, sizeof(float) * B * M),
                     "alloc scatter_Y_fp32");
        }
        if (!S.d_scatter_out_fp32) {
          cuda_check(cudaMalloc(&S.d_scatter_out_fp32, sizeof(float) * B * N),
                     "alloc scatter_out_fp32");
        }
        // Pack Y_fp64 -> Y_fp32 in (B, M) layout.
        const int TPB_K = 32;
        dim3 pack_blocks((nZeta + TPB_K - 1) / TPB_K, ns_local, n_cfg);
        dim3 pack_tpb(TPB_K, 1, 1);
        k_scatter_pack_Y_fp32<<<pack_blocks, pack_tpb, 0, st>>>(
            n_cfg, ns_local, mpol, kBatch, nZeta, S.d_Y, S.d_scatter_Y_fp32);
        cuda_check(cudaGetLastError(), "k_scatter_pack_Y_fp32 launch");
        // GEMM: out[B, N] = Y_packed[B, M] * W[M, N], with row-major buffers.
        // cuBLAS operates in column-major layout. A row-major matrix X(R, C)
        // is the column-major X^T(C, R) with leading dimension C, so the
        // row-major product out = Y * W is computed as the column-major
        // product out^T(N, B) = W^T(N, M) * Y^T(M, B) by passing the buffers
        // unchanged with op_A = op_B = N:
        //   m = N, n = B, k = M
        //   A = W   (N x M column-major), lda = N
        //   B = Y   (M x B column-major), ldb = M
        //   C = out (N x B column-major), ldc = N
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t cs = cublasGemmEx(
            S.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            N, (int)B, M,
            &alpha,
            S.d_scatter_basis_fp32, CUDA_R_32F, N,
            S.d_scatter_Y_fp32, CUDA_R_32F, M,
            &beta,
            S.d_scatter_out_fp32, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (cs != CUBLAS_STATUS_SUCCESS) {
          std::fprintf(stderr, "[fft_toroidal_cuda] cublasGemmEx scatter "
                               "failed: %d\n", (int)cs);
          std::abort();
        }
        // Surface any async error from the GEMM launch.
        static bool checked_once = false;
        if (!checked_once) {
          cudaError_t e = cudaStreamSynchronize(st);
          if (e != cudaSuccess) {
            std::fprintf(stderr, "[fft_toroidal_cuda] cuBLAS GEMM async "
                                 "error: %s\n", cudaGetErrorString(e));
            std::abort();
          }
          std::fprintf(stderr, "[fft_toroidal_cuda] cuBLAS GEMM first-call "
                               "sync OK\n");
          checked_once = true;
        }
        // Unpack out_fp32 -> 16 production buffers.
        dim3 unpack_blocks(((nZeta * nThetaReduced) + TPB_K - 1) / TPB_K,
                           ns_local, n_cfg);
        dim3 unpack_tpb(TPB_K, 1, 1);
        k_scatter_unpack_out_fp32<<<unpack_blocks, unpack_tpb, 0, st>>>(
            n_cfg, ns_local, nZeta, nThetaReduced, nThetaEff,
            S.d_scatter_out_fp32,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o);
        cuda_check(cudaGetLastError(), "k_scatter_unpack_out_fp32 launch");
        // rCon/zCon trailing kernel: FP64 mults.
        const int TPB_L_RC = 32;
        dim3 rcon_blocks((nThetaReduced + TPB_L_RC - 1) / TPB_L_RC,
                         nZeta, n_cfg * ns_local);
        dim3 rcon_tpb(TPB_L_RC, 1, 1);
        k_scatter_rcon_zcon_fp64<<<rcon_blocks, rcon_tpb, 0, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu,
            S.d_xmpq, S.d_sqrtSF,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(), "k_scatter_rcon_zcon_fp64 launch");
      } else if (scatter_dd_fp64mul_env) {
        k_scatter_main_and_con_dd_fp64mul<<<dd_blocks, dd_tpb, 0, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
            S.d_xmpq, S.d_sqrtSF,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(), "k_scatter_main_and_con_dd_fp64mul launch");
      } else if (scatter_dd_fp32_ddmul_env) {
        k_scatter_main_and_con_dd_fp32_ddmul<<<dd_blocks, dd_tpb, 0, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
            S.d_xmpq, S.d_sqrtSF,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(), "k_scatter_main_and_con_dd_fp32_ddmul launch");
      } else if (scatter_ozaki3_env) {
        k_scatter_main_and_con_ozaki3_fp32<<<dd_blocks, dd_tpb, 0, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
            S.d_xmpq, S.d_sqrtSF,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(), "k_scatter_main_and_con_ozaki3_fp32 launch");
      } else if (scatter_ozaki_env) {
        k_scatter_main_and_con_ozaki_fp32<<<dd_blocks, dd_tpb, 0, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
            S.d_xmpq, S.d_sqrtSF,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(), "k_scatter_main_and_con_ozaki_fp32 launch");
      } else {
        k_scatter_main_and_con_dd_fp32<<<dd_blocks, dd_tpb, 0, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
            S.d_xmpq, S.d_sqrtSF,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(), "k_scatter_main_and_con_dd_fp32 launch");
      }
    } else if (scatter_v5_env) {
      static const int scatter_pack_env = []() {
        const char* e = std::getenv("VMECPP_SCATTER_PACK");
        int v = (e && std::atoi(e) == 0) ? 0 : 1;  // default ON
        if (!v)
          std::fprintf(stderr, "[fft_toroidal_cuda] scatter warp-pack disabled "
                               "(VMECPP_SCATTER_PACK=0)\n");
        return v;
      }();
      const int ks_per_warp = (nThetaReduced > 0) ? (32 / nThetaReduced) : 0;
      if (scatter_pack_env && ks_per_warp >= 2 && (32 % nThetaReduced) == 0) {
        // Warp-pack ks_per_warp zeta planes so all 32 lanes compute; the Y
        // tile is replicated per packed plane (ks_per_warp * mpol * kBatch).
        dim3 pk_blocks(1, (nZeta + ks_per_warp - 1) / ks_per_warp, z_blocks);
        dim3 pk_tpb(32, WARPS_PER_BLOCK, 1);
        size_t pk_smem = sizeof(double) * (size_t)WARPS_PER_BLOCK *
                         (size_t)ks_per_warp * (size_t)mpol * (size_t)kBatch;
        k_scatter_main_and_con_v5_packed<<<pk_blocks, pk_tpb, pk_smem, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
            S.d_xmpq, S.d_sqrtSF,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(),
                   "k_scatter_main_and_con_v5_packed launch");
      } else {
        // Shared memory per block: blockDim.y warps * mpol * kBatch doubles.
        size_t smem_bytes = sizeof(double) * v4_tpb.y * mpol * kBatch;
        k_scatter_main_and_con_v5<<<v4_blocks, v4_tpb, smem_bytes, st>>>(
            S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
            S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
            S.d_xmpq, S.d_sqrtSF,
            S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
            S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
            S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
            S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
            S.d_rCon, S.d_zCon);
        cuda_check(cudaGetLastError(), "k_scatter_main_and_con_v5 launch");
      }
    } else {
      k_scatter_main_and_con_v4<<<v4_blocks, v4_tpb, 0, st>>>(
          S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
          S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
          S.d_xmpq, S.d_sqrtSF,
          S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
          S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
          S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
          S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o,
          S.d_rCon, S.d_zCon,
          S.d_active_per_cfg);
      cuda_check(cudaGetLastError(), "k_scatter_main_and_con_v4 launch");
    }
    S.TKEnd(CudaToroidalState::TK_SCATTER);
  } else {
    k_scatter_main<<<scat_blocks, scat_tpb, 0, st>>>(
        S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
        S.d_Y, S.d_cosmu, S.d_sinmu, S.d_cosmum, S.d_sinmum,
        S.d_r1_e, S.d_r1_o, S.d_ru_e, S.d_ru_o,
        S.d_rv_e, S.d_rv_o, S.d_z1_e, S.d_z1_o,
        S.d_zu_e, S.d_zu_o, S.d_zv_e, S.d_zv_o,
        S.d_lu_e, S.d_lu_o, S.d_lv_e, S.d_lv_o);
    cuda_check(cudaGetLastError(), "k_scatter_main launch");
    dim3 con_blocks((nThetaReduced + SCAT_TPB - 1) / SCAT_TPB, nZeta,
                    ns_con_local * S.n_config_max);
    k_scatter_con<<<con_blocks, scat_tpb, 0, st>>>(
        S.n_config_max, ns_local, ns_con_local,
        mpol, nZeta, nThetaReduced, nThetaEff,
        nsMinF_offset_in_local, S.d_Y, S.d_cosmu, S.d_sinmu,
        S.d_xmpq, S.d_sqrtSF, S.d_rCon, S.d_zCon);
    cuda_check(cudaGetLastError(), "k_scatter_con launch");
  }

scatter_done:;

  // Per-iteration D2H reduction: the previous 1.2 MB (at N=1) / 76 MB (at N=64) D2H +
  // 1.2 MB host scatter is replaced with a tiny 6-double extract. The only
  // live host consumers of r1_e/r1_o/z1_e under VMECPP_USE_CUDA are
  // SetRadialExtent (r_outer, r_inner at the LCFS, theta=0 and theta=last)
  // and SetGeometricOffset (r_00, z_00 at the axis). All other host reads
  // of m_geometry.* are inside CPU-fallback #else branches (dead code under
  // CUDA). The output phase reads device buffers via FlushForOutputCuda at
  // end-of-run. Indices use config 0's layout: (ns_local-1)*nZnT for the
  // LCFS surface base; +0 / +(nThetaReduced-1) for the two theta points; 0
  // for the axis r1_e[0]/z1_e[0].
  k_extract_geom_scalars<<<1, 1, 0, st>>>(
      S.d_r1_e, S.d_r1_o, S.d_z1_e,
      outer_idx_pre, inner_idx_pre, S.d_geom_scalars);
  cuda_check(cudaGetLastError(), "k_extract_geom_scalars launch");
  cuda_check(cudaMemcpyAsync(S.h_geom_scalars, S.d_geom_scalars,
                             6 * sizeof(double),
                             cudaMemcpyDeviceToHost, st),
             "d2h geom_scalars");

  if (capture_then_launch) {
    cuda_check(cudaStreamEndCapture(st, &S.fwd_graph), "end capture fwd graph");
    cuda_check(cudaGraphInstantiate(&S.fwd_graph_exec, S.fwd_graph, nullptr,
                                     nullptr, 0),
               "graph instantiate fwd");
    S.fwd_graph_captured = true;
    // Launch the graph to run the recorded work; during capture mode the
    // kernels were recorded but did not execute.
    cuda_check(cudaGraphLaunch(S.fwd_graph_exec, st), "graph launch fwd (first)");
  }

fwd_chain_done:
  // The cudaStreamSynchronize that would otherwise close the
  // forward chain is omitted here; the asynchronous device-to-host
  // transfer of S.h_geom_scalars remains queued on the stream and
  // is drained by the next natural synchronization point, namely
  // the tau-minmax synchronization that ComputeJacobianCuda
  // performs. The corresponding host writes into the RealSpaceGeometry
  // members r1_e, r1_o, and z1_e are emitted by
  // FlushFwdGeomScalarsToHost, which the IdealMhdModel update body
  // invokes after ComputeJacobianCuda has returned. The deferred
  // commit retains only the integer indices required to identify
  // the destination slots, since the RealSpaceGeometry container is
  // a stack-local in the caller and saving its address would
  // produce a dangling reference.
  S.fwd_geom_pending = true;
  S.fwd_geom_outer_idx = outer_idx_pre;
  S.fwd_geom_inner_idx = inner_idx_pre;
  DiagCfg01DiffCuda(S.d_r1_e, ns_local * nZeta * nThetaEff, "fwd:r1_e");
}

// ============================================================================
// ForcesToFourier3DSymmFastPoloidalCuda: real-kernel inverse FFT port.
// Reads device buffers populated by the device-resident chain (computeMHDForces,
// assembleTotalForces, hybridLambdaForce). The spectral outputs remain in the
// device shadow buffers for the device-resident preconditioner chain; host
// copies are refreshed at the consolidated flush sites.
// ============================================================================
void ForcesToFourier3DSymmFastPoloidalCuda(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq_host,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces) {
  auto& S = State();
  S.OneTimeInit(s.nZeta, s.nfp, s.mpol);

  const int ns_local = rp.nsMaxF1 - rp.nsMinF1;
  const int ns_force_local = rp.nsMaxF - rp.nsMinF;
  const int ns_con_local = rp.nsMaxFIncludingLcfs - rp.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  const int nhalf = s.nZeta / 2 + 1;
  const int nZeta = s.nZeta;
  const int nfp = s.nfp;
  const int nThetaReduced = s.nThetaReduced;
  const int nThetaEff = s.nThetaEff;

  if (ns_local <= 0) return;

  // Real-kernel inverse FFT path. The pre-real-kernel fallback (which called
  // out to the CPU FFTX or partial-DFT path) is removed since the device path
  // is the only consumer; gating switch retained for diagnostic asymmetry.
  constexpr bool kUseRealKernel = true;
  if (!kUseRealKernel) {
    (void)ns_force_local; (void)ns_con_local; (void)mpol; (void)ntor;
    (void)nhalf; (void)nZeta; (void)nfp; (void)nThetaReduced; (void)nThetaEff;
    return;
  }

  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  int nsMinF_to_nsMinF1 = rp.nsMinF - rp.nsMinF1;

  // Stage 1: k_inverse_fill populates Y[jF, m, q, k] from device force arrays
  // with poloidal-i basis projection.
  // Batched execution: z-dim = config * ns_local + jF_local.
  {
    const int TPB = 32;
    dim3 b((nZeta + TPB - 1) / TPB, mpol * kBatch,
           ns_local * S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_inverse_fill<<<b, t, 0, st>>>(
        S.n_config_max, ns_local, mpol, nZeta, nThetaReduced, nThetaEff,
        s.lthreed, nsMinF_to_nsMinF1, ns_force_local, ns_con_local,
        S.d_xmpq,
        S.d_cosmui, S.d_sinmui, S.d_cosmumi, S.d_sinmumi,
        S.d_armn_e, S.d_armn_o, S.d_azmn_e, S.d_azmn_o,
        S.d_brmn_e, S.d_brmn_o, S.d_bzmn_e, S.d_bzmn_o,
        S.d_blmn_e, S.d_blmn_o,
        S.d_crmn_e, S.d_crmn_o, S.d_czmn_e, S.d_czmn_o,
        S.d_clmn_e, S.d_clmn_o,
        S.d_frcon_e, S.d_frcon_o, S.d_fzcon_e, S.d_fzcon_o,
        S.d_Y);
    cuda_check(cudaGetLastError(), "k_inverse_fill launch");
  }

  // Stage 2: forward FFT (D2Z) on (ns_local × mpol × kBatch) batches of
  // length nZeta=24, producing complex output X[jF, m, q, n] for n in [0,
  // nhalf=13).
  //
  // Hand-coded radix-8x3 forward Fourier transform as an opt-in
  // alternative to cufftExecD2Z. The hand-coded kernel is amenable
  // to CUDA stream capture, whereas the cuFFT call is not, which
  // permits a graph-captured forward chain to enclose the
  // transform. Enablement is governed by the VMECPP_FWD_FFT_RADIX
  // environment variable; the default is disabled because the
  // hand-coded path does not match cuFFT's wall throughput at the
  // canonical problem shape under the current configuration. The
  // factorization is specific to transform length 24; other nZeta
  // values stay on cuFFT.
  static int fwd_fft_radix_env = -1;
  if (fwd_fft_radix_env < 0) {
    const char* e = std::getenv("VMECPP_FWD_FFT_RADIX");
    fwd_fft_radix_env = (e && std::atoi(e) > 0) ? 1 : 0;
    if (fwd_fft_radix_env) {
      std::fprintf(stderr, "[fft_toroidal_cuda] forward radix-8x3 DFT "
                           "ENABLED (VMECPP_FWD_FFT_RADIX=1)\n");
    }
  }
  if (fwd_fft_radix_env && nZeta != 24) {
    static bool fwd_radix_shape_warned = false;
    if (!fwd_radix_shape_warned) {
      fwd_radix_shape_warned = true;
      std::fprintf(stderr,
                   "[fft_toroidal_cuda] VMECPP_FWD_FFT_RADIX=1 requires "
                   "nZeta = 24 (this input has nZeta = %d); using cuFFT\n",
                   nZeta);
    }
  }
  if (fwd_fft_radix_env && nZeta == 24) {
    constexpr int FFTS_PER_BLOCK = 8;
    int total_batches = S.n_config_max * ns_local * mpol * kBatch;
    dim3 r_grid((total_batches + FFTS_PER_BLOCK - 1) / FFTS_PER_BLOCK, 1, 1);
    dim3 r_tpb(32, FFTS_PER_BLOCK, 1);
    // smem per FFT: 24 (real input) + 48 (T_re + T_im) = 72 doubles.
    size_t smem = sizeof(double) * 72 * FFTS_PER_BLOCK;
    k_forward_dft_24_radix83<<<r_grid, r_tpb, smem, st>>>(
        total_batches, nZeta, nhalf, S.d_Y, S.d_X);
    cuda_check(cudaGetLastError(), "k_forward_dft_24_radix83 launch");
  } else if (nZeta == 1) {
    // Length-1 toroidal transform: the identity (real value as the DC).
    int total_batches = S.n_config_max * ns_local * mpol * kBatch;
    int tpb = 256;
    k_d2z_identity_nzeta1<<<(total_batches + tpb - 1) / tpb, tpb, 0, st>>>(
        total_batches, S.d_Y, S.d_X);
    cuda_check(cudaGetLastError(), "k_d2z_identity_nzeta1");
  } else {
    S.TKBegin(CudaToroidalState::TK_CUFFT_FWD);
    cufft_check(cufftExecD2Z(S.cufft_plan_r2c, S.d_Y, S.d_X), "cufftExecD2Z");
    S.TKEnd(CudaToroidalState::TK_CUFFT_FWD);
  }

  // Stage 3: k_inverse_scatter populates spec arrays from X.
  // jMaxRZ from CPU: min(rp.nsMaxF, fc.ns - 1), bumped to ns on lfreeb+active.
  // jMinL is the lambda-write floor (constant = 1 in CPU code).
  int jMaxRZ_global = std::min(rp.nsMaxF, fc.ns - 1);
  if (fc.lfreeb &&
      (vacuum_pressure_state == VacuumPressureState::kInitialized ||
       vacuum_pressure_state == VacuumPressureState::kActive)) {
    jMaxRZ_global = std::min(rp.nsMaxF, fc.ns);
  }
  int jMaxRZ_local = jMaxRZ_global - rp.nsMinF1;
  int jMinL_local = 1 - rp.nsMinF1;
  if (jMinL_local < 0) jMinL_local = 0;
  if (jMaxRZ_local > ns_local) jMaxRZ_local = ns_local;
  if (jMaxRZ_local < 0) jMaxRZ_local = 0;
  // Batched execution: z-dim = config * ns_local + jF_local.
  {
    const int TPB = 32;
    dim3 b((ntor + TPB) / TPB, mpol, ns_local * S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_inverse_scatter<<<b, t, 0, st>>>(
        S.n_config_max, ns_local, mpol, ntor, nhalf, nfp, nZeta, s.lthreed,
        rp.nsMinF1, jMaxRZ_local, jMinL_local,
        S.d_X, S.d_nscale,
        S.d_frcc, S.d_frss, S.d_fzsc, S.d_fzcs, S.d_flsc, S.d_flcs);
    cuda_check(cudaGetLastError(), "k_inverse_scatter launch");
  }
  DiagCfg01DiffCuda(S.d_frcc, ns_local * mpol * (ntor + 1), "inv:frcc");

  // In CUDA mode, host m_physical_forces is never read again: DecomposeAndConstrainCuda
  // reads S.d_frcc/etc. on the same stream, so kernel ordering is enforced without a sync.
  (void)m_physical_forces;
}

// ============================================================================
// ComputeRuZuFullCuda: post-forward-FFT combine producing ruFull/zuFull on
// device + D2H to host (so downstream CPU code that reads these stays correct).
// Called once per geometryFromFourier after the forward FFT.
// ============================================================================
void ComputeRuZuFullCuda(const RadialPartitioning& r, const Sizes& s,
                          Eigen::VectorXd& ruFull, Eigen::VectorXd& zuFull) {
  auto& S = State();
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int nZnT = s.nZnT;
  if (ns_con_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  S.EnsureConstraintMultiplierBuffers(r.nsMaxF - r.nsMinF, ns_con_local, nZnT);

  int nsMinF_to_nsMinF1 = r.nsMinF - r.nsMinF1;
  // Batched execution: z-dim covers n_config_max configs.
  const int TPB = 64;
  dim3 b((nZnT + TPB - 1) / TPB, ns_con_local, S.n_config_max);
  dim3 t(TPB, 1, 1);
  k_compute_ru_zu_full<<<b, t, 0, S.stream>>>(
      S.n_config_max, S.ns_local_cached, ns_con_local, nZnT,
      nsMinF_to_nsMinF1,
      S.d_ru_e, S.d_ru_o, S.d_zu_e, S.d_zu_o, S.d_sqrtSF,
      S.d_ruFull, S.d_zuFull);
  cuda_check(cudaGetLastError(), "k_compute_ru_zu_full launch");

  // ruFull/zuFull stay on device; downstream ConstraintForceMultiplier /
  // EffectiveConstraintForce / AssembleTotalForces read d_ruFull / d_zuFull.
  (void)ruFull; (void)zuFull;
}

// ============================================================================
// DecomposeAndConstrainCuda
// Bridges S.d_frcc/etc. (m_physical_f device shadow from the inverse FFT) into
// the decomposed shadow S.d_decomposed_frcc/etc. (m_decomposed_f mirror) by
// running the three CPU-only steps from update():
//   m_physical_f.decomposeInto(m_decomposed_f, scalxc)
//   m_decomposed_f.m1Constraint(scalingFactor=1/sqrt(2))
//   m_decomposed_f.zeroZForceForM1()
// D2Hs the result to host m_decomposed_f for the subsequent CPU residuals call.
// stellarator-symmetric (lasym=false) only.
// ============================================================================
void DecomposeAndConstrainCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double m1ScalingFactor,
    const Eigen::VectorXd& scalxc,
    double* dec_frcc_host, double* dec_frss_host,
    double* dec_fzsc_host, double* dec_fzcs_host,
    double* dec_flsc_host, double* dec_flcs_host) {
  auto& S = State();
  int ns_dec_local = (r.nsMaxF1 == fc.ns) ? (fc.ns - r.nsMinF) : (r.nsMaxF - r.nsMinF);
  int ns_force_local = r.nsMaxF - r.nsMinF;
  int mpol = s.mpol;
  int ntor = s.ntor;
  int scalxc_len = static_cast<int>(scalxc.size());
  if (ns_dec_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;

  S.EnsureDecomposedForcesBuffers(ns_dec_local, mpol, ntor);
  S.EnsureScalxcBuffer(scalxc_len);

  if (!S.scalxc_staged) {
    for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
      cuda_check(cudaMemcpyAsync(S.d_scalxc + (size_t)cfg * scalxc_len,
                                  scalxc.data(),
                                  sizeof(double) * scalxc_len,
                                  cudaMemcpyHostToDevice, st),
                 "h2d scalxc (broadcast)");
    }
    S.scalxc_staged = true;
  }

  int nsMin_to_nsMinF1 = r.nsMinF - r.nsMinF1;

  // Stage 1: decompose (scale by scalxc).
  // Batched execution: z-dim = config * ns_dec_local + jF_dec.
  {
    const int TPB = 16;
    dim3 b((ntor + 1 + TPB - 1) / TPB, mpol,
           ns_dec_local * S.n_config_max);
    dim3 t(TPB, 1, 1);
    S.TKBegin(CudaToroidalState::TK_DECOMPOSE);
    k_decompose_into<<<b, t, 0, st>>>(
        S.n_config_max, ns_dec_local, S.ns_local_cached,
        mpol, ntor, nsMin_to_nsMinF1, s.lthreed, S.d_scalxc,
        S.d_frcc, S.d_frss, S.d_fzsc, S.d_fzcs, S.d_flsc, S.d_flcs,
        S.d_decomposed_frcc, S.d_decomposed_frss, S.d_decomposed_fzsc,
        S.d_decomposed_fzcs, S.d_decomposed_flsc, S.d_decomposed_flcs);
    cuda_check(cudaGetLastError(), "k_decompose_into launch");
    S.TKEnd(CudaToroidalState::TK_DECOMPOSE);
  }

  // Stage 2+3 fused: m=1 constraint (frss update) + zero Z force at m=1.
  // The original m1 kernel's fzcs output was dead code (overwritten by
  // zero_z_force in stage 3). The fused kernel skips that wasted store and
  // saves one launch per iter.
  // Batched execution: z-dim covers n_config_max configs.
  if (s.lthreed && ns_force_local > 0) {
    const int TPB = 32;
    dim3 b((ntor + 1 + TPB - 1) / TPB, ns_force_local, S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_m1_constraint_and_zero<<<b, t, 0, st>>>(
        S.n_config_max, S.ns_local_cached, ns_force_local, mpol, ntor,
        m1ScalingFactor,
        S.d_decomposed_frss, S.d_decomposed_fzcs);
    cuda_check(cudaGetLastError(), "k_m1_constraint_and_zero launch");
  }
  DiagCfg01DiffCuda(S.d_decomposed_frcc, S.ns_local_cached * mpol * (ntor + 1),
                    "dec:dec_frcc");

  // D2Hs deferred: device shadow S.d_decomposed_* is the source of truth
  // until FlushDecomposedToHostCuda runs at the end of residue(). Stream
  // ordering keeps subsequent wrappers (ApplyM1/Lambda/RZ on the same
  // stream) consistent without an explicit sync.
  (void)dec_frcc_host; (void)dec_frss_host; (void)dec_fzsc_host;
  (void)dec_fzcs_host; (void)dec_flsc_host; (void)dec_flcs_host;
}

// ============================================================================
// ApplyM1PreconditionerCuda
// ============================================================================
void ApplyM1PreconditionerCuda(
    const RadialPartitioning& r, const Sizes& s,
    const Eigen::VectorXd& ard, const Eigen::VectorXd& brd,
    const Eigen::VectorXd& azd, const Eigen::VectorXd& bzd,
    double* frss_host, double* fzcs_host) {
  if (!s.lthreed) return;  // quick return if neither lthreed nor lasym
  auto& S = State();
  const int ns_force_local = r.nsMaxF - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  if (ns_force_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  // The preconditioner matrix coefficients consumed by
  // k_apply_m1_preconditioner are produced upstream by
  // ComputePreconditioningMatrixCuda directly in the device buffers
  // d_pmat_ard, d_pmat_brd, d_pmat_azd, and d_pmat_bzd. The host
  // parameters ard, brd, azd, and bzd are retained in the function
  // signature for the sake of callers on the CPU-only path; under
  // CUDA they are read straight from device memory and the host
  // copies remain unused, which eliminates the four host-to-device
  // transfers that the prior path would have issued per iteration.
  (void)ard; (void)brd; (void)azd; (void)bzd;
  // frss/fzcs read/written directly on the DECOMPOSED shadow populated by
  // DecomposeAndConstrainCuda on the same stream; no H2D round-trip.
  // Batched execution: z-dim covers n_config_max configs.
  size_t spec_bytes = sizeof(double) * ns_force_local * mpol * (ntor + 1);
  const int TPB = 32;
  dim3 b((ntor + 1 + TPB - 1) / TPB, ns_force_local, S.n_config_max);
  dim3 t(TPB, 1, 1);
  // TK timing safe when graphs are auto-disabled (timing-on path).
  S.TKBegin(CudaToroidalState::TK_APPLY_M1);
  k_apply_m1_preconditioner<<<b, t, 0, st>>>(
      S.n_config_max, S.ns_local_cached, ns_force_local, mpol, ntor,
      S.d_pmat_ard, S.d_pmat_brd, S.d_pmat_azd, S.d_pmat_bzd,
      S.d_decomposed_frss, S.d_decomposed_fzcs,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_apply_m1 launch");
  S.TKEnd(CudaToroidalState::TK_APPLY_M1);
  DiagCfg01DiffCuda(S.d_decomposed_frss,
                    S.ns_local_cached * mpol * (ntor + 1), "m1:dec_frss");
  // D2H + sync deferred to end-of-residue() FlushDecomposedToHostCuda.
  (void)frss_host; (void)fzcs_host; (void)spec_bytes;
}

// Brings the device state into the new multigrid stage before iteration
// 1's geometry pipeline: the lazy Reshape (previous-stage d_pts_x and
// d_scalxc snapshots, stage-sized buffers including d_specs_block),
// scalxc staging, and PerformTimeStepCuda's init section (multigrid
// upscale / per-cfg dec_x load) without a time step. Idempotent: with
// the shape cache current and pts_x_initialized set, every section is a
// no-op.
void PrepareStagePtsXCuda(
    const RadialPartitioning& r, const Sizes& s,
    const FourierBasisFastPoloidal& fb, const FlowControl& fc,
    const Eigen::VectorXd& scalxc,
    double* m_dec_x_rcc, double* m_dec_x_rss,
    double* m_dec_x_zsc, double* m_dec_x_zcs,
    double* m_dec_x_lsc, double* m_dec_x_lcs) {
  auto& S = State();
  S.OneTimeInit(s.nZeta, s.nfp, s.mpol);
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  if (ns_local <= 0 || ns_con_local <= 0) return;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  const int nhalf = s.nZeta / 2 + 1;
  {
    // Same lazy-Reshape trigger as FourierToReal3DSymmFastPoloidalCuda;
    // the stage's first CUDA touch must be the Reshape, which snapshots
    // the previous stage's d_pts_x and d_scalxc and re-allocates the
    // stage-sized buffers.
    std::lock_guard<std::mutex> lk(S.mu);
    const int n_cfg = GetNConfigMaxCuda();
    if (S.ns_local_cached != ns_local ||
        S.ns_con_local_cached != ns_con_local ||
        S.mpol_cached != mpol || S.ntor_cached != ntor ||
        S.nhalf_cached != nhalf || S.nZeta_cached != s.nZeta ||
        S.nThetaReduced_cached != s.nThetaReduced ||
        S.nThetaEff_cached != s.nThetaEff ||
        S.n_config_max != n_cfg) {
      S.Reshape(ns_local, ns_con_local, mpol, ntor, nhalf, s.nZeta,
                s.nThetaReduced, s.nThetaEff, n_cfg);
      S.StageBasis(nhalf, mpol, s.nThetaReduced, fb.nscale.data(),
                   fb.cosmu.data(), fb.sinmu.data(), fb.cosmum.data(),
                   fb.sinmum.data());
      S.StageBasisI(mpol, s.nThetaReduced, fb.cosmui.data(),
                    fb.sinmui.data(), fb.cosmumi.data(), fb.sinmumi.data());
      S.StageToroidalBasis(s.nZeta, s.nnyq2 + 1, fb.cosnv.data(),
                           fb.sinnv.data());
      S.StageDftBasis(ntor, s.nZeta, fb.nscale.data());
      S.StageInverseDftBasis(nhalf, s.nZeta);
      S.EnsureFourierForcesBuffers(ns_local, mpol, ntor);
    }
    S.EnsurePTSBuffers(ns_con_local, ns_local, mpol, ntor);
    const int scalxc_len = static_cast<int>(scalxc.size());
    S.EnsureScalxcBuffer(scalxc_len);
    if (!S.scalxc_staged) {
      for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
        cuda_check(cudaMemcpyAsync(S.d_scalxc + (size_t)cfg * scalxc_len,
                                    scalxc.data(),
                                    sizeof(double) * scalxc_len,
                                    cudaMemcpyHostToDevice, S.stream),
                   "h2d scalxc (PrepareStagePtsX, broadcast)");
      }
      S.scalxc_staged = true;
    }
  }
  g_pts_init_only = true;
  PerformTimeStepCuda(r, s, fc,
                      /*velocity_scale=*/0.0, /*conjugation_parameter=*/0.0,
                      /*time_step=*/0.0, /*fnorm1=*/0.0, /*iter_phase=*/0,
                      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                      m_dec_x_rcc, m_dec_x_rss, m_dec_x_zsc, m_dec_x_zcs,
                      m_dec_x_lsc, m_dec_x_lcs);
  g_pts_init_only = false;
}

// ============================================================================
// RecomposeToPhysicalCuda
//
// Replaces the host triplet at the start of IdealMhdModel::update():
//   m_decomposed_x.decomposeInto(m_physical_x, m_p_.scalxc);
//   m_physical_x.m1Constraint(1.0);
//   m_physical_x.extrapolateTowardsAxis();
//
// Reads d_pts_x_* (kept device-resident across iters by PerformTimeStepCuda),
// writes d_specs_block sections (d_rmncc/d_rmnss/d_zmnsc/d_zmncs/d_lmnsc/
// d_lmncs). After this call, d_specs_block is the device-side m_physical_x
// and CudaForward's H2D specs_block can be skipped (specs_populated_from_device
// flag).
//
// First-call init: H2D host m_decomposed_x → d_pts_x. Subsequent calls read
// the device-resident d_pts_x updated by PerformTimeStepCuda at end of last
// iter. scalxc staging is shared with DecomposeAndConstrainCuda's stage flag.
// ============================================================================
void RecomposeToPhysicalCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    const Eigen::VectorXd& scalxc,
    const double* m_dec_x_rcc, const double* m_dec_x_rss,
    const double* m_dec_x_zsc, const double* m_dec_x_zcs,
    const double* m_dec_x_lsc, const double* m_dec_x_lcs) {
  auto& S = State();
  // Caller must gate on iter2 >= 2 so Reshape (which allocates d_specs_block
  // and sets d_rmncc/etc.) has already run from a previous CudaForward call.
  // Iter 1's CudaForward triggers Reshape; from iter 2 onward the spec
  // pointers are valid for our writes here.
  if (!S.d_specs_block) {
    // Defensive: caller violated gate. Fall through to nothing; the host
    // triplet in IdealMhdModel::update will still have computed m_physical_x
    // and CudaForward will do its full H2D.
    return;
  }
  const int ns_local = r.nsMaxF1 - r.nsMinF1;
  const int ns_con_local = r.nsMaxFIncludingLcfs - r.nsMinF;
  const int mpol = s.mpol;
  const int ntor = s.ntor;
  const int scalxc_len = static_cast<int>(scalxc.size());
  if (ns_local <= 0) return;
  std::lock_guard<std::mutex> lk(S.mu);
  cudaStream_t st = S.stream;
  S.EnsurePTSBuffers(ns_con_local, ns_local, mpol, ntor);
  S.EnsureScalxcBuffer(scalxc_len);
  (void)fc;


  // Stage scalxc (shared with DecomposeAndConstrainCuda's flag).
  if (!S.scalxc_staged) {
    for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
      cuda_check(cudaMemcpyAsync(S.d_scalxc + (size_t)cfg * scalxc_len,
                                  scalxc.data(),
                                  sizeof(double) * scalxc_len,
                                  cudaMemcpyHostToDevice, st),
                 "h2d scalxc (RecomposeToPhysical, broadcast)");
    }
    S.scalxc_staged = true;
  }

  // First-iter init: H2D host m_decomposed_x → d_pts_x. Subsequent iters
  // skip this; d_pts_x persists from PerformTimeStepCuda's update.
  //
  // Distinct-mode override (VMECPP_BATCH_DEC_X_FILE): when the per-cfg
  // decomposed_x file is set, pybind has already extracted the post-init
  // decomposed_x_[0] state for each cfg and written it to disk. We load
  // the full N * 6 * one_spec_doubles payload and memcpy per-cfg slices
  // into d_pts_x_*, overriding the seed-broadcast path that would otherwise
  // overwrite per-cfg initialization with single-cfg host data. The file
  // format mirrors batch_inputs: int32 header (N, ns_local, mpol, ntor)
  // followed by [sp][cfg][specs...] in row-major double-precision layout.
  size_t x_bytes_one = sizeof(double) * ns_local * mpol * (ntor + 1);
  if (!S.pts_x_initialized) {
    size_t one_spec_doubles = (size_t)ns_local * mpol * (ntor + 1);
    const char* dec_x_path = std::getenv("VMECPP_BATCH_DEC_X_FILE");
    bool dec_x_loaded = false;
    std::vector<double> dec_x_host_buf;
    if (!g_batch_dec_x_mem.empty() &&
        g_batch_mem_shape[0] == S.n_config_max &&
        g_batch_mem_shape[1] == ns_local && g_batch_mem_shape[2] == mpol &&
        g_batch_mem_shape[3] == ntor && S.n_config_max > 1) {
      dec_x_host_buf = g_batch_dec_x_mem;
      dec_x_loaded = true;
    }
    if (!dec_x_loaded && dec_x_path && *dec_x_path && S.n_config_max > 1) {
      FILE* f = std::fopen(dec_x_path, "rb");
      if (f) {
        int32_t header[4] = {0, 0, 0, 0};
        if (std::fread(header, sizeof(int32_t), 4, f) == 4 &&
            header[0] == S.n_config_max &&
            header[1] == ns_local &&
            header[2] == mpol &&
            header[3] == ntor) {
          size_t total_doubles = (size_t)6 * S.n_config_max * one_spec_doubles;
          dec_x_host_buf.resize(total_doubles);
          size_t got = std::fread(dec_x_host_buf.data(), sizeof(double),
                                   total_doubles, f);
          dec_x_loaded = (got == total_doubles);
        } else {
          std::fprintf(stderr,
              "[fft_toroidal_cuda] dec_x file header mismatch in %s "
              "(got N=%d ns=%d mpol=%d ntor=%d; expected N=%d ns=%d "
              "mpol=%d ntor=%d); falling back to broadcast\n",
              dec_x_path, header[0], header[1], header[2], header[3],
              S.n_config_max, ns_local, mpol, ntor);
        }
        std::fclose(f);
      }
    }
    double* dst_x[6] = {S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
                        S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs};
    if (dec_x_loaded) {
      for (int sp = 0; sp < 6; ++sp) {
        for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
          const double* src =
              dec_x_host_buf.data() +
              (size_t)sp * S.n_config_max * one_spec_doubles +
              (size_t)cfg * one_spec_doubles;
          cuda_check(cudaMemcpyAsync(
              dst_x[sp] + (size_t)cfg * one_spec_doubles,
              src, x_bytes_one, cudaMemcpyHostToDevice, st),
              "h2d per-cfg dec_x (Recompose)");
        }
      }
      std::fprintf(stderr,
          "[fft_toroidal_cuda] loaded per-cfg dec_x from %s into "
          "d_pts_x (N=%d ns=%d mpol=%d ntor=%d)\n",
          (dec_x_path && *dec_x_path) ? dec_x_path : "memory",
          S.n_config_max, ns_local, mpol, ntor);
    } else {
      const double* src_x[6] = {m_dec_x_rcc, m_dec_x_rss,
                                m_dec_x_zsc, m_dec_x_zcs,
                                m_dec_x_lsc, m_dec_x_lcs};
      for (int i = 0; i < 6; ++i) {
        for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
          cuda_check(cudaMemcpyAsync(dst_x[i] + (size_t)cfg * one_spec_doubles,
                                      src_x[i], x_bytes_one,
                                      cudaMemcpyHostToDevice, st),
                     "h2d pts x init (Recompose)");
        }
      }
    }
    S.pts_x_initialized = true;
    // Arm the device backup with the stage's initial state; same
    // contract as the matching call in PerformTimeStepCuda.
    EnsurePTSBackupBuffers(S);
  }

  // VMECPP_DEFENSIVE_BROADCAST=1: every RecomposeToPhysicalCuda entry
  // re-copies the cfg-0 slice of d_pts_x into all other slices.
  // Correct under broadcast inputs, redundant under per-cfg-correct
  // ones; an opt-in net for catching cfg-zero-only write regressions.
  static int defensive_broadcast_env = -1;
  if (defensive_broadcast_env < 0) {
    const char* e = std::getenv("VMECPP_DEFENSIVE_BROADCAST");
    defensive_broadcast_env = (e && std::atoi(e) > 0) ? 1 : 0;
    if (defensive_broadcast_env) {
      std::fprintf(stderr, "[fft_toroidal_cuda] Recompose defensive broadcast "
                           "ENABLED (VMECPP_DEFENSIVE_BROADCAST=1)\n");
    }
  }
  if (defensive_broadcast_env) {
    double* dst_x[6] = {S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
                        S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs};
    for (int i = 0; i < 6; ++i) {
      for (int cfg = 1; cfg < S.n_config_max; ++cfg) {
        cuda_check(cudaMemcpyAsync(
            dst_x[i] + (size_t)cfg * ns_local * mpol * (ntor + 1),
            dst_x[i],
            x_bytes_one,
            cudaMemcpyDeviceToDevice, st),
            "d2d pts x cfg=0 broadcast (Recompose)");
      }
    }
  }

  // Stage 1: decomposeInto → write d_specs_block sections from d_pts_x via
  // multiplication by scalxc. Reuse k_decompose_into; the math is the same
  // (dest = source * scal) regardless of geometry-vs-forces semantics. ns_dec
  // for the geometry case is ns_local (loop range [nsMinF1, ns) which equals
  // [0, ns_local) for single-rank).
  int nsMin_to_nsMinF1 = 0;  // for geometry: source is at full-grid index
  {
    const int TPB = 16;
    dim3 b((ntor + 1 + TPB - 1) / TPB, mpol, ns_local * S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_decompose_into<<<b, t, 0, st>>>(
        S.n_config_max, ns_local, ns_local,
        mpol, ntor, nsMin_to_nsMinF1, s.lthreed, S.d_scalxc,
        S.d_pts_x_rcc, S.d_pts_x_rss, S.d_pts_x_zsc,
        S.d_pts_x_zcs, S.d_pts_x_lsc, S.d_pts_x_lcs,
        S.d_rmncc, S.d_rmnss, S.d_zmnsc, S.d_zmncs, S.d_lmnsc, S.d_lmncs);
    cuda_check(cudaGetLastError(), "k_decompose_into (Recompose) launch");
  }

  // Stage 2: m1Constraint(1.0), physical_x in-place at m=1, mixing
  // rmnss/zmncs (and lasym rmnsc/zmncc pairs, which we skip). Reuse
  // standalone k_m1_constraint with scalingFactor=1.0.
  if (s.lthreed) {
    const int TPB = 32;
    int ns_for_m1 = ns_local;  // geometry loop is [nsMin_, nsMax_); for
                                // single-rank that's [0, ns_local)
    dim3 b((ntor + 1 + TPB - 1) / TPB, ns_for_m1, S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_m1_constraint<<<b, t, 0, st>>>(
        S.n_config_max, ns_local, ns_for_m1, mpol, ntor,
        /*scalingFactor=*/1.0,
        S.d_rmnss, S.d_zmncs);
    cuda_check(cudaGetLastError(), "k_m1_constraint (Recompose) launch");
  }

  // Stage 3: extrapolateTowardsAxis; only the nsMinF1==0 thread runs this.
  // For each n at m=1: copy from surface 1 to axis surface 0. Plus m=0 lmncs
  // (lthreed). Launch grid: (ntor+1, 1, n_config_max).
  if (r.nsMinF1 == 0) {
    const int TPB = 16;
    dim3 b((ntor + 1 + TPB - 1) / TPB, 1, S.n_config_max);
    dim3 t(TPB, 1, 1);
    k_extrapolate_towards_axis<<<b, t, 0, st>>>(
        S.n_config_max, ns_local, mpol, ntor, s.lthreed,
        S.d_rmncc, S.d_rmnss, S.d_zmnsc, S.d_zmncs, S.d_lmnsc, S.d_lmncs);
    cuda_check(cudaGetLastError(), "k_extrapolate_towards_axis launch");
  }

  // Signal CudaForward to skip its H2D specs_block; d_specs_block is now
  // populated from device.
  S.specs_populated_from_device = true;
  (void)fc;
}

}  // namespace vmecpp

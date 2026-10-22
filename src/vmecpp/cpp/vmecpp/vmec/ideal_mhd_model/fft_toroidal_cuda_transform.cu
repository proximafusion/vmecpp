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
// Selected when 32 is an exact multiple of nThetaReduced. Per-(jF, k, l)
// arithmetic is identical to v5.
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

  if (ns_local <= 0) {
    // Nothing to do for this MPI rank's local range.
    return;
  }

  // Effective configuration count for this run, frozen at run start by
  // ResetCudaStateForNewVmecRun. Default 1 keeps the single-config path
  // bit-exact. At N>1, distinct inputs arrive through the run_batched_gpu
  // staging; otherwise the same input is broadcast to all N slots.
  const int n_cfg = GetNConfigMaxCuda();

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
    // Distinct batches: the per-configuration spectral inputs come from the
    // run_batched_gpu staging block ([sp][cfg][spec] layout) instead of the
    // seed's physical_x broadcast to every slot. The staging cache lives in
    // the State so ResetForNewVmecRun rearms it; the consumed flag keeps it
    // one-shot within a run (the iteration-1 retry after the axis recompute
    // must see the host m_physical_x, not a reload of the pre-init staging).
    if (S.batch_inputs_loaded < 0) {
      if (!g_batch_inputs_mem.empty() && g_batch_mem_shape[0] == n_cfg &&
          g_batch_mem_shape[1] == ns_local && g_batch_mem_shape[2] == mpol &&
          g_batch_mem_shape[3] == ntor) {
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
    } else {
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

  // The scatter assigns every element of the output arrays, so they are not
  // cleared before it.

  // Launch fill kernel on stream.
  // Batched execution: z-dim is config * ns_local + jF_local. cuFFT batch dim
  // already covers n_config_max via the batched plan setup.
  const int FILL_TPB = 256;
  dim3 fill_blocks((kBatch * nhalf + FILL_TPB - 1) / FILL_TPB, mpol,
                   ns_local * S.n_config_max);
  dim3 fill_tpb(FILL_TPB, 1, 1);

  const int nZnT_local_pre = nZeta * nThetaEff;
  const int outer_idx_pre = (ns_local - 1) * nZnT_local_pre + 0;
  const int inner_idx_pre = (ns_local - 1) * nZnT_local_pre + (nThetaReduced - 1);

  k_fill_spectra<<<fill_blocks, fill_tpb, 0, st>>>(
      S.n_config_max, ns_local, mpol, ntor, nhalf, nfp, r.nsMinF1,
      S.d_rmncc, S.d_rmnss, S.d_zmnsc, S.d_zmncs, S.d_lmnsc, S.d_lmncs,
      S.d_nscale, S.d_X);
  cuda_check(cudaGetLastError(), "k_fill_spectra launch");

  if (nZeta == 1) {
    // Length-1 toroidal transform: the identity (DC real part). cuFFT's
    // length-1 plan is bypassed for the axisymmetric (ntor=0) case.
    int total_batches = n_cfg * ns_local * mpol * kBatch;
    int tpb = 256;
    k_z2d_identity_nzeta1<<<(total_batches + tpb - 1) / tpb, tpb, 0, st>>>(
        total_batches, S.d_X, S.d_Y);
    cuda_check(cudaGetLastError(), "k_z2d_identity_nzeta1");
  } else {
    cufft_check(cufftExecZ2D(S.cufft_plan, S.d_X, S.d_Y), "cufftExecZ2D");
  }

  // The fused scatter writes the main and the constraint outputs in one pass
  // over Y.
  {
    // One warp per (configuration, jF_local, k), four warps per block.
    constexpr int WARPS_PER_BLOCK = 4;
    int z_total = ns_local * S.n_config_max;
    int z_blocks = (z_total + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 v4_blocks((nThetaReduced + 31) / 32, nZeta, z_blocks);
    dim3 v4_tpb(32, WARPS_PER_BLOCK, 1);
    // Each warp caches its Y tile in shared memory; when 32 is a multiple of
    // nThetaReduced, a warp packs several zeta planes.
    const int ks_per_warp = (nThetaReduced > 0) ? (32 / nThetaReduced) : 0;
    if (ks_per_warp >= 2 && (32 % nThetaReduced) == 0) {
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
  }

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

  // Stage 2: forward FFT (D2Z) on (ns_local x mpol x kBatch) batches of
  // length nZeta=24, producing complex output X[jF, m, q, n] for n in [0,
  // nhalf=13).
  if (nZeta == 1) {
    // Length-1 toroidal transform: the identity (real value as the DC).
    int total_batches = S.n_config_max * ns_local * mpol * kBatch;
    int tpb = 256;
    k_d2z_identity_nzeta1<<<(total_batches + tpb - 1) / tpb, tpb, 0, st>>>(
        total_batches, S.d_Y, S.d_X);
    cuda_check(cudaGetLastError(), "k_d2z_identity_nzeta1");
  } else {
    cufft_check(cufftExecD2Z(S.cufft_plan_r2c, S.d_Y, S.d_X), "cufftExecD2Z");
  }

  // Stage 3: k_inverse_scatter populates spec arrays from X.
  // jMaxRZ from CPU: min(rp.nsMaxF, fc.ns - 1), bumped to ns on lfreeb+active.
  // jMinL is the lambda-write floor (constant = 1 in CPU code).
  int jMaxRZ_global = std::min(rp.nsMaxF, fc.ns - 1);
  if (fc.lfreeb &&
      vacuum_pressure_state >= VacuumPressureState::kInitialized) {
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
// MaskGeometryAboveCuda
// FourierCoeffs::maskGeometryAbove(mpolGeometry, ntorGeometry) on the device
// copy of either the decomposed forces (the surfaces decomposeInto covers) or
// the decomposed position state, which is authoritative once the first time
// step has staged it. A no-op without a geometry cap.
// ============================================================================
void MaskGeometryAboveCuda(const RadialPartitioning& r, const Sizes& s,
                           const FlowControl& fc, bool forces) {
  if (s.mpolGeometry >= s.mpol && s.ntorGeometry >= s.ntor) return;
  auto& S = State();
  std::lock_guard<std::mutex> lk(S.mu);
  const int modes = s.mpol * (s.ntor + 1);
  int ns_count = 0;
  size_t cfg_stride = 0;
  double *rcc = nullptr, *rss = nullptr, *zsc = nullptr, *zcs = nullptr;
  if (forces) {
    ns_count = (r.nsMaxF1 == fc.ns) ? (fc.ns - r.nsMinF) : (r.nsMaxF - r.nsMinF);
    cfg_stride = (size_t)S.ns_local_cached * (size_t)modes;
    rcc = S.d_decomposed_frcc;
    rss = S.d_decomposed_frss;
    zsc = S.d_decomposed_fzsc;
    zcs = S.d_decomposed_fzcs;
  } else {
    if (!S.pts_x_initialized || S.pts_x_size <= 0) return;
    ns_count = S.pts_x_size / modes;
    cfg_stride = (size_t)S.pts_x_size;
    rcc = S.d_pts_x_rcc;
    rss = S.d_pts_x_rss;
    zsc = S.d_pts_x_zsc;
    zcs = S.d_pts_x_zcs;
  }
  if (ns_count <= 0 || rcc == nullptr || zsc == nullptr) return;
  const int TPB = 64;
  dim3 b((modes + TPB - 1) / TPB, ns_count, S.n_config_max);
  dim3 t(TPB, 1, 1);
  k_mask_geometry_above<<<b, t, 0, S.stream>>>(
      S.n_config_max, cfg_stride, ns_count, s.mpol, s.ntor, s.mpolGeometry,
      s.ntorGeometry, s.lthreed ? 1 : 0, rcc, rss, zsc, zcs);
  cuda_check(cudaGetLastError(), "k_mask_geometry_above launch");
}

// ============================================================================
// DecomposeAndConstrainCuda
// Bridges S.d_frcc/etc. (m_physical_f device shadow from the inverse FFT) into
// the decomposed shadow S.d_decomposed_frcc/etc. (m_decomposed_f mirror) by
// running the three CPU-only steps from update():
//   m_physical_f.decomposeInto(m_decomposed_f, scalxc)
//   m_decomposed_f.m1Constraint(scalingFactor=1/sqrt(2), signOfJacobian)
//   if (always_fix_m1_gauge || fc.fsqz < 1e-6 || iter2 < 2)
//     m_decomposed_f.zeroZForceForM1()
// with the gate evaluated per configuration in a batch. The result stays in
// the device shadow for the subsequent residuals call.
// stellarator-symmetric (lasym=false) only.
// ============================================================================
void DecomposeAndConstrainCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    int iter2, bool always_fix_m1_gauge,
    double m1ScalingFactor, int sign_of_jacobian,
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
    k_decompose_into<<<b, t, 0, st>>>(
        S.n_config_max, ns_dec_local, S.ns_local_cached,
        mpol, ntor, nsMin_to_nsMinF1, s.lthreed, S.d_scalxc,
        S.d_frcc, S.d_frss, S.d_fzsc, S.d_fzcs, S.d_flsc, S.d_flcs,
        S.d_decomposed_frcc, S.d_decomposed_frss, S.d_decomposed_fzsc,
        S.d_decomposed_fzcs, S.d_decomposed_flsc, S.d_decomposed_flcs);
    cuda_check(cudaGetLastError(), "k_decompose_into launch");
  }

  // Stage 2+3: m=1 constraint, then the Z force at m=1 zeroed where the CPU
  // path zeroes it: under always_fix_m1_gauge, while the previous iteration's
  // fsqz is below 1e-6, and for iter2 < 2 (v8.50, so that reset=<WOUT_FILE>
  // works). A batch iterates in lockstep on iter2 and takes each
  // configuration's own fsqz.
  // Batched execution: z-dim covers n_config_max configs.
  if (s.lthreed && ns_force_local > 0) {
    const bool batched = S.n_config_max > 1;
    auto zeroes_m1_z_force = [&](int cfg) {
      const double fsqz = batched && cfg < static_cast<int>(fc.fsqz_per_cfg.size())
                              ? fc.fsqz_per_cfg[cfg]
                              : fc.fsqz;
      return always_fix_m1_gauge || fsqz < 1.0e-6 || iter2 < 2;
    };
    const unsigned char* zero_flags = nullptr;
    int zero_all = 0;
    if (batched) {
      S.h_m1_zero_flags.assign(S.n_config_max, 0);
      for (int cfg = 0; cfg < S.n_config_max; ++cfg) {
        S.h_m1_zero_flags[cfg] = zeroes_m1_z_force(cfg) ? 1 : 0;
      }
      if (S.m1_zero_flags_capacity < S.n_config_max) {
        if (S.d_m1_zero_flags != nullptr) {
          cuda_check(cudaFree(S.d_m1_zero_flags), "free m1 zero flags");
          S.d_m1_zero_flags = nullptr;
        }
        cuda_check(cudaMalloc(&S.d_m1_zero_flags, S.n_config_max),
                   "alloc m1 zero flags");
        S.m1_zero_flags_capacity = S.n_config_max;
      }
      cuda_check(cudaMemcpyAsync(S.d_m1_zero_flags, S.h_m1_zero_flags.data(),
                                  S.n_config_max, cudaMemcpyHostToDevice, st),
                 "h2d m1 zero flags");
      zero_flags = S.d_m1_zero_flags;
    } else {
      zero_all = zeroes_m1_z_force(0) ? 1 : 0;
    }
    const int TPB = 32;
    dim3 b((ntor + 1 + TPB - 1) / TPB, ns_force_local, S.n_config_max);
    dim3 t(TPB, 1, 1);
    const double sigma = -static_cast<double>(sign_of_jacobian);
    k_m1_constraint_gated<<<b, t, 0, st>>>(
        S.n_config_max, S.ns_local_cached, ns_force_local, mpol, ntor,
        m1ScalingFactor, sigma, zero_flags, zero_all,
        S.d_decomposed_frss, S.d_decomposed_fzcs);
    cuda_check(cudaGetLastError(), "k_m1_constraint_gated launch");
  }

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
  k_apply_m1_preconditioner<<<b, t, 0, st>>>(
      S.n_config_max, S.ns_local_cached, ns_force_local, mpol, ntor,
      S.d_pmat_ard, S.d_pmat_brd, S.d_pmat_azd, S.d_pmat_bzd,
      S.d_decomposed_frss, S.d_decomposed_fzcs,
      S.d_active_per_cfg);
  cuda_check(cudaGetLastError(), "k_apply_m1 launch");
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
//   m_physical_x.m1Constraint(1.0, signOfJacobian);
//   m_physical_x.extrapolateTowardsAxis();
//
// Reads d_pts_x_* (kept device-resident across iters by PerformTimeStepCuda),
// writes d_specs_block sections (d_rmncc/d_rmnss/d_zmnsc/d_zmncs/d_lmnsc/
// d_lmncs). After this call, d_specs_block is the device-side m_physical_x
// and CudaForward's H2D specs_block can be skipped (specs_populated_from_device
// flag).
//
// First-call init: H2D host m_decomposed_x -> d_pts_x. Subsequent calls read
// the device-resident d_pts_x updated by PerformTimeStepCuda at end of last
// iter. scalxc staging is shared with DecomposeAndConstrainCuda's stage flag.
// ============================================================================
void RecomposeToPhysicalCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    int sign_of_jacobian,
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

  // First-iter init: H2D host m_decomposed_x -> d_pts_x. Subsequent iters
  // skip this; d_pts_x persists from PerformTimeStepCuda's update. Distinct
  // batches load each configuration's decomposed position from the
  // run_batched_gpu staging block ([sp][cfg][spec] layout) instead, as
  // PerformTimeStepCuda's init does.
  size_t x_bytes_one = sizeof(double) * ns_local * mpol * (ntor + 1);
  if (!S.pts_x_initialized) {
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
              "h2d per-cfg dec_x (Recompose)");
        }
      }
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

  // Stage 1: decomposeInto -> write d_specs_block sections from d_pts_x via
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

  // Stage 2: m1Constraint(1.0, signOfJacobian), physical_x in-place at m=1,
  // mixing rmnss/zmncs (and lasym rmnsc/zmncc pairs, which we skip). Reuse
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
        /*sigma=*/-static_cast<double>(sign_of_jacobian),
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

#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"

namespace vmecpp {
// CUDA kernels

// k_fill_spectra: one thread per (jF_local, m, q, n) fills the cuFFT input X
// from the six spectra, scaled by nscale and xmpq per slot type.
__global__ void k_fill_spectra(
    int n_config, int ns_local, int mpol, int ntor, int nhalf, int nfp,
    int nsMinF1_offset,
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ nscale, cufftDoubleComplex* __restrict__ X) {
  int config = blockIdx.z / ns_local;
  int jF_local = blockIdx.z - config * ns_local;
  if (config >= n_config) return;
  int m = blockIdx.y;
  int qn = blockIdx.x * blockDim.x + threadIdx.x;
  if (qn >= kBatch * nhalf) return;
  int q = qn / nhalf;
  int n = qn % nhalf;
  size_t cfg_X    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nhalf;
  size_t cfg_spec = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)(ntor + 1);
  size_t dst_idx = cfg_X + (size_t)(((jF_local * mpol + m) * kBatch + q) * nhalf + n);

  // Per-m jMin handling.
  int jF_global = jF_local + nsMinF1_offset;
  int jMin = (m == 0 || m == 1) ? 0 : 1;
  if (jF_global < jMin) {
    X[dst_idx].x = 0.0;
    X[dst_idx].y = 0.0;
    return;
  }

  if (n > ntor) {
    X[dst_idx].x = 0.0;
    X[dst_idx].y = 0.0;
    return;
  }

  const double ns_n = nscale[n];
  size_t spec_base = cfg_spec + (size_t)((jF_local * mpol + m) * (ntor + 1) + n);

  double re = 0.0, im = 0.0;
  switch (q) {
    case kRmkcc: {  // DCT of rmncc
      double s = rmncc[spec_base];
      re = (n == 0) ? s * ns_n : s * ns_n * 0.5;
      im = 0.0;
      break;
    }
    case kRmkss: {  // DST of rmnss
      double s = rmnss[spec_base];
      re = 0.0;
      im = (n == 0) ? 0.0 : -s * ns_n * 0.5;
      break;
    }
    case kRmkccN: {  // DCT_DERIV of rmncc
      double s = rmncc[spec_base];
      re = 0.0;
      im = (n == 0) ? 0.0 : s * (double)n * nfp * ns_n * 0.5;
      break;
    }
    case kRmkssN: {  // DST_DERIV of rmnss
      double s = rmnss[spec_base];
      re = (n == 0) ? 0.0 : s * (double)n * nfp * ns_n * 0.5;
      im = 0.0;
      break;
    }
    case kZmksc: {  // DCT of zmnsc
      double s = zmnsc[spec_base];
      re = (n == 0) ? s * ns_n : s * ns_n * 0.5;
      im = 0.0;
      break;
    }
    case kZmkcs: {  // DST of zmncs
      double s = zmncs[spec_base];
      re = 0.0;
      im = (n == 0) ? 0.0 : -s * ns_n * 0.5;
      break;
    }
    case kZmkscN: {  // DCT_DERIV of zmnsc
      double s = zmnsc[spec_base];
      re = 0.0;
      im = (n == 0) ? 0.0 : s * (double)n * nfp * ns_n * 0.5;
      break;
    }
    case kZmkcsN: {  // DST_DERIV of zmncs
      double s = zmncs[spec_base];
      re = (n == 0) ? 0.0 : s * (double)n * nfp * ns_n * 0.5;
      im = 0.0;
      break;
    }
    case kLmksc: {  // DCT of lmnsc
      double s = lmnsc[spec_base];
      re = (n == 0) ? s * ns_n : s * ns_n * 0.5;
      im = 0.0;
      break;
    }
    case kLmkcs: {  // DST of lmncs
      double s = lmncs[spec_base];
      re = 0.0;
      im = (n == 0) ? 0.0 : -s * ns_n * 0.5;
      break;
    }
    case kLmkscN: {  // DCT_DERIV of lmnsc
      double s = lmnsc[spec_base];
      re = 0.0;
      im = (n == 0) ? 0.0 : s * (double)n * nfp * ns_n * 0.5;
      break;
    }
    case kLmkcsN: {  // DST_DERIV of lmncs
      double s = lmncs[spec_base];
      re = (n == 0) ? 0.0 : s * (double)n * nfp * ns_n * 0.5;
      im = 0.0;
      break;
    }
  }
  X[dst_idx].x = re;
  X[dst_idx].y = im;
}

// Fused main and constraint scatter: one warp per (config, jF_local, k), lanes
// over poloidal points, with Y staged in per-warp shared memory.
__global__ __launch_bounds__(128, 5) void k_scatter_main_and_con(
    int n_config, int ns_local, int mpol, int nZeta, int nThetaReduced, int nThetaEff,
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
  int z_base = blockIdx.z * blockDim.y;
  int z_global = z_base + warp_id;
  int config = z_global / ns_local;
  int jF_local = z_global - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  int lane = threadIdx.x;

  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;

  // Per-warp shared slot of mpol * kBatch doubles: s_Y[warp][m][q] = Y[cfg,
  // jF_local, m, q, k] for the warp's k.
  extern __shared__ double s_Y_block[];  // [blockDim.y * mpol * kBatch]
  double* s_Y = s_Y_block + (size_t)warp_id * (size_t)mpol * (size_t)kBatch;

  // Cooperative load: each lane handles a stride of 32 across (m * kBatch).
  // Total slots = mpol * kBatch = 120 for our shape. 32 lanes do ~4 each.
  const int total_slots = mpol * kBatch;
  #pragma unroll 4
  for (int t = lane; t < total_slots; t += 32) {
    int m_local = t / kBatch;
    int q_local = t - m_local * kBatch;
    size_t y_base_local = (size_t)((jF_local * mpol + m_local) * kBatch);
    s_Y[t] = Y[cfg_Y + (y_base_local + q_local) * nZeta + k];
  }
  __syncwarp();

  // One x-block per 32 poloidal points, covering
  // nThetaReduced > 32 across multiple warps.
  int l = blockIdx.x * blockDim.x + lane;
  if (l >= nThetaReduced) return;

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

  // mpol is 10 at the production call site; unrolling the m loop folds m_even
  // and the xmpq and sqrtSF products.
  #pragma unroll 10
  for (int m = 0; m < mpol; ++m) {
    // Read from shared memory (1 cycle latency vs L1's ~30 cycles).
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

// FourierGeometry::extrapolateTowardsAxis on the device: the axis takes the m =
// 1 coefficients, and in 3D the m = 0 lmncs, of surface one.
__global__ void k_extrapolate_towards_axis(
    int n_config, int ns_local, int mpol, int ntor, bool lthreed,
    double* __restrict__ rmncc, double* __restrict__ rmnss,
    double* __restrict__ zmnsc, double* __restrict__ zmncs,
    double* __restrict__ lmnsc, double* __restrict__ lmncs) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n > ntor) return;

  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);

  // axis = 0, firstSurface = 1.
  int axis0 = 0 * mpol * (ntor + 1) + 0 * (ntor + 1) + n;    // (jF=0, m=0, n)
  int axis1 = 0 * mpol * (ntor + 1) + 1 * (ntor + 1) + n;    // (jF=0, m=1, n)
  int firstSurface0 = 1 * mpol * (ntor + 1) + 0 * (ntor + 1) + n;
  int firstSurface1 = 1 * mpol * (ntor + 1) + 1 * (ntor + 1) + n;

  rmncc[cfg_spec + axis1] = rmncc[cfg_spec + firstSurface1];
  zmnsc[cfg_spec + axis1] = zmnsc[cfg_spec + firstSurface1];
  lmnsc[cfg_spec + axis1] = lmnsc[cfg_spec + firstSurface1];
  if (lthreed) {
    rmnss[cfg_spec + axis1] = rmnss[cfg_spec + firstSurface1];
    zmncs[cfg_spec + axis1] = zmncs[cfg_spec + firstSurface1];
    lmncs[cfg_spec + axis1] = lmncs[cfg_spec + firstSurface1];
    // m=0 component of lambda leftover from chi-force
    lmncs[cfg_spec + axis0] = lmncs[cfg_spec + firstSurface0];
  }
  // lasym branch omitted (our workload has lasym=false).
  (void)axis0;
}

// Multigrid upscale pre-pass: scales the previous-stage snapshot by its scalxc
// in place, as the host upscale receives X * SCALXC.
__global__ void k_scale_prev_by_scalxc(
    int n_config, int ns_old, int mpol, int ntor,
    int scalxc_old_len_per_cfg,
    double* prev_rcc, double* prev_rss,
    double* prev_zsc, double* prev_zcs,
    double* prev_lsc, double* prev_lcs,
    const double* __restrict__ scalxc_old) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n > ntor) return;
  int m = blockIdx.y;
  int j_cfg = blockIdx.z;
  int cfg = j_cfg / ns_old;
  int j = j_cfg % ns_old;
  if (cfg >= n_config || m >= mpol) return;
  size_t per_cfg = (size_t)ns_old * mpol * (ntor + 1);
  size_t idx = (size_t)cfg * per_cfg + ((size_t)j * mpol + m) * (ntor + 1) + n;
  size_t idx_scal = (size_t)cfg * (size_t)scalxc_old_len_per_cfg +
                    (size_t)j * 2 + (m & 1);
  const double s = scalxc_old[idx_scal];
  prev_rcc[idx] = __dmul_rn(prev_rcc[idx], s);
  prev_rss[idx] = __dmul_rn(prev_rss[idx], s);
  prev_zsc[idx] = __dmul_rn(prev_zsc[idx], s);
  prev_zcs[idx] = __dmul_rn(prev_zcs[idx], s);
  prev_lsc[idx] = __dmul_rn(prev_lsc[idx], s);
  prev_lcs[idx] = __dmul_rn(prev_lcs[idx], s);
}

// Upscale pre-pass: the odd-m axis value becomes 2 * old[js = 1] - old[js = 2]
// on the scaled values, as in the host upscale.
__global__ void k_axis_extrapolate_odd_m_prev(
    int n_config, int ns_old, int mpol, int ntor,
    double* prev_rcc, double* prev_rss,
    double* prev_zsc, double* prev_zcs,
    double* prev_lsc, double* prev_lcs) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n > ntor) return;
  int m_idx = blockIdx.y;  // 0, 1, 2, ... up to (mpol-1)/2
  int m = m_idx * 2 + 1;   // 1, 3, 5, ...
  if (m >= mpol) return;
  int cfg = blockIdx.z;
  if (cfg >= n_config) return;
  if (ns_old < 3) return;  // need js=0, js=1, js=2

  size_t per_cfg = (size_t)ns_old * mpol * (ntor + 1);
  size_t base = (size_t)cfg * per_cfg + (size_t)m * (ntor + 1) + n;
  size_t row_stride = (size_t)mpol * (ntor + 1);
  size_t idx_js0 = base + 0 * row_stride;
  size_t idx_js1 = base + 1 * row_stride;
  size_t idx_js2 = base + 2 * row_stride;

  prev_rcc[idx_js0] = 2.0 * prev_rcc[idx_js1] - prev_rcc[idx_js2];
  prev_rss[idx_js0] = 2.0 * prev_rss[idx_js1] - prev_rss[idx_js2];
  prev_zsc[idx_js0] = 2.0 * prev_zsc[idx_js1] - prev_zsc[idx_js2];
  prev_zcs[idx_js0] = 2.0 * prev_zcs[idx_js1] - prev_zcs[idx_js2];
  prev_lsc[idx_js0] = 2.0 * prev_lsc[idx_js1] - prev_lsc[idx_js2];
  prev_lcs[idx_js0] = 2.0 * prev_lcs[idx_js1] - prev_lcs[idx_js2];
}

// Per-configuration radial interpolation of d_pts_x at a stage boundary, the
// host upscale of Vmec::InterpolateToNextMultigridStep.
__global__ void k_radial_interpolate_pts_x(
    int n_config, int ns_old, int ns_new, int mpol, int ntor,
    int scalxc_len_per_cfg,
    const double* __restrict__ old_rcc, const double* __restrict__ old_rss,
    const double* __restrict__ old_zsc, const double* __restrict__ old_zcs,
    const double* __restrict__ old_lsc, const double* __restrict__ old_lcs,
    double* __restrict__ new_rcc, double* __restrict__ new_rss,
    double* __restrict__ new_zsc, double* __restrict__ new_zcs,
    double* __restrict__ new_lsc, double* __restrict__ new_lcs,
    const double* __restrict__ scalxc) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n > ntor) return;
  int m = blockIdx.y;
  int jNew_cfg = blockIdx.z;
  int cfg = jNew_cfg / ns_new;
  int jNew = jNew_cfg % ns_new;
  if (cfg >= n_config || jNew >= ns_new) return;

  size_t per_cfg_new = (size_t)ns_new * mpol * (ntor + 1);
  size_t idx_new = (size_t)cfg * per_cfg_new + ((size_t)jNew * mpol + m) * (ntor + 1) + n;
  int m_parity = m & 1;

  // Host final step: all odd-m modes are zeroed at the axis.
  if (m_parity == 1 && jNew == 0) {
    new_rcc[idx_new] = 0.0;
    new_rss[idx_new] = 0.0;
    new_zsc[idx_new] = 0.0;
    new_zcs[idx_new] = 0.0;
    new_lsc[idx_new] = 0.0;
    new_lcs[idx_new] = 0.0;
    return;
  }

  double hs_old = (ns_old > 1) ? 1.0 / (double)(ns_old - 1.0) : 1.0;
  int js1 = (jNew * (ns_old - 1)) / (ns_new - 1);
  int js2 = js1 + 1; if (js2 > ns_old - 1) js2 = ns_old - 1;
  double sj = (ns_new > 1) ? (double)jNew / (double)(ns_new - 1.0) : 0.0;
  double s1 = (double)js1 * hs_old;
  double xint = (sj - s1) / hs_old;
  if (xint > 1.0) xint = 1.0;
  if (xint < 0.0) xint = 0.0;

  size_t per_cfg_old = (size_t)ns_old * mpol * (ntor + 1);
  size_t idx_js1 = (size_t)cfg * per_cfg_old + ((size_t)js1 * mpol + m) * (ntor + 1) + n;
  size_t idx_js2 = (size_t)cfg * per_cfg_old + ((size_t)js2 * mpol + m) * (ntor + 1) + n;
  size_t idx_scal = (size_t)cfg * (size_t)scalxc_len_per_cfg + (size_t)jNew * 2 + m_parity;
  const double scal_new = scalxc[idx_scal];
  const double w0 = 1.0 - xint;
  const double w1 = xint;

  new_rcc[idx_new] = __ddiv_rn(
      __dadd_rn(__dmul_rn(w0, old_rcc[idx_js1]),
                __dmul_rn(w1, old_rcc[idx_js2])), scal_new);
  new_rss[idx_new] = __ddiv_rn(
      __dadd_rn(__dmul_rn(w0, old_rss[idx_js1]),
                __dmul_rn(w1, old_rss[idx_js2])), scal_new);
  new_zsc[idx_new] = __ddiv_rn(
      __dadd_rn(__dmul_rn(w0, old_zsc[idx_js1]),
                __dmul_rn(w1, old_zsc[idx_js2])), scal_new);
  new_zcs[idx_new] = __ddiv_rn(
      __dadd_rn(__dmul_rn(w0, old_zcs[idx_js1]),
                __dmul_rn(w1, old_zcs[idx_js2])), scal_new);
  new_lsc[idx_new] = __ddiv_rn(
      __dadd_rn(__dmul_rn(w0, old_lsc[idx_js1]),
                __dmul_rn(w1, old_lsc[idx_js2])), scal_new);
  new_lcs[idx_new] = __ddiv_rn(
      __dadd_rn(__dmul_rn(w0, old_lcs[idx_js1]),
                __dmul_rn(w1, old_lcs[idx_js2])), scal_new);
}

// Vmec::performTimeStep on the device: v = velocity_scale * (b1 * v + dt * f)
// and x = x + dt * v from the device forces.
__global__ void k_perform_time_step(
    int n_config, int ns_local, int ns_con_local,
    int mpol, int ntor, int nsMinF_to_nsMinF1, bool lthreed,
    double velocity_scale, double conjugation_parameter, double time_step,
    const double* __restrict__ f_rcc, const double* __restrict__ f_rss,
    const double* __restrict__ f_zsc, const double* __restrict__ f_zcs,
    const double* __restrict__ f_lsc, const double* __restrict__ f_lcs,
    double* __restrict__ v_rcc, double* __restrict__ v_rss,
    double* __restrict__ v_zsc, double* __restrict__ v_zcs,
    double* __restrict__ v_lsc, double* __restrict__ v_lcs,
    double* __restrict__ x_rcc, double* __restrict__ x_rss,
    double* __restrict__ x_zsc, double* __restrict__ x_zcs,
    double* __restrict__ x_lsc, double* __restrict__ x_lcs,
    const double* __restrict__ d_fac_b1,
    const std::uint8_t* __restrict__ d_active_per_cfg,
    const double* __restrict__ d_time_step_per_cfg) {
  int config = blockIdx.z / ns_con_local;
  int jF_v_local = blockIdx.z - config * ns_con_local;
  if (config >= n_config) return;
  // Inactive configurations hold their x and v, since shared host quantities
  // derive from configuration zero's live slot.
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int m = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_v_local >= ns_con_local || m >= mpol || n > ntor) return;

  // With d_fac_b1, each configuration takes its own fac and b1 from
  // k_update_timestep in place of the shared scalars.
  double fac_use = velocity_scale;
  double b1_use = conjugation_parameter;
  if (d_fac_b1 != nullptr) {
    fac_use = d_fac_b1[(size_t)config * 2 + 0];
    b1_use  = d_fac_b1[(size_t)config * 2 + 1];
  }
  // Per-configuration restart control gives each configuration its own
  // time step.
  const double dt =
      d_time_step_per_cfg != nullptr ? d_time_step_per_cfg[config] : time_step;

  // v and f (decomposed_f) share the same per-config layout: ns_con_local
  // (== ns_dec_local for the LCFS-owning thread in our single-rank setup).
  size_t cfg_v = (size_t)config * (size_t)ns_con_local * (size_t)mpol * (size_t)(ntor + 1);
  size_t cfg_x = (size_t)config * (size_t)ns_local     * (size_t)mpol * (size_t)(ntor + 1);
  int jF_full_local = jF_v_local + nsMinF_to_nsMinF1;
  size_t idx_v = cfg_v + (size_t)((jF_v_local * mpol + m) * (ntor + 1) + n);
  size_t idx_x = cfg_x + (size_t)((jF_full_local * mpol + m) * (ntor + 1) + n);

  // v_rcc / x_rcc (rmncc parity)
  {
    double v_new = fac_use *
                   (b1_use * v_rcc[idx_v] + dt * f_rcc[idx_v]);
    v_rcc[idx_v] = v_new;
    x_rcc[idx_x] += dt * v_new;
  }
  // v_zsc / x_zsc (zmnsc parity)
  {
    double v_new = fac_use *
                   (b1_use * v_zsc[idx_v] + dt * f_zsc[idx_v]);
    v_zsc[idx_v] = v_new;
    x_zsc[idx_x] += dt * v_new;
  }
  // v_lsc / x_lsc (lmnsc parity)
  {
    double v_new = fac_use *
                   (b1_use * v_lsc[idx_v] + dt * f_lsc[idx_v]);
    v_lsc[idx_v] = v_new;
    x_lsc[idx_x] += dt * v_new;
  }
  if (lthreed) {
    // v_rss / x_rss
    {
      double v_new = fac_use *
                     (b1_use * v_rss[idx_v] + dt * f_rss[idx_v]);
      v_rss[idx_v] = v_new;
      x_rss[idx_x] += dt * v_new;
    }
    // v_zcs / x_zcs
    {
      double v_new = fac_use *
                     (b1_use * v_zcs[idx_v] + dt * f_zcs[idx_v]);
      v_zcs[idx_v] = v_new;
      x_zcs[idx_x] += dt * v_new;
    }
    // v_lcs / x_lcs
    {
      double v_new = fac_use *
                     (b1_use * v_lcs[idx_v] + dt * f_lcs[idx_v]);
      v_lcs[idx_v] = v_new;
      x_lcs[idx_x] += dt * v_new;
    }
  }
}

// k_extract_geom_scalars: configuration zero's six geometry scalars for
// SetRadialExtent and SetGeometricOffset into one buffer.
__global__ void k_extract_geom_scalars(
    const double* __restrict__ d_r1_e,
    const double* __restrict__ d_r1_o,
    const double* __restrict__ d_z1_e,
    int outer_idx, int inner_idx, double* __restrict__ d_out) {
  d_out[0] = d_r1_e[outer_idx];
  d_out[1] = d_r1_o[outer_idx];
  d_out[2] = d_r1_e[inner_idx];
  d_out[3] = d_r1_o[inner_idx];
  d_out[4] = d_r1_e[0];
  d_out[5] = d_z1_e[0];
}

// k_compute_metric_elements: one thread per (jH_local, kl) computes gsqrt, guu,
// guv (3D) and gvv from the full-grid geometry, tau and r12.
__global__ void k_compute_metric_elements(
    int n_config, int ns_local,
    int ns_h, int jF_in_offset, int nZnT, bool lthreed,
    const double* __restrict__ r1_e, const double* __restrict__ r1_o,
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    const double* __restrict__ rv_e, const double* __restrict__ rv_o,
    const double* __restrict__ zv_e, const double* __restrict__ zv_o,
    const double* __restrict__ sqrtSF, const double* __restrict__ sqrtSH,
    const double* __restrict__ tau, const double* __restrict__ r12,
    double* __restrict__ gsqrt, double* __restrict__ guu,
    double* __restrict__ guv, double* __restrict__ gvv) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH_local = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  if (jH_local >= ns_h) return;

  size_t cfg_full = (size_t)config * (size_t)ns_local * (size_t)nZnT;
  size_t cfg_half = (size_t)config * (size_t)ns_h    * (size_t)nZnT;

  int jF_in = jH_local + jF_in_offset;
  int jF_out = jF_in + 1;
  size_t iHalf = cfg_half + (size_t)jH_local * (size_t)nZnT + (size_t)kl;

  double r1e_i = r1_e[cfg_full + jF_in  * nZnT + kl];
  double r1o_i = r1_o[cfg_full + jF_in  * nZnT + kl];
  double rue_i = ru_e[cfg_full + jF_in  * nZnT + kl];
  double ruo_i = ru_o[cfg_full + jF_in  * nZnT + kl];
  double zue_i = zu_e[cfg_full + jF_in  * nZnT + kl];
  double zuo_i = zu_o[cfg_full + jF_in  * nZnT + kl];
  double r1e_o = r1_e[cfg_full + jF_out * nZnT + kl];
  double r1o_o = r1_o[cfg_full + jF_out * nZnT + kl];
  double rue_o = ru_e[cfg_full + jF_out * nZnT + kl];
  double ruo_o = ru_o[cfg_full + jF_out * nZnT + kl];
  double zue_o = zu_e[cfg_full + jF_out * nZnT + kl];
  double zuo_o = zu_o[cfg_full + jF_out * nZnT + kl];

  double sqrtSF_i = sqrtSF[jF_in];
  double sqrtSF_o = sqrtSF[jF_out];
  double sF_i = sqrtSF_i * sqrtSF_i;
  double sF_o = sqrtSF_o * sqrtSF_o;
  double sH = sqrtSH[jH_local];

  gsqrt[iHalf] = tau[iHalf] * r12[iHalf];

  double guu_v = 0.5 * ((rue_i * rue_i + zue_i * zue_i) +
                         (rue_o * rue_o + zue_o * zue_o) +
                         sF_i * (ruo_i * ruo_i + zuo_i * zuo_i) +
                         sF_o * (ruo_o * ruo_o + zuo_o * zuo_o)) +
                  sH * ((rue_i * ruo_i + zue_i * zuo_i) +
                        (rue_o * ruo_o + zue_o * zuo_o));

  double gvv_v = 0.5 * (r1e_i * r1e_i + r1e_o * r1e_o +
                        sF_i * r1o_i * r1o_i + sF_o * r1o_o * r1o_o) +
                 sH * (r1e_i * r1o_i + r1e_o * r1o_o);

  double guv_v = 0.0;
  if (lthreed) {
    double rve_i = rv_e[cfg_full + jF_in  * nZnT + kl];
    double rvo_i = rv_o[cfg_full + jF_in  * nZnT + kl];
    double zve_i = zv_e[cfg_full + jF_in  * nZnT + kl];
    double zvo_i = zv_o[cfg_full + jF_in  * nZnT + kl];
    double rve_o = rv_e[cfg_full + jF_out * nZnT + kl];
    double rvo_o = rv_o[cfg_full + jF_out * nZnT + kl];
    double zve_o = zv_e[cfg_full + jF_out * nZnT + kl];
    double zvo_o = zv_o[cfg_full + jF_out * nZnT + kl];

    guv_v = 0.5 * ((rue_i * rve_i + zue_i * zve_i) +
                   (rue_o * rve_o + zue_o * zve_o) +
                   sF_i * (ruo_i * rvo_i + zuo_i * zvo_i) +
                   sF_o * (ruo_o * rvo_o + zuo_o * zvo_o) +
                   sH * ((rue_i * rvo_i + zue_i * zvo_i) +
                         (rue_o * rvo_o + zue_o * zvo_o) +
                         (rve_i * ruo_i + zve_i * zuo_i) +
                         (rve_o * ruo_o + zve_o * zuo_o)));

    gvv_v += 0.5 * ((rve_i * rve_i + zve_i * zve_i) +
                    (rve_o * rve_o + zve_o * zve_o) +
                    sF_i * (rvo_i * rvo_i + zvo_i * zvo_i) +
                    sF_o * (rvo_o * rvo_o + zvo_o * zvo_o)) +
             sH * ((rve_i * rvo_i + zve_i * zvo_i) +
                   (rve_o * rvo_o + zve_o * zvo_o));
  }

  guu[iHalf] = guu_v;
  gvv[iHalf] = gvv_v;
  guv[iHalf] = guv_v;
}

// Fused Jacobian and metric kernel, one thread per (configuration, jH_local,
// kl), writing r12, ru12, zu12, rs, zs, tau and the metric.
__global__ void k_jacobian_and_metric(
    int n_config, int ns_local,
    int ns_h, int jF_in_offset, int nZnT, bool lthreed,
    const double* __restrict__ r1_e, const double* __restrict__ r1_o,
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ z1_e, const double* __restrict__ z1_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    const double* __restrict__ rv_e, const double* __restrict__ rv_o,
    const double* __restrict__ zv_e, const double* __restrict__ zv_o,
    const double* __restrict__ sqrtSF, const double* __restrict__ sqrtSH,
    double deltaS, double dSHalfDsInterp,
    double* __restrict__ r12, double* __restrict__ ru12,
    double* __restrict__ zu12, double* __restrict__ rs,
    double* __restrict__ zs, double* __restrict__ tau,
    double* __restrict__ gsqrt, double* __restrict__ guu,
    double* __restrict__ guv, double* __restrict__ gvv,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jH_local = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  if (jH_local >= ns_h) return;

  size_t cfg_full = (size_t)config * (size_t)ns_local * (size_t)nZnT;
  size_t cfg_half = (size_t)config * (size_t)ns_h    * (size_t)nZnT;

  int jF_in = jH_local + jF_in_offset;
  int jF_out = jF_in + 1;

  // Shared loads (jacobian + metric both consume):
  double r1e_i = r1_e[cfg_full + jF_in  * nZnT + kl];
  double r1o_i = r1_o[cfg_full + jF_in  * nZnT + kl];
  double z1e_i = z1_e[cfg_full + jF_in  * nZnT + kl];
  double z1o_i = z1_o[cfg_full + jF_in  * nZnT + kl];
  double rue_i = ru_e[cfg_full + jF_in  * nZnT + kl];
  double ruo_i = ru_o[cfg_full + jF_in  * nZnT + kl];
  double zue_i = zu_e[cfg_full + jF_in  * nZnT + kl];
  double zuo_i = zu_o[cfg_full + jF_in  * nZnT + kl];
  double r1e_o = r1_e[cfg_full + jF_out * nZnT + kl];
  double r1o_o = r1_o[cfg_full + jF_out * nZnT + kl];
  double z1e_o = z1_e[cfg_full + jF_out * nZnT + kl];
  double z1o_o = z1_o[cfg_full + jF_out * nZnT + kl];
  double rue_o = ru_e[cfg_full + jF_out * nZnT + kl];
  double ruo_o = ru_o[cfg_full + jF_out * nZnT + kl];
  double zue_o = zu_e[cfg_full + jF_out * nZnT + kl];
  double zuo_o = zu_o[cfg_full + jF_out * nZnT + kl];

  double sH = sqrtSH[jH_local];

  size_t iHalf = cfg_half + (size_t)jH_local * (size_t)nZnT + (size_t)kl;

  // ===== Jacobian =====
  double r12_v  = 0.5 * ((r1e_i + r1e_o) + sH * (r1o_i + r1o_o));
  double ru12_v = 0.5 * ((rue_i + rue_o) + sH * (ruo_i + ruo_o));
  double zu12_v = 0.5 * ((zue_i + zue_o) + sH * (zuo_i + zuo_o));
  double rs_v   = ((r1e_o - r1e_i) + sH * (r1o_o - r1o_i)) / deltaS;
  double zs_v   = ((z1e_o - z1e_i) + sH * (z1o_o - z1o_i)) / deltaS;

  double tau1 = ru12_v * zs_v - rs_v * zu12_v;
  double tau2 = ruo_o * z1o_o + ruo_i * z1o_i -
                zuo_o * r1o_o - zuo_i * r1o_i +
                (rue_o * z1o_o + rue_i * z1o_i -
                 zue_o * r1o_o - zue_i * r1o_i) / sH;
  double tau_v = tau1 + dSHalfDsInterp * tau2;

  r12[iHalf]  = r12_v;
  ru12[iHalf] = ru12_v;
  zu12[iHalf] = zu12_v;
  rs[iHalf]   = rs_v;
  zs[iHalf]   = zs_v;
  tau[iHalf]  = tau_v;

  // ===== Metric =====
  double sqrtSF_i = sqrtSF[jF_in];
  double sqrtSF_o = sqrtSF[jF_out];
  double sF_i = sqrtSF_i * sqrtSF_i;
  double sF_o = sqrtSF_o * sqrtSF_o;

  double gsqrt_v = tau_v * r12_v;

  double guu_v = 0.5 * ((rue_i * rue_i + zue_i * zue_i) +
                         (rue_o * rue_o + zue_o * zue_o) +
                         sF_i * (ruo_i * ruo_i + zuo_i * zuo_i) +
                         sF_o * (ruo_o * ruo_o + zuo_o * zuo_o)) +
                  sH * ((rue_i * ruo_i + zue_i * zuo_i) +
                        (rue_o * ruo_o + zue_o * zuo_o));

  double gvv_v = 0.5 * (r1e_i * r1e_i + r1e_o * r1e_o +
                        sF_i * r1o_i * r1o_i + sF_o * r1o_o * r1o_o) +
                 sH * (r1e_i * r1o_i + r1e_o * r1o_o);

  double guv_v = 0.0;
  if (lthreed) {
    double rve_i = rv_e[cfg_full + jF_in  * nZnT + kl];
    double rvo_i = rv_o[cfg_full + jF_in  * nZnT + kl];
    double zve_i = zv_e[cfg_full + jF_in  * nZnT + kl];
    double zvo_i = zv_o[cfg_full + jF_in  * nZnT + kl];
    double rve_o = rv_e[cfg_full + jF_out * nZnT + kl];
    double rvo_o = rv_o[cfg_full + jF_out * nZnT + kl];
    double zve_o = zv_e[cfg_full + jF_out * nZnT + kl];
    double zvo_o = zv_o[cfg_full + jF_out * nZnT + kl];

    guv_v = 0.5 * ((rue_i * rve_i + zue_i * zve_i) +
                   (rue_o * rve_o + zue_o * zve_o) +
                   sF_i * (ruo_i * rvo_i + zuo_i * zvo_i) +
                   sF_o * (ruo_o * rvo_o + zuo_o * zvo_o) +
                   sH * ((rue_i * rvo_i + zue_i * zvo_i) +
                         (rue_o * rvo_o + zue_o * zvo_o) +
                         (rve_i * ruo_i + zve_i * zuo_i) +
                         (rve_o * ruo_o + zve_o * zuo_o)));

    gvv_v += 0.5 * ((rve_i * rve_i + zve_i * zve_i) +
                    (rve_o * rve_o + zve_o * zve_o) +
                    sF_i * (rvo_i * rvo_i + zvo_i * zvo_i) +
                    sF_o * (rvo_o * rvo_o + zvo_o * zvo_o)) +
             sH * ((rve_i * rvo_i + zve_i * zvo_i) +
                   (rve_o * rvo_o + zve_o * zvo_o));
  }

  gsqrt[iHalf] = gsqrt_v;
  guu[iHalf] = guu_v;
  gvv[iHalf] = gvv_v;
  guv[iHalf] = guv_v;
}

// Fused Jacobian, metric and dVdsH kernel with atomic accumulation, on
// k_jacobian_and_metric's launch geometry.
__global__ void k_jacobian_metric_dvdsh_atomic(
    int n_config, int ns_local,
    int ns_h, int jF_in_offset, int nZnT, int nThetaEff, bool lthreed,
    double signOfJacobian,
    const double* __restrict__ r1_e, const double* __restrict__ r1_o,
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ z1_e, const double* __restrict__ z1_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    const double* __restrict__ rv_e, const double* __restrict__ rv_o,
    const double* __restrict__ zv_e, const double* __restrict__ zv_o,
    const double* __restrict__ sqrtSF, const double* __restrict__ sqrtSH,
    const double* __restrict__ wInt,
    double deltaS, double dSHalfDsInterp,
    double* __restrict__ r12, double* __restrict__ ru12,
    double* __restrict__ zu12, double* __restrict__ rs,
    double* __restrict__ zs, double* __restrict__ tau,
    double* __restrict__ gsqrt, double* __restrict__ guu,
    double* __restrict__ guv, double* __restrict__ gvv,
    double* __restrict__ dVdsH) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH_local = blockIdx.y;
  if (jH_local >= ns_h) return;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;

  size_t cfg_full = (size_t)config * (size_t)ns_local * (size_t)nZnT;
  size_t cfg_half = (size_t)config * (size_t)ns_h    * (size_t)nZnT;

  int jF_in = jH_local + jF_in_offset;
  int jF_out = jF_in + 1;

  double r1e_i = r1_e[cfg_full + jF_in  * nZnT + kl];
  double r1o_i = r1_o[cfg_full + jF_in  * nZnT + kl];
  double z1e_i = z1_e[cfg_full + jF_in  * nZnT + kl];
  double z1o_i = z1_o[cfg_full + jF_in  * nZnT + kl];
  double rue_i = ru_e[cfg_full + jF_in  * nZnT + kl];
  double ruo_i = ru_o[cfg_full + jF_in  * nZnT + kl];
  double zue_i = zu_e[cfg_full + jF_in  * nZnT + kl];
  double zuo_i = zu_o[cfg_full + jF_in  * nZnT + kl];
  double r1e_o = r1_e[cfg_full + jF_out * nZnT + kl];
  double r1o_o = r1_o[cfg_full + jF_out * nZnT + kl];
  double z1e_o = z1_e[cfg_full + jF_out * nZnT + kl];
  double z1o_o = z1_o[cfg_full + jF_out * nZnT + kl];
  double rue_o = ru_e[cfg_full + jF_out * nZnT + kl];
  double ruo_o = ru_o[cfg_full + jF_out * nZnT + kl];
  double zue_o = zu_e[cfg_full + jF_out * nZnT + kl];
  double zuo_o = zu_o[cfg_full + jF_out * nZnT + kl];

  double sH = sqrtSH[jH_local];
  double sqrtSF_i = sqrtSF[jF_in];
  double sqrtSF_o = sqrtSF[jF_out];
  double sF_i = sqrtSF_i * sqrtSF_i;
  double sF_o = sqrtSF_o * sqrtSF_o;
  size_t iHalf = cfg_half + (size_t)jH_local * (size_t)nZnT + (size_t)kl;

  double r12_v  = 0.5 * ((r1e_i + r1e_o) + sH * (r1o_i + r1o_o));
  double ru12_v = 0.5 * ((rue_i + rue_o) + sH * (ruo_i + ruo_o));
  double zu12_v = 0.5 * ((zue_i + zue_o) + sH * (zuo_i + zuo_o));
  double rs_v   = ((r1e_o - r1e_i) + sH * (r1o_o - r1o_i)) / deltaS;
  double zs_v   = ((z1e_o - z1e_i) + sH * (z1o_o - z1o_i)) / deltaS;

  double tau1 = ru12_v * zs_v - rs_v * zu12_v;
  double tau2 = ruo_o * z1o_o + ruo_i * z1o_i -
                zuo_o * r1o_o - zuo_i * r1o_i +
                (rue_o * z1o_o + rue_i * z1o_i -
                 zue_o * r1o_o - zue_i * r1o_i) / sH;
  double tau_v = tau1 + dSHalfDsInterp * tau2;

  r12[iHalf]  = r12_v;
  ru12[iHalf] = ru12_v;
  zu12[iHalf] = zu12_v;
  rs[iHalf]   = rs_v;
  zs[iHalf]   = zs_v;
  tau[iHalf]  = tau_v;

  double gsqrt_v = tau_v * r12_v;
  double guu_v = 0.5 * ((rue_i * rue_i + zue_i * zue_i) +
                         (rue_o * rue_o + zue_o * zue_o) +
                         sF_i * (ruo_i * ruo_i + zuo_i * zuo_i) +
                         sF_o * (ruo_o * ruo_o + zuo_o * zuo_o)) +
                  sH * ((rue_i * ruo_i + zue_i * zuo_i) +
                        (rue_o * ruo_o + zue_o * zuo_o));
  double gvv_v = 0.5 * (r1e_i * r1e_i + r1e_o * r1e_o +
                        sF_i * r1o_i * r1o_i + sF_o * r1o_o * r1o_o) +
                 sH * (r1e_i * r1o_i + r1e_o * r1o_o);
  double guv_v = 0.0;
  if (lthreed) {
    double rve_i = rv_e[cfg_full + jF_in  * nZnT + kl];
    double rvo_i = rv_o[cfg_full + jF_in  * nZnT + kl];
    double zve_i = zv_e[cfg_full + jF_in  * nZnT + kl];
    double zvo_i = zv_o[cfg_full + jF_in  * nZnT + kl];
    double rve_o = rv_e[cfg_full + jF_out * nZnT + kl];
    double rvo_o = rv_o[cfg_full + jF_out * nZnT + kl];
    double zve_o = zv_e[cfg_full + jF_out * nZnT + kl];
    double zvo_o = zv_o[cfg_full + jF_out * nZnT + kl];
    guv_v = 0.5 * ((rue_i * rve_i + zue_i * zve_i) +
                   (rue_o * rve_o + zue_o * zve_o) +
                   sF_i * (ruo_i * rvo_i + zuo_i * zvo_i) +
                   sF_o * (ruo_o * rvo_o + zuo_o * zvo_o) +
                   sH * ((rue_i * rvo_i + zue_i * zvo_i) +
                         (rue_o * rvo_o + zue_o * zvo_o) +
                         (rve_i * ruo_i + zve_i * zuo_i) +
                         (rve_o * ruo_o + zve_o * zuo_o)));
    gvv_v += 0.5 * ((rve_i * rve_i + zve_i * zve_i) +
                    (rve_o * rve_o + zve_o * zve_o) +
                    sF_i * (rvo_i * rvo_i + zvo_i * zvo_i) +
                    sF_o * (rvo_o * rvo_o + zvo_o * zvo_o)) +
             sH * ((rve_i * rvo_i + zve_i * zvo_i) +
                   (rve_o * rvo_o + zve_o * zvo_o));
  }
  gsqrt[iHalf] = gsqrt_v;
  guu[iHalf]   = guu_v;
  gvv[iHalf]   = gvv_v;
  guv[iHalf]   = guv_v;

  // Atomic accumulator: signOfJacobian * gsqrt * wInt(l) -> dVdsH[cfg, jH].
  // Order-nondeterministic; relaxed-contract path.
  int l = kl % nThetaEff;
  double contrib = signOfJacobian * gsqrt_v * wInt[l];
  atomicAdd(&dVdsH[config * ns_h + jH_local], contrib);
}

// Half-grid pair coarsening of the fused kernel: a block serves two adjacent
// surfaces and caches their shared full-grid surface.
__global__ __launch_bounds__(128, 5) void k_jacobian_metric_dvdsh_atomic_pair(
    int n_config, int ns_local,
    int ns_h, int jF_in_offset, int nZnT, int nThetaEff, bool lthreed,
    double signOfJacobian,
    const double* __restrict__ r1_e, const double* __restrict__ r1_o,
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ z1_e, const double* __restrict__ z1_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    const double* __restrict__ rv_e, const double* __restrict__ rv_o,
    const double* __restrict__ zv_e, const double* __restrict__ zv_o,
    const double* __restrict__ sqrtSF, const double* __restrict__ sqrtSH,
    const double* __restrict__ wInt,
    double deltaS, double dSHalfDsInterp,
    double* __restrict__ r12, double* __restrict__ ru12,
    double* __restrict__ zu12, double* __restrict__ rs,
    double* __restrict__ zs, double* __restrict__ tau,
    double* __restrict__ gsqrt, double* __restrict__ guu,
    double* __restrict__ guv, double* __restrict__ gvv,
    double* __restrict__ dVdsH,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jH_pair = blockIdx.y;
  int my_jH_offset = threadIdx.y;  // 0 = lo, 1 = hi
  int jH_local = jH_pair * 2 + my_jH_offset;
  if (jH_local >= ns_h) return;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;

  size_t cfg_full = (size_t)config * (size_t)ns_local * (size_t)nZnT;
  size_t cfg_half = (size_t)config * (size_t)ns_h    * (size_t)nZnT;

  // SHARED jF index: jH_lo's jF_out == jH_hi's jF_in == jH_lo + 1 + offset.
  int jF_shared = jH_pair * 2 + 1 + jF_in_offset;

  // Shared layout: eight full-grid fields per x-lane, shared by the two jH
  // lanes at the same kl, sized to the block's x-extent.
  const int sj = threadIdx.x;
  const int sw = blockDim.x;
  extern __shared__ double s_jac_buf[];
  double* s_r1e = s_jac_buf + 0 * sw;
  double* s_r1o = s_jac_buf + 1 * sw;
  double* s_rue = s_jac_buf + 2 * sw;
  double* s_ruo = s_jac_buf + 3 * sw;
  double* s_z1e = s_jac_buf + 4 * sw;
  double* s_z1o = s_jac_buf + 5 * sw;
  double* s_zue = s_jac_buf + 6 * sw;
  double* s_zuo = s_jac_buf + 7 * sw;

  // Cooperative load: y=0 lanes populate shared from jF_shared.
  if (my_jH_offset == 0) {
    size_t i = cfg_full + (size_t)jF_shared * (size_t)nZnT + (size_t)kl;
    s_r1e[sj] = r1_e[i];
    s_r1o[sj] = r1_o[i];
    s_rue[sj] = ru_e[i];
    s_ruo[sj] = ru_o[i];
    s_z1e[sj] = z1_e[i];
    s_z1o[sj] = z1_o[i];
    s_zue[sj] = zu_e[i];
    s_zuo[sj] = zu_o[i];
  }
  __syncthreads();

  int jF_in  = jH_local + jF_in_offset;
  int jF_out = jF_in + 1;
  // For y=0: jF_out == jF_shared; for y=1: jF_in == jF_shared.
  bool in_is_shared  = (jF_in  == jF_shared);
  bool out_is_shared = (jF_out == jF_shared);

  double r1e_i, r1o_i, z1e_i, z1o_i, rue_i, ruo_i, zue_i, zuo_i;
  if (in_is_shared) {
    r1e_i = s_r1e[sj]; r1o_i = s_r1o[sj];
    rue_i = s_rue[sj]; ruo_i = s_ruo[sj];
    z1e_i = s_z1e[sj]; z1o_i = s_z1o[sj];
    zue_i = s_zue[sj]; zuo_i = s_zuo[sj];
  } else {
    size_t i_in = cfg_full + (size_t)jF_in * (size_t)nZnT + (size_t)kl;
    r1e_i = r1_e[i_in]; r1o_i = r1_o[i_in];
    rue_i = ru_e[i_in]; ruo_i = ru_o[i_in];
    z1e_i = z1_e[i_in]; z1o_i = z1_o[i_in];
    zue_i = zu_e[i_in]; zuo_i = zu_o[i_in];
  }

  double r1e_o, r1o_o, z1e_o, z1o_o, rue_o, ruo_o, zue_o, zuo_o;
  if (out_is_shared) {
    r1e_o = s_r1e[sj]; r1o_o = s_r1o[sj];
    rue_o = s_rue[sj]; ruo_o = s_ruo[sj];
    z1e_o = s_z1e[sj]; z1o_o = s_z1o[sj];
    zue_o = s_zue[sj]; zuo_o = s_zuo[sj];
  } else {
    size_t i_out = cfg_full + (size_t)jF_out * (size_t)nZnT + (size_t)kl;
    r1e_o = r1_e[i_out]; r1o_o = r1_o[i_out];
    rue_o = ru_e[i_out]; ruo_o = ru_o[i_out];
    z1e_o = z1_e[i_out]; z1o_o = z1_o[i_out];
    zue_o = zu_e[i_out]; zuo_o = zu_o[i_out];
  }

  double sH = sqrtSH[jH_local];
  double sqrtSF_i = sqrtSF[jF_in];
  double sqrtSF_o = sqrtSF[jF_out];
  double sF_i = sqrtSF_i * sqrtSF_i;
  double sF_o = sqrtSF_o * sqrtSF_o;
  size_t iHalf = cfg_half + (size_t)jH_local * (size_t)nZnT + (size_t)kl;

  double r12_v  = 0.5 * ((r1e_i + r1e_o) + sH * (r1o_i + r1o_o));
  double ru12_v = 0.5 * ((rue_i + rue_o) + sH * (ruo_i + ruo_o));
  double zu12_v = 0.5 * ((zue_i + zue_o) + sH * (zuo_i + zuo_o));
  double rs_v   = ((r1e_o - r1e_i) + sH * (r1o_o - r1o_i)) / deltaS;
  double zs_v   = ((z1e_o - z1e_i) + sH * (z1o_o - z1o_i)) / deltaS;

  double tau1 = ru12_v * zs_v - rs_v * zu12_v;
  double tau2 = ruo_o * z1o_o + ruo_i * z1o_i -
                zuo_o * r1o_o - zuo_i * r1o_i +
                (rue_o * z1o_o + rue_i * z1o_i -
                 zue_o * r1o_o - zue_i * r1o_i) / sH;
  double tau_v = tau1 + dSHalfDsInterp * tau2;

  r12[iHalf]  = r12_v;
  ru12[iHalf] = ru12_v;
  zu12[iHalf] = zu12_v;
  rs[iHalf]   = rs_v;
  zs[iHalf]   = zs_v;
  tau[iHalf]  = tau_v;

  double gsqrt_v = tau_v * r12_v;
  double guu_v = 0.5 * ((rue_i * rue_i + zue_i * zue_i) +
                         (rue_o * rue_o + zue_o * zue_o) +
                         sF_i * (ruo_i * ruo_i + zuo_i * zuo_i) +
                         sF_o * (ruo_o * ruo_o + zuo_o * zuo_o)) +
                  sH * ((rue_i * ruo_i + zue_i * zuo_i) +
                        (rue_o * ruo_o + zue_o * zuo_o));
  double gvv_v = 0.5 * (r1e_i * r1e_i + r1e_o * r1e_o +
                        sF_i * r1o_i * r1o_i + sF_o * r1o_o * r1o_o) +
                 sH * (r1e_i * r1o_i + r1e_o * r1o_o);
  double guv_v = 0.0;
  if (lthreed) {
    // rv/zv are not cached; both jF positions read global memory.
    double rve_i = rv_e[cfg_full + jF_in  * nZnT + kl];
    double rvo_i = rv_o[cfg_full + jF_in  * nZnT + kl];
    double zve_i = zv_e[cfg_full + jF_in  * nZnT + kl];
    double zvo_i = zv_o[cfg_full + jF_in  * nZnT + kl];
    double rve_o = rv_e[cfg_full + jF_out * nZnT + kl];
    double rvo_o = rv_o[cfg_full + jF_out * nZnT + kl];
    double zve_o = zv_e[cfg_full + jF_out * nZnT + kl];
    double zvo_o = zv_o[cfg_full + jF_out * nZnT + kl];
    guv_v = 0.5 * ((rue_i * rve_i + zue_i * zve_i) +
                   (rue_o * rve_o + zue_o * zve_o) +
                   sF_i * (ruo_i * rvo_i + zuo_i * zvo_i) +
                   sF_o * (ruo_o * rvo_o + zuo_o * zvo_o) +
                   sH * ((rue_i * rvo_i + zue_i * zvo_i) +
                         (rue_o * rvo_o + zue_o * zvo_o) +
                         (rve_i * ruo_i + zve_i * zuo_i) +
                         (rve_o * ruo_o + zve_o * zuo_o)));
    gvv_v += 0.5 * ((rve_i * rve_i + zve_i * zve_i) +
                    (rve_o * rve_o + zve_o * zve_o) +
                    sF_i * (rvo_i * rvo_i + zvo_i * zvo_i) +
                    sF_o * (rvo_o * rvo_o + zvo_o * zvo_o)) +
             sH * ((rve_i * rvo_i + zve_i * zvo_i) +
                   (rve_o * rvo_o + zve_o * zvo_o));
  }
  gsqrt[iHalf] = gsqrt_v;
  guu[iHalf]   = guu_v;
  gvv[iHalf]   = gvv_v;
  guv[iHalf]   = guv_v;

  int l = kl % nThetaEff;
  double contrib = signOfJacobian * gsqrt_v * wInt[l];
  atomicAdd(&dVdsH[config * ns_h + jH_local], contrib);
}

// k_update_dvdsh: one block per half-grid surface sums dVdsH = signOfJacobian *
// sum_kl gsqrt * wInt over kl.
__global__ void k_update_dvdsh(int n_config, int ns_h, int nZnT, int nThetaEff,
                                double signOfJacobian,
                                const double* __restrict__ gsqrt,
                                const double* __restrict__ wInt,
                                double* __restrict__ dVdsH) {
  // Serial single-thread kl accumulation matching CPU order.
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH_local = blockIdx.x;
  if (jH_local >= ns_h) return;
  if (threadIdx.x != 0) return;
  double acc = 0.0;
  size_t cfg_gsqrt = (size_t)config * ns_h * nZnT;
  for (int kl = 0; kl < nZnT; ++kl) {
    int l = kl % nThetaEff;
    acc += gsqrt[cfg_gsqrt + jH_local * nZnT + kl] * wInt[l];
  }
  dVdsH[config * ns_h + jH_local] = signOfJacobian * acc;
}

// Recomputes from gsqrt the dVdsH a configuration masked before the batch ended
// had zeroed; one block per configuration.
__global__ void k_recompute_dvdsh_if_zeroed(
    int n_config, int ns_h, int nZnT, int nThetaEff, double signOfJacobian,
    const double* __restrict__ gsqrt, const double* __restrict__ wInt,
    double* __restrict__ dVdsH) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (threadIdx.x != 0) return;
  if (dVdsH[(size_t)config * ns_h + 0] != 0.0) return;
  size_t cfg_gsqrt = (size_t)config * ns_h * nZnT;
  for (int jH = 0; jH < ns_h; ++jH) {
    double acc = 0.0;
    for (int kl = 0; kl < nZnT; ++kl) {
      acc += gsqrt[cfg_gsqrt + (size_t)jH * nZnT + kl] * wInt[kl % nThetaEff];
    }
    dVdsH[(size_t)config * ns_h + jH] = signOfJacobian * acc;
  }
}

// k_buco_bvco: bucoH and bvcoH as sums of bsubu and bsubv against wInt, one
// block per half-grid surface.
__global__ void k_buco_bvco(int n_config, int ns_h, int nZnT, int nThetaEff,
                              const double* __restrict__ bsubu,
                              const double* __restrict__ bsubv,
                              const double* __restrict__ wInt,
                              double* __restrict__ bucoH,
                              double* __restrict__ bvcoH) {
  // Serial single-thread kl accumulation matching CPU order.
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH_local = blockIdx.x;
  if (jH_local >= ns_h) return;
  if (threadIdx.x != 0) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  double accu = 0.0, accv = 0.0;
  for (int kl = 0; kl < nZnT; ++kl) {
    int l = kl % nThetaEff;
    double w = wInt[l];
    accu += bsubu[cfg_half + jH_local * nZnT + kl] * w;
    accv += bsubv[cfg_half + jH_local * nZnT + kl] * w;
  }
  bucoH[cfg_prof + jH_local] = accu;
  bvcoH[cfg_prof + jH_local] = accv;
}

// k_radial_interior: radial derivatives of bucoH, bvcoH and presH, the
// full-grid dVds and the residual equiF on interior surfaces.
__global__ void k_radial_interior(int n_config, int ns_h, int ns_local,
                                    int nsi, int nsMinFi_to_nsMinH_offset,
                                    int nsMinFi_to_nsMinF1_offset,
                                    double signByDeltaS, double invDeltaS,
                                    const double* __restrict__ bucoH,
                                    const double* __restrict__ bvcoH,
                                    const double* __restrict__ presH,
                                    const double* __restrict__ dVdsH,
                                    const double* __restrict__ chipF,
                                    const double* __restrict__ phipF,
                                    double* __restrict__ jcurvF,
                                    double* __restrict__ jcuruF,
                                    double* __restrict__ presgradF,
                                    double* __restrict__ dVdsF,
                                    double* __restrict__ equiF) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  int jFi_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (jFi_local >= nsi) return;
  size_t cfg_h = (size_t)config * (size_t)ns_h;
  size_t cfg_f = (size_t)config * (size_t)ns_local;
  size_t cfg_i = (size_t)config * (size_t)nsi;
  // jFi - nsMinH = jFi_local + nsMinFi_to_nsMinH_offset
  int jH_idx = jFi_local + nsMinFi_to_nsMinH_offset;
  int jF_idx = jFi_local + nsMinFi_to_nsMinF1_offset;
  double jcv = signByDeltaS * (bucoH[cfg_h + jH_idx] - bucoH[cfg_h + jH_idx - 1]);
  double jcu = -signByDeltaS * (bvcoH[cfg_h + jH_idx] - bvcoH[cfg_h + jH_idx - 1]);
  double pg = (presH[cfg_h + jH_idx] - presH[cfg_h + jH_idx - 1]) * invDeltaS;
  double dV = 0.5 * (dVdsH[cfg_h + jH_idx] + dVdsH[cfg_h + jH_idx - 1]);
  jcurvF[cfg_i + jFi_local] = jcv;
  jcuruF[cfg_i + jFi_local] = jcu;
  presgradF[cfg_i + jFi_local] = pg;
  dVdsF[cfg_i + jFi_local] = dV;
  double cp = chipF[cfg_f + jF_idx];
  double pp = phipF[cfg_f + jF_idx];
  equiF[cfg_i + jFi_local] = (cp * jcv - pp * jcu) / dV + pg;
}

// k_pm_half_reductions: half-grid reductions over kl of the
// preconditioner-matrix scratch ax, bx and cx, weighted by pTau.
__global__ void k_pm_half_reductions(int n_config, int ns_local, int ns_h,
                                       int nZnT, int nThetaEff,
                                       double pFactor, double deltaS,
                                       int nsMinH, int nsMinF1,
                                       const double* __restrict__ r12,
                                       const double* __restrict__ totalPressure,
                                       const double* __restrict__ tau,
                                       const double* __restrict__ wInt,
                                       const double* __restrict__ xu12,
                                       const double* __restrict__ xu_e,
                                       const double* __restrict__ xu_o,
                                       const double* __restrict__ x1_o,
                                       const double* __restrict__ xs,
                                       const double* __restrict__ sqrtSH,
                                       const double* __restrict__ bsupv,
                                       const double* __restrict__ gsqrt,
                                       double* __restrict__ ax_scratch,
                                       double* __restrict__ bx_scratch,
                                       double* __restrict__ cx_scratch) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH = blockIdx.x;
  if (jH >= ns_h) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h     * (size_t)nZnT;
  size_t cfg_full = (size_t)config * (size_t)ns_local * (size_t)nZnT;
  size_t cfg_ax   = (size_t)config * (size_t)ns_h * 4;
  size_t cfg_bx   = (size_t)config * (size_t)ns_h * 3;
  size_t cfg_cx   = (size_t)config * (size_t)ns_h;
  double sH = sqrtSH[jH];
  int jH_global = jH + nsMinH;
  int jF_in_local = jH_global - nsMinF1;
  int jF_out_local = jF_in_local + 1;
  __shared__ double s_ax0[32], s_ax1[32], s_ax2[32], s_ax3[32];
  __shared__ double s_bx0[32], s_bx1[32], s_bx2[32];
  __shared__ double s_cx[32];
  double ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
  double bx0 = 0.0, bx1 = 0.0, bx2 = 0.0;
  double cxv = 0.0;
  double invSH = 1.0 / sH;
  for (int kl = threadIdx.x; kl < nZnT; kl += blockDim.x) {
    size_t iHalf = cfg_half + (size_t)jH * (size_t)nZnT + (size_t)kl;
    size_t iFull_0 = cfg_full + (size_t)jF_in_local  * (size_t)nZnT + (size_t)kl;
    size_t iFull_1 = cfg_full + (size_t)jF_out_local * (size_t)nZnT + (size_t)kl;
    int l = kl % nThetaEff;
    double pTau = pFactor * r12[iHalf] * totalPressure[iHalf] / tau[iHalf]
                  * wInt[l];
    double t1a = xu12[iHalf] / deltaS;
    double t2a = 0.25 * (xu_e[iFull_1] * invSH + xu_o[iFull_1]) * invSH;
    double t3a = 0.25 * (xu_e[iFull_0] * invSH + xu_o[iFull_0]) * invSH;
    ax0 += pTau * t1a * t1a;
    ax1 += pTau * (t1a + t2a) * (-t1a + t3a);
    ax2 += pTau * (t1a + t2a) * (t1a + t2a);
    ax3 += pTau * (-t1a + t3a) * (-t1a + t3a);
    double t1b = 0.5 * (xs[iHalf] + 0.5 * invSH * x1_o[iFull_1]);
    double t2b = 0.5 * (xs[iHalf] + 0.5 * invSH * x1_o[iFull_0]);
    bx0 += pTau * t1b * t2b;
    bx1 += pTau * t1b * t1b;
    bx2 += pTau * t2b * t2b;
    double bv = bsupv[iHalf];
    cxv += 0.25 * pFactor * bv * bv * gsqrt[iHalf] * wInt[l];
  }
  s_ax0[threadIdx.x] = ax0; s_ax1[threadIdx.x] = ax1;
  s_ax2[threadIdx.x] = ax2; s_ax3[threadIdx.x] = ax3;
  s_bx0[threadIdx.x] = bx0; s_bx1[threadIdx.x] = bx1; s_bx2[threadIdx.x] = bx2;
  s_cx[threadIdx.x] = cxv;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_ax0[threadIdx.x] += s_ax0[threadIdx.x + stride];
      s_ax1[threadIdx.x] += s_ax1[threadIdx.x + stride];
      s_ax2[threadIdx.x] += s_ax2[threadIdx.x + stride];
      s_ax3[threadIdx.x] += s_ax3[threadIdx.x + stride];
      s_bx0[threadIdx.x] += s_bx0[threadIdx.x + stride];
      s_bx1[threadIdx.x] += s_bx1[threadIdx.x + stride];
      s_bx2[threadIdx.x] += s_bx2[threadIdx.x + stride];
      s_cx[threadIdx.x]  += s_cx[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    ax_scratch[cfg_ax + jH * 4 + 0] = s_ax0[0];
    ax_scratch[cfg_ax + jH * 4 + 1] = s_ax1[0];
    ax_scratch[cfg_ax + jH * 4 + 2] = s_ax2[0];
    ax_scratch[cfg_ax + jH * 4 + 3] = s_ax3[0];
    bx_scratch[cfg_bx + jH * 3 + 0] = s_bx0[0];
    bx_scratch[cfg_bx + jH * 3 + 1] = s_bx1[0];
    bx_scratch[cfg_bx + jH * 3 + 2] = s_bx2[0];
    cx_scratch[cfg_cx + jH] = s_cx[0];
  }
}

// k_pm_radial_assembly: per surface jH (half-grid), assemble axm/bxm.
// Batched execution: configuration axis on blockIdx.y. All buffers per-config.
__global__ void k_pm_assemble_half(int n_config, int ns_h,
                                     int kEven, int kOdd,
                                     const double* __restrict__ ax,
                                     const double* __restrict__ bx,
                                     const double* __restrict__ sm,
                                     const double* __restrict__ sp,
                                     double* __restrict__ m_axm,
                                     double* __restrict__ m_bxm) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  int jH = blockIdx.x * blockDim.x + threadIdx.x;
  if (jH >= ns_h) return;
  size_t cfg_ax  = (size_t)config * (size_t)ns_h * 4;
  size_t cfg_bx  = (size_t)config * (size_t)ns_h * 3;
  size_t cfg_sm  = (size_t)config * (size_t)ns_h;
  size_t cfg_m   = (size_t)config * (size_t)ns_h * 2;
  m_axm[cfg_m + jH * 2 + kEven] = -ax[cfg_ax + jH * 4 + 0];
  m_axm[cfg_m + jH * 2 + kOdd]  =  ax[cfg_ax + jH * 4 + 1] * sm[cfg_sm + jH] * sp[cfg_sm + jH];
  m_bxm[cfg_m + jH * 2 + kEven] = bx[cfg_bx + jH * 3 + 0];
  m_bxm[cfg_m + jH * 2 + kOdd]  = bx[cfg_bx + jH * 3 + 0] * sm[cfg_sm + jH] * sp[cfg_sm + jH];
}

// k_pm_assemble_full: full-grid axd, bxd and cxd from the inner and outer
// half-grid scratch, without the out-of-range halves.
__global__ void k_pm_assemble_full(int n_config, int ns_h, int ns_force_local,
                                     int ns_total,
                                     int kEven, int kOdd, int nsMinF, int nsMinH,
                                     const double* __restrict__ ax,
                                     const double* __restrict__ bx,
                                     const double* __restrict__ cx,
                                     const double* __restrict__ sm,
                                     const double* __restrict__ sp,
                                     double* __restrict__ m_axd,
                                     double* __restrict__ m_bxd,
                                     double* __restrict__ m_cxd) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  int jF_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_local >= ns_force_local) return;
  size_t cfg_ax  = (size_t)config * (size_t)ns_h * 4;
  size_t cfg_bx  = (size_t)config * (size_t)ns_h * 3;
  size_t cfg_cx  = (size_t)config * (size_t)ns_h;
  size_t cfg_sm  = (size_t)config * (size_t)ns_h;
  size_t cfg_dxx = (size_t)config * (size_t)ns_force_local * 2;
  size_t cfg_cxd = (size_t)config * (size_t)ns_force_local;
  int jF = jF_local + nsMinF;
  int jH_i_global = jF - 1;
  int jH_o_global = jF;
  int jH_i = jH_i_global - nsMinH;
  int jH_o = jH_o_global - nsMinH;
  bool i_valid = (jF > 0);
  bool o_valid = (jF < ns_total - 1);
  double axd_e = (i_valid ? ax[cfg_ax + jH_i * 4 + 0] : 0.0)
               + (o_valid ? ax[cfg_ax + jH_o * 4 + 0] : 0.0);
  double sm_i = i_valid ? sm[cfg_sm + jH_i] : 0.0;
  double sp_o = o_valid ? sp[cfg_sm + jH_o] : 0.0;
  double axd_o = (i_valid ? ax[cfg_ax + jH_i * 4 + 2] * sm_i * sm_i : 0.0)
               + (o_valid ? ax[cfg_ax + jH_o * 4 + 3] * sp_o * sp_o : 0.0);
  m_axd[cfg_dxx + jF_local * 2 + kEven] = axd_e;
  m_axd[cfg_dxx + jF_local * 2 + kOdd]  = axd_o;
  double bxd_e = (i_valid ? bx[cfg_bx + jH_i * 3 + 1] : 0.0)
               + (o_valid ? bx[cfg_bx + jH_o * 3 + 2] : 0.0);
  double bxd_o = (i_valid ? bx[cfg_bx + jH_i * 3 + 1] * sm_i * sm_i : 0.0)
               + (o_valid ? bx[cfg_bx + jH_o * 3 + 2] * sp_o * sp_o : 0.0);
  m_bxd[cfg_dxx + jF_local * 2 + kEven] = bxd_e;
  m_bxd[cfg_dxx + jF_local * 2 + kOdd]  = bxd_o;
  double cxd_v = (i_valid ? cx[cfg_cx + jH_i] : 0.0) + (o_valid ? cx[cfg_cx + jH_o] : 0.0);
  m_cxd[cfg_cxd + jF_local] = cxd_v;
}

// k_ulp_half_reductions: half-grid contributions to bLambda, dLambda and
// cLambda, written at jH + 1 with slot 0 for the axis.
__global__ void k_ulp_half_reductions(int n_config, int ns_h, int lambda_stride,
                                       int nZnT, int nThetaEff,
                                       bool lthreed,
                                       const double* __restrict__ guu,
                                       const double* __restrict__ guv,
                                       const double* __restrict__ gvv,
                                       const double* __restrict__ gsqrt,
                                       const double* __restrict__ wInt,
                                       double* __restrict__ bLambda,
                                       double* __restrict__ dLambda,
                                       double* __restrict__ cLambda) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH = blockIdx.x;
  if (jH >= ns_h) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t cfg_prof = (size_t)config * (size_t)lambda_stride;
  __shared__ double s_b[32], s_d[32], s_c[32];
  double accb = 0.0, accd = 0.0, accc = 0.0;
  for (int kl = threadIdx.x; kl < nZnT; kl += blockDim.x) {
    size_t i = cfg_half + (size_t)jH * (size_t)nZnT + (size_t)kl;
    int l = kl % nThetaEff;
    double w = wInt[l];
    double inv_g = 1.0 / gsqrt[i];
    accb += guu[i] * inv_g * w;
    accc += gvv[i] * inv_g * w;
    if (lthreed) {
      accd += guv[i] * inv_g * w;
    }
  }
  s_b[threadIdx.x] = accb;
  s_d[threadIdx.x] = accd;
  s_c[threadIdx.x] = accc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_b[threadIdx.x] += s_b[threadIdx.x + stride];
      s_d[threadIdx.x] += s_d[threadIdx.x + stride];
      s_c[threadIdx.x] += s_c[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    bLambda[cfg_prof + jH + 1] = s_b[0];
    cLambda[cfg_prof + jH + 1] = s_c[0];
    if (lthreed) {
      dLambda[cfg_prof + jH + 1] = s_d[0];
    } else {
      dLambda[cfg_prof + jH + 1] = 0.0;
    }
  }
}

// k_ulp_axis_extrap: the axis slot of each lambda profile takes the adjacent
// half-grid value, where the partition holds the axis.
__global__ void k_ulp_axis_extrap(int n_config, int lambda_stride, int axis_present,
                                    double* __restrict__ bLambda,
                                    double* __restrict__ dLambda,
                                    double* __restrict__ cLambda) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (threadIdx.x != 0) return;
  if (!axis_present) return;
  size_t cfg_prof = (size_t)config * (size_t)lambda_stride;
  bLambda[cfg_prof + 0] = bLambda[cfg_prof + 1];
  dLambda[cfg_prof + 0] = dLambda[cfg_prof + 1];
  cLambda[cfg_prof + 0] = cLambda[cfg_prof + 1];
}

// k_ulp_full_grid_average: averages the offset-by-one half-grid lambda profiles
// to the full grid on [jMin, ns_con_local).
__global__ void k_ulp_full_grid_average(int n_config, int lambda_stride,
                                          int ns_con_local, int jMin,
                                          int nsMinH_offset,  // nsMinF - nsMinH
                                          const double* __restrict__ bLambda_in,
                                          const double* __restrict__ dLambda_in,
                                          const double* __restrict__ cLambda_in,
                                          double* __restrict__ bLambda_out,
                                          double* __restrict__ dLambda_out,
                                          double* __restrict__ cLambda_out) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  int jF_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_local >= ns_con_local) return;
  if (jF_local < jMin) return;
  size_t cfg_in  = (size_t)config * (size_t)lambda_stride;
  size_t cfg_out = (size_t)config * (size_t)lambda_stride;
  // Offset-by-one input indices: the CPU's bLambda[jF + 1 - nsMinH] is
  // bLambda_in[jF_local + nsMinH_offset + 1].
  int jH_in_off = jF_local + nsMinH_offset;
  bLambda_out[cfg_out + jF_local] = 0.5 * (bLambda_in[cfg_in + jH_in_off + 1] +
                                  bLambda_in[cfg_in + jH_in_off]);
  dLambda_out[cfg_out + jF_local] = 0.5 * (dLambda_in[cfg_in + jH_in_off + 1] +
                                  dLambda_in[cfg_in + jH_in_off]);
  cLambda_out[cfg_out + jF_local] = 0.5 * (cLambda_in[cfg_in + jH_in_off + 1] +
                                  cLambda_in[cfg_in + jH_in_off]);
}

// k_ulp_assemble: the spectral lambdaPreconditioner from bLambda, dLambda and
// cLambda, one thread per (configuration, jF_local, n, m).
__global__ void k_ulp_assemble(int n_config, int ns_con_local, int lambda_stride,
                                int jMin,
                                int mpol, int ntor,
                                int nfp, double pFactor,
                                const double* __restrict__ bLambda,
                                const double* __restrict__ dLambda,
                                const double* __restrict__ cLambda,
                                const double* __restrict__ sqrtSF,
                                int sqrtSF_off,  // nsMinF - nsMinF1
                                double* __restrict__ lambdaPreconditioner,
                                const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z / ns_con_local;
  int jF_local = blockIdx.z - config * ns_con_local;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int n = blockIdx.y;
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_local >= ns_con_local || n > ntor || m >= mpol) return;
  size_t cfg_lp   = (size_t)config * (size_t)ns_con_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t cfg_blam = (size_t)config * (size_t)lambda_stride;
  int idx_mn = (jF_local * mpol + m) * (ntor + 1) + n;
  if (jF_local < jMin) {
    lambdaPreconditioner[cfg_lp + idx_mn] = 0.0;
    return;
  }
  if (m == 0 && n == 0) {
    lambdaPreconditioner[cfg_lp + idx_mn] = 0.0;
    return;
  }
  double tnn = (double)(n * nfp) * (double)(n * nfp);
  int tmm = m * m;
  double pwr = (double)tmm / (16.0 * 16.0);
  if (pwr > 8.0) pwr = 8.0;
  double tmn = 2.0 * m * n * nfp;
  double b = bLambda[cfg_blam + jF_local];
  double d = dLambda[cfg_blam + jF_local];
  double c = cLambda[cfg_blam + jF_local];
  double faclam = tnn * b + tmn * copysign(d, b) + (double)tmm * c;
  if (faclam == 0.0) faclam = -1.0e-10;
  double sFjF = sqrtSF[jF_local + sqrtSF_off];  // sqrtSF is shared (radial
  // grid invariant across configs in our broadcast execution mode; for distinct-input execution
  // with same radial grid still invariant).
  lambdaPreconditioner[cfg_lp + idx_mn] = pFactor / faclam * pow(sFjF, pwr);
}

// k_compute_mhd_forces: IdealMhdModel::computeMHDForces per (jF_local_force,
// kl); surfaces from jMaxRZ on write zeros.
__global__ __launch_bounds__(64, 12) void k_compute_mhd_forces(
    int n_config, int ns_local, int ns_force_local, int nZnT, bool lthreed,
    int nsMinF, int nsMinF1, int nsMinH, int nsMaxH, int jMaxRZ,
    double deltaS,
    const double* __restrict__ r1_e, const double* __restrict__ r1_o,
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ rv_e, const double* __restrict__ rv_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    const double* __restrict__ zv_e, const double* __restrict__ zv_o,
    const double* __restrict__ z1_o,
    const double* __restrict__ r12, const double* __restrict__ ru12,
    const double* __restrict__ zu12, const double* __restrict__ rs,
    const double* __restrict__ zs, const double* __restrict__ tau,
    const double* __restrict__ totalPressure,
    const double* __restrict__ gsqrt,
    const double* __restrict__ bsupu, const double* __restrict__ bsupv,
    const double* __restrict__ sqrtSF, const double* __restrict__ sqrtSH,
    double* __restrict__ armn_e, double* __restrict__ armn_o,
    double* __restrict__ azmn_e, double* __restrict__ azmn_o,
    double* __restrict__ brmn_e, double* __restrict__ brmn_o,
    double* __restrict__ bzmn_e, double* __restrict__ bzmn_o,
    double* __restrict__ crmn_e, double* __restrict__ crmn_o,
    double* __restrict__ czmn_e, double* __restrict__ czmn_o) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jF_local = blockIdx.y;
  if (jF_local >= ns_force_local) return;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  int ns_h_total = nsMaxH - nsMinH;
  size_t cfg_full  = (size_t)config * (size_t)ns_local       * (size_t)nZnT;
  size_t cfg_half  = (size_t)config * (size_t)ns_h_total     * (size_t)nZnT;
  size_t cfg_force = (size_t)config * (size_t)ns_force_local * (size_t)nZnT;
  size_t f_idx = cfg_force + (size_t)jF_local * (size_t)nZnT + (size_t)kl;
  int jF_global = jF_local + nsMinF;

  // Zero output for jF beyond jMaxRZ.
  if (jF_global >= jMaxRZ) {
    armn_e[f_idx] = 0.0; armn_o[f_idx] = 0.0;
    azmn_e[f_idx] = 0.0; azmn_o[f_idx] = 0.0;
    brmn_e[f_idx] = 0.0; brmn_o[f_idx] = 0.0;
    bzmn_e[f_idx] = 0.0; bzmn_o[f_idx] = 0.0;
    if (lthreed) {
      crmn_e[f_idx] = 0.0; crmn_o[f_idx] = 0.0;
      czmn_e[f_idx] = 0.0; czmn_o[f_idx] = 0.0;
    }
    return;
  }

  // Inside half-grid local index.
  int jH_in_local = jF_global - 1 - nsMinH;
  // Outside half-grid local index.
  int jH_out_local = jF_global - nsMinH;

  double sqrtSHi = 1.0, sqrtSHo = 1.0;
  double P_i = 0.0, rup_i = 0.0, zup_i = 0.0, rsp_i = 0.0, zsp_i = 0.0;
  double taup_i = 0.0;
  double gbubu_i = 0.0, gbubv_i = 0.0, gbvbv_i = 0.0;
  if (jF_global > 0 && jH_in_local >= 0 && jH_in_local < ns_h_total) {
    size_t i_in = cfg_half + (size_t)jH_in_local * (size_t)nZnT + (size_t)kl;
    double tp = totalPressure[i_in];
    P_i = r12[i_in] * tp;
    rup_i = ru12[i_in] * P_i;
    zup_i = zu12[i_in] * P_i;
    rsp_i = rs[i_in] * P_i;
    zsp_i = zs[i_in] * P_i;
    taup_i = tau[i_in] * tp;
    double g = gsqrt[i_in];
    double bu = bsupu[i_in];
    double bv = bsupv[i_in];
    gbubu_i = g * bu * bu;
    gbubv_i = g * bu * bv;
    gbvbv_i = g * bv * bv;
    sqrtSHi = sqrtSH[jH_in_local];
  }

  double P_o = 0.0, rup_o = 0.0, zup_o = 0.0, rsp_o = 0.0, zsp_o = 0.0;
  double taup_o = 0.0;
  double gbubu_o = 0.0, gbubv_o = 0.0, gbvbv_o = 0.0;
  if (jH_out_local >= 0 && jH_out_local < ns_h_total) {
    size_t i_out = cfg_half + (size_t)jH_out_local * (size_t)nZnT + (size_t)kl;
    double tp = totalPressure[i_out];
    P_o = r12[i_out] * tp;
    rup_o = ru12[i_out] * P_o;
    zup_o = zu12[i_out] * P_o;
    rsp_o = rs[i_out] * P_o;
    zsp_o = zs[i_out] * P_o;
    taup_o = tau[i_out] * tp;
    double g = gsqrt[i_out];
    double bu = bsupu[i_out];
    double bv = bsupv[i_out];
    gbubu_o = g * bu * bu;
    gbubv_o = g * bu * bv;
    gbvbv_o = g * bv * bv;
    sqrtSHo = sqrtSH[jH_out_local];
  }

  // Full-grid (jF) values; indexed by jF_global - nsMinF1.
  int jF_full_local = jF_global - nsMinF1;
  size_t g_idx = cfg_full + (size_t)jF_full_local * (size_t)nZnT + (size_t)kl;
  double r1e = r1_e[g_idx], r1o = r1_o[g_idx];
  double rue = ru_e[g_idx], ruo = ru_o[g_idx];
  double zue = zu_e[g_idx], zuo = zu_o[g_idx];
  double z1o = z1_o[g_idx];

  double sqrtSF_jF = sqrtSF[jF_full_local];
  double sFull = sqrtSF_jF * sqrtSF_jF;

  double invDS = 1.0 / deltaS;
  double invSHo = 1.0 / sqrtSHo;
  double invSHi = 1.0 / sqrtSHi;
  double P_avg = 0.5 * (P_o + P_i);
  double P_wavg = 0.5 * (P_o * invSHo + P_i * invSHi);
  double gbubu_avg = 0.5 * (gbubu_o + gbubu_i);
  double gbubu_wavg = 0.5 * (gbubu_o * sqrtSHo + gbubu_i * sqrtSHi);
  double gbvbv_avg = 0.5 * (gbvbv_o + gbvbv_i);
  double gbvbv_wavg = 0.5 * (gbvbv_o * sqrtSHo + gbvbv_i * sqrtSHi);

  // A_R
  double armn_e_v = (zup_o - zup_i) * invDS + 0.5 * (taup_o + taup_i)
                  - gbvbv_avg * r1e - gbvbv_wavg * r1o;
  double armn_o_v = (zup_o * sqrtSHo - zup_i * sqrtSHi) * invDS
                  - 0.5 * P_wavg * zue - 0.5 * P_avg * zuo
                  + 0.5 * (taup_o * sqrtSHo + taup_i * sqrtSHi)
                  - gbvbv_wavg * r1e - gbvbv_avg * r1o * sFull;

  // A_Z
  double azmn_e_v = -(rup_o - rup_i) * invDS;
  double azmn_o_v = -(rup_o * sqrtSHo - rup_i * sqrtSHi) * invDS
                  + 0.5 * P_wavg * rue + 0.5 * P_avg * ruo;

  // B_R
  double brmn_e_v = 0.5 * (zsp_o + zsp_i) + 0.5 * P_wavg * z1o
                  - gbubu_avg * rue - gbubu_wavg * ruo;
  double brmn_o_v = 0.5 * (zsp_o * sqrtSHo + zsp_i * sqrtSHi)
                  + 0.5 * P_avg * z1o
                  - gbubu_wavg * rue - gbubu_avg * ruo * sFull;

  // B_Z
  double bzmn_e_v = -0.5 * (rsp_o + rsp_i) - 0.5 * P_wavg * r1o
                  - gbubu_avg * zue - gbubu_wavg * zuo;
  double bzmn_o_v = -0.5 * (rsp_o * sqrtSHo + rsp_i * sqrtSHi)
                  - 0.5 * P_avg * r1o
                  - gbubu_wavg * zue - gbubu_avg * zuo * sFull;

  if (lthreed) {
    double gbubv_avg = 0.5 * (gbubv_o + gbubv_i);
    double gbubv_wavg = 0.5 * (gbubv_o * sqrtSHo + gbubv_i * sqrtSHi);
    double rve = rv_e[g_idx], rvo = rv_o[g_idx];
    double zve = zv_e[g_idx], zvo = zv_o[g_idx];

    // 3D contributions to B_R, B_Z
    brmn_e_v -= gbubv_avg * rve + gbubv_wavg * rvo;
    brmn_o_v -= gbubv_wavg * rve + gbubv_avg * rvo * sFull;
    bzmn_e_v -= gbubv_avg * zve + gbubv_wavg * zvo;
    bzmn_o_v -= gbubv_wavg * zve + gbubv_avg * zvo * sFull;

    // C_R
    double crmn_e_v = gbubv_avg * rue + gbubv_wavg * ruo
                    + gbvbv_avg * rve + gbvbv_wavg * rvo;
    double crmn_o_v = gbubv_wavg * rue + gbubv_avg * ruo * sFull
                    + gbvbv_wavg * rve + gbvbv_avg * rvo * sFull;

    // C_Z
    double czmn_e_v = gbubv_avg * zue + gbubv_wavg * zuo
                    + gbvbv_avg * zve + gbvbv_wavg * zvo;
    double czmn_o_v = gbubv_wavg * zue + gbubv_avg * zuo * sFull
                    + gbvbv_wavg * zve + gbvbv_avg * zvo * sFull;

    crmn_e[f_idx] = crmn_e_v; crmn_o[f_idx] = crmn_o_v;
    czmn_e[f_idx] = czmn_e_v; czmn_o[f_idx] = czmn_o_v;
  }

  armn_e[f_idx] = armn_e_v; armn_o[f_idx] = armn_o_v;
  azmn_e[f_idx] = azmn_e_v; azmn_o[f_idx] = azmn_o_v;
  brmn_e[f_idx] = brmn_e_v; brmn_o[f_idx] = brmn_o_v;
  bzmn_e[f_idx] = bzmn_e_v; bzmn_o[f_idx] = bzmn_o_v;
}

// Force-grid pair coarsening of k_compute_mhd_forces: a block serves two
// adjacent surfaces sharing the half-grid surface between them.
__global__ __launch_bounds__(128, 6) void k_compute_mhd_forces_pair(
    int n_config, int ns_local, int ns_force_local, int nZnT, bool lthreed,
    int nsMinF, int nsMinF1, int nsMinH, int nsMaxH, int jMaxRZ,
    double deltaS,
    const double* __restrict__ r1_e, const double* __restrict__ r1_o,
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ rv_e, const double* __restrict__ rv_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    const double* __restrict__ zv_e, const double* __restrict__ zv_o,
    const double* __restrict__ z1_o,
    const double* __restrict__ r12, const double* __restrict__ ru12,
    const double* __restrict__ zu12, const double* __restrict__ rs,
    const double* __restrict__ zs, const double* __restrict__ tau,
    const double* __restrict__ totalPressure,
    const double* __restrict__ gsqrt,
    const double* __restrict__ bsupu, const double* __restrict__ bsupv,
    const double* __restrict__ sqrtSF, const double* __restrict__ sqrtSH,
    double* __restrict__ armn_e, double* __restrict__ armn_o,
    double* __restrict__ azmn_e, double* __restrict__ azmn_o,
    double* __restrict__ brmn_e, double* __restrict__ brmn_o,
    double* __restrict__ bzmn_e, double* __restrict__ bzmn_o,
    double* __restrict__ crmn_e, double* __restrict__ crmn_o,
    double* __restrict__ czmn_e, double* __restrict__ czmn_o,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jF_pair = blockIdx.y;
  int my_jF_offset = threadIdx.y;  // 0 = lo, 1 = hi
  int jF_local = jF_pair * 2 + my_jF_offset;
  if (jF_local >= ns_force_local) return;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  int ns_h_total = nsMaxH - nsMinH;
  size_t cfg_full  = (size_t)config * (size_t)ns_local       * (size_t)nZnT;
  size_t cfg_half  = (size_t)config * (size_t)ns_h_total     * (size_t)nZnT;
  size_t cfg_force = (size_t)config * (size_t)ns_force_local * (size_t)nZnT;
  size_t f_idx = cfg_force + (size_t)jF_local * (size_t)nZnT + (size_t)kl;
  int jF_global = jF_local + nsMinF;

  // jF_lo's global jH is jF_lo + nsMinF (i.e., jH = jF_lo_global). Its local
  // index is jF_lo_global - nsMinH. This is the SHARED cache slot's jH.
  int jF_lo_global = jF_pair * 2 + nsMinF;
  int jH_shared_local = jF_lo_global - nsMinH;
  bool shared_valid = (jH_shared_local >= 0 && jH_shared_local < ns_h_total);

  // Shared layout: ten half-grid fields per x-lane, shared by the two jF lanes
  // at the same kl, sized to the block's x-extent.
  const int sj = threadIdx.x;
  const int sw = blockDim.x;
  extern __shared__ double s_pair_buf[];
  double* s_tp    = s_pair_buf + 0 * sw;
  double* s_r12   = s_pair_buf + 1 * sw;
  double* s_ru12  = s_pair_buf + 2 * sw;
  double* s_zu12  = s_pair_buf + 3 * sw;
  double* s_rs    = s_pair_buf + 4 * sw;
  double* s_zs    = s_pair_buf + 5 * sw;
  double* s_tau   = s_pair_buf + 6 * sw;
  double* s_gsqrt = s_pair_buf + 7 * sw;
  double* s_bsupu = s_pair_buf + 8 * sw;
  double* s_bsupv = s_pair_buf + 9 * sw;

  // The y = 0 thread loads the shared surface, which is also its own outer
  // half-grid surface.
  if (my_jF_offset == 0 && shared_valid) {
    size_t i_shared = cfg_half + (size_t)jH_shared_local * (size_t)nZnT + (size_t)kl;
    s_tp[sj]    = totalPressure[i_shared];
    s_r12[sj]   = r12[i_shared];
    s_ru12[sj]  = ru12[i_shared];
    s_zu12[sj]  = zu12[i_shared];
    s_rs[sj]    = rs[i_shared];
    s_zs[sj]    = zs[i_shared];
    s_tau[sj]   = tau[i_shared];
    s_gsqrt[sj] = gsqrt[i_shared];
    s_bsupu[sj] = bsupu[i_shared];
    s_bsupv[sj] = bsupv[i_shared];
  }
  __syncthreads();

  // Zero output for jF beyond jMaxRZ.
  if (jF_global >= jMaxRZ) {
    armn_e[f_idx] = 0.0; armn_o[f_idx] = 0.0;
    azmn_e[f_idx] = 0.0; azmn_o[f_idx] = 0.0;
    brmn_e[f_idx] = 0.0; brmn_o[f_idx] = 0.0;
    bzmn_e[f_idx] = 0.0; bzmn_o[f_idx] = 0.0;
    if (lthreed) {
      crmn_e[f_idx] = 0.0; crmn_o[f_idx] = 0.0;
      czmn_e[f_idx] = 0.0; czmn_o[f_idx] = 0.0;
    }
    return;
  }

  int jH_in_local  = jF_global - 1 - nsMinH;
  int jH_out_local = jF_global - nsMinH;

  // For y=0 (jF_lo): jH_out_local == jH_shared_local. Use shared if valid.
  // For y=1 (jF_hi): jH_in_local  == jH_shared_local. Use shared if valid.
  bool jH_in_is_shared  = (jH_in_local  == jH_shared_local) && shared_valid;
  bool jH_out_is_shared = (jH_out_local == jH_shared_local) && shared_valid;

  double sqrtSHi = 1.0, sqrtSHo = 1.0;
  double P_i = 0.0, rup_i = 0.0, zup_i = 0.0, rsp_i = 0.0, zsp_i = 0.0;
  double taup_i = 0.0;
  double gbubu_i = 0.0, gbubv_i = 0.0, gbvbv_i = 0.0;
  if (jF_global > 0 && jH_in_local >= 0 && jH_in_local < ns_h_total) {
    double tp, r12v, ru12v, zu12v, rsv, zsv, tauv, gv, bu, bv;
    if (jH_in_is_shared) {
      tp    = s_tp[sj];
      r12v  = s_r12[sj];
      ru12v = s_ru12[sj];
      zu12v = s_zu12[sj];
      rsv   = s_rs[sj];
      zsv   = s_zs[sj];
      tauv  = s_tau[sj];
      gv    = s_gsqrt[sj];
      bu    = s_bsupu[sj];
      bv    = s_bsupv[sj];
    } else {
      size_t i_in = cfg_half + (size_t)jH_in_local * (size_t)nZnT + (size_t)kl;
      tp    = totalPressure[i_in];
      r12v  = r12[i_in];
      ru12v = ru12[i_in];
      zu12v = zu12[i_in];
      rsv   = rs[i_in];
      zsv   = zs[i_in];
      tauv  = tau[i_in];
      gv    = gsqrt[i_in];
      bu    = bsupu[i_in];
      bv    = bsupv[i_in];
    }
    P_i = r12v * tp;
    rup_i = ru12v * P_i;
    zup_i = zu12v * P_i;
    rsp_i = rsv * P_i;
    zsp_i = zsv * P_i;
    taup_i = tauv * tp;
    gbubu_i = gv * bu * bu;
    gbubv_i = gv * bu * bv;
    gbvbv_i = gv * bv * bv;
    sqrtSHi = sqrtSH[jH_in_local];
  }

  double P_o = 0.0, rup_o = 0.0, zup_o = 0.0, rsp_o = 0.0, zsp_o = 0.0;
  double taup_o = 0.0;
  double gbubu_o = 0.0, gbubv_o = 0.0, gbvbv_o = 0.0;
  if (jH_out_local >= 0 && jH_out_local < ns_h_total) {
    double tp, r12v, ru12v, zu12v, rsv, zsv, tauv, gv, bu, bv;
    if (jH_out_is_shared) {
      tp    = s_tp[sj];
      r12v  = s_r12[sj];
      ru12v = s_ru12[sj];
      zu12v = s_zu12[sj];
      rsv   = s_rs[sj];
      zsv   = s_zs[sj];
      tauv  = s_tau[sj];
      gv    = s_gsqrt[sj];
      bu    = s_bsupu[sj];
      bv    = s_bsupv[sj];
    } else {
      size_t i_out = cfg_half + (size_t)jH_out_local * (size_t)nZnT + (size_t)kl;
      tp    = totalPressure[i_out];
      r12v  = r12[i_out];
      ru12v = ru12[i_out];
      zu12v = zu12[i_out];
      rsv   = rs[i_out];
      zsv   = zs[i_out];
      tauv  = tau[i_out];
      gv    = gsqrt[i_out];
      bu    = bsupu[i_out];
      bv    = bsupv[i_out];
    }
    P_o = r12v * tp;
    rup_o = ru12v * P_o;
    zup_o = zu12v * P_o;
    rsp_o = rsv * P_o;
    zsp_o = zsv * P_o;
    taup_o = tauv * tp;
    gbubu_o = gv * bu * bu;
    gbubv_o = gv * bu * bv;
    gbvbv_o = gv * bv * bv;
    sqrtSHo = sqrtSH[jH_out_local];
  }

  // Full-grid (jF) values; indexed by jF_global - nsMinF1.
  int jF_full_local = jF_global - nsMinF1;
  size_t g_idx = cfg_full + (size_t)jF_full_local * (size_t)nZnT + (size_t)kl;
  double r1e = r1_e[g_idx], r1o = r1_o[g_idx];
  double rue = ru_e[g_idx], ruo = ru_o[g_idx];
  double zue = zu_e[g_idx], zuo = zu_o[g_idx];
  double z1o = z1_o[g_idx];

  double sqrtSF_jF = sqrtSF[jF_full_local];
  double sFull = sqrtSF_jF * sqrtSF_jF;

  double invDS = 1.0 / deltaS;
  double invSHo = 1.0 / sqrtSHo;
  double invSHi = 1.0 / sqrtSHi;
  double P_avg = 0.5 * (P_o + P_i);
  double P_wavg = 0.5 * (P_o * invSHo + P_i * invSHi);
  double gbubu_avg = 0.5 * (gbubu_o + gbubu_i);
  double gbubu_wavg = 0.5 * (gbubu_o * sqrtSHo + gbubu_i * sqrtSHi);
  double gbvbv_avg = 0.5 * (gbvbv_o + gbvbv_i);
  double gbvbv_wavg = 0.5 * (gbvbv_o * sqrtSHo + gbvbv_i * sqrtSHi);

  // A_R
  double armn_e_v = (zup_o - zup_i) * invDS + 0.5 * (taup_o + taup_i)
                  - gbvbv_avg * r1e - gbvbv_wavg * r1o;
  double armn_o_v = (zup_o * sqrtSHo - zup_i * sqrtSHi) * invDS
                  - 0.5 * P_wavg * zue - 0.5 * P_avg * zuo
                  + 0.5 * (taup_o * sqrtSHo + taup_i * sqrtSHi)
                  - gbvbv_wavg * r1e - gbvbv_avg * r1o * sFull;

  // A_Z
  double azmn_e_v = -(rup_o - rup_i) * invDS;
  double azmn_o_v = -(rup_o * sqrtSHo - rup_i * sqrtSHi) * invDS
                  + 0.5 * P_wavg * rue + 0.5 * P_avg * ruo;

  // B_R
  double brmn_e_v = 0.5 * (zsp_o + zsp_i) + 0.5 * P_wavg * z1o
                  - gbubu_avg * rue - gbubu_wavg * ruo;
  double brmn_o_v = 0.5 * (zsp_o * sqrtSHo + zsp_i * sqrtSHi)
                  + 0.5 * P_avg * z1o
                  - gbubu_wavg * rue - gbubu_avg * ruo * sFull;

  // B_Z
  double bzmn_e_v = -0.5 * (rsp_o + rsp_i) - 0.5 * P_wavg * r1o
                  - gbubu_avg * zue - gbubu_wavg * zuo;
  double bzmn_o_v = -0.5 * (rsp_o * sqrtSHo + rsp_i * sqrtSHi)
                  - 0.5 * P_avg * r1o
                  - gbubu_wavg * zue - gbubu_avg * zuo * sFull;

  if (lthreed) {
    double gbubv_avg = 0.5 * (gbubv_o + gbubv_i);
    double gbubv_wavg = 0.5 * (gbubv_o * sqrtSHo + gbubv_i * sqrtSHi);
    double rve = rv_e[g_idx], rvo = rv_o[g_idx];
    double zve = zv_e[g_idx], zvo = zv_o[g_idx];

    brmn_e_v -= gbubv_avg * rve + gbubv_wavg * rvo;
    brmn_o_v -= gbubv_wavg * rve + gbubv_avg * rvo * sFull;
    bzmn_e_v -= gbubv_avg * zve + gbubv_wavg * zvo;
    bzmn_o_v -= gbubv_wavg * zve + gbubv_avg * zvo * sFull;

    double crmn_e_v = gbubv_avg * rue + gbubv_wavg * ruo
                    + gbvbv_avg * rve + gbvbv_wavg * rvo;
    double crmn_o_v = gbubv_wavg * rue + gbubv_avg * ruo * sFull
                    + gbvbv_wavg * rve + gbvbv_avg * rvo * sFull;

    double czmn_e_v = gbubv_avg * zue + gbubv_wavg * zuo
                    + gbvbv_avg * zve + gbvbv_wavg * zvo;
    double czmn_o_v = gbubv_wavg * zue + gbubv_avg * zuo * sFull
                    + gbvbv_wavg * zve + gbvbv_avg * zvo * sFull;

    crmn_e[f_idx] = crmn_e_v; crmn_o[f_idx] = crmn_o_v;
    czmn_e[f_idx] = czmn_e_v; czmn_o[f_idx] = czmn_o_v;
  }

  armn_e[f_idx] = armn_e_v; armn_o[f_idx] = armn_o_v;
  azmn_e[f_idx] = azmn_e_v; azmn_o[f_idx] = azmn_o_v;
  brmn_e[f_idx] = brmn_e_v; brmn_o[f_idx] = brmn_o_v;
  bzmn_e[f_idx] = bzmn_e_v; bzmn_o[f_idx] = bzmn_o_v;
}

// ComputeMHDForces stays separate from assembleTotalForces, whose brcon depends
// on DealiasInv, which runs between them.

// k_force_norm_partials: one thread per surface sums over kl in the CPU's
// order, giving partial_RZ and partial_L bit for bit.
__global__ void k_force_norm_partials(int n_config, int ns_h, int nZnT, int nThetaEff,
                                        int nsMinH, int nsMaxH_minus_1,
                                        int ns_minus_2,
                                        const double* __restrict__ guu,
                                        const double* __restrict__ r12,
                                        const double* __restrict__ bsubu,
                                        const double* __restrict__ bsubv,
                                        const double* __restrict__ wInt,
                                        double* __restrict__ partial_RZ,
                                        double* __restrict__ partial_L,
                                        const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jH = blockIdx.x;
  if (jH >= ns_h) return;
  if (threadIdx.x != 0) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  int jH_global = jH + nsMinH;
  bool unique = (jH_global < nsMaxH_minus_1) || (jH_global == ns_minus_2);
  double acc_rz = 0.0, acc_l = 0.0;
  if (unique) {
    for (int kl = 0; kl < nZnT; ++kl) {
      size_t i = cfg_half + (size_t)jH * (size_t)nZnT + (size_t)kl;
      int l = kl % nThetaEff;
      double w = wInt[l];
      double r12v = r12[i];
      acc_rz += guu[i] * r12v * r12v * w;
      double bu = bsubu[i], bv = bsubv[i];
      acc_l += (bu * bu + bv * bv) * w;
    }
  }
  partial_RZ[cfg_prof + jH] = acc_rz;
  partial_L[cfg_prof + jH] = acc_l;
}

// k_hybrid_lambda_force per (jF_local_con, kl) from the inside (jF - 1) and
// outside (jF) half-grid fields, without either beyond the range.
__global__ void k_hybrid_lambda_force(
    int n_config, int ns_local, int ns_h, int ns_con_local,
    int nZnT, bool lthreed,
    int nsMinF, int nsMinF1_off,  // nsMinF - nsMinF1
    int nsMinH_off,                // nsMinF - nsMinH (negative if nsMinF < nsMinH)
    int nsMaxH_minus_nsMinH,       // ns_h
    double lamscale,
    const double* __restrict__ bsubu, const double* __restrict__ bsubv,
    const double* __restrict__ gvv, const double* __restrict__ gsqrt,
    const double* __restrict__ guv, const double* __restrict__ bsupu,
    const double* __restrict__ lu_e, const double* __restrict__ lu_o,
    const double* __restrict__ sqrtSF, const double* __restrict__ sqrtSH,
    const double* __restrict__ radialBlending,
    double* __restrict__ blmn_e, double* __restrict__ blmn_o,
    double* __restrict__ clmn_e, double* __restrict__ clmn_o) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jF_local = blockIdx.y;
  if (jF_local >= ns_con_local) return;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;

  size_t cfg_half  = (size_t)config * (size_t)ns_h        * (size_t)nZnT;
  size_t cfg_full  = (size_t)config * (size_t)ns_local    * (size_t)nZnT;
  size_t cfg_con   = (size_t)config * (size_t)ns_con_local * (size_t)nZnT;

  int jF_global = jF_local + nsMinF;
  int jH_in = jF_global - 1;             // inside half-grid (global)
  int jH_out = jF_global;                // outside half-grid (global)
  int jH_in_local = jH_in - (nsMinF - nsMinH_off);   // half-grid local index inside
  int jH_out_local = jH_out - (nsMinF - nsMinH_off); // half-grid local index outside

  // Inside half-grid (j-1) values; default to 0 if jF_global == 0.
  double bsubu_i = 0.0, bsubv_i = 0.0;
  double gvv_gsqrt_i = 0.0, guv_bsupu_i = 0.0;
  double sqrtSHi = 0.0;
  if (jF_global > 0 && jH_in_local >= 0 && jH_in_local < nsMaxH_minus_nsMinH) {
    size_t i_in = cfg_half + (size_t)jH_in_local * (size_t)nZnT + (size_t)kl;
    bsubu_i = bsubu[i_in];
    bsubv_i = bsubv[i_in];
    double inv_g = 1.0 / gsqrt[i_in];
    gvv_gsqrt_i = gvv[i_in] * inv_g;
    if (lthreed) {
      guv_bsupu_i = guv[i_in] * bsupu[i_in];
    }
    sqrtSHi = sqrtSH[jH_in_local];
  }

  // Outside half-grid (j) values; default to 0 if jF_global >= nsMaxH.
  double bsubu_o = 0.0, bsubv_o = 0.0;
  double gvv_gsqrt_o = 0.0, guv_bsupu_o = 0.0;
  double sqrtSHo = 0.0;
  if (jH_out_local >= 0 && jH_out_local < nsMaxH_minus_nsMinH) {
    size_t i_out = cfg_half + (size_t)jH_out_local * (size_t)nZnT + (size_t)kl;
    bsubu_o = bsubu[i_out];
    bsubv_o = bsubv[i_out];
    double inv_g = 1.0 / gsqrt[i_out];
    gvv_gsqrt_o = gvv[i_out] * inv_g;
    if (lthreed) {
      guv_bsupu_o = guv[i_out] * bsupu[i_out];
    }
    sqrtSHo = sqrtSH[jH_out_local];
  }

  // Full-grid lu_e/o at jF (indexed by jF - nsMinF1).
  int jF_local_full = jF_global - nsMinF1_off;
  size_t i_full = cfg_full + (size_t)jF_local_full * (size_t)nZnT + (size_t)kl;
  double lue = lu_e[i_full];
  double luo = lu_o[i_full];

  double gvv_gsqrt_lu_e = 0.5 * (gvv_gsqrt_i + gvv_gsqrt_o) * lue;
  double gvv_gsqrt_lu_o = 0.5 * (gvv_gsqrt_i * sqrtSHi + gvv_gsqrt_o * sqrtSHo) * luo;
  double gvv_gsqrt_lu = gvv_gsqrt_lu_e + gvv_gsqrt_lu_o;
  double bsubv_alternative = gvv_gsqrt_lu;
  if (lthreed) {
    double guv_bsupu_avg = 0.5 * (guv_bsupu_i + guv_bsupu_o);
    bsubv_alternative += guv_bsupu_avg;
  }
  double bsubv_average = 0.5 * (bsubv_o + bsubv_i);
  double rb = radialBlending[jF_local_full];
  double _blmn = bsubv_average * (1.0 - rb) + bsubv_alternative * rb;
  if (jF_global > 0) {
    _blmn *= -lamscale;
  }
  double sqrtSF_jF = sqrtSF[jF_local_full];
  size_t out_idx = cfg_con + (size_t)jF_local * (size_t)nZnT + (size_t)kl;
  blmn_e[out_idx] = _blmn;
  blmn_o[out_idx] = _blmn * sqrtSF_jF;

  if (lthreed) {
    double _clmn = 0.5 * (bsubu_o + bsubu_i);
    if (jF_global > 0) {
      _clmn *= -lamscale;
    }
    clmn_e[out_idx] = _clmn;
    clmn_o[out_idx] = _clmn * sqrtSF_jF;
  }
}

// k_pres_compute_and_thermal: presH and the thermal partial in one launch.
__global__ void k_pres_compute_and_thermal(int n_config, int ns_h, double gamma,
                                            int nsMinH, int nsMaxH_minus_1,
                                            int ns_minus_2,
                                            const double* __restrict__ massH,
                                            const double* __restrict__ dVdsH,
                                            double* __restrict__ presH,
                                            double* __restrict__ thermal_partial) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  int jH = blockIdx.x * blockDim.x + threadIdx.x;
  if (jH >= ns_h) return;
  size_t cfg = (size_t)config * (size_t)ns_h;
  double dV = dVdsH[cfg + jH];
  double pres = massH[cfg + jH] / pow(dV, gamma);
  presH[cfg + jH] = pres;
  int jH_global = jH + nsMinH;
  bool unique = (jH_global < nsMaxH_minus_1) || (jH_global == ns_minus_2);
  thermal_partial[cfg + jH] = unique ? (pres * dV) : 0.0;
}

// k_pres_magnetic_partial_inline: the magnetic partial with totalPressure
// computed inline from bsupu * bsubu + bsupv * bsubv.
__global__ void k_pres_magnetic_partial_inline(int n_config, int ns_h, int nZnT,
                                                int nThetaEff, int nsMinH,
                                                int nsMaxH_minus_1, int ns_minus_2,
                                                const double* __restrict__ gsqrt,
                                                const double* __restrict__ bsupu,
                                                const double* __restrict__ bsubu,
                                                const double* __restrict__ bsupv,
                                                const double* __restrict__ bsubv,
                                                const double* __restrict__ wInt,
                                                double* __restrict__ magnetic_partial,
                                                const std::uint8_t* __restrict__ d_active_per_cfg) {
  // Serial single-thread kl accumulation to match CPU's reduction order.
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jH = blockIdx.x;
  if (jH >= ns_h) return;
  if (threadIdx.x != 0) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  int jH_global = jH + nsMinH;
  bool unique = (jH_global < nsMaxH_minus_1) || (jH_global == ns_minus_2);
  double acc = 0.0;
  if (unique) {
    for (int kl = 0; kl < nZnT; ++kl) {
      int l = kl % nThetaEff;
      size_t i = cfg_half + (size_t)jH * (size_t)nZnT + (size_t)kl;
      double mag_pressure = 0.5 * (bsupu[i] * bsubu[i] + bsupv[i] * bsubv[i]);
      acc += gsqrt[i] * mag_pressure * wInt[l];
    }
  }
  magnetic_partial[cfg_prof + jH] = acc;
}

// k_pres_totalpres_init_with_presH: the final totalPressure, magnetic plus
// thermal, in one pass.
__global__ void k_pres_totalpres_init_with_presH(int n_config, int ns_h, int nZnT,
                                                   const double* __restrict__ bsupu,
                                                   const double* __restrict__ bsubu,
                                                   const double* __restrict__ bsupv,
                                                   const double* __restrict__ bsubv,
                                                   const double* __restrict__ presH,
                                                   double* __restrict__ totalPressure,
                                                   const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jH = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (jH >= ns_h || kl >= nZnT) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  size_t i = cfg_half + (size_t)jH * (size_t)nZnT + (size_t)kl;
  totalPressure[i] = 0.5 * (bsupu[i] * bsubu[i] + bsupv[i] * bsubv[i])
                     + presH[cfg_prof + jH];
}

// k_volume_reduce: one scalar per configuration, the sum of dVdsH over the
// unique half-grid surfaces times multiplier.
__global__ void k_volume_reduce(int n_config, int ns_h, double multiplier,
                                  int nsMaxH_minus_1, int ns_minus_2,
                                  int nsMinH,
                                  const double* __restrict__ dVdsH,
                                  double* __restrict__ out_scalar,
                                  const std::uint8_t* __restrict__ d_active_per_cfg) {
  // Serial single-thread jH accumulation to match CPU's reduction order.
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  if (threadIdx.x != 0) return;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  double acc = 0.0;
  for (int jH = 0; jH < ns_h; ++jH) {
    int jH_global = jH + nsMinH;
    bool unique = (jH_global < nsMaxH_minus_1) || (jH_global == ns_minus_2);
    if (unique) acc += dVdsH[cfg_prof + jH];
  }
  out_scalar[config] = acc * multiplier;
}

// k_bcontra_mutate_lambda: lu and lv scaled by lamscale, with phipF added to
// lu_e, indexed jF - nsMinH as on the CPU.
__global__ void k_bcontra_mutate_lambda(
    int n_config, int ns_local,
    int jF_first, int jF_last_excl, int nZnT, int phipF_jOff,
    bool lthreed, double lamscale,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o,
    const double* __restrict__ phipF) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jF_local = blockIdx.y + jF_first;
  if (jF_local >= jF_last_excl) return;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  size_t cfg_full = (size_t)config * (size_t)ns_local * (size_t)nZnT;
  size_t cfg_prof = (size_t)config * (size_t)ns_local;
  size_t idx = cfg_full + (size_t)jF_local * (size_t)nZnT + (size_t)kl;
  double lue = lu_e[idx] * lamscale;
  double luo = lu_o[idx] * lamscale;
  lue += phipF[cfg_prof + (size_t)(jF_local - phipF_jOff)];
  lu_e[idx] = lue;
  lu_o[idx] = luo;
  if (lthreed) {
    lv_e[idx] *= lamscale;
    lv_o[idx] *= lamscale;
  }
}

// k_bcontra_bsupuv per (jH_local, kl): bsupu from the averaged lambda
// derivatives over gsqrt (zero in 2D) and bsupv; chip is added later.
__global__ void k_bcontra_bsupuv(
    int n_config, int ns_local, int ns_h,
    int jF_in_offset, int nZnT, bool lthreed,
    const double* __restrict__ lu_e, const double* __restrict__ lu_o,
    const double* __restrict__ lv_e, const double* __restrict__ lv_o,
    const double* __restrict__ sqrtSH, const double* __restrict__ gsqrt,
    double* __restrict__ bsupu, double* __restrict__ bsupv) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH_local = blockIdx.y;
  if (jH_local >= ns_h) return;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;

  size_t cfg_full = (size_t)config * (size_t)ns_local * (size_t)nZnT;
  size_t cfg_half = (size_t)config * (size_t)ns_h     * (size_t)nZnT;
  int jF_in = jH_local + jF_in_offset;
  int jF_out = jF_in + 1;
  size_t iHalf = cfg_half + (size_t)jH_local * (size_t)nZnT + (size_t)kl;

  double sH = sqrtSH[jH_local];
  double inv_g = 1.0 / gsqrt[iHalf];

  // Explicit linear indices, which nvcc does not hoist through the chained
  // subscripts.
  size_t i_in  = cfg_full + (size_t)jF_in  * (size_t)nZnT + (size_t)kl;
  size_t i_out = cfg_full + (size_t)jF_out * (size_t)nZnT + (size_t)kl;

  double lue_i = lu_e[i_in];
  double luo_i = lu_o[i_in];
  double lue_o = lu_e[i_out];
  double luo_o = lu_o[i_out];

  double bsupu_v = 0.0;
  if (lthreed) {
    double lve_i = lv_e[i_in];
    double lvo_i = lv_o[i_in];
    double lve_o = lv_e[i_out];
    double lvo_o = lv_o[i_out];
    bsupu_v = 0.5 * ((lve_i + lve_o) + sH * (lvo_i + lvo_o)) * inv_g;
  }
  double bsupv_v = 0.5 * ((lue_i + lue_o) + sH * (luo_i + luo_o)) * inv_g;
  bsupu[iHalf] = bsupu_v;
  bsupv[iHalf] = bsupv_v;
}

// k_bcontra_jvplasma_reduce (ncurr = 1): per surface jvPlasma = sum (guu bsupu
// + guv bsupv) wInt and avg_guu_gsqrt = sum guu / gsqrt wInt.
__global__ void k_bcontra_jvplasma_reduce(
    int n_config, int ns_h, int nZnT, int nThetaEff, bool lthreed,
    const double* __restrict__ guu, const double* __restrict__ guv,
    const double* __restrict__ bsupu, const double* __restrict__ bsupv,
    const double* __restrict__ gsqrt, const double* __restrict__ wInt,
    double* __restrict__ jvPlasma, double* __restrict__ avg_guu_gsqrt) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH_local = blockIdx.x;
  if (jH_local >= ns_h) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  __shared__ double s_jv[32];
  __shared__ double s_avg[32];
  double acc_jv = 0.0, acc_avg = 0.0;
  for (int kl = threadIdx.x; kl < nZnT; kl += blockDim.x) {
    size_t i = cfg_half + (size_t)jH_local * (size_t)nZnT + (size_t)kl;
    int l = kl % nThetaEff;
    double w = wInt[l];
    double g = gsqrt[i];
    double gu = guu[i];
    double bsu = bsupu[i];
    double term = gu * bsu;
    if (lthreed) {
      term += guv[i] * bsupv[i];
    }
    acc_jv += term * w;
    acc_avg += (gu / g) * w;
  }
  s_jv[threadIdx.x] = acc_jv;
  s_avg[threadIdx.x] = acc_avg;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_jv[threadIdx.x] += s_jv[threadIdx.x + stride];
      s_avg[threadIdx.x] += s_avg[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    jvPlasma[cfg_prof + jH_local] = s_jv[0];
    avg_guu_gsqrt[cfg_prof + jH_local] = s_avg[0];
  }
}

// k_bcontra_chipH_iotaH: chipH = (currH - jvPlasma) / avg_guu_gsqrt and iotaH =
// chipH / phipH for ncurr = 1, else chipH = iotaH * phipH.
__global__ void k_bcontra_chipH_iotaH(
    int n_config, int ns_h, int ncurr,
    const double* __restrict__ phipH, const double* __restrict__ currH,
    const double* __restrict__ iotaH_in, const double* __restrict__ jvPlasma,
    const double* __restrict__ avg_guu_gsqrt,
    double* __restrict__ chipH, double* __restrict__ iotaH) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  int jH_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (jH_local >= ns_h) return;
  size_t cfg = (size_t)config * (size_t)ns_h;
  double pH = phipH[cfg + jH_local];
  if (ncurr == 1) {
    double denom = avg_guu_gsqrt[cfg + jH_local];
    double newChip = chipH[cfg + jH_local];  // keep previous if denom == 0
    if (denom != 0.0) {
      newChip = (currH[cfg + jH_local] - jvPlasma[cfg + jH_local]) / denom;
    }
    chipH[cfg + jH_local] = newChip;
    double iv = iotaH[cfg + jH_local];
    if (pH != 0.0) {
      iv = newChip / pH;
    }
    iotaH[cfg + jH_local] = iv;
  } else {
    // ncurr==0: chipH = iotaH * phipH (using input iotaH).
    double iv = iotaH_in[cfg + jH_local];
    chipH[cfg + jH_local] = iv * pH;
    iotaH[cfg + jH_local] = iv;  // pass through unchanged
  }
}

// k_bcontra_chipF_iotaF: midpoint averages inside, linear extrapolation 1.5
// h[0] - 0.5 h[1] at the axis and LCFS, as computeBContra does.
__global__ void k_bcontra_chipF_iotaF(
    int n_config, int ns_h, int ns_local, int nsMinFi_off, int nsMaxFi_off,
    int axis_present, int lcfs_present, int last_jF_local,
    int last_jH_local,
    const double* __restrict__ chipH, const double* __restrict__ iotaH,
    double* __restrict__ chipF, double* __restrict__ iotaF) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  int jF_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_local >= ns_local) return;

  size_t cfg_h = (size_t)config * (size_t)ns_h;
  size_t cfg_f = (size_t)config * (size_t)ns_local;
  // Interior interpolation. nsMinFi/nsMaxFi are passed as local offsets
  // (nsMinFi - nsMinF1 .. nsMaxFi - nsMinF1).
  if (jF_local >= nsMinFi_off && jF_local < nsMaxFi_off) {
    // jH indices: (jFi - nsMinH) and (jFi - 1 - nsMinH). In single-rank
    // nsMinF1 == nsMinH == 0 so jH = jF_local and jF_local-1.
    int jH_o = jF_local;       // outside half-grid
    int jH_i = jF_local - 1;   // inside half-grid
    chipF[cfg_f + jF_local] = 0.5 * (chipH[cfg_h + jH_o] + chipH[cfg_h + jH_i]);
    iotaF[cfg_f + jF_local] = 0.5 * (iotaH[cfg_h + jH_o] + iotaH[cfg_h + jH_i]);
  }
  if (axis_present && jF_local == 0) {
    chipF[cfg_f + 0] = 1.5 * chipH[cfg_h + 0] - 0.5 * chipH[cfg_h + 1];
    iotaF[cfg_f + 0] = 1.5 * iotaH[cfg_h + 0] - 0.5 * iotaH[cfg_h + 1];
  }
  if (lcfs_present && jF_local == last_jF_local) {
    chipF[cfg_f + jF_local] = 1.5 * chipH[cfg_h + last_jH_local] - 0.5 * chipH[cfg_h + last_jH_local - 1];
    iotaF[cfg_f + jF_local] = 1.5 * iotaH[cfg_h + last_jH_local] - 0.5 * iotaH[cfg_h + last_jH_local - 1];
  }
}

// k_bcontra_bsupu_add_chip: bsupu += chipH / gsqrt per (jH, kl).
__global__ void k_bcontra_bsupu_add_chip(
    int n_config, int ns_h, int nZnT,
    const double* __restrict__ chipH, const double* __restrict__ gsqrt,
    double* __restrict__ bsupu) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH_local = blockIdx.y;
  if (jH_local >= ns_h) return;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  size_t iHalf = cfg_half + (size_t)jH_local * (size_t)nZnT + (size_t)kl;
  bsupu[iHalf] += chipH[cfg_prof + jH_local] / gsqrt[iHalf];
}

// k_rzcon_into_volume: rCon and zCon at the LCFS times sFull = sqrtSF^2 on
// [jMin_con, ns_con_local).
__global__ void k_rzcon_into_volume(
    int n_config, int ns_con_local, int nZnT, int jMin_con, int lcfs_con_local,
    int nsMinF_minus_nsMinF1,
    const double* __restrict__ rCon, const double* __restrict__ zCon,
    const double* __restrict__ sqrtSF,
    double* __restrict__ rCon0, double* __restrict__ zCon0) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jF_con_local = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  if (jF_con_local >= ns_con_local) return;
  if (jF_con_local < jMin_con) return;  // axis skipped (matches CPU max(1,nsMinF))
  size_t cfg_con = (size_t)config * (size_t)ns_con_local * (size_t)nZnT;
  int sqrtSF_idx = jF_con_local + nsMinF_minus_nsMinF1;
  double s = sqrtSF[sqrtSF_idx];
  double sFull = s * s;
  size_t idx = cfg_con + (size_t)jF_con_local * (size_t)nZnT + (size_t)kl;
  size_t lcfs_idx = cfg_con + (size_t)lcfs_con_local * (size_t)nZnT + (size_t)kl;
  rCon0[idx] = rCon[lcfs_idx] * sFull;
  zCon0[idx] = zCon[lcfs_idx] * sFull;
}

// Inverse FFT kernels: k_inverse_fill, cuFFT R2C and k_inverse_scatter, the
// forward pipeline in reverse.

// k_inverse_fill: real-space forces into Y[jF, m, q, k] of length nZeta, z =
// config * ns_local + jF_local.
__global__ void k_inverse_fill(
    int n_config, int ns_local, int mpol, int nZeta,
    int nThetaReduced, int nThetaEff,
    bool lthreed, int nsMinF_to_nsMinF1,
    int ns_force_local, int ns_con_local,
    const double* __restrict__ xmpq,
    const double* __restrict__ cosmui, const double* __restrict__ sinmui,
    const double* __restrict__ cosmumi, const double* __restrict__ sinmumi,
    const double* __restrict__ armn_e, const double* __restrict__ armn_o,
    const double* __restrict__ azmn_e, const double* __restrict__ azmn_o,
    const double* __restrict__ brmn_e, const double* __restrict__ brmn_o,
    const double* __restrict__ bzmn_e, const double* __restrict__ bzmn_o,
    const double* __restrict__ blmn_e, const double* __restrict__ blmn_o,
    const double* __restrict__ crmn_e, const double* __restrict__ crmn_o,
    const double* __restrict__ czmn_e, const double* __restrict__ czmn_o,
    const double* __restrict__ clmn_e, const double* __restrict__ clmn_o,
    const double* __restrict__ frcon_e, const double* __restrict__ frcon_o,
    const double* __restrict__ fzcon_e, const double* __restrict__ fzcon_o,
    double* __restrict__ Y) {
  int config = blockIdx.z / ns_local;
  int jF_local = blockIdx.z - config * ns_local;
  if (config >= n_config) return;
  int mq = blockIdx.y;
  int m = mq / kBatch;
  int q = mq % kBatch;
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nZeta || m >= mpol || jF_local >= ns_local) return;

  size_t cfg_force = (size_t)config * (size_t)ns_force_local *
                     (size_t)nZeta * (size_t)nThetaEff;
  size_t cfg_con   = (size_t)config * (size_t)ns_con_local *
                     (size_t)nZeta * (size_t)nThetaEff;
  size_t cfg_Y     = (size_t)config * (size_t)ns_local * (size_t)mpol *
                     (size_t)kBatch * (size_t)nZeta;

  int jF_force_local = jF_local - nsMinF_to_nsMinF1;
  bool force_valid = (jF_force_local >= 0 && jF_force_local < ns_force_local);
  bool con_valid   = (jF_force_local >= 0 && jF_force_local < ns_con_local);
  bool m_even = ((m & 1) == 0);
  double xmpq_m = xmpq[m];

  double acc = 0.0;
  for (int l = 0; l < nThetaReduced; ++l) {
    int basis_ml = m * nThetaReduced + l;
    double cmui  = cosmui[basis_ml];
    double smui  = sinmui[basis_ml];
    double cmumi = cosmumi[basis_ml];
    double smumi = sinmumi[basis_ml];

    size_t force_kl = force_valid ? (cfg_force + (size_t)(jF_force_local * nZeta * nThetaEff + k * nThetaEff + l)) : 0;
    size_t con_kl   = con_valid   ? (cfg_con   + (size_t)(jF_force_local * nZeta * nThetaEff + k * nThetaEff + l)) : 0;

    switch (q) {
      case kRmkcc: {
        if (force_valid) {
          double armn = m_even ? armn_e[force_kl] : armn_o[force_kl];
          double brmn = m_even ? brmn_e[force_kl] : brmn_o[force_kl];
          double frcon = m_even ? frcon_e[force_kl] : frcon_o[force_kl];
          acc += (armn + xmpq_m * frcon) * cmui + brmn * smumi;
        }
        break;
      }
      case kRmkss: {
        if (!lthreed) break;
        if (force_valid) {
          double armn = m_even ? armn_e[force_kl] : armn_o[force_kl];
          double brmn = m_even ? brmn_e[force_kl] : brmn_o[force_kl];
          double frcon = m_even ? frcon_e[force_kl] : frcon_o[force_kl];
          acc += (armn + xmpq_m * frcon) * smui + brmn * cmumi;
        }
        break;
      }
      case kRmkccN: {
        if (!lthreed) break;
        if (force_valid) {
          double crmn = m_even ? crmn_e[force_kl] : crmn_o[force_kl];
          acc -= crmn * cmui;  // CPU: -crmn_seg.dot(cosmui_seg)
        }
        break;
      }
      case kRmkssN: {
        if (!lthreed) break;
        if (force_valid) {
          double crmn = m_even ? crmn_e[force_kl] : crmn_o[force_kl];
          acc -= crmn * smui;  // CPU: -crmn_seg.dot(sinmui_seg)
        }
        break;
      }
      case kZmksc: {
        if (force_valid) {
          double azmn = m_even ? azmn_e[force_kl] : azmn_o[force_kl];
          double bzmn = m_even ? bzmn_e[force_kl] : bzmn_o[force_kl];
          double fzcon = m_even ? fzcon_e[force_kl] : fzcon_o[force_kl];
          acc += (azmn + xmpq_m * fzcon) * smui + bzmn * cmumi;
        }
        break;
      }
      case kZmkcs: {
        if (!lthreed) break;
        if (force_valid) {
          double azmn = m_even ? azmn_e[force_kl] : azmn_o[force_kl];
          double bzmn = m_even ? bzmn_e[force_kl] : bzmn_o[force_kl];
          double fzcon = m_even ? fzcon_e[force_kl] : fzcon_o[force_kl];
          acc += (azmn + xmpq_m * fzcon) * cmui + bzmn * smumi;
        }
        break;
      }
      case kZmkscN: {
        if (!lthreed) break;
        if (force_valid) {
          double czmn = m_even ? czmn_e[force_kl] : czmn_o[force_kl];
          acc -= czmn * smui;  // CPU: -czmn_seg.dot(sinmui_seg)
        }
        break;
      }
      case kZmkcsN: {
        if (!lthreed) break;
        if (force_valid) {
          double czmn = m_even ? czmn_e[force_kl] : czmn_o[force_kl];
          acc -= czmn * cmui;  // CPU: -czmn_seg.dot(cosmui_seg)
        }
        break;
      }
      case kLmksc: {
        if (con_valid) {
          double blmn = m_even ? blmn_e[con_kl] : blmn_o[con_kl];
          acc += blmn * cmumi;
        }
        break;
      }
      case kLmkcs: {
        if (!lthreed) break;
        if (con_valid) {
          double blmn = m_even ? blmn_e[con_kl] : blmn_o[con_kl];
          acc += blmn * smumi;
        }
        break;
      }
      case kLmkscN: {
        if (!lthreed) break;
        if (con_valid) {
          double clmn = m_even ? clmn_e[con_kl] : clmn_o[con_kl];
          acc -= clmn * smui;
        }
        break;
      }
      case kLmkcsN: {
        if (!lthreed) break;
        if (con_valid) {
          double clmn = m_even ? clmn_e[con_kl] : clmn_o[con_kl];
          acc -= clmn * cmui;
        }
        break;
      }
    }
  }
  size_t y_idx = cfg_Y + (size_t)(((jF_local * mpol + m) * kBatch + q) * nZeta + k);
  Y[y_idx] = acc;
}

// k_inverse_scatter: R2C output into the spectra over the CPU's RZ and lambda
// ranges, zero elsewhere as FourierForces.setZero().
__global__ void k_inverse_scatter(
    int n_config, int ns_local, int mpol, int ntor, int nhalf, int nfp, int nZeta,
    bool lthreed, int nsMinF1_offset,
    int jMaxRZ_local, int jMinL_local,
    const cufftDoubleComplex* __restrict__ X,
    const double* __restrict__ nscale,
    double* __restrict__ frcc, double* __restrict__ frss,
    double* __restrict__ fzsc, double* __restrict__ fzcs,
    double* __restrict__ flsc, double* __restrict__ flcs) {
  int config = blockIdx.z / ns_local;
  int jF_local = blockIdx.z - config * ns_local;
  if (config >= n_config) return;
  int m = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n > ntor || m >= mpol || jF_local >= ns_local) return;

  size_t cfg_X    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nhalf;
  size_t cfg_spec = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)(ntor + 1);
  size_t spec_idx = cfg_spec + (size_t)((jF_local * mpol + m) * (ntor + 1) + n);
  // At axis (jF_global=0) CPU sets mmax=1, so RZ writes only m=0.
  bool at_axis = (jF_local + nsMinF1_offset) == 0;
  bool write_rz = (jF_local < jMaxRZ_local) && (!at_axis || m == 0);
  bool write_lambda = (jF_local >= jMinL_local) && (!at_axis || m == 0);

  if (!write_rz) {
    frcc[spec_idx] = 0.0;
    fzsc[spec_idx] = 0.0;
    if (lthreed) { frss[spec_idx] = 0.0; fzcs[spec_idx] = 0.0; }
  }
  if (!write_lambda) {
    flsc[spec_idx] = 0.0;
    if (lthreed) flcs[spec_idx] = 0.0;
  }
  if (!write_rz && !write_lambda) return;

  const double ns_n = nscale[n];
  const double nfp_n = (double)(n * nfp);
  size_t x_base = cfg_X + (size_t)((jF_local * mpol + m) * kBatch * nhalf + n);

  // The CPU's FFTX accumulation with cuFFT's nscale[n], e.g. frcc = ns_n *
  // (X_rcc.re + nfp_n * X_rccN.im).
  if (write_rz) {
    cufftDoubleComplex x_rcc  = X[x_base + (size_t)kRmkcc  * nhalf];
    cufftDoubleComplex x_zsc  = X[x_base + (size_t)kZmksc  * nhalf];
    cufftDoubleComplex x_rccN = X[x_base + (size_t)kRmkccN * nhalf];
    cufftDoubleComplex x_zscN = X[x_base + (size_t)kZmkscN * nhalf];
    frcc[spec_idx] = ns_n * (x_rcc.x + nfp_n * x_rccN.y);
    fzsc[spec_idx] = ns_n * (x_zsc.x + nfp_n * x_zscN.y);
    if (lthreed) {
      cufftDoubleComplex x_rss  = X[x_base + (size_t)kRmkss  * nhalf];
      cufftDoubleComplex x_zcs  = X[x_base + (size_t)kZmkcs  * nhalf];
      cufftDoubleComplex x_rssN = X[x_base + (size_t)kRmkssN * nhalf];
      cufftDoubleComplex x_zcsN = X[x_base + (size_t)kZmkcsN * nhalf];
      frss[spec_idx] = ns_n * (-x_rss.y + nfp_n * x_rssN.x);
      fzcs[spec_idx] = ns_n * (-x_zcs.y + nfp_n * x_zcsN.x);
    }
  }
  if (write_lambda) {
    cufftDoubleComplex x_lsc  = X[x_base + (size_t)kLmksc  * nhalf];
    cufftDoubleComplex x_lscN = X[x_base + (size_t)kLmkscN * nhalf];
    flsc[spec_idx] = ns_n * (x_lsc.x + nfp_n * x_lscN.y);
    if (lthreed) {
      cufftDoubleComplex x_lcs  = X[x_base + (size_t)kLmkcs  * nhalf];
      cufftDoubleComplex x_lcsN = X[x_base + (size_t)kLmkcsN * nhalf];
      flcs[spec_idx] = ns_n * (-x_lcs.y + nfp_n * x_lcsN.x);
    }
  }
  (void)nZeta; (void)nsMinF1_offset;
}

// k_compute_ru_zu_full: ruFull = ru_e + sqrtSF * ru_o and zuFull alike per
// (jF_con, kl) on [nsMinF, nsMaxFIncludingLcfs).
__global__ void k_compute_ru_zu_full(int n_config, int ns_local,
                                      int ns_con_local, int nZnT,
                                      int nsMinF_to_nsMinF1,
                                      const double* __restrict__ ru_e,
                                      const double* __restrict__ ru_o,
                                      const double* __restrict__ zu_e,
                                      const double* __restrict__ zu_o,
                                      const double* __restrict__ sqrtSF,
                                      double* __restrict__ ruFull,
                                      double* __restrict__ zuFull) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jF_con = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  if (jF_con >= ns_con_local) return;
  size_t cfg_full = (size_t)config * (size_t)ns_local     * (size_t)nZnT;
  size_t cfg_con  = (size_t)config * (size_t)ns_con_local * (size_t)nZnT;
  int jF_full = jF_con + nsMinF_to_nsMinF1;
  size_t src = cfg_full + (size_t)jF_full * (size_t)nZnT + (size_t)kl;
  size_t dst = cfg_con  + (size_t)jF_con  * (size_t)nZnT + (size_t)kl;
  double s = sqrtSF[jF_full];
  ruFull[dst] = ru_e[src] + s * ru_o[src];
  zuFull[dst] = zu_e[src] + s * zu_o[src];
}

// k_constraint_force_multiplier: per surface reduces ruFull^2 wInt and zuFull^2
// wInt over kl into tcon.
__global__ void k_constraint_force_multiplier(int n_config,
                                                int ns_con_local,
                                                int ns_force_local, int nZnT,
                                                int nThetaEff, int jMin,
                                                int kEven, double tcon_factor,
                                                const double* __restrict__ ruFull,
                                                const double* __restrict__ zuFull,
                                                const double* __restrict__ ard,
                                                const double* __restrict__ azd,
                                                const double* __restrict__ wInt,
                                                double* __restrict__ tcon,
                                                const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jF = blockIdx.x;
  if (jF >= ns_force_local) return;
  size_t cfg_con  = (size_t)config * (size_t)ns_con_local   * (size_t)nZnT;
  // d_tcon is allocated as n_config_max * ns_con_local doubles (see
  // EnsureConstraintMultiplierBuffers). Per-config stride is ns_con_local.
  size_t cfg_prof = (size_t)config * (size_t)ns_con_local;
  if (jF < jMin) {
    if (threadIdx.x == 0) tcon[cfg_prof + jF] = 0.0;
    return;
  }
  __shared__ double s_ar[32], s_az[32];
  double acc_ar = 0.0, acc_az = 0.0;
  for (int kl = threadIdx.x; kl < nZnT; kl += blockDim.x) {
    size_t idx = cfg_con + (size_t)jF * (size_t)nZnT + (size_t)kl;
    int l = kl % nThetaEff;
    double w = wInt[l];
    double r = ruFull[idx];
    double z = zuFull[idx];
    acc_ar += r * r * w;
    acc_az += z * z * w;
  }
  s_ar[threadIdx.x] = acc_ar;
  s_az[threadIdx.x] = acc_az;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_ar[threadIdx.x] += s_ar[threadIdx.x + stride];
      s_az[threadIdx.x] += s_az[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double arN = s_ar[0], azN = s_az[0];
    // Per-configuration indexing extension: per-cfg pmat reads (was cfg=0's slot for all).
    size_t cfg_pmat = (size_t)config * (size_t)ns_force_local * 2;
    double ar_e = (arN != 0.0) ? fabs(ard[cfg_pmat + jF * 2 + kEven] / arN) : 0.0;
    double az_e = (azN != 0.0) ? fabs(azd[cfg_pmat + jF * 2 + kEven] / azN) : 0.0;
    double base = (ar_e < az_e) ? ar_e : az_e;
    tcon[cfg_prof + jF] = base * tcon_factor;
  }
}

// k_halve_tcon_lcfs: halves each configuration's last tcon on the rank owning
// the LCFS.
__global__ void k_halve_tcon_lcfs(int n_config, int tcon_stride,
                                    int last_idx,
                                    double* __restrict__ tcon,
                                    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  if (threadIdx.x != 0) return;
  size_t cfg = (size_t)config * (size_t)tcon_stride;
  tcon[cfg + last_idx] = 0.5 * tcon[cfg + last_idx - 1];
}

// k_effective_constraint_force: gConEff = (rCon - rCon0) * ruFull + (zCon -
// zCon0) * zuFull from jMin on.
__global__ void k_effective_constraint_force(int n_config, int ns_con_local,
                                              int nZnT, int jMin,
                                              const double* __restrict__ rCon,
                                              const double* __restrict__ rCon0,
                                              const double* __restrict__ zCon,
                                              const double* __restrict__ zCon0,
                                              const double* __restrict__ ruFull,
                                              const double* __restrict__ zuFull,
                                              double* __restrict__ gConEff) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jF = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  if (jF >= ns_con_local) return;
  size_t cfg = (size_t)config * (size_t)ns_con_local * (size_t)nZnT;
  size_t idx = cfg + (size_t)jF * (size_t)nZnT + (size_t)kl;
  if (jF < jMin) { gConEff[idx] = 0.0; return; }
  gConEff[idx] = (rCon[idx] - rCon0[idx]) * ruFull[idx] +
                 (zCon[idx] - zCon0[idx]) * zuFull[idx];
}

// k_assemble_total_forces: adds the constraint force to brmn and bzmn and
// writes frcon and fzcon; the free-boundary edge term is separate.
__global__ void k_assemble_total_forces(int n_config,
                                          int ns_con_local, int ns_force_local,
                                          int nZnT,
                                          int nsMinF_to_nsMinF1,
                                          const double* __restrict__ rCon,
                                          const double* __restrict__ rCon0,
                                          const double* __restrict__ zCon,
                                          const double* __restrict__ zCon0,
                                          const double* __restrict__ gCon,
                                          const double* __restrict__ ruFull,
                                          const double* __restrict__ zuFull,
                                          const double* __restrict__ sqrtSF,
                                          double* __restrict__ brmn_e,
                                          double* __restrict__ brmn_o,
                                          double* __restrict__ bzmn_e,
                                          double* __restrict__ bzmn_o,
                                          double* __restrict__ frcon_e,
                                          double* __restrict__ frcon_o,
                                          double* __restrict__ fzcon_e,
                                          double* __restrict__ fzcon_o,
                                          const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jF = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  if (jF >= ns_force_local) return;
  size_t cfg_con   = (size_t)config * (size_t)ns_con_local   * (size_t)nZnT;
  size_t cfg_force = (size_t)config * (size_t)ns_force_local * (size_t)nZnT;
  size_t idx_con   = cfg_con   + (size_t)jF * (size_t)nZnT + (size_t)kl;
  size_t idx_force = cfg_force + (size_t)jF * (size_t)nZnT + (size_t)kl;
  double rC = rCon[idx_con], rC0 = rCon0[idx_con];
  double zC = zCon[idx_con], zC0 = zCon0[idx_con];
  double gc = gCon[idx_con];
  double brcon = (rC - rC0) * gc;
  double bzcon = (zC - zC0) * gc;
  double sF = sqrtSF[jF + nsMinF_to_nsMinF1];
  brmn_e[idx_force] += brcon;
  bzmn_e[idx_force] += bzcon;
  brmn_o[idx_force] += brcon * sF;
  bzmn_o[idx_force] += bzcon * sF;
  double ru = ruFull[idx_con], zu = zuFull[idx_con];
  double frce = ru * gc;
  double fzce = zu * gc;
  frcon_e[idx_force] = frce;
  fzcon_e[idx_force] = fzce;
  frcon_o[idx_force] = frce * sF;
  fzcon_o[idx_force] = fzce * sF;
}

// k_compute_bco: bsubu = guu bsupu + guv bsupv and bsubv = guv bsupu + gvv
// bsupv in 3D, without the guv terms in 2D.
__global__ void k_compute_bco(int n_config, int ns_h, int nZnT, bool lthreed,
                               const double* __restrict__ guu,
                               const double* __restrict__ guv,
                               const double* __restrict__ gvv,
                               const double* __restrict__ bsupu,
                               const double* __restrict__ bsupv,
                               double* __restrict__ bsubu,
                               double* __restrict__ bsubv) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH_local = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  if (jH_local >= ns_h) return;
  size_t cfg = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t i = cfg + (size_t)jH_local * (size_t)nZnT + (size_t)kl;
  double bsupu_v = bsupu[i];
  double bsupv_v = bsupv[i];
  if (lthreed) {
    bsubu[i] = guu[i] * bsupu_v + guv[i] * bsupv_v;
    bsubv[i] = guv[i] * bsupu_v + gvv[i] * bsupv_v;
  } else {
    bsubu[i] = guu[i] * bsupu_v;
    bsubv[i] = gvv[i] * bsupv_v;
  }
}

// k_apply_m1_preconditioner: scales the m = 1 frss and fzcs by the force ratios
// from (ard + brd) / (ard + brd + azd + bzd).
__global__ void k_apply_m1_preconditioner(
    int n_config, int ns_local,
    int ns_force_local, int mpol, int ntor,
    const double* __restrict__ ard, const double* __restrict__ brd,
    const double* __restrict__ azd, const double* __restrict__ bzd,
    double* __restrict__ frss, double* __restrict__ fzcs,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jF_local = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_local >= ns_force_local || n > ntor) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  // d_pmat_ard/brd/azd/bzd are per-cfg snapshots from
  // ComputePreconditioningMatrixCuda, read at this configuration's offset.
  size_t cfg_pmat = (size_t)config * (size_t)ns_force_local * 2;
  const int mPar = 1;
  int ard_idx = cfg_pmat + (size_t)(jF_local * 2 + mPar);
  double a_r = ard[ard_idx], b_r = brd[ard_idx];
  double a_z = azd[ard_idx], b_z = bzd[ard_idx];
  double denom = a_r + b_r + a_z + b_z;
  if (denom == 0.0) return;
  double fsR = (a_r + b_r) / denom;
  double fsZ = (a_z + b_z) / denom;
  size_t idx_mn = cfg_spec + (size_t)(((jF_local * mpol + 1) * (ntor + 1)) + n);
  frss[idx_mn] *= fsR;
  fzcs[idx_mn] *= fsZ;
}

// k_apply_lambda_preconditioner: scales flsc and flcs by lambdaPreconditioner
// per (jF_local, m, n).
__global__ void k_apply_lambda_preconditioner(
    int n_config, int ns_local,
    int ns_con_local, int mpol, int ntor, bool lthreed,
    const double* __restrict__ lambdaPreconditioner,
    double* __restrict__ flsc, double* __restrict__ flcs,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z / ns_con_local;
  int jF_local = blockIdx.z - config * ns_con_local;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int m = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_local >= ns_con_local || m >= mpol || n > ntor) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  // lambdaPreconditioner is per-cfg (k_ulp_assemble writes one slot per
  // configuration), read at this configuration's offset.
  size_t cfg_lp = (size_t)config * (size_t)ns_con_local *
                  (size_t)mpol * (size_t)(ntor + 1);
  int local_mn = ((jF_local * mpol + m) * (ntor + 1)) + n;
  double scale = lambdaPreconditioner[cfg_lp + local_mn];
  size_t idx_mn = cfg_spec + (size_t)local_mn;
  flsc[idx_mn] *= scale;
  if (lthreed) flcs[idx_mn] *= scale;
}

// (k_apply_rz_thomas removed: superseded by k_apply_rz_pcr; the serial Thomas
//  was single-thread-per-block which was the largest GPU bottleneck.)

// k_rz_transpose_in: spectra (jF_local, m, n) into the Thomas layout (mn,
// basis, jF_global), zero outside the force range.
__global__ void k_rz_transpose_in(int n_config, int ns_local,
                                    int ns_force_local, int mpol, int ntor,
                                    int ns_total, int num_basis, int nsMinF,
                                    bool lthreed,
                                    const double* __restrict__ frcc,
                                    const double* __restrict__ frss,
                                    const double* __restrict__ fzsc,
                                    const double* __restrict__ fzcs,
                                    double* __restrict__ cR,
                                    double* __restrict__ cZ,
                                    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int mnsize = mpol * (ntor + 1);
  int mn = blockIdx.y;
  int jF = blockIdx.x * blockDim.x + threadIdx.x;
  if (mn >= mnsize || jF >= ns_total) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local * (size_t)mnsize;
  size_t cfg_thomas = (size_t)config * (size_t)mnsize *
                      (size_t)num_basis * (size_t)ns_total;
  int jF_local = jF - nsMinF;
  size_t idx_b0 = cfg_thomas + (size_t)((mn * num_basis + 0) * ns_total + jF);
  if (jF_local < 0 || jF_local >= ns_force_local) {
    cR[idx_b0] = 0.0;
    cZ[idx_b0] = 0.0;
    if (lthreed) {
      size_t idx_b1 = cfg_thomas + (size_t)((mn * num_basis + 1) * ns_total + jF);
      cR[idx_b1] = 0.0;
      cZ[idx_b1] = 0.0;
    }
    return;
  }
  size_t idx_spec = cfg_spec + (size_t)(jF_local * mnsize + mn);
  cR[idx_b0] = frcc[idx_spec];
  cZ[idx_b0] = fzsc[idx_spec];
  if (lthreed) {
    size_t idx_b1 = cfg_thomas + (size_t)((mn * num_basis + 1) * ns_total + jF);
    cR[idx_b1] = frss[idx_spec];
    cZ[idx_b1] = fzcs[idx_spec];
  }
}

// k_rz_transpose_out: the Thomas layout (mn, basis, jF_global) back into the
// spectra (jF_local, m, n).
__global__ void k_rz_transpose_out(int n_config, int ns_local,
                                     int ns_force_local, int mpol, int ntor,
                                     int ns_total, int num_basis, int nsMinF,
                                     bool lthreed,
                                     const double* __restrict__ cR,
                                     const double* __restrict__ cZ,
                                     double* __restrict__ frcc,
                                     double* __restrict__ frss,
                                     double* __restrict__ fzsc,
                                     double* __restrict__ fzcs,
                                     const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int mnsize = mpol * (ntor + 1);
  int mn = blockIdx.y;
  int jF_local = blockIdx.x * blockDim.x + threadIdx.x;
  if (mn >= mnsize || jF_local >= ns_force_local) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local * (size_t)mnsize;
  size_t cfg_thomas = (size_t)config * (size_t)mnsize *
                      (size_t)num_basis * (size_t)ns_total;
  int jF = jF_local + nsMinF;
  size_t idx_spec = cfg_spec + (size_t)(jF_local * mnsize + mn);
  frcc[idx_spec] = cR[cfg_thomas + (size_t)((mn * num_basis + 0) * ns_total + jF)];
  fzsc[idx_spec] = cZ[cfg_thomas + (size_t)((mn * num_basis + 0) * ns_total + jF)];
  if (lthreed) {
    frss[idx_spec] = cR[cfg_thomas + (size_t)((mn * num_basis + 1) * ns_total + jF)];
    fzcs[idx_spec] = cZ[cfg_thomas + (size_t)((mn * num_basis + 1) * ns_total + jF)];
  }
}

// k_dealias_fwd: gsc and gcs = tcon * sum gConEff * sinmui cosnv (cosmui sinnv)
// for 1 <= m < mpol - 1, zero at the ends.
__global__ void k_dealias_fwd(
    int n_config, int ns_force_local, int ns_con_local,
    int mpol, int ntor, int nZeta, int nThetaReduced,
    int nThetaEff, int nnyq2_plus_1,
    const double* __restrict__ gConEff, const double* __restrict__ tcon,
    const double* __restrict__ sinmui, const double* __restrict__ cosmui,
    const double* __restrict__ cosnv,  const double* __restrict__ sinnv,
    double* __restrict__ gsc, double* __restrict__ gcs) {
  int config = blockIdx.z / ns_force_local;
  int jF = blockIdx.z - config * ns_force_local;
  if (config >= n_config) return;
  int m = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF >= ns_force_local || m >= mpol || n > ntor) return;
  size_t cfg_con_g = (size_t)config * (size_t)ns_con_local *
                     (size_t)nZeta * (size_t)nThetaEff;
  size_t cfg_tcon  = (size_t)config * (size_t)ns_con_local;
  size_t cfg_spec  = (size_t)config * (size_t)ns_force_local *
                     (size_t)mpol * (size_t)(ntor + 1);
  size_t idx_mn = cfg_spec + (size_t)((jF * mpol + m) * (ntor + 1) + n);
  if (m == 0 || m >= mpol - 1) {
    gsc[idx_mn] = 0.0;
    gcs[idx_mn] = 0.0;
    return;
  }
  double t = tcon[cfg_tcon + jF];
  double acc_sc = 0.0, acc_cs = 0.0;
  #pragma unroll 24
  for (int k = 0; k < nZeta; ++k) {
    double w0 = 0.0, w1 = 0.0;
    #pragma unroll 14
    for (int l = 0; l < nThetaReduced; ++l) {
      double g = gConEff[cfg_con_g + (size_t)((jF * nZeta + k) * nThetaEff + l)];
      int bml = m * nThetaReduced + l;
      w0 += g * sinmui[bml];
      w1 += g * cosmui[bml];
    }
    int kn = k * nnyq2_plus_1 + n;
    acc_sc += cosnv[kn] * w0;
    acc_cs += sinnv[kn] * w1;
  }
  gsc[idx_mn] = acc_sc * t;
  gcs[idx_mn] = acc_cs * t;
}

// k_dealias_inv: gCon = sum over m of faccon * sum over n of gsc cosnv sinmu +
// gcs sinnv cosmu, per (jF, k, l).
__global__ void k_dealias_inv(
    int n_config, int ns_force_local, int ns_con_local,
    int mpol, int ntor, int nZeta, int nThetaReduced,
    int nThetaEff, int nnyq2_plus_1,
    const double* __restrict__ gsc, const double* __restrict__ gcs,
    const double* __restrict__ sinmu, const double* __restrict__ cosmu,
    const double* __restrict__ cosnv, const double* __restrict__ sinnv,
    const double* __restrict__ faccon,
    double* __restrict__ m_gCon) {
  int config = blockIdx.z / ns_force_local;
  int jF = blockIdx.z - config * ns_force_local;
  if (config >= n_config) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF >= ns_force_local || k >= nZeta || l >= nThetaReduced) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_force_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t cfg_grid = (size_t)config * (size_t)ns_con_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  double acc = 0.0;
  // Explicit unrolling, which the fac == 0 skip otherwise prevents, overlaps
  // the inner loop's cosnv and sinnv loads.
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

// k_decompose_into: FourierCoeffs::decomposeInto (lasym = false), physical
// times scalxc into decomposed on [nsMin, jMaxIncludingBoundary).
__global__ void k_decompose_into(int n_config,
                                  int ns_dec_local, int ns_local,
                                  int mpol, int ntor,
                                  int nsMin_to_nsMinF1, bool lthreed,
                                  const double* __restrict__ scalxc,
                                  const double* __restrict__ phys_frcc,
                                  const double* __restrict__ phys_frss,
                                  const double* __restrict__ phys_fzsc,
                                  const double* __restrict__ phys_fzcs,
                                  const double* __restrict__ phys_flsc,
                                  const double* __restrict__ phys_flcs,
                                  double* __restrict__ dec_frcc,
                                  double* __restrict__ dec_frss,
                                  double* __restrict__ dec_fzsc,
                                  double* __restrict__ dec_fzcs,
                                  double* __restrict__ dec_flsc,
                                  double* __restrict__ dec_flcs) {
  int config = blockIdx.z / ns_dec_local;
  int jF_dec = blockIdx.z - config * ns_dec_local;
  if (config >= n_config) return;
  int m = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_dec >= ns_dec_local || m >= mpol || n > ntor) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t idx = cfg_spec + (size_t)((jF_dec * mpol + m) * (ntor + 1) + n);
  int scalxc_row = jF_dec + nsMin_to_nsMinF1;  // jF - nsMinF1 in CPU index
  double scal = scalxc[scalxc_row * 2 + (m & 1)];
  dec_frcc[idx] = phys_frcc[idx] * scal;
  dec_fzsc[idx] = phys_fzsc[idx] * scal;
  dec_flsc[idx] = phys_flsc[idx] * scal;
  if (lthreed) {
    dec_frss[idx] = phys_frss[idx] * scal;
    dec_fzcs[idx] = phys_fzcs[idx] * scal;
    dec_flcs[idx] = phys_flcs[idx] * scal;
  }
}

// k_m1_constraint: FourierCoeffs::m1Constraint at m = 1 in 3D, mixing frss and
// fzcs with sigma = -sign_of_jacobian.
__global__ void k_m1_constraint(int n_config, int ns_local,
                                  int ns_force_local, int mpol, int ntor,
                                  double scalingFactor, double sigma,
                                  double* __restrict__ dec_frss,
                                  double* __restrict__ dec_fzcs) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jF_force = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_force >= ns_force_local || n > ntor) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t idx = cfg_spec + (size_t)((jF_force * mpol + 1) * (ntor + 1) + n);
  double old_rss = dec_frss[idx];
  double old_zcs = dec_fzcs[idx];
  dec_frss[idx] = (old_rss + sigma * old_zcs) * scalingFactor;
  dec_fzcs[idx] = (sigma * old_rss - old_zcs) * scalingFactor;
}

// k_m1_constraint_gated: k_m1_constraint, then the m = 1 Z force zeroed on the
// flagged configurations, or all when zero_all is set.
__global__ void k_m1_constraint_gated(int n_config, int ns_local,
                                      int ns_force_local, int mpol, int ntor,
                                      double scalingFactor, double sigma,
                                      const unsigned char* __restrict__ zero_flags,
                                      int zero_all,
                                      double* __restrict__ dec_frss,
                                      double* __restrict__ dec_fzcs) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jF_force = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_force >= ns_force_local || n > ntor) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t idx = cfg_spec + (size_t)((jF_force * mpol + 1) * (ntor + 1) + n);
  double old_rss = dec_frss[idx];
  double old_zcs = dec_fzcs[idx];
  dec_frss[idx] = (old_rss + sigma * old_zcs) * scalingFactor;
  const bool zeroed =
      zero_flags != nullptr ? zero_flags[config] != 0 : zero_all != 0;
  dec_fzcs[idx] = zeroed ? 0.0 : (sigma * old_rss - old_zcs) * scalingFactor;
}

// k_mask_geometry_above: zeroes R and Z at m >= mpol_geometry or n >
// ntor_geometry on the first ns_count surfaces; lambda is untouched.
__global__ void k_mask_geometry_above(int n_config, size_t cfg_stride,
                                      int ns_count, int mpol, int ntor,
                                      int mpol_geometry, int ntor_geometry,
                                      int lthreed,
                                      double* __restrict__ rcc,
                                      double* __restrict__ rss,
                                      double* __restrict__ zsc,
                                      double* __restrict__ zcs) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int j = blockIdx.y;
  int idx_mn = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= ns_count || idx_mn >= mpol * (ntor + 1)) return;
  int m = idx_mn / (ntor + 1);
  int n = idx_mn - m * (ntor + 1);
  if (m < mpol_geometry && n <= ntor_geometry) return;
  size_t idx = (size_t)config * cfg_stride +
               (size_t)j * (size_t)(mpol * (ntor + 1)) + (size_t)idx_mn;
  rcc[idx] = 0.0;
  zsc[idx] = 0.0;
  if (lthreed) {
    rss[idx] = 0.0;
    zcs[idx] = 0.0;
  }
}

// k_pres_final_reduce: one block per configuration sums the thermal and
// magnetic partials into [thermal, magnetic, mhd].
__global__ void k_pres_final_reduce(int n_config, int ns_h, double deltaS,
                                     double adiabaticIndex,
                                     const double* __restrict__ thermal_partial,
                                     const double* __restrict__ magnetic_partial,
                                     double* __restrict__ scalars_out,
                                     const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  __shared__ double s_t[256], s_m[256];
  double acc_t = 0.0, acc_m = 0.0;
  for (int jH = threadIdx.x; jH < ns_h; jH += blockDim.x) {
    acc_t += thermal_partial[cfg_prof + jH];
    acc_m += magnetic_partial[cfg_prof + jH];
  }
  s_t[threadIdx.x] = acc_t;
  s_m[threadIdx.x] = acc_m;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_t[threadIdx.x] += s_t[threadIdx.x + stride];
      s_m[threadIdx.x] += s_m[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    double thermal = s_t[0] * deltaS;
    double magnetic = fabs(s_m[0]) * deltaS;
    scalars_out[config * 3 + 0] = thermal;
    scalars_out[config * 3 + 1] = magnetic;
    scalars_out[config * 3 + 2] = magnetic + thermal / (adiabaticIndex - 1.0);
  }
}

// k_rz_norm_per_cfg: one serial thread per configuration sums the squared R and
// Z coefficients as rzNorm does and writes fNorm1.
__global__ void k_rz_norm_per_cfg(
    int n_config, int ns_local, int j_begin, int j_count, int mpol, int ntor,
    bool lthreed,
    const double* __restrict__ d_x_rcc, const double* __restrict__ d_x_rss,
    const double* __restrict__ d_x_zsc, const double* __restrict__ d_x_zcs,
    double* __restrict__ d_fnorm1,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  const int cfg = blockIdx.x;
  if (cfg >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[cfg]) return;
  if (threadIdx.x != 0) return;
  const size_t base = (size_t)cfg * ns_local * mpol * (ntor + 1);
  double norm2 = 0.0;
  for (int j = 0; j < j_count; ++j) {
    for (int m = 0; m < mpol; ++m) {
      for (int n = 0; n <= ntor; ++n) {
        const size_t idx =
            base + ((size_t)(j_begin + j) * mpol + m) * (ntor + 1) + n;
        if (n > 0 || m > 0) {
          norm2 += d_x_rcc[idx] * d_x_rcc[idx];
        }
        norm2 += d_x_zsc[idx] * d_x_zsc[idx];
        if (lthreed) {
          norm2 += d_x_rss[idx] * d_x_rss[idx];
          norm2 += d_x_zcs[idx] * d_x_zcs[idx];
        }
      }
    }
  }
  d_fnorm1[cfg] = 1.0 / norm2;
}

// k_force_norm_final_reduce: one block per configuration sums the per-surface
// force-norm partials into [sum_rz, sum_l].
__global__ void k_force_norm_final_reduce(
    int n_config, int ns_h,
    const double* __restrict__ rz_partial,
    const double* __restrict__ l_partial,
    double* __restrict__ scalars_out,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  __shared__ double s_rz[256], s_l[256];
  double acc_rz = 0.0, acc_l = 0.0;
  for (int jH = threadIdx.x; jH < ns_h; jH += blockDim.x) {
    acc_rz += rz_partial[cfg_prof + jH];
    acc_l  += l_partial[cfg_prof + jH];
  }
  s_rz[threadIdx.x] = acc_rz;
  s_l[threadIdx.x]  = acc_l;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_rz[threadIdx.x] += s_rz[threadIdx.x + stride];
      s_l[threadIdx.x]  += s_l[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    scalars_out[config * 2 + 0] = s_rz[0];
    scalars_out[config * 2 + 1] = s_l[0];
  }
}

// k_check_convergence: each configuration's residuals normalized as
// evalFResInvar does, all three compared with ftolv.
__global__ void k_check_convergence(
    int n_config,
    const double* __restrict__ scalars_out,
    const double* __restrict__ fnorm_scalars,
    const double* __restrict__ pressure_scalars,
    const double* __restrict__ volumes,
    double lamscale,
    double ftolv,
    std::uint8_t* __restrict__ conv_flag,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (threadIdx.x != 0) return;
  std::uint8_t active = (d_active_per_cfg ? d_active_per_cfg[config] : 1);
  if (!active) {
    conv_flag[config] = 1;  // inactive cfg is considered done
    return;
  }
  double r = scalars_out[config * 3 + 0];
  double z = scalars_out[config * 3 + 1];
  double l = scalars_out[config * 3 + 2];
  if (fnorm_scalars && pressure_scalars && volumes && lamscale > 0.0) {
    constexpr double r1scale = 0.25;
    const double sum_rz = fnorm_scalars[config * 2 + 0];
    const double sum_l = fnorm_scalars[config * 2 + 1];
    const double thermal = pressure_scalars[config * 3 + 0];
    const double magnetic = pressure_scalars[config * 3 + 1];
    const double pv = volumes[config];
    const double energy_density = fmax(magnetic, thermal) / pv;
    const double fnorm_rz =
        1.0 / (sum_rz * energy_density * energy_density);
    const double fnorm_l = 1.0 / (sum_l * lamscale * lamscale);
    r *= fnorm_rz * r1scale;
    z *= fnorm_rz * r1scale;
    l *= fnorm_l;
  }
  conv_flag[config] = (r <= ftolv && z <= ftolv && l <= ftolv) ? 1 : 0;
}

// k_residuals_par: one block of 256 threads per configuration reduces the
// residual sums with a shared-memory tree.
__global__ void k_residuals_par(
    int n_config, int ns_local,
    int jLocal_max_rz, int jLocal_max_boundary,
    int mpol, int ntor, bool lthreed,
    const double* __restrict__ frcc,
    const double* __restrict__ frss,
    const double* __restrict__ fzsc,
    const double* __restrict__ fzcs,
    const double* __restrict__ flsc,
    const double* __restrict__ flcs,
    double* __restrict__ scalars_out,
    const std::uint8_t* __restrict__ d_active_per_cfg,
    int jLocal_max_rz_edge,
    const std::uint8_t* __restrict__ edge_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  if (edge_per_cfg != nullptr && edge_per_cfg[config]) {
    jLocal_max_rz = jLocal_max_rz_edge;
  }
  int tid = threadIdx.x;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  int total = jLocal_max_boundary * mpol * (ntor + 1);
  double acc_R = 0.0, acc_Z = 0.0, acc_L = 0.0;
  // Strided sweep across the flattened (jLocal, m, n) index space.
  for (int i = tid; i < total; i += blockDim.x) {
    int n = i % (ntor + 1);
    int rest = i / (ntor + 1);
    int m = rest % mpol;
    int jLocal = rest / mpol;
    const bool in_rz = (jLocal < jLocal_max_rz);
    size_t cidx = cfg_spec + (size_t)i;
    double v;
    if (in_rz) {
      v = frcc[cidx]; acc_R += v * v;
      v = fzsc[cidx]; acc_Z += v * v;
    }
    v = flsc[cidx]; acc_L += v * v;
    if (lthreed) {
      if (in_rz) {
        v = frss[cidx]; acc_R += v * v;
        v = fzcs[cidx]; acc_Z += v * v;
      }
      v = flcs[cidx]; acc_L += v * v;
    }
  }
  // Shared-memory tree reduce across the block. Three accumulators ->
  // three separate trees laid out contiguously.
  __shared__ double s_R[256], s_Z[256], s_L[256];
  s_R[tid] = acc_R;
  s_Z[tid] = acc_Z;
  s_L[tid] = acc_L;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_R[tid] += s_R[tid + stride];
      s_Z[tid] += s_Z[tid + stride];
      s_L[tid] += s_L[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    scalars_out[config * 3 + 0] = s_R[0];
    scalars_out[config * 3 + 1] = s_Z[0];
    scalars_out[config * 3 + 2] = s_L[0];
  }
}

// Multi-block residuals: K sub-blocks per configuration each reduce a slice
// into partials_out[cfg * K * 3 + k * 3 + comp].
__global__ void k_residuals_par_K(
    int n_config, int ns_local,
    int jLocal_max_rz, int jLocal_max_boundary,
    int mpol, int ntor, bool lthreed,
    int n_partitions,
    const double* __restrict__ frcc,
    const double* __restrict__ frss,
    const double* __restrict__ fzsc,
    const double* __restrict__ fzcs,
    const double* __restrict__ flsc,
    const double* __restrict__ flcs,
    double* __restrict__ partials_out,
    const std::uint8_t* __restrict__ d_active_per_cfg,
    int jLocal_max_rz_edge,
    const std::uint8_t* __restrict__ edge_per_cfg) {
  int config = blockIdx.y;
  int partition = blockIdx.x;
  if (config >= n_config) return;
  if (edge_per_cfg != nullptr && edge_per_cfg[config]) {
    jLocal_max_rz = jLocal_max_rz_edge;
  }
  if (d_active_per_cfg && !d_active_per_cfg[config]) {
    if (threadIdx.x == 0) {
      size_t base = ((size_t)config * (size_t)n_partitions +
                     (size_t)partition) * 3;
      partials_out[base + 0] = 0.0;
      partials_out[base + 1] = 0.0;
      partials_out[base + 2] = 0.0;
    }
    return;
  }
  int tid = threadIdx.x;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  int total = jLocal_max_boundary * mpol * (ntor + 1);
  // Partition the flattened index space contiguously. Each block handles
  // indices in [part_lo, part_hi).
  int per_part = (total + n_partitions - 1) / n_partitions;
  int part_lo = partition * per_part;
  int part_hi = part_lo + per_part;
  if (part_hi > total) part_hi = total;
  double acc_R = 0.0, acc_Z = 0.0, acc_L = 0.0;
  for (int i = part_lo + tid; i < part_hi; i += blockDim.x) {
    int n = i % (ntor + 1);
    int rest = i / (ntor + 1);
    int m = rest % mpol;
    int jLocal = rest / mpol;
    const bool in_rz = (jLocal < jLocal_max_rz);
    size_t cidx = cfg_spec + (size_t)i;
    double v;
    if (in_rz) {
      v = frcc[cidx]; acc_R += v * v;
      v = fzsc[cidx]; acc_Z += v * v;
    }
    v = flsc[cidx]; acc_L += v * v;
    if (lthreed) {
      if (in_rz) {
        v = frss[cidx]; acc_R += v * v;
        v = fzcs[cidx]; acc_Z += v * v;
      }
      v = flcs[cidx]; acc_L += v * v;
    }
  }
  __shared__ double s_R[256], s_Z[256], s_L[256];
  s_R[tid] = acc_R;
  s_Z[tid] = acc_Z;
  s_L[tid] = acc_L;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_R[tid] += s_R[tid + stride];
      s_Z[tid] += s_Z[tid + stride];
      s_L[tid] += s_L[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    size_t base = ((size_t)config * (size_t)n_partitions +
                   (size_t)partition) * 3;
    partials_out[base + 0] = s_R[0];
    partials_out[base + 1] = s_Z[0];
    partials_out[base + 2] = s_L[0];
  }
}

// k_residuals_finalize_K: one warp per configuration reduces its K <= 32
// partials into one triple.
__global__ void k_residuals_finalize_K(
    int n_config, int n_partitions,
    const double* __restrict__ partials_in,
    double* __restrict__ scalars_out,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int tid = threadIdx.x;
  // Each thread loads one partition's triple (or zero if tid >= n_partitions).
  double v_R = 0.0, v_Z = 0.0, v_L = 0.0;
  if (tid < n_partitions) {
    size_t base = ((size_t)config * (size_t)n_partitions +
                   (size_t)tid) * 3;
    v_R = partials_in[base + 0];
    v_Z = partials_in[base + 1];
    v_L = partials_in[base + 2];
  }
  // Warp butterfly reduction in a fixed order, so runs repeat bit for bit.
  for (int delta = 16; delta > 0; delta >>= 1) {
    v_R += __shfl_xor_sync(0xffffffff, v_R, delta);
    v_Z += __shfl_xor_sync(0xffffffff, v_Z, delta);
    v_L += __shfl_xor_sync(0xffffffff, v_L, delta);
  }
  if (tid == 0) {
    scalars_out[config * 3 + 0] = v_R;
    scalars_out[config * 3 + 1] = v_Z;
    scalars_out[config * 3 + 2] = v_L;
  }
}

// k_update_timestep: the host damping of Vmec::Evolve on the device, from the
// preconditioned residual sum and the 1/tau ring.
__global__ void k_update_timestep(
    int n_config,
    int iter_phase,
    double time_step,
    const double* __restrict__ d_fnorm1,
    double fsql_scale,
    const double* __restrict__ d_residuals_partial,
    double* __restrict__ d_inv_tau,
    double* __restrict__ d_prev_fsq,
    double* __restrict__ d_fac_b1,
    const std::uint8_t* __restrict__ d_active_per_cfg,
    const double* __restrict__ d_time_step_per_cfg,
    const std::uint8_t* __restrict__ d_phase_per_cfg) {
  const int cfg = blockIdx.x;
  const int tid = threadIdx.x;
  if (cfg >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[cfg]) {
    // Inactive cfg: leave d_fac_b1 at its prior value; the per-cfg
    // active mask gates downstream kernel application.
    return;
  }
  // Per-configuration restart control gives each configuration its own
  // time step and ring reset.
  const double dt =
      d_time_step_per_cfg != nullptr ? d_time_step_per_cfg[cfg] : time_step;
  const int phase =
      d_phase_per_cfg != nullptr ? d_phase_per_cfg[cfg] : iter_phase;
  constexpr int kND = 10;  // matches host kNDamp
  // fsq1 with evalFResPrecd's normalization, fNorm1 on R and Z and deltaS on L,
  // normalized then summed left to right as the host does.
  double fsq1 = 0.0;
  if (tid == 0) {
    const double* res = d_residuals_partial + (size_t)cfg * 3;
    const double fnorm1 = d_fnorm1[cfg];
    fsq1 = res[0] * fnorm1 + res[1] * fnorm1 + res[2] * fsql_scale;
  }
  fsq1 = __shfl_sync(0xffffffff, fsq1, 0);
  // Load ring buffer entry into thread-local
  double* inv_tau_cfg = d_inv_tau + (size_t)cfg * kND;
  double entry = 0.0;
  if (tid < kND) entry = inv_tau_cfg[tid];
  // Phase 0: reset invTau to 0.15 / time_step (matches setConstant call)
  if (phase == 0) {
    if (tid < kND) entry = 0.15 / dt;
  } else {
    // Shift left: entry[i] <- entry[i+1] for i < kND-1; entry[kND-1] gets
    // the new sample. Use shfl_down for the shift.
    double next_entry = __shfl_down_sync(0xffffffff, entry, 1);
    if (tid < kND - 1) entry = next_entry;
    // Compute the new last entry (only tid==0 needs the result; broadcast)
    double new_entry = 0.0;
    if (tid == 0) {
      double prev_fsq_val = d_prev_fsq[cfg];
      double invtau_num = 0.0;
      if (fsq1 != 0.0 && prev_fsq_val != 0.0) {
        double ratio = fsq1 / prev_fsq_val;
        if (ratio > 0.0) {
          double logr = fabs(log(ratio));
          invtau_num = (logr < 0.15) ? logr : 0.15;
        }
      }
      new_entry = invtau_num / dt;
    }
    new_entry = __shfl_sync(0xffffffff, new_entry, 0);
    if (tid == kND - 1) entry = new_entry;
  }
  // Write back to ring buffer
  if (tid < kND) inv_tau_cfg[tid] = entry;
  __syncwarp();
  // A single thread sums left to right as Eigen's sum() does, keeping fac and
  // b1 bit-identical to the host.
  if (tid == 0) {
    double sum = 0.0;
    #pragma unroll
    for (int i = 0; i < kND; ++i) {
      sum += inv_tau_cfg[i];
    }
    double otav = sum / kND;
    double dtau = dt * otav / 2.0;
    double b1 = 1.0 - dtau;
    double fac = 1.0 / (1.0 + dtau);
    d_fac_b1[(size_t)cfg * 2 + 0] = fac;
    d_fac_b1[(size_t)cfg * 2 + 1] = b1;
    // Update prev_fsq for next iter
    d_prev_fsq[cfg] = fsq1;
  }
}

// k_tau_minmax: one block per configuration finds the min and max of its
// half-grid tau.
__global__ void k_tau_minmax(int n_config, int total,
                              const double* __restrict__ tau,
                              double* __restrict__ out2,
                              const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  size_t cfg = (size_t)config * (size_t)total;
  __shared__ double s_min[256], s_max[256];
  double mn = 1e300, mx = -1e300;
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    double t = tau[cfg + (size_t)i];
    if (t < mn) mn = t;
    if (t > mx) mx = t;
  }
  s_min[threadIdx.x] = mn;
  s_max[threadIdx.x] = mx;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (s_min[threadIdx.x + stride] < s_min[threadIdx.x]) {
        s_min[threadIdx.x] = s_min[threadIdx.x + stride];
      }
      if (s_max[threadIdx.x + stride] > s_max[threadIdx.x]) {
        s_max[threadIdx.x] = s_max[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    out2[config * 2 + 0] = s_min[0];
    out2[config * 2 + 1] = s_max[0];
  }
}

// k_apply_rz_pcr: parallel cyclic reduction of each (mn) tridiagonal system in
// shared memory, log2(N) passes and x = c / d.
__global__ void k_apply_rz_pcr(int n_config, int mnsize, int ns_total, int num_basis,
                                 const int* __restrict__ jMin, int jMax,
                                 const double* __restrict__ a_in,
                                 const double* __restrict__ d_in,
                                 const double* __restrict__ b_in,
                                 double* __restrict__ c_inout,
                                 const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int mn = blockIdx.x;
  if (mn >= mnsize) return;
  int j0 = jMin[mn];
  int j1 = jMax;
  int N = j1 - j0;
  if (N <= 0) return;
  int tid = threadIdx.x;
  size_t cfg_mat = (size_t)config * (size_t)mnsize * (size_t)ns_total;
  size_t cfg_c   = (size_t)config * (size_t)mnsize * (size_t)num_basis *
                   (size_t)ns_total;

  // Shared memory layout: [ns_total a][ns_total d][ns_total b][ns_total * 2 c]
  // 2 is the maximum num_basis for the stellarator-symmetric 3D case.
  extern __shared__ double smem[];
  double* s_a = smem;
  double* s_d = smem + ns_total;
  double* s_b = smem + 2 * ns_total;
  double* s_c = smem + 3 * ns_total;

  if (tid < N) {
    int gi = tid + j0;
    s_a[tid] = a_in[cfg_mat + (size_t)(mn * ns_total + gi)];
    s_d[tid] = d_in[cfg_mat + (size_t)(mn * ns_total + gi)];
    s_b[tid] = b_in[cfg_mat + (size_t)(mn * ns_total + gi)];
    for (int ib = 0; ib < num_basis; ++ib) {
      s_c[tid * 2 + ib] = c_inout[cfg_c + (size_t)((mn * num_basis + ib) * ns_total + gi)];
    }
  }
  __syncthreads();

  for (int k = 1; k < N; k <<= 1) {
    double a_new = 0.0, d_new = 0.0, b_new = 0.0;
    double c_new0 = 0.0, c_new1 = 0.0;
    if (tid < N) {
      int i_prev = tid - k;
      int i_next = tid + k;
      double alpha = (i_prev >= 0)     ? -s_b[tid] / s_d[i_prev] : 0.0;
      double beta  = (i_next <  N)     ? -s_a[tid] / s_d[i_next] : 0.0;
      d_new = s_d[tid];
      if (i_prev >= 0) d_new += alpha * s_a[i_prev];
      if (i_next <  N) d_new += beta  * s_b[i_next];
      b_new = (i_prev >= 0) ? alpha * s_b[i_prev] : 0.0;
      a_new = (i_next <  N) ? beta  * s_a[i_next] : 0.0;
      c_new0 = s_c[tid * 2 + 0];
      if (i_prev >= 0) c_new0 += alpha * s_c[i_prev * 2 + 0];
      if (i_next <  N) c_new0 += beta  * s_c[i_next * 2 + 0];
      if (num_basis == 2) {
        c_new1 = s_c[tid * 2 + 1];
        if (i_prev >= 0) c_new1 += alpha * s_c[i_prev * 2 + 1];
        if (i_next <  N) c_new1 += beta  * s_c[i_next * 2 + 1];
      }
    }
    __syncthreads();
    if (tid < N) {
      s_a[tid] = a_new;
      s_d[tid] = d_new;
      s_b[tid] = b_new;
      s_c[tid * 2 + 0] = c_new0;
      if (num_basis == 2) s_c[tid * 2 + 1] = c_new1;
    }
    __syncthreads();
  }

  if (tid < N) {
    int gi = tid + j0;
    for (int ib = 0; ib < num_basis; ++ib) {
      c_inout[cfg_c + (size_t)((mn * num_basis + ib) * ns_total + gi)] = s_c[tid * 2 + ib] / s_d[tid];
    }
  }
}

// k_apply_rz_thomas_block: serial Thomas elimination with one block per
// (config, mn) row and the ratios in dynamic shared memory.
__global__ void k_apply_rz_thomas_block(
    int n_config, int mnsize, int ns_total, int num_basis,
    const int* __restrict__ jMin, int jMax,
    const double* __restrict__ a_in, const double* __restrict__ d_in,
    const double* __restrict__ b_in, double* __restrict__ c_inout,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int row = blockIdx.x;
  int config = row / mnsize;
  int mn = row - config * mnsize;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int j0 = jMin[mn];
  if (jMax - j0 <= 0) return;
  size_t r0 = ((size_t)config * (size_t)mnsize + (size_t)mn) *
              (size_t)ns_total;
  size_t cfg_c = (size_t)config * (size_t)mnsize * (size_t)num_basis *
                 (size_t)ns_total;
  extern __shared__ double s_aloc[];  // [jMax], only [j0, jMax) is used

  // Cooperative load of the sub-diagonal into shared memory.
  for (int j = j0 + threadIdx.x; j < jMax; j += blockDim.x) {
    s_aloc[j] = a_in[r0 + j];
  }
  __syncthreads();

  // Forward elimination of the ratios (sequential recurrence).
  if (threadIdx.x == 0) {
    s_aloc[j0] /= d_in[r0 + j0];
    for (int j = j0 + 1; j < jMax - 1; ++j) {
      const double denominator = d_in[r0 + j] - s_aloc[j - 1] * b_in[r0 + j];
      s_aloc[j] /= denominator;
    }
  }
  __syncthreads();

  // Forward sweep and back substitution per right-hand side; the bases are
  // independent, so one thread each handles a basis column.
  if (threadIdx.x < num_basis) {
    int ib = threadIdx.x;
    double* c = c_inout + cfg_c + (size_t)(mn * num_basis + ib) * ns_total;
    c[j0] /= d_in[r0 + j0];
    for (int j = j0 + 1; j < jMax; ++j) {
      const double denominator = d_in[r0 + j] - s_aloc[j - 1] * b_in[r0 + j];
      c[j] = (c[j] - c[j - 1] * b_in[r0 + j]) / denominator;
    }
    for (int j = jMax - 2; j > j0 - 1; --j) {
      c[j] -= s_aloc[j] * c[j + 1];
    }
  }
}

// k_assemble_rz_preconditioner: IdealMhdModel::assembleRZPreconditioner on the
// device, writing (mn, jF_global) as k_apply_rz_pcr reads.
__global__ void k_assemble_rz_preconditioner(
    int n_config, int ns_h,
    int mpol, int ntor, int nfp,
    int ns_total, int ns_force_local, int nsMinF, int nsMinH, int nsMaxH,
    int jMax, int lcfs_owning,
    double edge_pedestal, double mult_fact_zc00,
    const double* __restrict__ d_arm, const double* __restrict__ d_brm,
    const double* __restrict__ d_azm, const double* __restrict__ d_bzm,
    const double* __restrict__ d_ard, const double* __restrict__ d_brd,
    const double* __restrict__ d_azd, const double* __restrict__ d_bzd,
    const double* __restrict__ d_cxd,
    double* __restrict__ d_aR, double* __restrict__ d_dR, double* __restrict__ d_bR,
    double* __restrict__ d_aZ, double* __restrict__ d_dZ, double* __restrict__ d_bZ,
    int* __restrict__ d_jMin,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int mnsize = mpol * (ntor + 1);
  int mn = blockIdx.x;
  int jF = blockIdx.y * blockDim.x + threadIdx.x;
  if (mn >= mnsize || jF >= ns_total) return;

  size_t cfg_half_prof  = (size_t)config * (size_t)ns_h * 2;
  size_t cfg_force_prof = (size_t)config * (size_t)ns_force_local * 2;
  size_t cfg_cxd        = (size_t)config * (size_t)ns_force_local;
  size_t cfg_mat        = (size_t)config * (size_t)mnsize * (size_t)ns_total;

  int m = mn / (ntor + 1);
  int n = mn % (ntor + 1);
  int m_parity = m & 1;
  int jMin_value = (m > 0) ? 1 : 0;

  // Write jMin once per mn; only config 0 writes (shared per-mn).
  if (config == 0 && jF == 0) d_jMin[mn] = jMin_value;

  size_t out_idx = cfg_mat + (size_t)(mn * ns_total + jF);

  // jF_upper = min(nsMinF + ns_force_local, jMax).
  int nsMaxF = nsMinF + ns_force_local;
  int jF_upper = (nsMaxF < jMax) ? nsMaxF : jMax;
  bool in_range = (jF >= nsMinF) && (jF < jF_upper);

  if (!in_range || jF < jMin_value) {
    d_aR[out_idx] = 0.0; d_aZ[out_idx] = 0.0;
    d_dR[out_idx] = 0.0; d_dZ[out_idx] = 0.0;
    d_bR[out_idx] = 0.0; d_bZ[out_idx] = 0.0;
    return;
  }

  size_t jF_local = (size_t)jF - (size_t)nsMinF;

  // sup-diagonal: half-grid pos OUTSIDE jF (jH = jF), only if jF < nsMaxH.
  double a_R = 0.0, a_Z = 0.0;
  if (jF < nsMaxH) {
    int jH_o = jF - nsMinH;
    a_R = -(d_arm[cfg_half_prof + jH_o * 2 + m_parity] +
            d_brm[cfg_half_prof + jH_o * 2 + m_parity] * m * m);
    a_Z = -(d_azm[cfg_half_prof + jH_o * 2 + m_parity] +
            d_bzm[cfg_half_prof + jH_o * 2 + m_parity] * m * m);
  }

  // The diagonal in the CPU's evaluation order, ((((cxd * n) * nfp) * n) *
  // nfp), for identical rounding.
  double d_R = -(d_ard[cfg_force_prof + jF_local * 2 + m_parity]
                 + d_brd[cfg_force_prof + jF_local * 2 + m_parity] * m * m
                 + d_cxd[cfg_cxd + jF_local] * n * nfp * n * nfp);
  double d_Z = -(d_azd[cfg_force_prof + jF_local * 2 + m_parity]
                 + d_bzd[cfg_force_prof + jF_local * 2 + m_parity] * m * m
                 + d_cxd[cfg_cxd + jF_local] * n * nfp * n * nfp);

  // sub-diagonal: half-grid pos INSIDE jF (jH = jF-1), only if jF > 0.
  double b_R = 0.0, b_Z = 0.0;
  if (jF > 0) {
    int jH_i = jF - 1 - nsMinH;
    b_R = -(d_arm[cfg_half_prof + jH_i * 2 + m_parity] +
            d_brm[cfg_half_prof + jH_i * 2 + m_parity] * m * m);
    b_Z = -(d_azm[cfg_half_prof + jH_i * 2 + m_parity] +
            d_bzm[cfg_half_prof + jH_i * 2 + m_parity] * m * m);
  }

  // Special: m=1 at jF=1 => dr += br, dz += bz.
  if (jF == 1 && m == 1) {
    d_R += b_R;
    d_Z += b_Z;
  }

  // Edge pedestal and ZC_00 stabilization at the LCFS row, which fixed boundary
  // never reaches, as on the CPU.
  if (lcfs_owning && jF == ns_total - 1) {
    double pedestal_mult = (m <= 1) ? (1.0 + edge_pedestal) : (1.0 + 2.0 * edge_pedestal);
    d_R *= pedestal_mult;
    d_Z *= pedestal_mult;
    if (m == 0 && n == 0) {
      d_Z *= (1.0 - mult_fact_zc00) / (1.0 + edge_pedestal);
    }
  }

  d_aR[out_idx] = a_R;
  d_aZ[out_idx] = a_Z;
  d_dR[out_idx] = d_R;
  d_dZ[out_idx] = d_Z;
  d_bR[out_idx] = b_R;
  d_bZ[out_idx] = b_Z;
}

}  // namespace vmecpp

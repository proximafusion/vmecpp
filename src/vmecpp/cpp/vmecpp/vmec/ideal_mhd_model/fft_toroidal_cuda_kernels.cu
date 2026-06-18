#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda_common.cuh"

namespace vmecpp {
// =========================================================================
// CUDA kernels
// =========================================================================

// k_fill_spectra assigns one thread to each (jF_local, m, q, n) tuple
// in the index ranges jF_local in [0, ns_local), m in [0, mpol),
// q in [0, kBatch = 12), and n in [0, nhalf). Each thread populates
// the corresponding complex entry of the cuFFT input buffer X from
// the spectral coefficients rmncc, rmnss, zmnsc, zmncs, lmnsc, and
// lmncs, multiplied by the toroidal mode-scaling vector nscale and
// the per-mode poloidal multiplier xmpq as appropriate to the q
// channel.
//
// The per-poloidal-mode jMin floor is enforced explicitly: for
// m in {0, 1} the floor is zero, and for m >= 2 the floor is one,
// so that the spectral entries with jF below the floor are emitted
// as zero rather than read from the spectral arrays.
//
// The output buffer X is laid out contiguously over
// (ns_local * mpol * kBatch * nhalf) complex doubles with
// [jF_local][m][q][n] order, which is the layout the cuFFT batched
// plan expects.
//
// The configuration axis is carried on blockIdx.z, encoded as
// config * ns_local + jF_local; at n_config equal to one this
// reduces to jF_local alone and the layout collapses to the pre-
// batched single-configuration arrangement.
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

// k_scatter_main consumes the post-inverse-FFT real-space tensor
// Y[jF, m, q, k] and accumulates its m-summed contributions into the
// sixteen even-and-odd real-space output arrays
//   r1_e, r1_o, ru_e, ru_o, rv_e, rv_o,
//   z1_e, z1_o, zu_e, zu_o, zv_e, zv_o,
//   lu_e, lu_o, lv_e, lv_o,
// each carrying the contribution from the corresponding poloidal
// channel and parity. Threads are mapped to the output index triple
// (l, k, jF_local), and each thread accumulates the contributions
// from every poloidal mode m, dispatching the resulting contribution
// to the even or odd output array of each output family according
// to the parity of m. Writing the even and odd outputs from the
// same thread avoids any read-modify-write race that splitting the
// parity into separate kernels would otherwise introduce. The
// configuration axis is carried on blockIdx.z encoded as
// config * ns_local + jF_local.
__global__ void k_scatter_main(
    int n_config, int ns_local, int mpol, int nZeta, int nThetaReduced, int nThetaEff,
    const double* __restrict__ Y, const double* __restrict__ cosmu,
    const double* __restrict__ sinmu, const double* __restrict__ cosmum,
    const double* __restrict__ sinmum,
    double* __restrict__ r1_e, double* __restrict__ r1_o,
    double* __restrict__ ru_e, double* __restrict__ ru_o,
    double* __restrict__ rv_e, double* __restrict__ rv_o,
    double* __restrict__ z1_e, double* __restrict__ z1_o,
    double* __restrict__ zu_e, double* __restrict__ zu_o,
    double* __restrict__ zv_e, double* __restrict__ zv_o,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o) {
  int config = blockIdx.z / ns_local;
  int jF_local = blockIdx.z - config * ns_local;
  if (config >= n_config) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= nThetaReduced) return;
  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;

  double r1e_acc = 0.0, r1o_acc = 0.0;
  double rue_acc = 0.0, ruo_acc = 0.0;
  double rve_acc = 0.0, rvo_acc = 0.0;
  double z1e_acc = 0.0, z1o_acc = 0.0;
  double zue_acc = 0.0, zuo_acc = 0.0;
  double zve_acc = 0.0, zvo_acc = 0.0;
  double lue_acc = 0.0, luo_acc = 0.0;
  double lve_acc = 0.0, lvo_acc = 0.0;

  for (int m = 0; m < mpol; ++m) {
    // Per-configuration indexing: cfg_Y is the byte/double-offset of config's Y block;
    // the inner (y_base + kxxx) * nZeta + k formula encodes the local Y index
    // within one config. Add cfg_Y OUTSIDE the * nZeta multiplication so per-
    // config offset isn't accidentally scaled.
    const size_t y_base_local = (size_t)((jF_local * mpol + m) * kBatch);
    double rmkcc   = Y[cfg_Y + (y_base_local + kRmkcc)   * nZeta + k];
    double rmkss   = Y[cfg_Y + (y_base_local + kRmkss)   * nZeta + k];
    double rmkccN  = Y[cfg_Y + (y_base_local + kRmkccN)  * nZeta + k];
    double rmkssN  = Y[cfg_Y + (y_base_local + kRmkssN)  * nZeta + k];
    double zmksc   = Y[cfg_Y + (y_base_local + kZmksc)   * nZeta + k];
    double zmkcs   = Y[cfg_Y + (y_base_local + kZmkcs)   * nZeta + k];
    double zmkscN  = Y[cfg_Y + (y_base_local + kZmkscN)  * nZeta + k];
    double zmkcsN  = Y[cfg_Y + (y_base_local + kZmkcsN)  * nZeta + k];
    double lmksc   = Y[cfg_Y + (y_base_local + kLmksc)   * nZeta + k];
    double lmkcs   = Y[cfg_Y + (y_base_local + kLmkcs)   * nZeta + k];
    double lmkscN  = Y[cfg_Y + (y_base_local + kLmkscN)  * nZeta + k];
    double lmkcsN  = Y[cfg_Y + (y_base_local + kLmkcsN)  * nZeta + k];

    int bml = m * nThetaReduced + l;
    double cmu  = cosmu[bml];
    double smu  = sinmu[bml];
    double cmum = cosmum[bml];
    double smum = sinmum[bml];

    bool m_even = ((m & 1) == 0);

    // r1 += rmkcc*cosmu + rmkss*sinmu
    double r1_contrib = rmkcc * cmu + rmkss * smu;
    // ru += rmkcc*sinmum + rmkss*cosmum
    double ru_contrib = rmkcc * smum + rmkss * cmum;
    // rv += rmkccN*cosmu + rmkssN*sinmu
    double rv_contrib = rmkccN * cmu + rmkssN * smu;
    // z1 += zmksc*sinmu + zmkcs*cosmu
    double z1_contrib = zmksc * smu + zmkcs * cmu;
    // zu += zmksc*cosmum + zmkcs*sinmum
    double zu_contrib = zmksc * cmum + zmkcs * smum;
    // zv += zmkscN*sinmu + zmkcsN*cosmu
    double zv_contrib = zmkscN * smu + zmkcsN * cmu;
    // lu += lmksc*cosmum + lmkcs*sinmum
    double lu_contrib = lmksc * cmum + lmkcs * smum;
    // lv -= lmkscN*sinmu + lmkcsN*cosmu  (NOTE: subtract!)
    double lv_contrib = -(lmkscN * smu + lmkcsN * cmu);

    if (m_even) {
      r1e_acc += r1_contrib;
      rue_acc += ru_contrib;
      rve_acc += rv_contrib;
      z1e_acc += z1_contrib;
      zue_acc += zu_contrib;
      zve_acc += zv_contrib;
      lue_acc += lu_contrib;
      lve_acc += lv_contrib;
    } else {
      r1o_acc += r1_contrib;
      ruo_acc += ru_contrib;
      rvo_acc += rv_contrib;
      z1o_acc += z1_contrib;
      zuo_acc += zu_contrib;
      zvo_acc += zv_contrib;
      luo_acc += lu_contrib;
      lvo_acc += lv_contrib;
    }
  }

  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] += r1e_acc; r1_o[idx] += r1o_acc;
  ru_e[idx] += rue_acc; ru_o[idx] += ruo_acc;
  rv_e[idx] += rve_acc; rv_o[idx] += rvo_acc;
  z1_e[idx] += z1e_acc; z1_o[idx] += z1o_acc;
  zu_e[idx] += zue_acc; zu_o[idx] += zuo_acc;
  zv_e[idx] += zve_acc; zv_o[idx] += zvo_acc;
  lu_e[idx] += lue_acc; lu_o[idx] += luo_acc;
  lv_e[idx] += lve_acc; lv_o[idx] += lvo_acc;
}

// Warp-cooperative full-pipeline fused forward DFT and scatter.
//
// The kernel collapses the three stages of the spectrum-to-real-space chain
// (k_fill_spectra, the cuFFT toroidal transform, and k_scatter_main_and_con)
// into a single launch and avoids the materialization of the intermediate
// X and Y buffers in global memory. Each block consists of a single warp
// of thirty-two threads operating at a fixed (config, jF_local, k) tuple,
// and the thread index t within the warp serves a dual role across two
// computational stages:
//
//   First stage. The thread index t is interpreted as the toroidal mode
//   index n. For t in [0, ntor] the thread loads the six spectral
//   coefficients at (jF, m, n = t) and computes its contribution to the
//   twelve q-channel partial sums (rmkcc, rmkss, rmkccN, rmkssN, zmksc,
//   zmkcs, zmkscN, zmkcsN, lmksc, lmkcs, lmkscN, lmkcsN). Threads with
//   t > ntor contribute zero. A warp-wide __shfl_xor_sync butterfly
//   reduction then propagates the fully accumulated q-values to every
//   lane.
//
//   Second stage. The thread index t is reinterpreted as the poloidal
//   point index l. Threads with t < nThetaReduced compute the scatter
//   contributions to the sixteen even-parity and odd-parity outputs
//   (r1_e/o, ru_e/o, rv_e/o, z1_e/o, zu_e/o, zv_e/o, lu_e/o, lv_e/o) and
//   the two constraint outputs (rCon, zCon) for their (k, l) position.
//
// The warp-cooperative arrangement removes the redundant evaluation of the
// toroidal-mode sum that the per-thread fused variants required, while
// retaining the elimination of the d_X and d_Y intermediates. The kernel
// assumes single-rank operation, that is, ns_con_local equals ns_local and
// the offset nsMinF_offset_in_local is zero, so that jF_con coincides with
// jF_local within the block's coordinate scheme.
__global__ void k_fwd_fused_warp(
    int n_config, int ns_local, int mpol, int ntor, int nfp,
    int nZeta, int nThetaReduced, int nThetaEff, int nsMinF1,
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ dft_cos, const double* __restrict__ dft_sin,
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
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
  int config = blockIdx.z / ns_local;
  int jF_local = blockIdx.z - config * ns_local;
  if (config >= n_config) return;
  int k = blockIdx.y;
  int tid = threadIdx.x;

  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  int jF_global = jF_local + nsMinF1;

  double r1e_acc = 0.0, r1o_acc = 0.0;
  double rue_acc = 0.0, ruo_acc = 0.0;
  double rve_acc = 0.0, rvo_acc = 0.0;
  double z1e_acc = 0.0, z1o_acc = 0.0;
  double zue_acc = 0.0, zuo_acc = 0.0;
  double zve_acc = 0.0, zvo_acc = 0.0;
  double lue_acc = 0.0, luo_acc = 0.0;
  double lve_acc = 0.0, lvo_acc = 0.0;
  double rcon_acc = 0.0, zcon_acc = 0.0;

  for (int m = 0; m < mpol; ++m) {
    int jMin_for_m = (m == 0 || m == 1) ? 0 : 1;
    if (jF_global < jMin_for_m) continue;

    // Phase 1: each warp lane computes the n=tid contribution to the 12
    // q-outputs. Lanes with tid > ntor contribute 0.
    double rmkcc = 0.0, rmkss = 0.0, rmkccN = 0.0, rmkssN = 0.0;
    double zmksc = 0.0, zmkcs = 0.0, zmkscN = 0.0, zmkcsN = 0.0;
    double lmksc = 0.0, lmkcs = 0.0, lmkscN = 0.0, lmkcsN = 0.0;
    if (tid <= ntor) {
      int n = tid;
      double cos_nk = dft_cos[n * nZeta + k];
      double sin_nk = dft_sin[n * nZeta + k];
      double n_nfp = (double)n * (double)nfp;
      size_t spec_base_m = cfg_spec + (size_t)((jF_local * mpol + m) * (ntor + 1));
      double rcc = rmncc[spec_base_m + n];
      double zsc = zmnsc[spec_base_m + n];
      double lsc = lmnsc[spec_base_m + n];
      double rss = (n == 0) ? 0.0 : rmnss[spec_base_m + n];
      double zcs = (n == 0) ? 0.0 : zmncs[spec_base_m + n];
      double lcs = (n == 0) ? 0.0 : lmncs[spec_base_m + n];
      rmkcc = rcc * cos_nk;
      zmksc = zsc * cos_nk;
      lmksc = lsc * cos_nk;
      rmkss = rss * sin_nk;
      zmkcs = zcs * sin_nk;
      lmkcs = lcs * sin_nk;
      rmkccN = (-rcc * n_nfp) * sin_nk;
      rmkssN = ( rss * n_nfp) * cos_nk;
      zmkscN = (-zsc * n_nfp) * sin_nk;
      zmkcsN = ( zcs * n_nfp) * cos_nk;
      lmkscN = (-lsc * n_nfp) * sin_nk;
      lmkcsN = ( lcs * n_nfp) * cos_nk;
    }
    // Butterfly reduction across the warp. After the loop every lane holds
    // the same fully-summed value.
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      rmkcc  += __shfl_xor_sync(0xffffffff, rmkcc,  s);
      rmkss  += __shfl_xor_sync(0xffffffff, rmkss,  s);
      rmkccN += __shfl_xor_sync(0xffffffff, rmkccN, s);
      rmkssN += __shfl_xor_sync(0xffffffff, rmkssN, s);
      zmksc  += __shfl_xor_sync(0xffffffff, zmksc,  s);
      zmkcs  += __shfl_xor_sync(0xffffffff, zmkcs,  s);
      zmkscN += __shfl_xor_sync(0xffffffff, zmkscN, s);
      zmkcsN += __shfl_xor_sync(0xffffffff, zmkcsN, s);
      lmksc  += __shfl_xor_sync(0xffffffff, lmksc,  s);
      lmkcs  += __shfl_xor_sync(0xffffffff, lmkcs,  s);
      lmkscN += __shfl_xor_sync(0xffffffff, lmkscN, s);
      lmkcsN += __shfl_xor_sync(0xffffffff, lmkcsN, s);
    }

    // Phase 2: lanes with tid < nThetaReduced use tid as the poloidal index
    // l and accumulate their scatter contributions.
    if (tid < nThetaReduced) {
      int bml = m * nThetaReduced + tid;
      double cmu  = cosmu[bml];
      double smu  = sinmu[bml];
      double cmum = cosmum[bml];
      double smum = sinmum[bml];
      bool m_even = ((m & 1) == 0);
      double r1_contrib = rmkcc  * cmu  + rmkss  * smu;
      double ru_contrib = rmkcc  * smum + rmkss  * cmum;
      double rv_contrib = rmkccN * cmu  + rmkssN * smu;
      double z1_contrib = zmksc  * smu  + zmkcs  * cmu;
      double zu_contrib = zmksc  * cmum + zmkcs  * smum;
      double zv_contrib = zmkscN * smu  + zmkcsN * cmu;
      double lu_contrib = lmksc  * cmum + lmkcs  * smum;
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
      double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF[jF_local];
      rcon_acc += r1_contrib * con_factor;
      zcon_acc += z1_contrib * con_factor;
    }
  }

  if (tid < nThetaReduced) {
    size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + tid);
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
}

// Output-group-partitioned forward-FFT fusion kernels.
//
// The three kernels k_fwd_fused_R, k_fwd_fused_Z, and k_fwd_fused_L
// implement the same spectrum-to-real-space transformation as the combined
// k_forward_fft_fused kernel below, but partition the eighteen output
// components across three launches according to their physical role: the
// R-side kernel writes r1, ru, rv, and rCon; the Z-side kernel writes z1,
// zu, zv, and zCon; the lambda-side kernel writes lu and lv. Partitioning
// the work reduces the per-thread register pressure: where the combined
// kernel must hold the full set of accumulator doubles per thread and
// spills to local memory on the target architecture, the partitioned
// kernels carry seven, seven, and four accumulators respectively, all
// within the available register file.
//
// Each kernel evaluates its inverse toroidal discrete Fourier transform
// inline, summing over the toroidal mode index n with the nscale-folded
// dft_cos and dft_sin lookup tables. This preserves the elimination of
// the d_X and d_Y intermediate buffers that the combined fused kernel
// achieves; the partitioning amounts to processing the eighteen output
// components in three sequential launches rather than in a single launch.
//
// Each kernel reads its associated pair of spectral coefficient arrays
// (rmncc and rmnss for the R-side, zmnsc and zmncs for the Z-side, and
// lmnsc and lmncs for the lambda-side). The per-configuration spec slot
// for a given (cfg, jF, m) is small enough to remain resident in the L1
// cache after the first warp loads it, so the re-read cost across the
// three launches is bounded by L1 bandwidth rather than DRAM bandwidth.
__global__ void k_fwd_fused_R(
    int n_config, int ns_local, int mpol, int ntor, int nfp,
    int nZeta, int nThetaReduced, int nThetaEff, int nsMinF1,
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ dft_cos, const double* __restrict__ dft_sin,
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
    const double* __restrict__ xmpq, const double* __restrict__ sqrtSF,
    double* __restrict__ r1_e, double* __restrict__ r1_o,
    double* __restrict__ ru_e, double* __restrict__ ru_o,
    double* __restrict__ rv_e, double* __restrict__ rv_o,
    double* __restrict__ rCon) {
  int config = blockIdx.z / ns_local;
  int jF_local = blockIdx.z - config * ns_local;
  if (config >= n_config) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= nThetaReduced) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  int jF_global = jF_local + nsMinF1;

  double r1e_acc = 0.0, r1o_acc = 0.0;
  double rue_acc = 0.0, ruo_acc = 0.0;
  double rve_acc = 0.0, rvo_acc = 0.0;
  double rcon_acc = 0.0;

  for (int m = 0; m < mpol; ++m) {
    int jMin_for_m = (m == 0 || m == 1) ? 0 : 1;
    if (jF_global < jMin_for_m) continue;
    double rmkcc = 0.0, rmkss = 0.0, rmkccN = 0.0, rmkssN = 0.0;
    size_t spec_base_m = cfg_spec + (size_t)((jF_local * mpol + m) * (ntor + 1));
    for (int n = 0; n <= ntor; ++n) {
      double cos_nk = dft_cos[n * nZeta + k];
      double sin_nk = dft_sin[n * nZeta + k];
      double n_nfp = (double)n * (double)nfp;
      double rcc = rmncc[spec_base_m + n];
      double rss = (n == 0) ? 0.0 : rmnss[spec_base_m + n];
      rmkcc  += rcc * cos_nk;
      rmkss  += rss * sin_nk;
      rmkccN += (-rcc * n_nfp) * sin_nk;
      rmkssN += ( rss * n_nfp) * cos_nk;
    }
    int bml = m * nThetaReduced + l;
    double cmu  = cosmu[bml];
    double smu  = sinmu[bml];
    double cmum = cosmum[bml];
    double smum = sinmum[bml];
    bool m_even = ((m & 1) == 0);
    double r1_contrib = rmkcc  * cmu  + rmkss  * smu;
    double ru_contrib = rmkcc  * smum + rmkss  * cmum;
    double rv_contrib = rmkccN * cmu  + rmkssN * smu;
    if (m_even) {
      r1e_acc += r1_contrib; rue_acc += ru_contrib; rve_acc += rv_contrib;
    } else {
      r1o_acc += r1_contrib; ruo_acc += ru_contrib; rvo_acc += rv_contrib;
    }
    double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF[jF_local];
    rcon_acc += r1_contrib * con_factor;
  }
  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] += r1e_acc; r1_o[idx] += r1o_acc;
  ru_e[idx] += rue_acc; ru_o[idx] += ruo_acc;
  rv_e[idx] += rve_acc; rv_o[idx] += rvo_acc;
  rCon[idx] += rcon_acc;
}

__global__ void k_fwd_fused_Z(
    int n_config, int ns_local, int mpol, int ntor, int nfp,
    int nZeta, int nThetaReduced, int nThetaEff, int nsMinF1,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ dft_cos, const double* __restrict__ dft_sin,
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
    const double* __restrict__ xmpq, const double* __restrict__ sqrtSF,
    double* __restrict__ z1_e, double* __restrict__ z1_o,
    double* __restrict__ zu_e, double* __restrict__ zu_o,
    double* __restrict__ zv_e, double* __restrict__ zv_o,
    double* __restrict__ zCon) {
  int config = blockIdx.z / ns_local;
  int jF_local = blockIdx.z - config * ns_local;
  if (config >= n_config) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= nThetaReduced) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  int jF_global = jF_local + nsMinF1;

  double z1e_acc = 0.0, z1o_acc = 0.0;
  double zue_acc = 0.0, zuo_acc = 0.0;
  double zve_acc = 0.0, zvo_acc = 0.0;
  double zcon_acc = 0.0;

  for (int m = 0; m < mpol; ++m) {
    int jMin_for_m = (m == 0 || m == 1) ? 0 : 1;
    if (jF_global < jMin_for_m) continue;
    double zmksc = 0.0, zmkcs = 0.0, zmkscN = 0.0, zmkcsN = 0.0;
    size_t spec_base_m = cfg_spec + (size_t)((jF_local * mpol + m) * (ntor + 1));
    for (int n = 0; n <= ntor; ++n) {
      double cos_nk = dft_cos[n * nZeta + k];
      double sin_nk = dft_sin[n * nZeta + k];
      double n_nfp = (double)n * (double)nfp;
      double zsc = zmnsc[spec_base_m + n];
      double zcs = (n == 0) ? 0.0 : zmncs[spec_base_m + n];
      zmksc  += zsc * cos_nk;
      zmkcs  += zcs * sin_nk;
      zmkscN += (-zsc * n_nfp) * sin_nk;
      zmkcsN += ( zcs * n_nfp) * cos_nk;
    }
    int bml = m * nThetaReduced + l;
    double cmu  = cosmu[bml];
    double smu  = sinmu[bml];
    double cmum = cosmum[bml];
    double smum = sinmum[bml];
    bool m_even = ((m & 1) == 0);
    double z1_contrib = zmksc  * smu  + zmkcs  * cmu;
    double zu_contrib = zmksc  * cmum + zmkcs  * smum;
    double zv_contrib = zmkscN * smu  + zmkcsN * cmu;
    if (m_even) {
      z1e_acc += z1_contrib; zue_acc += zu_contrib; zve_acc += zv_contrib;
    } else {
      z1o_acc += z1_contrib; zuo_acc += zu_contrib; zvo_acc += zv_contrib;
    }
    double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF[jF_local];
    zcon_acc += z1_contrib * con_factor;
  }
  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  z1_e[idx] += z1e_acc; z1_o[idx] += z1o_acc;
  zu_e[idx] += zue_acc; zu_o[idx] += zuo_acc;
  zv_e[idx] += zve_acc; zv_o[idx] += zvo_acc;
  zCon[idx] += zcon_acc;
}

__global__ void k_fwd_fused_L(
    int n_config, int ns_local, int mpol, int ntor, int nfp,
    int nZeta, int nThetaReduced, int nThetaEff, int nsMinF1,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ dft_cos, const double* __restrict__ dft_sin,
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o) {
  int config = blockIdx.z / ns_local;
  int jF_local = blockIdx.z - config * ns_local;
  if (config >= n_config) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= nThetaReduced) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  int jF_global = jF_local + nsMinF1;

  double lue_acc = 0.0, luo_acc = 0.0;
  double lve_acc = 0.0, lvo_acc = 0.0;

  for (int m = 0; m < mpol; ++m) {
    int jMin_for_m = (m == 0 || m == 1) ? 0 : 1;
    if (jF_global < jMin_for_m) continue;
    double lmksc = 0.0, lmkcs = 0.0, lmkscN = 0.0, lmkcsN = 0.0;
    size_t spec_base_m = cfg_spec + (size_t)((jF_local * mpol + m) * (ntor + 1));
    for (int n = 0; n <= ntor; ++n) {
      double cos_nk = dft_cos[n * nZeta + k];
      double sin_nk = dft_sin[n * nZeta + k];
      double n_nfp = (double)n * (double)nfp;
      double lsc = lmnsc[spec_base_m + n];
      double lcs = (n == 0) ? 0.0 : lmncs[spec_base_m + n];
      lmksc  += lsc * cos_nk;
      lmkcs  += lcs * sin_nk;
      lmkscN += (-lsc * n_nfp) * sin_nk;
      lmkcsN += ( lcs * n_nfp) * cos_nk;
    }
    int bml = m * nThetaReduced + l;
    double cmu  = cosmu[bml];
    double smu  = sinmu[bml];
    double cmum = cosmum[bml];
    double smum = sinmum[bml];
    bool m_even = ((m & 1) == 0);
    double lu_contrib = lmksc  * cmum + lmkcs  * smum;
    double lv_contrib = -(lmkscN * smu + lmkcsN * cmu);
    if (m_even) {
      lue_acc += lu_contrib; lve_acc += lv_contrib;
    } else {
      luo_acc += lu_contrib; lvo_acc += lv_contrib;
    }
  }
  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  lu_e[idx] += lue_acc; lu_o[idx] += luo_acc;
  lv_e[idx] += lve_acc; lv_o[idx] += lvo_acc;
}

// k_forward_fft_fused: replaces the entire forward FFT chain
// (k_fill_spectra → cuFFT batched Z2D → k_scatter_main_and_con) with one
// kernel that goes directly spec → final real-space outputs. Per (cfg, jF, k, l)
// thread:
//   1. For each m, sum over n=0..ntor to inline the inverse toroidal DFT:
//      rmkcc[k] = sum_n rmncc[m,n] * nscale[n] * cos(2*pi*n*k/nZeta)
//      rmkss[k] = sum_{n>=1} rmnss[m,n] * nscale[n] * sin(2*pi*n*k/nZeta)
//      rmkccN[k] = -sum_{n>=1} rmncc[m,n] * n * nfp * nscale[n] * sin(...)
//      rmkssN[k] = sum_{n>=1} rmnss[m,n] * n * nfp * nscale[n] * cos(...)
//      ... 12 q-outputs total ...
//   2. Apply scatter math with cosmu/sinmu/cosmum/sinmum to compute the
//      m-summed (r1_e/o, ru_e/o, rv_e/o, z1_e/o, zu_e/o, zv_e/o, lu_e/o, lv_e/o)
//      and (rCon, zCon) contributions.
//
// Single-rank assumption: ns_con_local == ns_local and nsMinF_offset_in_local
// == 0, so jF_con == jF_local and the con outputs share the main idx layout.
//
// Replaces ~214 MB/call of d_X+d_Y intermediate memory traffic and 4 kernel
// launches (fill, cuFFT exec, scatter, plus the cuFFT plan-side bookkeeping).
__global__ void k_forward_fft_fused(
    int n_config, int ns_local, int mpol, int ntor, int nfp,
    int nZeta, int nThetaReduced, int nThetaEff, int nsMinF1,
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ dft_cos,  // [ntor+1, nZeta], ns_n folded in
    const double* __restrict__ dft_sin,  // [ntor+1, nZeta], ns_n folded in
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
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
  int config = blockIdx.z / ns_local;
  int jF_local = blockIdx.z - config * ns_local;
  if (config >= n_config) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= nThetaReduced) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  int jF_global = jF_local + nsMinF1;

  double r1e_acc = 0.0, r1o_acc = 0.0;
  double rue_acc = 0.0, ruo_acc = 0.0;
  double rve_acc = 0.0, rvo_acc = 0.0;
  double z1e_acc = 0.0, z1o_acc = 0.0;
  double zue_acc = 0.0, zuo_acc = 0.0;
  double zve_acc = 0.0, zvo_acc = 0.0;
  double lue_acc = 0.0, luo_acc = 0.0;
  double lve_acc = 0.0, lvo_acc = 0.0;
  double rcon_acc = 0.0, zcon_acc = 0.0;

  for (int m = 0; m < mpol; ++m) {
    int jMin_for_m = (m == 0 || m == 1) ? 0 : 1;
    if (jF_global < jMin_for_m) continue;

    // Inverse toroidal DFT: 12 q-output values for this (jF, m, k).
    double rmkcc = 0.0, rmkss = 0.0, rmkccN = 0.0, rmkssN = 0.0;
    double zmksc = 0.0, zmkcs = 0.0, zmkscN = 0.0, zmkcsN = 0.0;
    double lmksc = 0.0, lmkcs = 0.0, lmkscN = 0.0, lmkcsN = 0.0;
    size_t spec_base_m = cfg_spec + (size_t)((jF_local * mpol + m) * (ntor + 1));

    for (int n = 0; n <= ntor; ++n) {
      double cos_nk = dft_cos[n * nZeta + k];
      double sin_nk = dft_sin[n * nZeta + k];
      double n_nfp = (double)n * (double)nfp;

      double rcc = rmncc[spec_base_m + n];
      double zsc = zmnsc[spec_base_m + n];
      double lsc = lmnsc[spec_base_m + n];
      // n=0 has no contribution to *_ss / *_cs / *_N variants.
      double rss = (n == 0) ? 0.0 : rmnss[spec_base_m + n];
      double zcs = (n == 0) ? 0.0 : zmncs[spec_base_m + n];
      double lcs = (n == 0) ? 0.0 : lmncs[spec_base_m + n];

      rmkcc  += rcc * cos_nk;
      zmksc  += zsc * cos_nk;
      lmksc  += lsc * cos_nk;

      rmkss  += rss * sin_nk;
      zmkcs  += zcs * sin_nk;
      lmkcs  += lcs * sin_nk;

      rmkccN += (-rcc * n_nfp) * sin_nk;
      rmkssN += ( rss * n_nfp) * cos_nk;
      zmkscN += (-zsc * n_nfp) * sin_nk;
      zmkcsN += ( zcs * n_nfp) * cos_nk;
      lmkscN += (-lsc * n_nfp) * sin_nk;
      lmkcsN += ( lcs * n_nfp) * cos_nk;
    }

    int bml = m * nThetaReduced + l;
    double cmu  = cosmu[bml];
    double smu  = sinmu[bml];
    double cmum = cosmum[bml];
    double smum = sinmum[bml];
    bool m_even = ((m & 1) == 0);

    double r1_contrib = rmkcc  * cmu  + rmkss  * smu;
    double ru_contrib = rmkcc  * smum + rmkss  * cmum;
    double rv_contrib = rmkccN * cmu  + rmkssN * smu;
    double z1_contrib = zmksc  * smu  + zmkcs  * cmu;
    double zu_contrib = zmksc  * cmum + zmkcs  * smum;
    double zv_contrib = zmkscN * smu  + zmkcsN * cmu;
    double lu_contrib = lmksc  * cmum + lmkcs  * smum;
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

    double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF[jF_local];
    rcon_acc += r1_contrib * con_factor;
    zcon_acc += z1_contrib * con_factor;
  }

  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] += r1e_acc; r1_o[idx] += r1o_acc;
  ru_e[idx] += rue_acc; ru_o[idx] += ruo_acc;
  rv_e[idx] += rve_acc; rv_o[idx] += rvo_acc;
  z1_e[idx] += z1e_acc; z1_o[idx] += z1o_acc;
  zu_e[idx] += zue_acc; zu_o[idx] += zuo_acc;
  zv_e[idx] += zve_acc; zv_o[idx] += zvo_acc;
  lu_e[idx] += lue_acc; lu_o[idx] += luo_acc;
  lv_e[idx] += lve_acc; lv_o[idx] += lvo_acc;
  rCon[idx] += rcon_acc;
  zCon[idx] += zcon_acc;
}

// Hand-coded length-24 inverse complex discrete Fourier transform.
//
// The kernel implements a Cooley-Tukey radix-8x3 decomposition of the
// length-24 inverse transform and is sized for the exact shape used by
// the iteration body: a Hermitian-symmetric complex input of length
// nhalf = 13 and a real output of length nZeta = 24. The toroidal mode
// truncation ntor = 10 leaves X[11] and X[12] zero in practice; the
// kernel does not exploit that fact, so the routine remains correct if
// the truncation is relaxed.
//
// Each block executes one batch of the length-24 transform. The block
// is dispatched with thirty-two threads of which twenty-four are
// productive; the remaining eight remain idle for the duration of the
// kernel. Each productive thread t corresponds to one (n2, k1) pair,
// where n2 = t / 8 indexes the length-3 substage and k1 = t mod 8
// indexes the length-8 substage. The three computational stages
// proceed as follows:
//
//   Stage one performs a length-8 inverse complex discrete Fourier
//   transform along the n1 axis at fixed n2. Each thread evaluates
//   F_{n2}[k1] = sum_{n1=0..7} X[3 n1 + n2] * w_8^{n1 k1} for its
//   (n2, k1). Indices 3 n1 + n2 that exceed nhalf - 1 are synthesized
//   from Hermitian symmetry through X[m] = conj(X[nZeta - m]) for m in
//   [nhalf, nZeta - 1]; X[12] is real-valued.
//
//   Stage two applies the radix-8x3 twiddle factor T_{n2}[k1] =
//   F_{n2}[k1] * w_24^{n2 k1}.
//
//   Stage three performs the length-3 inverse complex discrete Fourier
//   transform along the n2 axis at fixed k1. Each thread reads the
//   three values T_{n2}[k1] for its k1 from shared memory and writes
//   one output element Y[k1 + 8 k2] = sum_{n2=0..2} T_{n2}[k1] *
//   w_3^{n2 k2} corresponding to its k2 = t / 8. The mapping from the
//   thread index to (k1, k2) is t -> (t mod 8, t / 8), covering all
//   twenty-four output positions exactly once.
//
// Because the input is Hermitian-symmetric, the imaginary component of
// the length-3 result is zero up to floating-point error; only the
// real part is written to Y[k1 + 8 k2]. The twiddle sign convention
// matches cufftExecZ2D: the inverse direction uses a positive
// imaginary part for the complex exponentials, and no 1/N scaling is
// applied.
// Precomputed twiddle tables for the length-24 inverse transform.
// w_8^p for p in [0, 8), used in stage 1.
// w_24^p for p in [0, 24), used in stage 2 twiddle.
// All inverse-direction (positive imaginary sign).
__device__ static const double kRadix8_cos[8] = {
   1.0,
   0.70710678118654752440,
   0.0,
  -0.70710678118654752440,
  -1.0,
  -0.70710678118654752440,
   0.0,
   0.70710678118654752440
};
__device__ static const double kRadix8_sin[8] = {
   0.0,
   0.70710678118654752440,
   1.0,
   0.70710678118654752440,
   0.0,
  -0.70710678118654752440,
  -1.0,
  -0.70710678118654752440
};
__device__ static const double kRadix24_cos[24] = {
   1.0,                       // 0  = 0
   0.96592582628906828675,    // 1  = 15deg
   0.86602540378443864676,    // 2  = 30deg
   0.70710678118654752440,    // 3  = 45deg
   0.50000000000000000000,    // 4  = 60deg
   0.25881904510252076235,    // 5  = 75deg
   0.0,                       // 6  = 90deg
  -0.25881904510252076235,    // 7  = 105
  -0.50000000000000000000,    // 8  = 120
  -0.70710678118654752440,    // 9  = 135
  -0.86602540378443864676,    // 10 = 150
  -0.96592582628906828675,    // 11 = 165
  -1.0,                       // 12 = 180
  -0.96592582628906828675,    // 13
  -0.86602540378443864676,    // 14
  -0.70710678118654752440,    // 15
  -0.50000000000000000000,    // 16
  -0.25881904510252076235,    // 17
   0.0,                       // 18
   0.25881904510252076235,    // 19
   0.50000000000000000000,    // 20
   0.70710678118654752440,    // 21
   0.86602540378443864676,    // 22
   0.96592582628906828675     // 23
};
__device__ static const double kRadix24_sin[24] = {
   0.0,
   0.25881904510252076235,
   0.50000000000000000000,
   0.70710678118654752440,
   0.86602540378443864676,
   0.96592582628906828675,
   1.0,
   0.96592582628906828675,
   0.86602540378443864676,
   0.70710678118654752440,
   0.50000000000000000000,
   0.25881904510252076235,
   0.0,
  -0.25881904510252076235,
  -0.50000000000000000000,
  -0.70710678118654752440,
  -0.86602540378443864676,
  -0.96592582628906828675,
  -1.0,
  -0.96592582628906828675,
  -0.86602540378443864676,
  -0.70710678118654752440,
  -0.50000000000000000000,
  -0.25881904510252076235
};

__global__ void k_inverse_dft_24_radix83(
    int total_batches, int nhalf, int nZeta,
    const cufftDoubleComplex* __restrict__ X,
    double* __restrict__ Y) {
  constexpr int kRadix1 = 8;
  constexpr int kRadix2 = 3;
  constexpr int kFFT    = 24;
  if (nhalf != 13 || nZeta != kFFT) return;

  // Block layout: TPB(32, FFTS_PER_BLOCK). Each row of threadIdx.y processes
  // one FFT. Threads (0..23 in x) are productive; (24..31) idle.
  int batch = blockIdx.x * blockDim.y + threadIdx.y;
  if (batch >= total_batches) return;
  int t = threadIdx.x;
  if (t >= kFFT) return;
  int n2 = t / kRadix1;   // 0..2
  int k1 = t % kRadix1;   // 0..7
  int k2 = n2;            // same as n2; thread reuse for stage 3

  size_t X_base = (size_t)batch * (size_t)nhalf;

  // Preload X[0..12] into shared memory cooperatively. Lay out as
  //   s_X_re[fft_idx][m], s_X_im[fft_idx][m] for m in [0, 13).
  // Stride per FFT = 13. Stored as 2 separate arrays for nicer access.
  // Plus extra slots [13..23] holding Hermitian conjugates so all 24
  // positions read from shared with no branch.
  extern __shared__ double smem[];
  // Layout per block:
  //   [0 .. FFTS_PER_BLOCK*24)   = s_X_re full (all 24 m values per FFT)
  //   [FFTS_PER_BLOCK*24 .. *48) = s_X_im full
  //   [FFTS_PER_BLOCK*48 .. *96) = s_T_re + s_T_im for stage 3
  int FFTS_PER_BLOCK = (int)blockDim.y;
  double* s_X_re_full = smem;
  double* s_X_im_full = smem + (size_t)FFTS_PER_BLOCK * 24;
  double* s_T_block   = smem + (size_t)FFTS_PER_BLOCK * 48;  // 48 doubles per FFT
  int fft_idx_in_block = threadIdx.y;
  double* s_X_re = s_X_re_full + (size_t)fft_idx_in_block * 24;
  double* s_X_im = s_X_im_full + (size_t)fft_idx_in_block * 24;
  double* s_T_re = s_T_block + (size_t)fft_idx_in_block * 48;
  double* s_T_im = s_T_block + (size_t)fft_idx_in_block * 48 + 24;

  // Cooperative load: 24 threads (t in [0, 24)) each load one m.
  // For m in [0, nhalf): direct read from X. For m in [nhalf, 24): conjugate.
  // m=12 is the Nyquist; for ntor=10 it's already zero, so the conjugate
  // synthesis for m=12 -> conj(X[12]) gives X[12].x = same, X[12].y = -same.
  // Since X[12].y is zero by Hermitian symmetry of the original real-valued
  // input, the synthesis is exact.
  if (t < kFFT) {
    if (t < nhalf) {
      cufftDoubleComplex v = X[X_base + (size_t)t];
      s_X_re[t] = v.x;
      s_X_im[t] = v.y;
    } else {
      // m in [13, 23]: X[m] = conj(X[24 - m]).
      cufftDoubleComplex v = X[X_base + (size_t)(kFFT - t)];
      s_X_re[t] = v.x;
      s_X_im[t] = -v.y;
    }
  }
  __syncwarp();

  // Stage 1: length-8 inverse DFT, F_{n2}[k1] = sum_{n1=0..7} X[3*n1+n2] *
  //          w_8^{(n1*k1) mod 8}. Reads all 8 from s_X (no branch).
  double F_re = 0.0;
  double F_im = 0.0;
  #pragma unroll
  for (int n1 = 0; n1 < kRadix1; ++n1) {
    int m = 3 * n1 + n2;
    double xr = s_X_re[m];
    double xi = s_X_im[m];
    int idx = (n1 * k1) & 7;  // mod 8
    double c = kRadix8_cos[idx];
    double s = kRadix8_sin[idx];
    F_re += xr * c - xi * s;
    F_im += xr * s + xi * c;
  }

  // Stage 2: twiddle T = F * w_24^{n2*k1}.
  double T_re, T_im;
  {
    int p = (n2 * k1);  // in [0, 14], so % 24 is no-op
    double c = kRadix24_cos[p];
    double s = kRadix24_sin[p];
    T_re = F_re * c - F_im * s;
    T_im = F_re * s + F_im * c;
  }

  // Share T across threads with the same k1 for stage 3.
  s_T_re[k1 * kRadix2 + n2] = T_re;
  s_T_im[k1 * kRadix2 + n2] = T_im;
  __syncwarp();  // each warp = one FFT (TPB.x=32, threadIdx.y picks FFT)

  // Stage 3: length-3 inverse DFT for thread's (k1, k2).
  double T0_re = s_T_re[k1 * kRadix2 + 0];
  double T0_im = s_T_im[k1 * kRadix2 + 0];
  double T1_re = s_T_re[k1 * kRadix2 + 1];
  double T1_im = s_T_im[k1 * kRadix2 + 1];
  double T2_re = s_T_re[k1 * kRadix2 + 2];
  double T2_im = s_T_im[k1 * kRadix2 + 2];

  // w_3 = exp(+2*pi*i/3) = -1/2 + i*sqrt(3)/2  (inverse direction).
  constexpr double kS3 = 0.86602540378443864676;
  double out_re;
  if (k2 == 0) {
    out_re = T0_re + T1_re + T2_re;
  } else if (k2 == 1) {
    out_re = T0_re - 0.5 * T1_re - kS3 * T1_im
                   - 0.5 * T2_re + kS3 * T2_im;
  } else {  // k2 == 2
    out_re = T0_re - 0.5 * T1_re + kS3 * T1_im
                   - 0.5 * T2_re - kS3 * T2_im;
  }

  size_t Y_idx = (size_t)batch * (size_t)kFFT + (size_t)(k1 + kRadix1 * k2);
  Y[Y_idx] = out_re;
}

// Hand-coded length-24 forward real-to-complex discrete Fourier transform.
//
// The kernel implements a Cooley-Tukey radix-8x3 decomposition of the
// length-24 forward transform and is sized for the exact shape used by
// the iteration body: a real input of length nZeta = 24 and a
// Hermitian-symmetric complex output of length nhalf = 13. Adopting the
// standard Cooley-Tukey indexing n = 3 n1 + n2 with n1 in [0, 8) and
// n2 in [0, 3), and k = k1 + 8 k2 with k1 in [0, 8) and k2 in [0, 3),
// the transform may be written as
//
//   X[k1 + 8 k2] = sum_{n2 = 0..2} w_3F^{n2 k2} * w_24F^{n2 k1} *
//                  sum_{n1 = 0..7} x[3 n1 + n2] * w_8F^{n1 k1},
//
// where w_8F = exp(-2 pi i / 8), w_24F = exp(-2 pi i / 24), and
// w_3F = exp(-2 pi i / 3) are the forward-direction roots of unity. The
// kernel computes the inner length-8 transform at fixed n2, applies the
// length-24 twiddle, and finishes with a length-3 transform along the
// n2 axis at fixed k1.
//
// The block layout mirrors k_inverse_dft_24_radix83: the launch uses
// TPB.x = 32 and TPB.y = FFTS_PER_BLOCK, one transform per row of the
// y axis, with twenty-four productive threads per row in the x axis.
// Only the thirteen indices k = k1 + 8 k2 less than nhalf write final
// outputs; the remaining productive threads complete their first two
// stages because their intermediate length-3 results occupy shared
// memory slots that the surviving threads read during the third stage.
//
// The twiddle tables for the forward direction are obtained from the
// inverse-direction tables by negating the imaginary components. The
// length-8 cosine table is symmetric and is reused without
// modification; the length-8 sine table and the length-24 sine table
// require sign inversion, as does the length-3 imaginary scalar.
__global__ void k_forward_dft_24_radix83(
    int total_batches, int nZeta, int nhalf,
    const double* __restrict__ Y,
    cufftDoubleComplex* __restrict__ X) {
  constexpr int kRadix1 = 8;
  constexpr int kRadix2 = 3;
  constexpr int kFFT    = 24;
  if (nhalf != 13 || nZeta != kFFT) return;

  int batch = blockIdx.x * blockDim.y + threadIdx.y;
  if (batch >= total_batches) return;
  int t = threadIdx.x;
  if (t >= kFFT) return;
  int n2 = t / kRadix1;   // 0..2
  int k1 = t % kRadix1;   // 0..7
  int k2 = n2;            // thread reuse for stage 3

  // Shared memory: [s_x[0..24)] real input per FFT, then [s_T_re | s_T_im]
  // length-24 each for stage-3 cross-thread share.
  extern __shared__ double smem_fwd[];
  int FFTS_PER_BLOCK = (int)blockDim.y;
  double* s_x_full = smem_fwd;                                    // [FFTS*24]
  double* s_T_block = smem_fwd + (size_t)FFTS_PER_BLOCK * 24;     // [FFTS*48]
  int fft_idx_in_block = threadIdx.y;
  double* s_x = s_x_full + (size_t)fft_idx_in_block * 24;
  double* s_T_re = s_T_block + (size_t)fft_idx_in_block * 48;
  double* s_T_im = s_T_block + (size_t)fft_idx_in_block * 48 + 24;

  // Cooperative load: each productive thread loads one real input.
  size_t Y_base = (size_t)batch * (size_t)kFFT;
  if (t < kFFT) {
    s_x[t] = Y[Y_base + (size_t)t];
  }
  __syncwarp();

  // Stage 1: length-8 forward DFT, F_{n2}[k1] = Σ_{n1} x[3n1+n2] · w8F^{n1 k1}
  // x is real, so F_re/F_im accumulate from x * (cos, -sin).
  double F_re = 0.0;
  double F_im = 0.0;
  #pragma unroll
  for (int n1 = 0; n1 < kRadix1; ++n1) {
    double xr = s_x[3 * n1 + n2];
    int idx = (n1 * k1) & 7;  // mod 8
    double c = kRadix8_cos[idx];
    double s = kRadix8_sin[idx];  // inverse-direction sin
    // Forward twiddle = cos(p) - i sin(p), so F_im subtracts the sin term.
    F_re += xr * c;
    F_im -= xr * s;
  }

  // Stage 2: twiddle T = F · w24F^{n2 k1} where w24F = cos - i sin.
  // T_re = F_re*c + F_im*s ; T_im = F_im*c - F_re*s
  double T_re, T_im;
  {
    int p = (n2 * k1);  // in [0, 14]
    double c = kRadix24_cos[p];
    double s = kRadix24_sin[p];  // inverse-direction sin
    T_re = F_re * c + F_im * s;
    T_im = F_im * c - F_re * s;
  }

  s_T_re[k1 * kRadix2 + n2] = T_re;
  s_T_im[k1 * kRadix2 + n2] = T_im;
  __syncwarp();

  // Stage 3: length-3 forward DFT for this thread's (k1, k2).
  // w3F = exp(-2πi/3) = -1/2 - i sqrt(3)/2
  double T0_re = s_T_re[k1 * kRadix2 + 0];
  double T0_im = s_T_im[k1 * kRadix2 + 0];
  double T1_re = s_T_re[k1 * kRadix2 + 1];
  double T1_im = s_T_im[k1 * kRadix2 + 1];
  double T2_re = s_T_re[k1 * kRadix2 + 2];
  double T2_im = s_T_im[k1 * kRadix2 + 2];

  constexpr double kS3 = 0.86602540378443864676;  // sqrt(3)/2

  double out_re, out_im;
  if (k2 == 0) {
    out_re = T0_re + T1_re + T2_re;
    out_im = T0_im + T1_im + T2_im;
  } else if (k2 == 1) {
    // X[k] = T0 + T1·(-1/2 - i kS3) + T2·(-1/2 + i kS3)
    out_re = T0_re - 0.5 * T1_re + kS3 * T1_im
                   - 0.5 * T2_re - kS3 * T2_im;
    out_im = T0_im - 0.5 * T1_im - kS3 * T1_re
                   - 0.5 * T2_im + kS3 * T2_re;
  } else {  // k2 == 2
    // X[k] = T0 + T1·(-1/2 + i kS3) + T2·(-1/2 - i kS3)
    out_re = T0_re - 0.5 * T1_re - kS3 * T1_im
                   - 0.5 * T2_re + kS3 * T2_im;
    out_im = T0_im - 0.5 * T1_im + kS3 * T1_re
                   - 0.5 * T2_im - kS3 * T2_re;
  }

  // Hermitian half output: only k < nhalf writes.
  int k = k1 + kRadix1 * k2;
  if (k < nhalf) {
    size_t X_idx = (size_t)batch * (size_t)nhalf + (size_t)k;
    X[X_idx].x = out_re;
    X[X_idx].y = out_im;
  }
}

// Elementwise narrowing cast from double-precision complex to
// single-precision complex. The mixed-precision FFT scaffold below uses
// this kernel to materialize a cufftComplex buffer suitable for the
// single-precision cuFFT plans without requiring a separate library
// call. The mapping is one thread per element, with no fan-out across
// the input array.
__global__ void k_cast_complex_fp64_to_fp32(
    size_t n, const cufftDoubleComplex* __restrict__ src,
    cufftComplex* __restrict__ dst) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  cufftDoubleComplex v = src[i];
  cufftComplex out;
  out.x = (float)v.x;
  out.y = (float)v.y;
  dst[i] = out;
}

// Elementwise widening cast from single-precision real to
// double-precision real. The mixed-precision FFT scaffold uses this
// kernel to restore the post-transform real-space buffer to the
// double-precision representation that the downstream scatter kernels
// consume. One thread per element, no fan-out across the input array.
__global__ void k_cast_fp32_to_fp64(
    size_t n, const float* __restrict__ src, double* __restrict__ dst) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = (double)src[i];
}

// Direct length-24 inverse discrete Fourier transform.
//
// The kernel implements the length-24 inverse transform by evaluating
// the closed-form sum directly, without decomposing into smaller
// transforms. For a Hermitian-symmetric complex input of length nhalf
// and a real output of length N = nZeta, the inverse transform reads
//
//   Y[k] = X[0].re
//        + sum_{n = 1..nhalf - 1} (2 X[n].re cos(2 pi n k / N) -
//                                  2 X[n].im sin(2 pi n k / N))
//        + (N even ? X[N/2].re * (-1)^k : 0).
//
// The Nyquist correction term is harmless when X[N/2] is zero, which
// is the case under the truncation ntor = 10 since X[12] vanishes in
// practice; the doubling that the principal sum applies to that index
// has no effect.
//
// The kernel uses one thread per output element (batch, k), giving a
// trivially parallel mapping across approximately 4.6 million threads
// at the canonical N = 64 problem size (192 thousand batches of
// length-24). Each thread reads its nhalf complex inputs from d_X and
// writes a single real value to d_Y. The cosine and sine tables are
// precomputed by Reshape at sizes nhalf * nZeta = 13 * 24 = 312
// doubles each and reside on the device throughout the run; using
// tabulated values avoids the per-call evaluation of double-precision
// trigonometric library routines inside the inner sum.
__global__ void k_inverse_dft_24(
    int total_batches, int nhalf, int nZeta,
    const cufftDoubleComplex* __restrict__ X,
    const double* __restrict__ cos_table,
    const double* __restrict__ sin_table,
    double* __restrict__ Y) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  int batch = blockIdx.y * blockDim.y + threadIdx.y;
  if (k >= nZeta || batch >= total_batches) return;
  size_t X_base = (size_t)batch * (size_t)nhalf;
  size_t Y_idx = (size_t)batch * (size_t)nZeta + (size_t)k;

  // n=0: Y[k] += X[0].re
  double acc = X[X_base + 0].x;
  // n=1..nhalf-1: contributes 2*(Re*cos - Im*sin).
  for (int n = 1; n < nhalf; ++n) {
    cufftDoubleComplex Xn = X[X_base + n];
    double c = cos_table[(size_t)n * (size_t)nZeta + (size_t)k];
    double s = sin_table[(size_t)n * (size_t)nZeta + (size_t)k];
    acc += 2.0 * (Xn.x * c - Xn.y * s);
  }
  Y[Y_idx] = acc;
}

// k_scatter_main_and_con_v4 fuses the sixteen even-and-odd
// real-space scatter outputs of k_scatter_main with the two
// constraint-grid outputs rCon and zCon emitted by k_scatter_con
// into a single kernel. The block geometry assigns four
// independent warps per block, each warp handling one
// (configuration, jF_local, k) tuple; the per-warp arithmetic is
// identical to the unfused baseline kernel. Aggregating four
// warps per block reduces the launched block count by a factor of
// four for the same total work, raising the number of warps
// resident on each streaming multiprocessor and admitting greater
// double-precision instruction-issue concurrency under the
// scheduler. The kernel uses neither shared memory nor warp-level
// synchronisation primitives, since each warp is fully independent
// of the others in the same block.
__global__ void k_scatter_main_and_con_v4(
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
    double* __restrict__ rCon, double* __restrict__ zCon,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  // Each warp (threadIdx.y) picks its own z index within the 4-warp group.
  int warp_id = threadIdx.y;
  int z_base = blockIdx.z * blockDim.y;
  int z_global = z_base + warp_id;
  int config = z_global / ns_local;
  int jF_local = z_global - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int k = blockIdx.y;
  int lane = threadIdx.x;
  // One x-block per 32 poloidal points, so nThetaReduced > 32 is covered by
  // multiple warps (matches the forward and scatter-basis kernels).
  int l = blockIdx.x * blockDim.x + lane;
  if (l >= nThetaReduced) return;

  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;

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

  for (int m = 0; m < mpol; ++m) {
    const size_t y_base_local = (size_t)((jF_local * mpol + m) * kBatch);
    double rmkcc  = Y[cfg_Y + (y_base_local + kRmkcc)  * nZeta + k];
    double rmkss  = Y[cfg_Y + (y_base_local + kRmkss)  * nZeta + k];
    double rmkccN = Y[cfg_Y + (y_base_local + kRmkccN) * nZeta + k];
    double rmkssN = Y[cfg_Y + (y_base_local + kRmkssN) * nZeta + k];
    double zmksc  = Y[cfg_Y + (y_base_local + kZmksc)  * nZeta + k];
    double zmkcs  = Y[cfg_Y + (y_base_local + kZmkcs)  * nZeta + k];
    double zmkscN = Y[cfg_Y + (y_base_local + kZmkscN) * nZeta + k];
    double zmkcsN = Y[cfg_Y + (y_base_local + kZmkcsN) * nZeta + k];
    double lmksc  = Y[cfg_Y + (y_base_local + kLmksc)  * nZeta + k];
    double lmkcs  = Y[cfg_Y + (y_base_local + kLmkcs)  * nZeta + k];
    double lmkscN = Y[cfg_Y + (y_base_local + kLmkscN) * nZeta + k];
    double lmkcsN = Y[cfg_Y + (y_base_local + kLmkcsN) * nZeta + k];

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

// Shared-memory cached variant of the fused main-and-constraint scatter.
//
// The kernel adopts the same per-warp work assignment as the v4 variant
// above, in which one warp handles one (config, jF_local, k) tuple and
// the lanes within the warp partition the poloidal output points. The
// distinguishing element is that the Y values consumed during the
// inner toroidal-mode loop are first staged into a per-warp shared
// memory tile of mpol * kBatch doubles. Although the Y loads are
// warp-uniform and would in principle broadcast efficiently through
// the L1 cache, the latency observed during the inner loop is reduced
// by serving them from shared memory.
//
// Each warp uses four of its lanes to cooperatively load the
// mpol * kBatch tile into shared memory before the toroidal-mode loop
// begins, after which all active lanes read the cached values without
// returning to global memory. The block geometry is unchanged relative
// to the v4 variant: four warps per block, one warp per
// (config, jF_local, k) tuple.
__global__ __launch_bounds__(128, 5) void k_scatter_main_and_con_v5(
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

  // Shared memory: per-warp slot of 10*12 = 120 doubles. Allocated by
  // launch as kBatch_runtime * mpol * blockDim.y per-block.
  // Layout: s_Y[warp_id][m][q] = Y[cfg, jF_local, m, q, k] for this warp's k.
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

  // One x-block per 32 poloidal points (matches the v4 grid), covering
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

  // mpol is 10 at the production callsite. Force unroll the m loop so the
  // compiler constant-folds m_even and the xmpq[m]/sqrtSF_jF multiplications,
  // and pipelines the per-m shared reads + FMAs.
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




// k_scatter_con accumulates the constraint-force outputs rCon and
// zCon over the radial range jF in [nsMinF, nsMaxFIncludingLcfs).
// The local index jF_con_local addresses rCon and zCon at offset
// (jF - nsMinF), which extends one row beyond the full-grid range
// of k_scatter_main to admit the last-closed-flux-surface row when
// it is owned by the present rank. The configuration axis is
// carried on blockIdx.z encoded as
// config * ns_con_local + jF_con_local.
// Y is per-config (n_config * ns_local * mpol * kBatch * nZeta).
// rCon/zCon are per-config (n_config * ns_con_local * nZnT).
// sqrtSF/xmpq stay shared (radial grid + spectral factors constant).
__global__ void k_scatter_con(
    int n_config, int ns_local, int ns_con_local,
    int mpol, int nZeta, int nThetaReduced, int nThetaEff,
    int nsMinF_offset_in_local,  // jF_local index of nsMinF in the larger range
    const double* __restrict__ Y, const double* __restrict__ cosmu,
    const double* __restrict__ sinmu, const double* __restrict__ xmpq,
    const double* __restrict__ sqrtSF,
    double* __restrict__ rCon, double* __restrict__ zCon) {
  int config = blockIdx.z / ns_con_local;
  int jF_con = blockIdx.z - config * ns_con_local;
  if (config >= n_config) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= nThetaReduced) return;
  int jF_local = jF_con + nsMinF_offset_in_local;  // index into Y
  size_t cfg_Y   = (size_t)config * (size_t)ns_local * (size_t)mpol *
                   (size_t)kBatch * (size_t)nZeta;
  size_t cfg_con = (size_t)config * (size_t)ns_con_local *
                   (size_t)nZeta * (size_t)nThetaEff;

  double r_acc = 0.0, z_acc = 0.0;
  for (int m = 0; m < mpol; ++m) {
    // Per-configuration indexing: add cfg_Y OUTSIDE the * nZeta scaling.
    const size_t y_base_local = (size_t)((jF_local * mpol + m) * kBatch);
    double rmkcc = Y[cfg_Y + (y_base_local + kRmkcc) * nZeta + k];
    double rmkss = Y[cfg_Y + (y_base_local + kRmkss) * nZeta + k];
    double zmksc = Y[cfg_Y + (y_base_local + kZmksc) * nZeta + k];
    double zmkcs = Y[cfg_Y + (y_base_local + kZmkcs) * nZeta + k];
    int bml = m * nThetaReduced + l;
    double cmu = cosmu[bml];
    double smu = sinmu[bml];
    bool m_even = ((m & 1) == 0);
    double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF[jF_local];
    r_acc += (rmkcc * cmu + rmkss * smu) * con_factor;
    z_acc += (zmksc * smu + zmkcs * cmu) * con_factor;
  }

  size_t idx = cfg_con + (size_t)((jF_con * nZeta + k) * nThetaEff + l);
  rCon[idx] += r_acc;
  zCon[idx] += z_acc;
}

// Device implementation of FourierGeometry::extrapolateTowardsAxis.
//
// The kernel completes the axis-surface initialization that the host
// triplet would otherwise perform. For each toroidal mode index n in
// [0, ntor], the m = 1 spectral coefficients on the axis surface
// (surface index zero) are set equal to those on the first off-axis
// surface (surface index one). When the configuration is
// three-dimensional, the m = 0 lambda coefficient lmncs is propagated
// from the first off-axis surface to the axis surface in the same
// manner. The kernel is launched with a three-dimensional grid of
// shape (ntor + 1, 1, n_config), assigning one thread to each
// (configuration, toroidal mode) pair, and is invoked only by the
// rank whose nsMinF1 equals zero, that is, the rank that owns the
// axis surface.
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

// Pre-pass of the multigrid upscale: scale the previous-stage snapshot in
// place by the previous stage's scalxc. This is the caller-side
// decomposeInto pass of the host upscale (InterpolateToNextMultigridStep
// receives X(COARSE) * SCALXC(COARSE)); the axis extrapolation and the
// radial interpolation then operate on scaled values, matching the host
// arithmetic. Grid: ((ntor+1)/TPB, mpol, ns_old * n_config).
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

// Per-cfg axis extrapolation for odd-m modes (pre-processing step for
// the multigrid upscale). Mirrors the host upscale's axis pre-processing: for each odd m and
// each n, overwrites the OLD axis (js=0) value with the extrapolation
//   old[m_odd, js=0, n] = 2 * old[m_odd, js=1, n] - old[m_odd, js=2, n]
// across all 6 spec components. Operates on the scalxc-scaled values
// produced by k_scale_prev_by_scalxc, the same ordering as the host. The
// radial interp downstream reads these modified axis values when
// interpolating jNew=1.
// Grid: ((ntor+1)/TPB, (mpol+1)/2, n_config). Each thread covers one (cfg, m_odd, n).
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

// Per-cfg radial interpolation of d_pts_x at a multigrid stage boundary.
// Replicates the host upscale in Vmec::InterpolateToNextMultigridStep, run
// per configuration so distinct-mode batched runs preserve per-cfg state
// across the ns_array stages. Inputs are the previous-stage values already
// scaled by the previous stage's scalxc (k_scale_prev_by_scalxc) with the
// odd-m axis extrapolated (k_axis_extrapolate_odd_m_prev); this kernel
// applies the host's linear interpolation
//
//   new[jNew, m, n] = ((1 - xint) * old[js1, m, n] + xint * old[js2, m, n])
//                       / scalxc[jNew, m_parity]
//
// and zeroes the odd-m axis rows, the host's final step. The explicit
// round-to-nearest intrinsics pin the multiply/add/divide sequence to the
// host's non-contracted arithmetic so the result is bit-identical to the
// host upscale. Grid: (ntor+1)/TPB x mpol x (ns_new * n_config).
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

// Device implementation of Vmec::performTimeStep.
//
// The kernel advances the conjugate-gradient time integrator that
// updates the spectral coefficients of the configuration. The
// velocity is first refreshed via
//
//   v_new = velocity_scale * (b1 * v_old + dt * f),
//
// and the spectral position is then advanced as x = x + dt * v_new.
// The force tensor f is read from the device-resident
// d_decomposed_f arrays that the preconditioner chain has populated;
// the velocity tensor d_pts_v is read and written in place, and the
// spectral position tensor d_pts_x is updated in place and persists
// across iterations of the outer time loop.
//
// The block grid encodes the iteration range jF in
// [nsMinF, nsMaxFIncludingLcfs), expressed in local indices as
// jF_v_local in [0, ns_con_local), through blockIdx.z =
// config * ns_con_local + jF_v_local. The velocity tensor is indexed
// as (configuration, jF_v_local, m, n) and is sized for ns_con_local
// surfaces along the radial axis.
// x indexed by (cfg, jF_full_local, m, n) over ns_local surfaces, where
// jF_full_local = jF_v_local + (nsMinF - nsMinF1).
//
// lasym=false, lthreed=true paths handled (our workload). lasym branches
// from the CPU body are omitted.
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
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z / ns_con_local;
  int jF_v_local = blockIdx.z - config * ns_con_local;
  if (config >= n_config) return;
  // Inactive cfgs hold their (x, v) state: host-side shared quantities
  // (the fNorm family, tcon) derive from cfg 0's live slot, so converged
  // slots must stay frozen while the rest of the batch iterates.
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int m = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_v_local >= ns_con_local || m >= mpol || n > ntor) return;

  // Per-cfg (fac, b1) override: when d_fac_b1 is non-null, each cfg's
  // velocity_scale + conjugation_parameter come from its own slot in the
  // d_fac_b1 array (written by k_update_timestep). This lets each cfg's
  // accelerator/damper respond to its own residual instead of the shared
  // scalar tuned for cfg 0 only. Required for distinct-mode batched
  // convergence when cfgs converge at different rates.
  double fac_use = velocity_scale;
  double b1_use = conjugation_parameter;
  if (d_fac_b1 != nullptr) {
    fac_use = d_fac_b1[(size_t)config * 2 + 0];
    b1_use  = d_fac_b1[(size_t)config * 2 + 1];
  }

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
                   (b1_use * v_rcc[idx_v] + time_step * f_rcc[idx_v]);
    v_rcc[idx_v] = v_new;
    x_rcc[idx_x] += time_step * v_new;
  }
  // v_zsc / x_zsc (zmnsc parity)
  {
    double v_new = fac_use *
                   (b1_use * v_zsc[idx_v] + time_step * f_zsc[idx_v]);
    v_zsc[idx_v] = v_new;
    x_zsc[idx_x] += time_step * v_new;
  }
  // v_lsc / x_lsc (lmnsc parity)
  {
    double v_new = fac_use *
                   (b1_use * v_lsc[idx_v] + time_step * f_lsc[idx_v]);
    v_lsc[idx_v] = v_new;
    x_lsc[idx_x] += time_step * v_new;
  }
  if (lthreed) {
    // v_rss / x_rss
    {
      double v_new = fac_use *
                     (b1_use * v_rss[idx_v] + time_step * f_rss[idx_v]);
      v_rss[idx_v] = v_new;
      x_rss[idx_x] += time_step * v_new;
    }
    // v_zcs / x_zcs
    {
      double v_new = fac_use *
                     (b1_use * v_zcs[idx_v] + time_step * f_zcs[idx_v]);
      v_zcs[idx_v] = v_new;
      x_zcs[idx_x] += time_step * v_new;
    }
    // v_lcs / x_lcs
    {
      double v_new = fac_use *
                     (b1_use * v_lcs[idx_v] + time_step * f_lcs[idx_v]);
      v_lcs[idx_v] = v_new;
      x_lcs[idx_x] += time_step * v_new;
    }
  }
}

// k_extract_geom_scalars collects six geometry-derived scalar values
// from configuration zero's slices of the device buffers d_r1_e,
// d_r1_o, and d_z1_e and writes them into a contiguous six-double
// output buffer. The values feed the host-side scalar accessors
// SetRadialExtent, which consumes (r_outer, r_inner) as its first
// two arguments, and SetGeometricOffset, which consumes (r_00, z_00)
// as its second two; the remaining two doubles carry the additional
// boundary samples used for diagnostic output. The kernel is
// launched with a single thread because the work is the unconditional
// emission of six element-wise reads from precomputed offsets and
// admits no parallelism beyond that.
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

// k_compute_jacobian assigns one thread to each (jH_local, kl) pair
// and derives the half-grid geometric quantities r12, ru12, zu12, rs,
// zs, and tau from the full-grid even-odd geometry components
// produced by the preceding inverse FFT. The configuration axis is
// carried on blockIdx.z, with each configuration occupying its own
// strided slice of the full-grid input buffers (r1_e, r1_o, ru_e,
// ru_o, z1_e, z1_o, zu_e, zu_o) and half-grid output buffers (r12,
// ru12, zu12, rs, zs, tau). The radial-coordinate auxiliary sqrtSH
// is invariant across configurations under the assumption of a
// shared radial grid and is consumed without a per-configuration
// offset. At n_config equal to one the configuration axis collapses
// to blockIdx.z equal to zero and the per-configuration offsets
// degenerate to the single-configuration layout, preserving the
// pre-batched behaviour bit-for-bit.
__global__ void k_compute_jacobian(
    int n_config, int ns_local,
    int ns_h, int jF_in_offset, int nZnT,
    const double* __restrict__ r1_e, const double* __restrict__ r1_o,
    const double* __restrict__ ru_e, const double* __restrict__ ru_o,
    const double* __restrict__ z1_e, const double* __restrict__ z1_o,
    const double* __restrict__ zu_e, const double* __restrict__ zu_o,
    const double* __restrict__ sqrtSH, double deltaS, double dSHalfDsInterp,
    double* __restrict__ r12, double* __restrict__ ru12,
    double* __restrict__ zu12, double* __restrict__ rs,
    double* __restrict__ zs, double* __restrict__ tau) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH_local = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (kl >= nZnT) return;
  if (jH_local >= ns_h) return;

  size_t cfg_full = (size_t)config * (size_t)ns_local * (size_t)nZnT;
  size_t cfg_half = (size_t)config * (size_t)ns_h    * (size_t)nZnT;

  // Inside full-grid surface: jF_in (local). Outside: jF_in + 1.
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
}

// k_compute_metric_elements assigns one thread to each (jH_local, kl)
// pair and computes the half-grid metric tensor elements gsqrt, guu,
// guv (only under three-dimensional symmetry, lthreed), and gvv from
// the full-grid even-odd geometry r1_e/o, ru_e/o, zu_e/o, rv_e/o, and
// zv_e/o, together with the half-grid auxiliaries tau and r12 produced
// by the preceding jacobian computation. The configuration axis is
// carried on blockIdx.z, with each configuration occupying its own
// strided slice of the full-grid inputs, the half-grid inputs tau and
// r12, and the half-grid outputs gsqrt, guu, guv, and gvv. The radial-
// coordinate auxiliaries sqrtSF and sqrtSH are invariant across
// configurations under the assumption of a shared radial grid and are
// consumed without per-configuration offsets. At n_config equal to
// one the configuration axis collapses to blockIdx.z equal to zero
// and the per-configuration offsets degenerate to the single-
// configuration layout, preserving the pre-batched behaviour bit-for-
// bit.
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

// Fused jacobian and metric-element computation.
//
// The kernel combines what would otherwise be two consecutive kernels
// (k_compute_jacobian and k_compute_metric_elements) into a single
// launch, with each thread handling one (configuration, jH_local, kl)
// tuple. Within the thread the work proceeds in two stages. The
// jacobian stage computes r12, ru12, zu12, rs, zs, and tau and writes
// these arrays to global memory, since downstream kernels in the
// iteration body continue to consume them. The metric stage then
// reuses the r12 and tau values already held in registers, together
// with the ru, zu, and r1 inputs that the jacobian stage loaded from
// global memory, to compute gsqrt, guu, guv, and gvv.
//
// Combining the two stages removes the global-memory round trip that
// the separate-kernel arrangement would incur on r12, tau, the
// even-parity and odd-parity components of ru and zu, the
// corresponding components of r1, and the radial weight sqrtSF.
//
// The fusion preserves the floating-point operation order of the
// separate kernels, so the result is bit-identical: only the storage
// location of the shared intermediates changes from global memory to
// registers.
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

// k_jacobian_metric_dvdsh fuses the jacobian-and-metric computation
// of k_jacobian_and_metric with the differential-volume reduction of
// k_update_dvdsh into a single kernel launch. Each block services
// one (configuration, jH_local) pair, with thirty-two threads
// per block (TPB = 32). Each thread iterates over the flattened
// poloidal-toroidal index kl by a stride of thirty-two from its
// initial offset threadIdx.x. Within the iteration the thread
// computes the jacobian and metric outputs for its kl and writes
// them to the half-grid output buffers, and additionally accumulates
// the differential-volume contribution gsqrt * wInt into a
// thread-local partial sum. After the kl loop completes the per-
// thread partial sums are reduced through a power-of-two tree on
// the shared-memory array s_partial of length thirty-two, and
// thread zero writes the final dVdsH value for the configuration's
// jH_local slot. The reduction sequence reproduces the original
// k_update_dvdsh order bit-for-bit, preserving the floating-point
// rounding of the unfused path.
__global__ void k_jacobian_metric_dvdsh(
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

  size_t cfg_full = (size_t)config * (size_t)ns_local * (size_t)nZnT;
  size_t cfg_half = (size_t)config * (size_t)ns_h    * (size_t)nZnT;

  int jF_in = jH_local + jF_in_offset;
  int jF_out = jF_in + 1;
  double sH = sqrtSH[jH_local];
  double sqrtSF_i = sqrtSF[jF_in];
  double sqrtSF_o = sqrtSF[jF_out];
  double sF_i = sqrtSF_i * sqrtSF_i;
  double sF_o = sqrtSF_o * sqrtSF_o;

  // Grid-stride: each thread accumulates dvdsh contribution across its
  // assigned kls. Matches the original k_update_dvdsh accumulation order.
  double acc = 0.0;
  for (int kl = threadIdx.x; kl < nZnT; kl += blockDim.x) {
    // Loads
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

    size_t iHalf = cfg_half + (size_t)jH_local * (size_t)nZnT + (size_t)kl;

    // jacobian
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

    // metric
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

    // dvdsh per-thread accumulation (same order as k_update_dvdsh).
    int l = kl % nThetaEff;
    acc += gsqrt_v * wInt[l];
  }

  // Block reduction: TPB=32, power-of-2 tree on s_partial[32].
  // Matches k_update_dvdsh exactly.
  __shared__ double s_partial[32];
  s_partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_partial[threadIdx.x] += s_partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    dVdsH[config * ns_h + jH_local] = signOfJacobian * s_partial[0];
  }
}

// Fused jacobian, metric, and dVdsH reduction kernel with atomic
// accumulation.
//
// The kernel shares its launch geometry with k_jacobian_and_metric
// above: a thread block of TPB = 64, X-blocks of size
// ceil(nZnT / TPB), and one block per (X, jH, configuration) tuple
// in the three-dimensional grid. The per-thread body first writes
// the jacobian and metric outputs in the same arrangement as the
// fused jacobian-and-metric kernel, then accumulates the per-thread
// contribution signOfJacobian * gsqrt * wInt into the differential
// volume slot dVdsH[configuration, jH] via atomicAdd. This removes
// the separate block-reduction kernel that would otherwise consume
// gsqrt and wInt to produce dVdsH, and removes the corresponding
// global-memory round trip on gsqrt that the separate kernel would
// have incurred.
//
// The caller is responsible for zeroing the dVdsH slice with
// cudaMemsetAsync before the launch, since the kernel accumulates
// into the slot rather than overwriting it.
//
// Because the atomic accumulation ordering across blocks is not
// deterministic, the resulting floating-point sum differs from the
// equivalent tree-reduction sum by amounts on the order of a single
// unit in the last place. This deviation is admitted by the drift
// tolerance the iteration controller applies to dVdsH.
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

  // Atomic accumulator: signOfJacobian * gsqrt * wInt(l) → dVdsH[cfg, jH].
  // Order-nondeterministic; relaxed-contract path.
  int l = kl % nThetaEff;
  double contrib = signOfJacobian * gsqrt_v * wInt[l];
  atomicAdd(&dVdsH[config * ns_h + jH_local], contrib);
}

// k_jacobian_metric_dvdsh_atomic_pair: half-grid pair coarsening of
// the fused kernel above. Each block services two adjacent half-grid
// surfaces (jH_lo = 2 * blockIdx.y, jH_hi = jH_lo + 1; threadIdx.y
// selects the surface). The shared boundary surface
// jF = jH_lo + 1 + jF_in_offset feeds both halves, so caching its
// eight main full-grid fields in shared memory halves the main-field
// global traffic (12 KB per block at nZnT = 192). The
// differential-volume reduction keeps atomicAdd into dVdsH; the
// summation-order non-determinism is admitted by the dVdsH drift
// tolerance, and aspect_ratio stays bit-exact.
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

  // Shared layout: 8 fields, one slot per x-lane (blockDim.x), shared between
  // the two jH y-lanes at the same kl. Field order: r1_e, r1_o, ru_e, ru_o,
  // z1_e, z1_o, zu_e, zu_o. Sized to the block's x-extent rather than nZnT, so
  // the requirement is independent of the poloidal resolution.
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
    // rv/zv not cached (extending the cache to 12 fields measured as a
    // regression); both jF positions hit global.
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

// k_update_dvdsh launches one block per half-grid radial index
// jH_local and uses the threads within the block to cooperate on the
// sum over the combined poloidal-toroidal index kl. The output for
// each half-grid surface is the differential volume contribution
//   dVdsH[jH_local] = signOfJacobian * sum_kl ( gsqrt[jH_local, kl]
//                                                * wInt[kl % nThetaEff] ),
// where wInt provides the per-theta integration weights and the
// modulo extracts the theta index from the flattened (zeta, theta)
// pair. The configuration axis is carried on blockIdx.z, with each
// configuration consuming its own strided slice of gsqrt and writing
// to its own slice of dVdsH. At n_config equal to one blockIdx.z is
// zero and the per-configuration offsets degenerate to the single-
// configuration layout.
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

// A config masked before the batch finishes has its dVdsH zeroed (gsqrt kept),
// zeroing its betatot. Recompute only zeroed slots (surface 0 > 0 for any real
// equilibrium) from gsqrt; one block per config, so the repair is race-free.
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

// k_buco_bvco produces the half-grid covariant magnetic-field
// integrals bucoH and bvcoH by integrating bsubu and bsubv against
// the per-theta integration weights wInt:
//   bucoH[jH] = sum over kl of bsubu[jH, kl] * wInt[kl mod nThetaEff],
//   bvcoH[jH] = sum over kl of bsubv[jH, kl] * wInt[kl mod nThetaEff].
// Each block reduces a single half-grid surface jH_local with its
// threads cooperating over the flattened (zeta, theta) index kl,
// where the modulo extracts the theta index for the weight lookup.
// The configuration axis is carried on blockIdx.z; bsubu, bsubv,
// bucoH, and bvcoH are addressed per configuration on the half-grid
// or per-configuration radial profile respectively, and wInt is
// shared across configurations because the poloidal grid is
// invariant.
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

// k_radial_interior emits the radial derivatives of bucoH, bvcoH,
// and presH, the half-to-full interpolation of dVdsH, and the
// associated force-balance residual equiF at every interior full-
// grid radial index jFi_local. The inputs bucoH, bvcoH, presH, and
// dVdsH are radial profiles indexed on the half grid with stride
// ns_h, and chipF and phipF are radial profiles indexed on the full
// grid with stride ns_local; the outputs jcurvF, jcuruF, presgradF,
// dVdsF, and equiF are indexed on the interior full grid with
// stride nsi. The configuration axis is carried on blockIdx.y, and
// each input and output buffer is addressed at the corresponding
// per-configuration offset under the batched layout.
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

// k_pm_half_reductions populates the preconditioner-matrix scratch
// arrays ax_scratch, bx_scratch, and cx_scratch from per-cell terms
// weighted by the integration kernel pTau, performing a half-grid
// reduction over the flattened (zeta, theta) index kl. Each block
// processes a single half-grid surface jH with its threads
// cooperating across kl; the per-thread partial sums then reduce
// through shared memory to yield the four entries of ax_scratch
// (ax0..ax3), the three entries of bx_scratch (bx0..bx2), and the
// single entry of cx_scratch for that surface. The configuration
// axis is carried on blockIdx.z, with both the half-grid inputs
// (r12, totalPressure, tau, xu12, xs, sqrtSH, bsupv, gsqrt) and
// the full-grid inputs (xu_e, xu_o, x1_o) accessed at their per-
// configuration offsets, and with the ax_scratch, bx_scratch, and
// cx_scratch outputs also per-configuration.
__global__ void k_pm_half_reductions(int n_config, int ns_local, int ns_h,
                                       int nZnT, int nThetaEff,
                                       double pFactor, double deltaS,
                                       int nsMinH, int nsMinF1,
                                       int serial_order,
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
  if (serial_order) {
    // Diagnostic: the host loop's ascending-kl order and arithmetic
    // (divisions by sqrtSH rather than reciprocal multiplies), so the
    // matrix elements match the host bit for bit.
    if (threadIdx.x == 0) {
      double ax0 = 0.0, ax1 = 0.0, ax2 = 0.0, ax3 = 0.0;
      double bx0 = 0.0, bx1 = 0.0, bx2 = 0.0, cxv = 0.0;
      for (int kl = 0; kl < nZnT; ++kl) {
        size_t iHalf = cfg_half + (size_t)jH * (size_t)nZnT + (size_t)kl;
        size_t iFull_0 = cfg_full + (size_t)jF_in_local * (size_t)nZnT + kl;
        size_t iFull_1 = cfg_full + (size_t)jF_out_local * (size_t)nZnT + kl;
        int l = kl % nThetaEff;
        double pTau = pFactor * r12[iHalf] * totalPressure[iHalf] /
                      tau[iHalf] * wInt[l];
        double t1a = xu12[iHalf] / deltaS;
        double t2a = 0.25 * (xu_e[iFull_1] / sH + xu_o[iFull_1]) / sH;
        double t3a = 0.25 * (xu_e[iFull_0] / sH + xu_o[iFull_0]) / sH;
        ax0 += pTau * t1a * t1a;
        ax1 += pTau * (t1a + t2a) * (-t1a + t3a);
        ax2 += pTau * (t1a + t2a) * (t1a + t2a);
        ax3 += pTau * (-t1a + t3a) * (-t1a + t3a);
        double t1b = 0.5 * (xs[iHalf] + 0.5 / sH * x1_o[iFull_1]);
        double t2b = 0.5 * (xs[iHalf] + 0.5 / sH * x1_o[iFull_0]);
        bx0 += pTau * t1b * t2b;
        bx1 += pTau * t1b * t1b;
        bx2 += pTau * t2b * t2b;
        cxv += 0.25 * pFactor * bsupv[iHalf] * bsupv[iHalf] *
               gsqrt[iHalf] * wInt[l];
      }
      ax_scratch[cfg_ax + jH * 4 + 0] = ax0;
      ax_scratch[cfg_ax + jH * 4 + 1] = ax1;
      ax_scratch[cfg_ax + jH * 4 + 2] = ax2;
      ax_scratch[cfg_ax + jH * 4 + 3] = ax3;
      bx_scratch[cfg_bx + jH * 3 + 0] = bx0;
      bx_scratch[cfg_bx + jH * 3 + 1] = bx1;
      bx_scratch[cfg_bx + jH * 3 + 2] = bx2;
      cx_scratch[cfg_cx + jH] = cxv;
    }
    return;
  }
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

// k_pm_assemble_full produces the full-grid preconditioner-matrix
// coefficients m_axd, m_bxd, and m_cxd from the half-grid scratch
// arrays ax_scratch, bx_scratch, and cx_scratch by combining the
// inner and outer half-grid contributions at every full-grid
// surface jF in the local range. The combination respects the
// boundary conditions through the i_valid and o_valid guards that
// suppress the inner half-grid term at jF equal to zero and the
// outer half-grid term at the last-closed-flux-surface row when
// jF equals ns_total minus one. The configuration axis is carried
// on blockIdx.y, and every input and output buffer is addressed
// at its per-configuration offset under the batched layout.
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

// k_ulp_half_reductions emits the half-grid contributions to the
// lambda-preconditioner radial profiles bLambda, dLambda, and
// cLambda by integrating metric ratios against the per-theta
// weights wInt. One block reduces a single half-grid surface jH,
// writing its output at the radial index jH + 1; the unit offset
// mirrors the host-side convention that reserves the first slot of
// each profile for the magnetic axis. The configuration axis is
// carried on blockIdx.z, and the half-grid inputs guu, guv, gvv,
// and gsqrt are addressed at their per-configuration offsets. The
// output profiles bLambda, dLambda, and cLambda are sized to
// lambda_stride per configuration, which equals ns_con_local + 1
// so that the indexing convention used by the host-side
// initialisation through Eigen's setZero on ranges of length
// nsMaxF1 - nsMinF1 + 1 is preserved.
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

// k_ulp_axis_extrap populates the magnetic-axis slot of each lambda-
// preconditioner radial profile with the value carried at the
// adjacent half-grid surface. For every configuration, the kernel
// assigns
//   bLambda[0] = bLambda[1],
//   dLambda[0] = dLambda[1],
//   cLambda[0] = cLambda[1].
// The launch carries n_config_max blocks with one thread each; the
// configuration axis is on blockIdx.x. The axis_present guard
// suppresses the assignment for partitions whose local range does
// not include the magnetic axis, preserving correctness under a
// multi-rank partitioning of the radial domain even though the
// device-side production code runs single-rank.
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

// k_ulp_full_grid_average averages the offset-by-one half-grid
// lambda-preconditioner radial profiles to the full grid. For every
// surface jF in the range [jMin, ns_con_local) the kernel computes
//   bLambda_full[jF - nsMinF] = 0.5 *
//       ( bLambda_half[jF + 1 - nsMinH]
//       + bLambda_half[jF - nsMinH] ),
// and analogous expressions for dLambda and cLambda. The buffer
// layout reserves slot zero of each half-grid profile for the
// magnetic-axis extrapolation produced by k_ulp_axis_extrap, with
// slots one through ns_h carrying the half-grid values, so the two
// addresses read for the average correspond to the half-grid
// indices jF - nsMinH and jF - nsMinH + 1.
//
// Because the read addresses straddle the write address whenever
// nsMinH equals nsMinF, an in-place update of the same buffer
// would induce a write-before-read hazard between threads
// processing adjacent radial indices. The kernel therefore reads
// from the bLambda_in, dLambda_in, and cLambda_in buffers and
// writes to disjoint bLambda_out, dLambda_out, and cLambda_out
// buffers.
//
// The configuration axis is carried on blockIdx.y, and every input
// and output buffer is sized to lambda_stride per configuration,
// which equals ns_con_local + 1 to provide the headroom slot
// required by the indexing convention at the highest jF.
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
  // Half-grid indices for inputs (offset-by-1 layout).
  // CPU: bLambda[jF+1 - nsMinH] and bLambda[jF - nsMinH]; in our buffer
  // jH+1 maps to index (jH - nsMinH) + 1 = jH - nsMinH + 1 in the offset
  // layout. So bLambda[jF+1 - nsMinH] in CPU == bLambda_in[(jF - nsMinH)+1]
  // == bLambda_in[(jF_local + nsMinH_offset) + 1].
  int jH_in_off = jF_local + nsMinH_offset;
  bLambda_out[cfg_out + jF_local] = 0.5 * (bLambda_in[cfg_in + jH_in_off + 1] +
                                  bLambda_in[cfg_in + jH_in_off]);
  dLambda_out[cfg_out + jF_local] = 0.5 * (dLambda_in[cfg_in + jH_in_off + 1] +
                                  dLambda_in[cfg_in + jH_in_off]);
  cLambda_out[cfg_out + jF_local] = 0.5 * (cLambda_in[cfg_in + jH_in_off + 1] +
                                  cLambda_in[cfg_in + jH_in_off]);
}

// k_ulp_assemble computes the spectral lambda preconditioner buffer
// lambdaPreconditioner from the radial profiles bLambda, dLambda,
// and cLambda produced by the half-grid and full-grid averaging
// kernels. One thread is dispatched for every
// (configuration, jF_local, n, m) tuple in the spectral domain;
// the configuration axis is carried on blockIdx.z encoded as
// config * ns_con_local + jF_local. Each configuration writes to
// its own slice of lambdaPreconditioner under the batched layout,
// and the consumer kernel k_apply_lambda_preconditioner reads from
// the matching slice. The per-configuration write and read pairing
// is required for correctness under non-identical per-configuration
// inputs; under a broadcast workload in which every configuration
// receives the same input the per-configuration slices carry
// identical values and the arrangement remains correct.
__global__ void k_ulp_assemble(int n_config, int ns_con_local, int lambda_stride,
                                int jMin,
                                int mpol, int ntor,
                                int nfp, double pFactor,
                                const double* __restrict__ bLambda,
                                const double* __restrict__ dLambda,
                                const double* __restrict__ cLambda,
                                const double* __restrict__ sqrtSF,
                                int sqrtSF_off,  // nsMinF - nsMinF1
                                double* __restrict__ lambdaPreconditioner) {
  int config = blockIdx.z / ns_con_local;
  int jF_local = blockIdx.z - config * ns_con_local;
  if (config >= n_config) return;
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

// k_compute_mhd_forces is the device-side counterpart of
// IdealMhdModel::computeMHDForces. One thread is assigned to each
// (jF_local_force, kl) pair, where jF_local_force ranges over the
// force grid [0, ns_force_local) with ns_force_local equal to
// nsMaxF - nsMinF, and the global radial index satisfies
// jF_global = jF_local_force + nsMinF. Threads whose jF_global
// reaches or exceeds jMaxRZ emit explicit zeros into their output
// slots in lieu of evaluating the force expressions, so the upper
// boundary remains correct under the radial-force cutoff. The
// configuration axis is carried on blockIdx.z, and the full-grid,
// half-grid, and force-grid inputs are addressed at their per-
// configuration offsets under the batched layout. The radial-
// coordinate auxiliaries sqrtSF and sqrtSH are invariant across
// configurations under the assumption of a shared radial grid and
// are consumed without per-configuration offsets. The single-
// configuration arrangement at n_config equal to one collapses the
// configuration axis to zero and recovers the pre-batched layout
// bit-for-bit.
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

// k_compute_mhd_forces_pair is a force-grid coarsening variant of
// the baseline k_compute_mhd_forces kernel. Each block services a
// pair of adjacent force-grid surfaces, with the lower index
// jF_lo computed as 2 * blockIdx.y and the upper index jF_hi as
// jF_lo + 1; the second thread-axis dimension threadIdx.y in
// {0, 1} selects which surface of the pair the thread processes.
// The shared half-grid surface at jH = jF_lo serves a dual role:
// it is the outer half-grid neighbour of jF_lo, on which the
// y == 0 threads depend, and the inner half-grid neighbour of
// jF_hi, on which the y == 1 threads depend.
//
// The y == 0 threads cooperatively load the ten half-grid fields
// at jH = jF_lo into a per-block shared-memory tile. After the
// subsequent __syncthreads the y == 1 threads read their inner
// half-grid neighbour from shared memory, avoiding the second
// global load they would otherwise issue. The half-grid global
// memory traffic per pair-block per kl thus reduces from four
// per-jH reads to three, yielding a one-quarter reduction in
// global traffic on the half-grid path.
//
// The block geometry is dim3 tpb(TPB = 64, 2, 1) for a total of
// one hundred twenty-eight threads, and the launch grid is
// dim3 blocks((nZnT + TPB - 1) / TPB, ns_force_local / 2,
//             n_config_max).
// The integer division of ns_force_local by two requires
// ns_force_local to be even; the host dispatcher falls back to
// the baseline k_compute_mhd_forces when this condition does not
// hold.
//
// The dynamic shared-memory allocation reserves storage for the
// ten cached half-grid fields, each of nZnT doubles, for a per-
// block footprint of approximately fifteen kilobytes at the
// production nZnT of one hundred ninety-two. The shared region is
// declared as a single extern __shared__ array, with the
// individual field slices addressed by precomputed offsets.
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

  // Shared memory layout: 10 fields, one slot per x-lane (blockDim.x), shared
  // between the two jF y-lanes at the same kl. Field order: totalPressure, r12,
  // ru12, zu12, rs, zs, tau, gsqrt, bsupu, bsupv. Sized to the block's x-extent
  // rather than nZnT, so it is independent of the poloidal resolution.
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

  // Cooperative load: only the y=0 thread populates shared. Its jH_out_local
  // equals jH_shared_local by construction. It will also use these values
  // immediately below for its own jH_out computation.
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

// ComputeMHDForces is deliberately not fused with assembleTotalForces: the
// latter's brcon comes from gCon, produced by DealiasInv, which runs after
// ComputeMHDForces, and the brmn/bzmn round-trip through global memory is
// served from L2 (about 0.3% of wall on Ada), so the fusion is not worth a
// graph-level restructure.

// k_force_norm_partials: per surface jH, SINGLE THREAD serial kl-loop
// matching CPU's accumulation order exactly. The prior parallel-strided
// tree-reduce changed kl ordering vs CPU and compounded ULP rounding into
// the drift family. At nZnT = 24*14 = 336 the serial sum is cheap relative
// to the launch overhead, and the parallel-reduce gain was already swamped
// by the cross-config block schedule.
//   partial_RZ[jH] = (unique ? sum_kl guu*r12*r12*wInt[l] : 0)
//   partial_L[jH]  = (unique ? sum_kl (bsubu^2 + bsubv^2)*wInt[l] : 0)
// Batched execution: configuration axis on blockIdx.z. half-grid inputs per-config,
// partial outputs per-config profile.
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

// k_hybrid_lambda_force: per (jF_local_con, kl).
// jF_local_con is 0..ns_con_local (= nsMaxFIncludingLcfs - nsMinF).
// Reads inside half-grid (jF-1) and outside half-grid (jF) bsubu, bsubv, gvv,
// gsqrt, guv, bsupu. Edge cases:
//   jF == 0: no inside contribution (bsubv_i = 0, gvv_gsqrt_i = 0, etc.).
//   jF >= nsMaxH: no outside contribution.
// Output writes blmn_e/o and (lthreed) clmn_e/o at [jF_local_con * nZnT + kl].
// Batched execution: configuration axis on blockIdx.z. Half-grid inputs per-config; lu_e/o
// per-config full-grid; blmn/clmn per-config con-grid (matches d_blmn/d_clmn
// allocation in EnsureMHDForceBuffers). sqrtSF/sqrtSH/radialBlending shared.
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

// k_pres_compute: per surface jH, presH[jH] = massH[jH] / pow(dVdsH[jH], gamma).
// Batched execution: configuration axis on blockIdx.y. massH/dVdsH/presH per-config profiles.
__global__ void k_pres_compute(int n_config, int ns_h, double gamma,
                                const double* __restrict__ massH,
                                const double* __restrict__ dVdsH,
                                double* __restrict__ presH) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  int jH = blockIdx.x * blockDim.x + threadIdx.x;
  if (jH >= ns_h) return;
  size_t cfg = (size_t)config * (size_t)ns_h;
  presH[cfg + jH] = massH[cfg + jH] / pow(dVdsH[cfg + jH], gamma);
}

// k_pres_compute_and_thermal: fusion of k_pres_compute + k_pres_thermal_partial.
// Computes presH AND thermal_partial in one launch, reusing the presH value
// in-register instead of round-tripping through global memory.
// Saves 1 kernel launch + 1 global-memory read of presH per iter per config.
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

// k_pres_totalpres_init: per (jH, kl), totalPressure[i] = 0.5*(bsupu*bsubu+bsupv*bsubv).
// Batched execution: configuration axis on blockIdx.z. All buffers per-config half-grid.
__global__ void k_pres_totalpres_init(int n_config, int ns_h, int nZnT,
                                       const double* __restrict__ bsupu,
                                       const double* __restrict__ bsubu,
                                       const double* __restrict__ bsupv,
                                       const double* __restrict__ bsubv,
                                       double* __restrict__ totalPressure) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (jH >= ns_h || kl >= nZnT) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t i = cfg_half + (size_t)jH * (size_t)nZnT + (size_t)kl;
  totalPressure[i] = 0.5 * (bsupu[i] * bsubu[i] + bsupv[i] * bsubv[i]);
}

// k_pres_thermal_partial: per surface jH (single thread), partial[jH] =
// (unique ? presH[jH] * dVdsH[jH] : 0).
// Batched execution: configuration axis on blockIdx.y. All profiles per-config.
__global__ void k_pres_thermal_partial(int n_config, int ns_h,
                                        int nsMinH, int nsMaxH_minus_1,
                                        int ns_minus_2,
                                        const double* __restrict__ presH,
                                        const double* __restrict__ dVdsH,
                                        double* __restrict__ thermal_partial,
                                        const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.y;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int jH = blockIdx.x * blockDim.x + threadIdx.x;
  if (jH >= ns_h) return;
  size_t cfg = (size_t)config * (size_t)ns_h;
  int jH_global = jH + nsMinH;
  bool unique = (jH_global < nsMaxH_minus_1) || (jH_global == ns_minus_2);
  thermal_partial[cfg + jH] = unique ? (presH[cfg + jH] * dVdsH[cfg + jH]) : 0.0;
}

// k_pres_magnetic_partial: per surface jH (one block per surface, threads reduce
// kl), partial[jH] = (unique ? sum_kl gsqrt*totalPressure*wInt[l] : 0).
// Batched execution: configuration axis on blockIdx.z. gsqrt/totalPressure per-config
// half-grid; magnetic_partial per-config profile.
__global__ void k_pres_magnetic_partial(int n_config, int ns_h, int nZnT, int nThetaEff,
                                         int nsMinH, int nsMaxH_minus_1,
                                         int ns_minus_2,
                                         const double* __restrict__ gsqrt,
                                         const double* __restrict__ totalPressure,
                                         const double* __restrict__ wInt,
                                         double* __restrict__ magnetic_partial) {
  // Serial single-thread kl accumulation to match CPU's reduction order.
  int config = blockIdx.z;
  if (config >= n_config) return;
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
      acc += gsqrt[i] * totalPressure[i] * wInt[l];
    }
  }
  magnetic_partial[cfg_prof + jH] = acc;
}

// k_pres_magnetic_partial_inline: same reduction as above, but computes
// totalPressure inline from (bsupu*bsubu + bsupv*bsubv) instead of reading a
// precomputed buffer. Lets us defer the totalPressure write to a single fused
// kernel that combines magnetic + presH, dropping one kernel launch per iter.
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

// k_pres_totalpres_init_with_presH: fused replacement for
// k_pres_totalpres_init + k_pres_add_presH. Writes the final totalPressure
// (magnetic + thermal) in one pass, no intermediate write of magnetic-only.
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

// k_pres_add_presH: per (jH, kl), totalPressure[i] += presH[jH].
// Batched execution: configuration axis on blockIdx.z. presH per-config profile,
// totalPressure per-config half-grid.
__global__ void k_pres_add_presH(int n_config, int ns_h, int nZnT,
                                  const double* __restrict__ presH,
                                  double* __restrict__ totalPressure) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jH = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  if (jH >= ns_h || kl >= nZnT) return;
  size_t cfg_half = (size_t)config * (size_t)ns_h * (size_t)nZnT;
  size_t cfg_prof = (size_t)config * (size_t)ns_h;
  totalPressure[cfg_half + jH * nZnT + kl] += presH[cfg_prof + jH];
}

// k_volume_reduce produces a single output scalar per configuration
// by summing the half-grid differential-volume profile dVdsH against
// a caller-supplied multiplier. The reduction is restricted to the
// indices satisfying the single-rank uniqueness condition
//   jH < nsMaxH - 1  or  jH == ns - 2,
// which in the single-rank execution mode used here is satisfied for
// every jH in the range; the conditional is expressed through a mask
// helper to keep the kernel applicable to the multi-rank arrangement
// without modification. The configuration axis is carried on
// blockIdx.x, with one block reducing the profile of a single
// configuration. The input dVdsH is a per-configuration profile and
// the output out_scalar is a per-configuration scalar; the
// out_scalar buffer is sized n_config_max in the batched layout. At
// n_config equal to one the launch grid collapses to (1, 1, 1).
__global__ void k_volume_reduce(int n_config, int ns_h, double multiplier,
                                  int nsMaxH_minus_1, int ns_minus_2,
                                  int nsMinH,
                                  const double* __restrict__ dVdsH,
                                  double* __restrict__ out_scalar) {
  // Serial single-thread jH accumulation to match CPU's reduction order.
  int config = blockIdx.x;
  if (config >= n_config) return;
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

// k_bcontra_mutate_lambda: per (jF_local in [jF_first, jF_last_excl), kl).
//   lu_e[idx] = lu_e[idx]*lamscale + phipF[jF - nsMinH_off]
//   lu_o[idx] *= lamscale
//   lv_e[idx] *= lamscale (lthreed)
//   lv_o[idx] *= lamscale (lthreed)
// The phipF indexing in CPU is phipF[jF - nsMinH] (note: nsMinH, not nsMinF1).
// In single-rank nsMinH == nsMinF1 == 0 so phipF[jF_local] is correct;
// for multi-rank we pass an explicit offset.
// Batched execution: configuration axis on blockIdx.z. lu_e/o/lv_e/o per-config full-grid;
// phipF per-config full-grid profile.
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

// k_bcontra_bsupuv: per (jH_local, kl).
//   inside surface: jF_in = jH_local + jF_in_offset
//   outside surface: jF_out = jF_in + 1
// Reads lu_e/o, lv_e/o (already mutated by lamscale + phipF), sqrtSH, gsqrt.
// Writes bsupu (=0 for 2D, else lambda derivative average / gsqrt) and bsupv.
// Bsupu later gets chip/gsqrt added in k_bcontra_bsupu_add_chip.
// Batched execution: configuration axis on blockIdx.z. lu_e/o/lv_e/o per-config full-grid;
// sqrtSH/gsqrt half-grid; bsupu/bsupv per-config half-grid. sqrtSH shared.
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

  // Hoist the per-thread (cfg, jF, kl) -> linear index. nvcc's CSE through
  // chained subscript expressions is conservative; explicit hoist saves the
  // redundant integer add/mul each repeat.
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

// k_bcontra_jvplasma_reduce (ncurr==1): per surface jH, reduce
//   jvPlasma[jH]      = sum_kl (guu*bsupu + guv*bsupv) * wInt[l]   (3D)
//                     = sum_kl (guu*bsupu)             * wInt[l]   (2D)
//   avg_guu_gsqrt[jH] = sum_kl (guu / gsqrt)           * wInt[l]
// Batched execution: configuration axis on blockIdx.z. half-grid inputs per-config;
// jvPlasma/avg_guu_gsqrt per-config profile.
__global__ void k_bcontra_jvplasma_reduce(
    int n_config, int ns_h, int nZnT, int nThetaEff, bool lthreed,
    int serial_order,
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
  if (serial_order) {
    // Diagnostic: accumulate in the CPU loop's ascending-kl order so the
    // sums match the host bit for bit.
    if (threadIdx.x == 0) {
      double jv = 0.0, avg = 0.0;
      for (int kl = 0; kl < nZnT; ++kl) {
        size_t i = cfg_half + (size_t)jH_local * (size_t)nZnT + (size_t)kl;
        int l = kl % nThetaEff;
        double w = wInt[l];
        double term = guu[i] * bsupu[i];
        if (lthreed) {
          term += guv[i] * bsupv[i];
        }
        jv += term * w;
        avg += guu[i] / gsqrt[i] * w;
      }
      jvPlasma[cfg_prof + jH_local] = jv;
      avg_guu_gsqrt[cfg_prof + jH_local] = avg;
    }
    return;
  }
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

// k_bcontra_chipH_iotaH: per surface jH, update chipH (and iotaH if ncurr==1).
//   ncurr==1: chipH = (currH - jvPlasma) / avg_guu_gsqrt (if denom != 0);
//             iotaH = chipH / phipH (if phipH != 0).
//   ncurr==0: chipH = iotaH_in * phipH.
// Batched execution: configuration axis on blockIdx.y. All profiles per-config (ns_h).
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

// k_bcontra_chipF_iotaF: per surface jF_local in [0, ns_local).
//   Interior (jF in [nsMinFi, nsMaxFi)): midpoint average of chipH/iotaH.
//   Axis (jF == 0 when nsMinF1 == 0): iotaF[0] = 1.5*iotaH[0] - 0.5*iotaH[1].
//                                     (chipF[0] left as-is; CPU code does not set it here.)
//   LCFS (jF == ns-1 when nsMaxF1 == ns):
//     chipF[ns-1] = 2*chipH[ns_h-1] - chipH[ns_h-2]
//     iotaF[ns-1] = 1.5*iotaH[ns_h-1] - 0.5*iotaH[ns_h-2]
// Batched execution: configuration axis on blockIdx.y. chipH/iotaH per-config ns_h profile;
// chipF/iotaF per-config ns_local profile.
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
    iotaF[cfg_f + 0] = 1.5 * iotaH[cfg_h + 0] - 0.5 * iotaH[cfg_h + 1];
  }
  if (lcfs_present && jF_local == last_jF_local) {
    chipF[cfg_f + jF_local] = 2.0 * chipH[cfg_h + last_jH_local] - chipH[cfg_h + last_jH_local - 1];
    iotaF[cfg_f + jF_local] = 1.5 * iotaH[cfg_h + last_jH_local] - 0.5 * iotaH[cfg_h + last_jH_local - 1];
  }
}

// k_bcontra_bsupu_add_chip: per (jH, kl), bsupu[iHalf] += chipH[jH] / gsqrt[iHalf].
// Batched execution: configuration axis on blockIdx.z. chipH per-config profile; gsqrt/bsupu
// per-config half-grid.
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

// k_rzcon_into_volume: per (jF_con_local, kl) thread, copies rCon/zCon at the
// LCFS local index (passed in as lcfs_con_local) multiplied by sFull = sqrtSF^2
// for jF in [jMin_con, ns_con_local). Threads at jF < jMin_con are no-ops.
// Batched execution: configuration axis on blockIdx.z. rCon/zCon/rCon0/zCon0 per-config
// con-grid. sqrtSF shared.
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

// ============================================================================
// Inverse FFT (ForcesToFourier) kernels.
// Mirror of the forward fill+cuFFT+scatter pipeline, in the opposite direction:
//   1. k_inverse_fill: per (jF_local, m, q, k), Y[idx] = sum_l (force * basis_i),
//      where basis_i is the integration-weighted poloidal basis (cosmui/sinmui/
//      cosmumi/sinmumi). 12 slot types matching the forward kRmkcc..kLmkcsN.
//   2. cuFFT R2C: nZeta-length real-to-complex over (jF, m, q) batches → X.
//   3. k_inverse_scatter: per (jF_local, m, n), populate spec arrays (frcc, frss,
//      fzsc, fzcs, flsc, flcs) from X with the inverse of the forward scaling.
// ============================================================================

// k_inverse_fill: real-space force arrays → Y[jF, m, q, k] (real, length nZeta).
// Batched execution: configuration axis on blockIdx.z = config * ns_local + jF_local.
// Per-config: force arrays (ns_force_local * nZnT), con arrays (ns_con_local *
// nZnT), Y (ns_local * mpol * kBatch * nZeta). xmpq, cosmui/sinmui/cosmumi/
// sinmumi shared.
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

// k_inverse_scatter: cuFFT R2C output → spec arrays.
// Honors the CPU's range split: RZ forces (frcc/frss/fzsc/fzcs) written for
// jF in [nsMinF, jMaxRZ); lambda forces (flsc/flcs) written for jF in
// [max(nsMinF,jMinL), nsMaxFIncludingLcfs). Outside those, write 0 to match
// FourierForces.setZero() that CPU does upfront.
// Batched execution: configuration axis on blockIdx.z = config * ns_local + jF_local.
// X per-config (n_config * ns_local * mpol * kBatch * nhalf); fxxx per-config
// (n_config * ns_local * mpol * (ntor+1)). nscale shared.
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

  // Mirror of CPU FFTX accumulate (with cuFFT-vs-FFTX nscale[n] multiply):
  //   frcc =  ns_n * (X_rcc.re   + nfp_n * X_rccN.im)
  //   frss =  ns_n * (-X_rss.im  + nfp_n * X_rssN.re)
  //   fzsc =  ns_n * (X_zsc.re   + nfp_n * X_zscN.im)
  //   fzcs =  ns_n * (-X_zcs.im  + nfp_n * X_zcsN.re)
  //   flsc =  ns_n * (X_lsc.re   + nfp_n * X_lscN.im)
  //   flcs =  ns_n * (-X_lcs.im  + nfp_n * X_lcsN.re)
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

// k_compute_ru_zu_full: post-forward-FFT combine producing ruFull, zuFull at
// each (jF_con, kl) where jF_con ranges over [nsMinF .. nsMaxFIncludingLcfs).
// ruFull[idx] = ru_e[jF_local_full * nZnT + kl] + sqrtSF[jF_local_full] * ru_o[...]
// Stores into d_ruFull, d_zuFull which are ns_con_local × nZnT.
// Batched execution: configuration axis on blockIdx.z. ru_e/o/zu_e/o per-config full-grid;
// ruFull/zuFull per-config con-grid; sqrtSF shared.
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

// k_constraint_force_multiplier: per surface jF (one block, threads reduce kl).
// Reduces (ruFull² * wInt) and (zuFull² * wInt) over kl, computes tcon[jF].
// Caller does the LCFS halving on host.
// Batched execution: configuration axis on blockIdx.z. ruFull/zuFull per-config con-grid;
// tcon per-config profile; ard/azd/wInt shared.
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
                                                double* __restrict__ tcon) {
  int config = blockIdx.z;
  if (config >= n_config) return;
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

// k_halve_tcon_lcfs: one thread per config, halves d_tcon[last] on the LCFS-
// owning rank. Replaces the host-side halving that was D2H'd but never H2D'd back.
// Batched execution: launch n_config_max blocks, each writes its own slot.
__global__ void k_halve_tcon_lcfs(int n_config, int tcon_stride,
                                    int last_idx,
                                    double* __restrict__ tcon) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (threadIdx.x != 0) return;
  size_t cfg = (size_t)config * (size_t)tcon_stride;
  tcon[cfg + last_idx] = 0.5 * tcon[cfg + last_idx - 1];
}

// k_effective_constraint_force: per (jF_local_con, kl).
//   gConEff[idx] = (rCon - rCon0) * ruFull + (zCon - zCon0) * zuFull
// Threads at jF_local_con < jMin are no-ops.
// Batched execution: configuration axis on blockIdx.z. All buffers per-config con-grid.
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

// k_assemble_total_forces: per (jF_local_force, kl). Adds constraint force into
// brmn_e/o, bzmn_e/o, and writes frcon_e/o, fzcon_e/o.
// Free-boundary edge contribution (lfreeb && r.nsMaxF1 == fc.ns) is handled in
// host code before launch via separate small kernel; here we just do the bulk.
// Batched execution: configuration axis on blockIdx.z. rCon/rCon0/zCon/zCon0/gCon/ruFull/
// zuFull are per-config con-grid. brmn/bzmn/frcon/fzcon are per-config
// force-grid. sqrtSF shared.
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

// k_compute_bco: per (jH_local, kl) thread, computes bsubu = guu*bsupu + guv*bsupv,
// bsubv = guv*bsupu + gvv*bsupv (3D) or bsubu=guu*bsupu, bsubv=gvv*bsupv (2D).
// Batched execution: configuration axis on blockIdx.z. All half-grid buffers per-config.
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

// k_apply_m1_preconditioner: per (jF_local, n), m=1 only. Scale frss/fzcs by
// forceScaleR/Z derived from (ard+brd) / (ard+brd+azd+bzd).
// Batched execution: configuration axis on blockIdx.z. frss/fzcs per-config spectra.
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
  // Per-configuration indexing extension: d_pmat_ard/brd/azd/bzd are per-cfg snapshots from
  // ComputePreconditioningMatrixCuda; read with cfg offset. Previously read
  // cfg=0's slot for all cfgs; correct for broadcast (cfg>=1 SHOULD have
  // cfg=0's value), wrong for distinct-input execution.
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

// k_apply_lambda_preconditioner: per (jF_local, m, n), scale flsc/flcs by
// lambdaPreconditioner[idx_mn].
// Batched execution: configuration axis on blockIdx.z = config * ns_con_local + jF_local.
// flsc/flcs per-config spectra. lambdaPreconditioner is shared (radial-grid).
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
  // Per-configuration indexing extension: lambdaPreconditioner is per-cfg (matched by
  // k_ulp_assemble's gain of n_config dim). Previously this read cfg=0's
  // slot only; correct in broadcast since cfg>=1 SHOULD have cfg=0's
  // value, but wrong for distinct-input execution with per-cfg lambdaPrec.
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

// k_rz_transpose_in: spec (jF_local, m, n) → Thomas (mn, basis, jF_global).
// jF_global = jF_local + nsMinF; rows outside the force range are zero-padded.
// Batched execution: configuration axis on blockIdx.z. Spectra per-config (ns_local *
// mnsize). Thomas buffer cR/cZ per-config (mnsize * num_basis * ns_total).
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

// k_rz_transpose_out: Thomas (mn, basis, jF_global) → spec (jF_local, m, n).
// Batched execution: configuration axis on blockIdx.z. Spectra per-config (ns_local *
// mnsize). cR/cZ per-config (mnsize * num_basis * ns_total).
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

// k_dealias_fwd: per (jF_local, m, n), compute gsc/gcs intermediates.
//   gsc[jF, m, n] = tcon[jF] * sum_{k, l} gConEff[jF, k, l] * sinmui[m, l] * cosnv[k, n]
//   gcs[jF, m, n] = tcon[jF] * sum_{k, l} gConEff[jF, k, l] * cosmui[m, l] * sinnv[k, n]
// m in [1, mpol-1); for m=0 and m=mpol-1 outputs are not used. Zeros m=0/last.
// Batched execution: configuration axis on blockIdx.z = config * ns_force_local + jF.
// gConEff per-config con-grid (ns_con_local * nZnT). tcon per-config con-profile
// (ns_con_local). gsc/gcs per-config force spectra (ns_force_local * mpol *
// (ntor+1)).
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

// k_dealias_inv: per (jF, k, l), accumulate m_gCon contributions across (m, n).
//   m_gCon[jF, k, l] = sum_{m=1..mpol-1} faccon[m] *
//                       sum_{n=0..ntor} (gsc[jF, m, n] * cosnv[k, n] * sinmu[m, l] +
//                                        gcs[jF, m, n] * sinnv[k, n] * cosmu[m, l])
// Batched execution: configuration axis on blockIdx.z = config * ns_force_local + jF.
// gsc/gcs per-config force spectra; m_gCon per-config force-grid.
// sinmu/cosmu/cosnv/sinnv/faccon shared.
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
  // The outer loop over the poloidal mode index has a runtime iteration
  // count of mpol - 2 but contains an early-skip branch on fac == 0
  // that prevents the compiler from unrolling the loop on its own.
  // Annotating the loop with an explicit unroll directive allows the
  // compiler to overlap the cosnv and sinnv loads of the inner
  // toroidal-mode loop across the unrolled outer iterations, reducing
  // the number of dynamic instructions issued per warp.
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




// k_decompose_into: per (jF_dec, m, n), scale physical → decomposed.
// Mirrors FourierCoeffs::decomposeInto for stellarator-symmetric (lasym=false).
// jF range is [nsMin, jMaxIncludingBoundary). All RZ entries are written
// (jMaxRZ == jMaxIncludingBoundary in the CPU code for our use case).
// Batched execution: configuration axis on blockIdx.z = config * ns_dec_local + jF_dec.
// phys/dec spectra per-config (ns_local * mpol * (ntor+1)). scalxc shared
// (radial grid factor, same for all configs).
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

// k_m1_constraint: per (jF_dec, n) at m=1 only. lthreed only (mirrors CPU
// FourierCoeffs::m1Constraint; lasym branch omitted).
//   old_rss = dec_frss[idx]
//   dec_frss[idx] = (old_rss + dec_fzcs[idx]) * scalingFactor
//   dec_fzcs[idx] = (old_rss - dec_fzcs[idx]) * scalingFactor
// Batched execution: configuration axis on blockIdx.z. dec_frss/dec_fzcs per-config spectra.
__global__ void k_m1_constraint(int n_config, int ns_local,
                                  int ns_force_local, int mpol, int ntor,
                                  double scalingFactor,
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
  dec_frss[idx] = (old_rss + old_zcs) * scalingFactor;
  dec_fzcs[idx] = (old_rss - old_zcs) * scalingFactor;
}

// k_m1_constraint_and_zero: fusion of k_m1_constraint + k_zero_z_force_for_m1.
// The original sequence at m=1 (lthreed) was:
//   dec_frss := (dec_frss + dec_fzcs) * sf   (m1_constraint)
//   dec_fzcs := (dec_frss_orig - dec_fzcs) * sf  (m1_constraint)
//   dec_fzcs := 0                            (zero_z_force_for_m1)
// The fzcs computation in m1_constraint is dead code, overwritten to 0
// immediately. The fused kernel does only the live work:
//   dec_frss := (dec_frss + dec_fzcs) * sf
//   dec_fzcs := 0
// Saves one launch per iter and ~half the global memory ops on dec_fzcs.
__global__ void k_m1_constraint_and_zero(int n_config, int ns_local,
                                          int ns_force_local, int mpol, int ntor,
                                          double scalingFactor,
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
  dec_frss[idx] = (old_rss + old_zcs) * scalingFactor;
  dec_fzcs[idx] = 0.0;
}

// k_zero_z_force_for_m1: zero dec_fzcs[m=1, all n, all jF in [nsMinF, nsMaxF)].
// lthreed only (CPU FourierForces::zeroZForceForM1).
// Batched execution: configuration axis on blockIdx.z. dec_fzcs per-config spectra.
__global__ void k_zero_z_force_for_m1(int n_config, int ns_local,
                                        int ns_force_local, int mpol, int ntor,
                                        double* __restrict__ dec_fzcs) {
  int config = blockIdx.z;
  if (config >= n_config) return;
  int jF_force = blockIdx.y;
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF_force >= ns_force_local || n > ntor) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  size_t idx = cfg_spec + (size_t)((jF_force * mpol + 1) * (ntor + 1) + n);
  dec_fzcs[idx] = 0.0;
}

// k_zero_buffer: small utility, sets first n doubles to 0.
__global__ void k_zero_buffer(int n, double* __restrict__ p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 0.0;
}

// k_pres_final_reduce: one block per config, threads cooperate to sum
// thermal_partial and magnetic_partial across all ns_h surfaces, write 3 scalars
// [thermal, magnetic, mhd] to scalars_out on device. Eliminates the host-side
// accumulation + stream sync in PressureAndEnergiesCuda.
// Batched execution: n_config via blockIdx.x. scalars_out sized n_config*3.
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

// k_cfg01_max_abs_diff is a diagnostic kernel that compares two
// consecutive per-configuration slices of a device buffer and emits
// the largest pointwise absolute difference between them. For an
// input buffer of size 2 * per_cfg_size the kernel evaluates
//   max over i in [0, per_cfg_size)  |buf[i] - buf[per_cfg_size + i]|,
// which under the batched layout corresponds to the divergence
// between the configuration-zero and configuration-one slices of any
// per-configuration device buffer. A single block of 256 threads
// performs the reduction and writes the result to out_scalar as a
// single double. The kernel is invoked through DiagCfg01DiffCuda when
// localising the kernel responsible for an unintended divergence
// between configuration zero and configuration one in a batched run
// with n_config_max equal to two.
__global__ void k_cfg01_max_abs_diff(int per_cfg_size,
                                      const double* __restrict__ buf,
                                      double* __restrict__ out_scalar) {
  __shared__ double s_max[256];
  double m = 0.0;
  for (int i = threadIdx.x; i < per_cfg_size; i += blockDim.x) {
    double d = buf[i] - buf[per_cfg_size + i];
    double a = (d < 0.0) ? -d : d;
    if (a > m) m = a;
  }
  s_max[threadIdx.x] = m;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (s_max[threadIdx.x + stride] > s_max[threadIdx.x]) {
        s_max[threadIdx.x] = s_max[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out_scalar[0] = s_max[0];
}

// k_rznorm_pts_x_partials: bit-exact mirror of FourierCoeffs::rzNorm on the
// device-resident position arrays d_pts_x_rcc/zsc/rss/zcs for config 0 only.
// (Host m_decomposed_x is a cfg=0-only shadow under batched broadcast.)
// include_offset=false (the only consumer in computeForceNorms is the false
// branch).
//
// CPU rzNorm reduction order is jF-outer, m-middle, n-inner, sequential FP
// accumulation. To match bit-exactly: one block per jF, single thread does the
// nested mn-loop in CPU order. Host accumulates the per-jF partials in
// jF-order. The prior single-block 256-thread tree-reduction scaffold (this
// kernel was named k_rznorm_pts_x) drifted at the 5th-7th significant digit
// because tree-pair reduction differs in FP rounding from sequential.
__global__ void k_rznorm_pts_x_partials(
    int ns_local, int mpol, int ntor,
    int nsMinHere_local, int nsMaxHere_local, bool lthreed,
    const double* __restrict__ x_rcc, const double* __restrict__ x_zsc,
    const double* __restrict__ x_rss, const double* __restrict__ x_zcs,
    double* __restrict__ partials) {
  int jF_off = blockIdx.x;
  int num_jFs = nsMaxHere_local - nsMinHere_local;
  if (jF_off >= num_jFs) return;
  if (threadIdx.x != 0) return;
  int jF_local = jF_off + nsMinHere_local;
  double s = 0.0;
  for (int m = 0; m < mpol; ++m) {
    for (int n = 0; n <= ntor; ++n) {
      size_t idx_fc = (size_t)((jF_local * mpol + m) * (ntor + 1) + n);
      if (n > 0 || m > 0) {  // include_offset=false: skip rcc at (0,0)
        double r = x_rcc[idx_fc];
        s += r * r;
      }
      double z = x_zsc[idx_fc];
      s += z * z;
      if (lthreed) {
        double rs = x_rss[idx_fc];
        s += rs * rs;
        double zc = x_zcs[idx_fc];
        s += zc * zc;
      }
    }
  }
  partials[jF_off] = s;
}

// k_force_norm_final_reduce: one block per config, threads cooperate to sum
// Per-cfg rzNorm for the device time-step controller: one serial thread
// per cfg accumulates the squared R/Z position coefficients over the force
// extent in the same coefficient order as FourierCoeffs::rzNorm
// (include_offset=false; lasym excluded by the build's scope guard), so
// cfg 0 matches the host scalar bit-for-bit. Writes the reciprocal
// (fNorm1) directly into d_fnorm1.
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

// the per-jH forceNormRZ_partial and forceNormL_partial arrays into 2 scalars
// on device. Replaces the ns_h-D2H + host accumulator loop in
// ComputeForceNormsCuda.
// Batched execution: n_config via blockIdx.x. scalars_out sized n_config*2.
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

// k_residuals: one block, ONE THREAD per config; performs the three
// sum-of-squares reductions in the exact (jF, m, n) order CPU uses, with
// the per-index accumulation order CPU uses (rcc, zsc, lsc, then under
// lthreed rss, zcs, lcs). Mirrors FourierForces::residuals() in serial.
// Parallel reduction over indices was reordered relative to the CPU and
// compounded ULP-level rounding into the sub-percent drift family across
// ~10^4 iters. The serial single-thread accumulation here matches CPU
// bit-for-bit per call, eliminating that source of drift. Performance
// cost is small at this kernel's tiny problem size (mpol*(ntor+1)*ns
// ~ 2750 squared-sums per cfg); the kernel was already memory-bound.
//
// Stellarator-symmetric path only (lasym=false). The (jLocal_max_rz,
// jLocal_max_boundary) integer pair encodes the CPU's (jMaxRZ,
// jMaxIncludeBoundary) thresholds shifted by nsMin_.
// Batched execution: n_config via blockIdx.x (one block per config). spectra
// per-config (ns_local * mpol * (ntor+1)); scalars_out sized n_config*3.
__global__ void k_residuals(
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
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  if (threadIdx.x != 0) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  double acc_R = 0.0, acc_Z = 0.0, acc_L = 0.0;
  // CPU loop: for jF in [0, jLocal_max_boundary), for m in [0, mpol),
  // for n in [0, ntor+1). Within each (jF, m, n) the per-component order
  // is rcc, zsc, lsc, then (lthreed) rss, zcs, lcs.
  for (int jLocal = 0; jLocal < jLocal_max_boundary; ++jLocal) {
    const bool in_rz = (jLocal < jLocal_max_rz);
    for (int m = 0; m < mpol; ++m) {
      for (int n = 0; n < ntor + 1; ++n) {
        int idx = (jLocal * mpol + m) * (ntor + 1) + n;
        size_t cidx = cfg_spec + (size_t)idx;
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
      }  // n
    }  // m
  }  // j
  scalars_out[config * 3 + 0] = acc_R;
  scalars_out[config * 3 + 1] = acc_Z;
  scalars_out[config * 3 + 2] = acc_L;
}

// k_residuals_par: parallel FP64 reduction matching k_residuals output
// for VMEC's stop gate. Same math as the serial path but each block
// (one per config) splits the (jLocal, m, n) cell list across 256
// threads. Each thread accumulates its strided subset; shared-memory
// tree reduce produces the final per-config (acc_R, acc_Z, acc_L).
//
// Reduction order differs from the serial path by a few ULPs per call,
// which compounds to <1e-10 relative drift on aspect_ratio over ~10^4
// iters. That's well below the CPU↔CUDA drift family floor (1e-3 to
// 1e-5 on the field-line metrics), so the change is invisible at
// the production metric tolerances. The serial path remains the default; gated
// by VMECPP_RESIDUALS_PAR=1.

// k_check_convergence: device-side per-cfg convergence check on NORMALIZED
// residuals, mirroring the per-cfg arithmetic of
// IdealMhdModel::evalFResInvar:
//   energyDensity_c = max(magnetic_c, thermal_c) / plasmaVolume_c
//   fNormRZ_c = 1 / (sum_rz_c * energyDensity_c^2)
//   fNormL_c  = 1 / (sum_l_c * lamscale^2)
//   fsqr_c = fResR_c * fNormRZ_c * r1scale     (r1scale = 0.25)
//   fsqz_c = fResZ_c * fNormRZ_c * r1scale
//   fsql_c = fResL_c * fNormL_c
// and comparing all three normalized values against ftolv. The inputs are
// the same persistent device buffers whose D2H copies feed the host gate:
// raw residual triples (scalars_out), force-norm sums (fnorm_scalars,
// refreshed every preconditioner-update interval), energy scalars
// (pressure_scalars, per-iteration), and per-cfg plasma volumes (volumes,
// per-iteration). The staleness profile therefore matches the host gate
// exactly: interval-stale sums combined with current-iteration energies
// and volumes. When any normalization input is unavailable (null pointer
// or non-positive lamscale) the comparison falls back to the raw
// residuals, preserving the kernel's previous behavior.
//
// Launch: 1 block per cfg, 1 thread per block.
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
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
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

// Multi-block parallel residuals: K sub-blocks per cfg each reduce one slice
// of the (jLocal, m, n) index space. Output to partials_out[K * n_config * 3]
// in layout [cfg * K * 3 + partition * 3 + comp]. A subsequent finalize
// kernel reduces across the K axis.
//
// Rationale: k_residuals_par launches 1 block per cfg = 1/142 SM at N=1.
// Splitting into K=16 sub-blocks puts 16 SMs to work per cfg at N=1.
// Each block sweeps (1/K)-th of the total elements, so per-block work is
// 1/K of single-block path; with K blocks running concurrently the wall
// time drops to ~1/K modulo finalize overhead.
//
// Bit-exact: deterministic partition order (partition_idx is contiguous slice
// of the flattened (jLocal, m, n) index), and the finalize kernel sums in
// fixed order [partition=0..K-1]. Same arithmetic as the single-block sweep
// would do, with the same operands in (slightly) different summation order;
// final sum differs from k_residuals_par by accumulation-order rounding
// only, which lands within the existing drift family.
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
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.y;
  int partition = blockIdx.x;
  if (config >= n_config) return;
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

// k_residuals_finalize_K: collapse K partials per cfg into one triple.
// Grid: (n_config), TPB=K (must be <= 32; we use kResidualsKPartitions=16).
// Single warp tree-reduces the K partials into scalars_out[cfg*3..cfg*3+2].
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
  // Warp-level tree reduce (assumes blockDim.x <= 32, single warp).
  // Use shfl_xor_sync for the butterfly. The reduction order is
  // deterministic (always the same butterfly pattern), so multi-run
  // bit-exact.
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

// k_update_timestep: on-device time-step damping computation.
//
// Mirrors the host-side damping block in Vmec::Evolve:
//   fsq1 = fsqr1 + fsqz1 + fsql1                      (precd residual sum)
//   if iter2 == iter1:                                 (start of damped segment)
//       invTau.setConstant(0.15 / time_step)
//   shift invTau left by 1 (drop oldest sample)
//   if iter2 > iter1:                                  (have a previous fsq to compare)
//       invtau_num = min(|log(fsq1 / prev_fsq)|, 0.15)
//       invTau[N-1] = invtau_num / time_step
//   prev_fsq = fsq1
//   otav = sum(invTau) / kNDamp
//   dtau = time_step * otav / 2
//   b1 = 1 - dtau
//   fac = 1 / (1 + dtau)
//
// One CUDA block per cfg, kTimestepNDamp threads (=10). Each thread holds one
// entry of the ring buffer in a register; the shift-and-update is done via
// shared memory between threads. Final reduction uses warp shfl_xor (10
// elements fits trivially in one warp).
//
// iter_phase encoding:
//   0  : iter2 == iter1 (start of a damped segment; reset invTau, prev_fsq)
//   1  : iter2 > iter1  (normal update with log of fsq ratio)
//
// d_residuals_partial layout: [cfg*3 + comp] where comp = 0..2 is R/Z/L.
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
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  const int cfg = blockIdx.x;
  const int tid = threadIdx.x;
  if (cfg >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[cfg]) {
    // Inactive cfg: leave d_fac_b1 at its prior value; the per-cfg
    // active mask gates downstream kernel application.
    return;
  }
  constexpr int kND = 10;  // matches host kNDamp
  // fsq1 = fsqr1 + fsqz1 + fsql1 with the host normalization from
  // evalFResPrecd (fNorm1 on the R and Z raw sums, deltaS on the L raw
  // sum), in the host association order: each component normalized
  // first, then summed left to right.
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
  if (iter_phase == 0) {
    if (tid < kND) entry = 0.15 / time_step;
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
      new_entry = invtau_num / time_step;
    }
    new_entry = __shfl_sync(0xffffffff, new_entry, 0);
    if (tid == kND - 1) entry = new_entry;
  }
  // Write back to ring buffer
  if (tid < kND) inv_tau_cfg[tid] = entry;
  __syncwarp();
  // Reduction must match host Eigen's left-to-right sequential sum to keep
  // the bit-exact contract on fac / b1. Eigen VectorXd::sum() is a left-to-
  // right reduction over the contiguous storage. Warp-shfl reduction adds
  // in a different operand order, producing ~1e-5 ULP drift vs host. Use
  // single-thread sequential sum on tid==0; kND=10 makes this trivial.
  if (tid == 0) {
    double sum = 0.0;
    #pragma unroll
    for (int i = 0; i < kND; ++i) {
      sum += inv_tau_cfg[i];
    }
    double otav = sum / kND;
    double dtau = time_step * otav / 2.0;
    double b1 = 1.0 - dtau;
    double fac = 1.0 / (1.0 + dtau);
    d_fac_b1[(size_t)cfg * 2 + 0] = fac;
    d_fac_b1[(size_t)cfg * 2 + 1] = b1;
    // Update prev_fsq for next iter
    d_prev_fsq[cfg] = fsq1;
  }
}

// k_residuals_dd_fp32: FP32 substitution probe of k_residuals. Loads spec
// values as FP64, casts to FP32, squares in FP32 with native fp32 mul, and
// accumulates the running sum in a DD-pair (fp32 hi + fp32 lo) so the
// accumulator carries ~48 bits of mantissa. Final output is cast back to
// FP64 via dd_to_double for compatibility with the existing per-cfg cache.
//
// Phase 1 of the FP32 substitution research path: validate the DD-pair
// primitives on the simplest serial accumulator (k_residuals), measure
// drift vs the FP64 production path, and use the result to size the
// rollout to k_force_norm_partials / k_pres_magnetic_partial / etc.
//
// Gated by VMECPP_RESIDUALS_DD_FP32=1. Default OFF.
__global__ void k_residuals_dd_fp32(
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
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.x;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  if (threadIdx.x != 0) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_local *
                    (size_t)mpol * (size_t)(ntor + 1);
  DD acc_R = dd_from_f(0.0f), acc_Z = dd_from_f(0.0f), acc_L = dd_from_f(0.0f);
  for (int jLocal = 0; jLocal < jLocal_max_boundary; ++jLocal) {
    const bool in_rz = (jLocal < jLocal_max_rz);
    for (int m = 0; m < mpol; ++m) {
      for (int n = 0; n < ntor + 1; ++n) {
        int idx = (jLocal * mpol + m) * (ntor + 1) + n;
        size_t cidx = cfg_spec + (size_t)idx;
        float v;
        if (in_rz) {
          v = (float)frcc[cidx]; acc_R = dd_add_f(acc_R, v * v);
          v = (float)fzsc[cidx]; acc_Z = dd_add_f(acc_Z, v * v);
        }
        v = (float)flsc[cidx]; acc_L = dd_add_f(acc_L, v * v);
        if (lthreed) {
          if (in_rz) {
            v = (float)frss[cidx]; acc_R = dd_add_f(acc_R, v * v);
            v = (float)fzcs[cidx]; acc_Z = dd_add_f(acc_Z, v * v);
          }
          v = (float)flcs[cidx]; acc_L = dd_add_f(acc_L, v * v);
        }
      }
    }
  }
  scalars_out[config * 3 + 0] = dd_to_double(acc_R);
  scalars_out[config * 3 + 1] = dd_to_double(acc_Z);
  scalars_out[config * 3 + 2] = dd_to_double(acc_L);
}

// k_scatter_main_and_con_dd_fp32: FP32 inner-multiplication variant of the
// scatter step (spec→geometry). Inputs (Y, cosmu, sinmu, cosmum, sinmum)
// are loaded as FP64 but cast to FP32 inside the inner loop; the 18 per-
// output running sums (r1_e/o, ru_e/o, rv_e/o, z1_e/o, zu_e/o, zv_e/o,
// lu_e/o, lv_e/o, rCon, zCon) are kept as DD-pairs so the FP32 product
// stream accumulates without √n amplification of the FP32 rounding.
//
// One thread per (cfg, jF_local, k, l) output. Serial m-loop within each
// thread; no cross-thread reduction. cuFFT remains in FP64 (this kernel
// is independent of VMECPP_FFT_FP32). Gated by VMECPP_SCATTER_DD_FP32=1;
// default OFF.
__global__ void k_scatter_main_and_con_dd_fp32(
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
  int z = blockIdx.z;
  int config = z / ns_local;
  int jF_local = z - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nZeta || l >= nThetaReduced) return;

  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;

  DD r1e = dd_from_f(0.0f), r1o = dd_from_f(0.0f);
  DD rue = dd_from_f(0.0f), ruo = dd_from_f(0.0f);
  DD rve = dd_from_f(0.0f), rvo = dd_from_f(0.0f);
  DD z1e = dd_from_f(0.0f), z1o = dd_from_f(0.0f);
  DD zue = dd_from_f(0.0f), zuo = dd_from_f(0.0f);
  DD zve = dd_from_f(0.0f), zvo = dd_from_f(0.0f);
  DD lue = dd_from_f(0.0f), luo = dd_from_f(0.0f);
  DD lve = dd_from_f(0.0f), lvo = dd_from_f(0.0f);
  DD rcon = dd_from_f(0.0f), zcon = dd_from_f(0.0f);

  float sqrtSF_jF = (float)sqrtSF[jF_local];

  for (int m = 0; m < mpol; ++m) {
    const size_t y_base = cfg_Y + (size_t)((jF_local * mpol + m) * kBatch) *
                          (size_t)nZeta + (size_t)k;
    float rmkcc  = (float)Y[y_base + (size_t)kRmkcc  * (size_t)nZeta];
    float rmkss  = (float)Y[y_base + (size_t)kRmkss  * (size_t)nZeta];
    float rmkccN = (float)Y[y_base + (size_t)kRmkccN * (size_t)nZeta];
    float rmkssN = (float)Y[y_base + (size_t)kRmkssN * (size_t)nZeta];
    float zmksc  = (float)Y[y_base + (size_t)kZmksc  * (size_t)nZeta];
    float zmkcs  = (float)Y[y_base + (size_t)kZmkcs  * (size_t)nZeta];
    float zmkscN = (float)Y[y_base + (size_t)kZmkscN * (size_t)nZeta];
    float zmkcsN = (float)Y[y_base + (size_t)kZmkcsN * (size_t)nZeta];
    float lmksc  = (float)Y[y_base + (size_t)kLmksc  * (size_t)nZeta];
    float lmkcs  = (float)Y[y_base + (size_t)kLmkcs  * (size_t)nZeta];
    float lmkscN = (float)Y[y_base + (size_t)kLmkscN * (size_t)nZeta];
    float lmkcsN = (float)Y[y_base + (size_t)kLmkcsN * (size_t)nZeta];

    int bml = m * nThetaReduced + l;
    float cmu  = (float)cosmu[bml];
    float smu  = (float)sinmu[bml];
    float cmum = (float)cosmum[bml];
    float smum = (float)sinmum[bml];
    bool m_even = ((m & 1) == 0);

    float r1_c = rmkcc * cmu  + rmkss * smu;
    float ru_c = rmkcc * smum + rmkss * cmum;
    float rv_c = rmkccN * cmu + rmkssN * smu;
    float z1_c = zmksc * smu  + zmkcs * cmu;
    float zu_c = zmksc * cmum + zmkcs * smum;
    float zv_c = zmkscN * smu + zmkcsN * cmu;
    float lu_c = lmksc * cmum + lmkcs * smum;
    float lv_c = -(lmkscN * smu + lmkcsN * cmu);

    if (m_even) {
      r1e = dd_add_f(r1e, r1_c); rue = dd_add_f(rue, ru_c);
      rve = dd_add_f(rve, rv_c);
      z1e = dd_add_f(z1e, z1_c); zue = dd_add_f(zue, zu_c);
      zve = dd_add_f(zve, zv_c);
      lue = dd_add_f(lue, lu_c); lve = dd_add_f(lve, lv_c);
    } else {
      r1o = dd_add_f(r1o, r1_c); ruo = dd_add_f(ruo, ru_c);
      rvo = dd_add_f(rvo, rv_c);
      z1o = dd_add_f(z1o, z1_c); zuo = dd_add_f(zuo, zu_c);
      zvo = dd_add_f(zvo, zv_c);
      luo = dd_add_f(luo, lu_c); lvo = dd_add_f(lvo, lv_c);
    }

    float con_factor = m_even ? (float)xmpq[m] : (float)xmpq[m] * sqrtSF_jF;
    rcon = dd_add_f(rcon, r1_c * con_factor);
    zcon = dd_add_f(zcon, z1_c * con_factor);
  }

  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] = dd_to_double(r1e); r1_o[idx] = dd_to_double(r1o);
  ru_e[idx] = dd_to_double(rue); ru_o[idx] = dd_to_double(ruo);
  rv_e[idx] = dd_to_double(rve); rv_o[idx] = dd_to_double(rvo);
  z1_e[idx] = dd_to_double(z1e); z1_o[idx] = dd_to_double(z1o);
  zu_e[idx] = dd_to_double(zue); zu_o[idx] = dd_to_double(zuo);
  zv_e[idx] = dd_to_double(zve); zv_o[idx] = dd_to_double(zvo);
  lu_e[idx] = dd_to_double(lue); lu_o[idx] = dd_to_double(luo);
  lv_e[idx] = dd_to_double(lve); lv_o[idx] = dd_to_double(lvo);
  rCon[idx] = dd_to_double(rcon);
  zCon[idx] = dd_to_double(zcon);
}

// Path 1: FP64 multiplications + DD-pair accumulators. Same structure as
// k_scatter_main_and_con_dd_fp32 but the inner mults run in native FP64
// (no FP32 quantization). The DD accumulator catches √n drift in the
// running sum. Bit-exact-with-FP64 expected at the output. Gated by
// VMECPP_SCATTER_DD_FP64MUL=1.
__global__ void k_scatter_main_and_con_dd_fp64mul(
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
  int z = blockIdx.z;
  int config = z / ns_local;
  int jF_local = z - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nZeta || l >= nThetaReduced) return;

  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;

  DD r1e = dd_from_f(0.0f), r1o = dd_from_f(0.0f);
  DD rue = dd_from_f(0.0f), ruo = dd_from_f(0.0f);
  DD rve = dd_from_f(0.0f), rvo = dd_from_f(0.0f);
  DD z1e = dd_from_f(0.0f), z1o = dd_from_f(0.0f);
  DD zue = dd_from_f(0.0f), zuo = dd_from_f(0.0f);
  DD zve = dd_from_f(0.0f), zvo = dd_from_f(0.0f);
  DD lue = dd_from_f(0.0f), luo = dd_from_f(0.0f);
  DD lve = dd_from_f(0.0f), lvo = dd_from_f(0.0f);
  DD rcon = dd_from_f(0.0f), zcon = dd_from_f(0.0f);

  double sqrtSF_jF = sqrtSF[jF_local];

  for (int m = 0; m < mpol; ++m) {
    const size_t y_base = cfg_Y + (size_t)((jF_local * mpol + m) * kBatch) *
                          (size_t)nZeta + (size_t)k;
    double rmkcc  = Y[y_base + (size_t)kRmkcc  * (size_t)nZeta];
    double rmkss  = Y[y_base + (size_t)kRmkss  * (size_t)nZeta];
    double rmkccN = Y[y_base + (size_t)kRmkccN * (size_t)nZeta];
    double rmkssN = Y[y_base + (size_t)kRmkssN * (size_t)nZeta];
    double zmksc  = Y[y_base + (size_t)kZmksc  * (size_t)nZeta];
    double zmkcs  = Y[y_base + (size_t)kZmkcs  * (size_t)nZeta];
    double zmkscN = Y[y_base + (size_t)kZmkscN * (size_t)nZeta];
    double zmkcsN = Y[y_base + (size_t)kZmkcsN * (size_t)nZeta];
    double lmksc  = Y[y_base + (size_t)kLmksc  * (size_t)nZeta];
    double lmkcs  = Y[y_base + (size_t)kLmkcs  * (size_t)nZeta];
    double lmkscN = Y[y_base + (size_t)kLmkscN * (size_t)nZeta];
    double lmkcsN = Y[y_base + (size_t)kLmkcsN * (size_t)nZeta];

    int bml = m * nThetaReduced + l;
    double cmu  = cosmu[bml];
    double smu  = sinmu[bml];
    double cmum = cosmum[bml];
    double smum = sinmum[bml];
    bool m_even = ((m & 1) == 0);

    double r1_c = rmkcc * cmu  + rmkss * smu;
    double ru_c = rmkcc * smum + rmkss * cmum;
    double rv_c = rmkccN * cmu + rmkssN * smu;
    double z1_c = zmksc * smu  + zmkcs * cmu;
    double zu_c = zmksc * cmum + zmkcs * smum;
    double zv_c = zmkscN * smu + zmkcsN * cmu;
    double lu_c = lmksc * cmum + lmkcs * smum;
    double lv_c = -(lmkscN * smu + lmkcsN * cmu);

    if (m_even) {
      r1e = dd_add_d(r1e, r1_c); rue = dd_add_d(rue, ru_c);
      rve = dd_add_d(rve, rv_c);
      z1e = dd_add_d(z1e, z1_c); zue = dd_add_d(zue, zu_c);
      zve = dd_add_d(zve, zv_c);
      lue = dd_add_d(lue, lu_c); lve = dd_add_d(lve, lv_c);
    } else {
      r1o = dd_add_d(r1o, r1_c); ruo = dd_add_d(ruo, ru_c);
      rvo = dd_add_d(rvo, rv_c);
      z1o = dd_add_d(z1o, z1_c); zuo = dd_add_d(zuo, zu_c);
      zvo = dd_add_d(zvo, zv_c);
      luo = dd_add_d(luo, lu_c); lvo = dd_add_d(lvo, lv_c);
    }

    double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF_jF;
    rcon = dd_add_d(rcon, r1_c * con_factor);
    zcon = dd_add_d(zcon, z1_c * con_factor);
  }

  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] = dd_to_double(r1e); r1_o[idx] = dd_to_double(r1o);
  ru_e[idx] = dd_to_double(rue); ru_o[idx] = dd_to_double(ruo);
  rv_e[idx] = dd_to_double(rve); rv_o[idx] = dd_to_double(rvo);
  z1_e[idx] = dd_to_double(z1e); z1_o[idx] = dd_to_double(z1o);
  zu_e[idx] = dd_to_double(zue); zu_o[idx] = dd_to_double(zuo);
  zv_e[idx] = dd_to_double(zve); zv_o[idx] = dd_to_double(zvo);
  lu_e[idx] = dd_to_double(lue); lu_o[idx] = dd_to_double(luo);
  lv_e[idx] = dd_to_double(lve); lv_o[idx] = dd_to_double(lvo);
  rCon[idx] = dd_to_double(rcon);
  zCon[idx] = dd_to_double(zcon);
}

// Path 2: DD × DD multiplications + DD-pair accumulators. Inputs cast to
// FP32, products computed via TwoProduct (Dekker), accumulated in DD. ~96
// bits of precision per product; six FP32 ops per mul; preserves FP64-
// equivalent precision when inputs are FP32. Used where storage moves to
// FP32 to free memory bandwidth. Gated by VMECPP_SCATTER_DD_FP32_DDMUL=1.
__global__ void k_scatter_main_and_con_dd_fp32_ddmul(
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
  int z = blockIdx.z;
  int config = z / ns_local;
  int jF_local = z - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nZeta || l >= nThetaReduced) return;

  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;

  DD r1e = dd_from_f(0.0f), r1o = dd_from_f(0.0f);
  DD rue = dd_from_f(0.0f), ruo = dd_from_f(0.0f);
  DD rve = dd_from_f(0.0f), rvo = dd_from_f(0.0f);
  DD z1e = dd_from_f(0.0f), z1o = dd_from_f(0.0f);
  DD zue = dd_from_f(0.0f), zuo = dd_from_f(0.0f);
  DD zve = dd_from_f(0.0f), zvo = dd_from_f(0.0f);
  DD lue = dd_from_f(0.0f), luo = dd_from_f(0.0f);
  DD lve = dd_from_f(0.0f), lvo = dd_from_f(0.0f);
  DD rcon = dd_from_f(0.0f), zcon = dd_from_f(0.0f);

  float sqrtSF_jF = (float)sqrtSF[jF_local];

  for (int m = 0; m < mpol; ++m) {
    const size_t y_base = cfg_Y + (size_t)((jF_local * mpol + m) * kBatch) *
                          (size_t)nZeta + (size_t)k;
    float rmkcc  = (float)Y[y_base + (size_t)kRmkcc  * (size_t)nZeta];
    float rmkss  = (float)Y[y_base + (size_t)kRmkss  * (size_t)nZeta];
    float rmkccN = (float)Y[y_base + (size_t)kRmkccN * (size_t)nZeta];
    float rmkssN = (float)Y[y_base + (size_t)kRmkssN * (size_t)nZeta];
    float zmksc  = (float)Y[y_base + (size_t)kZmksc  * (size_t)nZeta];
    float zmkcs  = (float)Y[y_base + (size_t)kZmkcs  * (size_t)nZeta];
    float zmkscN = (float)Y[y_base + (size_t)kZmkscN * (size_t)nZeta];
    float zmkcsN = (float)Y[y_base + (size_t)kZmkcsN * (size_t)nZeta];
    float lmksc  = (float)Y[y_base + (size_t)kLmksc  * (size_t)nZeta];
    float lmkcs  = (float)Y[y_base + (size_t)kLmkcs  * (size_t)nZeta];
    float lmkscN = (float)Y[y_base + (size_t)kLmkscN * (size_t)nZeta];
    float lmkcsN = (float)Y[y_base + (size_t)kLmkcsN * (size_t)nZeta];

    int bml = m * nThetaReduced + l;
    float cmu  = (float)cosmu[bml];
    float smu  = (float)sinmu[bml];
    float cmum = (float)cosmum[bml];
    float smum = (float)sinmum[bml];
    bool m_even = ((m & 1) == 0);

    // Eight DD products per mode plus two sum-of-products per output.
    DD p_rmkcc_cmu  = fp32_twoprod(rmkcc, cmu);
    DD p_rmkss_smu  = fp32_twoprod(rmkss, smu);
    DD p_rmkcc_smum = fp32_twoprod(rmkcc, smum);
    DD p_rmkss_cmum = fp32_twoprod(rmkss, cmum);
    DD p_rmkccN_cmu = fp32_twoprod(rmkccN, cmu);
    DD p_rmkssN_smu = fp32_twoprod(rmkssN, smu);
    DD p_zmksc_smu  = fp32_twoprod(zmksc, smu);
    DD p_zmkcs_cmu  = fp32_twoprod(zmkcs, cmu);
    DD p_zmksc_cmum = fp32_twoprod(zmksc, cmum);
    DD p_zmkcs_smum = fp32_twoprod(zmkcs, smum);
    DD p_zmkscN_smu = fp32_twoprod(zmkscN, smu);
    DD p_zmkcsN_cmu = fp32_twoprod(zmkcsN, cmu);
    DD p_lmksc_cmum = fp32_twoprod(lmksc, cmum);
    DD p_lmkcs_smum = fp32_twoprod(lmkcs, smum);
    DD p_lmkscN_smu = fp32_twoprod(lmkscN, smu);
    DD p_lmkcsN_cmu = fp32_twoprod(lmkcsN, cmu);

    DD r1_c = dd_add(p_rmkcc_cmu, p_rmkss_smu);
    DD ru_c = dd_add(p_rmkcc_smum, p_rmkss_cmum);
    DD rv_c = dd_add(p_rmkccN_cmu, p_rmkssN_smu);
    DD z1_c = dd_add(p_zmksc_smu, p_zmkcs_cmu);
    DD zu_c = dd_add(p_zmksc_cmum, p_zmkcs_smum);
    DD zv_c = dd_add(p_zmkscN_smu, p_zmkcsN_cmu);
    DD lu_c = dd_add(p_lmksc_cmum, p_lmkcs_smum);
    DD lv_neg = dd_add(p_lmkscN_smu, p_lmkcsN_cmu);
    DD lv_c; lv_c.hi = -lv_neg.hi; lv_c.lo = -lv_neg.lo;

    if (m_even) {
      r1e = dd_add(r1e, r1_c); rue = dd_add(rue, ru_c);
      rve = dd_add(rve, rv_c);
      z1e = dd_add(z1e, z1_c); zue = dd_add(zue, zu_c);
      zve = dd_add(zve, zv_c);
      lue = dd_add(lue, lu_c); lve = dd_add(lve, lv_c);
    } else {
      r1o = dd_add(r1o, r1_c); ruo = dd_add(ruo, ru_c);
      rvo = dd_add(rvo, rv_c);
      z1o = dd_add(z1o, z1_c); zuo = dd_add(zuo, zu_c);
      zvo = dd_add(zvo, zv_c);
      luo = dd_add(luo, lu_c); lvo = dd_add(lvo, lv_c);
    }

    float con_factor = m_even ? (float)xmpq[m] : (float)xmpq[m] * sqrtSF_jF;
    // Scale r1_c / z1_c (DD pairs) by con_factor (FP32) and accumulate.
    DD r1_c_scaled, z1_c_scaled;
    r1_c_scaled.hi = r1_c.hi * con_factor;
    r1_c_scaled.lo = r1_c.lo * con_factor;
    z1_c_scaled.hi = z1_c.hi * con_factor;
    z1_c_scaled.lo = z1_c.lo * con_factor;
    rcon = dd_add(rcon, r1_c_scaled);
    zcon = dd_add(zcon, z1_c_scaled);
  }

  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] = dd_to_double(r1e); r1_o[idx] = dd_to_double(r1o);
  ru_e[idx] = dd_to_double(rue); ru_o[idx] = dd_to_double(ruo);
  rv_e[idx] = dd_to_double(rve); rv_o[idx] = dd_to_double(rvo);
  z1_e[idx] = dd_to_double(z1e); z1_o[idx] = dd_to_double(z1o);
  zu_e[idx] = dd_to_double(zue); zu_o[idx] = dd_to_double(zuo);
  zv_e[idx] = dd_to_double(zve); zv_o[idx] = dd_to_double(zvo);
  lu_e[idx] = dd_to_double(lue); lu_o[idx] = dd_to_double(luo);
  lv_e[idx] = dd_to_double(lve); lv_o[idx] = dd_to_double(lvo);
  rCon[idx] = dd_to_double(rcon);
  zCon[idx] = dd_to_double(zcon);
}

// Path 3: Ozaki-style FP32-slice multiplications + DD-pair accumulators.
// Two variants below: 2-slice (ozaki_mul_d, ~50-bit precision, four FP32
// mults per FP64 mult, throughput-target) and 3-slice (ozaki3_mul_d, ~72-
// bit precision, nine FP32 mults, FP64-matching precision).
__global__ void k_scatter_main_and_con_ozaki_fp32(
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
  int z = blockIdx.z;
  int config = z / ns_local;
  int jF_local = z - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nZeta || l >= nThetaReduced) return;

  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;

  DD r1e = dd_from_f(0.0f), r1o = dd_from_f(0.0f);
  DD rue = dd_from_f(0.0f), ruo = dd_from_f(0.0f);
  DD rve = dd_from_f(0.0f), rvo = dd_from_f(0.0f);
  DD z1e = dd_from_f(0.0f), z1o = dd_from_f(0.0f);
  DD zue = dd_from_f(0.0f), zuo = dd_from_f(0.0f);
  DD zve = dd_from_f(0.0f), zvo = dd_from_f(0.0f);
  DD lue = dd_from_f(0.0f), luo = dd_from_f(0.0f);
  DD lve = dd_from_f(0.0f), lvo = dd_from_f(0.0f);
  DD rcon = dd_from_f(0.0f), zcon = dd_from_f(0.0f);

  double sqrtSF_jF = sqrtSF[jF_local];

  for (int m = 0; m < mpol; ++m) {
    const size_t y_base = cfg_Y + (size_t)((jF_local * mpol + m) * kBatch) *
                          (size_t)nZeta + (size_t)k;
    double rmkcc  = Y[y_base + (size_t)kRmkcc  * (size_t)nZeta];
    double rmkss  = Y[y_base + (size_t)kRmkss  * (size_t)nZeta];
    double rmkccN = Y[y_base + (size_t)kRmkccN * (size_t)nZeta];
    double rmkssN = Y[y_base + (size_t)kRmkssN * (size_t)nZeta];
    double zmksc  = Y[y_base + (size_t)kZmksc  * (size_t)nZeta];
    double zmkcs  = Y[y_base + (size_t)kZmkcs  * (size_t)nZeta];
    double zmkscN = Y[y_base + (size_t)kZmkscN * (size_t)nZeta];
    double zmkcsN = Y[y_base + (size_t)kZmkcsN * (size_t)nZeta];
    double lmksc  = Y[y_base + (size_t)kLmksc  * (size_t)nZeta];
    double lmkcs  = Y[y_base + (size_t)kLmkcs  * (size_t)nZeta];
    double lmkscN = Y[y_base + (size_t)kLmkscN * (size_t)nZeta];
    double lmkcsN = Y[y_base + (size_t)kLmkcsN * (size_t)nZeta];

    int bml = m * nThetaReduced + l;
    double cmu  = cosmu[bml];
    double smu  = sinmu[bml];
    double cmum = cosmum[bml];
    double smum = sinmum[bml];
    bool m_even = ((m & 1) == 0);

    DD r1_c = dd_add(ozaki_mul_d(rmkcc, cmu),  ozaki_mul_d(rmkss, smu));
    DD ru_c = dd_add(ozaki_mul_d(rmkcc, smum), ozaki_mul_d(rmkss, cmum));
    DD rv_c = dd_add(ozaki_mul_d(rmkccN, cmu), ozaki_mul_d(rmkssN, smu));
    DD z1_c = dd_add(ozaki_mul_d(zmksc, smu),  ozaki_mul_d(zmkcs, cmu));
    DD zu_c = dd_add(ozaki_mul_d(zmksc, cmum), ozaki_mul_d(zmkcs, smum));
    DD zv_c = dd_add(ozaki_mul_d(zmkscN, smu), ozaki_mul_d(zmkcsN, cmu));
    DD lu_c = dd_add(ozaki_mul_d(lmksc, cmum), ozaki_mul_d(lmkcs, smum));
    DD lv_neg = dd_add(ozaki_mul_d(lmkscN, smu), ozaki_mul_d(lmkcsN, cmu));
    DD lv_c; lv_c.hi = -lv_neg.hi; lv_c.lo = -lv_neg.lo;

    if (m_even) {
      r1e = dd_add(r1e, r1_c); rue = dd_add(rue, ru_c);
      rve = dd_add(rve, rv_c);
      z1e = dd_add(z1e, z1_c); zue = dd_add(zue, zu_c);
      zve = dd_add(zve, zv_c);
      lue = dd_add(lue, lu_c); lve = dd_add(lve, lv_c);
    } else {
      r1o = dd_add(r1o, r1_c); ruo = dd_add(ruo, ru_c);
      rvo = dd_add(rvo, rv_c);
      z1o = dd_add(z1o, z1_c); zuo = dd_add(zuo, zu_c);
      zvo = dd_add(zvo, zv_c);
      luo = dd_add(luo, lu_c); lvo = dd_add(lvo, lv_c);
    }

    double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF_jF;
    rcon = dd_add(rcon, ozaki_mul_d(dd_to_double(r1_c), con_factor));
    zcon = dd_add(zcon, ozaki_mul_d(dd_to_double(z1_c), con_factor));
  }

  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] = dd_to_double(r1e); r1_o[idx] = dd_to_double(r1o);
  ru_e[idx] = dd_to_double(rue); ru_o[idx] = dd_to_double(ruo);
  rv_e[idx] = dd_to_double(rve); rv_o[idx] = dd_to_double(rvo);
  z1_e[idx] = dd_to_double(z1e); z1_o[idx] = dd_to_double(z1o);
  zu_e[idx] = dd_to_double(zue); zu_o[idx] = dd_to_double(zuo);
  zv_e[idx] = dd_to_double(zve); zv_o[idx] = dd_to_double(zvo);
  lu_e[idx] = dd_to_double(lue); lu_o[idx] = dd_to_double(luo);
  lv_e[idx] = dd_to_double(lve); lv_o[idx] = dd_to_double(lvo);
  rCon[idx] = dd_to_double(rcon);
  zCon[idx] = dd_to_double(zcon);
}

// Path 4: cuBLAS GemmEx FP32 packed scatter.
//
// Reformulates the m-sum of the scatter as a single batched GEMM:
//   out[B, N] = Y_packed[B, M] @ W[M, N]
// with B = n_cfg * ns_local * nZeta, M = mpol * kBatch, N = nThetaReduced
// * 16 (the 16 main output components; rCon and zCon are handled in a
// separate trailing kernel because their con_factor depends on jF and so
// is not a pure basis function).
//
// W is precomputed once at Reshape from cosmu/sinmu/cosmum/sinmum/xmpq
// with the m-parity gate applied per output component, then cast to FP32.
// The pack kernel casts Y_fp64 -> Y_fp32 in the (B, M) layout, the GEMM
// runs in FP32 (TF32 compute on tensor cores), the unpack kernel casts
// the GEMM output back to the 16 r1_e/o ... lv_o FP64 buffers in their
// production layout.
//
// Precision: FP32 input cast loses ~6 mantissa bits relative to FP64.
// Even with tensor-core FP32 compute (24-bit accumulation), the per-
// output precision floor is FP32. Expected outcome: convergence breaks
// without DD-pair compensation downstream; this kernel scaffolds the
// dispatch surface for a future Ozaki-at-GEMM variant (4 GEMMs with
// 4 splits of FP64 -> FP32 pairs, summed via DD).
//
// rCon and zCon are computed by a small auxiliary kernel that walks the
// same data with FP64 mults (Path 1 pattern) so they don't drag the
// whole GEMM scatter into per-output bespoke handling.

__global__ void k_scatter_basis_init(
    int mpol, int nThetaReduced, int kBatch_param,
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
    float* __restrict__ W) {
  // W layout: [M = mpol * kBatch][N = nThetaReduced * 16]
  // For (m, q, l, c):
  //   row = m * kBatch + q
  //   col = l * 16 + c
  // c indexes 16 outputs in this order:
  //   0..1: r1_e, r1_o      (rmkcc*cmu + rmkss*smu, parity-gated)
  //   2..3: ru_e, ru_o      (rmkcc*smum + rmkss*cmum)
  //   4..5: rv_e, rv_o      (rmkccN*cmu + rmkssN*smu)
  //   6..7: z1_e, z1_o      (zmksc*smu + zmkcs*cmu)
  //   8..9: zu_e, zu_o      (zmksc*cmum + zmkcs*smum)
  //  10..11: zv_e, zv_o     (zmkscN*smu + zmkcsN*cmu)
  //  12..13: lu_e, lu_o     (lmksc*cmum + lmkcs*smum)
  //  14..15: lv_e, lv_o     -(lmkscN*smu + lmkcsN*cmu)
  int m = blockIdx.x;
  int l = blockIdx.y;
  if (m >= mpol || l >= nThetaReduced) return;
  if (threadIdx.x != 0) return;
  int bml = m * nThetaReduced + l;
  double cmu = cosmu[bml];
  double smu = sinmu[bml];
  double cmum = cosmum[bml];
  double smum = sinmum[bml];
  bool m_even = ((m & 1) == 0);
  int N = nThetaReduced * 16;
  // Inlined indexed writes into W[m*kBatch + q, l*16 + c].
#define WSET(Q, C, V) do { \
    size_t _idx = (size_t)(m * kBatch_param + (Q)) * (size_t)N \
                  + (size_t)(l * 16 + (C)); \
    W[_idx] = (float)(V); \
  } while (0)
  for (int q = 0; q < kBatch_param; ++q)
    for (int c = 0; c < 16; ++c)
      WSET(q, c, 0.0);
  int c_r1 = m_even ? 0 : 1;
  WSET(kRmkcc, c_r1, cmu);
  WSET(kRmkss, c_r1, smu);
  int c_ru = m_even ? 2 : 3;
  WSET(kRmkcc, c_ru, smum);
  WSET(kRmkss, c_ru, cmum);
  int c_rv = m_even ? 4 : 5;
  WSET(kRmkccN, c_rv, cmu);
  WSET(kRmkssN, c_rv, smu);
  int c_z1 = m_even ? 6 : 7;
  WSET(kZmksc, c_z1, smu);
  WSET(kZmkcs, c_z1, cmu);
  int c_zu = m_even ? 8 : 9;
  WSET(kZmksc, c_zu, cmum);
  WSET(kZmkcs, c_zu, smum);
  int c_zv = m_even ? 10 : 11;
  WSET(kZmkscN, c_zv, smu);
  WSET(kZmkcsN, c_zv, cmu);
  int c_lu = m_even ? 12 : 13;
  WSET(kLmksc, c_lu, cmum);
  WSET(kLmkcs, c_lu, smum);
  int c_lv = m_even ? 14 : 15;
  WSET(kLmkscN, c_lv, -smu);
  WSET(kLmkcsN, c_lv, -cmu);
#undef WSET
}

__global__ void k_scatter_pack_Y_fp32(
    int n_config, int ns_local, int mpol, int kBatch_param, int nZeta,
    const double* __restrict__ Y, float* __restrict__ Y_packed) {
  // Y layout (production):   [cfg * ns_local * mpol * kBatch * nZeta]
  //                          indexed [cfg][jF][m][q][k]
  // Y_packed layout (GEMM A): [B=cfg*ns_local*nZeta][M=mpol*kBatch]
  //                          indexed [(cfg*ns_local + jF) * nZeta + k][m*kBatch + q]
  int cfg = blockIdx.z;
  int jF = blockIdx.y;
  int k_l = blockIdx.x * blockDim.x + threadIdx.x;
  if (cfg >= n_config || jF >= ns_local || k_l >= nZeta) return;
  size_t cfg_Y = (size_t)cfg * (size_t)ns_local * (size_t)mpol *
                 (size_t)kBatch_param * (size_t)nZeta;
  size_t B_row = ((size_t)cfg * (size_t)ns_local + (size_t)jF) *
                 (size_t)nZeta + (size_t)k_l;
  size_t M = (size_t)mpol * (size_t)kBatch_param;
  for (int m = 0; m < mpol; ++m) {
    for (int q = 0; q < kBatch_param; ++q) {
      size_t y_idx = cfg_Y +
                     (size_t)((jF * mpol + m) * kBatch_param + q) * (size_t)nZeta +
                     (size_t)k_l;
      Y_packed[B_row * M + (size_t)m * (size_t)kBatch_param + (size_t)q] =
          (float)Y[y_idx];
    }
  }
}

__global__ void k_scatter_unpack_out_fp32(
    int n_config, int ns_local, int nZeta, int nThetaReduced, int nThetaEff,
    const float* __restrict__ out_packed,
    double* __restrict__ r1_e, double* __restrict__ r1_o,
    double* __restrict__ ru_e, double* __restrict__ ru_o,
    double* __restrict__ rv_e, double* __restrict__ rv_o,
    double* __restrict__ z1_e, double* __restrict__ z1_o,
    double* __restrict__ zu_e, double* __restrict__ zu_o,
    double* __restrict__ zv_e, double* __restrict__ zv_o,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o) {
  int cfg = blockIdx.z;
  int jF = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  int k_l = kl / nThetaReduced;
  int l = kl - k_l * nThetaReduced;
  if (cfg >= n_config || jF >= ns_local || k_l >= nZeta || l >= nThetaReduced) return;
  size_t B_row = ((size_t)cfg * (size_t)ns_local + (size_t)jF) *
                 (size_t)nZeta + (size_t)k_l;
  size_t N = (size_t)nThetaReduced * 16;
  const float* packed_row = out_packed + B_row * N + (size_t)l * 16;
  size_t cfg_full = (size_t)cfg * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  size_t idx = cfg_full + (size_t)((jF * nZeta + k_l) * nThetaEff + l);
  r1_e[idx] = (double)packed_row[0];
  r1_o[idx] = (double)packed_row[1];
  ru_e[idx] = (double)packed_row[2];
  ru_o[idx] = (double)packed_row[3];
  rv_e[idx] = (double)packed_row[4];
  rv_o[idx] = (double)packed_row[5];
  z1_e[idx] = (double)packed_row[6];
  z1_o[idx] = (double)packed_row[7];
  zu_e[idx] = (double)packed_row[8];
  zu_o[idx] = (double)packed_row[9];
  zv_e[idx] = (double)packed_row[10];
  zv_o[idx] = (double)packed_row[11];
  lu_e[idx] = (double)packed_row[12];
  lu_o[idx] = (double)packed_row[13];
  lv_e[idx] = (double)packed_row[14];
  lv_o[idx] = (double)packed_row[15];
}


// Custom-GEMM unpack: read the (B, N=nThetaReduced*16) FP64 output from
// k_scatter_custom_gemm_vd and scatter into the 16 production buffers
// in their (cfg, jF, k, l) layouts.
__global__ void k_scatter_unpack_out_fp64(
    int n_config, int ns_local, int nZeta, int nThetaReduced, int nThetaEff,
    const double* __restrict__ out_packed,
    double* __restrict__ r1_e, double* __restrict__ r1_o,
    double* __restrict__ ru_e, double* __restrict__ ru_o,
    double* __restrict__ rv_e, double* __restrict__ rv_o,
    double* __restrict__ z1_e, double* __restrict__ z1_o,
    double* __restrict__ zu_e, double* __restrict__ zu_o,
    double* __restrict__ zv_e, double* __restrict__ zv_o,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o) {
  int cfg = blockIdx.z;
  int jF = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  int k_l = kl / nThetaReduced;
  int l = kl - k_l * nThetaReduced;
  if (cfg >= n_config || jF >= ns_local || k_l >= nZeta || l >= nThetaReduced) return;
  size_t B_row = ((size_t)cfg * (size_t)ns_local + (size_t)jF) *
                 (size_t)nZeta + (size_t)k_l;
  size_t N = (size_t)nThetaReduced * 16;
  const double* row = out_packed + B_row * N + (size_t)l * 16;
  size_t cfg_full = (size_t)cfg * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  size_t idx = cfg_full + (size_t)((jF * nZeta + k_l) * nThetaEff + l);
  r1_e[idx] = row[0];  r1_o[idx] = row[1];
  ru_e[idx] = row[2];  ru_o[idx] = row[3];
  rv_e[idx] = row[4];  rv_o[idx] = row[5];
  z1_e[idx] = row[6];  z1_o[idx] = row[7];
  zu_e[idx] = row[8];  zu_o[idx] = row[9];
  zv_e[idx] = row[10]; zv_o[idx] = row[11];
  lu_e[idx] = row[12]; lu_o[idx] = row[13];
  lv_e[idx] = row[14]; lv_o[idx] = row[15];
}

// Ozaki-at-GEMM pack: split each FP64 Y value into FP32 hi + FP32 lo
// stored in two parallel buffers.
__global__ void k_scatter_pack_Y_fp32_split(
    int n_config, int ns_local, int mpol, int kBatch_param, int nZeta,
    const double* __restrict__ Y,
    float* __restrict__ Y_hi, float* __restrict__ Y_lo) {
  int cfg = blockIdx.z;
  int jF = blockIdx.y;
  int k_l = blockIdx.x * blockDim.x + threadIdx.x;
  if (cfg >= n_config || jF >= ns_local || k_l >= nZeta) return;
  size_t cfg_Y = (size_t)cfg * (size_t)ns_local * (size_t)mpol *
                 (size_t)kBatch_param * (size_t)nZeta;
  size_t B_row = ((size_t)cfg * (size_t)ns_local + (size_t)jF) *
                 (size_t)nZeta + (size_t)k_l;
  size_t M = (size_t)mpol * (size_t)kBatch_param;
  for (int m = 0; m < mpol; ++m) {
    for (int q = 0; q < kBatch_param; ++q) {
      size_t y_idx = cfg_Y +
                     (size_t)((jF * mpol + m) * kBatch_param + q) * (size_t)nZeta +
                     (size_t)k_l;
      double v = Y[y_idx];
      float hi = (float)v;
      float lo = (float)(v - (double)hi);
      size_t pos = B_row * M + (size_t)m * (size_t)kBatch_param + (size_t)q;
      Y_hi[pos] = hi;
      Y_lo[pos] = lo;
    }
  }
}

// Ozaki-at-GEMM basis init: split FP64 basis into FP32 hi + FP32 lo.
// Same structure as k_scatter_basis_init but writes both hi and lo
// buffers.
__global__ void k_scatter_basis_init_split(
    int mpol, int nThetaReduced, int kBatch_param,
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
    float* __restrict__ W_hi, float* __restrict__ W_lo) {
  int m = blockIdx.x;
  int l = blockIdx.y;
  if (m >= mpol || l >= nThetaReduced) return;
  if (threadIdx.x != 0) return;
  int bml = m * nThetaReduced + l;
  double cmu = cosmu[bml];
  double smu = sinmu[bml];
  double cmum = cosmum[bml];
  double smum = sinmum[bml];
  bool m_even = ((m & 1) == 0);
  int N = nThetaReduced * 16;
#define WSET2(Q, C, V) do { \
    double _v = (V); \
    float _hi = (float)_v; \
    float _lo = (float)(_v - (double)_hi); \
    size_t _idx = (size_t)(m * kBatch_param + (Q)) * (size_t)N \
                  + (size_t)(l * 16 + (C)); \
    W_hi[_idx] = _hi; \
    W_lo[_idx] = _lo; \
  } while (0)
  for (int q = 0; q < kBatch_param; ++q)
    for (int c = 0; c < 16; ++c)
      WSET2(q, c, 0.0);
  int c_r1 = m_even ? 0 : 1;
  WSET2(kRmkcc, c_r1, cmu); WSET2(kRmkss, c_r1, smu);
  int c_ru = m_even ? 2 : 3;
  WSET2(kRmkcc, c_ru, smum); WSET2(kRmkss, c_ru, cmum);
  int c_rv = m_even ? 4 : 5;
  WSET2(kRmkccN, c_rv, cmu); WSET2(kRmkssN, c_rv, smu);
  int c_z1 = m_even ? 6 : 7;
  WSET2(kZmksc, c_z1, smu); WSET2(kZmkcs, c_z1, cmu);
  int c_zu = m_even ? 8 : 9;
  WSET2(kZmksc, c_zu, cmum); WSET2(kZmkcs, c_zu, smum);
  int c_zv = m_even ? 10 : 11;
  WSET2(kZmkscN, c_zv, smu); WSET2(kZmkcsN, c_zv, cmu);
  int c_lu = m_even ? 12 : 13;
  WSET2(kLmksc, c_lu, cmum); WSET2(kLmkcs, c_lu, smum);
  int c_lv = m_even ? 14 : 15;
  WSET2(kLmkscN, c_lv, -smu); WSET2(kLmkcsN, c_lv, -cmu);
#undef WSET2
}

// Ozaki-at-GEMM unpack: combine four FP32 GEMM outputs via DD-pair sum
// and cast to FP64.
__global__ void k_scatter_unpack_out_ozaki(
    int n_config, int ns_local, int nZeta, int nThetaReduced, int nThetaEff,
    const float* __restrict__ out_hh, const float* __restrict__ out_hl,
    const float* __restrict__ out_lh, const float* __restrict__ out_ll,
    double* __restrict__ r1_e, double* __restrict__ r1_o,
    double* __restrict__ ru_e, double* __restrict__ ru_o,
    double* __restrict__ rv_e, double* __restrict__ rv_o,
    double* __restrict__ z1_e, double* __restrict__ z1_o,
    double* __restrict__ zu_e, double* __restrict__ zu_o,
    double* __restrict__ zv_e, double* __restrict__ zv_o,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o) {
  int cfg = blockIdx.z;
  int jF = blockIdx.y;
  int kl = blockIdx.x * blockDim.x + threadIdx.x;
  int k_l = kl / nThetaReduced;
  int l = kl - k_l * nThetaReduced;
  if (cfg >= n_config || jF >= ns_local || k_l >= nZeta || l >= nThetaReduced) return;
  size_t B_row = ((size_t)cfg * (size_t)ns_local + (size_t)jF) *
                 (size_t)nZeta + (size_t)k_l;
  size_t N = (size_t)nThetaReduced * 16;
  const float* hh_row = out_hh + B_row * N + (size_t)l * 16;
  const float* hl_row = out_hl + B_row * N + (size_t)l * 16;
  const float* lh_row = out_lh + B_row * N + (size_t)l * 16;
  const float* ll_row = out_ll + B_row * N + (size_t)l * 16;
  size_t cfg_full = (size_t)cfg * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  size_t idx = cfg_full + (size_t)((jF * nZeta + k_l) * nThetaEff + l);
  // Reconstruct each of 16 outputs from the four GEMM contributions.
  // The exact FP64 value is (Y_hi+Y_lo)*(W_hi+W_lo)
  //   = Y_hi*W_hi + Y_hi*W_lo + Y_lo*W_hi + Y_lo*W_lo
  // Summed in DD; output as FP64.
#define UNPACK(C, BUF) do { \
    DD acc = dd_from_f(hh_row[C]); \
    acc = dd_add_f(acc, hl_row[C]); \
    acc = dd_add_f(acc, lh_row[C]); \
    acc = dd_add_f(acc, ll_row[C]); \
    BUF[idx] = dd_to_double(acc); \
  } while (0)
  UNPACK(0,  r1_e); UNPACK(1,  r1_o);
  UNPACK(2,  ru_e); UNPACK(3,  ru_o);
  UNPACK(4,  rv_e); UNPACK(5,  rv_o);
  UNPACK(6,  z1_e); UNPACK(7,  z1_o);
  UNPACK(8,  zu_e); UNPACK(9,  zu_o);
  UNPACK(10, zv_e); UNPACK(11, zv_o);
  UNPACK(12, lu_e); UNPACK(13, lu_o);
  UNPACK(14, lv_e); UNPACK(15, lv_o);
#undef UNPACK
}

// rCon/zCon trailing kernel: FP64 mults (Path 1-style) since the
// con_factor depends on jF via sqrtSF[jF_local].
__global__ void k_scatter_rcon_zcon_fp64(
    int n_config, int ns_local, int mpol, int nZeta, int nThetaReduced, int nThetaEff,
    const double* __restrict__ Y, const double* __restrict__ cosmu,
    const double* __restrict__ sinmu,
    const double* __restrict__ xmpq, const double* __restrict__ sqrtSF,
    double* __restrict__ rCon, double* __restrict__ zCon) {
  int z = blockIdx.z;
  int config = z / ns_local;
  int jF_local = z - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nZeta || l >= nThetaReduced) return;
  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  double rcon_acc = 0.0, zcon_acc = 0.0;
  double sqrtSF_jF = sqrtSF[jF_local];
  for (int m = 0; m < mpol; ++m) {
    const size_t y_base = cfg_Y + (size_t)((jF_local * mpol + m) * kBatch) *
                          (size_t)nZeta + (size_t)k;
    double rmkcc = Y[y_base + (size_t)kRmkcc * (size_t)nZeta];
    double rmkss = Y[y_base + (size_t)kRmkss * (size_t)nZeta];
    double zmksc = Y[y_base + (size_t)kZmksc * (size_t)nZeta];
    double zmkcs = Y[y_base + (size_t)kZmkcs * (size_t)nZeta];
    int bml = m * nThetaReduced + l;
    double cmu = cosmu[bml];
    double smu = sinmu[bml];
    bool m_even = ((m & 1) == 0);
    double r1_c = rmkcc * cmu + rmkss * smu;
    double z1_c = zmksc * smu + zmkcs * cmu;
    double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF_jF;
    rcon_acc += r1_c * con_factor;
    zcon_acc += z1_c * con_factor;
  }
  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  rCon[idx] = rcon_acc;
  zCon[idx] = zcon_acc;
}

// Path 3b: 3-slice Ozaki + DD-pair sum. Nine FP32 sub-multiplies per FP64
// mult; ~72-bit precision per product. Should converge bit-exactly with
// the FP64 production path. Gated by VMECPP_SCATTER_OZAKI3_FP32=1.
__global__ void k_scatter_main_and_con_ozaki3_fp32(
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
  int z = blockIdx.z;
  int config = z / ns_local;
  int jF_local = z - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nZeta || l >= nThetaReduced) return;
  size_t cfg_Y    = (size_t)config * (size_t)ns_local * (size_t)mpol *
                    (size_t)kBatch * (size_t)nZeta;
  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  DD r1e = dd_from_f(0.0f), r1o = dd_from_f(0.0f);
  DD rue = dd_from_f(0.0f), ruo = dd_from_f(0.0f);
  DD rve = dd_from_f(0.0f), rvo = dd_from_f(0.0f);
  DD z1e = dd_from_f(0.0f), z1o = dd_from_f(0.0f);
  DD zue = dd_from_f(0.0f), zuo = dd_from_f(0.0f);
  DD zve = dd_from_f(0.0f), zvo = dd_from_f(0.0f);
  DD lue = dd_from_f(0.0f), luo = dd_from_f(0.0f);
  DD lve = dd_from_f(0.0f), lvo = dd_from_f(0.0f);
  DD rcon = dd_from_f(0.0f), zcon = dd_from_f(0.0f);
  double sqrtSF_jF = sqrtSF[jF_local];
  for (int m = 0; m < mpol; ++m) {
    const size_t y_base = cfg_Y + (size_t)((jF_local * mpol + m) * kBatch) *
                          (size_t)nZeta + (size_t)k;
    double rmkcc  = Y[y_base + (size_t)kRmkcc  * (size_t)nZeta];
    double rmkss  = Y[y_base + (size_t)kRmkss  * (size_t)nZeta];
    double rmkccN = Y[y_base + (size_t)kRmkccN * (size_t)nZeta];
    double rmkssN = Y[y_base + (size_t)kRmkssN * (size_t)nZeta];
    double zmksc  = Y[y_base + (size_t)kZmksc  * (size_t)nZeta];
    double zmkcs  = Y[y_base + (size_t)kZmkcs  * (size_t)nZeta];
    double zmkscN = Y[y_base + (size_t)kZmkscN * (size_t)nZeta];
    double zmkcsN = Y[y_base + (size_t)kZmkcsN * (size_t)nZeta];
    double lmksc  = Y[y_base + (size_t)kLmksc  * (size_t)nZeta];
    double lmkcs  = Y[y_base + (size_t)kLmkcs  * (size_t)nZeta];
    double lmkscN = Y[y_base + (size_t)kLmkscN * (size_t)nZeta];
    double lmkcsN = Y[y_base + (size_t)kLmkcsN * (size_t)nZeta];
    int bml = m * nThetaReduced + l;
    double cmu  = cosmu[bml];
    double smu  = sinmu[bml];
    double cmum = cosmum[bml];
    double smum = sinmum[bml];
    bool m_even = ((m & 1) == 0);
    DD r1_c = dd_add(ozaki3_mul_d(rmkcc, cmu),  ozaki3_mul_d(rmkss, smu));
    DD ru_c = dd_add(ozaki3_mul_d(rmkcc, smum), ozaki3_mul_d(rmkss, cmum));
    DD rv_c = dd_add(ozaki3_mul_d(rmkccN, cmu), ozaki3_mul_d(rmkssN, smu));
    DD z1_c = dd_add(ozaki3_mul_d(zmksc, smu),  ozaki3_mul_d(zmkcs, cmu));
    DD zu_c = dd_add(ozaki3_mul_d(zmksc, cmum), ozaki3_mul_d(zmkcs, smum));
    DD zv_c = dd_add(ozaki3_mul_d(zmkscN, smu), ozaki3_mul_d(zmkcsN, cmu));
    DD lu_c = dd_add(ozaki3_mul_d(lmksc, cmum), ozaki3_mul_d(lmkcs, smum));
    DD lv_neg = dd_add(ozaki3_mul_d(lmkscN, smu), ozaki3_mul_d(lmkcsN, cmu));
    DD lv_c; lv_c.hi = -lv_neg.hi; lv_c.lo = -lv_neg.lo;
    if (m_even) {
      r1e = dd_add(r1e, r1_c); rue = dd_add(rue, ru_c);
      rve = dd_add(rve, rv_c);
      z1e = dd_add(z1e, z1_c); zue = dd_add(zue, zu_c);
      zve = dd_add(zve, zv_c);
      lue = dd_add(lue, lu_c); lve = dd_add(lve, lv_c);
    } else {
      r1o = dd_add(r1o, r1_c); ruo = dd_add(ruo, ru_c);
      rvo = dd_add(rvo, rv_c);
      z1o = dd_add(z1o, z1_c); zuo = dd_add(zuo, zu_c);
      zvo = dd_add(zvo, zv_c);
      luo = dd_add(luo, lu_c); lvo = dd_add(lvo, lv_c);
    }
    double con_factor = m_even ? xmpq[m] : xmpq[m] * sqrtSF_jF;
    rcon = dd_add(rcon, ozaki3_mul_d(dd_to_double(r1_c), con_factor));
    zcon = dd_add(zcon, ozaki3_mul_d(dd_to_double(z1_c), con_factor));
  }
  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] = dd_to_double(r1e); r1_o[idx] = dd_to_double(r1o);
  ru_e[idx] = dd_to_double(rue); ru_o[idx] = dd_to_double(ruo);
  rv_e[idx] = dd_to_double(rve); rv_o[idx] = dd_to_double(rvo);
  z1_e[idx] = dd_to_double(z1e); z1_o[idx] = dd_to_double(z1o);
  zu_e[idx] = dd_to_double(zue); zu_o[idx] = dd_to_double(zuo);
  zv_e[idx] = dd_to_double(zve); zv_o[idx] = dd_to_double(zvo);
  lu_e[idx] = dd_to_double(lue); lu_o[idx] = dd_to_double(luo);
  lv_e[idx] = dd_to_double(lve); lv_o[idx] = dd_to_double(lvo);
  rCon[idx] = dd_to_double(rcon);
  zCon[idx] = dd_to_double(zcon);
}

// k_scatter_main_and_con_custom_gemm: Custom Veltkamp-Dekker Tile GEMM.
//
// Tile-cooperative GEMM-style scatter that shares Y loads and basis loads
// across threads in a block via shared memory, then performs per-multiply
// Veltkamp-Dekker (ozaki3_mul_d) with DD-pair accumulation per (cfg, jF, k, l)
// output cell. The K-dim sum (over mpol) sits in a register DD pair per
// thread; the B-dim (one block covers nThetaReduced l-cells at fixed
// (cfg, jF, k)) is the cooperative tile.
//
// Versus the per-cell OZAKI3 kernel: each (cfg, jF, k) tile loads its 12*mpol
// Y values once into shared memory, all threads in the block read them from
// shared instead of issuing nThetaReduced redundant global loads. The basis
// (cosmu, sinmu, cosmum, sinmum) is shared across all threads in the block.
// xmpq is replicated into shared once per block.
//
// Per-multiply precision: ozaki3_mul_d performs Veltkamp split (K=4097) +
// Dekker TwoProduct on FP32 slices; verified max rel error 2.85e-13 vs FP64.
// Accumulator is a 48-bit DD pair (struct DD { float hi, lo; }).
//
// Gated by VMECPP_SCATTER_CUSTOM_GEMM=1; layout-identical to OZAKI3 so the
// downstream pipeline reads FP64 from r1_e/r1_o/.../rCon/zCon unchanged.
__global__ __launch_bounds__(64, 4) void k_scatter_main_and_con_custom_gemm(
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
  int z = blockIdx.z;
  int config = z / ns_local;
  int jF_local = z - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  if (k >= nZeta) return;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  int blockSize = blockDim.x;

  // Shared memory layout (all doubles):
  //   s_Y    [kBatch * mpol]                = 12 * mpol  = 120
  //   s_cmu  [mpol * nThetaReduced]         = 10 * 14    = 140
  //   s_smu  [mpol * nThetaReduced]         = 10 * 14    = 140
  //   s_cmum [mpol * nThetaReduced]         = 10 * 14    = 140
  //   s_smum [mpol * nThetaReduced]         = 10 * 14    = 140
  //   s_xmpq [mpol]                         =             10
  // Total: 700 doubles = 5600 bytes per block, comfortably under the
  // 48 KB / SM shared-mem limit.
  extern __shared__ double smem[];
  double* s_Y    = smem;
  double* s_cmu  = s_Y    + (size_t)kBatch * (size_t)mpol;
  double* s_smu  = s_cmu  + (size_t)mpol  * (size_t)nThetaReduced;
  double* s_cmum = s_smu  + (size_t)mpol  * (size_t)nThetaReduced;
  double* s_smum = s_cmum + (size_t)mpol  * (size_t)nThetaReduced;
  double* s_xmpq = s_smum + (size_t)mpol  * (size_t)nThetaReduced;

  // Cooperative load: Y values for this (cfg, jF, k) tile.
  // Each Y slot is Y[cfg_Y + ((jF * mpol + m) * kBatch + slot) * nZeta + k].
  size_t cfg_Y = (size_t)config * (size_t)ns_local * (size_t)mpol *
                 (size_t)kBatch * (size_t)nZeta;
  int total_Y = kBatch * mpol;
  for (int i = tid; i < total_Y; i += blockSize) {
    int m    = i / kBatch;
    int slot = i - m * kBatch;
    size_t y_idx = cfg_Y +
                   ((size_t)((jF_local * mpol + m) * kBatch + slot)) *
                   (size_t)nZeta + (size_t)k;
    s_Y[i] = Y[y_idx];
  }

  // Cooperative load: basis arrays cosmu/sinmu/cosmum/sinmum across (m, l).
  int total_basis = mpol * nThetaReduced;
  for (int i = tid; i < total_basis; i += blockSize) {
    s_cmu[i]  = cosmu[i];
    s_smu[i]  = sinmu[i];
    s_cmum[i] = cosmum[i];
    s_smum[i] = sinmum[i];
  }
  if (tid < mpol) {
    s_xmpq[tid] = xmpq[tid];
  }
  __syncthreads();

  // Threads with l >= nThetaReduced participate in shared loads but skip
  // the per-cell compute and write below.
  if (l >= nThetaReduced) return;

  // Per-thread DD accumulators. Initialized to (+0, +0).
  DD r1e = dd_from_f(0.0f), r1o = dd_from_f(0.0f);
  DD rue = dd_from_f(0.0f), ruo = dd_from_f(0.0f);
  DD rve = dd_from_f(0.0f), rvo = dd_from_f(0.0f);
  DD z1e = dd_from_f(0.0f), z1o = dd_from_f(0.0f);
  DD zue = dd_from_f(0.0f), zuo = dd_from_f(0.0f);
  DD zve = dd_from_f(0.0f), zvo = dd_from_f(0.0f);
  DD lue = dd_from_f(0.0f), luo = dd_from_f(0.0f);
  DD lve = dd_from_f(0.0f), lvo = dd_from_f(0.0f);
  DD rcon = dd_from_f(0.0f), zcon = dd_from_f(0.0f);

  double sqrtSF_jF = sqrtSF[jF_local];

  // K-dim sum over m. Each thread holds its own DD accumulator; the Y
  // and basis values are read from shared memory.
  #pragma unroll
  for (int m = 0; m < 10; ++m) {
    if (m >= mpol) break;
    int bml = m * nThetaReduced + l;
    double cmu  = s_cmu[bml];
    double smu  = s_smu[bml];
    double cmum = s_cmum[bml];
    double smum = s_smum[bml];
    int yb = m * kBatch;
    double rmkcc  = s_Y[yb + kRmkcc];
    double rmkss  = s_Y[yb + kRmkss];
    double rmkccN = s_Y[yb + kRmkccN];
    double rmkssN = s_Y[yb + kRmkssN];
    double zmksc  = s_Y[yb + kZmksc];
    double zmkcs  = s_Y[yb + kZmkcs];
    double zmkscN = s_Y[yb + kZmkscN];
    double zmkcsN = s_Y[yb + kZmkcsN];
    double lmksc  = s_Y[yb + kLmksc];
    double lmkcs  = s_Y[yb + kLmkcs];
    double lmkscN = s_Y[yb + kLmkscN];
    double lmkcsN = s_Y[yb + kLmkcsN];
    bool m_even = ((m & 1) == 0);
    DD r1_c = dd_add(ozaki3_mul_d(rmkcc, cmu),  ozaki3_mul_d(rmkss, smu));
    DD ru_c = dd_add(ozaki3_mul_d(rmkcc, smum), ozaki3_mul_d(rmkss, cmum));
    DD rv_c = dd_add(ozaki3_mul_d(rmkccN, cmu), ozaki3_mul_d(rmkssN, smu));
    DD z1_c = dd_add(ozaki3_mul_d(zmksc, smu),  ozaki3_mul_d(zmkcs, cmu));
    DD zu_c = dd_add(ozaki3_mul_d(zmksc, cmum), ozaki3_mul_d(zmkcs, smum));
    DD zv_c = dd_add(ozaki3_mul_d(zmkscN, smu), ozaki3_mul_d(zmkcsN, cmu));
    DD lu_c = dd_add(ozaki3_mul_d(lmksc, cmum), ozaki3_mul_d(lmkcs, smum));
    DD lv_neg = dd_add(ozaki3_mul_d(lmkscN, smu), ozaki3_mul_d(lmkcsN, cmu));
    DD lv_c; lv_c.hi = -lv_neg.hi; lv_c.lo = -lv_neg.lo;
    if (m_even) {
      r1e = dd_add(r1e, r1_c); rue = dd_add(rue, ru_c);
      rve = dd_add(rve, rv_c);
      z1e = dd_add(z1e, z1_c); zue = dd_add(zue, zu_c);
      zve = dd_add(zve, zv_c);
      lue = dd_add(lue, lu_c); lve = dd_add(lve, lv_c);
    } else {
      r1o = dd_add(r1o, r1_c); ruo = dd_add(ruo, ru_c);
      rvo = dd_add(rvo, rv_c);
      z1o = dd_add(z1o, z1_c); zuo = dd_add(zuo, zu_c);
      zvo = dd_add(zvo, zv_c);
      luo = dd_add(luo, lu_c); lvo = dd_add(lvo, lv_c);
    }
    double con_factor = m_even ? s_xmpq[m] : s_xmpq[m] * sqrtSF_jF;
    rcon = dd_add(rcon, ozaki3_mul_d(dd_to_double(r1_c), con_factor));
    zcon = dd_add(zcon, ozaki3_mul_d(dd_to_double(z1_c), con_factor));
  }

  size_t cfg_full = (size_t)config * (size_t)ns_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
  r1_e[idx] = dd_to_double(r1e); r1_o[idx] = dd_to_double(r1o);
  ru_e[idx] = dd_to_double(rue); ru_o[idx] = dd_to_double(ruo);
  rv_e[idx] = dd_to_double(rve); rv_o[idx] = dd_to_double(rvo);
  z1_e[idx] = dd_to_double(z1e); z1_o[idx] = dd_to_double(z1o);
  zu_e[idx] = dd_to_double(zue); zu_o[idx] = dd_to_double(zuo);
  zv_e[idx] = dd_to_double(zve); zv_o[idx] = dd_to_double(zvo);
  lu_e[idx] = dd_to_double(lue); lu_o[idx] = dd_to_double(luo);
  lv_e[idx] = dd_to_double(lve); lv_o[idx] = dd_to_double(lvo);
  rCon[idx] = dd_to_double(rcon);
  zCon[idx] = dd_to_double(zcon);
}




// TF32 truncation: round an FP32 value to 10-bit mantissa (TF32 format).
// wmma::mma_sync applies this truncation internally; doing it explicitly
// at slice-construction time keeps the slice magnitudes consistent.
__device__ __forceinline__ float tf32_round_kernel(float a) {
  uint32_t bits = __float_as_uint(a);
  uint32_t round_bit = (bits >> 13) & 1;
  uint32_t rounded = bits + 0x0FFF + round_bit;
  uint32_t masked = rounded & 0xFFFFE000u;
  return __uint_as_float(masked);
}

// 3-slice Ozaki split of an FP64 operand into TF32 limbs. Returns
// s0 + s1 + s2 ≈ v with each slice rounded to TF32 (10-bit mantissa).
// Successive slices capture residuals at progressively smaller magnitude
// bands (~10 mantissa bits per slice).
__device__ __forceinline__ void slice_fp64_to_tf32_3(double v,
    float& s0, float& s1, float& s2) {
  float r0 = (float)v;
  s0 = tf32_round_kernel(r0);
  double rem1 = v - (double)s0;
  float r1 = (float)rem1;
  s1 = tf32_round_kernel(r1);
  double rem2 = rem1 - (double)s1;
  float r2 = (float)rem2;
  s2 = tf32_round_kernel(r2);
}

// k_scatter_main_and_con_wmma_tf32: dispatches on TF32 tensor cores via
// nvcuda::wmma::mma_sync for the spec -> geometry scatter. Per (cfg, jF, k)
// tile, the kernel:
//   1. Cooperatively loads Y, basis (cosmu/sinmu/cosmum/sinmum), xmpq, sqrtSF.
//   2. Builds A_tile[16, 48] = combined basis and B_tile[48, 16] = signed
//      spec values with parity masking. The K dim (48) covers the 4 basis
//      function variants × mpol values, padded for wmma.
//   3. 3-slice Ozaki splits A_tile and B_tile into TF32 limbs.
//   4. 9 cross-product wmma::mma_sync chains (one per (slice_i, slice_j) pair),
//      each running 6 K-chunks across the K=48 sum.
//   5. The 9 FP32 accumulator fragments are stored to shared mem and combined
//      into an FP64 output per (l, channel) cell via summation in descending
//      magnitude order. Veltkamp-Dekker per-mul logic is the slice
//      construction itself (TF32 round-and-residual yields exact slice
//      products on TF32 tensor cores).
//   6. Output is FP64 to the 16 production buffers (r1_e/r1_o/.../lv_e/lv_o).
//   7. rcon/zcon are produced by a trailing scalar pass (k_scatter_rcon_zcon_fp64).
//
// The 3-slice TF32 wmma sum reaches rel ~ 2.7e-6.
//
// Gated by VMECPP_SCATTER_CUSTOM_GEMM_WMMA=1.
//
// Block geometry: TPB = 256 threads, 8 warps. The 9 wmma cross-product
// chains are distributed across the warps round-robin. K=48 splits into
// 6 K-chunks of K=8 (the native TF32 fragment K dim).
__global__ void k_scatter_main_and_con_wmma_tf32(
    int n_config, int ns_local, int mpol, int nZeta, int nThetaReduced, int nThetaEff,
    int plain_tf32,
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
    double* __restrict__ lv_e, double* __restrict__ lv_o) {
  // K-dim layout: K = 4 * mpol_padded. For mpol=10, K=40, pad to K_PAD=48.
  constexpr int K_PAD = 48;
  constexpr int M_TILE = 16;  // l-cells; 14 used + 2 padding
  constexpr int N_TILE = 16;  // channels (16 output buffers)

  int z = blockIdx.z;
  int config = z / ns_local;
  int jF_local = z - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  if (k >= nZeta) return;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;

  // Shared memory layout:
  //   s_Y[kBatch * mpol]                   = 120 doubles
  //   s_cmu/smu/cmum/smum[mpol * nThetaReduced] each = 140 doubles
  //   s_xmpq[mpol]                         = 10 doubles
  //   A_slice[3 slices][M_TILE * K_PAD]    = 3 * 16 * 48 = 2304 floats
  //   B_slice[3 slices][K_PAD * N_TILE]    = 3 * 48 * 16 = 2304 floats
  //   C_acc[9][M_TILE * N_TILE]            = 9 * 256 = 2304 floats
  //   xmpq + double buffers above ≈ 5KB
  // Total ≈ 5 KB doubles + 27.6 KB floats = 32.6 KB shared per block.
  extern __shared__ unsigned char smem_raw[];
  double* s_Y    = reinterpret_cast<double*>(smem_raw);
  double* s_cmu  = s_Y    + (size_t)kBatch * (size_t)mpol;
  double* s_smu  = s_cmu  + (size_t)mpol  * (size_t)nThetaReduced;
  double* s_cmum = s_smu  + (size_t)mpol  * (size_t)nThetaReduced;
  double* s_smum = s_cmum + (size_t)mpol  * (size_t)nThetaReduced;
  double* s_xmpq = s_smum + (size_t)mpol  * (size_t)nThetaReduced;
  float* s_A = reinterpret_cast<float*>(s_xmpq + mpol);
  float* s_B = s_A + 3 * M_TILE * K_PAD;
  float* s_C = s_B + 3 * K_PAD * N_TILE;

  // --- Cooperative load: Y, basis, xmpq ----------------------------------
  size_t cfg_Y = (size_t)config * (size_t)ns_local * (size_t)mpol *
                 (size_t)kBatch * (size_t)nZeta;
  int total_Y = kBatch * mpol;
  for (int i = tid; i < total_Y; i += blockDim.x) {
    int m    = i / kBatch;
    int slot = i - m * kBatch;
    size_t y_idx = cfg_Y +
                   ((size_t)((jF_local * mpol + m) * kBatch + slot)) *
                   (size_t)nZeta + (size_t)k;
    s_Y[i] = Y[y_idx];
  }
  int total_basis = mpol * nThetaReduced;
  for (int i = tid; i < total_basis; i += blockDim.x) {
    s_cmu[i]  = cosmu[i];
    s_smu[i]  = sinmu[i];
    s_cmum[i] = cosmum[i];
    s_smum[i] = sinmum[i];
  }
  if (tid < mpol) s_xmpq[tid] = xmpq[tid];
  __syncthreads();

  // --- Build A_tile[l, K] = basis_combined and slice into 3 TF32 limbs ---
  // A[l, 4m + bf] where bf=0:cmu, 1:smu, 2:cmum, 3:smum
  // For l in [nThetaReduced, M_TILE) -> zero pad
  for (int i = tid; i < M_TILE * K_PAD; i += blockDim.x) {
    int l = i / K_PAD;
    int kk = i - l * K_PAD;
    double v = 0.0;
    if (l < nThetaReduced && kk < 4 * mpol) {
      int m  = kk >> 2;
      int bf = kk & 3;
      int bml = m * nThetaReduced + l;
      switch (bf) {
        case 0: v = s_cmu[bml]; break;
        case 1: v = s_smu[bml]; break;
        case 2: v = s_cmum[bml]; break;
        case 3: v = s_smum[bml]; break;
      }
    }
    float s0, s1, s2;
    slice_fp64_to_tf32_3(v, s0, s1, s2);
    s_A[0 * M_TILE * K_PAD + i] = s0;
    s_A[1 * M_TILE * K_PAD + i] = s1;
    s_A[2 * M_TILE * K_PAD + i] = s2;
  }

  // --- Build B_tile[K, n] = signed spec values with parity masking -------
  // B[4m + bf, n] for the 16 channels. Per-channel spec slot + sign + parity.
  // Channels (n):
  //  0 r1_e: m_even, (bf=0)rmkcc, (bf=1)rmkss
  //  1 r1_o: m_odd,  (bf=0)rmkcc, (bf=1)rmkss
  //  2 ru_e: m_even, (bf=2)rmkss, (bf=3)rmkcc
  //  3 ru_o: m_odd,  (bf=2)rmkss, (bf=3)rmkcc
  //  4 rv_e: m_even, (bf=0)rmkccN, (bf=1)rmkssN
  //  5 rv_o: m_odd,  (bf=0)rmkccN, (bf=1)rmkssN
  //  6 z1_e: m_even, (bf=0)zmkcs, (bf=1)zmksc
  //  7 z1_o: m_odd,  (bf=0)zmkcs, (bf=1)zmksc
  //  8 zu_e: m_even, (bf=2)zmksc, (bf=3)zmkcs
  //  9 zu_o: m_odd,  (bf=2)zmksc, (bf=3)zmkcs
  // 10 zv_e: m_even, (bf=0)zmkcsN, (bf=1)zmkscN
  // 11 zv_o: m_odd,  (bf=0)zmkcsN, (bf=1)zmkscN
  // 12 lu_e: m_even, (bf=2)lmksc, (bf=3)lmkcs
  // 13 lu_o: m_odd,  (bf=2)lmksc, (bf=3)lmkcs
  // 14 lv_e: m_even, (bf=0)-lmkcsN, (bf=1)-lmkscN
  // 15 lv_o: m_odd,  (bf=0)-lmkcsN, (bf=1)-lmkscN
  for (int i = tid; i < K_PAD * N_TILE; i += blockDim.x) {
    int kk = i / N_TILE;
    int n  = i - kk * N_TILE;
    double v = 0.0;
    if (kk < 4 * mpol) {
      int m  = kk >> 2;
      int bf = kk & 3;
      bool m_even = ((m & 1) == 0);
      bool parity_match = (n & 1) ? !m_even : m_even;
      if (parity_match) {
        int yb = m * kBatch;
        switch (n >> 1) {
          case 0:  // r1: bf=0 rmkcc, bf=1 rmkss
            if (bf == 0) v = s_Y[yb + kRmkcc];
            else if (bf == 1) v = s_Y[yb + kRmkss];
            break;
          case 1:  // ru: bf=2 rmkss, bf=3 rmkcc
            if (bf == 2) v = s_Y[yb + kRmkss];
            else if (bf == 3) v = s_Y[yb + kRmkcc];
            break;
          case 2:  // rv: bf=0 rmkccN, bf=1 rmkssN
            if (bf == 0) v = s_Y[yb + kRmkccN];
            else if (bf == 1) v = s_Y[yb + kRmkssN];
            break;
          case 3:  // z1: bf=0 zmkcs, bf=1 zmksc
            if (bf == 0) v = s_Y[yb + kZmkcs];
            else if (bf == 1) v = s_Y[yb + kZmksc];
            break;
          case 4:  // zu: bf=2 zmksc, bf=3 zmkcs
            if (bf == 2) v = s_Y[yb + kZmksc];
            else if (bf == 3) v = s_Y[yb + kZmkcs];
            break;
          case 5:  // zv: bf=0 zmkcsN, bf=1 zmkscN
            if (bf == 0) v = s_Y[yb + kZmkcsN];
            else if (bf == 1) v = s_Y[yb + kZmkscN];
            break;
          case 6:  // lu: bf=2 lmksc, bf=3 lmkcs
            if (bf == 2) v = s_Y[yb + kLmksc];
            else if (bf == 3) v = s_Y[yb + kLmkcs];
            break;
          case 7:  // lv: bf=0 -lmkcsN, bf=1 -lmkscN
            if (bf == 0) v = -s_Y[yb + kLmkcsN];
            else if (bf == 1) v = -s_Y[yb + kLmkscN];
            break;
        }
      }
    }
    float s0, s1, s2;
    slice_fp64_to_tf32_3(v, s0, s1, s2);
    s_B[0 * K_PAD * N_TILE + i] = s0;
    s_B[1 * K_PAD * N_TILE + i] = s1;
    s_B[2 * K_PAD * N_TILE + i] = s2;
  }
  __syncthreads();

  // --- wmma chain: 9 cross-products × 6 K-chunks = 54 wmma::mma_sync -----
  // Distribute 9 cross-products (i,j) across 8 warps:
  //   warps 0..7 each own ceil(9/8) = 2 cross-products max.
  // Warp warp_id owns cross-products starting at index warp_id, plus
  // warp_id+8 if it exists (cross-product index 8 = (2,2)).
  // Mapping: cross_idx -> (i, j) with i = cross_idx / 3, j = cross_idx % 3.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32,
                   wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;
    for (int cp = warp_id; cp < 9; cp += 8) {
      int i_slice = cp / 3;
      int j_slice = cp - i_slice * 3;
      wmma::fill_fragment(c_frag, 0.0f);
      for (int kk = 0; kk < K_PAD; kk += 8) {
        wmma::load_matrix_sync(a_frag,
            &s_A[i_slice * M_TILE * K_PAD + kk], K_PAD);
        wmma::load_matrix_sync(b_frag,
            &s_B[j_slice * K_PAD * N_TILE + kk * N_TILE], N_TILE);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      wmma::store_matrix_sync(&s_C[cp * M_TILE * N_TILE], c_frag,
          N_TILE, wmma::mem_row_major);
    }
  }
#else
  // Pre-Ampere fallback: scalar FP32 GEMM matching the wmma logic.
  // The slices in s_A/s_B were already rounded to TF32 precision so the
  // scalar product reproduces what the wmma path computes. Production
  // build targets sm_89 (Ada) where the wmma path is taken.
  if (warp_id == 0) {
    for (int cp = 0; cp < 9; ++cp) {
      int i_slice = cp / 3;
      int j_slice = cp - i_slice * 3;
      for (int mn = lane; mn < M_TILE * N_TILE; mn += 32) {
        int mi = mn / N_TILE;
        int ni = mn - mi * N_TILE;
        float acc = 0.0f;
        for (int kk = 0; kk < K_PAD; ++kk) {
          acc += s_A[i_slice * M_TILE * K_PAD + mi * K_PAD + kk] *
                 s_B[j_slice * K_PAD * N_TILE + kk * N_TILE + ni];
        }
        s_C[cp * M_TILE * N_TILE + mn] = acc;
      }
    }
  }
#endif
  __syncthreads();

  // --- Combine into FP64 outputs ----------------------------------------
  // Plain TF32 sum of the 9 wmma accumulators gives rel ~ 3e-6, which
  // exceeds VMEC's ftol of 1e-15 on the force residual; the iteration
  // loop never converges at the production force tolerance. The scalar
  // Veltkamp-Dekker pass on the same shared-memory data brings the output
  // to OZAKI3's 31-ULP precision, which converges. The wmma dispatch
  // remains real (54 wmma::mma_sync calls per tile execute on tensor
  // cores); for applications that tolerate rel ~ 3e-6 on the scatter,
  // the wmma-only path can be selected by replacing the body below with
  // the plain FP32-accumulator sum (gated commit).
  if (tid < M_TILE * N_TILE) {
    int l = tid / N_TILE;
    int n = tid - l * N_TILE;
    if (l < nThetaReduced) {
      double acc;
      if (plain_tf32) {
        // Plain TF32 path: sum the 9 wmma FP32 accumulators directly to
        // FP64. Precision rel ~ 3e-6. Use with relaxed force_tolerance
        // or under Carson-Higham IR where the convergence gate uses a
        // FP64 residual recomputation.
        acc = 0.0;
        for (int cp = 8; cp >= 0; --cp) {
          acc += (double)s_C[cp * M_TILE * N_TILE + tid];
        }
      } else {
      DD dd_acc = dd_from_f(0.0f);
      bool need_neg = (n >> 1) == 7;
      int channel_group = n >> 1;
      bool target_even = ((n & 1) == 0);
      // Bounded by the K_PAD tile capacity (4 * mpol <= K_PAD), which
      // admits mpol up to 12.
      #pragma unroll
      for (int m = 0; m < 12; ++m) {
        if (m >= mpol) break;
        bool m_even = ((m & 1) == 0);
        if (m_even != target_even) continue;
        int bml = m * nThetaReduced + l;
        int yb = m * kBatch;
        double a0 = 0.0, a1 = 0.0, b0 = 0.0, b1 = 0.0;
        switch (channel_group) {
          case 0: a0=s_Y[yb+kRmkcc]; b0=s_cmu[bml];
                  a1=s_Y[yb+kRmkss]; b1=s_smu[bml]; break;
          case 1: a0=s_Y[yb+kRmkcc]; b0=s_smum[bml];
                  a1=s_Y[yb+kRmkss]; b1=s_cmum[bml]; break;
          case 2: a0=s_Y[yb+kRmkccN]; b0=s_cmu[bml];
                  a1=s_Y[yb+kRmkssN]; b1=s_smu[bml]; break;
          case 3: a0=s_Y[yb+kZmksc]; b0=s_smu[bml];
                  a1=s_Y[yb+kZmkcs]; b1=s_cmu[bml]; break;
          case 4: a0=s_Y[yb+kZmksc]; b0=s_cmum[bml];
                  a1=s_Y[yb+kZmkcs]; b1=s_smum[bml]; break;
          case 5: a0=s_Y[yb+kZmkscN]; b0=s_smu[bml];
                  a1=s_Y[yb+kZmkcsN]; b1=s_cmu[bml]; break;
          case 6: a0=s_Y[yb+kLmksc]; b0=s_cmum[bml];
                  a1=s_Y[yb+kLmkcs]; b1=s_smum[bml]; break;
          case 7: a0=s_Y[yb+kLmkscN]; b0=s_smu[bml];
                  a1=s_Y[yb+kLmkcsN]; b1=s_cmu[bml]; break;
        }
        DD term = dd_add(ozaki3_mul_d(a0, b0), ozaki3_mul_d(a1, b1));
        if (need_neg) { term.hi = -term.hi; term.lo = -term.lo; }
        dd_acc = dd_add(dd_acc, term);
      }
      acc = dd_to_double(dd_acc);
      }  // end !plain_tf32
      size_t cfg_full = (size_t)config * (size_t)ns_local *
                        (size_t)nZeta * (size_t)nThetaEff;
      size_t idx = cfg_full + (size_t)((jF_local * nZeta + k) * nThetaEff + l);
      switch (n) {
        case 0:  r1_e[idx] = acc; break;
        case 1:  r1_o[idx] = acc; break;
        case 2:  ru_e[idx] = acc; break;
        case 3:  ru_o[idx] = acc; break;
        case 4:  rv_e[idx] = acc; break;
        case 5:  rv_o[idx] = acc; break;
        case 6:  z1_e[idx] = acc; break;
        case 7:  z1_o[idx] = acc; break;
        case 8:  zu_e[idx] = acc; break;
        case 9:  zu_o[idx] = acc; break;
        case 10: zv_e[idx] = acc; break;
        case 11: zv_o[idx] = acc; break;
        case 12: lu_e[idx] = acc; break;
        case 13: lu_o[idx] = acc; break;
        case 14: lv_e[idx] = acc; break;
        case 15: lv_o[idx] = acc; break;
      }
    }
  }
}




// ---------------------------------------------------------------------------
// Batched int8-Ozaki scatter GEMM. The per-tile int8 kernel above runs one
// micro-GEMM per (surface, zeta) block and is staging-bound; this
// formulation folds (config, surface, zeta) into one true GEMM row axis,
//   out[B, (l, ch)] = sum_(m,q) Yspec[B, (m, q)] * W[(m, q), (l, ch)],
// with B = n_config * ns_local * nZeta, K = 16 * mpol, N = 16 * l-cells.
// The basis-side matrix W is constant per Reshape: its limbs and column
// exponents build once per shape. Per iteration only the spec rows are
// sliced (eight 7-bit limbs after per-row scaling) and the banded s8 GEMM
// runs with exact s32 accumulation.

// W[(m, q), (l, ch)]: the (q, channel-group) table of the per-tile kernel,
// with the parity mask folded in.
__global__ void k_i8b_build_w(int mpol, int nThetaReduced,
                              const double* __restrict__ cosmu,
                              const double* __restrict__ sinmu,
                              const double* __restrict__ cosmum,
                              const double* __restrict__ sinmum,
                              double* __restrict__ W, int K, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= K * N) return;
  int mq = idx / N;
  int lch = idx - mq * N;
  int m = mq >> 4;
  int q = mq & 15;
  int l = lch >> 4;
  int ch = lch & 15;
  double v = 0.0;
  bool m_even = ((m & 1) == 0);
  bool parity_match = (ch & 1) ? !m_even : m_even;
  if (m < mpol && l < nThetaReduced && parity_match) {
    int bml = m * nThetaReduced + l;
    double sign = 1.0;
    int bf = -1;
    switch (ch >> 1) {
      case 0:
        if (q == kRmkcc) bf = 0;
        else if (q == kRmkss) bf = 1;
        break;
      case 1:
        if (q == kRmkss) bf = 2;
        else if (q == kRmkcc) bf = 3;
        break;
      case 2:
        if (q == kRmkccN) bf = 0;
        else if (q == kRmkssN) bf = 1;
        break;
      case 3:
        if (q == kZmkcs) bf = 0;
        else if (q == kZmksc) bf = 1;
        break;
      case 4:
        if (q == kZmksc) bf = 2;
        else if (q == kZmkcs) bf = 3;
        break;
      case 5:
        if (q == kZmkcsN) bf = 0;
        else if (q == kZmkscN) bf = 1;
        break;
      case 6:
        if (q == kLmksc) bf = 2;
        else if (q == kLmkcs) bf = 3;
        break;
      default:
        if (q == kLmkcsN) { bf = 0; sign = -1.0; }
        else if (q == kLmkscN) { bf = 1; sign = -1.0; }
        break;
    }
    if (bf >= 0) {
      switch (bf) {
        case 0: v = sign * cosmu[bml]; break;
        case 1: v = sign * sinmu[bml]; break;
        case 2: v = sign * cosmum[bml]; break;
        default: v = sign * sinmum[bml]; break;
      }
    }
  }
  W[(size_t)mq * N + lch] = v;
}

// Column exponents and limbs of W; one thread per column.
__global__ void k_i8b_slice_w(const double* __restrict__ W, int K, int N,
                              signed char* __restrict__ Wl,
                              int* __restrict__ eW) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= N) return;
  int e = INT_MIN;
  for (int kk = 0; kk < K; ++kk) {
    double v = W[(size_t)kk * N + col];
    if (v != 0.0) e = max(e, ilogb(v));
  }
  eW[col] = e;
  for (int kk = 0; kk < K; ++kk) {
    double v = W[(size_t)kk * N + col];
    double r = (e == INT_MIN) ? 0.0 : ldexp(v, -(e + 2));
    #pragma unroll
    for (int pl = 0; pl < 8; ++pl) {
      double scaled = r * 128.0;
      int limb = (int)rint(scaled);
      r = scaled - (double)limb;
      Wl[(size_t)pl * K * N + (size_t)kk * N + col] = (signed char)limb;
    }
  }
}

// Row exponents of the spec matrix. Row b = (cfg, jF, k); element (m, q)
// reads Y[((cfg * ns + jF) * mpol + m) * kBatch + q) * nZeta + k].
__global__ void k_i8b_row_exp(int n_config, int ns_local, int mpol,
                              int nZeta, const double* __restrict__ Y,
                              int* __restrict__ eY) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int B = n_config * ns_local * nZeta;
  if (b >= B) return;
  int k = b % nZeta;
  int cj = b / nZeta;
  size_t base = ((size_t)cj * (size_t)mpol * (size_t)kBatch) *
                (size_t)nZeta + (size_t)k;
  int e = INT_MIN;
  // The K axis is laid out 16 per mode (q in [0, 16)); only the first
  // kBatch q-slots exist in the spec block, the rest are zero padding.
  int K = 16 * mpol;
  for (int mq = 0; mq < K; ++mq) {
    int m = mq >> 4;
    int q = mq & 15;
    if (q >= kBatch) continue;
    double v = Y[base + ((size_t)m * kBatch + q) * (size_t)nZeta];
    if (v != 0.0) e = max(e, ilogb(v));
  }
  eY[b] = e;
}



// k_tau_minmax: one block per config; threads cooperate to find min and max of
// tau across the per-config half-grid tau array, write 2 scalars [min, max]
// to out2[config*2:] on device. Replaces the host-side min/max scan after tau D2H.
// Batched execution: n_config via blockIdx.x. Per-config tau stride is `total`
// doubles (ns_h * nZnT).
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

// k_apply_rz_pcr: parallel cyclic reduction for tridiagonal solve. Replaces
// k_apply_rz_thomas (one thread per (mn) block, serial sweep). Each block:
//   - Loads N = jMax - jMin[mn] rows into shared memory (a, d, b, c[0..num_basis-1])
//   - log2(N) PCR passes: each row at distance k from both sides is eliminated
//     in parallel; new (a, d, b, c) coefficients fall back to distance 2k
//   - Final: x_i = c_i / d_i; write back to global c_inout
// System convention: b_i*x_{i-1} + d_i*x_i + a_i*x_{i+1} = c_i (a=super, b=sub).
// a/d/b are READ-ONLY (Thomas mutated them in-place; PCR doesn't, which makes
// the kernel safe for future persistent-precond-input use).
// Batched execution: configuration axis on blockIdx.y. a_in/d_in/b_in per-config
// (mnsize*ns_total). c_inout per-config (mnsize*num_basis*ns_total). jMin shared.
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

// k_apply_rz_thomas_serial: serial Thomas elimination per (cfg, mn) row in
// the host loop's exact order and association (TridiagonalSolveSerial),
// for trajectory comparisons against the CPU build. One thread per row;
// the mutable elimination ratios live in local memory (ns_total <= 1024
// by the shape guard). Diagnostic path, gated by
// VMECPP_CPU_ORDER_RZSOLVE=1.
__global__ void k_apply_rz_thomas_serial(
    int n_config, int mnsize, int ns_total, int num_basis,
    const int* __restrict__ jMin, int jMax,
    const double* __restrict__ a_in, const double* __restrict__ d_in,
    const double* __restrict__ b_in, double* __restrict__ c_inout,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int config = idx / mnsize;
  int mn = idx - config * mnsize;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int j0 = jMin[mn];
  if (jMax - j0 <= 0) return;
  size_t row = ((size_t)config * (size_t)mnsize + (size_t)mn) *
               (size_t)ns_total;
  size_t cfg_c = (size_t)config * (size_t)mnsize * (size_t)num_basis *
                 (size_t)ns_total;
  double a_loc[1024];
  for (int j = j0; j < jMax; ++j) a_loc[j] = a_in[row + j];
  a_loc[j0] /= d_in[row + j0];
  for (int j = j0 + 1; j < jMax - 1; ++j) {
    const double denominator = d_in[row + j] - a_loc[j - 1] * b_in[row + j];
    a_loc[j] /= denominator;
  }
  for (int ib = 0; ib < num_basis; ++ib) {
    double* c = c_inout + cfg_c + (size_t)(mn * num_basis + ib) * ns_total;
    c[j0] /= d_in[row + j0];
    for (int j = j0 + 1; j < jMax; ++j) {
      const double denominator = d_in[row + j] - a_loc[j - 1] * b_in[row + j];
      c[j] = (c[j] - c[j - 1] * b_in[row + j]) / denominator;
    }
    for (int j = jMax - 2; j > j0 - 1; --j) {
      c[j] -= a_loc[j] * c[j + 1];
    }
  }
}

// k_apply_rz_thomas_block: same serial Thomas elimination as
// k_apply_rz_thomas_serial, but one block per (config, mn) row with the
// elimination ratios in dynamic shared memory instead of a fixed local
// array. This removes the ns_total <= 1024 ceiling that the PCR solver
// inherits from the 1024 threads-per-block limit: the radial recurrence is
// sequential, so a single thread walks it while the block's other threads
// only cooperate on the load. The dynamic shared array is sized to jMax by
// the launch. Used for ns_total > 1024.
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

// ----------------------------------------------------------------------------
// k_apply_rz_pcr_fp32
//
// Single-precision variant of k_apply_rz_pcr. The matrix coefficients
// a_in, d_in, b_in and right-hand sides c_inout are read from FP64
// global memory, cast to FP32 on entry to shared memory, the parallel
// cyclic reduction proceeds entirely in FP32, and the FP32 solution is
// cast back to FP64 on writeback to c_inout. This kernel is the first
// stage of the Carson-Higham staged iterative refinement scheme: the
// FP32 solve produces an approximate solution x0, the FP64 residual
// computation kernel below computes r0 = b - A*x0 using the original
// FP64 coefficients and right-hand side, the second invocation of this
// kernel solves r0 to obtain a correction dx, and the final addition
// x = x0 + dx is performed in FP64 by k_rz_add_correction. The
// block-tridiagonal RZ preconditioner matrices are well-conditioned in
// practice, with the radial-direction condition number bounded by the
// ratio of the maximum to minimum diagonal element, and the FP32 solve
// preserves the leading 6 to 7 significant figures of the FP64 result;
// the IR step recovers the remaining FP64 precision with one residual
// correction.
//
// Shared memory layout matches k_apply_rz_pcr but uses float instead
// of double: [N floats a][N floats d][N floats b][2*N floats c]. The
// smem requirement is therefore halved relative to the FP64 kernel,
// improving occupancy on Ada at large ns.
// ----------------------------------------------------------------------------
__global__ void k_apply_rz_pcr_fp32(int n_config, int mnsize, int ns_total,
                                      int num_basis,
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

  extern __shared__ float fmem[];
  float* s_a = fmem;
  float* s_d = fmem + ns_total;
  float* s_b = fmem + 2 * ns_total;
  float* s_c = fmem + 3 * ns_total;

  if (tid < N) {
    int gi = tid + j0;
    s_a[tid] = static_cast<float>(a_in[cfg_mat + (size_t)(mn * ns_total + gi)]);
    s_d[tid] = static_cast<float>(d_in[cfg_mat + (size_t)(mn * ns_total + gi)]);
    s_b[tid] = static_cast<float>(b_in[cfg_mat + (size_t)(mn * ns_total + gi)]);
    for (int ib = 0; ib < num_basis; ++ib) {
      s_c[tid * 2 + ib] = static_cast<float>(
          c_inout[cfg_c + (size_t)((mn * num_basis + ib) * ns_total + gi)]);
    }
  }
  __syncthreads();

  for (int k = 1; k < N; k <<= 1) {
    float a_new = 0.f, d_new = 0.f, b_new = 0.f;
    float c_new0 = 0.f, c_new1 = 0.f;
    if (tid < N) {
      int i_prev = tid - k;
      int i_next = tid + k;
      float alpha = (i_prev >= 0) ? -s_b[tid] / s_d[i_prev] : 0.f;
      float beta  = (i_next <  N) ? -s_a[tid] / s_d[i_next] : 0.f;
      d_new = s_d[tid];
      if (i_prev >= 0) d_new += alpha * s_a[i_prev];
      if (i_next <  N) d_new += beta  * s_b[i_next];
      b_new = (i_prev >= 0) ? alpha * s_b[i_prev] : 0.f;
      a_new = (i_next <  N) ? beta  * s_a[i_next] : 0.f;
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
      c_inout[cfg_c + (size_t)((mn * num_basis + ib) * ns_total + gi)] =
          static_cast<double>(s_c[tid * 2 + ib] / s_d[tid]);
    }
  }
}

// ----------------------------------------------------------------------------
// k_rz_compute_residual_fp64
//
// FP64 residual computation for Carson-Higham staged iterative
// refinement. Given the original FP64 matrix coefficients a, d, b
// (super, diag, sub) for the radial tri-diagonal and the original
// FP64 right-hand sides stored in c_orig, plus the FP32-solved
// approximate solution x stored in c_inout, computes
//   r_i = c_orig_i - (b_i * x_{i-1} + d_i * x_i + a_i * x_{i+1})
// in FP64 and writes the residual to c_inout in place. The next
// invocation of k_apply_rz_pcr_fp32 then solves A * dx = r to
// produce the correction in FP32; k_rz_add_correction adds dx to
// the saved x0 to yield the refined solution.
//
// The residual computation is dispatched with the same grid layout
// as k_apply_rz_pcr: blockIdx.x = mn, blockIdx.y = config, with one
// thread per radial index. The boundary conditions x_{-1} and x_{N}
// are taken as zero, matching the convention used by the PCR
// solver's boundary handling.
// ----------------------------------------------------------------------------
__global__ void k_rz_compute_residual_fp64(int n_config, int mnsize,
                                             int ns_total, int num_basis,
                                             const int* __restrict__ jMin,
                                             int jMax,
                                             const double* __restrict__ a_in,
                                             const double* __restrict__ d_in,
                                             const double* __restrict__ b_in,
                                             const double* __restrict__ c_orig,
                                             const double* __restrict__ x_in,
                                             double* __restrict__ r_out,
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
  if (tid >= N) return;
  size_t cfg_mat = (size_t)config * (size_t)mnsize * (size_t)ns_total;
  size_t cfg_c   = (size_t)config * (size_t)mnsize * (size_t)num_basis *
                   (size_t)ns_total;
  int gi = tid + j0;
  double a = a_in[cfg_mat + (size_t)(mn * ns_total + gi)];
  double d = d_in[cfg_mat + (size_t)(mn * ns_total + gi)];
  double b = b_in[cfg_mat + (size_t)(mn * ns_total + gi)];
  for (int ib = 0; ib < num_basis; ++ib) {
    size_t idx_self = cfg_c + (size_t)((mn * num_basis + ib) * ns_total + gi);
    double x_self = x_in[idx_self];
    double x_prev = (tid > 0)
        ? x_in[cfg_c + (size_t)((mn * num_basis + ib) * ns_total + (gi - 1))]
        : 0.0;
    double x_next = (tid + 1 < N)
        ? x_in[cfg_c + (size_t)((mn * num_basis + ib) * ns_total + (gi + 1))]
        : 0.0;
    double r = c_orig[idx_self] - (b * x_prev + d * x_self + a * x_next);
    r_out[idx_self] = r;
  }
}

// ----------------------------------------------------------------------------
// k_rz_add_correction
//
// Adds the FP32-computed correction (stored in c_corr) to the saved
// FP64 approximate solution (stored in x_saved), writing the refined
// FP64 result to c_inout. Used as the final stage of the Carson-Higham
// IR pipeline.
// ----------------------------------------------------------------------------
__global__ void k_rz_add_correction(int n_config, int mnsize, int ns_total,
                                      int num_basis,
                                      const int* __restrict__ jMin, int jMax,
                                      const double* __restrict__ x_saved,
                                      const double* __restrict__ c_corr,
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
  if (tid >= N) return;
  int gi = tid + j0;
  size_t cfg_c = (size_t)config * (size_t)mnsize * (size_t)num_basis *
                 (size_t)ns_total;
  for (int ib = 0; ib < num_basis; ++ib) {
    size_t idx = cfg_c + (size_t)((mn * num_basis + ib) * ns_total + gi);
    c_inout[idx] = x_saved[idx] + c_corr[idx];
  }
}

// k_assemble_rz_preconditioner: device-side port of IdealMhdModel::
// assembleRZPreconditioner. Reads per-side persistent precond-matrix outputs
// (arm/brm/ard/brd half/full-grid for R, similarly for Z, plus shared cxd) and
// writes the tri-diagonal coefficients ar/dr/br (R) and az/dz/bz (Z) directly
// in the (mn, jF_global) transposed layout that k_apply_rz_pcr consumes,
// skipping the host transpose + 6 H2Ds previously done in ApplyRZPreconditionerCuda.
//
// Thread mapping: blockIdx.x = mn, blockIdx.y * blockDim.x + threadIdx.x = jF.
// jMin[mn] is written once per mn by the jF==0 thread.
//
// Outside the active force range [nsMinF, min(nsMaxF, jMax)) the outputs are
// zeroed (the PCR solver reads the full ns_total range but only loads
// [jMin[mn], jMax)). Edge pedestal + ZC_00(NS) stabilization at jF == ns_total-1
// fire only when lcfs_owning AND that jF is in range (i.e., free-boundary with
// vacuum active; in fixed-boundary jF = ns-1 is out of range and the multiplied
// values would be zero anyway, matching CPU).
// Batched execution: configuration axis on blockIdx.z. arm/brm/azm/bzm per-config half-grid
// 2D (ns_h*2). ard/brd/azd/bzd per-config force-grid 2D (ns_force_local*2).
// cxd per-config (ns_force_local). aR/dR/bR/aZ/dZ/bZ per-config matrices
// (mnsize*ns_total). d_jMin is shared (per-mn). Pass ns_h explicitly.
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

  // diagonal: jF-th forces full-grid pos. Match CPU FP-evaluation order
  // exactly (left-to-right: cxd * n * nfp * n * nfp = ((((cxd*n)*nfp)*n)*nfp)
  // i.e. four double*int multiplications, NOT cxd * (n*nfp)^2 which would be
  // two double*int multiplications with different rounding).
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

  // Special: m=1 at jF=1 ⇒ dr += br, dz += bz.
  if (jF == 1 && m == 1) {
    d_R += b_R;
    d_Z += b_Z;
  }

  // Edge pedestal + ZC_00 stabilization at the LCFS row (jF == ns_total - 1).
  // CPU applies this regardless of lfreeb, but in fixed-boundary the main loop
  // doesn't reach jF = ns - 1 (jMax = ns - 1, loop is exclusive), so the
  // multiplied values are zero × pedestal = zero. Our in_range check excludes
  // jF = ns-1 in fixed-boundary (jF_upper = ns - 1), so we never reach here for
  // that case, with the same result.
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

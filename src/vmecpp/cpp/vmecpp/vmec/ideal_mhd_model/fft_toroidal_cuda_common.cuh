// Internal header for the CUDA iteration body: shared device
// helpers, the persistent device-state struct, kernel
// declarations, and the run-scoped globals. Included by the
// fft_toroidal_cuda_*.cu translation units only.
#pragma once

// CUDA implementation of the toroidal Fourier transforms used by the vmecpp
// ideal-MHD iteration body. Both directions of the transform are resident on
// the device: the spectrum-to-real-space mapping is implemented by
// FourierToReal3DSymmFastPoloidalCuda, and the real-space-to-spectrum mapping
// by ForcesToFourier3DSymmFastPoloidalCuda. The two paths share the device
// state held in CudaToroidalState, including the persistent intermediate
// buffers d_X and d_Y, the cuFFT plans, and the poloidal basis tables.
//
// Forward pipeline (spectrum-to-real-space, mirroring the structure of
// fft_toroidal.cc's FourierToReal3DSymmFastPoloidalFft):
//   1. Stage the six spectral coefficient arrays (rmncc, rmnss, zmnsc, zmncs,
//      lmnsc, lmncs) into d_specs_block on the device.
//   2. k_fill_spectra populates X_batch[jF][m][q][n] for q in [0, 12). Empty
//      combinations (jF below jMin for the given mode) are written as zero
//      directly by the kernel rather than requiring a separate memset.
//   3. A cufftExecZ2D batched over (jF, m, q) executes all toroidal
//      transforms in a single call, producing the real-space intermediate
//      Y. An equivalent hand-coded radix-8x3 decomposition is available
//      behind VMECPP_FFT_RADIX.
//   4. k_scatter_main_and_con accumulates the sixteen even-parity and
//      odd-parity outputs (r1_e/o, ru_e/o, rv_e/o, z1_e/o, zu_e/o, zv_e/o,
//      lu_e/o, lv_e/o) and the two constraint outputs (rCon, zCon) in a
//      single pass over Y.
//   5. The eighteen output arrays are exposed to the iteration controller
//      through pointers held by CudaToroidalState; copy-back to host is
//      performed only at the surfaces where the host iteration controller
//      genuinely consumes the values.
//
// The poloidal basis arrays (fb.cosmu, fb.sinmu, fb.cosmum, fb.sinmum) and
// the toroidal mode-scaling factors nscale are invariant across the run and
// are staged once in CudaToroidalState::Init rather than reissued each
// invocation. Per-configuration extensions to all of the above are handled
// by the n_config_max dimension on the device buffers and by the
// configuration axis applied to each kernel's grid; the single-configuration
// path remains a special case at n_config_max = 1.
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda.h"

#ifdef VMECPP_USE_CUDA

#ifdef VMECPP_USE_HIP
// cooperative_groups and mma.h are used only by the NVIDIA-only kernels,
// which are compiled out below.
#include "vmecpp/vmec/ideal_mhd_model/hip_compat.cuh"
#else
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <mma.h>
#endif  // VMECPP_USE_HIP

#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_toroidal.h"  // partial-DFT fallback
#include "vmecpp/vmec/ideal_mhd_model/phase_timer.h"

namespace vmecpp {

// File-scope per-configuration caches populated as side effects of the
// device-to-host transfers already performed by the corresponding kernel
// wrappers. The iteration controller consumes the cached values through
// the GetResidualsPerCfgCache* and related accessors declared in
// fft_toroidal_cuda.h. Maintaining these caches at file scope avoids
// reissuing the per-configuration transfer separately for each consumer
// and permits the synchronization that the wrapper already performs to
// serve as the cache-validity boundary.
//
// Layout of the residual caches: 3 * n_cfg doubles, ordered as
// [fResR_0, fResZ_0, fResL_0, fResR_1, fResZ_1, fResL_1, ...]. The
// caches are resized lazily by the corresponding wrapper when n_cfg
// changes between successive iterations.
extern std::vector<double> g_residuals_invar_cache;
extern std::vector<double> g_residuals_precd_cache;
// Force-norm scalar cache. Layout: 2 * n_cfg doubles ordered as
// [sum_rz_0, sum_l_0, sum_rz_1, sum_l_1, ...]. Populated by
// ComputeForceNormsCuda. The iteration controller derives per-configuration
// normalization factors fNormRZ and fNormL by combining these sums with
// the per-configuration energyDensity values that the pressure-and-energy
// cache below holds.
extern std::vector<double> g_fnorm_scalars_cache;
// Jacobian-extrema cache. Layout: 2 * n_cfg doubles ordered as
// [minTau_0, maxTau_0, minTau_1, maxTau_1, ...]. Populated by
// ComputeJacobianCuda. The host-side per-configuration bad-Jacobian
// decision is computed from these values as
// (minTau[c] * maxTau[c] < 0) || !std::isfinite(minTau[c] * maxTau[c]).
extern std::vector<double> g_jac_minmax_cache;
// 3*n_cfg [thermalEnergy_0, magneticEnergy_0, mhdEnergy_0,
//          thermalEnergy_1, magneticEnergy_1, mhdEnergy_1, ...].
// Populated by PressureAndEnergiesCuda's added per-cfg D2H. Becomes valid
// after the caller's next stream sync (same as the existing single-cfg
// scalar writes).
extern std::vector<double> g_pressure_scalars_cache;
// n_cfg per-cfg plasmaVolume. Populated by UpdateVolumeCuda's added per-cfg
// D2H. Becomes valid after the existing single-cfg sync that
// UpdateVolumeCuda already does.
extern std::vector<double> g_plasma_volume_cache;

const std::vector<double>& GetResidualsPerCfgCacheInvar();

const std::vector<double>& GetResidualsPerCfgCachePrecd();

const std::vector<double>& GetFnormScalarsPerCfgCache();

const std::vector<double>& GetJacMinmaxPerCfgCache();

const std::vector<double>& GetPressureScalarsPerCfgCache();

// n_cfg per-cfg fNorm1 (reciprocal rzNorm over each configuration's own
// device position state). Populated by ComputeForceNormsCuda at the
// force-norm cadence; cfg 0 matches the host scalar bit for bit.
extern std::vector<double> g_fnorm1_per_cfg_cache;

const std::vector<double>& GetPlasmaVolumePerCfgCache();

const std::vector<double>& GetFnorm1PerCfgCache();

// Effective configuration count for the current run. Negative means
// unread; GetNConfigMaxCuda resolves it from VMECPP_N_CONFIG_MAX on
// first use and ResetCudaStateForNewVmecRun re-reads it at the start
// of every Vmec::run, so the count can change between runs in one
// process while staying frozen within a run.
extern int g_n_config_run;

// Run-scoped boolean gates for the per-configuration multigrid upscale.
// Re-read at the start of every Vmec::run alongside the configuration
// count, so mixed batched and single runs in one process do not freeze
// the first run's setting: a process whose first run leaves the upscale
// unset would otherwise disable the per-configuration stage transition
// for every later distinct-mode run.
extern int g_batch_upscale_env;
extern int g_batch_upscale_kernel_env;
static int RunEnvFlag(int* cache, const char* name) {
  if (*cache < 0) {
    const char* e = std::getenv(name);
    *cache = (e != nullptr && std::atoi(e) > 0) ? 1 : 0;
  }
  return *cache;
}

// ============================================================================
// Double-single (DD) FP32 primitives for the FP32 substitution research path.
// Two FP32 numbers (hi, lo) represent a single quantity with ~48 bits of
// mantissa (vs 23 for plain FP32 and 52 for FP64). The high part holds the
// "rounded" value, the low part captures the rounding error that ordinary
// FP32 arithmetic would discard.
//
// Used to recover FP64-equivalent precision on the accumulators that break
// convergence under naive VMECPP_FFT_FP32=1: force-residual norm, the near-
// axis Jacobian sum, and the spectral inverse-transform reductions in
// k_inverse_scatter / k_scatter_main_and_con. Reference: Knuth's TwoSum and
// the QDP / double-single literature.
//
// These primitives are __host__ __device__ so the same code can be tested
// from CPU. They assume IEEE 754 round-to-nearest semantics on FP32 and
// that the compiler does NOT fuse the additions into FMAs (which would
// invalidate the TwoSum trick). NVCC: pass -fmad=false to the relevant
// translation unit when this path is active, OR use __fadd_rn explicitly.
// ============================================================================
__device__ __forceinline__ void fp32_twosum(float a, float b,
                                                       float& s, float& e) {
  // Knuth's TwoSum: s = a + b (rounded), e = exact error so that
  // a + b == s + e to infinite precision.
  s = __fadd_rn(a, b);
  float a_prime = __fadd_rn(s, -b);
  float b_prime = __fadd_rn(s, -a_prime);
  float delta_a = __fadd_rn(a, -a_prime);
  float delta_b = __fadd_rn(b, -b_prime);
  e = __fadd_rn(delta_a, delta_b);
}

__device__ __forceinline__ void fp32_quicktwosum(float a, float b,
                                                            float& s, float& e) {
  // Fast TwoSum assuming |a| >= |b|. Saves three FLOPs vs the symmetric form.
  s = __fadd_rn(a, b);
  e = __fadd_rn(b, __fadd_rn(a, -s));
}

// DD pair (FP32-hi, FP32-lo) representing one ~48-bit value. The invariant
// |lo| <= 0.5 * ulp(hi) is preserved by renormalize after each operation.
struct DD {
  float hi;
  float lo;
};

__device__ __forceinline__ DD dd_add_f(DD a, float b) {
  // Add a plain FP32 to a DD pair. Knuth's TwoSum on the hi parts, then
  // accumulate the low correction. Standard double-single add-FP32 routine.
  float s, e;
  fp32_twosum(a.hi, b, s, e);
  float t = __fadd_rn(e, a.lo);
  DD r;
  fp32_quicktwosum(s, t, r.hi, r.lo);
  return r;
}

__device__ __forceinline__ DD dd_add(DD a, DD b) {
  // Add two DD pairs. Six TwoSums in the general case, renormalized at
  // the end. Matches Dekker's add2 from the original double-double paper.
  float s, e;
  fp32_twosum(a.hi, b.hi, s, e);
  float t = __fadd_rn(__fadd_rn(a.lo, b.lo), e);
  DD r;
  fp32_quicktwosum(s, t, r.hi, r.lo);
  return r;
}

__device__ __forceinline__ DD dd_from_f(float x) {
  DD r; r.hi = x; r.lo = 0.0f; return r;
}

__device__ __forceinline__ double dd_to_double(DD a) {
  // Promote the DD pair to FP64 for output. The (hi + lo) sum is computed
  // in FP64 so the lo's contribution is preserved.
  return (double)a.hi + (double)a.lo;
}

// dd_add_d: add an FP64 value to a DD-pair accumulator. The FP64 input is
// effectively split into FP32 hi/lo via cast + residual, then the standard
// DD add is applied. Used by Path 1 (FP64 mults, DD-pair sums) where the
// inputs are FP64-precise but the accumulator wants DD compensation against
// √n error growth across the inner loop.
__device__ __forceinline__ DD dd_add_d(DD a, double b) {
  float b_hi = (float)b;
  float b_lo = (float)(b - (double)b_hi);
  // (b_hi, b_lo) is a DD-pair representing b to ~48-bit precision.
  DD bd; bd.hi = b_hi; bd.lo = b_lo;
  return dd_add(a, bd);
}

// TwoProduct via Dekker: splits a, b into FP32 hi/lo halves, then computes
// the exact product as a DD pair (hi, lo). Six FP32 multiplies + a few
// adds; ~96-bit precision when both operands are FP32. Used by Path 2
// (DD × DD multiply) where inputs are FP32 and we want to recover full
// precision in the product before accumulating.
__device__ __forceinline__ void fp32_twoprod_split(float a, float& ahi, float& alo) {
  // Dekker split with K = 2^12 + 1 = 4097 (single-precision boundary).
  constexpr float K = 4097.0f;
  float c = __fmul_rn(K, a);
  float a_big = __fadd_rn(c, __fadd_rn(a, -c));
  ahi = a_big;
  alo = __fadd_rn(a, -a_big);
}

__device__ __forceinline__ DD fp32_twoprod(float a, float b) {
  float ahi, alo, bhi, blo;
  fp32_twoprod_split(a, ahi, alo);
  fp32_twoprod_split(b, bhi, blo);
  float p = __fmul_rn(a, b);
  float e = __fadd_rn(
      __fadd_rn(
        __fadd_rn(__fmul_rn(ahi, bhi), -p),
        __fmul_rn(ahi, blo)),
      __fadd_rn(__fmul_rn(alo, bhi), __fmul_rn(alo, blo)));
  DD r; r.hi = p; r.lo = e; return r;
}

// Ozaki-style FP64 multiply by 2-slice FP32 splitting. Splits each FP64
// operand into FP32 hi + FP32 lo residual, computes the four cross-
// products in FP32 hardware, then sums into a DD pair. ~50-bit precision
// per product when both operands are FP64; ~26-bit if both are FP32.
// Four FP32 mults + four adds, vs one FP64 mult.
__device__ __forceinline__ DD ozaki_mul_d(double a, double b) {
  float ahi = (float)a;
  float alo = (float)(a - (double)ahi);
  float bhi = (float)b;
  float blo = (float)(b - (double)bhi);
  float p_hh = __fmul_rn(ahi, bhi);
  float p_hl = __fmul_rn(ahi, blo);
  float p_lh = __fmul_rn(alo, bhi);
  float p_ll = __fmul_rn(alo, blo);
  // Sum in descending magnitude order with TwoSum to preserve precision.
  float t0, e0;
  fp32_twosum(p_hh, p_hl, t0, e0);
  float t1, e1;
  fp32_twosum(t0, p_lh, t1, e1);
  float t2 = __fadd_rn(t1, p_ll);
  // Pack {t2, e0 + e1} as the DD result.
  DD r;
  r.hi = t2;
  r.lo = __fadd_rn(e0, e1);
  return r;
}

// Veltkamp split: a (FP32) -> (hi, lo) where hi has 12 mantissa bits and
// lo has 12 mantissa bits, and hi + lo == a exactly. Guarantees that any
// FP32 product hi_a * hi_b, hi_a * lo_b, lo_a * hi_b, lo_a * lo_b is
// EXACT in FP32 (24-bit mantissa holds 12+12 bits without rounding).
__device__ __forceinline__ void veltkamp_split(float a, float& hi, float& lo) {
  constexpr float K = 4097.0f;  // 2^12 + 1
  float c = __fmul_rn(K, a);
  hi = __fadd_rn(c, -__fadd_rn(c, -a));
  lo = __fadd_rn(a, -hi);
}

// Dekker TwoProduct: exact product of two FP32 numbers as a DD pair.
// p + e = a*b to infinite precision; p = round(a*b). Uses Veltkamp to
// make the four sub-products exact in FP32.
__device__ __forceinline__ DD two_product_dekker(float a, float b) {
  float p = __fmul_rn(a, b);
  float a_hi, a_lo, b_hi, b_lo;
  veltkamp_split(a, a_hi, a_lo);
  veltkamp_split(b, b_hi, b_lo);
  float e1 = __fadd_rn(__fmul_rn(a_hi, b_hi), -p);
  float e2 = __fadd_rn(e1, __fmul_rn(a_hi, b_lo));
  float e3 = __fadd_rn(e2, __fmul_rn(a_lo, b_hi));
  float e  = __fadd_rn(e3, __fmul_rn(a_lo, b_lo));
  DD r; r.hi = p; r.lo = e; return r;
}

// 2-slice Ozaki with FP64 inputs and Veltkamp-Dekker exact partial products.
// Verified standalone vs FP64 reference: max relative error 2.8e-13, mean
// 1e-15 across 10000 random pairs spanning 1e-15 to 10 in magnitude.
__device__ __forceinline__ DD ozaki3_mul_d(double a, double b) {
  float a32 = (float)a;
  float a_r = (float)(a - (double)a32);
  float b32 = (float)b;
  float b_r = (float)(b - (double)b32);
  DD p00 = two_product_dekker(a32, b32);
  DD p01 = two_product_dekker(a32, b_r);
  DD p10 = two_product_dekker(a_r, b32);
  DD p11 = two_product_dekker(a_r, b_r);
  // Accumulate four DD pairs via dd_add_f cascade.
  DD acc = p00;
  acc = dd_add_f(acc, p01.hi); acc = dd_add_f(acc, p01.lo);
  acc = dd_add_f(acc, p10.hi); acc = dd_add_f(acc, p10.lo);
  acc = dd_add_f(acc, p11.hi); acc = dd_add_f(acc, p11.lo);
  return acc;
}

// Carson-Higham IR state. Lives at vmecpp:: file-scope (not anonymous
// namespace) so the host-side iteration controller in ideal_mhd_model.cc
// can update it via SetIRResidualSum each iter. The kernel dispatchers
// read these to switch between FP32/TF32 fast paths and FP64 precise
// paths based on the current residual sum. Threshold and gating come
// from env (VMECPP_IR_STAGED, VMECPP_IR_THRESHOLD).
extern double g_ir_residual_sum;
extern double g_ir_threshold;
extern int    g_ir_staged;

static inline void init_ir_env() {
  if (g_ir_staged < 0) {
    const char* e_staged = std::getenv("VMECPP_IR_STAGED");
    g_ir_staged = (e_staged && std::atoi(e_staged) > 0) ? 1 : 0;
    const char* e_thr = std::getenv("VMECPP_IR_THRESHOLD");
    if (e_thr) g_ir_threshold = std::strtod(e_thr, nullptr);
    if (g_ir_staged) {
      std::fprintf(stderr, "[fft_toroidal_cuda] Carson-Higham IR ENABLED "
                           "(VMECPP_IR_STAGED=1, threshold=%.3e)\n",
                           g_ir_threshold);
    }
  }
}

// (anonymous namespace hoisted: the split translation units share these symbols)

// Slot indices for the 12 quantities transformed per (jF, m).
enum Slot {
  kRmkcc = 0,  kRmkss = 1,  kRmkccN = 2, kRmkssN = 3,
  kZmksc = 4,  kZmkcs = 5,  kZmkscN = 6, kZmkcsN = 7,
  kLmksc = 8,  kLmkcs = 9,  kLmkscN = 10, kLmkcsN = 11,
};
constexpr int kBatch = 12;

// CUDA and cuFFT failures throw instead of aborting so an embedding
// process (the Python interpreter in particular) receives a catchable
// error rather than dying. The throw unwinds out of Vmec::run; pybind
// translates it into a Python RuntimeError, and vmec_standalone
// terminates with the message. Device state after a thrown CUDA error
// is unspecified; the next run's ResetCudaStateForNewVmecRun plus the
// shape-triggered Reshape rebuild it.
static void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    char msg[256];
    std::snprintf(msg, sizeof(msg), "[fft_toroidal_cuda] CUDA error at %s: %s",
                  what, cudaGetErrorString(err));
    std::fprintf(stderr, "%s\n", msg);
    throw std::runtime_error(msg);
  }
}
static void cufft_check(cufftResult res, const char* what) {
  if (res != CUFFT_SUCCESS) {
    char msg[256];
    std::snprintf(msg, sizeof(msg), "[fft_toroidal_cuda] cuFFT error at %s: %d",
                  what, (int)res);
    std::fprintf(stderr, "%s\n", msg);
    throw std::runtime_error(msg);
  }
}

// Raw upper estimate, in bytes, of the persistent device allocation at the
// given shape and configuration count, without the safety margin or the
// context cushion that the admission pre-flight adds on top. Shared by
// CudaVramBudgetCuda and by the per-Reshape bookkeeping that credits a
// follow-up run with the memory the next Reshape frees. The coefficient
// families: spectral-coefficient blocks (specs, force spectra, decomposed
// shadow, dealias intermediates, position/velocity/backup/final/prev
// state, lambda preconditioner, RZ tridiagonal rows), full-grid real-space
// arrays (outputs, forces, constraint terms, preconditioner inputs),
// half-grid arrays (jacobian, metric, fields, pressure), and the cuFFT
// scratch with its plan workspace.
long long CudaBudgetRawBytes(long long n_cfg, long long ns, long long mpol,
                             long long ntor, long long nZeta,
                             long long nThetaEff);


// State-region globals and helpers referenced by the struct below
// (defined in fft_toroidal_cuda_state.cu).
extern bool g_free_boundary_run;
extern bool g_vacuum_edge_run;
extern bool g_sync_elide_run;

// =========================================================================
// Process-static CUDA state
// =========================================================================
struct CudaToroidalState {
  bool initialized = false;
  // Cached shape parameters.
  int n_cached = -1, nfp_cached = -1, mpol_cached = -1;
  int ns_local_cached = -1, ntor_cached = -1, nhalf_cached = -1;
  int nThetaReduced_cached = -1, nThetaEff_cached = -1, nZeta_cached = -1;
  int ns_con_local_cached = -1;

  // Maximum concurrent equilibria the device buffers are sized for. Each
  // per-config buffer's size is multiplied by n_config_max and every kernel
  // carries the configuration axis on its launch grid; at the default of 1
  // the single-call path is preserved bit-exact.
  int n_config_max = 1;

  // Persistent device buffers + stream.
  cudaStream_t stream = nullptr;
  cufftHandle cufft_plan = 0;
#ifndef VMECPP_USE_HIP
  // cuBLAS handle for the FP32 GEMM-based scatter (Path 4 of the FP32
  // substitution research). Created lazily when first needed. Stream-bound to
  // S.stream so GEMM ops serialize naturally with the surrounding kernels.
  // Part of the NVIDIA-only experiments annex; not built under HIP.
  cublasHandle_t cublas = nullptr;
#endif  // VMECPP_USE_HIP
  // Precomputed basis matrix W[M=mpol*kBatch, N=nThetaReduced*18] for the
  // scatter GEMM. Allocated and populated once per Reshape. FP32 layout.
  float* d_scatter_basis_fp32 = nullptr;
  // Packed FP32 buffers used by the GEMM scatter path. Y_packed has shape
  // (B=n_cfg*ns_local*nZeta, M); out_packed has shape (B, N).
  float* d_scatter_Y_fp32 = nullptr;
  float* d_scatter_out_fp32 = nullptr;
  // Ozaki-at-GEMM-level buffers: each FP64 operand is split into FP32
  // hi/lo slices, four GEMMs are dispatched (hh, hl, lh, ll), and the
  // results summed via DD-pair to recover ~48-bit precision per output.
  float* d_scatter_basis_hi = nullptr;  // W_hi[M, N] FP32
  float* d_scatter_basis_lo = nullptr;  // W_lo[M, N] FP32
  float* d_scatter_Y_hi = nullptr;      // Y_hi[B, M] FP32
  float* d_scatter_Y_lo = nullptr;      // Y_lo[B, M] FP32
  float* d_scatter_out_hh = nullptr;    // GEMM(Y_hi, W_hi) FP32
  float* d_scatter_out_hl = nullptr;    // GEMM(Y_hi, W_lo) FP32
  float* d_scatter_out_lh = nullptr;    // GEMM(Y_lo, W_hi) FP32
  float* d_scatter_out_ll = nullptr;    // GEMM(Y_lo, W_lo) FP32
  size_t scatter_basis_M = 0;  // mpol * kBatch
  size_t scatter_basis_N = 0;  // nThetaReduced * 18

  // Per-kernel cudaEvent timing harness (env-gated, VMECPP_KERNEL_TIMING=1).
  // Each slot: 2 events (start/stop) + accumulated ms + call count. Recorded
  // around the major per-iter kernels; dumped to stderr at program exit
  // via atexit. Slow when enabled (per-call sync); diagnostic only.
  static constexpr int kNumTimedKernels = 19;
  enum TimedKernel {
    TK_CUFFT_INV = 0,        // cufftExecZ2D
    TK_SCATTER = 1,          // k_scatter_main_and_con_v4
    TK_JAC_METRIC_DVDSH = 2, // k_jacobian_metric_dvdsh_atomic
    TK_BCONTRA = 3,          // k_bcontra_bsupuv (heaviest bcontra kernel)
    TK_PRES = 4,             // pressureAndEnergies kernels
    TK_RADIAL_FB = 5,        // k_radial_interior
    TK_CUFFT_FWD = 6,        // cufftExecD2Z
    TK_DECOMPOSE = 7,        // k_decompose_into
    TK_RESIDUALS = 8,        // k_residuals (both calls)
    TK_APPLY_M1 = 9,         // k_apply_m1_preconditioner
    TK_APPLY_RZ = 10,        // k_apply_rz_pcr
    TK_APPLY_LAMBDA = 11,    // k_apply_lambda_preconditioner
    TK_EFFECTIVE_CONSTRAINT = 12,  // k_effective_constraint_force
    TK_DEALIAS = 13,         // k_dealias_inv (main dealias kernel)
    TK_COMPUTE_MHD = 14,     // k_compute_mhd_forces
    TK_ASSEMBLE_TOTAL = 15,  // k_assemble_total_forces
    TK_PCONDITION_MAT = 16,  // ComputePreconditioningMatrix (every 25 iters)
    TK_ASSEMBLE_RZ = 17,     // k_assemble_rz_preconditioner
    TK_BACKUP_PTS = 18,      // k_backup_pts_x (state backup at store cadence)
  };
  cudaEvent_t tk_start[kNumTimedKernels] = {};
  cudaEvent_t tk_stop[kNumTimedKernels] = {};
  double tk_total_ms[kNumTimedKernels] = {};
  long long tk_calls[kNumTimedKernels] = {};
  bool tk_initialized = false;
  int tk_env = -1;  // -1 unread, 0 disabled, 1 enabled

  // K-window sync elision (VMECPP_SYNC_ELIDE=K). When nonzero for the
  // current iteration, the per-iteration scalar D2H + stream-sync sites
  // (jacobian tau extrema, residual triples, plasma volume) launch their
  // reduction kernels as usual but skip the transfer and sync; host
  // callers receive the last boundary-synced values from the static
  // caches. Set per iteration by Vmec::Evolve via SetSyncElideIterCuda.
  int sync_elide_iter = 0;

  void TKInit() {
    if (tk_initialized) return;
    for (int i = 0; i < kNumTimedKernels; ++i) {
      cuda_check(cudaEventCreate(&tk_start[i]), "tk start event");
      cuda_check(cudaEventCreate(&tk_stop[i]), "tk stop event");
    }
    tk_initialized = true;
  }

  void TKBegin(int slot) {
    if (tk_env <= 0 || !tk_initialized) return;
    cudaEventRecord(tk_start[slot], stream);
  }
  void TKEnd(int slot) {
    if (tk_env <= 0 || !tk_initialized) return;
    cudaEventRecord(tk_stop[slot], stream);
    cudaEventSynchronize(tk_stop[slot]);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, tk_start[slot], tk_stop[slot]);
    tk_total_ms[slot] += ms;
    tk_calls[slot] += 1;
    // Periodic dump so partial data survives abnormal process termination
    // that skips the atexit handler. Path via VMECPP_KERNEL_TIMING_PATH
    // (defaults to /tmp/vmecpp_kernel_timing.log). Dump every 10000 TKEnd
    // calls.
    static long long dump_counter = 0;
    ++dump_counter;
    if ((dump_counter % 10000) == 0) {
      const char* path = std::getenv("VMECPP_KERNEL_TIMING_PATH");
      if (!path) path = "/tmp/vmecpp_kernel_timing.log";
      FILE* f = std::fopen(path, "w");
      if (f) { TKDump(f); std::fclose(f); }
    }
  }
  void TKDump(FILE* f) {
    static const char* names[kNumTimedKernels] = {
      "cufftExecZ2D (inverse)", "k_scatter_main_and_con_v4",
      "k_jacobian_metric_dvdsh_atomic", "k_bcontra_bsupuv",
      "pressureAndEnergies (3 kernels)", "k_radial_interior",
      "cufftExecD2Z (forward)", "k_decompose_into",
      "k_residuals", "k_apply_m1_preconditioner",
      "k_apply_rz_pcr", "k_apply_lambda_preconditioner",
      "k_effective_constraint_force", "k_dealias_inv",
      "k_compute_mhd_forces", "k_assemble_total_forces",
      "ComputePreconditioningMatrix", "k_assemble_rz_preconditioner",
      "k_backup_pts_x",
    };
    double total = 0.0;
    for (int i = 0; i < kNumTimedKernels; ++i) total += tk_total_ms[i];
    std::fprintf(f, "===== VMECPP per-kernel timing =====\n");
    for (int i = 0; i < kNumTimedKernels; ++i) {
      double pct = (total > 0) ? (tk_total_ms[i] / total) * 100.0 : 0.0;
      std::fprintf(f,
          "  %-40s  total=%8.3fs  calls=%lld  avg=%.4fms  pct=%5.2f%%\n",
          names[i], tk_total_ms[i] / 1000.0,
          (long long)tk_calls[i],
          (tk_calls[i] > 0) ? tk_total_ms[i] / tk_calls[i] : 0.0,
          pct);
    }
    std::fprintf(f, "  ---\n  cumulative=%.3fs (per-call sync overhead included)\n",
                 total / 1000.0);
  }

  // Forward-FFT CUDA graph state.
  //
  // The forward graph captures the chain consisting of k_fill_spectra,
  // the toroidal Fourier transform (either cufftExecZ2D or the
  // hand-coded radix-8x3 inverse DFT), k_scatter_main_and_con, the
  // geometric scalar extraction, and the device-to-host transfers
  // that follow. Capture occurs once per shape and replay is used
  // thereafter; the captured graph removes the per-iteration kernel
  // launch overhead of approximately five separate launches.
  cudaGraph_t fwd_graph = nullptr;
  cudaGraphExec_t fwd_graph_exec = nullptr;
  bool fwd_graph_captured = false;
  // Whole-iteration graph (VMECPP_ITER_GRAPH=1 under sync elision): one
  // cudaGraphLaunch replays the complete elided iteration body. Captured
  // after kIterGraphWarmups eligible iterations; invalidated on Reshape,
  // on restarts, and when the segment-4 graph re-captures on a jMax
  // change.
  cudaGraph_t iter_graph = nullptr;
  cudaGraphExec_t iter_graph_exec = nullptr;
  bool iter_graph_captured = false;
  int iter_graph_warmups = 0;
  static constexpr int kIterGraphWarmups = 2;

  // Segment-3 graph state.
  //
  // The segment-3 graph captures the chain from
  // effectiveConstraintForceCuda through DecomposeAndConstrainCuda,
  // comprising approximately six CUDA wrappers and on the order of
  // ten to fifteen launches in total. The chain runs entirely on a
  // single stream with no host scalar reads or stream
  // synchronizations between the boundary points, so the capture
  // window is well defined. The graph is captured on the first call
  // following a Reshape and replayed on subsequent invocations until
  // a Reshape reissues the underlying allocations and invalidates
  // the captured pointers.
  cudaGraph_t seg3_graph = nullptr;
  bool seg3_vacuum_edge_at_capture = false;
  cudaGraphExec_t seg3_graph_exec = nullptr;
  bool seg3_graph_captured = false;
  bool seg3_in_capture = false;  // true while between BeginSeg3 and EndSeg3 in capture mode
  int seg3_warmup_calls = 0;     // capture begins only after the first warmup invocation completes lazy allocations

  // Segment-4 graph state.
  //
  // The segment-4 graph captures the preconditioner chain consisting
  // of ApplyM1PreconditionerCuda, AssembleRZPreconditionerCuda,
  // ApplyRZPreconditionerCuda, and ApplyLambdaPreconditionerCuda.
  // Each of these wrappers is kernel-only and contains no host
  // synchronization or host reads, and the chain runs between the
  // two ResidualsCuda invocations within each iteration. The single
  // kernel argument that varies across iterations is jMax, which
  // depends on lfreeb together with the vacuum-pressure-state
  // transitions; the last-captured value of jMax is retained, and a
  // change in that value triggers re-capture. Reshape invalidates
  // the captured pointers and resets the capture state.
  cudaGraph_t seg4_graph = nullptr;
  cudaGraphExec_t seg4_graph_exec = nullptr;
  bool seg4_graph_captured = false;
  bool seg4_in_capture = false;
  int seg4_warmup_calls = 0;
  int seg4_last_jMax = -1;       // re-capture if jMax changes

  // Segment-2 graph state.
  //
  // The segment-2 graph captures the six kernel-only wrappers that
  // run between ComputeJacobianCuda's stream synchronization and the
  // preconditioner-update block, namely
  // ComputeMetricElementsCuda, UpdateDifferentialVolumeCuda,
  // ComputeBContraCuda, ComputeBCoCuda, PressureAndEnergiesCuda, and
  // RadialForceBalanceCuda. No host synchronization or host read
  // occurs in this window. Reshape invalidates the captured pointers
  // and resets the capture state.
  cudaGraph_t seg2_graph = nullptr;
  cudaGraphExec_t seg2_graph_exec = nullptr;
  bool seg2_graph_captured = false;
  bool seg2_in_capture = false;
  int seg2_warmup_calls = 0;

  // Single contiguous buffer for the 6 spec arrays + xmpq + sqrtSF.
  double* d_specs_block = nullptr;
  // Pointers into d_specs_block:
  double* d_rmncc = nullptr;
  double* d_rmnss = nullptr;
  double* d_zmnsc = nullptr;
  double* d_zmncs = nullptr;
  double* d_lmnsc = nullptr;
  double* d_lmncs = nullptr;
  double* d_xmpq = nullptr;
  double* d_sqrtSF = nullptr;

  // Pinned host staging buffer mirroring d_specs_block.
  double* h_specs_pinned = nullptr;
  size_t specs_block_bytes = 0;

  // Basis arrays (constant per Reshape).
  double* d_nscale = nullptr;
  double* d_cosmu = nullptr;
  double* d_sinmu = nullptr;
  double* d_cosmum = nullptr;
  double* d_sinmum = nullptr;
  // Integration-weighted basis variants for the inverse FFT.
  double* d_cosmui = nullptr;
  double* d_sinmui = nullptr;
  double* d_cosmumi = nullptr;
  double* d_sinmumi = nullptr;

  // Toroidal discrete Fourier transform basis tables used by the
  // fused-single-pass forward-FFT kernels. Each entry stores the
  // nscale-folded cosine or sine evaluated at the corresponding
  // (n, k) lattice point,
  //   d_dft_cos[n * nZeta + k] = nscale[n] * cos(2 pi n k / nZeta)
  //   d_dft_sin[n * nZeta + k] = nscale[n] * sin(2 pi n k / nZeta).
  // Each table holds (ntor + 1) * nZeta doubles. When a fused-pass
  // forward-FFT variant is selected, these tables provide the
  // toroidal transform basis in place of a separate cuFFT batched
  // inverse call.
  double* d_dft_cos = nullptr;
  double* d_dft_sin = nullptr;
  int dft_basis_ntor_cached = -1;
  int dft_basis_nZeta_cached = -1;

  // Raw cosine and sine tables for the direct length-24 inverse
  // discrete Fourier transform. The entries are
  //   d_idft_cos[n * nZeta + k] = cos(2 pi n k / nZeta)
  //   d_idft_sin[n * nZeta + k] = sin(2 pi n k / nZeta),
  // each table holding nhalf * nZeta doubles. The k_inverse_dft_24
  // kernel reads from these tables when it replaces cufftExecZ2D in
  // the forward-FFT chain.
  double* d_idft_cos = nullptr;
  double* d_idft_sin = nullptr;
  int idft_basis_nhalf_cached = -1;
  int idft_basis_nZeta_cached = -1;

  // Inverse FFT: R2C cuFFT plan + Y reused as input (real, length nZeta), X
  // reused as output (complex, length nhalf).
  cufftHandle cufft_plan_r2c = 0;

  // FourierForces spec array device shadows (output of inverse FFT).
  double* d_frcc = nullptr;
  double* d_frss = nullptr;
  double* d_fzsc = nullptr;
  double* d_fzcs = nullptr;
  double* d_flsc = nullptr;
  double* d_flcs = nullptr;

  // FFT scratch.
  cufftDoubleComplex* d_X = nullptr;
  double* d_Y = nullptr;

  // Mixed-precision FFT scratch buffers. The complex input d_X_fp32
  // and the real output d_Y_fp32 hold single-precision copies of d_X
  // and d_Y respectively, and the cuFFT plan cufft_plan_c2r_fp32 maps
  // between them. The single-precision path exploits the substantially
  // higher single-precision floating-point throughput of the target
  // architecture relative to double precision, at the cost of
  // reduced numerical fidelity. Selection is governed by the
  // VMECPP_FFT_FP32 environment variable.
  cufftComplex* d_X_fp32 = nullptr;
  float* d_Y_fp32 = nullptr;
  cufftHandle cufft_plan_c2r_fp32 = 0;
  size_t fft_x_elems = 0;  // element count of the complex buffer d_X / d_X_fp32
  size_t fft_y_elems = 0;  // element count of the real buffer d_Y / d_Y_fp32

  // Single contiguous buffer for the 18 output arrays. Each output is a
  // contiguous slice of (ns_local * nZeta * nThetaEff) (16 main) or
  // (ns_con_local * nZeta * nThetaEff) (2 con).
  double* d_outputs_block = nullptr;
  double* h_outputs_pinned = nullptr;
  size_t outputs_block_bytes = 0;
  // Six-double extract for the geometric-scalar consumers
  // SetRadialExtent and SetGeometricOffset, which are the only host
  // sites that read r1_e, r1_o, and z1_e under VMECPP_USE_CUDA. The
  // extract is staged on the device in d_geom_scalars and copied to a
  // pinned host buffer h_geom_scalars; the larger device-to-host
  // transfer of d_outputs_block together with the host-side scatter
  // that would otherwise follow each forward FFT is unnecessary, since
  // the downstream output phase reads the device buffers directly via
  // FlushForOutputQuantitiesCuda at end-of-run.
  double* d_geom_scalars = nullptr;  // six doubles resident on the device
  double* h_geom_scalars = nullptr;  // six doubles in pinned host memory
  // Deferred-commit state for the geometric-scalar host writes. The
  // SetRadialExtent and SetGeometricOffset writes are deferred until
  // after the next natural stream synchronization, namely the
  // tau-minmax synchronization that ComputeJacobianCuda performs.
  // The flag fwd_geom_pending records that a deferred write is
  // outstanding, and the index fields hold the outputs-block offsets
  // that FlushFwdGeomScalarsToHost will commit. The commit call is
  // valid only after a stream synchronization has occurred.
  bool fwd_geom_pending = false;
  int fwd_geom_outer_idx = -1;
  int fwd_geom_inner_idx = -1;
  // Pointers into d_outputs_block:
  double* d_r1_e = nullptr; double* d_r1_o = nullptr;
  double* d_ru_e = nullptr; double* d_ru_o = nullptr;
  double* d_rv_e = nullptr; double* d_rv_o = nullptr;
  double* d_z1_e = nullptr; double* d_z1_o = nullptr;
  double* d_zu_e = nullptr; double* d_zu_o = nullptr;
  double* d_zv_e = nullptr; double* d_zv_o = nullptr;
  double* d_lu_e = nullptr; double* d_lu_o = nullptr;
  double* d_lv_e = nullptr; double* d_lv_o = nullptr;
  double* d_rCon = nullptr; double* d_zCon = nullptr;
  // Sizes (bytes) of each sub-block in outputs.
  size_t main_array_bytes = 0;  // ns_local * nZeta * nThetaEff * sizeof(double)
  size_t con_array_bytes = 0;   // ns_con_local * nZeta * nThetaEff * sizeof(double)

  // Persistent jacobian buffers (half-grid). Allocated lazily.
  double* d_r12 = nullptr;
  double* d_ru12 = nullptr;
  double* d_zu12 = nullptr;
  double* d_rs = nullptr;
  double* d_zs = nullptr;
  double* d_tau = nullptr;
  double* d_sqrtSH = nullptr;
  // Staging optimization: track whether d_sqrtSH has been staged since the last
  // Reshape. sqrtSH is invariant for a given ns (radial grid), so the per-iter
  // H2Ds in ComputeJacobianCuda + ComputeMetricElementsCuda are redundant.
  // Reset in EnsureJacobianBuffers when ns_h changes.
  bool sqrtSH_staged = false;

  // When ComputeJacobianCuda dispatches the fused jacobian-and-metric
  // kernel, the metric outputs gsqrt, guu, guv, and gvv are produced
  // alongside the jacobian outputs. This per-iteration flag is raised
  // by the jacobian wrapper to inform ComputeMetricElementsCuda that
  // its kernel launch may be elided. The flag is cleared inside
  // ComputeMetricElementsCuda after the elision has taken effect, so
  // it does not persist across iterations.
  bool jac_metric_fused_this_iter = false;

  // Persistent-kernel direction: when the 3-way jacobian+metric+dvdsh fused
  // kernel runs, dvdsh outputs are also done. This per-iter flag tells
  // UpdateDifferentialVolumeCuda to skip its kernel launch (work already done).
  // Reset to false at end of UpdateDifferentialVolumeCuda (consumed).
  bool dvdsh_fused_this_iter = false;
  // Same caching pattern for the other per-iter-invariant radial profiles:
  // massH/currH/phipF/phipH/radialBlending all depend only on radial grid +
  // input parameters (fixed for a given multigrid level). Flags reset in
  // Reshape (which is also where ns_h_cached etc. change).
  bool massH_staged = false;
  bool currH_staged = false;
  bool phipF_staged = false;
  bool phipH_staged = false;
  bool radialBlending_staged = false;
  // pm_sm/pm_sp are radial scaling factors from m_p_.sm / m_p_.sp; invariant
  // per Reshape. Cache so the 2 H2Ds (R-side and Z-side calls) become 1 once.
  bool pm_sm_staged = false;
  bool pm_sp_staged = false;
  // scalxc is the radial scaling for FourierCoeffs decomposeInto. Function of
  // radial grid; invariant per Reshape.
  bool scalxc_staged = false;
  // dealias faccon is `-0.25 * signOfJacobian / xmpq[m]^2`, set once in
  // IdealMhdModel ctor and never changed. The per-iter H2D in
  // DeAliasConstraintForceCuda is redundant.
  bool dealias_faccon_staged = false;
  // BContra iotaH seeding/input: d_iotaH (ncurr=1) and d_iotaH_in (ncurr=0)
  // only need to be seeded once. For ncurr=1 the device value is updated by
  // k_bcontra_chipH_iotaH each iter; the host m_p_.iotaH is a stale D2H copy
  // that we'd re-upload, contributing nothing. For ncurr=0 iotaH_in is a
  // prescribed profile that doesn't change.
  bool iotaH_seeded = false;
  double* h_jac_pinned = nullptr;  // staging for D2H of 6 jacobian outputs
  int ns_h_cached = -1;
  int nZnT_cached = -1;
  size_t jac_array_bytes = 0;

  // Metric-element buffers (half-grid). 4 outputs.
  double* d_gsqrt = nullptr;
  double* d_guu = nullptr;
  double* d_guv = nullptr;
  double* d_gvv = nullptr;
  double* h_metric_pinned = nullptr;

  // dVdsH integration weights + scalar output.
  double* d_wInt = nullptr;     // size nThetaEff, staged once at Reshape
  double* d_dVdsH = nullptr;    // size ns_h
  int nThetaEff_for_wInt = -1;

  // computeBCo outputs (size ns_h * nZnT).
  double* d_bsubu = nullptr;
  double* d_bsubv = nullptr;

  // radialForceBalance outputs (half-grid ns_h: bucoH, bvcoH; interior ns_fi:
  // jcurvF, jcuruF, presgradF, dVdsF, equiF). Plus inputs presH, chipF, phipF
  // copied per-call.
  double* d_bucoH = nullptr;
  double* d_bvcoH = nullptr;
  double* d_jcurvF = nullptr;
  double* d_jcuruF = nullptr;
  double* d_presgradF = nullptr;
  double* d_dVdsF = nullptr;
  double* d_equiF = nullptr;
  double* d_presH = nullptr;
  double* d_chipF = nullptr;
  double* d_phipF = nullptr;

  // rzConIntoVolume outputs (full-grid con range, ns_con_local × nZnT).
  double* d_rCon0 = nullptr;
  double* d_zCon0 = nullptr;
  int rzcon0_ns_con_cached = -1;
  int rzcon0_nZnT_cached = -1;

  // computeBContra: bsupu, bsupv (half-grid, persistent), chip/iota profiles,
  // plus per-call H2D inputs (phipF, phipH, currH, iotaH_in) and ncurr=1
  // reduction scratch (jvPlasma, avg_guu_gsqrt).
  double* d_bsupu = nullptr;
  double* d_bsupv = nullptr;
  double* d_chipH = nullptr;
  double* d_iotaH = nullptr;
  double* d_iotaF = nullptr;
  double* d_phipH = nullptr;
  double* d_currH = nullptr;
  double* d_iotaH_in = nullptr;   // input iotaH (when ncurr==0)
  double* d_jvPlasma = nullptr;
  double* d_avg_guu_gsqrt = nullptr;

  // pressureAndEnergies buffers.
  double* d_massH = nullptr;
  double* d_totalPressure = nullptr;     // ns_h * nZnT
  double* d_thermal_partial = nullptr;   // ns_h
  double* d_magnetic_partial = nullptr;  // ns_h

  // hybridLambdaForce buffers.
  double* d_radialBlending = nullptr;    // ns_local
  double* d_blmn_e = nullptr;            // ns_con_local * nZnT
  double* d_blmn_o = nullptr;
  double* d_clmn_e = nullptr;
  double* d_clmn_o = nullptr;

  // computeForceNorms reductions.
  double* d_forceNormRZ_partial = nullptr;  // ns_h
  double* d_forceNormL_partial = nullptr;   // ns_h

  // updateLambdaPreconditioner buffers.
  // bLambda/dLambda/cLambda: size ns_h + 1 (offset-1 indexing to mirror CPU).
  // lambdaPreconditioner: size ns_con_local * mpol * (ntor+1).
  double* d_bLambda = nullptr;
  double* d_dLambda = nullptr;
  double* d_cLambda = nullptr;
  double* d_lambdaPreconditioner = nullptr;

  // computePreconditioningMatrix scratch.
  // ax: ns_h × 4, bx: ns_h × 3, cx: ns_h.
  double* d_ax_scratch = nullptr;
  double* d_bx_scratch = nullptr;
  double* d_cx_scratch = nullptr;
  // Outputs are per-call (xs/xu12/xu_e/xu_o/x1_o inputs, axm/axd/bxm/bxd/cxd
  // outputs); allocated lazily by EnsurePrecondMatrixBuffers.
  double* d_pm_xs = nullptr;
  double* d_pm_xu12 = nullptr;
  double* d_pm_xu_e = nullptr;
  double* d_pm_xu_o = nullptr;
  double* d_pm_x1_o = nullptr;
  double* d_pm_sm = nullptr;
  double* d_pm_sp = nullptr;
  double* d_pm_axm = nullptr;  // ns_h * 2
  double* d_pm_axd = nullptr;  // ns_force_local * 2
  double* d_pm_bxm = nullptr;  // ns_h * 2
  double* d_pm_bxd = nullptr;  // ns_force_local * 2
  double* d_pm_cxd = nullptr;  // ns_force_local

  // Persistent per-side snapshots of the preconditioner-matrix
  // coefficients produced by computePreconditioningMatrix. The shared
  // scratch buffers d_pm_axm, d_pm_axd, d_pm_bxm, d_pm_bxd, and
  // d_pm_cxd are overwritten on the second (Z-side) invocation of
  // ComputePreconditioningMatrixCuda, so dedicated R-side and Z-side
  // destinations are required if AssembleRZPreconditionerCuda is to
  // read both halves. The snapshots are populated through a device-
  // to-device memcpy issued at the end of each
  // ComputePreconditioningMatrixCuda call, supplanting the host-side
  // m_axm and related arrays that the CPU implementation of
  // assembleRZPreconditioner consumes.
  double* d_pmat_arm = nullptr;  // R-side half-grid coefficient: ns_h * 2
  double* d_pmat_brm = nullptr;  // R-side half-grid coefficient: ns_h * 2
  double* d_pmat_ard = nullptr;  // R-side full-grid coefficient: ns_force_local * 2
  double* d_pmat_brd = nullptr;  // R-side full-grid coefficient: ns_force_local * 2
  double* d_pmat_azm = nullptr;  // Z-side half-grid coefficient: ns_h * 2
  double* d_pmat_bzm = nullptr;  // Z-side half-grid coefficient: ns_h * 2
  double* d_pmat_azd = nullptr;  // Z-side full-grid coefficient: ns_force_local * 2
  double* d_pmat_bzd = nullptr;  // Z-side full-grid coefficient: ns_force_local * 2
  double* d_pmat_cxd = nullptr;  // shared full-grid coefficient (identical R/Z): ns_force_local
  // Sizes cached so ResetForNewVmecRun can memset without arg threading.
  int pmat_ns_h_cached = -1;
  int pmat_ns_force_local_cached = -1;

  // constraintForceMultiplier buffers (per-surface reductions + outputs).
  double* d_arNorm = nullptr;
  double* d_azNorm = nullptr;
  double* d_tcon = nullptr;
  double* d_ruFull = nullptr;
  double* d_zuFull = nullptr;
  // Free-boundary vacuum edge term. d_rbsq holds the host-computed rBSq
  // profile (nZnT doubles, single configuration); rbsq_staged marks it
  // current for the iteration's force assembly.
  double* d_rbsq = nullptr;
  int rbsq_size = 0;
  bool rbsq_staged = false;
  // Batched int8-Ozaki scatter state. The W limbs and column exponents
  // are shape-constant (built once after Reshape); the Y limbs and row
  // exponents refresh every iteration.
  double* d_i8b_W = nullptr;
  signed char* d_i8b_Wl = nullptr;
  int* d_i8b_eW = nullptr;
  signed char* d_i8b_Yl = nullptr;
  int* d_i8b_eY = nullptr;
  int i8b_B_pad = 0;
  bool i8b_w_built = false;
  double* d_ard = nullptr;  // per-call H2D
  double* d_azd = nullptr;  // per-call H2D

  // effectiveConstraintForce + assembleTotalForces helpers.
  double* d_gConEff = nullptr;
  double* d_gCon = nullptr;       // host H2D per call
  double* d_rCon_in = nullptr;    // d_rCon already exists; alias not needed
  double* d_frcon_e = nullptr;
  double* d_frcon_o = nullptr;
  double* d_fzcon_e = nullptr;
  double* d_fzcon_o = nullptr;

  // deAliasConstraintForce persistent buffers.
  double* d_dealias_gsc = nullptr;
  double* d_dealias_gcs = nullptr;
  double* d_dealias_faccon = nullptr;
  double* d_dealias_cosnv = nullptr;
  double* d_dealias_sinnv = nullptr;
  int dealias_nnyq2_plus_1_cached = -1;

  // Decomposed FourierForces shadow (m_decomposed_f mirror). Populated by
  // DecomposeAndConstrainCuda from S.d_frcc/etc. (physical) via decomposeInto
  // + m1Constraint + zeroZForceForM1. Read by M1/Lambda/RZ preconditioners.
  double* d_decomposed_frcc = nullptr;
  double* d_decomposed_frss = nullptr;
  double* d_decomposed_fzsc = nullptr;
  double* d_decomposed_fzcs = nullptr;
  double* d_decomposed_flsc = nullptr;
  double* d_decomposed_flcs = nullptr;
  double* d_scalxc = nullptr;        // host m_p_.scalxc staged per call
  int decomposed_size_cached = -1;

  // Velocity and decomposed-position state for the device-resident
  // conjugate-gradient time integrator implemented by
  // PerformTimeStepCuda. The velocity tensor d_pts_v_* is sized
  // ns_con_local * mpol * (ntor + 1) and the position tensor
  // d_pts_x_* is sized ns_local * mpol * (ntor + 1); each carries
  // six spectral coefficient fields under the three-dimensional
  // stellarator-symmetric workload (lthreed = true, lasym = false).
  // Both tensors are extended along the configuration axis to
  // n_config_max.
  //
  // The state is allocated lazily on the first call to
  // PerformTimeStepCuda. On that first call the host buffers are
  // copied to the device once: the host velocity m_decomposed_v
  // begins zeroed, and the host position m_decomposed_x carries the
  // initial boundary spectra. After each kernel invocation the
  // device position d_pts_x_* is copied back to the host
  // m_decomposed_x so that the host triplet of
  // decomposeInto, m1Constraint, and extrapolateTowardsAxis at the
  // start of the next iteration's update operates on the most
  // recent decomposed position.
  double* d_pts_v_rcc = nullptr;
  double* d_pts_v_rss = nullptr;
  double* d_pts_v_zsc = nullptr;
  double* d_pts_v_zcs = nullptr;
  double* d_pts_v_lsc = nullptr;
  double* d_pts_v_lcs = nullptr;
  double* d_pts_x_rcc = nullptr;
  double* d_pts_x_rss = nullptr;
  double* d_pts_x_zsc = nullptr;
  double* d_pts_x_zcs = nullptr;
  double* d_pts_x_lsc = nullptr;
  double* d_pts_x_lcs = nullptr;
  // Device twin of host physical_x_backup. RestartIteration periodically
  // saves d_pts_x → d_pts_x_backup on the NO_RESTART path and restores on
  // BAD_JACOBIAN/BAD_PROGRESS. Mirrors the host backup mechanism so the
  // device state participates in rollback; required when the per-iter D2H
  // of d_pts_x → host m_decomposed_x is removed.
  double* d_pts_x_backup_rcc = nullptr;
  double* d_pts_x_backup_rss = nullptr;
  double* d_pts_x_backup_zsc = nullptr;
  double* d_pts_x_backup_zcs = nullptr;
  double* d_pts_x_backup_lsc = nullptr;
  double* d_pts_x_backup_lcs = nullptr;
  bool pts_x_backup_initialized = false;
  // Per-cfg converged-state snapshots. When the iteration controller marks
  // a cfg inactive (converged or timed out), its d_pts_x slice is copied
  // here and the batch outputs dump prefers the snapshot. The live
  // d_pts_x slice of an inactive cfg continues to be modified by
  // mask-agnostic kernels while the rest of the batch iterates and is not
  // trustworthy at end of run.
  double* d_pts_x_final_rcc = nullptr;
  double* d_pts_x_final_rss = nullptr;
  double* d_pts_x_final_zsc = nullptr;
  double* d_pts_x_final_zcs = nullptr;
  double* d_pts_x_final_lsc = nullptr;
  double* d_pts_x_final_lcs = nullptr;
  std::vector<std::uint8_t> pts_x_final_taken;
  // rzNorm partials: per-jF doubles. Sized to ns_local since
  // (nsMaxFIncludingLcfs - nsMinF) <= ns_local. Pinned host counterpart so
  // D2H completes without an additional copy.
  double* d_rznorm_partials = nullptr;
  double* h_rznorm_partials = nullptr;
  int pts_v_size = -1;        // ns_con_local * mpol * (ntor+1) (per cfg)
  int pts_x_size = -1;        // ns_local * mpol * (ntor+1) (per cfg)
  int pts_x_ns = -1;          // ns_local (for the latest EnsurePTSBuffers call)
  bool pts_v_initialized = false;
  bool pts_x_initialized = false;
  // Multigrid-stage transition state. Captured BEFORE freeing d_pts_x in
  // EnsurePTSBuffers when the new ns_local differs from the old; the per-cfg
  // radial-interp kernel in PerformTimeStepCuda's init branch reads from
  // these and writes into the freshly allocated d_pts_x at the new ns_local.
  // This is the device-side analogue of the host m_decomposed_x upscale that
  // runs at each multigrid stage boundary in vmec.cc, but operating per cfg
  // so distinct-mode batched runs preserve per-cfg state across stages.
  double* d_pts_x_prev_rcc = nullptr;
  double* d_pts_x_prev_rss = nullptr;
  double* d_pts_x_prev_zsc = nullptr;
  double* d_pts_x_prev_zcs = nullptr;
  double* d_pts_x_prev_lsc = nullptr;
  double* d_pts_x_prev_lcs = nullptr;
  int pts_x_prev_ns = -1;
  int pts_x_prev_size = -1;
  int pts_x_prev_mpol = -1;
  int pts_x_prev_ntor = -1;
  bool pts_x_prev_valid = false;
  // d_scalxc snapshot at the old ns, captured at the same Reshape transition
  // point as d_pts_x_prev. The upscale kernel multiplies d_pts_x_prev by
  // this OLD scalxc to recover physical-space coefficients (matching the
  // host's old_xc_scaled_), interpolates radially, then divides by the
  // freshly-staged d_scalxc at the new ns. Sized n_config_max * old_ns * 2.
  double* d_scalxc_prev = nullptr;
  int scalxc_prev_len = -1;
  bool scalxc_prev_valid = false;
  // Per-cfg time-step controller state. timestep_first_call_after_reset
  // flips to true whenever EnsurePTSBuffers reallocates (a new multigrid
  // stage starts) or Reshape resets state. The k_update_timestep dispatch
  // uses this to pass iter_phase=0 (resets inv_tau ring) on that first
  // call, then iter_phase=1 on subsequent calls.
  bool timestep_first_call_after_reset = true;
  // Producer-consumer signal between RecomposeToPhysicalCuda and
  // CudaForward. RecomposeToPhysicalCuda raises this flag after it
  // has written d_specs_block on the device, having executed the
  // device-side decomposeInto, m1Constraint, and extrapolation
  // sequence on d_pts_x. CudaForward clears the flag after consuming
  // d_specs_block, and, while the flag is set, skips the
  // host-to-device transfer of the spectral block that would
  // otherwise occur at the start of each forward FFT call.
  bool specs_populated_from_device = false;

  // Persistent preconditioner-input buffers.
  double* d_m1_ard = nullptr;
  double* d_m1_brd = nullptr;
  double* d_m1_azd = nullptr;
  double* d_m1_bzd = nullptr;
  double* d_lambda_lp = nullptr;
  double* d_rz_aR = nullptr;
  double* d_rz_dR = nullptr;
  double* d_rz_bR = nullptr;
  double* d_rz_cR = nullptr;
  double* d_rz_aZ = nullptr;
  double* d_rz_dZ = nullptr;
  double* d_rz_bZ = nullptr;
  double* d_rz_cZ = nullptr;
  int* d_rz_jMin = nullptr;
  int rz_mnsize_cached = -1;
  int rz_ns_total_cached = -1;
  int rz_num_basis_cached = -1;

  // Carson-Higham staged FP32 iterative refinement scratch buffers.
  // d_rz_c_orig_R/Z hold a copy of the original FP64 right-hand side
  // captured immediately before the first FP32 PCR launch; the FP64
  // residual kernel reads them to form r = b - A*x without depending
  // on the FP32 solve being non-destructive. d_rz_x_saved_R/Z hold the
  // FP64 approximate solution returned by the first FP32 PCR pass so
  // the final correction kernel can compute x_refined = x_saved + dx.
  // All four buffers are sized identically to d_rz_cR/cZ and are
  // (re)allocated whenever EnsureRZBuffers reallocates the main c
  // buffers. They are only allocated and used when VMECPP_RZ_IR_FP32
  // is set; under default execution they remain nullptr.
  double* d_rz_c_orig_R = nullptr;
  double* d_rz_c_orig_Z = nullptr;
  double* d_rz_x_saved_R = nullptr;
  double* d_rz_x_saved_Z = nullptr;

  // computeMHDForces outputs (force grid, ns_force_local × nZnT each).
  double* d_armn_e = nullptr;
  double* d_armn_o = nullptr;
  double* d_azmn_e = nullptr;
  double* d_azmn_o = nullptr;
  double* d_brmn_e = nullptr;
  double* d_brmn_o = nullptr;
  double* d_bzmn_e = nullptr;
  double* d_bzmn_o = nullptr;
  double* d_crmn_e = nullptr;
  double* d_crmn_o = nullptr;
  double* d_czmn_e = nullptr;
  double* d_czmn_o = nullptr;

  // Scalar scratch buffer (one double on device for scalar reductions).
  double* d_scalar = nullptr;

  // PressureAndEnergies: 3 device scalars [thermal, magnetic, mhd]. Avoids
  // the host-side accumulation + sync; D2H'd async only.
  double* d_pressure_scalars = nullptr;

  // ComputeJacobian: 2 device scalars [tau_min, tau_max]. Replaces the full
  // tau D2H + host min/max scan.
  double* d_jac_minmax = nullptr;

  // Residuals: 3 device scalars [fResR, fResZ, fResL]. Read once at end of
  // ResidualsCuda via small D2H.
  double* d_residuals_partial = nullptr;
  // Multi-block partials buffer for k_residuals_par_K: K * n_config_max * 3
  // doubles. K sub-blocks per cfg each write one triple; finalize kernel
  // reduces across the K axis into d_residuals_partial. Sized in
  // EnsureResidualsBuffer with K_partitions = kResidualsKPartitions.
  double* d_residuals_partials_K = nullptr;
  static constexpr int kResidualsKPartitions = 16;

  // Time-step controller on device.
  // Layout (per-cfg, sized in EnsureTimestepBuffers):
  //   d_inv_tau     : [n_config_max * kNDamp] doubles; ring buffer of 1/tau
  //                   samples. Each iter shifts left by 1 and writes the new
  //                   sample at position kNDamp - 1.
  //   d_prev_fsq    : [n_config_max] doubles; fc.fsq from previous iter
  //                   (initialized to 1.0 on first call per cfg).
  //   d_fac_b1      : [n_config_max * 2] doubles; laid out as
  //                   [cfg0.fac, cfg0.b1, cfg1.fac, cfg1.b1, ...].
  //                   k_perform_time_step_devfac reads this in place of the
  //                   scalar velocity_scale / conjugation_parameter args.
  // Driven by the per-cfg time-step controller (VMECPP_BATCH_PER_CFG_TIMESTEP)
  // and by sync-elided iterations, where the device controller is
  // authoritative. The host fc_.invTau_ stays the source of truth when
  // neither is active, so the existing convergence/restart logic in
  // vmec.cc is unaffected.
  double* d_inv_tau = nullptr;
  double* d_prev_fsq = nullptr;
  double* d_fac_b1 = nullptr;
  // d_fnorm1: [n_config_max] doubles; the evalFResPrecd force-norm factor
  // consumed by k_update_timestep. Staged by StageFnorm1 whenever the host
  // value changes (preconditioner boundaries), so the kernel carries no
  // baked host-scalar argument and the launch is stream-capturable across
  // boundaries. One slot per cfg, broadcast today; ready for per-cfg
  // force norms.
  double* d_fnorm1 = nullptr;
  double fnorm1_staged = 0.0;
  bool fnorm1_staged_valid = false;
  // Set once k_rz_norm_per_cfg starts filling d_fnorm1 with per-cfg values
  // at the force-norm cadence; StageFnorm1's host broadcast then stands
  // down so it cannot overwrite the per-cfg values.
  bool fnorm1_device_filled = false;
  // kNDamp matches FlowControl::kNDamp on host (10 by convention; if the
  // host constant changes, this needs to match).
  static constexpr int kTimestepNDamp = 10;

  // Batched-input staging cache, one-shot per run. State members rather
  // than function-local statics so ResetForNewVmecRun rearms the staging
  // for the next run in the same process.
  double* batch_inputs_pinned = nullptr;
  int batch_inputs_n_cfg = 0;
  size_t batch_inputs_one_spec_doubles = 0;
  int batch_inputs_loaded = -1;  // -1 unread, 0 absent, 1 loaded
  bool batch_inputs_consumed = false;

  // ComputeForceNorms: 2 device scalars [sum_rz, sum_l] after per-jH
  // reduction. Replaces the ns_h-D2H + host accumulator.
  double* d_fnorm_scalars = nullptr;

  // RZ-preconditioner transpose cache. The host hAR/hDR/hBR/hAZ/hDZ/hBZ
  // transpose work + 6 H2Ds only need to happen when the preconditioner
  // updates (every kPreconditionerUpdateInterval iters). We sentinel on
  // ar[0]: when it matches the cached value, d_rz_aR/dR/bR/aZ/dZ/bZ are
  // already up to date and we skip the transpose + H2D entirely.
  double rz_cache_ar_sentinel = std::numeric_limits<double>::quiet_NaN();

  // Raw byte estimate of the persistent allocation made by the most
  // recent Reshape, credited by the admission pre-flight as memory the
  // next Reshape would free.
  long long reshape_budget_raw_bytes = 0;

  std::mutex mu;

  // Reallocate device buffers for a new shape. Frees previous buffers.
  // Batched layout: optional n_config_max parameter (default 1) sizes
  // per-config buffers for N concurrent equilibria. At n_config_max=1
  // the layout and behavior are identical to single-call.
  void Reshape(int ns_local, int ns_con_local, int mpol, int ntor, int nhalf,
               int nZeta, int nThetaReduced, int nThetaEff,
               int n_config_max_in = 1) {
    n_config_max = n_config_max_in;
    // Invariant-staging caches reset on every Reshape (which is the only
    // time the underlying radial grid / problem size changes).
    sqrtSH_staged = false;
    massH_staged = false;
    currH_staged = false;
    phipF_staged = false;
    phipH_staged = false;
    radialBlending_staged = false;
    pm_sm_staged = false;
    pm_sp_staged = false;
    scalxc_staged = false;
    dealias_faccon_staged = false;
    iotaH_seeded = false;
    auto cuda_free_if = [](void*& p) {
      if (p) { cudaFree(p); p = nullptr; }
    };
    auto pinned_free_if = [](void*& p) {
      if (p) { cudaFreeHost(p); p = nullptr; }
    };
    cuda_free_if((void*&)d_specs_block);
    cuda_free_if((void*&)d_X);     cuda_free_if((void*&)d_Y);
    cuda_free_if((void*&)d_X_fp32); cuda_free_if((void*&)d_Y_fp32);
    if (cufft_plan_c2r_fp32) { cufftDestroy(cufft_plan_c2r_fp32); cufft_plan_c2r_fp32 = 0; }
    // The GEMM-scatter scratch is sized by ns_local and the cuBLAS
    // handle is bound to the stream this Reshape destroys below; free
    // and reset both so the next dispatch rebuilds them at the new
    // shape on the new stream.
    cuda_free_if((void*&)d_scatter_basis_fp32);
    cuda_free_if((void*&)d_scatter_Y_fp32);
    cuda_free_if((void*&)d_scatter_out_fp32);
    cuda_free_if((void*&)d_scatter_basis_hi);
    cuda_free_if((void*&)d_scatter_basis_lo);
    cuda_free_if((void*&)d_scatter_Y_hi);
    cuda_free_if((void*&)d_scatter_Y_lo);
    cuda_free_if((void*&)d_scatter_out_hh);
    cuda_free_if((void*&)d_scatter_out_hl);
    cuda_free_if((void*&)d_scatter_out_lh);
    cuda_free_if((void*&)d_scatter_out_ll);
    scatter_basis_M = 0;
    scatter_basis_N = 0;
#ifndef VMECPP_USE_HIP
    if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
#endif  // VMECPP_USE_HIP
    cuda_free_if((void*&)d_outputs_block);
    cuda_free_if((void*&)d_geom_scalars);
    cuda_free_if((void*&)d_nscale);
    cuda_free_if((void*&)d_cosmu);
    cuda_free_if((void*&)d_sinmu);
    cuda_free_if((void*&)d_cosmum);
    cuda_free_if((void*&)d_sinmum);
    cuda_free_if((void*&)d_cosmui);
    cuda_free_if((void*&)d_sinmui);
    cuda_free_if((void*&)d_cosmumi);
    cuda_free_if((void*&)d_sinmumi);
    cuda_free_if((void*&)d_frcc);
    cuda_free_if((void*&)d_frss);
    cuda_free_if((void*&)d_fzsc);
    cuda_free_if((void*&)d_fzcs);
    cuda_free_if((void*&)d_flsc);
    cuda_free_if((void*&)d_flcs);
    if (cufft_plan_r2c) { cufftDestroy(cufft_plan_r2c); cufft_plan_r2c = 0; }
    pinned_free_if((void*&)h_specs_pinned);
    pinned_free_if((void*&)h_outputs_pinned);
    pinned_free_if((void*&)h_geom_scalars);
    // Jacobian buffers too.
    cuda_free_if((void*&)d_r12);
    cuda_free_if((void*&)d_ru12);
    cuda_free_if((void*&)d_zu12);
    cuda_free_if((void*&)d_rs);
    cuda_free_if((void*&)d_zs);
    cuda_free_if((void*&)d_tau);
    cuda_free_if((void*&)d_sqrtSH);
    pinned_free_if((void*&)h_jac_pinned);
    // Metric-element buffers too.
    cuda_free_if((void*&)d_gsqrt);
    cuda_free_if((void*&)d_guu);
    cuda_free_if((void*&)d_guv);
    cuda_free_if((void*&)d_gvv);
    pinned_free_if((void*&)h_metric_pinned);
    cuda_free_if((void*&)d_wInt);
    cuda_free_if((void*&)d_dVdsH);
    cuda_free_if((void*&)d_bsubu);
    cuda_free_if((void*&)d_bsubv);
    cuda_free_if((void*&)d_bucoH);
    cuda_free_if((void*&)d_bvcoH);
    cuda_free_if((void*&)d_jcurvF);
    cuda_free_if((void*&)d_jcuruF);
    cuda_free_if((void*&)d_presgradF);
    cuda_free_if((void*&)d_dVdsF);
    cuda_free_if((void*&)d_equiF);
    cuda_free_if((void*&)d_presH);
    cuda_free_if((void*&)d_chipF);
    cuda_free_if((void*&)d_phipF);
    cuda_free_if((void*&)d_rCon0);
    cuda_free_if((void*&)d_zCon0);
    rzcon0_ns_con_cached = -1;
    rzcon0_nZnT_cached = -1;
    cuda_free_if((void*&)d_rbsq);
    rbsq_staged = false;
    cuda_free_if((void*&)d_i8b_W);
    cuda_free_if((void*&)d_i8b_Wl);
    cuda_free_if((void*&)d_i8b_eW);
    cuda_free_if((void*&)d_i8b_Yl);
    cuda_free_if((void*&)d_i8b_eY);
    i8b_B_pad = 0;
    i8b_w_built = false;
    cuda_free_if((void*&)d_bsupu);
    cuda_free_if((void*&)d_bsupv);
    cuda_free_if((void*&)d_chipH);
    cuda_free_if((void*&)d_iotaH);
    cuda_free_if((void*&)d_iotaF);
    cuda_free_if((void*&)d_phipH);
    cuda_free_if((void*&)d_currH);
    cuda_free_if((void*&)d_iotaH_in);
    cuda_free_if((void*&)d_jvPlasma);
    cuda_free_if((void*&)d_avg_guu_gsqrt);
    cuda_free_if((void*&)d_massH);
    cuda_free_if((void*&)d_totalPressure);
    cuda_free_if((void*&)d_thermal_partial);
    cuda_free_if((void*&)d_magnetic_partial);
    cuda_free_if((void*&)d_radialBlending);
    cuda_free_if((void*&)d_blmn_e);
    cuda_free_if((void*&)d_blmn_o);
    cuda_free_if((void*&)d_clmn_e);
    cuda_free_if((void*&)d_clmn_o);
    cuda_free_if((void*&)d_forceNormRZ_partial);
    cuda_free_if((void*&)d_forceNormL_partial);
    cuda_free_if((void*&)d_armn_e);
    cuda_free_if((void*&)d_armn_o);
    cuda_free_if((void*&)d_azmn_e);
    cuda_free_if((void*&)d_azmn_o);
    cuda_free_if((void*&)d_brmn_e);
    cuda_free_if((void*&)d_brmn_o);
    cuda_free_if((void*&)d_bzmn_e);
    cuda_free_if((void*&)d_bzmn_o);
    cuda_free_if((void*&)d_crmn_e);
    cuda_free_if((void*&)d_crmn_o);
    cuda_free_if((void*&)d_czmn_e);
    cuda_free_if((void*&)d_czmn_o);
    cuda_free_if((void*&)d_bLambda);
    cuda_free_if((void*&)d_dLambda);
    cuda_free_if((void*&)d_cLambda);
    cuda_free_if((void*&)d_lambdaPreconditioner);
    cuda_free_if((void*&)d_ax_scratch);
    cuda_free_if((void*&)d_bx_scratch);
    cuda_free_if((void*&)d_cx_scratch);
    cuda_free_if((void*&)d_pm_xs);
    cuda_free_if((void*&)d_pm_xu12);
    cuda_free_if((void*&)d_pm_xu_e);
    cuda_free_if((void*&)d_pm_xu_o);
    cuda_free_if((void*&)d_pm_x1_o);
    cuda_free_if((void*&)d_pm_sm);
    cuda_free_if((void*&)d_pm_sp);
    cuda_free_if((void*&)d_pm_axm);
    cuda_free_if((void*&)d_pm_axd);
    cuda_free_if((void*&)d_pm_bxm);
    cuda_free_if((void*&)d_pm_bxd);
    cuda_free_if((void*&)d_pm_cxd);
    cuda_free_if((void*&)d_pmat_arm);
    cuda_free_if((void*&)d_pmat_brm);
    cuda_free_if((void*&)d_pmat_ard);
    cuda_free_if((void*&)d_pmat_brd);
    cuda_free_if((void*&)d_pmat_azm);
    cuda_free_if((void*&)d_pmat_bzm);
    cuda_free_if((void*&)d_pmat_azd);
    cuda_free_if((void*&)d_pmat_bzd);
    cuda_free_if((void*&)d_pmat_cxd);
    cuda_free_if((void*&)d_arNorm);
    cuda_free_if((void*&)d_azNorm);
    cuda_free_if((void*&)d_tcon);
    cuda_free_if((void*&)d_ruFull);
    cuda_free_if((void*&)d_zuFull);
    cuda_free_if((void*&)d_ard);
    cuda_free_if((void*&)d_azd);
    cuda_free_if((void*&)d_gConEff);
    cuda_free_if((void*&)d_gCon);
    cuda_free_if((void*&)d_frcon_e);
    cuda_free_if((void*&)d_frcon_o);
    cuda_free_if((void*&)d_fzcon_e);
    cuda_free_if((void*&)d_fzcon_o);
    cuda_free_if((void*&)d_dealias_gsc);
    cuda_free_if((void*&)d_dealias_gcs);
    cuda_free_if((void*&)d_dealias_faccon);
    cuda_free_if((void*&)d_dealias_cosnv);
    cuda_free_if((void*&)d_dealias_sinnv);
    dealias_nnyq2_plus_1_cached = -1;
    cuda_free_if((void*&)d_decomposed_frcc);
    cuda_free_if((void*&)d_decomposed_frss);
    cuda_free_if((void*&)d_decomposed_fzsc);
    cuda_free_if((void*&)d_decomposed_fzcs);
    cuda_free_if((void*&)d_decomposed_flsc);
    cuda_free_if((void*&)d_decomposed_flcs);
    // Snapshot d_scalxc into d_scalxc_prev BEFORE freeing it. Pairs with the
    // d_pts_x_prev snapshot below so the upscale kernel can interpolate in
    // physical space (decomposed * scalxc_OLD) matching the host upscale.
    // Gated on the same VMECPP_BATCH_MULTIGRID_UPSCALE knob; otherwise the
    // existing free + re-alloc path is unchanged.
    {
      const int scalxc_upscale_env =
          RunEnvFlag(&g_batch_upscale_env, "VMECPP_BATCH_MULTIGRID_UPSCALE");
      // scalxc_staged is reset earlier in Reshape (line ~9580), so don't
      // check it here. d_scalxc != null is the right guard for the data:
      // the buffer still holds the OLD ns staged values until the
      // cuda_free_if below. pts_x_initialized distinguishes a genuine
      // stage transition (position state live) from the first Reshape of
      // a fresh run after ResetForNewVmecRun, whose leftover scalxc
      // belongs to the prior run and must not arm the upscale.
      if (scalxc_upscale_env > 0 && d_scalxc && pts_x_initialized &&
          pts_x_ns > 0 && pts_x_ns != ns_local) {
        int scalxc_len_old = pts_x_ns * 2;
        size_t bytes_prev = sizeof(double) * (size_t)n_config_max *
                             (size_t)scalxc_len_old;
        if (d_scalxc_prev) cudaFree(d_scalxc_prev);
        cuda_check(cudaMalloc(&d_scalxc_prev, bytes_prev),
                   "alloc d_scalxc_prev");
        cuda_check(cudaMemcpyAsync(d_scalxc_prev, d_scalxc, bytes_prev,
                                    cudaMemcpyDeviceToDevice, stream),
                   "d2d scalxc → scalxc_prev (Reshape snapshot)");
        scalxc_prev_len = scalxc_len_old;
        scalxc_prev_valid = true;
        std::fprintf(stderr,
            "[fft_toroidal_cuda] Reshape scalxc snapshot: scalxc_len %d "
            "(n_cfg=%d), %zu bytes\n",
            scalxc_len_old, n_config_max, bytes_prev);
      }
    }
    cuda_free_if((void*&)d_scalxc);
    decomposed_size_cached = -1;
    // PerformTimeStep persistent state.
    cuda_free_if((void*&)d_pts_v_rcc);
    cuda_free_if((void*&)d_pts_v_rss);
    cuda_free_if((void*&)d_pts_v_zsc);
    cuda_free_if((void*&)d_pts_v_zcs);
    cuda_free_if((void*&)d_pts_v_lsc);
    cuda_free_if((void*&)d_pts_v_lcs);
    // Multigrid-stage transition snapshot: capture per-cfg d_pts_x slices
    // into d_pts_x_prev BEFORE freeing d_pts_x, so PerformTimeStepCuda's
    // init branch can dispatch the radial-interp kernel into the freshly
    // allocated buffers at the new ns_local. Required for distinct-mode
    // batched runs to preserve per-cfg state across stages.
    const int upscale_env_reshape =
        RunEnvFlag(&g_batch_upscale_env, "VMECPP_BATCH_MULTIGRID_UPSCALE");
    if (upscale_env_reshape > 0 && pts_x_initialized && d_pts_x_rcc &&
        pts_x_size > 0 && pts_x_ns > 0 &&
        ns_local != pts_x_ns) {
      auto cuda_free_inline = [](double*& p) {
        if (p) { cudaFree(p); p = nullptr; }
      };
      auto realloc_prev = [&](double*& p) {
        cuda_free_inline(p);
        size_t bytes_prev =
            sizeof(double) * (size_t)n_config_max * pts_x_size;
        cuda_check(cudaMalloc(&p, bytes_prev), "alloc d_pts_x_prev (Reshape)");
      };
      realloc_prev(d_pts_x_prev_rcc); realloc_prev(d_pts_x_prev_rss);
      realloc_prev(d_pts_x_prev_zsc); realloc_prev(d_pts_x_prev_zcs);
      realloc_prev(d_pts_x_prev_lsc); realloc_prev(d_pts_x_prev_lcs);
      size_t bytes_prev = sizeof(double) * (size_t)n_config_max * pts_x_size;
      double* src[6] = {d_pts_x_rcc, d_pts_x_rss, d_pts_x_zsc,
                        d_pts_x_zcs, d_pts_x_lsc, d_pts_x_lcs};
      double* dst[6] = {d_pts_x_prev_rcc, d_pts_x_prev_rss, d_pts_x_prev_zsc,
                        d_pts_x_prev_zcs, d_pts_x_prev_lsc, d_pts_x_prev_lcs};
      for (int i = 0; i < 6; ++i) {
        cuda_check(cudaMemcpyAsync(dst[i], src[i], bytes_prev,
                                    cudaMemcpyDeviceToDevice, stream),
                   "d2d pts_x → pts_x_prev (Reshape snapshot)");
      }
      pts_x_prev_size = pts_x_size;
      pts_x_prev_ns = pts_x_ns;
      pts_x_prev_valid = true;
      std::fprintf(stderr,
          "[fft_toroidal_cuda] Reshape multigrid snapshot: ns %d → %d "
          "(n_cfg=%d), %zu bytes/spec captured\n",
          pts_x_ns, ns_local, n_config_max, bytes_prev);
    }
    cuda_free_if((void*&)d_pts_x_rcc);
    cuda_free_if((void*&)d_pts_x_rss);
    cuda_free_if((void*&)d_pts_x_zsc);
    cuda_free_if((void*&)d_pts_x_zcs);
    cuda_free_if((void*&)d_pts_x_lsc);
    cuda_free_if((void*&)d_pts_x_lcs);
    cuda_free_if((void*&)d_pts_x_backup_rcc);
    cuda_free_if((void*&)d_pts_x_backup_rss);
    cuda_free_if((void*&)d_pts_x_backup_zsc);
    cuda_free_if((void*&)d_pts_x_backup_zcs);
    cuda_free_if((void*&)d_pts_x_backup_lsc);
    cuda_free_if((void*&)d_pts_x_backup_lcs);
    pts_x_backup_initialized = false;
    cuda_free_if((void*&)d_pts_x_final_rcc);
    cuda_free_if((void*&)d_pts_x_final_rss);
    cuda_free_if((void*&)d_pts_x_final_zsc);
    cuda_free_if((void*&)d_pts_x_final_zcs);
    cuda_free_if((void*&)d_pts_x_final_lsc);
    cuda_free_if((void*&)d_pts_x_final_lcs);
    pts_x_final_taken.clear();
    cuda_free_if((void*&)d_rznorm_partials);
    pinned_free_if((void*&)h_rznorm_partials);
    pts_v_size = -1;
    pts_x_size = -1;
    pts_v_initialized = false;
    pts_x_initialized = false;
    // Persistent preconditioner-input buffers.
    cuda_free_if((void*&)d_m1_ard);
    cuda_free_if((void*&)d_m1_brd);
    cuda_free_if((void*&)d_m1_azd);
    cuda_free_if((void*&)d_m1_bzd);
    cuda_free_if((void*&)d_lambda_lp);
    cuda_free_if((void*&)d_rz_aR);
    cuda_free_if((void*&)d_rz_dR);
    cuda_free_if((void*&)d_rz_bR);
    cuda_free_if((void*&)d_rz_cR);
    cuda_free_if((void*&)d_rz_aZ);
    cuda_free_if((void*&)d_rz_dZ);
    cuda_free_if((void*&)d_rz_bZ);
    cuda_free_if((void*&)d_rz_cZ);
    cuda_free_if((void*&)d_rz_c_orig_R);
    cuda_free_if((void*&)d_rz_c_orig_Z);
    cuda_free_if((void*&)d_rz_x_saved_R);
    cuda_free_if((void*&)d_rz_x_saved_Z);
    if (d_rz_jMin) { cudaFree(d_rz_jMin); d_rz_jMin = nullptr; }
    rz_mnsize_cached = -1;
    rz_ns_total_cached = -1;
    rz_num_basis_cached = -1;
    rz_cache_ar_sentinel = std::numeric_limits<double>::quiet_NaN();
    cuda_free_if((void*&)d_scalar);
    cuda_free_if((void*&)d_pressure_scalars);
    cuda_free_if((void*&)d_jac_minmax);
    cuda_free_if((void*&)d_residuals_partial);
    cuda_free_if((void*&)d_residuals_partials_K);
    cuda_free_if((void*&)d_inv_tau);
    cuda_free_if((void*&)d_prev_fsq);
    cuda_free_if((void*&)d_fac_b1);
    cuda_free_if((void*&)d_fnorm1);
    fnorm1_staged_valid = false;
    fnorm1_device_filled = false;
    if (batch_inputs_pinned) {
      cudaFreeHost(batch_inputs_pinned);
      batch_inputs_pinned = nullptr;
    }
    batch_inputs_n_cfg = 0;
    batch_inputs_one_spec_doubles = 0;
    cuda_free_if((void*&)d_fnorm_scalars);
    cuda_free_if((void*&)d_active_per_cfg);
    // Convergence-flag, deferred-residual, and restart-mask buffers are
    // sized by n_config_max and allocated behind null guards outside
    // Reshape; free them here so a run with a different configuration
    // count reallocates them at the new size instead of overrunning the
    // old allocations.
    cuda_free_if((void*&)d_conv_flag);
    pinned_free_if((void*&)h_conv_flag_pinned);
    pinned_free_if((void*&)h_residuals_pinned);
    if (residuals_d2h_event) {
      cudaEventDestroy(residuals_d2h_event);
      residuals_d2h_event = nullptr;
    }
    residuals_d2h_pending = false;
    cuda_free_if((void*&)d_restart_mask);
    ns_h_cached = -1;
    nZnT_cached = -1;
    nThetaEff_for_wInt = -1;
    // Sub-pointers become invalid once their backing block is freed.
    d_rmncc = d_rmnss = d_zmnsc = d_zmncs = d_lmnsc = d_lmncs = nullptr;
    d_xmpq = d_sqrtSF = nullptr;
    d_r1_e = d_r1_o = d_ru_e = d_ru_o = d_rv_e = d_rv_o = nullptr;
    d_z1_e = d_z1_o = d_zu_e = d_zu_o = d_zv_e = d_zv_o = nullptr;
    d_lu_e = d_lu_o = d_lv_e = d_lv_o = nullptr;
    d_rCon = d_zCon = nullptr;
    if (cufft_plan) { cufftDestroy(cufft_plan); cufft_plan = 0; }
    // Reshape may change the device pointers that the captured CUDA
    // graphs reference, so the executable graphs and the underlying
    // graph descriptors are destroyed here and rebuilt lazily on the
    // next capture attempt.
    if (fwd_graph_exec) { cudaGraphExecDestroy(fwd_graph_exec); fwd_graph_exec = nullptr; }
    if (fwd_graph) { cudaGraphDestroy(fwd_graph); fwd_graph = nullptr; }
    fwd_graph_captured = false;
    if (iter_graph_exec) { cudaGraphExecDestroy(iter_graph_exec); iter_graph_exec = nullptr; }
    if (iter_graph) { cudaGraphDestroy(iter_graph); iter_graph = nullptr; }
    iter_graph_captured = false;
    iter_graph_warmups = 0;
    if (seg3_graph_exec) { cudaGraphExecDestroy(seg3_graph_exec); seg3_graph_exec = nullptr; }
    if (seg3_graph) { cudaGraphDestroy(seg3_graph); seg3_graph = nullptr; }
    seg3_graph_captured = false;
    seg3_in_capture = false;
    seg3_warmup_calls = 0;
    if (seg4_graph_exec) { cudaGraphExecDestroy(seg4_graph_exec); seg4_graph_exec = nullptr; }
    if (seg4_graph) { cudaGraphDestroy(seg4_graph); seg4_graph = nullptr; }
    seg4_graph_captured = false;
    seg4_in_capture = false;
    seg4_warmup_calls = 0;
    seg4_last_jMax = -1;
    if (seg2_graph_exec) { cudaGraphExecDestroy(seg2_graph_exec); seg2_graph_exec = nullptr; }
    if (seg2_graph) { cudaGraphDestroy(seg2_graph); seg2_graph = nullptr; }
    seg2_graph_captured = false;
    seg2_in_capture = false;
    seg2_warmup_calls = 0;
    if (stream) { cudaStreamDestroy(stream); stream = nullptr; }

    // ----- Single contiguous specs block: 6 spec arrays + xmpq + sqrtSF -----
    // Batched layout: per-config buffers sized by n_config_max. xmpq stays
    // shared across configs (constant per shape). At n_config_max=1 the
    // layout is identical to the single-configuration arrangement.
    size_t one_spec_bytes = sizeof(double) * n_config_max * ns_local * mpol * (ntor + 1);
    specs_block_bytes = 6 * one_spec_bytes + sizeof(double) * mpol +
                        sizeof(double) * ns_local * n_config_max;
    cuda_check(cudaMalloc(&d_specs_block, specs_block_bytes),
               "alloc d_specs_block");
    cuda_check(cudaMallocHost(&h_specs_pinned, specs_block_bytes),
               "alloc h_specs_pinned");
    size_t off = 0;
    d_rmncc = d_specs_block + off / sizeof(double); off += one_spec_bytes;
    d_rmnss = d_specs_block + off / sizeof(double); off += one_spec_bytes;
    d_zmnsc = d_specs_block + off / sizeof(double); off += one_spec_bytes;
    d_zmncs = d_specs_block + off / sizeof(double); off += one_spec_bytes;
    d_lmnsc = d_specs_block + off / sizeof(double); off += one_spec_bytes;
    d_lmncs = d_specs_block + off / sizeof(double); off += one_spec_bytes;
    d_xmpq  = d_specs_block + off / sizeof(double); off += sizeof(double) * mpol;
    d_sqrtSF = d_specs_block + off / sizeof(double);

    // ----- FFT scratch (X, Y) -----
    // Batched layout: scratch sized by n_config_max so cuFFT batch covers
    // N configs. cuFFT plan also takes n_config_max via the batch arg below.
    fft_x_elems = (size_t)n_config_max * ns_local * mpol * kBatch * nhalf;
    fft_y_elems = (size_t)n_config_max * ns_local * mpol * kBatch * nZeta;
    size_t X_bytes = sizeof(cufftDoubleComplex) * fft_x_elems;
    size_t Y_bytes = sizeof(double) * fft_y_elems;
    cuda_check(cudaMalloc(&d_X, X_bytes), "alloc X");
    cuda_check(cudaMalloc(&d_Y, Y_bytes), "alloc Y");
    cuda_check(cudaMalloc(&d_X_fp32, sizeof(cufftComplex) * fft_x_elems),
               "alloc d_X_fp32");
    cuda_check(cudaMalloc(&d_Y_fp32, sizeof(float) * fft_y_elems),
               "alloc d_Y_fp32");

    // ----- Single contiguous outputs block: 16 main + 2 con -----
    // Batched layout: per-config outputs sized by n_config_max.
    main_array_bytes = sizeof(double) * n_config_max * ns_local * nZeta * nThetaEff;
    con_array_bytes  = sizeof(double) * n_config_max * ns_con_local * nZeta * nThetaEff;
    outputs_block_bytes = 16 * main_array_bytes + 2 * con_array_bytes;
    cuda_check(cudaMalloc(&d_outputs_block, outputs_block_bytes),
               "alloc d_outputs_block");
    cuda_check(cudaMallocHost(&h_outputs_pinned, outputs_block_bytes),
               "alloc h_outputs_pinned");
    // 6-double scratch for SetRadialExtent + SetGeometricOffset extract.
    cuda_check(cudaMalloc(&d_geom_scalars, 6 * sizeof(double)),
               "alloc d_geom_scalars");
    cuda_check(cudaMallocHost(&h_geom_scalars, 6 * sizeof(double)),
               "alloc h_geom_scalars");
    off = 0;
    d_r1_e = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_r1_o = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_ru_e = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_ru_o = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_rv_e = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_rv_o = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_z1_e = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_z1_o = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_zu_e = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_zu_o = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_zv_e = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_zv_o = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_lu_e = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_lu_o = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_lv_e = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_lv_o = d_outputs_block + off / sizeof(double); off += main_array_bytes;
    d_rCon = d_outputs_block + off / sizeof(double); off += con_array_bytes;
    d_zCon = d_outputs_block + off / sizeof(double);

    cuda_check(cudaStreamCreate(&stream), "stream create");

    int n_dim[1] = {nZeta};
    // Batched layout: batch dim multiplied by n_config_max. At
    // n_config_max=1 this is identical to the single-configuration plan.
    int batch = n_config_max * ns_local * mpol * kBatch;
    cufft_check(cufftPlanMany(&cufft_plan, 1, n_dim,
                              nullptr, 1, nhalf,
                              nullptr, 1, nZeta,
                              CUFFT_Z2D, batch),
                "cufftPlanMany");
    cufft_check(cufftSetStream(cufft_plan, stream), "cufftSetStream");

    // Single-precision complex-to-real plan companion to the
    // double-precision inverse plan. The batch shape is identical;
    // the plan is used by the mixed-precision path that the
    // VMECPP_FFT_FP32 environment variable selects.
    cufft_check(cufftPlanMany(&cufft_plan_c2r_fp32, 1, n_dim,
                              nullptr, 1, nhalf,
                              nullptr, 1, nZeta,
                              CUFFT_C2R, batch),
                "cufftPlanMany C2R fp32");
    cufft_check(cufftSetStream(cufft_plan_c2r_fp32, stream),
                "cufftSetStream C2R fp32");

    // Inverse R2C plan: same batch shape, D2Z direction.
    cufft_check(cufftPlanMany(&cufft_plan_r2c, 1, n_dim,
                              nullptr, 1, nZeta,
                              nullptr, 1, nhalf,
                              CUFFT_D2Z, batch),
                "cufftPlanMany R2C");
    cufft_check(cufftSetStream(cufft_plan_r2c, stream), "cufftSetStream R2C");

    ns_local_cached = ns_local;
    ns_con_local_cached = ns_con_local;
    mpol_cached = mpol;
    ntor_cached = ntor;
    nhalf_cached = nhalf;
    nZeta_cached = nZeta;
    nThetaReduced_cached = nThetaReduced;
    nThetaEff_cached = nThetaEff;
    // Footprint bookkeeping for the admission pre-flight: the next
    // Reshape frees this much before a new run's allocations land, so
    // CudaVramBudgetCuda credits it against the free-memory query.
    reshape_budget_raw_bytes = CudaBudgetRawBytes(
        n_config_max, ns_local, mpol, ntor, nZeta, nThetaEff);
  }

  // Stage constant basis arrays once (or when shape changes).
  void StageBasis(int nhalf, int mpol, int nThetaReduced,
                  const double* nscale, const double* cosmu, const double* sinmu,
                  const double* cosmum, const double* sinmum) {
    auto alloc_if_needed = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc basis");
    };
    alloc_if_needed(d_nscale, sizeof(double) * nhalf);
    alloc_if_needed(d_cosmu,  sizeof(double) * mpol * nThetaReduced);
    alloc_if_needed(d_sinmu,  sizeof(double) * mpol * nThetaReduced);
    alloc_if_needed(d_cosmum, sizeof(double) * mpol * nThetaReduced);
    alloc_if_needed(d_sinmum, sizeof(double) * mpol * nThetaReduced);
    cuda_check(cudaMemcpyAsync(d_nscale, nscale, sizeof(double) * nhalf,
                          cudaMemcpyHostToDevice, stream), "h2d nscale");
    cuda_check(cudaMemcpyAsync(d_cosmu, cosmu, sizeof(double) * mpol * nThetaReduced,
                          cudaMemcpyHostToDevice, stream), "h2d cosmu");
    cuda_check(cudaMemcpyAsync(d_sinmu, sinmu, sizeof(double) * mpol * nThetaReduced,
                          cudaMemcpyHostToDevice, stream), "h2d sinmu");
    cuda_check(cudaMemcpyAsync(d_cosmum, cosmum,
                          sizeof(double) * mpol * nThetaReduced,
                          cudaMemcpyHostToDevice, stream), "h2d cosmum");
    cuda_check(cudaMemcpyAsync(d_sinmum, sinmum,
                          sizeof(double) * mpol * nThetaReduced,
                          cudaMemcpyHostToDevice, stream), "h2d sinmum");
  }

  // Stage the toroidal discrete Fourier transform basis tables for
  // the fused single-pass forward-FFT kernels. The host computes the
  // cosine and sine of 2 pi n k / nZeta over the (n, k) lattice with
  // the toroidal mode-scaling factor nscale[n] folded into the
  // amplitude, copies the resulting tables to the device, and caches
  // the (ntor, nZeta) extents to skip the staging on subsequent
  // invocations with unchanged shape.
  void StageDftBasis(int ntor, int nZeta, const double* nscale) {
    if (d_dft_cos && dft_basis_ntor_cached == ntor &&
        dft_basis_nZeta_cached == nZeta) return;
    auto cuda_free_if = [](void*& p) {
      if (p) { cudaFree(p); p = nullptr; }
    };
    cuda_free_if((void*&)d_dft_cos);
    cuda_free_if((void*&)d_dft_sin);
    size_t bytes = sizeof(double) * (size_t)(ntor + 1) * (size_t)nZeta;
    cuda_check(cudaMalloc(&d_dft_cos, bytes), "alloc d_dft_cos");
    cuda_check(cudaMalloc(&d_dft_sin, bytes), "alloc d_dft_sin");
    std::vector<double> h_cos((ntor + 1) * nZeta);
    std::vector<double> h_sin((ntor + 1) * nZeta);
    const double two_pi = 6.283185307179586476925286766559;
    for (int n = 0; n <= ntor; ++n) {
      double ns_n = nscale[n];
      for (int k = 0; k < nZeta; ++k) {
        double angle = two_pi * (double)n * (double)k / (double)nZeta;
        h_cos[n * nZeta + k] = ns_n * std::cos(angle);
        h_sin[n * nZeta + k] = ns_n * std::sin(angle);
      }
    }
    cuda_check(cudaMemcpyAsync(d_dft_cos, h_cos.data(), bytes,
                                cudaMemcpyHostToDevice, stream), "h2d d_dft_cos");
    cuda_check(cudaMemcpyAsync(d_dft_sin, h_sin.data(), bytes,
                                cudaMemcpyHostToDevice, stream), "h2d d_dft_sin");
    cuda_check(cudaStreamSynchronize(stream), "dft basis stage sync");
    dft_basis_ntor_cached = ntor;
    dft_basis_nZeta_cached = nZeta;
  }

  // Stage the cosine and sine tables consumed by the direct
  // length-24 inverse discrete Fourier transform kernel
  // k_inverse_dft_24. The tables hold the raw values cos(2 pi n k / nZeta)
  // and sin(2 pi n k / nZeta) over the (n, k) lattice with no
  // toroidal mode-scaling factor folded in, since the kernel reads
  // Hermitian-symmetric complex spectra in the format cufftExecZ2D
  // would consume and is intended to produce a real output
  // mathematically equivalent to that of cufftExecZ2D.
  void StageInverseDftBasis(int nhalf, int nZeta) {
    if (d_idft_cos && idft_basis_nhalf_cached == nhalf &&
        idft_basis_nZeta_cached == nZeta) return;
    auto cuda_free_if = [](void*& p) {
      if (p) { cudaFree(p); p = nullptr; }
    };
    cuda_free_if((void*&)d_idft_cos);
    cuda_free_if((void*&)d_idft_sin);
    size_t bytes = sizeof(double) * (size_t)nhalf * (size_t)nZeta;
    cuda_check(cudaMalloc(&d_idft_cos, bytes), "alloc d_idft_cos");
    cuda_check(cudaMalloc(&d_idft_sin, bytes), "alloc d_idft_sin");
    std::vector<double> h_cos((size_t)nhalf * (size_t)nZeta);
    std::vector<double> h_sin((size_t)nhalf * (size_t)nZeta);
    const double two_pi = 6.283185307179586476925286766559;
    for (int n = 0; n < nhalf; ++n) {
      for (int k = 0; k < nZeta; ++k) {
        double angle = two_pi * (double)n * (double)k / (double)nZeta;
        h_cos[(size_t)n * (size_t)nZeta + (size_t)k] = std::cos(angle);
        h_sin[(size_t)n * (size_t)nZeta + (size_t)k] = std::sin(angle);
      }
    }
    cuda_check(cudaMemcpyAsync(d_idft_cos, h_cos.data(), bytes,
                                cudaMemcpyHostToDevice, stream), "h2d d_idft_cos");
    cuda_check(cudaMemcpyAsync(d_idft_sin, h_sin.data(), bytes,
                                cudaMemcpyHostToDevice, stream), "h2d d_idft_sin");
    cuda_check(cudaStreamSynchronize(stream), "idft basis stage sync");
    idft_basis_nhalf_cached = nhalf;
    idft_basis_nZeta_cached = nZeta;
  }

  // Stage the integration-weighted basis variants used by the inverse FFT.
  void StageBasisI(int mpol, int nThetaReduced,
                    const double* cosmui, const double* sinmui,
                    const double* cosmumi, const double* sinmumi) {
    auto alloc_if_needed = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc basis_i");
    };
    size_t bytes = sizeof(double) * mpol * nThetaReduced;
    alloc_if_needed(d_cosmui,  bytes);
    alloc_if_needed(d_sinmui,  bytes);
    alloc_if_needed(d_cosmumi, bytes);
    alloc_if_needed(d_sinmumi, bytes);
    cuda_check(cudaMemcpyAsync(d_cosmui,  cosmui,  bytes,
                                cudaMemcpyHostToDevice, stream), "h2d cosmui");
    cuda_check(cudaMemcpyAsync(d_sinmui,  sinmui,  bytes,
                                cudaMemcpyHostToDevice, stream), "h2d sinmui");
    cuda_check(cudaMemcpyAsync(d_cosmumi, cosmumi, bytes,
                                cudaMemcpyHostToDevice, stream), "h2d cosmumi");
    cuda_check(cudaMemcpyAsync(d_sinmumi, sinmumi, bytes,
                                cudaMemcpyHostToDevice, stream), "h2d sinmumi");
  }

  // Toroidal basis arrays (cosnv/sinnv) for deAlias and elsewhere. Staged once
  // per shape at Reshape-time (called from FourierToReal3DSymmFastPoloidalCuda
  // after StageBasis/StageBasisI). Removes the per-call sentinel check that
  // previously lived in EnsureDealiasBuffers.
  void StageToroidalBasis(int nZeta, int nnyq2_plus_1,
                          const double* cosnv, const double* sinnv) {
    size_t cv_bytes = sizeof(double) * nZeta * nnyq2_plus_1;
    if (d_dealias_cosnv) { cudaFree(d_dealias_cosnv); d_dealias_cosnv = nullptr; }
    if (d_dealias_sinnv) { cudaFree(d_dealias_sinnv); d_dealias_sinnv = nullptr; }
    cuda_check(cudaMalloc(&d_dealias_cosnv, cv_bytes), "alloc dealias cosnv");
    cuda_check(cudaMalloc(&d_dealias_sinnv, cv_bytes), "alloc dealias sinnv");
    cuda_check(cudaMemcpyAsync(d_dealias_cosnv, cosnv, cv_bytes,
                                cudaMemcpyHostToDevice, stream), "h2d dealias cosnv");
    cuda_check(cudaMemcpyAsync(d_dealias_sinnv, sinnv, cv_bytes,
                                cudaMemcpyHostToDevice, stream), "h2d dealias sinnv");
    dealias_nnyq2_plus_1_cached = nnyq2_plus_1;
  }

  // FourierForces spec array device shadows.
  void EnsureFourierForcesBuffers(int ns_local, int mpol, int ntor) {
    auto alloc_if_null = [this](double*& p, size_t bytes) {
      if (!p) {
        cuda_check(cudaMalloc(&p, bytes), "alloc fForces buf");
        // Zero-initialize the freshly allocated device buffer so that
        // configuration slots whose corresponding kernels fail to
        // write them present a deterministic zero rather than the
        // uninitialized bit pattern that may otherwise decode as a
        // signaling NaN. The configuration slots beyond cfg = 0 are
        // consumed by RecomposeToPhysicalCuda, which reads each
        // d_pts_x configuration slot derived from these forces; an
        // uninitialized NaN would propagate through the subsequent
        // arithmetic and contaminate downstream kernel outputs.
        cuda_check(cudaMemsetAsync(p, 0, bytes, stream),
                   "memset fForces buf zero-init");
      }
    };
    // Batched layout: per-config inverse-FFT spec arrays.
    size_t bytes = sizeof(double) * n_config_max * ns_local * mpol * (ntor + 1);
    alloc_if_null(d_frcc, bytes);
    alloc_if_null(d_frss, bytes);
    alloc_if_null(d_fzsc, bytes);
    alloc_if_null(d_fzcs, bytes);
    alloc_if_null(d_flsc, bytes);
    alloc_if_null(d_flcs, bytes);
  }

  // Allocate/copy per-call host-mutable arrays (xmpq, sqrtSF).
  void StagePerCall(int mpol, int ns_local, const double* xmpq,
                    const double* sqrtSF) {
    if (!d_xmpq) {
      cuda_check(cudaMalloc((void**)&d_xmpq, sizeof(double) * mpol),
                 "alloc d_xmpq");
    }
    if (!d_sqrtSF) {
      cuda_check(cudaMalloc((void**)&d_sqrtSF, sizeof(double) * ns_local),
                 "alloc d_sqrtSF");
    }
    cuda_check(cudaMemcpyAsync(d_xmpq, xmpq, sizeof(double) * mpol,
                          cudaMemcpyHostToDevice, stream), "h2d xmpq");
    cuda_check(cudaMemcpyAsync(d_sqrtSF, sqrtSF, sizeof(double) * ns_local,
                          cudaMemcpyHostToDevice, stream), "h2d sqrtSF");
  }

  // Allocate persistent jacobian-output buffers for given (ns_h, nZnT). Only
  // re-allocates when the shape changes.
  void EnsureJacobianBuffers(int ns_h, int nZnT) {
    if (ns_h_cached == ns_h && nZnT_cached == nZnT && d_r12 != nullptr) return;
    auto cuda_free_if = [](void*& p) {
      if (p) { cudaFree(p); p = nullptr; }
    };
    auto pinned_free_if = [](void*& p) {
      if (p) { cudaFreeHost(p); p = nullptr; }
    };
    cuda_free_if((void*&)d_r12);
    cuda_free_if((void*&)d_ru12);
    cuda_free_if((void*&)d_zu12);
    cuda_free_if((void*&)d_rs);
    cuda_free_if((void*&)d_zs);
    cuda_free_if((void*&)d_tau);
    cuda_free_if((void*&)d_sqrtSH);
    pinned_free_if((void*&)h_jac_pinned);

    // Batched layout: per-config buffers sized by n_config_max. sqrtSH is
    // shape-constant (radial sqrt grid values) so shared across configs.
    jac_array_bytes = sizeof(double) * n_config_max * ns_h * nZnT;
    cuda_check(cudaMalloc(&d_r12, jac_array_bytes), "alloc d_r12");
    cuda_check(cudaMalloc(&d_ru12, jac_array_bytes), "alloc d_ru12");
    cuda_check(cudaMalloc(&d_zu12, jac_array_bytes), "alloc d_zu12");
    cuda_check(cudaMalloc(&d_rs, jac_array_bytes), "alloc d_rs");
    cuda_check(cudaMalloc(&d_zs, jac_array_bytes), "alloc d_zs");
    cuda_check(cudaMalloc(&d_tau, jac_array_bytes), "alloc d_tau");
    cuda_check(cudaMalloc(&d_sqrtSH, sizeof(double) * ns_h), "alloc d_sqrtSH");
    cuda_check(cudaMallocHost(&h_jac_pinned, 6 * jac_array_bytes),
               "alloc h_jac_pinned");
    sqrtSH_staged = false;  // newly-allocated buffer, no data yet
    ns_h_cached = ns_h;
    nZnT_cached = nZnT;
  }

  // Allocate metric-element output buffers (gsqrt, guu, guv, gvv) at half-grid.
  // Shares jac_array_bytes (same ns_h × nZnT shape).
  void EnsureMetricBuffers(int ns_h, int nZnT) {
    EnsureJacobianBuffers(ns_h, nZnT);  // ensures jac arrays + d_sqrtSH exist
    auto cuda_free_if = [](void*& p) {
      if (p) { cudaFree(p); p = nullptr; }
    };
    auto pinned_free_if = [](void*& p) {
      if (p) { cudaFreeHost(p); p = nullptr; }
    };
    if (d_gsqrt == nullptr) {
      cuda_check(cudaMalloc(&d_gsqrt, jac_array_bytes), "alloc d_gsqrt");
    }
    if (d_guu == nullptr) {
      cuda_check(cudaMalloc(&d_guu, jac_array_bytes), "alloc d_guu");
    }
    if (d_guv == nullptr) {
      cuda_check(cudaMalloc(&d_guv, jac_array_bytes), "alloc d_guv");
    }
    if (d_gvv == nullptr) {
      cuda_check(cudaMalloc(&d_gvv, jac_array_bytes), "alloc d_gvv");
    }
    if (h_metric_pinned == nullptr) {
      cuda_check(cudaMallocHost(&h_metric_pinned, 4 * jac_array_bytes),
                 "alloc h_metric_pinned");
    }
  }

  // Stage wInt (poloidal integration weights, size nThetaEff). Constant.
  void EnsureWIntStaged(int nThetaEff_in, const double* wInt) {
    if (nThetaEff_for_wInt == nThetaEff_in && d_wInt != nullptr) return;
    if (d_wInt) { cudaFree(d_wInt); d_wInt = nullptr; }
    cuda_check(cudaMalloc(&d_wInt, sizeof(double) * nThetaEff_in),
               "alloc d_wInt");
    cuda_check(cudaMemcpyAsync(d_wInt, wInt, sizeof(double) * nThetaEff_in,
                                cudaMemcpyHostToDevice, stream), "h2d wInt");
    nThetaEff_for_wInt = nThetaEff_in;
  }

  // dVdsH buffer (per-config radial profile, size ns_h per config).
  void EnsureDVdsHBuffer(int ns_h) {
    if (d_dVdsH != nullptr) return;
    cuda_check(cudaMalloc(&d_dVdsH, sizeof(double) * n_config_max * ns_h), "alloc d_dVdsH");
  }

  // BCo output buffers (bsubu, bsubv); half-grid arrays.
  void EnsureBCoBuffers() {
    if (d_bsubu == nullptr) {
      cuda_check(cudaMalloc(&d_bsubu, jac_array_bytes), "alloc d_bsubu");
    }
    if (d_bsubv == nullptr) {
      cuda_check(cudaMalloc(&d_bsubv, jac_array_bytes), "alloc d_bsubv");
    }
  }

  // Batched layout: scalar reductions are per-config (each config has its
  // own min/max/sum). At n_config_max=1 these are 1-3 doubles, same as before.
  void EnsureScalarBuffer() {
    if (!d_scalar) {
      cuda_check(cudaMalloc(&d_scalar, sizeof(double) * n_config_max), "alloc d_scalar");
    }
  }

  void EnsurePressureScalarsBuffer() {
    if (!d_pressure_scalars) {
      cuda_check(cudaMalloc(&d_pressure_scalars, 3 * sizeof(double) * n_config_max),
                 "alloc d_pressure_scalars");
    }
  }

  void EnsureJacMinmaxBuffer() {
    if (!d_jac_minmax) {
      cuda_check(cudaMalloc(&d_jac_minmax, 2 * sizeof(double) * n_config_max),
                 "alloc d_jac_minmax");
    }
  }

  void EnsureTimestepBuffers(double time_step_init) {
    if (!d_inv_tau) {
      cuda_check(cudaMalloc(&d_inv_tau,
                             (size_t)kTimestepNDamp * sizeof(double) * n_config_max),
                 "alloc d_inv_tau");
      // Initialize all entries to 0.15 / time_step (matches host
      // invTau_.setConstant(0.15 / time_step) at iter1 == iter2).
      // We do this once per allocation; the per-iter init logic inside
      // k_update_timestep also resets when iter_idx == 0.
      const double init_val = 0.15 / time_step_init;
      std::vector<double> host_init((size_t)kTimestepNDamp * n_config_max,
                                     init_val);
      cuda_check(cudaMemcpyAsync(d_inv_tau, host_init.data(),
                                  host_init.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                 "h2d d_inv_tau init");
    }
    if (!d_prev_fsq) {
      cuda_check(cudaMalloc(&d_prev_fsq,
                             sizeof(double) * n_config_max),
                 "alloc d_prev_fsq");
      // Init to 1.0 (matches FlowControl::fsq default).
      std::vector<double> ones(n_config_max, 1.0);
      cuda_check(cudaMemcpyAsync(d_prev_fsq, ones.data(),
                                  ones.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                 "h2d d_prev_fsq init");
    }
    if (!d_fac_b1) {
      cuda_check(cudaMalloc(&d_fac_b1,
                             2 * sizeof(double) * n_config_max),
                 "alloc d_fac_b1");
    }
    EnsureFnorm1Buffer();
  }

  void EnsureFnorm1Buffer() {
    if (!d_fnorm1) {
      cuda_check(cudaMalloc(&d_fnorm1, sizeof(double) * n_config_max),
                 "alloc d_fnorm1");
      cuda_check(cudaMemsetAsync(d_fnorm1, 0,
                                  sizeof(double) * n_config_max, stream),
                 "memset d_fnorm1");
      fnorm1_staged_valid = false;
      fnorm1_device_filled = false;
    }
  }

  // Refresh the device fnorm1 slots when the host value changes (the host
  // recomputes fnorm1 at preconditioner boundaries). Mid-window calls hit
  // the staged cache and enqueue nothing, so the controller launch stays
  // capturable; the boundary refresh itself must run outside capture.
  // Stands down once k_rz_norm_per_cfg fills the slots per cfg.
  void StageFnorm1(double fnorm1) {
    if (fnorm1_device_filled) return;
    if (fnorm1_staged_valid && fnorm1 == fnorm1_staged) return;
    std::vector<double> vals(n_config_max, fnorm1);
    cuda_check(cudaMemcpyAsync(d_fnorm1, vals.data(),
                                vals.size() * sizeof(double),
                                cudaMemcpyHostToDevice, stream),
               "h2d d_fnorm1");
    fnorm1_staged = fnorm1;
    fnorm1_staged_valid = true;
  }

  void EnsureResidualsBuffer() {
    if (!d_residuals_partial) {
      cuda_check(cudaMalloc(&d_residuals_partial, 3 * sizeof(double) * n_config_max),
                 "alloc d_residuals_partial");
    }
    if (!d_residuals_partials_K) {
      cuda_check(cudaMalloc(&d_residuals_partials_K,
                             (size_t)kResidualsKPartitions * 3 *
                             sizeof(double) * n_config_max),
                 "alloc d_residuals_partials_K");
    }
    if (!h_residuals_pinned) {
      // Pinned host buffer for deferred-sync residuals D2H. Allocated
      // once per Reshape; freed in the dtor. Size matches d_residuals_partial.
      cuda_check(cudaMallocHost(&h_residuals_pinned,
                                 3 * sizeof(double) * n_config_max),
                 "alloc h_residuals_pinned");
      cuda_check(cudaEventCreateWithFlags(&residuals_d2h_event,
                                           cudaEventDisableTiming),
                 "create residuals_d2h_event");
      residuals_d2h_pending = false;
    }
    if (!d_conv_flag) {
      // Device + pinned-host buffers for k_check_convergence flag. The
      // device kernel writes a per-cfg byte (1 = converged, 0 = not) and
      // an async memcpy copies it to the pinned host buffer for non-
      // blocking polling by the iter loop control.
      cuda_check(cudaMalloc(&d_conv_flag,
                             sizeof(std::uint8_t) * n_config_max),
                 "alloc d_conv_flag");
      cuda_check(cudaMallocHost(&h_conv_flag_pinned,
                                 sizeof(std::uint8_t) * n_config_max),
                 "alloc h_conv_flag_pinned");
    }
  }
  double* h_residuals_pinned = nullptr;
  cudaEvent_t residuals_d2h_event = nullptr;
  bool residuals_d2h_pending = false;
  std::uint8_t* d_conv_flag = nullptr;
  std::uint8_t* h_conv_flag_pinned = nullptr;
  // Per-run lamscale, cached by ComputeForceNormsCuda for the device-side
  // normalized convergence check (k_check_convergence). Zero means "not
  // yet seen this run"; the kernel falls back to raw-residual comparison.
  double lamscale_cached = 0.0;

  void EnsureFnormScalarsBuffer() {
    if (!d_fnorm_scalars) {
      cuda_check(cudaMalloc(&d_fnorm_scalars, 2 * sizeof(double) * n_config_max),
                 "alloc d_fnorm_scalars");
    }
  }

  // Per-config control: device-resident per-cfg active mask. 1 byte per cfg.
  // Mask-aware kernels early-return when d_active_per_cfg[blockIdx.z] == 0,
  // eliminating GPU work for already-converged cfgs. nullptr until first
  // EnsureActivePerCfgBuffer() call; kernels treat nullptr as "all active"
  // (preserves single-cfg behavior).
  std::uint8_t* d_active_per_cfg = nullptr;
  std::vector<std::uint8_t> h_active_staged;  // tracks last H2D'd state
  void EnsureActivePerCfgBuffer() {
    if (!d_active_per_cfg) {
      cuda_check(cudaMalloc(&d_active_per_cfg,
                             sizeof(std::uint8_t) * n_config_max),
                 "alloc d_active_per_cfg");
      // Initialize all active (1). H2D once; thereafter only when changed.
      h_active_staged.assign(n_config_max, 1);
      cuda_check(cudaMemcpyAsync(d_active_per_cfg, h_active_staged.data(),
                                  sizeof(std::uint8_t) * n_config_max,
                                  cudaMemcpyHostToDevice, stream),
                 "h2d d_active_per_cfg (initial)");
    }
  }

  // Per-cfg restart mask: used by RestorePtsXFromBackupPerCfgCuda to gate
  // the d_pts_x backup→main copy on a per-cfg basis. nullptr until the first
  // EnsureRestartMaskBuffer() call.
  std::uint8_t* d_restart_mask = nullptr;
  void EnsureRestartMaskBuffer() {
    if (!d_restart_mask) {
      cuda_check(cudaMalloc(&d_restart_mask,
                             sizeof(std::uint8_t) * n_config_max),
                 "alloc d_restart_mask");
    }
  }

  void EnsureLambdaPrecondBuffers(int ns_h, int ns_con_local, int mpol, int ntor) {
    // Batched layout: per-config lambda preconditioner buffers.
    // Per-cfg stride = ns_con_local + 1, mirroring CPU's
    // bLambda.setZero(nsMaxF1 - nsMinF1 + 1). The +1 is a headroom slot the
    // CPU's full-grid average reads at jF = ns_con_local - 1 (which dereferences
    // bLambda[ns_con_local]). On the GPU at N>1, omitting it lets cfg=0's read
    // fall into cfg=1's slot[0], corrupting FGA's last output.
    size_t lambda_stride = (size_t)(ns_con_local + 1);
    size_t bytes_prof = sizeof(double) * (size_t)n_config_max * lambda_stride;
    auto alloc_zero_if_null = [bytes_prof](double*& p) {
      if (!p) {
        cuda_check(cudaMalloc(&p, bytes_prof), "alloc lambda profile buf");
        cuda_check(cudaMemset(p, 0, bytes_prof), "zero lambda profile buf");
      }
    };
    alloc_zero_if_null(d_bLambda);
    alloc_zero_if_null(d_dLambda);
    alloc_zero_if_null(d_cLambda);
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc lambda precond buf");
    };
    alloc_if_null(d_lambdaPreconditioner,
                  sizeof(double) * n_config_max * ns_con_local * mpol * (ntor + 1));
    (void)ns_h;
  }

  void EnsurePrecondMatrixBuffers(int ns_h, int ns_force_local, int ns_local,
                                    int nZnT) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc precond mat buf");
    };
    // Batched layout: per-config preconditioner-matrix scratch + outputs.
    // sm, sp are radial sqrt grid values (shape-constant), but we still
    // size them per-config for layout uniformity at the cost of small mem.
    alloc_if_null(d_ax_scratch, sizeof(double) * n_config_max * ns_h * 4);
    alloc_if_null(d_bx_scratch, sizeof(double) * n_config_max * ns_h * 3);
    alloc_if_null(d_cx_scratch, sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_pm_xs,     sizeof(double) * n_config_max * ns_h * nZnT);
    alloc_if_null(d_pm_xu12,   sizeof(double) * n_config_max * ns_h * nZnT);
    alloc_if_null(d_pm_xu_e,   sizeof(double) * n_config_max * ns_local * nZnT);
    alloc_if_null(d_pm_xu_o,   sizeof(double) * n_config_max * ns_local * nZnT);
    alloc_if_null(d_pm_x1_o,   sizeof(double) * n_config_max * ns_local * nZnT);
    alloc_if_null(d_pm_sm,     sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_pm_sp,     sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_pm_axm,    sizeof(double) * n_config_max * ns_h * 2);
    alloc_if_null(d_pm_axd,    sizeof(double) * n_config_max * ns_force_local * 2);
    alloc_if_null(d_pm_bxm,    sizeof(double) * n_config_max * ns_h * 2);
    alloc_if_null(d_pm_bxd,    sizeof(double) * n_config_max * ns_force_local * 2);
    alloc_if_null(d_pm_cxd,    sizeof(double) * n_config_max * ns_force_local);
    // Per-side snapshot buffers consumed by AssembleRZPreconditionerCuda.
    // Each pmat_* allocation mirrors the layout of its scratch counterpart
    // pm_* but with the configuration axis incorporated so that the R-side
    // and Z-side coefficients computed by successive
    // ComputePreconditioningMatrixCuda calls remain accessible to the
    // tridiagonal assembly downstream.
    alloc_if_null(d_pmat_arm,  sizeof(double) * n_config_max * ns_h * 2);
    alloc_if_null(d_pmat_brm,  sizeof(double) * n_config_max * ns_h * 2);
    alloc_if_null(d_pmat_ard,  sizeof(double) * n_config_max * ns_force_local * 2);
    alloc_if_null(d_pmat_brd,  sizeof(double) * n_config_max * ns_force_local * 2);
    alloc_if_null(d_pmat_azm,  sizeof(double) * n_config_max * ns_h * 2);
    alloc_if_null(d_pmat_bzm,  sizeof(double) * n_config_max * ns_h * 2);
    alloc_if_null(d_pmat_azd,  sizeof(double) * n_config_max * ns_force_local * 2);
    alloc_if_null(d_pmat_bzd,  sizeof(double) * n_config_max * ns_force_local * 2);
    alloc_if_null(d_pmat_cxd,  sizeof(double) * n_config_max * ns_force_local);
    pmat_ns_h_cached = ns_h;
    pmat_ns_force_local_cached = ns_force_local;
  }

  // Reinitialisation at the entry to a new Vmec::run: zeroes the
  // preconditioner-matrix snapshots and the RZ-preconditioner
  // outputs. The CPU build gets the equivalent reset implicitly by
  // re-constructing IdealMhdModel per Vmec instance; the persistent
  // device buffers survive instances, so the reset is explicit here.
  void ResetForNewVmecRun() {
    std::lock_guard<std::mutex> lk(mu);
    if (!stream) return;
    // Force the next vmecpp::run to H2D its host decomposed_x and
    // velocity vector into the per-cfg device buffers from scratch.
    // Without this, the per-cfg recompute path in pybind's
    // run_batched_gpu sees the batched run's stale d_pts_x slots and
    // RecomposeToPhysicalCuda produces a d_specs_block inconsistent
    // with the new Vmec's host state, which CudaForward then
    // misinterprets as a bad-Jacobian initial geometry.
    pts_x_initialized = false;
    pts_v_initialized = false;
    pts_x_backup_initialized = false;
    // Invariant-staging caches: the profiles they guard derive from the
    // input, which can differ between runs at an unchanged shape where
    // no Reshape fires to clear them. The next run restages each.
    sqrtSH_staged = false;
    massH_staged = false;
    currH_staged = false;
    phipF_staged = false;
    phipH_staged = false;
    radialBlending_staged = false;
    pm_sm_staged = false;
    pm_sp_staged = false;
    scalxc_staged = false;
    dealias_faccon_staged = false;
    iotaH_seeded = false;
    lamscale_cached = 0.0;
    auto zero_if = [this](double* p, size_t bytes) {
      if (p) cudaMemsetAsync(p, 0, bytes, stream);
    };
    // Batched layout: scale memset sizes by n_config_max so all per-config
    // slices get zeroed (not just the first).
    if (pmat_ns_h_cached > 0 && pmat_ns_force_local_cached > 0) {
      size_t half_bytes = sizeof(double) * n_config_max * pmat_ns_h_cached * 2;
      size_t full_bytes = sizeof(double) * n_config_max * pmat_ns_force_local_cached * 2;
      size_t cxd_bytes = sizeof(double) * n_config_max * pmat_ns_force_local_cached;
      zero_if(d_pmat_arm, half_bytes);
      zero_if(d_pmat_brm, half_bytes);
      zero_if(d_pmat_azm, half_bytes);
      zero_if(d_pmat_bzm, half_bytes);
      zero_if(d_pmat_ard, full_bytes);
      zero_if(d_pmat_brd, full_bytes);
      zero_if(d_pmat_azd, full_bytes);
      zero_if(d_pmat_bzd, full_bytes);
      zero_if(d_pmat_cxd, cxd_bytes);
    }
    if (rz_mnsize_cached > 0 && rz_ns_total_cached > 0) {
      size_t row_bytes = sizeof(double) * n_config_max * rz_mnsize_cached * rz_ns_total_cached;
      zero_if(d_rz_aR, row_bytes);
      zero_if(d_rz_dR, row_bytes);
      zero_if(d_rz_bR, row_bytes);
      zero_if(d_rz_aZ, row_bytes);
      zero_if(d_rz_dZ, row_bytes);
      zero_if(d_rz_bZ, row_bytes);
      if (d_rz_jMin) {
        cudaMemsetAsync(d_rz_jMin, 0, sizeof(int) * n_config_max * rz_mnsize_cached, stream);
      }
    }
    rz_cache_ar_sentinel = std::numeric_limits<double>::quiet_NaN();

    // Run-scoped staging, capture, and gate state. A second Vmec::run in
    // the same process restages everything the previous run consumed.
    specs_populated_from_device = false;
    fnorm1_staged_valid = false;
    fnorm1_device_filled = false;
    timestep_first_call_after_reset = true;
    pts_x_prev_valid = false;
    scalxc_prev_valid = false;
    std::fill(pts_x_final_taken.begin(), pts_x_final_taken.end(),
              static_cast<std::uint8_t>(0));
    if (d_conv_flag) {
      cudaMemsetAsync(d_conv_flag, 0,
                      sizeof(std::uint8_t) * n_config_max, stream);
    }
    if (h_conv_flag_pinned) {
      std::memset(h_conv_flag_pinned, 0,
                  sizeof(std::uint8_t) * n_config_max);
    }
    if (batch_inputs_pinned) {
      cudaFreeHost(batch_inputs_pinned);
      batch_inputs_pinned = nullptr;
    }
    batch_inputs_n_cfg = 0;
    batch_inputs_one_spec_doubles = 0;
    batch_inputs_loaded = -1;
    batch_inputs_consumed = false;
    auto drop_graph = [](cudaGraphExec_t& exec, cudaGraph_t& graph,
                         bool& captured) {
      if (exec) {
        cudaGraphExecDestroy(exec);
        exec = nullptr;
      }
      if (graph) {
        cudaGraphDestroy(graph);
        graph = nullptr;
      }
      captured = false;
    };
    drop_graph(seg2_graph_exec, seg2_graph, seg2_graph_captured);
    drop_graph(seg3_graph_exec, seg3_graph, seg3_graph_captured);
    drop_graph(seg4_graph_exec, seg4_graph, seg4_graph_captured);
    drop_graph(fwd_graph_exec, fwd_graph, fwd_graph_captured);
    drop_graph(iter_graph_exec, iter_graph, iter_graph_captured);
    seg2_in_capture = false;
    seg3_in_capture = false;
    seg4_in_capture = false;
    seg2_warmup_calls = 0;
    seg3_warmup_calls = 0;
    seg4_warmup_calls = 0;
    iter_graph_warmups = 0;
  }

  void EnsureConstraintMultiplierBuffers(int ns_force_local, int ns_con_local,
                                          int nZnT) {
    // Zero-fill on allocation: the CPU counterparts are setZero'd at
    // construction, and the writers skip rows outside their domain (the
    // axis row under jMin = 1, the LCFS row of the force-local arrays).
    // Without the fill those rows read allocator residue, which varies
    // with the process's prior allocation history.
    auto alloc_if_null = [this](double*& p, size_t bytes) {
      if (!p) {
        cuda_check(cudaMalloc(&p, bytes), "alloc cm buf");
        cuda_check(cudaMemsetAsync(p, 0, bytes, stream), "zero cm buf");
      }
    };
    // Batched layout: per-config constraint-multiplier buffers.
    alloc_if_null(d_arNorm, sizeof(double) * n_config_max * ns_force_local);
    alloc_if_null(d_azNorm, sizeof(double) * n_config_max * ns_force_local);
    alloc_if_null(d_tcon,   sizeof(double) * n_config_max * ns_con_local);
    alloc_if_null(d_ruFull, sizeof(double) * n_config_max * ns_con_local * nZnT);
    alloc_if_null(d_zuFull, sizeof(double) * n_config_max * ns_con_local * nZnT);
    alloc_if_null(d_ard,    sizeof(double) * n_config_max * ns_force_local * 2);
    alloc_if_null(d_azd,    sizeof(double) * n_config_max * ns_force_local * 2);
    alloc_if_null(d_gConEff, sizeof(double) * n_config_max * ns_con_local * nZnT);
    alloc_if_null(d_gCon,    sizeof(double) * n_config_max * ns_con_local * nZnT);
    alloc_if_null(d_frcon_e, sizeof(double) * n_config_max * ns_force_local * nZnT);
    alloc_if_null(d_frcon_o, sizeof(double) * n_config_max * ns_force_local * nZnT);
    alloc_if_null(d_fzcon_e, sizeof(double) * n_config_max * ns_force_local * nZnT);
    alloc_if_null(d_fzcon_o, sizeof(double) * n_config_max * ns_force_local * nZnT);
  }

  // Decomposed FourierForces shadow + scalxc.
  // Batched layout: per-config decomposed force spec arrays.
  void EnsureDecomposedForcesBuffers(int ns_dec_local, int mpol, int ntor) {
    int size = n_config_max * ns_dec_local * mpol * (ntor + 1);
    if (decomposed_size_cached == size && d_decomposed_frcc) return;
    auto cuda_free_if = [](void*& p) {
      if (p) { cudaFree(p); p = nullptr; }
    };
    cuda_free_if((void*&)d_decomposed_frcc);
    cuda_free_if((void*&)d_decomposed_frss);
    cuda_free_if((void*&)d_decomposed_fzsc);
    cuda_free_if((void*&)d_decomposed_fzcs);
    cuda_free_if((void*&)d_decomposed_flsc);
    cuda_free_if((void*&)d_decomposed_flcs);
    size_t bytes = sizeof(double) * size;
    cuda_check(cudaMalloc(&d_decomposed_frcc, bytes), "alloc d_decomposed_frcc");
    cuda_check(cudaMalloc(&d_decomposed_frss, bytes), "alloc d_decomposed_frss");
    cuda_check(cudaMalloc(&d_decomposed_fzsc, bytes), "alloc d_decomposed_fzsc");
    cuda_check(cudaMalloc(&d_decomposed_fzcs, bytes), "alloc d_decomposed_fzcs");
    cuda_check(cudaMalloc(&d_decomposed_flsc, bytes), "alloc d_decomposed_flsc");
    cuda_check(cudaMalloc(&d_decomposed_flcs, bytes), "alloc d_decomposed_flcs");
    // Zero-initialize the freshly allocated decomposed-force buffers
    // so that configuration slots whose corresponding force-chain
    // kernels do not write them present a deterministic zero rather
    // than an uninitialized bit pattern that may decode as a
    // signaling NaN.
    cuda_check(cudaMemsetAsync(d_decomposed_frcc, 0, bytes, stream), "memset dec_frcc");
    cuda_check(cudaMemsetAsync(d_decomposed_frss, 0, bytes, stream), "memset dec_frss");
    cuda_check(cudaMemsetAsync(d_decomposed_fzsc, 0, bytes, stream), "memset dec_fzsc");
    cuda_check(cudaMemsetAsync(d_decomposed_fzcs, 0, bytes, stream), "memset dec_fzcs");
    cuda_check(cudaMemsetAsync(d_decomposed_flsc, 0, bytes, stream), "memset dec_flsc");
    cuda_check(cudaMemsetAsync(d_decomposed_flcs, 0, bytes, stream), "memset dec_flcs");
    decomposed_size_cached = size;
  }

  void EnsureScalxcBuffer(int len) {
    if (d_scalxc) return;
    // Batched layout: per-config scalxc scaling factors.
    cuda_check(cudaMalloc(&d_scalxc, sizeof(double) * n_config_max * len), "alloc d_scalxc");
  }

  void EnsurePTSBuffers(int ns_con_local, int ns_local, int mpol, int ntor) {
    int v_size = ns_con_local * mpol * (ntor + 1);
    int x_size = ns_local * mpol * (ntor + 1);
    if (d_pts_v_rcc && pts_v_size == v_size && pts_x_size == x_size) return;
    // Multigrid stage transition snapshot: if d_pts_x is currently allocated,
    // initialized, AND its size is changing (ns went up), copy each cfg's
    // d_pts_x slice into d_pts_x_prev so the per-cfg radial-interp kernel
    // in PerformTimeStepCuda's init branch can upscale per cfg into the
    // newly-allocated d_pts_x. Without this, the host m_decomposed_x
    // broadcast at the new ns washes out per-cfg state for cfg != 0.
    const int upscale_env =
        RunEnvFlag(&g_batch_upscale_env, "VMECPP_BATCH_MULTIGRID_UPSCALE");
    bool snapshot = (upscale_env > 0 && pts_x_initialized && d_pts_x_rcc &&
                      pts_x_size > 0 && pts_x_size != x_size &&
                      pts_x_ns > 0);
    if (snapshot) {
      auto realloc_prev = [&](double*& p) {
        if (p) cudaFree(p);
        size_t bytes_prev =
            sizeof(double) * (size_t)n_config_max * pts_x_size;
        cuda_check(cudaMalloc(&p, bytes_prev), "alloc d_pts_x_prev");
      };
      realloc_prev(d_pts_x_prev_rcc); realloc_prev(d_pts_x_prev_rss);
      realloc_prev(d_pts_x_prev_zsc); realloc_prev(d_pts_x_prev_zcs);
      realloc_prev(d_pts_x_prev_lsc); realloc_prev(d_pts_x_prev_lcs);
      size_t bytes_prev =
          sizeof(double) * (size_t)n_config_max * pts_x_size;
      double* src[6] = {d_pts_x_rcc, d_pts_x_rss, d_pts_x_zsc,
                        d_pts_x_zcs, d_pts_x_lsc, d_pts_x_lcs};
      double* dst[6] = {d_pts_x_prev_rcc, d_pts_x_prev_rss, d_pts_x_prev_zsc,
                        d_pts_x_prev_zcs, d_pts_x_prev_lsc, d_pts_x_prev_lcs};
      for (int i = 0; i < 6; ++i) {
        cuda_check(cudaMemcpyAsync(dst[i], src[i], bytes_prev,
                                    cudaMemcpyDeviceToDevice, stream),
                   "d2d pts_x → pts_x_prev (multigrid snapshot)");
      }
      pts_x_prev_size = pts_x_size;
      pts_x_prev_ns = pts_x_ns;
      pts_x_prev_mpol = mpol;
      pts_x_prev_ntor = ntor;
      pts_x_prev_valid = true;
      std::fprintf(stderr,
          "[fft_toroidal_cuda] multigrid snapshot: ns %d → %d (mpol=%d ntor=%d "
          "n_cfg=%d), %zu bytes/spec\n",
          pts_x_ns, ns_local, mpol, ntor, n_config_max, bytes_prev);
    }
    auto free_if = [](double*& p) { if (p) { cudaFree(p); p = nullptr; } };
    free_if(d_pts_v_rcc); free_if(d_pts_v_rss);
    free_if(d_pts_v_zsc); free_if(d_pts_v_zcs);
    free_if(d_pts_v_lsc); free_if(d_pts_v_lcs);
    free_if(d_pts_x_rcc); free_if(d_pts_x_rss);
    free_if(d_pts_x_zsc); free_if(d_pts_x_zcs);
    free_if(d_pts_x_lsc); free_if(d_pts_x_lcs);
    // Multigrid-stage transition: pts_x shape changed, so the backup buffers
    // (sized at pts_x_size) must also be freed and re-lazied. Without this,
    // BackupPtsXCuda would memcpy NEW size into OLD-sized buffer.
    free_if(d_pts_x_backup_rcc); free_if(d_pts_x_backup_rss);
    free_if(d_pts_x_backup_zsc); free_if(d_pts_x_backup_zcs);
    free_if(d_pts_x_backup_lsc); free_if(d_pts_x_backup_lcs);
    pts_x_backup_initialized = false;
    free_if(d_pts_x_final_rcc); free_if(d_pts_x_final_rss);
    free_if(d_pts_x_final_zsc); free_if(d_pts_x_final_zcs);
    free_if(d_pts_x_final_lsc); free_if(d_pts_x_final_lcs);
    pts_x_final_taken.clear();
    size_t v_bytes = sizeof(double) * (size_t)n_config_max * v_size;
    size_t x_bytes = sizeof(double) * (size_t)n_config_max * x_size;
    cuda_check(cudaMalloc(&d_pts_v_rcc, v_bytes), "alloc d_pts_v_rcc");
    cuda_check(cudaMalloc(&d_pts_v_rss, v_bytes), "alloc d_pts_v_rss");
    cuda_check(cudaMalloc(&d_pts_v_zsc, v_bytes), "alloc d_pts_v_zsc");
    cuda_check(cudaMalloc(&d_pts_v_zcs, v_bytes), "alloc d_pts_v_zcs");
    cuda_check(cudaMalloc(&d_pts_v_lsc, v_bytes), "alloc d_pts_v_lsc");
    cuda_check(cudaMalloc(&d_pts_v_lcs, v_bytes), "alloc d_pts_v_lcs");
    cuda_check(cudaMalloc(&d_pts_x_rcc, x_bytes), "alloc d_pts_x_rcc");
    cuda_check(cudaMalloc(&d_pts_x_rss, x_bytes), "alloc d_pts_x_rss");
    cuda_check(cudaMalloc(&d_pts_x_zsc, x_bytes), "alloc d_pts_x_zsc");
    cuda_check(cudaMalloc(&d_pts_x_zcs, x_bytes), "alloc d_pts_x_zcs");
    cuda_check(cudaMalloc(&d_pts_x_lsc, x_bytes), "alloc d_pts_x_lsc");
    cuda_check(cudaMalloc(&d_pts_x_lcs, x_bytes), "alloc d_pts_x_lcs");
    pts_v_size = v_size;
    pts_x_size = x_size;
    pts_x_ns = ns_local;
    pts_v_initialized = false;  // will be cudaMemset to 0 on first call
    pts_x_initialized = false;  // will be H2D from host m_decomposed_x on first call
    timestep_first_call_after_reset = true;
  }

  // Persistent preconditioner-input buffers.
  void EnsureM1InputBuffers(int ns_force_local) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc m1 input");
    };
    // Batched layout: per-config M1 preconditioner-input coefficients.
    size_t coef_bytes = sizeof(double) * n_config_max * ns_force_local * 2;
    alloc_if_null(d_m1_ard, coef_bytes);
    alloc_if_null(d_m1_brd, coef_bytes);
    alloc_if_null(d_m1_azd, coef_bytes);
    alloc_if_null(d_m1_bzd, coef_bytes);
  }
  void EnsureLambdaInputBuffer(int ns_con_local, int mpol, int ntor) {
    if (d_lambda_lp) return;
    // Batched layout: per-config lambda preconditioner spec array.
    size_t spec_bytes = sizeof(double) * n_config_max * ns_con_local * mpol * (ntor + 1);
    cuda_check(cudaMalloc(&d_lambda_lp, spec_bytes), "alloc lambda lp");
  }
  void EnsureRZBuffers(int mnsize, int ns_total, int num_basis) {
    if (rz_mnsize_cached == mnsize && rz_ns_total_cached == ns_total &&
        rz_num_basis_cached == num_basis && d_rz_aR) {
      return;
    }
    auto cuda_free_if = [](void*& p) {
      if (p) { cudaFree(p); p = nullptr; }
    };
    cuda_free_if((void*&)d_rz_aR);
    cuda_free_if((void*&)d_rz_dR);
    cuda_free_if((void*&)d_rz_bR);
    cuda_free_if((void*&)d_rz_cR);
    cuda_free_if((void*&)d_rz_aZ);
    cuda_free_if((void*&)d_rz_dZ);
    cuda_free_if((void*&)d_rz_bZ);
    cuda_free_if((void*&)d_rz_cZ);
    cuda_free_if((void*&)d_rz_c_orig_R);
    cuda_free_if((void*&)d_rz_c_orig_Z);
    cuda_free_if((void*&)d_rz_x_saved_R);
    cuda_free_if((void*&)d_rz_x_saved_Z);
    if (d_rz_jMin) { cudaFree(d_rz_jMin); d_rz_jMin = nullptr; }
    // Batched layout: per-config RZ-preconditioner matrices and RHS.
    size_t row_bytes = sizeof(double) * n_config_max * mnsize * ns_total;
    size_t c_bytes = sizeof(double) * n_config_max * mnsize * num_basis * ns_total;
    cuda_check(cudaMalloc(&d_rz_aR, row_bytes), "alloc rz aR");
    cuda_check(cudaMalloc(&d_rz_dR, row_bytes), "alloc rz dR");
    cuda_check(cudaMalloc(&d_rz_bR, row_bytes), "alloc rz bR");
    cuda_check(cudaMalloc(&d_rz_cR, c_bytes),   "alloc rz cR");
    cuda_check(cudaMalloc(&d_rz_aZ, row_bytes), "alloc rz aZ");
    cuda_check(cudaMalloc(&d_rz_dZ, row_bytes), "alloc rz dZ");
    cuda_check(cudaMalloc(&d_rz_bZ, row_bytes), "alloc rz bZ");
    cuda_check(cudaMalloc(&d_rz_cZ, c_bytes),   "alloc rz cZ");
    cuda_check(cudaMalloc(&d_rz_jMin, sizeof(int) * n_config_max * mnsize), "alloc rz jMin");
    // Carson-Higham IR scratch buffers, allocated only when the
    // VMECPP_RZ_IR_FP32 toggle is active. The four-buffer footprint is
    // 4 * c_bytes, which at the canonical production shape is roughly
    // 32 MiB total at single-cfg execution and scales linearly with
    // n_config_max. Allocating them unconditionally would waste VRAM
    // under the default FP64 PCR path; the env check defers the cost.
    {
      const char* e = std::getenv("VMECPP_RZ_IR_FP32");
      const bool ir_enabled = (e != nullptr && std::atoi(e) > 0);
      if (ir_enabled) {
        cuda_check(cudaMalloc(&d_rz_c_orig_R, c_bytes),  "alloc rz c_orig_R");
        cuda_check(cudaMalloc(&d_rz_c_orig_Z, c_bytes),  "alloc rz c_orig_Z");
        cuda_check(cudaMalloc(&d_rz_x_saved_R, c_bytes), "alloc rz x_saved_R");
        cuda_check(cudaMalloc(&d_rz_x_saved_Z, c_bytes), "alloc rz x_saved_Z");
      }
    }
    rz_mnsize_cached = mnsize;
    rz_ns_total_cached = ns_total;
    rz_num_basis_cached = num_basis;
    rz_cache_ar_sentinel = std::numeric_limits<double>::quiet_NaN();
  }

  // deAliasConstraintForce buffers. gsc/gcs/faccon are per-call sizes but
  // allocated once. cosnv/sinnv are staged by StageToroidalBasis at Reshape
  // time; no per-call check needed here.
  void EnsureDealiasBuffers(int mpol, int ntor, int ns_force_local) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc dealias buf");
    };
    // Batched layout: per-config dealias gsc/gcs spec arrays + per-config faccon.
    size_t gsc_bytes = sizeof(double) * n_config_max * ns_force_local * mpol * (ntor + 1);
    alloc_if_null(d_dealias_gsc,    gsc_bytes);
    alloc_if_null(d_dealias_gcs,    gsc_bytes);
    alloc_if_null(d_dealias_faccon, sizeof(double) * n_config_max * mpol);
  }

  void EnsureMHDForceBuffers(int ns_force_local, int nZnT, bool lthreed) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc mhd force buf");
    };
    // Batched layout: per-config MHD force outputs.
    size_t b = sizeof(double) * n_config_max * ns_force_local * nZnT;
    alloc_if_null(d_armn_e, b); alloc_if_null(d_armn_o, b);
    alloc_if_null(d_azmn_e, b); alloc_if_null(d_azmn_o, b);
    alloc_if_null(d_brmn_e, b); alloc_if_null(d_brmn_o, b);
    alloc_if_null(d_bzmn_e, b); alloc_if_null(d_bzmn_o, b);
    if (lthreed) {
      alloc_if_null(d_crmn_e, b); alloc_if_null(d_crmn_o, b);
      alloc_if_null(d_czmn_e, b); alloc_if_null(d_czmn_o, b);
    }
  }

  void EnsureForceNormBuffers(int ns_h) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc fnorm buf");
    };
    // Batched layout: per-config force-norm partials.
    alloc_if_null(d_forceNormRZ_partial, sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_forceNormL_partial,  sizeof(double) * n_config_max * ns_h);
  }

  void EnsureHybridLambdaBuffers(int ns_local, int ns_con_local, int nZnT) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc hlf buf");
    };
    // Batched layout: per-config radialBlending + hybrid lambda force outputs.
    alloc_if_null(d_radialBlending, sizeof(double) * n_config_max * ns_local);
    alloc_if_null(d_blmn_e, sizeof(double) * n_config_max * ns_con_local * nZnT);
    alloc_if_null(d_blmn_o, sizeof(double) * n_config_max * ns_con_local * nZnT);
    alloc_if_null(d_clmn_e, sizeof(double) * n_config_max * ns_con_local * nZnT);
    alloc_if_null(d_clmn_o, sizeof(double) * n_config_max * ns_con_local * nZnT);
  }

  void EnsurePressureBuffers(int ns_h, int nZnT) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc pressure buf");
    };
    // Batched layout: per-config pressure / totalPressure / energy partials.
    if (!d_presH) {
      cuda_check(cudaMalloc(&d_presH, sizeof(double) * n_config_max * ns_h), "alloc d_presH (pres)");
    }
    alloc_if_null(d_massH,            sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_totalPressure,    sizeof(double) * n_config_max * ns_h * nZnT);
    alloc_if_null(d_thermal_partial,  sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_magnetic_partial, sizeof(double) * n_config_max * ns_h);
  }

  // computeBContra buffers.
  void EnsureBContraBuffers(int ns_h, int nZnT, int ns_local) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc bcontra buf");
    };
    // Batched layout: per-config BContra radial profiles + 2D fields.
    alloc_if_null(d_bsupu,         sizeof(double) * n_config_max * ns_h * nZnT);
    alloc_if_null(d_bsupv,         sizeof(double) * n_config_max * ns_h * nZnT);
    alloc_if_null(d_chipH,         sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_iotaH,         sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_iotaF,         sizeof(double) * n_config_max * ns_local);
    alloc_if_null(d_phipH,         sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_currH,         sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_iotaH_in,      sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_jvPlasma,      sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_avg_guu_gsqrt, sizeof(double) * n_config_max * ns_h);
    if (!d_phipF) {
      cuda_check(cudaMalloc(&d_phipF, sizeof(double) * n_config_max * ns_local), "alloc d_phipF (bcontra)");
    }
    if (!d_chipF) {
      cuda_check(cudaMalloc(&d_chipF, sizeof(double) * n_config_max * ns_local), "alloc d_chipF (bcontra)");
    }
  }

  // rzConIntoVolume buffers (ns_con_local × nZnT each for rCon0, zCon0).
  void EnsureRzCon0Buffers(int ns_con_local, int nZnT) {
    if (rzcon0_ns_con_cached == ns_con_local && rzcon0_nZnT_cached == nZnT &&
        d_rCon0 != nullptr) {
      return;
    }
    auto cuda_free_if = [](void*& p) {
      if (p) { cudaFree(p); p = nullptr; }
    };
    cuda_free_if((void*&)d_rCon0);
    cuda_free_if((void*&)d_zCon0);
    // Batched layout: per-config rCon0/zCon0. Fresh buffers start at
    // zero, matching the host arrays at a multigrid stage where
    // rzConIntoVolume does not run (the vacuum pressure is already
    // active there); RzConIntoVolumeCuda overwrites them in full when
    // it does run.
    size_t bytes = sizeof(double) * n_config_max * ns_con_local * nZnT;
    cuda_check(cudaMalloc(&d_rCon0, bytes), "alloc d_rCon0");
    cuda_check(cudaMalloc(&d_zCon0, bytes), "alloc d_zCon0");
    cuda_check(cudaMemsetAsync(d_rCon0, 0, bytes, stream), "zero d_rCon0");
    cuda_check(cudaMemsetAsync(d_zCon0, 0, bytes, stream), "zero d_zCon0");
    rzcon0_ns_con_cached = ns_con_local;
    rzcon0_nZnT_cached = nZnT;
  }

  // radialForceBalance buffers (half-grid ns_h scalars + interior ns_fi scalars).
  void EnsureRadialForceBalanceBuffers(int ns_h, int nsi, int ns_local) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc radial fb buf");
    };
    // Batched layout: per-config radial half-grid + interior scalars + full-grid profiles.
    alloc_if_null(d_bucoH,    sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_bvcoH,    sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_presH,    sizeof(double) * n_config_max * ns_h);
    alloc_if_null(d_jcurvF,   sizeof(double) * n_config_max * nsi);
    alloc_if_null(d_jcuruF,   sizeof(double) * n_config_max * nsi);
    alloc_if_null(d_presgradF,sizeof(double) * n_config_max * nsi);
    alloc_if_null(d_dVdsF,    sizeof(double) * n_config_max * nsi);
    alloc_if_null(d_equiF,    sizeof(double) * n_config_max * nsi);
    alloc_if_null(d_chipF,    sizeof(double) * n_config_max * ns_local);
    alloc_if_null(d_phipF,    sizeof(double) * n_config_max * ns_local);
  }

  void OneTimeInit(int n, int nfp, int mpol) {
    std::lock_guard<std::mutex> lk(mu);
    if (initialized) return;
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count == 0) {
      throw std::runtime_error("[fft_toroidal_cuda] no CUDA device");
    }
    // VMECPP_CUDA_DEVICE selects the device ordinal for this process;
    // unset or invalid values fall back to device 0. Per-device batches
    // across multiple GPUs run one process per device.
    int device_index = 0;
    if (const char* e = std::getenv("VMECPP_CUDA_DEVICE")) {
      const int requested = std::atoi(e);
      if (requested >= 0 && requested < device_count) {
        device_index = requested;
      } else {
        std::fprintf(stderr,
                     "[fft_toroidal_cuda] VMECPP_CUDA_DEVICE=%s out of range "
                     "(%d device(s)); using device 0\n",
                     e, device_count);
      }
    }
    cuda_check(cudaSetDevice(device_index), "cudaSetDevice");
    cudaDeviceProp prop;
    cuda_check(cudaGetDeviceProperties(&prop, device_index),
               "cudaGetDeviceProperties");
    std::fprintf(stderr,
                 "[fft_toroidal_cuda] using device %d: %s (sm_%d%d), "
                 "real-kernel forward path active\n",
                 device_index, prop.name, prop.major, prop.minor);
    n_cached = n;
    nfp_cached = nfp;
    mpol_cached = mpol;
    initialized = true;
  }
};

CudaToroidalState& State();
// Backup-mirror arming helper (defined in fft_toroidal_cuda_io.cu).
void EnsurePTSBackupBuffers(CudaToroidalState& S);

// Device helpers shared with the template kernels below.
// Element generators shared by the int8-Ozaki scatter passes. A is the
// combined poloidal basis [l, 4m + bf]; B is the signed, parity-masked
// spec [4m + bf, channel] in the same channel layout as the wmma tile.
__device__ __forceinline__ double i8oz_a_elem(
    int l, int kk, int mpol, int nThetaReduced, const double* s_cmu,
    const double* s_smu, const double* s_cmum, const double* s_smum) {
  if (l >= nThetaReduced || kk >= 4 * mpol) return 0.0;
  int m  = kk >> 2;
  int bf = kk & 3;
  int bml = m * nThetaReduced + l;
  switch (bf) {
    case 0: return s_cmu[bml];
    case 1: return s_smu[bml];
    case 2: return s_cmum[bml];
    default: return s_smum[bml];
  }
}
__device__ __forceinline__ double i8oz_b_elem(int kk, int n, int mpol,
                                              const double* s_Y) {
  if (kk >= 4 * mpol) return 0.0;
  int m  = kk >> 2;
  int bf = kk & 3;
  bool m_even = ((m & 1) == 0);
  bool parity_match = (n & 1) ? !m_even : m_even;
  if (!parity_match) return 0.0;
  int yb = m * kBatch;
  switch (n >> 1) {
    case 0:
      if (bf == 0) return s_Y[yb + kRmkcc];
      if (bf == 1) return s_Y[yb + kRmkss];
      return 0.0;
    case 1:
      if (bf == 2) return s_Y[yb + kRmkss];
      if (bf == 3) return s_Y[yb + kRmkcc];
      return 0.0;
    case 2:
      if (bf == 0) return s_Y[yb + kRmkccN];
      if (bf == 1) return s_Y[yb + kRmkssN];
      return 0.0;
    case 3:
      if (bf == 0) return s_Y[yb + kZmkcs];
      if (bf == 1) return s_Y[yb + kZmksc];
      return 0.0;
    case 4:
      if (bf == 2) return s_Y[yb + kZmksc];
      if (bf == 3) return s_Y[yb + kZmkcs];
      return 0.0;
    case 5:
      if (bf == 0) return s_Y[yb + kZmkcsN];
      if (bf == 1) return s_Y[yb + kZmkscN];
      return 0.0;
    case 6:
      if (bf == 2) return s_Y[yb + kLmksc];
      if (bf == 3) return s_Y[yb + kLmkcs];
      return 0.0;
    default:
      if (bf == 0) return -s_Y[yb + kLmkcsN];
      if (bf == 1) return -s_Y[yb + kLmkscN];
      return 0.0;
  }
}

// Kernel declarations (definitions in fft_toroidal_cuda_kernels.cu).
__global__ void k_fill_spectra(
    int n_config, int ns_local, int mpol, int ntor, int nhalf, int nfp,
    int nsMinF1_offset,
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ nscale, cufftDoubleComplex* __restrict__ X);
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
    double* __restrict__ lv_e, double* __restrict__ lv_o);
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
    double* __restrict__ rCon, double* __restrict__ zCon);
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
    double* __restrict__ rCon);
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
    double* __restrict__ zCon);
__global__ void k_fwd_fused_L(
    int n_config, int ns_local, int mpol, int ntor, int nfp,
    int nZeta, int nThetaReduced, int nThetaEff, int nsMinF1,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ dft_cos, const double* __restrict__ dft_sin,
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o);
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
    double* __restrict__ rCon, double* __restrict__ zCon);
__global__ void k_inverse_dft_24_radix83(
    int total_batches, int nhalf, int nZeta,
    const cufftDoubleComplex* __restrict__ X,
    double* __restrict__ Y);
__global__ void k_forward_dft_24_radix83(
    int total_batches, int nZeta, int nhalf,
    const double* __restrict__ Y,
    cufftDoubleComplex* __restrict__ X);
__global__ void k_cast_complex_fp64_to_fp32(
    size_t n, const cufftDoubleComplex* __restrict__ src,
    cufftComplex* __restrict__ dst);
__global__ void k_cast_fp32_to_fp64(
    size_t n, const float* __restrict__ src, double* __restrict__ dst);
__global__ void k_inverse_dft_24(
    int total_batches, int nhalf, int nZeta,
    const cufftDoubleComplex* __restrict__ X,
    const double* __restrict__ cos_table,
    const double* __restrict__ sin_table,
    double* __restrict__ Y);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
    double* __restrict__ rCon, double* __restrict__ zCon);
__global__ void k_scatter_con(
    int n_config, int ns_local, int ns_con_local,
    int mpol, int nZeta, int nThetaReduced, int nThetaEff,
    int nsMinF_offset_in_local,  // jF_local index of nsMinF in the larger range
    const double* __restrict__ Y, const double* __restrict__ cosmu,
    const double* __restrict__ sinmu, const double* __restrict__ xmpq,
    const double* __restrict__ sqrtSF,
    double* __restrict__ rCon, double* __restrict__ zCon);
__global__ void k_extrapolate_towards_axis(
    int n_config, int ns_local, int mpol, int ntor, bool lthreed,
    double* __restrict__ rmncc, double* __restrict__ rmnss,
    double* __restrict__ zmnsc, double* __restrict__ zmncs,
    double* __restrict__ lmnsc, double* __restrict__ lmncs);
__global__ void k_scale_prev_by_scalxc(
    int n_config, int ns_old, int mpol, int ntor,
    int scalxc_old_len_per_cfg,
    double* prev_rcc, double* prev_rss,
    double* prev_zsc, double* prev_zcs,
    double* prev_lsc, double* prev_lcs,
    const double* __restrict__ scalxc_old);
__global__ void k_axis_extrapolate_odd_m_prev(
    int n_config, int ns_old, int mpol, int ntor,
    double* prev_rcc, double* prev_rss,
    double* prev_zsc, double* prev_zcs,
    double* prev_lsc, double* prev_lcs);
__global__ void k_radial_interpolate_pts_x(
    int n_config, int ns_old, int ns_new, int mpol, int ntor,
    int scalxc_len_per_cfg,
    const double* __restrict__ old_rcc, const double* __restrict__ old_rss,
    const double* __restrict__ old_zsc, const double* __restrict__ old_zcs,
    const double* __restrict__ old_lsc, const double* __restrict__ old_lcs,
    double* __restrict__ new_rcc, double* __restrict__ new_rss,
    double* __restrict__ new_zsc, double* __restrict__ new_zcs,
    double* __restrict__ new_lsc, double* __restrict__ new_lcs,
    const double* __restrict__ scalxc);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_extract_geom_scalars(
    const double* __restrict__ d_r1_e,
    const double* __restrict__ d_r1_o,
    const double* __restrict__ d_z1_e,
    int outer_idx, int inner_idx, double* __restrict__ d_out);
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
    double* __restrict__ zs, double* __restrict__ tau);
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
    double* __restrict__ guv, double* __restrict__ gvv);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
    double* __restrict__ dVdsH);
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
    double* __restrict__ dVdsH);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_update_dvdsh(int n_config, int ns_h, int nZnT, int nThetaEff,
                                double signOfJacobian,
                                const double* __restrict__ gsqrt,
                                const double* __restrict__ wInt,
                                double* __restrict__ dVdsH);
__global__ void k_recompute_dvdsh_if_zeroed(
    int n_config, int ns_h, int nZnT, int nThetaEff, double signOfJacobian,
    const double* __restrict__ gsqrt, const double* __restrict__ wInt,
    double* __restrict__ dVdsH);
__global__ void k_buco_bvco(int n_config, int ns_h, int nZnT, int nThetaEff,
                              const double* __restrict__ bsubu,
                              const double* __restrict__ bsubv,
                              const double* __restrict__ wInt,
                              double* __restrict__ bucoH,
                              double* __restrict__ bvcoH);
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
                                    double* __restrict__ equiF);
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
                                       double* __restrict__ cx_scratch);
__global__ void k_pm_assemble_half(int n_config, int ns_h,
                                     int kEven, int kOdd,
                                     const double* __restrict__ ax,
                                     const double* __restrict__ bx,
                                     const double* __restrict__ sm,
                                     const double* __restrict__ sp,
                                     double* __restrict__ m_axm,
                                     double* __restrict__ m_bxm);
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
                                     double* __restrict__ m_cxd);
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
                                       double* __restrict__ cLambda);
__global__ void k_ulp_axis_extrap(int n_config, int lambda_stride, int axis_present,
                                    double* __restrict__ bLambda,
                                    double* __restrict__ dLambda,
                                    double* __restrict__ cLambda);
__global__ void k_ulp_full_grid_average(int n_config, int lambda_stride,
                                          int ns_con_local, int jMin,
                                          int nsMinH_offset,  // nsMinF - nsMinH
                                          const double* __restrict__ bLambda_in,
                                          const double* __restrict__ dLambda_in,
                                          const double* __restrict__ cLambda_in,
                                          double* __restrict__ bLambda_out,
                                          double* __restrict__ dLambda_out,
                                          double* __restrict__ cLambda_out);
__global__ void k_ulp_assemble(int n_config, int ns_con_local, int lambda_stride,
                                int jMin,
                                int mpol, int ntor,
                                int nfp, double pFactor,
                                const double* __restrict__ bLambda,
                                const double* __restrict__ dLambda,
                                const double* __restrict__ cLambda,
                                const double* __restrict__ sqrtSF,
                                int sqrtSF_off,  // nsMinF - nsMinF1
                                double* __restrict__ lambdaPreconditioner);
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
    double* __restrict__ czmn_e, double* __restrict__ czmn_o);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
                                        const std::uint8_t* __restrict__ d_active_per_cfg);
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
    double* __restrict__ clmn_e, double* __restrict__ clmn_o);
__global__ void k_pres_compute(int n_config, int ns_h, double gamma,
                                const double* __restrict__ massH,
                                const double* __restrict__ dVdsH,
                                double* __restrict__ presH);
__global__ void k_pres_compute_and_thermal(int n_config, int ns_h, double gamma,
                                            int nsMinH, int nsMaxH_minus_1,
                                            int ns_minus_2,
                                            const double* __restrict__ massH,
                                            const double* __restrict__ dVdsH,
                                            double* __restrict__ presH,
                                            double* __restrict__ thermal_partial);
__global__ void k_pres_totalpres_init(int n_config, int ns_h, int nZnT,
                                       const double* __restrict__ bsupu,
                                       const double* __restrict__ bsubu,
                                       const double* __restrict__ bsupv,
                                       const double* __restrict__ bsubv,
                                       double* __restrict__ totalPressure);
__global__ void k_pres_thermal_partial(int n_config, int ns_h,
                                        int nsMinH, int nsMaxH_minus_1,
                                        int ns_minus_2,
                                        const double* __restrict__ presH,
                                        const double* __restrict__ dVdsH,
                                        double* __restrict__ thermal_partial,
                                        const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_pres_magnetic_partial(int n_config, int ns_h, int nZnT, int nThetaEff,
                                         int nsMinH, int nsMaxH_minus_1,
                                         int ns_minus_2,
                                         const double* __restrict__ gsqrt,
                                         const double* __restrict__ totalPressure,
                                         const double* __restrict__ wInt,
                                         double* __restrict__ magnetic_partial);
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
                                                const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_pres_totalpres_init_with_presH(int n_config, int ns_h, int nZnT,
                                                   const double* __restrict__ bsupu,
                                                   const double* __restrict__ bsubu,
                                                   const double* __restrict__ bsupv,
                                                   const double* __restrict__ bsubv,
                                                   const double* __restrict__ presH,
                                                   double* __restrict__ totalPressure,
                                                   const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_pres_add_presH(int n_config, int ns_h, int nZnT,
                                  const double* __restrict__ presH,
                                  double* __restrict__ totalPressure);
__global__ void k_volume_reduce(int n_config, int ns_h, double multiplier,
                                  int nsMaxH_minus_1, int ns_minus_2,
                                  int nsMinH,
                                  const double* __restrict__ dVdsH,
                                  double* __restrict__ out_scalar);
__global__ void k_bcontra_mutate_lambda(
    int n_config, int ns_local,
    int jF_first, int jF_last_excl, int nZnT, int phipF_jOff,
    bool lthreed, double lamscale,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o,
    const double* __restrict__ phipF);
__global__ void k_bcontra_bsupuv(
    int n_config, int ns_local, int ns_h,
    int jF_in_offset, int nZnT, bool lthreed,
    const double* __restrict__ lu_e, const double* __restrict__ lu_o,
    const double* __restrict__ lv_e, const double* __restrict__ lv_o,
    const double* __restrict__ sqrtSH, const double* __restrict__ gsqrt,
    double* __restrict__ bsupu, double* __restrict__ bsupv);
__global__ void k_bcontra_jvplasma_reduce(
    int n_config, int ns_h, int nZnT, int nThetaEff, bool lthreed,
    int serial_order,
    const double* __restrict__ guu, const double* __restrict__ guv,
    const double* __restrict__ bsupu, const double* __restrict__ bsupv,
    const double* __restrict__ gsqrt, const double* __restrict__ wInt,
    double* __restrict__ jvPlasma, double* __restrict__ avg_guu_gsqrt);
__global__ void k_bcontra_chipH_iotaH(
    int n_config, int ns_h, int ncurr,
    const double* __restrict__ phipH, const double* __restrict__ currH,
    const double* __restrict__ iotaH_in, const double* __restrict__ jvPlasma,
    const double* __restrict__ avg_guu_gsqrt,
    double* __restrict__ chipH, double* __restrict__ iotaH);
__global__ void k_bcontra_chipF_iotaF(
    int n_config, int ns_h, int ns_local, int nsMinFi_off, int nsMaxFi_off,
    int axis_present, int lcfs_present, int last_jF_local,
    int last_jH_local,
    const double* __restrict__ chipH, const double* __restrict__ iotaH,
    double* __restrict__ chipF, double* __restrict__ iotaF);
__global__ void k_bcontra_bsupu_add_chip(
    int n_config, int ns_h, int nZnT,
    const double* __restrict__ chipH, const double* __restrict__ gsqrt,
    double* __restrict__ bsupu);
__global__ void k_rzcon_into_volume(
    int n_config, int ns_con_local, int nZnT, int jMin_con, int lcfs_con_local,
    int nsMinF_minus_nsMinF1,
    const double* __restrict__ rCon, const double* __restrict__ zCon,
    const double* __restrict__ sqrtSF,
    double* __restrict__ rCon0, double* __restrict__ zCon0);
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
    double* __restrict__ Y);
__global__ void k_inverse_scatter(
    int n_config, int ns_local, int mpol, int ntor, int nhalf, int nfp, int nZeta,
    bool lthreed, int nsMinF1_offset,
    int jMaxRZ_local, int jMinL_local,
    const cufftDoubleComplex* __restrict__ X,
    const double* __restrict__ nscale,
    double* __restrict__ frcc, double* __restrict__ frss,
    double* __restrict__ fzsc, double* __restrict__ fzcs,
    double* __restrict__ flsc, double* __restrict__ flcs);
__global__ void k_compute_ru_zu_full(int n_config, int ns_local,
                                      int ns_con_local, int nZnT,
                                      int nsMinF_to_nsMinF1,
                                      const double* __restrict__ ru_e,
                                      const double* __restrict__ ru_o,
                                      const double* __restrict__ zu_e,
                                      const double* __restrict__ zu_o,
                                      const double* __restrict__ sqrtSF,
                                      double* __restrict__ ruFull,
                                      double* __restrict__ zuFull);
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
                                                double* __restrict__ tcon);
__global__ void k_halve_tcon_lcfs(int n_config, int tcon_stride,
                                    int last_idx,
                                    double* __restrict__ tcon);
__global__ void k_effective_constraint_force(int n_config, int ns_con_local,
                                              int nZnT, int jMin,
                                              const double* __restrict__ rCon,
                                              const double* __restrict__ rCon0,
                                              const double* __restrict__ zCon,
                                              const double* __restrict__ zCon0,
                                              const double* __restrict__ ruFull,
                                              const double* __restrict__ zuFull,
                                              double* __restrict__ gConEff);
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
                                          const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_compute_bco(int n_config, int ns_h, int nZnT, bool lthreed,
                               const double* __restrict__ guu,
                               const double* __restrict__ guv,
                               const double* __restrict__ gvv,
                               const double* __restrict__ bsupu,
                               const double* __restrict__ bsupv,
                               double* __restrict__ bsubu,
                               double* __restrict__ bsubv);
__global__ void k_apply_m1_preconditioner(
    int n_config, int ns_local,
    int ns_force_local, int mpol, int ntor,
    const double* __restrict__ ard, const double* __restrict__ brd,
    const double* __restrict__ azd, const double* __restrict__ bzd,
    double* __restrict__ frss, double* __restrict__ fzcs,
    const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_apply_lambda_preconditioner(
    int n_config, int ns_local,
    int ns_con_local, int mpol, int ntor, bool lthreed,
    const double* __restrict__ lambdaPreconditioner,
    double* __restrict__ flsc, double* __restrict__ flcs,
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
                                    const std::uint8_t* __restrict__ d_active_per_cfg);
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
                                     const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_dealias_fwd(
    int n_config, int ns_force_local, int ns_con_local,
    int mpol, int ntor, int nZeta, int nThetaReduced,
    int nThetaEff, int nnyq2_plus_1,
    const double* __restrict__ gConEff, const double* __restrict__ tcon,
    const double* __restrict__ sinmui, const double* __restrict__ cosmui,
    const double* __restrict__ cosnv,  const double* __restrict__ sinnv,
    double* __restrict__ gsc, double* __restrict__ gcs);
__global__ void k_dealias_inv(
    int n_config, int ns_force_local, int ns_con_local,
    int mpol, int ntor, int nZeta, int nThetaReduced,
    int nThetaEff, int nnyq2_plus_1,
    const double* __restrict__ gsc, const double* __restrict__ gcs,
    const double* __restrict__ sinmu, const double* __restrict__ cosmu,
    const double* __restrict__ cosnv, const double* __restrict__ sinnv,
    const double* __restrict__ faccon,
    double* __restrict__ m_gCon);
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
                                  double* __restrict__ dec_flcs);
__global__ void k_m1_constraint(int n_config, int ns_local,
                                  int ns_force_local, int mpol, int ntor,
                                  double scalingFactor,
                                  double* __restrict__ dec_frss,
                                  double* __restrict__ dec_fzcs);
__global__ void k_m1_constraint_and_zero(int n_config, int ns_local,
                                          int ns_force_local, int mpol, int ntor,
                                          double scalingFactor,
                                          double* __restrict__ dec_frss,
                                          double* __restrict__ dec_fzcs);
__global__ void k_zero_z_force_for_m1(int n_config, int ns_local,
                                        int ns_force_local, int mpol, int ntor,
                                        double* __restrict__ dec_fzcs);
__global__ void k_zero_buffer(int n, double* __restrict__ p);
__global__ void k_pres_final_reduce(int n_config, int ns_h, double deltaS,
                                     double adiabaticIndex,
                                     const double* __restrict__ thermal_partial,
                                     const double* __restrict__ magnetic_partial,
                                     double* __restrict__ scalars_out,
                                     const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_cfg01_max_abs_diff(int per_cfg_size,
                                      const double* __restrict__ buf,
                                      double* __restrict__ out_scalar);
__global__ void k_rznorm_pts_x_partials(
    int ns_local, int mpol, int ntor,
    int nsMinHere_local, int nsMaxHere_local, bool lthreed,
    const double* __restrict__ x_rcc, const double* __restrict__ x_zsc,
    const double* __restrict__ x_rss, const double* __restrict__ x_zcs,
    double* __restrict__ partials);
__global__ void k_rz_norm_per_cfg(
    int n_config, int ns_local, int j_begin, int j_count, int mpol, int ntor,
    bool lthreed,
    const double* __restrict__ d_x_rcc, const double* __restrict__ d_x_rss,
    const double* __restrict__ d_x_zsc, const double* __restrict__ d_x_zcs,
    double* __restrict__ d_fnorm1,
    const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_force_norm_final_reduce(
    int n_config, int ns_h,
    const double* __restrict__ rz_partial,
    const double* __restrict__ l_partial,
    double* __restrict__ scalars_out,
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_check_convergence(
    int n_config,
    const double* __restrict__ scalars_out,
    const double* __restrict__ fnorm_scalars,
    const double* __restrict__ pressure_scalars,
    const double* __restrict__ volumes,
    double lamscale,
    double ftolv,
    std::uint8_t* __restrict__ conv_flag,
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_residuals_finalize_K(
    int n_config, int n_partitions,
    const double* __restrict__ partials_in,
    double* __restrict__ scalars_out,
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);
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
    double* __restrict__ rCon, double* __restrict__ zCon);
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
    double* __restrict__ rCon, double* __restrict__ zCon);
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
    double* __restrict__ rCon, double* __restrict__ zCon);
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
    double* __restrict__ rCon, double* __restrict__ zCon);
__global__ void k_scatter_basis_init(
    int mpol, int nThetaReduced, int kBatch_param,
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
    float* __restrict__ W);
__global__ void k_scatter_pack_Y_fp32(
    int n_config, int ns_local, int mpol, int kBatch_param, int nZeta,
    const double* __restrict__ Y, float* __restrict__ Y_packed);
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
    double* __restrict__ lv_e, double* __restrict__ lv_o);
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
    double* __restrict__ lv_e, double* __restrict__ lv_o);
__global__ void k_scatter_pack_Y_fp32_split(
    int n_config, int ns_local, int mpol, int kBatch_param, int nZeta,
    const double* __restrict__ Y,
    float* __restrict__ Y_hi, float* __restrict__ Y_lo);
__global__ void k_scatter_basis_init_split(
    int mpol, int nThetaReduced, int kBatch_param,
    const double* __restrict__ cosmu, const double* __restrict__ sinmu,
    const double* __restrict__ cosmum, const double* __restrict__ sinmum,
    float* __restrict__ W_hi, float* __restrict__ W_lo);
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
    double* __restrict__ lv_e, double* __restrict__ lv_o);
__global__ void k_scatter_rcon_zcon_fp64(
    int n_config, int ns_local, int mpol, int nZeta, int nThetaReduced, int nThetaEff,
    const double* __restrict__ Y, const double* __restrict__ cosmu,
    const double* __restrict__ sinmu,
    const double* __restrict__ xmpq, const double* __restrict__ sqrtSF,
    double* __restrict__ rCon, double* __restrict__ zCon);
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
    double* __restrict__ rCon, double* __restrict__ zCon);
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
    double* __restrict__ rCon, double* __restrict__ zCon);
#ifndef VMECPP_USE_HIP  // NVIDIA-only; see the definitions in kernels.cu
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
    double* __restrict__ lv_e, double* __restrict__ lv_o);
__global__ void k_i8b_build_w(int mpol, int nThetaReduced,
                              const double* __restrict__ cosmu,
                              const double* __restrict__ sinmu,
                              const double* __restrict__ cosmum,
                              const double* __restrict__ sinmum,
                              double* __restrict__ W, int K, int N);
__global__ void k_i8b_slice_w(const double* __restrict__ W, int K, int N,
                              signed char* __restrict__ Wl,
                              int* __restrict__ eW);
__global__ void k_i8b_row_exp(int n_config, int ns_local, int mpol,
                              int nZeta, const double* __restrict__ Y,
                              int* __restrict__ eY);
#endif  // VMECPP_USE_HIP
__global__ void k_tau_minmax(int n_config, int total,
                              const double* __restrict__ tau,
                              double* __restrict__ out2,
                              const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_apply_rz_pcr(int n_config, int mnsize, int ns_total, int num_basis,
                                 const int* __restrict__ jMin, int jMax,
                                 const double* __restrict__ a_in,
                                 const double* __restrict__ d_in,
                                 const double* __restrict__ b_in,
                                 double* __restrict__ c_inout,
                                 const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_apply_rz_thomas_serial(
    int n_config, int mnsize, int ns_total, int num_basis,
    const int* __restrict__ jMin, int jMax,
    const double* __restrict__ a_in, const double* __restrict__ d_in,
    const double* __restrict__ b_in, double* __restrict__ c_inout,
    const std::uint8_t* __restrict__ d_active_per_cfg);
// One block per (config, mn) row; the elimination ratios live in dynamic
// shared memory sized to jMax, so this handles ns_total beyond the 1024
// threads-per-block limit of the PCR solver. Used for ns_total > 1024.
__global__ void k_apply_rz_thomas_block(
    int n_config, int mnsize, int ns_total, int num_basis,
    const int* __restrict__ jMin, int jMax,
    const double* __restrict__ a_in, const double* __restrict__ d_in,
    const double* __restrict__ b_in, double* __restrict__ c_inout,
    const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_apply_rz_pcr_fp32(int n_config, int mnsize, int ns_total,
                                      int num_basis,
                                      const int* __restrict__ jMin, int jMax,
                                      const double* __restrict__ a_in,
                                      const double* __restrict__ d_in,
                                      const double* __restrict__ b_in,
                                      double* __restrict__ c_inout,
                                      const std::uint8_t* __restrict__ d_active_per_cfg);
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
                                             const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_rz_add_correction(int n_config, int mnsize, int ns_total,
                                      int num_basis,
                                      const int* __restrict__ jMin, int jMax,
                                      const double* __restrict__ x_saved,
                                      const double* __restrict__ c_corr,
                                      double* __restrict__ c_inout,
                                      const std::uint8_t* __restrict__ d_active_per_cfg);
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
    const std::uint8_t* __restrict__ d_active_per_cfg);

// Template kernels (instantiated at their launch sites).
// k_dealias_inv_tpl_mixed: mixed-precision variant of k_dealias_inv_tpl.
// Inner (m, n) loop multiplies gsc/gcs × cosnv/sinnv at FP32 (Ada FP32 cores
// outnumber FP64 cores ~4:1), with the result cast to FP64 and added to a
// FP64 accumulator. Outer accumulator + writes remain FP64. The sinmu/cosmu
// product into acc also stays FP64.
//
// Tolerance: each FP32 mult has ~1e-7 relative error; with NTOR+1=11 mults
// per (m) chain, w0/w1 accumulator error is ~1e-6 relative. The downstream
// gCon feeds back through AssembleTotalForces → forces → residual norm; if
// residual must converge below the FP32 noise floor, the iterative solver
// will fail to terminate (the same failure mode observed with the FP32
// cuFFT path). The relaxed contract here
// is: aspect_ratio remains bit-exact at 14 sig figs, and the run completes.
// Env-gate VMECPP_DEALIAS_MIXED, default OFF.
template <int MPOL, int NTOR>
__global__ __launch_bounds__(32, 16) void k_dealias_inv_tpl_mixed(
    int n_config, int ns_force_local, int ns_con_local,
    int nZeta, int nThetaReduced,
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
                    (size_t)MPOL * (size_t)(NTOR + 1);
  size_t cfg_grid = (size_t)config * (size_t)ns_con_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  double acc = 0.0;
  #pragma unroll
  for (int m = 1; m < MPOL - 1; ++m) {
    double fac = faccon[m];
    if (fac == 0.0) continue;
    double w0 = 0.0, w1 = 0.0;
    size_t idx_base = cfg_spec + (size_t)((jF * MPOL + m) * (NTOR + 1));
    #pragma unroll
    for (int n = 0; n <= NTOR; ++n) {
      int kn = k * nnyq2_plus_1 + n;
      float g0  = (float)gsc[idx_base + n];
      float c   = (float)cosnv[kn];
      float g1  = (float)gcs[idx_base + n];
      float s   = (float)sinnv[kn];
      // FP32 multiply, cast to FP64, accumulate at FP64 precision.
      w0 += (double)(g0 * c);
      w1 += (double)(g1 * s);
    }
    int bml = m * nThetaReduced + l;
    acc += fac * (w0 * sinmu[bml] + w1 * cosmu[bml]);
  }
  size_t dst = cfg_grid + (size_t)((jF * nZeta + k) * nThetaEff + l);
  m_gCon[dst] = acc;
}
// k_dealias_inv_tpl_split: same template specialization as k_dealias_inv_tpl
// but breaks the per-(m) n-loop dependency chain with 4 partial accumulators
// (w0a/w0b/w0c/w0d and w1a/w1b/w1c/w1d). Reduces FP dep chain from NTOR+1=11
// dependent FMAs to ceil((NTOR+1)/4)=3 per chain, freeing the warp scheduler
// to issue 4 independent FMAs in parallel. Targets the ILP-starved bottleneck
// observed in k_dealias_inv_tpl profiling.
//
// Tolerance: FP accumulator order changes -> ULP-level intermediate differences.
// Iterative solver re-converges; under the relaxed 12-sig-fig contract the
// final aspect_ratio is bit-exact (same logic as atomicAdd jacobian_metric).
//
// Env-gate VMECPP_DEALIAS_SPLIT (default OFF; measured as a small regression
// at the dispatch site, retained for further ILP experimentation).
template <int MPOL, int NTOR>
__global__ __launch_bounds__(32, 16) void k_dealias_inv_tpl_split(
    int n_config, int ns_force_local, int ns_con_local,
    int nZeta, int nThetaReduced,
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
                    (size_t)MPOL * (size_t)(NTOR + 1);
  size_t cfg_grid = (size_t)config * (size_t)ns_con_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  double acc = 0.0;
  #pragma unroll
  for (int m = 1; m < MPOL - 1; ++m) {
    double fac = faccon[m];
    if (fac == 0.0) continue;
    // 4 partial accumulators per w to break the dep chain.
    double w0a = 0.0, w0b = 0.0, w0c = 0.0, w0d = 0.0;
    double w1a = 0.0, w1b = 0.0, w1c = 0.0, w1d = 0.0;
    size_t idx_base = cfg_spec + (size_t)((jF * MPOL + m) * (NTOR + 1));
    #pragma unroll
    for (int n = 0; n <= NTOR; ++n) {
      int kn = k * nnyq2_plus_1 + n;
      double gv = gsc[idx_base + n] * cosnv[kn];
      double sv = gcs[idx_base + n] * sinnv[kn];
      switch (n & 3) {
        case 0: w0a += gv; w1a += sv; break;
        case 1: w0b += gv; w1b += sv; break;
        case 2: w0c += gv; w1c += sv; break;
        case 3: w0d += gv; w1d += sv; break;
      }
    }
    double w0 = (w0a + w0b) + (w0c + w0d);
    double w1 = (w1a + w1b) + (w1c + w1d);
    int bml = m * nThetaReduced + l;
    acc += fac * (w0 * sinmu[bml] + w1 * cosmu[bml]);
  }
  size_t dst = cfg_grid + (size_t)((jF * nZeta + k) * nThetaEff + l);
  m_gCon[dst] = acc;
}
// Templated specialization for compile-time-known (MPOL, NTOR). Lets the
// compiler fully unroll the m and n loops and pipeline the cosnv/sinnv +
// gsc/gcs loads. Used for the production shape (mpol=10, ntor=10).
template <int MPOL, int NTOR>
__global__ __launch_bounds__(32, 16) void k_dealias_inv_tpl(
    int n_config, int ns_force_local, int ns_con_local,
    int nZeta, int nThetaReduced,
    int nThetaEff, int nnyq2_plus_1,
    const double* __restrict__ gsc, const double* __restrict__ gcs,
    const double* __restrict__ sinmu, const double* __restrict__ cosmu,
    const double* __restrict__ cosnv, const double* __restrict__ sinnv,
    const double* __restrict__ faccon,
    double* __restrict__ m_gCon,
    const std::uint8_t* __restrict__ d_active_per_cfg) {
  int config = blockIdx.z / ns_force_local;
  int jF = blockIdx.z - config * ns_force_local;
  if (config >= n_config) return;
  if (d_active_per_cfg && !d_active_per_cfg[config]) return;
  int k = blockIdx.y;
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (jF >= ns_force_local || k >= nZeta || l >= nThetaReduced) return;
  size_t cfg_spec = (size_t)config * (size_t)ns_force_local *
                    (size_t)MPOL * (size_t)(NTOR + 1);
  size_t cfg_grid = (size_t)config * (size_t)ns_con_local *
                    (size_t)nZeta * (size_t)nThetaEff;
  double acc = 0.0;
  #pragma unroll
  for (int m = 1; m < MPOL - 1; ++m) {
    double fac = faccon[m];
    if (fac == 0.0) continue;
    double w0 = 0.0, w1 = 0.0;
    size_t idx_base = cfg_spec + (size_t)((jF * MPOL + m) * (NTOR + 1));
    #pragma unroll
    for (int n = 0; n <= NTOR; ++n) {
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
// Custom tile-cooperative GEMM with Veltkamp-Dekker per-multiply. Each
// thread accumulates one output element in a DD-pair register, four
// Veltkamp-Dekker exact-FP32 sub-products per k-step (A_hi*W_hi,
// A_hi*W_lo, A_lo*W_hi, A_lo*W_lo). Cooperative tile load of A_hi/A_lo
// and W_hi/W_lo into shared memory. Output cast DD->FP64 for the
// production scatter buffers.
//
// Math: A (FP64) ~= A_hi + A_lo (FP32 pair); B (FP64) ~= B_hi + B_lo.
// A*B = A_hi*B_hi + A_hi*B_lo + A_lo*B_hi + A_lo*B_lo. Each FP32 mul
// gets a Veltkamp split inside two_product_dekker so it's exact in
// FP32; the four exact products are summed in DD-pair. Result has
// ~48-bit precision (matches scalar Path 3b's Veltkamp-Dekker Ozaki).
//
// Templated tile sizes. Default TM=32, TN=32, TK=32 with 256 threads
// (16x16) computing 2x2 outputs per thread.
template <int TM, int TN, int TK>
__global__ void k_scatter_custom_gemm_vd(
    int B, int M, int N,
    const float* __restrict__ A_hi, const float* __restrict__ A_lo,
    const float* __restrict__ W_hi, const float* __restrict__ W_lo,
    double* __restrict__ C) {
  constexpr int TPB_X = 16;
  constexpr int TPB_Y = 16;
  constexpr int OUT_X = TN / TPB_X;  // 2
  constexpr int OUT_Y = TM / TPB_Y;  // 2
  int tile_b = blockIdx.y * TM;
  int tile_n = blockIdx.x * TN;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  __shared__ float Ash_hi[TM][TK];
  __shared__ float Ash_lo[TM][TK];
  __shared__ float Wsh_hi[TK][TN];
  __shared__ float Wsh_lo[TK][TN];
  // Per-thread DD accumulators: OUT_Y * OUT_X = 4 outputs.
  DD acc[OUT_Y][OUT_X];
  #pragma unroll
  for (int oy = 0; oy < OUT_Y; ++oy)
    #pragma unroll
    for (int ox = 0; ox < OUT_X; ++ox)
      acc[oy][ox] = dd_from_f(0.0f);
  for (int k_tile = 0; k_tile < M; k_tile += TK) {
    // Cooperative tile load. Each thread loads OUT_Y*TK/TPB_X = 4 A
    // elements (per hi/lo) and OUT_X*TK/TPB_Y = 4 W elements (per hi/lo).
    #pragma unroll
    for (int oy = 0; oy < OUT_Y; ++oy) {
      int rowA = tile_b + ty * OUT_Y + oy;
      #pragma unroll
      for (int kk = tx; kk < TK; kk += TPB_X) {
        int gA_k = k_tile + kk;
        if (rowA < B && gA_k < M) {
          size_t idxA = (size_t)rowA * M + (size_t)gA_k;
          Ash_hi[ty * OUT_Y + oy][kk] = A_hi[idxA];
          Ash_lo[ty * OUT_Y + oy][kk] = A_lo[idxA];
        } else {
          Ash_hi[ty * OUT_Y + oy][kk] = 0.0f;
          Ash_lo[ty * OUT_Y + oy][kk] = 0.0f;
        }
      }
    }
    #pragma unroll
    for (int ox = 0; ox < OUT_X; ++ox) {
      int colW = tile_n + tx * OUT_X + ox;
      #pragma unroll
      for (int kk = ty; kk < TK; kk += TPB_Y) {
        int gW_k = k_tile + kk;
        if (colW < N && gW_k < M) {
          size_t idxW = (size_t)gW_k * N + (size_t)colW;
          Wsh_hi[kk][tx * OUT_X + ox] = W_hi[idxW];
          Wsh_lo[kk][tx * OUT_X + ox] = W_lo[idxW];
        } else {
          Wsh_hi[kk][tx * OUT_X + ox] = 0.0f;
          Wsh_lo[kk][tx * OUT_X + ox] = 0.0f;
        }
      }
    }
    __syncthreads();
    // Inner-product over K.
    int k_end = (k_tile + TK <= M) ? TK : (M - k_tile);
    #pragma unroll 8
    for (int kk = 0; kk < k_end; ++kk) {
      #pragma unroll
      for (int oy = 0; oy < OUT_Y; ++oy) {
        float a_hi = Ash_hi[ty * OUT_Y + oy][kk];
        float a_lo = Ash_lo[ty * OUT_Y + oy][kk];
        #pragma unroll
        for (int ox = 0; ox < OUT_X; ++ox) {
          float w_hi = Wsh_hi[kk][tx * OUT_X + ox];
          float w_lo = Wsh_lo[kk][tx * OUT_X + ox];
          DD p1 = two_product_dekker(a_hi, w_hi);
          DD p2 = two_product_dekker(a_hi, w_lo);
          DD p3 = two_product_dekker(a_lo, w_hi);
          DD p4 = two_product_dekker(a_lo, w_lo);
          DD s12 = dd_add(p1, p2);
          DD s34 = dd_add(p3, p4);
          DD s_all = dd_add(s12, s34);
          acc[oy][ox] = dd_add(acc[oy][ox], s_all);
        }
      }
    }
    __syncthreads();
  }
  // Write outputs.
  #pragma unroll
  for (int oy = 0; oy < OUT_Y; ++oy) {
    int rowC = tile_b + ty * OUT_Y + oy;
    if (rowC >= B) continue;
    #pragma unroll
    for (int ox = 0; ox < OUT_X; ++ox) {
      int colC = tile_n + tx * OUT_X + ox;
      if (colC >= N) continue;
      C[(size_t)rowC * N + (size_t)colC] = dd_to_double(acc[oy][ox]);
    }
  }
}
// The three kernels below reach the tensor cores through nvcuda::wmma s8
// fragments, so they are NVIDIA-only.
#ifndef VMECPP_USE_HIP
// k_scatter_main_and_con_i8ozaki: the scatter GEMM on int8 tensor cores
// with exact integer accumulation (the Ozaki construction). Each FP64
// operand is scaled per A-row / per B-column to (-0.5, 0.5), split into
// eight 7-bit signed limbs (56 bits, covering the FP64 mantissa), and
// the limb cross-products accumulate through wmma s8 x s8 -> s32
// fragments, which are exact: no scalar recovery pass is needed. Bands
// b = p + q share one fragment chain (equal scale); per-band sums stay
// far below the s32 range (<= 8 pairs x 48 x 127^2 ~ 6e6). The FP64
// output is the band sum scaled by 2^(eA + eB - 7(b + 2)).
//
// Gated by VMECPP_SCATTER_I8OZAKI=1. Tile geometry matches the wmma
// kernel: mpol <= 12 (4 * mpol <= K_PAD) and nThetaReduced <= 16.
// LIMBS_T selects the operand width: 8 limbs cover the FP64 mantissa
// (56 bits); 4 limbs carry 28-bit operands (rel ~4e-9) at half the
// limb-plane traffic and half the mma work. Launch with 32 * LIMBS_T
// threads (one band per warp).
template <int LIMBS_T>
__global__ void k_scatter_main_and_con_i8ozaki(
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
    double* __restrict__ lv_e, double* __restrict__ lv_o) {
  constexpr int K_PAD = 48;
  constexpr int M_TILE = 16;
  constexpr int N_TILE = 16;
  constexpr int LIMBS = LIMBS_T;
  constexpr int BANDS = LIMBS_T;  // p + q in [0, LIMBS); deeper bands truncated

  int z = blockIdx.z;
  int config = z / ns_local;
  int jF_local = z - config * ns_local;
  if (config >= n_config || jF_local >= ns_local) return;
  int k = blockIdx.y;
  if (k >= nZeta) return;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;

  extern __shared__ unsigned char smem_raw[];
  double* s_Y    = reinterpret_cast<double*>(smem_raw);
  double* s_cmu  = s_Y    + (size_t)kBatch * (size_t)mpol;
  double* s_smu  = s_cmu  + (size_t)mpol  * (size_t)nThetaReduced;
  double* s_cmum = s_smu  + (size_t)mpol  * (size_t)nThetaReduced;
  double* s_smum = s_cmum + (size_t)mpol  * (size_t)nThetaReduced;
  double* s_xmpq = s_smum + (size_t)mpol  * (size_t)nThetaReduced;
  // ldmatrix requires 32-byte tile bases; the FP64 staging length is not
  // a 32-byte multiple at every shape.
  size_t a8_off = (((size_t)(s_xmpq + mpol) - (size_t)smem_raw) + 31u) &
                  ~(size_t)31u;
  signed char* s_A8 = reinterpret_cast<signed char*>(smem_raw + a8_off);
  signed char* s_B8 = s_A8 + LIMBS * M_TILE * K_PAD;
  int* s_band = reinterpret_cast<int*>(s_B8 + LIMBS * K_PAD * N_TILE);
  int* s_eA = s_band + BANDS * M_TILE * N_TILE;
  int* s_eB = s_eA + M_TILE;

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
  for (int i = tid; i < mpol; i += blockDim.x) s_xmpq[i] = xmpq[i];
  if (tid < M_TILE) s_eA[tid] = INT_MIN;
  if (tid < N_TILE) s_eB[tid] = INT_MIN;
  __syncthreads();

  // Pass 1: per-row (A) and per-column (B) max exponents.
  for (int i = tid; i < M_TILE * K_PAD; i += blockDim.x) {
    int l = i / K_PAD;
    int kk = i - l * K_PAD;
    double v = i8oz_a_elem(l, kk, mpol, nThetaReduced,
                           s_cmu, s_smu, s_cmum, s_smum);
    if (v != 0.0) atomicMax(&s_eA[l], ilogb(v));
  }
  for (int i = tid; i < K_PAD * N_TILE; i += blockDim.x) {
    int kk = i / N_TILE;
    int n  = i - kk * N_TILE;
    double v = i8oz_b_elem(kk, n, mpol, s_Y);
    if (v != 0.0) atomicMax(&s_eB[n], ilogb(v));
  }
  __syncthreads();

  // Pass 2: limb extraction at the row/column scale. The +2 keeps the
  // scaled magnitude at or below 0.5 so rint(r * 128) stays within the
  // signed 8-bit range at every limb.
  for (int i = tid; i < M_TILE * K_PAD; i += blockDim.x) {
    int l = i / K_PAD;
    int kk = i - l * K_PAD;
    double v = i8oz_a_elem(l, kk, mpol, nThetaReduced,
                           s_cmu, s_smu, s_cmum, s_smum);
    double r = (s_eA[l] == INT_MIN) ? 0.0 : ldexp(v, -(s_eA[l] + 2));
    #pragma unroll
    for (int pl = 0; pl < LIMBS; ++pl) {
      double scaled = r * 128.0;
      int limb = (int)rint(scaled);
      r = scaled - (double)limb;
      s_A8[pl * M_TILE * K_PAD + i] = (signed char)limb;
    }
  }
  for (int i = tid; i < K_PAD * N_TILE; i += blockDim.x) {
    int kk = i / N_TILE;
    int n  = i - kk * N_TILE;
    double v = i8oz_b_elem(kk, n, mpol, s_Y);
    double r = (s_eB[n] == INT_MIN) ? 0.0 : ldexp(v, -(s_eB[n] + 2));
    #pragma unroll
    for (int pl = 0; pl < LIMBS; ++pl) {
      double scaled = r * 128.0;
      int limb = (int)rint(scaled);
      r = scaled - (double)limb;
      s_B8[pl * K_PAD * N_TILE + i] = (signed char)limb;
    }
  }
  __syncthreads();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  {
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                   wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                   wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag;
    // One band per warp; pairs (p, q) with p + q == band chain into the
    // same exact s32 accumulator.
    int band = warp_id;
    wmma::fill_fragment(c_frag, 0);
    for (int pq = 0; pq <= band; ++pq) {
      int pa = pq, qb = band - pq;
      if (pa >= LIMBS || qb >= LIMBS) continue;
      for (int kk = 0; kk < K_PAD; kk += 16) {
        wmma::load_matrix_sync(a_frag,
            s_A8 + pa * M_TILE * K_PAD + kk, K_PAD);
        wmma::load_matrix_sync(b_frag,
            s_B8 + qb * K_PAD * N_TILE + kk * N_TILE, N_TILE);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }
    wmma::store_matrix_sync(&s_band[band * M_TILE * N_TILE], c_frag,
                            N_TILE, wmma::mem_row_major);
  }
#else
  // Pre-Turing fallback: scalar integer accumulation, same banding.
  if (warp_id == 0) {
    for (int band = 0; band < BANDS; ++band) {
      for (int mn = lane; mn < M_TILE * N_TILE; mn += 32) {
        int mi = mn / N_TILE;
        int ni = mn - mi * N_TILE;
        int acc = 0;
        for (int pq = 0; pq <= band; ++pq) {
          int pa = pq, qb = band - pq;
          if (pa >= LIMBS || qb >= LIMBS) continue;
          for (int kk = 0; kk < K_PAD; ++kk) {
            acc += (int)s_A8[pa * M_TILE * K_PAD + mi * K_PAD + kk] *
                   (int)s_B8[qb * K_PAD * N_TILE + kk * N_TILE + ni];
          }
        }
        s_band[band * M_TILE * N_TILE + mn] = acc;
      }
    }
  }
#endif
  __syncthreads();

  for (int i = tid; i < M_TILE * N_TILE; i += blockDim.x) {
    int l = i / N_TILE;
    int n = i - l * N_TILE;
    if (l < nThetaReduced && s_eA[l] != INT_MIN && s_eB[n] != INT_MIN) {
      // Ascending magnitude: deepest band first.
      double acc = 0.0;
      int e_base = s_eA[l] + s_eB[n] + 4;
      for (int band = BANDS - 1; band >= 0; --band) {
        acc += ldexp((double)s_band[band * M_TILE * N_TILE + i],
                     e_base - 7 * (band + 2));
      }
      size_t cfg_full = (size_t)config * (size_t)ns_local *
                        (size_t)nZeta * (size_t)nThetaEff;
      size_t idx = cfg_full +
                   (size_t)((jF_local * nZeta + k) * nThetaEff + l);
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
    } else if (l < nThetaReduced) {
      size_t cfg_full = (size_t)config * (size_t)ns_local *
                        (size_t)nZeta * (size_t)nThetaEff;
      size_t idx = cfg_full +
                   (size_t)((jF_local * nZeta + k) * nThetaEff + l);
      double zero = 0.0;
      switch (n) {
        case 0:  r1_e[idx] = zero; break;
        case 1:  r1_o[idx] = zero; break;
        case 2:  ru_e[idx] = zero; break;
        case 3:  ru_o[idx] = zero; break;
        case 4:  rv_e[idx] = zero; break;
        case 5:  rv_o[idx] = zero; break;
        case 6:  z1_e[idx] = zero; break;
        case 7:  z1_o[idx] = zero; break;
        case 8:  zu_e[idx] = zero; break;
        case 9:  zu_o[idx] = zero; break;
        case 10: zv_e[idx] = zero; break;
        case 11: zv_o[idx] = zero; break;
        case 12: lu_e[idx] = zero; break;
        case 13: lu_o[idx] = zero; break;
        case 14: lv_e[idx] = zero; break;
        case 15: lv_o[idx] = zero; break;
      }
    }
  }
}
// Spec limbs, row-major [B, K] per limb plane. LIMBS_T limits the
// slice to the leading (most-significant) planes; the buffer is sized
// for eight so the limb mode can change between iterations.
template <int LIMBS_T>
__global__ void k_i8b_slice_y(int n_config, int ns_local, int mpol,
                              int nZeta, const double* __restrict__ Y,
                              const int* __restrict__ eY,
                              signed char* __restrict__ Yl, int B_pad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int K = 16 * mpol;
  int B = n_config * ns_local * nZeta;
  if (idx >= B_pad * K) return;
  int b = idx / K;
  int mq = idx - b * K;
  double r = 0.0;
  int m = mq >> 4;
  int q = mq & 15;
  if (b < B && q < kBatch) {
    int k = b % nZeta;
    int cj = b / nZeta;
    size_t y_idx = (((size_t)cj * (size_t)mpol * (size_t)kBatch) +
                    ((size_t)m * kBatch + q)) * (size_t)nZeta + (size_t)k;
    double v = Y[y_idx];
    int e = eY[b];
    r = (e == INT_MIN) ? 0.0 : ldexp(v, -(e + 2));
  }
  size_t plane = (size_t)B_pad * (size_t)K;
  #pragma unroll
  for (int pl = 0; pl < LIMBS_T; ++pl) {
    double scaled = r * 128.0;
    int limb = (int)rint(scaled);
    r = scaled - (double)limb;
    Yl[(size_t)pl * plane + idx] = (signed char)limb;
  }
}
// Banded s8 GEMM: one block per (64-row stripe, 16-column tile). One
// warp per band; bands chain their (p, q) limb pairs into one exact
// s32 accumulator. The combine scales bands into FP64 and writes the
// 16 channel arrays directly (a 16-column tile is one poloidal cell).
// LIMBS_T as in the per-tile kernel: 8 covers the FP64 mantissa, 4
// halves the staged limb traffic at 28-bit operand width. The W limb
// planes are built most-significant first, so a 4-limb consumer reads
// the leading planes of the 8-limb build. Launch with 32 * LIMBS_T
// threads.
template <int LIMBS_T>
__global__ void k_i8b_gemm(
    int n_config, int ns_local, int mpol, int nZeta, int nThetaReduced,
    int nThetaEff, int B_pad,
    const signed char* __restrict__ Yl, const int* __restrict__ eY,
    const signed char* __restrict__ Wl, const int* __restrict__ eW,
    double* __restrict__ r1_e, double* __restrict__ r1_o,
    double* __restrict__ ru_e, double* __restrict__ ru_o,
    double* __restrict__ rv_e, double* __restrict__ rv_o,
    double* __restrict__ z1_e, double* __restrict__ z1_o,
    double* __restrict__ zu_e, double* __restrict__ zu_o,
    double* __restrict__ zv_e, double* __restrict__ zv_o,
    double* __restrict__ lu_e, double* __restrict__ lu_o,
    double* __restrict__ lv_e, double* __restrict__ lv_o) {
  constexpr int ROWS = 64;
  constexpr int NT = 16;
  constexpr int LIMBS = LIMBS_T;
  constexpr int BANDS = LIMBS_T;
  int K = 16 * mpol;
  int N = 16 * nThetaReduced;
  int row0 = blockIdx.x * ROWS;
  int col0 = blockIdx.y * NT;  // one l-cell: l = blockIdx.y
  int l = blockIdx.y;
  int tid = threadIdx.x;
  int warp_id = tid >> 5;

  extern __shared__ unsigned char sm[];
  // Per K-chunk staging: Y limbs [LIMBS][ROWS][16], W limbs [LIMBS][16][NT].
  signed char* sY = reinterpret_cast<signed char*>(sm);
  signed char* sW = sY + LIMBS * ROWS * 16;
  int* sBand = reinterpret_cast<int*>(sW + LIMBS * 16 * NT);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750)
  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, signed char,
                 wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, signed char,
                 wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, int> c_frag[ROWS / 16];
  int band = warp_id;
  #pragma unroll
  for (int rf = 0; rf < ROWS / 16; ++rf) wmma::fill_fragment(c_frag[rf], 0);

  for (int kc = 0; kc < K; kc += 16) {
    // Stage this K-chunk's limbs.
    for (int i = tid; i < LIMBS * ROWS * 16; i += blockDim.x) {
      int pl = i / (ROWS * 16);
      int rr = (i / 16) % ROWS;
      int kk = i & 15;
      sY[i] = Yl[(size_t)pl * B_pad * K + (size_t)(row0 + rr) * K +
                 (kc + kk)];
    }
    for (int i = tid; i < LIMBS * 16 * NT; i += blockDim.x) {
      int pl = i / (16 * NT);
      int kk = (i / NT) & 15;
      int nn = i & 15;
      sW[i] = Wl[(size_t)pl * K * N + (size_t)(kc + kk) * N + (col0 + nn)];
    }
    __syncthreads();
    for (int pq = 0; pq <= band; ++pq) {
      int pa = pq, qb = band - pq;
      if (pa >= LIMBS || qb >= LIMBS) continue;
      wmma::load_matrix_sync(b_frag, sW + qb * 16 * NT, NT);
      #pragma unroll
      for (int rf = 0; rf < ROWS / 16; ++rf) {
        wmma::load_matrix_sync(a_frag, sY + pa * ROWS * 16 + rf * 16 * 16,
                               16);
        wmma::mma_sync(c_frag[rf], a_frag, b_frag, c_frag[rf]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int rf = 0; rf < ROWS / 16; ++rf) {
    wmma::store_matrix_sync(
        sBand + (size_t)band * ROWS * NT + rf * 16 * NT, c_frag[rf], NT,
        wmma::mem_row_major);
  }
  __syncthreads();
#else
  if (warp_id == 0 && tid == 0) {
    for (int band = 0; band < BANDS; ++band) {
      for (int rr = 0; rr < ROWS; ++rr) {
        for (int nn = 0; nn < NT; ++nn) {
          int acc = 0;
          for (int pq = 0; pq <= band; ++pq) {
            int pa = pq, qb = band - pq;
            if (pa >= LIMBS || qb >= LIMBS) continue;
            for (int kk = 0; kk < K; ++kk) {
              acc += (int)Yl[(size_t)pa * B_pad * K +
                             (size_t)(row0 + rr) * K + kk] *
                     (int)Wl[(size_t)qb * K * N + (size_t)kk * N +
                             (col0 + nn)];
            }
          }
          sBand[(size_t)band * ROWS * NT + rr * NT + nn] = acc;
        }
      }
    }
  }
  __syncthreads();
#endif

  int B = n_config * ns_local * nZeta;
  for (int i = tid; i < ROWS * NT; i += blockDim.x) {
    int rr = i / NT;
    int ch = i & 15;
    int b = row0 + rr;
    if (b >= B) continue;
    int eyv = eY[b];
    int ewv = eW[col0 + ch];
    double acc = 0.0;
    if (eyv != INT_MIN && ewv != INT_MIN) {
      int e_base = eyv + ewv + 4;
      for (int band = BANDS - 1; band >= 0; --band) {
        acc += ldexp((double)sBand[(size_t)band * ROWS * NT + i],
                     e_base - 7 * (band + 2));
      }
    }
    int k = b % nZeta;
    int cj = b / nZeta;
    int config = cj / ns_local;
    int jF_local = cj - config * ns_local;
    size_t idx = (size_t)config * (size_t)ns_local * (size_t)nZeta *
                     (size_t)nThetaEff +
                 (size_t)((jF_local * nZeta + k) * nThetaEff + l);
    switch (ch) {
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
#endif  // VMECPP_USE_HIP (int8-Ozaki tensor-core kernels)

// Run-scoped and process-scoped globals (defined in fft_toroidal_cuda_state.cu).
extern bool g_iter_graph_capturing;
extern bool g_pts_init_only;
extern int g_batch_mem_shape[4];
extern int g_batch_outputs_shape[4];
extern int g_i8_limbs_env;
extern int g_i8_limbs_last;
extern int g_iter_graph_env;
extern int g_residuals_k_run;
extern std::vector<double> g_batch_dec_x_mem;
extern std::vector<double> g_batch_inputs_mem;
extern std::vector<double> g_batch_outputs_mem;
extern std::vector<double> g_batch_phipF, g_batch_phipH, g_batch_currH,
    g_batch_iotaH, g_batch_massH;
extern int g_batch_prof_ncfg, g_batch_prof_nsh, g_batch_prof_nsf;

// Cross-unit internal helpers.
void DiagCfg01DiffCuda(const double* d_buf, int per_cfg_size,
                       const char* label);

}  // namespace vmecpp

#endif  // VMECPP_USE_CUDA

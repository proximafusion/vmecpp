// Internal header of the CUDA iteration body: device helpers, the persistent
// device state, kernel declarations and run-scoped globals.
#pragma once

// Both toroidal transform directions run on the device and share the persistent
// buffers of CudaToroidalState.
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal_cuda.h"

#ifdef VMECPP_USE_CUDA

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cufft.h>

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

namespace vmecpp {

// Per-configuration caches filled by the device-to-host transfers the kernel
// wrappers already make, read through the accessors in fft_toroidal_cuda.h.
extern std::vector<double> g_residuals_invar_cache;
extern std::vector<double> g_residuals_precd_cache;
// Force-norm sums [sum_rz, sum_l] per configuration from ComputeForceNormsCuda.
extern std::vector<double> g_fnorm_scalars_cache;
// Jacobian extrema [minTau, maxTau] per configuration from ComputeJacobianCuda.
extern std::vector<double> g_jac_minmax_cache;
// PressureAndEnergiesCuda's energies [thermal, magnetic, mhd] per
// configuration, valid after the caller's next synchronization.
extern std::vector<double> g_pressure_scalars_cache;
// Plasma volume per configuration from UpdateVolumeCuda, valid after its
// synchronization.
extern std::vector<double> g_plasma_volume_cache;

const std::vector<double>& GetResidualsPerCfgCacheInvar();

const std::vector<double>& GetResidualsPerCfgCachePrecd();

const std::vector<double>& GetFnormScalarsPerCfgCache();

const std::vector<double>& GetJacMinmaxPerCfgCache();

const std::vector<double>& GetPressureScalarsPerCfgCache();

// fNorm1 per configuration, the reciprocal rzNorm of its device positions, from
// ComputeForceNormsCuda at the force-norm cadence.
extern std::vector<double> g_fnorm1_per_cfg_cache;

const std::vector<double>& GetPlasmaVolumePerCfgCache();

const std::vector<double>& GetFnorm1PerCfgCache();

// Configuration count of the current run, set by ResetCudaStateForNewVmecRun
// at the start of every Vmec::run and stable within the run.
extern int g_n_config_run;

// Whether the run's configurations carry their own inputs (distinct mode), so
// the device interpolates each at the multigrid transitions.
extern bool g_batch_distinct_run;

// Slot indices for the 12 quantities transformed per (jF, m).
enum Slot {
  kRmkcc = 0,  kRmkss = 1,  kRmkccN = 2, kRmkssN = 3,
  kZmksc = 4,  kZmkcs = 5,  kZmkscN = 6, kZmkcsN = 7,
  kLmksc = 8,  kLmkcs = 9,  kLmkscN = 10, kLmkcsN = 11,
};
constexpr int kBatch = 12;

// CUDA and cuFFT failures throw, so an embedding process gets a catchable
// error; the next run's reset and Reshape rebuild the device state.
static void cuda_check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    char msg[256];
    std::snprintf(msg, sizeof(msg), "[fft_toroidal_cuda] CUDA error at %s: %s",
                  what, cudaGetErrorString(err));
    throw std::runtime_error(msg);
  }
}
static void cufft_check(cufftResult res, const char* what) {
  if (res != CUFFT_SUCCESS) {
    char msg[256];
    std::snprintf(msg, sizeof(msg), "[fft_toroidal_cuda] cuFFT error at %s: %d",
                  what, (int)res);
    throw std::runtime_error(msg);
  }
}

// Raw upper estimate in bytes of the persistent device allocation at this shape
// and count, without the pre-flight's margins.
long long CudaBudgetRawBytes(long long n_cfg, long long ns, long long mpol,
                             long long ntor, long long nZeta,
                             long long nThetaEff);

// State-region globals and helpers referenced by the struct below
// (defined in fft_toroidal_cuda_state.cu).
extern bool g_free_boundary_run;
extern bool g_vacuum_edge_run;

// Process-static CUDA state.
struct CudaToroidalState {
  bool initialized = false;
  // Cached shape parameters.
  int n_cached = -1, nfp_cached = -1, mpol_cached = -1;
  int ns_local_cached = -1, ntor_cached = -1, nhalf_cached = -1;
  int nThetaReduced_cached = -1, nThetaEff_cached = -1, nZeta_cached = -1;
  int ns_con_local_cached = -1;

  // Equilibria the device buffers are sized for, carried on every kernel's
  // launch grid; 1 keeps the single-call path bit-exact.
  int n_config_max = 1;

  // Persistent device buffers + stream.
  cudaStream_t stream = nullptr;
  cufftHandle cufft_plan = 0;
  // Segment-three graph: effectiveConstraintForceCuda through
  // DecomposeAndConstrainCuda on one stream without host reads.
  cudaGraph_t seg3_graph = nullptr;
  bool seg3_vacuum_edge_at_capture = false;
  cudaGraphExec_t seg3_graph_exec = nullptr;
  bool seg3_graph_captured = false;
  bool seg3_in_capture = false;  // true while between BeginSeg3 and EndSeg3 in capture mode
  int seg3_warmup_calls = 0;     // capture begins only after the first warmup invocation completes lazy allocations

  // Segment-four graph: the four preconditioner wrappers between the two
  // ResidualsCuda calls, re-captured when jMax changes.
  cudaGraph_t seg4_graph = nullptr;
  cudaGraphExec_t seg4_graph_exec = nullptr;
  bool seg4_graph_captured = false;
  bool seg4_in_capture = false;
  int seg4_warmup_calls = 0;
  int seg4_last_jMax = -1;       // re-capture if jMax changes

  // Segment-two graph: the six kernel-only wrappers from ComputeJacobianCuda's
  // synchronization to the preconditioner update.
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

  size_t fft_x_elems = 0;  // element count of the complex buffer d_X
  size_t fft_y_elems = 0;  // element count of the real buffer d_Y

  // One buffer for the 18 output arrays: 16 slices of ns_local * nZeta *
  // nThetaEff and 2 of ns_con_local * nZeta * nThetaEff.
  double* d_outputs_block = nullptr;
  double* h_outputs_pinned = nullptr;
  size_t outputs_block_bytes = 0;
  // The six doubles SetRadialExtent and SetGeometricOffset read, copied to
  // pinned host memory in place of the whole outputs block.
  double* d_geom_scalars = nullptr;  // six doubles resident on the device
  double* h_geom_scalars = nullptr;  // six doubles in pinned host memory
  // Deferred geometry-scalar writes, committed by FlushFwdGeomScalarsToHost
  // after ComputeJacobianCuda's synchronization.
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
  // d_sqrtSH depends on ns alone, so it is staged once per Reshape.
  bool sqrtSH_staged = false;

  // Raised when the fused Jacobian and metric kernel produced the metric, so
  // ComputeMetricElementsCuda skips its launch this iteration.
  bool jac_metric_fused_this_iter = false;

  // Raised when the fused kernel also produced dVdsH, so
  // UpdateDifferentialVolumeCuda skips its launch this iteration.
  bool dvdsh_fused_this_iter = false;
  // massH, currH, phipF, phipH and the radial blending depend on the grid and
  // the inputs only; staged once, reset in Reshape.
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
  // The dealias faccon is fixed by the IdealMhdModel constructor, so it is
  // staged once.
  bool dealias_faccon_staged = false;
  // iotaH needs seeding once: with ncurr = 1 the device updates it each
  // iteration, with ncurr = 0 it is the prescribed profile.
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

  // radialForceBalance outputs: half-grid bucoH, bvcoH; interior jcurvF,
  // jcuruF, presgradF, dVdsF, equiF; and the presH, chipF, phipF inputs.
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

  // rzConIntoVolume outputs (full-grid con range, ns_con_local x nZnT).
  double* d_rCon0 = nullptr;
  double* d_zCon0 = nullptr;
  int rzcon0_ns_con_cached = -1;
  int rzcon0_nZnT_cached = -1;

  // computeBContra: half-grid bsupu, bsupv, the chip and iota profiles, their
  // inputs and the ncurr = 1 reduction scratch.
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

  // updateLambdaPreconditioner: bLambda, dLambda, cLambda of ns_h + 1 (offset
  // by one as on the CPU), and lambdaPreconditioner.
  double* d_bLambda = nullptr;
  double* d_dLambda = nullptr;
  double* d_cLambda = nullptr;
  double* d_lambdaPreconditioner = nullptr;
  // Tridiagonal m = 0 lambda preconditioner: the half-grid <g_uu / sqrt g> per
  // configuration and, per (cfg, n), the right side and upper diagonal.
  double* d_lambda_b_half = nullptr;
  double* d_lambda_m0_c = nullptr;
  double* d_lambda_m0_a = nullptr;
  int lambda_m0_stride_cached = -1;
  int lambda_m0_ns_cached = -1;
  int lambda_m0_ntor_cached = -1;

  // computePreconditioningMatrix scratch.
  // ax: ns_h x 4, bx: ns_h x 3, cx: ns_h.
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

  // R-side and Z-side snapshots of the preconditioning-matrix coefficients,
  // since the second call overwrites the shared scratch.
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
  // The host-computed rBSq profile of the free-boundary edge term, nZnT
  // doubles; rbsq_staged marks it current for this iteration.
  double* d_rbsq = nullptr;
  int rbsq_size = 0;
  bool rbsq_staged = false;
  double* d_ard = nullptr;  // per-call H2D
  double* d_azd = nullptr;  // per-call H2D

  // effectiveConstraintForce + assembleTotalForces helpers.
  double* d_gConEff = nullptr;
  double* d_gCon = nullptr;       // host H2D per call
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

  // DecomposeAndConstrainCuda fills this shadow of m_decomposed_f, which the
  // M1, lambda and RZ preconditioners read.
  double* d_decomposed_frcc = nullptr;
  double* d_decomposed_frss = nullptr;
  double* d_decomposed_fzsc = nullptr;
  double* d_decomposed_fzcs = nullptr;
  double* d_decomposed_flsc = nullptr;
  double* d_decomposed_flcs = nullptr;
  double* d_scalxc = nullptr;        // host m_p_.scalxc staged per call
  int decomposed_size_cached = -1;

  // Velocity (ns_con_local) and position (ns_local) of the device time
  // integrator, six spectral fields each, per configuration.
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
  // Device twin of the host physical_x_backup: stored on NO_RESTART, restored
  // on BAD_JACOBIAN and BAD_PROGRESS.
  double* d_pts_x_backup_rcc = nullptr;
  double* d_pts_x_backup_rss = nullptr;
  double* d_pts_x_backup_zsc = nullptr;
  double* d_pts_x_backup_zcs = nullptr;
  double* d_pts_x_backup_lsc = nullptr;
  double* d_pts_x_backup_lcs = nullptr;
  bool pts_x_backup_initialized = false;
  // Device twin of the host last_evaluated_x, the state of the latest force
  // evaluation, which the backup takes under backup_evaluated_state.
  double* d_pts_x_evaluated_rcc = nullptr;
  double* d_pts_x_evaluated_rss = nullptr;
  double* d_pts_x_evaluated_zsc = nullptr;
  double* d_pts_x_evaluated_zcs = nullptr;
  double* d_pts_x_evaluated_lsc = nullptr;
  double* d_pts_x_evaluated_lcs = nullptr;
  bool pts_x_evaluated_valid = false;
  // Per-configuration converged-state snapshots, which the outputs prefer,
  // since kernels that ignore the mask keep modifying the live slice.
  double* d_pts_x_final_rcc = nullptr;
  double* d_pts_x_final_rss = nullptr;
  double* d_pts_x_final_zsc = nullptr;
  double* d_pts_x_final_zcs = nullptr;
  double* d_pts_x_final_lsc = nullptr;
  double* d_pts_x_final_lcs = nullptr;
  std::vector<std::uint8_t> pts_x_final_taken;
  int pts_v_size = -1;        // ns_con_local * mpol * (ntor+1) (per cfg)
  int pts_x_size = -1;        // ns_local * mpol * (ntor+1) (per cfg)
  int pts_x_ns = -1;          // ns_local (for the latest EnsurePTSBuffers call)
  bool pts_v_initialized = false;
  bool pts_x_initialized = false;
  // Each configuration's positions at the old ns, which the radial
  // interpolation of the stage transition upscales into the new d_pts_x.
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
  // d_scalxc at the old ns: the upscale multiplies by it, interpolates radially
  // and divides by the new scalxc, as the host's old_xc_scaled_.
  double* d_scalxc_prev = nullptr;
  int scalxc_prev_len = -1;
  bool scalxc_prev_valid = false;
  // True on the first time step after a reallocation or reset, which resets the
  // damping ring (iter_phase = 0).
  bool timestep_first_call_after_reset = true;
  // Raised when RecomposeToPhysicalCuda has written d_specs_block on the
  // device, so CudaForward skips the host transfer and clears it.
  bool specs_populated_from_device = false;

  // Per-configuration flags of the m = 1 Z-force zeroing in a batch, staged
  // from h_m1_zero_flags each iteration by DecomposeAndConstrainCuda.
  unsigned char* d_m1_zero_flags = nullptr;
  int m1_zero_flags_capacity = 0;
  std::vector<unsigned char> h_m1_zero_flags;

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

  // computeMHDForces outputs (force grid, ns_force_local x nZnT each).
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
  // Partial residual triples of the K sub-blocks per configuration, which the
  // finalize kernel reduces into d_residuals_partial.
  double* d_residuals_partials_K = nullptr;
  static constexpr int kResidualsKPartitions = 16;

  // Device time-step controller, per configuration: the 1/tau ring of kNDamp
  // samples, the previous fsq, and fac and b1.
  double* d_inv_tau = nullptr;
  double* d_prev_fsq = nullptr;
  double* d_fac_b1 = nullptr;
  // The evalFResPrecd normalization per configuration for k_update_timestep,
  // staged when it changes so the launch stays capturable.
  double* d_fnorm1 = nullptr;
  double fnorm1_staged = 0.0;
  bool fnorm1_staged_valid = false;
  // Set once k_rz_norm_per_cfg fills d_fnorm1 per configuration, after which
  // StageFnorm1 no longer broadcasts the host value.
  bool fnorm1_device_filled = false;
  // kNDamp matches FlowControl::kNDamp on host (10 by convention; if the
  // host constant changes, this needs to match).
  static constexpr int kTimestepNDamp = 10;

  // Batched-input staging, once per run; members rather than statics so
  // ResetForNewVmecRun rearms it.
  double* batch_inputs_pinned = nullptr;
  int batch_inputs_n_cfg = 0;
  size_t batch_inputs_one_spec_doubles = 0;
  int batch_inputs_loaded = -1;  // -1 unread, 0 absent, 1 loaded
  bool batch_inputs_consumed = false;

  // ComputeForceNorms: 2 device scalars [sum_rz, sum_l] after per-jH
  // reduction. Replaces the ns_h-D2H + host accumulator.
  double* d_fnorm_scalars = nullptr;

  // The RZ transpose runs when the preconditioner changes, detected by ar[0]
  // differing from this sentinel.
  double rz_cache_ar_sentinel = std::numeric_limits<double>::quiet_NaN();

  // Raw bytes allocated by the latest Reshape, which the pre-flight credits as
  // freed by the next one.
  long long reshape_budget_raw_bytes = 0;

  std::mutex mu;

  // Reallocates the device buffers for a new shape, sized for n_config_max
  // configurations; one configuration keeps the single-call layout.
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
    cuda_free_if((void*&)d_lambda_b_half);
    cuda_free_if((void*&)d_lambda_m0_c);
    cuda_free_if((void*&)d_lambda_m0_a);
    lambda_m0_stride_cached = -1;
    lambda_m0_ns_cached = -1;
    lambda_m0_ntor_cached = -1;
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
    // Snapshots the old d_scalxc before it is freed, so the upscale
    // interpolates in physical space as the host does; distinct batches only.
    {
      // d_scalxc still holds the old ns values; pts_x_initialized tells a stage
      // transition from a fresh run, whose scalxc is the prior run's.
      if (g_batch_distinct_run && d_scalxc && pts_x_initialized &&
          pts_x_ns > 0 && pts_x_ns != ns_local) {
        int scalxc_len_old = pts_x_ns * 2;
        size_t bytes_prev = sizeof(double) * (size_t)n_config_max *
                             (size_t)scalxc_len_old;
        if (d_scalxc_prev) cudaFree(d_scalxc_prev);
        cuda_check(cudaMalloc(&d_scalxc_prev, bytes_prev),
                   "alloc d_scalxc_prev");
        cuda_check(cudaMemcpyAsync(d_scalxc_prev, d_scalxc, bytes_prev,
                                    cudaMemcpyDeviceToDevice, stream),
                   "d2d scalxc -> scalxc_prev (Reshape snapshot)");
        scalxc_prev_len = scalxc_len_old;
        scalxc_prev_valid = true;
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
    // Snapshots each configuration's positions before d_pts_x is freed, for the
    // stage-transition upscale of a distinct batch.
    if (g_batch_distinct_run && pts_x_initialized && d_pts_x_rcc &&
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
                   "d2d pts_x -> pts_x_prev (Reshape snapshot)");
      }
      pts_x_prev_size = pts_x_size;
      pts_x_prev_ns = pts_x_ns;
      pts_x_prev_valid = true;
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
    cuda_free_if((void*&)d_pts_x_evaluated_rcc);
    cuda_free_if((void*&)d_pts_x_evaluated_rss);
    cuda_free_if((void*&)d_pts_x_evaluated_zsc);
    cuda_free_if((void*&)d_pts_x_evaluated_zcs);
    cuda_free_if((void*&)d_pts_x_evaluated_lsc);
    cuda_free_if((void*&)d_pts_x_evaluated_lcs);
    pts_x_evaluated_valid = false;
    cuda_free_if((void*&)d_pts_x_final_rcc);
    cuda_free_if((void*&)d_pts_x_final_rss);
    cuda_free_if((void*&)d_pts_x_final_zsc);
    cuda_free_if((void*&)d_pts_x_final_zcs);
    cuda_free_if((void*&)d_pts_x_final_lsc);
    cuda_free_if((void*&)d_pts_x_final_lcs);
    pts_x_final_taken.clear();
    pts_v_size = -1;
    pts_x_size = -1;
    pts_v_initialized = false;
    pts_x_initialized = false;
    cuda_free_if((void*&)d_m1_zero_flags);
    m1_zero_flags_capacity = 0;
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
    // Frees the flag and mask buffers sized by n_config_max, so a run with
    // another configuration count reallocates them.
    cuda_free_if((void*&)d_conv_flag);
    pinned_free_if((void*&)h_conv_flag_pinned);
    cuda_free_if((void*&)d_restart_mask);
    cuda_free_if((void*&)d_copy_mask);
    cuda_free_if((void*&)d_time_step_per_cfg);
    cuda_free_if((void*&)d_phase_per_cfg);
    cuda_free_if((void*&)d_edge_per_cfg);
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
    // Reshape may move the pointers the captured graphs reference, so they are
    // destroyed and recaptured lazily.
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

    // Specs block of the six spectra, xmpq and sqrtSF, per configuration except
    // the shared xmpq.
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

    // FFT scratch X and Y for n_config_max configurations, which the cuFFT
    // batch covers.
    fft_x_elems = (size_t)n_config_max * ns_local * mpol * kBatch * nhalf;
    fft_y_elems = (size_t)n_config_max * ns_local * mpol * kBatch * nZeta;
    size_t X_bytes = sizeof(cufftDoubleComplex) * fft_x_elems;
    size_t Y_bytes = sizeof(double) * fft_y_elems;
    cuda_check(cudaMalloc(&d_X, X_bytes), "alloc X");
    cuda_check(cudaMalloc(&d_Y, Y_bytes), "alloc Y");

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
    // The next Reshape frees this much before a new run allocates, which
    // CudaVramBudgetCuda credits against the free memory.
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

  // Toroidal basis cosnv and sinnv, staged once per shape.
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
        // Zero-fills the new buffer, so slots no kernel writes read zero and
        // not a NaN bit pattern that would reach RecomposeToPhysicalCuda.
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
  // Shares jac_array_bytes (same ns_h x nZnT shape).
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
      // All entries start at 0.15 / time_step, as the host invTau_ does at
      // iter1 == iter2.
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

  // Refreshes the device fnorm1 when the host value changes; unchanged calls
  // enqueue nothing, keeping the controller capturable.
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
    if (!d_conv_flag) {
      // Device and pinned-host convergence flags: k_check_convergence writes a
      // byte per configuration, copied asynchronously for polling.
      cuda_check(cudaMalloc(&d_conv_flag,
                             sizeof(std::uint8_t) * n_config_max),
                 "alloc d_conv_flag");
      cuda_check(cudaMallocHost(&h_conv_flag_pinned,
                                 sizeof(std::uint8_t) * n_config_max),
                 "alloc h_conv_flag_pinned");
    }
  }
  std::uint8_t* d_conv_flag = nullptr;
  std::uint8_t* h_conv_flag_pinned = nullptr;
  // lamscale cached by ComputeForceNormsCuda for k_check_convergence; zero
  // until seen this run, when the kernel compares raw residuals.
  double lamscale_cached = 0.0;

  void EnsureFnormScalarsBuffer() {
    if (!d_fnorm_scalars) {
      cuda_check(cudaMalloc(&d_fnorm_scalars, 2 * sizeof(double) * n_config_max),
                 "alloc d_fnorm_scalars");
    }
  }

  // Device active mask, one byte per configuration; mask-aware kernels return
  // early for a zero byte, and nullptr means all active.
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

  // Per-configuration restart mask gating RestorePtsXFromBackupPerCfgCuda;
  // nullptr until EnsureRestartMaskBuffer.
  std::uint8_t* d_restart_mask = nullptr;
  void EnsureRestartMaskBuffer() {
    if (!d_restart_mask) {
      cuda_check(cudaMalloc(&d_restart_mask,
                             sizeof(std::uint8_t) * n_config_max),
                 "alloc d_restart_mask");
    }
  }

  // Per-configuration restart control: the snapshot and store mask, each
  // configuration's time step and ring flag, and its edge-force mask.
  std::uint8_t* d_copy_mask = nullptr;
  double* d_time_step_per_cfg = nullptr;
  std::uint8_t* d_phase_per_cfg = nullptr;
  std::uint8_t* d_edge_per_cfg = nullptr;
  void EnsurePerCfgControlBuffers() {
    if (!d_copy_mask) {
      cuda_check(cudaMalloc(&d_copy_mask, sizeof(std::uint8_t) * n_config_max),
                 "alloc d_copy_mask");
    }
    if (!d_time_step_per_cfg) {
      cuda_check(
          cudaMalloc(&d_time_step_per_cfg, sizeof(double) * n_config_max),
          "alloc d_time_step_per_cfg");
    }
    if (!d_phase_per_cfg) {
      cuda_check(
          cudaMalloc(&d_phase_per_cfg, sizeof(std::uint8_t) * n_config_max),
          "alloc d_phase_per_cfg");
    }
    if (!d_edge_per_cfg) {
      cuda_check(
          cudaMalloc(&d_edge_per_cfg, sizeof(std::uint8_t) * n_config_max),
          "alloc d_edge_per_cfg");
    }
  }

  void EnsureLambdaPrecondBuffers(int ns_h, int ns_con_local, int mpol, int ntor) {
    // Per-configuration lambda preconditioner stride ns_con_local + 1, the slot
    // the full-grid average reads at its last surface.
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

  // Buffers of the tridiagonal m = 0 lambda preconditioner, zero-filled on
  // allocation so the unwritten end slots of the half-grid profile read zero.
  void EnsureLambdaM0Buffers(int lambda_stride, int ns, int ntor) {
    if (d_lambda_b_half && lambda_m0_stride_cached == lambda_stride &&
        lambda_m0_ns_cached == ns && lambda_m0_ntor_cached == ntor) {
      return;
    }
    auto realloc_zero = [this](double*& p, size_t bytes) {
      if (p) {
        cudaFree(p);
        p = nullptr;
      }
      cuda_check(cudaMalloc(&p, bytes), "alloc lambda m0 buf");
      cuda_check(cudaMemsetAsync(p, 0, bytes, stream), "zero lambda m0 buf");
    };
    const size_t rows = (size_t)n_config_max * (size_t)(ntor + 1);
    realloc_zero(d_lambda_b_half,
                 sizeof(double) * (size_t)n_config_max * lambda_stride);
    realloc_zero(d_lambda_m0_c, sizeof(double) * rows * ns);
    realloc_zero(d_lambda_m0_a, sizeof(double) * rows * ns);
    lambda_m0_stride_cached = lambda_stride;
    lambda_m0_ns_cached = ns;
    lambda_m0_ntor_cached = ntor;
  }

  void EnsurePrecondMatrixBuffers(int ns_h, int ns_force_local, int ns_local,
                                    int nZnT) {
    auto alloc_if_null = [](double*& p, size_t bytes) {
      if (!p) cuda_check(cudaMalloc(&p, bytes), "alloc precond mat buf");
    };
    // Per-configuration preconditioning-matrix scratch and outputs; sm and sp
    // are sized per configuration for a uniform layout.
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
    // Per-side snapshots of the preconditioning matrices, laid out as their
    // scratch with the configuration axis, for the RZ assembly.
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

  // Zeroes the preconditioning-matrix snapshots and RZ outputs for a new
  // Vmec::run, which survive across Vmec instances on the device.
  void ResetForNewVmecRun() {
    std::lock_guard<std::mutex> lk(mu);
    if (!stream) return;
    // The next run loads its positions and velocity from the host afresh
    // instead of reading the previous run's device slots.
    pts_x_initialized = false;
    pts_v_initialized = false;
    pts_x_backup_initialized = false;
    pts_x_evaluated_valid = false;
    // The input-derived profiles can change between runs of one shape without a
    // Reshape, so the next run restages each.
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
    seg2_in_capture = false;
    seg3_in_capture = false;
    seg4_in_capture = false;
    seg2_warmup_calls = 0;
    seg3_warmup_calls = 0;
    seg4_warmup_calls = 0;
  }

  void EnsureConstraintMultiplierBuffers(int ns_force_local, int ns_con_local,
                                          int nZnT) {
    // Zero-fills on allocation, as the CPU arrays are, since writers skip rows
    // outside their domain such as the axis or LCFS row.
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
    // Zero-fills the decomposed-force buffers so slots no kernel writes read
    // zero and not a NaN bit pattern.
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
    // On a stage transition of a distinct batch, snapshots each configuration's
    // positions for the per-configuration upscale.
    bool snapshot = (g_batch_distinct_run && pts_x_initialized && d_pts_x_rcc &&
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
                   "d2d pts_x -> pts_x_prev (multigrid snapshot)");
      }
      pts_x_prev_size = pts_x_size;
      pts_x_prev_ns = pts_x_ns;
      pts_x_prev_mpol = mpol;
      pts_x_prev_ntor = ntor;
      pts_x_prev_valid = true;
    }
    auto free_if = [](double*& p) { if (p) { cudaFree(p); p = nullptr; } };
    free_if(d_pts_v_rcc); free_if(d_pts_v_rss);
    free_if(d_pts_v_zsc); free_if(d_pts_v_zcs);
    free_if(d_pts_v_lsc); free_if(d_pts_v_lcs);
    free_if(d_pts_x_rcc); free_if(d_pts_x_rss);
    free_if(d_pts_x_zsc); free_if(d_pts_x_zcs);
    free_if(d_pts_x_lsc); free_if(d_pts_x_lcs);
    // The backup buffers follow the new position size, so they are freed and
    // reallocated lazily.
    free_if(d_pts_x_backup_rcc); free_if(d_pts_x_backup_rss);
    free_if(d_pts_x_backup_zsc); free_if(d_pts_x_backup_zcs);
    free_if(d_pts_x_backup_lsc); free_if(d_pts_x_backup_lcs);
    pts_x_backup_initialized = false;
    free_if(d_pts_x_evaluated_rcc); free_if(d_pts_x_evaluated_rss);
    free_if(d_pts_x_evaluated_zsc); free_if(d_pts_x_evaluated_zcs);
    free_if(d_pts_x_evaluated_lsc); free_if(d_pts_x_evaluated_lcs);
    pts_x_evaluated_valid = false;
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
    rz_mnsize_cached = mnsize;
    rz_ns_total_cached = ns_total;
    rz_num_basis_cached = num_basis;
    rz_cache_ar_sentinel = std::numeric_limits<double>::quiet_NaN();
  }

  // deAliasConstraintForce buffers, allocated once; StageToroidalBasis stages
  // cosnv and sinnv at Reshape.
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

  // rzConIntoVolume buffers (ns_con_local x nZnT each for rCon0, zCon0).
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
    // Per-configuration rCon0 and zCon0 start at zero, as the host arrays at a
    // stage where rzConIntoVolume does not run.
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
    // The run uses the first visible device; CUDA_VISIBLE_DEVICES selects
    // it, one process per device.
    const int device_index = 0;
    cuda_check(cudaSetDevice(device_index), "cudaSetDevice");
    n_cached = n;
    nfp_cached = nfp;
    mpol_cached = mpol;
    initialized = true;
  }
};

CudaToroidalState& State();
// Backup-mirror arming helper (defined in fft_toroidal_cuda_io.cu).
void EnsurePTSBackupBuffers(CudaToroidalState& S);

// Kernel declarations (definitions in fft_toroidal_cuda_kernels.cu).
__global__ void k_fill_spectra(
    int n_config, int ns_local, int mpol, int ntor, int nhalf, int nfp,
    int nsMinF1_offset,
    const double* __restrict__ rmncc, const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc, const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc, const double* __restrict__ lmncs,
    const double* __restrict__ nscale, cufftDoubleComplex* __restrict__ X);
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
    const std::uint8_t* __restrict__ d_active_per_cfg,
    const double* __restrict__ d_time_step_per_cfg);
__global__ void k_extract_geom_scalars(
    const double* __restrict__ d_r1_e,
    const double* __restrict__ d_r1_o,
    const double* __restrict__ d_z1_e,
    int outer_idx, int inner_idx, double* __restrict__ d_out);
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
                                double* __restrict__ lambdaPreconditioner,
                                const std::uint8_t* __restrict__ d_active_per_cfg);
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
__global__ void k_pres_compute_and_thermal(int n_config, int ns_h, double gamma,
                                            int nsMinH, int nsMaxH_minus_1,
                                            int ns_minus_2,
                                            const double* __restrict__ massH,
                                            const double* __restrict__ dVdsH,
                                            double* __restrict__ presH,
                                            double* __restrict__ thermal_partial);
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
__global__ void k_volume_reduce(int n_config, int ns_h, double multiplier,
                                  int nsMaxH_minus_1, int ns_minus_2,
                                  int nsMinH,
                                  const double* __restrict__ dVdsH,
                                  double* __restrict__ out_scalar,
                                  const std::uint8_t* __restrict__ d_active_per_cfg);
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
                                                double* __restrict__ tcon,
                                                const std::uint8_t* __restrict__ d_active_per_cfg);
__global__ void k_halve_tcon_lcfs(int n_config, int tcon_stride,
                                    int last_idx,
                                    double* __restrict__ tcon,
                                    const std::uint8_t* __restrict__ d_active_per_cfg);
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
                                  double scalingFactor, double sigma,
                                  double* __restrict__ dec_frss,
                                  double* __restrict__ dec_fzcs);
__global__ void k_m1_constraint_gated(int n_config, int ns_local,
                                      int ns_force_local, int mpol, int ntor,
                                      double scalingFactor, double sigma,
                                      const unsigned char* __restrict__ zero_flags,
                                      int zero_all,
                                      double* __restrict__ dec_frss,
                                      double* __restrict__ dec_fzcs);
__global__ void k_mask_geometry_above(int n_config, size_t cfg_stride,
                                      int ns_count, int mpol, int ntor,
                                      int mpol_geometry, int ntor_geometry,
                                      int lthreed,
                                      double* __restrict__ rcc,
                                      double* __restrict__ rss,
                                      double* __restrict__ zsc,
                                      double* __restrict__ zcs);
__global__ void k_pres_final_reduce(int n_config, int ns_h, double deltaS,
                                     double adiabaticIndex,
                                     const double* __restrict__ thermal_partial,
                                     const double* __restrict__ magnetic_partial,
                                     double* __restrict__ scalars_out,
                                     const std::uint8_t* __restrict__ d_active_per_cfg);
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
    const std::uint8_t* __restrict__ d_active_per_cfg,
    int jLocal_max_rz_edge,
    const std::uint8_t* __restrict__ edge_per_cfg);
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
    const std::uint8_t* __restrict__ edge_per_cfg);
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
    const std::uint8_t* __restrict__ d_active_per_cfg,
    const double* __restrict__ d_time_step_per_cfg,
    const std::uint8_t* __restrict__ d_phase_per_cfg);
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
// One block per (config, mn) row with the elimination ratios in dynamic shared
// memory, for ns_total beyond the PCR solver's 1024.
__global__ void k_apply_rz_thomas_block(
    int n_config, int mnsize, int ns_total, int num_basis,
    const int* __restrict__ jMin, int jMax,
    const double* __restrict__ a_in, const double* __restrict__ d_in,
    const double* __restrict__ b_in, double* __restrict__ c_inout,
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

// Kernels specialized for compile-time (MPOL, NTOR), unrolling the m and n
// loops; instantiated for the production shape mpol = ntor = 10.
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

// Run-scoped and process-scoped globals (defined in fft_toroidal_cuda_state.cu).
extern bool g_pts_init_only;
extern int g_batch_mem_shape[4];
extern int g_batch_outputs_shape[4];
extern int g_residuals_k_run;
extern std::vector<double> g_batch_dec_x_mem;
extern std::vector<double> g_batch_inputs_mem;
extern std::vector<double> g_batch_outputs_mem;
extern std::vector<double> g_batch_phipF, g_batch_phipH, g_batch_currH,
    g_batch_iotaH, g_batch_massH;
extern int g_batch_prof_ncfg, g_batch_prof_nsh, g_batch_prof_nsf;

}  // namespace vmecpp

#endif  // VMECPP_USE_CUDA

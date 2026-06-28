// CUDA-accelerated implementations of the VMEC++ IdealMhdModel iteration
// body. The interface declared in this header is conditionally compiled
// when the build option VMECPP_USE_CUDA is enabled; consumers obtain the
// CUDA paths by including this header and dispatching at the call sites
// in IdealMhdModel and Vmec::run.
//
// The CUDA implementations are intended as drop-in replacements for the
// equivalent CPU routines (FourierToReal3DSymmFastPoloidalFft,
// ForcesToFourier3DSymmFastPoloidalFft, and the per-quantity update
// functions of IdealMhdModel). Each function preserves the data interface
// of its CPU counterpart so that the surrounding control flow remains
// unchanged.
//
// State that persists across iterations (cuFFT plans, persistent device
// buffers, precomputed basis tables, and the per-configuration shadow
// vectors used by the batched execution mode) is held in a thread-local
// CudaToroidalState instance whose lifetime spans the host thread that
// drives the iteration body. The state is reset by ResetCudaStateForNewVmecRun
// at the start of each Vmec::run invocation so that successive runs within
// a single process do not inherit stale device buffers from a prior run.
#pragma once

#ifdef VMECPP_USE_CUDA

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {

// CUDA-accelerated geometry-from-Fourier toroidal DFT. Consumes the
// spectral representation of the plasma boundary in physical_x and produces
// the real-space geometry components (R, Z, lambda and their poloidal and
// toroidal derivatives, in even and odd parity decompositions) into the
// RealSpaceGeometry output. The semantics match the CPU routine
// FourierToReal3DSymmFastPoloidalFft; the cuFFT plan and the persistent
// device buffers required for the radial-batched inverse DFT are managed
// internally by the thread-local CudaToroidalState.
//
// The CUDA path does not consult the host-resident ToroidalFftPlans, so the
// signature here drops it. The CPU FFTX path keeps its own signature with
// ToroidalFftPlans in fft_toroidal.h; ideal_mhd_model.cc dispatches between
// the two via the macro chain.
void FourierToReal3DSymmFastPoloidalCuda(
    const FourierGeometry& physical_x, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, RealSpaceGeometry& m_geometry);

// Iterative-refinement precision signal. ideal_mhd_model.cc calls this
// after every evalFResInvar with the current sum (fsqr + fsqz + fsql).
// The fft_toroidal_cuda dispatchers compare against the env-controlled
// threshold (VMECPP_IR_THRESHOLD, default 1e-5) and switch hot kernels
// between FP32/TF32 (residual above threshold) and FP64 (below) so the
// descent runs cheap while the final refinement reaches ftol=1e-15.
void SetIRResidualSum(double sum);
int GetIRPhase();  // 1 = FP32 phase, 0 = FP64 phase

// CUDA-accelerated forces-to-Fourier toroidal DFT. Consumes the real-space
// MHD force residuals in RealSpaceForces and produces the spectral
// representation of the force in FourierForces. Semantically equivalent to
// the CPU routine ForcesToFourier3DSymmFastPoloidalFft. Free-boundary mode
// is signalled via the vacuum_pressure_state argument, which selects the
// appropriate boundary-edge contribution.
void ForcesToFourier3DSymmFastPoloidalCuda(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

// CUDA port of IdealMhdModel::computeJacobian. Consumes the device-resident
// real-space geometry buffers (r1_e/o, ru_e/o, z1_e/o, zu_e/o) populated by
// the preceding FourierToReal3DSymmFastPoloidalCuda call and writes the
// half-grid quantities r12, ru12, zu12, rs, zs, and tau. The bad_jacobian
// out-parameter is set to true when the half-grid Jacobian tau changes
// sign across the grid (the product of minimum and maximum is negative) or
// contains non-finite values; the caller uses this flag to drive the
// magnetic-axis recomputation and time-step-reduction recovery logic.
//
// The optional signOfJacobian and wInt arguments are consumed only when
// the three-way fusion of jacobian, metric, and dVdsH kernels is enabled
// via the VMECPP_JAC_METRIC_DVDSH_FUSE environment variable. With fusion
// active, the dVdsH output that ComputeJacobianCuda would otherwise leave
// to a subsequent UpdateDifferentialVolumeCuda call is produced inline,
// and the subsequent call becomes a no-op.
void ComputeJacobianCuda(const RadialPartitioning& r, const Sizes& s,
                         const Eigen::VectorXd& sqrtSH, double deltaS,
                         double dSHalfDsInterp, Eigen::VectorXd& r12,
                         Eigen::VectorXd& ru12, Eigen::VectorXd& zu12,
                         Eigen::VectorXd& rs, Eigen::VectorXd& zs,
                         Eigen::VectorXd& tau, bool& bad_jacobian,
                         int signOfJacobian = 0,
                         const Eigen::VectorXd* wInt = nullptr);

void ComputeMetricElementsCuda(const RadialPartitioning& r, const Sizes& s,
                               const Eigen::VectorXd& sqrtSF,
                               const Eigen::VectorXd& sqrtSH,
                               Eigen::VectorXd& gsqrt, Eigen::VectorXd& guu,
                               Eigen::VectorXd& guv, Eigen::VectorXd& gvv);

void UpdateDifferentialVolumeCuda(const RadialPartitioning& r, const Sizes& s,
                                  double signOfJacobian,
                                  const Eigen::VectorXd& wInt,
                                  Eigen::VectorXd& dVdsH);

void ComputeBCoCuda(const RadialPartitioning& r, const Sizes& s,
                    const Eigen::VectorXd& guu, const Eigen::VectorXd& guv,
                    const Eigen::VectorXd& gvv, const Eigen::VectorXd& bsupu,
                    const Eigen::VectorXd& bsupv, Eigen::VectorXd& bsubu,
                    Eigen::VectorXd& bsubv);

void RadialForceBalanceCuda(
    const RadialPartitioning& r, const Sizes& s, double signOfJacobian,
    double deltaS, const Eigen::VectorXd& wInt, const Eigen::VectorXd& presH,
    const Eigen::VectorXd& chipF, const Eigen::VectorXd& phipF,
    Eigen::VectorXd& bucoH, Eigen::VectorXd& bvcoH, Eigen::VectorXd& jcurvF,
    Eigen::VectorXd& jcuruF, Eigen::VectorXd& presgradF, Eigen::VectorXd& dVdsF,
    Eigen::VectorXd& equiF);

void RzConIntoVolumeCuda(const RadialPartitioning& r, const Sizes& s,
                         const FlowControl& fc, Eigen::VectorXd& rCon0,
                         Eigen::VectorXd& zCon0);

void ComputeBContraCuda(const RadialPartitioning& r, const Sizes& s,
                        const FlowControl& fc, int ncurr, double lamscale,
                        const Eigen::VectorXd& phipF,
                        const Eigen::VectorXd& phipH,
                        const Eigen::VectorXd& currH,
                        const Eigen::VectorXd& iotaH_in, Eigen::VectorXd& bsupu,
                        Eigen::VectorXd& bsupv, Eigen::VectorXd& chipH_out,
                        Eigen::VectorXd& iotaH_out, Eigen::VectorXd& chipF_out,
                        Eigen::VectorXd& iotaF_out);

void ComputeInitialVolumeCuda(const RadialPartitioning& r,
                              const FlowControl& fc, double deltaS,
                              double& voli_out);

void UpdateVolumeCuda(const RadialPartitioning& r, const FlowControl& fc,
                      double deltaS, double& plasmaVolume_out);

void PressureAndEnergiesCuda(const RadialPartitioning& r, const Sizes& s,
                             const FlowControl& fc, double deltaS,
                             double adiabaticIndex,
                             const Eigen::VectorXd& massH,
                             Eigen::VectorXd& presH_out,
                             Eigen::VectorXd& totalPressure_out,
                             double& thermalEnergy_out,
                             double& magneticEnergy_out, double& mhdEnergy_out);

void HybridLambdaForceCuda(const RadialPartitioning& r, const Sizes& s,
                           double lamscale,
                           const Eigen::VectorXd& radialBlending,
                           Eigen::VectorXd& blmn_e, Eigen::VectorXd& blmn_o,
                           Eigen::VectorXd& clmn_e, Eigen::VectorXd& clmn_o);

void ComputeForceNormsCuda(const RadialPartitioning& r, const Sizes& s,
                           const FlowControl& fc, double magneticEnergy,
                           double thermalEnergy, double plasmaVolume,
                           double lamscale, double forceNorm1_host,
                           double& fNormRZ_out, double& fNormL_out,
                           double& fNorm1_out);

void ComputeMHDForcesCuda(const RadialPartitioning& r, const Sizes& s,
                          const FlowControl& fc, bool lfreeb, double deltaS,
                          Eigen::VectorXd& armn_e, Eigen::VectorXd& armn_o,
                          Eigen::VectorXd& azmn_e, Eigen::VectorXd& azmn_o,
                          Eigen::VectorXd& brmn_e, Eigen::VectorXd& brmn_o,
                          Eigen::VectorXd& bzmn_e, Eigen::VectorXd& bzmn_o,
                          Eigen::VectorXd& crmn_e, Eigen::VectorXd& crmn_o,
                          Eigen::VectorXd& czmn_e, Eigen::VectorXd& czmn_o);

void UpdateLambdaPreconditionerCuda(const RadialPartitioning& r, const Sizes& s,
                                    double dampingFactor, double lamscale,
                                    double* bLambda_out, double* dLambda_out,
                                    double* cLambda_out,
                                    double* lambdaPreconditioner_host);

// CUDA port of IdealMhdModel::computePreconditioningMatrix. Computes the
// preconditioning-matrix coefficient arrays for one of the two coordinate
// directions (R or Z) selected by the side argument: side == 0 populates
// the R-side device buffers d_pmat_arm, d_pmat_brm, d_pmat_ard, and
// d_pmat_brd; side == 1 populates the Z-side equivalents d_pmat_azm,
// d_pmat_bzm, d_pmat_azd, and d_pmat_bzd. The shared coefficient buffer
// d_pmat_cxd is written by both invocations with identical content, which
// is consistent with the upstream definition of the radial preconditioner.
// The wrapper is invoked twice per preconditioner-update interval, once
// per coordinate direction; downstream consumers (AssembleRZPreconditionerCuda
// and the apply-RZ Thomas/PCR path) read all three coefficient sets to
// construct the tridiagonal system that drives the per-iteration apply
// step.
void ComputePreconditioningMatrixCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double deltaS, int kEvenParity, int kOddParity, const Eigen::VectorXd& xs,
    const Eigen::VectorXd& xu12, const Eigen::VectorXd& xu_e,
    const Eigen::VectorXd& xu_o, const Eigen::VectorXd& x1_o,
    const Eigen::VectorXd& sm, const Eigen::VectorXd& sp,
    Eigen::VectorXd& m_axm, Eigen::VectorXd& m_axd, Eigen::VectorXd& m_bxm,
    Eigen::VectorXd& m_bxd, Eigen::VectorXd& m_cxd, int side);

// Device-side assembly of the tridiagonal coefficients consumed by the
// RZ-preconditioner apply step. Reads the persistent
// preconditioning-matrix buffers populated by the two
// ComputePreconditioningMatrixCuda invocations of the most recent
// preconditioner-update interval and produces the coefficient arrays
// d_rz_aR, d_rz_dR, d_rz_bR (R direction), d_rz_aZ, d_rz_dZ, d_rz_bZ
// (Z direction), and the per-mode lower-bound array d_rz_jMin, all in
// the (mn, jF_global) transposed layout consumed by the parallel-cyclic-
// reduction solver in ApplyRZPreconditionerCuda. This function replaces
// the per-iteration host-side assembly loop and the six host-to-device
// transfers that the apply-RZ path would otherwise perform on every
// preconditioner cache miss. The jMax argument supplies the upper bound
// of the radial loop and equals (ns - 1) in fixed-boundary mode or ns in
// free-boundary mode with an active vacuum-pressure state.
void AssembleRZPreconditionerCuda(const RadialPartitioning& r, const Sizes& s,
                                  const FlowControl& fc, int jMax);

// Computes the full-grid radial derivatives ruFull and zuFull by combining
// the even and odd-parity components on the constraint range:
// ruFull = ru_e + sqrtSF * ru_o, zuFull = zu_e + sqrtSF * zu_o. Invoked
// once per iteration after FourierToReal3DSymmFastPoloidalCuda. The
// outputs are transferred to the host so that downstream CPU paths that
// still consume the full-grid derivatives (notably the boundary-condition
// evaluators in the free-boundary path) observe the correct values.
void ComputeRuZuFullCuda(const RadialPartitioning& r, const Sizes& s,
                         Eigen::VectorXd& ruFull, Eigen::VectorXd& zuFull);

void ConstraintForceMultiplierCuda(const RadialPartitioning& r, const Sizes& s,
                                   const FlowControl& fc, double tcon0,
                                   const Eigen::VectorXd& ard,
                                   const Eigen::VectorXd& azd,
                                   Eigen::VectorXd& tcon_out);

void EffectiveConstraintForceCuda(const RadialPartitioning& r, const Sizes& s,
                                  Eigen::VectorXd& gConEff_out);

// vacuum_edge applies the staged rBSq profile (StageRbsqCuda) to the
// LCFS force row ahead of the constraint assembly; pass true only when
// the vacuum pressure contribution is initialized or active.
void AssembleTotalForcesCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    const Eigen::VectorXd& gCon, Eigen::VectorXd& brmn_e_out,
    Eigen::VectorXd& brmn_o_out, Eigen::VectorXd& bzmn_e_out,
    Eigen::VectorXd& bzmn_o_out, Eigen::VectorXd& frcon_e_out,
    Eigen::VectorXd& frcon_o_out, Eigen::VectorXd& fzcon_e_out,
    Eigen::VectorXd& fzcon_o_out, bool vacuum_edge);

// Free-boundary bridges. The NESTOR vacuum solve stays on the host;
// these carry the per-iteration traffic between the device-resident
// iteration state and the vacuum block in IdealMhdModel::update.
//
// Records whether the vacuum edge force is active this iteration; the
// segment-3 graph recaptures when the value changes.
void SetVacuumEdgeCuda(int active);
// Marks the current Vmec::run as free-boundary; the segment and
// whole-iteration CUDA graphs are disabled for the run. Reset by
// ResetCudaStateForNewVmecRun.
void SetFreeBoundaryRunCuda(int enabled);
// Scales the device rCon0/zCon0 volume profiles in place.
void ScaleRZCon0Cuda(double factor);
// One consolidated D2H flush per vacuum iteration and configuration:
// the axis and LCFS geometry rows, the outermost totalPressure rows,
// and the presH/bucoH/bvcoH profiles consumed by the edge-pressure
// extrapolation and the toroidal-current scalars, with a single
// synchronize. The host destinations are the single-configuration
// arrays; the batched vacuum loop consumes one configuration at a time.
void FlushVacuumHostDataCuda(int cfg, const RadialPartitioning& r,
                             const Sizes& s, Eigen::VectorXd& m_r1_e,
                             Eigen::VectorXd& m_r1_o, Eigen::VectorXd& m_z1_e,
                             Eigen::VectorXd& m_totalPressure,
                             Eigen::VectorXd& m_presH, Eigen::VectorXd& m_bucoH,
                             Eigen::VectorXd& m_bvcoH);
// H2D stage of one configuration's host-computed rBSq profile for the
// edge application.
void StageRbsqCuda(int cfg, const Eigen::VectorXd& rBSq);

// Device-side composition of the three steps that bridge the physical
// force representation produced by the inverse FFT
// (S.d_frcc, S.d_frss, S.d_fzsc, S.d_fzcs, S.d_flsc, S.d_flcs) to the
// parity-typed decomposed force representation consumed by the
// preconditioner-apply chain (S.d_decomposed_frcc and the analogous
// components). The composition applies the scalxc weighting in the
// decomposeInto step, enforces the m1 constraint on the resulting
// poloidal-mode-one coefficients via the supplied m1ScalingFactor, and
// zeroes the Z-force component at poloidal mode one to maintain the
// quasi-polar boundary parameterization. The decomposed force values stay
// in the device shadow buffers; host m_decomposed_f copies are refreshed
// through the consolidated FlushDecomposedToHostCuda entry point.
void DecomposeAndConstrainCuda(const RadialPartitioning& r, const Sizes& s,
                               const FlowControl& fc, double m1ScalingFactor,
                               const Eigen::VectorXd& scalxc,
                               double* dec_frcc_host, double* dec_frss_host,
                               double* dec_fzsc_host, double* dec_fzcs_host,
                               double* dec_flsc_host, double* dec_flcs_host);

// CUDA port of IdealMhdModel::applyM1Preconditioner. Scales the
// poloidal-mode-one components of the radial and Z force-spectrum
// coefficients (frss and fzcs under the three-dimensional configuration)
// by the force-ratio derived from the preconditioner-matrix coefficients
// ard, brd, azd, and bzd. The transformation maintains the consistency
// of the m1 constraint imposed by the quasi-polar boundary
// parameterization across the preconditioner-apply step.
void ApplyM1PreconditionerCuda(const RadialPartitioning& r, const Sizes& s,
                               const Eigen::VectorXd& ard,
                               const Eigen::VectorXd& brd,
                               const Eigen::VectorXd& azd,
                               const Eigen::VectorXd& bzd, double* frss_host,
                               double* fzcs_host);

// CUDA port of IdealMhdModel::applyLambdaPreconditioner. Scales the
// lambda-force coefficient buffers element-wise by the radial-and-modal
// lambdaPreconditioner array that
// IdealMhdModel::updateLambdaPreconditioner populates each
// preconditioner-update interval.
void ApplyLambdaPreconditionerCuda(const RadialPartitioning& r, const Sizes& s,
                                   const Eigen::VectorXd& lambdaPreconditioner,
                                   double* flsc_host, double* flcs_host);

// CUDA port of IdealMhdModel::applyRZPreconditioner. Solves the
// tridiagonal system that the preconditioner-update path has assembled
// (via AssembleRZPreconditionerCuda) for each (m, n) Fourier mode and
// applies the result to the four radial and Z force-spectrum coefficient
// arrays. The implementation uses a parallel-cyclic-reduction solver that
// processes one tridiagonal system per CUDA block; the per-mode jMin
// array supplies the lower bound of each system, and jMaxRZ supplies the
// upper bound. The host pointer arguments receive the post-apply force
// coefficients via the consolidated FlushDecomposedToHostCuda call later
// in the iteration.
int ApplyRZPreconditionerCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    const Eigen::VectorXd& ar, const Eigen::VectorXd& dr,
    const Eigen::VectorXd& br_in, const Eigen::VectorXd& az,
    const Eigen::VectorXd& dz, const Eigen::VectorXd& bz_in,
    const int* jMin_arr, int jMin_size, int jMaxRZ, double* frcc_host,
    double* frss_host, double* fzsc_host, double* fzcs_host);

// CUDA port of IdealMhdModel::deAliasConstraintForce. Executes a
// poloidal-and-toroidal Fourier round-trip on the constraint-force buffer:
// the constraint force is transformed to real space, scaled by the tcon
// constraint-force-multiplier produced by ConstraintForceMultiplierCuda
// and by the faccon mode-amplitude weights, and transformed back to
// spectral space. The procedure attenuates high-frequency aliasing
// content in the constraint contribution without affecting the
// low-frequency physical components.
void DeAliasConstraintForceCuda(const RadialPartitioning& r,
                                const FourierBasisFastPoloidal& fb,
                                const Sizes& s, const Eigen::VectorXd& faccon,
                                Eigen::VectorXd& m_gCon_host);

// Device-side mirror of FourierForces::residuals. Operates on the device
// shadow buffers S.d_decomposed_* populated by the inverse-FFT path and
// produces the three sum-of-squares residual components [fResR, fResZ,
// fResL]. The reduction range honors the same jMaxRZ logic (modified by
// includeEdgeRZForces) and jMaxIncludeBoundary range as the equivalent
// CPU loop, ensuring that the residual sum-of-squares is taken over the
// identical set of radial surfaces.
//
// The is_precd argument selects between two synchronization regimes. The
// invariant residual call (is_precd == false) synchronizes the device
// stream immediately after the reduction so that the host-visible output
// reflects the current iteration's geometry. This is required because the
// convergence test that drives multigrid stage termination consumes the
// invariant residual immediately. The preconditioned residual call
// (is_precd == true) defers the synchronization by using a cudaEvent and
// returns the prior iteration's reduced value. The deferral is acceptable
// because the preconditioned residual drives only the time-step controller
// and tau acceleration, both of which are tolerant of single-iteration
// staleness; the immediate convergence test for sign-flipped Jacobians
// is the responsibility of the subsequent ComputeJacobianCuda call and
// is not affected by the deferral.
void ResidualsCuda(const RadialPartitioning& r, const Sizes& s,
                   const FlowControl& fc, bool includeEdgeRZForces,
                   double& fResR_out, double& fResZ_out, double& fResL_out,
                   bool is_precd = false);

// =============================================================================
// Per-configuration accessor entry points for the batched execution mode.
// =============================================================================
// The batched kernels write per-configuration outputs to device buffers of
// size proportional to n_config_max. The entry points in this section
// transfer the full per-configuration arrays to the host so that the
// iteration controller can consume them when sizing convergence decisions
// per configuration. The transfer is performed via either an explicit
// device-to-host copy (for the direct accessors) or as a side effect of
// the existing per-iteration single-configuration copy (for the cached
// accessors). Under single-configuration execution the arrays remain
// available and contain a single slot whose contents match the equivalent
// scalar quantities returned by the single-cfg paths.

// Copy the per-configuration tau extrema from the device buffer populated
// by ComputeJacobianCuda into the supplied host vectors. The caller derives
// the bad-Jacobian flag for each configuration host-side as
// (minTau[c] * maxTau[c] < 0.0) || !std::isfinite(minTau[c] * maxTau[c]).
// The function must be called after ComputeJacobianCuda has queued the
// device-side reduction; invoking it earlier yields stale or undefined
// values.
void ComputeJacobianCudaPerCfgD2H(std::vector<double>* minTau_per_cfg,
                                  std::vector<double>* maxTau_per_cfg);

// Copy the per-configuration force-norm scalars from the device buffer
// populated by ComputeForceNormsCuda into the supplied host vectors. The
// caller derives the normalized residuals as
// fNormRZ[c] = 1.0 / (sum_rz[c] * energyDensity^2) and
// fNormL[c] = 1.0 / (sum_l[c] * lamscale^2). Requires a prior invocation
// of ComputeForceNormsCuda in the current iteration.
void ComputeForceNormsCudaPerCfgD2H(std::vector<double>* sumRZ_per_cfg,
                                    std::vector<double>* sumL_per_cfg);

// Copy the per-configuration residual triples populated by ResidualsCuda
// into the supplied host vectors. The vectors are resized to n_cfg and
// populated with the values [fResR_c, fResZ_c, fResL_c] for each
// configuration c. Requires a prior invocation of ResidualsCuda in the
// current iteration.
void ResidualsCudaPerCfgD2H(std::vector<double>* fResR_per_cfg,
                            std::vector<double>* fResZ_per_cfg,
                            std::vector<double>* fResL_per_cfg);

// Returns the effective n_config_max for the current process. The value is
// read once from the VMECPP_N_CONFIG_MAX environment variable and cached
// for the lifetime of the process; a return value of one indicates
// single-configuration execution. The accessor is invoked from Vmec::run
// at run initialization to size the per-configuration vectors in
// FlowControl before any CUDA kernel has fired, since the static cache
// inside CudaForward is populated only on the first kernel invocation.
int GetNConfigMaxCuda();

// Largest ns the CUDA radial tridiagonal solver supports on the active device.
// ns <= 1024 uses PCR; larger ns uses block-Thomas with the elimination ratios
// in dynamic shared memory, so the device's opt-in shared-memory capacity sets
// the ceiling (sizeof(double) per radial row). Returns 1024 if the device
// query fails.
int CudaMaxRadialResolution();

// Upper estimate of the persistent device allocation for one run at the
// given shape and configuration count, compared against the device's
// free memory. Returns true when the run fits (or when the device cannot
// be queried, in which case the allocations themselves decide); fills
// needed_bytes and free_bytes for the caller's diagnostics. Vmec::run
// consults this before the first allocation so an oversized batch is
// rejected with a budget message instead of failing mid-iteration.
bool CudaVramBudgetCuda(long long n_cfg, long long ns, long long mpol,
                        long long ntor, long long nZeta, long long nThetaEff,
                        long long* needed_bytes, long long* free_bytes);

// Reads the device-side convergence flag for the given cfg from the
// pinned-host shadow buffer that the k_check_convergence kernel populates
// every iter. NO sync; relies on natural stream ordering: by the time
// the host calls this accessor (in the iter loop control after
// IdealMhdModel::update() returns), all prior stream work (including the
// k_check_convergence kernel and its async memcpy) has completed via the
// existing cudaStreamSynchronize inside ResidualsCuda. Returns 1 if the
// cfg's residuals are all below the current stage's ftolv, 0 otherwise.
// Returns -1 if the convergence-flag infrastructure is unallocated or the
// cfg index is out of range; callers fall back to the host comparison.
int GetConvergenceFlag(int cfg);

// Whole-iteration CUDA graph (VMECPP_ITER_GRAPH=1, requires active sync
// elision). One cudaGraphLaunch replays the complete elided iteration
// body: forward transforms, jacobian/metric, the segment-graph kernel
// sequences in raw form, force transforms, residuals, the convergence
// flag, and the time step. Capture brackets one normal elided iteration
// in Vmec::Evolve after two warmup iterations; the segment-graph
// Begin/End functions run in passthrough while the capture is open.
// Invalidated on Reshape, on restarts (the baked time_step changes), and
// on a segment-4 jMax re-capture.
bool IterGraphEnabledCuda();
bool IterGraphCapturingCuda();
bool IterGraphReplayCuda();
bool IterGraphBeginCaptureCuda();
void IterGraphEndCaptureCuda();
void AbortIterGraphCaptureCuda();
void InvalidateIterationGraphCuda();

// Hands the distinct-mode per-cfg staging blocks to the CUDA layer in
// memory: the spectral inputs consumed by the first forward transform and
// the decomposed-position block consumed by the time integrator's init,
// both in the [sp][cfg][spec] layout of the batch files. The staging
// loaders consume these ahead of the VMECPP_BATCH_INPUTS_FILE /
// VMECPP_BATCH_DEC_X_FILE fallback, which remains for external drivers.
void SetBatchStagingCuda(const double* inputs, const double* dec_x, int n_cfg,
                         int ns, int mpol, int ntor);

// Drops the staged batch-input blocks. run_batched_gpu calls this once
// its seed run has finished so the staging cannot outlive the batched
// run that owns it.
void ClearBatchStagingCuda();

// Per-config evaluated input profiles for a distinct-mode batch at the current
// multigrid level (flat [cfg][ns]); Vmec fills these from per-config
// RadialProfiles each level so the device stages each config's profiles
// instead of broadcasting the seed's. Cleared when the batched run finishes.
void SetBatchProfilesCuda(int n_cfg, int ns_h, int ns_f, const double* phipF,
                          const double* phipH, const double* currH,
                          const double* iotaH, const double* massH);
void ClearBatchProfilesCuda();

// Reads back the converged per-cfg spectra of the last multi-configuration
// run, the same [sp][cfg][spec] block the VMECPP_BATCH_OUTPUTS_FILE dump
// writes. Returns false when no multi-configuration run has completed.
bool GetBatchOutputSpectraCuda(std::vector<double>* out, int* n_cfg, int* ns,
                               int* mpol, int* ntor);

// Copies the given cfg's d_pts_x slice into the per-cfg converged-state
// snapshot buffers. Called by the iteration controller at the moment a
// cfg transitions to inactive (converged or timed out); the batch outputs
// dump prefers the snapshot over the live slice, which mask-agnostic
// kernels continue to modify while the rest of the batch iterates.
void SnapshotInactiveCfgCuda(int cfg);

// Writes the full batched decomposed-x state (all configuration slots,
// six spectral components) to a raw binary file. Diagnostic for cross-cfg
// contamination A/B runs (VMECPP_STATE_DUMP_ITERS hook in vmec.cc).
void DumpPtsXAllCfgsCuda(const char* path, long long iter);

// Same for the decomposed-forces buffers (per-cfg stride pts_v_size).
void DumpDecomposedFAllCfgsCuda(const char* path, long long iter);

// Per-cfg half-grid radial profiles (chipH, iotaH, jvPlasma,
// avg_guu_gsqrt) at stride ns_h per cfg.
void DumpBContraProfilesAllCfgsCuda(const char* path, long long iter, int ns_h);

// Transfers the host active-mask byte vector to the device-resident
// active-mask buffer. The batched kernels read this buffer at their
// blockIdx.z slot and return early when the byte is zero, providing the
// per-configuration kernel-skip mechanism that drives the throughput
// benefit of the batched mode under per-configuration convergence. The
// implementation compares against the most recently staged buffer and
// omits the transfer when the contents are unchanged, since the mask
// changes only at convergence events (typically a handful of times per
// run). The call is a no-op when n_config_max is one.
void StagePhaseDActiveCuda(const std::vector<std::uint8_t>& active_per_cfg);

// Cached per-configuration residual accessors. The two functions return
// references to host-side caches populated as a side effect of the
// corresponding ResidualsCuda call: GetResidualsPerCfgCacheInvar
// corresponds to the invariant residual call (is_precd == false) and
// GetResidualsPerCfgCachePrecd corresponds to the preconditioned residual
// call (is_precd == true). The cache layout is a contiguous array of
// 3*n_cfg doubles ordered [fResR_0, fResZ_0, fResL_0, fResR_1, ...]. The
// caches are populated after the synchronization in ResidualsCuda returns
// and remain valid until the next invocation of the corresponding
// ResidualsCuda call site.
const std::vector<double>& GetResidualsPerCfgCacheInvar();
const std::vector<double>& GetResidualsPerCfgCachePrecd();

// Cached per-configuration force-norm accessor. Layout: 2*n_cfg doubles
// ordered [sum_rz_0, sum_l_0, sum_rz_1, sum_l_1, ...]. Populated by
// ComputeForceNormsCuda. Caller derives the normalized residuals as
// fNormRZ[c] = 1.0 / (sum_rz[c] * energyDensity^2) and
// fNormL[c] = 1.0 / (sum_l[c] * lamscale^2).
const std::vector<double>& GetFnormScalarsPerCfgCache();

// Cached per-configuration Jacobian extrema accessor. Layout: 2*n_cfg
// doubles ordered [minTau_0, maxTau_0, minTau_1, maxTau_1, ...]. Populated
// by ComputeJacobianCuda. Caller derives per-configuration bad_jacobian
// host-side from the sign and finiteness of the product.
const std::vector<double>& GetJacMinmaxPerCfgCache();

// Cached per-configuration energy-scalar accessor. Layout: 3*n_cfg doubles
// ordered [thermal_0, magnetic_0, mhd_0, thermal_1, ...]. Populated by
// PressureAndEnergiesCuda's per-configuration device-to-host transfer.
// Caller derives per-configuration energyDensity as
// max(magnetic_c, thermal_c) / plasmaVolume_c.
const std::vector<double>& GetPressureScalarsPerCfgCache();

// Cached per-configuration plasma-volume accessor. Layout: n_cfg doubles
// ordered by configuration index. Populated by UpdateVolumeCuda's
// per-configuration device-to-host transfer. Consumed by evalFResInvar to
// compute the per-configuration energy density used in residual
// normalization.
const std::vector<double>& GetPlasmaVolumePerCfgCache();

// Cached per-configuration fNorm1 accessor. Layout: n_cfg doubles, each
// the reciprocal rzNorm of that configuration's device-resident position
// state, refreshed at the force-norm cadence by ComputeForceNormsCuda.
// Configuration zero matches the shared host fNorm1 bit for bit, so the
// per-configuration normalization in evalFResPrecd reduces to the legacy
// scalar arithmetic under single-configuration and broadcast execution.
const std::vector<double>& GetFnorm1PerCfgCache();

// Transfers the device-resident decomposed force buffers back to the
// host-side m_decomposed_f arrays and synchronizes the device stream.
// Consolidates the per-wrapper transfers that the apply-preconditioner
// chain (ApplyM1PreconditionerCuda, ApplyLambdaPreconditionerCuda,
// ApplyRZPreconditionerCuda) and DecomposeAndConstrainCuda would
// otherwise pay individually. The expected invocation point is once at
// the end of IdealMhdModel::update so that the host buffers reflect the
// completed iteration state before any downstream host-resident reader
// observes them.
void FlushDecomposedToHostCuda(const RadialPartitioning& r, const Sizes& s,
                               const FlowControl& fc, double* dec_frcc_host,
                               double* dec_frss_host, double* dec_fzsc_host,
                               double* dec_fzcs_host, double* dec_flsc_host,
                               double* dec_flcs_host);

// Device-side replacement for Vmec::performTimeStep. The velocity and
// decomposed-position state vectors remain resident on the device across
// iterations, removing the per-iteration device-to-host transfer that the
// host-resident path would require. The device position is the
// authoritative state: the host m_decomposed_x arrays are refreshed at the
// controller's explicit flush sites, and RecomposeToPhysicalCuda consumes
// d_pts_x directly from the second iteration onward. The pointer arguments
// correspond to the std::span representations of the FourierVelocity and
// FourierGeometry containers.
// fnorm1 and iter_phase feed the device time-step controller
// (k_update_timestep): fnorm1 is the host's preconditioned-residual
// normalization for the R/Z components (the L component scales by
// fc.deltaS), and iter_phase is 0 when iter2 == iter1 (ring reset at
// stage starts and restarts), 1 otherwise.
void PerformTimeStepCuda(const RadialPartitioning& r, const Sizes& s,
                         const FlowControl& fc, double velocity_scale,
                         double conjugation_parameter, double time_step,
                         double fnorm1, int iter_phase, double* m_dec_v_rcc,
                         double* m_dec_v_rss, double* m_dec_v_zsc,
                         double* m_dec_v_zcs, double* m_dec_v_lsc,
                         double* m_dec_v_lcs, double* m_dec_x_rcc,
                         double* m_dec_x_rss, double* m_dec_x_zsc,
                         double* m_dec_x_zcs, double* m_dec_x_lsc,
                         double* m_dec_x_lcs);

// Brings the device state into the new multigrid stage before iteration
// 1's geometry pipeline: lazy Reshape (snapshots + stage-sized buffers),
// scalxc staging, and PerformTimeStepCuda's init section (multigrid
// upscale / per-cfg dec_x load) without a time step. Idempotent.
void PrepareStagePtsXCuda(const RadialPartitioning& r, const Sizes& s,
                          const FourierBasisFastPoloidal& fb,
                          const FlowControl& fc, const Eigen::VectorXd& scalxc,
                          double* m_dec_x_rcc, double* m_dec_x_rss,
                          double* m_dec_x_zsc, double* m_dec_x_zcs,
                          double* m_dec_x_lsc, double* m_dec_x_lcs);

// Device-resident time-step damping computation. Computes per-cfg fac / b1
// from the on-device precd residuals (d_residuals_partial) without round-
// tripping through host. Standalone entry point retained for validation;
// the production dispatch of k_update_timestep lives inside
// PerformTimeStepCuda, gated by VMECPP_BATCH_PER_CFG_TIMESTEP and by sync
// elision. The host invTau_ ring buffer stays the source of truth when
// neither is active.
void UpdateTimestepDeviceCuda(const FlowControl& fc, int iter1, int iter2,
                              double time_step, double fnorm1);

// Device-side composition of the host triplet that previously executed at
// the start of IdealMhdModel::update: decomposeInto applied to
// m_decomposed_x with the scalxc weighting, m1Constraint on the result with
// unit scaling factor, and extrapolateTowardsAxis. Reads the
// device-resident d_pts_x buffers maintained by PerformTimeStepCuda and
// writes the d_specs_block sections that CudaForward would otherwise
// populate via a host-to-device transfer. Setting this device-resident
// state allows CudaForward to skip its specs transfer entirely on
// iterations where the routine has run.
void RecomposeToPhysicalCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    const Eigen::VectorXd& scalxc, const double* m_dec_x_rcc,
    const double* m_dec_x_rss, const double* m_dec_x_zsc,
    const double* m_dec_x_zcs, const double* m_dec_x_lsc,
    const double* m_dec_x_lcs);

// Resets the thread-local CudaToroidalState's persistent device buffers
// at the start of each Vmec::run invocation. The persistent buffers
// include the preconditioning-matrix snapshots (d_pmat_*), the tridiagonal
// coefficient arrays consumed by the RZ-preconditioner solver
// (d_rz_aR/dR/bR/aZ/dZ/bZ and d_rz_jMin), and any associated bookkeeping
// state. Without this reset, a second VMEC run within the same process
// inherits the prior run's final values for these buffers; the
// preconditioner-update path refreshes only the half-grid contributions
// at the start of each new run, so stale full-grid contributions persist
// and cause divergence before the first preconditioner refresh of the
// new run completes. The routine is safe to call before any FFT call,
// returning as a no-op when the device stream has not yet been allocated.
// The expected call site is Vmec::run, immediately before the first
// IdealMhdModel iteration.
void ResetCudaStateForNewVmecRun();

// Commits the host-side writes that FourierToReal3DSymmFastPoloidalCuda
// defers as part of its no-end-sync optimization. The forward FFT routine
// returns without synchronizing the device stream and without populating
// the host-side geometry scalars (m_geometry.r1_e at the outer edge,
// m_geometry.r1_o, m_geometry.z1_e), since those host values are consumed
// only by subsequent code paths that themselves perform a stream
// synchronization. This routine performs the deferred host writes after
// the caller's stream synchronization completes. The expected call site
// is immediately after ComputeJacobianCuda's tau-extrema synchronization,
// which is the first stream synchronization following the forward FFT.
// A call with no pending deferred writes is a no-op.
void FlushFwdGeomScalarsToHost(double* r1_e, double* r1_o, double* z1_e);

// CUDA Graph capture and replay infrastructure for contiguous segments of
// the post-Jacobian portion of IdealMhdModel::update. Each segment spans a
// sequence of asynchronous kernel launches with no intervening host
// synchronization or device-to-host transfer, so the entire sequence can
// be recorded into a cudaGraph_t once and replayed thereafter with a
// single cudaGraphLaunch call. This amortizes the kernel-launch overhead
// across the captured sequence, which is particularly beneficial at low
// N_config where the per-kernel host-driver cost is a significant fraction
// of the kernel's effective wall time.
//
// Each segment exposes a paired Begin/End entry point. The Begin call
// returns true if a previously captured graph was replayed (in which case
// the caller should skip the individual kernel-wrapper invocations that
// the graph represents) and false if the segment is either in capture
// mode for the current iteration or has been disabled at runtime via the
// corresponding environment variable. The End call closes the capture
// boundary and instantiates the graph; in replay mode it is a no-op.

bool BeginUpdateSegment3GraphOrReplay();
void EndUpdateSegment3GraphOrLaunch();

// Segment-4 graph: covers ApplyM1PreconditionerCuda,
// AssembleRZPreconditionerCuda, ApplyRZPreconditionerCuda, and
// ApplyLambdaPreconditionerCuda, all of which fall between the two
// ResidualsCuda synchronization points. The jMax argument controls the
// PCR solver's loop bound, which can change between iterations when the
// free-boundary vacuum-pressure state transitions; the graph is
// re-captured when jMax differs from its previous value.
bool BeginUpdateSegment4GraphOrReplay(int jMax);
void EndUpdateSegment4GraphOrLaunch();

// Segment-2 graph: covers ComputeMetricElementsCuda,
// UpdateDifferentialVolumeCuda, ComputeBContraCuda, ComputeBCoCuda,
// PressureAndEnergiesCuda, RadialForceBalanceCuda, and
// HybridLambdaForceCuda. Spans from the synchronization point at the end
// of ComputeJacobianCuda to the preconditioner-update block that opens
// segment three. defer_capture: no capture is begun on iterations whose
// segment body synchronizes the stream; replay is unaffected.
bool BeginUpdateSegment2GraphOrReplay(bool defer_capture);
void EndUpdateSegment2GraphOrLaunch();

// Device-side mirror of the host physical_x_backup mechanism employed by
// Vmec::RestartIteration. On the NO_RESTART code path, the controller
// periodically copies m_decomposed_x into m_physical_x_backup so that
// subsequent BAD_JACOBIAN or BAD_PROGRESS events can revert to the
// preserved state. Under the CUDA path the authoritative iteration state
// resides in the device buffer d_pts_x; the backup and restore operations
// must therefore execute device-side as well in order for the rollback
// semantics to propagate correctly. The presence of a device-resident
// backup also permits eliminating the per-iteration device-to-host
// transfer that PerformTimeStepCuda would otherwise require to maintain
// a fresh host shadow.
void BackupPtsXCuda();
void RestorePtsXFromBackupCuda();
// Per-cfg variant: restore d_pts_x and zero d_pts_v only for cfgs whose
// mask byte is non-zero. mask.size() must equal n_config_max. Whole-batch
// behavior is recovered by passing a mask of all 1's.
void RestorePtsXFromBackupPerCfgCuda(const std::vector<std::uint8_t>& mask);

// Invalidates the device-resident position, velocity, and restart-backup
// state so the next initialization re-stages them from the host
// m_decomposed_x. Called by the iteration-1 recovery path after the
// magnetic-axis recomputation: the recovery re-interpolates the host
// state from the boundary and the improved axis, and the device-side
// per-stage initialization is gated on pts_x_initialized, which the
// failed attempt left set.
void InvalidatePtsXCuda();

// Transfers the device-resident decomposed-position state (configuration
// zero only) to the host m_decomposed_x arrays. The expected call site is
// once at end-of-run, alongside FlushForOutputQuantitiesCuda, so that the
// post-iteration output-derivation path observes the final iteration
// state on the host. Selecting configuration zero is consistent with the
// remainder of the end-of-run output-derivation logic, which produces a
// single output equilibrium even when the batched execution mode has
// solved multiple configurations in parallel.
void FlushDecomposedXToHostCuda(int cfg, int ns_local, int mpol, int ntor,
                                bool lthreed, double* m_dec_x_rcc,
                                double* m_dec_x_rss, double* m_dec_x_zsc,
                                double* m_dec_x_zcs, double* m_dec_x_lsc,
                                double* m_dec_x_lcs);

// Per-configuration variant of FlushForOutputQuantitiesCuda. Transfers
// every metric-input array from device to host for all n_cfg
// configurations in a single batched operation. Each host buffer is sized
// to n_cfg multiplied by the per-configuration element count, with the
// configuration axis stored in the outer (slowest-varying) position. The
// routine enables the Python output-processing layer to derive the
// converged metrics for all N equilibria from a single batched VMEC run.
void FlushAllConfigsForOutputCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    int n_cfg, double* gsqrt_host, double* guu_host, double* guv_host,
    double* gvv_host, double* bsubu_host, double* bsubv_host,
    double* bsupu_host, double* bsupv_host, double* totalPressure_host,
    double* r12_host, double* ru12_host, double* zu12_host, double* rs_host,
    double* zs_host, double* r1_e_host, double* r1_o_host, double* z1_e_host,
    double* z1_o_host, double* ru_e_host, double* ru_o_host, double* zu_e_host,
    double* zu_o_host, double* rv_e_host, double* rv_o_host, double* zv_e_host,
    double* zv_o_host, double* ruFull_host, double* zuFull_host,
    double* blmn_e_host, double* presH_host, double* dVdsH_host,
    double* bvcoH_host, double* bucoH_host, double* jcurvF_host,
    double* jcuruF_host, double* presgradF_host, double* dVdsF_host,
    double* equiF_host, double* chipH_host, double* iotaH_host,
    double* chipF_host, double* iotaF_host, double* pts_x_rcc_host,
    double* pts_x_rss_host, double* pts_x_zsc_host, double* pts_x_zcs_host,
    double* pts_x_lsc_host, double* pts_x_lcs_host);

// Repair per-config dVdsH zeroed by pre-final masking (which zeroes betatot);
// rewrites only zeroed configs. Call before FlushAllConfigsForOutputCudaNs.
void RecomputeZeroedDVdsHForOutputCuda(int n_cfg, int ns_h, int nZnT,
                                       int nThetaEff, double signOfJacobian);

// Single-rank variant of the per-configuration flush: the radial extents
// derive from ns alone, so callers that hold no RadialPartitioning or
// FlowControl (the post-run pybind path) can flush with just the converged
// ns and the Sizes. The interior full-grid arrays (jcurvF and friends) are
// not flushed by this variant.
void FlushAllConfigsForOutputCudaNs(
    int ns, const Sizes& s, int n_cfg, double* gsqrt_host, double* guu_host,
    double* guv_host, double* gvv_host, double* bsubu_host, double* bsubv_host,
    double* bsupu_host, double* bsupv_host, double* totalPressure_host,
    double* r12_host, double* ru12_host, double* zu12_host, double* rs_host,
    double* zs_host, double* r1_e_host, double* r1_o_host, double* z1_e_host,
    double* z1_o_host, double* ru_e_host, double* ru_o_host, double* zu_e_host,
    double* zu_o_host, double* rv_e_host, double* rv_o_host, double* zv_e_host,
    double* zv_o_host, double* ruFull_host, double* zuFull_host,
    double* blmn_e_host, double* presH_host, double* dVdsH_host,
    double* bvcoH_host, double* bucoH_host, double* chipH_host,
    double* iotaH_host, double* chipF_host, double* iotaF_host,
    double* pts_x_rcc_host, double* pts_x_rss_host, double* pts_x_zsc_host,
    double* pts_x_zcs_host, double* pts_x_lsc_host, double* pts_x_lcs_host);

// D2H of the configuration-zero differential-volume profile (half grid),
// consumed by the printout-cadence spectral-width volume average. No-op
// when the device buffer is absent.
void FlushDVdsHToHostCuda(int ns_h, double* dVdsH_host);

// Device-side mirror of FourierCoeffs::rzNorm invoked with
// include_offset == false. The implementation reads the configuration-zero
// slice of the device-resident d_pts_x buffers, computes the sum of
// squared coefficients on the device, and returns the resulting scalar
// to the host. The reduction is carried out in two stages: a per-radial
// partial kernel emits one double per radial surface, and the host
// accumulates the partials in radial order. The ordering matches the
// outer loop of the CPU implementation, so the routine is bit-exact with
// FourierCoeffs::rzNorm under nominally identical inputs. Reading from
// d_pts_x rather than the host m_decomposed_x shadow eliminates the
// per-iteration device-to-host synchronization that PerformTimeStepCuda
// would otherwise need to maintain a fresh host shadow.
double ComputeForceNorm1FromPtsXCuda(int ns_local, int mpol, int ntor,
                                     bool lthreed, int nsMinHere_local,
                                     int nsMaxHere_local);

// Reports whether the device-resident d_pts_x buffers have been
// initialized by the first invocation of PerformTimeStepCuda. The
// buffers are zero-initialized at cudaMalloc time but do not contain a
// valid decomposed-position state until PerformTimeStepCuda has copied
// the initial host state. Callers that invoke
// ComputeForceNorm1FromPtsXCuda before this condition holds must instead
// fall back to the host-resident rzNorm computation, since the device
// buffer state is unrepresentative of the iteration's starting condition.
bool PtsXInitializedCuda();

// True when d_pts_x is initialized AND sized for the given stage geometry
// (ns_local * mpol * (ntor+1) elements per cfg). At iteration 1 of a
// multigrid stage this distinguishes the post-upscale authoritative
// device state from a stale previous-stage buffer.
bool PtsXMatchesCuda(int ns_local, int mpol, int ntor);

// Marks the current iteration as sync-elided (1) or a sync boundary (0).
// On elided iterations the per-iteration scalar D2H + stream-sync sites
// (jacobian tau extrema, residual triples, plasma volume) launch their
// reduction kernels but skip the transfer and sync; host callers receive
// the last boundary-synced values. Driven by VMECPP_SYNC_ELIDE=K
// orchestration in Vmec::Evolve.
void SetSyncElideIterCuda(int elide);
// Reads the flag staged by SetSyncElideIterCuda: true on iterations
// whose host syncs are elided, false on boundary iterations and when
// elision is off. The free-boundary path consults it to skip the host
// vacuum block on elided iterations.
bool SyncElidedIterCuda();
// Run-level counterpart, set once per run by Vmec::Evolve when the
// VMECPP_SYNC_ELIDE value resolves: true when the run carries an active
// elision window, independent of the current iteration. The
// free-boundary vacuum block consults it to force full NESTOR updates
// at the window boundaries. Reset by ResetCudaStateForNewVmecRun.
void SetSyncElideRunCuda(int active);
bool SyncElideRunActiveCuda();

// Diagnostic probe that reports the maximum absolute difference between
// configuration zero and configuration one of a device-resident buffer.
// For each index i in [0, per_cfg_size), the routine computes
// |d_buf[i] - d_buf[per_cfg_size + i]| and prints the maximum value to
// the diagnostic stream, prefixed with the supplied label so that
// multiple invocations within the same iteration can be distinguished.
// The routine is synchronous: it waits for the device stream to complete
// before issuing the device-to-host transfer required to evaluate the
// reduction on the host. The probe is gated by the VMECPP_TRACE_CFG_DIFF
// environment variable and is a no-op when the variable is unset or zero.
// Intended use is the empirical localization of the first kernel in the
// iteration body at which two configurations begin to diverge when the
// batched execution mode is supplied with identical inputs.
void DiagCfg01DiffCuda(const double* d_buf, int per_cfg_size,
                       const char* label);

// Single-configuration final-iteration flush of all device-resident
// quantities consumed by ComputeOutputQuantities. The routine is invoked
// once at the end of IdealMhdModel::update and transfers each
// device-resident metric-input array to its host counterpart so that the
// post-iteration output-derivation path (GatherDataFromThreads, the
// half-grid wout buffer assembly, and the radial-derivative metric
// calculations including L_grad_B,
// flux_compression_in_regions_of_bad_curvature, and vacuum_well) observes the
// converged state. Without this consolidated flush the host metric-input arrays
// contain only the values populated by the corresponding per-iteration
// transfers from earlier in the iteration, which are reset to zero or to
// per-iteration partials and would produce degenerate output values when
// consumed by the output-derivation path. Any host pointer argument may be
// supplied as nullptr to omit the transfer of the corresponding field; this is
// appropriate when the field is not consumed by the configured output mode (for
// example, the toroidal-flux derivatives are needed only by certain
// free-boundary diagnostics). The expected call site is the same as that of
// FlushDecomposedToHostCuda.
void FlushForOutputQuantitiesCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    // half-grid scalar fields (ns_h * nZnT each)
    double* gsqrt_host, double* guu_host, double* guv_host, double* gvv_host,
    double* bsubu_host, double* bsubv_host, double* bsupu_host,
    double* bsupv_host, double* totalPressure_host, double* r12_host,
    double* ru12_host, double* zu12_host, double* rs_host, double* zs_host,
    // full-grid R/Z and derivatives (ns_local * nZnT each)
    double* r1_e_host, double* r1_o_host, double* z1_e_host, double* z1_o_host,
    double* ru_e_host, double* ru_o_host, double* zu_e_host, double* zu_o_host,
    double* rv_e_host, double* rv_o_host, double* zv_e_host, double* zv_o_host,
    double* ruFull_host, double* zuFull_host,
    // full-grid force-local (ns_force_local * nZnT)
    double* blmn_e_host,
    // radial half-grid scalars (ns_h)
    double* presH_host, double* dVdsH_host, double* bvcoH_host,
    // radial force-local scalars (ns_force_local)
    double* jcurvF_host, double* jcuruF_host, double* presgradF_host,
    double* dVdsF_host, double* equiF_host,
    // half-grid + full-grid magnetic profile scalars (ns_h / ns_local).
    // Previously D2H'd per-iter in ComputeBContraCuda; consolidated here so
    // the per-iter D2H+launch overhead is eliminated.
    double* chipH_host, double* iotaH_host, double* chipF_host,
    double* iotaF_host);

}  // namespace vmecpp

#endif  // VMECPP_USE_CUDA

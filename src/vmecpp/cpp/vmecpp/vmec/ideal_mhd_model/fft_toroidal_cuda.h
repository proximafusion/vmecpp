// CUDA implementations of the IdealMhdModel iteration body, compiled with
// VMECPP_USE_CUDA and dispatched from IdealMhdModel and Vmec::run.
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

// Geometry from the Fourier coefficients of physical_x on the device, as
// FourierToReal3DSymmFastPoloidalFft computes it on the CPU.
void FourierToReal3DSymmFastPoloidalCuda(
    const FourierGeometry& physical_x, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, RealSpaceGeometry& m_geometry);

// ForcesToFourier3DSymmFastPoloidalFft on the device; vacuum_pressure_state
// selects the boundary-edge contribution.
void ForcesToFourier3DSymmFastPoloidalCuda(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

// computeJacobian on the device: half-grid r12, ru12, zu12, rs, zs and tau;
// bad_jacobian when tau changes sign or is not finite.
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

// With checkerboard_terms >= 0 the half-grid <g_uu / sqrt g> profile is also
// kept for the tridiagonal m = 0 block of ApplyLambdaPreconditionerCuda.
void UpdateLambdaPreconditionerCuda(const RadialPartitioning& r, const Sizes& s,
                                    double lambdaPreconditionerScale,
                                    double lamscale, double checkerboard_terms,
                                    double* bLambda_out, double* dLambda_out,
                                    double* cLambda_out,
                                    double* lambdaPreconditioner_host);

// computePreconditioningMatrix on the device for one side: R (side 0) into
// d_pmat_arm, brm, ard, brd, or Z (side 1) into the azm set.
void ComputePreconditioningMatrixCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    double deltaS, int kEvenParity, int kOddParity, const Eigen::VectorXd& xs,
    const Eigen::VectorXd& xu12, const Eigen::VectorXd& xu_e,
    const Eigen::VectorXd& xu_o, const Eigen::VectorXd& x1_o,
    const Eigen::VectorXd& sm, const Eigen::VectorXd& sp,
    Eigen::VectorXd& m_axm, Eigen::VectorXd& m_axd, Eigen::VectorXd& m_bxm,
    Eigen::VectorXd& m_bxd, Eigen::VectorXd& m_cxd, int side);

// Assembles the tridiagonal RZ-preconditioner coefficients and d_rz_jMin on the
// device from the latest preconditioning matrices.
void AssembleRZPreconditionerCuda(const RadialPartitioning& r, const Sizes& s,
                                  const FlowControl& fc, int jMax);

// ruFull = ru_e + sqrtSF * ru_o and zuFull = zu_e + sqrtSF * zu_o on the
// constraint range, copied to the host for the free-boundary path.
void ComputeRuZuFullCuda(const RadialPartitioning& r, const Sizes& s,
                         Eigen::VectorXd& ruFull, Eigen::VectorXd& zuFull);

void ConstraintForceMultiplierCuda(const RadialPartitioning& r, const Sizes& s,
                                   const FlowControl& fc, double tcon0,
                                   const Eigen::VectorXd& ard,
                                   const Eigen::VectorXd& azd,
                                   Eigen::VectorXd& tcon_out);

void EffectiveConstraintForceCuda(const RadialPartitioning& r, const Sizes& s,
                                  Eigen::VectorXd& gConEff_out);

// vacuum_edge applies the staged rBSq profile to the LCFS force row; true only
// while the vacuum pressure is initialized or active.
void AssembleTotalForcesCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    const Eigen::VectorXd& gCon, Eigen::VectorXd& brmn_e_out,
    Eigen::VectorXd& brmn_o_out, Eigen::VectorXd& bzmn_e_out,
    Eigen::VectorXd& bzmn_o_out, Eigen::VectorXd& frcon_e_out,
    Eigen::VectorXd& frcon_o_out, Eigen::VectorXd& fzcon_e_out,
    Eigen::VectorXd& fzcon_o_out, bool vacuum_edge);

// Free-boundary bridges to the host NESTOR solve. Whether the vacuum edge force
// is active; a change recaptures the segment-three graph.
void SetVacuumEdgeCuda(int active);
// Marks the current Vmec::run as free-boundary, whose vacuum block reads
// bucoH on the host every iteration. Reset by ResetCudaStateForNewVmecRun.
void SetFreeBoundaryRunCuda(int enabled);
// Scales the device rCon0/zCon0 volume profiles in place.
void ScaleRZCon0Cuda(double factor);
// One flush per vacuum iteration and configuration of the axis and LCFS rows,
// outer totalPressure rows and the presH, bucoH, bvcoH profiles.
void FlushVacuumHostDataCuda(int cfg, const RadialPartitioning& r,
                             const Sizes& s, Eigen::VectorXd& m_r1_e,
                             Eigen::VectorXd& m_r1_o, Eigen::VectorXd& m_z1_e,
                             Eigen::VectorXd& m_totalPressure,
                             Eigen::VectorXd& m_presH, Eigen::VectorXd& m_bucoH,
                             Eigen::VectorXd& m_bvcoH);
// H2D stage of one configuration's host-computed rBSq profile for the
// edge application.
void StageRbsqCuda(int cfg, const Eigen::VectorXd& rBSq);

// Physical to decomposed forces on the device: decomposeInto with scalxc, the m
// = 1 constraint, and the m = 1 Z force zeroed where the CPU zeroes it.
void DecomposeAndConstrainCuda(const RadialPartitioning& r, const Sizes& s,
                               const FlowControl& fc, int iter2,
                               bool always_fix_m1_gauge, double m1ScalingFactor,
                               int sign_of_jacobian,
                               const Eigen::VectorXd& scalxc,
                               double* dec_frcc_host, double* dec_frss_host,
                               double* dec_fzsc_host, double* dec_fzcs_host,
                               double* dec_flsc_host, double* dec_flcs_host);

// applyM1Preconditioner on the device: scales the m = 1 R and Z force
// coefficients by the ratio from ard, brd, azd and bzd.
void ApplyM1PreconditionerCuda(const RadialPartitioning& r, const Sizes& s,
                               const Eigen::VectorXd& ard,
                               const Eigen::VectorXd& brd,
                               const Eigen::VectorXd& azd,
                               const Eigen::VectorXd& bzd, double* frss_host,
                               double* fzcs_host);

// applyLambdaPreconditioner on the device; with checkerboard_terms >= 0 in 3D
// the m = 0, n >= 1 forces take the tridiagonal solve as on the CPU.
void ApplyLambdaPreconditionerCuda(const RadialPartitioning& r, const Sizes& s,
                                   const Eigen::VectorXd& lambdaPreconditioner,
                                   double checkerboard_terms, double step_scale,
                                   double* flsc_host, double* flcs_host);

// applyRZPreconditioner on the device: a PCR solve per (m, n) of the assembled
// tridiagonal system, from jMin to jMaxRZ.
int ApplyRZPreconditionerCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    const Eigen::VectorXd& ar, const Eigen::VectorXd& dr,
    const Eigen::VectorXd& br_in, const Eigen::VectorXd& az,
    const Eigen::VectorXd& dz, const Eigen::VectorXd& bz_in,
    const int* jMin_arr, int jMin_size, int jMaxRZ, double* frcc_host,
    double* frss_host, double* fzsc_host, double* fzcs_host);

// deAliasConstraintForce on the device: a Fourier round trip of the constraint
// force, scaled by tcon and the faccon weights.
void DeAliasConstraintForceCuda(const RadialPartitioning& r,
                                const FourierBasisFastPoloidal& fb,
                                const Sizes& s, const Eigen::VectorXd& faccon,
                                Eigen::VectorXd& m_gCon_host);

// FourierForces::residuals on the device: [fResR, fResZ, fResL] over the CPU's
// radial range; edge_per_cfg selects the edge forces per configuration.
void ResidualsCuda(const RadialPartitioning& r, const Sizes& s,
                   const FlowControl& fc, bool includeEdgeRZForces,
                   double& fResR_out, double& fResZ_out, double& fResL_out,
                   bool is_precd = false,
                   const std::vector<std::uint8_t>* edge_per_cfg = nullptr);

// Per-configuration accessors of the batched mode: copies of the device's
// per-configuration outputs that the iteration controller reads.

// ResetCudaStateForNewVmecRun sets this run's configuration count, to which
// FlowControl sizes its per-configuration vectors.
int GetNConfigMaxCuda();

// Largest ns the radial solver supports on this device: PCR to 1024, then
// block-Thomas limited by shared memory; 1024 if the query fails.
int CudaMaxRadialResolution();

// Whether a run's persistent device allocation fits the free memory, filling
// needed_bytes and free_bytes; true when the device cannot be queried.
bool CudaVramBudgetCuda(long long n_cfg, long long ns, long long mpol,
                        long long ntor, long long nZeta, long long nThetaEff,
                        long long* needed_bytes, long long* free_bytes);

// The device convergence flag of cfg from its pinned shadow, current after
// ResidualsCuda's synchronization: 1, 0, or -1 when unavailable.
int GetConvergenceFlag(int cfg);

// Hands the distinct batch's staged spectra and decomposed positions,
// [sp][cfg][spec], to the CUDA layer.
void SetBatchStagingCuda(const double* inputs, const double* dec_x, int n_cfg,
                         int ns, int mpol, int ntor);

// Drops the staged batch inputs once run_batched_gpu's seed run has finished.
void ClearBatchStagingCuda();

// Each configuration's evaluated profiles at the current level (flat [cfg][ns])
// for a distinct batch; cleared when the batched run finishes.
void SetBatchProfilesCuda(int n_cfg, int ns_h, int ns_f, const double* phipF,
                          const double* phipH, const double* currH,
                          const double* iotaH, const double* massH);
void ClearBatchProfilesCuda();

// The converged per-configuration spectra of the last multi-configuration run,
// [sp][cfg][spec]; false when none has completed.
bool GetBatchOutputSpectraCuda(std::vector<double>* out, int* n_cfg, int* ns,
                               int* mpol, int* ntor);

// Snapshots cfg's position slice when it goes inactive, since kernels that
// ignore the mask keep modifying the live slice.
void SnapshotInactiveCfgCuda(int cfg);

// Copies the host active mask to the device, where batched kernels skip a
// configuration whose byte is zero; an unchanged mask is not copied.
void StagePhaseDActiveCuda(const std::vector<std::uint8_t>& active_per_cfg);

// Host caches of the per-configuration residuals of the invariant and
// preconditioned ResidualsCuda calls, [fResR, fResZ, fResL] per configuration.
const std::vector<double>& GetResidualsPerCfgCacheInvar();
const std::vector<double>& GetResidualsPerCfgCachePrecd();

// Per-configuration force-norm sums [sum_rz, sum_l] from ComputeForceNormsCuda,
// from which the caller derives fNormRZ and fNormL.
const std::vector<double>& GetFnormScalarsPerCfgCache();

// Per-configuration [minTau, maxTau] from ComputeJacobianCuda.
const std::vector<double>& GetJacMinmaxPerCfgCache();

// The configurations whose Jacobian changed sign in the latest force
// evaluation; empty when the cache does not hold n_cfg configurations.
std::vector<std::uint8_t> BadJacobianPerCfgCuda(int n_cfg);

// Per-configuration [thermal, magnetic, mhd] energies from
// PressureAndEnergiesCuda.
const std::vector<double>& GetPressureScalarsPerCfgCache();

// Per-configuration plasma volume from UpdateVolumeCuda, for the energy density
// of the residual normalization.
const std::vector<double>& GetPlasmaVolumePerCfgCache();

// Per-configuration {fsqr, fsqz, fsql} of each configuration's last active
// iteration.
const std::vector<double>& GetFsqrPerCfgCache();

// Per-configuration fNorm1, the reciprocal rzNorm of each configuration's
// device positions at the force-norm cadence.
const std::vector<double>& GetFnorm1PerCfgCache();

// Vmec::performTimeStep on the device, whose positions are the iteration state;
// fnorm1 and iter_phase feed the device time-step controller.
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

// Brings the device state into the new multigrid stage before iteration 1:
// reshape, scalxc staging and the initial positions. Idempotent.
void PrepareStagePtsXCuda(const RadialPartitioning& r, const Sizes& s,
                          const FourierBasisFastPoloidal& fb,
                          const FlowControl& fc, const Eigen::VectorXd& scalxc,
                          double* m_dec_x_rcc, double* m_dec_x_rss,
                          double* m_dec_x_zsc, double* m_dec_x_zcs,
                          double* m_dec_x_lsc, double* m_dec_x_lcs);

// The host triplet on the device: decomposeInto, m1Constraint and
// extrapolateTowardsAxis from d_pts_x into the specs CudaForward reads.
void RecomposeToPhysicalCuda(
    const RadialPartitioning& r, const Sizes& s, const FlowControl& fc,
    int sign_of_jacobian, const Eigen::VectorXd& scalxc,
    const double* m_dec_x_rcc, const double* m_dec_x_rss,
    const double* m_dec_x_zsc, const double* m_dec_x_zcs,
    const double* m_dec_x_lsc, const double* m_dec_x_lcs);

// Resets the persistent device state at the start of each Vmec::run; n_config
// sets the configuration count and distinct their own inputs.
void ResetCudaStateForNewVmecRun(int n_config, bool distinct);

// Writes the geometry scalars that the forward transform defers, after the
// first stream synchronization following it; otherwise a no-op.
void FlushFwdGeomScalarsToHost(double* r1_e, double* r1_o, double* z1_e);

// CUDA graphs of synchronization-free segments of update(): Begin returns true
// after replaying, and the caller skips those kernels.

bool BeginUpdateSegment3GraphOrReplay();
void EndUpdateSegment3GraphOrLaunch();

// Segment-four graph over the four preconditioner applications; a change of
// jMax at a vacuum-pressure transition re-captures it.
bool BeginUpdateSegment4GraphOrReplay(int jMax);
void EndUpdateSegment4GraphOrLaunch();

// Segment-two graph from the end of ComputeJacobianCuda through
// HybridLambdaForceCuda; defer_capture skips a synchronizing iteration.
bool BeginUpdateSegment2GraphOrReplay(bool defer_capture);
void EndUpdateSegment2GraphOrLaunch();

// Device backup and restore of d_pts_x for the restart protocol; with
// from_evaluated the backup takes the state of the latest force evaluation.
void SnapshotEvaluatedPtsXCuda();
void BackupPtsXCuda(bool from_evaluated);

// Per-configuration variants of the two calls above: only the configurations
// whose mask byte is nonzero take the snapshot or the store.
void SnapshotEvaluatedPtsXPerCfgCuda(const std::vector<std::uint8_t>& mask);
void BackupPtsXPerCfgCuda(const std::vector<std::uint8_t>& mask,
                          bool from_evaluated);
void RestorePtsXFromBackupCuda();
// Restores d_pts_x and zeroes d_pts_v for the configurations whose mask byte is
// nonzero; the mask holds n_config_max bytes.
void RestorePtsXFromBackupPerCfgCuda(const std::vector<std::uint8_t>& mask);

// Invalidates the device positions, velocity and backup, so the axis-recovery
// retry re-stages them from the host.
void InvalidatePtsXCuda();

// Copies the device positions of configuration cfg to the host m_decomposed_x
// arrays.
void FlushDecomposedXToHostCuda(int cfg, int ns_local, int mpol, int ntor,
                                bool lthreed, double* m_dec_x_rcc,
                                double* m_dec_x_rss, double* m_dec_x_zsc,
                                double* m_dec_x_zcs, double* m_dec_x_lsc,
                                double* m_dec_x_lcs);

// Copies the first configuration's device backup, the last accepted state, to
// the host; a no-op before the backup is armed or across a shape change.
void FlushBackupXToHostCuda(int ns_local, int mpol, int ntor, bool lthreed,
                            double* rcc, double* rss, double* zsc, double* zcs,
                            double* lsc, double* lcs);

// Marks the device copy of currH stale, so that the next force evaluation
// stages the host values again: an iteration callback may change them.
void InvalidateStagedCurrHCuda();

// maskGeometryAbove on the device copy of the forces or positions; a no-op
// unless mpol_geometry or ntor_geometry caps the geometry.
void MaskGeometryAboveCuda(const RadialPartitioning& r, const Sizes& s,
                           const FlowControl& fc, bool forces);

// Copies the output-derivation arrays of all n_cfg configurations to the host,
// configuration outermost.
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

// Single-rank variant sized from ns alone, for callers without
// RadialPartitioning or FlowControl; it skips the interior full-grid arrays.
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

// Copies configuration zero's half-grid dVdsH to the host for the
// spectral-width average; a no-op without the device buffer.
void FlushDVdsHToHostCuda(int ns_h, double* dVdsH_host);

// Whether the first PerformTimeStepCuda has loaded d_pts_x; until then callers
// read the host state.
bool PtsXInitializedCuda();

// Whether d_pts_x is loaded and sized for this stage, telling the upscaled
// state at a stage's first iteration from a stale buffer.
bool PtsXMatchesCuda(int ns_local, int mpol, int ntor);

// Copies every device field ComputeOutputQuantities reads to the host once, at
// the end of the run; a nullptr argument skips its field.
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
    double* bucoH_host,
    // radial force-local scalars (ns_force_local)
    double* jcurvF_host, double* jcuruF_host, double* presgradF_host,
    double* dVdsF_host, double* equiF_host,
    // half-grid + full-grid magnetic profile scalars (ns_h / ns_local).
    double* chipH_host, double* iotaH_host, double* chipF_host,
    double* iotaF_host);

}  // namespace vmecpp

#endif  // VMECPP_USE_CUDA

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_

#include <Eigen/Dense>
#include <climits>
#include <memory>
#include <span>

#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP

#include "absl/status/statusor.h"
#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/free_boundary_base/free_boundary_base.h"
#include "vmecpp/vmec/boundaries/boundaries.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/handover_storage/handover_storage.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/ideal_mhd_model/fft_toroidal.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"
#include "vmecpp/vmec/thread_local_storage/thread_local_storage.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

namespace vmecpp {

// Implemented as a free function for easier testing and benchmarking.
// "FastPoloidal" indicates that, in real space, iterations use the
// poloidal coordinate as the fast index.
void ForcesToFourier3DSymmFastPoloidal(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

// Implemented as a free function for easier testing and benchmarking.
// "FastPoloidal" indicates that, in real space, iterations use the
// poloidal coordinate as the fast index.
void FourierToReal3DSymmFastPoloidal(const FourierGeometry& physical_x,
                                     const Eigen::VectorXd& xmpq,
                                     const RadialPartitioning& r,
                                     const Sizes& s, const RadialProfiles& rp,
                                     const FourierBasisFastPoloidal& fb,
                                     RealSpaceGeometry& m_geometry);

// Implemented as a free function for easier testing and benchmarking.
void deAliasConstraintForce(const RadialPartitioning& rp,
                            const FourierBasisFastPoloidal& fb, const Sizes& s_,
                            const Eigen::VectorXd& faccon,
                            const Eigen::VectorXd& tcon,
                            const Eigen::VectorXd& gConEff,
                            Eigen::VectorXd& m_gsc, Eigen::VectorXd& m_gcs,
                            Eigen::VectorXd& m_gCon);

class IdealMhdModel {
 public:
  IdealMhdModel(FlowControl* m_fc, const Sizes* s,
                const FourierBasisFastPoloidal* t, RadialProfiles* m_p,
                const VmecConstants* constants, ThreadLocalStorage* m_ls,
                HandoverStorage* m_h, const RadialPartitioning* r,
                FreeBoundaryBase* m_fb, int signOfJacobian, int nvacskip,
                VacuumPressureState* m_vacuum_pressure_state);

  void setFromINDATA(int ncurr, double adiabaticIndex, double tCon0);

  // Compute the invariant (i.e., not preconditioned yet) force residuals.
  // Will put them into the provided array as { fsqr, fsqz, fsql }.
  void evalFResInvar(const Eigen::VectorXd& localFResInvar);

  // Compute the preconditioned force residuals.
  // Will put them into the provided array as { fsqr1, fsqz1, fsql1 }.
  void evalFResPrecd(const Eigen::VectorXd& localFResPrecd);

  // Return true/false depending on whether the VmecCheckpoint was reached,
  // or an error status if something went wrong.
  absl::StatusOr<bool> update(
      FourierGeometry& m_decomposed_x, FourierGeometry& m_physical_x,
      FourierForces& m_decomposed_f, FourierForces& m_physical_f,
      bool& m_need_restart, int& m_last_preconditioner_update,
      int& m_last_full_update_nestor, FlowControl& m_fc, const int iter1,
      const int iter2, const VmecCheckpoint& checkpoint = VmecCheckpoint::NONE,
      const int iterations_before_checkpointing = INT_MAX, bool verbose = true);

  // Coordinates which inverse-DFT routine to call for computing
  // the flux surface geometry and lambda on it from the provided Fourier
  // coefficients. Also computes the net dR/dTheta and dZ/dTheta, without the
  // even-m/odd-m split. Also computes the radial extent and geometric offset of
  // the flux surface geometry.
  void geometryFromFourier(const FourierGeometry& physical_x);

  // Inverse-DFT for flux surface geometry and lambda, 3D (Stellarator) case
  // Dispatching dft_FourierToReal_3d_symm
  void dft_FourierToReal_3d_symm(const FourierGeometry& physical_x);

  // Inverse-DFT for flux surface geometry and lambda, 2D axisymmetric (Tokamak)
  // case
  void dft_FourierToReal_2d_symm(const FourierGeometry& physical_x);

  // Extrapolates ingredients for the spectral condensation force
  // from the LCFS into the plasma volume.
  void rzConIntoVolume();

  // Computes the Jacobian sqrt(g) and its ingredients.
  void computeJacobian();

  // Computes the metric elements g_uu, g_uv, g_vv.
  void computeMetricElements();

  // Computes the differential volume profile dV/ds.
  void updateDifferentialVolume();

  // Computes the plasma volume of the initial guess, i.e.,
  // assuming the LCFS geometry provided in the input file.
  void computeInitialVolume();

  // Computes the plasma volume during the iterations.
  void updateVolume();

  // Computes the contravariant magnetic field components B^theta and B^zeta.
  // This also applies the toroidal current profile constraint if `ncurr==1` in
  // the input.
  void computeBContra();

  // Computes the covariant magnetic field components
  // from the contravariant magnetic field components and the metric elements.
  void computeBCo();

  // Computes total pressure (kinetic plus magnetic) as well as the
  // kinetic/thermal and magnetic energy.
  void pressureAndEnergies();

  // Computes the radial force balance (or better: residual imbalance)
  void radialForceBalance();

  // Computes the force on the lambda state variable (which is the covariant
  // magnetic field on the full-grid) using a mixture of two different numerical
  // approaches for increased numerical accuracy.
  void hybridLambdaForce();

  // Computes normalizing factors for the force residuals.
  void computeForceNorms(const FourierGeometry& decomposed_x);

  // Computes the MHD forces in realspace.
  void computeMHDForces();

  // Computes a radial profile of a scaling factor for the constraint force.
  // Current working hypothesis: This is used to make the constraint force "look
  // similar" to the MHD forces for improved numerical stability.
  absl::Status constraintForceMultiplier();

  // Computes the effective constraint force that actually enters the iterative
  // scheme.
  void effectiveConstraintForce();

  // De-aliases the effective constraint force by bandpass filtering in Fourier
  // space. Think of aliasing in terms of Fourier components higher than the
  // Nyquist frequency.
  void deAliasConstraintForce();

  // Assembles the total forces (MHD, spectral constraint, free-boundary).
  void assembleTotalForces();

  // Coordinates the forward-DFT to transform the total force in realspace into
  // Fourier space.
  void forcesToFourier(FourierForces& m_physical_f);

  // Computes the forward-DFT of forces for the 3D (Stellarator) case.
  // Dispatching dft_ForcesToFourier_3d_symm
  void dft_ForcesToFourier_3d_symm(FourierForces& m_physical_f);

  // Computes the forward-DFT of forces for the 2D axisymmetric (Tokamak) case.
  void dft_ForcesToFourier_2d_symm(FourierForces& m_physical_f);

  // Checks if the radial preconditioner matrix elements should be updated.
  // They don't change so much during iterations, so one can get away with
  // computing them only ever so often (as of now: every 25 iterations).
  bool shouldUpdateRadialPreconditioner(int iter1, int iter2) const;

  // Computes the radial preconditioner matrix elements for R and Z.
  void updateRadialPreconditioner();

  // Computes the radial preconditioner matrix elements for lambda.
  void updateLambdaPreconditioner();

  // Support function for computing the radial preconditioner matrix elements
  // for R and Z.
  void computePreconditioningMatrix(
      const Eigen::VectorXd& xs, const Eigen::VectorXd& xu12,
      const Eigen::VectorXd& xu_e, const Eigen::VectorXd& xu_o,
      const Eigen::VectorXd& x1_o, Eigen::VectorXd& m_axm,
      Eigen::VectorXd& m_axd, Eigen::VectorXd& m_bxm, Eigen::VectorXd& m_bxd,
      Eigen::VectorXd& m_cxd);

  // Applies the radial preconditioner for the m=1 Fourier coefficients of R and
  // Z.
  void applyM1Preconditioner(FourierForces& m_decomposed_f);

  // Assembles the preconditioner matrix elements for R and Z into the actual
  // preconditioner matrix.
  void assembleRZPreconditioner();

  // Applies the radial preconditioner for R and Z (solves a tri-diagonal system
  // of equations).
  absl::Status applyRZPreconditioner(FourierForces& m_decomposed_f);

  // Applies the radial preconditioner for lambda.
  void applyLambdaPreconditioner(FourierForces& m_decomposed_f);

  // Computes the mismatch in |B|^2 at the LCFS.
  double get_delbsq() const;

  // `ivacskip` is the current counter that controls whether a full update or a
  // partial update of the Nestor free boundary force contribution is computed.
  int get_ivacskip() const;

  /**********************************************/

  // R on full-grid
  Eigen::VectorXd r1_e;
  Eigen::VectorXd r1_o;

  // dRdTheta on full-grid
  Eigen::VectorXd ru_e;
  Eigen::VectorXd ru_o;

  // dRdZeta on full-grid
  Eigen::VectorXd rv_e;
  Eigen::VectorXd rv_o;

  // Z on full-grid
  Eigen::VectorXd z1_e;
  Eigen::VectorXd z1_o;

  // dZdTheta on full-grid
  Eigen::VectorXd zu_e;
  Eigen::VectorXd zu_o;

  // dZdZeta on full-grid
  Eigen::VectorXd zv_e;
  Eigen::VectorXd zv_o;

  // d(lambda)dTheta on full-grid
  Eigen::VectorXd lu_e;
  Eigen::VectorXd lu_o;

  // d(lambda)dZeta on full-grid
  Eigen::VectorXd lv_e;
  Eigen::VectorXd lv_o;

  // constraint force contribution X on full-grid
  Eigen::VectorXd rCon;

  // constraint force contribution Y on full-grid
  // In free-boundary this starts as a large value and is slowly reduced to zero
  // to gradually increase the vacuum pressure constraint (force felt from the
  // B^2 contribution).
  Eigen::VectorXd zCon;

  // initial constraint force contribution X on full-grid.
  // In free-boundary this starts as a large value and is slowly reduced to zero
  // to gradually increase the vacuum pressure constraint (force felt from the
  // B^2 contribution).
  Eigen::VectorXd rCon0;

  // initial constraint force contribution Y on full-grid
  Eigen::VectorXd zCon0;

  // dRdTheta combined on full-grid
  Eigen::VectorXd ruFull;

  // dRdZeta combined on full-grid
  Eigen::VectorXd zuFull;

  /**********************************************/

  // R on half-grid
  Eigen::VectorXd r12;

  // dRdTheta on half-grid
  Eigen::VectorXd ru12;

  // dZdTheta on half-grid
  Eigen::VectorXd zu12;

  // dRdS on half-grid (without 0.5/sqrt(s) contrib)
  Eigen::VectorXd rs;

  // dZdS on half-grid (without 0.5/sqrt(s) contrib)
  Eigen::VectorXd zs;

  // sqrt(g)/R on half-grid
  Eigen::VectorXd tau;

  /**********************************************/

  // sqrt(g) == Jacobian on half-grid
  Eigen::VectorXd gsqrt;

  // metric elements
  Eigen::VectorXd guu;
  Eigen::VectorXd guv;
  Eigen::VectorXd gvv;

  /**********************************************/

  // contravariant magnetic field components
  Eigen::VectorXd bsupu;
  Eigen::VectorXd bsupv;

  /**********************************************/

  // covariant magnetic field components
  Eigen::VectorXd bsubu;
  Eigen::VectorXd bsubv;

  /**********************************************/

  // |B|^2/(2 mu_0) + p
  Eigen::VectorXd totalPressure;

  // r * |B_vac|^2 at LCFS
  Eigen::VectorXd rBSq;

  // (|B|^2/(2 mu_0) + p) on inside of LCFS
  Eigen::VectorXd insideTotalPressure;

  // mismatch in |B|^2 between plasma and vacuum regions at LCFS
  Eigen::VectorXd delBSq;

  /**********************************************/

  // real-space forces
  Eigen::VectorXd armn_e;
  Eigen::VectorXd armn_o;
  Eigen::VectorXd brmn_e;
  Eigen::VectorXd brmn_o;
  Eigen::VectorXd crmn_e;
  Eigen::VectorXd crmn_o;
  // ---------
  Eigen::VectorXd azmn_e;
  Eigen::VectorXd azmn_o;
  Eigen::VectorXd bzmn_e;
  Eigen::VectorXd bzmn_o;
  Eigen::VectorXd czmn_e;
  Eigen::VectorXd czmn_o;
  // ---------
  Eigen::VectorXd blmn_e;
  Eigen::VectorXd blmn_o;
  Eigen::VectorXd clmn_e;
  Eigen::VectorXd clmn_o;

  /**********************************************/

  // lambda preconditioner
  Eigen::VectorXd bLambda;
  Eigen::VectorXd dLambda;
  Eigen::VectorXd cLambda;
  Eigen::VectorXd lambdaPreconditioner;

  // R,Z preconditioner
  Eigen::VectorXd ax;
  Eigen::VectorXd bx;
  Eigen::VectorXd cx;

  Eigen::VectorXd arm;
  Eigen::VectorXd ard;
  Eigen::VectorXd brm;
  Eigen::VectorXd brd;
  Eigen::VectorXd azm;
  Eigen::VectorXd azd;
  Eigen::VectorXd bzm;
  Eigen::VectorXd bzd;
  // crd == czd --> cxd
  Eigen::VectorXd cxd;

  Eigen::VectorXd ar;
  Eigen::VectorXd dr;
  Eigen::VectorXd br;
  Eigen::VectorXd az;
  Eigen::VectorXd dz;
  Eigen::VectorXd bz;

  /**********************************************/

  // constraint force ingredients
  Eigen::VectorXd xmpq;
  Eigen::VectorXd faccon;

  // radial profile of constraint force multiplier
  Eigen::VectorXd tcon;

  // effective constraint force - still to be de-aliased
  Eigen::VectorXd gConEff;

  // Fourier coefficients of constraint force - used during de-aliasing
  Eigen::VectorXd gsc;
  Eigen::VectorXd gcs;

  // de-aliased constraint force - what enters the Fourier coefficients of the
  // forces
  Eigen::VectorXd gCon;

  // Fourier coefficients of constraint force, de-aliased
  Eigen::VectorXd frcon_e;
  Eigen::VectorXd frcon_o;
  Eigen::VectorXd fzcon_e;
  Eigen::VectorXd fzcon_o;

 private:
  FlowControl& m_fc_;
  const Sizes& s_;
  const FourierBasisFastPoloidal& t_;
  RadialProfiles& m_p_;
  const VmecConstants& constants_;
  ThreadLocalStorage& m_ls_;
  HandoverStorage& m_h_;
  const RadialPartitioning& r_;
  FreeBoundaryBase* m_fb_;
  VacuumPressureState& m_vacuum_pressure_state_;

  // Pre-computed FFTW plans for the toroidal (zeta) Fourier transforms,
  // allocated only when mpol*(ntor+1) > kFftThreshold (see ideal_mhd_model.cc);
  // otherwise nullptr and the dft_FourierToReal_3d_symm path falls back to the
  // partial-DFT routine.  At small spectral resolutions the FFT plan-execute
  // dispatch overhead exceeds the asymptotic savings, so DFT is faster there.
  // Created once at construction (single-threaded context) and reused across
  // iterations.  Execution is thread-safe when using separate input/output
  // buffers (which the FFT transform functions allocate locally).
  std::unique_ptr<ToroidalFftPlans> fft_plans_;

  int signOfJacobian;

  // 1/4: 1/2 from d(sHalf)/ds and 1/2 from interpolation
  static constexpr double dSHalfDsInterp = 0.25;

  // TODO(jons): understand what this is (related to radial preconditioner)
  static constexpr double dampingFactor = 2.0;

  // from INDATA: flag to select between constrained-iota and
  // constrained-toroidal-current
  int ncurr;

  // from INDATA: adiabatic index == gamma
  double adiabaticIndex;

  // from INDATA: constraint force scaling parameter; between 0 and 1
  // 0 -- no spectral condensation constraint force
  // 1 (default) -- full spectral condensation constraint force
  double tcon0;

  // [mnsize] minimum flux surface index for which to apply radial
  // preconditioner for R and Z
  Eigen::VectorXi jMin;

  // ****** IDENTICAL THREAD LOCALS *******
  // In multi-thread runs, the following data members
  // will take identical values in all instances of
  // IdealMHDModel.
  //
  // Having one copy of the data member per thread has
  // a negligible memory cost and removes the need of
  // synchronization around a single global copy.

  // on-the-fly adjusted (--> <= nvskip0) nvacskip
  int nvacskip;

  // counter how many vacuum iterations have passed since last full update
  // --> counts modulo nvacskip
  int ivacskip;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_

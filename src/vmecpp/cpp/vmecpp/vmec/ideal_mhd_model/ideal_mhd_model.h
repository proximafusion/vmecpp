// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_

#include <Eigen/Dense>
#include <climits>
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
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"
#include "vmecpp/vmec/thread_local_storage/thread_local_storage.h"
#include "vmecpp/vmec/vmec_constants/vmec_constants.h"

namespace vmecpp {

// Implemented as a free function for easier testing and benchmarking.
// "FastPoloidal" indicates that, in real space, iterations use the
// poloidal coordinate as the fast index.
void ForcesToFourier3DSymmFastPoloidal(
    const RealSpaceForces& d,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

// Implemented as a free function for easier testing and benchmarking.
// "FastPoloidal" indicates that, in real space, iterations use the
// poloidal coordinate as the fast index.
void FourierToReal3DSymmFastPoloidal(
    const FourierGeometry& physical_x,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, RealSpaceGeometry& m_geometry);

// Implemented as a free function for easier testing and benchmarking.
void deAliasConstraintForce(
    const RadialPartitioning& rp, const FourierBasisFastPoloidal& fb,
    const Sizes& s_, const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& faccon,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& tcon,
    const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& gConEff,
    Eigen::Matrix<real_t, Eigen::Dynamic, 1>& m_gsc,
    Eigen::Matrix<real_t, Eigen::Dynamic, 1>& m_gcs,
    Eigen::Matrix<real_t, Eigen::Dynamic, 1>& m_gCon);

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
  void evalFResInvar(
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& localFResInvar);

  // Compute the preconditioned force residuals.
  // Will put them into the provided array as { fsqr1, fsqz1, fsql1 }.
  void evalFResPrecd(
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& localFResPrecd);

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
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& xs,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& xu12,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& xu_e,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& xu_o,
      const Eigen::Matrix<real_t, Eigen::Dynamic, 1>& x1_o,
      Eigen::Matrix<real_t, Eigen::Dynamic, 1>& m_axm,
      Eigen::Matrix<real_t, Eigen::Dynamic, 1>& m_axd,
      Eigen::Matrix<real_t, Eigen::Dynamic, 1>& m_bxm,
      Eigen::Matrix<real_t, Eigen::Dynamic, 1>& m_bxd,
      Eigen::Matrix<real_t, Eigen::Dynamic, 1>& m_cxd);

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
  real_t get_delbsq() const;

  // `ivacskip` is the current counter that controls whether a full update or a
  // partial update of the Nestor free boundary force contribution is computed.
  int get_ivacskip() const;

  /**********************************************/

  // R on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> r1_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> r1_o;

  // dRdTheta on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ru_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ru_o;

  // dRdZeta on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rv_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rv_o;

  // Z on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> z1_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> z1_o;

  // dZdTheta on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zu_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zu_o;

  // dZdZeta on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zv_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zv_o;

  // d(lambda)dTheta on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> lu_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> lu_o;

  // d(lambda)dZeta on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> lv_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> lv_o;

  // constraint force contribution X on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rCon;

  // constraint force contribution Y on full-grid
  // In free-boundary this starts as a large value and is slowly reduced to zero
  // to gradually increase the vacuum pressure constraint (force felt from the
  // B^2 contribution).
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zCon;

  // initial constraint force contribution X on full-grid.
  // In free-boundary this starts as a large value and is slowly reduced to zero
  // to gradually increase the vacuum pressure constraint (force felt from the
  // B^2 contribution).
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rCon0;

  // initial constraint force contribution Y on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zCon0;

  // dRdTheta combined on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ruFull;

  // dRdZeta combined on full-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zuFull;

  /**********************************************/

  // R on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> r12;

  // dRdTheta on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ru12;

  // dZdTheta on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zu12;

  // dRdS on half-grid (without 0.5/sqrt(s) contrib)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rs;

  // dZdS on half-grid (without 0.5/sqrt(s) contrib)
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zs;

  // sqrt(g)/R on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> tau;

  /**********************************************/

  // sqrt(g) == Jacobian on half-grid
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gsqrt;

  // metric elements
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> guu;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> guv;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gvv;

  /**********************************************/

  // contravariant magnetic field components
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bsupu;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bsupv;

  /**********************************************/

  // covariant magnetic field components
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bsubu;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bsubv;

  /**********************************************/

  // |B|^2/(2 mu_0) + p
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> totalPressure;

  // r * |B_vac|^2 at LCFS
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rBSq;

  // (|B|^2/(2 mu_0) + p) on inside of LCFS
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> insideTotalPressure;

  // mismatch in |B|^2 between plasma and vacuum regions at LCFS
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> delBSq;

  /**********************************************/

  // real-space forces
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> armn_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> armn_o;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> brmn_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> brmn_o;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> crmn_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> crmn_o;
  // ---------
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> azmn_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> azmn_o;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bzmn_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bzmn_o;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> czmn_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> czmn_o;
  // ---------
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> blmn_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> blmn_o;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> clmn_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> clmn_o;

  /**********************************************/

  // lambda preconditioner
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bLambda;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> dLambda;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> cLambda;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> lambdaPreconditioner;

  // R,Z preconditioner
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ax;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bx;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> cx;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> arm;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ard;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> brm;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> brd;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> azm;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> azd;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bzm;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bzd;
  // crd == czd --> cxd
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> cxd;

  Eigen::Matrix<real_t, Eigen::Dynamic, 1> ar;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> dr;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> br;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> az;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> dz;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> bz;

  /**********************************************/

  // constraint force ingredients
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> xmpq;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> faccon;

  // radial profile of constraint force multiplier
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> tcon;

  // effective constraint force - still to be de-aliased
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gConEff;

  // Fourier coefficients of constraint force - used during de-aliasing
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gsc;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gcs;

  // de-aliased constraint force - what enters the Fourier coefficients of the
  // forces
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> gCon;

  // Fourier coefficients of constraint force, de-aliased
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> frcon_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> frcon_o;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> fzcon_e;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> fzcon_o;

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

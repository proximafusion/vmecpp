// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_IDEAL_MHD_MODEL_H_

#include <climits>
#include <span>
#include <vector>

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
    const RealSpaceForces& d, const std::vector<double>& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

// Implemented as a free function for easier testing and benchmarking.
// "FastPoloidal" indicates that, in real space, iterations use the
// poloidal coordinate as the fast index.
void FourierToReal3DSymmFastPoloidal(const FourierGeometry& physical_x,
                                     const std::vector<double>& xmpq,
                                     const RadialPartitioning& r,
                                     const Sizes& s, const RadialProfiles& rp,
                                     const FourierBasisFastPoloidal& fb,
                                     RealSpaceGeometry& m_geometry);

// Implemented as a free function for easier testing and benchmarking.
void deAliasConstraintForce(const RadialPartitioning& rp,
                            const FourierBasisFastPoloidal& fb, const Sizes& s_,
                            const std::vector<double>& faccon,
                            const std::vector<double>& tcon,
                            const std::vector<double>& gConEff,
                            std::vector<double>& m_gsc,
                            std::vector<double>& m_gcs,
                            std::vector<double>& m_gCon);

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
  void evalFResInvar(const std::vector<double>& localFResInvar);

  // Compute the preconditioned force residuals.
  // Will put them into the provided array as { fsqr1, fsqz1, fsql1 }.
  void evalFResPrecd(const std::vector<double>& localFResPrecd);

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
      const std::vector<double>& xs, const std::vector<double>& xu12,
      const std::vector<double>& xu_e, const std::vector<double>& xu_o,
      const std::vector<double>& x1_o, std::vector<double>& m_axm,
      std::vector<double>& m_axd, std::vector<double>& m_bxm,
      std::vector<double>& m_bxd, std::vector<double>& m_cxd);

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
  std::vector<double> r1_e;
  std::vector<double> r1_o;

  // dRdTheta on full-grid
  std::vector<double> ru_e;
  std::vector<double> ru_o;

  // dRdZeta on full-grid
  std::vector<double> rv_e;
  std::vector<double> rv_o;

  // Z on full-grid
  std::vector<double> z1_e;
  std::vector<double> z1_o;

  // dZdTheta on full-grid
  std::vector<double> zu_e;
  std::vector<double> zu_o;

  // dZdZeta on full-grid
  std::vector<double> zv_e;
  std::vector<double> zv_o;

  // d(lambda)dTheta on full-grid
  std::vector<double> lu_e;
  std::vector<double> lu_o;

  // d(lambda)dZeta on full-grid
  std::vector<double> lv_e;
  std::vector<double> lv_o;

  // constraint force contribution X on full-grid
  std::vector<double> rCon;

  // constraint force contribution Y on full-grid
  // In free-boundary this starts as a large value and is slowly reduced to zero
  // to gradually increase the vacuum pressure constraint (force felt from the
  // B^2 contribution).
  std::vector<double> zCon;

  // initial constraint force contribution X on full-grid.
  // In free-boundary this starts as a large value and is slowly reduced to zero
  // to gradually increase the vacuum pressure constraint (force felt from the
  // B^2 contribution).
  std::vector<double> rCon0;

  // initial constraint force contribution Y on full-grid
  std::vector<double> zCon0;

  // dRdTheta combined on full-grid
  std::vector<double> ruFull;

  // dRdZeta combined on full-grid
  std::vector<double> zuFull;

  /**********************************************/

  // R on half-grid
  std::vector<double> r12;

  // dRdTheta on half-grid
  std::vector<double> ru12;

  // dZdTheta on half-grid
  std::vector<double> zu12;

  // dRdS on half-grid (without 0.5/sqrt(s) contrib)
  std::vector<double> rs;

  // dZdS on half-grid (without 0.5/sqrt(s) contrib)
  std::vector<double> zs;

  // sqrt(g)/R on half-grid
  std::vector<double> tau;

  /**********************************************/

  // sqrt(g) == Jacobian on half-grid
  std::vector<double> gsqrt;

  // metric elements
  std::vector<double> guu;
  std::vector<double> guv;
  std::vector<double> gvv;

  /**********************************************/

  // contravariant magnetic field components
  std::vector<double> bsupu;
  std::vector<double> bsupv;

  /**********************************************/

  // covariant magnetic field components
  std::vector<double> bsubu;
  std::vector<double> bsubv;

  /**********************************************/

  // |B|^2/(2 mu_0) + p
  std::vector<double> totalPressure;

  // r * |B_vac|^2 at LCFS
  std::vector<double> rBSq;

  // (|B|^2/(2 mu_0) + p) on inside of LCFS
  std::vector<double> insideTotalPressure;

  // mismatch in |B|^2 between plasma and vacuum regions at LCFS
  std::vector<double> delBSq;

  /**********************************************/

  // real-space forces
  std::vector<double> armn_e;
  std::vector<double> armn_o;
  std::vector<double> brmn_e;
  std::vector<double> brmn_o;
  std::vector<double> crmn_e;
  std::vector<double> crmn_o;
  // ---------
  std::vector<double> azmn_e;
  std::vector<double> azmn_o;
  std::vector<double> bzmn_e;
  std::vector<double> bzmn_o;
  std::vector<double> czmn_e;
  std::vector<double> czmn_o;
  // ---------
  std::vector<double> blmn_e;
  std::vector<double> blmn_o;
  std::vector<double> clmn_e;
  std::vector<double> clmn_o;

  /**********************************************/

  // lambda preconditioner
  std::vector<double> bLambda;
  std::vector<double> dLambda;
  std::vector<double> cLambda;
  std::vector<double> lambdaPreconditioner;

  // R,Z preconditioner
  std::vector<double> ax;
  std::vector<double> bx;
  std::vector<double> cx;

  std::vector<double> arm;
  std::vector<double> ard;
  std::vector<double> brm;
  std::vector<double> brd;
  std::vector<double> azm;
  std::vector<double> azd;
  std::vector<double> bzm;
  std::vector<double> bzd;
  // crd == czd --> cxd
  std::vector<double> cxd;

  std::vector<double> ar;
  std::vector<double> dr;
  std::vector<double> br;
  std::vector<double> az;
  std::vector<double> dz;
  std::vector<double> bz;

  /**********************************************/

  // constraint force ingredients
  std::vector<double> xmpq;
  std::vector<double> faccon;

  // radial profile of constraint force multiplier
  std::vector<double> tcon;

  // effective constraint force - still to be de-aliased
  std::vector<double> gConEff;

  // Fourier coefficients of constraint force - used during de-aliasing
  std::vector<double> gsc;
  std::vector<double> gcs;

  // de-aliased constraint force - what enters the Fourier coefficients of the
  // forces
  std::vector<double> gCon;

  // Fourier coefficients of constraint force, de-aliased
  std::vector<double> frcon_e;
  std::vector<double> frcon_o;
  std::vector<double> fzcon_e;
  std::vector<double> fzcon_o;

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
  std::vector<int> jMin;

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

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_HANDOVER_STORAGE_HANDOVER_STORAGE_H_
#define VMECPP_VMEC_HANDOVER_STORAGE_HANDOVER_STORAGE_H_

#include <Eigen/Dense>
#include <cstddef>
#include <span>
#include <vector>

#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {

// Default values are set for accumulation.
// Note that these correspond to an invalid spectral width,
// as a division-by-zero would occur.
struct SpectralWidthContribution {
  double numerator = 0.0;
  double denominator = 0.0;
};

// Size of the plasma in radial direction.
struct RadialExtent {
  double r_outer = 0.0;
  double r_inner = 0.0;
};

struct GeometricOffset {
  double r_00 = 0.0;
  double z_00 = 0.0;
};

class HandoverStorage {
 public:
  explicit HandoverStorage(const Sizes* s);

  void allocate(const RadialPartitioning& r, int ns);

  // Each thread writes its contribution into its own slot row. There is no
  // separate reset step: the next iteration's writes overwrite the slot in
  // full. The volume-averaged spectral width is computed by summing the
  // slot columns lazily inside `VolumeAveragedSpectralWidth()`. This means
  // the previous reset/critical/barrier triplet around the spectral-width
  // accumulation is replaced by the outer barrier already present in the
  // caller (`Vmec::Printout`).
  void RegisterSpectralWidthContribution(
      int thread_id,
      const SpectralWidthContribution& spectral_width_contribution);
  double VolumeAveragedSpectralWidth() const;

  void SetRadialExtent(const RadialExtent& radial_extent);
  void SetGeometricOffset(const GeometricOffset& geometric_offset);

  RadialExtent GetRadialExtent() const;
  GeometricOffset GetGeometricOffset() const;

  // -------------------

  double thermalEnergy;
  double magneticEnergy;
  double mhdEnergy;

  /** plasma volume in m^3/(2pi)^2 */
  double plasmaVolume;

  // initial plasma volume (at start of multi-grid step) in m^3
  double voli;

  // force residual normalization factor for R and Z
  double fNormRZ;

  // force residual normalization factor for lambda
  double fNormL;

  // preconditioned force residual normalization factor for R, Z and lambda
  double fNorm1;

  // poloidal current at axis
  double rBtor0;

  // poloidal current at LCFS; rBtor / MU_0 is in Amperes
  double rBtor;

  // net enclosed toroidal current at LCFS; cTor / MU_0 is in Amperes
  double cTor;

  // net toroidal current from vacuum; bSubUVac / MU_0 is in Amperes
  double bSubUVac;

  // poloidal current at LCFS from vacuum; bSubVVac * 2 * pi / MU_0 is in
  // Amperes
  double bSubVVac;

  // Used only in rzConIntoVolume() to extrapolate the constraint force
  // contribution from the LCFS into the plasma volume.
  // TODO(jurasic) this should have a smaller scope.
  Eigen::VectorXd rCon_LCFS;
  Eigen::VectorXd zCon_LCFS;

  // Inter-thread handover storage: RowMatrixXd [num_threads, mnsize]
  // _i arrays: inside boundary, _o arrays: outside boundary

  RowMatrixXd rmncc_i;
  RowMatrixXd rmnss_i;
  RowMatrixXd zmnsc_i;
  RowMatrixXd zmncs_i;
  RowMatrixXd lmnsc_i;
  RowMatrixXd lmncs_i;
  // Asymmetric arrays for lasym=true
  RowMatrixXd rmnsc_i;
  RowMatrixXd rmncs_i;
  RowMatrixXd zmncc_i;
  RowMatrixXd zmnss_i;
  RowMatrixXd lmncc_i;
  RowMatrixXd lmnss_i;

  RowMatrixXd rmncc_o;
  RowMatrixXd rmnss_o;
  RowMatrixXd zmnsc_o;
  RowMatrixXd zmncs_o;
  RowMatrixXd lmnsc_o;
  RowMatrixXd lmncs_o;
  // Asymmetric arrays for lasym=true
  RowMatrixXd rmnsc_o;
  RowMatrixXd rmncs_o;
  RowMatrixXd zmncc_o;
  RowMatrixXd zmnss_o;
  RowMatrixXd lmncc_o;
  RowMatrixXd lmnss_o;

  // Serial tri-diagonal solver storage
  // Matrix: RowMatrixXd [mn, j], RHS: vector of RowMatrixXd [mn][basis, j]

  int mnsize;
  RowMatrixXd all_ar;  // [mnsize, ns]
  RowMatrixXd all_az;
  RowMatrixXd all_dr;
  RowMatrixXd all_dz;
  RowMatrixXd all_br;
  RowMatrixXd all_bz;

  // [mnsize] -> [num_basis, ns]
  std::vector<RowMatrixXd> all_cr;
  std::vector<RowMatrixXd> all_cz;

  // Parallel tri-diagonal solver storage
  // handover_cR/cZ: RowMatrixXd [num_basis, mnsize], handover_aR/aZ: flat
  // [mnsize]

  RowMatrixXd handover_cR;      // [num_basis, mnsize]
  Eigen::VectorXd handover_aR;  // [mnsize]
  RowMatrixXd handover_cZ;
  Eigen::VectorXd handover_aZ;
  // magnetic axis geometry for NESTOR
  Eigen::VectorXd rAxis;
  Eigen::VectorXd zAxis;

  // LCFS geometry for NESTOR
  Eigen::VectorXd rCC_LCFS;
  Eigen::VectorXd rSS_LCFS;
  Eigen::VectorXd rSC_LCFS;
  Eigen::VectorXd rCS_LCFS;
  Eigen::VectorXd zSC_LCFS;
  Eigen::VectorXd zCS_LCFS;
  Eigen::VectorXd zCC_LCFS;
  Eigen::VectorXd zSS_LCFS;

  // [nZnT] vacuum magnetic pressure |B_vac^2|/2 at the plasma boundary
  Eigen::VectorXd vacuum_magnetic_pressure;

  // [nZnT] cylindrical B^R of Nestor's vacuum magnetic field
  Eigen::VectorXd vacuum_b_r;

  // [nZnT] cylindrical B^phi of Nestor's vacuum magnetic field
  Eigen::VectorXd vacuum_b_phi;

  // [nZnT] cylindrical B^Z of Nestor's vacuum magnetic field
  Eigen::VectorXd vacuum_b_z;

  // Per-thread reduction slots for the residual reductions performed at the
  // end of IdealMhdModel::update(). Each thread writes its three local
  // residual contributions into row [thread_id]; one team barrier publishes
  // the writes; the global totals are then computed by a column-wise sum
  // (executed redundantly on every thread to avoid an extra rendezvous).
  // Using per-thread slots replaces the previous
  //   `single { reset } / atomic / barrier / single { finalize }`
  // pattern (three rendezvous) with a single explicit barrier plus a
  // `single nowait` publish (one rendezvous), which scales much better
  // under OMP_WAIT_POLICY=passive.
  RowMatrixXd fres_invar_slots_;  // [num_threads x 3]
  RowMatrixXd fres_precd_slots_;  // [num_threads x 3]
  RowMatrixXd energy_slots_;      // [num_threads x 2] -- {thermal, magnetic}
  RowMatrixXd spectral_width_slots_;  // [num_threads x 2] -- {numer, denom}

 private:
  const Sizes& s_;

  int num_threads_;
  int num_basis_;

  // Spectral-width accumulators were previously two scalars updated under a
  // critical section. They are now derived from `spectral_width_slots_` on
  // demand inside `VolumeAveragedSpectralWidth()`.

  RadialExtent radial_extent_;
  GeometricOffset geometric_offset_;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_HANDOVER_STORAGE_HANDOVER_STORAGE_H_

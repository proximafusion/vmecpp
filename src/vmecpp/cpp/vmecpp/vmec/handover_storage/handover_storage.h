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
#include "vmecpp/common/util/real_type.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {

// Default values are set for accumulation.
// Note that these correspond to an invalid spectral width,
// as a division-by-zero would occur.
struct SpectralWidthContribution {
  real_t numerator = 0.0;
  real_t denominator = 0.0;
};

// Size of the plasma in radial direction.
struct RadialExtent {
  real_t r_outer = 0.0;
  real_t r_inner = 0.0;
};

struct GeometricOffset {
  real_t r_00 = 0.0;
  real_t z_00 = 0.0;
};

class HandoverStorage {
 public:
  explicit HandoverStorage(const Sizes* s);

  void allocate(const RadialPartitioning& r, int ns);

  void ResetSpectralWidthAccumulators();
  void RegisterSpectralWidthContribution(
      const SpectralWidthContribution& spectral_width_contribution);
  real_t VolumeAveragedSpectralWidth() const;

  void SetRadialExtent(const RadialExtent& radial_extent);
  void SetGeometricOffset(const GeometricOffset& geometric_offset);

  RadialExtent GetRadialExtent() const;
  GeometricOffset GetGeometricOffset() const;

  // -------------------

  real_t thermalEnergy;
  real_t magneticEnergy;
  real_t mhdEnergy;

  /** plasma volume in m^3/(2pi)^2 */
  real_t plasmaVolume;

  // initial plasma volume (at start of multi-grid step) in m^3
  real_t voli;

  // force residual normalization factor for R and Z
  real_t fNormRZ;

  // force residual normalization factor for lambda
  real_t fNormL;

  // preconditioned force residual normalization factor for R, Z and lambda
  real_t fNorm1;

  // poloidal current at axis
  real_t rBtor0;

  // poloidal current at LCFS; rBtor / MU_0 is in Amperes
  real_t rBtor;

  // net enclosed toroidal current at LCFS; cTor / MU_0 is in Amperes
  real_t cTor;

  // net toroidal current from vacuum; bSubUVac / MU_0 is in Amperes
  real_t bSubUVac;

  // poloidal current at LCFS from vacuum; bSubVVac * 2 * pi / MU_0 is in
  // Amperes
  real_t bSubVVac;

  // Used only in rzConIntoVolume() to extrapolate the constraint force
  // contribution from the LCFS into the plasma volume.
  // TODO(jurasic) this should have a smaller scope.
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rCon_LCFS;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zCon_LCFS;

  // Inter-thread handover storage: RowMatrixXr [num_threads, mnsize]
  // _i arrays: inside boundary, _o arrays: outside boundary

  RowMatrixXr rmncc_i;
  RowMatrixXr rmnss_i;
  RowMatrixXr zmnsc_i;
  RowMatrixXr zmncs_i;
  RowMatrixXr lmnsc_i;
  RowMatrixXr lmncs_i;
  // Asymmetric arrays for lasym=true
  RowMatrixXr rmnsc_i;
  RowMatrixXr rmncs_i;
  RowMatrixXr zmncc_i;
  RowMatrixXr zmnss_i;
  RowMatrixXr lmncc_i;
  RowMatrixXr lmnss_i;

  RowMatrixXr rmncc_o;
  RowMatrixXr rmnss_o;
  RowMatrixXr zmnsc_o;
  RowMatrixXr zmncs_o;
  RowMatrixXr lmnsc_o;
  RowMatrixXr lmncs_o;
  // Asymmetric arrays for lasym=true
  RowMatrixXr rmnsc_o;
  RowMatrixXr rmncs_o;
  RowMatrixXr zmncc_o;
  RowMatrixXr zmnss_o;
  RowMatrixXr lmncc_o;
  RowMatrixXr lmnss_o;

  // Serial tri-diagonal solver storage
  // Matrix: RowMatrixXr [mn, j], RHS: vector of RowMatrixXr [mn][basis, j]

  int mnsize;
  RowMatrixXr all_ar;  // [mnsize, ns]
  RowMatrixXr all_az;
  RowMatrixXr all_dr;
  RowMatrixXr all_dz;
  RowMatrixXr all_br;
  RowMatrixXr all_bz;

  // [mnsize] -> [num_basis, ns]
  std::vector<RowMatrixXr> all_cr;
  std::vector<RowMatrixXr> all_cz;

  // Parallel tri-diagonal solver storage
  // handover_cR/cZ: RowMatrixXr [num_basis, mnsize], handover_aR/aZ: flat
  // [mnsize]

  RowMatrixXr handover_cR;                               // [num_basis, mnsize]
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> handover_aR;  // [mnsize]
  RowMatrixXr handover_cZ;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> handover_aZ;
  // magnetic axis geometry for NESTOR
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rAxis;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zAxis;

  // LCFS geometry for NESTOR
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rCC_LCFS;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rSS_LCFS;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rSC_LCFS;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> rCS_LCFS;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zSC_LCFS;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zCS_LCFS;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zCC_LCFS;
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> zSS_LCFS;

  // [nZnT] vacuum magnetic pressure |B_vac^2|/2 at the plasma boundary
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> vacuum_magnetic_pressure;

  // [nZnT] cylindrical B^R of Nestor's vacuum magnetic field
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> vacuum_b_r;

  // [nZnT] cylindrical B^phi of Nestor's vacuum magnetic field
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> vacuum_b_phi;

  // [nZnT] cylindrical B^Z of Nestor's vacuum magnetic field
  Eigen::Matrix<real_t, Eigen::Dynamic, 1> vacuum_b_z;

 private:
  const Sizes& s_;

  int num_threads_;
  int num_basis_;

  real_t spectral_width_numerator_;
  real_t spectral_width_denominator_;

  RadialExtent radial_extent_;
  GeometricOffset geometric_offset_;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_HANDOVER_STORAGE_HANDOVER_STORAGE_H_

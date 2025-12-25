// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_HANDOVER_STORAGE_HANDOVER_STORAGE_H_
#define VMECPP_VMEC_HANDOVER_STORAGE_HANDOVER_STORAGE_H_

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

  void ResetSpectralWidthAccumulators();
  void RegisterSpectralWidthContribution(
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
  std::vector<double> rCon_LCFS;
  std::vector<double> zCon_LCFS;

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

  RowMatrixXd handover_cR;          // [num_basis, mnsize]
  std::vector<double> handover_aR;  // [mnsize]
  RowMatrixXd handover_cZ;
  std::vector<double> handover_aZ;
  // magnetic axis geometry for NESTOR
  std::vector<double> rAxis;
  std::vector<double> zAxis;

  // LCFS geometry for NESTOR
  std::vector<double> rCC_LCFS;
  std::vector<double> rSS_LCFS;
  std::vector<double> rSC_LCFS;
  std::vector<double> rCS_LCFS;
  std::vector<double> zSC_LCFS;
  std::vector<double> zCS_LCFS;
  std::vector<double> zCC_LCFS;
  std::vector<double> zSS_LCFS;

  // [nZnT] vacuum magnetic pressure |B_vac^2|/2 at the plasma boundary
  std::vector<double> vacuum_magnetic_pressure;

  // [nZnT] cylindrical B^R of Nestor's vacuum magnetic field
  std::vector<double> vacuum_b_r;

  // [nZnT] cylindrical B^phi of Nestor's vacuum magnetic field
  std::vector<double> vacuum_b_phi;

  // [nZnT] cylindrical B^Z of Nestor's vacuum magnetic field
  std::vector<double> vacuum_b_z;

 private:
  const Sizes& s_;

  int num_threads_;
  int num_basis_;

  double spectral_width_numerator_;
  double spectral_width_denominator_;

  RadialExtent radial_extent_;
  GeometricOffset geometric_offset_;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_HANDOVER_STORAGE_HANDOVER_STORAGE_H_

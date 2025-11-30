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

  // =========================================================================
  // Fourier coefficient handover storage for inter-thread communication
  // =========================================================================
  // Layout: flat [thread * mnsize + mn], contiguous per thread
  // Access: FourierCoeff(thread, mn) for element access
  //
  // _i arrays: inside boundary of target thread (from previous thread)
  // _o arrays: outside boundary of target thread (from next thread)

  std::vector<double> rmncc_i;
  std::vector<double> rmnss_i;
  std::vector<double> zmnsc_i;
  std::vector<double> zmncs_i;
  std::vector<double> lmnsc_i;
  std::vector<double> lmncs_i;
  // Asymmetric arrays for lasym=true
  std::vector<double> rmnsc_i;
  std::vector<double> rmncs_i;
  std::vector<double> zmncc_i;
  std::vector<double> zmnss_i;
  std::vector<double> lmncc_i;
  std::vector<double> lmnss_i;

  std::vector<double> rmncc_o;
  std::vector<double> rmnss_o;
  std::vector<double> zmnsc_o;
  std::vector<double> zmncs_o;
  std::vector<double> lmnsc_o;
  std::vector<double> lmncs_o;
  // Asymmetric arrays for lasym=true
  std::vector<double> rmnsc_o;
  std::vector<double> rmncs_o;
  std::vector<double> zmncc_o;
  std::vector<double> zmnss_o;
  std::vector<double> lmncc_o;
  std::vector<double> lmnss_o;

  // Index helper for Fourier coefficient arrays: [thread][mn]
  inline std::size_t IdxFourier(int thread, int mn) const noexcept {
    return static_cast<std::size_t>(thread) * mnsize + mn;
  }

  // =========================================================================
  // Radial preconditioner storage for serial tri-diagonal solver
  // =========================================================================
  // Tri-diagonal matrix arrays: layout [mn * ns + j]
  // Radial dimension (j) is contiguous for optimal solver access.
  //
  // RHS arrays: layout [(mn * num_basis + k) * ns + j]
  // For each mode mn, the num_basis RHS vectors are stored contiguously,
  // with radial dimension innermost.

  int mnsize;
  int ns_ = 0;  // number of radial surfaces (set in allocate)

  std::vector<double> all_ar;  // [mnsize * ns]
  std::vector<double> all_az;
  std::vector<double> all_dr;
  std::vector<double> all_dz;
  std::vector<double> all_br;
  std::vector<double> all_bz;
  std::vector<double> all_cr;  // [mnsize * num_basis * ns]
  std::vector<double> all_cz;

  // Index helper for tri-diagonal matrix arrays: [mn][j]
  inline std::size_t IdxTridiag(int mn, int j) const noexcept {
    return static_cast<std::size_t>(mn) * ns_ + j;
  }

  // Index helper for tri-diagonal RHS arrays: [mn][k][j]
  inline std::size_t IdxTridiagRhs(int mn, int k, int j) const noexcept {
    return (static_cast<std::size_t>(mn) * num_basis_ + k) * ns_ + j;
  }

  // Span accessors for tri-diagonal solver (returns view of radial slice)
  inline std::span<double> TridiagAr(int mn) noexcept {
    return std::span<double>(all_ar.data() + mn * ns_, ns_);
  }
  inline std::span<double> TridiagAz(int mn) noexcept {
    return std::span<double>(all_az.data() + mn * ns_, ns_);
  }
  inline std::span<double> TridiagDr(int mn) noexcept {
    return std::span<double>(all_dr.data() + mn * ns_, ns_);
  }
  inline std::span<double> TridiagDz(int mn) noexcept {
    return std::span<double>(all_dz.data() + mn * ns_, ns_);
  }
  inline std::span<double> TridiagBr(int mn) noexcept {
    return std::span<double>(all_br.data() + mn * ns_, ns_);
  }
  inline std::span<double> TridiagBz(int mn) noexcept {
    return std::span<double>(all_bz.data() + mn * ns_, ns_);
  }

  // RHS span accessor: returns pointer to start of RHS block for mode mn
  // The solver accesses c[k][j] as base[k * ns + j]
  inline double* TridiagCrData(int mn) noexcept {
    return all_cr.data() + static_cast<std::size_t>(mn) * num_basis_ * ns_;
  }
  inline double* TridiagCzData(int mn) noexcept {
    return all_cz.data() + static_cast<std::size_t>(mn) * num_basis_ * ns_;
  }

  // Dimension accessor for solver
  int GetNumBasis() const noexcept { return num_basis_; }
  int GetNs() const noexcept { return ns_; }

  // =========================================================================
  // Radial preconditioner storage for parallel tri-diagonal solver
  // =========================================================================
  // handover_cR/cZ: layout [k * mnsize + mn]
  // handover_aR/aZ: already flat [mnsize]

  std::vector<double> handover_cR;  // [num_basis * mnsize]
  std::vector<double> handover_aR;  // [mnsize]
  std::vector<double> handover_cZ;
  std::vector<double> handover_aZ;

  // Index helper for parallel handover arrays: [k][mn]
  inline std::size_t IdxHandover(int k, int mn) const noexcept {
    return static_cast<std::size_t>(k) * mnsize + mn;
  }

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

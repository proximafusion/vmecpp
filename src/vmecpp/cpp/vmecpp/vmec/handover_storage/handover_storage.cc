// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/handover_storage/handover_storage.h"

#include <iostream>

namespace vmecpp {

HandoverStorage::HandoverStorage(const Sizes* s) : s_(*s) {
  plasmaVolume = 0.0;
  voli = 0.0;

  fNormRZ = 0.0;
  fNormL = 0.0;
  fNorm1 = 0.0;

  thermalEnergy = 0.0;
  magneticEnergy = 0.0;
  mhdEnergy = 0.0;

  rBtor0 = 0.0;
  rBtor = 0.0;
  cTor = 0.0;

  bSubUVac = 0.0;
  bSubVVac = 0.0;

  rCon_LCFS.resize(s_.nZnT);
  zCon_LCFS.resize(s_.nZnT);
  vacuum_magnetic_pressure.resize(s_.nZnT);
  vacuum_b_r.resize(s_.nZnT);
  vacuum_b_phi.resize(s_.nZnT);
  vacuum_b_z.resize(s_.nZnT);

  num_threads_ = 1;
  num_basis_ = 0;

  mnsize = s_.mnsize;

  // Default values for accumulation.
  // Note that these correspond to an invalid spectral width,
  // as a division-by-zero would occur.
  spectral_width_numerator_ = 0.0;
  spectral_width_denominator_ = 0.0;

  rAxis.resize(s_.nZeta);
  zAxis.resize(s_.nZeta);

  rCC_LCFS.resize(mnsize);
  rSS_LCFS.resize(mnsize);
  zSC_LCFS.resize(mnsize);
  zCS_LCFS.resize(mnsize);
  if (s_.lasym) {
    rSC_LCFS.resize(mnsize);
    rCS_LCFS.resize(mnsize);
    zCC_LCFS.resize(mnsize);
    zSS_LCFS.resize(mnsize);
  }
}

// called from serial region now
void HandoverStorage::allocate(const RadialPartitioning& r, int ns) {
  // only 1 thread allocates all storage
  if (r.get_thread_id() == 0) {
    num_threads_ = r.get_num_threads();
    num_basis_ = s_.num_basis;
    ns_ = ns;

    // =========================================================================
    // Fourier coefficient handover storage
    // =========================================================================
    // Layout: flat [thread * mnsize + mn]
    // Allocate full array for all threads; unused portions remain zero.
    const std::size_t fourier_size =
        static_cast<std::size_t>(num_threads_) * mnsize;

    rmncc_i.assign(fourier_size, 0.0);
    zmnsc_i.assign(fourier_size, 0.0);
    lmnsc_i.assign(fourier_size, 0.0);

    rmncc_o.assign(fourier_size, 0.0);
    zmnsc_o.assign(fourier_size, 0.0);
    lmnsc_o.assign(fourier_size, 0.0);

    if (s_.lthreed) {
      rmnss_i.assign(fourier_size, 0.0);
      zmncs_i.assign(fourier_size, 0.0);
      lmncs_i.assign(fourier_size, 0.0);

      rmnss_o.assign(fourier_size, 0.0);
      zmncs_o.assign(fourier_size, 0.0);
      lmncs_o.assign(fourier_size, 0.0);
    }

    if (s_.lasym) {
      rmnsc_i.assign(fourier_size, 0.0);
      zmncc_i.assign(fourier_size, 0.0);
      lmncc_i.assign(fourier_size, 0.0);

      rmnsc_o.assign(fourier_size, 0.0);
      zmncc_o.assign(fourier_size, 0.0);
      lmncc_o.assign(fourier_size, 0.0);

      if (s_.lthreed) {
        rmncs_i.assign(fourier_size, 0.0);
        zmnss_i.assign(fourier_size, 0.0);
        lmnss_i.assign(fourier_size, 0.0);

        rmncs_o.assign(fourier_size, 0.0);
        zmnss_o.assign(fourier_size, 0.0);
        lmnss_o.assign(fourier_size, 0.0);
      }
    }

    // =========================================================================
    // Tri-diagonal solver storage
    // =========================================================================
    // Matrix arrays: layout [mn * ns + j], size = mnsize * ns
    const std::size_t tridiag_size = static_cast<std::size_t>(mnsize) * ns;
    all_ar.assign(tridiag_size, 0.0);
    all_az.assign(tridiag_size, 0.0);
    all_dr.assign(tridiag_size, 0.0);
    all_dz.assign(tridiag_size, 0.0);
    all_br.assign(tridiag_size, 0.0);
    all_bz.assign(tridiag_size, 0.0);

    // RHS arrays: layout [(mn * num_basis + k) * ns + j]
    // size = mnsize * num_basis * ns
    const std::size_t tridiag_rhs_size = tridiag_size * num_basis_;
    all_cr.assign(tridiag_rhs_size, 0.0);
    all_cz.assign(tridiag_rhs_size, 0.0);

    // =========================================================================
    // Parallel tri-diagonal solver handover storage
    // =========================================================================
    // handover_cR/cZ: layout [k * mnsize + mn], size = num_basis * mnsize
    const std::size_t handover_c_size =
        static_cast<std::size_t>(num_basis_) * mnsize;
    handover_cR.assign(handover_c_size, 0.0);
    handover_cZ.assign(handover_c_size, 0.0);

    // handover_aR/aZ: flat [mnsize]
    handover_aR.assign(mnsize, 0.0);
    handover_aZ.assign(mnsize, 0.0);
  }

  // Note: With flat storage, individual thread allocation is no longer needed.
  // All threads access their portions via index helpers:
  //   IdxFourier(thread, mn) for Fourier coefficients
  //   IdxTridiag(mn, j) for tri-diagonal matrix elements
  //   IdxTridiagRhs(mn, k, j) for tri-diagonal RHS elements
}  // allocate

void HandoverStorage::ResetSpectralWidthAccumulators() {
  spectral_width_numerator_ = 0.0;
  spectral_width_denominator_ = 0.0;
}  // ResetSpectralWidthAccumulators

void HandoverStorage::RegisterSpectralWidthContribution(
    const SpectralWidthContribution& spectral_width_contribution) {
  spectral_width_numerator_ += spectral_width_contribution.numerator;
  spectral_width_denominator_ += spectral_width_contribution.denominator;
}  // RegisterSpectralWidthContribution

double HandoverStorage::VolumeAveragedSpectralWidth() const {
  return spectral_width_numerator_ / spectral_width_denominator_;
}  // VolumeAveragedSpectralWidth

void HandoverStorage::SetRadialExtent(const RadialExtent& radial_extent) {
  radial_extent_ = radial_extent;
}  // SetRadialExtent

void HandoverStorage::SetGeometricOffset(
    const GeometricOffset& geometric_offset) {
  geometric_offset_ = geometric_offset;
}  // SetGeometricOffset

RadialExtent HandoverStorage::GetRadialExtent() const {
  return radial_extent_;
}  // GetRadialExtent

GeometricOffset HandoverStorage::GetGeometricOffset() const {
  return geometric_offset_;
}  // GetGeometricOffset

}  // namespace vmecpp

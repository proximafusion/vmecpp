// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/vmec/handover_storage/handover_storage.h"

#include <iostream>

namespace vmecpp {

HandoverStorage::HandoverStorage(const Sizes* s) : s_(*s) {
  plasmaVolume = 0.0;

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
  // only 1 thread allocates
  if (r.get_thread_id() == 0) {
    num_threads_ = r.get_num_threads();
    num_basis_ = s_.num_basis;

    // -----------
    // Fourier coefficient handover storage
    // -----------
    // Layout: RowMatrixXd [num_threads, mnsize]

    rmncc_i.resize(num_threads_, mnsize);
    rmncc_i.setZero();
    zmnsc_i.resize(num_threads_, mnsize);
    zmnsc_i.setZero();
    lmnsc_i.resize(num_threads_, mnsize);
    lmnsc_i.setZero();

    rmncc_o.resize(num_threads_, mnsize);
    rmncc_o.setZero();
    zmnsc_o.resize(num_threads_, mnsize);
    zmnsc_o.setZero();
    lmnsc_o.resize(num_threads_, mnsize);
    lmnsc_o.setZero();

    if (s_.lthreed) {
      rmnss_i.resize(num_threads_, mnsize);
      rmnss_i.setZero();
      zmncs_i.resize(num_threads_, mnsize);
      zmncs_i.setZero();
      lmncs_i.resize(num_threads_, mnsize);
      lmncs_i.setZero();

      rmnss_o.resize(num_threads_, mnsize);
      rmnss_o.setZero();
      zmncs_o.resize(num_threads_, mnsize);
      zmncs_o.setZero();
      lmncs_o.resize(num_threads_, mnsize);
      lmncs_o.setZero();
    }

    if (s_.lasym) {
      rmnsc_i.resize(num_threads_, mnsize);
      rmnsc_i.setZero();
      zmncc_i.resize(num_threads_, mnsize);
      zmncc_i.setZero();
      lmncc_i.resize(num_threads_, mnsize);
      lmncc_i.setZero();

      rmnsc_o.resize(num_threads_, mnsize);
      rmnsc_o.setZero();
      zmncc_o.resize(num_threads_, mnsize);
      zmncc_o.setZero();
      lmncc_o.resize(num_threads_, mnsize);
      lmncc_o.setZero();

      if (s_.lthreed) {
        rmncs_i.resize(num_threads_, mnsize);
        rmncs_i.setZero();
        zmnss_i.resize(num_threads_, mnsize);
        zmnss_i.setZero();
        lmnss_i.resize(num_threads_, mnsize);
        lmnss_i.setZero();

        rmncs_o.resize(num_threads_, mnsize);
        rmncs_o.setZero();
        zmnss_o.resize(num_threads_, mnsize);
        zmnss_o.setZero();
        lmnss_o.resize(num_threads_, mnsize);
        lmnss_o.setZero();
      }
    }

    // =========================================================================
    // Tri-diagonal solver storage
    // =========================================================================
    // Matrix arrays: RowMatrixXd [mn, ns]
    all_ar.resize(mnsize, ns);
    all_ar.setZero();
    all_az.resize(mnsize, ns);
    all_az.setZero();
    all_dr.resize(mnsize, ns);
    all_dr.setZero();
    all_dz.resize(mnsize, ns);
    all_dz.setZero();
    all_br.resize(mnsize, ns);
    all_br.setZero();
    all_bz.resize(mnsize, ns);
    all_bz.setZero();

    // RHS arrays: vector of RowMatrixXd [mn][num_basis, ns]
    all_cr.resize(mnsize);
    all_cz.resize(mnsize);
    for (int mn = 0; mn < mnsize; ++mn) {
      all_cr[mn].resize(num_basis_, ns);
      all_cr[mn].setZero();
      all_cz[mn].resize(num_basis_, ns);
      all_cz[mn].setZero();
    }

    // =========================================================================
    // Parallel tri-diagonal solver handover storage
    // =========================================================================
    // handover_cR/cZ: RowMatrixXd [num_basis, mnsize]
    handover_cR.resize(num_basis_, mnsize);
    handover_cR.setZero();
    handover_cZ.resize(num_basis_, mnsize);
    handover_cZ.setZero();

    // handover_aR/aZ: flat [mnsize]
    handover_aR.assign(mnsize, 0.0);
    handover_aZ.assign(mnsize, 0.0);
  }

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

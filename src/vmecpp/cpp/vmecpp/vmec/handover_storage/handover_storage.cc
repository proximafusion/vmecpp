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

    rmncc_i.resize(num_threads_);
    rmnss_i.resize(num_threads_);
    zmnsc_i.resize(num_threads_);
    zmncs_i.resize(num_threads_);
    lmnsc_i.resize(num_threads_);
    lmncs_i.resize(num_threads_);

    rmncc_o.resize(num_threads_);
    rmnss_o.resize(num_threads_);
    zmnsc_o.resize(num_threads_);
    zmncs_o.resize(num_threads_);
    lmnsc_o.resize(num_threads_);
    lmncs_o.resize(num_threads_);

    // global accumulator for serial tri-diagonal solver
    all_ar.resize(mnsize);
    all_az.resize(mnsize);
    all_dr.resize(mnsize);
    all_dz.resize(mnsize);
    all_br.resize(mnsize);
    all_bz.resize(mnsize);
    all_cr.resize(mnsize);
    all_cz.resize(mnsize);
    for (int mn = 0; mn < mnsize; ++mn) {
      all_ar[mn].resize(ns);
      all_az[mn].resize(ns);
      all_dr[mn].resize(ns);
      all_dz[mn].resize(ns);
      all_br[mn].resize(ns);
      all_bz[mn].resize(ns);
      all_cr[mn].resize(num_basis_);
      all_cz[mn].resize(num_basis_);
      for (int k = 0; k < num_basis_; ++k) {
        all_cr[mn][k].resize(ns);
        all_cz[mn][k].resize(ns);
      }
    }

    // storage to hand over data between ranks
    handover_aR.resize(mnsize);
    handover_aZ.resize(mnsize);
    handover_cR.resize(num_basis_);
    handover_cZ.resize(num_basis_);
    for (int k = 0; k < num_basis_; ++k) {
      handover_cR[k].resize(mnsize);
      handover_cZ[k].resize(mnsize);
    }
  }

  if (r.nsMinF1 > 0) {
    // has inside
    rmncc_i[r.get_thread_id()].resize(s_.mnsize);
    rmnss_i[r.get_thread_id()].resize(s_.mnsize);
    zmnsc_i[r.get_thread_id()].resize(s_.mnsize);
    zmncs_i[r.get_thread_id()].resize(s_.mnsize);
    lmnsc_i[r.get_thread_id()].resize(s_.mnsize);
    lmncs_i[r.get_thread_id()].resize(s_.mnsize);
  }

  if (r.nsMaxF1 < ns) {
    // has outside
    rmncc_o[r.get_thread_id()].resize(s_.mnsize);
    rmnss_o[r.get_thread_id()].resize(s_.mnsize);
    zmnsc_o[r.get_thread_id()].resize(s_.mnsize);
    zmncs_o[r.get_thread_id()].resize(s_.mnsize);
    lmnsc_o[r.get_thread_id()].resize(s_.mnsize);
    lmncs_o[r.get_thread_id()].resize(s_.mnsize);
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

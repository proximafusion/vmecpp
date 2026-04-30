// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
//
// This header is only compiled when VMECPP_USE_MKL is defined.
// When MKL is not available the ideal_mhd_model falls back to the DFT path
// and this file is not included.
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_

#include <mkl_dfti.h>

#include <Eigen/Dense>

#include "vmecpp/common/flow_control/flow_control.h"
#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/fourier_forces/fourier_forces.h"
#include "vmecpp/vmec/fourier_geometry/fourier_geometry.h"
#include "vmecpp/vmec/ideal_mhd_model/dft_data.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {

// RAII holder for Intel MKL DFTI descriptors used by the toroidal
// (zeta-direction) Fourier transforms.
//
// Descriptors are created once during construction (before any parallel
// execution) and can be safely re-executed concurrently from multiple threads
// by providing thread-private input/output buffers to
// DftiComputeBackward / DftiComputeForward.
//
// Sizing:
//   n     = nZeta  (number of toroidal grid points, transform length)
//   nhalf = n/2+1  (half-spectrum size, CCE packing)
//   nfp   = number of toroidal field periods (factor in derivative spectra)
//
// CCE (complex conjugate-even) half-spectrum layout (2*nhalf doubles):
//   [re0, 0, re1, im1, re2, im2, ..., re(N/2), 0]
// Fill* helpers write to CplxBuf (double[2]) regardless of backend.
class ToroidalFftPlans {
 public:
  // Create synthesis (c2r) and analysis (r2c) descriptors for transforms of
  // length n.
  explicit ToroidalFftPlans(int n, int nfp);
  ~ToroidalFftPlans();

  // Non-copyable, non-movable (descriptors hold raw MKL resources).
  ToroidalFftPlans(const ToroidalFftPlans&) = delete;
  ToroidalFftPlans& operator=(const ToroidalFftPlans&) = delete;
  ToroidalFftPlans(ToroidalFftPlans&&) = delete;
  ToroidalFftPlans& operator=(ToroidalFftPlans&&) = delete;

  // Transform length (= nZeta).
  int n;

  // Half-spectrum size: n/2 + 1.
  int nhalf;

  // Number of field periods.
  int nfp;

  // Synthesis (c2r / backward): CCE input -> real output.
  DFTI_DESCRIPTOR_HANDLE desc_c2r;

  // Analysis (r2c / forward): real input -> CCE output.
  DFTI_DESCRIPTOR_HANDLE desc_r2c;
};

// MKL-accelerated forward transform: Fourier coefficients -> real space.
//
// Drop-in replacement for FourierToReal3DSymmFastPoloidal.
// Replaces the O(nZeta * ntor) inner-n dot-product loop with
// O(nZeta * log(nZeta)) MKL DFTI c2r transforms.
//
// The toroidal basis arrays cosnv/sinnv/cosnvn/sinnvn in
// FourierBasisFastPoloidal incorporate nscale[n] = sqrt(2) for n > 0. The FFT
// counterpart uses the same normalization: X[0] = spec[0] * nscale[0] and
// X[n] = spec[n]*nscale[n]/2 for n >= 1.
void FourierToReal3DSymmFastPoloidalFft(
    const FourierGeometry& physical_x, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, const ToroidalFftPlans& plans,
    RealSpaceGeometry& m_geometry);

// MKL-accelerated inverse transform: real-space forces -> Fourier coefficients.
//
// Drop-in replacement for ForcesToFourier3DSymmFastPoloidal.
// Replaces the O(nZeta * ntor) toroidal scatter loop with
// O(nZeta * log(nZeta)) MKL DFTI r2c transforms.
void ForcesToFourier3DSymmFastPoloidalFft(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb, const ToroidalFftPlans& plans,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_

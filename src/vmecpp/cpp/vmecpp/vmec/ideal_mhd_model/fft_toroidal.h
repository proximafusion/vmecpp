// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_

#include <fftw3.h>

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

// RAII holder for FFTW plans used by the toroidal (zeta-direction) Fourier
// transforms.
//
// Plans are created once during construction (which must occur before any
// parallel execution) and can be safely re-executed concurrently from multiple
// threads by providing thread-private input/output buffers to
// fftw_execute_dft_c2r / fftw_execute_dft_r2c.
//
// Sizing:
//   n     = nZeta  (number of toroidal grid points, transform length)
//   nhalf = n/2+1  (half-spectrum size used by c2r and r2c plans)
//   nfp   = number of toroidal field periods (factor in derivative spectra)
class ToroidalFftPlans {
 public:
  // Create c2r (synthesis) and r2c (analysis) plans for transforms of length n.
  // n is the number of toroidal grid points (Sizes::nZeta).
  // nfp is the number of field periods (Sizes::nfp).
  explicit ToroidalFftPlans(int n, int nfp);
  ~ToroidalFftPlans();

  // Non-copyable, non-movable (plans hold raw FFTW resources).
  ToroidalFftPlans(const ToroidalFftPlans&) = delete;
  ToroidalFftPlans& operator=(const ToroidalFftPlans&) = delete;
  ToroidalFftPlans(ToroidalFftPlans&&) = delete;
  ToroidalFftPlans& operator=(ToroidalFftPlans&&) = delete;

  // Transform length (= nZeta).
  int n;

  // Half-spectrum size: n/2 + 1 (input size for c2r, output size for r2c).
  int nhalf;

  // Number of field periods.
  int nfp;

  // Synthesis plan: half-complex (size nhalf) -> real (size n).
  // Used in FourierToReal (Fourier coefficients -> real-space values).
  fftw_plan plan_c2r;

  // Analysis plan: real (size n) -> half-complex (size nhalf).
  // Used in ForcesToFourier (real-space forces -> Fourier coefficients).
  fftw_plan plan_r2c;
};

// FFT-accelerated forward transform: Fourier coefficients -> real space.
//
// Drop-in replacement for FourierToReal3DSymmFastPoloidal.
// Replaces the O(nZeta * ntor) toroidal dot-product inner loop with
// O(nZeta * log(nZeta)) FFTW c2r IFFTs, giving a ~2-3x speedup.
//
// The toroidal basis arrays cosnv/sinnv/cosnvn/sinnvn in
// FourierBasisFastPoloidal encorporate nscale[n] = sqrt(2) for n > 0. The FFT
// counterpart uses the same normalization: X[0] = spec[0] * nscale[0] and X[n]
// = spec[n]*nscale[n]/2 for n >= 1, so that the c2r output matches the original
// dot-product result.
void FourierToReal3DSymmFastPoloidalFft(
    const FourierGeometry& physical_x, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, const ToroidalFftPlans& plans,
    RealSpaceGeometry& m_geometry);

// FFT-accelerated inverse transform: real-space forces -> Fourier coefficients.
//
// Drop-in replacement for ForcesToFourier3DSymmFastPoloidal.
// Replaces the O(nZeta * ntor) toroidal scatter inner loop with
// O(nZeta * log(nZeta)) FFTW r2c DFTs.
void ForcesToFourier3DSymmFastPoloidalFft(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb, const ToroidalFftPlans& plans,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_

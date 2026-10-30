// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_

// FFT path is optional: gated on VMECPP_USE_FFTX which is set by the build
// system (CMake / Bazel) when FFTX/SPIRAL kernels are available.  Without
// FFTX, this header expands to nothing -- IdealMhdModel falls back to the
// partial-DFT path unconditionally; see ideal_mhd_model.cc for the dispatch.
#ifdef VMECPP_USE_FFTX

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

// SPIRAL/FFTX kernel function pointers for batched toroidal DFTs.
//
// Plans are created once during construction (which must occur before any
// parallel execution). SPIRAL-generated kernels are thread-safe: twiddle
// tables are plain static (init-once, read-only); per-call workspace uses
// thread_local storage.
//
// Sizing:
//   n     = nZeta  (number of toroidal grid points, transform length)
//   nhalf = n/2+1  (half-spectrum size used by c2r and r2c plans)
//   nfp   = number of toroidal field periods (factor in derivative spectra)
class ToroidalFftPlans {
 public:
  // Create c2r (synthesis) and r2c (analysis) FFTX kernels for transforms of
  // length n.
  // n is the number of toroidal grid points (Sizes::nZeta).
  // nfp is the number of field periods (Sizes::nfp).
  // mpol is the number of poloidal modes; used to size the "full" batched
  // kernels that pack 12 transforms per m for all m in [0, mpol) into one
  // call.
  ToroidalFftPlans(int n, int nfp, int mpol);
  ~ToroidalFftPlans();

  // Non-copyable, non-movable (kernel pointers hold FFTX resources).
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

  // Number of poloidal modes (set at construction).
  int mpol;

  // kBatch = 12 covers the 12 quantities transformed per (jF, m) pair:
  //   {R_cc, R_ss, dR_cc, dR_ss, Z_sc, Z_cs, dZ_sc, dZ_cs,
  //    L_sc, L_cs, dL_sc, dL_cs}.
  static constexpr int kBatch = 12;

  // FFTX runtime function pointers for the full-surface batched transforms:
  //   full_count = 12 * mpol transforms in one call.
  // Call signatures: run(double* output, double* input).
  // c2r input layout: full_count contiguous half-spectra of (n/2+1)*2 doubles.
  // c2r output layout: full_count contiguous real signals of n doubles.
  // r2c is the mirror image.
  using FftxRunFn = void (*)(double* output, double* input);
  using FftxLifecycleFn = void (*)();
  FftxRunFn fftx_full_c2r_run = nullptr;
  FftxLifecycleFn fftx_full_c2r_destroy = nullptr;
  FftxRunFn fftx_full_r2c_run = nullptr;
  FftxLifecycleFn fftx_full_r2c_destroy = nullptr;

  // True iff vendored FFTX codelets exist for this (nZeta, 12*mpol) shape.
  // Both forward and inverse must be present; partial coverage falls back to
  // the partial-DFT path so the whole solver stays on one consistent backend.
  bool kernels_available() const {
    return fftx_full_c2r_run != nullptr && fftx_full_r2c_run != nullptr;
  }
};

// FFT-accelerated forward transform: Fourier coefficients -> real space.
//
// Drop-in replacement for FourierToReal3DSymmFastPoloidal.
// Replaces the O(nZeta * ntor) toroidal dot-product inner loop with
// O(nZeta * log(nZeta)) FFTX c2r IFFTs, giving a ~2-3x speedup.
//
// The toroidal basis arrays cosnv/sinnv/cosnvn/sinnvn in
// FourierBasisFastPoloidal encorporate nscale[n] = sqrt(2) for n > 0. The FFT
// counterpart uses the same normalization: X[0] = spec[0] * nscale[0] and X[n]
// = spec[n]*nscale[n]/2 for n >= 1, so that the c2r output matches the
// original dot-product result.
void FourierToReal3DSymmFastPoloidalFft(
    const FourierGeometry& physical_x, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& r, const Sizes& s, const RadialProfiles& rp,
    const FourierBasisFastPoloidal& fb, const ToroidalFftPlans& plans,
    RealSpaceGeometry& m_geometry);

// FFT-accelerated inverse transform: real-space forces -> Fourier coefficients.
//
// Drop-in replacement for ForcesToFourier3DSymmFastPoloidal.
// Replaces the O(nZeta * ntor) toroidal scatter inner loop with
// O(nZeta * log(nZeta)) FFTX r2c DFTs.
void ForcesToFourier3DSymmFastPoloidalFft(
    const RealSpaceForces& d, const Eigen::VectorXd& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb, const ToroidalFftPlans& plans,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces);

}  // namespace vmecpp

#endif  // VMECPP_USE_FFTX

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_FFT_TOROIDAL_H_

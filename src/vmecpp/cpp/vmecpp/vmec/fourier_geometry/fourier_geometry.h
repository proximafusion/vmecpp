// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_FOURIER_GEOMETRY_FOURIER_GEOMETRY_H_
#define VMECPP_VMEC_FOURIER_GEOMETRY_FOURIER_GEOMETRY_H_

#include <span>

#include "vmecpp/common/fourier_basis_fast_poloidal/fourier_basis_fast_poloidal.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/boundaries/boundaries.h"
#include "vmecpp/vmec/fourier_coefficients/fourier_coefficients.h"
#include "vmecpp/vmec/radial_profiles/radial_profiles.h"

namespace vmecpp {

class FourierGeometry : public FourierCoeffs {
 public:
  FourierGeometry(const Sizes* s, const RadialPartitioning* r, int ns);
  FourierGeometry(const FourierGeometry& other);
  FourierGeometry& operator=(const FourierGeometry& other);
  FourierGeometry(FourierGeometry&& other) noexcept;
  FourierGeometry& operator=(FourierGeometry&& other) noexcept;

  void interpFromBoundaryAndAxis(const FourierBasisFastPoloidal& t,
                                 const Boundaries& b, const RadialProfiles& p);

  // Initial guess for a plasma with an inner boundary: every coefficient
  // interpolated between the inner boundary and the boundary b with the
  // weights of InterpolationWeight, which are those of
  // interpFromBoundaryAndAxis at inner_boundary_s = 0, and the m=1 gauge
  // combinations set by setM1GaugeFromBoundaries.
  void interpFromBoundaries(const FourierBasisFastPoloidal& t,
                            const Boundaries& inner, const Boundaries& b,
                            const RadialProfiles& p);

  // Set the m=1 gauge combinations (the zmncs and, for lasym, zmncc slots
  // that FourierCoeffs::m1Constraint couples to rmnss / rmnsc) on every
  // owned surface to their interpFromBoundaryAndAxis value, the boundary
  // gauge scaled by sqrt(s). Leaves all other coefficients untouched.
  void setM1GaugeFromBoundary(const FourierBasisFastPoloidal& t,
                              const Boundaries& b, const RadialProfiles& p);

  // The same for a plasma with an inner boundary: the gauge linear in sqrt(s)
  // between the inner boundary and the boundary b, which is sqrt(s) times the
  // gauge of b when the inner boundary carries sqrt(s) times it.
  void setM1GaugeFromBoundaries(const FourierBasisFastPoloidal& t,
                                const Boundaries& inner, const Boundaries& b,
                                const RadialProfiles& p);

  // Initialize the state of this FourierGeometry with the given Fourier
  // coefficients. If a Boundaries object is specified (defaults to nullptr; in
  // order to avoid a copy when using std::optional), the geometry of the
  // outermost flux surface (at ns-1) is taken from that Boundaries object
  // instead of from the Fourier coefficient matrices, and the inner surfaces
  // are shifted by the difference between the two with the radial weights of
  // interpFromBoundaryAndAxis.
  // This latter use case applies to fixed-boundary hot-restart operation of
  // VMEC++.
  // rmns, zmnc and lmnc_full carry the non-stellarator-symmetric half of the
  // state and are read only when lasym is set; a stellarator-symmetric run
  // leaves them empty.
  // For a plasma with an inner boundary, the geometry of surface 0 is taken
  // from inner, and the surfaces in between are shifted by the differences
  // between inner and surface 0 of the state and between b and the outermost
  // surface of the state, with the weights of the initial guess
  // interpFromBoundaries.
  void InitFromState(const FourierBasisFastPoloidal& fb,
                     const RowMatrixXd& rmnc, const RowMatrixXd& zmns,
                     const RowMatrixXd& lmns_full, const RowMatrixXd& rmns,
                     const RowMatrixXd& zmnc, const RowMatrixXd& lmnc_full,
                     const RadialProfiles& p, const VmecConstants& constants,
                     int sign_of_jacobian, const Boundaries* b = nullptr,
                     const Boundaries* inner = nullptr);

  void extrapolateTowardsAxis();

  // Set R and Z of surface 0 to the inner boundary inner, which carries the
  // m=1 constraint.
  void SetInnerBoundary(const FourierBasisFastPoloidal& t,
                        const Boundaries& inner);

  // On the inner boundary of a plasma without a magnetic axis, lambda is not a
  // free variable: it enters the half-grid field only through its average
  // with the first interior surface, so it is extrapolated linearly from the
  // first two interior surfaces.
  void ExtrapolateLambdaToInnerBoundary();

  // Transpose of extrapolateTowardsAxis: the forward copies surface-1 m=1 (and
  // m=0 lambda) coefficients onto the axis, so the adjoint folds the axis
  // cotangent back into surface 1 and zeroes the axis.
  void extrapolateTowardsAxisTranspose();

  // Compute the spectral width of the R and Z Fourier coefficients
  // and write it into the spectral_width vector in the given RadialProfiles.
  void ComputeSpectralWidth(const FourierBasisFastPoloidal& fourier_basis,
                            RadialProfiles& m_radial_profiles, int p = 4,
                            int q = 1) const;

  // N.B. all raw pointers below are non-owning pointers to data in
  // FourierCoeffs

  // contrib to R ~ cos(m * theta) * cos(n * zeta)
  std::span<double> rmncc;

  // contrib to R ~ sin(m * theta) * sin(n * zeta)
  std::span<double> rmnss;

  // contrib to R ~ sin(m * theta) * cos(n * zeta)
  std::span<double> rmnsc;

  // contrib to R ~ cos(m * theta) * sin(n * zeta)
  std::span<double> rmncs;

  // -----------

  // contrib to Z ~ sin(m * theta) * cos(n * zeta)
  std::span<double> zmnsc;

  // contrib to Z ~ cos(m * theta) * sin(n * zeta)
  std::span<double> zmncs;

  // contrib to Z ~ cos(m * theta) * cos(n * zeta)
  std::span<double> zmncc;

  // contrib to Z ~ sin(m * theta) * sin(n * zeta)
  std::span<double> zmnss;

  // -----------

  // contrib to lambda ~ sin(m * theta) * cos(n * zeta)
  std::span<double> lmnsc;

  // contrib to lambda ~ cos(m * theta) * sin(n * zeta)
  std::span<double> lmncs;

  // contrib to lambda ~ cos(m * theta) * cos(n * zeta)
  std::span<double> lmncc;

  // contrib to lambda ~ sin(m * theta) * sin(n * zeta)
  std::span<double> lmnss;

 private:
  void BindSpans();

  // Weight of the boundary against the inner boundary of mode m on surface jF
  // of a plasma without a magnetic axis: the weight of the boundary in
  // interpFromBoundaryAndAxis, s for m = 0 and s^(m/2) for m > 0, rescaled to
  // run from 0 on the inner boundary to 1 on the boundary.
  double InterpolationWeight(int jF, int m, const RadialProfiles& p) const;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_GEOMETRY_FOURIER_GEOMETRY_H_

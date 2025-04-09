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
  FourierGeometry(const Sizes *s, const RadialPartitioning *r, int ns);

  void interpFromBoundaryAndAxis(const FourierBasisFastPoloidal &t,
                                 const Boundaries &b, const RadialProfiles &p);

  // Initialize the state of this FourierGeometry with the given Fourier
  // coefficients. If a Boundaries object is specified (defaults to nullptr; in
  // order to avoid a copy when using std::optional), the geometry of the
  // outermost flux surface (at ns-1) is taken from that Boundaries object
  // instead of from the Fourier coefficient matrices.
  // This latter use case applies to fixed-boundary hot-restart operation of
  // VMEC++.
  void InitFromState(const FourierBasisFastPoloidal &fb,
                     const RowMatrixXd &rmnc, const RowMatrixXd &zmns,
                     const RowMatrixXd &lmns_full, const RadialProfiles &p,
                     const VmecConstants &constants,
                     const Boundaries *b = nullptr);

  void extrapolateTowardsAxis();

  void copyFrom(const FourierGeometry &src);

  // Compute the spectral width of the R and Z Fourier coefficients
  // and write it into the spectral_width vector in the given RadialProfiles.
  void ComputeSpectralWidth(const FourierBasisFastPoloidal &fourier_basis,
                            RadialProfiles &m_radial_profiles, int p = 4,
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
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_GEOMETRY_FOURIER_GEOMETRY_H_

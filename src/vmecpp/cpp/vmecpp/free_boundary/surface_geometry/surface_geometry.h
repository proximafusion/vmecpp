// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_SURFACE_GEOMETRY_H_
#define VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_SURFACE_GEOMETRY_H_

#include <span>
#include <vector>

#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

class SurfaceGeometry {
 public:
  SurfaceGeometry(const Sizes* s, const FourierBasisFastToroidal* fb,
                  const TangentialPartitioning* tp);

  void update(
      const std::span<const real_t> rCC, const std::span<const real_t> rSS,
      const std::span<const real_t> rSC, const std::span<const real_t> rCS,
      const std::span<const real_t> zSC, const std::span<const real_t> zCS,
      const std::span<const real_t> zCC, const std::span<const real_t> zSS,
      int signOfJacobian, bool fullUpdate);

  // [nfp] cos(2 pi / nfp * p)
  std::vector<real_t> cos_per;

  // [nfp] sin(2 pi / nfp * p)
  std::vector<real_t> sin_per;

  // [nZeta] cos(phi)
  std::vector<real_t> cos_phi;

  // [nZeta] sin(phi)
  std::vector<real_t> sin_phi;

  // -----------------

  // R
  // full surface
  std::vector<real_t> r1b;

  // dR/dTheta
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> rub;

  // dR/dPhi
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> rvb;

  // Z
  // full surface
  std::vector<real_t> z1b;

  // dZ/dTheta
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> zub;

  // dZ/dPhi
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> zvb;

  // d^2R/dTheta^2
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  std::vector<real_t> ruu;

  // d^2R/(dTheta dPhi)
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  std::vector<real_t> ruv;

  // d^2R/dPhi^2
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  std::vector<real_t> rvv;

  // d^2Z/dTheta^2
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  std::vector<real_t> zuu;

  // d^2Z/(dTheta dPhi)
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  std::vector<real_t> zuv;

  // d^2Z/dPhi^2
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  std::vector<real_t> zvv;

  // N^r * signOfJacobian
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> snr;

  // N^phi * signOfJacobian
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> snv;

  // N^z * signOfJacobian
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> snz;

  // g_{theta,theta}
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> guu;

  // 2 * g_{theta,zeta} = 2/nfp * g_{theta,phi}
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> guv;

  // g_{zeta,zeta} = 1/(nfp*nfp) g_{phi,phi}
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> gvv;

  // 1/2 d^2X/dTheta^2 dot N
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> auu;

  // d^2X/(dTheta dZeta) dot N
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> auv;

  // 1/2 d^2X/dZeta^2 dot N
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> avv;

  // - (R N^R + Z N^Z)
  // needed for dsave --> (X - X') dot N'
  // thread-local effective poloidal range (tp->numZT)
  std::vector<real_t> drv;

  // R^2 + Z^2
  // needed for gsave --> |X - X'|^2
  // full surface
  std::vector<real_t> rzb2;

  // x
  // full surface
  std::vector<real_t> rcosuv;

  // y
  // full surface
  std::vector<real_t> rsinuv;

 private:
  const Sizes& s_;
  const FourierBasisFastToroidal& fb_;
  const TangentialPartitioning& tp_;

  void computeConstants();

  void inverseDFT(const std::span<const real_t> rCC,
                  const std::span<const real_t> rSS,
                  const std::span<const real_t> rSC,
                  const std::span<const real_t> rCS,
                  const std::span<const real_t> zSC,
                  const std::span<const real_t> zCS,
                  const std::span<const real_t> zCC,
                  const std::span<const real_t> zSS, bool fullUpdate);

  void derivedSurfaceQuantities(int signOfJacobian, bool fullUpdate);

  std::vector<real_t> r1b_asym;
  std::vector<real_t> z1b_asym;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_SURFACE_GEOMETRY_H_

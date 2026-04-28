// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_SURFACE_GEOMETRY_H_
#define VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_SURFACE_GEOMETRY_H_

#include <Eigen/Dense>
#include <span>

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
      const std::span<const double> rCC, const std::span<const double> rSS,
      const std::span<const double> rSC, const std::span<const double> rCS,
      const std::span<const double> zSC, const std::span<const double> zCS,
      const std::span<const double> zCC, const std::span<const double> zSS,
      int signOfJacobian, bool fullUpdate);

  // [nfp] cos(2 pi / nfp * p)
  Eigen::VectorXd cos_per;

  // [nfp] sin(2 pi / nfp * p)
  Eigen::VectorXd sin_per;

  // [nZeta] cos(phi)
  Eigen::VectorXd cos_phi;

  // [nZeta] sin(phi)
  Eigen::VectorXd sin_phi;

  // -----------------

  // R
  // full surface
  Eigen::VectorXd r1b;

  // dR/dTheta
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd rub;

  // dR/dPhi
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd rvb;

  // Z
  // full surface
  Eigen::VectorXd z1b;

  // dZ/dTheta
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd zub;

  // dZ/dPhi
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd zvb;

  // d^2R/dTheta^2
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  Eigen::VectorXd ruu;

  // d^2R/(dTheta dPhi)
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  Eigen::VectorXd ruv;

  // d^2R/dPhi^2
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  Eigen::VectorXd rvv;

  // d^2Z/dTheta^2
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  Eigen::VectorXd zuu;

  // d^2Z/(dTheta dPhi)
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  Eigen::VectorXd zuv;

  // d^2Z/dPhi^2
  // thread-local effective poloidal range (tp->numZT)
  // only needed within SurfaceGeometry() but public for testing
  Eigen::VectorXd zvv;

  // N^r * signOfJacobian
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd snr;

  // N^phi * signOfJacobian
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd snv;

  // N^z * signOfJacobian
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd snz;

  // g_{theta,theta}
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd guu;

  // 2 * g_{theta,zeta} = 2/nfp * g_{theta,phi}
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd guv;

  // g_{zeta,zeta} = 1/(nfp*nfp) g_{phi,phi}
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd gvv;

  // 1/2 d^2X/dTheta^2 dot N
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd auu;

  // d^2X/(dTheta dZeta) dot N
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd auv;

  // 1/2 d^2X/dZeta^2 dot N
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd avv;

  // - (R N^R + Z N^Z)
  // needed for dsave --> (X - X') dot N'
  // thread-local effective poloidal range (tp->numZT)
  Eigen::VectorXd drv;

  // R^2 + Z^2
  // needed for gsave --> |X - X'|^2
  // full surface
  Eigen::VectorXd rzb2;

  // x
  // full surface
  Eigen::VectorXd rcosuv;

  // y
  // full surface
  Eigen::VectorXd rsinuv;

 private:
  const Sizes& s_;
  const FourierBasisFastToroidal& fb_;
  const TangentialPartitioning& tp_;

  void computeConstants();

  void inverseDFT(const std::span<const double> rCC,
                  const std::span<const double> rSS,
                  const std::span<const double> rSC,
                  const std::span<const double> rCS,
                  const std::span<const double> zSC,
                  const std::span<const double> zCS,
                  const std::span<const double> zCC,
                  const std::span<const double> zSS, bool fullUpdate);

  void derivedSurfaceQuantities(int signOfJacobian, bool fullUpdate);

  Eigen::VectorXd r1b_asym;
  Eigen::VectorXd z1b_asym;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_SURFACE_GEOMETRY_SURFACE_GEOMETRY_H_

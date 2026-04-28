// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_
#define VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_

#include <Eigen/Dense>

#include "vmecpp/common/fourier_basis_fast_toroidal/fourier_basis_fast_toroidal.h"
#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"
#include "vmecpp/free_boundary/tangential_partitioning/tangential_partitioning.h"

namespace vmecpp {

class SingularIntegrals {
 public:
  SingularIntegrals(const Sizes* s, const FourierBasisFastToroidal* fb,
                    const TangentialPartitioning* tp, const SurfaceGeometry* sg,
                    int nf, int mf);

  void update(const Eigen::VectorXd& bDotN, bool fullUpdate);

  int numSC;
  int numCS;
  int nzLen;  // non-zero length

  Eigen::VectorXd cmn;
  Eigen::VectorXd cmns;

  Eigen::VectorXd ap;
  Eigen::VectorXd am;
  Eigen::VectorXd d;
  Eigen::VectorXd sqrtc2;
  Eigen::VectorXd sqrta2;
  Eigen::VectorXd delta4;

  Eigen::VectorXd Ap;
  Eigen::VectorXd Am;
  Eigen::VectorXd D;

  Eigen::VectorXd R1p;
  Eigen::VectorXd R1m;
  Eigen::VectorXd R0p;
  Eigen::VectorXd R0m;
  Eigen::VectorXd Ra1p;
  Eigen::VectorXd Ra1m;

  // l-2
  Eigen::VectorXd Tl2p;
  // l-2
  Eigen::VectorXd Tl2m;
  // l-1
  Eigen::VectorXd Tl1p;
  // l-1
  Eigen::VectorXd Tl1m;
  // l
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Tlp;
  // l
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Tlm;

  // l
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Slp;
  // l
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Slm;

  // sum_kl { Tlm * sin(mu + nv), Tlp * sin(mu - nv) }
  Eigen::VectorXd bvec_sin;

  // sum_kl { Tlm * cos(mu + nv), Tlp * cos(mu - nv) }
  Eigen::VectorXd bvec_cos;

  // Slm * sin(mu + nv), Slp * sin(mu - nv)
  Eigen::VectorXd grpmn_sin;

  // Slm * cos(mu + nv), Slp * cos(mu - nv)
  Eigen::VectorXd grpmn_cos;

  void prepareUpdate(const Eigen::VectorXd& a, const Eigen::VectorXd& b2,
                     const Eigen::VectorXd& c, const Eigen::VectorXd& A,
                     const Eigen::VectorXd& B2, const Eigen::VectorXd& C,
                     bool fullUpdate);

 private:
  const Sizes& s_;
  const FourierBasisFastToroidal& fb_;
  const TangentialPartitioning& tp_;
  const SurfaceGeometry& sg_;

  void computeCoefficients();

  void performUpdate(const Eigen::VectorXd& bDotN, bool fullUpdate);

  int nf;
  int mf;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_

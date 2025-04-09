// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_
#define VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_

#include <vector>

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

  void update(const std::vector<double>& bDotN, bool fullUpdate);

  int numSC;
  int numCS;
  int nzLen;  // non-zero length

  std::vector<double> cmn;
  std::vector<double> cmns;

  std::vector<double> ap;
  std::vector<double> am;
  std::vector<double> d;
  std::vector<double> sqrtc2;
  std::vector<double> sqrta2;
  std::vector<double> delta4;

  std::vector<double> Ap;
  std::vector<double> Am;
  std::vector<double> D;

  std::vector<double> R1p;
  std::vector<double> R1m;
  std::vector<double> R0p;
  std::vector<double> R0m;
  std::vector<double> Ra1p;
  std::vector<double> Ra1m;

  // l-2
  std::vector<double> Tl2p;
  // l-2
  std::vector<double> Tl2m;
  // l-1
  std::vector<double> Tl1p;
  // l-1
  std::vector<double> Tl1m;
  // l
  std::vector<std::vector<double> > Tlp;
  // l
  std::vector<std::vector<double> > Tlm;

  // l
  std::vector<std::vector<double> > Slp;
  // l
  std::vector<std::vector<double> > Slm;

  // sum_kl { Tlm * sin(mu + nv), Tlp * sin(mu - nv) }
  std::vector<double> bvec_sin;

  // sum_kl { Tlm * cos(mu + nv), Tlp * cos(mu - nv) }
  std::vector<double> bvec_cos;

  // Slm * sin(mu + nv), Slp * sin(mu - nv)
  std::vector<double> grpmn_sin;

  // Slm * cos(mu + nv), Slp * cos(mu - nv)
  std::vector<double> grpmn_cos;

 private:
  const Sizes& s_;
  const FourierBasisFastToroidal& fb_;
  const TangentialPartitioning& tp_;
  const SurfaceGeometry& sg_;

  void computeCoefficients();

  void prepareUpdate(const std::vector<double>& a,
                     const std::vector<double>& b2,
                     const std::vector<double>& c, const std::vector<double>& A,
                     const std::vector<double>& B2,
                     const std::vector<double>& C, bool fullUpdate);
  void performUpdate(const std::vector<double>& bDotN, bool fullUpdate);

  int nf;
  int mf;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_

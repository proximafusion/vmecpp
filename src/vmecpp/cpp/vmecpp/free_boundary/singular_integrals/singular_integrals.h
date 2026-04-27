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

  void update(const std::vector<real_t>& bDotN, bool fullUpdate);

  int numSC;
  int numCS;
  int nzLen;  // non-zero length

  std::vector<real_t> cmn;
  std::vector<real_t> cmns;

  std::vector<real_t> ap;
  std::vector<real_t> am;
  std::vector<real_t> d;
  std::vector<real_t> sqrtc2;
  std::vector<real_t> sqrta2;
  std::vector<real_t> delta4;

  std::vector<real_t> Ap;
  std::vector<real_t> Am;
  std::vector<real_t> D;

  std::vector<real_t> R1p;
  std::vector<real_t> R1m;
  std::vector<real_t> R0p;
  std::vector<real_t> R0m;
  std::vector<real_t> Ra1p;
  std::vector<real_t> Ra1m;

  // l-2
  std::vector<real_t> Tl2p;
  // l-2
  std::vector<real_t> Tl2m;
  // l-1
  std::vector<real_t> Tl1p;
  // l-1
  std::vector<real_t> Tl1m;
  // l
  std::vector<std::vector<real_t> > Tlp;
  // l
  std::vector<std::vector<real_t> > Tlm;

  // l
  std::vector<std::vector<real_t> > Slp;
  // l
  std::vector<std::vector<real_t> > Slm;

  // sum_kl { Tlm * sin(mu + nv), Tlp * sin(mu - nv) }
  std::vector<real_t> bvec_sin;

  // sum_kl { Tlm * cos(mu + nv), Tlp * cos(mu - nv) }
  std::vector<real_t> bvec_cos;

  // Slm * sin(mu + nv), Slp * sin(mu - nv)
  std::vector<real_t> grpmn_sin;

  // Slm * cos(mu + nv), Slp * cos(mu - nv)
  std::vector<real_t> grpmn_cos;

  void prepareUpdate(const std::vector<real_t>& a,
                     const std::vector<real_t>& b2,
                     const std::vector<real_t>& c, const std::vector<real_t>& A,
                     const std::vector<real_t>& B2,
                     const std::vector<real_t>& C, bool fullUpdate);

 private:
  const Sizes& s_;
  const FourierBasisFastToroidal& fb_;
  const TangentialPartitioning& tp_;
  const SurfaceGeometry& sg_;

  void computeCoefficients();

  void performUpdate(const std::vector<real_t>& bDotN, bool fullUpdate);

  int nf;
  int mf;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_SINGULAR_INTEGRALS_SINGULAR_INTEGRALS_H_

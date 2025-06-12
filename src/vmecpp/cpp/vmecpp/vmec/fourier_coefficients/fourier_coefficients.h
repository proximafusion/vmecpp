// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_FOURIER_COEFFICIENTS_FOURIER_COEFFICIENTS_H_
#define VMECPP_VMEC_FOURIER_COEFFICIENTS_FOURIER_COEFFICIENTS_H_

#include <cstdio>
#include <optional>
#include <vector>

#include "vmecpp/common/sizes/sizes.h"
#include "vmecpp/common/util/util.h"
#include "vmecpp/vmec/radial_partitioning/radial_partitioning.h"

namespace vmecpp {

class FourierCoeffs {
 public:
  FourierCoeffs(const Sizes* s, const RadialPartitioning* r, int nsMin,
                int nsMax, int ns);

  void setZero();

  void decomposeInto(FourierCoeffs& m_x,
                     const std::vector<double>& scalxc) const;
  void m1Constraint(double scalingFactor,
                    std::optional<int> jMax = std::nullopt);

  // Get the sum of squared coefficients for R and Z.
  // If includeOffset is false, the (0,0)-coefficients for cos(mu)*cos(nv) are
  // left out. The range of flux surface to count in is specified as [nsMinHere,
  // nsMaxHere[ in order to allow to not count stuff twice at the borders in
  // different threads.
  double rzNorm(bool includeOffset, int nsMinHere, int nsMaxHere) const;

  double GetXcElement(int rzl, int basis_index, int j, int n, int m) const;

  int nsMin() const;
  int nsMax() const;

 protected:
  const Sizes& s_;
  const RadialPartitioning& r_;

  const int nsMin_;
  const int nsMax_;
  const int ns;

  // [ns x numFC] R ~ cos(m*theta)*cos(n*zeta)
  std::vector<double> rcc;

  // [ns x numFC] R ~ sin(m*theta)*sin(n*zeta)
  std::vector<double> rss;

  // [ns x numFC] R ~ sin(m*theta)*cos(n*zeta)
  std::vector<double> rsc;

  // [ns x numFC] R ~ cos(m*theta)*sin(n*zeta)
  std::vector<double> rcs;

  //***************/

  // [ns x numFC] Z ~ sin(m*theta)*cos(n*zeta)
  std::vector<double> zsc;

  // [ns x numFC] Z ~ cos(m*theta)*sin(n*zeta)
  std::vector<double> zcs;

  // [ns x numFC] Z ~ cos(m*theta)*cos(n*zeta)
  std::vector<double> zcc;

  // [ns x numFC] Z ~ sin(m*theta)*sin(n*zeta)
  std::vector<double> zss;

  //***************/

  // [ns x numFC] lambda ~ sin(m*theta)*cos(n*zeta)
  std::vector<double> lsc;

  // [ns x numFC] lambda ~ cos(m*theta)*sin(n*zeta)
  std::vector<double> lcs;

  // [ns x numFC] lambda ~ cos(m*theta)*cos(n*zeta)
  std::vector<double> lcc;

  // [ns x numFC] lambda ~ sin(m*theta)*sin(n*zeta)
  std::vector<double> lss;
};

}  // namespace vmecpp

#endif  // VMECPP_VMEC_FOURIER_COEFFICIENTS_FOURIER_COEFFICIENTS_H_

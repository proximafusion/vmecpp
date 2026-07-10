// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_FREE_BOUNDARY_AEGIS_AEGIS_H_
#define VMECPP_FREE_BOUNDARY_AEGIS_AEGIS_H_

#include <Eigen/Dense>
#include <vector>

namespace vmecpp {

// AEGIS (Accurate Exterior Green's Integral Solver): a virtual-casing
// exterior-magnetic-field evaluator, a higher-order alternative to NESTOR's
// scalar-potential vacuum field.
//
// Given a discretized surface (typically the LCFS) with the equivalent surface
// current of the plasma field,
//   K     = n x B_plasma   (surface_current)
//   sigma = n . B_plasma   (normal_field, the monopole term)
// and area weights dA, VirtualCasing evaluates the plasma's exterior field
//   B_plasma_ext(r) = (1/4pi) sum_i [ K_i x (r - X_i) + sigma_i (r - X_i) ]
//                                    / |r - X_i|^3 * dA_i .
// The total exterior vacuum field on the boundary is B_coil + B_plasma_ext, and
// |B_exterior|^2 / 2 is the free-boundary vacuum pressure that couples to VMEC.
//
// Unlike NESTOR, which LU-solves a dense boundary-integral system for a scalar
// potential every ivacskip iterations, AEGIS evaluates the Green's integral
// directly from the known surface current, with no linear solve. The
// near-singular on-surface limit is taken by Quadrature By Expansion (QBX):
// the field is evaluated at expansion centers a few source-spacings off the
// surface, where the integral is smooth, then extrapolated to the surface.
//
// This is the C++ core validated against the Python reference in
// examples/aegis_virtual_casing.py (jump condition to <1%, magnetized-sphere
// self-test to <0.1%). See issue #628.
class VirtualCasing {
 public:
  // surface_points X_i, surface_current K_i (n x B_plasma), normal_field
  // sigma_i (n . B_plasma), and area weights dA_i, one entry per surface point.
  VirtualCasing(std::vector<Eigen::Vector3d> surface_points,
                std::vector<Eigen::Vector3d> surface_current,
                std::vector<double> normal_field, std::vector<double> area);

  // Non-singular Biot-Savart surface integral, valid away from the surface.
  Eigen::Vector3d Raw(const Eigen::Vector3d& r) const;

  // Exterior on-surface limit at r0 with outward unit normal, via QBX. The
  // expansion centers are placed at multiples of expansion_scale (a length of
  // order the source-point spacing) along the normal. Pass a negated normal to
  // obtain the interior limit.
  Eigen::Vector3d OnSurface(const Eigen::Vector3d& r0,
                            const Eigen::Vector3d& outward_normal,
                            double expansion_scale) const;

  int num_points() const { return static_cast<int>(x_.size()); }

 private:
  std::vector<Eigen::Vector3d> x_;  // surface points
  std::vector<Eigen::Vector3d> k_;  // surface current n x B_plasma
  std::vector<double> sigma_;       // n . B_plasma
  std::vector<double> dA_;          // area weights
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_AEGIS_AEGIS_H_

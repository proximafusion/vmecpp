#ifndef VMECPP_FREE_BOUNDARY_VAC2_BASIS_INTEGRALS_H_
#define VMECPP_FREE_BOUNDARY_VAC2_BASIS_INTEGRALS_H_

#include <span>
#include <vector>

#include "absl/status/statusor.h"

namespace vmecpp {

// Output of BasisIntegrals::Compute. Tp/Tm hold the basis integrals
// T^+_l(a,b,c) / T^-_l(a,b,c) at every collocation point (a_i, b_i, c_i),
// for l = 0, ..., lmax. Layout: Tp[l * ndim + i] = T^+_l at point i.
//
// Interpretation (b >= 0 at point i):
//
//   T^+_0(a, b, c) = integral_0^1 dt / sqrt(a*(1-t)^2 + 2*b*t*(1-t) + c*t^2)
//
// For b < 0 the algorithm computes with |b| and swaps T^+ <-> T^- at the end
// so that T^+ and T^- are labelled consistently with the t-substitution
// convention (see vac2/rinteg.f90 lines 95-103).
struct BasisIntegralsOutput {
  std::vector<double> Tp;  // [(lmax+1) * ndim]
  std::vector<double> Tm;
  int ndim = 0;
  int lmax = 0;
};

// Shared T^+_l / T^-_l basis-integral engine. This is the numerical core that
// the vac1 and vac2 mode-assembly modules both depend on. Ported from
// vac2/rinteg.f90 including the two-leg (forward + stabilised backward)
// recurrence structure and the sign-of-b swap that handles both branches of
// the t-substitution without losing digits to catastrophic cancellation at
// large l. See the plan file for the motivation.
class BasisIntegrals {
 public:
  BasisIntegrals() = default;

  // Compute the full batched T^+_l / T^-_l table for l = 0, ..., lmax at every
  // collocation point. a, b, c must be of equal length (the "ndim"); lmax >= 0.
  // a > 0, c > 0 and a*c > b^2 at every point; otherwise an InvalidArgument
  // status is returned.
  absl::StatusOr<BasisIntegralsOutput> Compute(std::span<const double> a,
                                               std::span<const double> b,
                                               std::span<const double> c,
                                               int lmax) const;
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_VAC2_BASIS_INTEGRALS_H_

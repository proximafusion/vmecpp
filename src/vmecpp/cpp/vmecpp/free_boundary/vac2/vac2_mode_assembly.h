#ifndef VMECPP_FREE_BOUNDARY_VAC2_VAC2_MODE_ASSEMBLY_H_
#define VMECPP_FREE_BOUNDARY_VAC2_VAC2_MODE_ASSEMBLY_H_

#include <span>
#include <vector>

#include "absl/status/statusor.h"
#include "vmecpp/free_boundary/vac2/basis_integrals.h"

namespace vmecpp {

// Vac2-style mode-space assembly: given the metric coefficients (a, b, c) and
// the "perturbation" metric (az, bz, cz) at ndim collocation points, produce
// the four-kernel mode-space matrix I_{m, n} = fk(i, m, n) in the convention
// of vac2/analin.f90.
//
// Output layout:
//   fk[(m * (2*mnf + 1) + (n + mnf)) * ndim + i]
//   with m in [0, mnf], n in [-mnf, mnf], i in [0, ndim).
//
// Algorithm (port of vac2/analin.f90):
//   1) Precompute the three-index binomial table aco(l, m, n) once per mnf.
//   2) Call BasisIntegrals::Compute(a, b, c, lmax=2*mnf+2) to get T^{+-}_l.
//   3) Apply the (az, bz, cz) differential combination
//         rp_l <- azm*rp_l + dz*rp_{l+1} + azp*rp_{l+2}
//      (and analogous for rm with azp<->azm), where azp = az+2bz+cz,
//      dz = 2(cz-az), azm = az-2bz+cz.
//   4) Reduce into cp(i, m, n), cm(i, m, n) for m in [0, mnf], n in [0, m]
//      via cp(i,m,n) = sum_l aco(l,m,n) * rp(m-n+2l, i).
//   5) Repeat steps (2)-(4) with metric arguments swapped (c, b, a) and the
//      dz sign flipped, reducing into cp(i, n, m) for m > n (the upper
//      triangle).
//   6) Assemble fk from averaged cp/cm entries.
//
// This is the vac2 half of the user's architectural requirement that both the
// vac1 and vac2 paths share the same stable T_l^{+-} engine (BasisIntegrals).
//
// NOTE: the large-|m,n| asymptotic closed form at vac2/analin.f90 lines
// 108-139 is not yet included; it affects only m > 4 AND n > 4 AND
// m^2 + n^2 > 144 and is marked TODO below. Covered by a follow-up test layer.
class Vac2ModeAssembly {
 public:
  explicit Vac2ModeAssembly(int mnf);

  struct Output {
    // Layout: [(m * (2*mnf+1) + (n+mnf)) * ndim + i]
    std::vector<double> fk;
    int mnf = 0;
    int ndim = 0;
  };

  // Compute fk for the given metric and perturbation-metric arrays (all of
  // length ndim).
  absl::StatusOr<Output> Compute(std::span<const double> a,
                                 std::span<const double> b,
                                 std::span<const double> c,
                                 std::span<const double> az,
                                 std::span<const double> bz,
                                 std::span<const double> cz) const;

  // aco(l, m, n) accessor -- exposed for Layer 1 tests that assert hand-
  // computed binomial identities.
  double aco(int l, int m, int n) const;

  int mnf() const { return mnf_; }

 private:
  int mnf_;
  // Row-major [l][m][n] layout of size (mnf+1)^3.
  std::vector<double> aco_;
  BasisIntegrals basis_integrals_;

  void BuildAcoTable();
};

}  // namespace vmecpp

#endif  // VMECPP_FREE_BOUNDARY_VAC2_VAC2_MODE_ASSEMBLY_H_

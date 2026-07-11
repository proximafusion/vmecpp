// Port of vac2/rinteg.f90. The Fortran source is the ground-truth reference.
// Indexing: Fortran tp(i, l) with column-major storage is at linear offset
// i + l * ndim, which matches Tp[l * ndim + i] in C++.

#include "vmecpp/free_boundary/vac2/basis_integrals.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace vmecpp {

namespace {

constexpr int kBufferLen = 30;  // "lasym" in the Fortran; see rinteg.f90:13

inline std::size_t idx(int l, int i, int ndim) {
  return static_cast<std::size_t>(l) * static_cast<std::size_t>(ndim) +
         static_cast<std::size_t>(i);
}

}  // namespace

absl::StatusOr<BasisIntegralsOutput> BasisIntegrals::Compute(
    std::span<const double> a, std::span<const double> b,
    std::span<const double> c, int lmax) const {
  if (a.size() != b.size() || b.size() != c.size()) {
    return absl::InvalidArgumentError("a, b, c must have equal length");
  }
  if (lmax < 0) {
    return absl::InvalidArgumentError("lmax must be >= 0");
  }
  const int ndim = static_cast<int>(a.size());
  if (ndim == 0) {
    return absl::InvalidArgumentError("empty input");
  }
  for (int i = 0; i < ndim; ++i) {
    if (!(a[i] > 0.0) || !(c[i] > 0.0)) {
      return absl::InvalidArgumentError(
          absl::StrCat("a, c must be positive (i=", i, ")"));
    }
    if (!(a[i] * c[i] - b[i] * b[i] > 0.0)) {
      return absl::InvalidArgumentError(
          absl::StrCat("a*c - b^2 must be positive (i=", i, ")"));
    }
  }

  const int nup = lmax + 3 + kBufferLen;
  const int l_alloc = lmax + 4;  // indices 0..lmax+3

  // Working buffers.
  std::vector<double> d(ndim), sa(ndim), sc(ndim), acb(ndim);
  std::vector<double> rpc(ndim), rmc(ndim), rpa(ndim), rma(ndim);
  std::vector<double> rpz(ndim), rmz(ndim), ca(ndim);
  std::vector<double> ap(ndim), sqa(ndim), sqc(ndim), r1(ndim), r0(ndim);

  for (int i = 0; i < ndim; ++i) {
    d[i] = std::abs(b[i]);
    sa[i] = std::sqrt(a[i]);
    sc[i] = std::sqrt(c[i]);
    acb[i] = 0.25 / (a[i] * c[i] - b[i] * b[i]);
    rpc[i] = (c[i] + d[i]) * acb[i] / sc[i];
    rmc[i] = (c[i] - d[i]) * acb[i] / sc[i];
    rpa[i] = (a[i] + d[i]) * acb[i] / sa[i];
    rma[i] = (a[i] - d[i]) * acb[i] / sa[i];
    const double rpz_u = a[i] + 2.0 * d[i] + c[i];
    const double rmz_u = a[i] - 2.0 * d[i] + c[i];
    ca[i] = (c[i] - a[i]) * acb[i];
    ap[i] = 1.0 / rpz_u;
    sqa[i] = 2.0 * sa[i] * ap[i];
    sqc[i] = 2.0 * sc[i] * ap[i];
    r1[i] = (c[i] - a[i]) * ap[i];
    r0[i] = rmz_u * ap[i];
    rpz[i] = rpz_u * acb[i];
    rmz[i] = rmz_u * acb[i];
  }

  // Allocate tp, tm with space for indices 0..lmax+3.
  BasisIntegralsOutput out;
  out.ndim = ndim;
  out.lmax = lmax;
  // Intermediate V^+, V^- need size (lmax+4). We compute in-place and then
  // shrink at the end.
  std::vector<double> Tp(static_cast<std::size_t>(l_alloc) * ndim, 0.0);
  std::vector<double> Tm(static_cast<std::size_t>(l_alloc) * ndim, 0.0);

  // ---- Base case -----------------------------------------------------------
  for (int i = 0; i < ndim; ++i) {
    const double sqrt_ap = std::sqrt(ap[i]);
    const double num = sc[i] + (c[i] + d[i]) * sqrt_ap;
    const double den = sa[i] - (a[i] + d[i]) * sqrt_ap;
    Tp[idx(0, i, ndim)] = sqrt_ap * std::log(num / den);
    Tp[idx(1, i, ndim)] = -r1[i] * Tp[idx(0, i, ndim)] + sqc[i] - sqa[i];
  }

  // ---- Forward recurrence for V^+, n = 0..lmax+1 writes Tp[2..lmax+3] ------
  for (int n = 0; n <= lmax + 1; ++n) {
    const double fl2 = 1.0 / static_cast<double>(n + 2);
    const double sign_n = (n % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < ndim; ++i) {
      Tp[idx(n + 2, i, ndim)] =
          (sqc[i] + sign_n * sqa[i] -
           (2.0 * n + 3.0) * r1[i] * Tp[idx(n + 1, i, ndim)] -
           (n + 1.0) * r0[i] * Tp[idx(n, i, ndim)]) *
          fl2;
    }
  }

  // ---- Buffer extension (3-slot sliding window) ----------------------------
  const int k = lmax + 1;
  for (int n = lmax + 2; n <= lmax + 1 + kBufferLen; ++n) {
    const double fl2 = 1.0 / static_cast<double>(n + 2);
    const double sign_n = (n % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < ndim; ++i) {
      Tp[idx(k, i, ndim)] = Tp[idx(k + 1, i, ndim)];
      Tp[idx(k + 1, i, ndim)] = Tp[idx(k + 2, i, ndim)];
      Tp[idx(k + 2, i, ndim)] =
          (sqc[i] + sign_n * sqa[i] -
           (2.0 * n + 3.0) * r1[i] * Tp[idx(k + 1, i, ndim)] -
           (n + 1.0) * r0[i] * Tp[idx(k, i, ndim)]) *
          fl2;
    }
  }

  // ---- Initialize V^- at the top of the buffer ----------------------------
  const int num_ = nup - 1;
  const double sign_nup = (nup % 2 == 0) ? 1.0 : -1.0;
  const double sign_num = (num_ % 2 == 0) ? 1.0 : -1.0;
  for (int i = 0; i < ndim; ++i) {
    const double inv_sc3 = 0.5 / sc[i];
    const double inv_sa3 = 0.5 / sa[i];
    const double cube_c = inv_sc3 * inv_sc3 * inv_sc3;
    const double cube_a = inv_sa3 * inv_sa3 * inv_sa3;
    Tm[idx(k + 2, i, ndim)] =
        nup * (Tp[idx(k + 2, i, ndim)] - 4.0 * std::abs(b[i]) /
                                             ((nup + 1.0) * (nup + 3.0)) *
                                             (cube_c + sign_nup * cube_a));
    Tm[idx(k + 1, i, ndim)] =
        num_ * (Tp[idx(k + 1, i, ndim)] - 4.0 * std::abs(b[i]) /
                                              ((num_ + 1.0) * (num_ + 3.0)) *
                                              (cube_c + sign_num * cube_a));
  }

  // ---- Backward recurrence for V^- through the buffer ---------------------
  for (int n = lmax + 1 + kBufferLen; n >= lmax + 2; --n) {
    const double fl1 = 1.0 / static_cast<double>(n + 1);
    const double fln = n * fl1;
    const double sign_n = (n % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < ndim; ++i) {
      Tm[idx(k, i, ndim)] =
          (sqc[i] + sign_n * sqa[i] -
           (2.0 * n + 3.0) * fl1 * r1[i] * Tm[idx(k + 1, i, ndim)] -
           r0[i] * Tm[idx(k + 2, i, ndim)]) *
          fln;
      Tm[idx(k + 2, i, ndim)] = Tm[idx(k + 1, i, ndim)];
      Tm[idx(k + 1, i, ndim)] = Tm[idx(k, i, ndim)];
    }
  }

  // ---- Backward recurrence for V^- down to l = 0 --------------------------
  for (int n = lmax + 1; n >= 0; --n) {
    const double fl1 = 1.0 / static_cast<double>(n + 1);
    const double fln = (n == 0) ? 1.0 : n * fl1;
    const double sign_n = (n % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < ndim; ++i) {
      Tm[idx(n, i, ndim)] =
          (sqc[i] + sign_n * sqa[i] -
           (2.0 * n + 3.0) * fl1 * r1[i] * Tm[idx(n + 1, i, ndim)] -
           r0[i] * Tm[idx(n + 2, i, ndim)]) *
          fln;
    }
  }

  // ---- Transformation V^+-, V^-, -> T^+, T^- ------------------------------
  // Walk n from lmax down to 1 so that Tp[n-1] reads the original value before
  // it is overwritten at the next iteration.
  for (int n = lmax; n >= 1; --n) {
    double fln_denom = static_cast<double>(n - 1);
    if (n == 1) fln_denom = 1.0;
    const double fl = static_cast<double>(n) / fln_denom;
    const double sign_n = (n % 2 == 0) ? 1.0 : -1.0;
    for (int i = 0; i < ndim; ++i) {
      const double Vp_n = Tp[idx(n, i, ndim)];
      const double Vp_nm1 = Tp[idx(n - 1, i, ndim)];
      const double Vm_n = Tm[idx(n, i, ndim)];
      const double Vm_nm1 = Tm[idx(n - 1, i, ndim)];
      Tp[idx(n, i, ndim)] =
          rpc[i] + sign_n * rpa[i] - n * (rpz[i] * Vp_n + ca[i] * Vp_nm1);
      Tm[idx(n, i, ndim)] =
          rmc[i] + sign_n * rma[i] - rmz[i] * Vm_n - ca[i] * Vm_nm1 * fl;
    }
  }
  for (int i = 0; i < ndim; ++i) {
    Tp[idx(0, i, ndim)] = rpc[i] + rpa[i];
    Tm[idx(0, i, ndim)] = rmc[i] + rma[i];
  }

  // ---- Sign-of-b swap -----------------------------------------------------
  for (int l = 0; l <= lmax; ++l) {
    for (int i = 0; i < ndim; ++i) {
      if (b[i] < 0.0) {
        std::swap(Tp[idx(l, i, ndim)], Tm[idx(l, i, ndim)]);
      }
    }
  }

  // Shrink to just the requested range [0, lmax].
  out.Tp.assign(Tp.begin(),
                Tp.begin() + static_cast<std::ptrdiff_t>((lmax + 1)) * ndim);
  out.Tm.assign(Tm.begin(),
                Tm.begin() + static_cast<std::ptrdiff_t>((lmax + 1)) * ndim);
  return out;
}

}  // namespace vmecpp

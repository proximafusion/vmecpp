// Port of vac2/analin.f90 (except the large-|m,n| asymptotic block at
// lines 108-139, which is a later refinement). The Fortran source is the
// authoritative reference for the reduction structure and sign conventions.

#include "vmecpp/free_boundary/vac2/vac2_mode_assembly.h"

#include <cstddef>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace vmecpp {

namespace {

inline std::size_t Aco3Idx(int l, int m, int n, int mnf) {
  const int stride = mnf + 1;
  return (static_cast<std::size_t>(l) * stride + m) * stride + n;
}

}  // namespace

Vac2ModeAssembly::Vac2ModeAssembly(int mnf) : mnf_(mnf) { BuildAcoTable(); }

double Vac2ModeAssembly::aco(int l, int m, int n) const {
  return aco_[Aco3Idx(l, m, n, mnf_)];
}

void Vac2ModeAssembly::BuildAcoTable() {
  const int S = mnf_ + 1;
  aco_.assign(static_cast<std::size_t>(S) * S * S, 0.0);
  aco_[Aco3Idx(0, 0, 0, mnf_)] = 1.0;
  for (int m = 1; m <= mnf_; ++m) {
    aco_[Aco3Idx(0, m, 0, mnf_)] = 1.0;
    for (int n = 1; n <= m; ++n) {
      aco_[Aco3Idx(0, m, n, mnf_)] = aco_[Aco3Idx(0, m, n - 1, mnf_)] *
                                     (m + 1 - n) / static_cast<double>(n);
      for (int l = 1; l <= n; ++l) {
        aco_[Aco3Idx(l, m, n, mnf_)] =
            -aco_[Aco3Idx(l - 1, m, n, mnf_)] *
            static_cast<double>((m + l) * (n + 1 - l)) /
            static_cast<double>((m - n + l) * l);
      }
    }
  }
}

absl::StatusOr<Vac2ModeAssembly::Output> Vac2ModeAssembly::Compute(
    std::span<const double> a, std::span<const double> b,
    std::span<const double> c, std::span<const double> az,
    std::span<const double> bz, std::span<const double> cz) const {
  if (a.size() != b.size() || b.size() != c.size() || c.size() != az.size() ||
      az.size() != bz.size() || bz.size() != cz.size()) {
    return absl::InvalidArgumentError(
        "Vac2ModeAssembly: a, b, c, az, bz, cz must all have equal length");
  }
  const int ndim = static_cast<int>(a.size());
  const int mnf = mnf_;
  const int lmax = 2 * mnf + 2;
  const int n_range = 2 * mnf + 1;

  // Lambda that computes T^+-_l on the given metric, then applies the
  // (az, bz, cz) 3-term differential combination in-place. The dz_sign flag
  // flips the middle-term sign on the second call (vac2/analin.f90:67-68).
  auto compute_and_diff = [&](std::span<const double> ma,
                              std::span<const double> mb,
                              std::span<const double> mc, double dz_sign,
                              std::vector<double>& rp_out,
                              std::vector<double>& rm_out) -> absl::Status {
    auto res = basis_integrals_.Compute(ma, mb, mc, lmax);
    if (!res.ok()) return res.status();
    rp_out = std::move(res->Tp);
    rm_out = std::move(res->Tm);
    const int l_alloc = lmax + 1;  // BasisIntegrals returns indices 0..lmax

    std::vector<double> azp(ndim), dz(ndim), azm(ndim);
    for (int i = 0; i < ndim; ++i) {
      azp[i] = az[i] + 2.0 * bz[i] + cz[i];
      dz[i] = 2.0 * (cz[i] - az[i]) * dz_sign;
      azm[i] = az[i] - 2.0 * bz[i] + cz[i];
    }
    // In vac2/analin.f90 the 3-term combination reads indices l, l+1, l+2.
    // BasisIntegrals fills l in [0, lmax], so we can safely run l in
    // [0, lmax-2].
    std::vector<double> rp_new(static_cast<std::size_t>(l_alloc) * ndim, 0.0);
    std::vector<double> rm_new(static_cast<std::size_t>(l_alloc) * ndim, 0.0);
    for (int l = 0; l <= lmax - 2; ++l) {
      for (int i = 0; i < ndim; ++i) {
        rp_new[l * ndim + i] = azm[i] * rp_out[l * ndim + i] +
                               dz[i] * rp_out[(l + 1) * ndim + i] +
                               azp[i] * rp_out[(l + 2) * ndim + i];
        rm_new[l * ndim + i] = azp[i] * rm_out[l * ndim + i] +
                               dz[i] * rm_out[(l + 1) * ndim + i] +
                               azm[i] * rm_out[(l + 2) * ndim + i];
      }
    }
    rp_out = std::move(rp_new);
    rm_out = std::move(rm_new);
    return absl::OkStatus();
  };

  // cp/cm layout: [m * (mnf+1) + n] * ndim + i, covering (m, n) in
  // [0, mnf] x [0, mnf]. The first call fills the lower triangle (n <= m);
  // the second call fills the upper triangle (n > m, swapped indices).
  const int cpn = mnf + 1;
  std::vector<double> cp(static_cast<std::size_t>(cpn) * cpn * ndim, 0.0);
  std::vector<double> cm(static_cast<std::size_t>(cpn) * cpn * ndim, 0.0);
  auto cp_idx = [&](int i, int m, int n) {
    return (static_cast<std::size_t>(m) * cpn + n) * ndim + i;
  };

  // ---- First call: rinteg(a, b, c) then lower-triangle reduction ----------
  std::vector<double> rp, rm;
  if (auto st = compute_and_diff(a, b, c, +1.0, rp, rm); !st.ok()) return st;
  for (int m = 0; m <= mnf; ++m) {
    for (int n = 0; n <= m; ++n) {
      for (int l = 0; l <= n; ++l) {
        const double w = aco(l, m, n);
        const int l_idx = m - n + 2 * l;
        for (int i = 0; i < ndim; ++i) {
          cp[cp_idx(i, m, n)] += w * rp[l_idx * ndim + i];
          cm[cp_idx(i, m, n)] += w * rm[l_idx * ndim + i];
        }
      }
    }
  }

  // ---- Second call: rinteg(c, b, a) + upper-triangle reduction ------------
  if (auto st = compute_and_diff(c, b, a, -1.0, rp, rm); !st.ok()) return st;
  for (int m = 1; m <= mnf; ++m) {
    for (int n = 0; n < m; ++n) {
      for (int l = n; l >= 0; --l) {
        const double w = aco(l, m, n);
        const int l_idx = m - n + 2 * l;
        for (int i = 0; i < ndim; ++i) {
          cp[cp_idx(i, n, m)] += w * rp[l_idx * ndim + i];
          cm[cp_idx(i, n, m)] += w * rm[l_idx * ndim + i];
        }
      }
    }
  }

  // ---- Assemble fk(i, m, n) from averaged cp/cm entries -------------------
  Output out;
  out.mnf = mnf;
  out.ndim = ndim;
  out.fk.assign(static_cast<std::size_t>(cpn) * n_range * ndim, 0.0);
  auto fk_idx = [&](int i, int m, int n) {
    return (static_cast<std::size_t>(m) * n_range + (n + mnf)) * ndim + i;
  };

  for (int i = 0; i < ndim; ++i) {
    out.fk[fk_idx(i, 0, 0)] = cp[cp_idx(i, 0, 0)] + cm[cp_idx(i, 0, 0)];
  }
  for (int n = 1; n <= mnf; ++n) {
    for (int i = 0; i < ndim; ++i) {
      const double v = 0.5 * (cp[cp_idx(i, 0, n)] + cp[cp_idx(i, 0, n - 1)] +
                              cm[cp_idx(i, 0, n)] + cm[cp_idx(i, 0, n - 1)]);
      out.fk[fk_idx(i, 0, n)] = v;
      out.fk[fk_idx(i, 0, -n)] = v;
    }
  }
  for (int m = 1; m <= mnf; ++m) {
    for (int i = 0; i < ndim; ++i) {
      out.fk[fk_idx(i, m, 0)] =
          0.5 * (cp[cp_idx(i, m, 0)] + cp[cp_idx(i, m - 1, 0)] +
                 cm[cp_idx(i, m, 0)] + cm[cp_idx(i, m - 1, 0)]);
    }
  }
  for (int m = 1; m <= mnf; ++m) {
    for (int n = 1; n <= mnf; ++n) {
      for (int i = 0; i < ndim; ++i) {
        out.fk[fk_idx(i, m, n)] =
            0.5 * (cp[cp_idx(i, m, n)] + cp[cp_idx(i, m - 1, n)] +
                   cp[cp_idx(i, m, n - 1)] + cp[cp_idx(i, m - 1, n - 1)]);
        out.fk[fk_idx(i, m, -n)] =
            0.5 * (cm[cp_idx(i, m, n)] + cm[cp_idx(i, m - 1, n)] +
                   cm[cp_idx(i, m, n - 1)] + cm[cp_idx(i, m - 1, n - 1)]);
      }
    }
  }

  // TODO(nestor): port the large-|m,n| asymptotic closed-form block at
  // vac2/analin.f90:108-139. It replaces fk for m^2+n^2 > 144 AND m>4 AND n>4
  // with a far-field formula involving the metric components directly. Not
  // required for correctness at small modes; add with its own test layer.

  return out;
}

}  // namespace vmecpp

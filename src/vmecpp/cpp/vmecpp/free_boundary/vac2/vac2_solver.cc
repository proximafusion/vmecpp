// Port of vac2/{precal, matrix, fourier, foumat, bexmat, solver}.f90 into a
// single C++ class. The Fortran source is the authoritative reference; see
// inline comments for algorithmic anchors.
//
// High-level pipeline (mirroring the Fortran):
//   1. Precal (constructor)     -- trig tables, mode index arrays, tau/tav.
//   2. Matrix (Solve stage 1)   -- ga, gc, gd on nuv*nuv grid with period
//                                  folding and log-singular tangent-plane
//                                  subtraction. Note: gb is folded into gc's
//                                  upper triangle during symmetrisation.
//   3. Fourier (Solve stage 2)  -- first-pass DFT reducing ga/gc/gd along the
//                                  "spatial j" index to produce
//                                  (ac, as, bc, bs, dc, ds), each (nuv, mnpot).
//   4. Foumat (Solve stage 3)   -- second-pass DFT reducing along the
//                                  remaining spatial index to produce the
//                                  twelve mode-space matrices
//                                  (acc, asc, acs, ass, bcc, bsc, bcs, bss,
//                                   dcc, dsc, dcs, dss), each (mnpot, mnpot).
//   5. Bexmat (Solve stage 5)   -- Fourier-project bexn into benf[nd].
//   6. Assemble (stage 4)       -- build nd*nd block matrix g from 12 mode
//                                  matrices following solver.f90:37-48.
//   7. Assemble cpol, ctor      -- derived from the (k=1) row/column of the
//                                  mode matrices (solver.f90:55-60).
//   8. RHS and solve (stage 7)  -- rhs = cpol*curpol + ctor*curtor - benf;
//                                  dpotrf, dpotrs (Cholesky).
//   9. Reconstruct (stage 8)    -- potU, potV on the nu*nv grid by mode sum
//                                  with the `-curtor, +curpol` secular base.
//  10. bsqvac (stage 9)         -- pointwise 1/2 |B_vac|^2 from the surface
//                                  metric.
//
// Mode layout (cf. precal.f90:103-115 and header):
//   First (ntor+1) modes: (m=0, n=0..ntor).
//   Then, for m = 1..mpol-1: (m, n=-ntor..ntor).
//   mnpot = (ntor+1) + (mpol-1)*(2*ntor+1).
//   nd = 2 * (mnpot - 1) (the m=n=0 entry is a constant-shift mode handled
//   by the secular `-curtor, +curpol` base, so it is skipped in the solve).

#include "vmecpp/free_boundary/vac2/vac2_solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef VMECPP_VAC2_HAVE_FFTW3
#include <fftw3.h>
#endif

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

// LAPACK Cholesky factor (dpotrf) and solve (dpotrs).
extern "C" {
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda,
             int* info);
void dpotrs_(const char* uplo, const int* n, const int* nrhs, const double* a,
             const int* lda, double* b, const int* ldb, int* info);
}

namespace vmecpp {

namespace {
constexpr double kPi = 3.14159265358979323846;
constexpr double kTwoPi = 2.0 * kPi;

// mnpot formula: (ntor+1) + (mpol-1)*(2*ntor+1). Encapsulated so the
// constructor initializer list can call it.
int ComputeMnpot(int mpol, int ntor) {
  return (ntor + 1) + (mpol - 1) * (2 * ntor + 1);
}
}  // namespace

// ----------------------------------------------------------------------------
// Constructor: port of vac2_precal.
// ----------------------------------------------------------------------------
Vac2Solver::Vac2Solver(int mpol, int ntor, int nu, int nv, int nfp)
    : mpol_(mpol),
      ntor_(ntor),
      nu_(nu),
      nv_(nv),
      nfp_(nfp),
      mnpot_(ComputeMnpot(mpol, ntor)),
      nd_(2 * (mnpot_ - 1)) {
  // Mode index tables ma_[k], na_[k] for k in [0, mnpot_).
  ma_.assign(mnpot_, 0);
  na_.assign(mnpot_, 0);
  int k = 0;
  // First block: (m=0, n=0..ntor_), that's (ntor_+1) entries.
  for (int n = 0; n <= ntor_; ++n) {
    ma_[k] = 0;
    na_[k] = n;
    ++k;
  }
  // Remaining blocks: (m=1..mpol_-1, n=-ntor_..ntor_).
  for (int m = 1; m <= mpol_ - 1; ++m) {
    for (int n = -ntor_; n <= ntor_; ++n) {
      ma_[k] = m;
      na_[k] = n;
      ++k;
    }
  }
  // k should equal mnpot_ here.

  // Trig tables: conu_[ku * mpol_ + m] = cos(2pi * m * ku / nu) for
  // ku in [0, nu_), m in [0, mpol_).
  conu_.assign(static_cast<std::size_t>(nu_) * mpol_, 0.0);
  sinu_.assign(static_cast<std::size_t>(nu_) * mpol_, 0.0);
  const double alu = kTwoPi / static_cast<double>(nu_);
  for (int ku = 0; ku < nu_; ++ku) {
    for (int m = 0; m < mpol_; ++m) {
      const double arg = alu * m * ku;
      conu_[static_cast<std::size_t>(ku) * mpol_ + m] = std::cos(arg);
      sinu_[static_cast<std::size_t>(ku) * mpol_ + m] = std::sin(arg);
    }
  }
  // conv_[kv * (ntor_+1) + n] = cos(2pi * n * kv / nv) for kv in [0, nv_),
  // n in [0, ntor_]. Signed n handled via conv_ being even and sinv_ being
  // odd under n -> -n (see accessor helpers below).
  conv_.assign(static_cast<std::size_t>(nv_) * (ntor_ + 1), 0.0);
  sinv_.assign(static_cast<std::size_t>(nv_) * (ntor_ + 1), 0.0);
  const double alv = kTwoPi / static_cast<double>(nv_);
  for (int kv = 0; kv < nv_; ++kv) {
    for (int n = 0; n <= ntor_; ++n) {
      const double arg = alv * n * kv;
      conv_[static_cast<std::size_t>(kv) * (ntor_ + 1) + n] = std::cos(arg);
      sinv_[static_cast<std::size_t>(kv) * (ntor_ + 1) + n] = std::sin(arg);
    }
  }
}

namespace {

// Accessors for signed-n trig tables.
inline double ConV(const std::vector<double>& conv, int ntor, int kv, int n) {
  // cos(-n v) = cos(n v)
  return conv[static_cast<std::size_t>(kv) * (ntor + 1) + std::abs(n)];
}
inline double SinV(const std::vector<double>& sinv, int ntor, int kv, int n) {
  // sin(-n v) = -sin(n v)
  const double s =
      sinv[static_cast<std::size_t>(kv) * (ntor + 1) + std::abs(n)];
  return (n < 0) ? -s : s;
}

// Build tau(ku) = tan(0.5 * alu * ku) / pi for |ku| != nu/2, else 1e20.
// Stored with offset so the argument ku in [-nu+1, nu-1] maps to
// tau_flat[ku + (nu-1)].
void BuildTauTable(int nu, std::vector<double>* tau) {
  tau->assign(2 * nu - 1, 0.0);
  const double alu = kTwoPi / static_cast<double>(nu);
  for (int ku = -(nu - 1); ku <= nu - 1; ++ku) {
    const int idx = ku + (nu - 1);
    if (std::abs(ku) == nu / 2) {
      (*tau)[idx] = 1.0e20;
    } else {
      (*tau)[idx] = std::tan(0.5 * alu * ku) / kPi;
    }
  }
}

// Flat index access for nuv-sized row-major matrices of shape (nuv, nuv).
inline std::size_t Idx2(int i, int j, int nuv) {
  return static_cast<std::size_t>(i) * static_cast<std::size_t>(nuv) +
         static_cast<std::size_t>(j);
}

// Convert a flat (ku, kv) grid index `[ku * nv + kv]` to poloidal / toroidal
// components.
inline int Iu(int i, int nv) { return i / nv; }  // ku
inline int Iv(int i, int nv) { return i % nv; }  // kv

// --------------------------------------------------------------------------
// Port of vac2_rinteg. Given per-point arrays a[ndim], b[ndim], c[ndim]
// (typically (guu, guv, gvv) at each grid point), fill tp[ndim][0..lmax+3]
// and tm[ndim][0..lmax+3] with the recurrence terms that represent the
// analytic contribution of the log-singular tangent-plane kernel to each
// Fourier mode.
//
// tp/tm are stored flat with stride (lmax+4): tp[i*(lmax+4) + l].
//
// Fortran source: vac2/rinteg.f90 (faithful line-by-line port).
// --------------------------------------------------------------------------
void Rinteg(const std::vector<double>& a, const std::vector<double>& b,
            const std::vector<double>& c, int lmax, int ndim,
            std::vector<double>* tp, std::vector<double>* tm,
            bool stable = true) {
  constexpr int kLasym = 30;
  const int stride = lmax + 4;
  tp->assign(static_cast<std::size_t>(ndim) * stride, 0.0);
  tm->assign(static_cast<std::size_t>(ndim) * stride, 0.0);

  std::vector<double> d(ndim), sqa(ndim), sqc(ndim), r1(ndim), r0(ndim),
      ap(ndim), rpc(ndim), rmc(ndim), rpa(ndim), rma(ndim);
  std::vector<double> sa(ndim), sc(ndim), ca(ndim), rpz(ndim), rmz(ndim),
      acb(ndim);

  for (int i = 0; i < ndim; ++i) {
    d[i] = std::abs(b[i]);
    sa[i] = std::sqrt(a[i]);
    sc[i] = std::sqrt(c[i]);
    acb[i] = 0.25 / (a[i] * c[i] - b[i] * b[i]);
    rpc[i] = (c[i] + d[i]) * acb[i] / sc[i];
    rmc[i] = (c[i] - d[i]) * acb[i] / sc[i];
    rpa[i] = (a[i] + d[i]) * acb[i] / sa[i];
    rma[i] = (a[i] - d[i]) * acb[i] / sa[i];
    rpz[i] = a[i] + 2.0 * d[i] + c[i];
    rmz[i] = a[i] - 2.0 * d[i] + c[i];
    ca[i] = (c[i] - a[i]) * acb[i];
    ap[i] = 1.0 / rpz[i];
    sqa[i] = 2.0 * sa[i] * ap[i];
    sqc[i] = 2.0 * sc[i] * ap[i];
    r1[i] = (c[i] - a[i]) * ap[i];
    r0[i] = rmz[i] * ap[i];
    rpz[i] = rpz[i] * acb[i];
    rmz[i] = rmz[i] * acb[i];
  }

  // Save unscaled a^+ and a^- BEFORE the *= acb overwrote rpz/rmz above.
  // These are needed for the unstable forward T^- recurrence.
  // Recompute from the original a, b, c since rpz/rmz are already scaled.
  std::vector<double> rpz_raw(ndim), rmz_raw(ndim);
  if (!stable) {
    for (int i = 0; i < ndim; ++i) {
      rpz_raw[i] = a[i] + 2.0 * d[i] + c[i];  // a^+ = a + 2|b| + c
      rmz_raw[i] = a[i] - 2.0 * d[i] + c[i];  // a^- = a - 2|b| + c
    }
  }

  auto TP = [&](int i, int l) -> double& {
    return (*tp)[static_cast<std::size_t>(i) * stride + l];
  };
  auto TM = [&](int i, int l) -> double& {
    return (*tm)[static_cast<std::size_t>(i) * stride + l];
  };

  for (int i = 0; i < ndim; ++i) {
    const double sap = std::sqrt(ap[i]);
    TP(i, 0) = sap * std::log((sc[i] + (c[i] + d[i]) * sap) /
                              (sa[i] - (a[i] + d[i]) * sap));
    TP(i, 1) = -r1[i] * TP(i, 0) + sqc[i] - sqa[i];
  }

  // Forward recurrence for tp: n = 0..lmax+1 fills tp(.,n+2).
  for (int n = 0; n <= lmax + 1; ++n) {
    const double fl2 = 1.0 / static_cast<double>(n + 2);
    const double sgn = ((n & 1) == 0) ? 1.0 : -1.0;
    for (int i = 0; i < ndim; ++i) {
      TP(i, n + 2) =
          (sqc[i] + sgn * sqa[i] - (2.0 * n + 3.0) * r1[i] * TP(i, n + 1) -
           (n + 1.0) * r0[i] * TP(i, n)) *
          fl2;
    }
  }

  if (stable) {
    // Continued forward recurrence for asymptotic init of tm, using a
    // 3-slot sliding window at index k. Fortran writes this as tp(:,k),
    // tp(:,k+1), tp(:,k+2) with k = lmax+1; we mirror that.
    const int k = lmax + 1;
    std::vector<double> tp_k(ndim), tp_kp1(ndim), tp_kp2(ndim);
    for (int i = 0; i < ndim; ++i) {
      tp_k[i] = TP(i, k);
      tp_kp1[i] = TP(i, k + 1);
      tp_kp2[i] = TP(i, k + 2);
    }
    for (int n = lmax + 2; n <= lmax + 1 + kLasym; ++n) {
      const double fl2 = 1.0 / static_cast<double>(n + 2);
      const double sgn = ((n & 1) == 0) ? 1.0 : -1.0;
      for (int i = 0; i < ndim; ++i) {
        // Shift: tp(:,k) <- tp(:,k+1); tp(:,k+1) <- tp(:,k+2); then
        // compute new tp(:,k+2).
        tp_k[i] = tp_kp1[i];
        tp_kp1[i] = tp_kp2[i];
        tp_kp2[i] =
            (sqc[i] + sgn * sqa[i] - (2.0 * n + 3.0) * r1[i] * tp_kp1[i] -
             (n + 1.0) * r0[i] * tp_k[i]) *
            fl2;
      }
    }
    // At this point tp_k, tp_kp1, tp_kp2 hold the values at the final n.
    // We need to seed tm at indices k+1 and k+2 (these are the top end).
    const int nup = lmax + 3 + kLasym;
    const int num = nup - 1;
    const double sgn_nup = ((nup & 1) == 0) ? 1.0 : -1.0;
    const double sgn_num = ((num & 1) == 0) ? 1.0 : -1.0;
    std::vector<double> tm_k(ndim), tm_kp1(ndim), tm_kp2(ndim);
    for (int i = 0; i < ndim; ++i) {
      const double factor_nup =
          4.0 * std::abs(b[i]) / static_cast<double>((nup + 1) * (nup + 3));
      const double factor_num =
          4.0 * std::abs(b[i]) / static_cast<double>((num + 1) * (num + 3));
      const double inv_2sc_cubed =
          (0.5 / sc[i]) * (0.5 / sc[i]) * (0.5 / sc[i]);
      const double inv_2sa_cubed =
          (0.5 / sa[i]) * (0.5 / sa[i]) * (0.5 / sa[i]);
      tm_kp2[i] =
          nup *
          (tp_kp2[i] - factor_nup * (inv_2sc_cubed + sgn_nup * inv_2sa_cubed));
      tm_kp1[i] =
          num *
          (tp_kp1[i] - factor_num * (inv_2sc_cubed + sgn_num * inv_2sa_cubed));
    }

    // Backward recurrence for tm from n=lmax+1+lasym down to n=lmax+2.
    // Sliding 3-window logic: compute tm(:,k) = f(tm(:,k+1), tm(:,k+2));
    // then shift tm(:,k+2)<-tm(:,k+1); tm(:,k+1)<-tm(:,k).
    for (int n = lmax + 1 + kLasym; n >= lmax + 2; --n) {
      const double fl1 = 1.0 / static_cast<double>(n + 1);
      const double fln = n * fl1;
      const double sgn = ((n & 1) == 0) ? 1.0 : -1.0;
      for (int i = 0; i < ndim; ++i) {
        tm_k[i] =
            (sqc[i] + sgn * sqa[i] - (2.0 * n + 3.0) * fl1 * r1[i] * tm_kp1[i] -
             r0[i] * tm_kp2[i]) *
            fln;
        // Shift:
        tm_kp2[i] = tm_kp1[i];
        tm_kp1[i] = tm_k[i];
      }
    }
    // After loop: tm_kp1 holds tm at n=lmax+2. For the next loop we need
    // tm(:, n+1) and tm(:, n+2) starting from n=lmax+1. These are exactly
    // tm_kp1 (= n=lmax+2) and tm_kp2 (= n=lmax+3). Initialize tm(:,n+1)
    // = tm_kp1, tm(:,n+2) = tm_kp2 as we step n down into the stored tm
    // array.
    // Store them into TM at indices lmax+2 and lmax+3 so the next loop
    // can read them.
    for (int i = 0; i < ndim; ++i) {
      TM(i, lmax + 2) = tm_kp1[i];
      TM(i, lmax + 3) = tm_kp2[i];
    }

    // Backward recurrence from n=lmax+1 down to n=0 storing into TM.
    for (int n = lmax + 1; n >= 0; --n) {
      const double fl1 = 1.0 / static_cast<double>(n + 1);
      double fln = n * fl1;
      if (n == 0) fln = 1.0;
      const double sgn = ((n & 1) == 0) ? 1.0 : -1.0;
      for (int i = 0; i < ndim; ++i) {
        TM(i, n) = (sqc[i] + sgn * sqa[i] -
                    (2.0 * n + 3.0) * fl1 * r1[i] * TM(i, n + 1) -
                    r0[i] * TM(i, n + 2)) *
                   fln;
      }
    }
  } else {
    // =========================================================================
    // Unstable (forward-only) recurrence for T^- (TM).
    //
    // From STARWALL eq. D.15, the T^- forward recurrence uses a^- = rmz_raw
    // as the denominator factor and a^+ = rpz_raw in the T_{l-2} coupling:
    //
    //   T^-_0 = (1/sqrt(a^-)) * log((sqrt(c)*a^- + c - |b|) /
    //                                (sqrt(a)*a^- - a + |b|))
    //   T^-_1 = (1/a^-) * (2*(sqrt(c) - sqrt(a)) - (c - a)*T^-_0)
    //   T^-_l = (1/(l*a^-)) * (2*(sqrt(c) + (-1)^l*sqrt(a))
    //           - (2l-1)*(c-a)*T^-_{l-1} - (l-1)*a^+*T^-_{l-2})   for l >= 2
    //
    // Storage convention: TM(i, l) = l * T^-_l for l >= 1, TM(i, 0) = T^-_0
    // (matching the backward recurrence convention where fln=1 at n=0).
    // =========================================================================

    // T^-_0 initial condition.
    // Use the Fortran analyt.f90 formula (numerically stable):
    //   T^-_0 = (1/sqrt(adm)) * log((sqrt(adm)*2*sqrt(c) + adm + c-a) /
    //                                (sqrt(adm)*2*sqrt(a) - adm + c-a))
    // NOT the D.15 form (sqrt(c)*adm + c-|b|) / (sqrt(a)*adm - a+|b|)
    // which suffers from sign cancellation in the denominator.
    for (int i = 0; i < ndim; ++i) {
      const double arm = rmz_raw[i];           // a^- = a - 2|b| + c
      const double sqrt_arm = std::sqrt(arm);  // sqrt(a^-)
      const double sqrtc2 = 2.0 * sc[i];       // 2*sqrt(c)
      const double sqrta2 = 2.0 * sa[i];       // 2*sqrt(a)
      const double cma = c[i] - a[i];          // c - a
      TM(i, 0) = (1.0 / sqrt_arm) * std::log((sqrt_arm * sqrtc2 + arm + cma) /
                                             (sqrt_arm * sqrta2 - arm + cma));
    }

    // T^-_1: matching Fortran analyt.f90 convention.
    // T^-_1 = (1/a^-) * (2*(sqrt(c) - sqrt(a)) - (c-a)*T^-_0)
    // But using the Fortran's sqrtc = 2*sqrt(c), sqrta = 2*sqrt(a):
    //   = (1/adm) * (sqrtc - sqrta - cma*T^-_0)
    // Note: 2*(sqrt(c)-sqrt(a)) = sqrtc2 - sqrta2, and (c-a) = cma.
    for (int i = 0; i < ndim; ++i) {
      const double sqrtc2 = 2.0 * sc[i];
      const double sqrta2 = 2.0 * sa[i];
      const double cma = c[i] - a[i];
      const double Tm1 = (sqrtc2 - sqrta2 - cma * TM(i, 0)) / rmz_raw[i];
      TM(i, 1) = Tm1;  // 1 * T^-_1
    }

    // Forward recurrence for l = 2 .. lmax+3.
    // From D.15: T^-_l = (1/(l*a^-)) * (2*(sqrt(c)+(-1)^l*sqrt(a))
    //                     - (2l-1)*(c-a)*T^-_{l-1} - (l-1)*a^+*T^-_{l-2})
    // Using sqrtc2=2*sqrt(c), sqrta2=2*sqrt(a):
    //   T^-_l = (1/(l*adm)) * (sqrtc2 + (-1)^l*sqrta2
    //           - (2l-1)*cma*T^-_{l-1} - (l-1)*adp*T^-_{l-2})
    for (int i = 0; i < ndim; ++i) {
      const double sqrtc2 = 2.0 * sc[i];
      const double sqrta2 = 2.0 * sa[i];
      const double cma = c[i] - a[i];
      double Tm_prev2 = TM(i, 0);  // T^-_0
      double Tm_prev1 = TM(i, 1);  // T^-_1
      for (int l = 2; l <= lmax + 3; ++l) {
        const double sgn = ((l & 1) == 0) ? 1.0 : -1.0;
        const double Tm_l =
            (sqrtc2 + sgn * sqrta2 - (2.0 * l - 1.0) * cma * Tm_prev1 -
             static_cast<double>(l - 1) * rpz_raw[i] * Tm_prev2) /
            (static_cast<double>(l) * rmz_raw[i]);
        TM(i, l) = static_cast<double>(l) * Tm_l;  // store l * T^-_l
        Tm_prev2 = Tm_prev1;
        Tm_prev1 = Tm_l;
      }
    }
  }  // end if (stable) ... else

  // Final transformation of tp and tm for n = lmax down to 1.
  for (int n = lmax; n >= 1; --n) {
    double fln = static_cast<double>(n - 1);
    if (n == 1) fln = 1.0;
    const double fl = n / fln;
    const double sgn = ((n & 1) == 0) ? 1.0 : -1.0;
    for (int i = 0; i < ndim; ++i) {
      TP(i, n) = rpc[i] + sgn * rpa[i] -
                 n * (rpz[i] * TP(i, n) + ca[i] * TP(i, n - 1));
      TM(i, n) =
          rmc[i] + sgn * rma[i] - rmz[i] * TM(i, n) - ca[i] * TM(i, n - 1) * fl;
    }
  }
  for (int i = 0; i < ndim; ++i) {
    TP(i, 0) = rpc[i] + rpa[i];
    TM(i, 0) = rmc[i] + rma[i];
  }

  // Branch swap: when b(i) < 0, swap tp and tm.
  for (int l = 0; l <= lmax; ++l) {
    for (int i = 0; i < ndim; ++i) {
      if (b[i] < 0.0) {
        std::swap(TP(i, l), TM(i, l));
      }
    }
  }
}

// --------------------------------------------------------------------------
// Port of vac2_analin. Compute fk[ndim, 0..mnf, -mnf..mnf] — the analytic
// Fourier mode coefficients of the regularized log-singular kernel.
//
// fk is stored flat with layout [i * (mnf+1) * (2*mnf+1) + m * (2*mnf+1)
// + (n + mnf)].
//
// Fortran source: vac2/analin.f90 (faithful port).
// --------------------------------------------------------------------------
void Analin(int mnf, int ndim, const std::vector<double>& a,
            const std::vector<double>& b, const std::vector<double>& c,
            const std::vector<double>& az, const std::vector<double>& bz,
            const std::vector<double>& cz, std::vector<double>* fk,
            bool stable = true) {
  const int lmax = 2 * mnf + 2;
  const std::size_t fk_stride_m = static_cast<std::size_t>(2 * mnf + 1);
  const std::size_t fk_stride_i =
      fk_stride_m * static_cast<std::size_t>(mnf + 1);
  fk->assign(static_cast<std::size_t>(ndim) * fk_stride_i, 0.0);
  auto FK = [&](int i, int m, int n) -> double& {
    return (*fk)[static_cast<std::size_t>(i) * fk_stride_i +
                 static_cast<std::size_t>(m) * fk_stride_m +
                 static_cast<std::size_t>(n + mnf)];
  };

  std::vector<double> rp, rm;
  Rinteg(a, b, c, lmax, ndim, &rp, &rm, stable);
  const int stride = lmax + 4;
  auto RP = [&](int i, int l) -> double& {
    return rp[static_cast<std::size_t>(i) * stride + l];
  };
  auto RM = [&](int i, int l) -> double& {
    return rm[static_cast<std::size_t>(i) * stride + l];
  };

  // Pascal-like coefficient array aco(l, m, n) for 0 <= l <= n <= m <= mnf.
  // Stored flat with layout [l * (mnf+1)^2 + m * (mnf+1) + n].
  const std::size_t aco_stride = static_cast<std::size_t>(mnf + 1);
  std::vector<double> aco(aco_stride * aco_stride * aco_stride, 0.0);
  auto ACO = [&](int l, int m, int n) -> double& {
    return aco[static_cast<std::size_t>(l) * aco_stride * aco_stride +
               static_cast<std::size_t>(m) * aco_stride +
               static_cast<std::size_t>(n)];
  };
  ACO(0, 0, 0) = 1.0;
  for (int m = 1; m <= mnf; ++m) {
    ACO(0, m, 0) = 1.0;
    for (int n = 1; n <= m; ++n) {
      ACO(0, m, n) = ACO(0, m, n - 1) * static_cast<double>(m + 1 - n) /
                     static_cast<double>(n);
      for (int l = 1; l <= n; ++l) {
        ACO(l, m, n) = -ACO(l - 1, m, n) *
                       static_cast<double>((m + l) * (n + 1 - l)) /
                       static_cast<double>((m - n + l) * l);
      }
    }
  }

  // Compute azp, dz, azm.
  std::vector<double> azp(ndim), dz(ndim), azm(ndim);
  for (int i = 0; i < ndim; ++i) {
    azp[i] = az[i] + 2.0 * bz[i] + cz[i];
    dz[i] = 2.0 * (cz[i] - az[i]);
    azm[i] = az[i] - 2.0 * bz[i] + cz[i];
  }

  // Transform rp, rm via three-term combination. The Fortran overwrites
  // rp(:,l) in place using rp(:,l), rp(:,l+1), rp(:,l+2). Since the
  // recurrence reads l+1 and l+2 (which haven't been overwritten yet at
  // this step), the loop order (ascending l) is fine.
  for (int l = 0; l <= lmax - 2; ++l) {
    for (int i = 0; i < ndim; ++i) {
      RP(i, l) =
          azm[i] * RP(i, l) + dz[i] * RP(i, l + 1) + azp[i] * RP(i, l + 2);
      RM(i, l) =
          azp[i] * RM(i, l) + dz[i] * RM(i, l + 1) + azm[i] * RM(i, l + 2);
    }
  }

  // cp, cm arrays: (ndim, 0..mnf, 0..mnf). Stored flat [i*(mnf+1)^2 +
  // r1*(mnf+1) + r2]. We store cp(i, m, n) and cm(i, m, n). Later we'll
  // also fill cp(i, n, m) for n < m (the "second pass").
  const std::size_t cp_stride_r1 = static_cast<std::size_t>(mnf + 1);
  const std::size_t cp_stride_i = cp_stride_r1 * cp_stride_r1;
  std::vector<double> cp(static_cast<std::size_t>(ndim) * cp_stride_i, 0.0);
  std::vector<double> cm(static_cast<std::size_t>(ndim) * cp_stride_i, 0.0);
  auto CP = [&](int i, int a_idx, int b_idx) -> double& {
    return cp[static_cast<std::size_t>(i) * cp_stride_i +
              static_cast<std::size_t>(a_idx) * cp_stride_r1 +
              static_cast<std::size_t>(b_idx)];
  };
  auto CM = [&](int i, int a_idx, int b_idx) -> double& {
    return cm[static_cast<std::size_t>(i) * cp_stride_i +
              static_cast<std::size_t>(a_idx) * cp_stride_r1 +
              static_cast<std::size_t>(b_idx)];
  };

  // Diagnostic: dump aco-weighted accumulation at corner mode for i=0.
  // Activated by env VAC2_ACO_DUMP=1.
  static const bool aco_dump = std::getenv("VAC2_ACO_DUMP") &&
                               std::atoi(std::getenv("VAC2_ACO_DUMP")) != 0;
  // Corner mode for the dump: (m=mnf, n=mnf) — the highest (m, n).
  const int aco_dump_m = mnf;
  const int aco_dump_n = mnf;
  const int aco_dump_i = 0;

  if (aco_dump && ndim > 0) {
    std::fprintf(stderr, "ACO_DUMP mnf=%d  corner=(m=%d, n=%d)  i=%d\n", mnf,
                 aco_dump_m, aco_dump_n, aco_dump_i);
    // Dump the post-transformed rp values at relevant l indices.
    // For corner (m=mnf, n=mnf), l ranges 0..mnf, r_idx = m-n+2l = 2l.
    std::fprintf(stderr, "ACO_DUMP rp[i=0] values at relevant r_idx:\n");
    for (int l = 0; l <= aco_dump_n; ++l) {
      const int r_idx = aco_dump_m - aco_dump_n + 2 * l;
      std::fprintf(stderr,
                   "ACO_DUMP   rp[i=0, r_idx=%d] = %.17e  "
                   "rm[i=0, r_idx=%d] = %.17e\n",
                   r_idx, RP(aco_dump_i, r_idx), r_idx, RM(aco_dump_i, r_idx));
    }
    // Dump aco coefficients at corner.
    std::fprintf(stderr, "ACO_DUMP aco coefficients at (m=%d, n=%d):\n",
                 aco_dump_m, aco_dump_n);
    for (int l = 0; l <= aco_dump_n; ++l) {
      std::fprintf(stderr, "ACO_DUMP   aco[l=%d, m=%d, n=%d] = %.17e\n", l,
                   aco_dump_m, aco_dump_n, ACO(l, aco_dump_m, aco_dump_n));
    }
  }

  // First pass: cp(i, m, n), cm(i, m, n) for 0 <= n <= m <= mnf.
  for (int m = 0; m <= mnf; ++m) {
    for (int n = 0; n <= m; ++n) {
      // Diagnostic: term-by-term dump at corner mode for i=0.
      const bool dump_this_mn =
          aco_dump && m == aco_dump_m && n == aco_dump_n && ndim > 0;
      double cp_partial = 0.0, cm_partial = 0.0;

      for (int l = 0; l <= n; ++l) {
        const double acol = ACO(l, m, n);
        const int r_idx = m - n + 2 * l;
        for (int i = 0; i < ndim; ++i) {
          CP(i, m, n) += acol * RP(i, r_idx);
          CM(i, m, n) += acol * RM(i, r_idx);
        }
        if (dump_this_mn) {
          const double term_cp = acol * RP(aco_dump_i, r_idx);
          const double term_cm = acol * RM(aco_dump_i, r_idx);
          cp_partial += term_cp;
          cm_partial += term_cm;
          std::fprintf(stderr,
                       "ACO_DUMP PASS1 l=%2d  aco=%.17e  rp[%d]=%.17e  "
                       "term_cp=%.17e  partial_cp=%.17e  |  "
                       "rm[%d]=%.17e  term_cm=%.17e  partial_cm=%.17e\n",
                       l, acol, r_idx, RP(aco_dump_i, r_idx), term_cp,
                       cp_partial, r_idx, RM(aco_dump_i, r_idx), term_cm,
                       cm_partial);
        }
      }
      if (dump_this_mn) {
        std::fprintf(stderr,
                     "ACO_DUMP PASS1 FINAL cp[i=0, m=%d, n=%d] = %.17e  "
                     "cm[i=0, m=%d, n=%d] = %.17e\n",
                     m, n, CP(aco_dump_i, m, n), m, n, CM(aco_dump_i, m, n));
      }
    }
  }

  // Second pass: recompute rp, rm by calling rinteg with a, c swapped.
  Rinteg(c, b, a, lmax, ndim, &rp, &rm, stable);
  for (int l = 0; l <= lmax - 2; ++l) {
    for (int i = 0; i < ndim; ++i) {
      RP(i, l) =
          azm[i] * RP(i, l) - dz[i] * RP(i, l + 1) + azp[i] * RP(i, l + 2);
      RM(i, l) =
          azp[i] * RM(i, l) - dz[i] * RM(i, l + 1) + azm[i] * RM(i, l + 2);
    }
  }
  // cp(i, n, m), cm(i, n, m) for n < m (stored into the upper triangle).
  for (int m = 1; m <= mnf; ++m) {
    for (int n = 0; n <= m - 1; ++n) {
      // Diagnostic: second pass uses SWAPPED rp/rm. Dump for the modes
      // that feed into fk[corner]. For fk(i, mnf, mnf) we need:
      //   CP(i, mnf, mnf)     -- first pass (m=n=mnf)
      //   CP(i, mnf-1, mnf)   -- second pass (m=mnf, n=mnf-1) stored as CP(i,
      //   mnf-1, mnf) CP(i, mnf, mnf-1)   -- first pass (m=mnf, n=mnf-1) CP(i,
      //   mnf-1, mnf-1) -- first pass (m=n=mnf-1)
      // The second pass fills CP(i, n, m) for n < m, so we need (n=mnf-1,
      // m=mnf).
      const bool dump_this_mn =
          aco_dump && m == aco_dump_m && n == aco_dump_n - 1 && ndim > 0;
      double cp_partial = 0.0, cm_partial = 0.0;

      // Note: Fortran loop is "do l=n,0,-1" — descending. The direction
      // doesn't affect the final summed value so we can loop ascending.
      for (int l = 0; l <= n; ++l) {
        const double acol = ACO(l, m, n);
        const int r_idx = m - n + 2 * l;
        for (int i = 0; i < ndim; ++i) {
          CP(i, n, m) += acol * RP(i, r_idx);
          CM(i, n, m) += acol * RM(i, r_idx);
        }
        if (dump_this_mn) {
          const double term_cp = acol * RP(aco_dump_i, r_idx);
          const double term_cm = acol * RM(aco_dump_i, r_idx);
          cp_partial += term_cp;
          cm_partial += term_cm;
          std::fprintf(stderr,
                       "ACO_DUMP PASS2 l=%2d  aco=%.17e  rp[%d]=%.17e  "
                       "term_cp=%.17e  partial_cp=%.17e  |  "
                       "rm[%d]=%.17e  term_cm=%.17e  partial_cm=%.17e\n",
                       l, acol, r_idx, RP(aco_dump_i, r_idx), term_cp,
                       cp_partial, r_idx, RM(aco_dump_i, r_idx), term_cm,
                       cm_partial);
        }
      }
      if (dump_this_mn) {
        std::fprintf(stderr,
                     "ACO_DUMP PASS2 FINAL cp[i=0, n=%d, m=%d] = %.17e  "
                     "cm[i=0, n=%d, m=%d] = %.17e\n",
                     n, m, CP(aco_dump_i, n, m), n, m, CM(aco_dump_i, n, m));
      }
    }
  }

  // Final assembly of fk.
  for (int i = 0; i < ndim; ++i) {
    FK(i, 0, 0) = CP(i, 0, 0) + CM(i, 0, 0);
  }
  for (int n = 1; n <= mnf; ++n) {
    for (int i = 0; i < ndim; ++i) {
      FK(i, 0, n) =
          0.5 * (CP(i, 0, n) + CP(i, 0, n - 1) + CM(i, 0, n) + CM(i, 0, n - 1));
      FK(i, 0, -n) = FK(i, 0, n);
    }
  }
  for (int m = 1; m <= mnf; ++m) {
    for (int i = 0; i < ndim; ++i) {
      FK(i, m, 0) =
          0.5 * (CP(i, m, 0) + CP(i, m - 1, 0) + CM(i, m, 0) + CM(i, m - 1, 0));
    }
  }
  for (int m = 1; m <= mnf; ++m) {
    for (int n = 1; n <= mnf; ++n) {
      for (int i = 0; i < ndim; ++i) {
        FK(i, m, n) = 0.5 * (CP(i, m, n) + CP(i, m - 1, n) + CP(i, m, n - 1) +
                             CP(i, m - 1, n - 1));
        FK(i, m, -n) = 0.5 * (CM(i, m, n) + CM(i, m - 1, n) + CM(i, m, n - 1) +
                              CM(i, m - 1, n - 1));
      }
      // Asymptotic replacement for large (m, n) where the recurrence is
      // numerically unstable.
      if (m * m + n * n > 144 && m > 4 && n > 4) {
        const double fm = static_cast<double>(m);
        for (int kk = 0; kk <= 1; ++kk) {
          const double fn = static_cast<double>(n) * ((kk == 0) ? 1.0 : -1.0);
          const double eps1 = 1.0 / std::sqrt(fm * fm + fn * fn);
          const double si = eps1 * fn;
          const double co = eps1 * fm;
          for (int i = 0; i < ndim; ++i) {
            const double ai = a[i], bi = b[i], ci = c[i];
            const double azi = az[i], bzi = bz[i], czi = cz[i];
            const double an2 =
                1.0 / (ai * si * si - 2.0 * bi * co * si + ci * co * co);
            const double an = std::sqrt(an2);
            const double cb = ci * co - bi * si;
            const double ab = ai * si - bi * co;
            const double z2 =
                (azi * si * si - 2.0 * bzi * co * si + czi * co * co) * an2;
            const double cbz = czi * co - bzi * si;
            const double abz = azi * si - bzi * co;
            // xn, yn, xyn are computed but unused in the final expression
            // (Fortran reads them via cb**2 etc, not via xn/yn). We retain
            // the bare expressions used below.
            const int n_signed = (kk == 0) ? n : -n;
            const double an3 = an2 * an;
            const double an4 = an2 * an2;
            // Fortran expression (analin.f90:129-136):
            //   eps1*an*z2 + eps1^3 * 0.5 * an^3 * (
            //     (az+cz)
            //     - 3*an2 * (
            //         (2*cbz*cb + cz*co*cb + c*co*cbz
            //          + 2*abz*ab + az*si*ab + a*si*abz)
            //         - 5*an2 * (co*cb^2*cbz + si*ab^2*abz)
            //       )
            //     - z2*0.5 * (
            //         3*(a+c)
            //         - 15*an2 * (cb^2 + ab^2 + co*c*cb + si*a*ab)
            //         + 35*an2^2 * (co*cb^3 + si*ab^3)
            //       )
            //   )
            const double X1 = 2.0 * cbz * cb + czi * co * cb + ci * co * cbz +
                              2.0 * abz * ab + azi * si * ab + ai * si * abz;
            const double X2 = co * cb * cb * cbz + si * ab * ab * abz;
            const double Y1 = cb * cb + ab * ab + co * ci * cb + si * ai * ab;
            const double Y2 = co * cb * cb * cb + si * ab * ab * ab;
            const double inner =
                (azi + czi) - 3.0 * an2 * (X1 - 5.0 * an2 * X2) -
                z2 * 0.5 *
                    (3.0 * (ai + ci) - 15.0 * an2 * Y1 + 35.0 * an4 * Y2);
            FK(i, m, n_signed) =
                eps1 * an * z2 + eps1 * eps1 * eps1 * 0.5 * an3 * inner;
          }
        }
      }
    }
  }

  // Diagnostic: dump the four CP/CM entries and the final fk at the corner.
  if (aco_dump && ndim > 0 && mnf >= 1) {
    const int m = aco_dump_m, n = aco_dump_n;
    std::fprintf(stderr,
                 "ACO_DUMP FK_ASSEMBLY (m=%d, n=%d):\n"
                 "ACO_DUMP   CP(i=0, m=%d, n=%d)   = %.17e\n"
                 "ACO_DUMP   CP(i=0, m=%d, n=%d) = %.17e\n"
                 "ACO_DUMP   CP(i=0, m=%d, n=%d) = %.17e\n"
                 "ACO_DUMP   CP(i=0, m=%d, n=%d) = %.17e\n"
                 "ACO_DUMP   CM(i=0, m=%d, n=%d)   = %.17e\n"
                 "ACO_DUMP   CM(i=0, m=%d, n=%d) = %.17e\n"
                 "ACO_DUMP   CM(i=0, m=%d, n=%d) = %.17e\n"
                 "ACO_DUMP   CM(i=0, m=%d, n=%d) = %.17e\n",
                 m, n, m, n, CP(0, m, n), m - 1, n, CP(0, m - 1, n), m, n - 1,
                 CP(0, m, n - 1), m - 1, n - 1, CP(0, m - 1, n - 1), m, n,
                 CM(0, m, n), m - 1, n, CM(0, m - 1, n), m, n - 1,
                 CM(0, m, n - 1), m - 1, n - 1, CM(0, m - 1, n - 1));
    std::fprintf(stderr,
                 "ACO_DUMP   fk(i=0, m=%d, n=+%d) = %.17e\n"
                 "ACO_DUMP   fk(i=0, m=%d, n=-%d) = %.17e\n",
                 m, n, FK(0, m, n), m, n, FK(0, m, -n));
    // Also report the cancellation ratio: max|term| vs |final sum| for CP.
    double max_cp_term = 0;
    for (int l = 0; l <= n; ++l) {
      const int r_idx = m - n + 2 * l;
      double t = std::abs(ACO(l, m, n) * RP(0, r_idx));
      if (t > max_cp_term) max_cp_term = t;
    }
    std::fprintf(stderr,
                 "ACO_DUMP   cancellation: max|aco*rp_term|=%.3e  "
                 "|CP(0,m,n)|=%.3e  ratio=%.3e\n",
                 max_cp_term, std::abs(CP(0, m, n)),
                 max_cp_term > 0 ? std::abs(CP(0, m, n)) / max_cp_term : 0.0);
  }
}

}  // namespace

// ----------------------------------------------------------------------------
// Solve
// ----------------------------------------------------------------------------
absl::StatusOr<std::unique_ptr<Vac2Solver::Output>> Vac2Solver::Solve(
    const Input& in, bool reuse_operator) {
  const int nuv = nu_ * nv_;
  const int mnpot = mnpot_;
  const int nd = nd_;
  const double fnuv = 1.0 / static_cast<double>(nuv);

  // Sanity-check input sizes.
  auto check = [&](const std::vector<double>& v,
                   const char* name) -> absl::Status {
    if (static_cast<int>(v.size()) != nuv) {
      return absl::InvalidArgumentError(
          absl::StrCat(name, " size ", v.size(), " != nuv=", nuv));
    }
    return absl::OkStatus();
  };
  for (const auto* p :
       {&in.x, &in.y, &in.z, &in.xu, &in.yu, &in.zu, &in.xv, &in.yv, &in.zv,
        &in.guu, &in.guv, &in.gvv, &in.snx, &in.sny, &in.snz, &in.bexn}) {
    if (auto st = check(*p, (p == &in.x     ? "x"
                             : p == &in.y   ? "y"
                             : p == &in.z   ? "z"
                             : p == &in.xu  ? "xu"
                             : p == &in.yu  ? "yu"
                             : p == &in.zu  ? "zu"
                             : p == &in.xv  ? "xv"
                             : p == &in.yv  ? "yv"
                             : p == &in.zv  ? "zv"
                             : p == &in.guu ? "guu"
                             : p == &in.guv ? "guv"
                             : p == &in.gvv ? "gvv"
                             : p == &in.snx ? "snx"
                             : p == &in.sny ? "sny"
                             : p == &in.snz ? "snz"
                                            : "bexn"));
        !st.ok()) {
      return st;
    }
  }

  auto out = std::make_unique<Output>();

  // Fast path: re-solve for the new right-hand side on the frozen operator
  // cached by the last full Solve (NESTOR-style partial update).
  if (reuse_operator && has_cached_operator_ && !in.lasym) {
    out->cpol = cached_cpol_;
    out->ctor = cached_ctor_;

    ComputeBenf(in, out.get());

    const int mnpot1 = mnpot - 1;
    std::vector<double> hs_cc(mnpot1, 0.0);
    for (int k = 0; k < mnpot1; ++k) {
      hs_cc[k] =
          out->cpol[k] * in.curpol + out->ctor[k] * in.curtor - out->benf[k];
    }
    const char uplo = 'U';
    const int nrhs = 1;
    int info = 0;
    dpotrs_(&uplo, &mnpot1, &nrhs, cached_cc_factor_.data(), &mnpot1,
            hs_cc.data(), &mnpot1, &info);
    if (info != 0) {
      return absl::InternalError(
          absl::StrCat("dpotrs(cached cc) failed with info=", info));
    }
    out->hs.assign(nd, 0.0);
    for (int k = 0; k < mnpot1; ++k) out->hs[k] = hs_cc[k];

    ReconstructPotentials(in, out.get());
    return out;
  }

  const bool vac2_progress = std::getenv("VAC2_PROGRESS") &&
                             std::atoi(std::getenv("VAC2_PROGRESS")) != 0;
  using clk = std::chrono::steady_clock;
  const auto t_solve_start = clk::now();
  auto now_ms = [&]() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() -
                                                                 t_solve_start)
        .count();
  };
  if (vac2_progress) {
    int omp_threads = 1;
#ifdef _OPENMP
    omp_threads = omp_get_max_threads();
#endif
    const char* fftw_env = std::getenv("VAC2_FFTW");
    const bool fftw_requested = fftw_env && fftw_env[0] == '1';
#ifdef VMECPP_VAC2_HAVE_FFTW3
    const char* fftw_state = fftw_requested ? "ON" : "off";
#else
    const char* fftw_state =
        fftw_requested ? "requested-but-NOT-COMPILED" : "off";
#endif
    std::fprintf(stderr,
                 "[Vac2] Solve start: mpol=%d ntor=%d nu=%d nv=%d nfp=%d "
                 "(nuv=%d, mnpot=%d, nd=%d)  OMP=%d  FFTW=%s\n",
                 mpol_, ntor_, nu_, nv_, nfp_, nuv, mnpot, nd, omp_threads,
                 fftw_state);
  }

  // =========================================================================
  // Stage 1+2 (matrix-free): fused port of matrix.f90 + fourier.f90 first pass.
  //
  // The Fortran builds the full `ga, gb, gc, gd` arrays (each nuv × nuv) and
  // then FFTs each column. At (nu=512, nv=256) this would be 4·131072² · 8 B ≈
  // 550 GB per matrix — infeasible. Instead, we stream per column (c):
  //
  //   1. Assemble `ga_col[j], gc_col[j], gd_col[j]` for j = 0..nuv-1 on the
  //      fly (all contributions: intra-period 1/|X_j − X_c|, k≥2 period
  //      rotations, tangent-plane log-singular subtraction).
  //   2. 2D r2c FFT of the three nuv-length rows over the (nu, nv) grid.
  //   3. Extract the Fourier coefficients (ac[c,k], as[c,k], bc[c,k], bs[c,k],
  //      dc[c,k], ds[c,k]) and discard the FFT output.
  //
  // Storage budget is now O(nuv · mnpot) for the six output arrays plus per-
  // thread scratch (three nuv vectors + one FFTW complex buffer).
  //
  // Correctness anchor: the kernel matrices `ga, gc, gd` produced by Fortran's
  // matrix.f90 are algebraically equal to what this matrix-free version
  // produces column-by-column. Details:
  //   - `ga` and `gd` are symmetric: the per-column formula
  //       ga_col_c[j] = (xu[j]·xu[c]) * inv_r_jc                    (k=1, j≠c)
  //                    + Σ_{k≥2} (xu[j]·R_k·xu[c]) * inv_r_{j,R_k c}  (all j)
  //     matches what Fortran writes at (j, c) for j≥c and at (c, j) for j<c
  //     by symmetrise+rotation-set closure (see comment below).
  //   - `gc` is NOT symmetric but its per-column formula with column-rotation
  //     also collapses to the simple form
  //       gc_col_c[j] = (xu[j]·xv[c]) * inv_r_jc                    (k=1, j≠c)
  //                    + Σ_{k≥2} (xu[j]·R_k·xv[c]) * inv_r_{j,R_k c}  (all j)
  //     The reason: for the upper triangle j<c, Fortran writes gc[j,c] =
  //     gb[c,j] whose k=1 piece is (xv[c]·xu[j]) * inv_r (same as xu[j]·xv[c]
  //     by commutativity) and whose k≥2 piece (with rotation of j, not c)
  //     equals the rotation-of-c form summed over the rotation set {non-zero 2π
  //     j/nfp} because orthogonality `a·R_θ·b = R_{-θ}·a·b` and the set is
  //     closed under inversion.
  //
  // Tangent subtraction (matrix.f90:74-104): subtract guu[c]·sq, guv[c]·sq,
  // gvv[c]·sq from the three rows at j≠c where
  //   sq = 1 / sqrt(guu[c]·tu² + 2·guv[c]·tu·tv + gvv[c]·tv²),
  //   tu = tau[iu(c) − iu(j)], tv = tav[iv(c) − iv(j)].
  // =========================================================================

#ifndef VMECPP_VAC2_HAVE_FFTW3
#error "Vac2Solver requires FFTW3 (compile with VMECPP_VAC2_HAVE_FFTW3=1)."
#endif

  // ga_row[j]/gc_row[j]/gd_row[j] precomputed tangent subtraction and FFTW r2c.
  // Set up mode extraction tables.
  const bool no_sub = []() {
    const char* p = std::getenv("VAC2_NOSUB");
    return p != nullptr && p[0] == '1';
  }();
  std::vector<double> tau, tav;
  BuildTauTable(nu_, &tau);
  BuildTauTable(nv_, &tav);

  // Output mode-space arrays (nuv × mnpot each).
  const std::size_t nuv_mnpot =
      static_cast<std::size_t>(nuv) * static_cast<std::size_t>(mnpot);
  std::vector<double> ac(nuv_mnpot, 0.0);
  std::vector<double> as(nuv_mnpot, 0.0);
  std::vector<double> bc(nuv_mnpot, 0.0);
  std::vector<double> bs(nuv_mnpot, 0.0);
  std::vector<double> dc(nuv_mnpot, 0.0);
  std::vector<double> ds(nuv_mnpot, 0.0);

  // Precompute the basis functions cos_k(j), sin_k(j) on the full grid:
  // basis_c[j * mnpot + k] = cos(2pi*(m_k * u_j + n_k * v_j)).
  // Still needed downstream by stage 2.5 (analyt add-back), stage 5 (bexn),
  // stage 8 (potU/potV reconstruction). O(nuv · mnpot) storage — small.
  cached_basis_c_.assign(nuv_mnpot, 0.0);
  cached_basis_s_.assign(nuv_mnpot, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int j = 0; j < nuv; ++j) {
    const int ku = Iu(j, nv_);
    const int kv = Iv(j, nv_);
    for (int k = 0; k < mnpot; ++k) {
      const int m = ma_[k];
      const int n = na_[k];
      const double cum = conu_[static_cast<std::size_t>(ku) * mpol_ + m];
      const double sum_ = sinu_[static_cast<std::size_t>(ku) * mpol_ + m];
      const double cvn = ConV(conv_, ntor_, kv, n);
      const double svn = SinV(sinv_, ntor_, kv, n);
      cached_basis_c_[static_cast<std::size_t>(j) * mnpot + k] =
          cum * cvn - sum_ * svn;
      cached_basis_s_[static_cast<std::size_t>(j) * mnpot + k] =
          sum_ * cvn + cum * svn;
    }
  }

  // Precompute period-rotation tables cs_per[k], sn_per[k] for k=1..nfp-1.
  std::vector<double> cs_per(nfp_, 0.0), sn_per(nfp_, 0.0);
  const double alp = kTwoPi / static_cast<double>(nfp_);
  for (int kp = 0; kp < nfp_; ++kp) {
    cs_per[kp] = std::cos(alp * kp);
    sn_per[kp] = std::sin(alp * kp);
  }

  // Build FFTW plan (shared; thread-safe via fftw_execute_dft_r2c).
  const int nu_f = nu_;
  const int nv_f = nv_;
  const int nvh = nv_f / 2 + 1;
  std::vector<double> scratch_in(static_cast<std::size_t>(nuv), 0.0);
  std::vector<fftw_complex> scratch_out(static_cast<std::size_t>(nu_f) * nvh);
  fftw_plan plan_fwd = fftw_plan_dft_r2c_2d(nu_f, nv_f, scratch_in.data(),
                                            scratch_out.data(), FFTW_ESTIMATE);
  if (!plan_fwd) {
    return absl::InternalError("FFTW plan_dft_r2c_2d failed (stage 1+2)");
  }

  // Stellarator-symmetry column fold: for lasym = false the kernel rows of
  // the mirrored column c' (theta -> 2 pi - theta, phi -> -phi) are the
  // mirrored rows of column c (the reflection is an isometry, so all three
  // kernel dot products are invariant), and mirroring the input of a real
  // 2D FFT conjugates its output. The mirrored column's mode arrays are
  // therefore sign-flipped copies of the canonical column's: ac/bc/dc copy,
  // as/bs/ds negate. This halves the dominant kernel-assembly work.
  // Disable with VAC2_NO_SYMM_FOLD=1 for A/B validation.
  const bool symm_fold = !in.lasym && []() {
    const char* p = std::getenv("VAC2_NO_SYMM_FOLD");
    return p == nullptr || p[0] != '1';
  }();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // Per-thread scratch: three nuv-length rows + three complex FFT buffers.
    std::vector<double> ga_row(nuv);
    std::vector<double> gc_row(nuv);
    std::vector<double> gd_row(nuv);
    std::vector<fftw_complex> out_a(static_cast<std::size_t>(nu_f) * nvh);
    std::vector<fftw_complex> out_c(static_cast<std::size_t>(nu_f) * nvh);
    std::vector<fftw_complex> out_d(static_cast<std::size_t>(nu_f) * nvh);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, 1)
#endif
    for (int c = 0; c < nuv; ++c) {
      // mirrored partner column (see symm_fold above)
      const int c_mirror =
          ((nu_ - Iu(c, nv_)) % nu_) * nv_ + ((nv_ - Iv(c, nv_)) % nv_);
      if (symm_fold && c_mirror < c) {
        continue;  // filled from its canonical partner below
      }
      const double xc = in.x[c];
      const double yc = in.y[c];
      const double zc = in.z[c];
      const double xuc = in.xu[c];
      const double yuc = in.yu[c];
      const double zuc = in.zu[c];
      const double xvc = in.xv[c];
      const double yvc = in.yv[c];
      const double zvc = in.zv[c];

      // --- Intra-period (k=1): `r_kernel[j] = 1/|X_j - X_c|` for j != c,
      // with diagonal j==c cleared to zero (no self-interaction).
      for (int j = 0; j < nuv; ++j) {
        if (j == c) {
          ga_row[j] = 0.0;
          gc_row[j] = 0.0;
          gd_row[j] = 0.0;
          continue;
        }
        const double dx = in.x[j] - xc;
        const double dy = in.y[j] - yc;
        const double dz = in.z[j] - zc;
        const double inv_r = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
        const double xuj = in.xu[j];
        const double yuj = in.yu[j];
        const double zuj = in.zu[j];
        const double xvj = in.xv[j];
        const double yvj = in.yv[j];
        const double zvj = in.zv[j];
        ga_row[j] = (xuj * xuc + yuj * yuc + zuj * zuc) * inv_r;
        gc_row[j] = (xuj * xvc + yuj * yvc + zuj * zvc) * inv_r;
        gd_row[j] = (xvj * xvc + yvj * yvc + zvj * zvc) * inv_r;
      }

      // --- Period contributions k=2..nfp (rotate column c by angle
      // 2π·kp/nfp for kp=1..nfp-1). Applied to ALL j (including diagonal).
      for (int kp = 1; kp < nfp_; ++kp) {
        const double cs = cs_per[kp];
        const double sn = sn_per[kp];
        const double xcR = cs * xc - sn * yc;
        const double ycR = sn * xc + cs * yc;
        const double zcR = zc;
        const double xucR = cs * xuc - sn * yuc;
        const double yucR = sn * xuc + cs * yuc;
        const double zucR = zuc;
        const double xvcR = cs * xvc - sn * yvc;
        const double yvcR = sn * xvc + cs * yvc;
        const double zvcR = zvc;
        for (int j = 0; j < nuv; ++j) {
          const double dx = in.x[j] - xcR;
          const double dy = in.y[j] - ycR;
          const double dz = in.z[j] - zcR;
          const double inv_r = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
          const double xuj = in.xu[j];
          const double yuj = in.yu[j];
          const double zuj = in.zu[j];
          const double xvj = in.xv[j];
          const double yvj = in.yv[j];
          const double zvj = in.zv[j];
          ga_row[j] += (xuj * xucR + yuj * yucR + zuj * zucR) * inv_r;
          gc_row[j] += (xuj * xvcR + yuj * yvcR + zuj * zvcR) * inv_r;
          gd_row[j] += (xvj * xvcR + yvj * yvcR + zvj * zvcR) * inv_r;
        }
      }

      // --- Tangent-plane log-singular subtraction at column c (j != c).
      if (!no_sub) {
        const double guu_c = in.guu[c];
        const double guv_c = in.guv[c];
        const double gvv_c = in.gvv[c];
        const int iu_c = Iu(c, nv_);
        const int iv_c = Iv(c, nv_);
        for (int j = 0; j < nuv; ++j) {
          if (j == c) continue;
          const int iu_j = Iu(j, nv_);
          const int iv_j = Iv(j, nv_);
          const double tu = tau[(iu_c - iu_j) + (nu_ - 1)];
          const double tv = tav[(iv_c - iv_j) + (nv_ - 1)];
          const double denom =
              guu_c * tu * tu + 2.0 * guv_c * tu * tv + gvv_c * tv * tv;
          const double sq = 1.0 / std::sqrt(denom);
          ga_row[j] -= guu_c * sq;
          gc_row[j] -= guv_c * sq;
          gd_row[j] -= gvv_c * sq;
        }
      }

      // --- 2D r2c FFT of the three rows over (nu, nv).
      fftw_execute_dft_r2c(plan_fwd, ga_row.data(), out_a.data());
      fftw_execute_dft_r2c(plan_fwd, gc_row.data(), out_c.data());
      fftw_execute_dft_r2c(plan_fwd, gd_row.data(), out_d.data());

      // --- Extract the mnpot modes (same (m, n) → FFT index mapping used by
      // the original Vac2 FFTW path — verified bit-for-bit against direct-sum
      // DFT at correctness-check resolution).
      for (int k = 0; k < mnpot; ++k) {
        const int m = ma_[k];
        const int n = na_[k];
        double re_a, im_a, re_c, im_c, re_d, im_d;
        if (n >= 0) {
          const std::size_t p = static_cast<std::size_t>(m) * nvh + n;
          re_a = out_a[p][0];
          im_a = out_a[p][1];
          re_c = out_c[p][0];
          im_c = out_c[p][1];
          re_d = out_d[p][0];
          im_d = out_d[p][1];
        } else {
          const int abs_n = -n;
          const int m_conj = (nu_f - m) % nu_f;
          const std::size_t p = static_cast<std::size_t>(m_conj) * nvh + abs_n;
          re_a = out_a[p][0];
          im_a = -out_a[p][1];
          re_c = out_c[p][0];
          im_c = -out_c[p][1];
          re_d = out_d[p][0];
          im_d = -out_d[p][1];
        }
        const std::size_t idx = static_cast<std::size_t>(c) * mnpot + k;
        ac[idx] = fnuv * re_a;
        as[idx] = -fnuv * im_a;
        bc[idx] = fnuv * re_c;
        bs[idx] = -fnuv * im_c;
        dc[idx] = fnuv * re_d;
        ds[idx] = -fnuv * im_d;
        if (symm_fold && c_mirror != c) {
          // mirrored column: conjugated FFT -> cos parts copy, sin parts flip
          const std::size_t idx_m =
              static_cast<std::size_t>(c_mirror) * mnpot + k;
          ac[idx_m] = fnuv * re_a;
          as[idx_m] = fnuv * im_a;
          bc[idx_m] = fnuv * re_c;
          bs[idx_m] = fnuv * im_c;
          dc[idx_m] = fnuv * re_d;
          ds[idx_m] = fnuv * im_d;
        }
      }
    }
  }
  fftw_destroy_plan(plan_fwd);

  if (vac2_progress)
    std::fprintf(
        stderr,
        "[Vac2] %6lld ms — stage 1+2 (matrix-free): nuv=%d × mnpot=%d\n",
        now_ms(), nuv, mnpot);

  // =========================================================================
  // Stage 2.5: port of analyt.f90 — add back the analytic Fourier
  // coefficients of the log-singular tangent-plane kernel subtracted in
  // matrix.f90. Without this step, g ends up strongly non-positive-definite
  // in the 3D path (nv>1 or ntor>0) because the tangent-plane subtraction
  // over-regularises the kernel. The add-back compensates exactly.
  //
  // This step can be skipped via VAC2_NOANALYT=1 for diagnostic purposes.
  // =========================================================================
  const bool no_analyt = []() {
    const char* p = std::getenv("VAC2_NOANALYT");
    return p != nullptr && p[0] == '1';
  }();
  // VAC2_UNSTABLE=1: use forward-only (unstable) T^- recurrence in Rinteg
  // instead of the default backward (stable) recurrence. This serves as a
  // reference for debugging the Vac1 stable recurrence.
  const bool vac2_unstable = []() {
    const char* p = std::getenv("VAC2_UNSTABLE");
    return p != nullptr && p[0] == '1';
  }();
  if (!no_sub && !no_analyt) {
    if (vac2_unstable && vac2_progress) {
      std::fprintf(stderr,
                   "[Vac2] VAC2_UNSTABLE=1: using forward (unstable) T^- "
                   "recurrence in Rinteg\n");
    }
    // mnax per vacuum_me.f90:41: max(mpot, npot). In C++ convention,
    // the highest m in the mode table is (mpol_-1) and the highest |n| is
    // ntor_. So mnax = max(mpol_-1, ntor_). Use at least 1 to keep the
    // arrays well-formed.
    int mnax = std::max(mpol_ - 1, ntor_);
    if (mnax < 1) mnax = 1;
    std::vector<double> f_an;
    Analin(mnax, nuv, in.guu, in.guv, in.gvv, in.guu, in.guv, in.gvv, &f_an,
           /*stable=*/!vac2_unstable);
    const std::size_t fan_stride_m = static_cast<std::size_t>(2 * mnax + 1);
    const std::size_t fan_stride_i =
        fan_stride_m * static_cast<std::size_t>(mnax + 1);
    auto FAN = [&](int i, int m, int n) {
      return f_an[static_cast<std::size_t>(i) * fan_stride_i +
                  static_cast<std::size_t>(m) * fan_stride_m +
                  static_cast<std::size_t>(n + mnax)];
    };
    // Parallelise over `i` (spatial axis) rather than `kk` since mnpot
    // can be small and the (i, kk) matrix cells are disjoint across threads
    // when we pick `i` as the outer parallel axis.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < nuv; ++i) {
      const int ku = Iu(i, nv_);
      const int kv = Iv(i, nv_);
      const double guu_i = in.guu[i];
      const double guv_i = in.guv[i];
      const double gvv_i = in.gvv[i];
      for (int kk = 0; kk < mnpot; ++kk) {
        const int m = ma_[kk];
        const int n = na_[kk];
        const double cum = conu_[static_cast<std::size_t>(ku) * mpol_ + m];
        const double sum_ = sinu_[static_cast<std::size_t>(ku) * mpol_ + m];
        const double cvn = ConV(conv_, ntor_, kv, n);
        const double svn = SinV(sinv_, ntor_, kv, n);
        const double f = FAN(i, m, n);
        const double co = f * (cum * cvn - sum_ * svn);
        const double si = f * (sum_ * cvn + cum * svn);
        const std::size_t idx = static_cast<std::size_t>(i) * mnpot + kk;
        ac[idx] += co * guu_i;
        as[idx] += si * guu_i;
        bc[idx] += co * guv_i;
        bs[idx] += si * guv_i;
        dc[idx] += co * gvv_i;
        ds[idx] += si * gvv_i;
      }
    }
  }
  if (vac2_progress)
    std::fprintf(stderr, "[Vac2] %6lld ms — stage 2.5: analyt add-back\n",
                 now_ms());

  // =========================================================================
  // Stage 3: port of foumat.f90 second pass. Reduce ac/as/bc/bs/dc/ds along
  // the remaining spatial index to produce the 12 mode-space matrices
  //   acc, asc (from ac), acs, ass (from as),
  //   bcc, bsc (from bc), bcs, bss (from bs),
  //   dcc, dsc (from dc), dcs, dss (from ds).
  // Each is (mnpot x mnpot) with row index = outer (first-pass) mode,
  // column index = inner (second-pass) mode.
  //   acc(k_o, k_i) = fnuv * sum_i ac(i, k_o) * cos(2pi*(m_{k_i}*u_i +
  //   n_{k_i}*v_i)) asc(k_o, k_i) = fnuv * sum_i ac(i, k_o) * sin(...)
  // =========================================================================
  const std::size_t mnpot_sq = static_cast<std::size_t>(mnpot) * mnpot;
  std::vector<double> acc_m(mnpot_sq, 0.0);
  std::vector<double> asc_m(mnpot_sq, 0.0);
  std::vector<double> acs_m(mnpot_sq, 0.0);
  std::vector<double> ass_m(mnpot_sq, 0.0);
  std::vector<double> bcc_m(mnpot_sq, 0.0);
  std::vector<double> bsc_m(mnpot_sq, 0.0);
  std::vector<double> bcs_m(mnpot_sq, 0.0);
  std::vector<double> bss_m(mnpot_sq, 0.0);
  std::vector<double> dcc_m(mnpot_sq, 0.0);
  std::vector<double> dsc_m(mnpot_sq, 0.0);
  std::vector<double> dcs_m(mnpot_sq, 0.0);
  std::vector<double> dss_m(mnpot_sq, 0.0);

  // FFTW r2c for each outer mode ko. Each ko transforms 6 spatial arrays
  // (ac, as, bc, bs, dc, ds) at column ko into (mnpot) inner-mode pairs.
  // nu_f, nv_f, nvh already defined above in stage 1+2.
  {
    std::vector<double> scratch_in2(static_cast<std::size_t>(nuv), 0.0);
    std::vector<fftw_complex> scratch_out2(static_cast<std::size_t>(nu_f) *
                                           nvh);
    fftw_plan plan_fwd2 = fftw_plan_dft_r2c_2d(
        nu_f, nv_f, scratch_in2.data(), scratch_out2.data(), FFTW_ESTIMATE);
    if (!plan_fwd2) {
      return absl::InternalError("FFTW plan_dft_r2c_2d failed (stage 3)");
    }

    // Parallel over ko: each ko writes to its own row of the 12 output
    // matrices and uses thread-local input and output buffers.
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      std::vector<double> in_ac(nuv);
      std::vector<double> in_as(nuv);
      std::vector<double> in_bc(nuv);
      std::vector<double> in_bs(nuv);
      std::vector<double> in_dc(nuv);
      std::vector<double> in_ds(nuv);
      std::vector<fftw_complex> out_ac(static_cast<std::size_t>(nu_f) * nvh);
      std::vector<fftw_complex> out_as(static_cast<std::size_t>(nu_f) * nvh);
      std::vector<fftw_complex> out_bc(static_cast<std::size_t>(nu_f) * nvh);
      std::vector<fftw_complex> out_bs(static_cast<std::size_t>(nu_f) * nvh);
      std::vector<fftw_complex> out_dc(static_cast<std::size_t>(nu_f) * nvh);
      std::vector<fftw_complex> out_ds(static_cast<std::size_t>(nu_f) * nvh);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (int ko = 0; ko < mnpot; ++ko) {
        for (int i = 0; i < nuv; ++i) {
          const std::size_t iko = static_cast<std::size_t>(i) * mnpot + ko;
          in_ac[i] = ac[iko];
          in_as[i] = as[iko];
          in_bc[i] = bc[iko];
          in_bs[i] = bs[iko];
          in_dc[i] = dc[iko];
          in_ds[i] = ds[iko];
        }
        fftw_execute_dft_r2c(plan_fwd2, in_ac.data(), out_ac.data());
        fftw_execute_dft_r2c(plan_fwd2, in_as.data(), out_as.data());
        fftw_execute_dft_r2c(plan_fwd2, in_bc.data(), out_bc.data());
        fftw_execute_dft_r2c(plan_fwd2, in_bs.data(), out_bs.data());
        fftw_execute_dft_r2c(plan_fwd2, in_dc.data(), out_dc.data());
        fftw_execute_dft_r2c(plan_fwd2, in_ds.data(), out_ds.data());

        // For inner mode ki=(m, n):
        //   s_acc = sum_i ac[i, ko] * cos(2π(m*u + n*v))  = Re(fft_ac[m, n])
        //   s_asc = sum_i ac[i, ko] * sin(2π(m*u + n*v))  = -Im(fft_ac[m, n])
        // And s_cos[*] = Re(fft_*), s_sin[*] = -Im(fft_*).
        for (int ki = 0; ki < mnpot; ++ki) {
          const int m = ma_[ki];
          const int n = na_[ki];
          double re_ac, im_ac, re_as, im_as;
          double re_bc, im_bc, re_bs, im_bs;
          double re_dc, im_dc, re_ds, im_ds;
          if (n >= 0) {
            const std::size_t p = static_cast<std::size_t>(m) * nvh + n;
            re_ac = out_ac[p][0];
            im_ac = out_ac[p][1];
            re_as = out_as[p][0];
            im_as = out_as[p][1];
            re_bc = out_bc[p][0];
            im_bc = out_bc[p][1];
            re_bs = out_bs[p][0];
            im_bs = out_bs[p][1];
            re_dc = out_dc[p][0];
            im_dc = out_dc[p][1];
            re_ds = out_ds[p][0];
            im_ds = out_ds[p][1];
          } else {
            const int abs_n = -n;
            const int m_conj = (nu_f - m) % nu_f;
            const std::size_t p =
                static_cast<std::size_t>(m_conj) * nvh + abs_n;
            re_ac = out_ac[p][0];
            im_ac = -out_ac[p][1];
            re_as = out_as[p][0];
            im_as = -out_as[p][1];
            re_bc = out_bc[p][0];
            im_bc = -out_bc[p][1];
            re_bs = out_bs[p][0];
            im_bs = -out_bs[p][1];
            re_dc = out_dc[p][0];
            im_dc = -out_dc[p][1];
            re_ds = out_ds[p][0];
            im_ds = -out_ds[p][1];
          }
          const std::size_t idx = static_cast<std::size_t>(ko) * mnpot + ki;
          acc_m[idx] = fnuv * re_ac;
          asc_m[idx] = fnuv * -im_ac;
          acs_m[idx] = fnuv * re_as;
          ass_m[idx] = fnuv * -im_as;
          bcc_m[idx] = fnuv * re_bc;
          bsc_m[idx] = fnuv * -im_bc;
          bcs_m[idx] = fnuv * re_bs;
          bss_m[idx] = fnuv * -im_bs;
          dcc_m[idx] = fnuv * re_dc;
          dsc_m[idx] = fnuv * -im_dc;
          dcs_m[idx] = fnuv * re_ds;
          dss_m[idx] = fnuv * -im_ds;
        }
      }
    }
    fftw_destroy_plan(plan_fwd2);
  }

  // Release the first-pass matrices.
  std::vector<double>().swap(ac);
  std::vector<double>().swap(as);
  std::vector<double>().swap(bc);
  std::vector<double>().swap(bs);
  std::vector<double>().swap(dc);
  std::vector<double>().swap(ds);
  if (vac2_progress)
    std::fprintf(stderr,
                 "[Vac2] %6lld ms — stage 3: 2nd-pass DFT → 12 mode matrices\n",
                 now_ms());

  // =========================================================================
  // Stage 4: port of solver.f90:37-48 -- assemble g[nd, nd] from 12 matrices.
  // Index convention: Fortran k, i in [2, mnpot] maps to C++ ko, ki in
  // [1, mnpot-1] (skipping the m=0,n=0 mode at index 0); the four blocks
  // occupy g's quadrants of size mn2+1 = mnpot-1 each.
  //
  // Let mn2 = mnpot - 2 (the Fortran offset). In C++ 0-based indexing,
  // block boundaries are:
  //   - top-left     : rows [0, mnpot-1), cols [0, mnpot-1)
  //   - bottom-right : rows [mnpot-1, 2(mnpot-1)), cols same
  //   - top-right    : rows [0, mnpot-1), cols [mnpot-1, 2(mnpot-1))
  //   - bottom-left  : rows [mnpot-1, 2(mnpot-1)), cols [0, mnpot-1)
  //
  // We iterate ko, ki in [1, mnpot); the Fortran index i corresponds to ki+1
  // (row mode) and Fortran k to ko+1 (col mode). Look at the Fortran:
  //   g(i-1,   k-1)   : row uses i (= ki+1), col uses k (= ko+1).
  // So in C++: g[ki (row), ko (col)] for the top-left block, row = ki-1 + 0,
  // col = ko-1 + 0. But it's clearer to keep the Fortran loop variables
  // `i_f = ki + 1` (row) and `k_f = ko + 1` (col) and just subtract 1 for
  // C++ indices.
  //
  // fa = pi (from solver.f90:34 comment: fa = 4 pi^2 / (4 pi) = pi).
  // =========================================================================
  out->g.assign(static_cast<std::size_t>(nd) * nd, 0.0);
  const double fa_g = kPi;  // solver.f90:34

  // Fortran loop variables (1-based): i in [2, mnpot], k in [2, mnpot]
  // (i is the row mode index, k is the col mode index).
  for (int i_f = 2; i_f <= mnpot; ++i_f) {
    // C++ 0-based mode index for the Fortran i:
    const int i_c = i_f - 1;
    const int ma_i = ma_[i_c];
    const int na_i = na_[i_c];
    for (int k_f = 2; k_f <= mnpot; ++k_f) {
      const int k_c = k_f - 1;
      const int ma_k = ma_[k_c];
      const int na_k = na_[k_c];

      // Block positions (top-left origin is (0, 0)):
      //   top-left    :  row = i_f - 2, col = k_f - 2            (in [0, mn2])
      //   bot-right   :  row = i_f - 2 + mnpot1, col = k_f - 2 + mnpot1
      //   top-right   :  row = i_f - 2, col = k_f - 2 + mnpot1
      //   bot-left    :  row = i_f - 2 + mnpot1, col = k_f - 2
      // where mnpot1 = mnpot - 1.
      const int mnpot1 = mnpot - 1;
      const int row_top = i_f - 2;
      const int col_top = k_f - 2;
      const int row_bot = row_top + mnpot1;
      const int col_bot = col_top + mnpot1;

      // Fortran uses 1-based indexing on 12 matrices, so Fortran (k, i) maps
      // to C++ (k_f - 1, i_f - 1) = (k_c, i_c). Similarly (i, k) -> (i_c, k_c).
      auto M = [&](const std::vector<double>& Mat, int a, int b) {
        return Mat[static_cast<std::size_t>(a) * mnpot + b];
      };

      const double g_tl =
          fa_g *
          (na_i * na_k * M(acc_m, k_c, i_c) + ma_i * ma_k * M(dcc_m, k_c, i_c) -
           na_i * ma_k * M(bcc_m, i_c, k_c) - ma_i * na_k * M(bcc_m, k_c, i_c));
      const double g_br =
          fa_g *
          (na_i * na_k * M(ass_m, k_c, i_c) + ma_i * ma_k * M(dss_m, k_c, i_c) -
           na_i * ma_k * M(bss_m, i_c, k_c) - ma_i * na_k * M(bss_m, k_c, i_c));
      const double g_tr = fa_g * (-na_i * na_k * M(acs_m, k_c, i_c) -
                                  ma_i * ma_k * M(dcs_m, k_c, i_c) +
                                  na_i * ma_k * M(bsc_m, i_c, k_c) +
                                  ma_i * na_k * M(bcs_m, k_c, i_c));
      const double g_bl = fa_g * (-na_i * na_k * M(asc_m, k_c, i_c) -
                                  ma_i * ma_k * M(dsc_m, k_c, i_c) +
                                  na_i * ma_k * M(bcs_m, i_c, k_c) +
                                  ma_i * na_k * M(bsc_m, k_c, i_c));

      out->g[static_cast<std::size_t>(row_top) * nd + col_top] = g_tl;
      out->g[static_cast<std::size_t>(row_bot) * nd + col_bot] = g_br;
      out->g[static_cast<std::size_t>(row_top) * nd + col_bot] = g_tr;
      out->g[static_cast<std::size_t>(row_bot) * nd + col_top] = g_bl;
    }
  }

  // =========================================================================
  // Stage 6: cpol and ctor from the (k=1) row of the mode matrices
  // (solver.f90:55-60). Note: Fortran uses (1, i) and (i, 1), i.e. the
  // m=0,n=0 row / column, which in C++ is index 0.
  //   cpol(i-1       ) = -0.5 * (na(i) * acc(1,i) - ma(i) * bcc(1,i))
  //   cpol(i-1+mnpot1) = +0.5 * (na(i) * asc(1,i) - ma(i) * bsc(1,i))
  //   ctor(i-1       ) = -0.5 * (na(i) * bcc(i,1) - ma(i) * dcc(i,1))
  //   ctor(i-1+mnpot1) = +0.5 * (na(i) * bcs(i,1) - ma(i) * dcs(i,1))
  // fa = 0.5 (= 2*pi / (4*pi)).
  // =========================================================================
  out->cpol.assign(nd, 0.0);
  out->ctor.assign(nd, 0.0);
  const double fa_c = 0.5;
  const int mnpot1 = mnpot - 1;
  for (int i_f = 2; i_f <= mnpot; ++i_f) {
    const int i_c = i_f - 1;
    const int ma_i = ma_[i_c];
    const int na_i = na_[i_c];
    auto M = [&](const std::vector<double>& Mat, int a, int b) {
      return Mat[static_cast<std::size_t>(a) * mnpot + b];
    };
    out->cpol[i_f - 2] =
        -fa_c * (na_i * M(acc_m, 0, i_c) - ma_i * M(bcc_m, 0, i_c));
    out->cpol[i_f - 2 + mnpot1] =
        +fa_c * (na_i * M(asc_m, 0, i_c) - ma_i * M(bsc_m, 0, i_c));
    out->ctor[i_f - 2] =
        -fa_c * (na_i * M(bcc_m, i_c, 0) - ma_i * M(dcc_m, i_c, 0));
    out->ctor[i_f - 2 + mnpot1] =
        +fa_c * (na_i * M(bcs_m, i_c, 0) - ma_i * M(dcs_m, i_c, 0));
  }

  // =========================================================================
  // Stage 5: benf -- Fourier projection of bexn (port of bexmat.f90).
  //   bnc[k] = fnuv * sum_i bexn[i] * cos(2*pi*(m_k*u_i + n_k*v_i))
  //   bns[k] = fnuv * sum_i bexn[i] * sin(2*pi*(m_k*u_i + n_k*v_i))
  //   benf[i-1]          = bns[i]            (first half: sin of bexn)
  //   benf[mnpot-2+i]    = bnc[i]            (second half: cos of bexn)
  // so the first half of benf corresponds to the same "sin-of-phi" block
  // as the first half of hs.
  // =========================================================================
  ComputeBenf(in, out.get());

  // =========================================================================
  // Diagnostic dump (VAC2_DIAG=1): mode tables, diagonal, and upper-left
  // submatrix of g pre-symmetrise, to understand PD failures.
  // =========================================================================
  const bool diag = []() {
    const char* d = std::getenv("VAC2_DIAG");
    return d != nullptr && d[0] == '1';
  }();
  if (diag) {
    std::fprintf(
        stderr,
        "[VAC2_DIAG] mpol=%d ntor=%d nu=%d nv=%d nfp=%d mnpot=%d nd=%d\n",
        mpol_, ntor_, nu_, nv_, nfp_, mnpot_, nd);
    std::fprintf(stderr, "[VAC2_DIAG] mode table (k: m, n):\n");
    for (int k = 0; k < mnpot_; ++k) {
      std::fprintf(stderr, "  k=%d: m=%d n=%d\n", k, ma_[k], na_[k]);
    }
    std::fprintf(stderr, "[VAC2_DIAG] g diagonal (pre-symm):\n");
    for (int i = 0; i < nd; ++i) {
      std::fprintf(stderr, "  g[%d,%d] = %.6e\n", i, i,
                   out->g[static_cast<std::size_t>(i) * nd + i]);
    }
    const int dim = std::min(nd, 6);
    std::fprintf(stderr, "[VAC2_DIAG] g(0..%d, 0..%d) pre-symm:\n", dim - 1,
                 dim - 1);
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        std::fprintf(stderr, " %10.3e",
                     out->g[static_cast<std::size_t>(i) * nd + j]);
      }
      std::fprintf(stderr, "\n");
    }
  }

  // =========================================================================
  // Symmetrise the g-matrix (solver.f90:69-73):
  //   g(i, k) = 0.5 * (g(i, k) + g(k, i))  for i < k
  // Then fill the lower triangle from the upper triangle so g is exactly
  // symmetric. (The lower triangle could also be kept and averaged; Fortran
  // averages the upper triangle in place and then LAPACK dpotrf reads UPLO='U',
  // so the lower triangle contents are irrelevant to the factorisation.)
  // =========================================================================
  for (int k = 1; k < nd; ++k) {
    for (int i = 0; i < k; ++i) {
      const double avg = 0.5 * (out->g[static_cast<std::size_t>(i) * nd + k] +
                                out->g[static_cast<std::size_t>(k) * nd + i]);
      out->g[static_cast<std::size_t>(i) * nd + k] = avg;
      out->g[static_cast<std::size_t>(k) * nd + i] = avg;
    }
  }
  if (diag) {
    const int dim = std::min(nd, 6);
    std::fprintf(stderr, "[VAC2_DIAG] g(0..%d, 0..%d) post-symm:\n", dim - 1,
                 dim - 1);
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        std::fprintf(stderr, " %10.3e",
                     out->g[static_cast<std::size_t>(i) * nd + j]);
      }
      std::fprintf(stderr, "\n");
    }
  }

  // =========================================================================
  // Stage 7: solve. LAPACK dpotrf (Cholesky 'U') and dpotrs.
  //
  // For lasym = false (default, stellarator-symmetric):
  //   - Zero the cs/sc/ss blocks of g and the cosine half of benf; only the
  //     top-left cc block (size mnpot1 x mnpot1) and the sine half of benf
  //     carry data.
  //   - Factor the cc block via a single Cholesky of size mnpot1.
  //   - Solve for hs's first half; hs's second half is set to zero.
  // For lasym = true, factor the full nd x nd matrix.
  // =========================================================================
  const char uplo = 'U';
  const int nrhs = 1;
  int info = 0;

  if (!in.lasym) {
    // Zero the cs (top-right), sc (bottom-left), and ss (bottom-right) blocks.
    // (These blocks should be structurally zero under stellarator symmetry
    // anyway; we zero them to avoid any numerical pollution from apparent-
    // zero entries.)
    for (int r = 0; r < mnpot1; ++r) {
      // top-right row r: cols [mnpot1, nd)
      for (int c = mnpot1; c < nd; ++c) {
        out->g[static_cast<std::size_t>(r) * nd + c] = 0.0;
      }
    }
    for (int r = mnpot1; r < nd; ++r) {
      // bottom half: cols [0, nd)
      for (int c = 0; c < nd; ++c) {
        out->g[static_cast<std::size_t>(r) * nd + c] = 0.0;
      }
    }
    // Also zero the cosine (second) half of benf.
    for (int k = mnpot1; k < nd; ++k) {
      out->benf[k] = 0.0;
    }

    // Extract the cc block (size mnpot1 x mnpot1) into a standalone matrix
    // for factorisation.
    std::vector<double> cc(static_cast<std::size_t>(mnpot1) * mnpot1, 0.0);
    for (int r = 0; r < mnpot1; ++r) {
      for (int c = 0; c < mnpot1; ++c) {
        cc[static_cast<std::size_t>(r) * mnpot1 + c] =
            out->g[static_cast<std::size_t>(r) * nd + c];
      }
    }

    // Build rhs on the cc block: rhs = cpol*curpol + ctor*curtor - benf.
    std::vector<double> rhs_cc(mnpot1, 0.0);
    for (int k = 0; k < mnpot1; ++k) {
      rhs_cc[k] =
          out->cpol[k] * in.curpol + out->ctor[k] * in.curtor - out->benf[k];
    }

    // Cholesky factor + solve.
    dpotrf_(&uplo, &mnpot1, cc.data(), &mnpot1, &info);
    if (info != 0) {
      return absl::InternalError(
          absl::StrCat("dpotrf(cc) failed with info=", info));
    }

    // Cache the frozen operator for reuse_operator solves.
    cached_cc_factor_ = cc;
    cached_cpol_ = out->cpol;
    cached_ctor_ = out->ctor;
    has_cached_operator_ = true;

    std::vector<double> hs_cc = rhs_cc;
    dpotrs_(&uplo, &mnpot1, &nrhs, cc.data(), &mnpot1, hs_cc.data(), &mnpot1,
            &info);
    if (info != 0) {
      return absl::InternalError(
          absl::StrCat("dpotrs(cc) failed with info=", info));
    }

    // Pack hs: first half from cc solve, second half zero.
    out->hs.assign(nd, 0.0);
    for (int k = 0; k < mnpot1; ++k) out->hs[k] = hs_cc[k];
  } else {
    // Full asymmetric path: factor the full nd x nd g.
    std::vector<double> g_fact = out->g;  // dpotrf overwrites in place

    // Build full rhs.
    std::vector<double> rhs(nd, 0.0);
    for (int k = 0; k < nd; ++k) {
      rhs[k] =
          out->cpol[k] * in.curpol + out->ctor[k] * in.curtor - out->benf[k];
    }

    dpotrf_(&uplo, &nd, g_fact.data(), &nd, &info);
    if (info != 0) {
      return absl::InternalError(
          absl::StrCat("dpotrf failed with info=", info));
    }
    std::vector<double> hs = rhs;
    dpotrs_(&uplo, &nd, &nrhs, g_fact.data(), &nd, hs.data(), &nd, &info);
    if (info != 0) {
      return absl::InternalError(
          absl::StrCat("dpotrs failed with info=", info));
    }
    out->hs = std::move(hs);
  }
  if (vac2_progress)
    std::fprintf(stderr,
                 "[Vac2] %6lld ms — stage 4-7: g-assembly + Cholesky (nd=%d)\n",
                 now_ms(), nd);

  // =========================================================================
  // Stage 8: reconstruct potU, potV (port of solver.f90:87-96).
  // The base secular term is (-curtor, +curpol).
  //   potU(i) = -curtor + 2*pi * sum_{j=1..nd/2} m_{j+1} *
  //             (hs[j-1] * cos(2pi*(m u + n v))
  //              - hs[j-1 + nd/2] * sin(2pi*(m u + n v)))
  //   potV(i) = +curpol + 2*pi * sum_{j=1..nd/2} n_{j+1} *
  //             (hs[j-1] * cos(...) - hs[j-1 + nd/2] * sin(...))
  // In C++ indexing: for j0 in [0, nd/2), mode index i_mode = j0 + 1 (since
  // we skip the m=0,n=0 mode). hs[j0] is sin-of-phi amplitude, hs[j0 + nd/2]
  // is cos-of-phi amplitude.
  // =========================================================================
  ReconstructPotentials(in, out.get());
  if (vac2_progress)
    std::fprintf(
        stderr,
        "[Vac2] %6lld ms — stage 8-9: potU/potV + bsqvac. total=%lld ms\n",
        now_ms(), now_ms());

  return out;
}

void Vac2Solver::ComputeBenf(const Input& in, Output* out) const {
  const int nuv = nu_ * nv_;
  const int mnpot = mnpot_;
  const int mnpot1 = mnpot - 1;
  const int nd = nd_;
  const double fnuv = 1.0 / static_cast<double>(nuv);

  // benf -- Fourier projection of bexn (port of bexmat.f90).
  //   bnc[k] = fnuv * sum_i bexn[i] * cos(2*pi*(m_k*u_i + n_k*v_i))
  //   bns[k] = fnuv * sum_i bexn[i] * sin(2*pi*(m_k*u_i + n_k*v_i))
  //   benf[i-1]          = bns[i]            (first half: sin of bexn)
  //   benf[mnpot-2+i]    = bnc[i]            (second half: cos of bexn)
  out->benf.assign(nd, 0.0);
  std::vector<double> bnc(mnpot, 0.0);
  std::vector<double> bns(mnpot, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int k = 0; k < mnpot; ++k) {
    double s_c = 0.0, s_s = 0.0;
    for (int i = 0; i < nuv; ++i) {
      const double b = in.bexn[i];
      s_c += b * cached_basis_c_[static_cast<std::size_t>(i) * mnpot + k];
      s_s += b * cached_basis_s_[static_cast<std::size_t>(i) * mnpot + k];
    }
    bnc[k] = fnuv * s_c;
    bns[k] = fnuv * s_s;
  }
  // Fortran vac2 reference stores benf's sine block with a net sign flip
  // relative to the "natural" bns = sum bexn * sin(m u + n v) / nuv computed
  // above. That flip is load-bearing: the rhs of the linear system is
  // (cpol*curpol + ctor*curtor - benf) and the curpol/curtor terms DO NOT
  // flip. Verified against the Fortran vac2 reference: with the flip all
  // intermediates agree to machine precision. Env `VAC2_NO_BENF_FLIP=1`
  // restores the "natural" convention for comparisons only.
  const bool no_flip = []() {
    const char* p = std::getenv("VAC2_NO_BENF_FLIP");
    return p != nullptr && p[0] == '1';
  }();
  const double sign_bns = no_flip ? +1.0 : -1.0;
  for (int i_f = 2; i_f <= mnpot; ++i_f) {
    out->benf[i_f - 2] = sign_bns * bns[i_f - 1];
    out->benf[i_f - 2 + mnpot1] = bnc[i_f - 1];
  }
  if (!in.lasym) {
    // under stellarator symmetry only the sine half carries data
    for (int k = mnpot1; k < nd; ++k) {
      out->benf[k] = 0.0;
    }
  }
}  // ComputeBenf

void Vac2Solver::ReconstructPotentials(const Input& in, Output* out) const {
  const int nuv = nu_ * nv_;
  const int nd = nd_;

  // Stage 8: reconstruct potU, potV (port of solver.f90:87-96).
  // The base secular term is (-curtor, +curpol).
  out->potU.assign(nuv, -in.curtor);
  out->potV.assign(nuv, in.curpol);
  const int nd_half = nd / 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < nuv; ++i) {
    const int ku = Iu(i, nv_);
    const int kv = Iv(i, nv_);
    double accU = 0.0, accV = 0.0;
    for (int j0 = 0; j0 < nd_half; ++j0) {
      const int i_mode = j0 + 1;  // skip (m=0, n=0)
      const int m = ma_[i_mode];
      const int n = na_[i_mode];
      const double hs_c = out->hs[j0];
      const double hs_s = out->hs[j0 + nd_half];
      const double cum = conu_[static_cast<std::size_t>(ku) * mpol_ + m];
      const double sum_ = sinu_[static_cast<std::size_t>(ku) * mpol_ + m];
      const double cvn = ConV(conv_, ntor_, kv, n);
      const double svn = SinV(sinv_, ntor_, kv, n);
      const double co = cum * cvn - sum_ * svn;
      const double si = sum_ * cvn + cum * svn;
      const double wt = kTwoPi * (hs_c * co - hs_s * si);
      accU += m * wt;
      accV += n * wt;
    }
    out->potU[i] += accU;
    out->potV[i] += accV;
  }

  // Stage 9: bsqvac = (0.5 potU^2 gvv - potU potV guv + 0.5 potV^2 guu)
  //                   / (guu gvv - guv^2)   -- port of solver.f90:98-106.
  out->bsqvac.assign(nuv, 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < nuv; ++i) {
    const double guu = in.guu[i];
    const double guv = in.guv[i];
    const double gvv = in.gvv[i];
    const double pu = out->potU[i];
    const double pv = out->potV[i];
    const double det = guu * gvv - guv * guv;
    out->bsqvac[i] =
        (0.5 * pu * pu * gvv - pu * pv * guv + 0.5 * pv * pv * guu) / det;
  }
}  // ReconstructPotentials

}  // namespace vmecpp

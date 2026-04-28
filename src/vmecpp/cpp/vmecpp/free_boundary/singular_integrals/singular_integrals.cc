// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/singular_integrals/singular_integrals.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace vmecpp {

SingularIntegrals::SingularIntegrals(const Sizes* s,
                                     const FourierBasisFastToroidal* fb,
                                     const TangentialPartitioning* tp,
                                     const SurfaceGeometry* sg, int nf, int mf)
    : s_(*s), fb_(*fb), tp_(*tp), sg_(*sg), nf(nf), mf(mf) {
  numSC = mf * (nf + 1);
  numCS = (mf + 1) * nf;
  nzLen = numSC + numCS;

  cmn.resize((1 + nf + mf) * (nf + 1) * (mf + 1));
  cmns.resize((1 + nf + mf) * (nf + 1) * (mf + 1));

  // -------------

  // thread-local tangential grid point range
  int numLocal = tp_.ztMax - tp_.ztMin;

  ap.resize(numLocal);
  am.resize(numLocal);
  d.resize(numLocal);
  sqrtc2.resize(numLocal);
  sqrta2.resize(numLocal);
  delta4.resize(numLocal);

  Ap.resize(numLocal);
  Am.resize(numLocal);
  D.resize(numLocal);
  R1p.resize(numLocal);
  R1m.resize(numLocal);
  R0p.resize(numLocal);
  R0m.resize(numLocal);
  Ra1p.resize(numLocal);
  Ra1m.resize(numLocal);

  Tl2p.resize(numLocal);
  Tl2m.resize(numLocal);
  Tl1p.resize(numLocal);
  Tl1m.resize(numLocal);

  // (mf + nf + 1) + 1 -> one extra entry for last iteration of fl loop
  Tlp.resize(mf + nf + 2, numLocal);
  Tlm.resize(mf + nf + 2, numLocal);

  Slp.resize(mf + nf + 1, numLocal);
  Slm.resize(mf + nf + 1, numLocal);

  const int mnfull = (2 * nf + 1) * (mf + 1);
  bvec_sin.setZero(mnfull);
  grpmn_sin.setZero(mnfull * numLocal);
  if (s->lasym) {
    bvec_cos.setZero(mnfull);
    grpmn_cos.setZero(mnfull * numLocal);
  }

  // -------------

  computeCoefficients();
}

void SingularIntegrals::computeCoefficients() {
  // below loop sets only parts of cmn,
  // so initialize all entries to zero once here
  cmn.setZero();

  // cmn from scratch: Algorithm 1 in TNOV
  for (int n = 0; n < nf + 1; ++n) {
    for (int m = 0; m < mf + 1; ++m) {
      int i_mn = m - n;
      int j_mn = m + n;
      int k_mn = abs(i_mn);

      // originally: s_mn = (j_mn + k_mn) / 2
      // (j+k) is always even, so dividing by 2 is always possible
      // also: s_mn = 0.5*(m + n + abs(m - n)) == max(m, n)
      int s_mn = std::max(m, n);

      double f1 = 1.0;
      double f2 = 1.0;
      double f3 = 1.0;

      for (int i = 1; i <= k_mn; ++i) {
        f1 *= s_mn - i + 1;
        f2 *= i;
      }

      // (-1)^{(l-i_mn)/2} == (-1)^{(k_mn-i_mn)/2} at beginning of l-loop;
      // note (6.182 in TNOV) that (k_mn - i_mn) / 2 == max(0, n - m)
      // --> compute initial value and then reverse on each iteration of l loop
      // l gets increased by 2 --> l/2 gets increased by 1 per iteration
      // --> (-1)^{(l-i_mn)/2} == (-1)^{l/2 - i_mn/2} == (-1)^{l/2} /
      // (-1)^{i_mn/2} and since i_mn is constant during the l-iterations, the
      // sign reversed in each iteration
      int cmnSign = (std::max(0, n - m) % 2 == 0) ? 1 : -1;

      for (int l = k_mn; l <= j_mn; l += 2) {
        int lnm = (l * (nf + 1) + n) * (mf + 1) + m;

        cmn[lnm] = f1 / (f2 * f3) * cmnSign;

        f1 *= (l + 2 + j_mn) * (j_mn - l) * 0.25;
        f2 *= (l + 2 + k_mn) * 0.5;
        f3 *= (l + 2 - k_mn) * 0.5;

        cmnSign = -cmnSign;
      }  // l
    }  // m
  }  // n

  // cmns from cmn: (6.291) in TNOV
  for (int n = 0; n < nf + 1; ++n) {
    for (int m = 0; m < mf + 1; ++m) {
      int n_m_ = n * (mf + 1) + m;
      int n1m_ = (n - 1) * (mf + 1) + m;
      int n_m1 = n * (mf + 1) + (m - 1);
      int n1m1 = (n - 1) * (mf + 1) + (m - 1);
      for (int l = 0; l < 1 + mf + nf; ++l) {
        int ln_m_ = l * (mf + 1) * (nf + 1) + n_m_;
        int ln1m_ = l * (mf + 1) * (nf + 1) + n1m_;
        int ln_m1 = l * (mf + 1) * (nf + 1) + n_m1;
        int ln1m1 = l * (mf + 1) * (nf + 1) + n1m1;
        if (m == 0 && n == 0) {
          cmns[ln_m_] = cmn[ln_m_];
        } else if (m == 0 && n > 0) {
          cmns[ln_m_] = (cmn[ln_m_] + cmn[ln1m_]) / 2;
        } else if (m > 0 && n == 0) {
          cmns[ln_m_] = (cmn[ln_m_] + cmn[ln_m1]) / 2;
        } else {
          cmns[ln_m_] = (cmn[ln_m_] + cmn[ln1m_] + cmn[ln_m1] + cmn[ln1m1]) / 2;
        }
      }  // l
    }  // m
  }  // n
}  // computeCoefficients

void SingularIntegrals::update(const Eigen::VectorXd& bDotN, bool fullUpdate) {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  prepareUpdate(sg_.guu, sg_.guv, sg_.gvv, sg_.auu, sg_.auv, sg_.avv,
                fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  performUpdate(bDotN, fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}  // update

void SingularIntegrals::prepareUpdate(
    const Eigen::VectorXd& a, const Eigen::VectorXd& b2,
    const Eigen::VectorXd& c, const Eigen::VectorXd& A,
    const Eigen::VectorXd& B2, const Eigen::VectorXd& C, bool fullUpdate) {
  int numLocal = tp_.ztMax - tp_.ztMin;
  for (int kl = 0; kl < numLocal; ++kl) {
    // initialize constants (along expansion in l)
    ap[kl] = a[kl] + b2[kl] + c[kl];
    am[kl] = a[kl] - b2[kl] + c[kl];
    d[kl] = c[kl] - a[kl];
    sqrtc2[kl] = 2.0 * sqrt(c[kl]);
    sqrta2[kl] = 2.0 * sqrt(a[kl]);

    if (fullUpdate) {
      delta4[kl] = ap[kl] * am[kl] - d[kl] * d[kl];

      Ap[kl] = A[kl] + B2[kl] + C[kl];
      Am[kl] = A[kl] - B2[kl] + C[kl];
      D[kl] = C[kl] - A[kl];

      R1p[kl] = (Ap[kl] * (delta4[kl] - d[kl] * d[kl]) / ap[kl] -
                 Am[kl] * ap[kl] + 2 * D[kl] * d[kl]) /
                delta4[kl];
      R1m[kl] = (Am[kl] * (delta4[kl] - d[kl] * d[kl]) / am[kl] -
                 Ap[kl] * am[kl] + 2 * D[kl] * d[kl]) /
                delta4[kl];
      R0p[kl] = (-Ap[kl] * am[kl] * d[kl] / ap[kl] - Am[kl] * d[kl] +
                 2 * D[kl] * am[kl]) /
                delta4[kl];
      R0m[kl] = (-Am[kl] * ap[kl] * d[kl] / am[kl] - Ap[kl] * d[kl] +
                 2 * D[kl] * ap[kl]) /
                delta4[kl];
      Ra1p[kl] = Ap[kl] / ap[kl];
      Ra1m[kl] = Am[kl] / am[kl];
    }  // fullUpdate

    const double sqrtap = sqrt(ap[kl]);
    const double sqrtam = sqrt(am[kl]);

    // Compute T^{\pm}_0 analytically (eq. 6.207 in the_numerics_of_vmecpp.pdf).
    const double T0p = log((sqrtap * sqrtc2[kl] + ap[kl] + d[kl]) /
                           (sqrtap * sqrta2[kl] - ap[kl] + d[kl])) /
                       sqrtap;
    const double T0m = log((sqrtam * sqrtc2[kl] + am[kl] + d[kl]) /
                           (sqrtam * sqrta2[kl] - am[kl] + d[kl])) /
                       sqrtam;

    // Fill all Tlp[0..L] and Tlm[0..L] by picking the numerically stable
    // direction of the three-term recurrence on a per-(+/-), per-kl basis.
    //
    // The characteristic roots of the homogeneous recurrence satisfy
    //   A*r^2 + 2*d*r + B = 0  -> |r1 r2| = B/A.
    // For T^+: (A, B) = (ap, am), so |r1 r2| = am/ap.
    // For T^-: (A, B) = (am, ap), so |r1 r2| = ap/am.
    // If B > A (at least one |r| > 1), forward iteration is unstable and
    // backward (Miller's algorithm) is used instead; otherwise forward is fine.
    //
    // T^{\pm}_0 is analytic (above); T^{\pm}_{-1} = 0. Forward produces
    // T_{l+1} from T_l and T_{l-1}; backward produces T_{l-1} from T_l and
    // T_{l+1} via the same recurrence solved in reverse. For backward,
    // iteration starts from a zero seed far above the required L; the result
    // is then normalized to match the analytic T^{\pm}_0.
    //
    // rhs(l+1) = sqrtc2 + (-1)^{l+1}*sqrta2  (same for T^+ and T^-).
    const int kL = mf + nf;
    // The spurious solution is damped by (A/B)^kTailExtra per pass.
    // For the worst realistic ratio (A/B ~ 0.5) suppression is ~0.5^50 ~ 1e-16.
    const int kTailExtra = 50;
    const int kLtail = kL + kTailExtra;

    // Only switch to backward when the forward spurious-mode growth
    // (|r1 r2| = B/A) would actually exceed double precision over kL steps.
    // Threshold: forward is considered stable as long as (B/A)^kL < 1e10,
    // i.e. spurious amplitude stays within ~1e10 of the particular solution.
    // Near-degenerate kl (|r1|~|r2|~1) fall in the forward branch, where
    // zero-seed Miller is known to misconverge (spurious modes never damp).
    // Formula: kL * ln(B/A) < ln(1e10) -> B/A < exp(ln(1e10)/kL).
    constexpr double kLogGrowthThreshold = 10.0 * 2.30258509299;  // ln(1e10)
    const double logRatioP =
        (am[kl] > ap[kl] && ap[kl] > 0.0) ? std::log(am[kl] / ap[kl]) : 0.0;
    const bool useBackwardP =
        static_cast<double>(kL) * logRatioP > kLogGrowthThreshold;
    const double logRatioM =
        (ap[kl] > am[kl] && am[kl] > 0.0) ? std::log(ap[kl] / am[kl]) : 0.0;
    const bool useBackwardM =
        static_cast<double>(kL) * logRatioM > kLogGrowthThreshold;

    // --- T^+: A = ap, B = am ---
    Tlp(0, kl) = T0p;
    if (useBackwardP) {
      // forward unstable -> use backward recurrence.
      double T_hi = 0.0;
      double T_cur = 1.0e-300;
      for (int l = kLtail; l >= 1; --l) {
        const double rhs = sqrtc2[kl] + (l % 2 == 0 ? -1.0 : 1.0) * sqrta2[kl];
        const double T_lo =
            (rhs - (2 * l + 1) * d[kl] * T_cur - (l + 1) * ap[kl] * T_hi) /
            (l * am[kl]);
        T_hi = T_cur;
        T_cur = T_lo;
        if (l - 1 <= kL) {
          Tlp(l - 1, kl) = T_lo;
        }
      }
      const double scaleP = T0p / Tlp(0, kl);
      for (int l = 0; l <= kL; ++l) {
        Tlp(l, kl) *= scaleP;
      }
    } else {
      // forward stable.
      double T_prev = 0.0;  // T^+_{-1}
      int sgn = 1;
      for (int fl = 0; fl < kL; ++fl) {
        sgn = -sgn;
        const double rhs = sqrtc2[kl] + sgn * sqrta2[kl];
        const double T_next =
            (rhs - (2 * fl + 1) * d[kl] * Tlp(fl, kl) - fl * am[kl] * T_prev) /
            (ap[kl] * (fl + 1));
        T_prev = Tlp(fl, kl);
        Tlp(fl + 1, kl) = T_next;
      }
    }

    // --- T^-: A = am, B = ap ---
    Tlm(0, kl) = T0m;
    if (useBackwardM) {
      // forward unstable -> use backward recurrence.
      double T_hi = 0.0;
      double T_cur = 1.0e-300;
      for (int l = kLtail; l >= 1; --l) {
        const double rhs = sqrtc2[kl] + (l % 2 == 0 ? -1.0 : 1.0) * sqrta2[kl];
        const double T_lo =
            (rhs - (2 * l + 1) * d[kl] * T_cur - (l + 1) * am[kl] * T_hi) /
            (l * ap[kl]);
        T_hi = T_cur;
        T_cur = T_lo;
        if (l - 1 <= kL) {
          Tlm(l - 1, kl) = T_lo;
        }
      }
      const double scaleM = T0m / Tlm(0, kl);
      for (int l = 0; l <= kL; ++l) {
        Tlm(l, kl) *= scaleM;
      }
    } else {
      // forward stable.
      double T_prev = 0.0;  // T^-_{-1}
      int sgn = 1;
      for (int fl = 0; fl < kL; ++fl) {
        sgn = -sgn;
        const double rhs = sqrtc2[kl] + sgn * sqrta2[kl];
        const double T_next =
            (rhs - (2 * fl + 1) * d[kl] * Tlm(fl, kl) - fl * ap[kl] * T_prev) /
            (am[kl] * (fl + 1));
        T_prev = Tlm(fl, kl);
        Tlm(fl + 1, kl) = T_next;
      }
    }
  }  // kl
}  // prepareUpdate

void SingularIntegrals::performUpdate(const Eigen::VectorXd& bDotN,
                                      bool fullUpdate) {
  const int numLocal = tp_.ztMax - tp_.ztMin;

  bvec_sin.setZero();
  if (s_.lasym) {
    bvec_cos.setZero();
  }

  if (fullUpdate) {
    grpmn_sin.setZero();
    if (s_.lasym) {
      grpmn_cos.setZero();
    }
  }

  // Tl1p/Tl1m hold T^{\pm}_{fl-1} for the Slp/Slm formula; T^{\pm}_{-1} = 0.
  Tl1p.setZero();
  Tl1m.setZero();

  int sgn = 1;
  for (int fl = 0; fl < 1 + nf + mf; ++fl) {
    // COMPUTE SL+ and SL- , Eq (A17)
    // SLP(M): SL+(-)
    if (fullUpdate) {
      for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
        const int klRel = kl - tp_.ztMin;

        Slp(fl, klRel) = (R1p[klRel] * fl + Ra1p[klRel]) * Tlp(fl, klRel) +
                         R0p[klRel] * fl * Tl1p[klRel] -
                         (R0p[klRel] + R1p[klRel]) / sqrtc2[klRel] +
                         sgn * (R0p[klRel] - R1p[klRel]) / sqrta2[klRel];
        Slm(fl, klRel) = (R1m[klRel] * fl + Ra1m[klRel]) * Tlm(fl, klRel) +
                         R0m[klRel] * fl * Tl1m[klRel] -
                         (R0m[klRel] + R1m[klRel]) / sqrtc2[klRel] +
                         sgn * (R0m[klRel] - R1m[klRel]) / sqrta2[klRel];
      }  // kl
    }  // fullUpdate

    for (int n = 0; n < nf + 1; ++n) {
      for (int m = 0; m < mf + 1; ++m) {
        const int idx_m_posn = (nf + n) * (mf + 1) + m;
        const int idx_m_negn = (nf - n) * (mf + 1) + m;

        const int idx_lnm = (fl * (nf + 1) + n) * (mf + 1) + m;
        const double cmns_factor =
            cmns[idx_lnm] / (fb_.mscale[m] * fb_.nscale[n]);

        if (n == 0 || m == 0) {
          // analysum

          for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
            const int l = kl / s_.nZeta;
            const int k = kl % s_.nZeta;
            const int klRel = kl - tp_.ztMin;

            const int idx_lm = l * (s_.mnyq2 + 1) + m;
            const int idx_nk = n * s_.nZeta + k;

            // sin(mu - |n|v) * cmns(l,n,m)
            const double sinp = (fb_.sinmu[idx_lm] * fb_.cosnv[idx_nk] -
                                 fb_.cosmu[idx_lm] * fb_.sinnv[idx_nk]) *
                                cmns_factor;

            bvec_sin[idx_m_posn] += (Tlp(fl, klRel) + Tlm(fl, klRel)) *
                                    bDotN[klRel] * s_.wInt[l] * sinp;
            if (fullUpdate) {
              grpmn_sin[idx_m_posn * numLocal + klRel] +=
                  (Slp(fl, klRel) + Slm(fl, klRel)) * sinp;
            }

            if (s_.lasym) {
              // cos(mu - |n|v) * cmns(l,n,m)
              const double cosp = (fb_.cosmu[idx_lm] * fb_.cosnv[idx_nk] +
                                   fb_.sinmu[idx_lm] * fb_.sinnv[idx_nk]) *
                                  cmns_factor;

              bvec_cos[idx_m_posn] += (Tlp(fl, klRel) + Tlm(fl, klRel)) *
                                      bDotN[klRel] * s_.wInt[l] * cosp;
              if (fullUpdate) {
                grpmn_cos[idx_m_posn * numLocal + klRel] +=
                    (Slp(fl, klRel) + Slm(fl, klRel)) * cosp;
              }
            }
          }  // kl

        } else {
          // analysum2

          for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
            const int l = kl / s_.nZeta;
            int k = kl % s_.nZeta;
            const int idx_lm = l * (s_.mnyq2 + 1) + m;
            const int remaining = std::min(s_.nZeta - k, tp_.ztMax - kl);

            const double coeff1 = fb_.sinmu[idx_lm] * cmns_factor;
            const double coeff2 = fb_.cosmu[idx_lm] * cmns_factor;

            std::array<double, 4> buf_m_posn{};
            std::array<double, 4> buf_m_negn{};

            int i = 0;
            for (; i + 3 < remaining; i += 4, k += 4, kl += 4) {
              // in here l is constant and k always increases
              const int klRel = kl - tp_.ztMin;
              const int idx_nk = n * s_.nZeta + k;

              const double c0 = bDotN[klRel + 0] * s_.wInt[l];
              const double c1 = bDotN[klRel + 1] * s_.wInt[l];
              const double c2 = bDotN[klRel + 2] * s_.wInt[l];
              const double c3 = bDotN[klRel + 3] * s_.wInt[l];

              // sin(mu - |n|v) * cmns(l,n,m)
              const double sinp0 = coeff1 * fb_.cosnv[idx_nk + 0] -
                                   coeff2 * fb_.sinnv[idx_nk + 0];
              const double sinp1 = coeff1 * fb_.cosnv[idx_nk + 1] -
                                   coeff2 * fb_.sinnv[idx_nk + 1];
              const double sinp2 = coeff1 * fb_.cosnv[idx_nk + 2] -
                                   coeff2 * fb_.sinnv[idx_nk + 2];
              const double sinp3 = coeff1 * fb_.cosnv[idx_nk + 3] -
                                   coeff2 * fb_.sinnv[idx_nk + 3];

              buf_m_posn[0] += Tlp(fl, klRel + 0) * c0 * sinp0;
              buf_m_posn[1] += Tlp(fl, klRel + 1) * c1 * sinp1;
              buf_m_posn[2] += Tlp(fl, klRel + 2) * c2 * sinp2;
              buf_m_posn[3] += Tlp(fl, klRel + 3) * c3 * sinp3;

              // sin(mu + |n|v) * cmns(l,n,m)
              const double sinm0 = coeff1 * fb_.cosnv[idx_nk + 0] +
                                   coeff2 * fb_.sinnv[idx_nk + 0];
              const double sinm1 = coeff1 * fb_.cosnv[idx_nk + 1] +
                                   coeff2 * fb_.sinnv[idx_nk + 1];
              const double sinm2 = coeff1 * fb_.cosnv[idx_nk + 2] +
                                   coeff2 * fb_.sinnv[idx_nk + 2];
              const double sinm3 = coeff1 * fb_.cosnv[idx_nk + 3] +
                                   coeff2 * fb_.sinnv[idx_nk + 3];

              buf_m_negn[0] += Tlm(fl, klRel + 0) * c0 * sinm0;
              buf_m_negn[1] += Tlm(fl, klRel + 1) * c1 * sinm1;
              buf_m_negn[2] += Tlm(fl, klRel + 2) * c2 * sinm2;
              buf_m_negn[3] += Tlm(fl, klRel + 3) * c3 * sinm3;

              if (fullUpdate) {
                grpmn_sin[idx_m_posn * numLocal + klRel + 0] +=
                    Slp(fl, klRel + 0) * sinp0;
                grpmn_sin[idx_m_posn * numLocal + klRel + 1] +=
                    Slp(fl, klRel + 1) * sinp1;
                grpmn_sin[idx_m_posn * numLocal + klRel + 2] +=
                    Slp(fl, klRel + 2) * sinp2;
                grpmn_sin[idx_m_posn * numLocal + klRel + 3] +=
                    Slp(fl, klRel + 3) * sinp3;

                grpmn_sin[idx_m_negn * numLocal + klRel + 0] +=
                    Slm(fl, klRel + 0) * sinm0;
                grpmn_sin[idx_m_negn * numLocal + klRel + 1] +=
                    Slm(fl, klRel + 1) * sinm1;
                grpmn_sin[idx_m_negn * numLocal + klRel + 2] +=
                    Slm(fl, klRel + 2) * sinm2;
                grpmn_sin[idx_m_negn * numLocal + klRel + 3] +=
                    Slm(fl, klRel + 3) * sinm3;
              }
            }

            bvec_sin[idx_m_posn] +=
                buf_m_posn[0] + buf_m_posn[1] + buf_m_posn[2] + buf_m_posn[3];
            bvec_sin[idx_m_negn] +=
                buf_m_negn[0] + buf_m_negn[1] + buf_m_negn[2] + buf_m_negn[3];

            if (i != remaining) {
              for (; i < remaining; ++i, ++k, ++kl) {
                // in here l is constant and k always increases
                const int klRel = kl - tp_.ztMin;
                const int idx_nk = n * s_.nZeta + k;

                const double coeff1 =
                    fb_.sinmu[idx_lm] * fb_.cosnv[idx_nk] * cmns_factor;
                const double coeff2 =
                    fb_.cosmu[idx_lm] * fb_.sinnv[idx_nk] * cmns_factor;

                // sin(mu + |n|v) * cmns(l,n,m)
                const double sinm = coeff1 + coeff2;

                // sin(mu - |n|v) * cmns(l,n,m)
                const double sinp = coeff1 - coeff2;

                const double c = bDotN[klRel] * s_.wInt[l];
                bvec_sin[idx_m_posn] += Tlp(fl, klRel) * c * sinp;
                bvec_sin[idx_m_negn] += Tlm(fl, klRel) * c * sinm;

                if (fullUpdate) {
                  grpmn_sin[idx_m_posn * numLocal + klRel] +=
                      Slp(fl, klRel) * sinp;
                  grpmn_sin[idx_m_negn * numLocal + klRel] +=
                      Slm(fl, klRel) * sinm;
                }
              }
            }

            // adjust for the ++kl that's coming from the outer loop
            --kl;
          }  // kl

          if (s_.lasym) {
            for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
              const int l = kl / s_.nZeta;
              const int k = kl % s_.nZeta;
              const int klRel = kl - tp_.ztMin;

              const int idx_lm = l * (s_.mnyq2 + 1) + m;
              const int idx_nk = n * s_.nZeta + k;

              const double coeff1 =
                  fb_.cosmu[idx_lm] * fb_.cosnv[idx_nk] * cmns_factor;
              const double coeff2 =
                  fb_.sinmu[idx_lm] * fb_.sinnv[idx_nk] * cmns_factor;

              // cos(mu + |n|v) * cmns(l,n,m)
              const double cosm = coeff1 - coeff2;

              // cos(mu - |n|v) * cmns(l,n,m)
              const double cosp = coeff1 + coeff2;

              bvec_cos[idx_m_posn] +=
                  Tlp(fl, klRel) * bDotN[klRel] * s_.wInt[l] * cosp;
              bvec_cos[idx_m_negn] +=
                  Tlm(fl, klRel) * bDotN[klRel] * s_.wInt[l] * cosm;
              if (fullUpdate) {
                grpmn_cos[idx_m_posn * numLocal + klRel] +=
                    Slp(fl, klRel) * cosp;
                grpmn_cos[idx_m_negn * numLocal + klRel] +=
                    Slm(fl, klRel) * cosm;
              }
            }
          }  // kl
        }  // m == 0 or n == 0
      }  // m
    }  // n

    // T^{\pm}_l are pre-computed in prepareUpdate via backward recurrence.
    // Update sgn and shift Tl1p/Tl1m to T^{\pm}_{fl} for the Slp/Slm formula
    // in the next iteration.
    sgn = -sgn;
    Tl1p = Tlp.row(fl).transpose();
    Tl1m = Tlm.row(fl).transpose();
  }  // fl
}  // performUpdate

}  // namespace vmecpp

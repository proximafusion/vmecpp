// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#include "vmecpp/free_boundary/surface_geometry/surface_geometry.h"

namespace vmecpp {

SurfaceGeometry::SurfaceGeometry(const Sizes* s,
                                 const FourierBasisFastToroidal* fb,
                                 const TangentialPartitioning* tp)
    : s_(*s), fb_(*fb), tp_(*tp) {
  cos_per.resize(s_.nfp);
  sin_per.resize(s_.nfp);

  cos_phi.resize(s_.nZeta);
  sin_phi.resize(s_.nZeta);

  // -----------------

  // full surface
  r1b.resize(s_.nThetaEven * s_.nZeta);
  z1b.resize(s_.nThetaEven * s_.nZeta);
  rcosuv.resize(s_.nThetaEven * s_.nZeta);
  rsinuv.resize(s_.nThetaEven * s_.nZeta);
  rzb2.resize(s_.nThetaEven * s_.nZeta);

  // thread-local tangential grid point range
  int numLocal = tp_.ztMax - tp_.ztMin;

  // For lasym the surface derivatives are needed over the full poloidal range
  // (so the antisymmetric pieces can be mirrored into ]pi,2pi[), so each thread
  // holds the full set, like r1b/z1b. For the stellarator-symmetric case they
  // stay thread-local.
  const int derivSize = s_.lasym ? s_.nZnT : numLocal;

  rub.resize(derivSize);
  rvb.resize(derivSize);
  zub.resize(derivSize);
  zvb.resize(derivSize);

  ruu.resize(derivSize);
  ruv.resize(derivSize);
  rvv.resize(derivSize);
  zuu.resize(derivSize);
  zuv.resize(derivSize);
  zvv.resize(derivSize);

  snr.resize(numLocal);
  snv.resize(numLocal);
  snz.resize(numLocal);

  guu.resize(numLocal);
  guv.resize(numLocal);
  gvv.resize(numLocal);

  auu.resize(numLocal);
  auv.resize(numLocal);
  avv.resize(numLocal);

  drv.resize(numLocal);

  // Non-stellarator-symmetric antisymmetric pieces (only allocated for lasym).
  if (s_.lasym) {
    r1b_asym.resize(s_.nZnT);
    z1b_asym.resize(s_.nZnT);
    rub_asym.resize(s_.nZnT);
    rvb_asym.resize(s_.nZnT);
    zub_asym.resize(s_.nZnT);
    zvb_asym.resize(s_.nZnT);
    ruu_asym.resize(s_.nZnT);
    ruv_asym.resize(s_.nZnT);
    rvv_asym.resize(s_.nZnT);
    zuu_asym.resize(s_.nZnT);
    zuv_asym.resize(s_.nZnT);
    zvv_asym.resize(s_.nZnT);
  }

  // -----------------

  computeConstants();
}

void SurfaceGeometry::computeConstants() {
  double omega_per = 2.0 * M_PI / s_.nfp;
  for (int p = 0; p < s_.nfp; ++p) {
    double phi_per = omega_per * p;
    cos_per[p] = cos(phi_per);
    sin_per[p] = sin(phi_per);
  }

  double omega_phi = 2.0 * M_PI / (s_.nfp * s_.nZeta);
  for (int k = 0; k < s_.nZeta; ++k) {
    double phi = omega_phi * k;
    cos_phi[k] = cos(phi);
    sin_phi[k] = sin(phi);
  }
}

// Evaluate the Fourier series for the surface geometry
// and compute quantities depending on it.
void SurfaceGeometry::update(
    const std::span<const double> rCC, const std::span<const double> rSS,
    const std::span<const double> rSC, const std::span<const double> rCS,
    const std::span<const double> zSC, const std::span<const double> zCS,
    const std::span<const double> zCC, const std::span<const double> zSS,
    int signOfJacobian, bool fullUpdate) {
#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  inverseDFT(rCC, rSS, rSC, rCS, zSC, zCS, zCC, zSS, fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP

  derivedSurfaceQuantities(signOfJacobian, fullUpdate);

#ifdef _OPENMP
#pragma omp barrier
#endif  // _OPENMP
}

// Perform inverse Fourier transform from Fourier coefficients of surface
// geometry into realspace and compute 1st- and 2nd-order tangential
// derivatives.
void SurfaceGeometry::inverseDFT(
    const std::span<const double> rCC, const std::span<const double> rSS,
    const std::span<const double> rSC, const std::span<const double> rCS,
    const std::span<const double> zSC, const std::span<const double> zCS,
    const std::span<const double> zCC, const std::span<const double> zSS,
    bool fullUpdate) {
  // The non-stellarator-symmetric coefficients rSC/rCS/zCC/zSS are folded in
  // below, guarded by s_.lasym. For stellarator-symmetric runs those spans are
  // empty and the antisymmetric arrays are neither allocated nor touched.

  // ----------------

  r1b.setZero();
  z1b.setZero();
  if (s_.lasym) {
    r1b_asym.setZero();
    z1b_asym.setZero();
  }

  // ----------------

  // For lasym the derivative arrays span the full poloidal range and are
  // indexed by the absolute tangential index (offset 0); for the symmetric case
  // they are thread-local and indexed relative to tp_.ztMin.
  const int derivOffset = s_.lasym ? 0 : tp_.ztMin;

  rub.setZero();
  rvb.setZero();
  zub.setZero();
  zvb.setZero();
  if (s_.lasym) {
    rub_asym.setZero();
    rvb_asym.setZero();
    zub_asym.setZero();
    zvb_asym.setZero();
  }

  if (fullUpdate) {
    ruu.setZero();
    ruv.setZero();
    rvv.setZero();
    zuu.setZero();
    zuv.setZero();
    zvv.setZero();
    if (s_.lasym) {
      ruu_asym.setZero();
      ruv_asym.setZero();
      rvv_asym.setZero();
      zuu_asym.setZero();
      zuv_asym.setZero();
      zvv_asym.setZero();
    }
  }

  for (int n = 0; n < s_.ntor + 1; ++n) {
    // needed for second-order toroidal derivatives
    int nSq = n * s_.nfp * n * s_.nfp;

    int lMin = tp_.ztMin / s_.nZeta;
    int lMax = tp_.ztMax / s_.nZeta;

    for (int l = 0; l < s_.nThetaReduced; ++l) {
      double rmkcc = 0.0;
      double rmkss = 0.0;
      double zmksc = 0.0;
      double zmkcs = 0.0;

      // ----------------

      double rmkcc_m = 0.0;
      double rmkcc_mm = 0.0;
      double rmkss_m = 0.0;
      double rmkss_mm = 0.0;

      double zmksc_m = 0.0;
      double zmksc_mm = 0.0;
      double zmkcs_m = 0.0;
      double zmkcs_mm = 0.0;

      // antisymmetric (lasym) accumulators: R uses rSC*sin(mu)*cos(nv) +
      // rCS*cos(mu)*sin(nv); Z uses zCC*cos(mu)*cos(nv) + zSS*sin(mu)*sin(nv).
      double rmksc = 0.0;
      double rmkcs = 0.0;
      double zmkcc = 0.0;
      double zmkss = 0.0;
      double rmksc_m = 0.0;
      double rmksc_mm = 0.0;
      double rmkcs_m = 0.0;
      double rmkcs_mm = 0.0;
      double zmkcc_m = 0.0;
      double zmkcc_mm = 0.0;
      double zmkss_m = 0.0;
      double zmkss_mm = 0.0;

      for (int m = 0; m < s_.mpol; ++m) {
        int idx_mn = n * s_.mpol + m;

        // needed for second-order poloidal derivatives
        int mSq = m * m;

        double cosmu = fb_.cosmu[l * (s_.mnyq2 + 1) + m];
        double sinmu = fb_.sinmu[l * (s_.mnyq2 + 1) + m];

        rmkcc += rCC[idx_mn] * cosmu;
        rmkss += rSS[idx_mn] * sinmu;
        zmksc += zSC[idx_mn] * sinmu;
        zmkcs += zCS[idx_mn] * cosmu;

        if (s_.lasym) {
          rmksc += rSC[idx_mn] * sinmu;
          rmkcs += rCS[idx_mn] * cosmu;
          zmkcc += zCC[idx_mn] * cosmu;
          zmkss += zSS[idx_mn] * sinmu;
        }

        // ----------------

        if (lMin <= l && l <= lMax) {
          // TODO(jons): in asymmetric case, some processors will have local
          // poloidal ranges outside the first half-module
          // --> these would be excluded here, but they still need to do some
          // work here!

          double cosmum = fb_.cosmum[l * (s_.mnyq2 + 1) + m];
          double sinmum = fb_.sinmum[l * (s_.mnyq2 + 1) + m];
          double cosmumm = -mSq * fb_.cosmu[l * (s_.mnyq2 + 1) + m];
          double sinmumm = -mSq * fb_.sinmu[l * (s_.mnyq2 + 1) + m];

          rmkcc_m += rCC[idx_mn] * sinmum;
          rmkcc_mm += rCC[idx_mn] * cosmumm;
          rmkss_m += rSS[idx_mn] * cosmum;
          rmkss_mm += rSS[idx_mn] * sinmumm;

          zmksc_m += zSC[idx_mn] * cosmum;
          zmksc_mm += zSC[idx_mn] * sinmumm;
          zmkcs_m += zCS[idx_mn] * sinmum;
          zmkcs_mm += zCS[idx_mn] * cosmumm;

          if (s_.lasym) {
            rmksc_m += rSC[idx_mn] * cosmum;
            rmksc_mm += rSC[idx_mn] * sinmumm;
            rmkcs_m += rCS[idx_mn] * sinmum;
            rmkcs_mm += rCS[idx_mn] * cosmumm;
            zmkcc_m += zCC[idx_mn] * sinmum;
            zmkcc_mm += zCC[idx_mn] * cosmumm;
            zmkss_m += zSS[idx_mn] * cosmum;
            zmkss_mm += zSS[idx_mn] * sinmumm;
          }
        }
      }  // m

      for (int k = 0; k < s_.nZeta; ++k) {
        int idx_kl = l * s_.nZeta + k;

        double cosnv = fb_.cosnv[n * s_.nZeta + k];
        double sinnv = fb_.sinnv[n * s_.nZeta + k];

        r1b[idx_kl] += rmkcc * cosnv + rmkss * sinnv;
        z1b[idx_kl] += zmksc * cosnv + zmkcs * sinnv;

        if (s_.lasym) {
          r1b_asym[idx_kl] += rmksc * cosnv + rmkcs * sinnv;
          z1b_asym[idx_kl] += zmkcc * cosnv + zmkss * sinnv;
        }

        // ----------------

        // For lasym every thread computes the derivatives over the full reduced
        // poloidal range (so it can mirror them itself); the symmetric case
        // only computes its thread-local slice.
        if (s_.lasym || (tp_.ztMin <= idx_kl && idx_kl < tp_.ztMax)) {
          double cosnvn = fb_.cosnvn[n * s_.nZeta + k];
          double sinnvn = fb_.sinnvn[n * s_.nZeta + k];

          rub[idx_kl - derivOffset] += rmkcc_m * cosnv + rmkss_m * sinnv;
          rvb[idx_kl - derivOffset] += rmkcc * sinnvn + rmkss * cosnvn;
          zub[idx_kl - derivOffset] += zmksc_m * cosnv + zmkcs_m * sinnv;
          zvb[idx_kl - derivOffset] += zmksc * sinnvn + zmkcs * cosnvn;

          if (s_.lasym) {
            rub_asym[idx_kl - derivOffset] += rmksc_m * cosnv + rmkcs_m * sinnv;
            rvb_asym[idx_kl - derivOffset] += rmksc * sinnvn + rmkcs * cosnvn;
            zub_asym[idx_kl - derivOffset] += zmkcc_m * cosnv + zmkss_m * sinnv;
            zvb_asym[idx_kl - derivOffset] += zmkcc * sinnvn + zmkss * cosnvn;
          }

          if (fullUpdate) {
            double cosnvnn = -nSq * fb_.cosnv[n * s_.nZeta + k];
            double sinnvnn = -nSq * fb_.sinnv[n * s_.nZeta + k];

            ruu[idx_kl - derivOffset] += rmkcc_mm * cosnv + rmkss_mm * sinnv;
            ruv[idx_kl - derivOffset] += rmkcc_m * sinnvn + rmkss_m * cosnvn;
            rvv[idx_kl - derivOffset] += rmkcc * cosnvnn + rmkss * sinnvnn;
            zuu[idx_kl - derivOffset] += zmksc_mm * cosnv + zmkcs_mm * sinnv;
            zuv[idx_kl - derivOffset] += zmksc_m * sinnvn + zmkcs_m * cosnvn;
            zvv[idx_kl - derivOffset] += zmksc * cosnvnn + zmkcs * sinnvnn;

            if (s_.lasym) {
              ruu_asym[idx_kl - derivOffset] +=
                  rmksc_mm * cosnv + rmkcs_mm * sinnv;
              ruv_asym[idx_kl - derivOffset] +=
                  rmksc_m * sinnvn + rmkcs_m * cosnvn;
              rvv_asym[idx_kl - derivOffset] +=
                  rmksc * cosnvnn + rmkcs * sinnvnn;
              zuu_asym[idx_kl - derivOffset] +=
                  zmkcc_mm * cosnv + zmkss_mm * sinnv;
              zuv_asym[idx_kl - derivOffset] +=
                  zmkcc_m * sinnvn + zmkss_m * cosnvn;
              zvv_asym[idx_kl - derivOffset] +=
                  zmkcc * cosnvnn + zmkss * sinnvnn;
            }
          }
        }
      }  // k
    }  // l
  }  // n

  if (s_.lasym) {
    // Build the full poloidal range from the symmetric and antisymmetric pieces
    // (cf. educational_VMEC symrzl). The second poloidal half ]pi,2pi[ is the
    // parity-signed mirror of (symmetric - antisymmetric) at the reflected
    // point (theta -> 2pi-theta, zeta -> 2pi-zeta); the first half [0,pi] is
    // the sum. R, Ruu, Ruv, Rvv, Zu, Zv are even under that reflection (mirror
    // = +sym - asym); Ru, Rv, Z, Zuu, Zuv, Zvv are odd (mirror = -sym + asym).
    // The second half is done first, while the arrays still hold the pure
    // symmetric values.
    for (int l = 1; l < s_.nThetaReduced - 1; ++l) {
      const int lRev = (s_.nThetaEven - l) % s_.nThetaEven;
      for (int k = 0; k < s_.nZeta; ++k) {
        const int kRev = (s_.nZeta - k) % s_.nZeta;
        const int kl = l * s_.nZeta + k;
        const int klRev = lRev * s_.nZeta + kRev;

        r1b[klRev] = r1b[kl] - r1b_asym[kl];
        z1b[klRev] = -z1b[kl] + z1b_asym[kl];

        rub[klRev] = -rub[kl] + rub_asym[kl];
        rvb[klRev] = -rvb[kl] + rvb_asym[kl];
        zub[klRev] = zub[kl] - zub_asym[kl];
        zvb[klRev] = zvb[kl] - zvb_asym[kl];

        if (fullUpdate) {
          ruu[klRev] = ruu[kl] - ruu_asym[kl];
          ruv[klRev] = ruv[kl] - ruv_asym[kl];
          rvv[klRev] = rvv[kl] - rvv_asym[kl];
          zuu[klRev] = -zuu[kl] + zuu_asym[kl];
          zuv[klRev] = -zuv[kl] + zuv_asym[kl];
          zvv[klRev] = -zvv[kl] + zvv_asym[kl];
        }
      }  // k
    }  // l

    // first poloidal half [0,pi]: symmetric + antisymmetric
    for (int l = 0; l < s_.nThetaReduced; ++l) {
      for (int k = 0; k < s_.nZeta; ++k) {
        const int kl = l * s_.nZeta + k;

        r1b[kl] += r1b_asym[kl];
        z1b[kl] += z1b_asym[kl];

        rub[kl] += rub_asym[kl];
        rvb[kl] += rvb_asym[kl];
        zub[kl] += zub_asym[kl];
        zvb[kl] += zvb_asym[kl];

        if (fullUpdate) {
          ruu[kl] += ruu_asym[kl];
          ruv[kl] += ruv_asym[kl];
          rvv[kl] += rvv_asym[kl];
          zuu[kl] += zuu_asym[kl];
          zuv[kl] += zuv_asym[kl];
          zvv[kl] += zvv_asym[kl];
        }
      }  // k
    }  // l
  }
}

void SurfaceGeometry::derivedSurfaceQuantities(int signOfJacobian,
                                               bool fullUpdate) {
  // Derivatives are full-range (offset 0) for lasym and thread-local for the
  // symmetric case; the derived per-slice outputs stay thread-local.
  const int derivOffset = s_.lasym ? 0 : tp_.ztMin;
  for (int kl = tp_.ztMin; kl < tp_.ztMax; ++kl) {
    const int o = kl - tp_.ztMin;    // thread-local output index
    const int d = kl - derivOffset;  // derivative index (full-range for lasym)

    // surface normal vector components
    snr[o] = signOfJacobian * r1b[kl] * zub[d];
    snv[o] = signOfJacobian * (rub[d] * zvb[d] - zub[d] * rvb[d]);
    snz[o] = -signOfJacobian * r1b[kl] * rub[d];

    // metric elements; used in Imn and Kmn
    guu[o] = rub[d] * rub[d] + zub[d] * zub[d];
    guv[o] = 2.0 * (rub[d] * rvb[d] + zub[d] * zvb[d]) / s_.nfp;
    gvv[o] = (rvb[d] * rvb[d] + r1b[kl] * r1b[kl] + zvb[d] * zvb[d]) /
             (s_.nfp * s_.nfp);

    if (fullUpdate) {
      // d^2X/d(ij) . N (used in Kmn)
      auu[o] = (ruu[d] * snr[o] + zuu[d] * snz[o]) / 2;
      auv[o] = (ruv[d] * snr[o] + rub[d] * snv[o] + zuv[d] * snz[o]) / s_.nfp;
      avv[o] = (rvb[d] * snv[o] +
                ((rvv[d] - r1b[kl]) * snr[o] + zvv[d] * snz[o]) / 2) /
               (s_.nfp * s_.nfp);

      // -(R N^R + Z N^Z)
      drv[o] = -(r1b[kl] * snr[o] + z1b[kl] * snz[o]);
    }
  }  // kl

  if (fullUpdate) {
    // R^2 + Z^2
    for (int kl = 0; kl < s_.nZnT; ++kl) {
      rzb2[kl] = r1b[kl] * r1b[kl] + z1b[kl] * z1b[kl];
    }

    if (!s_.lasym) {
      // mirror into non-stellarator-symmetric half of poloidal range
      for (int l = 1; l < s_.nThetaReduced - 1; ++l) {
        int lRev = (s_.nThetaEven - l) % s_.nThetaEven;
        for (int k = 0; k < s_.nZeta; ++k) {
          int kRev = (s_.nZeta - k) % s_.nZeta;

          int kl = l * s_.nZeta + k;
          int klRev = lRev * s_.nZeta + kRev;

          r1b[klRev] = r1b[kl];
          z1b[klRev] = -z1b[kl];

          rzb2[klRev] = rzb2[kl];
        }  // k
      }  // l
    }

    // x and y
    for (int kl = 0; kl < s_.nThetaEven * s_.nZeta; ++kl) {
      int k = kl % s_.nZeta;
      rcosuv[kl] = r1b[kl] * cos_phi[k];
      rsinuv[kl] = r1b[kl] * sin_phi[k];
    }  // kl
  }
}

}  // namespace vmecpp

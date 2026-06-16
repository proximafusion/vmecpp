// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Flat-buffer, allocation-free forward transform of the half-grid field
// quantities SIMSOPT's QuasisymmetryRatioResidual needs: gmnc (Jacobian),
// bmnc (|B|), bsubumnc, bsubvmnc, bsupumnc, bsupvmnc. It mirrors the
// per-surface Fourier integral in output_quantities.cc bit for bit, but over
// plain double* buffers with explicit reductions, so Enzyme can differentiate
// it (the Eigen RowMatrixXd path in output_quantities allocates heap
// temporaries Enzyme cannot handle). Symmetric (lasym=false) case.
//
// Real-space inputs are the half-grid fields the force chain already produces
// (gsqrt, bsupu, bsupv, bsubu, bsubv); |B| is formed here from the total
// pressure as modB = sqrt(2 * |total_pressure - presH[jH]|), matching
// output_quantities. All real-space fields use the half-grid index
// idx = (jH*nZeta + k)*nThetaEff + l.

#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_QS_HARMONICS_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_QS_HARMONICS_KERNEL_H_

#include <cmath>

namespace vmecpp {

// Geometry/basis description for the half-grid QS forward transform. All arrays
// are caller-owned; the kernel reads them and writes only the output harmonics.
struct QsHarmonicsConfig {
  int nsH;            // number of half-grid surfaces (ns - 1)
  int nZeta;          // toroidal grid points
  int nThetaReduced;  // poloidal grid points on the reduced [0, pi] interval
  int nThetaEff;      // effective poloidal stride of the real-space buffers
  int nnyq2;          // toroidal Nyquist bound (cosnv stride is nnyq2 + 1)
  int mnyq;           // poloidal Nyquist mode (cos basis halved there)
  int nnyq;           // toroidal Nyquist mode (cos basis halved there)
  int mnmax_nyq;      // number of (m, n) Nyquist modes
  double tmult;       // overall transform normalization

  const int* xm_nyq;     // [mnmax_nyq] poloidal mode number per mode
  const int* xn_nyq;     // [mnmax_nyq] toroidal mode number per mode (n*nfp)
  int nfp;               // field periods (xn_nyq[.]/nfp gives n)
  const double* mscale;  // [mpol_nyq+1]
  const double* nscale;  // [nnyq2+1]
  const double* cosmui;  // [nThetaReduced*(mpol_nyq+1)] weighted cos basis
  const double* sinmui;  // [nThetaReduced*(mpol_nyq+1)] weighted sin basis
  const double* cosnv;   // [nZeta*(nnyq2+1)]
  const double* sinnv;   // [nZeta*(nnyq2+1)]
};

// Output harmonic layout: each quantity is a [mnmax_nyq * nsH] block, indexed
// harm[mn * nsH + jH]. Field input layout: each real-space field is a half-grid
// block of nsH*nZeta*nThetaEff doubles in the order described above. The six
// field pointers and six harmonic pointers are passed explicitly so the Enzyme
// JVP can mark exactly the active buffers.
// modB is formed from the total pressure exactly as output_quantities does:
//   |B| = sqrt(2 * |total_pressure - presH[jH]|),
// where total_pressure (per point) and presH (per half surface) are force-chain
// outputs. total_pressure[idx] uses the same half-grid index as the fields;
// presH[jH] is the kinetic pressure on half surface jH.
inline void ComputeQsHarmonics(const double* gsqrt,
                               const double* total_pressure,
                               const double* presH, const double* bsupu,
                               const double* bsupv, const double* bsubu,
                               const double* bsubv, double* gmnc, double* bmnc,
                               double* bsubumnc, double* bsubvmnc,
                               double* bsupumnc, double* bsupvmnc,
                               const QsHarmonicsConfig* c) {
  const int nsH = c->nsH;
  const int nZeta = c->nZeta;
  const int nThetaR = c->nThetaReduced;
  const int nThetaEff = c->nThetaEff;
  const int nnv = c->nnyq2 + 1;

  for (int jH = 0; jH < nsH; ++jH) {
    for (int mn = 0; mn < c->mnmax_nyq; ++mn) {
      const int m = c->xm_nyq[mn];
      const int n = c->xn_nyq[mn] / c->nfp;
      const int abs_n = n < 0 ? -n : n;
      const int sign_n = n < 0 ? -1 : 1;

      double dmult = c->mscale[m] * c->nscale[abs_n] * c->tmult;
      if (m == 0 || n == 0) {
        dmult *= 2.0;
      }

      // Nyquist modes carry half the cos-basis weight (output_quantities halves
      // cosmui at m==mnyq and cosnv at n==nnyq); the sin basis is unscaled.
      const double mhalf = (c->mnyq != 0 && m == c->mnyq) ? 0.5 : 1.0;
      const double nhalf = (c->nnyq != 0 && abs_n == c->nnyq) ? 0.5 : 1.0;

      double acc_g = 0.0, acc_b = 0.0, acc_su = 0.0, acc_sv = 0.0;
      double acc_pu = 0.0, acc_pv = 0.0;
      for (int l = 0; l < nThetaR; ++l) {
        const int ml = m * nThetaR + l;
        const double cmu = c->cosmui[ml] * mhalf;
        const double smu = c->sinmui[ml];
        for (int k = 0; k < nZeta; ++k) {
          const int kn = k * nnv + abs_n;
          const double tcosi = dmult * (cmu * c->cosnv[kn] * nhalf +
                                        sign_n * smu * c->sinnv[kn]);

          const int idx = (jH * nZeta + k) * nThetaEff + l;
          const double bu = bsubu[idx];
          const double bv = bsubv[idx];
          const double su = bsupu[idx];
          const double sv = bsupv[idx];
          const double modB =
              std::sqrt(2.0 * std::fabs(total_pressure[idx] - presH[jH]));

          acc_g += tcosi * gsqrt[idx];
          acc_b += tcosi * modB;
          acc_su += tcosi * bu;
          acc_sv += tcosi * bv;
          acc_pu += tcosi * su;
          acc_pv += tcosi * sv;
        }
      }
      const int o = mn * nsH + jH;
      gmnc[o] = acc_g;
      bmnc[o] = acc_b;
      bsubumnc[o] = acc_su;
      bsubvmnc[o] = acc_sv;
      bsupumnc[o] = acc_pu;
      bsupvmnc[o] = acc_pv;
    }
  }
}

// Adjoint (vector-Jacobian product) of ComputeQsHarmonics. Given a cotangent on
// each harmonic block, scatters the field-space gradient into the *_bar
// buffers. The forward transform is linear in every field except |B|, so the
// adjoint is the same basis contraction transposed (harmonic -> grid), with the
// single nonlinearity handled in closed form: modB = sqrt(2(total_pressure -
// presH)) gives d(modB)/d(total_pressure) = 1/modB, so bmnc's cotangent enters
// total_pressure_bar weighted by 1/modB. presH is a fixed profile and is not
// differentiated. This analytic adjoint replaces a reverse-mode AD pass over a
// linear map. The six *_bar field buffers (each nsH*nZeta*nThetaEff) are zeroed
// here and then accumulated.
inline void ComputeQsHarmonicsVjp(
    const double* gmnc_bar, const double* bmnc_bar, const double* bsubumnc_bar,
    const double* bsubvmnc_bar, const double* bsupumnc_bar,
    const double* bsupvmnc_bar, const double* total_pressure,
    const double* presH, double* gsqrt_bar, double* total_pressure_bar,
    double* bsupu_bar, double* bsupv_bar, double* bsubu_bar, double* bsubv_bar,
    const QsHarmonicsConfig* c) {
  const int nsH = c->nsH;
  const int nZeta = c->nZeta;
  const int nThetaR = c->nThetaReduced;
  const int nThetaEff = c->nThetaEff;
  const int nnv = c->nnyq2 + 1;
  const int npts = nsH * nZeta * nThetaEff;
  for (int i = 0; i < npts; ++i) {
    gsqrt_bar[i] = 0.0;
    total_pressure_bar[i] = 0.0;
    bsupu_bar[i] = 0.0;
    bsupv_bar[i] = 0.0;
    bsubu_bar[i] = 0.0;
    bsubv_bar[i] = 0.0;
  }

  for (int jH = 0; jH < nsH; ++jH) {
    for (int mn = 0; mn < c->mnmax_nyq; ++mn) {
      const int m = c->xm_nyq[mn];
      const int n = c->xn_nyq[mn] / c->nfp;
      const int abs_n = n < 0 ? -n : n;
      const int sign_n = n < 0 ? -1 : 1;

      double dmult = c->mscale[m] * c->nscale[abs_n] * c->tmult;
      if (m == 0 || n == 0) {
        dmult *= 2.0;
      }
      const double mhalf = (c->mnyq != 0 && m == c->mnyq) ? 0.5 : 1.0;
      const double nhalf = (c->nnyq != 0 && abs_n == c->nnyq) ? 0.5 : 1.0;

      const int o = mn * nsH + jH;
      const double g_b = gmnc_bar[o];
      const double b_b = bmnc_bar[o];
      const double su_b = bsubumnc_bar[o];
      const double sv_b = bsubvmnc_bar[o];
      const double pu_b = bsupumnc_bar[o];
      const double pv_b = bsupvmnc_bar[o];
      for (int l = 0; l < nThetaR; ++l) {
        const int ml = m * nThetaR + l;
        const double cmu = c->cosmui[ml] * mhalf;
        const double smu = c->sinmui[ml];
        for (int k = 0; k < nZeta; ++k) {
          const int kn = k * nnv + abs_n;
          const double tcosi = dmult * (cmu * c->cosnv[kn] * nhalf +
                                        sign_n * smu * c->sinnv[kn]);
          const int idx = (jH * nZeta + k) * nThetaEff + l;
          gsqrt_bar[idx] += tcosi * g_b;
          bsubu_bar[idx] += tcosi * su_b;
          bsubv_bar[idx] += tcosi * sv_b;
          bsupu_bar[idx] += tcosi * pu_b;
          bsupv_bar[idx] += tcosi * pv_b;
          const double diff = total_pressure[idx] - presH[jH];
          const double modB = std::sqrt(2.0 * std::fabs(diff));
          if (modB > 0.0) {
            const double dmodB = (diff < 0.0 ? -1.0 : 1.0) / modB;
            total_pressure_bar[idx] += tcosi * b_b * dmodB;
          }
        }
      }
    }
  }
}

// Forward tangent (Jacobian-vector product) of ComputeQsHarmonics. Given the
// real-space field tangents (the linear fields directly; |B| through the total
// pressure), projects them to the harmonic tangents with the same weighted
// basis the forward transform uses. The transform is linear in every field but
// |B|; for |B| the chain rule gives d|B| = d(total_pressure)/|B| (sign of
// total_pressure - presH), exactly the slope ComputeQsHarmonicsVjp transposes.
// presH is a fixed profile and contributes no tangent.
inline void ComputeQsHarmonicsTangent(
    const double* gsqrt_t, const double* total_pressure_t, const double* bsupu_t,
    const double* bsupv_t, const double* bsubu_t, const double* bsubv_t,
    const double* total_pressure, const double* presH, double* gmnc_t,
    double* bmnc_t, double* bsubumnc_t, double* bsubvmnc_t, double* bsupumnc_t,
    double* bsupvmnc_t, const QsHarmonicsConfig* c) {
  const int nsH = c->nsH;
  const int nZeta = c->nZeta;
  const int nThetaR = c->nThetaReduced;
  const int nThetaEff = c->nThetaEff;
  const int nnv = c->nnyq2 + 1;

  for (int jH = 0; jH < nsH; ++jH) {
    for (int mn = 0; mn < c->mnmax_nyq; ++mn) {
      const int m = c->xm_nyq[mn];
      const int n = c->xn_nyq[mn] / c->nfp;
      const int abs_n = n < 0 ? -n : n;
      const int sign_n = n < 0 ? -1 : 1;

      double dmult = c->mscale[m] * c->nscale[abs_n] * c->tmult;
      if (m == 0 || n == 0) {
        dmult *= 2.0;
      }
      const double mhalf = (c->mnyq != 0 && m == c->mnyq) ? 0.5 : 1.0;
      const double nhalf = (c->nnyq != 0 && abs_n == c->nnyq) ? 0.5 : 1.0;

      double acc_g = 0.0, acc_b = 0.0, acc_su = 0.0, acc_sv = 0.0;
      double acc_pu = 0.0, acc_pv = 0.0;
      for (int l = 0; l < nThetaR; ++l) {
        const int ml = m * nThetaR + l;
        const double cmu = c->cosmui[ml] * mhalf;
        const double smu = c->sinmui[ml];
        for (int k = 0; k < nZeta; ++k) {
          const int kn = k * nnv + abs_n;
          const double tcosi = dmult * (cmu * c->cosnv[kn] * nhalf +
                                        sign_n * smu * c->sinnv[kn]);
          const int idx = (jH * nZeta + k) * nThetaEff + l;
          const double diff = total_pressure[idx] - presH[jH];
          const double modB = std::sqrt(2.0 * std::fabs(diff));
          const double dmodB =
              modB > 0.0 ? (diff < 0.0 ? -1.0 : 1.0) / modB : 0.0;
          acc_g += tcosi * gsqrt_t[idx];
          acc_b += tcosi * dmodB * total_pressure_t[idx];
          acc_su += tcosi * bsubu_t[idx];
          acc_sv += tcosi * bsubv_t[idx];
          acc_pu += tcosi * bsupu_t[idx];
          acc_pv += tcosi * bsupv_t[idx];
        }
      }
      const int o = mn * nsH + jH;
      gmnc_t[o] = acc_g;
      bmnc_t[o] = acc_b;
      bsubumnc_t[o] = acc_su;
      bsubvmnc_t[o] = acc_sv;
      bsupumnc_t[o] = acc_pu;
      bsupvmnc_t[o] = acc_pv;
    }
  }
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_QS_HARMONICS_KERNEL_H_

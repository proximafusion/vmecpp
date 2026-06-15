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

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_QS_HARMONICS_KERNEL_H_

// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_CONSTRAINT_FORCE_KERNEL_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_CONSTRAINT_FORCE_KERNEL_H_

#include <algorithm>

namespace vmecpp {

// The two local (non-transform) pieces of VMEC's spectral-condensation
// constraint force. The Fourier-space bandpass between them is the separate
// shared free function deAliasConstraintForce. All full-grid buffers are
// indexed (jF-nsMinF)*nZnT except sqrtSF, which is indexed (jF-nsMinF1).
// Allocation-free, shared between the solver and the Enzyme autodiff path.

// Effective constraint force gConEff = (rCon - rCon0) ru + (zCon - zCon0) zu.
// No constraint on the axis (it has no poloidal angle), so the axis surface is
// skipped, matching deAliasConstraintForce.
inline void ComputeEffectiveConstraintForce(
    const double* rCon, const double* rCon0, const double* zCon,
    const double* zCon0, const double* ruFull, const double* zuFull, int nZnT,
    int nsMinF, int nsMaxFIncludingLcfs, double* gConEff) {
  int jMin = 0;
  if (nsMinF == 0) {
    jMin = 1;
  }
  for (int jF = std::max(jMin, nsMinF); jF < nsMaxFIncludingLcfs; ++jF) {
    for (int kl = 0; kl < nZnT; ++kl) {
      int idx_kl = (jF - nsMinF) * nZnT + kl;
      gConEff[idx_kl] = (rCon[idx_kl] - rCon0[idx_kl]) * ruFull[idx_kl] +
                        (zCon[idx_kl] - zCon0[idx_kl]) * zuFull[idx_kl];
    }  // kl
  }  // jF
}

// Add the bandpass-filtered constraint force gCon back into the MHD R/Z forces
// (brmn, bzmn) and write the constraint-force outputs (frcon, fzcon).
inline void AddConstraintForces(
    const double* rCon, const double* rCon0, const double* zCon,
    const double* zCon0, const double* ruFull, const double* zuFull,
    const double* gCon, const double* sqrtSF, int nZnT, int nsMinF, int nsMinF1,
    int nsMaxF, double* brmn_e, double* brmn_o, double* bzmn_e, double* bzmn_o,
    double* frcon_e, double* frcon_o, double* fzcon_e, double* fzcon_o) {
  for (int jF = nsMinF; jF < nsMaxF; ++jF) {
    for (int kl = 0; kl < nZnT; ++kl) {
      int idx_kl = (jF - nsMinF) * nZnT + kl;

      double brcon = (rCon[idx_kl] - rCon0[idx_kl]) * gCon[idx_kl];
      double bzcon = (zCon[idx_kl] - zCon0[idx_kl]) * gCon[idx_kl];

      brmn_e[idx_kl] += brcon;
      bzmn_e[idx_kl] += bzcon;
      brmn_o[idx_kl] += brcon * sqrtSF[jF - nsMinF1];
      bzmn_o[idx_kl] += bzcon * sqrtSF[jF - nsMinF1];

      frcon_e[idx_kl] = ruFull[idx_kl] * gCon[idx_kl];
      fzcon_e[idx_kl] = zuFull[idx_kl] * gCon[idx_kl];
      frcon_o[idx_kl] = frcon_e[idx_kl] * sqrtSF[jF - nsMinF1];
      fzcon_o[idx_kl] = fzcon_e[idx_kl] * sqrtSF[jF - nsMinF1];
    }  // kl
  }  // jF
}

// Fourier-space bandpass of the constraint force: forward transform gConEff to
// the (m, n) coefficients gsc/gcs scaled by tcon, then inverse transform back
// to real space scaled by faccon[m]. Bandpass keeps m in [1, mpol-1). The axis
// surface has no poloidal angle and is skipped. Allocation-free over flat
// buffers with explicit reductions (no Eigen temporaries), so it differentiates
// under Enzyme; the free function vmecpp::deAliasConstraintForce wraps this
// with the partition/basis structs. Basis layout: sinmui/cosmui/sinmu/cosmu
// indexed m*nThetaReduced + l; cosnv/sinnv indexed k*(nnyq2+1) + n;
// gConEff/gCon indexed
// ((jF-nsMinF)*nZeta + k)*nThetaEff + l; all Fourier scratch arrays have
// length ntor+1. In the non-symmetric case, gConAsym has nZeta*nThetaEff
// entries for one surface and refl has nThetaReduced entries.
inline void ComputeDeAliasConstraintForce(
    const double* gConEff, const double* faccon, const double* tcon,
    const double* sinmui, const double* cosmui, const double* cosnv,
    const double* sinnv, const double* sinmu, const double* cosmu, int nsMinF,
    int nsMaxF, int nZeta, int nThetaEff, int nThetaReduced, int nThetaEven,
    int mpol, int ntor, int nnyq2, bool lasym, double* m_gsc, double* m_gcs,
    double* m_gcc, double* m_gss, double* m_gConAsym, double* m_refl,
    double* m_gCon) {
  for (int i = 0; i < (nsMaxF - nsMinF) * nZeta * nThetaEff; ++i) {
    m_gCon[i] = 0.0;
  }
  const int jMin = (nsMinF == 0) ? 1 : 0;
  for (int jF = (jMin > nsMinF ? jMin : nsMinF); jF < nsMaxF; ++jF) {
    if (lasym) {
      for (int i = 0; i < nZeta * nThetaEff; ++i) {
        m_gConAsym[i] = 0.0;
      }
    }
    for (int m = 1; m < mpol - 1; ++m) {
      for (int n = 0; n < ntor + 1; ++n) {
        m_gsc[n] = 0.0;
        m_gcs[n] = 0.0;
        if (lasym) {
          m_gcc[n] = 0.0;
          m_gss[n] = 0.0;
        }
      }
      for (int k = 0; k < nZeta; ++k) {
        const int kl_base = ((jF - nsMinF) * nZeta + k) * nThetaEff;
        const int ml_base = m * nThetaReduced;
        double w0 = 0.0;
        double w1 = 0.0;
        for (int l = 0; l < nThetaReduced; ++l) {
          w0 += gConEff[kl_base + l] * sinmui[ml_base + l];
          w1 += gConEff[kl_base + l] * cosmui[ml_base + l];
        }
        const double tc = tcon[jF - nsMinF];
        if (!lasym) {
          for (int n = 0; n < ntor + 1; ++n) {
            const int idx_kn = k * (nnyq2 + 1) + n;
            m_gsc[n] += cosnv[idx_kn] * w0 * tc;
            m_gcs[n] += sinnv[idx_kn] * w1 * tc;
          }
        } else {
          const int k_rev = (nZeta - k) % nZeta;
          const int refl_base = ((jF - nsMinF) * nZeta + k_rev) * nThetaEff;
          for (int l = 0; l < nThetaReduced; ++l) {
            const int l_rev = (nThetaEven - l) % nThetaEven;
            m_refl[l] = gConEff[refl_base + l_rev];
          }
          double w3 = 0.0;
          double w4 = 0.0;
          for (int l = 0; l < nThetaReduced; ++l) {
            w3 += m_refl[l] * cosmui[ml_base + l];
            w4 += m_refl[l] * sinmui[ml_base + l];
          }
          for (int n = 0; n < ntor + 1; ++n) {
            const int idx_kn = k * (nnyq2 + 1) + n;
            const double cosnv_value = cosnv[idx_kn];
            const double sinnv_value = sinnv[idx_kn];
            m_gcs[n] += 0.5 * tc * sinnv_value * (w1 - w3);
            m_gsc[n] += 0.5 * tc * cosnv_value * (w0 - w4);
            m_gss[n] += 0.5 * tc * sinnv_value * (w0 + w4);
            m_gcc[n] += 0.5 * tc * cosnv_value * (w1 + w3);
          }
        }
      }

      for (int k = 0; k < nZeta; ++k) {
        const int kn_base = k * (nnyq2 + 1);
        double w0 = 0.0;
        double w1 = 0.0;
        double a0 = 0.0;
        double a1 = 0.0;
        for (int n = 0; n < ntor + 1; ++n) {
          w0 += m_gsc[n] * cosnv[kn_base + n];
          w1 += m_gcs[n] * sinnv[kn_base + n];
          if (lasym) {
            a0 += m_gcc[n] * cosnv[kn_base + n];
            a1 += m_gss[n] * sinnv[kn_base + n];
          }
        }
        for (int l = 0; l < nThetaReduced; ++l) {
          const int idx_kl = ((jF - nsMinF) * nZeta + k) * nThetaEff + l;
          const int idx_ml = m * nThetaReduced + l;
          m_gCon[idx_kl] +=
              faccon[m] * (w0 * sinmu[idx_ml] + w1 * cosmu[idx_ml]);
          if (lasym) {
            const int within = k * nThetaEff + l;
            m_gConAsym[within] +=
                faccon[m] * (a0 * cosmu[idx_ml] + a1 * sinmu[idx_ml]);
          }
        }
      }
    }
    if (lasym) {
      const int surface_base = (jF - nsMinF) * nZeta * nThetaEff;
      for (int k = 0; k < nZeta; ++k) {
        const int k_rev = (nZeta - k) % nZeta;
        for (int l = nThetaReduced; l < nThetaEven; ++l) {
          const int target = surface_base + k * nThetaEff + l;
          const int reflected = k_rev * nThetaEff + (nThetaEven - l);
          m_gCon[target] =
              -m_gCon[surface_base + reflected] + m_gConAsym[reflected];
        }
      }
      for (int k = 0; k < nZeta; ++k) {
        for (int l = 0; l < nThetaReduced; ++l) {
          const int within = k * nThetaEff + l;
          m_gCon[surface_base + within] += m_gConAsym[within];
        }
      }
    }
  }
}

// Symmetric convenience overload used by standalone tests and benchmarks.
inline void ComputeDeAliasConstraintForce(
    const double* gConEff, const double* faccon, const double* tcon,
    const double* sinmui, const double* cosmui, const double* cosnv,
    const double* sinnv, const double* sinmu, const double* cosmu, int nsMinF,
    int nsMaxF, int nZeta, int nThetaEff, int nThetaReduced, int mpol, int ntor,
    int nnyq2, double* m_gsc, double* m_gcs, double* m_gCon) {
  ComputeDeAliasConstraintForce(
      gConEff, faccon, tcon, sinmui, cosmui, cosnv, sinnv, sinmu, cosmu, nsMinF,
      nsMaxF, nZeta, nThetaEff, nThetaReduced, 2 * (nThetaReduced - 1), mpol,
      ntor, nnyq2, false, m_gsc, m_gcs, nullptr, nullptr, nullptr, nullptr,
      m_gCon);
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_CONSTRAINT_FORCE_KERNEL_H_

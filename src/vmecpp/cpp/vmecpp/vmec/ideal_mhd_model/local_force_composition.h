// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_LOCAL_FORCE_COMPOSITION_H_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_LOCAL_FORCE_COMPOSITION_H_

#include "vmecpp/vmec/ideal_mhd_model/bco_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/bcontra_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/constraint_force_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/jacobian_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/lambda_force_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/metric_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/mhdforce_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/pressure_kernel.h"

namespace vmecpp {

// Composition of the local force-density chain as a single allocation-free map
// g: real-space geometry -> real-space force density. This is the nonlinear
// core of VMEC's force; the spectral transforms around it are linear and
// applied separately. Shared between the Enzyme autodiff validation and the
// exact Hessian-vector product. Covers the MHD force and the hybrid lambda
// force; when with_constraint is set it also computes the spectral-condensation
// constraint force (effective force, Fourier bandpass, assembly into the R/Z
// force), holding the multiplier tcon frozen.
//
// Geometry layout (each block GeomStride doubles, index (jF-nsMinF1)*nZnT):
//   r1_e r1_o z1_e z1_o ru_e ru_o zu_e zu_o rv_e rv_o zv_e zv_o lu_e lu_o lv_e
//   lv_o
// Force layout (each block ForceStride doubles): the 12 MHD densities then
//   blmn_e blmn_o clmn_e clmn_o.
struct LocalForceComposition {
  int nZnT;
  int geom_stride;   // doubles per geometry block (>= (nsMaxF1-nsMinF1)*nZnT)
  int force_stride;  // doubles per force block (>=
                     // (nsMaxFIncludingLcfs-nsMinF)*nZnT)
  int nsMinF, nsMinF1, nsMinH, nsMaxH;
  int jMaxRZ;                    // MHD force surfaces: [nsMinF, jMaxRZ)
  int nsMaxFIncludingLcfs;       // lambda force surfaces: [nsMinF,
                                 // nsMaxFIncludingLcfs)
  const double* sqrtSF;          // index jF-nsMinF1
  const double* sqrtSH;          // index jH-nsMinH
  const double* chipH;           // index jH-nsMinH (frozen for ncurr==0)
  const double* presH;           // index jH-nsMinH
  const double* radialBlending;  // index jF-nsMinF1
  double deltaS;
  double dSHalfDsInterp;
  double lamscale;
  bool lthreed;

  // Constrained-current profile (ncurr==1): chi' is a function of geometry,
  // recomputed each step, so it is differentiated in place. ncurr==0 uses the
  // frozen chipH above.
  int ncurr = 0;
  int nThetaEff = 0;
  const double* currH = nullptr;  // index jH-nsMinH
  const double* wInt = nullptr;   // index kl % nThetaEff

  // Spectral-condensation constraint force. Enabled only when with_constraint
  // is set; then geometry blocks 16-19 hold rCon, zCon, ruFull, zuFull and
  // force blocks 16-19 receive frcon_e/o, fzcon_e/o. The bandpass uses the
  // Fourier basis arrays and the tcon/faccon profiles. rCon0/zCon0 are
  // recomputed in place from the live geometry (so they are differentiated);
  // tcon is held frozen (see freeze_constraint_multiplier).
  bool with_constraint = false;
  int nsMaxF = 0;  // constraint RZ range upper bound
  int nZeta = 0, nThetaReduced = 0, mpol = 0, ntor = 0, nnyq2 = 0;
  const double* rCon0 = nullptr;
  const double* zCon0 = nullptr;
  const double* faccon = nullptr;
  const double* tcon = nullptr;
  const double* sinmui = nullptr;
  const double* cosmui = nullptr;
  const double* cosnv = nullptr;
  const double* sinnv = nullptr;
  const double* sinmu = nullptr;
  const double* cosmu = nullptr;
};

// work must hold 15*nHalf + 30*nZnT doubles, where nHalf=(nsMaxH-nsMinH)*nZnT.
inline void ComputeLocalForceDensity(const double* geom, double* work,
                                     double* force,
                                     const LocalForceComposition* c) {
  const int nZnT = c->nZnT;
  const int gS = c->geom_stride;
  const int fS = c->force_stride;
  const int nH = (c->nsMaxH - c->nsMinH) * nZnT;
  const double* r1e = geom + 0 * gS;
  const double* r1o = geom + 1 * gS;
  const double* z1e = geom + 2 * gS;
  const double* z1o = geom + 3 * gS;
  const double* rue = geom + 4 * gS;
  const double* ruo = geom + 5 * gS;
  const double* zue = geom + 6 * gS;
  const double* zuo = geom + 7 * gS;
  const double* rve = geom + 8 * gS;
  const double* rvo = geom + 9 * gS;
  const double* zve = geom + 10 * gS;
  const double* zvo = geom + 11 * gS;
  const double* lue = geom + 12 * gS;
  const double* luo = geom + 13 * gS;
  const double* lve = geom + 14 * gS;
  const double* lvo = geom + 15 * gS;

  double* p = work;
  double* r12 = p;
  p += nH;
  double* ru12 = p;
  p += nH;
  double* zu12 = p;
  p += nH;
  double* rs = p;
  p += nH;
  double* zs = p;
  p += nH;
  double* tau = p;
  p += nH;
  double* gsqrt = p;
  p += nH;
  double* guu = p;
  p += nH;
  double* guv = p;
  p += nH;
  double* gvv = p;
  p += nH;
  double* bsupu = p;
  p += nH;
  double* bsupv = p;
  p += nH;
  double* bsubu = p;
  p += nH;
  double* bsubv = p;
  p += nH;
  double* tp = p;
  p += nH;
  double* s = p;  // 30 * nZnT

  ComputeHalfGridJacobian(r1e, r1o, z1e, z1o, rue, ruo, zue, zuo, c->sqrtSH,
                          c->deltaS, c->dSHalfDsInterp, nZnT, c->nsMinF1,
                          c->nsMinH, c->nsMaxH, r12, ru12, zu12, rs, zs, tau);
  ComputeMetricElements(r1e, r1o, rue, ruo, zue, zuo, rve, rvo, zve, zvo, tau,
                        r12, c->sqrtSF, c->sqrtSH, c->lthreed, nZnT, c->nsMinF1,
                        c->nsMinH, c->nsMaxH, gsqrt, guu, guv, gvv);
  ComputeBsupContra(lue, luo, lve, lvo, gsqrt, c->sqrtSH, c->lthreed, nZnT,
                    c->nsMinF1, c->nsMinH, c->nsMaxH, bsupu, bsupv);
  for (int jH = c->nsMinH; jH < c->nsMaxH; ++jH) {
    // For a prescribed-current profile (ncurr==1), chi' is recomputed from the
    // geometry each step (constrained toroidal current), so differentiate it
    // here rather than freezing it. For ncurr==0 chi' = iota*phi' is a fixed
    // profile, so use the frozen c->chipH.
    double chip = c->chipH[jH - c->nsMinH];
    if (c->ncurr == 1) {
      double jvPlasma = 0.0;
      double avg_guu_gsqrt = 0.0;
      for (int kl = 0; kl < nZnT; ++kl) {
        const int ih = (jH - c->nsMinH) * nZnT + kl;
        const int l = kl % c->nThetaEff;
        if (c->lthreed) {
          jvPlasma += (guu[ih] * bsupu[ih] + guv[ih] * bsupv[ih]) * c->wInt[l];
        } else {
          jvPlasma += guu[ih] * bsupu[ih] * c->wInt[l];
        }
        avg_guu_gsqrt += guu[ih] / gsqrt[ih] * c->wInt[l];
      }
      if (avg_guu_gsqrt != 0.0) {
        chip = (c->currH[jH - c->nsMinH] - jvPlasma) / avg_guu_gsqrt;
      }
    }
    for (int kl = 0; kl < nZnT; ++kl) {
      const int ih = (jH - c->nsMinH) * nZnT + kl;
      bsupu[ih] += chip / gsqrt[ih];
    }
  }
  ComputeBCo(guu, guv, gvv, bsupu, bsupv, c->lthreed, nH, bsubu, bsubv);
  ComputeMagneticPressure(bsupu, bsubu, bsupv, bsubv, nH, tp);
  for (int jH = c->nsMinH; jH < c->nsMaxH; ++jH) {
    for (int kl = 0; kl < nZnT; ++kl)
      tp[(jH - c->nsMinH) * nZnT + kl] += c->presH[jH - c->nsMinH];
  }

  double* P_i = s;
  s += nZnT;
  double* rup_i = s;
  s += nZnT;
  double* zup_i = s;
  s += nZnT;
  double* rsp_i = s;
  s += nZnT;
  double* zsp_i = s;
  s += nZnT;
  double* taup_i = s;
  s += nZnT;
  double* gbubu_i = s;
  s += nZnT;
  double* gbubv_i = s;
  s += nZnT;
  double* gbvbv_i = s;
  s += nZnT;
  double* P_o = s;
  s += nZnT;
  double* rup_o = s;
  s += nZnT;
  double* zup_o = s;
  s += nZnT;
  double* rsp_o = s;
  s += nZnT;
  double* zsp_o = s;
  s += nZnT;
  double* taup_o = s;
  s += nZnT;
  double* gbubu_o = s;
  s += nZnT;
  double* gbubv_o = s;
  s += nZnT;
  double* gbvbv_o = s;
  s += nZnT;
  double* P_avg = s;
  s += nZnT;
  double* P_wavg = s;
  s += nZnT;
  double* gbubu_avg = s;
  s += nZnT;
  double* gbubu_wavg = s;
  s += nZnT;
  double* gbvbv_avg = s;
  s += nZnT;
  double* gbvbv_wavg = s;
  s += nZnT;
  double* gbubv_avg = s;
  s += nZnT;
  double* gbubv_wavg = s;
  s += nZnT;
  double* bsubu_i = s;
  s += nZnT;
  double* bsubv_i = s;
  s += nZnT;
  double* gvv_gsqrt_i = s;
  s += nZnT;
  double* guv_bsupu_i = s;
  s += nZnT;

  double* armn_e = force + 0 * fS;
  double* armn_o = force + 1 * fS;
  double* azmn_e = force + 2 * fS;
  double* azmn_o = force + 3 * fS;
  double* brmn_e = force + 4 * fS;
  double* brmn_o = force + 5 * fS;
  double* bzmn_e = force + 6 * fS;
  double* bzmn_o = force + 7 * fS;
  double* crmn_e = force + 8 * fS;
  double* crmn_o = force + 9 * fS;
  double* czmn_e = force + 10 * fS;
  double* czmn_o = force + 11 * fS;
  ComputeMHDForceDensity(
      r1e, r1o, rue, ruo, zue, zuo, z1o, rve, rvo, zve, zvo, r12, ru12, zu12,
      rs, zs, tau, tp, gsqrt, bsupu, bsupv, c->sqrtSF, c->sqrtSH, P_i, rup_i,
      zup_i, rsp_i, zsp_i, taup_i, gbubu_i, gbubv_i, gbvbv_i, P_o, rup_o, zup_o,
      rsp_o, zsp_o, taup_o, gbubu_o, gbubv_o, gbvbv_o, P_avg, P_wavg, gbubu_avg,
      gbubu_wavg, gbvbv_avg, gbvbv_wavg, gbubv_avg, gbubv_wavg, c->deltaS, nZnT,
      c->nsMinF, c->nsMinF1, c->nsMinH, c->nsMaxH, c->jMaxRZ, c->lthreed,
      armn_e, armn_o, azmn_e, azmn_o, brmn_e, brmn_o, bzmn_e, bzmn_o, crmn_e,
      crmn_o, czmn_e, czmn_o);

  double* blmn_e = force + 12 * fS;
  double* blmn_o = force + 13 * fS;
  double* clmn_e = force + 14 * fS;
  double* clmn_o = force + 15 * fS;
  ComputeHybridLambdaForce(
      bsubu, bsubv, gvv, gsqrt, guv, bsupu, lue, luo, c->sqrtSH, c->sqrtSF,
      c->radialBlending, c->lamscale, c->lthreed, nZnT, c->nsMinF, c->nsMinF1,
      c->nsMinH, c->nsMaxH, c->nsMaxFIncludingLcfs, bsubu_i, bsubv_i,
      gvv_gsqrt_i, guv_bsupu_i, blmn_e, blmn_o, clmn_e, clmn_o);

  if (c->with_constraint) {
    // geometry blocks 16-19 carry the constraint coordinates and full-grid
    // derivatives; force blocks 16-19 receive the constraint outputs.
    const double* rCon = geom + 16 * gS;
    const double* zCon = geom + 17 * gS;
    const double* ruFull = geom + 18 * gS;
    const double* zuFull = geom + 19 * gS;
    double* gConEff = s;
    s += (c->nsMaxFIncludingLcfs - c->nsMinF) * nZnT;
    double* gCon = s;
    s += (c->nsMaxF - c->nsMinF) * nZnT;
    double* gsc = s;
    s += c->ntor + 1;
    double* gcs = s;
    s += c->ntor + 1;
    // Constraint reference rCon0/zCon0 extrapolated from the LCFS into the
    // volume (rzConIntoVolume): rCon0[jF] = rCon[LCFS] * s_full. This is linear
    // in the geometry, so computing it here (rather than freezing it) keeps the
    // exact HVP consistent with re-evaluating rzConIntoVolume each step.
    double* rCon0 = s;
    s += (c->nsMaxFIncludingLcfs - c->nsMinF) * nZnT;
    double* zCon0 = s;  // last slice of the work buffer
    const int lcfs = (c->nsMaxFIncludingLcfs - 1 - c->nsMinF) * nZnT;
    for (int jF = (c->nsMinF > 1 ? c->nsMinF : 1); jF < c->nsMaxFIncludingLcfs;
         ++jF) {
      const double sf = c->sqrtSF[jF - c->nsMinF1] * c->sqrtSF[jF - c->nsMinF1];
      for (int kl = 0; kl < nZnT; ++kl) {
        const int idx = (jF - c->nsMinF) * nZnT + kl;
        rCon0[idx] = rCon[lcfs + kl] * sf;
        zCon0[idx] = zCon[lcfs + kl] * sf;
      }
    }
    ComputeEffectiveConstraintForce(rCon, rCon0, zCon, zCon0, ruFull, zuFull,
                                    nZnT, c->nsMinF, c->nsMaxFIncludingLcfs,
                                    gConEff);
    ComputeDeAliasConstraintForce(
        gConEff, c->faccon, c->tcon, c->sinmui, c->cosmui, c->cosnv, c->sinnv,
        c->sinmu, c->cosmu, c->nsMinF, c->nsMaxF, c->nZeta, c->nThetaEff,
        c->nThetaReduced, c->mpol, c->ntor, c->nnyq2, gsc, gcs, gCon);
    double* frcon_e = force + 16 * fS;
    double* frcon_o = force + 17 * fS;
    double* fzcon_e = force + 18 * fS;
    double* fzcon_o = force + 19 * fS;
    AddConstraintForces(rCon, rCon0, zCon, zCon0, ruFull, zuFull, gCon,
                        c->sqrtSF, nZnT, c->nsMinF, c->nsMinF1, c->nsMaxF,
                        brmn_e, brmn_o, bzmn_e, bzmn_o, frcon_e, frcon_o,
                        fzcon_e, fzcon_o);
  }
}

}  // namespace vmecpp

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_LOCAL_FORCE_COMPOSITION_H_

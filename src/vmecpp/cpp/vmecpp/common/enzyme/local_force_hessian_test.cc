// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Exact Hessian of VMEC's local force map via composed autodiff.
//
// The shared force-chain kernels (Jacobian, metric, B^contra, B_cov, magnetic
// pressure, MHD force density, and the hybrid lambda force) compose into the
// local map
//   g : real-space geometry -> real-space force density,
// the nonlinear core of VMEC's force. The full force is Tᵀ . g . T with the
// linear spectral transforms T, Tᵀ; the exact force Hessian-vector product is
// therefore Tᵀ . J_g . T . v, and J_g is what Enzyme provides here. The
// remaining augmented term, the spectral-condensation constraint force, also
// carries a linear Fourier bandpass and is validated end-to-end against the
// finite-difference HVP in the pybind exact-HVP path.
//
// This test composes the production kernels over flat buffers and takes the
// Jacobian of g by forward and reverse mode, checks both against central finite
// differences and against each other.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "vmecpp/common/enzyme/enzyme.h"
#include "vmecpp/vmec/ideal_mhd_model/bco_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/bcontra_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/jacobian_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/lambda_force_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/metric_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/mhdforce_kernel.h"
#include "vmecpp/vmec/ideal_mhd_model/pressure_kernel.h"

// Problem dimensions and the constant (non-differentiated) context.
struct Ctx {
  int nZnT, nsH;                 // half-grid surfaces; full-grid has nsH + 1
  int nFull, nHalf;              // (nsH+1)*nZnT, nsH*nZnT
  const double* sqrtSF;          // [nsH+1]
  const double* sqrtSH;          // [nsH]
  const double* chipH;           // [nsH]
  const double* presH;           // [nsH]
  const double* radialBlending;  // [nsH+1]
  double deltaS;
  double lamscale;
  bool lthreed;
};

// 16 geometry blocks (each nFull) packed into geom; force blocks (each nHalf):
// 12 MHD force-density blocks + 4 lambda-force blocks (blmn_e/o, clmn_e/o).
// Everything else is intermediate work sliced from work.
enum { kGeomBlocks = 16, kForceBlocks = 16 };

// g: geometry -> force density, composing the MHD and lambda-force kernels.
__attribute__((noinline)) void LocalForce(const double* geom, double* work,
                                          double* force, const Ctx* c) {
  const int nF = c->nFull;
  const int nH = c->nHalf;
  const int nZnT = c->nZnT, nsH = c->nsH;
  // geometry blocks
  const double* r1e = geom + 0 * nF;
  const double* r1o = geom + 1 * nF;
  const double* z1e = geom + 2 * nF;
  const double* z1o = geom + 3 * nF;
  const double* rue = geom + 4 * nF;
  const double* ruo = geom + 5 * nF;
  const double* zue = geom + 6 * nF;
  const double* zuo = geom + 7 * nF;
  const double* rve = geom + 8 * nF;
  const double* rvo = geom + 9 * nF;
  const double* zve = geom + 10 * nF;
  const double* zvo = geom + 11 * nF;
  const double* lue = geom + 12 * nF;
  const double* luo = geom + 13 * nF;
  const double* lve = geom + 14 * nF;
  const double* lvo = geom + 15 * nF;
  // half-grid intermediates from work
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
  // per-nZnT scratch for the force kernel (26 blocks)
  double* sc = p;  // 26 * nZnT

  vmecpp::ComputeHalfGridJacobian(
      r1e, r1o, z1e, z1o, rue, ruo, zue, zuo, c->sqrtSH, c->deltaS,
      /*dSHalfDsInterp=*/0.25, nZnT, 0, 0, nsH, r12, ru12, zu12, rs, zs, tau);
  vmecpp::ComputeMetricElements(r1e, r1o, rue, ruo, zue, zuo, rve, rvo, zve,
                                zvo, tau, r12, c->sqrtSF, c->sqrtSH, c->lthreed,
                                nZnT, 0, 0, nsH, gsqrt, guu, guv, gvv);
  vmecpp::ComputeBsupContra(lue, luo, lve, lvo, gsqrt, c->sqrtSH, c->lthreed,
                            nZnT, 0, 0, nsH, bsupu, bsupv);
  for (int jH = 0; jH < nsH; ++jH) {
    for (int kl = 0; kl < nZnT; ++kl) {
      const int ih = jH * nZnT + kl;
      bsupu[ih] += c->chipH[jH] / gsqrt[ih];
    }
  }
  vmecpp::ComputeBCo(guu, guv, gvv, bsupu, bsupv, c->lthreed, nH, bsubu, bsubv);
  vmecpp::ComputeMagneticPressure(bsupu, bsubu, bsupv, bsubv, nH, tp);
  for (int jH = 0; jH < nsH; ++jH) {
    for (int kl = 0; kl < nZnT; ++kl) tp[jH * nZnT + kl] += c->presH[jH];
  }
  double* s = sc;
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
  // lambda-force radial-sweep scratch (carried inside half-grid point)
  double* bsubu_i = s;
  s += nZnT;
  double* bsubv_i = s;
  s += nZnT;
  double* gvv_gsqrt_i = s;
  s += nZnT;
  double* guv_bsupu_i = s;
  s += nZnT;
  double* armn_e = force + 0 * nH;
  double* armn_o = force + 1 * nH;
  double* azmn_e = force + 2 * nH;
  double* azmn_o = force + 3 * nH;
  double* brmn_e = force + 4 * nH;
  double* brmn_o = force + 5 * nH;
  double* bzmn_e = force + 6 * nH;
  double* bzmn_o = force + 7 * nH;
  double* crmn_e = force + 8 * nH;
  double* crmn_o = force + 9 * nH;
  double* czmn_e = force + 10 * nH;
  double* czmn_o = force + 11 * nH;
  vmecpp::ComputeMHDForceDensity(
      r1e, r1o, rue, ruo, zue, zuo, z1o, rve, rvo, zve, zvo, r12, ru12, zu12,
      rs, zs, tau, tp, gsqrt, bsupu, bsupv, c->sqrtSF, c->sqrtSH, P_i, rup_i,
      zup_i, rsp_i, zsp_i, taup_i, gbubu_i, gbubv_i, gbvbv_i, P_o, rup_o, zup_o,
      rsp_o, zsp_o, taup_o, gbubu_o, gbubv_o, gbvbv_o, P_avg, P_wavg, gbubu_avg,
      gbubu_wavg, gbvbv_avg, gbvbv_wavg, gbubv_avg, gbubv_wavg, c->deltaS, nZnT,
      /*nsMinF=*/0, 0, 0, nsH, /*jMaxRZ=*/nsH, c->lthreed, armn_e, armn_o,
      azmn_e, azmn_o, brmn_e, brmn_o, bzmn_e, bzmn_o, crmn_e, crmn_o, czmn_e,
      czmn_o);
  // lambda force (blmn_e/o, clmn_e/o) from the shared kernel
  double* blmn_e = force + 12 * nH;
  double* blmn_o = force + 13 * nH;
  double* clmn_e = force + 14 * nH;
  double* clmn_o = force + 15 * nH;
  vmecpp::ComputeHybridLambdaForce(
      bsubu, bsubv, gvv, gsqrt, guv, bsupu, lue, luo, c->sqrtSH, c->sqrtSF,
      c->radialBlending, c->lamscale, c->lthreed, nZnT, /*nsMinF=*/0,
      /*nsMinF1=*/0, /*nsMinH=*/0, nsH, /*nsMaxFIncludingLcfs=*/nsH, bsubu_i,
      bsubv_i, gvv_gsqrt_i, guv_bsupu_i, blmn_e, blmn_o, clmn_e, clmn_o);
}

// Scalar objective L = 0.5 ||force||^2; work and force are caller-owned
// scratch.
__attribute__((noinline)) double Loss(const double* geom, double* work,
                                      double* force, const Ctx* c) {
  LocalForce(geom, work, force, c);
  const int n = kForceBlocks * c->nHalf;
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += 0.5 * force[i] * force[i];
  return s;
}

int main() {
  const int nZnT = 24, nsH = 8;
  Ctx c;
  c.nZnT = nZnT;
  c.nsH = nsH;
  c.nFull = (nsH + 1) * nZnT;
  c.nHalf = nsH * nZnT;
  c.deltaS = 0.1;
  c.lamscale = 0.7;
  c.lthreed = true;
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> d(0.5, 1.5), s(-1.0, 1.0);
  std::vector<double> sqrtSF(nsH + 1), sqrtSH(nsH), chipH(nsH), presH(nsH),
      radialBlending(nsH + 1);
  for (int j = 0; j <= nsH; ++j) {
    sqrtSF[j] = std::sqrt(0.05 + 0.9 * j / nsH);
    radialBlending[j] = 0.3 + 0.4 * j / nsH;
  }
  for (int j = 0; j < nsH; ++j) {
    sqrtSH[j] = std::sqrt(0.05 + 0.9 * (j + 0.5) / nsH);
    chipH[j] = 0.3 + 0.1 * j;
    presH[j] = 0.2 + 0.05 * j;
  }
  c.sqrtSF = sqrtSF.data();
  c.sqrtSH = sqrtSH.data();
  c.chipH = chipH.data();
  c.presH = presH.data();
  c.radialBlending = radialBlending.data();

  const int nG = kGeomBlocks * c.nFull;
  const int nW = 15 * c.nHalf + 30 * nZnT;
  const int nFc = kForceBlocks * c.nHalf;
  std::vector<double> geom(nG), v(nG);
  for (double& x : geom) x = d(rng);
  for (double& x : v) x = s(rng);
  std::vector<double> work(nW, 0.0), dwork(nW, 0.0), force(nFc, 0.0),
      dforce(nFc, 0.0), grev(nG, 0.0);

  // Reverse mode: full gradient dL/dgeom.
  __enzyme_autodiff((void*)Loss, enzyme_dup, geom.data(), grev.data(),
                    enzyme_dup, work.data(), dwork.data(), enzyme_dup,
                    force.data(), dforce.data(), enzyme_const, &c);
  // Forward mode: directional derivative dL.v.
  const double dfwd = __enzyme_fwddiff<double>(
      (void*)Loss, enzyme_dup, geom.data(), v.data(), enzyme_dup, work.data(),
      dwork.data(), enzyme_dup, force.data(), dforce.data(), enzyme_const, &c);

  const double h = 1e-6;
  std::vector<double> gp(geom), gm(geom), w2(nW), f2(nFc);
  for (int i = 0; i < nG; ++i) {
    gp[i] = geom[i] + h * v[i];
    gm[i] = geom[i] - h * v[i];
  }
  const double dfd = (Loss(gp.data(), w2.data(), f2.data(), &c) -
                      Loss(gm.data(), w2.data(), f2.data(), &c)) /
                     (2 * h);
  const double drev =
      std::inner_product(grev.begin(), grev.end(), v.begin(), 0.0);
  const double scale = std::fabs(dfd) + 1e-300;

  printf("exact Hessian of VMEC local force map (MHD + lambda kernels)\n");
  printf("  geom dofs=%d  force outputs=%d\n", nG, nFc);
  printf("  reverse dL.v vs finite-diff : %.2e\n",
         std::fabs(drev - dfd) / scale);
  printf("  forward dL.v vs finite-diff : %.2e\n",
         std::fabs(dfwd - dfd) / scale);
  printf("  forward / reverse agreement : %.2e\n",
         std::fabs(dfwd - drev) / (std::fabs(drev) + 1e-300));

  const bool ok = std::fabs(drev - dfd) < 1e-5 * scale &&
                  std::fabs(dfwd - dfd) < 1e-5 * scale &&
                  std::fabs(dfwd - drev) < 1e-9 * (std::fabs(drev) + 1e-300);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

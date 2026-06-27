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
#include "vmecpp/vmec/ideal_mhd_model/exact_force_jvp.h"
#include "vmecpp/vmec/ideal_mhd_model/local_force_composition.h"

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
  vmecpp::LocalForceComposition lc;
  lc.nZnT = c->nZnT;
  lc.geom_stride = c->nFull;
  lc.force_stride = c->nHalf;
  lc.nsMinF = 0;
  lc.nsMinF1 = 0;
  lc.nsMinH = 0;
  lc.nsMaxH = c->nsH;
  lc.jMaxRZ = c->nsH;
  lc.nsMaxFIncludingLcfs = c->nsH;
  lc.sqrtSF = c->sqrtSF;
  lc.sqrtSH = c->sqrtSH;
  lc.chipH = c->chipH;
  lc.presH = c->presH;
  lc.radialBlending = c->radialBlending;
  lc.deltaS = c->deltaS;
  lc.dSHalfDsInterp = 0.25;
  lc.lamscale = c->lamscale;
  lc.lthreed = c->lthreed;
  vmecpp::ComputeLocalForceDensity(geom, work, force, &lc);
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

  // Validate the standalone plugin JVP wrapper (exact_force_jvp.cc, the same
  // entry point the exact Hessian-vector product calls) against a finite
  // difference of the composition's force-density output.
  vmecpp::LocalForceComposition lc;
  lc.nZnT = c.nZnT;
  lc.geom_stride = c.nFull;
  lc.force_stride = c.nHalf;
  lc.nsMinF = 0;
  lc.nsMinF1 = 0;
  lc.nsMinH = 0;
  lc.nsMaxH = c.nsH;
  lc.jMaxRZ = c.nsH;
  lc.nsMaxFIncludingLcfs = c.nsH;
  lc.sqrtSF = c.sqrtSF;
  lc.sqrtSH = c.sqrtSH;
  lc.chipH = c.chipH;
  lc.presH = c.presH;
  lc.radialBlending = c.radialBlending;
  lc.deltaS = c.deltaS;
  lc.dSHalfDsInterp = 0.25;
  lc.lamscale = c.lamscale;
  lc.lthreed = c.lthreed;
  std::vector<double> jf(nFc, 0.0), jdf(nFc, 0.0), jw(nW, 0.0), jdw(nW, 0.0);
  vmecpp::ExactForceDensityJvp(geom.data(), v.data(), jw.data(), jdw.data(),
                               jf.data(), jdf.data(), &lc);
  std::vector<double> fpf(nFc, 0.0), fmf(nFc, 0.0), fw(nW, 0.0);
  vmecpp::ComputeLocalForceDensity(gp.data(), fw.data(), fpf.data(), &lc);
  vmecpp::ComputeLocalForceDensity(gm.data(), fw.data(), fmf.data(), &lc);
  double jvp_err = 0.0, jvp_scale = 1e-300;
  for (int i = 0; i < nFc; ++i) {
    const double fd = (fpf[i] - fmf[i]) / (2 * h);
    jvp_err = std::max(jvp_err, std::fabs(jdf[i] - fd));
    jvp_scale = std::max(jvp_scale, std::fabs(fd));
  }
  printf("  plugin JVP wrapper vs finite-diff : %.2e\n", jvp_err / jvp_scale);

  const bool ok = std::fabs(drev - dfd) < 1e-5 * scale &&
                  std::fabs(dfwd - dfd) < 1e-5 * scale &&
                  std::fabs(dfwd - drev) < 1e-9 * (std::fabs(drev) + 1e-300) &&
                  jvp_err < 1e-5 * jvp_scale;
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

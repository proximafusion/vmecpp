// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Exact autodiff of the QS field-harmonics kernel, forward vs reverse mode.
//
// ComputeQsHarmonics (qs_harmonics_kernel.h) maps the half-grid real-space
// fields the force chain produces (gsqrt, total_pressure, presH, bsupu, bsupv,
// bsubu, bsubv) to the Fourier harmonics SIMSOPT's QuasisymmetryRatioResidual
// consumes (gmnc, bmnc, bsub{u,v}mnc, bsup{u,v}mnc). A scalar quasisymmetry
// objective is a smooth function of those harmonics, so its sensitivity to the
// state is the chain of this kernel's Jacobian with the (already exact) force
// chain. This is the piece that lets the QS adjoint drop the finite-difference
// objective_state_gradient: the gradient w.r.t. all field inputs comes from a
// single reverse pass, FD-free.
//
// The kernel is allocation-free over flat buffers, the form Enzyme
// differentiates. We take:
//   * a Jacobian-vector product  J v   by forward mode  (__enzyme_fwddiff), and
//   * the full gradient          J^T u by reverse mode (__enzyme_autodiff),
// check both against central finite differences and each other (adjoint
// identity), and time the reverse pass against the finite-difference full
// gradient (the legacy path) so the speedup is explicit.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "vmecpp/common/enzyme/enzyme.h"
#include "vmecpp/vmec/ideal_mhd_model/qs_harmonics_kernel.h"

// Types used as Enzyme intrinsic arguments need external linkage (an anonymous
// namespace type cannot instantiate the __enzyme_* templates).

// Synthetic geometry/basis: the AD check tests d(harmonics)/d(fields) with the
// Fourier basis held constant, so the basis only needs to be in-bounds and
// finite. A real cos/sin grid keeps the kernel well conditioned.
struct Problem {
  int npts, nsH, nh, n_in, n_out;
  vmecpp::QsHarmonicsConfig c;
  std::vector<int> xm_nyq, xn_nyq;
  std::vector<double> mscale, nscale, cosmui, sinmui, cosnv, sinnv;
};

Problem MakeProblem() {
  const int nsH = 12, nZeta = 8, nThetaR = 9, nThetaEff = nThetaR;
  const int nfp = 1, mnyq = 4, nnyq = 3, nnyq2 = 3;
  Problem p{};
  p.nsH = nsH;
  p.npts = nsH * nZeta * nThetaEff;

  // Nyquist mode enumeration (m, n): m in [0, mnyq], n in [-nnyq, nnyq],
  // dropping the n<0 half of m==0, as VMEC lays out the Nyquist harmonics.
  for (int m = 0; m <= mnyq; ++m)
    for (int n = -nnyq; n <= nnyq; ++n) {
      if (m == 0 && n < 0) continue;
      p.xm_nyq.push_back(m);
      p.xn_nyq.push_back(n * nfp);
    }
  const int mnmax_nyq = static_cast<int>(p.xm_nyq.size());
  p.nh = mnmax_nyq * nsH;
  p.n_in = 6 * p.npts + nsH;  // 6 full fields + presH (length nsH)
  p.n_out = 6 * p.nh;

  const double pi = std::acos(-1.0);
  p.mscale.assign(mnyq + 1, std::sqrt(2.0));
  p.mscale[0] = 1.0;
  p.nscale.assign(nnyq2 + 1, std::sqrt(2.0));
  p.nscale[0] = 1.0;
  p.cosmui.resize(nThetaR * (mnyq + 1));
  p.sinmui.resize(nThetaR * (mnyq + 1));
  for (int m = 0; m <= mnyq; ++m)
    for (int l = 0; l < nThetaR; ++l) {
      const double th = pi * l / (nThetaR - 1);
      p.cosmui[m * nThetaR + l] = std::cos(m * th) * (2.0 / (nThetaR - 1));
      p.sinmui[m * nThetaR + l] = std::sin(m * th) * (2.0 / (nThetaR - 1));
    }
  const int nnv = nnyq2 + 1;
  p.cosnv.resize(nZeta * nnv);
  p.sinnv.resize(nZeta * nnv);
  for (int k = 0; k < nZeta; ++k)
    for (int n = 0; n < nnv; ++n) {
      const double ze = 2.0 * pi * k / nZeta;
      p.cosnv[k * nnv + n] = std::cos(n * ze) / nZeta;
      p.sinnv[k * nnv + n] = std::sin(n * ze) / nZeta;
    }

  p.c.nsH = nsH;
  p.c.nZeta = nZeta;
  p.c.nThetaReduced = nThetaR;
  p.c.nThetaEff = nThetaEff;
  p.c.nnyq2 = nnyq2;
  p.c.mnyq = mnyq;
  p.c.nnyq = nnyq;
  p.c.mnmax_nyq = mnmax_nyq;
  p.c.tmult = 0.5;
  p.c.xm_nyq = p.xm_nyq.data();
  p.c.xn_nyq = p.xn_nyq.data();
  p.c.nfp = nfp;
  p.c.mscale = p.mscale.data();
  p.c.nscale = p.nscale.data();
  p.c.cosmui = p.cosmui.data();
  p.c.sinmui = p.sinmui.data();
  p.c.cosnv = p.cosnv.data();
  p.c.sinnv = p.sinnv.data();
  return p;
}

// Initial field vector. modB = sqrt(2*(total_pressure - presH)) must stay
// smooth and real, so keep total_pressure comfortably above presH everywhere.
std::vector<double> MakeFields(const Problem& p) {
  std::vector<double> f(p.n_in);
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> d(0.5, 1.5);
  const int np = p.npts;
  for (int i = 0; i < np; ++i) f[i] = d(rng);             // gsqrt
  for (int i = 0; i < np; ++i) f[np + i] = 2.0 + d(rng);  // total_pressure
  for (int i = 2 * np; i < 6 * np; ++i) f[i] = d(rng);    // bsup/bsub u,v
  for (int j = 0; j < p.nsH; ++j) f[6 * np + j] = 0.2 + 0.1 * d(rng);  // presH
  return f;
}

// Evaluate the kernel from the packed field buffer into the packed harmonics
// buffer. Layout: [gsqrt | total_pressure | bsupu | bsupv | bsubu | bsubv |
// presH], each full field npts long, presH nsH long.
__attribute__((noinline)) void Eval(const double* fields, const Problem& p,
                                    double* harm) {
  const int np = p.npts;
  vmecpp::ComputeQsHarmonics(
      fields, fields + np, fields + 6 * np, fields + 2 * np, fields + 3 * np,
      fields + 4 * np, fields + 5 * np, harm, harm + p.nh, harm + 2 * p.nh,
      harm + 3 * p.nh, harm + 4 * p.nh, harm + 5 * p.nh, &p.c);
}

// Scalar QS-like objective L(fields) = 0.5 * sum of squares of all harmonics.
// harm is caller-owned scratch (an intermediate of the differentiated call).
__attribute__((noinline)) double Loss(const double* fields, double* harm,
                                      const Problem* p) {
  Eval(fields, *p, harm);
  double s = 0.0;
  for (int i = 0; i < p->n_out; ++i) s += 0.5 * harm[i] * harm[i];
  return s;
}

int main() {
  const Problem p = MakeProblem();
  const std::vector<double> fields = MakeFields(p);
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> v(p.n_in);
  for (double& x : v) x = dist(rng);

  // Reverse mode: full gradient dL/dfields in one pass (FD-free).
  std::vector<double> g_rev(p.n_in, 0.0), harm_s(p.n_out, 0.0),
      harm_sh(p.n_out, 0.0);
  __enzyme_autodiff((void*)Loss, enzyme_dup, fields.data(), g_rev.data(),
                    enzyme_dup, harm_s.data(), harm_sh.data(), enzyme_const,
                    &p);

  // Forward mode: directional derivative dL . v in one pass.
  std::vector<double> harm_t(p.n_out, 0.0);
  const double d_fwd = __enzyme_fwddiff<double>(
      (void*)Loss, enzyme_dup, fields.data(), v.data(), enzyme_dup,
      harm_s.data(), harm_t.data(), enzyme_const, &p);

  // Central finite differences of the directional derivative (correctness ref).
  const double h = 1e-6;
  std::vector<double> fp(fields), fm(fields), hs(p.n_out);
  for (int i = 0; i < p.n_in; ++i) {
    fp[i] = fields[i] + h * v[i];
    fm[i] = fields[i] - h * v[i];
  }
  const double d_fd =
      (Loss(fp.data(), hs.data(), &p) - Loss(fm.data(), hs.data(), &p)) /
      (2 * h);
  const double d_rev =
      std::inner_product(g_rev.begin(), g_rev.end(), v.begin(), 0.0);

  const double scale = std::fabs(d_fd) + 1e-300;
  printf("exact autodiff of QS harmonics kernel (n_in=%d n_out=%d)\n", p.n_in,
         p.n_out);
  printf("  reverse  dL.v = %.10e   (rel err vs FD %.2e)\n", d_rev,
         std::fabs(d_rev - d_fd) / scale);
  printf("  forward  dL.v = %.10e   (rel err vs FD %.2e)\n", d_fwd,
         std::fabs(d_fwd - d_fd) / scale);
  printf("  forward/reverse agreement: %.2e\n",
         std::fabs(d_fwd - d_rev) / (std::fabs(d_rev) + 1e-300));

  // Performance: the legacy path forms the full field-gradient by finite
  // differences (2*n_in kernel evaluations); the exact path is a single reverse
  // pass. Time both so the speedup at this problem size is explicit.
  const int reps = 200;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    std::fill(g_rev.begin(), g_rev.end(), 0.0);
    __enzyme_autodiff((void*)Loss, enzyme_dup, fields.data(), g_rev.data(),
                      enzyme_dup, harm_s.data(), harm_sh.data(), enzyme_const,
                      &p);
  }
  auto t1 = std::chrono::steady_clock::now();
  std::vector<double> g_fd(p.n_in), fpe(fields), fme(fields), hsa(p.n_out),
      hsb(p.n_out);
  for (int i = 0; i < p.n_in; ++i) {
    fpe[i] = fields[i] + h;
    fme[i] = fields[i] - h;
    g_fd[i] =
        (Loss(fpe.data(), hsa.data(), &p) - Loss(fme.data(), hsb.data(), &p)) /
        (2 * h);
    fpe[i] = fields[i];
    fme[i] = fields[i];
  }
  auto t2 = std::chrono::steady_clock::now();
  const double us_rev =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
  const double us_fd =
      std::chrono::duration<double, std::micro>(t2 - t1).count();
  // Agreement of the reverse full gradient with the FD full gradient.
  double g_relerr = 0.0, g_scale = 0.0;
  for (int i = 0; i < p.n_in; ++i) {
    g_relerr = std::fmax(g_relerr, std::fabs(g_rev[i] - g_fd[i]));
    g_scale = std::fmax(g_scale, std::fabs(g_fd[i]));
  }
  g_relerr /= (g_scale + 1e-300);
  printf("  full-gradient reverse vs FD: rel err %.2e\n", g_relerr);
  printf(
      "  performance: reverse %.2f us/pass (full gradient), FD full gradient "
      "%.1f us  -> %.0fx speedup\n",
      us_rev, us_fd, us_fd / us_rev);

  const bool ok =
      std::fabs(d_rev - d_fd) < 1e-5 * scale &&
      std::fabs(d_fwd - d_fd) < 1e-5 * scale &&
      std::fabs(d_fwd - d_rev) < 1e-9 * (std::fabs(d_rev) + 1e-300) &&
      g_relerr < 1e-5;
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

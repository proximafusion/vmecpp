// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Exact autodiff of a real VMEC nonlinear kernel, forward vs reverse mode.
//
// JacobianKernel reproduces IdealMhdModel::computeJacobian: it maps the
// full-grid geometry (R, Z and their poloidal derivatives, even/odd parity) to
// the half-grid metric quantities r12, ru12, zu12, rs, zs and the Jacobian tau.
// tau is nonlinear in the geometry (products ru12*zs - rs*zu12, ...), so its
// Jacobian is a genuine building block of the MHD force Hessian (the force is
// the gradient of VMEC's functional; chain rule composes this kernel's Jacobian
// with the linear spectral transforms to give the Hessian-vector product).
//
// The kernel is written allocation-free over flat buffers (scalar locals,
// Eigen-free), which is the form Enzyme differentiates. We take:
//   * a Jacobian-vector product  J v   by forward mode  (__enzyme_fwddiff), and
//   * a vector-Jacobian product  J^T u  by reverse mode (__enzyme_autodiff),
// check both against central finite differences, and time them so the
// forward/reverse trade-off is explicit.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "vmecpp/common/enzyme/enzyme.h"

// Types used as Enzyme intrinsic arguments need external linkage (an anonymous
// namespace type cannot instantiate the __enzyme_* templates).

// Half-grid Jacobian kernel. Geometry inputs are (nsH+1) full-grid surfaces of
// nZnT angular points each; outputs are nsH half-grid surfaces. Transcribed
// from IdealMhdModel::computeJacobian (direct full-grid reads instead of the
// in/out handover; identical arithmetic).
__attribute__((noinline)) void JacobianKernel(
    const double* r1e, const double* r1o, const double* z1e, const double* z1o,
    const double* rue, const double* ruo, const double* zue, const double* zuo,
    const double* sqrtSH, double deltaS, double dSHalfDsInterp, int nZnT,
    int nsH, double* r12, double* ru12, double* zu12, double* rs, double* zs,
    double* tau) {
  for (int jH = 0; jH < nsH; ++jH) {
    const double sH = sqrtSH[jH];
    for (int kl = 0; kl < nZnT; ++kl) {
      const int i_in = jH * nZnT + kl;
      const int i_out = (jH + 1) * nZnT + kl;
      const int ih = jH * nZnT + kl;

      const double r1e_i = r1e[i_in], r1e_o = r1e[i_out];
      const double r1o_i = r1o[i_in], r1o_o = r1o[i_out];
      const double z1e_i = z1e[i_in], z1e_o = z1e[i_out];
      const double z1o_i = z1o[i_in], z1o_o = z1o[i_out];
      const double rue_i = rue[i_in], rue_o = rue[i_out];
      const double ruo_i = ruo[i_in], ruo_o = ruo[i_out];
      const double zue_i = zue[i_in], zue_o = zue[i_out];
      const double zuo_i = zuo[i_in], zuo_o = zuo[i_out];

      r12[ih] = 0.5 * ((r1e_i + r1e_o) + sH * (r1o_i + r1o_o));
      ru12[ih] = 0.5 * ((rue_i + rue_o) + sH * (ruo_i + ruo_o));
      zu12[ih] = 0.5 * ((zue_i + zue_o) + sH * (zuo_i + zuo_o));
      rs[ih] = ((r1e_o - r1e_i) + sH * (r1o_o - r1o_i)) / deltaS;
      zs[ih] = ((z1e_o - z1e_i) + sH * (z1o_o - z1o_i)) / deltaS;

      const double tau1 = ru12[ih] * zs[ih] - rs[ih] * zu12[ih];
      const double tau2 =
          ruo_o * z1o_o + ruo_i * z1o_i - zuo_o * r1o_o - zuo_i * r1o_i +
          (rue_o * z1o_o + rue_i * z1o_i - zue_o * r1o_o - zue_i * r1o_i) / sH;
      tau[ih] = tau1 + dSHalfDsInterp * tau2;
    }
  }
}

constexpr int kNgeom = 8;  // r1e r1o z1e z1o rue ruo zue zuo
constexpr int kNout = 6;   // r12 ru12 zu12 rs zs tau

struct Problem {
  int nZnT, nsH, n_in, n_out;
  std::vector<double> geom;  // kNgeom blocks of (nsH+1)*nZnT
  std::vector<double> sqrtSH;
  double deltaS, dSHalfDsInterp;
};

Problem MakeProblem(int nZnT, int nsH) {
  Problem p{nZnT, nsH, kNgeom * (nsH + 1) * nZnT, kNout * nsH * nZnT, {}, {},
            0.0,  0.0};
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> d(0.5, 1.5);
  p.geom.resize(p.n_in);
  for (double& x : p.geom) x = d(rng);
  p.sqrtSH.resize(nsH);
  for (int j = 0; j < nsH; ++j) p.sqrtSH[j] = std::sqrt(0.05 + 0.9 * j / nsH);
  p.deltaS = 0.1;
  p.dSHalfDsInterp = 0.25;
  return p;
}

// Evaluate the kernel from a flat geometry buffer into a flat output buffer.
__attribute__((noinline)) void Eval(const double* geom, const Problem& p,
                                    double* out) {
  const int s = (p.nsH + 1) * p.nZnT;
  const int o = p.nsH * p.nZnT;
  JacobianKernel(geom, geom + s, geom + 2 * s, geom + 3 * s, geom + 4 * s,
                 geom + 5 * s, geom + 6 * s, geom + 7 * s, p.sqrtSH.data(),
                 p.deltaS, p.dSHalfDsInterp, p.nZnT, p.nsH, out, out + o,
                 out + 2 * o, out + 3 * o, out + 4 * o, out + 5 * o);
}

double MaxAbs(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}

// Scalar objective L(geom) = 0.5 * sum of squares of all kernel outputs. out is
// caller-owned scratch (an intermediate of the differentiated call). This is
// the scalar-over-buffers form Enzyme differentiates in both modes.
__attribute__((noinline)) double Loss(const double* geom, double* out,
                                      const Problem* p) {
  Eval(geom, *p, out);
  double s = 0.0;
  for (int i = 0; i < p->n_out; ++i) s += 0.5 * out[i] * out[i];
  return s;
}

int main() {
  const Problem p = MakeProblem(/*nZnT=*/32, /*nsH=*/12);
  std::mt19937 rng(11);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> v(p.n_in);
  for (double& x : v) x = dist(rng);

  // Reverse mode: full gradient dL/dgeom in one pass.
  std::vector<double> g_rev(p.n_in, 0.0), out_s(p.n_out, 0.0),
      out_sh(p.n_out, 0.0);
  __enzyme_autodiff((void*)Loss, enzyme_dup, p.geom.data(), g_rev.data(),
                    enzyme_dup, out_s.data(), out_sh.data(), enzyme_const, &p);

  // Forward mode: directional derivative dL . v in one pass.
  std::vector<double> out_t(p.n_out, 0.0);
  const double d_fwd = __enzyme_fwddiff<double>(
      (void*)Loss, enzyme_dup, p.geom.data(), v.data(), enzyme_dup,
      out_s.data(), out_t.data(), enzyme_const, &p);

  // Central finite differences of the directional derivative.
  const double h = 1e-6;
  std::vector<double> gp(p.geom), gm(p.geom), os(p.n_out);
  for (int i = 0; i < p.n_in; ++i) {
    gp[i] = p.geom[i] + h * v[i];
    gm[i] = p.geom[i] - h * v[i];
  }
  const double d_fd =
      (Loss(gp.data(), os.data(), &p) - Loss(gm.data(), os.data(), &p)) /
      (2 * h);
  const double d_rev =
      std::inner_product(g_rev.begin(), g_rev.end(), v.begin(), 0.0);

  const double scale = std::fabs(d_fd) + 1e-300;
  printf("exact autodiff of VMEC Jacobian kernel (n_in=%d n_out=%d)\n", p.n_in,
         p.n_out);
  printf("  reverse  dL.v = %.10e   (rel err vs FD %.2e)\n", d_rev,
         std::fabs(d_rev - d_fd) / scale);
  printf("  forward  dL.v = %.10e   (rel err vs FD %.2e)\n", d_fwd,
         std::fabs(d_fwd - d_fd) / scale);
  printf("  forward/reverse agreement: %.2e\n",
         std::fabs(d_fwd - d_rev) / (std::fabs(d_rev) + 1e-300));

  // Performance: reverse returns the whole gradient (n_in) in one pass; forward
  // returns one directional derivative per pass. For a scalar objective reverse
  // wins; for a single Jacobian/Hessian-vector product forward is the cheaper
  // primitive. Time both.
  const int reps = 2000;
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    std::fill(g_rev.begin(), g_rev.end(), 0.0);
    __enzyme_autodiff((void*)Loss, enzyme_dup, p.geom.data(), g_rev.data(),
                      enzyme_dup, out_s.data(), out_sh.data(), enzyme_const,
                      &p);
  }
  auto t1 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r) {
    out_t.assign(p.n_out, 0.0);
    volatile double s = __enzyme_fwddiff<double>(
        (void*)Loss, enzyme_dup, p.geom.data(), v.data(), enzyme_dup,
        out_s.data(), out_t.data(), enzyme_const, &p);
    (void)s;
  }
  auto t2 = std::chrono::steady_clock::now();
  const double us_rev =
      std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
  const double us_fwd =
      std::chrono::duration<double, std::micro>(t2 - t1).count() / reps;
  printf(
      "  performance: reverse %.2f us/pass (full gradient), forward %.2f "
      "us/pass (one direction)\n",
      us_rev, us_fwd);

  const bool ok = std::fabs(d_rev - d_fd) < 1e-5 * scale &&
                  std::fabs(d_fwd - d_fd) < 1e-5 * scale &&
                  std::fabs(d_fwd - d_rev) < 1e-9 * (std::fabs(d_rev) + 1e-300);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

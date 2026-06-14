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
#include "vmecpp/vmec/ideal_mhd_model/jacobian_kernel.h"

// Types used as Enzyme intrinsic arguments need external linkage (an anonymous
// namespace type cannot instantiate the __enzyme_* templates).

// JacobianKernel is the shared production kernel ComputeHalfGridJacobian
// (jacobian_kernel.h); differentiated here exactly as the solver uses it.

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
  vmecpp::ComputeHalfGridJacobian(
      geom, geom + s, geom + 2 * s, geom + 3 * s, geom + 4 * s, geom + 5 * s,
      geom + 6 * s, geom + 7 * s, p.sqrtSH.data(), p.deltaS, p.dSHalfDsInterp,
      p.nZnT, /*nsMinF1=*/0, /*nsMinH=*/0,
      /*nsMaxH=*/p.nsH, out, out + o, out + 2 * o, out + 3 * o, out + 4 * o,
      out + 5 * o);
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

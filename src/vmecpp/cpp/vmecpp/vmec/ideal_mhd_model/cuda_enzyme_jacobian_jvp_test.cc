// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Deterministic GPU test for the device Enzyme JVP of the half-grid Jacobian.
//
// It seeds smooth nonzero geometry, runs the GPU-Enzyme forward-mode JVP
// (cuda_enzyme_jacobian_jvp.cu) on device buffers, and checks every output
// tangent for r12, ru12, zu12, rs, zs and tau against central finite
// differences of the same shared kernel (jacobian_kernel.h). Tolerances are
// tight enough to catch missing activity markers, wrong shadow strides, and
// host/device layout drift. A second phase seeds one geometry block at a time
// to verify that outputs which do not depend on that block get exactly zero
// tangent.
//
// Built only under VMECPP_ENABLE_CUDA_ENZYME; not run in CI unless a GPU runner
// with the matching ClangEnzyme plugin is available.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "vmecpp/vmec/ideal_mhd_model/jacobian_kernel.h"

namespace vmecpp {
void LaunchJacobianJvp(const JacobianPointInput* d_x,
                       const JacobianPointInput* d_dx, JacobianPointOutput* d_y,
                       JacobianPointOutput* d_dy, int n);
}  // namespace vmecpp

// Host Enzyme forward-mode intrinsic and activity marker: the exact CPU oracle.
// It differentiates the identical HalfGridJacobianPoint on host IR, so the GPU
// device-IR JVP must agree with it to machine precision -- the check that pins
// activity markers, shadow strides, and host/device struct layout. Resolved by
// the same ClangEnzyme plugin the .cu uses.
extern int enzyme_dup;
template <typename Return, typename... Args>
Return __enzyme_fwddiff(  // NOLINT(bugprone-reserved-identifier)
    void*, Args...);

using vmecpp::HalfGridJacobianPoint;
using vmecpp::JacobianPointInput;
using vmecpp::JacobianPointOutput;

namespace {

constexpr int kNInDoubles = 19;  // 16 geometry + sH, deltaS, dSHalfDsInterp
constexpr int kNGeomDoubles = 16;
constexpr int kNOutDoubles = 6;

double* AsDoubles(JacobianPointInput& x) {
  return reinterpret_cast<double*>(&x);
}
const double* AsDoubles(const JacobianPointOutput& y) {
  return reinterpret_cast<const double*>(&y);
}

bool CudaOk(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::printf("CUDA error (%s): %s\n", what, cudaGetErrorString(e));
    return false;
  }
  return true;
}

// Run the device JVP for a whole array of points and a whole tangent array.
bool GpuJvp(const std::vector<JacobianPointInput>& x,
            const std::vector<JacobianPointInput>& dx,
            std::vector<JacobianPointOutput>& dy) {
  const int n = static_cast<int>(x.size());
  JacobianPointInput *d_x = nullptr, *d_dx = nullptr;
  JacobianPointOutput *d_y = nullptr, *d_dy = nullptr;
  const size_t in_bytes = n * sizeof(JacobianPointInput);
  const size_t out_bytes = n * sizeof(JacobianPointOutput);
  bool ok =
      CudaOk(cudaMalloc(&d_x, in_bytes), "malloc x") &&
      CudaOk(cudaMalloc(&d_dx, in_bytes), "malloc dx") &&
      CudaOk(cudaMalloc(&d_y, out_bytes), "malloc y") &&
      CudaOk(cudaMalloc(&d_dy, out_bytes), "malloc dy") &&
      CudaOk(cudaMemcpy(d_x, x.data(), in_bytes, cudaMemcpyHostToDevice),
             "H2D x") &&
      CudaOk(cudaMemcpy(d_dx, dx.data(), in_bytes, cudaMemcpyHostToDevice),
             "H2D dx");
  if (ok) {
    vmecpp::LaunchJacobianJvp(d_x, d_dx, d_y, d_dy, n);
    ok = CudaOk(cudaGetLastError(), "launch") &&
         CudaOk(cudaDeviceSynchronize(), "sync") &&
         CudaOk(cudaMemcpy(dy.data(), d_dy, out_bytes, cudaMemcpyDeviceToHost),
                "D2H dy");
  }
  cudaFree(d_x);
  cudaFree(d_dx);
  cudaFree(d_y);
  cudaFree(d_dy);
  return ok;
}

// Central difference of the shared kernel at step h, per point, into dy.
void CentralDiff(const std::vector<JacobianPointInput>& x,
                 const std::vector<JacobianPointInput>& dx, double h,
                 std::vector<JacobianPointOutput>& dy) {
  const int n = static_cast<int>(x.size());
  for (int i = 0; i < n; ++i) {
    JacobianPointInput xp = x[i], xm = x[i];
    double* pp = AsDoubles(xp);
    double* pm = AsDoubles(xm);
    const double* pd = reinterpret_cast<const double*>(&dx[i]);
    for (int k = 0; k < kNInDoubles; ++k) {
      pp[k] += h * pd[k];
      pm[k] -= h * pd[k];
    }
    JacobianPointOutput yp{}, ym{};
    HalfGridJacobianPoint(xp, &yp);
    HalfGridJacobianPoint(xm, &ym);
    double* out = reinterpret_cast<double*>(&dy[i]);
    const double* op = AsDoubles(yp);
    const double* om = AsDoubles(ym);
    for (int k = 0; k < kNOutDoubles; ++k) out[k] = (op[k] - om[k]) / (2.0 * h);
  }
}

// Richardson-extrapolated central-difference JVP: (4 D(h/2) - D(h)) / 3 cancels
// the O(h^2) truncation term, leaving an O(h^4) oracle accurate to ~1e-12. The
// residual against the exact Enzyme JVP then reflects Enzyme and the
// host/device layout, not the finite-difference step. An input block that an
// output does not depend on gives bit-identical +/- evaluations, hence an
// exactly-zero oracle tangent, so the per-block phase checks true zeros.
std::vector<JacobianPointOutput> FdJvp(
    const std::vector<JacobianPointInput>& x,
    const std::vector<JacobianPointInput>& dx, double h) {
  const int n = static_cast<int>(x.size());
  std::vector<JacobianPointOutput> d1(n), d2(n), dy(n);
  CentralDiff(x, dx, h, d1);
  CentralDiff(x, dx, 0.5 * h, d2);
  for (int i = 0; i < n; ++i) {
    double* out = reinterpret_cast<double*>(&dy[i]);
    const double* p1 = AsDoubles(d1[i]);
    const double* p2 = AsDoubles(d2[i]);
    for (int k = 0; k < kNOutDoubles; ++k) out[k] = (4.0 * p2[k] - p1[k]) / 3.0;
  }
  return dy;
}

double MaxRelErr(const std::vector<JacobianPointOutput>& a,
                 const std::vector<JacobianPointOutput>& b, double abs_floor) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double* pa = AsDoubles(a[i]);
    const double* pb = AsDoubles(b[i]);
    for (int k = 0; k < kNOutDoubles; ++k) {
      const double denom = std::fabs(pb[k]) + abs_floor;
      worst = std::fmax(worst, std::fabs(pa[k] - pb[k]) / denom);
    }
  }
  return worst;
}

// Exact CPU forward-mode JVP of the shared kernel, per point: the trusted
// oracle. HalfGridJacobianPoint is host code here (VMECPP_HD is empty without a
// CUDA compiler), so Enzyme differentiates it the same way it does on the GPU.
std::vector<JacobianPointOutput> CpuEnzymeJvp(
    const std::vector<JacobianPointInput>& x,
    const std::vector<JacobianPointInput>& dx) {
  const int n = static_cast<int>(x.size());
  std::vector<JacobianPointOutput> dy(n);
  for (int i = 0; i < n; ++i) {
    JacobianPointOutput y{}, d{};
    __enzyme_fwddiff<void>(reinterpret_cast<void*>(HalfGridJacobianPoint),
                           enzyme_dup, &x[i], &dx[i], enzyme_dup, &y, &d);
    dy[i] = d;
  }
  return dy;
}

}  // namespace

int main() {
  // Small deterministic problem: nZnT = 8, two half-grid points per "surface"
  // pair, giving n independent per-point Jacobian evaluations.
  constexpr int kNZnT = 8;
  constexpr int kNsH = 2;
  const int n = kNZnT * kNsH;

  std::vector<JacobianPointInput> x(n);
  for (int i = 0; i < n; ++i) {
    double* p = AsDoubles(x[i]);
    // Smooth, nonzero geometry so every derivative path is exercised.
    for (int k = 0; k < kNGeomDoubles; ++k) {
      p[k] = 1.0 + 0.3 * std::sin(0.7 * i + 1.1 * k) + 0.1 * (k % 3);
    }
    p[16] = 0.5 + 0.4 * ((i % kNsH) + 1) / kNsH;  // sH in (0, 1]
    p[17] = 0.1;                                  // deltaS
    p[18] = 0.25;                                 // dSHalfDsInterp
  }

  const double h = 1e-4;  // Richardson base step; h/2 is also evaluated
  const double abs_floor = 1e-9;
  const double kTolExact =
      1e-8;                    // GPU-Enzyme vs exact CPU-Enzyme (issue target)
  const double kTolFd = 1e-6;  // GPU-Enzyme vs Richardson FD (absolute sanity)
  bool ok = true;

  // Phase 1: full random tangent on the 16 geometry components. The exact
  // CPU-Enzyme oracle is the tight gate; Richardson FD is an independent check
  // that both agree with the true derivative, not just with each other.
  {
    std::vector<JacobianPointInput> dx(n);
    for (int i = 0; i < n; ++i) {
      double* p = reinterpret_cast<double*>(&dx[i]);
      for (int k = 0; k < kNInDoubles; ++k) p[k] = 0.0;
      for (int k = 0; k < kNGeomDoubles; ++k) {
        p[k] = std::sin(1.7 * i + 0.9 * k + 0.3);
      }
    }
    std::vector<JacobianPointOutput> dy_gpu(n);
    if (!GpuJvp(x, dx, dy_gpu)) return 2;
    const double rel_cpu = MaxRelErr(dy_gpu, CpuEnzymeJvp(x, dx), abs_floor);
    const double rel_fd = MaxRelErr(dy_gpu, FdJvp(x, dx, h), abs_floor);
    std::printf("phase 1 (full tangent):   vs CPU-Enzyme %.2e, vs FD %.2e\n",
                rel_cpu, rel_fd);
    ok = ok && rel_cpu < kTolExact && rel_fd < kTolFd;
  }

  // Phase 2: seed one geometry block at a time. Outputs that do not depend on a
  // block get exactly-zero tangent from both Enzyme and FD (bit-identical +/-
  // evaluations); the GPU JVP must match, catching wrong activity markers or
  // shadow strides.
  const char* kBlocks[8] = {"r1e", "r1o", "z1e", "z1o",
                            "rue", "ruo", "zue", "zuo"};
  for (int b = 0; b < 8; ++b) {
    std::vector<JacobianPointInput> dx(n);
    for (int i = 0; i < n; ++i) {
      double* p = reinterpret_cast<double*>(&dx[i]);
      for (int k = 0; k < kNInDoubles; ++k) p[k] = 0.0;
      // Each geometry block occupies two adjacent doubles (_i, _o).
      p[2 * b] = std::cos(0.5 * i + b);
      p[2 * b + 1] = std::sin(0.5 * i + b);
    }
    std::vector<JacobianPointOutput> dy_gpu(n);
    if (!GpuJvp(x, dx, dy_gpu)) return 2;
    const double rel_cpu = MaxRelErr(dy_gpu, CpuEnzymeJvp(x, dx), abs_floor);
    const double rel_fd = MaxRelErr(dy_gpu, FdJvp(x, dx, h), abs_floor);
    std::printf("phase 2 (seed %-3s only):  vs CPU-Enzyme %.2e, vs FD %.2e\n",
                kBlocks[b], rel_cpu, rel_fd);
    ok = ok && rel_cpu < kTolExact && rel_fd < kTolFd;
  }

  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

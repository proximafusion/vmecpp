// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

// Device-side Enzyme forward-mode JVP of the half-grid Jacobian kernel.
//
// This is the first proof that a VMEC++ CUDA physics kernel can be
// differentiated by Enzyme on the GPU. It differentiates exactly one pure
// device function, HalfGridJacobianPoint (jacobian_kernel.h) -- the same
// per-point arithmetic the CPU solver runs -- and touches nothing else on the
// #561 CUDA path: no cuFFT, no CUDA graphs, no persistent CudaToroidalState, no
// NESTOR bridge, no host/device transfers inside the differentiated region.
//
// Built only under the opt-in VMECPP_ENABLE_CUDA_ENZYME CMake path, which
// requires VMECPP_USE_CUDA=ON, a Clang CUDA compiler, and the ClangEnzyme
// plugin. The default CPU, CPU-Enzyme, and CUDA builds are unaffected.

#include "vmecpp/vmec/ideal_mhd_model/jacobian_kernel.h"

// Enzyme forward-mode intrinsic and the per-argument activity marker, resolved
// by the ClangEnzyme plugin during the device-IR optimization pass. Declared
// __device__ (not via common/enzyme/enzyme.h, whose markers are host globals)
// so the __global__ kernel below can reference them; the plugin matches
// `enzyme_dup` by symbol name and rewrites the call away before device linking.
extern __device__ int enzyme_dup;

template <typename Return, typename... Args>
__device__ Return __enzyme_fwddiff(  // NOLINT(bugprone-reserved-identifier)
    void*, Args...);

namespace vmecpp {

// One thread per half-grid point. Forward-mode JVP of HalfGridJacobianPoint:
// x[i]/y[i] are the primal input/output, dx[i] is the seeded input tangent, and
// dy[i] receives the output tangent J(x[i]) . dx[i]. All buffers are device
// resident; Enzyme differentiates the pure device function over them.
__global__ void JacobianJvpKernel(const JacobianPointInput* x,
                                  const JacobianPointInput* dx,
                                  JacobianPointOutput* y,
                                  JacobianPointOutput* dy, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  __enzyme_fwddiff<void>(reinterpret_cast<void*>(HalfGridJacobianPoint),
                         enzyme_dup, x + i, dx + i, enzyme_dup, y + i, dy + i);
}

// Host launcher for the JVP kernel. Pointers are device resident; the caller
// owns allocation and host/device transfers, keeping the differentiated region
// pure.
void LaunchJacobianJvp(const JacobianPointInput* d_x,
                       const JacobianPointInput* d_dx, JacobianPointOutput* d_y,
                       JacobianPointOutput* d_dy, int n) {
  constexpr int kThreads = 64;
  const int blocks = (n + kThreads - 1) / kThreads;
  JacobianJvpKernel<<<blocks, kThreads>>>(d_x, d_dx, d_y, d_dy, n);
}

}  // namespace vmecpp

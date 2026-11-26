// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
//
// SPDX-License-Identifier: MIT
#ifndef VMECPP_VMEC_IDEAL_MHD_MODEL_HIP_COMPAT_CUH_
#define VMECPP_VMEC_IDEAL_MHD_MODEL_HIP_COMPAT_CUH_

// Maps the CUDA runtime, cuFFT, and warp-intrinsic surface used by the GPU
// translation units onto HIP.

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>

// --- runtime types ---------------------------------------------------------
#define cudaError_t hipError_t
#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaDeviceProp hipDeviceProp_t

// --- enumerators and constants ---------------------------------------------
#define cudaSuccess hipSuccess
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaStreamCaptureModeGlobal hipStreamCaptureModeGlobal
#define cudaDevAttrMaxSharedMemoryPerBlockOptin \
  hipDeviceAttributeSharedMemPerBlockOptin
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
  hipFuncAttributeMaxDynamicSharedMemorySize

// --- memory ----------------------------------------------------------------
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo

// hipHostMalloc requires a flags argument; cudaMallocHost takes two.
template <typename T>
static inline hipError_t VmecppHipMallocHost(T** ptr, size_t size) {
  return hipHostMalloc(reinterpret_cast<void**>(ptr), size,
                       hipHostMallocDefault);
}
#define cudaMallocHost VmecppHipMallocHost
#define cudaFreeHost hipHostFree

// --- error handling --------------------------------------------------------
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString

// --- device management -----------------------------------------------------
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute

// clang requires the cast to const void* that nvcc performs implicitly.
template <typename KernelT>
static inline hipError_t VmecppHipFuncSetAttribute(KernelT kernel,
                                                   hipFuncAttribute attr,
                                                   int value) {
  return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attr,
                             value);
}
#define cudaFuncSetAttribute VmecppHipFuncSetAttribute

// --- streams and events ----------------------------------------------------
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime

// --- graphs ----------------------------------------------------------------
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy

// --- FFT -------------------------------------------------------------------
#define cufftHandle hipfftHandle
#define cufftResult hipfftResult
#define cufftDoubleComplex hipfftDoubleComplex
#define cufftComplex hipfftComplex
#define cufftPlanMany hipfftPlanMany
#define cufftExecD2Z hipfftExecD2Z
#define cufftExecZ2D hipfftExecZ2D
#define cufftExecC2R hipfftExecC2R
#define cufftSetStream hipfftSetStream
#define cufftDestroy hipfftDestroy
#define CUFFT_D2Z HIPFFT_D2Z
#define CUFFT_Z2D HIPFFT_Z2D
#define CUFFT_C2R HIPFFT_C2R
#define CUFFT_SUCCESS HIPFFT_SUCCESS

// --- warp intrinsics -------------------------------------------------------
// Width-32 shuffles partition the 64-lane wavefront into 32-lane groups,
// preserving the 32-lane warp semantics the kernels assume.
#define __shfl_xor_sync(mask, var, lane_mask) __shfl_xor((var), (lane_mask), 32)
#define __shfl_sync(mask, var, src_lane) __shfl((var), (src_lane), 32)
#define __shfl_down_sync(mask, var, delta) __shfl_down((var), (delta), 32)

// Wave-level barrier standing in for __syncwarp.
__device__ __forceinline__ void VmecppHipSyncwarp() {
  __builtin_amdgcn_wave_barrier();
}
#define __syncwarp(...) VmecppHipSyncwarp()

// --- launch bounds ---------------------------------------------------------
// Keeps max threads per block; drops the second parameter, which is min
// blocks per multiprocessor in CUDA but min waves per execution unit in HIP.
#undef __launch_bounds__
#define VMECPP_HIP_MAX_THREADS(first, ...) first
#define __launch_bounds__(...) \
  __attribute__((launch_bounds(VMECPP_HIP_MAX_THREADS(__VA_ARGS__, 0))))

#endif  // VMECPP_VMEC_IDEAL_MHD_MODEL_HIP_COMPAT_CUH_

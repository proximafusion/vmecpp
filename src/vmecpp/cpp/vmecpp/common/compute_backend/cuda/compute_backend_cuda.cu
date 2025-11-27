// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/common/compute_backend/cuda/compute_backend_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace vmecpp {

// Helper macro for CUDA error checking.
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("CUDA error in ") + __FILE__ +  \
                               ":" + std::to_string(__LINE__) + ": " +     \
                               cudaGetErrorString(err));                   \
    }                                                                      \
  } while (0)

// Device memory buffer that automatically manages allocation/deallocation.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() : data_(nullptr), size_(0) {}

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  // Non-copyable.
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  // Movable.
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        cudaFree(data_);
      }
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  void Resize(size_t new_size) {
    if (new_size > size_) {
      if (data_ != nullptr) {
        cudaFree(data_);
      }
      CUDA_CHECK(cudaMalloc(&data_, new_size * sizeof(T)));
      size_ = new_size;
    }
  }

  void CopyFromHost(const T* host_data, size_t count,
                    cudaStream_t stream = nullptr) {
    Resize(count);
    if (stream != nullptr) {
      CUDA_CHECK(cudaMemcpyAsync(data_, host_data, count * sizeof(T),
                                 cudaMemcpyHostToDevice, stream));
    } else {
      CUDA_CHECK(
          cudaMemcpy(data_, host_data, count * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

  void CopyFromHost(const std::vector<T>& host_vec,
                    cudaStream_t stream = nullptr) {
    CopyFromHost(host_vec.data(), host_vec.size(), stream);
  }

  void CopyToHost(T* host_data, size_t count,
                  cudaStream_t stream = nullptr) const {
    if (stream != nullptr) {
      CUDA_CHECK(cudaMemcpyAsync(host_data, data_, count * sizeof(T),
                                 cudaMemcpyDeviceToHost, stream));
    } else {
      CUDA_CHECK(
          cudaMemcpy(host_data, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
  }

  void CopyToHost(std::vector<T>& host_vec, cudaStream_t stream = nullptr) const {
    CopyToHost(host_vec.data(), host_vec.size(), stream);
  }

  void SetZero(size_t count, cudaStream_t stream = nullptr) {
    Resize(count);
    if (stream != nullptr) {
      CUDA_CHECK(cudaMemsetAsync(data_, 0, count * sizeof(T), stream));
    } else {
      CUDA_CHECK(cudaMemset(data_, 0, count * sizeof(T)));
    }
  }

  T* Data() { return data_; }
  const T* Data() const { return data_; }
  size_t Size() const { return size_; }

 private:
  T* data_;
  size_t size_;
};

// Grid and problem size parameters passed to kernels.
struct KernelParams {
  int ns_min_f;
  int ns_max_f;
  int ns_min_f1;
  int ns_max_f1;
  int ns_max_f_including_lcfs;
  int mpol;
  int ntor;
  int n_zeta;
  int n_theta_eff;
  int n_theta_reduced;
  int nnyq2;
  int ns;
  bool lfreeb;
};

// =============================================================================
// CUDA Kernels for FourierToReal
// =============================================================================

// Kernel for inverse DFT: Fourier coefficients -> real-space geometry.
// Each thread handles one (jF, k, l) triple.
__global__ void FourierToRealKernel(
    const double* __restrict__ rmncc,
    const double* __restrict__ rmnss,
    const double* __restrict__ zmnsc,
    const double* __restrict__ zmncs,
    const double* __restrict__ lmnsc,
    const double* __restrict__ lmncs,
    const double* __restrict__ xmpq,
    const double* __restrict__ sqrt_sf,
    const double* __restrict__ cosmu,
    const double* __restrict__ sinmu,
    const double* __restrict__ cosmum,
    const double* __restrict__ sinmum,
    const double* __restrict__ cosnv,
    const double* __restrict__ sinnv,
    const double* __restrict__ cosnvn,
    const double* __restrict__ sinnvn,
    double* __restrict__ r1_e,
    double* __restrict__ r1_o,
    double* __restrict__ ru_e,
    double* __restrict__ ru_o,
    double* __restrict__ rv_e,
    double* __restrict__ rv_o,
    double* __restrict__ z1_e,
    double* __restrict__ z1_o,
    double* __restrict__ zu_e,
    double* __restrict__ zu_o,
    double* __restrict__ zv_e,
    double* __restrict__ zv_o,
    double* __restrict__ lu_e,
    double* __restrict__ lu_o,
    double* __restrict__ lv_e,
    double* __restrict__ lv_o,
    double* __restrict__ r_con,
    double* __restrict__ z_con,
    KernelParams params) {

  const int jF = blockIdx.x + params.ns_min_f1;
  const int k = blockIdx.y;
  const int l = threadIdx.x;

  if (jF >= params.ns_max_f1 || k >= params.n_zeta ||
      l >= params.n_theta_reduced) {
    return;
  }

  const int ntorp1 = params.ntor + 1;
  const int nnyq2p1 = params.nnyq2 + 1;

  // Loop over poloidal modes m.
  for (int m = 0; m < params.mpol; ++m) {
    const bool m_even = (m % 2 == 0);

    // Axis only gets contributions from m=0,1.
    int jMin = 1;
    if (m == 0 || m == 1) {
      jMin = 0;
    }
    if (jF < jMin) {
      continue;
    }

    const int idx_ml = m * params.n_theta_reduced + l;
    const double cosmu_val = cosmu[idx_ml];
    const double sinmu_val = sinmu[idx_ml];
    const double cosmum_val = cosmum[idx_ml];
    const double sinmum_val = sinmum[idx_ml];

    // Accumulate from toroidal modes n.
    double rmkcc = 0.0, rmkcc_n = 0.0;
    double rmkss = 0.0, rmkss_n = 0.0;
    double zmksc = 0.0, zmksc_n = 0.0;
    double zmkcs = 0.0, zmkcs_n = 0.0;
    double lmksc = 0.0, lmksc_n = 0.0;
    double lmkcs = 0.0, lmkcs_n = 0.0;

    for (int n = 0; n < ntorp1; ++n) {
      const int idx_kn = k * nnyq2p1 + n;
      const double cosnv_val = cosnv[idx_kn];
      const double sinnv_val = sinnv[idx_kn];
      const double cosnvn_val = cosnvn[idx_kn];
      const double sinnvn_val = sinnvn[idx_kn];

      const int idx_mn =
          ((jF - params.ns_min_f1) * params.mpol + m) * ntorp1 + n;

      rmkcc += rmncc[idx_mn] * cosnv_val;
      rmkcc_n += rmncc[idx_mn] * sinnvn_val;
      rmkss += rmnss[idx_mn] * sinnv_val;
      rmkss_n += rmnss[idx_mn] * cosnvn_val;
      zmksc += zmnsc[idx_mn] * cosnv_val;
      zmksc_n += zmnsc[idx_mn] * sinnvn_val;
      zmkcs += zmncs[idx_mn] * sinnv_val;
      zmkcs_n += zmncs[idx_mn] * cosnvn_val;
      lmksc += lmnsc[idx_mn] * cosnv_val;
      lmksc_n += lmnsc[idx_mn] * sinnvn_val;
      lmkcs += lmncs[idx_mn] * sinnv_val;
      lmkcs_n += lmncs[idx_mn] * cosnvn_val;
    }

    // Compute output index.
    const int idx_kl =
        ((jF - params.ns_min_f1) * params.n_zeta + k) * params.n_theta_eff + l;

    // Select even/odd arrays based on m parity.
    double* r1 = m_even ? r1_e : r1_o;
    double* ru = m_even ? ru_e : ru_o;
    double* rv = m_even ? rv_e : rv_o;
    double* z1 = m_even ? z1_e : z1_o;
    double* zu = m_even ? zu_e : zu_o;
    double* zv = m_even ? zv_e : zv_o;
    double* lu = m_even ? lu_e : lu_o;
    double* lv = m_even ? lv_e : lv_o;

    // Use atomic adds since multiple m values write to the same output location.
    atomicAdd(&r1[idx_kl], rmkcc * cosmu_val + rmkss * sinmu_val);
    atomicAdd(&ru[idx_kl], rmkcc * sinmum_val + rmkss * cosmum_val);
    atomicAdd(&rv[idx_kl], rmkcc_n * cosmu_val + rmkss_n * sinmu_val);
    atomicAdd(&z1[idx_kl], zmksc * sinmu_val + zmkcs * cosmu_val);
    atomicAdd(&zu[idx_kl], zmksc * cosmum_val + zmkcs * sinmum_val);
    atomicAdd(&zv[idx_kl], zmksc_n * sinmu_val + zmkcs_n * cosmu_val);
    atomicAdd(&lu[idx_kl], lmksc * cosmum_val + lmkcs * sinmum_val);
    // lv gets a negative sign.
    atomicAdd(&lv[idx_kl], -(lmksc_n * sinmu_val + lmkcs_n * cosmu_val));

    // Constraint force computation.
    if (params.ns_min_f <= jF && jF < params.ns_max_f_including_lcfs) {
      const double sqrt_s =
          m_even ? 1.0 : sqrt_sf[jF - params.ns_min_f1];
      const double con_factor = xmpq[m] * sqrt_s;

      const int idx_con =
          ((jF - params.ns_min_f) * params.n_zeta + k) * params.n_theta_eff + l;

      atomicAdd(&r_con[idx_con],
                (rmkcc * cosmu_val + rmkss * sinmu_val) * con_factor);
      atomicAdd(&z_con[idx_con],
                (zmksc * sinmu_val + zmkcs * cosmu_val) * con_factor);
    }
  }
}

// =============================================================================
// CUDA Kernels for ForcesToFourier
// =============================================================================

// Kernel for forward DFT: real-space forces -> Fourier coefficients.
// Each thread block handles one (jF, m) pair.
__global__ void ForcesToFourierKernel(
    const double* __restrict__ armn_e,
    const double* __restrict__ armn_o,
    const double* __restrict__ azmn_e,
    const double* __restrict__ azmn_o,
    const double* __restrict__ blmn_e,
    const double* __restrict__ blmn_o,
    const double* __restrict__ brmn_e,
    const double* __restrict__ brmn_o,
    const double* __restrict__ bzmn_e,
    const double* __restrict__ bzmn_o,
    const double* __restrict__ clmn_e,
    const double* __restrict__ clmn_o,
    const double* __restrict__ crmn_e,
    const double* __restrict__ crmn_o,
    const double* __restrict__ czmn_e,
    const double* __restrict__ czmn_o,
    const double* __restrict__ frcon_e,
    const double* __restrict__ frcon_o,
    const double* __restrict__ fzcon_e,
    const double* __restrict__ fzcon_o,
    const double* __restrict__ xmpq,
    const double* __restrict__ cosmui,
    const double* __restrict__ sinmui,
    const double* __restrict__ cosmumi,
    const double* __restrict__ sinmumi,
    const double* __restrict__ cosnv,
    const double* __restrict__ sinnv,
    const double* __restrict__ cosnvn,
    const double* __restrict__ sinnvn,
    double* __restrict__ frcc,
    double* __restrict__ frss,
    double* __restrict__ fzsc,
    double* __restrict__ fzcs,
    double* __restrict__ flsc,
    double* __restrict__ flcs,
    KernelParams params,
    int j_max_rz) {

  // Each block handles one (jF, m) pair.
  const int jF = blockIdx.x + params.ns_min_f;
  const int m = blockIdx.y;
  const int k = threadIdx.x;

  if (jF >= j_max_rz || k >= params.n_zeta) {
    return;
  }

  // On axis (jF=0), only m<1 contributes.
  const int mmax = (jF == 0) ? 1 : params.mpol;
  if (m >= mmax) {
    return;
  }

  const bool m_even = (m % 2 == 0);
  const int ntorp1 = params.ntor + 1;
  const int nnyq2p1 = params.nnyq2 + 1;

  // Select even/odd input arrays.
  const double* armn = m_even ? armn_e : armn_o;
  const double* azmn = m_even ? azmn_e : azmn_o;
  const double* blmn = m_even ? blmn_e : blmn_o;
  const double* brmn = m_even ? brmn_e : brmn_o;
  const double* bzmn = m_even ? bzmn_e : bzmn_o;
  const double* clmn = m_even ? clmn_e : clmn_o;
  const double* crmn = m_even ? crmn_e : crmn_o;
  const double* czmn = m_even ? czmn_e : czmn_o;
  const double* frcon = m_even ? frcon_e : frcon_o;
  const double* fzcon = m_even ? fzcon_e : fzcon_o;

  // Accumulate over poloidal grid points l.
  double rmkcc = 0.0, rmkcc_n = 0.0;
  double rmkss = 0.0, rmkss_n = 0.0;
  double zmksc = 0.0, zmksc_n = 0.0;
  double zmkcs = 0.0, zmkcs_n = 0.0;
  double lmksc = 0.0, lmksc_n = 0.0;
  double lmkcs = 0.0, lmkcs_n = 0.0;

  const int idx_kl_base =
      ((jF - params.ns_min_f) * params.n_zeta + k) * params.n_theta_eff;
  const int idx_ml_base = m * params.n_theta_reduced;

  for (int l = 0; l < params.n_theta_reduced; ++l) {
    const int idx_kl = idx_kl_base + l;
    const int idx_ml = idx_ml_base + l;

    const double cosmui_val = cosmui[idx_ml];
    const double sinmui_val = sinmui[idx_ml];
    const double cosmumi_val = cosmumi[idx_ml];
    const double sinmumi_val = sinmumi[idx_ml];

    lmksc += blmn[idx_kl] * cosmumi_val;
    lmkcs += blmn[idx_kl] * sinmumi_val;
    lmkcs_n -= clmn[idx_kl] * cosmui_val;
    lmksc_n -= clmn[idx_kl] * sinmui_val;

    rmkcc_n -= crmn[idx_kl] * cosmui_val;
    zmkcs_n -= czmn[idx_kl] * cosmui_val;
    rmkss_n -= crmn[idx_kl] * sinmui_val;
    zmksc_n -= czmn[idx_kl] * sinmui_val;

    const double tempR = armn[idx_kl] + xmpq[m] * frcon[idx_kl];
    const double tempZ = azmn[idx_kl] + xmpq[m] * fzcon[idx_kl];

    rmkcc += tempR * cosmui_val + brmn[idx_kl] * sinmumi_val;
    rmkss += tempR * sinmui_val + brmn[idx_kl] * cosmumi_val;
    zmksc += tempZ * sinmui_val + bzmn[idx_kl] * cosmumi_val;
    zmkcs += tempZ * cosmui_val + bzmn[idx_kl] * sinmumi_val;
  }

  // Write to Fourier coefficients.
  for (int n = 0; n < ntorp1; ++n) {
    const int idx_mn =
        ((jF - params.ns_min_f) * params.mpol + m) * ntorp1 + n;
    const int idx_kn = k * nnyq2p1 + n;

    const double cosnv_val = cosnv[idx_kn];
    const double sinnv_val = sinnv[idx_kn];
    const double cosnvn_val = cosnvn[idx_kn];
    const double sinnvn_val = sinnvn[idx_kn];

    atomicAdd(&frcc[idx_mn], rmkcc * cosnv_val + rmkcc_n * sinnvn_val);
    atomicAdd(&frss[idx_mn], rmkss * sinnv_val + rmkss_n * cosnvn_val);
    atomicAdd(&fzsc[idx_mn], zmksc * cosnv_val + zmksc_n * sinnvn_val);
    atomicAdd(&fzcs[idx_mn], zmkcs * sinnv_val + zmkcs_n * cosnvn_val);

    // Lambda forces only for jF >= 1.
    if (jF >= 1) {
      atomicAdd(&flsc[idx_mn], lmksc * cosnv_val + lmksc_n * sinnvn_val);
      atomicAdd(&flcs[idx_mn], lmkcs * sinnv_val + lmkcs_n * cosnvn_val);
    }
  }
}

// =============================================================================
// ComputeBackendCudaImpl (PIMPL)
// =============================================================================

class ComputeBackendCudaImpl {
 public:
  explicit ComputeBackendCudaImpl(int device_id, int num_streams)
      : device_id_(device_id), num_streams_(num_streams), available_(false) {
    // Check if device is available.
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0 || device_id >= device_count) {
      return;
    }

    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
      return;
    }

    err = cudaGetDeviceProperties(&device_props_, device_id);
    if (err != cudaSuccess) {
      return;
    }

    // Create streams.
    streams_.resize(num_streams);
    for (int i = 0; i < num_streams; ++i) {
      err = cudaStreamCreate(&streams_[i]);
      if (err != cudaSuccess) {
        // Clean up any streams we created.
        for (int j = 0; j < i; ++j) {
          cudaStreamDestroy(streams_[j]);
        }
        streams_.clear();
        return;
      }
    }

    available_ = true;
  }

  ~ComputeBackendCudaImpl() {
    for (auto& stream : streams_) {
      cudaStreamDestroy(stream);
    }
  }

  bool IsAvailable() const { return available_; }

  int GetDeviceId() const { return device_id_; }

  std::string GetDeviceName() const {
    return available_ ? device_props_.name : "";
  }

  std::string GetComputeCapability() const {
    if (!available_) return "";
    return std::to_string(device_props_.major) + "." +
           std::to_string(device_props_.minor);
  }

  size_t GetDeviceMemoryBytes() const {
    return available_ ? device_props_.totalGlobalMem : 0;
  }

  void Synchronize() {
    if (available_) {
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  void FourierToReal(const FourierGeometry& physical_x,
                     const std::vector<double>& xmpq,
                     const RadialPartitioning& rp, const Sizes& s,
                     const RadialProfiles& profiles,
                     const FourierBasisFastPoloidal& fb,
                     RealSpaceGeometry& m_geometry);

  void ForcesToFourier(const RealSpaceForces& forces,
                       const std::vector<double>& xmpq,
                       const RadialPartitioning& rp, const FlowControl& fc,
                       const Sizes& s, const FourierBasisFastPoloidal& fb,
                       VacuumPressureState vacuum_pressure_state,
                       FourierForces& m_physical_forces);

 private:
  int device_id_;
  int num_streams_;
  bool available_;
  cudaDeviceProp device_props_;
  std::vector<cudaStream_t> streams_;

  // Device memory buffers (reused across calls).
  DeviceBuffer<double> d_rmncc_, d_rmnss_, d_zmnsc_, d_zmncs_;
  DeviceBuffer<double> d_lmnsc_, d_lmncs_;
  DeviceBuffer<double> d_xmpq_, d_sqrt_sf_;
  DeviceBuffer<double> d_cosmu_, d_sinmu_, d_cosmum_, d_sinmum_;
  DeviceBuffer<double> d_cosmui_, d_sinmui_, d_cosmumi_, d_sinmumi_;
  DeviceBuffer<double> d_cosnv_, d_sinnv_, d_cosnvn_, d_sinnvn_;

  // Output buffers.
  DeviceBuffer<double> d_r1_e_, d_r1_o_, d_ru_e_, d_ru_o_;
  DeviceBuffer<double> d_rv_e_, d_rv_o_, d_z1_e_, d_z1_o_;
  DeviceBuffer<double> d_zu_e_, d_zu_o_, d_zv_e_, d_zv_o_;
  DeviceBuffer<double> d_lu_e_, d_lu_o_, d_lv_e_, d_lv_o_;
  DeviceBuffer<double> d_r_con_, d_z_con_;

  // Force input buffers.
  DeviceBuffer<double> d_armn_e_, d_armn_o_, d_azmn_e_, d_azmn_o_;
  DeviceBuffer<double> d_blmn_e_, d_blmn_o_, d_brmn_e_, d_brmn_o_;
  DeviceBuffer<double> d_bzmn_e_, d_bzmn_o_, d_clmn_e_, d_clmn_o_;
  DeviceBuffer<double> d_crmn_e_, d_crmn_o_, d_czmn_e_, d_czmn_o_;
  DeviceBuffer<double> d_frcon_e_, d_frcon_o_, d_fzcon_e_, d_fzcon_o_;

  // Force output buffers.
  DeviceBuffer<double> d_frcc_, d_frss_, d_fzsc_, d_fzcs_;
  DeviceBuffer<double> d_flsc_, d_flcs_;
};

void ComputeBackendCudaImpl::FourierToReal(const FourierGeometry& physical_x,
                                           const std::vector<double>& xmpq,
                                           const RadialPartitioning& rp,
                                           const Sizes& s,
                                           const RadialProfiles& profiles,
                                           const FourierBasisFastPoloidal& fb,
                                           RealSpaceGeometry& m_geometry) {
  cudaStream_t stream = streams_[0];

  // Prepare kernel parameters.
  KernelParams params;
  params.ns_min_f = rp.nsMinF;
  params.ns_max_f = rp.nsMaxF;
  params.ns_min_f1 = rp.nsMinF1;
  params.ns_max_f1 = rp.nsMaxF1;
  params.ns_max_f_including_lcfs = rp.nsMaxFIncludingLcfs;
  params.mpol = s.mpol;
  params.ntor = s.ntor;
  params.n_zeta = s.nZeta;
  params.n_theta_eff = s.nThetaEff;
  params.n_theta_reduced = s.nThetaReduced;
  params.nnyq2 = s.nnyq2;

  // Calculate sizes.
  const int num_surfaces = rp.nsMaxF1 - rp.nsMinF1;
  const int coeff_size = num_surfaces * s.mpol * (s.ntor + 1);
  const int grid_size = num_surfaces * s.nZeta * s.nThetaEff;
  const int con_size = (rp.nsMaxFIncludingLcfs - rp.nsMinF) * s.nZeta * s.nThetaEff;

  // Copy input data to device.
  d_rmncc_.CopyFromHost(physical_x.rmncc.data(), coeff_size, stream);
  d_rmnss_.CopyFromHost(physical_x.rmnss.data(), coeff_size, stream);
  d_zmnsc_.CopyFromHost(physical_x.zmnsc.data(), coeff_size, stream);
  d_zmncs_.CopyFromHost(physical_x.zmncs.data(), coeff_size, stream);
  d_lmnsc_.CopyFromHost(physical_x.lmnsc.data(), coeff_size, stream);
  d_lmncs_.CopyFromHost(physical_x.lmncs.data(), coeff_size, stream);

  d_xmpq_.CopyFromHost(xmpq, stream);
  d_sqrt_sf_.CopyFromHost(profiles.sqrtSF, stream);

  // Copy basis functions.
  d_cosmu_.CopyFromHost(fb.cosmu, stream);
  d_sinmu_.CopyFromHost(fb.sinmu, stream);
  d_cosmum_.CopyFromHost(fb.cosmum, stream);
  d_sinmum_.CopyFromHost(fb.sinmum, stream);
  d_cosnv_.CopyFromHost(fb.cosnv, stream);
  d_sinnv_.CopyFromHost(fb.sinnv, stream);
  d_cosnvn_.CopyFromHost(fb.cosnvn, stream);
  d_sinnvn_.CopyFromHost(fb.sinnvn, stream);

  // Initialize output buffers to zero.
  d_r1_e_.SetZero(grid_size, stream);
  d_r1_o_.SetZero(grid_size, stream);
  d_ru_e_.SetZero(grid_size, stream);
  d_ru_o_.SetZero(grid_size, stream);
  d_rv_e_.SetZero(grid_size, stream);
  d_rv_o_.SetZero(grid_size, stream);
  d_z1_e_.SetZero(grid_size, stream);
  d_z1_o_.SetZero(grid_size, stream);
  d_zu_e_.SetZero(grid_size, stream);
  d_zu_o_.SetZero(grid_size, stream);
  d_zv_e_.SetZero(grid_size, stream);
  d_zv_o_.SetZero(grid_size, stream);
  d_lu_e_.SetZero(grid_size, stream);
  d_lu_o_.SetZero(grid_size, stream);
  d_lv_e_.SetZero(grid_size, stream);
  d_lv_o_.SetZero(grid_size, stream);
  d_r_con_.SetZero(con_size, stream);
  d_z_con_.SetZero(con_size, stream);

  // Launch kernel.
  dim3 grid_dim(num_surfaces, s.nZeta, 1);
  dim3 block_dim(s.nThetaReduced, 1, 1);

  FourierToRealKernel<<<grid_dim, block_dim, 0, stream>>>(
      d_rmncc_.Data(), d_rmnss_.Data(), d_zmnsc_.Data(), d_zmncs_.Data(),
      d_lmnsc_.Data(), d_lmncs_.Data(), d_xmpq_.Data(), d_sqrt_sf_.Data(),
      d_cosmu_.Data(), d_sinmu_.Data(), d_cosmum_.Data(), d_sinmum_.Data(),
      d_cosnv_.Data(), d_sinnv_.Data(), d_cosnvn_.Data(), d_sinnvn_.Data(),
      d_r1_e_.Data(), d_r1_o_.Data(), d_ru_e_.Data(), d_ru_o_.Data(),
      d_rv_e_.Data(), d_rv_o_.Data(), d_z1_e_.Data(), d_z1_o_.Data(),
      d_zu_e_.Data(), d_zu_o_.Data(), d_zv_e_.Data(), d_zv_o_.Data(),
      d_lu_e_.Data(), d_lu_o_.Data(), d_lv_e_.Data(), d_lv_o_.Data(),
      d_r_con_.Data(), d_z_con_.Data(), params);

  // Copy results back to host.
  d_r1_e_.CopyToHost(m_geometry.r1_e.data(), grid_size, stream);
  d_r1_o_.CopyToHost(m_geometry.r1_o.data(), grid_size, stream);
  d_ru_e_.CopyToHost(m_geometry.ru_e.data(), grid_size, stream);
  d_ru_o_.CopyToHost(m_geometry.ru_o.data(), grid_size, stream);
  d_rv_e_.CopyToHost(m_geometry.rv_e.data(), grid_size, stream);
  d_rv_o_.CopyToHost(m_geometry.rv_o.data(), grid_size, stream);
  d_z1_e_.CopyToHost(m_geometry.z1_e.data(), grid_size, stream);
  d_z1_o_.CopyToHost(m_geometry.z1_o.data(), grid_size, stream);
  d_zu_e_.CopyToHost(m_geometry.zu_e.data(), grid_size, stream);
  d_zu_o_.CopyToHost(m_geometry.zu_o.data(), grid_size, stream);
  d_zv_e_.CopyToHost(m_geometry.zv_e.data(), grid_size, stream);
  d_zv_o_.CopyToHost(m_geometry.zv_o.data(), grid_size, stream);
  d_lu_e_.CopyToHost(m_geometry.lu_e.data(), grid_size, stream);
  d_lu_o_.CopyToHost(m_geometry.lu_o.data(), grid_size, stream);
  d_lv_e_.CopyToHost(m_geometry.lv_e.data(), grid_size, stream);
  d_lv_o_.CopyToHost(m_geometry.lv_o.data(), grid_size, stream);
  d_r_con_.CopyToHost(m_geometry.rCon.data(), con_size, stream);
  d_z_con_.CopyToHost(m_geometry.zCon.data(), con_size, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void ComputeBackendCudaImpl::ForcesToFourier(
    const RealSpaceForces& forces, const std::vector<double>& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces) {
  cudaStream_t stream = streams_[0];

  // Zero the output.
  m_physical_forces.setZero();

  // Determine j_max_rz.
  int j_max_rz = std::min(rp.nsMaxF, fc.ns - 1);
  if (fc.lfreeb &&
      (vacuum_pressure_state == VacuumPressureState::kInitialized ||
       vacuum_pressure_state == VacuumPressureState::kActive)) {
    j_max_rz = std::min(rp.nsMaxF, fc.ns);
  }

  // Prepare kernel parameters.
  KernelParams params;
  params.ns_min_f = rp.nsMinF;
  params.ns_max_f = rp.nsMaxF;
  params.ns_min_f1 = rp.nsMinF1;
  params.ns_max_f1 = rp.nsMaxF1;
  params.ns_max_f_including_lcfs = rp.nsMaxFIncludingLcfs;
  params.mpol = s.mpol;
  params.ntor = s.ntor;
  params.n_zeta = s.nZeta;
  params.n_theta_eff = s.nThetaEff;
  params.n_theta_reduced = s.nThetaReduced;
  params.nnyq2 = s.nnyq2;
  params.ns = fc.ns;
  params.lfreeb = fc.lfreeb;

  const int num_surfaces = j_max_rz - rp.nsMinF;
  if (num_surfaces <= 0) {
    return;
  }

  const int grid_size = num_surfaces * s.nZeta * s.nThetaEff;
  const int coeff_size = num_surfaces * s.mpol * (s.ntor + 1);

  // Copy input forces to device.
  auto copy_span = [&](DeviceBuffer<double>& buf, std::span<const double> sp) {
    std::vector<double> vec(sp.begin(), sp.end());
    buf.CopyFromHost(vec, stream);
  };

  copy_span(d_armn_e_, forces.armn_e);
  copy_span(d_armn_o_, forces.armn_o);
  copy_span(d_azmn_e_, forces.azmn_e);
  copy_span(d_azmn_o_, forces.azmn_o);
  copy_span(d_blmn_e_, forces.blmn_e);
  copy_span(d_blmn_o_, forces.blmn_o);
  copy_span(d_brmn_e_, forces.brmn_e);
  copy_span(d_brmn_o_, forces.brmn_o);
  copy_span(d_bzmn_e_, forces.bzmn_e);
  copy_span(d_bzmn_o_, forces.bzmn_o);
  copy_span(d_clmn_e_, forces.clmn_e);
  copy_span(d_clmn_o_, forces.clmn_o);
  copy_span(d_crmn_e_, forces.crmn_e);
  copy_span(d_crmn_o_, forces.crmn_o);
  copy_span(d_czmn_e_, forces.czmn_e);
  copy_span(d_czmn_o_, forces.czmn_o);
  copy_span(d_frcon_e_, forces.frcon_e);
  copy_span(d_frcon_o_, forces.frcon_o);
  copy_span(d_fzcon_e_, forces.fzcon_e);
  copy_span(d_fzcon_o_, forces.fzcon_o);

  d_xmpq_.CopyFromHost(xmpq, stream);

  // Copy basis functions (with integration weights).
  d_cosmui_.CopyFromHost(fb.cosmui, stream);
  d_sinmui_.CopyFromHost(fb.sinmui, stream);
  d_cosmumi_.CopyFromHost(fb.cosmumi, stream);
  d_sinmumi_.CopyFromHost(fb.sinmumi, stream);
  d_cosnv_.CopyFromHost(fb.cosnv, stream);
  d_sinnv_.CopyFromHost(fb.sinnv, stream);
  d_cosnvn_.CopyFromHost(fb.cosnvn, stream);
  d_sinnvn_.CopyFromHost(fb.sinnvn, stream);

  // Initialize output buffers to zero.
  d_frcc_.SetZero(coeff_size, stream);
  d_frss_.SetZero(coeff_size, stream);
  d_fzsc_.SetZero(coeff_size, stream);
  d_fzcs_.SetZero(coeff_size, stream);
  d_flsc_.SetZero(coeff_size, stream);
  d_flcs_.SetZero(coeff_size, stream);

  // Launch kernel.
  dim3 grid_dim(num_surfaces, s.mpol, 1);
  dim3 block_dim(s.nZeta, 1, 1);

  ForcesToFourierKernel<<<grid_dim, block_dim, 0, stream>>>(
      d_armn_e_.Data(), d_armn_o_.Data(), d_azmn_e_.Data(), d_azmn_o_.Data(),
      d_blmn_e_.Data(), d_blmn_o_.Data(), d_brmn_e_.Data(), d_brmn_o_.Data(),
      d_bzmn_e_.Data(), d_bzmn_o_.Data(), d_clmn_e_.Data(), d_clmn_o_.Data(),
      d_crmn_e_.Data(), d_crmn_o_.Data(), d_czmn_e_.Data(), d_czmn_o_.Data(),
      d_frcon_e_.Data(), d_frcon_o_.Data(), d_fzcon_e_.Data(), d_fzcon_o_.Data(),
      d_xmpq_.Data(), d_cosmui_.Data(), d_sinmui_.Data(), d_cosmumi_.Data(),
      d_sinmumi_.Data(), d_cosnv_.Data(), d_sinnv_.Data(), d_cosnvn_.Data(),
      d_sinnvn_.Data(), d_frcc_.Data(), d_frss_.Data(), d_fzsc_.Data(),
      d_fzcs_.Data(), d_flsc_.Data(), d_flcs_.Data(), params, j_max_rz);

  // Copy results back to host.
  d_frcc_.CopyToHost(m_physical_forces.frcc.data(), coeff_size, stream);
  d_frss_.CopyToHost(m_physical_forces.frss.data(), coeff_size, stream);
  d_fzsc_.CopyToHost(m_physical_forces.fzsc.data(), coeff_size, stream);
  d_fzcs_.CopyToHost(m_physical_forces.fzcs.data(), coeff_size, stream);
  d_flsc_.CopyToHost(m_physical_forces.flsc.data(), coeff_size, stream);
  d_flcs_.CopyToHost(m_physical_forces.flcs.data(), coeff_size, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
}

// =============================================================================
// ComputeBackendCuda
// =============================================================================

ComputeBackendCuda::ComputeBackendCuda(int device_id, int num_streams)
    : impl_(std::make_unique<ComputeBackendCudaImpl>(device_id, num_streams)) {}

ComputeBackendCuda::~ComputeBackendCuda() = default;

ComputeBackendCuda::ComputeBackendCuda(ComputeBackendCuda&&) noexcept = default;
ComputeBackendCuda& ComputeBackendCuda::operator=(ComputeBackendCuda&&) noexcept = default;

std::string ComputeBackendCuda::GetName() const {
  if (impl_ && impl_->IsAvailable()) {
    return "CUDA (" + impl_->GetDeviceName() + ")";
  }
  return "CUDA (unavailable)";
}

void ComputeBackendCuda::FourierToReal(const FourierGeometry& physical_x,
                                       const std::vector<double>& xmpq,
                                       const RadialPartitioning& rp,
                                       const Sizes& s,
                                       const RadialProfiles& profiles,
                                       const FourierBasisFastPoloidal& fb,
                                       RealSpaceGeometry& m_geometry) {
  if (!impl_ || !impl_->IsAvailable()) {
    throw std::runtime_error("CUDA backend is not available");
  }
  impl_->FourierToReal(physical_x, xmpq, rp, s, profiles, fb, m_geometry);
}

void ComputeBackendCuda::ForcesToFourier(
    const RealSpaceForces& forces, const std::vector<double>& xmpq,
    const RadialPartitioning& rp, const FlowControl& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb,
    VacuumPressureState vacuum_pressure_state,
    FourierForces& m_physical_forces) {
  if (!impl_ || !impl_->IsAvailable()) {
    throw std::runtime_error("CUDA backend is not available");
  }
  impl_->ForcesToFourier(forces, xmpq, rp, fc, s, fb, vacuum_pressure_state,
                         m_physical_forces);
}

void ComputeBackendCuda::Synchronize() {
  if (impl_) {
    impl_->Synchronize();
  }
}

bool ComputeBackendCuda::IsAvailable() const {
  return impl_ && impl_->IsAvailable();
}

int ComputeBackendCuda::GetDeviceId() const {
  return impl_ ? impl_->GetDeviceId() : -1;
}

std::string ComputeBackendCuda::GetDeviceName() const {
  return impl_ ? impl_->GetDeviceName() : "";
}

std::string ComputeBackendCuda::GetComputeCapability() const {
  return impl_ ? impl_->GetComputeCapability() : "";
}

size_t ComputeBackendCuda::GetDeviceMemoryBytes() const {
  return impl_ ? impl_->GetDeviceMemoryBytes() : 0;
}

}  // namespace vmecpp

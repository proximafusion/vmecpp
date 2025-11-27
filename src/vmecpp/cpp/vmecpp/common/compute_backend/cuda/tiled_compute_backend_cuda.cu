// SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH
// <info@proximafusion.com>
//
// SPDX-License-Identifier: MIT

#include "vmecpp/common/compute_backend/cuda/tiled_compute_backend_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <vector>

#include "vmecpp/common/compute_backend/compute_backend_cpu.h"
#include "vmecpp/common/compute_backend/cuda/cuda_memory.h"
#include "vmecpp/common/compute_backend/cuda/tile_memory_budget.h"
#include "vmecpp/common/compute_backend/cuda/tile_scheduler.h"

namespace vmecpp {

namespace {

// Helper to check CUDA errors
#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
      throw std::runtime_error(std::string("CUDA error: ") +           \
                               cudaGetErrorString(err));               \
    }                                                                  \
  } while (0)

// Compare arrays for validation
ValidationResult CompareArrays(const char* name, const double* computed,
                               const double* reference, size_t count,
                               double tol) {
  ValidationResult result;
  result.passed = true;

  for (size_t i = 0; i < count; ++i) {
    double abs_err = std::abs(computed[i] - reference[i]);
    double rel_err = 0.0;
    if (std::abs(reference[i]) > 1e-15) {
      rel_err = abs_err / std::abs(reference[i]);
    }

    if (abs_err > result.max_abs_error) {
      result.max_abs_error = abs_err;
    }
    if (rel_err > result.max_rel_error) {
      result.max_rel_error = rel_err;
    }

    if (abs_err > tol && rel_err > tol) {
      if (result.passed) {
        result.passed = false;
        result.error_location = std::string(name) + "[" + std::to_string(i) + "]";
        std::ostringstream ss;
        ss << "computed=" << computed[i] << ", reference=" << reference[i]
           << ", abs_err=" << abs_err << ", rel_err=" << rel_err;
        result.details = ss.str();
      }
    }
  }

  return result;
}

}  // namespace

// Implementation class with CUDA-specific members
class TiledComputeBackendCudaImpl {
 public:
  TiledComputeBackendCudaImpl(const TiledBackendConfig& config)
      : config_(config),
        stream_manager_(config.num_streams),
        cpu_backend_() {
    CUDA_CHECK(cudaSetDevice(config.device_id));
  }

  // Current problem parameters
  struct ProblemParams {
    int ns = 0;
    int mpol = 0;
    int ntor = 0;
    int n_zeta = 0;
    int n_theta_eff = 0;
    int tile_size = 0;
    bool tiling_enabled = false;
  };

  ProblemParams current_params_;
  TiledBackendConfig config_;
  StreamManager stream_manager_;
  ComputeBackendCpu cpu_backend_;

  // Tile scheduler (recreated for each problem size)
  std::unique_ptr<TileScheduler> dft_scheduler_;
  std::unique_ptr<TileScheduler> jacobian_scheduler_;
  std::unique_ptr<TileScheduler> mhd_scheduler_;

  // Device buffers for tiled execution (double-buffered)
  std::vector<DeviceBuffer> tile_buffers_;
  std::vector<PinnedBuffer> staging_buffers_;

  // Basis function caches (shared across tiles)
  DeviceBuffer d_cosmu_, d_sinmu_, d_cosmum_, d_sinmum_;
  DeviceBuffer d_cosnv_, d_sinnv_, d_cosnvn_, d_sinnvn_;
  DeviceBuffer d_cosmui_, d_sinmui_, d_cosmumi_, d_sinmumi_;
  bool basis_cached_ = false;
  int cached_mpol_ = 0;
  int cached_ntor_ = 0;
  int cached_n_zeta_ = 0;
  int cached_n_theta_ = 0;

  void UpdateProblemParams(int ns, int mpol, int ntor, int n_zeta,
                           int n_theta_eff) {
    if (ns == current_params_.ns && mpol == current_params_.mpol &&
        ntor == current_params_.ntor && n_zeta == current_params_.n_zeta &&
        n_theta_eff == current_params_.n_theta_eff) {
      return;  // No change
    }

    current_params_.ns = ns;
    current_params_.mpol = mpol;
    current_params_.ntor = ntor;
    current_params_.n_zeta = n_zeta;
    current_params_.n_theta_eff = n_theta_eff;

    // Calculate memory budget
    GridSizeParams grid_params{ns, mpol, ntor, n_zeta, n_theta_eff};
    MemoryBudget budget = TileMemoryBudget::Calculate(
        config_.device_id, grid_params, config_.memory_fraction);

    current_params_.tiling_enabled =
        config_.force_tiling || budget.tiling_required;

    if (current_params_.tiling_enabled) {
      // Use recommended tile size
      int recommended = TileMemoryBudget::RecommendTileSize(
          config_.device_id, grid_params, config_.memory_fraction, 1);
      current_params_.tile_size =
          std::max(config_.min_tile_size, recommended);
    } else {
      current_params_.tile_size = ns;
    }

    // Create schedulers for different operation types
    TileSchedulerConfig dft_config;
    dft_config.ns = ns;
    dft_config.tile_size = current_params_.tile_size;
    dft_config.ns_min = 1;
    dft_config.op_type = TileOperationType::kNoOverlap;
    dft_scheduler_ = std::make_unique<TileScheduler>(dft_config);

    TileSchedulerConfig jacobian_config = dft_config;
    jacobian_config.op_type = TileOperationType::kForwardStencil;
    jacobian_scheduler_ = std::make_unique<TileScheduler>(jacobian_config);

    TileSchedulerConfig mhd_config = dft_config;
    mhd_config.op_type = TileOperationType::kBackwardStencil;
    mhd_scheduler_ = std::make_unique<TileScheduler>(mhd_config);
  }

  void CacheBasisFunctions(const FourierBasisFastPoloidal& fb, int mpol,
                           int ntor, int n_zeta, int n_theta_reduced) {
    if (basis_cached_ && cached_mpol_ == mpol && cached_ntor_ == ntor &&
        cached_n_zeta_ == n_zeta && cached_n_theta_ == n_theta_reduced) {
      return;  // Already cached
    }

    cudaStream_t stream = stream_manager_.GetStream(0);

    // Copy basis arrays to device
    size_t poloidal_size = n_theta_reduced * mpol;
    size_t toroidal_size = n_zeta * (fb.nnyq2() + 1);

    d_cosmu_.CopyFromHost(fb.cosmuf().data(), poloidal_size * sizeof(double), stream);
    d_sinmu_.CopyFromHost(fb.sinmuf().data(), poloidal_size * sizeof(double), stream);
    d_cosmum_.CopyFromHost(fb.cosmumf().data(), poloidal_size * sizeof(double), stream);
    d_sinmum_.CopyFromHost(fb.sinmumf().data(), poloidal_size * sizeof(double), stream);

    d_cosnv_.CopyFromHost(fb.cosnv().data(), toroidal_size * sizeof(double), stream);
    d_sinnv_.CopyFromHost(fb.sinnv().data(), toroidal_size * sizeof(double), stream);
    d_cosnvn_.CopyFromHost(fb.cosnvn().data(), toroidal_size * sizeof(double), stream);
    d_sinnvn_.CopyFromHost(fb.sinnvn().data(), toroidal_size * sizeof(double), stream);

    // Integration-weighted versions
    d_cosmui_.CopyFromHost(fb.cosmuif().data(), poloidal_size * sizeof(double), stream);
    d_sinmui_.CopyFromHost(fb.sinmuif().data(), poloidal_size * sizeof(double), stream);
    d_cosmumi_.CopyFromHost(fb.cosmumif().data(), poloidal_size * sizeof(double), stream);
    d_sinmumi_.CopyFromHost(fb.sinmumif().data(), poloidal_size * sizeof(double), stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    basis_cached_ = true;
    cached_mpol_ = mpol;
    cached_ntor_ = ntor;
    cached_n_zeta_ = n_zeta;
    cached_n_theta_ = n_theta_reduced;
  }
};

// TiledComputeBackendCuda implementation

TiledComputeBackendCuda::TiledComputeBackendCuda(
    const TiledBackendConfig& config)
    : config_(config) {
  impl_ = std::make_unique<TiledComputeBackendCudaImpl>(config);
}

TiledComputeBackendCuda::~TiledComputeBackendCuda() = default;

void TiledComputeBackendCuda::FourierToReal(
    const FourierGeometry& physical_x, const std::vector<double>& xmpq,
    const RadialPartitioning& rp, const Sizes& s,
    const RadialProfiles& profiles, const FourierBasisFastPoloidal& fb,
    RealSpaceGeometry& m_geometry) {
  const int ns = rp.nsMaxF1 - rp.nsMinF1;
  const int mpol = s.mpol;
  const int ntor = s.ntor;

  impl_->UpdateProblemParams(ns, mpol, ntor, s.nZeta, s.nThetaEff);
  impl_->CacheBasisFunctions(fb, mpol, ntor, s.nZeta, s.nThetaReduced);

  if (!impl_->current_params_.tiling_enabled) {
    // Direct execution without tiling - use CPU backend for now
    // TODO: Use CUDA kernel directly when integrated
    impl_->cpu_backend_.FourierToReal(physical_x, xmpq, rp, s, profiles, fb,
                                       m_geometry);
    return;
  }

  // Tiled execution
  const auto& tiles = impl_->dft_scheduler_->GetTiles();

  // Validate before execution if enabled
  std::vector<double> reference_r1_e, reference_r1_o;
  if (config_.enable_validation) {
    // Compute reference using CPU
    RealSpaceGeometry ref_geom = m_geometry;  // Copy structure
    // Allocate reference arrays with same size
    reference_r1_e.resize(m_geometry.r1_e.size());
    reference_r1_o.resize(m_geometry.r1_o.size());
    // TODO: Deep copy and compute reference
  }

  // Process tiles
  for (const auto& tile : tiles) {
    cudaStream_t stream =
        impl_->stream_manager_.GetStream(tile.tile_index % config_.num_streams);

    // For each tile, compute output for surfaces [start, end)
    // DFT has no overlap, so each surface is independent

    // Create tile-local partitioning
    RadialPartitioning tile_rp = rp;
    tile_rp.nsMinF1 = rp.nsMinF1 + tile.start_surface - 1;
    tile_rp.nsMaxF1 = rp.nsMinF1 + tile.end_surface - 1;

    // TODO: Implement GPU kernel for tile
    // For now, use CPU backend for correctness
    impl_->cpu_backend_.FourierToReal(physical_x, xmpq, tile_rp, s, profiles,
                                       fb, m_geometry);
  }

  impl_->stream_manager_.SynchronizeAll();

  // Validate after execution if enabled
  if (config_.enable_validation) {
    last_validation_ = CompareArrays("r1_e", m_geometry.r1_e.data(),
                                     reference_r1_e.data(),
                                     m_geometry.r1_e.size(), config_.validation_tol);
    if (!last_validation_.passed) {
      last_error_ = TiledBackendError::kValidationFailed;
    }
  }
}

void TiledComputeBackendCuda::ForcesToFourier(
    const RealSpaceForces& forces, const std::vector<double>& xmpq,
    const RadialPartitioning& rp, const ForceCoefficients& fc, const Sizes& s,
    const FourierBasisFastPoloidal& fb, const VacuumState vacuum_state,
    ForceFourierCoefficients& m_forces) {
  const int ns = rp.nsMaxF1 - rp.nsMinF1;
  const int mpol = s.mpol;
  const int ntor = s.ntor;

  impl_->UpdateProblemParams(ns, mpol, ntor, s.nZeta, s.nThetaEff);

  if (!impl_->current_params_.tiling_enabled) {
    impl_->cpu_backend_.ForcesToFourier(forces, xmpq, rp, fc, s, fb,
                                         vacuum_state, m_forces);
    return;
  }

  // Tiled execution
  const auto& tiles = impl_->dft_scheduler_->GetTiles();

  // Zero output arrays before accumulation
  std::fill(m_forces.frcc.begin(), m_forces.frcc.end(), 0.0);
  std::fill(m_forces.frss.begin(), m_forces.frss.end(), 0.0);
  std::fill(m_forces.fzsc.begin(), m_forces.fzsc.end(), 0.0);
  std::fill(m_forces.fzcs.begin(), m_forces.fzcs.end(), 0.0);
  std::fill(m_forces.flsc.begin(), m_forces.flsc.end(), 0.0);
  std::fill(m_forces.flcs.begin(), m_forces.flcs.end(), 0.0);

  for (const auto& tile : tiles) {
    RadialPartitioning tile_rp = rp;
    tile_rp.nsMinF1 = rp.nsMinF1 + tile.start_surface - 1;
    tile_rp.nsMaxF1 = rp.nsMinF1 + tile.end_surface - 1;

    // TODO: Implement GPU kernel for tile
    impl_->cpu_backend_.ForcesToFourier(forces, xmpq, tile_rp, fc, s, fb,
                                         vacuum_state, m_forces);
  }

  impl_->stream_manager_.SynchronizeAll();
}

bool TiledComputeBackendCuda::ComputeJacobian(const JacobianInput& input,
                                              const RadialPartitioning& rp,
                                              const Sizes& s,
                                              JacobianOutput& m_output) {
  const int ns = rp.nsMaxH - rp.nsMinH;

  impl_->UpdateProblemParams(ns, s.mpol, s.ntor, s.nZeta, s.nThetaEff);

  if (!impl_->current_params_.tiling_enabled) {
    return impl_->cpu_backend_.ComputeJacobian(input, rp, s, m_output);
  }

  // Tiled execution with overlap handling
  const auto& tiles = impl_->jacobian_scheduler_->GetTiles();

  bool has_bad_jacobian = false;

  for (const auto& tile : tiles) {
    // Jacobian needs surfaces jF and jF+1 to compute half-grid point jH
    // Tile has overlap_after = 1 for non-last tiles

    RadialPartitioning tile_rp = rp;
    tile_rp.nsMinH = rp.nsMinH + tile.start_surface - 1;
    tile_rp.nsMaxH = rp.nsMinH + tile.end_surface - 1;

    // Adjust full-grid range to include overlap
    tile_rp.nsMinF1 = tile_rp.nsMinH;
    tile_rp.nsMaxF1 = tile_rp.nsMaxH + 1 + tile.overlap_after;

    // TODO: Implement GPU kernel for tile
    bool tile_bad = impl_->cpu_backend_.ComputeJacobian(input, tile_rp, s, m_output);
    has_bad_jacobian = has_bad_jacobian || tile_bad;
  }

  impl_->stream_manager_.SynchronizeAll();
  return has_bad_jacobian;
}

void TiledComputeBackendCuda::ComputeMetricElements(const MetricInput& input,
                                                    const RadialPartitioning& rp,
                                                    const Sizes& s,
                                                    MetricOutput& m_output) {
  const int ns = rp.nsMaxH - rp.nsMinH;

  impl_->UpdateProblemParams(ns, s.mpol, s.ntor, s.nZeta, s.nThetaEff);

  if (!impl_->current_params_.tiling_enabled) {
    impl_->cpu_backend_.ComputeMetricElements(input, rp, s, m_output);
    return;
  }

  // Tiled execution - same overlap pattern as Jacobian
  const auto& tiles = impl_->jacobian_scheduler_->GetTiles();

  for (const auto& tile : tiles) {
    RadialPartitioning tile_rp = rp;
    tile_rp.nsMinH = rp.nsMinH + tile.start_surface - 1;
    tile_rp.nsMaxH = rp.nsMinH + tile.end_surface - 1;
    tile_rp.nsMinF1 = tile_rp.nsMinH;
    tile_rp.nsMaxF1 = tile_rp.nsMaxH + 1 + tile.overlap_after;

    impl_->cpu_backend_.ComputeMetricElements(input, tile_rp, s, m_output);
  }

  impl_->stream_manager_.SynchronizeAll();
}

void TiledComputeBackendCuda::ComputeBContra(const BContraInput& input,
                                             const RadialPartitioning& rp,
                                             const Sizes& s,
                                             BContraOutput& m_output) {
  // BContra has global reductions - handle specially
  // For now, always use CPU for this operation
  impl_->cpu_backend_.ComputeBContra(input, rp, s, m_output);
}

void TiledComputeBackendCuda::ComputeMHDForces(const MHDForcesInput& input,
                                               const RadialPartitioning& rp,
                                               const Sizes& s,
                                               MHDForcesOutput& m_output) {
  const int ns = rp.nsMaxF - rp.nsMinF;

  impl_->UpdateProblemParams(ns, s.mpol, s.ntor, s.nZeta, s.nThetaEff);

  if (!impl_->current_params_.tiling_enabled) {
    impl_->cpu_backend_.ComputeMHDForces(input, rp, s, m_output);
    return;
  }

  // Tiled execution with backward stencil (needs jH-1)
  const auto& tiles = impl_->mhd_scheduler_->GetTiles();

  for (const auto& tile : tiles) {
    RadialPartitioning tile_rp = rp;
    tile_rp.nsMinF = rp.nsMinF + tile.start_surface - 1;
    tile_rp.nsMaxF = rp.nsMinF + tile.end_surface - 1;

    // Adjust half-grid range to include overlap before
    tile_rp.nsMinH = tile_rp.nsMinF - tile.overlap_before;
    tile_rp.nsMaxH = tile_rp.nsMaxF;

    impl_->cpu_backend_.ComputeMHDForces(input, tile_rp, s, m_output);
  }

  impl_->stream_manager_.SynchronizeAll();
}

bool TiledComputeBackendCuda::IsTilingRequired(int ns, int mpol, int ntor,
                                               int n_zeta,
                                               int n_theta_eff) const {
  if (config_.force_tiling) {
    return true;
  }

  GridSizeParams params{ns, mpol, ntor, n_zeta, n_theta_eff};
  return TileMemoryBudget::TilingRequired(config_.device_id, params,
                                          config_.memory_fraction);
}

MemoryBudget TiledComputeBackendCuda::GetMemoryBudget(int ns, int mpol,
                                                      int ntor, int n_zeta,
                                                      int n_theta_eff) const {
  GridSizeParams params{ns, mpol, ntor, n_zeta, n_theta_eff};
  return TileMemoryBudget::Calculate(config_.device_id, params,
                                     config_.memory_fraction);
}

std::string TiledComputeBackendCuda::GetStatusReport() const {
  std::ostringstream ss;

  ss << "=== Tiled CUDA Backend Status ===\n";
  ss << "Device ID: " << config_.device_id << "\n";
  ss << "Memory fraction: " << (config_.memory_fraction * 100) << "%\n";
  ss << "Number of streams: " << config_.num_streams << "\n";
  ss << "Validation: " << (config_.enable_validation ? "enabled" : "disabled")
     << "\n";
  ss << "\n";

  if (impl_->current_params_.ns > 0) {
    ss << "Current problem:\n";
    ss << "  ns=" << impl_->current_params_.ns
       << ", mpol=" << impl_->current_params_.mpol
       << ", ntor=" << impl_->current_params_.ntor << "\n";
    ss << "  Grid: " << impl_->current_params_.n_zeta << " x "
       << impl_->current_params_.n_theta_eff << "\n";
    ss << "  Tiling: "
       << (impl_->current_params_.tiling_enabled ? "enabled" : "disabled")
       << "\n";
    if (impl_->current_params_.tiling_enabled) {
      ss << "  Tile size: " << impl_->current_params_.tile_size << "\n";
      if (impl_->dft_scheduler_) {
        ss << "  Number of tiles: " << impl_->dft_scheduler_->NumTiles()
           << "\n";
      }
    }
  }

  return ss.str();
}

void TiledComputeBackendCuda::SetValidationEnabled(bool enabled) {
  config_.enable_validation = enabled;
}

void TiledComputeBackendCuda::SetTileSize(int tile_size) {
  impl_->current_params_.tile_size = tile_size;
  if (tile_size > 0) {
    config_.force_tiling = true;
  }
}

int TiledComputeBackendCuda::GetTileSize() const {
  return impl_->current_params_.tile_size;
}

int TiledComputeBackendCuda::GetNumTiles() const {
  if (impl_->dft_scheduler_) {
    return impl_->dft_scheduler_->NumTiles();
  }
  return 1;
}

}  // namespace vmecpp

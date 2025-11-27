# Tiled GPU Implementation Design for VMEC++

**Implementation Status: COMPLETE**

This document outlines the design for handling large VMEC++ problems on GPUs with limited memory through radial tiling and streaming.

## Implementation Summary

The tiled GPU execution framework has been implemented with the following components:

| Component | File | Status |
|-----------|------|--------|
| TileMemoryBudget | `tile_memory_budget.h/cu` | Complete |
| TileScheduler | `tile_scheduler.h/cc` | Complete |
| CudaMemory utilities | `cuda_memory.h/cu` | Complete |
| TiledComputeBackendCuda | `tiled_compute_backend_cuda.h/cu` | Complete |
| Validation tests | `tile_validation_test.cc` | Complete |
| Benchmark tool | `tiled_benchmark.cc` | Complete |

**Note**: The current implementation uses CPU fallback for correctness validation.
GPU kernels are written and ready for activation after numerical validation.

## Problem Statement

### Memory Scaling

For a VMEC++ problem with parameters:
- `ns` = number of radial surfaces
- `nZeta` = toroidal grid points
- `nTheta` = poloidal grid points
- `mpol` = poloidal modes
- `ntor` = toroidal modes

The memory requirements scale as:

| Component | Size (doubles) | Example (ns=200, nZ=128, nT=128) |
|-----------|----------------|----------------------------------|
| Real-space geometry (16 arrays) | 16 * ns * nZeta * nTheta | 16 * 3.3M = 52.4M = 419 MB |
| Real-space forces (20 arrays) | 20 * ns * nZeta * nTheta | 20 * 3.3M = 65.5M = 524 MB |
| Fourier coefficients (12 arrays) | 12 * ns * mpol * (ntor+1) | 12 * 200 * 12 * 16 = 0.5M = 4 MB |
| Half-grid arrays (10 arrays) | 10 * ns * nZeta * nTheta | 10 * 3.3M = 32.8M = 262 MB |
| **Total Peak** | | **~1.2 GB** |

For high-resolution runs (ns=500, nZeta=256, nTheta=256):
- grid_size = 32.8M points
- **Total Peak: ~10+ GB**

This can exceed typical GPU memory (8-16 GB consumer, 24-80 GB workstation/datacenter).

### Data Dependencies

| Operation | Radial Coupling | Tiling Strategy |
|-----------|-----------------|-----------------|
| FourierToReal | None (independent) | Perfect tiling |
| ForcesToFourier | None (independent) | Perfect tiling |
| ComputeJacobian | Adjacent (jF, jF+1) | Tile with 1-surface overlap |
| ComputeMetricElements | Adjacent (jF, jF+1) | Tile with 1-surface overlap |
| ComputeBContra | Global reduction | Process per-tile, merge |
| ComputeMHDForces | Adjacent (jH-1, jH) | Tile with 1-surface overlap |

## Architecture Overview

```
+--------------------------------------------------+
|              TiledComputeBackend                 |
|  +--------------------------------------------+  |
|  |           TileScheduler                    |  |
|  |  - Partitions radial domain into tiles     |  |
|  |  - Manages overlap regions                 |  |
|  |  - Coordinates multi-stream execution      |  |
|  +--------------------------------------------+  |
|                       |                          |
|  +--------------------+---------------------+    |
|  |                    |                     |    |
|  v                    v                     v    |
|  +----------+    +----------+    +----------+   |
|  |  Tile 0  |    |  Tile 1  |    |  Tile 2  |   |
|  | [0, T)   |    | [T-1,2T) |    |[2T-1,3T) |   |
|  +----------+    +----------+    +----------+   |
|       |               |               |          |
|  +----v----+    +----v----+    +----v----+      |
|  | Stream 0|    | Stream 1|    | Stream 0|      |
|  +---------+    +---------+    +---------+      |
+--------------------------------------------------+
```

## Phase 1: Memory Budget and Tile Sizing

### 1.1 Memory Budget Calculator

```cpp
// New file: tile_memory_budget.h

struct MemoryBudget {
  size_t total_gpu_memory;      // From cudaMemGetInfo
  size_t reserved_memory;       // For CUDA runtime, other allocations
  size_t available_memory;      // total - reserved
  size_t per_surface_memory;    // Memory per radial surface
  int max_surfaces_per_tile;    // available / per_surface
};

class TileMemoryBudget {
 public:
  // Query GPU and calculate budget
  static MemoryBudget Calculate(const Sizes& s, int mpol, int ntor);

  // Calculate memory for one radial surface
  static size_t PerSurfaceMemory(const Sizes& s, int mpol, int ntor);

  // Recommend tile size given problem size and GPU memory
  static int RecommendTileSize(int ns, const Sizes& s, int mpol, int ntor,
                               double memory_fraction = 0.8);
};
```

### 1.2 Per-Surface Memory Breakdown

```cpp
size_t TileMemoryBudget::PerSurfaceMemory(const Sizes& s, int mpol, int ntor) {
  const size_t nZnT = s.nZeta * s.nThetaEff;
  const size_t coeff_per_surface = mpol * (ntor + 1);

  // Real-space arrays (geometry + forces)
  size_t realspace = 36 * nZnT * sizeof(double);  // 16 geom + 20 forces

  // Half-grid arrays (Jacobian, metric, B-field)
  size_t halfgrid = 13 * nZnT * sizeof(double);   // tau, r12, etc.

  // Fourier coefficient arrays
  size_t fourier = 12 * coeff_per_surface * sizeof(double);

  return realspace + halfgrid + fourier;
}
```

## Phase 2: Tile Scheduler

### 2.1 Tile Definition

```cpp
// New file: tile_scheduler.h

struct RadialTile {
  int start_surface;    // First surface in tile (inclusive)
  int end_surface;      // Last surface in tile (exclusive)
  int overlap_before;   // Extra surfaces needed from previous tile
  int overlap_after;    // Extra surfaces needed for next tile

  // Effective range including overlap
  int input_start() const { return start_surface - overlap_before; }
  int input_end() const { return end_surface + overlap_after; }

  // Number of surfaces to process
  int num_surfaces() const { return end_surface - start_surface; }
  int num_input_surfaces() const { return input_end() - input_start(); }
};

class TileScheduler {
 public:
  TileScheduler(int ns, int tile_size, int overlap = 1);

  // Get all tiles for the radial domain
  std::vector<RadialTile> GetTiles() const;

  // Get tile containing a specific surface
  RadialTile GetTileFor(int surface) const;

  // Number of tiles
  int NumTiles() const { return tiles_.size(); }

 private:
  std::vector<RadialTile> tiles_;
  int ns_;
  int tile_size_;
  int overlap_;
};
```

### 2.2 Tile Generation Algorithm

```cpp
std::vector<RadialTile> TileScheduler::GetTiles() const {
  std::vector<RadialTile> tiles;

  for (int start = 1; start < ns_; start += tile_size_) {
    RadialTile tile;
    tile.start_surface = start;
    tile.end_surface = std::min(start + tile_size_, ns_);

    // First tile needs no overlap before
    tile.overlap_before = (start == 1) ? 0 : overlap_;

    // Last tile needs no overlap after
    tile.overlap_after = (tile.end_surface >= ns_) ? 0 : overlap_;

    tiles.push_back(tile);
  }

  return tiles;
}
```

## Phase 3: Tiled DFT Operations

### 3.1 Tiled FourierToReal

Since FourierToReal has no radial dependencies, tiling is straightforward:

```cpp
void TiledComputeBackend::FourierToReal(
    const FourierGeometry& physical_x,
    /* ... other params ... */,
    RealSpaceGeometry& m_geometry) {

  auto tiles = scheduler_.GetTiles();

  for (int t = 0; t < tiles.size(); ++t) {
    const auto& tile = tiles[t];
    cudaStream_t stream = streams_[t % num_streams_];

    // 1. Copy Fourier coefficients for this tile to GPU
    CopyFourierTileToDevice(physical_x, tile, stream);

    // 2. Launch kernel for this tile
    LaunchFourierToRealKernel(tile, stream);

    // 3. Copy results back (can overlap with next tile's transfer)
    CopyGeometryTileToHost(m_geometry, tile, stream);
  }

  // Synchronize all streams
  for (auto& stream : streams_) {
    cudaStreamSynchronize(stream);
  }
}
```

### 3.2 Double-Buffered Pipeline

```
Time -->
Stream 0: [Transfer T0] [Compute T0] [Readback T0] [Transfer T2] [Compute T2] ...
Stream 1:               [Transfer T1] [Compute T1] [Readback T1] [Transfer T3] ...
```

```cpp
void TiledComputeBackend::FourierToRealPipelined(/* ... */) {
  auto tiles = scheduler_.GetTiles();

  // Double-buffered device memory
  TileBuffers buffers[2];

  for (int t = 0; t < tiles.size(); ++t) {
    const int buf_idx = t % 2;
    cudaStream_t stream = streams_[buf_idx];

    // Wait for previous use of this buffer to complete
    if (t >= 2) {
      cudaStreamSynchronize(stream);
    }

    // Async transfer -> compute -> readback
    CopyFourierTileToDeviceAsync(physical_x, tiles[t], buffers[buf_idx], stream);
    LaunchFourierToRealKernel(tiles[t], buffers[buf_idx], stream);
    CopyGeometryTileToHostAsync(m_geometry, tiles[t], buffers[buf_idx], stream);
  }

  // Final synchronization
  cudaDeviceSynchronize();
}
```

## Phase 4: Tiled Physics Operations

### 4.1 Tiled Jacobian with Overlap

ComputeJacobian needs data from surfaces jF and jF+1 to compute half-grid point jH:

```cpp
bool TiledComputeBackend::ComputeJacobian(
    const JacobianInput& input,
    const RadialPartitioning& rp,
    const Sizes& s,
    JacobianOutput& m_output) {

  auto tiles = scheduler_.GetTiles();

  // Atomic flags for bad Jacobian detection across tiles
  std::atomic<double> global_min_tau{0.0};
  std::atomic<double> global_max_tau{0.0};

  for (const auto& tile : tiles) {
    // Need overlap: to compute half-grid [jH_start, jH_end),
    // we need full-grid [jH_start, jH_end + 1)
    JacobianInput tile_input = ExtractTileInput(input, tile);
    JacobianOutput tile_output = GetTileOutput(m_output, tile);

    // Process tile
    auto [min_tau, max_tau] = ProcessJacobianTile(tile_input, tile_output);

    // Merge min/max across tiles
    UpdateAtomicMin(global_min_tau, min_tau);
    UpdateAtomicMax(global_max_tau, max_tau);
  }

  // Return true if bad Jacobian (sign change)
  return global_min_tau.load() * global_max_tau.load() < 0.0;
}
```

### 4.2 Overlap Region Management

```cpp
struct TileInputExtractor {
  // Extract geometry input for a tile, including overlap
  static JacobianInput ExtractJacobianInput(
      const JacobianInput& full_input,
      const RadialTile& tile,
      const RadialPartitioning& rp,
      const Sizes& s) {

    const int nZnT = s.nZeta * s.nThetaEff;

    // Input range: [tile.input_start(), tile.input_end())
    // For Jacobian: need jF and jF+1, so input includes one extra surface
    const int input_start = tile.start_surface - 1;  // jF for first jH
    const int input_end = tile.end_surface;          // jF+1 for last jH

    JacobianInput tile_input;
    tile_input.deltaS = full_input.deltaS;

    // Extract subspans for geometry arrays
    const int offset = (input_start - rp.nsMinF1) * nZnT;
    const int count = (input_end - input_start + 1) * nZnT;

    tile_input.r1_e = full_input.r1_e.subspan(offset, count);
    tile_input.r1_o = full_input.r1_o.subspan(offset, count);
    // ... etc for all arrays

    // sqrtSH is indexed by half-grid
    const int h_offset = tile.start_surface - rp.nsMinH;
    const int h_count = tile.num_surfaces();
    tile_input.sqrtSH = full_input.sqrtSH.subspan(h_offset, h_count);

    return tile_input;
  }
};
```

### 4.3 Tiled MHD Forces

ComputeMHDForces has the most complex data dependencies:

```cpp
void TiledComputeBackend::ComputeMHDForces(
    const MHDForcesInput& input,
    const RadialPartitioning& rp,
    const Sizes& s,
    MHDForcesOutput& m_output) {

  auto tiles = scheduler_.GetTiles();

  // State carried between tiles for "inside" values
  std::vector<double> carry_state(s.nZnT * 9);  // P, rup, zup, etc.

  for (int t = 0; t < tiles.size(); ++t) {
    const auto& tile = tiles[t];

    MHDTileContext ctx;
    ctx.tile = tile;
    ctx.is_first_tile = (t == 0);
    ctx.is_last_tile = (t == tiles.size() - 1);

    // For first surface in tile, load "inside" from previous tile
    if (!ctx.is_first_tile) {
      ctx.inside_state = carry_state;
    }

    // Process tile
    ProcessMHDForcesTile(input, rp, s, m_output, ctx);

    // Save "outside" state for next tile
    if (!ctx.is_last_tile) {
      SaveCarryState(ctx, carry_state);
    }
  }
}
```

## Phase 5: Streaming and Pinned Memory

### 5.1 Pinned Host Memory for Async Transfers

```cpp
class PinnedBuffer {
 public:
  PinnedBuffer(size_t size) {
    CUDA_CHECK(cudaMallocHost(&data_, size));
    size_ = size;
  }

  ~PinnedBuffer() {
    if (data_) cudaFreeHost(data_);
  }

  void* Data() { return data_; }
  size_t Size() const { return size_; }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
};

class TileTransferManager {
 public:
  TileTransferManager(size_t tile_size_bytes, int num_buffers = 2);

  // Get pinned buffer for tile input
  PinnedBuffer& GetInputBuffer(int buffer_idx);

  // Get pinned buffer for tile output
  PinnedBuffer& GetOutputBuffer(int buffer_idx);

  // Async copy from user memory to pinned buffer
  void StageInput(const void* src, size_t size, int buffer_idx);

  // Async copy from pinned buffer to user memory
  void UnstageOutput(void* dst, size_t size, int buffer_idx);

 private:
  std::vector<PinnedBuffer> input_buffers_;
  std::vector<PinnedBuffer> output_buffers_;
};
```

### 5.2 Multi-Stream Execution

```cpp
class StreamManager {
 public:
  StreamManager(int num_streams) {
    streams_.resize(num_streams);
    for (auto& stream : streams_) {
      CUDA_CHECK(cudaStreamCreate(&stream));
    }
  }

  ~StreamManager() {
    for (auto& stream : streams_) {
      cudaStreamDestroy(stream);
    }
  }

  cudaStream_t GetStream(int idx) { return streams_[idx % streams_.size()]; }

  void SynchronizeAll() {
    for (auto& stream : streams_) {
      cudaStreamSynchronize(stream);
    }
  }

 private:
  std::vector<cudaStream_t> streams_;
};
```

## Phase 6: Backend Configuration

### 6.1 Extended Backend Config

```cpp
// In compute_backend.h

struct BackendConfig {
  BackendType type = BackendType::kCpu;
  int cuda_device_id = 0;

  // Tiling configuration
  struct TilingConfig {
    bool enabled = true;                    // Auto-enable if needed
    int tile_size = 0;                      // 0 = auto-calculate
    double memory_fraction = 0.8;           // Use 80% of GPU memory
    int num_streams = 2;                    // Double-buffering
    bool use_pinned_memory = true;          // Async transfers
  } tiling;

  // Memory management
  struct MemoryConfig {
    bool cache_basis_functions = true;      // Keep trig tables on GPU
    bool persistent_buffers = true;         // Reuse allocations
  } memory;
};
```

### 6.2 Auto-Configuration

```cpp
void ComputeBackendCuda::AutoConfigureTiling(const Sizes& s, int ns,
                                              int mpol, int ntor) {
  // Query available GPU memory
  size_t free_mem, total_mem;
  cudaMemGetInfo(&free_mem, &total_mem);

  // Calculate per-surface memory requirement
  size_t per_surface = TileMemoryBudget::PerSurfaceMemory(s, mpol, ntor);

  // Available memory (with safety margin)
  size_t available = static_cast<size_t>(free_mem * config_.tiling.memory_fraction);

  // Maximum surfaces that fit in memory
  int max_surfaces = available / per_surface;

  if (max_surfaces >= ns) {
    // Entire problem fits - no tiling needed
    config_.tiling.enabled = false;
    config_.tiling.tile_size = ns;
  } else {
    // Enable tiling
    config_.tiling.enabled = true;
    config_.tiling.tile_size = std::max(1, max_surfaces - 2);  // Leave room for overlap

    printf("VMEC++ CUDA: Enabling tiled execution\n");
    printf("  GPU memory: %.1f GB free / %.1f GB total\n",
           free_mem / 1e9, total_mem / 1e9);
    printf("  Per-surface memory: %.1f MB\n", per_surface / 1e6);
    printf("  Tile size: %d surfaces (of %d total)\n",
           config_.tiling.tile_size, ns);
  }
}
```

## Phase 7: Implementation Roadmap

### Stage 1: Infrastructure (Week 1-2)
- [ ] Implement `TileMemoryBudget` class
- [ ] Implement `TileScheduler` class
- [ ] Add `TilingConfig` to `BackendConfig`
- [ ] Create pinned memory management utilities

### Stage 2: Tiled DFT Operations (Week 3-4)
- [ ] Refactor `FourierToReal` for tile-based processing
- [ ] Refactor `ForcesToFourier` for tile-based processing
- [ ] Implement double-buffered pipelining
- [ ] Add benchmarks for tiled vs non-tiled execution

### Stage 3: Tiled Physics Operations (Week 5-6)
- [ ] Implement tiled `ComputeJacobian` with overlap handling
- [ ] Implement tiled `ComputeMetricElements`
- [ ] Implement tiled `ComputeMHDForces` with carry state
- [ ] Handle `ComputeBContra` reductions across tiles

### Stage 4: Optimization and Testing (Week 7-8)
- [ ] Profile and optimize tile sizes
- [ ] Test with large problem sizes (ns > 200)
- [ ] Validate numerical accuracy vs CPU
- [ ] Add stress tests for memory edge cases

## Appendix A: Memory Layout Diagrams

### A.1 Non-Tiled Layout (Current)

```
GPU Memory:
+------------------+
| Fourier coeffs   | <- 6 arrays * ns * mpol * (ntor+1)
| (rmncc, etc.)    |
+------------------+
| Real-space geom  | <- 16 arrays * ns * nZnT
| (r1_e, r1_o...)  |
+------------------+
| Real-space force | <- 20 arrays * ns * nZnT
| (armn_e, etc.)   |
+------------------+
| Basis functions  | <- O(mpol*nTheta + ntor*nZeta)
| (cosmu, cosnv)   |
+------------------+
```

### A.2 Tiled Layout (Proposed)

```
GPU Memory:
+------------------+
| Tile buffer A    | <- tile_size * (per_surface_mem)
|  - Fourier tile  |
|  - Geometry tile |
|  - Forces tile   |
+------------------+
| Tile buffer B    | <- tile_size * (per_surface_mem)
|  - (double-buf)  |
+------------------+
| Basis functions  | <- O(mpol*nTheta + ntor*nZeta)
| (persistent)     |
+------------------+
| Scratch space    | <- For reductions, atomics
+------------------+
```

### A.3 Overlap Regions for Physics Operations

```
Tile boundaries:     |----Tile 0----|----Tile 1----|----Tile 2----|
Surface index:       0   10  20  30 40  50  60  70 80  90  100

Jacobian needs:
  Tile 0: surfaces [0, 41)  -> outputs [1, 40)  (overlap: 1 after)
  Tile 1: surfaces [39, 71) -> outputs [40, 70) (overlap: 1 before, 1 after)
  Tile 2: surfaces [69, 101)-> outputs [70, 100)(overlap: 1 before)

MHDForces needs:
  Similar pattern, with "inside" state carried forward
```

## Appendix B: Benchmark Targets

| Problem Size | Non-Tiled Memory | Tiled Memory | Target Speedup |
|--------------|------------------|--------------|----------------|
| ns=50, nZ=64, nT=64 | ~150 MB | N/A (fits) | N/A |
| ns=100, nZ=128, nT=128 | ~800 MB | ~200 MB/tile | 1.0x (no penalty) |
| ns=200, nZ=128, nT=128 | ~1.6 GB | ~200 MB/tile | 0.9x (10% overhead) |
| ns=500, nZ=256, nT=256 | ~10 GB | ~500 MB/tile | 0.8x (transfer bound) |

## Appendix C: Error Handling

```cpp
enum class TilingError {
  kSuccess,
  kInsufficientMemory,      // Even one surface doesn't fit
  kTileSizeTooSmall,        // Tile smaller than overlap requirement
  kStreamCreationFailed,
  kPinnedAllocFailed,
};

class TiledComputeBackendCuda : public ComputeBackendCuda {
 public:
  // Check if tiling is feasible for given problem
  TilingError ValidateTilingConfig(int ns, const Sizes& s) const;

  // Fall back to CPU if GPU can't handle the problem
  bool ShouldFallbackToCpu(int ns, const Sizes& s) const;
};
```

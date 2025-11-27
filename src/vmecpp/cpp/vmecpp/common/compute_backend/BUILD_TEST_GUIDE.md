# CUDA Compute Backend Build and Test Guide

This guide provides step-by-step instructions for building, testing, validating, and benchmarking the CUDA compute backends for VMEC++.

## Quick Start

### 1. Standalone Validation (No Dependencies)

The fastest way to validate the core tiling logic:

```bash
# Compile and run standalone validation tests
g++ -std=c++20 -O2 -I src/vmecpp/cpp \
  src/vmecpp/cpp/vmecpp/common/compute_backend/cuda/standalone_validation.cc \
  src/vmecpp/cpp/vmecpp/common/compute_backend/cuda/tile_scheduler.cc \
  -o standalone_validation

./standalone_validation
```

Expected output:
```
==============================================
  VMEC++ Tiled GPU Backend Validation Tests
==============================================

Running TileScheduler tests:
  Testing single tile covers entire domain... done
  ...

Running numerical validation tests:
  Testing tiled summation matches non-tiled... done
  ...

==============================================
  Results: 15424/15424 tests passed
==============================================
```

## Full Build Instructions

### Prerequisites

- CMake 3.18+
- GCC 11+ or Clang 14+ (C++20 support required)
- CUDA Toolkit 11.0+ (for GPU support)
- HDF5 development libraries
- netCDF development libraries
- LAPACK

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake \
  libhdf5-dev libnetcdf-dev \
  liblapack-dev libopenmp-dev

# For CUDA support (adjust version as needed)
# Follow: https://developer.nvidia.com/cuda-downloads
```

### Build Without CUDA (CPU Only)

```bash
# Configure
cmake -B build -DVMECPP_ENABLE_CUDA=OFF

# Build
cmake --build build --parallel

# Run Python tests
pytest tests/
```

### Build With CUDA

```bash
# Configure with CUDA
cmake -B build -DVMECPP_ENABLE_CUDA=ON

# Build
cmake --build build --parallel

# Verify CUDA backend is enabled
./build/backend_benchmark --help
```

## Testing

### 1. Python Test Suite

```bash
# Full test suite
pytest tests/

# Specific test
pytest tests/test_simsopt_compat.py -v
```

### 2. Standalone Tile Scheduler Tests

These tests validate the core tiling logic without requiring the full build:

```bash
g++ -std=c++20 -O2 -I src/vmecpp/cpp \
  src/vmecpp/cpp/vmecpp/common/compute_backend/cuda/standalone_validation.cc \
  src/vmecpp/cpp/vmecpp/common/compute_backend/cuda/tile_scheduler.cc \
  -o standalone_validation

./standalone_validation
```

#### What's Tested

| Test Category | Count | Description |
|---------------|-------|-------------|
| TileScheduler logic | 9 | Coverage, overlap, iteration |
| Numerical validation | 6 | Summation, stencil, reduction |
| Stress test configs | 140 | Various ns, tile_size combinations |
| **Total** | **15,424** | All edge cases |

### 3. GTest Suite (Full Build)

```bash
# Build tests
cmake -B build -DVMECPP_ENABLE_CUDA=ON
cmake --build build --target tile_validation_test

# Run
./build/tile_validation_test
```

## Benchmarking

### CPU vs CUDA Backend Benchmark

```bash
# Build with CUDA
cmake -B build -DVMECPP_ENABLE_CUDA=ON
cmake --build build --parallel

# Run default benchmark
./build/backend_benchmark

# Custom configuration
./build/backend_benchmark \
  --ns=100 \
  --mpol=20 \
  --ntor=20 \
  --nzeta=64 \
  --ntheta=64 \
  --iterations=50
```

### Tiled Execution Benchmark

```bash
# Build
cmake -B build -DVMECPP_ENABLE_CUDA=ON
cmake --build build --parallel

# Run tiled benchmark
./build/tiled_benchmark

# Custom configuration
./build/tiled_benchmark \
  --ns-min=50 \
  --ns-max=500 \
  --ns-step=50 \
  --mpol=12 \
  --ntor=12 \
  --iterations=10
```

#### Sample Output

```
========================================
  VMEC++ Tiled Execution Benchmark
========================================

Configuration:
  mpol=12, ntor=12
  Grid: 36 x 36
  Iterations: 10

      ns tile_size   tiles non-tiled(us)  tiled(us) overhead(%)   mem/surf need_tile
--------------------------------------------------------------------------------------
      50        50       1         1234.5     1234.5       +0.00     2.5MB        NO
      50        10       5         1234.5     1248.2       +1.11     2.5MB        NO
      50        25       2         1234.5     1240.1       +0.45     2.5MB        NO

     100       100       1         2468.9     2468.9       +0.00     2.5MB        NO
     100        10      10         2468.9     2520.3       +2.08     2.5MB        NO
     ...
```

## Validation

### Memory Budget Validation

To verify memory calculations for a given problem size:

```cpp
#include "vmecpp/common/compute_backend/cuda/tile_memory_budget.h"

// Query memory requirements
GridSizeParams params{100, 12, 12, 64, 64};  // ns, mpol, ntor, nzeta, ntheta
MemoryBudget budget = TileMemoryBudget::Calculate(0, params);

std::cout << TileMemoryBudget::GetMemoryReport(budget, params);
```

### Tile Coverage Validation

To verify tiles cover the domain correctly:

```cpp
#include "vmecpp/common/compute_backend/cuda/tile_scheduler.h"

TileSchedulerConfig config;
config.ns = 100;
config.tile_size = 25;
config.ns_min = 1;
config.op_type = TileOperationType::kForwardStencil;

TileScheduler scheduler(config);

// Validate coverage
assert(scheduler.ValidateCoverage());

// Print tile report
std::cout << scheduler.GetTileReport();
```

### Numerical Validation

The TiledComputeBackendCuda supports runtime validation:

```cpp
TiledBackendConfig config;
config.enable_validation = true;
config.validation_tol = 1e-10;

auto backend = std::make_unique<TiledComputeBackendCuda>(config);

// Operations will compare tiled vs non-tiled results
backend->FourierToReal(...);

// Check validation result
if (!backend->GetLastValidationResult().passed) {
  std::cerr << "Validation failed: "
            << backend->GetLastValidationResult().details << "\n";
}
```

## Troubleshooting

### Common Issues

1. **HDF5 not found**
   ```bash
   sudo apt-get install libhdf5-dev
   # Or on CentOS/RHEL:
   sudo yum install hdf5-devel
   ```

2. **CUDA not detected**
   ```bash
   # Check CUDA installation
   nvcc --version

   # Ensure CUDA is in PATH
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. **Out of GPU memory**
   ```bash
   # Check available memory
   nvidia-smi

   # Reduce problem size or enable tiling
   export VMECPP_FORCE_TILING=1
   export VMECPP_TILE_SIZE=25
   ```

### Debug Builds

```bash
cmake -B build_debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DVMECPP_ENABLE_CUDA=ON

cmake --build build_debug --parallel
```

### Sanitizers

```bash
# Address sanitizer
cmake -B build_asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"

# Thread sanitizer (for OpenMP issues)
cmake -B build_tsan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread"
```

## File Reference

| File | Description |
|------|-------------|
| `compute_backend.h` | Abstract interface for all backends |
| `compute_backend_cpu.h/cc` | CPU implementation |
| `cuda/compute_backend_cuda.h/cu` | CUDA implementation |
| `cuda/tiled_compute_backend_cuda.h/cu` | Tiled CUDA implementation |
| `cuda/tile_scheduler.h/cc` | Tile partitioning logic |
| `cuda/tile_memory_budget.h/cu` | Memory budget calculations |
| `cuda/cuda_memory.h/cu` | CUDA memory utilities |
| `cuda/standalone_validation.cc` | Standalone tests |
| `cuda/tile_validation_test.cc` | GTest-based tests |
| `benchmark/backend_benchmark.cc` | CPU vs CUDA benchmark |
| `benchmark/tiled_benchmark.cc` | Tiled execution benchmark |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VMECPP_BACKEND` | Backend selection: `cpu`, `cuda` | auto |
| `VMECPP_CUDA_DEVICE` | CUDA device ID | 0 |
| `VMECPP_FORCE_TILING` | Force tiled execution | 0 |
| `VMECPP_TILE_SIZE` | Override tile size | auto |
| `VMECPP_MEMORY_FRACTION` | GPU memory fraction to use | 0.8 |

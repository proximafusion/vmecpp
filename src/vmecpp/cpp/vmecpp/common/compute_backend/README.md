# Compute Backend for VMEC++

This directory contains the compute backend abstraction layer that enables GPU acceleration for VMEC++ using CUDA.

## Architecture

The compute backend provides an abstract interface (`ComputeBackend`) for the computationally intensive operations:

**DFT Operations (Primary):**
- `FourierToReal`: Inverse DFT (Fourier coefficients -> real-space geometry)
- `ForcesToFourier`: Forward DFT (real-space forces -> Fourier coefficients)

**Physics Operations (Secondary):**
- `ComputeJacobian`: Jacobian sqrt(g)/R and half-grid geometry
- `ComputeMetricElements`: Metric tensor (guu, guv, gvv) and Jacobian gsqrt
- `ComputeBContra`: Contravariant magnetic field (B^theta, B^zeta)
- `ComputeMHDForces`: MHD force calculation on real-space grid

### Available Backends

1. **CPU Backend** (`ComputeBackendCpu`): Wraps the existing CPU implementations. Always available.

2. **CUDA Backend** (`ComputeBackendCuda`): GPU-accelerated implementation using NVIDIA CUDA. Requires building with `-DVMECPP_ENABLE_CUDA=ON`.

## Coverage Analysis

### Currently Accelerated Operations

The CUDA backend accelerates the **two most computationally intensive operations** in VMEC++:

| Operation | Description | Complexity | Status |
|-----------|-------------|------------|--------|
| `FourierToReal` | Inverse DFT: Fourier coefficients to real-space geometry | O(ns * mpol * ntor * nzeta * ntheta) | GPU Accelerated |
| `ForcesToFourier` | Forward DFT: Real-space forces to Fourier coefficients | O(ns * mpol * ntor * nzeta * ntheta) | GPU Accelerated |

These DFT operations typically account for **30-50%** of total VMEC iteration time on CPU, making them the highest-impact targets for GPU acceleration.

### Operations with Interface Support (CPU Fallback)

The following operations have backend interface support and CPU implementations. The CUDA backend currently uses CPU fallback for these operations, with GPU kernels ready for future activation:

| Operation | Description | Complexity | Status |
|-----------|-------------|------------|--------|
| `ComputeMHDForces` | MHD force calculation on real-space grid | O(ns * nzeta * ntheta) | Interface Ready |
| `ComputeJacobian` | Jacobian sqrt(g)/R and half-grid geometry | O(ns * nzeta * ntheta) | Interface Ready |
| `ComputeMetricElements` | Metric tensor (guu, guv, gvv) and gsqrt | O(ns * nzeta * ntheta) | Interface Ready |
| `ComputeBContra` | Contravariant magnetic field (B^theta, B^zeta) | O(ns * nzeta * ntheta) | Interface Ready |

These operations have CUDA kernels written but currently execute on CPU to ensure correctness. Future versions will enable GPU execution after validation.

### Operations Not Yet in Backend

The following operations remain outside the backend interface:

| Operation | Description | Potential Benefit |
|-----------|-------------|-------------------|
| `pressureAndEnergies()` | Pressure and energy integrals | Low |
| `deAliasConstraintForce()` | Constraint force filtering | Low |

These operations have lower computational density and minimal impact on overall performance.

### Estimated Speedup

For typical stellarator configurations (ns=50, mpol=12, ntor=12):
- **DFT-only speedup**: 5-20x on modern GPUs (RTX 3080, A100)
- **Overall iteration speedup**: 1.5-3x (depends on problem size and CPU performance)

Speedup increases with problem size. For high-resolution runs (ns>100, mpol>20), GPU acceleration provides greater benefit.

## Building with CUDA Support

### CMake

```bash
cmake -B build -DVMECPP_ENABLE_CUDA=ON
cmake --build build --parallel
```

### Requirements

- CUDA Toolkit 11.0 or later
- NVIDIA GPU with compute capability 6.0 or later (Pascal architecture)
- CMake 3.18 or later (for improved CUDA support)

## Benchmarking

A benchmark tool is provided to compare CPU vs CUDA backend performance:

```bash
# Build with CUDA support
cmake -B build -DVMECPP_ENABLE_CUDA=ON
cmake --build build --parallel

# Run benchmark with default settings
./build/backend_benchmark

# Run with custom configuration
./build/backend_benchmark --ns=100 --mpol=20 --ntor=20 --iterations=50
```

### Benchmark Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ns=<int>` | Number of radial surfaces | 50 |
| `--mpol=<int>` | Number of poloidal modes | 12 |
| `--ntor=<int>` | Number of toroidal modes | 12 |
| `--nzeta=<int>` | Toroidal grid points | 36 |
| `--ntheta=<int>` | Poloidal grid points | 36 |
| `--iterations=<int>` | Benchmark iterations | 100 |
| `--warmup=<int>` | Warmup iterations | 10 |

### Sample Output

```
========================================
  VMEC++ Compute Backend Benchmark
========================================

Configuration:
  Radial surfaces (ns):     50
  Poloidal modes (mpol):    12
  Toroidal modes (ntor):    12
  ...

Results (times in microseconds):

Backend             FourierToReal  ForcesToFourier          Total      Status
-----------------------------------------------------------------------------
CPU                        1250.3           1380.5         2630.8          OK
CUDA (RTX 3080)             125.2            142.1          267.3          OK

Speedup vs CPU:
  CUDA (RTX 3080): 9.84x
```

## Usage

### Basic Usage

```cpp
#include "vmecpp/common/compute_backend/compute_backend_factory.h"

// Create backend from environment (respects VMECPP_BACKEND env var)
auto backend = ComputeBackendFactory::CreateFromEnvironment();

// Or create with explicit configuration
BackendConfig config;
config.type = BackendType::kCuda;
config.cuda_device_id = 0;
auto cuda_backend = ComputeBackendFactory::Create(config);

// Use the backend
backend->FourierToReal(physical_x, xmpq, rp, s, profiles, fb, m_geometry);
backend->ForcesToFourier(forces, xmpq, rp, fc, s, fb, vacuum_state, m_forces);
```

### Environment Variables

- `VMECPP_BACKEND`: Set to `cpu` or `cuda` to select backend
- `VMECPP_CUDA_DEVICE`: Set to GPU device ID (default: 0)

## Integration with IdealMhdModel

To integrate the compute backend with the existing `IdealMhdModel` class, modify the DFT dispatch methods:

```cpp
// In ideal_mhd_model.cc

void IdealMhdModel::dft_FourierToReal_3d_symm(const FourierGeometry& physical_x) {
  RealSpaceGeometry geom{r1_e, r1_o, ru_e, ru_o, rv_e, rv_o,
                         z1_e, z1_o, zu_e, zu_o, zv_e, zv_o,
                         lu_e, lu_o, lv_e, lv_o, rCon, zCon};

  if (compute_backend_) {
    compute_backend_->FourierToReal(physical_x, xmpq, r_, s_, m_p_, t_, geom);
  } else {
    FourierToReal3DSymmFastPoloidal(physical_x, xmpq, r_, s_, m_p_, t_, geom);
  }
}
```

## Performance Considerations

The CUDA backend is most effective when:

1. **Grid sizes are large**: The DFT operations scale with O(ns * mpol * ntor * nZeta * nTheta). For small problems, the CPU backend may be faster due to GPU memory transfer overhead.

2. **Multiple iterations are performed**: Device memory is reused across iterations, amortizing the allocation cost.

3. **Using a modern GPU**: Pascal (GTX 1000 series) or newer GPUs provide better double-precision performance.

## File Structure

```
compute_backend/
  compute_backend.h           # Abstract interface
  compute_backend_cpu.h       # CPU backend header
  compute_backend_cpu.cc      # CPU backend implementation
  compute_backend_factory.h   # Factory for backend creation
  compute_backend_factory.cc  # Factory implementation
  BUILD.bazel                 # Bazel build (CPU only)
  CMakeLists.txt              # CMake build (CPU + CUDA)
  README.md                   # This file
  benchmark/
    backend_benchmark.cc      # Benchmark executable
    CMakeLists.txt            # Benchmark build
  cuda/
    compute_backend_cuda.h    # CUDA backend header
    compute_backend_cuda.cu   # CUDA backend implementation
```

## Bazel Support

Currently, the Bazel build only supports the CPU backend. For CUDA support, use CMake.

## Extending to Other Backends

To add a new backend (e.g., OpenCL, SYCL, ROCm):

1. Create a new class inheriting from `ComputeBackend`
2. Implement the virtual methods
3. Add the backend type to `BackendType` enum
4. Update `ComputeBackendFactory` to create instances
5. Add appropriate build configuration

## Future Work

Potential enhancements:

1. **Enable GPU kernels for physics operations**: Activate CUDA kernels for ComputeJacobian, ComputeMetricElements, ComputeBContra, and ComputeMHDForces after validation
2. **Multi-GPU support**: Distribute radial surfaces across multiple GPUs
3. **Mixed-precision computation**: Use FP32 for intermediate calculations
4. **Persistent GPU memory**: Keep data on GPU across multiple VMEC iterations
5. **Bazel CUDA support**: Add rules_cuda integration
6. **Integrate backend into IdealMhdModel**: Wire up the physics operations to use the compute backend

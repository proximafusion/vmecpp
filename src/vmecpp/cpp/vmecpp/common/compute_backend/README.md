# Compute Backend for VMEC++

This directory contains the compute backend abstraction layer that enables GPU acceleration for VMEC++ using CUDA.

## Architecture

The compute backend provides an abstract interface (`ComputeBackend`) for the computationally intensive DFT operations:

- `FourierToReal`: Inverse DFT (Fourier coefficients -> real-space geometry)
- `ForcesToFourier`: Forward DFT (real-space forces -> Fourier coefficients)

### Available Backends

1. **CPU Backend** (`ComputeBackendCpu`): Wraps the existing CPU implementations. Always available.

2. **CUDA Backend** (`ComputeBackendCuda`): GPU-accelerated implementation using NVIDIA CUDA. Requires building with `-DVMECPP_ENABLE_CUDA=ON`.

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
  CMakeLists.txt             # CMake build (CPU + CUDA)
  README.md                  # This file
  cuda/
    compute_backend_cuda.h   # CUDA backend header
    compute_backend_cuda.cu  # CUDA backend implementation
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

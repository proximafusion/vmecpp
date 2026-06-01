# AGENTS.md

## Development Commands

### Building and Installation
```bash
# Install as editable Python package (rebuilds C++ automatically on changes)
pip install -e .

# Build C++ core with CMake manually
cmake -B build
cmake --build build --parallel
```

### Testing
```bash
# Run Python tests
pytest

# Run specific test file
pytest tests/test_simsopt_compat.py
```

### Code Quality
```bash
ruff check
ruff format
pyright
pre-commit run --all-files
```

### C++ Development (Bazel)
```bash
# Build C++ core with Bazel (from src/vmecpp/cpp/)
bazel build //...

# Run C++ tests
bazel test //vmecpp/...

# Build specific target
bazel build //vmecpp/vmec/vmec:vmec
```

### Running VMEC++
```bash
# Command line usage
python -m vmecpp examples/data/input.w7x
python -m vmecpp examples/data/w7x.json

# Run C++ standalone executable
./build/vmec_standalone examples/data/solovev.json
```

## High-Level Architecture

VMEC++ is a modern C++ reimplementation of the VMEC magnetohydrodynamic equilibrium solver with a Python interface.

### Core Components

**C++ Core Computations** (`src/vmecpp/cpp/vmecpp/`):
- **VMEC Solver** (`vmec/vmec/`): Main iterative equilibrium solver using multigrid methods
- **Ideal MHD Model** (`vmec/ideal_mhd_model/`): Physics equations and force calculations
- **Fourier Transforms** (`common/fourier_basis_fast_*`): Fast transforms for spectral decomposition
- **Free Boundary Solver** (`free_boundary/`): NESTOR/BIEST methods for plasma-vacuum interface
- **Geometry Engine** (`vmec/fourier_geometry/`): Flux surface geometry and coordinate transformations

**Python Interface Layer** (`src/vmecpp/`):
- **VmecInput**: Pydantic model for input validation (profiles, boundary, parameters)
- **VmecOutput/VmecWOut**: Output data structures with equilibrium results
- **run()**: Primary entry point for computations
- **Free Boundary Support**: External magnetic field handling

**Python-C++ Bridge** (`src/vmecpp/cpp/vmecpp/vmec/pybind11/`):
- Automatic NumPy ↔ Eigen conversion
- Exception translation from C++ to Python
- In-Memory data sharing

**SIMSOPT Compatibility** (`src/vmecpp/simsopt_compat.py`):
- Drop-in replacement for SIMSOPT's Vmec class
- Optimization workflow integration
- Hot restart support for parameter scans

### Data Flow

1. **Input**: JSON (VMEC++) or INDATA (Fortran) formats → VmecInput validation → C++ VmecINDATA
2. **Computation**: Multigrid setup → Fourier decomposition → Force balance iteration → Convergence
3. **Output**: C++ results → Python data structures → Multiple formats (HDF5, NetCDF, JSON)

### Key Features

- **Zero-crash policy**: All errors reported as Python exceptions
- **Hot restart**: Initialize from previous converged state for efficient parameter scans
- **OpenMP parallelization**: Multi-threaded force calculations
- **Dual input formats**: Classic INDATA and modern JSON
- **SIMSOPT integration**: Seamless optimization workflow support

## Coding Standards and Guidelines

### C++ Code (Google Style with Physics Domain Adaptations)

**Naming Conventions**:
- **Namespaces**: `snake_case` (e.g., `vmec_algorithm_constants`)
- **Classes**: `CamelCase` (e.g., `IdealMhdModel`)
- **Functions**: `CamelCase` (e.g., `ComputeGeometry()`)
- **Constants**: `kCamelCase` (e.g., `kSignOfJacobian`)
- **Member variables**: `snake_case_` with trailing underscore
- **Physics variables**: Preserve traditional names (e.g., `bsupu_`, `iotaf_`, `presf_`)

**Function Parameters**:
- Use `m_` prefix for parameters that **will be modified** by the function
- Example: `void UpdateForces(const RadialProfiles& profiles, FourierGeometry& m_geometry)`

**Modern C++ Practices**:
- Use `std::array<>` instead of C-style arrays
- Include `<array>` header when using `std::array<>`
- Follow clang-format Google style

**Pre-commit Validation**:
- All C++ code must pass `clang-format` (Google style)
- Must pass `readability-identifier-naming` checks
- Must pass `modernize-avoid-c-arrays` checks
- Files must end with newline (`end-of-file-fixer`)

**Incremental development**: Make small, focused changes that can be validated independently

## Physics Domain Knowledge

**Essential Reading for Fourier Operations**: `docs/fourier_basis_implementation.md` provides comprehensive coverage of DFT implementation, basis conversions, and mathematical foundations. Required reading before working with any Fourier-related code.

### Fourier Basis Architecture

VMEC++ uses **two different Fourier representations**:

**Internal Product Basis** (Computational efficiency):
- `rmncc_`: R \times cos(m\theta) \times cos(n\zeta) coefficients
- `rmnss_`: R \times sin(m\theta) \times sin(n\zeta) coefficients
- Enables separable DFT operations using pre-computed basis arrays

**External Combined Basis** (Researcher interface):
- `rmnc`: R \times cos(m\theta - n\zeta) coefficients (traditional VMEC format)
- `zmns`: Z \times sin(m\theta - n\zeta) coefficients
- Compatible with research literature and SIMSOPT

**Key Physics Variables** (preserve traditional names):
- `bsupu_`: B^\theta contravariant magnetic field component
- `bsupv_`: B^\zeta contravariant magnetic field component
- `iotaf_`: Rotational transform on full grid
- `presf_`: Pressure on full grid

**Localized domain docs**: deeper notes on the solver control flow live next to the code:
- `src/vmecpp/cpp/vmecpp/vmec/vmec/AGENTS.md` -- multigrid iteration, descent algorithm,
  restart/hot-restart logic, convergence.
- `src/vmecpp/cpp/vmecpp/vmec/ideal_mhd_model/AGENTS.md` -- staggered full/half radial grid,
  the forward/inverse DFT/FFT and de-alias hot kernels, post-processing boundary.

## Development Notes

- Uses **scikit-build-core** for Python packaging with CMake backend
- **Editable installs** (`pip install -e .`) automatically rebuild C++ on changes
- **Multi-threading**: OpenMP parallelization, not MPI (unlike Fortran VMEC)
- **Dependencies**: Eigen (linear algebra), abseil (utilities), pybind11 (Python binding)
- **Input validation**: Pydantic ensures type safety and automatic validation
- **Hot restart**: Pass previous VmecOutput as `restart_from` parameter to run()

## Common Workflows

**Fixed Boundary Run**:
```python
import vmecpp

input = vmecpp.VmecInput.from_file("input.w7x")
output = vmecpp.run(input)
output.wout.save("wout_result.nc")
```

**Free Boundary Run**:
```python
# Requires mgrid file for external magnetic field
input = vmecpp.VmecInput.from_file("free_boundary_config.json")
output = vmecpp.run(input)  # Automatically detects free boundary mode
```

**Hot Restart**:
```python
base_output = vmecpp.run(base_input)
# Modify input parameters
perturbed_input.rbc[0, 0] *= 1.1
# Must use single multigrid step for hot restart
perturbed_input.ns_array = perturbed_input.ns_array[-1:]
hot_output = vmecpp.run(perturbed_input, restart_from=base_output)
```

**SIMSOPT Optimization**:
```python
import vmecpp.simsopt_compat

vmec = vmecpp.simsopt_compat.Vmec("input.w7x")
# Use in SIMSOPT optimization workflows
```

## Naming and Conventions

**For Code Changes**:
1. **MANDATORY: Always check compliance with VMECPP_NAMING_GUIDE.md before proposing ANY changes**
   - Verify naming conventions: classes, functions, variables, constants
   - Ensure physics variable names are preserved (e.g., `bsupu_`, `iotaf_`, `presf_`)
   - Check function parameter conventions (`m_` prefix for mutable parameters)
   - Validate against Google C++ Style Guide adaptations
2. **MANDATORY: Only use ASCII characters in ALL changes**
   - Never use Unicode, special symbols, or non-ASCII characters in code,comments or documentation
   - Use LaTeX notation for mathematics (e.g., `\nabla p`, `\sum_{m,n}`, `\lambda`)
3. Always validate changes with pre-commit hooks before suggesting commits
4. Use incremental development approach (small, testable changes)
5. Respect the physics domain knowledge embedded in variable names
6. Follow the naming guide strictly for new code

**For Testing**:
1. Always build and test C++ changes: `bazel build //... && bazel test //vmecpp/...`
2. Run pre-commit checks: `pre-commit run --files <modified_files>`
3. Test Python integration when modifying C++ interfaces

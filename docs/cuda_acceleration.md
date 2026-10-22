# CUDA Acceleration

## Overview

VMEC++ can execute the iteration body on an NVIDIA GPU. The CUDA path is a
device-resident port of `IdealMhdModel::update` and the surrounding
time-stepping loop: the spectral state, geometry, forces, and preconditioners
live in persistent device buffers across iterations, and the per-iteration
kernel chain (forward toroidal transform, Jacobian and metric elements, MHD
forces, inverse transform, preconditioning, residuals, time step) runs on the
GPU with a small number of scalar transfers per iteration. A batched mode
solves N equilibria concurrently inside one CUDA context, with
per-configuration residual evolution, convergence gates, and restart
handling; its entry point is exposed to Python as `run_batched_gpu`.

## Building

The CUDA path is a CMake option, default off:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVMECPP_USE_CUDA=ON
cmake --build build
```

Requirements: the CUDA toolkit (nvcc and cuFFT) and a GPU of compute
capability 7.0 or newer. `CMAKE_CUDA_ARCHITECTURES` defaults to `70;80;89`.
The CUDA units compile as C++20; nvcc 12.0 rejects the `std::ranges`
specialization in abseil's `span.h`, and nvcc 12.6 compiles it.
With the option off, the build contains no CUDA reference and needs no CUDA
toolkit. With it on, the CUDA implementation replaces the CPU iteration body
at compile time, and `vmecpp.run` and `vmec_standalone` use it for every
supported input. `VMECPP_HWCAPS_DISPATCH` defaults off under the CUDA build.

### Windows (MSVC)

The option builds natively with the MSVC host compiler. HDF5, netCDF-C and
zlib come from vcpkg (`vcpkg install hdf5[cpp] netcdf-c zlib`); the
from-source dependency build the Linux and macOS configurations use is
skipped on MSVC, and the dense solves use Eigen. The build uses the Ninja
generator, which honors the CUDA compiler launcher that adapts the MSVC
command line:

```bat
cmake -B build -G Ninja -DVMECPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

FFTX and the `indata2json` namelist converter default off on MSVC; the CUDA
path uses neither. At runtime the vcpkg and CUDA `bin` directories must be on
the module loader's search path; from Python, add them with
`os.add_dll_directory` before importing `vmecpp`.

## Scope

The CUDA build supports fixed-boundary and free-boundary,
stellarator-symmetric (`lasym = false`) configurations, axisymmetric
(`ntor = 0`, one toroidal plane) and three-dimensional (`ntor >= 1`), on a
single radial rank, in the single and the batched execution modes. Batched
free-boundary runs use the NESTOR vacuum solver, one instance per
configuration slot sharing one loaded mgrid: every input in the batch carries
the same `mgrid_file` and `extcur`, the boundaries and profiles may differ in
distinct mode, and the vacuum activation iteration, the `nvacskip` cadence,
and the soft-start restart are batch-wide. The per-iteration vacuum solves
run on the host, serialized across the configurations. `lasym` and `lforbal`
inputs are rejected with `absl::UnimplementedError`. The radial domain is not
partitioned across OpenMP threads; `num_threads` is forced to 1. The Python
iteration API (`VmecModel`, `vmecpp.iterate`, `solve_equilibrium`,
`solve_multigrid`, `vmecpp.autodiff`) evaluates the forward model state by
state through the host arrays and is not available in the CUDA build:
`VmecModel.create` raises. `vmecpp.has_cuda()` reports the build.

## Correctness contract

The CUDA path runs the algorithm of the CPU path: the m = 1 Z force is zeroed
under the same gate (`always_fix_m1_gauge`, `fsqz < 1e-6` on the previous
iteration, or `iter2 < 2`), a restart rewinds to the same backup (the state of
the latest force evaluation under `backup_evaluated_state`), and the
convergence test and the multigrid transitions are the CPU's. It therefore
takes the CPU build's iterations, restart for restart, on every input of
`examples/cuda_cross_validation.py`: axisymmetric and three-dimensional
boundaries, prescribed current and prescribed iota, power-series and spline
pressure, vacuum, `lbsubs`, nfp from 1 to 5, an odd nzeta, a missing axis
guess, a bad initial Jacobian, W7-X with its mid-run restarts, 1101 radial
surfaces, the largest poloidal grid the device kernels accept, and free
boundary at two vacuum cadences and through a multigrid transition.

The toroidal transforms run in cuFFT, whose summation order differs from the
CPU's: the first force residuals agree to 4e-13 relative or better in three
dimensions, 1e-11 on cma, and to 2e-15 in two. The difference grows slowly
enough that every `wout` field, the spectral width included, agrees with the
CPU build within ten times the spread between CPU runs on one and on eight
threads, which partition the radial reductions differently, or to 1e-11
relative where that spread is smaller, except the force-balance diagnostic
`equif`, which agrees to 3e-9 relative; the interior quantities that
`tests/test_cuda_gpu.py` checks agree to 5e-10 relative or better. The
residual triples that feed the time-step controller and the convergence gate
come from device reductions of fixed order, so a run repeats to the last bit.

## Validation

`examples/cuda_cross_validation.py` compares the CUDA build with the CPU build
of the same commit: every `wout` field against the spread between CPU runs on
one and on eight threads, hot restarts in both directions, and the residual
traces of the first iterations. Its docstring lists the commands.

`pytest tests/` runs the whole suite on a CUDA build: every test of an input
the build accepts solves on the GPU and is held to the CPU's references, and
a test that needs a `lasym` or `lforbal` input or the `VmecModel` iteration
API is reported as skipped with the build's refusal (`tests/conftest.py`).
`tests/test_cuda_gpu.py` adds the CUDA-specific cases, which run in every
CUDA build; the W7-X case carries the `slow` marker and runs under
`pytest -m slow`. The `Build the CUDA path` workflow compiles the
CUDA iteration body and links the Python module on every pull request; its
runners have no GPU.

## Batched execution

`run_batched_gpu(indata_list, ...)` solves N equilibria in one CUDA-resident
iteration loop, and `vmecpp.run_batch` wraps it. All inputs share `mpol`,
`ntor`, `nfp`, `lasym`, and the multigrid schedule; in distinct mode,
`always_fix_m1_gauge` inputs take a single-entry `ns_array`. N is at most
65535 / ns at the largest ns of the schedule. The `distinct` argument selects
the mode. Broadcast (`distinct=False`, the default of `run_batched_gpu`): the
first input's spectra fill all N slots and the call returns the single
converged result. Distinct (`distinct=True`, the default of `run_batch`):
each configuration keeps its own boundary, magnetic axis, and profiles, with
per-configuration pre-initialization, first iteration, residual evolution,
restart control, convergence gating, and outputs, and the call returns one
`OutputQuantities` per configuration derived from the batched run's flushed
device state, with that configuration's residual traces and iteration count;
a derivation that fails raises. As in a single run, a configuration that
misses the tolerance of the last multigrid stage raises unless
`return_outputs_even_if_not_converged` is set. The converged
per-configuration spectra are also returned with `return_spectra=True`.

A run occupies the first visible CUDA device. `CUDA_VISIBLE_DEVICES` selects
it, and batches on several GPUs run one process per device.

## Free-boundary execution

The NESTOR vacuum solve stays on the host. Once the vacuum block is live,
bridges carry the per-iteration traffic: the decomposed position state
flushes to the host every iteration; the axis and LCFS geometry rows, the
outermost totalPressure rows, and the bucoH/bvcoH profiles flush ahead of
their host reads; and the host-computed `rBSq` profile stages back to the
device, where a kernel applies the vacuum edge force to the LCFS row ahead of
the constraint assembly. The rCon0/zCon0 turn-off decay runs as a device
kernel. The segment CUDA graphs run on free-boundary runs as well: the
segment-3 graph, which holds the vacuum edge-force kernel, and the segment-4
graph, whose `jMax` follows the vacuum pressure state, are re-captured when
that state changes.

## Restart and state backup

The CPU controller's restart protocol (`RestartReason` in `FlowControl`)
rewinds the position state to a backup copy and reduces the time step when
the Jacobian changes sign or the residual stops improving. Under the CUDA
build the protocol operates on device-resident state. The device backup of
the six spectral position components is armed at the start of every
multigrid stage and refreshed at the host backup cadence by one fused copy
kernel (`k_backup_pts_x`). Under `backup_evaluated_state`, the default, the
backup takes the state of the latest force evaluation, which
`SnapshotEvaluatedPtsXCuda` saves before each time step advances the position
buffers, as the CPU path keeps `last_evaluated_x`; the free-boundary soft
start at vacuum activation rewinds to it as well. The progress printout's
`<M>` column and the spectral width in the output file are taken from this
backup, as on the CPU. A restore replays the backup into the position
buffers and zeroes the integrator velocity. A bad Jacobian during the first
iterations triggers the magnetic-axis recomputation and a retry from scratch,
which invalidates the device position, velocity, and backup state so the
retry re-stages from the recomputed host axis.

Each configuration of a distinct batch takes the restart control of its own
single run: its own time step, damping, restart markers, residual minimum
and backup, so that a bad Jacobian, a growing residual or a stall rewinds
that configuration alone, and its iteration count advances only in the
iterations it does not restart. Before the batch starts, each configuration
runs its first iteration alone, which recomputes its magnetic axis and starts
over when the first force evaluation shows a bad Jacobian or huge forces, as
in its single run. A configuration whose residuals stop being finite, whose
Jacobian changes sign in the first iteration of a stage, or whose Jacobian
changed sign 75 times ends as not converged and sits out the later stages; a
distinct batch never retries from three surfaces. The batch counts the
cadences of the radial preconditioner and of the vacuum solve from the
latest restart of any configuration, takes its largest residual to switch
on the vacuum pressure and the edge forces of a nearly converged residual,
and skips the force evaluation of every configuration in an iteration in
which configuration zero's Jacobian changes sign; it normalizes each
configuration's residuals with the current energy density, where a single run
keeps the one of its latest preconditioner update. A broadcast batch shares
one restart control.

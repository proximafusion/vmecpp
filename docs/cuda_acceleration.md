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

Requirements: the CUDA toolkit (nvcc, cuFFT, cuBLAS) and a GPU of compute
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

### AMD GPUs (HIP/ROCm)

The same GPU iteration body builds for AMD GPUs through HIP, selected by a
separate option (Linux only, mutually exclusive with `VMECPP_USE_CUDA`):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVMECPP_USE_HIP=ON
cmake --build build
```

Requirements: ROCm with `hipcc` and hipFFT. `CMAKE_HIP_ARCHITECTURES`
defaults to `gfx90a;gfx942` (MI210/MI250 and MI300), and
`VMECPP_HWCAPS_DISPATCH` defaults off as under the CUDA build.

The `.cu` translation units are compiled as HIP; `hip_compat.cuh` maps the
CUDA runtime, cuFFT, and warp-intrinsic surface they use onto HIP, 1:1 for
the runtime, stream, event, graph, and FFT calls. The masked `__shfl_*_sync`
intrinsics map to the width-32 legacy forms, which partition the 64-lane
wavefront into 32-lane groups and preserve the 32-lane warp semantics the
kernels assume; `__syncwarp` maps to a wave-level barrier. The second
`__launch_bounds__` parameter is min blocks per multiprocessor in CUDA and
min waves per execution unit in HIP, so only the first parameter is kept and
the occupancy hint is left to the AMD register allocator.

The five tensor-core and cuBLAS scatter experiments
(`VMECPP_SCATTER_CUSTOM_GEMM_WMMA`, `VMECPP_SCATTER_I8GEMM`,
`VMECPP_SCATTER_I8OZAKI`, `VMECPP_SCATTER_CUBLAS_FP32`,
`VMECPP_SCATTER_CUBLAS_OZAKI`) are NVIDIA-only and compiled out; setting one
falls back to the production scatter with a one-time notice. The remaining
runtime controls, the batched mode, and the graphs apply unchanged.

The HIP build defines `VMECPP_USE_CUDA`, which every host dispatch site tests
to mean the GPU iteration body is built, together with `VMECPP_USE_HIP`,
which selects the backend inside the GPU translation units. The host solver
code is identical between the two backends.

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

The `<M>` column of the progress printout evaluates the current iteration
state under the CUDA build and the restart backup under the CPU build, so the
printed column can differ in the trailing digit at a given iteration. The
spectral width in the output file is computed by the output phase from the
converged state.

## Correctness contract

Against the CPU implementation, on the fixed-boundary test cases through the
full multigrid ramp: `aspect_ratio` and the volume agree to 2e-15 relative;
the volume-averaged and field-line-derived output quantities match within a
drift family of 1e-5 to 3e-3 relative (the widest is the axis rotational
transform on `cma`), from summation-order differences in the reductions
marked at their kernel definitions in `fft_toroidal_cuda_kernels.cu`; the
Mercier profile, a second radial derivative, amplifies the drift to 36% at
mid-radius on `cma`. Restart-reason sequences and multigrid stage transitions
match the CPU trajectory. Iteration counts match on the axisymmetric cases
and differ by a few percent on three-dimensional ones (`cma` converges in
214 iterations against the CPU's 207, `cth_like_fixed_bdy` in 118 against
121, W7-X in 3146 against 2954), to the same equilibrium within the drift
family. On free-boundary runs the boundary responds to the vacuum
field through the drift-sensitive trajectory, so every converged output
carries the drift family. The residual triples that feed the time-step
controller and the convergence gate use serial or order-preserving device
reductions. `tests/test_cuda_gpu.py` holds the regression battery, opt-in
via `VMECPP_TEST_CUDA=1`; the W7-X case additionally needs
`VMECPP_TEST_CUDA_SLOW=1`.

## Batched execution

`run_batched_gpu(indata_list, ...)` solves N equilibria in one CUDA-resident
iteration loop. All inputs share `mpol`, `ntor`, `nfp`, `lasym`, and the
multigrid schedule. `VMECPP_BATCH_DISTINCT` selects the mode. Broadcast, the
default: the first input's spectra fill all N slots and the call returns the
single converged result. Distinct (`=1`): each configuration keeps its own
boundary, with per-configuration pre-initialization, axis recomputation,
residual evolution, time-step control, convergence gating, and outputs, and
the call returns one `OutputQuantities` per configuration derived from the
batched run's flushed device state (`VMECPP_PER_CFG_RECOMPUTE=0` opts out).
Per-configuration converged spectra are also available via
`return_spectra=True` or `VMECPP_BATCH_OUTPUTS_FILE` plus
`recompute_outputs_from_spectra`. A configuration that exceeds
`VMECPP_PER_CFG_NITER_CAP` is marked timed out without failing the rest of
the batch. Every environment variable the binding sets is restored when the
call returns.

## Free-boundary execution

The NESTOR vacuum solve stays on the host. Once the vacuum block is live,
bridges carry the per-iteration traffic: the decomposed position state
flushes to the host every iteration; the axis and LCFS geometry rows, the
outermost totalPressure rows, and the bucoH/bvcoH profiles flush ahead of
their host reads; and the host-computed `rBSq` profile stages back to the
device, where a kernel applies the vacuum edge force to the LCFS row ahead of
the constraint assembly. The rCon0/zCon0 turn-off decay runs as a device
kernel. Single-configuration NESTOR runs can overlap the vacuum solve with
the device iteration on a worker thread (`VMECPP_FB_ASYNC_NESTOR=1`); the
solve and the rBSq assembly are then one iteration stale. The segment
and whole-iteration CUDA graphs are disabled on free-boundary runs. Under
sync elision (`VMECPP_SYNC_ELIDE=K`) iterations run live until the vacuum
state machine reaches its active state; from then on the per-iteration
scalar sync sites elide, with the device time-step controller authoritative
and the convergence gate on the K-boundaries, while the vacuum block keeps
its per-iteration cadence.

## Restart and state backup

The CPU controller's restart protocol (`RestartReason` in `FlowControl`)
rewinds the position state to a backup copy and reduces the time step when
the Jacobian changes sign or the residual stops improving. Under the CUDA
build the protocol operates on device-resident state. The device backup of
the six spectral position components is armed at the start of every
multigrid stage and refreshed at the host backup cadence by one fused copy
kernel (`k_backup_pts_x`). A restore replays the backup into the position
buffers and zeroes the integrator velocity; in batched mode the restore is
gated per configuration by the restart mask. A bad Jacobian during the first
iterations triggers the magnetic-axis recomputation and a retry from scratch,
which invalidates the device position, velocity, and backup state so the
retry re-stages from the recomputed host axis. Under sync elision the backup
refresh and the restart bookkeeping evaluate on the K-window boundaries, so a
restart rewinds at most K-1 iterations.

## Runtime controls

Every environment variable the CUDA path reads, grouped by role. Boolean
knobs parse as `atoi(value) > 0`; "default ON" means active when unset, with
`=0` disabling.

### Device and batched execution

| Variable | Default | Effect |
|---|---|---|
| `VMECPP_CUDA_DEVICE` | `0` | Device ordinal for this process. Multi-GPU batches run one process per device. |
| `VMECPP_N_CONFIG_MAX` | `1` | Number of configuration slots the persistent device buffers are dimensioned for. At 1 the layout is identical to the single-configuration path. Read once per process. |
| `VMECPP_BATCH_DISTINCT` | OFF | Distinct mode, one boundary per slot. |
| `VMECPP_BATCH_MULTIGRID_UPSCALE` | OFF | Per-configuration multigrid stage transition: each configuration's device state is snapshotted at the stage boundary instead of broadcasting configuration 0's host upscale. Distinct mode needs it across the `ns_array` ramp. |
| `VMECPP_BATCH_UPSCALE_KERNEL` | OFF | Host-exact per-configuration radial interpolation at each stage transition: scale to physical, odd-m axis extrapolation, linear interpolation in s, division by the new stage's scalxc. |
| `VMECPP_BATCH_PER_CFG_TIMESTEP` | ON | Per-configuration time-step controller: each configuration's `(fac, b1)` derive from its own residual. Active when the batch holds more than one slot. |
| `VMECPP_BATCH_AXIS_RECOMPUTE` | ON | Per-configuration magnetic-axis recomputation during distinct-mode pre-initialization. `=0` leaves axis recovery to the iteration body's bad-Jacobian path. |
| `VMECPP_BATCH_INPUTS_FILE` | unset | Binary file of per-configuration spectral inputs (`[component][cfg][spectra]`, int32 shape header), written by the distinct-mode pre-initialization and read once by the first forward transform. |
| `VMECPP_BATCH_DEC_X_FILE` | unset | Companion file carrying each configuration's pre-triplet decomposed position state for the device time integrator's first iteration. |
| `VMECPP_BATCH_OUTPUTS_FILE` | unset | Destination for the end-of-run dump of every configuration's converged decomposed spectra, same layout as the inputs file. |
| `VMECPP_KEEP_BATCH_FILES` | OFF | Keeps the three batch files after `run_batched_gpu` returns. |
| `VMECPP_PER_CFG_NITER_CAP` | unset | Per-configuration iteration ceiling. A configuration that reaches it is marked timed out; the batch succeeds if at least one configuration converged. |
| `VMECPP_ACTIVE_PER_CFG_OVERRIDE_BITS` | unset | Bit mask overriding the active-configuration mask at run start. |
| `VMECPP_PER_CFG_RECOMPUTE` | OFF | In-process per-configuration `OutputQuantities` after a distinct-mode batched run: each configuration's converged device state flushes once and the standard output-derivation chain runs per configuration. Each derived `wout` carries that configuration's own iteration count and final `fsqr`/`fsqz`/`fsql`, snapshotted from its last active iteration. |
| `VMECPP_RECOMPUTE_FTOL` | last `ftol_array` entry | Tolerance for the single-stage hot-restart recompute run. |
| `VMECPP_RECOMPUTE_NITER` | unset | Iteration cap for the recompute run. |
| `VMECPP_RECOMPUTE_LAMBDA` | ON | Seeds the recompute hot restart with the converged lambda spectra. `=0` re-converges lambda from zero. |
| `VMECPP_RECOMPUTE_LAMBDA_SCALE` | `1.0` | Uniform factor on the lambda seed. |

### Production-path switches (default ON)

`=0` selects the alternative in the second column. Every setting preserves
the bit-exact `aspect_ratio` contract.

| Variable | `=0` selects |
|---|---|
| `VMECPP_SCATTER_V5` | The L1-broadcast scatter (v4) instead of the shared-memory-cached fused scatter (v5). |
| `VMECPP_JAC_METRIC_FUSE` | Two launches for the Jacobian and the metric elements instead of the fused kernel. |
| `VMECPP_JAC_METRIC_DVDSH_FUSE` | Fused Jacobian and metric plus a tree-reduced dVdsH instead of the three-way atomicAdd fusion. |
| `VMECPP_JAC_PAIR` | Per-surface Jacobian blocks instead of the jH-pair-coarsened variant. Requires an even half-grid extent; falls back automatically otherwise. |
| `VMECPP_MHD_PAIR` | Per-surface MHD-force blocks instead of the jF-pair-coarsened variant. Requires an even force-grid extent; falls back automatically otherwise. |
| `VMECPP_RESIDUALS_PAR` | The serial residual reduction, bit-identical to the CPU per call, instead of the 256-thread parallel one. |
| `VMECPP_RESIDUALS_K` | Multi-block residual partitioning; auto picks `K = max(1, 16 / n_config)` capped at 16, an explicit value overrides, `=1` forces a single block. |
| `VMECPP_DEALIAS_PACK` | One zeta plane per warp in the dealias inverse instead of `32 / nThetaReduced` planes. Packing is selected only when 32 is a multiple of `nThetaReduced`. |
| `VMECPP_SCATTER_PACK` | One zeta plane per warp in the main-and-constraint scatter. Same selection rule. |
| `VMECPP_UPDATE_GRAPH` | Per-kernel dispatch instead of the segment-3 CUDA graph (effectiveConstraintForce through assembleTotalForces). |
| `VMECPP_SEG2_GRAPH` | Per-kernel dispatch instead of the segment-2 CUDA graph (computeMetricElements through radialForceBalance plus hybridLambdaForce). |
| `VMECPP_SEG4_GRAPH` | Per-kernel dispatch instead of the segment-4 CUDA graph (the four preconditioner-apply wrappers, re-captured when `jMax` changes). |
| `VMECPP_CONV_FLAG_AUTH` | The host residual comparison as the termination gate instead of the device-side `k_check_convergence` flag; the host comparison is also used when the flag buffers are absent. |

### Opt-in throughput modes (default OFF)

| Variable | Effect |
|---|---|
| `VMECPP_FB_ASYNC_NESTOR` | `=1`: on single-configuration NESTOR free-boundary runs, the vacuum solve runs on a worker thread while the device iterates; the device applies the previous iteration's edge force, and the worker runs a full vacuum update every iteration. |
| `VMECPP_SYNC_ELIDE` | `=K`: K-window sync elision. The per-iteration scalar D2H transfers and stream syncs (tau extrema, residual triples, plasma volume) are skipped on non-boundary iterations; the device time-step controller is authoritative, and the convergence gate, restart bookkeeping, and device-state backups evaluate every K-th iteration. `K=25` matches the preconditioner cadence. |
| `VMECPP_RESIDUALS_DEFER` | Deferred-sync residuals: the iteration consumes one-iteration-stale residual values; within 10x ftolv the value is force-synced so the gate never fires on a stale read. |
| `VMECPP_ITER_GRAPH` | Whole-iteration CUDA graph under sync elision: each captured elided iteration replays as one `cudaGraphLaunch`, including both cuFFT execs. Captured after two eligible iterations, invalidated on multigrid stage transitions and restarts. No effect without `VMECPP_SYNC_ELIDE`. |
| `VMECPP_FWD_GRAPH` | CUDA graph over the forward-FFT chain. |

### Precision and transform variants (default OFF)

The DD-pair primitives and Ozaki-slice multiplications are defined in
`fft_toroidal_cuda_common.cuh`, the Carson-Higham refinement at its kernels
in `fft_toroidal_cuda_kernels.cu`.

| Variable | Effect |
|---|---|
| `VMECPP_FFT_FP32` | FP32 cuFFT. The force residual floors above production ftol. |
| `VMECPP_FFT_RADIX` | Hand-coded radix-8x3 inverse DFT replacing cuFFT Z2D, for transform length 24; other lengths stay on cuFFT with a one-time notice. Accumulation order sits outside the bit-exact contract. |
| `VMECPP_FWD_FFT_RADIX` | Forward-direction (D2Z) counterpart, stream-capturable, same length coverage. |
| `VMECPP_DEALIAS_MIXED` | FP32 inner multiplies with an FP64 accumulator in the dealias inverse. Same floor as FP32 cuFFT. |
| `VMECPP_DEALIAS_SPLIT` | Four partial accumulators in the dealias inverse. |
| `VMECPP_SCATTER_DD_FP32` | FP32 multiplies with DD-pair accumulators in the scatter. |
| `VMECPP_SCATTER_DD_FP64MUL` | FP64 multiplies with DD-pair accumulators. |
| `VMECPP_SCATTER_DD_FP32_DDMUL` | Dekker TwoProduct DD x DD multiplies on FP32 operands, ~96-bit products. |
| `VMECPP_SCATTER_OZAKI_FP32` | 2-slice Ozaki FP32 multiplications, ~50-bit precision. |
| `VMECPP_SCATTER_OZAKI3_FP32` | 3-slice Ozaki, ~72-bit precision; converges within a few ULP of the FP64 equilibrium. |
| `VMECPP_SCATTER_CUBLAS_FP32` | Scatter as one cuBLAS GemmEx FP32 GEMM. FP32 floor. |
| `VMECPP_SCATTER_CUBLAS_OZAKI` | Four-GEMM Ozaki: FP32 hi/lo slices per operand, DD-pair reassembly, ~48-bit precision. |
| `VMECPP_SCATTER_CUSTOM_GEMM` | Tile-cooperative GEMM with per-multiply Veltkamp-Dekker and DD accumulation. |
| `VMECPP_SCATTER_CUSTOM_GEMM_WMMA` | TF32 tensor-core dispatch: 3-slice Ozaki limbs, 54 `wmma::mma_sync` per tile, with a scalar Veltkamp-Dekker pass over the same shared-memory data for production precision. Covers `mpol <= 12` and `nThetaReduced <= 16`; larger inputs fall back to the production scatter with a one-time notice. |
| `VMECPP_SCATTER_I8OZAKI` | int8 tensor-core scatter via the Ozaki construction: eight 7-bit limbs per FP64 operand, exact s32 accumulation. Converges within a few ULP of FP64. Same tile coverage as the wmma path. |
| `VMECPP_SCATTER_I8GEMM` | Batched int8-Ozaki GEMM with (config, surface, zeta) folded into one GEMM row axis; the basis-side limb matrix builds once per shape. No tile shape limits. Converges within a few ULP of FP64. |
| `VMECPP_SCATTER_TF32_PLAIN` | Plain TF32 accumulator sum, rel ~3e-6. |
| `VMECPP_SCATTER_I8_LIMBS` | Limb width for the int8 scatter paths: `4` selects 28-bit operands (rel ~4e-9), the default `8` covers the FP64 mantissa. Under `VMECPP_IR_STAGED` the width follows the residual phase, 4 above the threshold with a decade hysteresis band and 8 below; a width change drops the whole-iteration graph. |
| `VMECPP_RESIDUALS_DD_FP32` | DD-pair FP32 accumulator in the residual reduction. |
| `VMECPP_RZ_IR_FP32` | Carson-Higham iterative refinement on the RZ tridiagonal solve: FP32 PCR, FP64 residual, FP32 correction, FP64 combine. Halves PCR shared memory. |
| `VMECPP_IR_STAGED` | Staged-precision descent: hot kernels run FP32/TF32 while the residual is above the threshold, FP64 below it. |
| `VMECPP_IR_THRESHOLD` | Crossover residual for the staged descent. Default `1e-5`. |
| `VMECPP_IR_LOG_EVERY` | Logging cadence for staged-precision phase transitions. |

### Diagnostics (default OFF)

| Variable | Effect |
|---|---|
| `VMECPP_KERNEL_TIMING` | Per-kernel cudaEvent timing, dumped at exit and every 10k events. Disables the CUDA graphs and adds per-call syncs. |
| `VMECPP_KERNEL_TIMING_PATH` | Dump destination. Default `/tmp/vmecpp_kernel_timing.log`. |
| `VMECPP_PHASE_TIMING_PATH` | Destination for the phase-timer report. Default `/tmp/vmecpp_phase_timing.txt`. |
| `VMECPP_FFT_DUMP` | One-shot dump of the cuFFT input and output plus the radix-8x3 recomputation on the same input; needs `nZeta = 24`. |
| `VMECPP_CPU_ORDER_BCONTRA` | Serial ascending-kl accumulation of the jvPlasma and avg_guu_gsqrt reductions, matching the host loop bit for bit. |
| `VMECPP_CPU_ORDER_PRECOND` | Host-order serial accumulation of the radial-preconditioner matrix elements, including the host's division forms. |
| `VMECPP_CPU_ORDER_RZSOLVE` | Serial Thomas elimination in the host order instead of parallel cyclic reduction. |
| `VMECPP_RZ_FORCE_BLOCK` | The block-Thomas radial solver (the `ns > 1024` path) at any `ns`. |
| `VMECPP_FB_ASYNC_INLINE` | Runs the asynchronous NESTOR worker's solve inline from the submit call. |
| `VMECPP_DUMP_TCON` | One-shot full-precision print of the first constraint-multiplier profile. |
| `VMECPP_DUMP_GCON` | One-shot print of the effective-constraint-force checksum and the per-surface sums of the dealiased constraint force. |
| `VMECPP_DUMP_SPECS` | One-shot dump of the staged spectral input (configuration 0). |
| `VMECPP_STATE_DUMP_ITERS` | Comma-separated `iter2` values at which the full batched decomposed-x state is written to disk. |
| `VMECPP_STATE_DUMP_F` / `VMECPP_STATE_DUMP_PROF` | Add the decomposed forces / the per-configuration radial profiles to the state dumps. |
| `VMECPP_STATE_DUMP_PATH` | Filename prefix for the state dumps. |
| `VMECPP_PERCFG_RESIDUAL_DUMP` | `=K`: logs each configuration's residual triple every K iterations. |
| `VMECPP_TRACE_RESTART` | Logs every backup store and restore event with the controller inputs that drove it. |
| `VMECPP_TRACE_CFG_DIFF` | Per-call max-abs-difference probe between the configuration 0 and 1 slices of named device buffers. |
| `VMECPP_CONV_FLAG_DEBUG` | Logs any disagreement between the device convergence flag and the host gate. |
| `VMECPP_VALIDATE_DEVICE_TIMESTEP` | One-shot comparison of the device time-step controller's `(fac, b1)` against the host-computed values. |
| `VMECPP_DEFENSIVE_BROADCAST` | Re-broadcasts configuration 0's position state into all slots on every recompose. |
| `VMECPP_BATCH_PERTURB` | Scales each configuration's input spectra by `1 + scale * cfg / n_cfg`. |

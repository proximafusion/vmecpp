# CUDA Acceleration

## Overview

VMEC++ can execute the fixed-boundary iteration body on an NVIDIA GPU. The
CUDA path is a device-resident port of `IdealMhdModel::update` and the
surrounding time-stepping loop: the spectral state, geometry, forces, and
preconditioners live in persistent device buffers across iterations, and the
per-iteration kernel chain (forward toroidal transform, Jacobian and metric
elements, MHD forces, inverse transform, preconditioning, residuals, time
step) runs on the GPU with a small number of scalar transfers per iteration.

The port also provides a batched execution mode that solves N fixed-boundary
equilibria concurrently inside one CUDA context, with per-configuration
residual evolution, convergence gates, and restart handling. The batched
entry point is exposed to Python as `run_batched_gpu`.

## Building

The CUDA path is a CMake option, default off:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVMECPP_USE_CUDA=ON
cmake --build build
```

Requirements: the CUDA toolkit (nvcc, cuFFT, cuBLAS) and a GPU of compute
capability 7.0 or newer. `CMAKE_CUDA_ARCHITECTURES` defaults to `70;80;89`.
Without the option, the build contains no CUDA reference and needs no CUDA
toolkit; the CPU code path is unchanged.

With the option on, the CUDA implementation replaces the CPU iteration body
at compile time. `vmecpp.run` and `vmec_standalone` use it transparently for
supported inputs.

### Windows (MSVC)

The same option builds natively on Windows with the MSVC host compiler. The
system dependencies come from vcpkg (`vcpkg install hdf5 netcdf-c clapack`;
clapack is the LAPACK provider that needs no Fortran compiler), and the build
uses the Ninja generator, since the CUDA compiler launcher that adapts the
MSVC command line is honored by Ninja but not by the Visual Studio generator:

```bat
cmake -B build -G Ninja -DVMECPP_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build
```

FFTX and the `indata2json` namelist converter default off on MSVC (the former
is generated code MSVC rejects, the latter needs a Fortran toolchain), and
neither is used by the CUDA path. At runtime the vcpkg and CUDA `bin`
directories must be on the module loader's search path; from Python, add them
with `os.add_dll_directory` before importing `vmecpp`.

### AMD GPUs (HIP/ROCm)

The same GPU iteration body builds for AMD GPUs through HIP, selected by a
separate option (Linux only, mutually exclusive with `VMECPP_USE_CUDA`):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DVMECPP_USE_HIP=ON
cmake --build build
```

Requirements: ROCm with `hipcc` and hipFFT. `CMAKE_HIP_ARCHITECTURES`
defaults to `gfx90a;gfx942` (MI210/MI250 and MI300).

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
stellarator-symmetric (`lasym = false`), three-dimensional (`ntor >= 1`)
configurations, executed on a single radial rank, in both the single and
the batched execution modes. Batched free-boundary runs use the NESTOR
vacuum solver, one instance per configuration slot sharing one loaded
mgrid: every input in the batch must carry the same `mgrid_file` and
`extcur` (the boundaries and plasma profiles may differ in distinct
mode), and the vacuum activation iteration, the `nvacskip` cadence, and
the soft-start restart are batch-wide decisions, so each configuration's
trajectory sits in the drift family of its single run (a broadcast batch
agrees with the single run to roughly 1e-13 relative). The per-iteration
vacuum solves serialize on the host across the configurations, which
bounds the batched free-boundary speedup well below the fixed-boundary
one. `lasym` and axisymmetric (`ntor = 0`) inputs are rejected with
`absl::UnimplementedError`, and the radial domain is not partitioned
across OpenMP threads (`num_threads` is forced to 1 under the CUDA build;
host threading offers no benefit when the iteration body runs on the
device).

The `<M>` (volume-averaged spectral width) column of the progress
printout evaluates the current iteration state under the CUDA build; the
CPU path evaluates the restart backup, which lags by one iteration, so
the printed column can differ in the trailing digit at a given iteration.
The spectral width in the output file is computed by the output phase
from the converged state and is unaffected.

## Correctness contract

The CUDA path is held to the following contract against the CPU
implementation, verified on fixed-boundary stellarator configurations
through the full multigrid ramp:

- `aspect_ratio` matches the CPU result bit-for-bit.
- The volume-averaged and field-line-derived output quantities match within
  a drift family of 1e-5 to 1e-3 relative, originating from
  floating-point summation-order differences in reductions that are
  documented at each kernel.
- Iteration counts, restart-reason sequences, and multigrid stage
  transitions match the CPU trajectory. On inputs that trigger mid-run
  bad-Jacobian restarts, the timing of the restart events is sensitive
  to the documented drift, so iteration counts there match approximately
  rather than exactly (W7-X converges in 3146 iterations against the
  CPU's 2954, to the same equilibrium).

On free-boundary runs the boundary itself responds to the vacuum field
through the drift-sensitive trajectory (the vacuum-pressure activation
iteration shifts with the residual history), so every converged output
carries the drift family there; the bit-exact `aspect_ratio` clause
applies to fixed-boundary runs.

Reductions that must match the CPU bit-for-bit (the residual triples that
feed the time-step controller and the convergence gate) use serial or
order-preserving device implementations; reductions covered by the drift
tolerance (for example the differential volume under the fused
atomicAdd kernel) are documented as such at their kernel definitions in
`fft_toroidal_cuda_kernels.cu`.

## Batched execution

`run_batched_gpu(indata_list, ...)` solves N fixed-boundary equilibria in
one CUDA-resident iteration loop. All inputs must share `mpol`, `ntor`,
`nfp`, `lasym`, and the multigrid schedule, since the persistent device
buffers are dimensioned once. Two modes exist, selected by
`VMECPP_BATCH_DISTINCT`:

- Broadcast (default): the first input's spectra fill all N configuration
  slots; the call returns the single converged result. This mode
  exercises the batched kernel chain for measurement.
- Distinct (`VMECPP_BATCH_DISTINCT=1`): each configuration keeps its own
  boundary, with per-configuration pre-initialization, axis recomputation,
  residual evolution, time-step control, convergence gating, and outputs.
  The call returns one `OutputQuantities` per configuration, derived
  directly from the batched run's flushed device state
  (`VMECPP_PER_CFG_RECOMPUTE=0` opts out). Per-configuration converged
  spectra are also available via `return_spectra=True` or
  `VMECPP_BATCH_OUTPUTS_FILE` plus `recompute_outputs_from_spectra`.

A configuration that exceeds `VMECPP_PER_CFG_NITER_CAP` is marked timed out
without failing the rest of the batch. Every environment variable the
binding sets is restored when the call returns.

## Free-boundary execution

The NESTOR vacuum solve stays on the host. Bridges carry the
per-iteration traffic once the vacuum block is live: the decomposed
position state flushes to the host every iteration (the host spectral
path stays alive so `HandOverBoundaryGeometry` reads a current
`m_physical_x`), the axis and LCFS geometry rows, the outermost
totalPressure rows, and the bucoH/bvcoH profiles flush ahead of their
host reads, and the host-computed `rBSq` profile stages back to the
device, where a kernel applies the vacuum edge force to the LCFS row
ahead of the constraint assembly. The rCon0/zCon0 turn-off decay runs
as a device kernel.

The segment and whole-iteration CUDA graphs are disabled on
free-boundary runs: the edge-force kernel toggles with the vacuum
pressure state, which would invalidate a captured kernel sequence.

Sync elision (`VMECPP_SYNC_ELIDE=K`) covers free-boundary runs.
Iterations run live until the vacuum state machine reaches its active
state, so the activation check reads fresh residuals and the soft-start
restart replays the live sequence. Once active, the elision covers the
per-iteration scalar sync sites (the tau extrema, the residual triples,
the plasma volume) with the device time-step controller authoritative
and the convergence gate on the K-boundaries, while the vacuum block
(geometry flush, host triplet, NESTOR solve, edge-pressure staging)
keeps its per-iteration cadence and arithmetic. The converged outputs
sit in the documented free-boundary drift family.

The vacuum block does not join the elision window: the NESTOR response
must track the boundary every iteration. Running the vacuum solve only
on the K-boundaries leaves the iteration orbiting its window-stale
vacuum target instead of converging (the invariant force residuals
floor around 1e-4 on the cth-like case at K = 25 while the
preconditioned residuals keep descending), and rebuilding the staged
edge force on the device from the current LCFS geometry between
boundaries does not close the orbit either; both forms were measured
and rejected.

## Restart and state backup

The CPU controller's restart protocol (Fortran `irst`, typed as
`RestartReason` in `FlowControl`) rewinds the position state to a backup
copy and reduces the time step when the Jacobian changes sign or the
residual stops improving. Under the CUDA build the protocol operates on
device-resident state:

- The device backup of the six spectral position components is armed at
  the start of every multigrid stage and refreshed at the host backup
  cadence on subsequent iterations. The refresh is one fused copy kernel
  (`k_backup_pts_x`) on the iteration stream; under `VMECPP_KERNEL_TIMING`
  it measures 0.0048 ms per call and under 1% of iteration time on
  production-scale inputs (W7-X, sm_89), so the rollback target stays
  current at every improving iteration without a meaningful cost.
- A restore replays the backup into the position buffers and zeroes the
  integrator velocity, mirroring the host `decomposed_v.setZero()`. In
  batched mode the restore is gated per configuration by the restart
  mask, so one configuration's bad-Jacobian event does not rewind a
  neighbor that reported no restart.
- A bad Jacobian during the first iterations triggers the magnetic-axis
  recomputation and a retry from scratch; the recovery invalidates the
  device position, velocity, and backup state so the retry re-stages
  everything from the recomputed host axis.
- Under sync elision (`VMECPP_SYNC_ELIDE=K`) the backup refresh and the
  restart bookkeeping evaluate on the K-window boundaries, so a restart
  rewinds at most K-1 iterations.

## Runtime controls

Every environment variable read by the CUDA path, grouped by role. Boolean
knobs parse as `atoi(value) > 0`; "default ON" means active when unset, with
`=0` disabling.

### Batched execution

| Variable | Default | Effect |
|---|---|---|
| `VMECPP_N_CONFIG_MAX` | `1` | Number of configuration slots the persistent device buffers are dimensioned for. At 1 the layout and behavior are identical to the single-configuration path, bit-exact. Read once per process. |
| `VMECPP_BATCH_DISTINCT` | OFF | Selects distinct mode (see above). |
| `VMECPP_BATCH_MULTIGRID_UPSCALE` | OFF | Per-configuration multigrid stage transition: snapshots each configuration's device state at the stage boundary instead of broadcasting configuration 0's host upscale. Distinct mode requires this to carry per-configuration state across the `ns_array` ramp. |
| `VMECPP_BATCH_UPSCALE_KERNEL` | OFF | Host-exact per-configuration radial interpolation at each stage transition: scale to physical, odd-m axis extrapolation, linear interpolation in s, divide by the new stage's scalxc. Companion to the snapshot above. |
| `VMECPP_BATCH_PER_CFG_TIMESTEP` | ON | Per-configuration time-step controller: each configuration's `(fac, b1)` derive from its own residual instead of sharing the scalar tuned by configuration 0. Active when the batch holds more than one slot; distinct-mode convergence requires it when configurations converge at different rates. `=0` restores the shared scalar. |
| `VMECPP_BATCH_AXIS_RECOMPUTE` | ON | Proactive per-configuration magnetic-axis recomputation during distinct-mode pre-initialization. `=0` leaves axis recovery to the iteration body's bad-Jacobian path. |
| `VMECPP_BATCH_INPUTS_FILE` | unset | Binary file of per-configuration spectral inputs (`[component][cfg][spectra]`, int32 shape header). Written by the distinct-mode pre-initialization, read once by the first forward transform. |
| `VMECPP_BATCH_DEC_X_FILE` | unset | Companion file carrying each configuration's pre-triplet decomposed position state for the device time integrator's first iteration. |
| `VMECPP_BATCH_OUTPUTS_FILE` | unset | Destination for the end-of-run dump of every configuration's converged decomposed spectra, same layout as the inputs file. |
| `VMECPP_KEEP_BATCH_FILES` | OFF | Preserves the three batch files after `run_batched_gpu` returns. |
| `VMECPP_PER_CFG_NITER_CAP` | unset | Per-configuration iteration ceiling. A configuration that reaches the cap is marked timed out instead of holding the whole batch; the batch succeeds if at least one configuration converged. |
| `VMECPP_ACTIVE_PER_CFG_OVERRIDE_BITS` | unset | Bit mask overriding the active-configuration mask at run start. Pins a follow-up single-configuration run to slot 0 after a batched run. |
| `VMECPP_PER_CFG_RECOMPUTE` | OFF | In-process per-configuration `OutputQuantities` after a distinct-mode batched run: each configuration's converged device state flushes once and the standard output-derivation chain runs per configuration, with no additional iterations. Falls back to a single-stage hot-restart reconstruction per configuration, and the subprocess recompute via `recompute_outputs_from_spectra` remains available. Each derived `wout` carries that configuration's own iteration count and final invariant force residuals (`fsqr`/`fsqz`/`fsql`), snapshotted from the last iteration it was active before convergence masking zeros the live residual caches; compare them against `wout.ftolv` for per-configuration convergence. |
| `VMECPP_RECOMPUTE_FTOL` | last `ftol_array` entry | Tolerance for the single-stage hot-restart recompute run. |
| `VMECPP_RECOMPUTE_NITER` | unset | Iteration cap for the recompute run. |
| `VMECPP_RECOMPUTE_LAMBDA` | ON | Seeds the recompute hot restart with the converged lambda spectra. `=0` re-converges lambda from zero. |
| `VMECPP_RECOMPUTE_LAMBDA_SCALE` | `1.0` | Uniform factor on the lambda seed. |

### Production-path switches (default ON)

Each `=0` falls back to a slower equivalent path. All settings preserve the
bit-exact `aspect_ratio` contract.

| Variable | Effect of `=0` |
|---|---|
| `VMECPP_SCATTER_V5` | Shared-memory-cached fused scatter (v5) falls back to the L1-broadcast variant (v4). |
| `VMECPP_JAC_METRIC_FUSE` | The fused jacobian-and-metric kernel splits back into two launches. |
| `VMECPP_JAC_METRIC_DVDSH_FUSE` | The three-way jacobian+metric+dVdsH fusion (atomicAdd accumulation) splits into fused jacobian+metric plus a tree-reduced dVdsH. The atomic path's summation-order nondeterminism lands within the dVdsH drift tolerance. |
| `VMECPP_JAC_PAIR` | The jH-pair-coarsened jacobian variant (shared middle surface cached in shared memory) falls back to per-surface blocks. Requires even half-grid extent; automatic fallback otherwise. |
| `VMECPP_MHD_PAIR` | The jF-pair-coarsened MHD-forces variant falls back to per-surface blocks. Requires even force-grid extent; automatic fallback otherwise. |
| `VMECPP_RESIDUALS_PAR` | The 256-thread parallel residual reduction (summation-order difference within the drift family) falls back to the serial reduction that matches the CPU bit-for-bit per call. |
| `VMECPP_RESIDUALS_K` | Multi-block residual partitioning. Auto picks `K = max(1, 16 / n_config)` capped at 16; an explicit value overrides. `=1` forces single-block. |
| `VMECPP_DEALIAS_PACK` | The dealias inverse packs `32 / nThetaReduced` zeta planes onto each 32-lane warp (16 reduced thetas pack two planes per warp), filling lanes the one-plane kernel left idle; `=0` falls back to that kernel. Selected only when 32 is an exact multiple of `nThetaReduced`. |
| `VMECPP_SCATTER_PACK` | The main-and-constraint scatter packs zeta planes onto each warp the same way; `=0` falls back to the one-plane-per-warp kernel. Selected only when 32 is an exact multiple of `nThetaReduced`. |
| `VMECPP_UPDATE_GRAPH` | Disables the segment-3 CUDA graph (effectiveConstraintForce through assembleTotalForces). |
| `VMECPP_SEG2_GRAPH` | Disables the segment-2 CUDA graph (computeMetricElements through radialForceBalance plus hybridLambdaForce). |
| `VMECPP_SEG4_GRAPH` | Disables the segment-4 CUDA graph (the four preconditioner-apply wrappers; re-captured automatically when `jMax` changes). |
| `VMECPP_CONV_FLAG_AUTH` | The termination gate uses the host residual comparison instead of the device-side `k_check_convergence` flag. The gate also falls back to the host comparison automatically when the flag buffers are absent. |

### Opt-in throughput modes (default OFF)

| Variable | Effect |
|---|---|
| `VMECPP_SYNC_ELIDE` | `=K`: K-window sync elision. Per-iteration scalar D2H + stream syncs (tau extrema, residual triples, plasma volume) are skipped on non-boundary iterations; the device time-step controller is authoritative and the convergence gate and restart bookkeeping evaluate every K-th iteration. `K=25` aligns with the preconditioner cadence. Device-state backups move to the same cadence, so a bad-Jacobian event rewinds at most K-1 iterations. Free-boundary runs are covered: iterations run live until the vacuum contribution is fully active, and the vacuum block keeps its per-iteration cadence throughout (see the free-boundary section). |
| `VMECPP_RESIDUALS_DEFER` | Deferred-sync residuals: the iteration consumes one-iteration-stale residual values, eliminating a per-iteration sync stall. Within 10x ftolv the value is force-synced so the gate never fires on a stale read. |
| `VMECPP_ITER_GRAPH` | Whole-iteration CUDA graph. With sync elision active, each captured elided iteration replays as one `cudaGraphLaunch` instead of the per-kernel dispatch sequence (including both cuFFT execs). Captured after two eligible iterations; invalidated on multigrid stage transitions and restarts. No effect without `VMECPP_SYNC_ELIDE`. Replays bit-identically to plain elision; wall-neutral at the canonical shape because elided iterations already pipeline launches ahead of the GPU. Retained for dispatch-bound hosts and shapes. |
| `VMECPP_FWD_GRAPH` | CUDA graph over the forward-FFT chain. Graph-mode cuFFT shows no improvement on current toolkits. |

### Mixed-precision and alternate-FFT experiments (default OFF)

Measured outcomes per gate. The DD-pair primitives and Ozaki-slice
multiplications are documented at their definitions in
`fft_toroidal_cuda_common.cuh`, the Carson-Higham refinement at its
kernels in `fft_toroidal_cuda_kernels.cu`.

| Variable | Outcome |
|---|---|
| `VMECPP_FFT_FP32` | FP32 cuFFT. The force residual cannot converge below the FP32 noise floor; runs do not terminate at production ftol. |
| `VMECPP_FFT_RADIX` | Hand-coded radix-8x3 inverse DFT replacing cuFFT Z2D. Slower than cuFFT at the production shape; accumulation order falls outside the bit-exact contract. The factorization covers transform length 24 only; inputs with `nZeta != 24` stay on cuFFT, with a one-time notice. |
| `VMECPP_FWD_FFT_RADIX` | Forward-direction (D2Z) counterpart. Same result; stream-capturable, unlike cuFFT. Same length-24 coverage. |
| `VMECPP_DEALIAS_MIXED` | FP32 inner multiplies in the dealias inverse, FP64 accumulator. Same convergence floor as FP32 cuFFT. |
| `VMECPP_DEALIAS_SPLIT` | Four partial accumulators breaking the 11-deep FP dependency chain. Measured as a small regression; the compiler already extracts the ILP. |
| `VMECPP_SCATTER_DD_FP32` | FP32 multiplies with DD-pair accumulators in the scatter. |
| `VMECPP_SCATTER_DD_FP64MUL` | FP64 multiplies with DD-pair accumulators; isolates accumulator-order drift from multiply quantization. |
| `VMECPP_SCATTER_DD_FP32_DDMUL` | Dekker TwoProduct DD x DD multiplies on FP32 operands, ~96-bit products. |
| `VMECPP_SCATTER_OZAKI_FP32` | 2-slice Ozaki FP32 multiplications, ~50-bit precision. |
| `VMECPP_SCATTER_OZAKI3_FP32` | 3-slice Ozaki, ~72-bit precision; converges to the FP64 equilibrium within a few ULP of the converged scalars. |
| `VMECPP_SCATTER_CUBLAS_FP32` | Scatter as one cuBLAS GemmEx FP32 GEMM. FP32 precision floor; convergence breaks without downstream compensation. |
| `VMECPP_SCATTER_CUBLAS_OZAKI` | Four-GEMM Ozaki: FP32 hi/lo slices per operand, DD-pair reassembly, ~48-bit precision. |
| `VMECPP_SCATTER_CUSTOM_GEMM` | Tile-cooperative GEMM with per-multiply Veltkamp-Dekker and DD accumulation. |
| `VMECPP_SCATTER_CUSTOM_GEMM_WMMA` | TF32 tensor-core dispatch: 3-slice Ozaki limbs, 54 `wmma::mma_sync` per tile. The wmma-only sum reaches rel ~3e-6; production precision comes from the scalar Veltkamp-Dekker pass over the same shared-memory data. The tile geometry covers `mpol <= 12` and `nThetaReduced <= 16`; larger inputs fall back to the production scatter with a one-time notice. |
| `VMECPP_SCATTER_I8OZAKI` | int8 tensor-core scatter via the Ozaki construction: eight 7-bit limbs per FP64 operand, exact s32 accumulation, no scalar recovery pass. Converges within a few ULP of FP64. Same tile coverage as the wmma path. |
| `VMECPP_SCATTER_I8GEMM` | Batched int8-Ozaki GEMM: (config, surface, zeta) fold into one GEMM row axis, the basis-side limb matrix builds once per shape, and only the spec rows slice per iteration. No tile shape limits. Converges within a few ULP of FP64. At the scatter's K (at most 16 mpol) the eight limb planes carry as many bytes as the FP64 operands, so the kernel stays memory-bound and the production scatter remains faster at saturated shapes. Measured on sm_89 (RTX 6000 Ada): the int8 scatter runs at 0.084 ms per call against the production scatter's 0.013 ms (6.5x), and the higher int8-tensor throughput of the larger part does not help a bandwidth-bound kernel; the run still converges to the canonical result. |
| `VMECPP_SCATTER_TF32_PLAIN` | Plain TF32 accumulator sum, rel ~3e-6. |
| `VMECPP_SCATTER_I8_LIMBS` | Limb width for the int8 scatter paths: `4` selects 28-bit operands (rel ~4e-9) at half the limb-plane traffic and half the mma work; the default `8` covers the FP64 mantissa. Pure 4-limb runs stall above the convergence tolerance. Under `VMECPP_IR_STAGED` the residual phase routes the width per iteration: 4 above the threshold with a decade hysteresis band, 8 below; a width change drops the whole-iteration graph. The 4-limb descent inflates iteration counts past its bandwidth saving, so the staged mode trails the pure 8-limb path, and inputs that recover from mid-run bad Jacobians do not converge under it. |
| `VMECPP_RESIDUALS_DD_FP32` | DD-pair FP32 accumulator in the residual reduction. |
| `VMECPP_RZ_IR_FP32` | Carson-Higham iterative refinement on the RZ tridiagonal solve: FP32 PCR, FP64 residual, FP32 correction, FP64 combine. Halves PCR shared memory. |
| `VMECPP_IR_STAGED` | Staged-precision descent: hot kernels run FP32/TF32 while the residual is above threshold, FP64 below it. |
| `VMECPP_IR_THRESHOLD` | Crossover residual for the staged descent. Default `1e-5`. |
| `VMECPP_IR_LOG_EVERY` | Logging cadence for staged-precision phase transitions. |

### Diagnostics (default OFF)

| Variable | Effect |
|---|---|
| `VMECPP_KERNEL_TIMING` | Per-kernel cudaEvent timing, dumped at exit and every 10k events. Auto-disables the CUDA graphs (cudaEventRecord is illegal inside graph capture) and adds per-call syncs. |
| `VMECPP_KERNEL_TIMING_PATH` | Dump destination. Default `/tmp/vmecpp_kernel_timing.log`. |
| `VMECPP_PHASE_TIMING_PATH` | Destination for the phase-timer report. Default `/tmp/vmecpp_phase_timing.txt`. |
| `VMECPP_FFT_DUMP` | One-shot dump of the cuFFT input/output plus the radix-8x3 recomputation on the same input. Requires `nZeta = 24` (the radix kernel's coverage); skipped otherwise with a notice. |
| `VMECPP_CPU_ORDER_BCONTRA` | Serial ascending-kl accumulation of the jvPlasma and avg_guu_gsqrt reductions, matching the host loop bit for bit. Trajectory-comparison diagnostic against the CPU build. |
| `VMECPP_CPU_ORDER_PRECOND` | Host-order serial accumulation of the radial-preconditioner matrix elements, including the host's division forms. Trajectory-comparison diagnostic. |
| `VMECPP_CPU_ORDER_RZSOLVE` | Serial Thomas elimination in the host order instead of parallel cyclic reduction. Trajectory-comparison diagnostic. |
| `VMECPP_DUMP_TCON` | One-shot full-precision print of the first constraint-multiplier profile. |
| `VMECPP_DUMP_GCON` | One-shot print of the effective-constraint-force checksum and the per-surface sums of the dealiased constraint force. |
| `VMECPP_DUMP_SPECS` | One-shot dump of the staged spectral input (configuration 0) for distinct-vs-broadcast bit-equivalence checks. |
| `VMECPP_STATE_DUMP_ITERS` | Comma-separated `iter2` values at which the full batched decomposed-x state is written to disk. |
| `VMECPP_STATE_DUMP_F` / `VMECPP_STATE_DUMP_PROF` | Adds the decomposed forces / per-configuration radial profiles to the state dumps. |
| `VMECPP_STATE_DUMP_PATH` | Filename prefix for the state dumps. |
| `VMECPP_PERCFG_RESIDUAL_DUMP` | `=K`: logs each configuration's residual triple every K iterations. |
| `VMECPP_TRACE_RESTART` | Logs every backup store/restore event with the controller inputs that drove it. |
| `VMECPP_TRACE_CFG_DIFF` | Per-call max-abs-difference probe between configuration 0 and 1 slices of named device buffers. |
| `VMECPP_CONV_FLAG_DEBUG` | Logs any disagreement between the device convergence flag and the host gate. |
| `VMECPP_VALIDATE_DEVICE_TIMESTEP` | One-shot comparison of the device time-step controller's `(fac, b1)` against the host-computed values. |
| `VMECPP_DEFENSIVE_BROADCAST` | Re-broadcasts configuration 0's position state into all slots on every recompose. Correct under broadcast inputs, redundant under distinct; catches configuration-zero-only write regressions. |
| `VMECPP_BATCH_PERTURB` | Scales each configuration's input spectra by `1 + scale * cfg / n_cfg` to exercise the per-configuration path with non-identical inputs. |

# OpenMP Synchronisation in VMEC++ — Investigation Report

This report tracks the diagnostic and optimisation steps from the plan at
`~/.claude/plans/the-vmec-flow-control-compiled-boot.md`. Each section
captures what was measured, what we learned, and what to do next.

Benchmark workload (unless noted): `examples/data/input.w7x` with
`NS_ARRAY=300`, `FTOL_ARRAY=1.E-9` (heavier than the original committed
`NS_ARRAY=99`/`FTOL=1.E-12` — local-only edit so barrier savings show up
above the bench noise floor).

Common environment for every run: `OMP_WAIT_POLICY=passive`,
`OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

---

## Baseline — `omp-atomics` tip (commit dae2b92)

Branch state: atomics + per-thread slot reductions for residuals,
energies, and spectral width. **All hot-path reductions are on the slot
pattern; no `critical` sections remain on the hot path.** This is what
we are starting from.

### Wall time (heavy input.w7x, passive)

Branch: `omp-atomics` (dae2b92), venv `.venv-stage2b`.
3 reps per thread count, `OMP_WAIT_POLICY=passive`.

| Threads | Rep 1 (s) | Rep 2 (s) | Rep 3 (s) | Median (s) | T(1)/T(N)/N |
|---------|-----------|-----------|-----------|------------|-------------|
| 1       | 89.90     | 88.28     | 89.33     | 89.33      | 1.00        |
| 4       | 27.88     | 27.46     | 27.59     | 27.59      | 0.81        |
| 10      | 17.49     | 17.32     | 17.27     | 17.32      | 0.52        |
| 20      | 19.69     | 20.50     | 37.79     | 20.50      | 0.22        |
| 30      | 21.47     | 20.06     | 25.89     | 21.47      | 0.14        |

Scaling efficiency T(1)/(T(N)\*N): t=4 is 81%, t=10 drops to 52%, t=20
to 22%, t=30 to 14%. The scaling plateau at t=10 confirms the barrier
imbalance finding — adding more threads past ~10 yields diminishing returns
because load imbalance dominates.

### Barrier wait totals (re-measured on omp-instrument-v2, commit 45719ac)

Measured on branch `omp-instrument-v2` (same TIMED_BARRIER sites as
`omp-instrument`, just extended with new instrumentation).

| Site | total@t=10 (s) | total@t=30 (s) | avg-us t10 | avg-us t30 | ratio |
|------|---------------|---------------|-----------|-----------|-------|
| `computeJacobian.post_critical`    | 15.12 | 27.97 | 1306 | 805 | 0.62x |
| `evalFResInvar.slot_publish`       | 13.31 | 24.05 | 1158 | 698 | 0.60x |
| `assembleTotalForces.entry`        |  9.91 | 21.00 |  862 | 609 | 0.71x |
| `pressureAndEnergies.slot_publish` |  6.85 | 14.37 |  596 | 417 | 0.70x |
| `hybridLambdaForce.exit`           |  3.50 | 10.48 |  304 | 304 | 1.00x |
| `hybridLambdaForce.entry`          |  3.33 |  8.71 |  290 | 253 | 0.87x |
| `applyRZPreconditioner.A_gather`   |  3.10 |  8.61 |  270 | 250 | 0.93x |
| `applyM1Preconditioner.exit`       |  2.92 |  7.35 |  254 | 213 | 0.84x |
| `evalFResPrecd.slot_publish`       |  2.62 |  7.79 |  228 | 226 | 0.99x |
| `applyRZPreconditioner.B_solve`    |  2.73 |  8.04 |  238 | 233 | 0.98x |

Note: avg_us decreases t10->t30 because total time is spread across more
threads (each thread hits fewer barriers per unit wall-clock). Absolute
total time is the load-imbalance signal.

Aggregate: ~25% thread-time wasted at barriers at t=10, higher at t=30.

---

## Step 2a — Which thread executes each `single` block?

**Hypothesis.** Under libgomp, `#pragma omp single` is "first arriver
wins". Conventional wisdom says the master thread (tid 0) wins almost
every race, accumulating extra non-radial work and consistently
arriving at every downstream barrier last. Verify or refute.

### Method

Extended `omp_barrier_timing.h` with `RecordSingleExecutor(site_id, tid)`
and `RECORD_SINGLE_EXECUTOR(site_id)` macro. Added per-(site, tid) plain
`int64_t` counter array (single-writer-per-tid, no atomics needed). Wrapped
all hot-path `#pragma omp single` bodies in `ideal_mhd_model.cc` and
`vmec.cc` with `RECORD_SINGLE_EXECUTOR`. Dump triggered by
`VMECPP_DUMP_BARRIER_TIMINGS=1`. Branch: `omp-instrument-v2`, commit 45719ac.

### Results — t=10 (10 threads)

The hypothesis that tid 0 monopolises is **refuted**. Execution counts are
broadly uniform across threads, with tid 0 winning 7-14% of races —
close to the expected 1/10 = 10% for a fair lottery.

| site | tid0 | tid1 | tid2 | tid3 | tid4 | tid5 | tid6 | tid7 | tid8 | tid9 | total | tid0% |
|------|------|------|------|------|------|------|------|------|------|------|-------|-------|
| Evolve.clear_restart_reason_single | 109 | 101 | 127 | 119 | 125 | 120 | 110 | 118 | 105 | 124 | 1158 | 9.4% |
| pressureAndEnergies.publish_single | 89 | 107 | 125 | 137 | 107 | 130 | 101 | 105 | 122 | 126 | 1149 | 7.7% |
| evalFResInvar.slot_publish_single | 91 | 121 | 115 | 107 | 128 | 149 | 109 | 110 | 100 | 119 | 1149 | 7.9% |
| evalFResPrecd.slot_publish_single | 92 | 108 | 120 | 116 | 120 | 104 | 133 | 114 | 125 | 117 | 1149 | 8.0% |
| update.huge_initial_forces_check_single | 100 | 108 | 110 | 117 | 116 | 101 | 139 | 114 | 131 | 113 | 1149 | 8.7% |
| Evolve.stopping_criterion_single | 90 | 115 | 127 | 118 | 116 | 106 | 118 | 120 | 124 | 124 | 1158 | 7.8% |
| Evolve.invtau_update_single | 93 | 116 | 138 | 102 | 113 | 114 | 118 | 121 | 119 | 123 | 1157 | 8.0% |
| SolveEqLoop.time_step_control_single | 150 | 106 | 124 | 121 | 94 | 103 | 109 | 110 | 87 | 154 | 1158 | 13.0% |
| SolveEqLoop.vac_state_activate_single | 111 | 109 | 121 | 108 | 114 | 123 | 101 | 121 | 117 | 133 | 1158 | 9.6% |
| computeForceNorms.zero_single | 8 | 5 | 0 | 6 | 6 | 3 | 9 | 10 | 5 | 6 | 58 | 13.8% |
| computeForceNorms.normalize_single | 5 | 5 | 7 | 8 | 5 | 4 | 7 | 3 | 8 | 6 | 58 | 8.6% |
| updateVolume.zero_single | 4 | 6 | 6 | 4 | 5 | 5 | 7 | 10 | 4 | 7 | 58 | 6.9% |
| update.preconditioner_update_single | 0 | 4 | 9 | 6 | 4 | 4 | 7 | 11 | 9 | 4 | 58 | 0.0% |
| computeInitialVolume.zero_single | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0.0% |

### Results — t=30 (30 threads)

At t=30, distribution across 30 threads is even more uniform. A few selected
rows (only first 10 tids shown for brevity — see full log at
`/tmp/run_t30.log`):

| site | tid0 | tid1 | ... | tid29 | total | tid0% |
|------|------|------|-----|-------|-------|-------|
| Evolve.clear_restart_reason_single | 38 | 35 | ... | 38 | 1158 | 3.3% |
| evalFResPrecd.slot_publish_single | 25 | 19 | ... | 41 | 1149 | 2.2% |
| SolveEqLoop.time_step_control_single | 38 | 40 | ... | 37 | 1158 | 3.3% |
| pressureAndEnergies.publish_single | 84 | 49 | ... | 65 | 1149 | 7.3% |

`pressureAndEnergies.publish_single` stands out slightly at t=30 (tid 0
wins 84/1149 = 7.3%, vs expected 3.3%), but it is a nowait single so
imbalance here does not create a downstream barrier. All guarded (non-nowait)
singles show distributions close to uniform.

### Overhead comparison (t=10, passive)

| Build | Rep 1 (s) | Rep 2 (s) | Rep 3 (s) | Median |
|-------|-----------|-----------|-----------|--------|
| baseline (.venv-stage2b) | 17.49 | 17.32 | 17.27 | 17.32 |
| instrumented (.venv-instr2, no dump) | 17.83 | 17.54 | 17.59 | 17.59 |

Overhead: +1.6% (within noise; 0.27 s absolute on a ~17 s run). Instrumentation
does not measurably perturb the system.

### Insight / next step

**The tid-0-monopolises-singles hypothesis is false.** Under libgomp with
`OMP_WAIT_POLICY=passive`, single blocks are won by whichever thread wakes
from its idle-wait first — a near-uniform lottery across all tids. No single
site shows >15% concentration in any one thread. This means the observed
load imbalance at barriers is NOT caused by extra serial work accumulating
on one thread via `omp single`. The root cause must be in the parallel
radial-loop work distribution itself (computeJacobian, assembleTotalForces,
hybridLambdaForce) — see Step 2b and 2c below.

---

## Step 2b — Per-thread time inside `computeJacobian`

**Hypothesis.** `computeJacobian.post_critical` is the largest absolute
sink (35 s thread-wait at t=30) but flat-ratio with thread count, so
the slow thread is doing more *work*, not waiting longer. Identify
which thread.

### Method

Added `vmecpp::omp_timing::FunctionTimer` RAII class to the instrumentation
header. It records `omp_get_wtime()` at construction and accumulates elapsed
time into per-(site, tid) plain `int64_t` slots at destruction (single-writer-
per-tid, no atomics needed). Placed `vmecpp::omp_timing::FunctionTimer
_t("computeJacobian.body");` at the top of `IdealMhdModel::computeJacobian()`.
Output dumped alongside barrier and single tables at exit.

Radial slice sizes for NS=300 (fixed boundary, 299 surfaces distributed):
- t=10: tids 0-8 each get 30 surfaces; tid 9 gets 29. Equal distribution.
- t=30: tids 0-28 each get 10 surfaces; tid 29 gets 9. Equal distribution.

### Results — t=10 (10 threads)

| tid | count | total_s | avg_us |
|-----|-------|---------|--------|
| 0   | 1158  | 2.143   | 1850 |
| 1   | 1158  | 1.905   | 1645 |
| 2   | 1158  | 1.921   | 1659 |
| 3   | 1158  | 1.919   | 1658 |
| 4   | 1157  | 1.902   | 1644 |
| 5   | 1158  | 1.975   | 1705 |
| 6   | 1158  | 1.923   | 1661 |
| 7   | 1158  | 1.960   | 1692 |
| 8   | 1158  | 1.884   | 1627 |
| 9   | 1158  | 2.117   | 1828 |

Slowest threads: **tid 0 (1850 us/call)** and **tid 9 (1828 us/call)**.
Fastest: tid 8 (1627 us/call). Ratio slowest/fastest = 1.137x.
Both slow threads own boundary surfaces: tid 0 owns the magnetic axis
(nsMinF = 0), tid 9 owns the LCFS (nsMaxF1 = 300).

### Results — t=30 (30 threads)

| tid | count | total_s | avg_us |
|-----|-------|---------|--------|
| 0   | 1158  | 1.327   | 1146 |
| 1   | 1158  | 1.125   | 971 |
| 2-28 (range) | 1158 | 1.073-1.137 | 926-981 |
| 29  | 1158  | 1.211   | 1046 |

Slowest: **tid 0 (1146 us/call)** and **tid 29 (1046 us/call)**.
Fastest: tid 21 (926 us/call). Ratio slowest/fastest = 1.238x.
Again tid 0 (axis owner) and tid 29 (LCFS owner) are the slowest.
The pattern persists regardless of thread count.

### Insight / next step

**Tid 0 (axis owner) and the LCFS-owner thread are systematically 12-24%
slower in `computeJacobian` than interior threads, despite having equal or
smaller radial-slice sizes (30 vs 29 for t=10, 10 vs 9 for t=30).** This
strongly implicates boundary-surface extra work: at the axis the code must
handle the degenerate (r=0) case and the `r1e_i`/`r1o_i` initialisation from
the innermost full-grid surface; at the LCFS thread the `rzConIntoVolume`
handover and free-boundary force contributions run on the same thread.
These are genuinely more expensive per-surface operations, not a partitioning
unfairness in surface count. The fix must reduce the slice count for tids 0
and the LCFS owner to compensate.

---

## Step 2c — Per-thread time inside `assembleTotalForces` and `hybridLambdaForce`

**Hypothesis.** Same load-imbalance pattern as `computeJacobian` —
combined ~39 s wait at t=30, similar 1.3-1.6× ratio.

### Method

Reused `FunctionTimer`. Placed `vmecpp::omp_timing::FunctionTimer
_t("assembleTotalForces.body")` at the top of `assembleTotalForces()` and
`vmecpp::omp_timing::FunctionTimer _t("hybridLambdaForce.body")` at the top
of `hybridLambdaForce()`. Same run as Step 2b.

### Results — assembleTotalForces, t=10

| tid | count | total_s | avg_us |
|-----|-------|---------|--------|
| 0   | 1149  | 1.435   | 1249 |
| 1   | 1149  | 1.393   | 1213 |
| 2   | 1149  | 1.455   | 1266 |
| 3   | 1149  | 1.454   | 1265 |
| 4   | 1149  | 1.426   | 1241 |
| 5   | 1149  | 1.452   | 1263 |
| 6   | 1149  | 1.419   | 1235 |
| 7   | 1149  | 1.398   | 1216 |
| 8   | 1149  | 1.410   | 1227 |
| 9   | 1149  | 1.501   | 1306 |

Slowest: **tid 9 (1306 us/call)**. Second slowest: tid 0 (1249 us/call).
Fastest: tid 1 (1213 us/call). Ratio 9/1 = 1.077x (mild).

### Results — assembleTotalForces, t=30

| tid | count | total_s | avg_us |
|-----|-------|---------|--------|
| 0   | 1149  | 0.981   | 854 |
| 1-28 (range) | 1149 | 0.880-0.944 | 766-822 |
| 29  | 1149  | 0.968   | 842 |

Slowest: **tid 0 (854 us/call)** and tid 29 (842 us/call).
Fastest: tid 14 (766 us/call). Ratio 0/14 = 1.115x.

### Results — hybridLambdaForce, t=10

| tid | count | total_s | avg_us |
|-----|-------|---------|--------|
| 0   | 1149  | 1.077   | 937 |
| 1   | 1149  | 1.057   | 920 |
| 2   | 1149  | 1.050   | 914 |
| 3   | 1149  | 1.019   | 887 |
| 4   | 1149  | 1.064   | 926 |
| 5   | 1149  | 1.069   | 930 |
| 6   | 1149  | 1.062   | 924 |
| 7   | 1149  | 1.050   | 913 |
| 8   | 1149  | 1.058   | 921 |
| 9   | 1149  | 1.064   | 926 |

Slowest: **tid 0 (937 us/call)**. Fastest: tid 3 (887 us/call).
Ratio = 1.056x (mild — axis overhead smaller here than in computeJacobian).

### Results — hybridLambdaForce, t=30

All 30 threads tightly clustered: 692-704 us/call, ratio 1.017x.
`hybridLambdaForce` distributes nearly perfectly at t=30 — the axis/LCFS
overhead is proportionally smaller here.

### Insight / next step

**`assembleTotalForces` shows the same tid 0 (axis owner) and LCFS-owner
slowdown pattern as `computeJacobian`, confirming that the axis and LCFS
threads consistently carry extra work across all radial-loop kernels.**
`hybridLambdaForce` has a milder imbalance (~5% at t=10, <2% at t=30),
suggesting the boundary extra work is smaller in that function. The primary
targets for imbalance-aware partitioning are `computeJacobian` and
`assembleTotalForces`, where tid 0 and LCFS-owner are 8-24% slower than
interior threads.

### Overhead comparison (t=10, passive)

Instrumentation runs (with `VMECPP_DUMP_BARRIER_TIMINGS=1` off) vs baseline:

| Build | Median (s) |
|-------|-----------|
| baseline | 17.32 |
| instrumented | 17.59 |

Overhead: +1.6% — well within the ~3% budget. The instrumentation does not
perturb the system.

---

## Step 3 — Imbalance-aware partitioning fix

_Not started yet — depends on Step 2 findings._

---

## Step 4 — Re-measure and decide

_Not started yet._

---

## Bench-script noise floor reminder

`scripts/bench_omp_barriers.py` on `examples/data/w7x.json` has ~3 %
noise floor at -t 4. The heavier `examples/data/input.w7x` (NS=300)
should give a tighter relative signal because the iteration count is
much higher and barrier savings compound.

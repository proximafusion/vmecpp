# Initial Jacobian recovery

This experiment measures recovery from a bad initial Jacobian while retaining
the input boundary coefficients, profiles, flux and final resolution/tolerance.
The change enables the existing three-surface bootstrap after the first cold
grid fails, then resumes the requested grid sequence.

```text
requested first grid -> initial Jacobian failure -> cold three-surface solve
                                                            |
                                                            v
                                  original grids and final force tolerance
```

`cases.csv` freezes 53 historical QUASR initialization failures by case ID and
source-file SHA256. Twelve equally spaced entries in the sorted list formed the
discovery set; the remaining 41 were held out while the recovery was selected.
This is a selected stress set, not a population estimate of VMEC++ failures.
`controls.csv` contains five existing successful free-boundary configurations.

## Reproduce

Build baseline commit `5611d9f3c39b73b5706845e481037a9b54322d45` and this revision
in separate Python environments with the same compiler, dependencies and build
options. Include the `benchmark` dependencies. Run the script from this checkout
for both environments so input generation and validation are shared.

```sh
BASELINE_PY=/path/to/baseline/bin/python
CANDIDATE_PY=/path/to/candidate/bin/python
SCRIPT=benchmarks/initial_jacobian_recovery/run.py
WORK=/path/to/recovery-inputs

# Download the frozen serial files and prepare each field table once.
"$CANDIDATE_PY" "$SCRIPT" prepare --work-dir "$WORK"
"$CANDIDATE_PY" "$SCRIPT" prepare --work-dir "$WORK" \
  --manifest benchmarks/initial_jacobian_recovery/controls.csv

# Paired, interleaved fresh processes. Preserve failures and timeouts.
"$CANDIDATE_PY" "$SCRIPT" run --work-dir "$WORK" \
  --runner baseline "$BASELINE_PY" --runner candidate "$CANDIDATE_PY" \
  --repeats 3 --threads 4 --output /path/to/recovery-results

"$CANDIDATE_PY" "$SCRIPT" run --work-dir "$WORK" \
  --manifest benchmarks/initial_jacobian_recovery/controls.csv \
  --runner baseline "$BASELINE_PY" --runner candidate "$CANDIDATE_PY" \
  --repeats 6 --threads 4 --output /path/to/control-results
```

An existing source cache can be supplied with `--serial-cache`. Preparation
refuses changed source hashes and existing prepared input directories. Each run
has a 120-second process limit. The default inputs use `mpol=ntor=6`,
`ns_array=[8,16,31]`, `ftol=1e-9`, 4,000 iterations per requested grid and
`nvacskip=6`, with the field-table settings of `test_free_boundary_quasr.py`.
The automatic bootstrap uses the existing `ns=3`, `ftol=1e-4` settings.

The result directories contain the exact inputs, full successful outputs,
force histories, process logs, binary fingerprints and an aggregate JSONL
ledger. Ordinary error handling is enabled. Success requires all three finite
force residuals below the requested tolerance at the final requested radial
resolution; a normal return or `ier_flag=0` alone is insufficient.

## Measurement scope

Solver time includes initialization, recovery attempts, the requested solves
and output-quantity computation. Field-table preparation, file writing and
independent validation are outside that timer and are recorded separately.
`iterations` is `wout.itfsq`, the stored force-history length; the final
tolerance-satisfying observation is not included in that history.

The control timing comparison retains all six randomly ordered repetitions.
The recovery comparison retains all three repetitions.
Failure latency and per-case outcomes must be inspected alongside aggregate
timings. No speedup is claimed for this recovery change.

The direct coil normal-field diagnostic uses an area-weighted RMS of
`B_coil . n / |B_coil|` on a shifted 48-by-96 angular grid. It measures agreement
with a pure-vacuum target. Zero prescribed pressure and net toroidal current do
not by themselves exclude force-free currents or interface current sheets, so
this diagnostic does not certify a general VMEC equilibrium. Reaching the
discrete force tolerance is also distinct from resolving physical error.

The accompanying fixed-boundary regression is checked against a regenerated
educational_VMEC 8.52 result, including the full geometry, lambda and iota arrays.
The explicit cold-three control checks the recovered force history and final
state. Tests exercise ordinary and best-effort error handling, every output
mode, insufficient budgets, and one/four-thread Python execution.

Measured results and environment details are recorded alongside this protocol.

## Recorded results

The same 15 configurations converge in all three candidate repetitions; none
of the 53 converge in any baseline repetition. The split is 5/12 discovery
cases and 10/41 held-out cases. All 60 successful-control runs converge.

| Control | Baseline median (s) | Candidate median (s) |
|---|---:|---:|
| 954 | 1.162 | 1.163 |
| 9914 | 1.111 | 1.001 |
| 19940 | 1.225 | 1.118 |
| 29346 | 1.044 | 1.029 |
| 65579 | 2.361 | 2.326 |

The per-case one-sided 95% Student-t upper bounds on the mean paired
log-time ratio correspond to candidate/baseline ratios of 1.065, 1.063,
0.993, 1.097 and 1.015, respectively. Each lies below the predeclared 1.10
material-slowdown margin. These are separate bounds for six pairs per case
under the paired log-time model.

Unrecovered candidate runs take a median 0.0047 seconds, with a maximum
3.273 seconds. The latter is additional recovery work on a case that still
fails; the repair does not make every failed input cheaper.

The measurements used Apple M4, macOS 26.5, Python 3.14.5, Apple Clang 17,
Release/LTO builds with FFTX enabled, and four solver threads.
`environment.json` records dependency and binary provenance;
`recovery-results.csv` and `control-results.csv` retain every measured run.

Validation on the candidate also includes 23 supported native VMEC test
targets, the two new Python reference checks, and independent fixed/free
boundary API probes across output modes and thread counts. The paired Python regression run had 201 passing tests and the same six
failures on baseline and the recovery implementation, plus two skips and
36 deselections. This preceded the final interface-preserving wrapper
refactor; native tests, Python reference checks and independent API probes
were repeated on the final build. A separate LI383 control at 4.26% beta retains
an identical complete output with the input tolerance of `1e-6`.

# Benchmarks

Performance improvements and regressions of `vmecpp` can be tracked below on a per-commit basis. A larger set of benchmarks against VMEC2000 can be found at [`proximafusion/vmecpp-benchmarks`](https://github.com/proximafusion/vmecpp-benchmarks).
A small but representative set of benchmarks runs automatically on every push to `main` and on pull requests, using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) (Python, end-to-end) and [Google Benchmark](https://github.com/google/benchmark) (C++, function-level), both tracked via [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark/).

## End-to-end benchmark suite (Python)

| Benchmark | Description |
|-----------|-------------|
| `test_bench_cli_startup` | CLI startup time (`vmecpp -h`) |
| `test_bench_cli_invalid_input` | CLI error path (`vmecpp invalid_input`) |
| `test_bench_fixed_boundary_w7x` | Fixed-boundary W7-X equilibrium (5-period stellarator, mpol=12, ntor=12, ns=99) |
| `test_bench_fixed_boundary_cma` | Fixed-boundary CMA equilibrium (stellarator, ntor=6, mpol=5) |
| `test_bench_response_table_from_coils` | Magnetic field response table creation from coils file |
| `test_bench_free_boundary` | Free-boundary solve with pre-computed response table |

## Microbenchmark suite (C++)

These target individual hot functions so a regression can be attributed to a specific kernel rather than only showing up in the end-to-end timings above. Each is a Google Benchmark `cc_binary` under `src/vmecpp/cpp/`, swept over several `(mpol, ntor)` resolutions.

| Benchmark | Description |
|-----------|-------------|
| `DftFourierToReal` / `FftFourierToReal` | Inverse transform (Fourier -> real-space geometry), DFT fallback vs FFTX path |
| `Dft` / `Fft` (real-space sweep) | Inverse transform cost as the real-space grid (`ntheta`, `nzeta`) is scaled |
| `BM_FourierToReal_Parallel_W7x_4t` | Inverse transform under VMEC's persistent multi-threaded OpenMP call pattern |
| `DeAliasConstraintForce` | Spectral-condensation constraint-force de-aliasing (forward+inverse transform pair) |
| `LaplaceSolve` / `LaplaceDecompose` | Free-boundary (NESTOR) dense Laplace solve: assemble + LU factorize + back-substitute |
| `TransformGreensFunctionDerivative` | Free-boundary Green's-function-derivative Fourier transform |
| `ComputeOutputQuantities` | Post-solve output computation (wout, jxbout, Mercier, ...) -- time spent after the solve has converged |

FFT-path benchmarks are skipped at resolutions for which no vendored FFTX codelet exists (the solver falls back to the DFT path there), so only resolutions with an actual FFT kernel appear in the FFT charts.

```{raw} html
<script src="https://cdn.jsdelivr.net/npm/chart.js@2.9.2/dist/Chart.min.js"></script>
<script src="benchmarks/data.js"></script>
<script src="_static/benchmarks.js"></script>
<div id="benchmark-charts">
  <p><em>Loading benchmark data...</em></p>
</div>
```

Click on any data point to open the corresponding commit.

## Running locally

Python end-to-end benchmarks:

```bash
pip install -e .[benchmark]
pytest benchmarks/test_benchmarks.py -v
```

To produce a JSON report:

```bash
pytest benchmarks/test_benchmarks.py --benchmark-json=benchmark_results.json
```

C++ microbenchmarks are built and run via Bazel, e.g.:

```bash
cd src/vmecpp/cpp
bazel run --config=perf -- //vmecpp/vmec/ideal_mhd_model:fft_toroidal_bench
```

The other three targets are `//vmecpp/vmec/ideal_mhd_model:dealias_constraint_force_bench`,
`//vmecpp/free_boundary/laplace_solver:laplace_solver_bench`, and
`//vmecpp/vmec/output_quantities:output_quantities_bench`. CI builds, runs, and merges
all four into a single JSON report for the regression tracker (see
`.github/actions/run-benchmarks/action.yml`).

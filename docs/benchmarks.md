# Benchmarks

VMEC++ tracks performance of key operations using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/).
Benchmarks run automatically on every push to `main` and on pull requests.

## Benchmark suite

| Benchmark | Description |
|-----------|-------------|
| `test_bench_cli_startup` | CLI startup time (`vmecpp -h`) |
| `test_bench_cli_invalid_input` | CLI error path (`vmecpp invalid_input`) |
| `test_bench_fixed_boundary_w7x` | Fixed-boundary W7-X equilibrium (5-period stellarator, mpol=12, ntor=12, ns=99) |
| `test_bench_fixed_boundary_cma` | Fixed-boundary CMA equilibrium (stellarator, ntor=6, mpol=5) |
| `test_bench_response_table_from_coils` | Magnetic field response table creation from coils file |
| `test_bench_free_boundary` | Free-boundary solve with pre-computed response table |

All solver benchmarks use `max_threads=1` and `verbose=False` for reproducibility.

## Interactive dashboard

The interactive benchmark dashboard with historical trends is available at
[benchmarks/index.html](benchmarks/index.html).

The dashboard is populated after the first benchmark run on `main` completes.

## Running locally

```bash
pip install -e .[benchmark]
pytest benchmarks/test_benchmarks.py -v
```

To produce a JSON report:

```bash
pytest benchmarks/test_benchmarks.py --benchmark-json=benchmark_results.json
```

# Benchmarks

Performance improvements and regressions of `vmecpp` can be tracked below on a per-commit basis. A larger set of benchmarks against VMEC2000 can be found at [`proximafusion/vmecpp-benchmarks`](https://github.com/proximafusion/vmecpp-benchmarks).
A small but representative set of benchmarks runs automatically on every push to `main` and on pull requests, using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) and [github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark/).

## Benchmark suite

| Benchmark | Description |
|-----------|-------------|
| `test_bench_cli_startup` | CLI startup time (`vmecpp -h`) |
| `test_bench_cli_invalid_input` | CLI error path (`vmecpp invalid_input`) |
| `test_bench_fixed_boundary_w7x` | Fixed-boundary W7-X equilibrium (5-period stellarator, mpol=12, ntor=12, ns=99) |
| `test_bench_fixed_boundary_cma` | Fixed-boundary CMA equilibrium (stellarator, ntor=6, mpol=5) |
| `test_bench_response_table_from_coils` | Magnetic field response table creation from coils file |
| `test_bench_free_boundary` | Free-boundary solve with pre-computed response table |

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

```bash
pip install -e .[benchmark]
pytest benchmarks/test_benchmarks.py -v
```

To produce a JSON report:

```bash
pytest benchmarks/test_benchmarks.py --benchmark-json=benchmark_results.json
```

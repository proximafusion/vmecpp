window.BENCHMARK_DATA = {
  "lastUpdate": 1774536031108,
  "repoUrl": "https://github.com/proximafusion/vmecpp",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "name": "Philipp Jurašić",
            "username": "jurasic-pf"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a5682f80e0acc586de0b0ae8f5b751dc12f89238",
          "message": "Fix auto benchmarks (#449)",
          "timestamp": "2026-03-26T15:33:28+01:00",
          "tree_id": "26c20171c6d12ed2706e72c307c69cdde94da07c",
          "url": "https://github.com/proximafusion/vmecpp/commit/a5682f80e0acc586de0b0ae8f5b751dc12f89238"
        },
        "date": 1774536030110,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.942191616692392,
            "unit": "iter/sec",
            "range": "stddev: 0.0012402140429368682",
            "extra": "mean: 339.88268960000596 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8985393001921063,
            "unit": "iter/sec",
            "range": "stddev: 0.0042896725521008134",
            "extra": "mean: 345.00135979999413 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27895326288408384,
            "unit": "iter/sec",
            "range": "stddev: 0.0820586075318222",
            "extra": "mean: 3.5848299089999878 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7173935145748672,
            "unit": "iter/sec",
            "range": "stddev: 0.034082260479289486",
            "extra": "mean: 1.393935099333324 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.27758115671229155,
            "unit": "iter/sec",
            "range": "stddev: 0.016730031139211843",
            "extra": "mean: 3.6025500139999926 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10571575002582835,
            "unit": "iter/sec",
            "range": "stddev: 0.005114991849708583",
            "extra": "mean: 9.459328432666666 sec\nrounds: 3"
          }
        ]
      }
    ]
  }
}
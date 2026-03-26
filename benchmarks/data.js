window.BENCHMARK_DATA = {
  "lastUpdate": 1774538140696,
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
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurasic",
            "username": "jurasic-pf",
            "email": "jurasic@proximafusion.com"
          },
          "committer": {
            "name": "Philipp Jurasic",
            "username": "jurasic-pf",
            "email": "jurasic@proximafusion.com"
          },
          "id": "7b4fd271e90c502a9c8342d5d6beab7ab604ef7d",
          "message": "Cleaner benchmark online overview",
          "timestamp": "2026-03-26T15:03:27Z",
          "url": "https://github.com/proximafusion/vmecpp/commit/7b4fd271e90c502a9c8342d5d6beab7ab604ef7d"
        },
        "date": 1774538138948,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.902307998733303,
            "unit": "iter/sec",
            "range": "stddev: 0.004562993686744183",
            "extra": "mean: 344.55336939995505 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8930144174535286,
            "unit": "iter/sec",
            "range": "stddev: 0.009243419840870824",
            "extra": "mean: 345.6602199999452 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.28532876722873546,
            "unit": "iter/sec",
            "range": "stddev: 0.04066073227389749",
            "extra": "mean: 3.5047289823333663 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.726978400065979,
            "unit": "iter/sec",
            "range": "stddev: 0.007429922474252053",
            "extra": "mean: 1.3755566876667065 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2786053711891718,
            "unit": "iter/sec",
            "range": "stddev: 0.009535161631149511",
            "extra": "mean: 3.589306249666682 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1045598743048834,
            "unit": "iter/sec",
            "range": "stddev: 0.016281502360103086",
            "extra": "mean: 9.56389826066667 sec\nrounds: 3"
          }
        ]
      }
    ]
  }
}
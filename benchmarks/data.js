window.BENCHMARK_DATA = {
  "lastUpdate": 1776330678315,
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
      },
      {
        "commit": {
          "author": {
            "email": "jurasic@proximafusion.com",
            "name": "jurasic-pf",
            "username": "jurasic-pf"
          },
          "committer": {
            "email": "jurasic@proximafusion.com",
            "name": "jurasic-pf",
            "username": "jurasic-pf"
          },
          "distinct": false,
          "id": "e1d17b9da212722e00f01fef1a5d36af4b8ce119",
          "message": "Ignore git worktrees for Claude (#455)",
          "timestamp": "2026-03-26T17:14:25Z",
          "tree_id": "42a875b6850c90124f283db079a1c159aa7463ff",
          "url": "https://github.com/proximafusion/vmecpp/commit/e1d17b9da212722e00f01fef1a5d36af4b8ce119"
        },
        "date": 1774546647026,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9158392681160503,
            "unit": "iter/sec",
            "range": "stddev: 0.003020167502226743",
            "extra": "mean: 342.9544319999877 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.929484098251977,
            "unit": "iter/sec",
            "range": "stddev: 0.0019279600571041184",
            "extra": "mean: 341.3570330000084 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.28223388143117967,
            "unit": "iter/sec",
            "range": "stddev: 0.10590272277185879",
            "extra": "mean: 3.5431607110000414 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7260095874560796,
            "unit": "iter/sec",
            "range": "stddev: 0.004671793006981951",
            "extra": "mean: 1.3773922786666997 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2781184843290902,
            "unit": "iter/sec",
            "range": "stddev: 0.01272217124798357",
            "extra": "mean: 3.595589852333319 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10553971961222995,
            "unit": "iter/sec",
            "range": "stddev: 0.028601622457251754",
            "extra": "mean: 9.475105710666677 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jurasic@proximafusion.com",
            "name": "jurasic-pf",
            "username": "jurasic-pf"
          },
          "committer": {
            "email": "jurasic@proximafusion.com",
            "name": "jurasic-pf",
            "username": "jurasic-pf"
          },
          "distinct": false,
          "id": "14c9c876477f74fc42a46a1e94a8fcd69afad6b4",
          "message": "high verbosity CI logs for docs were enabled for debugging, no longer needed. (#450)",
          "timestamp": "2026-03-26T17:14:25Z",
          "tree_id": "80c8918132fc1bc43f96c90db3159e62b8e8b1a6",
          "url": "https://github.com/proximafusion/vmecpp/commit/14c9c876477f74fc42a46a1e94a8fcd69afad6b4"
        },
        "date": 1774547127599,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9691844045760933,
            "unit": "iter/sec",
            "range": "stddev: 0.002731248080450877",
            "extra": "mean: 336.79282379996494 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9729036993351947,
            "unit": "iter/sec",
            "range": "stddev: 0.00047862880454434064",
            "extra": "mean: 336.37147419999565 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.28815865639401067,
            "unit": "iter/sec",
            "range": "stddev: 0.032224865380086794",
            "extra": "mean: 3.4703104620000054 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7219733952948133,
            "unit": "iter/sec",
            "range": "stddev: 0.014810607032413999",
            "extra": "mean: 1.3850925900000182 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.27901883897672713,
            "unit": "iter/sec",
            "range": "stddev: 0.008234796283750831",
            "extra": "mean: 3.583987388333336 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10612768316978004,
            "unit": "iter/sec",
            "range": "stddev: 0.01971315567664394",
            "extra": "mean: 9.422612179333347 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jurasic@proximafusion.com",
            "name": "jurasic-pf",
            "username": "jurasic-pf"
          },
          "committer": {
            "email": "jurasic@proximafusion.com",
            "name": "jurasic-pf",
            "username": "jurasic-pf"
          },
          "distinct": false,
          "id": "27a12a3ee727a764f58c759a333579c1557af738",
          "message": "Cleaner benchmark online overview (#451)\n\n![image.png](https://app.graphite.com/user-attachments/assets/9915494a-25e2-4f71-94d9-60bb50e243a5.png)\n\nNice online view for the benchmarks, inlined with the other docs.",
          "timestamp": "2026-03-26T17:37:22Z",
          "tree_id": "022a42e50d70191af020a2783e5aa82dc45cb9ca",
          "url": "https://github.com/proximafusion/vmecpp/commit/27a12a3ee727a764f58c759a333579c1557af738"
        },
        "date": 1774547582039,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0958689758405837,
            "unit": "iter/sec",
            "range": "stddev: 0.002711228228765889",
            "extra": "mean: 323.01108600000816 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.1266781037716904,
            "unit": "iter/sec",
            "range": "stddev: 0.002237279676083818",
            "extra": "mean: 319.8282543999994 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24770180788815604,
            "unit": "iter/sec",
            "range": "stddev: 0.01264086909322307",
            "extra": "mean: 4.0371122380000015 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6679750283155638,
            "unit": "iter/sec",
            "range": "stddev: 0.013847145550703563",
            "extra": "mean: 1.497061952333316 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.3040653410519669,
            "unit": "iter/sec",
            "range": "stddev: 0.019759694111681016",
            "extra": "mean: 3.2887668043333256 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.11334443818791759,
            "unit": "iter/sec",
            "range": "stddev: 0.011416062849415093",
            "extra": "mean: 8.82266493166666 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jurasic@proximafusion.com",
            "name": "jurasic-pf",
            "username": "jurasic-pf"
          },
          "committer": {
            "email": "jurasic@proximafusion.com",
            "name": "jurasic-pf",
            "username": "jurasic-pf"
          },
          "distinct": false,
          "id": "a304c1df8c7f1e53648449541dce0c291856b82e",
          "message": "Migrate FourierBasisFastPoloidal and FourierBasisFastToroidal to Eigen3 (#410)\n\nPart of the larger migration of `vmecpp` to Eigen datatypes instead of std::vectors.",
          "timestamp": "2026-03-26T23:59:51Z",
          "tree_id": "7cfcf9d3a35c7021e12bb29c7280017a59f39996",
          "url": "https://github.com/proximafusion/vmecpp/commit/a304c1df8c7f1e53648449541dce0c291856b82e"
        },
        "date": 1774570723028,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.910577815976096,
            "unit": "iter/sec",
            "range": "stddev: 0.0022402754904197438",
            "extra": "mean: 343.5743908000063 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9039757486336586,
            "unit": "iter/sec",
            "range": "stddev: 0.003010388550123228",
            "extra": "mean: 344.3554928000026 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2451544522594961,
            "unit": "iter/sec",
            "range": "stddev: 0.034814709304288045",
            "extra": "mean: 4.079061141999982 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5940994908332135,
            "unit": "iter/sec",
            "range": "stddev: 0.0007362544415360937",
            "extra": "mean: 1.6832197559999902 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2770968305588864,
            "unit": "iter/sec",
            "range": "stddev: 0.014761498826641357",
            "extra": "mean: 3.608846763000012 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10095290907287849,
            "unit": "iter/sec",
            "range": "stddev: 0.010964357285937431",
            "extra": "mean: 9.905608557333343 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "69142e7a02b5457b323fff1fc4512a1abf0df591",
          "message": "Vmec constructor -> Factory for error handling during initialization. (#244)\n\nTo get proper exception handling during file IO and mgrid construction, we should move to a factory pattern.\n\nBuilding a Factory function is complicated by the current situation with move/copy assignment operators, and the fact that we have const references in our objects.\n\nRevisit once this is resolved.\nhttps://github.com/proximafusion/vmecpp/issues/164",
          "timestamp": "2026-03-27T00:38:24Z",
          "tree_id": "d1c86ca8ebee63e151d20fc645b5bae901836a2d",
          "url": "https://github.com/proximafusion/vmecpp/commit/69142e7a02b5457b323fff1fc4512a1abf0df591"
        },
        "date": 1774572149725,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0007013029038445,
            "unit": "iter/sec",
            "range": "stddev: 0.0022316056287117175",
            "extra": "mean: 333.25542899997345 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9876823973533093,
            "unit": "iter/sec",
            "range": "stddev: 0.0008221596337784202",
            "extra": "mean: 334.707598399973 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2496837922475478,
            "unit": "iter/sec",
            "range": "stddev: 0.009670402162005456",
            "extra": "mean: 4.005065731333313 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.60246478101846,
            "unit": "iter/sec",
            "range": "stddev: 0.0051946245564453075",
            "extra": "mean: 1.6598480633332808 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.27820582237098607,
            "unit": "iter/sec",
            "range": "stddev: 0.01777270412715383",
            "extra": "mean: 3.5944610773332593 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10280459167241082,
            "unit": "iter/sec",
            "range": "stddev: 0.04241070451910032",
            "extra": "mean: 9.727191983666671 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "2d8888e1defbd4d5f2949047dfa742495854ca1f",
          "message": "SImsopt keep dependency chain even when resolution is inconsistent (#436)\n\nRelated to https://github.com/proximafusion/vmecpp/issues/427",
          "timestamp": "2026-03-27T00:57:22Z",
          "tree_id": "3b8c2e550b838b75d8fc95055d8f597c69aa717d",
          "url": "https://github.com/proximafusion/vmecpp/commit/2d8888e1defbd4d5f2949047dfa742495854ca1f"
        },
        "date": 1774573340955,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.931847132955567,
            "unit": "iter/sec",
            "range": "stddev: 0.004240172575010991",
            "extra": "mean: 341.0819032000177 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.925490089099671,
            "unit": "iter/sec",
            "range": "stddev: 0.003045399639364468",
            "extra": "mean: 341.82306879998805 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2453390218864563,
            "unit": "iter/sec",
            "range": "stddev: 0.028926613381809265",
            "extra": "mean: 4.075992446333316 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5993525732370025,
            "unit": "iter/sec",
            "range": "stddev: 0.0018944765137329516",
            "extra": "mean: 1.6684670169999738 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2784411905724834,
            "unit": "iter/sec",
            "range": "stddev: 0.00729835812165881",
            "extra": "mean: 3.591422655333323 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10185277045853065,
            "unit": "iter/sec",
            "range": "stddev: 0.02049071876904756",
            "extra": "mean: 9.818093268333334 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "1ba362029017a7ee7ffc3dc1c8ed77fb8e209a1a",
          "message": "Migrate ideal MHD core to eigen3 (#452)\n\n* Migrate ideal MHD core to eigen3\n\n* Update vmec.h",
          "timestamp": "2026-03-27T09:50:07+01:00",
          "tree_id": "1595f0d2dadf78d0732658ccb4881db013369167",
          "url": "https://github.com/proximafusion/vmecpp/commit/1ba362029017a7ee7ffc3dc1c8ed77fb8e209a1a"
        },
        "date": 1774601668169,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.7178737653457867,
            "unit": "iter/sec",
            "range": "stddev: 0.00840107494990392",
            "extra": "mean: 367.9346748000171 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.6887895423217336,
            "unit": "iter/sec",
            "range": "stddev: 0.007298024289776985",
            "extra": "mean: 371.9145676000039 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2373292138878116,
            "unit": "iter/sec",
            "range": "stddev: 0.07918133344068164",
            "extra": "mean: 4.213556281666665 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5918530338513334,
            "unit": "iter/sec",
            "range": "stddev: 0.018162485485199963",
            "extra": "mean: 1.6896086406666768 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2780440167664485,
            "unit": "iter/sec",
            "range": "stddev: 0.005889990560710803",
            "extra": "mean: 3.596552846666649 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10045897926824829,
            "unit": "iter/sec",
            "range": "stddev: 0.005505363364384548",
            "extra": "mean: 9.954311772666662 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "df0547f8fbd75d5c34354b4148de251a5f365491",
          "message": "Simplify a few more ideal mhd expressions with Eigen3 (#453)",
          "timestamp": "2026-03-27T09:07:38Z",
          "tree_id": "59491cd9d769f611dc4d46f48ef19e3d7f9b388e",
          "url": "https://github.com/proximafusion/vmecpp/commit/df0547f8fbd75d5c34354b4148de251a5f365491"
        },
        "date": 1774602711642,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9363142433776366,
            "unit": "iter/sec",
            "range": "stddev: 0.0030670384955502885",
            "extra": "mean: 340.56300420002117 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9666399263631305,
            "unit": "iter/sec",
            "range": "stddev: 0.0013017848504486636",
            "extra": "mean: 337.0816900000136 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24695621775621324,
            "unit": "iter/sec",
            "range": "stddev: 0.04328242196180794",
            "extra": "mean: 4.049300758999986 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6189751959039602,
            "unit": "iter/sec",
            "range": "stddev: 0.001504065503079652",
            "extra": "mean: 1.6155736233332998 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.27875389985881444,
            "unit": "iter/sec",
            "range": "stddev: 0.006598923725209605",
            "extra": "mean: 3.5873937566666805 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10232305071881385,
            "unit": "iter/sec",
            "range": "stddev: 0.06385182748735631",
            "extra": "mean: 9.772968974000037 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "159fcbc0380fd0f104d7a272ff18403b5ee8f6ad",
          "message": "Optimize hot loops further, redundant evaluations (#454)",
          "timestamp": "2026-03-27T10:09:00+01:00",
          "tree_id": "95f95268707df1057a1a620917b75aab8bdf025b",
          "url": "https://github.com/proximafusion/vmecpp/commit/159fcbc0380fd0f104d7a272ff18403b5ee8f6ad"
        },
        "date": 1774603184690,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.863631431968405,
            "unit": "iter/sec",
            "range": "stddev: 0.0010713457928530911",
            "extra": "mean: 349.2069506000007 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.854935932936242,
            "unit": "iter/sec",
            "range": "stddev: 0.0020320104247298553",
            "extra": "mean: 350.2705572000423 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23481132852028985,
            "unit": "iter/sec",
            "range": "stddev: 0.028596520222884194",
            "extra": "mean: 4.258738308333325 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5628878087678447,
            "unit": "iter/sec",
            "range": "stddev: 0.00006149865163903662",
            "extra": "mean: 1.7765529550000185 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2780605710427257,
            "unit": "iter/sec",
            "range": "stddev: 0.019172516253982787",
            "extra": "mean: 3.5963387266666587 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10019459872782738,
            "unit": "iter/sec",
            "range": "stddev: 0.009397925972032671",
            "extra": "mean: 9.98057792233332 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "c9181b56ab9c9f097eb576b58d8a86462659f748",
          "message": "Rename WOutFileContents scalar/1D fields to match Python VmecWOut (#442)\n\n(Expected CI failiures for the intermediate commits until PR #446, on large CPP tests)\n\nRename 22 fields in the C++ WOutFileContents struct to match the\nPython VmecWOut field names, eliminating name-mapping conversion\nlogic in _from_cpp_wout and _to_cpp_wout.\n\nKey renames: version->version_ (also string->double), sign_of_jacobian->signgs,\nmaximum_iterations->niter, betatot->betatotal, VolAvgB->volavgB,\nvolume_p->volume, iota_full->iotaf, safety_factor->q_factor,\npressure_full->presf, toroidal_flux->phi, poloidal_flux->chi,\nspectral_width->specw, Dshear->DShear, Dwell->DWell, Dcurr->DCurr,\nDgeod->DGeod, raxis_c->raxis_cc, zaxis_s->zaxis_cs, raxis_s->raxis_cs,\nzaxis_c->zaxis_cc, restart_reasons->restart_reason_timetrace.\n\n_CPP_WOUT_SPECIAL_HANDLING reduced from 56 to 34 entries.",
          "timestamp": "2026-03-27T11:10:05+01:00",
          "tree_id": "350559997c660245c118de00e7f79b90aa1a5712",
          "url": "https://github.com/proximafusion/vmecpp/commit/c9181b56ab9c9f097eb576b58d8a86462659f748"
        },
        "date": 1774606456940,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9099889634303953,
            "unit": "iter/sec",
            "range": "stddev: 0.0057497998317706375",
            "extra": "mean: 343.64391499999556 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.893320833377173,
            "unit": "iter/sec",
            "range": "stddev: 0.006417176634364154",
            "extra": "mean: 345.6236130000036 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23464622638151153,
            "unit": "iter/sec",
            "range": "stddev: 0.010664362216718971",
            "extra": "mean: 4.261734848333333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5721445868393417,
            "unit": "iter/sec",
            "range": "stddev: 0.0041427692167804045",
            "extra": "mean: 1.7478099469999886 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.27828500592554445,
            "unit": "iter/sec",
            "range": "stddev: 0.0065946328107481375",
            "extra": "mean: 3.5934383049999874 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10045047977627423,
            "unit": "iter/sec",
            "range": "stddev: 0.04737034148020323",
            "extra": "mean: 9.955154044333332 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "684058866cbb7969ca2e90bcb8c01b6388e3265c",
          "message": "Right-pad profile arrays in C++, add lrfp field, final cleanup (#445)\n\n* Right-pad profile arrays in C++, add lrfp field, final cleanup\n\n* Update __init__.py",
          "timestamp": "2026-03-27T11:17:46+01:00",
          "tree_id": "e091a056a071f389319fdb60afead4147c667238",
          "url": "https://github.com/proximafusion/vmecpp/commit/684058866cbb7969ca2e90bcb8c01b6388e3265c"
        },
        "date": 1774606939663,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9338594716623336,
            "unit": "iter/sec",
            "range": "stddev: 0.0012380790942025115",
            "extra": "mean: 340.8479546000194 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.873073252073852,
            "unit": "iter/sec",
            "range": "stddev: 0.004191656659944429",
            "extra": "mean: 348.0593470000031 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23432559561522262,
            "unit": "iter/sec",
            "range": "stddev: 0.08430126904711714",
            "extra": "mean: 4.267566235666645 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5706875871868932,
            "unit": "iter/sec",
            "range": "stddev: 0.002878398485235706",
            "extra": "mean: 1.7522722106666606 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2792164403054774,
            "unit": "iter/sec",
            "range": "stddev: 0.00864058926554246",
            "extra": "mean: 3.5814510023333432 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1011601204673713,
            "unit": "iter/sec",
            "range": "stddev: 0.004779294621861399",
            "extra": "mean: 9.885318397999981 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "16e7d0e3171a58d62b6fb12fe3e800e56eda11f8",
          "message": "Add HDF5 backwards compatibility for renamed/resized wout fields (#446)",
          "timestamp": "2026-03-27T11:21:40+01:00",
          "tree_id": "53fa7dee60337426e3269ec381c9c2296e887308",
          "url": "https://github.com/proximafusion/vmecpp/commit/16e7d0e3171a58d62b6fb12fe3e800e56eda11f8"
        },
        "date": 1774607405051,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.978227207776783,
            "unit": "iter/sec",
            "range": "stddev: 0.001249436165069204",
            "extra": "mean: 335.77021839998906 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9565086147001813,
            "unit": "iter/sec",
            "range": "stddev: 0.0020550348050118704",
            "extra": "mean: 338.2367956000053 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23095484423866391,
            "unit": "iter/sec",
            "range": "stddev: 0.008825581031545963",
            "extra": "mean: 4.329850726000018 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5693504149766464,
            "unit": "iter/sec",
            "range": "stddev: 0.00855285390420487",
            "extra": "mean: 1.7563875843333108 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2773846773564516,
            "unit": "iter/sec",
            "range": "stddev: 0.020582770688828432",
            "extra": "mean: 3.6051018013333 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10067706606872306,
            "unit": "iter/sec",
            "range": "stddev: 0.03527315159997423",
            "extra": "mean: 9.932748728666676 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "c81b6dd298d1937ed3eeac79e23d00a1e3234ea9",
          "message": "Pass through outer pydantic serialization contexts (#457)",
          "timestamp": "2026-03-27T21:51:03+01:00",
          "tree_id": "2fb6ae029d04b3512849a84ff0805dddc450076d",
          "url": "https://github.com/proximafusion/vmecpp/commit/c81b6dd298d1937ed3eeac79e23d00a1e3234ea9"
        },
        "date": 1774644918968,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.8253020421638175,
            "unit": "iter/sec",
            "range": "stddev: 0.004602533749929204",
            "extra": "mean: 353.9444580000122 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8129932970532647,
            "unit": "iter/sec",
            "range": "stddev: 0.004656936276401357",
            "extra": "mean: 355.49320400000397 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2299602955812068,
            "unit": "iter/sec",
            "range": "stddev: 0.015572207868512667",
            "extra": "mean: 4.3485767726666795 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5650108821934045,
            "unit": "iter/sec",
            "range": "stddev: 0.010380646706194572",
            "extra": "mean: 1.769877415666656 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2780397748635872,
            "unit": "iter/sec",
            "range": "stddev: 0.009065247048421562",
            "extra": "mean: 3.5966077173333324 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09976075801655974,
            "unit": "iter/sec",
            "range": "stddev: 0.018805242324691905",
            "extra": "mean: 10.023981572333335 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "3d6cbed0f25c1e5a296d1befb8b7ca15cca07997",
          "message": "[Bugfix] xm, xn should be double but were serialized as int (#459)\n\n[Bugfix] xm, xn should be double but were serialized as int. Add testcase to catch such regressions",
          "timestamp": "2026-03-28T16:13:28+01:00",
          "tree_id": "822a97757f797559878c6a25e283b92420d4c609",
          "url": "https://github.com/proximafusion/vmecpp/commit/3d6cbed0f25c1e5a296d1befb8b7ca15cca07997"
        },
        "date": 1774711061326,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.902297039941741,
            "unit": "iter/sec",
            "range": "stddev: 0.0028635108179312996",
            "extra": "mean: 344.55467039999235 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.898164035631086,
            "unit": "iter/sec",
            "range": "stddev: 0.0022002017203393604",
            "extra": "mean: 345.04603180000686 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2326732936241909,
            "unit": "iter/sec",
            "range": "stddev: 0.016620784680583364",
            "extra": "mean: 4.297871854666653 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.568839962139555,
            "unit": "iter/sec",
            "range": "stddev: 0.006151376999728751",
            "extra": "mean: 1.7579636920000137 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.27887178497336196,
            "unit": "iter/sec",
            "range": "stddev: 0.004313832157065651",
            "extra": "mean: 3.585877288000006 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10072246364169234,
            "unit": "iter/sec",
            "range": "stddev: 0.04847067795032895",
            "extra": "mean: 9.92827184566668 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "edo.alvarezr@gmail.com",
            "name": "EdoAlvarezR",
            "username": "EdoAlvarezR"
          },
          "committer": {
            "email": "130992531+jons-pf@users.noreply.github.com",
            "name": "Jonathan Schilling",
            "username": "jons-pf"
          },
          "distinct": true,
          "id": "279080900a2abedea6a3f478a1218321cd40b385",
          "message": "Automatic changes by pre-commit run --all-files",
          "timestamp": "2026-04-08T10:43:45+02:00",
          "tree_id": "0659fdd49f80b3d0af9f2ee548f1a7880f20869c",
          "url": "https://github.com/proximafusion/vmecpp/commit/279080900a2abedea6a3f478a1218321cd40b385"
        },
        "date": 1775638084435,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9302596984500178,
            "unit": "iter/sec",
            "range": "stddev: 0.0016102702073630404",
            "extra": "mean: 341.2666804000196 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.93016256004484,
            "unit": "iter/sec",
            "range": "stddev: 0.0013529417896189484",
            "extra": "mean: 341.2779938000085 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23137084676228722,
            "unit": "iter/sec",
            "range": "stddev: 0.04823497368246868",
            "extra": "mean: 4.32206569666666 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5702858496972092,
            "unit": "iter/sec",
            "range": "stddev: 0.006401314283780715",
            "extra": "mean: 1.7535065976666715 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.27911031627704347,
            "unit": "iter/sec",
            "range": "stddev: 0.007578489778930305",
            "extra": "mean: 3.582812750666676 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10129373958904249,
            "unit": "iter/sec",
            "range": "stddev: 0.005803871927889995",
            "extra": "mean: 9.87227842566665 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "130992531+jons-pf@users.noreply.github.com",
            "name": "Jonathan Schilling",
            "username": "jons-pf"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bb623a411e381b7a45c3905502ce63c2d6886b56",
          "message": "Implement the current density Fourier coefficients. (#474)",
          "timestamp": "2026-04-10T10:26:12Z",
          "tree_id": "27605f632f483ab8cb9a4a98f54cbd75d36dee9c",
          "url": "https://github.com/proximafusion/vmecpp/commit/bb623a411e381b7a45c3905502ce63c2d6886b56"
        },
        "date": 1775817020231,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.7474776222578354,
            "unit": "iter/sec",
            "range": "stddev: 0.0035124411644543246",
            "extra": "mean: 363.97020739998425 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.7147479807961377,
            "unit": "iter/sec",
            "range": "stddev: 0.003977850771133096",
            "extra": "mean: 368.35831800001415 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23076889890580482,
            "unit": "iter/sec",
            "range": "stddev: 0.034842879959478795",
            "extra": "mean: 4.33333956499996 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5611759204971993,
            "unit": "iter/sec",
            "range": "stddev: 0.012841338055319782",
            "extra": "mean: 1.7819723966666363 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2772501402377176,
            "unit": "iter/sec",
            "range": "stddev: 0.014743401853566063",
            "extra": "mean: 3.606851196333347 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.0998568255099256,
            "unit": "iter/sec",
            "range": "stddev: 0.0035784047592257186",
            "extra": "mean: 10.014337977333374 sec\nrounds: 3"
          }
        ]
      },
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
          "id": "d79e972b074dedc10976091e00e12f68551d5fb3",
          "message": "Restrict pydantic version, breaking change (#480)\n\npydantic 2.13 is causing validation failiures, restrict it until we understand the issue",
          "timestamp": "2026-04-16T11:06:55+02:00",
          "tree_id": "c081f0908ae5d478b1d3e6f9b2b2ae2457d276fb",
          "url": "https://github.com/proximafusion/vmecpp/commit/d79e972b074dedc10976091e00e12f68551d5fb3"
        },
        "date": 1776330677246,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.5069642200014006,
            "unit": "iter/sec",
            "range": "stddev: 0.007022561567535208",
            "extra": "mean: 398.8888202000112 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.5931517266232045,
            "unit": "iter/sec",
            "range": "stddev: 0.005103190661552017",
            "extra": "mean: 385.6311182000127 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2299238303108003,
            "unit": "iter/sec",
            "range": "stddev: 0.0373332178785665",
            "extra": "mean: 4.349266444666682 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5630004394083282,
            "unit": "iter/sec",
            "range": "stddev: 0.006764010403762522",
            "extra": "mean: 1.7761975480000085 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.277674342773856,
            "unit": "iter/sec",
            "range": "stddev: 0.003867373101805961",
            "extra": "mean: 3.6013410169999815 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09924296737695544,
            "unit": "iter/sec",
            "range": "stddev: 0.01469764029796653",
            "extra": "mean: 10.076280732333316 sec\nrounds: 3"
          }
        ]
      }
    ]
  }
}
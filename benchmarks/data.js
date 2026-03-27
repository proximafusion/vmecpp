window.BENCHMARK_DATA = {
  "lastUpdate": 1774601669039,
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
      }
    ]
  }
}
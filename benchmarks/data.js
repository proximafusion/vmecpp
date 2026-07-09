window.BENCHMARK_DATA = {
  "lastUpdate": 1783604631967,
  "repoUrl": "https://github.com/proximafusion/vmecpp",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c798c70845a8ad5b5d848abbe249278f6c97f753",
          "message": "Add continuous benchmarking (#426)",
          "timestamp": "2026-02-25T13:44:39+01:00",
          "tree_id": "dc0f6863d3a544bc3fed16e17e42dcebcbef34b3",
          "url": "https://github.com/proximafusion/vmecpp/commit/c798c70845a8ad5b5d848abbe249278f6c97f753"
        },
        "date": 1783604247859,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3804127304000303,
            "range": "stddev: 0.012297982947006252",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3779994149999311,
            "range": "stddev: 0.015403018005997426",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.4173382336666691,
            "range": "stddev: 0.006461779444669496",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.2708967629999584,
            "range": "stddev: 0.027928454969843107",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8910479266667153,
            "range": "stddev: 0.034332397793565395",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.682315444999935,
            "range": "stddev: 0.05340215561138927",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5476218163334277,
            "range": "stddev: 0.00916379525480342",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "eguiraud-pf",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "eguiraud-pf",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c125adfa5f46fe55ce9e0183576bbaeeb11387c3",
          "message": "Add code owners file (#428)",
          "timestamp": "2026-03-09T10:22:20Z",
          "tree_id": "0a89cb0e32c0ae9230b9f02283da421a2a3341e0",
          "url": "https://github.com/proximafusion/vmecpp/commit/c125adfa5f46fe55ce9e0183576bbaeeb11387c3"
        },
        "date": 1783604247851,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3514097108000442,
            "range": "stddev: 0.0029679205687764294",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34866736499998296,
            "range": "stddev: 0.0007627785036230066",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.4130694299999504,
            "range": "stddev: 0.0030998189320737483",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.253605028000038,
            "range": "stddev: 0.004412120620249038",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.838018288999971,
            "range": "stddev: 0.007768249418966232",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.696578128999969,
            "range": "stddev: 0.04312428107989624",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5544929509999292,
            "range": "stddev: 0.008849328798300065",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "660fa04e463483d97bca326a6ef28174924dee41",
          "message": "hotfix pre-commit on Arch: use Python 3.10 (not 3.14) (#432)",
          "timestamp": "2026-03-11T18:46:04Z",
          "tree_id": "7ff8697b6377fbbfa65b2654714f0818cfbd2ae4",
          "url": "https://github.com/proximafusion/vmecpp/commit/660fa04e463483d97bca326a6ef28174924dee41"
        },
        "date": 1783604247727,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3870300542000223,
            "range": "stddev: 0.00540979579773853",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3743833664000249,
            "range": "stddev: 0.013894916039983662",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.42758129799995,
            "range": "stddev: 0.013062758094166845",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.268885725333272,
            "range": "stddev: 0.03680455801231114",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.869074308666692,
            "range": "stddev: 0.026243380045135725",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.684823200999896,
            "range": "stddev: 0.028816978967411526",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.55450604699998,
            "range": "stddev: 0.008561022055058947",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "07b8a0f980019f295ebfdb15a6681704bc58a30c",
          "message": "Add stand-alone makegrid executable (#431)",
          "timestamp": "2026-03-11T18:57:23Z",
          "tree_id": "0cbc84d805949211ff1cb9f082a3702d6f73341d",
          "url": "https://github.com/proximafusion/vmecpp/commit/07b8a0f980019f295ebfdb15a6681704bc58a30c"
        },
        "date": 1783604247637,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3643721124000422,
            "range": "stddev: 0.023325839205993824",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.35397316299995507,
            "range": "stddev: 0.005905050035861718",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.4153924706665748,
            "range": "stddev: 0.005349315576847951",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.255754037333342,
            "range": "stddev: 0.009296489011894058",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.877499781333275,
            "range": "stddev: 0.02588745506234584",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.646581489666687,
            "range": "stddev: 0.013086590970827375",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5722759089999272,
            "range": "stddev: 0.025726187434563126",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9e57835a6772309ef0d07ba35fdedbbfd915ea59",
          "message": "ignore .cache (#430)",
          "timestamp": "2026-03-11T18:57:23Z",
          "tree_id": "62d5ab9148dafc000a305f8eb9983ec3fc8e4da8",
          "url": "https://github.com/proximafusion/vmecpp/commit/9e57835a6772309ef0d07ba35fdedbbfd915ea59"
        },
        "date": 1783604247809,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35751364640004796,
            "range": "stddev: 0.013227948294767002",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3730073872000048,
            "range": "stddev: 0.01245578510408214",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.4207697043333003,
            "range": "stddev: 0.000605517534465868",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.289545655999973,
            "range": "stddev: 0.08687554341893572",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.892848983999935,
            "range": "stddev: 0.03542991984352223",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.564281870333312,
            "range": "stddev: 0.083600132007938",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5496539436667263,
            "range": "stddev: 0.008963668876954716",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jungdaesuh",
            "email": "78460559+jungdaesuh@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "e755dc32c9ce57a0c85c6c7f7fe6f10cf4ccf84a",
          "message": "Preserve assigned boundary during run (#429)",
          "timestamp": "2026-03-12T07:25:59+09:00",
          "tree_id": "5fe728871e4ab55bd1f1b9748adfb5bd6a9b28cc",
          "url": "https://github.com/proximafusion/vmecpp/commit/e755dc32c9ce57a0c85c6c7f7fe6f10cf4ccf84a"
        },
        "date": 1783604247920,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35274824080001965,
            "range": "stddev: 0.009968714614879923",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.38048238420001324,
            "range": "stddev: 0.010374145637289629",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.424282334666638,
            "range": "stddev: 0.00573040915469908",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.2451902036665765,
            "range": "stddev: 0.009850280041980543",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.89389819433336,
            "range": "stddev: 0.011024581700344078",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.678224729999783,
            "range": "stddev: 0.028481151650667225",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.560021613999955,
            "range": "stddev: 0.011955238834021015",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c8a471ba878f4b3f92db1bdcd6228b5f86698a31",
          "message": "Add only_coils-related cases in indata tests (#433)",
          "timestamp": "2026-03-11T22:50:42Z",
          "tree_id": "fd642d74d332af5a203c80b5f2a6e72b40ef4c4a",
          "url": "https://github.com/proximafusion/vmecpp/commit/c8a471ba878f4b3f92db1bdcd6228b5f86698a31"
        },
        "date": 1783604247864,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.36627380079999056,
            "range": "stddev: 0.0012463139715390976",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3585645965999902,
            "range": "stddev: 0.0020754736064511037",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.418047384999909,
            "range": "stddev: 0.005209441586963816",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.2724064690000128,
            "range": "stddev: 0.055368490717586476",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8681143433333696,
            "range": "stddev: 0.022876201578281862",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.686925556666742,
            "range": "stddev: 0.016632755423324402",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5516614216666464,
            "range": "stddev: 0.009420964319179775",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cf64eaaeafae5ff91dd8e90a2f538732812be877",
          "message": "Fix ci error in pyright.yaml (#440)",
          "timestamp": "2026-03-18T14:31:35+01:00",
          "tree_id": "922e038285fc3406734ee9fca6b80eb396c0e453",
          "url": "https://github.com/proximafusion/vmecpp/commit/cf64eaaeafae5ff91dd8e90a2f538732812be877"
        },
        "date": 1783604247891,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35955951119995005,
            "range": "stddev: 0.0030645753599391995",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3623730333999447,
            "range": "stddev: 0.002656769353527634",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.4229323056667151,
            "range": "stddev: 0.0025437303760654797",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.2845438266666256,
            "range": "stddev: 0.03711280182908252",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8756794490000934,
            "range": "stddev: 0.04274011516265186",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.696921121999972,
            "range": "stddev: 0.11169091031228107",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5528314756666077,
            "range": "stddev: 0.013094018454744323",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jungdaesuh",
            "email": "78460559+jungdaesuh@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "40426da882d554bfa1da117458a424d564a53957",
          "message": "Write back asymmetric boundary coefficients (#439)",
          "timestamp": "2026-03-18T23:48:56+09:00",
          "tree_id": "8f738e16be93c53bc11f563ef75ed996e9a48a7f",
          "url": "https://github.com/proximafusion/vmecpp/commit/40426da882d554bfa1da117458a424d564a53957"
        },
        "date": 1783604247694,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35229050640000426,
            "range": "stddev: 0.0041837613964047235",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3518250810000154,
            "range": "stddev: 0.0035619437141165846",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.416833608999923,
            "range": "stddev: 0.0007514887003177135",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.267324664333349,
            "range": "stddev: 0.014248999090283918",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.845194559666652,
            "range": "stddev: 0.0191758007751769",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.618819106333452,
            "range": "stddev: 0.009560332116372512",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5659085603333551,
            "range": "stddev: 0.019560089830312245",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8856974a7b20bccb2a2369c40e44ed5cb610bbd5",
          "message": "Globally prevent 3.14 python (#435)",
          "timestamp": "2026-03-18T16:06:34+01:00",
          "tree_id": "07d365c528c63f38b69a3f0811dd98318c653934",
          "url": "https://github.com/proximafusion/vmecpp/commit/8856974a7b20bccb2a2369c40e44ed5cb610bbd5"
        },
        "date": 1783604247770,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.354021843999999,
            "range": "stddev: 0.001372170248863942",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34921128320002026,
            "range": "stddev: 0.002453747619466562",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.4164703906667455,
            "range": "stddev: 0.007543468814718891",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.2734773136666413,
            "range": "stddev: 0.012605148079267612",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.85673725933331,
            "range": "stddev: 0.014720021856340838",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.610129715666593,
            "range": "stddev: 0.025517986866532966",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5501767879999686,
            "range": "stddev: 0.009421853422065534",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "827c4b0ed5d4c5c4305b79e25b000291ebfe0413",
          "message": "DNDEBUG for faster eigen3 performance (#447)",
          "timestamp": "2026-03-26T14:50:12+01:00",
          "tree_id": "bdac70f73639a81791d026c842d96a260943c70e",
          "url": "https://github.com/proximafusion/vmecpp/commit/827c4b0ed5d4c5c4305b79e25b000291ebfe0413"
        },
        "date": 1783604247765,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3537024102000032,
            "range": "stddev: 0.004036427154070721",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.35513883439998606,
            "range": "stddev: 0.004367827626405494",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3994507136666623,
            "range": "stddev: 0.008269441978242364",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.250712181333256,
            "range": "stddev: 0.026022135882950753",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.641733084666612,
            "range": "stddev: 0.021288977187554577",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.518989577000033,
            "range": "stddev: 0.024460392385308873",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5541425023335858,
            "range": "stddev: 0.009571293928935212",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5c527e1ec36b148acecd0cf59dc2f8418f6bf791",
          "message": "apt get update in docs CI (#448)",
          "timestamp": "2026-03-26T15:32:42+01:00",
          "tree_id": "25ec0c21ffc071b90ef07bd6beb8f7c6f95541ef",
          "url": "https://github.com/proximafusion/vmecpp/commit/5c527e1ec36b148acecd0cf59dc2f8418f6bf791"
        },
        "date": 1783604247719,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3503321686000163,
            "range": "stddev: 0.002525737763020919",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3477617843999724,
            "range": "stddev: 0.0009552185289847676",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3942412486665792,
            "range": "stddev: 0.003434063067600643",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.206737957666519,
            "range": "stddev: 0.01192485306576418",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.6386525466667385,
            "range": "stddev: 0.009269415550872436",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.451066632666576,
            "range": "stddev: 0.03874020395279849",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5615696729999702,
            "range": "stddev: 0.010864891506571672",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a5682f80e0acc586de0b0ae8f5b751dc12f89238",
          "message": "Fix auto benchmarks (#449)",
          "timestamp": "2026-03-26T15:33:28+01:00",
          "tree_id": "26c20171c6d12ed2706e72c307c69cdde94da07c",
          "url": "https://github.com/proximafusion/vmecpp/commit/a5682f80e0acc586de0b0ae8f5b751dc12f89238"
        },
        "date": 1783604247824,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3425879535999229,
            "range": "stddev: 0.0024893035945333696",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3418394945999353,
            "range": "stddev: 0.0021334081705652944",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.38536056166663,
            "range": "stddev: 0.002226319018872906",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.2299386126666527,
            "range": "stddev: 0.04750200661561449",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.6088625899999442,
            "range": "stddev: 0.026258667436399468",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.438725542999919,
            "range": "stddev: 0.02036830345708905",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5501808110000184,
            "range": "stddev: 0.011656220807177289",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "e1d17b9da212722e00f01fef1a5d36af4b8ce119",
          "message": "Ignore git worktrees for Claude (#455)",
          "timestamp": "2026-03-26T17:14:25Z",
          "tree_id": "42a875b6850c90124f283db079a1c159aa7463ff",
          "url": "https://github.com/proximafusion/vmecpp/commit/e1d17b9da212722e00f01fef1a5d36af4b8ce119"
        },
        "date": 1783604247917,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34545544379998316,
            "range": "stddev: 0.006994291257834655",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34472269339999,
            "range": "stddev: 0.004992482959670683",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3869061763333168,
            "range": "stddev: 0.0070211347757148735",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.2314515803333657,
            "range": "stddev: 0.031178364098798702",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.6116414439999667,
            "range": "stddev: 0.02109104653744289",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.371925265999986,
            "range": "stddev: 0.04371162532415942",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5511200343332803,
            "range": "stddev: 0.012252848351352761",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "14c9c876477f74fc42a46a1e94a8fcd69afad6b4",
          "message": "high verbosity CI logs for docs were enabled for debugging, no longer needed. (#450)",
          "timestamp": "2026-03-26T17:14:25Z",
          "tree_id": "80c8918132fc1bc43f96c90db3159e62b8e8b1a6",
          "url": "https://github.com/proximafusion/vmecpp/commit/14c9c876477f74fc42a46a1e94a8fcd69afad6b4"
        },
        "date": 1783604247655,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34394017220001843,
            "range": "stddev: 0.0016871815337238206",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3433981822000078,
            "range": "stddev: 0.0016102891614947826",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.381501286666662,
            "range": "stddev: 0.002636480006084772",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.23413130066668,
            "range": "stddev: 0.020746990476720427",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.611337965333329,
            "range": "stddev: 0.01729159529622167",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.442174256333374,
            "range": "stddev: 0.012842700527564997",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5483477370000098,
            "range": "stddev: 0.011804603370155204",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "27a12a3ee727a764f58c759a333579c1557af738",
          "message": "Cleaner benchmark online overview (#451)",
          "timestamp": "2026-03-26T17:37:22Z",
          "tree_id": "022a42e50d70191af020a2783e5aa82dc45cb9ca",
          "url": "https://github.com/proximafusion/vmecpp/commit/27a12a3ee727a764f58c759a333579c1557af738"
        },
        "date": 1783604247672,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3457825147999756,
            "range": "stddev: 0.003818300635431751",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34371402099995974,
            "range": "stddev: 0.004103762469725922",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3853941039999427,
            "range": "stddev: 0.002586658063376587",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.246547894999973,
            "range": "stddev: 0.03268739796306055",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.6008077649999755,
            "range": "stddev: 0.02506860604645364",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.446259116333295,
            "range": "stddev: 0.0027593312665962783",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5484023936666442,
            "range": "stddev: 0.010264391999743992",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "a304c1df8c7f1e53648449541dce0c291856b82e",
          "message": "Migrate FourierBasisFastPoloidal and FourierBasisFastToroidal to Eigen3 (#410)",
          "timestamp": "2026-03-26T23:59:51Z",
          "tree_id": "7cfcf9d3a35c7021e12bb29c7280017a59f39996",
          "url": "https://github.com/proximafusion/vmecpp/commit/a304c1df8c7f1e53648449541dce0c291856b82e"
        },
        "date": 1783604247822,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34374799259999234,
            "range": "stddev: 0.001959117459969092",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3422135578000507,
            "range": "stddev: 0.0007829230870701179",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6901022786666242,
            "range": "stddev: 0.0020270338227641933",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6055618423333726,
            "range": "stddev: 0.011443608741498218",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.179520299000008,
            "range": "stddev: 0.026844920873677288",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.800229640666734,
            "range": "stddev: 0.019902456034534877",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.552354080666646,
            "range": "stddev: 0.013724385716513951",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "69142e7a02b5457b323fff1fc4512a1abf0df591",
          "message": "Vmec constructor -> Factory for error handling during initialization. (#244)",
          "timestamp": "2026-03-27T01:38:24+01:00",
          "tree_id": "d1c86ca8ebee63e151d20fc645b5bae901836a2d",
          "url": "https://github.com/proximafusion/vmecpp/commit/69142e7a02b5457b323fff1fc4512a1abf0df591"
        },
        "date": 1783604247734,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3463995860000068,
            "range": "stddev: 0.0017766854219023202",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3443981330000042,
            "range": "stddev: 0.0032587592712776314",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.67803793566668,
            "range": "stddev: 0.005539639425972589",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6224424343333035,
            "range": "stddev: 0.016369838953687155",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.170236897999909,
            "range": "stddev: 0.023755674039310126",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.787552515000016,
            "range": "stddev: 0.018007458888602286",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5454099413332945,
            "range": "stddev: 0.01374287273249687",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "2d8888e1defbd4d5f2949047dfa742495854ca1f",
          "message": "SImsopt keep dependency chain even when resolution is inconsistent (#436)",
          "timestamp": "2026-03-27T01:57:22+01:00",
          "tree_id": "3b8c2e550b838b75d8fc95055d8f597c69aa717d",
          "url": "https://github.com/proximafusion/vmecpp/commit/2d8888e1defbd4d5f2949047dfa742495854ca1f"
        },
        "date": 1783604247675,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34116057879991785,
            "range": "stddev: 0.0009230194839172546",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3402538392000679,
            "range": "stddev: 0.0013596497206527023",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6759533333332304,
            "range": "stddev: 0.0028184301072773036",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6578330759998607,
            "range": "stddev: 0.0054886941155622825",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.185340312333362,
            "range": "stddev: 0.03382989411607352",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.693031936000049,
            "range": "stddev: 0.08309419350011286",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5444252676667627,
            "range": "stddev: 0.011063101646755835",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "1ba362029017a7ee7ffc3dc1c8ed77fb8e209a1a",
          "message": "Migrate ideal MHD core to eigen3 (#452)",
          "timestamp": "2026-03-27T09:50:07+01:00",
          "tree_id": "1595f0d2dadf78d0732658ccb4881db013369167",
          "url": "https://github.com/proximafusion/vmecpp/commit/1ba362029017a7ee7ffc3dc1c8ed77fb8e209a1a"
        },
        "date": 1783604247665,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3406983161999051,
            "range": "stddev: 0.001784790427885504",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3396449410000969,
            "range": "stddev: 0.002692482707256359",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6850875716666753,
            "range": "stddev: 0.0012145930840919044",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.662688615333385,
            "range": "stddev: 0.023913040596178676",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.254817894999936,
            "range": "stddev: 0.016777191493231574",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.740202664999893,
            "range": "stddev: 0.000587907132426864",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5508650879999852,
            "range": "stddev: 0.009748233252699535",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "df0547f8fbd75d5c34354b4148de251a5f365491",
          "message": "Simplify a few more ideal mhd expressions with Eigen3 (#453)",
          "timestamp": "2026-03-27T10:07:38+01:00",
          "tree_id": "59491cd9d769f611dc4d46f48ef19e3d7f9b388e",
          "url": "https://github.com/proximafusion/vmecpp/commit/df0547f8fbd75d5c34354b4148de251a5f365491"
        },
        "date": 1783604247912,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3415748987999905,
            "range": "stddev: 0.0028883386855863206",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3401117627999156,
            "range": "stddev: 0.0018606177389522674",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6269435073333323,
            "range": "stddev: 0.007440481057334383",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6185461589999854,
            "range": "stddev: 0.04145962510952665",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.1953809416666745,
            "range": "stddev: 0.0376356909800945",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.634740211333488,
            "range": "stddev: 0.00341577767675961",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5463320059999812,
            "range": "stddev: 0.013691751552702889",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "159fcbc0380fd0f104d7a272ff18403b5ee8f6ad",
          "message": "Optimize hot loops further, redundant evaluations (#454)",
          "timestamp": "2026-03-27T10:09:00+01:00",
          "tree_id": "95f95268707df1057a1a620917b75aab8bdf025b",
          "url": "https://github.com/proximafusion/vmecpp/commit/159fcbc0380fd0f104d7a272ff18403b5ee8f6ad"
        },
        "date": 1783604247657,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34027762220002844,
            "range": "stddev: 0.0018633840224073727",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3407645196000885,
            "range": "stddev: 0.0010345567211468849",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7887830593334304,
            "range": "stddev: 0.005214515411884262",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.677221241333276,
            "range": "stddev: 0.029729623063935357",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.422498679666508,
            "range": "stddev: 0.008533681623292216",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.903646579666656,
            "range": "stddev: 0.010697100179276237",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5466277506667818,
            "range": "stddev: 0.014418019711672319",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c9181b56ab9c9f097eb576b58d8a86462659f748",
          "message": "Rename WOutFileContents scalar/1D fields to match Python VmecWOut (#442)",
          "timestamp": "2026-03-27T11:10:05+01:00",
          "tree_id": "350559997c660245c118de00e7f79b90aa1a5712",
          "url": "https://github.com/proximafusion/vmecpp/commit/c9181b56ab9c9f097eb576b58d8a86462659f748"
        },
        "date": 1783604247869,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3449329275999844,
            "range": "stddev: 0.0016384531108364162",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.341707805400074,
            "range": "stddev: 0.001679267010196436",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7501637636664782,
            "range": "stddev: 0.0019306929140504201",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6821584833332963,
            "range": "stddev: 0.0048510550641453105",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.423305349666559,
            "range": "stddev: 0.03728982595372502",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.92025108233338,
            "range": "stddev: 0.0561087456650638",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5458438336666707,
            "range": "stddev: 0.01202277726247177",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0e531fae6fa57f52c0a52d8400ff9c0e0c76ca0b",
          "message": "Rename and pad half-grid 1D arrays to match Python VmecWOut (#443)",
          "timestamp": "2026-03-27T11:14:09+01:00",
          "tree_id": "a015fbee1637174e551bd2cae6554899ad72e0f6",
          "url": "https://github.com/proximafusion/vmecpp/commit/0e531fae6fa57f52c0a52d8400ff9c0e0c76ca0b"
        },
        "date": 1783604247642,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34393341219983997,
            "range": "stddev: 0.0012264340721555556",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34764108119998127,
            "range": "stddev: 0.004934687913671468",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7721083403331856,
            "range": "stddev: 0.007898383194730007",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.7161133473332484,
            "range": "stddev: 0.027479756222609737",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.444320603000051,
            "range": "stddev: 0.022593848985631104",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.918765741666752,
            "range": "stddev: 0.024187232570254187",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5682886206665596,
            "range": "stddev: 0.023760438535418718",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ab4e491bed0ce79324ae8112abf1f8424b9342ed",
          "message": "Transpose 2D Fourier arrays to (mnmax, ns) layout (#444)",
          "timestamp": "2026-03-27T11:15:18+01:00",
          "tree_id": "c9712b333516c57c9f32b284646dd9145bc5c50c",
          "url": "https://github.com/proximafusion/vmecpp/commit/ab4e491bed0ce79324ae8112abf1f8424b9342ed"
        },
        "date": 1783604247832,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34627686580001865,
            "range": "stddev: 0.004336771366571061",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3477997659999801,
            "range": "stddev: 0.0035887346306731696",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.763799304666615,
            "range": "stddev: 0.004661746493027807",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.69988315166673,
            "range": "stddev: 0.052893502748992",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.435051566666668,
            "range": "stddev: 0.026928216678735153",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.89762493699997,
            "range": "stddev: 0.04889334824569403",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5513566009999522,
            "range": "stddev: 0.0075537560546080715",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "684058866cbb7969ca2e90bcb8c01b6388e3265c",
          "message": "Right-pad profile arrays in C++, add lrfp field, final cleanup (#445)",
          "timestamp": "2026-03-27T11:17:46+01:00",
          "tree_id": "e091a056a071f389319fdb60afead4147c667238",
          "url": "https://github.com/proximafusion/vmecpp/commit/684058866cbb7969ca2e90bcb8c01b6388e3265c"
        },
        "date": 1783604247729,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3401575355999739,
            "range": "stddev: 0.0009331209581921033",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.33946072259996074,
            "range": "stddev: 0.0011169738787739753",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7543826146666106,
            "range": "stddev: 0.0032533323891807686",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6737246953333247,
            "range": "stddev: 0.009012536302678795",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.449955996999961,
            "range": "stddev: 0.014155875868414027",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.965716088999974,
            "range": "stddev: 0.009786016069420842",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5542130966666718,
            "range": "stddev: 0.015154235101290226",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "16e7d0e3171a58d62b6fb12fe3e800e56eda11f8",
          "message": "Add HDF5 backwards compatibility for renamed/resized wout fields (#446)",
          "timestamp": "2026-03-27T11:21:40+01:00",
          "tree_id": "53fa7dee60337426e3269ec381c9c2296e887308",
          "url": "https://github.com/proximafusion/vmecpp/commit/16e7d0e3171a58d62b6fb12fe3e800e56eda11f8"
        },
        "date": 1783604247660,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3403345372000331,
            "range": "stddev: 0.0011888872750560441",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3428147391999573,
            "range": "stddev: 0.0018781087080609034",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.8156511883333526,
            "range": "stddev: 0.0020993160155525627",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.762314609666646,
            "range": "stddev: 0.01932861857037906",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.588572392333238,
            "range": "stddev: 0.05659652471624497",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.96150817866669,
            "range": "stddev: 0.07631274559435086",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5585838913333039,
            "range": "stddev: 0.004606924970796606",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c81b6dd298d1937ed3eeac79e23d00a1e3234ea9",
          "message": "Pass through outer pydantic serialization contexts (#457)",
          "timestamp": "2026-03-27T21:51:03+01:00",
          "tree_id": "2fb6ae029d04b3512849a84ff0805dddc450076d",
          "url": "https://github.com/proximafusion/vmecpp/commit/c81b6dd298d1937ed3eeac79e23d00a1e3234ea9"
        },
        "date": 1783604247861,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3415320382000573,
            "range": "stddev: 0.0012484061584300227",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.340229739400047,
            "range": "stddev: 0.0008911884051481143",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7933036099999906,
            "range": "stddev: 0.0032299837646690647",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.7604639483333813,
            "range": "stddev: 0.01709069461803831",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.542803759000132,
            "range": "stddev: 0.020748453191731263",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.942859782000065,
            "range": "stddev: 0.08231702866717794",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5513746609999544,
            "range": "stddev: 0.014871880809219308",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "3d6cbed0f25c1e5a296d1befb8b7ca15cca07997",
          "message": "[Bugfix] xm, xn should be double but were serialized as int (#459)",
          "timestamp": "2026-03-28T16:13:28+01:00",
          "tree_id": "822a97757f797559878c6a25e283b92420d4c609",
          "url": "https://github.com/proximafusion/vmecpp/commit/3d6cbed0f25c1e5a296d1befb8b7ca15cca07997"
        },
        "date": 1783604247689,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.350037219800015,
            "range": "stddev: 0.003062707373938375",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34866086399997587,
            "range": "stddev: 0.0039019361614466247",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.8144889376666622,
            "range": "stddev: 0.062406487445138666",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.737547619666581,
            "range": "stddev: 0.056200592506070686",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.544186877000054,
            "range": "stddev: 0.014516394941775366",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.922449600333342,
            "range": "stddev: 0.021175231520188863",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.549034989000044,
            "range": "stddev: 0.00935588186510566",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8f8383a4aef70c8dbde420ecbcc7853b25db06ff",
          "message": "Port calculate_magnetic_field into example",
          "timestamp": "2026-04-06T09:31:46-05:00",
          "tree_id": "a2d3875f78fe4375d9830a215540823e4ab554bd",
          "url": "https://github.com/proximafusion/vmecpp/commit/8f8383a4aef70c8dbde420ecbcc7853b25db06ff"
        },
        "date": 1783604247785,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3502240382000309,
            "range": "stddev: 0.001428939801889732",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34832052319998186,
            "range": "stddev: 0.0008663746320499146",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.771779969666568,
            "range": "stddev: 0.005235323853640703",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.7026615083332217,
            "range": "stddev: 0.028310981757830773",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.542600911333314,
            "range": "stddev: 0.027466802287493427",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.918068820000068,
            "range": "stddev: 0.03431422628002521",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.554739292666606,
            "range": "stddev: 0.008362931239229663",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "1dd86d0442c7be39d04adf32829ca1a4c130c07a",
          "message": "Constructing magnetic field over a flux surface",
          "timestamp": "2026-04-06T09:36:39-05:00",
          "tree_id": "224725e2b7b592bc4ddd6850cc09d8a5ebc29d3a",
          "url": "https://github.com/proximafusion/vmecpp/commit/1dd86d0442c7be39d04adf32829ca1a4c130c07a"
        },
        "date": 1783604247667,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.26928929359992254,
            "range": "stddev: 0.001433511949230608",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.26794645760001,
            "range": "stddev: 0.003151165258709685",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3279450399999557,
            "range": "stddev: 0.003823382214775648",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.066418851666716,
            "range": "stddev: 0.030697046365091643",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.4976569976665814,
            "range": "stddev: 0.0017858831948027966",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 7.136915215666629,
            "range": "stddev: 0.04686185321927313",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.1775414793333614,
            "range": "stddev: 0.012520858702233401",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cce98fe2a193219e631dbf0e78ad8d948c7d74ed",
          "message": "Visualize with pyVista",
          "timestamp": "2026-04-06T09:46:03-05:00",
          "tree_id": "a9c11fa677dc527a7d905a9639d6c593e2fae463",
          "url": "https://github.com/proximafusion/vmecpp/commit/cce98fe2a193219e631dbf0e78ad8d948c7d74ed"
        },
        "date": 1783604247883,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.2776955161999922,
            "range": "stddev: 0.0005561222917487482",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.2755188439999984,
            "range": "stddev: 0.0022629927045652687",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3584366619999553,
            "range": "stddev: 0.005748148278063028",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.1459400223332827,
            "range": "stddev: 0.05666042177105466",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.6338439903333133,
            "range": "stddev: 0.034089266693772484",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 7.302031949333316,
            "range": "stddev: 0.1710806114882748",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.1863816960000502,
            "range": "stddev: 0.011606636859754672",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "52f46df618511ba29fc7e1823f8f3d7c4c77b036",
          "message": "Attempt to construct current density field",
          "timestamp": "2026-04-06T10:15:06-05:00",
          "tree_id": "8f1060b34cdeefd2ad8122f35b042d036d5dc016",
          "url": "https://github.com/proximafusion/vmecpp/commit/52f46df618511ba29fc7e1823f8f3d7c4c77b036"
        },
        "date": 1783604247712,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.2718072176000078,
            "range": "stddev: 0.002460687463413686",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.2755125294000663,
            "range": "stddev: 0.0022287922510116007",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3456808049998774,
            "range": "stddev: 0.0033602259235292788",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.1198096593333653,
            "range": "stddev: 0.07441247634130745",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.554003204333488,
            "range": "stddev: 0.021611135378675084",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 7.097713333333256,
            "range": "stddev: 0.034533238208421295",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.1737543840001006,
            "range": "stddev: 0.0072973881460710625",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a079cd5f6ec56873e5ea97a9920fd51947bb3122",
          "message": "Add visualize_magnetic_field.py to index of examples",
          "timestamp": "2026-04-06T10:24:37-05:00",
          "tree_id": "68ab4f0aae4a221c4f477dafcc0de9d9c2d2b035",
          "url": "https://github.com/proximafusion/vmecpp/commit/a079cd5f6ec56873e5ea97a9920fd51947bb3122"
        },
        "date": 1783604247811,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.27374822179981495,
            "range": "stddev: 0.0025632274173665154",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.2763728498001001,
            "range": "stddev: 0.004109630812755024",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3475907106665848,
            "range": "stddev: 0.014867699590473944",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.103329898666592,
            "range": "stddev: 0.017895346027042097",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.541183165666704,
            "range": "stddev: 0.0159020041265972",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 7.095939210000021,
            "range": "stddev: 0.03477966146478198",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.1775912133333197,
            "range": "stddev: 0.005967805874655219",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "93fc5bedd34a265c1afe235e08d01407f3f448fc",
          "message": "Rename suptht -> supu and supzta -> supv for consistency",
          "timestamp": "2026-04-07T21:06:10-05:00",
          "tree_id": "4d41779cd8bb0e4f6aa7a20f4412bf9742c12056",
          "url": "https://github.com/proximafusion/vmecpp/commit/93fc5bedd34a265c1afe235e08d01407f3f448fc"
        },
        "date": 1783604247793,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.27574248799983253,
            "range": "stddev: 0.004667093140902399",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.27322065800008205,
            "range": "stddev: 0.0018715998277316036",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3345218373331893,
            "range": "stddev: 0.0093290715528652",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.072130252333409,
            "range": "stddev: 0.03884447084757496",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.549474245999894,
            "range": "stddev: 0.02671092147507278",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 7.068237897666677,
            "range": "stddev: 0.01754475401997681",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.1618974330000735,
            "range": "stddev: 0.008554109921381147",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "82727604d701fd1202fef14aa41b6f177a249ecb",
          "message": "Tentative J calculation for when currumnc and currvmnc are available",
          "timestamp": "2026-04-07T21:12:58-05:00",
          "tree_id": "9f75260c54505d58b2787a498f5c970e5e7b6983",
          "url": "https://github.com/proximafusion/vmecpp/commit/82727604d701fd1202fef14aa41b6f177a249ecb"
        },
        "date": 1783604247762,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.2725162427999749,
            "range": "stddev: 0.0021493346662938883",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.27203979020005137,
            "range": "stddev: 0.0039954231491877625",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3267457376667455,
            "range": "stddev: 0.002588903404984964",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.0980816403334757,
            "range": "stddev: 0.05067146295787343",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.5115064556666766,
            "range": "stddev: 0.0318473344378503",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 7.014179090333225,
            "range": "stddev: 0.03216828536878027",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.158192694666771,
            "range": "stddev: 0.010231718413721905",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "daed90fa13b4c18db1121abaf2ff08a9374ff53a",
          "message": "Calculate Cartesian coordinates only if requested",
          "timestamp": "2026-04-07T21:17:41-05:00",
          "tree_id": "7c87983e2932e664481d10b0187f1cb7d2e62be9",
          "url": "https://github.com/proximafusion/vmecpp/commit/daed90fa13b4c18db1121abaf2ff08a9374ff53a"
        },
        "date": 1783604247907,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.2684377424000104,
            "range": "stddev: 0.0032170772819786263",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.26745020779999323,
            "range": "stddev: 0.0017611726926646046",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3135414966665546,
            "range": "stddev: 0.002668752489525365",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.032728579666658,
            "range": "stddev: 0.028435620579143204",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.4827152810000066,
            "range": "stddev: 0.014736843680813804",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 7.020300478333259,
            "range": "stddev: 0.05293096039935851",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.1536597283332715,
            "range": "stddev: 0.006910381919323868",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cf0ae1fbc9884abd5b7e4db7a7f3dc7dcd6b93a6",
          "message": "Precompute and reuse sin and cos terms",
          "timestamp": "2026-04-07T21:20:24-05:00",
          "tree_id": "2dea4c1b4d9b05940012201307a99708d64d4fa7",
          "url": "https://github.com/proximafusion/vmecpp/commit/cf0ae1fbc9884abd5b7e4db7a7f3dc7dcd6b93a6"
        },
        "date": 1783604247888,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.2698812892000205,
            "range": "stddev: 0.011907202294854367",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.2599521770000138,
            "range": "stddev: 0.0009198639435330141",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3101059826666035,
            "range": "stddev: 0.00552310087755559",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.0815265453332663,
            "range": "stddev: 0.055501037580543526",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.4736153869999575,
            "range": "stddev: 0.03235873139877782",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 6.981996525000038,
            "range": "stddev: 0.02626489628932233",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.154580082333344,
            "range": "stddev: 0.006421673674719169",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7e673720abc3c6c283ee636f91470b540f7b71aa",
          "message": "Remove unnecessary comment",
          "timestamp": "2026-04-07T21:23:05-05:00",
          "tree_id": "0fab048c9a714e1df431ec297d2e69ce4ef99b45",
          "url": "https://github.com/proximafusion/vmecpp/commit/7e673720abc3c6c283ee636f91470b540f7b71aa"
        },
        "date": 1783604247760,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.2663590069999827,
            "range": "stddev: 0.0021611598171060267",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.2604071819999717,
            "range": "stddev: 0.002436195515219578",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3133471986667093,
            "range": "stddev: 0.005120154746240165",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.097323705333338,
            "range": "stddev: 0.03855459635478948",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.457674969666641,
            "range": "stddev: 0.015346902936742767",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 6.988782621000003,
            "range": "stddev: 0.017618710972026233",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.152683174333409,
            "range": "stddev: 0.008522393322811827",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "3ec621eb49fd0d908f96157352da9aed133f8c82",
          "message": "Automatic changes by pre-commit run --all-files",
          "timestamp": "2026-04-07T21:28:26-05:00",
          "tree_id": "4da7da2b437c6e1a3277f6bfb356e29f89d93514",
          "url": "https://github.com/proximafusion/vmecpp/commit/3ec621eb49fd0d908f96157352da9aed133f8c82"
        },
        "date": 1783604247692,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.26792220979991727,
            "range": "stddev: 0.013214902061537968",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.25887818100000004,
            "range": "stddev: 0.00120362233436416",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3116029953333357,
            "range": "stddev: 0.0023883096300435767",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.047673739333201,
            "range": "stddev: 0.028617345988267213",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.4840374769999776,
            "range": "stddev: 0.02285842792455382",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 6.984805083666591,
            "range": "stddev: 0.02229528437306716",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.158489845666736,
            "range": "stddev: 0.005596204210374146",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "32790ff82a678f7d666fbef11845d30b5ca78fce",
          "message": "Import numpy so example function is callable as a module",
          "timestamp": "2026-04-07T22:34:55-05:00",
          "tree_id": "e4299d9e5d858c85b7679e831d795e0288d5c253",
          "url": "https://github.com/proximafusion/vmecpp/commit/32790ff82a678f7d666fbef11845d30b5ca78fce"
        },
        "date": 1783604247680,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.2607118530000207,
            "range": "stddev: 0.000701188348308614",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.26253920020003535,
            "range": "stddev: 0.003837586711929165",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.3206520960000034,
            "range": "stddev: 0.007291050767740363",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.028976361333283,
            "range": "stddev: 0.013664557212100675",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.479447784000058,
            "range": "stddev: 0.002122995193816038",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 6.963905313666753,
            "range": "stddev: 0.011598514292470488",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.1721968006666732,
            "range": "stddev: 0.02749511161894496",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "dcaa3059ffab97b6805c2d9afdb3c90909fa5bc9",
          "message": "Reshape B and J field to be compatible with a volumetric grid",
          "timestamp": "2026-04-07T23:24:05-05:00",
          "tree_id": "4160e98df87f6b374707cba451ad7e585d9d7718",
          "url": "https://github.com/proximafusion/vmecpp/commit/dcaa3059ffab97b6805c2d9afdb3c90909fa5bc9"
        },
        "date": 1783604247910,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.25924731499999326,
            "range": "stddev: 0.0009110254039812662",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.2617288598000414,
            "range": "stddev: 0.0014208999789013172",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.312393545666661,
            "range": "stddev: 0.0034551888895606917",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.0581993699999352,
            "range": "stddev: 0.03210469571681794",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.4537038949999137,
            "range": "stddev: 0.008311891848414892",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 6.957239578000099,
            "range": "stddev: 0.015657884084398662",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.1510186833335563,
            "range": "stddev: 0.007978362795075595",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "279080900a2abedea6a3f478a1218321cd40b385",
          "message": "Automatic changes by pre-commit run --all-files",
          "timestamp": "2026-04-07T23:27:35-05:00",
          "tree_id": "0659fdd49f80b3d0af9f2ee548f1a7880f20869c",
          "url": "https://github.com/proximafusion/vmecpp/commit/279080900a2abedea6a3f478a1218321cd40b385"
        },
        "date": 1783604247670,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34226769860001693,
            "range": "stddev: 0.000825578475610224",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3409562819999792,
            "range": "stddev: 0.0012962997060694675",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7460041063332785,
            "range": "stddev: 0.004210669145385721",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.660656835999968,
            "range": "stddev: 0.03081229342408632",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.456038765333308,
            "range": "stddev: 0.012294684184768144",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.73360688100009,
            "range": "stddev: 0.048474613757614964",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5257938796667077,
            "range": "stddev: 0.00328453723329001",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "bb623a411e381b7a45c3905502ce63c2d6886b56",
          "message": "Implement the current density Fourier coefficients. (#474)",
          "timestamp": "2026-04-10T12:26:12+02:00",
          "tree_id": "27605f632f483ab8cb9a4a98f54cbd75d36dee9c",
          "url": "https://github.com/proximafusion/vmecpp/commit/bb623a411e381b7a45c3905502ce63c2d6886b56"
        },
        "date": 1783604247845,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3356525623999914,
            "range": "stddev: 0.005033209613820676",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.33703337920001103,
            "range": "stddev: 0.0045818434827125",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7211751063333243,
            "range": "stddev: 0.016305136098159142",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6191271706666917,
            "range": "stddev: 0.00934394119112885",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.439308074666694,
            "range": "stddev: 0.01758919830766406",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.84562580233334,
            "range": "stddev: 0.050072553148645935",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5493210796666972,
            "range": "stddev: 0.007548453200586942",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d79e972b074dedc10976091e00e12f68551d5fb3",
          "message": "Restrict pydantic version, breaking change (#480)",
          "timestamp": "2026-04-16T11:06:55+02:00",
          "tree_id": "c081f0908ae5d478b1d3e6f9b2b2ae2457d276fb",
          "url": "https://github.com/proximafusion/vmecpp/commit/d79e972b074dedc10976091e00e12f68551d5fb3"
        },
        "date": 1783604247905,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.33215283600002293,
            "range": "stddev: 0.0011580881654347163",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.33561646319999455,
            "range": "stddev: 0.003787593364277135",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7784143373332881,
            "range": "stddev: 0.01875966015826798",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6907293513333266,
            "range": "stddev: 0.0183439477000992",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.4411808819999505,
            "range": "stddev: 0.04612274399188821",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.784593686333286,
            "range": "stddev: 0.05353674063179743",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5649303173333162,
            "range": "stddev: 0.01750808545932296",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "48dd1a146c28f3a06e53af22dbc2e8d2c32c38cc",
          "message": "Also add .github/workflows/copilot-setup-steps.yml to pre-install vmecpp and its Ubuntu system dependencies in the Copilot cloud agent environment. (#483)",
          "timestamp": "2026-04-16T13:33:19+02:00",
          "tree_id": "1aed1e09f2dffc1c1fdbc2e0601a7f79d30d2d0d",
          "url": "https://github.com/proximafusion/vmecpp/commit/48dd1a146c28f3a06e53af22dbc2e8d2c32c38cc"
        },
        "date": 1783604247704,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3426873528000215,
            "range": "stddev: 0.002690517550203128",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3376536604000194,
            "range": "stddev: 0.003324570206572298",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7510951043332927,
            "range": "stddev: 0.016850685808656576",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6273048593332837,
            "range": "stddev: 0.02952116490315292",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.3626337066666565,
            "range": "stddev: 0.025783322897859996",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.586234159999853,
            "range": "stddev: 0.04560479072597765",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.517261615000128,
            "range": "stddev: 0.024140528177392526",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8adda439309a04d379648e25c4f64f2c5f3bdedb",
          "message": "Add a Python 3.13 Nix development shell (#479)",
          "timestamp": "2026-04-16T14:44:28+02:00",
          "tree_id": "e8e3f2cfad15e4222d2e4ee2fbb465245fd99bf3",
          "url": "https://github.com/proximafusion/vmecpp/commit/8adda439309a04d379648e25c4f64f2c5f3bdedb"
        },
        "date": 1783604247775,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3464541012000154,
            "range": "stddev: 0.0026988486960750823",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3392206149999765,
            "range": "stddev: 0.0033077067942056575",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7281280549999185,
            "range": "stddev: 0.016449399289380908",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6382089443332006,
            "range": "stddev: 0.030937547518905886",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.346995621333311,
            "range": "stddev: 0.033090729296901565",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.716781777000051,
            "range": "stddev: 0.12856658205730598",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5478604840000116,
            "range": "stddev: 0.009525286984412231",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "fa1b951045b2243a0c9340d0b16bec4a60e5136e",
          "message": "Update README.md",
          "timestamp": "2026-04-23T10:12:38+02:00",
          "tree_id": "f3fac97cd72cc9ec7d5fd85ca957778291adbb65",
          "url": "https://github.com/proximafusion/vmecpp/commit/fa1b951045b2243a0c9340d0b16bec4a60e5136e"
        },
        "date": 1783604247933,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34201667920006,
            "range": "stddev: 0.0015436354850875762",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3428847340000175,
            "range": "stddev: 0.0017609186136768196",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7734222770000088,
            "range": "stddev: 0.004171987864049519",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.665689145999901,
            "range": "stddev: 0.028497737326997813",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.434352813666767,
            "range": "stddev: 0.00999861459818134",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.883008994333371,
            "range": "stddev: 0.0039801935854302485",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.561599104000076,
            "range": "stddev: 0.03031040727678281",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "51527dfb71c83aee7a8d83df6203758e2c78ef18",
          "message": "Miller recurrence relation for numerically stable NESTOR at high mpol/ntor (#484)",
          "timestamp": "2026-04-24T14:28:08+02:00",
          "tree_id": "46ee572b3021f833c339ffbbac9effe65b0f54e2",
          "url": "https://github.com/proximafusion/vmecpp/commit/51527dfb71c83aee7a8d83df6203758e2c78ef18"
        },
        "date": 1783604247709,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35508324540001013,
            "range": "stddev: 0.005692823747880829",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34467902220008,
            "range": "stddev: 0.004687610229451174",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.8267098639999706,
            "range": "stddev: 0.010349967485896493",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.70548533933326,
            "range": "stddev: 0.024093582984277436",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.515540096999985,
            "range": "stddev: 0.049464377657989625",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 10.41009158166662,
            "range": "stddev: 0.052198123510415445",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6661704883332884,
            "range": "stddev: 0.00914160489253006",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "f3f1a1068147acd40e45ed847d5a8cc06015b2a3",
          "message": "fix: write test output files to temp directories instead of CWD (#489)",
          "timestamp": "2026-04-27T10:41:19+02:00",
          "tree_id": "d77249fb049a39855dbb393d8b2771a785f8e3b0",
          "url": "https://github.com/proximafusion/vmecpp/commit/f3f1a1068147acd40e45ed847d5a8cc06015b2a3"
        },
        "date": 1783604247928,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3406154724000771,
            "range": "stddev: 0.007348550800356999",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3316996270000345,
            "range": "stddev: 0.0012853808710730512",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.8099286046667658,
            "range": "stddev: 0.008803886409588875",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6962726173333826,
            "range": "stddev: 0.015365848558774502",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.446686613666695,
            "range": "stddev: 0.01826737744932415",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 10.333990330333412,
            "range": "stddev: 0.0010490909451816564",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6657679793333198,
            "range": "stddev: 0.010570591001546068",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9c2041a1d23a6d9c69f9d98caff279d7166f4282",
          "message": "Fix normalize_by_currents having no effect on MagneticFieldResponseTable (#487)",
          "timestamp": "2026-04-27T14:21:51Z",
          "tree_id": "b2a4004f87cba6621d18e876a2da25becd03f3da",
          "url": "https://github.com/proximafusion/vmecpp/commit/9c2041a1d23a6d9c69f9d98caff279d7166f4282"
        },
        "date": 1783604247804,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.33714670759995896,
            "range": "stddev: 0.003614347194063574",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.333480796799995,
            "range": "stddev: 0.003031261008101719",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.802686408666735,
            "range": "stddev: 0.010686016490134703",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6661531256666726,
            "range": "stddev: 0.002302860284312016",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.42143992233332,
            "range": "stddev: 0.04407542424849031",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 10.299734745000023,
            "range": "stddev: 0.024775978756825044",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6884839753334593,
            "range": "stddev: 0.04632544033164859",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d6875c57529a77d7e32e6845b9d3ee40afdda5b3",
          "message": "Abseil errors instead of LOG(FATAL) (#493)",
          "timestamp": "2026-04-27T17:58:22+02:00",
          "tree_id": "a4313fa42f5838bf688910bffd3f9f3d68f9827b",
          "url": "https://github.com/proximafusion/vmecpp/commit/d6875c57529a77d7e32e6845b9d3ee40afdda5b3"
        },
        "date": 1783604247902,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3333142051999857,
            "range": "stddev: 0.001103513065998704",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3319817692000015,
            "range": "stddev: 0.0010015780912046668",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.8201397520000075,
            "range": "stddev: 0.005521275358709083",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6449079009999727,
            "range": "stddev: 0.01806791486214739",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.428115264333353,
            "range": "stddev: 0.0361276879493191",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 10.308118097999946,
            "range": "stddev: 0.013475167824159421",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6685597623333404,
            "range": "stddev: 0.01396832009678859",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "97cb9d20e1c85fdf05f9dbfae0549754daacbf4a",
          "message": "Scientific variable names get flagged incorrectly too often by clang-tidy (#491)",
          "timestamp": "2026-04-27T18:18:23+02:00",
          "tree_id": "b23af3b37dab107f49bb1792b9e40ac499ab3af8",
          "url": "https://github.com/proximafusion/vmecpp/commit/97cb9d20e1c85fdf05f9dbfae0549754daacbf4a"
        },
        "date": 1783604247796,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.33684889679998375,
            "range": "stddev: 0.011105708816765458",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3363489777999803,
            "range": "stddev: 0.007610863647197629",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.8197276270000202,
            "range": "stddev: 0.01585867194798235",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6570322246667124,
            "range": "stddev: 0.031421223723975274",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.447469250333294,
            "range": "stddev: 0.06556164442025969",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 10.32112995299993,
            "range": "stddev: 0.01911761815433989",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6711112520000218,
            "range": "stddev: 0.014705092459561598",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "42e3ec5f8b9b731a5ad717a06ec19adf7131111f",
          "message": "Use std::unreachable instead of LOG(FATAL) for unreachable code (#494)",
          "timestamp": "2026-04-27T23:01:30+02:00",
          "tree_id": "c57ee3789ceb1a61f7bbd68a6322c8d76d8bac12",
          "url": "https://github.com/proximafusion/vmecpp/commit/42e3ec5f8b9b731a5ad717a06ec19adf7131111f"
        },
        "date": 1783604247697,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3385337648000132,
            "range": "stddev: 0.006092554487067483",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.33623720199998386,
            "range": "stddev: 0.001623031195935577",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.8076121606666977,
            "range": "stddev: 0.0025281439864628945",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6880052686666054,
            "range": "stddev: 0.0034705759540062326",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.417343168333143,
            "range": "stddev: 0.040132342452364786",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 10.403331226000015,
            "range": "stddev: 0.09519293504915324",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.671162548333238,
            "range": "stddev: 0.01797540217649977",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b24222e51f3d3ab1f9acbff4139fbca21783a0b7",
          "message": "Terminate on NaN or inf, return NaN instead of terminating in tridiagonal solve (#496)",
          "timestamp": "2026-04-28T09:42:10+02:00",
          "tree_id": "77e85f47c89c700befb46c86153741f0fbeb0185",
          "url": "https://github.com/proximafusion/vmecpp/commit/b24222e51f3d3ab1f9acbff4139fbca21783a0b7"
        },
        "date": 1783604247840,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34624938360007035,
            "range": "stddev: 0.002543509410659308",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34680358600003275,
            "range": "stddev: 0.0012382530140137436",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7427365230000003,
            "range": "stddev: 0.01699638173771146",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.650717005000009,
            "range": "stddev: 0.012279102191401709",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.429213836333322,
            "range": "stddev: 0.019851073664199456",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.991559757999994,
            "range": "stddev: 0.07195018347330791",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.551548060333289,
            "range": "stddev: 0.009099880942686931",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "6f846de63275c49dc5f60da16e2dfb052055c50d",
          "message": "TSAN docker setup (#492)",
          "timestamp": "2026-04-28T12:23:06+02:00",
          "tree_id": "6ea1141f29156aa8a27ad98b5fa089076d812218",
          "url": "https://github.com/proximafusion/vmecpp/commit/6f846de63275c49dc5f60da16e2dfb052055c50d"
        },
        "date": 1783604247742,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34664910660003445,
            "range": "stddev: 0.0036610757422583544",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3453176641999562,
            "range": "stddev: 0.0015320868952512184",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7132650710000992,
            "range": "stddev: 0.0044077222738314265",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.632345539333452,
            "range": "stddev: 0.0030927632605059078",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.412890842333279,
            "range": "stddev: 0.031617803613752604",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.992957224666648,
            "range": "stddev: 0.09430977636315496",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5527552736665864,
            "range": "stddev: 0.007984430492103236",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c9772aed7885bd7ed62e4fbaae99d8a30915e74c",
          "message": "Niter wording improved (#504)",
          "timestamp": "2026-04-28T16:59:51+02:00",
          "tree_id": "7cc623a63d553fa23218de359440aa6bc4c7d20e",
          "url": "https://github.com/proximafusion/vmecpp/commit/c9772aed7885bd7ed62e4fbaae99d8a30915e74c"
        },
        "date": 1783604247875,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34866723619998086,
            "range": "stddev: 0.003332755072838524",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3495743492000202,
            "range": "stddev: 0.003078819597790143",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7197257376666737,
            "range": "stddev: 0.009304815104863669",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.637145248666684,
            "range": "stddev: 0.015521758280680201",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.411118192333333,
            "range": "stddev: 0.020753869835488013",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.92765959366678,
            "range": "stddev: 0.0174883679630289",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5478676016667425,
            "range": "stddev: 0.00805664600781462",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ea76d2124c0a47e294f7369d8fa36781d5f7d2fe",
          "message": "FFTW3 dependencies (#501)",
          "timestamp": "2026-04-28T17:32:26+02:00",
          "tree_id": "e7fb40efbab4d320cc28192ea0b0a916d6b7500d",
          "url": "https://github.com/proximafusion/vmecpp/commit/ea76d2124c0a47e294f7369d8fa36781d5f7d2fe"
        },
        "date": 1783604247922,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3465079548000176,
            "range": "stddev: 0.0008858936656849247",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3471547562000524,
            "range": "stddev: 0.0022817684454694664",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7106209063332851,
            "range": "stddev: 0.0018702797556289883",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6530380320000404,
            "range": "stddev: 0.02708522132009589",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.4242306316667355,
            "range": "stddev: 0.021200457496750213",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.885730655333191,
            "range": "stddev: 0.013812750345565877",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5484192133332424,
            "range": "stddev: 0.010268231920610915",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a2b31a8383c130052994620d0585b075b7a13fa6",
          "message": "Refactor LaplaceSolver to use Eigen3 for matrix operations (#499)",
          "timestamp": "2026-04-28T15:52:23Z",
          "tree_id": "7cdf4e474746688ac76d605713e9bb783da32bd4",
          "url": "https://github.com/proximafusion/vmecpp/commit/a2b31a8383c130052994620d0585b075b7a13fa6"
        },
        "date": 1783604247819,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3487790344000132,
            "range": "stddev: 0.001728912719432096",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3447047592000672,
            "range": "stddev: 0.0008981182780669299",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7450734576664217,
            "range": "stddev: 0.00438620177677473",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.672634482666581,
            "range": "stddev: 0.03217938345378433",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.407075229333411,
            "range": "stddev: 0.02346539700988785",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.231338382333282,
            "range": "stddev: 0.016146317726002623",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5554957169999095,
            "range": "stddev: 0.013658408969449368",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "12b7f60d9a4c5417ceddaa4d5ea5ab0e25febf6a",
          "message": "Laplace solver stellarator asym added (#502)",
          "timestamp": "2026-04-28T18:12:25+02:00",
          "tree_id": "2f14fa7d2c8decb9c434dc27157c2c2916187b00",
          "url": "https://github.com/proximafusion/vmecpp/commit/12b7f60d9a4c5417ceddaa4d5ea5ab0e25febf6a"
        },
        "date": 1783604247652,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34606510559997333,
            "range": "stddev: 0.001345741347791347",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3464710322000428,
            "range": "stddev: 0.002774248987059563",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7354843803332187,
            "range": "stddev: 0.0034184816031015784",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.684634402000029,
            "range": "stddev: 0.003996375586799178",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.4129850233331736,
            "range": "stddev: 0.03457917149349925",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.22800749233329,
            "range": "stddev: 0.015615586473757046",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5534065520000695,
            "range": "stddev: 0.00958929138857708",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c8e567fc623c429024c11a4555e0fd8a58b71679",
          "message": "Revert \"FFTW3 dependencies (#501)\" (#506)",
          "timestamp": "2026-04-29T15:14:21+02:00",
          "tree_id": "4895e3fe13cfbe6572522fbff370fdb42ef68df8",
          "url": "https://github.com/proximafusion/vmecpp/commit/c8e567fc623c429024c11a4555e0fd8a58b71679"
        },
        "date": 1783604247867,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35332981240001116,
            "range": "stddev: 0.008004854445833123",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34706427239996174,
            "range": "stddev: 0.002282164870180415",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7777856753333101,
            "range": "stddev: 0.0751263020965832",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.667606668333292,
            "range": "stddev: 0.04618595501976926",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.441614011666691,
            "range": "stddev: 0.014595652808145546",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.251139641333339,
            "range": "stddev: 0.013346671567572948",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5570387209999883,
            "range": "stddev: 0.01761391306547995",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "378b4ff6edc6dd41d3afde48fc66f38723726297",
          "message": "Reduce free boundary oversubscription due to Eigen low level parallelism on some platforms (#507)",
          "timestamp": "2026-04-29T15:33:24+02:00",
          "tree_id": "32526f65a9fdcc89d31c91b4fdbf874ceb6c27b9",
          "url": "https://github.com/proximafusion/vmecpp/commit/378b4ff6edc6dd41d3afde48fc66f38723726297"
        },
        "date": 1783604247682,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3475187991999974,
            "range": "stddev: 0.003035086687604592",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3481676178000271,
            "range": "stddev: 0.005513872269170873",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7582403363333015,
            "range": "stddev: 0.0022683807176667077",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6716974753332656,
            "range": "stddev: 0.0167160113068018",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.47093546366663,
            "range": "stddev: 0.018034099493851374",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.314333008666722,
            "range": "stddev: 0.010901494801263949",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.549898378333296,
            "range": "stddev: 0.011810098336811332",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7bb4582991a98dc91f165f9659e9338fefb7e8bb",
          "message": "Reduce thread spinning on first parallel region enter (#508)",
          "timestamp": "2026-04-30T01:39:45+02:00",
          "tree_id": "c102485b620e8b188776c29b801d5c85342061c5",
          "url": "https://github.com/proximafusion/vmecpp/commit/7bb4582991a98dc91f165f9659e9338fefb7e8bb"
        },
        "date": 1783604247749,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34789456819996756,
            "range": "stddev: 0.003704555163319854",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3473307778000162,
            "range": "stddev: 0.0032476033142448944",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7645947653332996,
            "range": "stddev: 0.004497968702691829",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.652329379999969,
            "range": "stddev: 0.017795227539608587",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.4644349983333695,
            "range": "stddev: 0.03821320453431211",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.290222442666655,
            "range": "stddev: 0.015756820949319936",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.552192198000019,
            "range": "stddev: 0.006688189721896287",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "01b1fcb56a26188eb71a74fcf9cdf0c7349e5c04",
          "message": "Diagnose how much force residual is being truncated away (#495)",
          "timestamp": "2026-04-30T21:49:38+02:00",
          "tree_id": "89a1ebe9dc8cda139a3ada16a72c43556f11f0ff",
          "url": "https://github.com/proximafusion/vmecpp/commit/01b1fcb56a26188eb71a74fcf9cdf0c7349e5c04"
        },
        "date": 1783604247632,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3487984150000102,
            "range": "stddev: 0.00634335240818549",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3459593035999887,
            "range": "stddev: 0.003665107779742392",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7499901680000296,
            "range": "stddev: 0.00404115779433167",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.65322560300001,
            "range": "stddev: 0.011433988421769516",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.461131650333338,
            "range": "stddev: 0.012549027226227796",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.293990005333399,
            "range": "stddev: 0.005686421431860945",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5475087833332661,
            "range": "stddev: 0.011369765964691378",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "3ac242a3d87feef9b57e348f50131d9754ca43d1",
          "message": "Batched FFT, 15% expected gain for w7x (#503)",
          "timestamp": "2026-05-05T15:22:26+02:00",
          "tree_id": "b48a2de956f8741c9aecfcbf733013e1a4b4f1da",
          "url": "https://github.com/proximafusion/vmecpp/commit/3ac242a3d87feef9b57e348f50131d9754ca43d1"
        },
        "date": 1783604247685,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3458225659999698,
            "range": "stddev: 0.00204086027779178",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34461655560007787,
            "range": "stddev: 0.0016378354248590165",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7164883553333918,
            "range": "stddev: 0.011694347908352956",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4328445076666108,
            "range": "stddev: 0.009134495470182314",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.836967750333315,
            "range": "stddev: 0.026395616928002727",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.259293888666738,
            "range": "stddev: 0.023297938929718962",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5498550053334081,
            "range": "stddev: 0.008476209524730237",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8ed41a4ebc3aac15f0356bc75d4f8cebbac6ef0c",
          "message": "More realistic wait policy, additional fixed boundary benchmark (#515)",
          "timestamp": "2026-05-05T15:48:58+02:00",
          "tree_id": "217c3b116ed2219ed36cb2ea3d15e5d7f56c8ef8",
          "url": "https://github.com/proximafusion/vmecpp/commit/8ed41a4ebc3aac15f0356bc75d4f8cebbac6ef0c"
        },
        "date": 1783604247783,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3486312370000178,
            "range": "stddev: 0.008299667533711487",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3469027872000424,
            "range": "stddev: 0.00257568015621949",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7117643376666365,
            "range": "stddev: 0.009329599379751965",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4394759046667027,
            "range": "stddev: 0.03547556369234785",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.86080453300004,
            "range": "stddev: 0.036383956081344496",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.246493558333214,
            "range": "stddev: 0.03322194303379532",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.548703686333359,
            "range": "stddev: 0.00906407782357297",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "04158aa5f9eae86e399a928db56afef3413dabb1",
          "message": "Update README.md",
          "timestamp": "2026-05-12T21:26:02+02:00",
          "tree_id": "4901ac7979c957946219f2483da33f2eab28d01f",
          "url": "https://github.com/proximafusion/vmecpp/commit/04158aa5f9eae86e399a928db56afef3413dabb1"
        },
        "date": 1783604247634,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34495523980003784,
            "range": "stddev: 0.0026604927979930184",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3445664628000486,
            "range": "stddev: 0.00039033936920035705",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7082556513333504,
            "range": "stddev: 0.0034151357574800935",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4312935406666534,
            "range": "stddev: 0.03600134697161343",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8196919080000193,
            "range": "stddev: 0.028341113580143518",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.232229785333326,
            "range": "stddev: 0.02116947739750256",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5504532399998727,
            "range": "stddev: 0.011506608212506235",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9b2e1e066106a9100d31aad811677c8f23fb8456",
          "message": "Update copilot-setup-steps.yml",
          "timestamp": "2026-05-13T09:45:01+02:00",
          "tree_id": "213971aa8b812e90eea7f78beeed6a4d98f68681",
          "url": "https://github.com/proximafusion/vmecpp/commit/9b2e1e066106a9100d31aad811677c8f23fb8456"
        },
        "date": 1783604247801,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34673475039999174,
            "range": "stddev: 0.002873027872931421",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3576699441999608,
            "range": "stddev: 0.0040620982052141895",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.709357956333406,
            "range": "stddev: 0.003097770350238373",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3953177270000197,
            "range": "stddev: 0.024068384960375184",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.812655920666733,
            "range": "stddev: 0.008715171488605376",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.223388248999981,
            "range": "stddev: 0.0007952559683574496",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.547752255999967,
            "range": "stddev: 0.011824617026036512",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5f23d714753bb808bf5ca2675374e1385ae1468a",
          "message": "Remove FFTW from container installs, since we settled on FFTX now (#517)",
          "timestamp": "2026-05-13T10:01:17+02:00",
          "tree_id": "3aa86ce29736387d10a101cb18432185447d9955",
          "url": "https://github.com/proximafusion/vmecpp/commit/5f23d714753bb808bf5ca2675374e1385ae1468a"
        },
        "date": 1783604247722,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3575138845999845,
            "range": "stddev: 0.01594374419788809",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3495566034000149,
            "range": "stddev: 0.0033631458566466066",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7140624510000937,
            "range": "stddev: 0.006445364444444394",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4140217320000374,
            "range": "stddev: 0.021187831346912468",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.83784131400004,
            "range": "stddev: 0.03423194851679487",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.283310056666702,
            "range": "stddev: 0.025814857857056735",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5507455673334032,
            "range": "stddev: 0.014385030579379916",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "6a5bee6ee5669c5a77face38d1e8056bba120abd",
          "message": "Enable pre-commit for copilot agents (#520)",
          "timestamp": "2026-05-13T13:44:26+02:00",
          "tree_id": "265592f4b5820720cc5c7fbe0372796b1e38086b",
          "url": "https://github.com/proximafusion/vmecpp/commit/6a5bee6ee5669c5a77face38d1e8056bba120abd"
        },
        "date": 1783604247737,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3446598374000132,
            "range": "stddev: 0.0006589355117272339",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34352489060001973,
            "range": "stddev: 0.0014889827565561832",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7126917986667347,
            "range": "stddev: 0.001236518288950752",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.405991514666615,
            "range": "stddev: 0.03032400642301806",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.81226263299997,
            "range": "stddev: 0.01858844466809309",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.229105757666654,
            "range": "stddev: 0.00940152568414102",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5666859603332643,
            "range": "stddev: 0.022332168687109726",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Asim Arshad",
            "email": "asim48@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "4667d5dfa147f908c9f8d6235977e5536d70a1d5",
          "message": "Fix force balance spectrum example shape",
          "timestamp": "2026-05-14T02:00:00+01:00",
          "tree_id": "aafb2c13b94ea6c1640ce5ac74433f563d72b95d",
          "url": "https://github.com/proximafusion/vmecpp/commit/4667d5dfa147f908c9f8d6235977e5536d70a1d5"
        },
        "date": 1783604247699,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3449035527999513,
            "range": "stddev: 0.0024973821038396776",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34638588159996286,
            "range": "stddev: 0.003518323748829976",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.704834754333433,
            "range": "stddev: 0.0011068008270745154",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.395668969666682,
            "range": "stddev: 0.0042495125488868695",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.82186437866676,
            "range": "stddev: 0.020087335497286446",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.210315512999918,
            "range": "stddev: 0.010204616052113586",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5476164636667515,
            "range": "stddev: 0.012823394114349964",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7e06d9300a2b0afadf440c9b557fe4bfab5c3c79",
          "message": "Update README.md (#516)",
          "timestamp": "2026-05-14T23:54:23+02:00",
          "tree_id": "0b15c8f66daa0128501442e828e3d74f0cb119b3",
          "url": "https://github.com/proximafusion/vmecpp/commit/7e06d9300a2b0afadf440c9b557fe4bfab5c3c79"
        },
        "date": 1783604247757,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35274056060002296,
            "range": "stddev: 0.007080998155489675",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34941276520003156,
            "range": "stddev: 0.003997428712838251",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.735141628333319,
            "range": "stddev: 0.021855027199296707",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.408282987333299,
            "range": "stddev: 0.006274173591809101",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.797721156333258,
            "range": "stddev: 0.02989030999746141",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.265496991666547,
            "range": "stddev: 0.01762634378795883",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5496048470000308,
            "range": "stddev: 0.01090591654219504",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "1721a87219664eea872c668e74189a3eea115d94",
          "message": "Add CI coverage for example scripts via dedicated pytest job (#523)",
          "timestamp": "2026-05-16T15:56:22Z",
          "tree_id": "3f864c1d77b0ca3397cff9c28da416dc5109f32c",
          "url": "https://github.com/proximafusion/vmecpp/commit/1721a87219664eea872c668e74189a3eea115d94"
        },
        "date": 1783604247662,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.36508047859997533,
            "range": "stddev: 0.003987958237365752",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.36570622119993457,
            "range": "stddev: 0.0024680359994731717",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7279056639999908,
            "range": "stddev: 0.007638101113329095",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4656198096665776,
            "range": "stddev: 0.02770150342878277",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.89887327533332,
            "range": "stddev: 0.03606858318967029",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.312658129333386,
            "range": "stddev: 0.06524455348772627",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5535739063332887,
            "range": "stddev: 0.01027568678311432",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "bfc276c7f512c3abab65c6760cfaf3fb7c231357",
          "message": "README: fix missing word + 'suggest to give' phrasing",
          "timestamp": "2026-05-18T05:17:51-04:00",
          "tree_id": "8bf398f0174c87a944caff9c9b33117e0b9a715c",
          "url": "https://github.com/proximafusion/vmecpp/commit/bfc276c7f512c3abab65c6760cfaf3fb7c231357"
        },
        "date": 1783604247848,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.36715096899988564,
            "range": "stddev: 0.0009452515304040408",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3677157747999445,
            "range": "stddev: 0.003689690971564901",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.740989407333321,
            "range": "stddev: 0.024894510572195473",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.6287823996666098,
            "range": "stddev: 0.3212842944591567",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.903872294000242,
            "range": "stddev: 0.017157399026918225",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.304166970333577,
            "range": "stddev: 0.009510909212399705",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5514904883334566,
            "range": "stddev: 0.008274372813362758",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8d14954abd147b871352ae1522eb3973c5ba3081",
          "message": "bazel: make vmecpp usable as a Bazel module (#532)",
          "timestamp": "2026-05-28T17:12:08-04:00",
          "tree_id": "87a632621a184995f103204598cb64adbcc5d063",
          "url": "https://github.com/proximafusion/vmecpp/commit/8d14954abd147b871352ae1522eb3973c5ba3081"
        },
        "date": 1783604247778,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35082460699995865,
            "range": "stddev: 0.0023541772863426933",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3513098151999657,
            "range": "stddev: 0.0014649478310097674",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7333492036667242,
            "range": "stddev: 0.030977337022359115",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.458318295333356,
            "range": "stddev: 0.023983687328529242",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8507874633332904,
            "range": "stddev: 0.0218336875273648",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.236634452333268,
            "range": "stddev: 0.01721592835771713",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5523069610001887,
            "range": "stddev: 0.005924520308838677",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ab117cb23c72e537b233ebb59597598a600f5a3a",
          "message": "output_quantities: zero-initialize jPS2 axis/edge entries (#529)",
          "timestamp": "2026-05-28T17:16:42-04:00",
          "tree_id": "d05dd4fd3583aae9c378126982384bdfac009e1e",
          "url": "https://github.com/proximafusion/vmecpp/commit/ab117cb23c72e537b233ebb59597598a600f5a3a"
        },
        "date": 1783604247830,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3625118458000543,
            "range": "stddev: 0.0037834775448490014",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3588724421999359,
            "range": "stddev: 0.0023061196428591224",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7380114656666592,
            "range": "stddev: 0.0015640495413289301",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.44899311733343,
            "range": "stddev: 0.02247155614084818",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8586660989999473,
            "range": "stddev: 0.023442543263795287",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.279504130000078,
            "range": "stddev: 0.004014857007634471",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5525176126666338,
            "range": "stddev: 0.009263560763665713",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8596e70339522c3c4959a1af1e278c758d6c04bb",
          "message": "tests: cross-check vmecpp/test_data and vmecpp_large_cpp_tests inputs via indata2json (#534)",
          "timestamp": "2026-05-28T17:19:33-04:00",
          "tree_id": "a1625da2614ebf437f5631901fb16e3abf41914f",
          "url": "https://github.com/proximafusion/vmecpp/commit/8596e70339522c3c4959a1af1e278c758d6c04bb"
        },
        "date": 1783604247767,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3566831724001531,
            "range": "stddev: 0.002395432600468801",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3532161343999178,
            "range": "stddev: 0.0025705268921989983",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7210816163331704,
            "range": "stddev: 0.003983784544930758",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.446666174000105,
            "range": "stddev: 0.01712613837034987",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.865620724333288,
            "range": "stddev: 0.03073173592945996",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.236233771666624,
            "range": "stddev: 0.025486031741849765",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5472504753333851,
            "range": "stddev: 0.00876046813860202",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ca4fc48fa3f627ded8128565b54c00f068a0b80b",
          "message": "Populate full-grid bsubsmns_full in wout (#530)",
          "timestamp": "2026-05-28T17:40:09-04:00",
          "tree_id": "b0eb42db2d5fdc07733bcbb41e64fd47d67a2129",
          "url": "https://github.com/proximafusion/vmecpp/commit/ca4fc48fa3f627ded8128565b54c00f068a0b80b"
        },
        "date": 1783604247880,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3466719478000414,
            "range": "stddev: 0.0020862317369491536",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3443147383999531,
            "range": "stddev: 0.0008384795119975474",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7520515600000788,
            "range": "stddev: 0.014203719994158438",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4459381093335346,
            "range": "stddev: 0.01813810058084074",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8827957366667456,
            "range": "stddev: 0.028863828362059737",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.309574397000082,
            "range": "stddev: 0.014647991680142145",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5472935253331646,
            "range": "stddev: 0.012817194269362004",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a14f9448e88d417947535cb4ab0453a2fc620b0b",
          "message": "radial_profiles: implement spline and analytic profile evaluators (#531)",
          "timestamp": "2026-05-29T06:53:35-04:00",
          "tree_id": "551fabc736ae3a92dda5670a0e3d44ba063c0663",
          "url": "https://github.com/proximafusion/vmecpp/commit/a14f9448e88d417947535cb4ab0453a2fc620b0b"
        },
        "date": 1783604247814,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3676424051999675,
            "range": "stddev: 0.005163125139520124",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.35824341079996885,
            "range": "stddev: 0.006258680708164232",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7413345766666832,
            "range": "stddev: 0.0017582637957343052",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4342300996666686,
            "range": "stddev: 0.03480465865606966",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8861724040000354,
            "range": "stddev: 0.022388076401113975",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.308489638333413,
            "range": "stddev: 0.02177341057238767",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5626093146666638,
            "range": "stddev: 0.020774727617336175",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "4d8a378fdd27c735c568b187e2bcf5107cb5ea78",
          "message": "Complete the Eigen3 port of the free-boundary code (#543)",
          "timestamp": "2026-05-31T07:00:34-04:00",
          "tree_id": "2a4b84595103ac514d38fbc52f6506f9416de2e7",
          "url": "https://github.com/proximafusion/vmecpp/commit/4d8a378fdd27c735c568b187e2bcf5107cb5ea78"
        },
        "date": 1783604247707,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35208138600000893,
            "range": "stddev: 0.004305146637352694",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34587727539997104,
            "range": "stddev: 0.0026435552501249443",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.713474670666604,
            "range": "stddev: 0.005684626743723472",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4171353770000223,
            "range": "stddev: 0.011380078016086245",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.9359887803332945,
            "range": "stddev: 0.07839283101901981",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.296385672000042,
            "range": "stddev: 0.03546602690286609",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5546825536666802,
            "range": "stddev: 0.0038342004055311016",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "10027e5d5ec4f31fcd863663cf6c8d4a41b45b5d",
          "message": "Cache TSAN Docker image in GHCR (#553)",
          "timestamp": "2026-05-31T14:42:53+02:00",
          "tree_id": "b8cc6a49e078feacb3fcf762afc3455a317e1624",
          "url": "https://github.com/proximafusion/vmecpp/commit/10027e5d5ec4f31fcd863663cf6c8d4a41b45b5d"
        },
        "date": 1783604247647,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35846439900001315,
            "range": "stddev: 0.0056607889191285215",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3635202880000179,
            "range": "stddev: 0.00454321240947847",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7151753456666938,
            "range": "stddev: 0.01279775668417154",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.440078344000085,
            "range": "stddev: 0.028615171586038237",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.9107961243333498,
            "range": "stddev: 0.02244306466208408",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.321349473000131,
            "range": "stddev: 0.06102281607214961",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5496999080000933,
            "range": "stddev: 0.0108976095377969",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c945b86dc692a2c60b1d3832095c2d68f719e25f",
          "message": "free_boundary: multi-grid test case and opt-in to persist the Nestor solver across grid steps (#535)",
          "timestamp": "2026-05-31T09:54:14-04:00",
          "tree_id": "d1ce5591064c5a20b3c3b95506696129531a660c",
          "url": "https://github.com/proximafusion/vmecpp/commit/c945b86dc692a2c60b1d3832095c2d68f719e25f"
        },
        "date": 1783604247872,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34868683499989855,
            "range": "stddev: 0.002432815044142967",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.35311801359994205,
            "range": "stddev: 0.0066887818644516634",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.705671671000194,
            "range": "stddev: 0.005102050039286197",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4326005023332677,
            "range": "stddev: 0.005474851585334404",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.89091130099996,
            "range": "stddev: 0.027608006462560845",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.305033900999812,
            "range": "stddev: 0.01759612842420742",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5534330423333813,
            "range": "stddev: 0.0011952558778271876",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0138a54cf5c8b36d5b9229f68685f8cc331a2c4d",
          "message": "free_boundary: implement non-stellarator-symmetric surface geometry (#533)",
          "timestamp": "2026-06-01T18:59:48-04:00",
          "tree_id": "d39ce9ccbe6b761888e0a2bafef155883520004b",
          "url": "https://github.com/proximafusion/vmecpp/commit/0138a54cf5c8b36d5b9229f68685f8cc331a2c4d"
        },
        "date": 1783604247629,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3468934879999324,
            "range": "stddev: 0.00749656999186771",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.35183859460003075,
            "range": "stddev: 0.010881310510838104",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.705503144333458,
            "range": "stddev: 0.011883745041457847",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4342632716667745,
            "range": "stddev: 0.013834258861589316",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.9149831900000813,
            "range": "stddev: 0.03741345972699845",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.220177506333433,
            "range": "stddev: 0.013155041097021487",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5419435136667137,
            "range": "stddev: 0.01228924664820955",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8dca9e9711204f7a75df4bad46a1e847447d8c59",
          "message": "Streamlined AGENTS.md (#539)",
          "timestamp": "2026-06-02T01:10:12+02:00",
          "tree_id": "6991bf52fdd0f0ecde43b759bf11bd23f9a74f76",
          "url": "https://github.com/proximafusion/vmecpp/commit/8dca9e9711204f7a75df4bad46a1e847447d8c59"
        },
        "date": 1783604247780,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3434951840001304,
            "range": "stddev: 0.0027021805605447085",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34113240879996737,
            "range": "stddev: 0.001205286346742885",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.705865538333607,
            "range": "stddev: 0.006193865284089178",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4258353236667367,
            "range": "stddev: 0.023858234954082118",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8718917596667475,
            "range": "stddev: 0.053096692725626164",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.236119993666762,
            "range": "stddev: 0.03347500171597112",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5419107596668862,
            "range": "stddev: 0.009849819528799002",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a8268b821e9e2570f33ec47323823c90e873313e",
          "message": "free_boundary: implement axisymmetric (nzeta=1) free-boundary support (#536)",
          "timestamp": "2026-06-02T02:18:05-04:00",
          "tree_id": "3ba41eaa4b12a01ed15c5589262cdda109e0830b",
          "url": "https://github.com/proximafusion/vmecpp/commit/a8268b821e9e2570f33ec47323823c90e873313e"
        },
        "date": 1783604247827,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.31375467320000894,
            "range": "stddev: 0.0012365573704380996",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3133782372000155,
            "range": "stddev: 0.003932047531882607",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7233283253333411,
            "range": "stddev: 0.004456404877393303",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.308745446333328,
            "range": "stddev: 0.024431376333433265",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.499013339666675,
            "range": "stddev: 0.022358787399425536",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 8.525262567333337,
            "range": "stddev: 0.04251248744531726",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.4559872006666599,
            "range": "stddev: 0.01218722268449764",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5f3407175584ab13b2f0c1b5a5623f207e5ea654",
          "message": "Expose the forward model and drive the equilibrium iteration from Python (#546)",
          "timestamp": "2026-06-02T02:49:24-04:00",
          "tree_id": "b911431cff5a48c2af7038c5e16230a4306b0977",
          "url": "https://github.com/proximafusion/vmecpp/commit/5f3407175584ab13b2f0c1b5a5623f207e5ea654"
        },
        "date": 1783604247724,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.31536818240001596,
            "range": "stddev: 0.00165982490249083",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.31489312200000086,
            "range": "stddev: 0.0009186497566242351",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7113886346666618,
            "range": "stddev: 0.0030986032009818944",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.298240510000028,
            "range": "stddev: 0.020856089723162143",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.434673788333328,
            "range": "stddev: 0.023213437077060445",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 8.44521462100001,
            "range": "stddev: 0.018471165860193553",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.4489418859999812,
            "range": "stddev: 0.007618496388979927",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "3ca4231d1dfaa8696acf81af115e5206507d1d4d",
          "message": "Expose the full threed1 output quantities in the Python interface (#542)",
          "timestamp": "2026-06-02T03:23:15-04:00",
          "tree_id": "f9315d08af9a48356fc6e9d9f49a1c40b4156a26",
          "url": "https://github.com/proximafusion/vmecpp/commit/3ca4231d1dfaa8696acf81af115e5206507d1d4d"
        },
        "date": 1783604247687,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.32798612660001253,
            "range": "stddev: 0.0006438450115003101",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.32793704880000407,
            "range": "stddev: 0.0004419770450119297",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7120469816666741,
            "range": "stddev: 0.002483489664928455",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.308138907000019,
            "range": "stddev: 0.026739441677399344",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.399501650333339,
            "range": "stddev: 0.023614128441952188",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 8.454809145333305,
            "range": "stddev: 0.024638494384334356",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.447779019000033,
            "range": "stddev: 0.010448653962795443",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "55c569444c82fc93b55b2c752c7f793d841625fa",
          "message": "Python iteration debug script (#556)",
          "timestamp": "2026-06-03T08:58:09+02:00",
          "tree_id": "10550af4bf37180818ab1e0ca3487939aacef66c",
          "url": "https://github.com/proximafusion/vmecpp/commit/55c569444c82fc93b55b2c752c7f793d841625fa"
        },
        "date": 1783604247717,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.33084215240005505,
            "range": "stddev: 0.00202369600244569",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3284075487999871,
            "range": "stddev: 0.00145560469475152",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7101285390000005,
            "range": "stddev: 0.0023135590597897324",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.317311234333298,
            "range": "stddev: 0.019390943248909928",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.508667065666676,
            "range": "stddev: 0.013640944015673607",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 8.488607061666661,
            "range": "stddev: 0.007661464063109261",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.4580668500000609,
            "range": "stddev: 0.01074851665721997",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0ff3e7c9fb29385e6a91753ee320c983cff5e6e0",
          "message": "Update fourier_basis_implementation.md (#559)",
          "timestamp": "2026-06-05T10:02:40+02:00",
          "tree_id": "4be8128819c0b9b38cd83ea5c1989c69e26f42ec",
          "url": "https://github.com/proximafusion/vmecpp/commit/0ff3e7c9fb29385e6a91753ee320c983cff5e6e0"
        },
        "date": 1783604247645,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.33329349939997427,
            "range": "stddev: 0.001601797553223102",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3328982918000065,
            "range": "stddev: 0.0024595708410792947",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7147690650000034,
            "range": "stddev: 0.0021127371364506172",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3389685350000113,
            "range": "stddev: 0.03238916697376248",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.502991537333325,
            "range": "stddev: 0.028714110227811784",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 8.482008147666647,
            "range": "stddev: 0.029132544320951668",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.451721513666674,
            "range": "stddev: 0.007402244102901715",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a248cd9869c53281ca99b2485c75de3dd5851153",
          "message": "build+ci: abseil commit pin for Clang>=21, VMEC2000-from-source, benchmark fork guard (#564)",
          "timestamp": "2026-06-15T08:57:27+02:00",
          "tree_id": "13ea6b4473301c940018476545b224d738c209bd",
          "url": "https://github.com/proximafusion/vmecpp/commit/a248cd9869c53281ca99b2485c75de3dd5851153"
        },
        "date": 1783604247817,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.32934317119998013,
            "range": "stddev: 0.0021662262477944966",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.32890577939997456,
            "range": "stddev: 0.0016907253267798364",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7116814463333487,
            "range": "stddev: 0.004681643476513215",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.2729217730000073,
            "range": "stddev: 0.037870402879875094",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.460943387333373,
            "range": "stddev: 0.02292897898351874",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 8.453430116666633,
            "range": "stddev: 0.014586061849710268",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.4547331759999906,
            "range": "stddev: 0.007402818677794227",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c3ec179b9fa09fc043444ed60a46fafa726d620b",
          "message": "Handle jaxtyping anonymous-dimension API drift in wout serialization (#562)",
          "timestamp": "2026-06-15T17:12:05+02:00",
          "tree_id": "5c962a9aab90d2912a1034d8f8fc1c75980d8db6",
          "url": "https://github.com/proximafusion/vmecpp/commit/c3ec179b9fa09fc043444ed60a46fafa726d620b"
        },
        "date": 1783604247853,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34897293300000454,
            "range": "stddev: 0.0020884012389383857",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.346812259599983,
            "range": "stddev: 0.0016990170106700084",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7347295893333314,
            "range": "stddev: 0.008146118746780623",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4180783066666436,
            "range": "stddev: 0.05914596648124857",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.886478883666655,
            "range": "stddev: 0.031451708942024134",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.6108581793333,
            "range": "stddev: 0.03974757517464493",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6703742533333827,
            "range": "stddev: 0.0028030236690515017",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b0876027f1c16d0910a80564118dae42161d0cad",
          "message": "Update abseil-cpp Bazel dependency to 20260107.1 LTS (#586)",
          "timestamp": "2026-06-15T16:31:54Z",
          "tree_id": "3b32e93144a6947920897c3c354a59cf41e9dcc5",
          "url": "https://github.com/proximafusion/vmecpp/commit/b0876027f1c16d0910a80564118dae42161d0cad"
        },
        "date": 1783604247838,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3477319393999778,
            "range": "stddev: 0.0013925535597838828",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34660039659997893,
            "range": "stddev: 0.0012639151847193173",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7480159343333526,
            "range": "stddev: 0.03592807303173435",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3829343329999424,
            "range": "stddev: 0.016393557181876326",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8995881673333392,
            "range": "stddev: 0.02791657311703356",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.574898020333345,
            "range": "stddev: 0.012841445811187189",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.662871305333321,
            "range": "stddev: 0.01317420654984787",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "559e230b6dde38c018c3fb62994750e2bbef007c",
          "message": "ideal_mhd_model: make computeMHDForces allocation-free (#566)",
          "timestamp": "2026-06-17T09:52:05+02:00",
          "tree_id": "33bc7ad8a8a303b6a6c44a48d490975675a44ddd",
          "url": "https://github.com/proximafusion/vmecpp/commit/559e230b6dde38c018c3fb62994750e2bbef007c"
        },
        "date": 1783604247714,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3475284878000139,
            "range": "stddev: 0.000818073196923465",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34682922179999875,
            "range": "stddev: 0.0006381994698135346",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6359254646666084,
            "range": "stddev: 0.013279420581190906",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.356277637666684,
            "range": "stddev: 0.00432831031552762",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.863281490333293,
            "range": "stddev: 0.014500765450227006",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.531275373666707,
            "range": "stddev: 0.07623442326316529",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.662915695333292,
            "range": "stddev: 0.015552541378601827",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "73f541fb66f34267d3dffcd45c8682b01aa9813c",
          "message": "Make the iteration hot loop allocation-free (#595)",
          "timestamp": "2026-06-17T17:42:06-04:00",
          "tree_id": "cda00ec27b0a41c3214f6c10e4a9bf8d0f0e73d2",
          "url": "https://github.com/proximafusion/vmecpp/commit/73f541fb66f34267d3dffcd45c8682b01aa9813c"
        },
        "date": 1783604247744,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.36014288000001216,
            "range": "stddev: 0.0051096491355913",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3552260739999838,
            "range": "stddev: 0.005344778809123012",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.703185769333383,
            "range": "stddev: 0.0043196322250616094",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.380064431000013,
            "range": "stddev: 0.023255783986478276",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.0391687180000035,
            "range": "stddev: 0.09691137311507823",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.82524902166665,
            "range": "stddev: 0.05707834060744081",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6688730113334411,
            "range": "stddev: 0.011880236898275219",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7d186f2c1ac2f889f9778d292a58f4158b10b0e3",
          "message": "iteration: fix post-reguess lambda scaling divergence, add Python multigrid driver (#560)",
          "timestamp": "2026-06-23T19:23:01-04:00",
          "tree_id": "93262fcd05de06ef9ce49df381d121b66eccd959",
          "url": "https://github.com/proximafusion/vmecpp/commit/7d186f2c1ac2f889f9778d292a58f4158b10b0e3"
        },
        "date": 1783604247754,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35312167560005037,
            "range": "stddev: 0.005286144749873287",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34859277180007664,
            "range": "stddev: 0.001811896864393607",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7017138323334013,
            "range": "stddev: 0.015291324949421927",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.338318641333293,
            "range": "stddev: 0.018159337598690695",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.914207402000026,
            "range": "stddev: 0.07737378522966806",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.595461303333346,
            "range": "stddev: 0.029091458124066487",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6694041016666386,
            "range": "stddev: 0.008180561610727183",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "2e7c3d039e5571c3e8f3a2f17207cfa8f5903e84",
          "message": "enzyme: opt-in Clang/Enzyme build option + AD smoke test (#565)",
          "timestamp": "2026-06-24T08:42:22+02:00",
          "tree_id": "813c275e8dbd0aa8c2f0bc854d6e614292584fab",
          "url": "https://github.com/proximafusion/vmecpp/commit/2e7c3d039e5571c3e8f3a2f17207cfa8f5903e84"
        },
        "date": 1783604247677,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35420603720003785,
            "range": "stddev: 0.005238486271089121",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.36926067380004496,
            "range": "stddev: 0.006464564174212067",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7069006599999739,
            "range": "stddev: 0.014816515028470484",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.369024070000023,
            "range": "stddev: 0.007566828061375807",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 4.009709399333285,
            "range": "stddev: 0.048451614120047344",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.567729257333363,
            "range": "stddev: 0.054779512206087455",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6686564213333288,
            "range": "stddev: 0.012751101701243167",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "776d247ba8021d6be42573c4660bd973f346c13a",
          "message": "Migrate large cpp tests to the main repository (#596)",
          "timestamp": "2026-06-25T16:27:28+02:00",
          "tree_id": "cdbb8a7eda60cd8ae4970b3f1f88bcdea908d11e",
          "url": "https://github.com/proximafusion/vmecpp/commit/776d247ba8021d6be42573c4660bd973f346c13a"
        },
        "date": 1783604247747,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3742296965999685,
            "range": "stddev: 0.0034491495275418968",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3695825601999786,
            "range": "stddev: 0.00264489452900075",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6972249533333372,
            "range": "stddev: 0.003570523542076213",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.398203075000083,
            "range": "stddev: 0.009964234258644603",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.839535380333397,
            "range": "stddev: 0.033993731495861546",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.239327502333404,
            "range": "stddev: 0.009446623075943274",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5569571436667502,
            "range": "stddev: 0.010712841023805342",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b25cfdde81133d5420f7e87744f0b392952d5861",
          "message": "Make the lambda Fourier resolution independent of the geometry (#598)",
          "timestamp": "2026-06-26T16:01:27-04:00",
          "tree_id": "797b573c496ca778c519d25d96e62b0c37626d56",
          "url": "https://github.com/proximafusion/vmecpp/commit/b25cfdde81133d5420f7e87744f0b392952d5861"
        },
        "date": 1783604247843,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3734904857999936,
            "range": "stddev: 0.002942286820168852",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.376996985400001,
            "range": "stddev: 0.003928897601274389",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.7092148510000698,
            "range": "stddev: 0.0037903383461529855",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.398624092666599,
            "range": "stddev: 0.01921635424461097",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.902127201999974,
            "range": "stddev: 0.04264511966869249",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.247591063999911,
            "range": "stddev: 0.037222222098813644",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.555052044999987,
            "range": "stddev: 0.012165324277989503",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "484b8fe6e4637e0102bb201926eccd6ebe7dd2e4",
          "message": "Update bazel lock (#604)",
          "timestamp": "2026-07-06T11:26:00+02:00",
          "tree_id": "ece6e2bcf41bbd1e91c7a4448987c5cf5ac2049c",
          "url": "https://github.com/proximafusion/vmecpp/commit/484b8fe6e4637e0102bb201926eccd6ebe7dd2e4"
        },
        "date": 1783604247702,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3769140669999615,
            "range": "stddev: 0.0025040412196535335",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.37419763379998583,
            "range": "stddev: 0.0029072912878929983",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.712262894666689,
            "range": "stddev: 0.013742445213950318",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.400739174666645,
            "range": "stddev: 0.013045456300501385",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.9303921976667575,
            "range": "stddev: 0.03690425872243059",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.279904338333305,
            "range": "stddev: 0.008573291474396592",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5671875416666505,
            "range": "stddev: 0.015929767598271558",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d0e474d4230f3d421a34715808e10e4f356d9f5e",
          "message": "Benchmark CI action remove code duplication (#605)",
          "timestamp": "2026-07-08T00:34:04+02:00",
          "tree_id": "7a02f9ebab9973f3b10d0358b82819547ff89b91",
          "url": "https://github.com/proximafusion/vmecpp/commit/d0e474d4230f3d421a34715808e10e4f356d9f5e"
        },
        "date": 1783604247894,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3852493721999963,
            "range": "stddev: 0.0020209240725248137",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3840600600000016,
            "range": "stddev: 0.004264862413261546",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.705766123666611,
            "range": "stddev: 0.007118936043675135",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.406219250333303,
            "range": "stddev: 0.02600822867691598",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.981388715666602,
            "range": "stddev: 0.014753877751571541",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.287233328333286,
            "range": "stddev: 0.012538170667922476",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5579656393333607,
            "range": "stddev: 0.013385479238257768",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "93456948ee947ff235d9bc22fc8ad074ec7a8978",
          "message": "Continue to higher multigrid resolutions from hot restart (#603)",
          "timestamp": "2026-07-08T00:34:44+02:00",
          "tree_id": "c25024a7689b4e887598aa33971acd52d1d28f13",
          "url": "https://github.com/proximafusion/vmecpp/commit/93456948ee947ff235d9bc22fc8ad074ec7a8978"
        },
        "date": 1783604247791,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.37946772160003095,
            "range": "stddev: 0.0037450478397191727",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.37923596239998003,
            "range": "stddev: 0.004913784838359745",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6995526669998071,
            "range": "stddev: 0.006423044979255954",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.4709070489999854,
            "range": "stddev: 0.06454774924711223",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8583046186666556,
            "range": "stddev: 0.028216979542647794",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.203278104333398,
            "range": "stddev: 0.02212132442037606",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.574880824333377,
            "range": "stddev: 0.04010262600391496",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "6d4a597cb2654a8f8ca5689cb2a238c567624d3c",
          "message": "Python-side resolution continuation (#544)",
          "timestamp": "2026-07-07T19:01:18-04:00",
          "tree_id": "aaccd48ea2a365ca0f8e62d108f9b107ddcfca5c",
          "url": "https://github.com/proximafusion/vmecpp/commit/6d4a597cb2654a8f8ca5689cb2a238c567624d3c"
        },
        "date": 1783604247739,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3703277768000589,
            "range": "stddev: 0.0033747926723803833",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.37043987000006384,
            "range": "stddev: 0.0022162425242764072",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6930525746667324,
            "range": "stddev: 0.0027235029292154386",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3835345259999485,
            "range": "stddev: 0.025731073268072342",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8456722666664973,
            "range": "stddev: 0.005570606171094114",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.210341400333315,
            "range": "stddev: 0.022053416974727964",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5519681789999897,
            "range": "stddev: 0.016156950155223837",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "f5b7ac74dc86f226bfb74060f7865784d29cbf78",
          "message": "ideal_mhd_model: share the Jacobian kernel with exact autodiff (fwd vs rev) (#567)",
          "timestamp": "2026-07-08T01:21:52+02:00",
          "tree_id": "1cb265e12f30abd17a9e48f5b70d530ab62ab9bf",
          "url": "https://github.com/proximafusion/vmecpp/commit/f5b7ac74dc86f226bfb74060f7865784d29cbf78"
        },
        "date": 1783604247931,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3540689509999993,
            "range": "stddev: 0.001369666637769654",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.35280560540004446,
            "range": "stddev: 0.002308433587542325",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6724263653332703,
            "range": "stddev: 0.006037456787071679",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3741968930000135,
            "range": "stddev: 0.029021465288193507",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8777289180000025,
            "range": "stddev: 0.027568984817669326",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.562172560999974,
            "range": "stddev: 0.005283706436181407",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6642784713333185,
            "range": "stddev: 0.014001616388513391",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "07ba4fdd96acd6e5b44d62e9c8ccdcbeef493273",
          "message": "ideal_mhd_model: share the metric kernel (gsqrt, guu, guv, gvv) (#568)",
          "timestamp": "2026-07-08T02:45:04+02:00",
          "tree_id": "462646ea4dbacc91f75b7d99278e90b96dee1a0e",
          "url": "https://github.com/proximafusion/vmecpp/commit/07ba4fdd96acd6e5b44d62e9c8ccdcbeef493273"
        },
        "date": 1783604247639,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.349618794800017,
            "range": "stddev: 0.0022592015174313284",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3504325620000145,
            "range": "stddev: 0.0023220098815688642",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6705741273333388,
            "range": "stddev: 0.0076969718031650565",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.383943820333343,
            "range": "stddev: 0.022336303474118645",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8608496739999887,
            "range": "stddev: 0.026273642516529867",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.526125513333303,
            "range": "stddev: 0.01716691921841237",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6701018973333248,
            "range": "stddev: 0.019639150454213033",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "10938d76de9e7d96d5913d43abef20e5b7049646",
          "message": "ideal_mhd_model: share the contravariant-field kernel (bsupu, bsupv) (#569)",
          "timestamp": "2026-07-08T08:21:57+02:00",
          "tree_id": "5d0e9ccf0341e22da884a9204450edf363d88016",
          "url": "https://github.com/proximafusion/vmecpp/commit/10938d76de9e7d96d5913d43abef20e5b7049646"
        },
        "date": 1783604247650,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3520365864000269,
            "range": "stddev: 0.0011837653373134025",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.36147838299998514,
            "range": "stddev: 0.008725977200169734",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6612443196667452,
            "range": "stddev: 0.005730211900593901",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.329014811000055,
            "range": "stddev: 0.01628826066239636",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.807933345666773,
            "range": "stddev: 0.018850937898920794",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.507845189999975,
            "range": "stddev: 0.03989479347499869",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.665235508999937,
            "range": "stddev: 0.011838059376134344",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d1c2328aee3ee8e7613ebcba1b023bcbcf041454",
          "message": "Add C++ Google Benchmark microbenchmarks for critical hot functions (#606)",
          "timestamp": "2026-07-08T08:48:02+02:00",
          "tree_id": "ffb12cfbe7d56b38c4799b2acec380d6a211006f",
          "url": "https://github.com/proximafusion/vmecpp/commit/d1c2328aee3ee8e7613ebcba1b023bcbcf041454"
        },
        "date": 1783604247896,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35673363479991166,
            "range": "stddev: 0.002277117700751577",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3526396492000458,
            "range": "stddev: 0.0020924535686699425",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6604333266666345,
            "range": "stddev: 0.0010579065062082157",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.320745666999983,
            "range": "stddev: 0.006551646326305574",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8765656813332803,
            "range": "stddev: 0.027089187840111998",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.503791134666775,
            "range": "stddev: 0.021843939811358248",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6649474819999643,
            "range": "stddev: 0.013357473722166358",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c55210cb72406cfe184665b7bdf8832936426543",
          "message": "ideal_mhd_model: share the covariant-field kernel (bsubu, bsubv) (#570)",
          "timestamp": "2026-07-08T08:50:13+02:00",
          "tree_id": "e8171646866b31b130b72d4100fe89b85bfd7838",
          "url": "https://github.com/proximafusion/vmecpp/commit/c55210cb72406cfe184665b7bdf8832936426543"
        },
        "date": 1783604247856,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.34932990719998996,
            "range": "stddev: 0.0012075738859214135",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.34937398540000686,
            "range": "stddev: 0.003619518688998026",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.661755666666674,
            "range": "stddev: 0.002173727332253369",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3853404113332695,
            "range": "stddev: 0.03057539725205914",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8733037856666215,
            "range": "stddev: 0.009309461063014274",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.51274887500002,
            "range": "stddev: 0.012589911451548823",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6652095203333677,
            "range": "stddev: 0.013980531864144554",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d2033339109a71943f87d81046b6acac19763ce0",
          "message": "ideal_mhd_model: share the magnetic-pressure kernel (#571)",
          "timestamp": "2026-07-08T09:08:12+02:00",
          "tree_id": "5034e4f34354d10e6f9d9ebbef2cf71276f069d9",
          "url": "https://github.com/proximafusion/vmecpp/commit/d2033339109a71943f87d81046b6acac19763ce0"
        },
        "date": 1783604247899,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3509956520000287,
            "range": "stddev: 0.0011876855347328979",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3487768127999516,
            "range": "stddev: 0.0032418042385145866",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6657925380000052,
            "range": "stddev: 0.00269355511644131",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.338472584000025,
            "range": "stddev: 0.010434482460593102",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.885013921000033,
            "range": "stddev: 0.051611938425702025",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.52252108633335,
            "range": "stddev: 0.014060509507637157",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6657190790000034,
            "range": "stddev: 0.01480052975561255",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "acd99d524bc9ca92528cc031bc5d5518feb33259",
          "message": "ideal_mhd_model: share the MHD force-density kernel (6th/last) (#572)",
          "timestamp": "2026-07-08T09:30:41+02:00",
          "tree_id": "909aa733d849bf68c7e22d8da94d11507e914ecf",
          "url": "https://github.com/proximafusion/vmecpp/commit/acd99d524bc9ca92528cc031bc5d5518feb33259"
        },
        "date": 1783604247835,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3546082106000085,
            "range": "stddev: 0.0017862874248500474",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.35410369239998546,
            "range": "stddev: 0.0017678869668037462",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6797592740000102,
            "range": "stddev: 0.005584558864997532",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3461609613333017,
            "range": "stddev: 0.019112467930940936",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.9283119203333094,
            "range": "stddev: 0.060167378535876705",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.595195381666676,
            "range": "stddev: 0.026356657319083376",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6671802599999712,
            "range": "stddev: 0.014268514833652879",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Wenyin Wei",
            "email": "wenyin.wei.ww@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "99d2e45ea900bcf817edfe841abdac53f92de5fb",
          "message": "Fix current density in magnetic field visualization example (#607)",
          "timestamp": "2026-07-08T15:48:59+08:00",
          "tree_id": "5e3ee6c5b68a715dfc9b315615daaffaafe9e473",
          "url": "https://github.com/proximafusion/vmecpp/commit/99d2e45ea900bcf817edfe841abdac53f92de5fb"
        },
        "date": 1783604247799,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.36497228860002906,
            "range": "stddev: 0.0035036919052372924",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.36455798680001406,
            "range": "stddev: 0.003766869278093796",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.714273043999962,
            "range": "stddev: 0.010965656854370651",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.344840813000019,
            "range": "stddev: 0.020224351346844353",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.9029167186666505,
            "range": "stddev: 0.022919335204268265",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.630664544000032,
            "range": "stddev: 0.017334473185708552",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6677918819999984,
            "range": "stddev: 0.01140876767281423",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ca0156935152b9d594f2223916985897c9d3ce9b",
          "message": "enzyme: exact Hessian of the composed local force map (#573)",
          "timestamp": "2026-07-08T09:49:17+02:00",
          "tree_id": "3f61c23861be10fb2def771ff724e1f0aba592c8",
          "url": "https://github.com/proximafusion/vmecpp/commit/ca0156935152b9d594f2223916985897c9d3ce9b"
        },
        "date": 1783604247877,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3919811848000336,
            "range": "stddev: 0.01903617694116188",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.37780719920001504,
            "range": "stddev: 0.007662955399127839",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.670903302333348,
            "range": "stddev: 0.00460175574880893",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3574532530000547,
            "range": "stddev: 0.04490369622724899",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.9034678970000036,
            "range": "stddev: 0.017262827712967187",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.613225110999943,
            "range": "stddev: 0.006939674594406505",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6710372286665915,
            "range": "stddev: 0.011438837144750883",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "888163596accf7515c0a55eb28affd33ba45c048",
          "message": "ideal_mhd_model: share the hybrid lambda-force kernel (#574)",
          "timestamp": "2026-07-08T10:09:55+02:00",
          "tree_id": "94572c488fe96cabd2e5e3e30c1e88ff0ce12d54",
          "url": "https://github.com/proximafusion/vmecpp/commit/888163596accf7515c0a55eb28affd33ba45c048"
        },
        "date": 1783604247772,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3598093405999407,
            "range": "stddev: 0.002907885855344302",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.358822728000041,
            "range": "stddev: 0.006269753928682135",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.674333711333323,
            "range": "stddev: 0.013642376763464989",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.337782290333356,
            "range": "stddev: 0.029250277724342886",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.88711330033334,
            "range": "stddev: 0.0503577873536418",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.553217710333229,
            "range": "stddev: 0.012942292375351232",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6718189516667128,
            "range": "stddev: 0.013437551861085728",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "f36757d8ec444b5f5ba0d2211442f9af5b5cccfc",
          "message": "ideal_mhd_model: share the constraint-force kernels (#575)",
          "timestamp": "2026-07-08T11:20:18+02:00",
          "tree_id": "8a175395c8e040b020b87da2468d50fa80eeceec",
          "url": "https://github.com/proximafusion/vmecpp/commit/f36757d8ec444b5f5ba0d2211442f9af5b5cccfc"
        },
        "date": 1783604247925,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.36178328560004047,
            "range": "stddev: 0.004130680102210417",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3614596478000294,
            "range": "stddev: 0.0026547336694828258",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6780827076666658,
            "range": "stddev: 0.0033272712808492893",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3758293986666104,
            "range": "stddev: 0.017484539520010806",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.9525327489999804,
            "range": "stddev: 0.032259607309488604",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.500221146666641,
            "range": "stddev: 0.04273651195095964",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6752306540000366,
            "range": "stddev: 0.010847175131016944",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9de6072722f45ab6ad10b1d315baa05115137e9a",
          "message": "Merge fast_poloidal and fast_toroidal Fourier bases into one template (#611)",
          "timestamp": "2026-07-08T09:20:31-04:00",
          "tree_id": "a2d28fa6321018bda16919a4bf4a1aea2adffc3e",
          "url": "https://github.com/proximafusion/vmecpp/commit/9de6072722f45ab6ad10b1d315baa05115137e9a"
        },
        "date": 1783604247806,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.35725293040009093,
            "range": "stddev: 0.0008739023381207104",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.35486596799996734,
            "range": "stddev: 0.002457962547320972",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6659717806666474,
            "range": "stddev: 0.005306294652774926",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.342486016333396,
            "range": "stddev: 0.01805775944904309",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8264746903332707,
            "range": "stddev: 0.028150687005476304",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.544700249666676,
            "range": "stddev: 0.016731868804875677",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.6661765116665872,
            "range": "stddev: 0.012760603684887638",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "902d93092f4836c542ee9e6042c5acf3432a05d3",
          "message": "pybind: expose the unpreconditioned internal-basis gradient (#577)",
          "timestamp": "2026-07-08T20:12:48+02:00",
          "tree_id": "1d2259537eb1696ba807972e3fa3a69aed39b7ed",
          "url": "https://github.com/proximafusion/vmecpp/commit/902d93092f4836c542ee9e6042c5acf3432a05d3"
        },
        "date": 1783604247788,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.3873007341999937,
            "range": "stddev: 0.003517176404532911",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.38549243679998424,
            "range": "stddev: 0.012275817089099783",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6386366793333498,
            "range": "stddev: 0.00782834153341839",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.366698501999963,
            "range": "stddev: 0.046080207968501104",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.825544923333325,
            "range": "stddev: 0.024675949294155827",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.250368576666668,
            "range": "stddev: 0.0322523109052952",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5643511196666584,
            "range": "stddev: 0.026145900707132547",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cd3dfc2d6facd85e77f076120b479ad1c92a5dbc",
          "message": "Clean up benchmark on demandplots (#615)",
          "timestamp": "2026-07-08T23:12:19+02:00",
          "tree_id": "ab8b5c4ded4e43c9df5c47f8b23e03b29fe692e2",
          "url": "https://github.com/proximafusion/vmecpp/commit/cd3dfc2d6facd85e77f076120b479ad1c92a5dbc"
        },
        "date": 1783604247885,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.36492685600001096,
            "range": "stddev: 0.0019318605350626054",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3656056374000173,
            "range": "stddev: 0.003755453944332879",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6351496573333104,
            "range": "stddev: 0.004465944927482612",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3838445626666953,
            "range": "stddev: 0.024116442282378487",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.80217148600002,
            "range": "stddev: 0.0463907112790665",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.208666440000002,
            "range": "stddev: 0.09286544705895217",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5453473863333709,
            "range": "stddev: 0.011749938132244078",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7c811775db796064e7f36e54a6ecedf0258d9ed0",
          "message": "Report less, more targeted results in FFT bench (#617)",
          "timestamp": "2026-07-09T11:25:07+02:00",
          "tree_id": "64bbb7827ce86119c7c8e7a5fcb32ce919582512",
          "url": "https://github.com/proximafusion/vmecpp/commit/7c811775db796064e7f36e54a6ecedf0258d9ed0"
        },
        "date": 1783604247752,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.36461982760001777,
            "range": "stddev: 0.0008488764405678706",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.36417921720001234,
            "range": "stddev: 0.0015758468265879302",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6332609583333806,
            "range": "stddev: 0.00301554230744372",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3515131153333564,
            "range": "stddev: 0.03502296746735608",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.7910803040000096,
            "range": "stddev: 0.02701778032267226",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.187676621999989,
            "range": "stddev: 0.031297888802353235",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5486326940000101,
            "range": "stddev: 0.011882466636687315",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "686bd269a67c9e0ab0bf5e8b30c23ed97a38486d",
          "message": "Honor iteration_style=parvmec in the native solver (#612)",
          "timestamp": "2026-07-09T08:15:48-04:00",
          "tree_id": "cabfc258e275f96602f96983d37d7c8f9b16915d",
          "url": "https://github.com/proximafusion/vmecpp/commit/686bd269a67c9e0ab0bf5e8b30c23ed97a38486d"
        },
        "date": 1783604247732,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.37457303299997874,
            "range": "stddev: 0.0038277284251547507",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.37320478620001724,
            "range": "stddev: 0.00443287631835257",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.6749463483333404,
            "range": "stddev: 0.01179208777323452",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.358436452000054,
            "range": "stddev: 0.009675165209345514",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.824429056333315,
            "range": "stddev: 0.062093767869090884",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.218261985666649,
            "range": "stddev: 0.02103659874811191",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5517681666666476,
            "range": "stddev: 0.012573358600893423",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "e0f1985e4ff668a5c80b92561bb610d93b59f861",
          "message": "Abseil status handling for mgrid errors (#613)",
          "timestamp": "2026-07-09T14:23:00+02:00",
          "tree_id": "f772e916641ec1e556c4a112c885018f6c8f57a6",
          "url": "https://github.com/proximafusion/vmecpp/commit/e0f1985e4ff668a5c80b92561bb610d93b59f861"
        },
        "date": 1783604247915,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "unit": "seconds",
            "value": 0.38333449979998024,
            "range": "stddev: 0.006176094065228246",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "unit": "seconds",
            "value": 0.3796252110000296,
            "range": "stddev: 0.0028710715154384252",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "unit": "seconds",
            "value": 1.662361120333306,
            "range": "stddev: 0.0009913713673956315",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "unit": "seconds",
            "value": 2.3821971883333313,
            "range": "stddev: 0.020318119324778422",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "unit": "seconds",
            "value": 3.8258576886666256,
            "range": "stddev: 0.016042909364381762",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "unit": "seconds",
            "value": 9.310848989666738,
            "range": "stddev: 0.03175019553497513",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "unit": "seconds",
            "value": 1.5474930919999679,
            "range": "stddev: 0.012032673192996491",
            "extra": "rounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "albert@tugraz.at",
            "name": "Christopher Albert",
            "username": "krystophny"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2449ddc9c59d0a4cbc676fbe30b0dc1dfdc72089",
          "message": "pybind: Hessian-vector product inside VMEC++ + internal Newton-Krylov (#580)\n\n* build: bump CMake abseil pin to 20260107.1 for Clang >= 21\n\nThe CMake FetchContent abseil pin (2024-08) fails to compile under\nClang >= 21: absl::Nonnull SFINAE in absl/strings/ascii.cc and the\nnumbers.cc nullability annotations are rejected by the newer frontend.\nBump to the 20260107.1 LTS, which compiles cleanly under Clang 21.1.8\nand GCC. Clang is the compiler required for the Enzyme autodiff build.\n\nThe Bazel build keeps its own (BCR) abseil pin and is unaffected.\n\n* enzyme: opt-in Clang/Enzyme build option and AD smoke test\n\nAdd VMECPP_ENABLE_ENZYME (OFF by default), which requires a Clang\ncompiler and a ClangEnzyme plugin path and builds a self-contained\nautodiff smoke test. The test differentiates a scalar objective written\nover Eigen::Map'd caller buffers and checks reverse- and forward-mode\nEnzyme gradients against the closed form and central finite differences.\n\nenzyme.h documents the intrinsic ABI and the allocation constraint that\nshapes the differentiable kernels: Enzyme cannot track Eigen's aligned\nallocator, so differentiable paths use Eigen::Map over caller-owned\nbuffers and avoid heap expression temporaries.\n\nWith the option off the build is unchanged.\n\n* pybind: expose the unpreconditioned internal-basis gradient\n\nAdd a precondition flag to VmecModel.evaluate (default true, unchanged\nbehaviour). With precondition=false the forward model returns at the\nINVARIANT_RESIDUALS checkpoint, so get_forces() yields the raw,\nunpreconditioned force: the gradient of VMEC's augmented functional (MHD\nenergy plus the spectral-condensation and lambda constraints) with\nrespect to the decomposed internal-basis state.\n\nThis is the consistent state/gradient pair an external optimizer needs\nto minimise in VMEC's own basis. The native solver's preconditioned\nsearch direction (precondition=true) is a different vector; the raw\ngradient is the equilibrium residual and vanishes at convergence.\n\nTests: raw force is finite and differs in direction from the\npreconditioned force, and drops by >1e6 from the initial guess to the\nconverged equilibrium.\n\n* examples: drive VMEC++ from external optimizers in the internal basis\n\nTreat the equilibrium as the root problem F(x) = 0, where F is the raw\ninternal-basis force (gradient of VMEC's augmented functional) exposed by\nevaluate(precondition=False). Wire it to two solvers that reuse VMEC++'s\nforward model: native-style preconditioned descent and Jacobian-free\nNewton-Krylov (matrix-free Hessian information). Both reach the native\nsolver's equilibrium.\n\nThis is the external-differentiability path: VMEC++ as a differentiable\nequilibrium component an outside optimizer can drive. Quasi-Newton\nroot-finders without a preconditioner diverge on this stiff system, which\nmotivates exposing VMEC's preconditioner as an operator next.\n\nTests assert both solvers reach force balance and recover the native\nenergy and state.\n\n* pybind: expose VMEC preconditioner as an operator; preconditioned JFNK\n\nAdd VmecModel.apply_preconditioner(v): applies VMEC's preconditioner\nM^-1 (m=1, radial, lambda steps) to a vector in the decomposed basis.\nM^-1 is VMEC's hand-built approximate inverse Hessian; this exposes it\nas a reusable linear operator for preconditioned Krylov / quasi-Newton\nand for the Hessian solve in adjoint sensitivities. It requires a prior\nevaluate(precondition=true), which assembles the radial preconditioner.\n\nValidated exactly: apply_preconditioner(raw force) equals the native\npreconditioned search direction; the operator is linear and, once\nassembled, state-invariant.\n\nUse it as the inner Krylov preconditioner in Newton-Krylov: on solovev\n(ns=11) this cuts force evaluations from 2242 to 505 (4.4x) versus\nunpreconditioned JFNK, converging to the same equilibrium.\n\n* pybind: Hessian-vector product inside VMEC++; internal Newton-Krylov\n\nAdd VmecModel.hessian_vector_product(v): the curvature of VMEC's\naugmented functional, computed inside VMEC++ as a central directional\nderivative of the analytic force (its gradient). The force is exact; only\nthe directional step is finite-differenced. Add a force_eval_count for\nfair cross-optimizer cost comparison (counts evaluations hidden in the\nHessian-vector products).\n\nDrive a true Newton-Krylov from this HVP plus the preconditioner: it\nreaches the equilibrium in ~7 outer iterations (second order) versus\n~1300 descent steps. This is the inside-the-solver Hessian path; together\nwith the external optimizers it gives differentiability inside and out.\n\nBenchmark (solovev, ns=11, force evals counted in VMEC++):\n  preconditioned descent          2606 evals  1302 iters\n  Newton-Krylov (JFNK)            2243 evals\n  Newton-Krylov (preconditioned)   507 evals\n  Newton (VMEC++ HVP + M^-1)      9194 evals     7 iters\n\nThe HVP-Newton's higher force-eval count (two evals per finite-difference\nHVP) is what the exact Enzyme Hessian will remove.\n\n* ideal_mhd_model: make computeMHDForces allocation-free\n\nThe force kernel allocated 17 dynamic Eigen vectors per radial surface (the\n_o half-grid quantities and the avg/wavg surface averages). Move them to\npreallocated per-thread ThreadLocalStorage scratch and assign in place, so\nthe radial loop allocates nothing.\n\nTwo benefits: it removes per-surface heap churn from the hot force loop, and\nit makes the kernel differentiable by Enzyme, which cannot trace dynamic\nEigen temporaries (forward and reverse mode both abort on them). This is the\nallocation-free prerequisite for an exact autodiff Hessian.\n\nPure refactor, identical arithmetic. Verified bit-for-bit: vmec_standalone\nMHD energy unchanged on solovev (2.548352e+00) and cth_like_fixed_bdy\n(5.057191e-02).\n\n* examples: globalize HVP Newton with a backtracking line search\n\nThe full Newton step overshoots on stiff 3D equilibria (cth_like stalled\nat the iteration cap with ||F|| ~ 5e-2). Add a backtracking line search on\n||F|| so each step is damped to a decrease. With it the HVP-Newton\nconverges on cth_like in 9 outer iterations (||F|| = 1.8e-10) and still\nconverges solovev in 8.\n\n* dft_toroidal: make ForcesToFourier allocation-free\n\nThe forces transform materialized two per-(surface,m,zeta) Eigen temporaries\n(tempR_seg, tempZ_seg) inside the inner loop. Reuse per-thread scratch\ninstead, so the whole FFTX-off force path (geometryFromFourier,\ncomputeJacobian/Metric/BContra/BCo, pressureAndEnergies, computeMHDForces,\nforcesToFourier) is now allocation-free end to end.\n\nSame arithmetic as the previous .eval(); verified bit-for-bit: solovev\n2.548352e+00, cth_like_fixed_bdy 5.057191e-02.\n\n* enzyme: exact autodiff of the VMEC Jacobian kernel (forward vs reverse)\n\nDemonstrate exact automatic differentiation of a real VMEC nonlinear\nkernel. JacobianKernel reproduces IdealMhdModel::computeJacobian (half-grid\nr12/ru12/zu12/rs/zs and the Jacobian tau), written allocation-free over flat\nbuffers, which is the form Enzyme differentiates.\n\nFor L = 0.5||outputs||^2 the test computes dL/dgeom by reverse mode and the\ndirectional derivative dL.v by forward mode, checks both against central\nfinite differences, and against each other:\n\n  reverse dL.v vs FD : 1.9e-9\n  forward dL.v vs FD : 1.9e-9\n  forward vs reverse : 2.9e-15\n  performance: reverse ~16 us/pass (full gradient), forward ~16 us/pass\n               (one direction)\n\nReverse returns the whole gradient per pass and wins for a scalar gradient;\nforward is the cheaper primitive for a single Jacobian/Hessian-vector\nproduct. tau is nonlinear in the geometry, so this kernel's Jacobian is a\ngenuine building block of the exact MHD force Hessian; the remaining force\nchain follows the same allocation-free pattern.\n\n* ideal_mhd_model: share the Jacobian kernel between solver and autodiff\n\nMove the half-grid Jacobian arithmetic into jacobian_kernel.h\n(ComputeHalfGridJacobian), allocation-free over flat buffers. Production\ncomputeJacobian now calls it (followed by the unchanged Jacobian-sign\ncheck), and the Enzyme forward/reverse test differentiates the same\nkernel: one implementation, no duplication.\n\nBit-exact: vmec_standalone MHD energy unchanged on solovev\n(2.548352e+00) and cth_like_fixed_bdy (5.057191e-02). Autodiff test still\nmatches finite differences and agrees forward vs reverse to 3e-15.\n\n* ideal_mhd_model: share the metric kernel (gsqrt, guu, guv, gvv)\n\nExtract computeMetricElements into the shared, allocation-free kernel\nComputeMetricElements (metric_kernel.h), over flat buffers, and call it\nfrom the solver. guv and the 3D part of gvv are computed only when\nlthreed, matching the original. This is the second force-chain kernel made\nEnzyme-differentiable (composed into the exact Hessian-vector product\nlater), following the Jacobian kernel pattern.\n\nBit-exact: vmec_standalone MHD energy unchanged on solovev (2.548352e+00,\n2D) and cth_like_fixed_bdy (5.057191e-02, 3D path with guv/gvv).\n\n* ideal_mhd_model: share the contravariant-field kernel (bsupu, bsupv)\n\nFactor the bsupu/bsupv arithmetic out of computeBContra into the shared,\nallocation-free kernel ComputeBsupContra (bcontra_kernel.h). The lambda\nnormalization (lamscale, + phi') and the chi'/iota profile and\ntoroidal-current-constraint logic stay in the solver verbatim, since they\nmutate state and update profiles; only the differentiable field arithmetic\nmoves to the shared kernel.\n\nBit-exact across 1 and 4 threads (so the ghost-cell radial partitioning is\nexercised) on solovev (2.548352e+00, 2D) and cth_like_fixed_bdy\n(5.057191e-02, 3D).\n\n* ideal_mhd_model: share the covariant-field kernel (bsubu, bsubv)\n\nExtract the metric index-lowering (bsubu = guu B^u + guv B^v, bsubv = guv\nB^u + gvv B^v; guv absent in 2D) from computeBCo into the shared,\nallocation-free kernel ComputeBCo (bco_kernel.h).\n\nBit-exact across 1 and 4 threads on solovev (2.548352e+00) and\ncth_like_fixed_bdy (5.057191e-02).\n\n* ideal_mhd_model: share the magnetic-pressure kernel\n\nExtract the field-dependent magnetic pressure |B|^2/2 = 0.5(B^u B_u + B^v\nB_v) from pressureAndEnergies into the shared, allocation-free kernel\nComputeMagneticPressure (pressure_kernel.h). The kinetic-pressure profile\nand the energy volume integrals stay in the solver.\n\nBit-exact across 1 and 4 threads on solovev (2.548352e+00) and\ncth_like_fixed_bdy (5.057191e-02). Completes the point-local nonlinear\nforce-chain kernels (Jacobian, metric, B^contra, B_cov, pressure).\n\n* ideal_mhd_model: share the MHD force-density kernel\n\nExtract computeMHDForces' real-space force-density assembly (armn/azmn/\nbrmn/bzmn, and crmn/czmn in 3D, even+odd) into the shared, allocation-free\nkernel ComputeMHDForceDensity (mhdforce_kernel.h). The Eigen arithmetic is\npreserved verbatim over flat-buffer Eigen::Map views with caller-owned\nhandover/average scratch, so it is bit-for-bit identical.\n\nThis is the sixth and final point-local force-chain kernel; the six\n(Jacobian, metric, B^contra, B_cov, pressure, force) now form the local map\ngeometry -> force density, ready to compose into the exact Hessian-vector\nproduct. (This branch also merges the allocation-free force kernel, #12,\nwhich removes the per-surface heap temporaries this extraction relies on.)\n\nBit-exact across 1 and 4 threads on solovev (2.548352e+00) and\ncth_like_fixed_bdy (5.057191e-02).\n\n* enzyme: exact Hessian of the composed local force map\n\nCompose the six shared force-chain kernels (Jacobian, metric, B^contra,\nB_cov, magnetic pressure, MHD force density) into the single local map\ng: real-space geometry -> real-space force density, the nonlinear core of\nVMEC's force. The full MHD force is T^T . g . T with the linear spectral\ntransforms; the exact force Hessian-vector product is therefore\nT^T . J_g . T . v, and this provides J_g by autodiff.\n\nThe new test takes the Jacobian of g by forward and reverse Enzyme modes\nover flat allocation-free buffers, checks both against central finite\ndifferences and against each other, and times one forward Jacobian-vector\npass against the two force evaluations a finite-difference HVP costs.\n\n* ideal_mhd_model: share the hybrid lambda-force kernel\n\nExtract hybridLambdaForce's full-grid lambda force (blmn, and clmn in 3D)\ninto lambda_force_kernel.h (ComputeHybridLambdaForce), shared between the\nsolver and the Enzyme autodiff path. The method drops from 115 lines to a\nsingle kernel call; the OpenMP barriers stay in the method.\n\nThe kernel is allocation-free over flat buffers and preserves the radial\nsweep that carries the inside half-grid point in scratch and shifts it\noutward each surface, plus the blend of the two bsubv interpolations.\n\nThis is the lambda-force piece of the augmented functional, the second\nnonlinear force-density term after the MHD force chain.\n\n* ideal_mhd_model: share the constraint-force kernels\n\nExtract the two local (non-transform) pieces of the spectral-condensation\nconstraint force into constraint_force_kernel.h, shared between the solver\nand the Enzyme autodiff path:\n\n- ComputeEffectiveConstraintForce: gConEff = (rCon-rCon0) ru + (zCon-zCon0) zu\n  (effectiveConstraintForce), skipping the axis surface.\n- AddConstraintForces: add the bandpass-filtered gCon back into the MHD R/Z\n  forces and write frcon/fzcon (the constraint part of assembleTotalForces).\n\nThe Fourier-space bandpass between them stays the shared free function\ndeAliasConstraintForce; the free-boundary rBSq contribution stays in\nassembleTotalForces. Allocation-free over flat buffers.\n\nThis completes the local force-density terms of the augmented functional\n(MHD + lambda + constraint), the nonlinear core of the exact Hessian.\n\n* enzyme: extend the composed-force Hessian test with the lambda force\n\nAdd the hybrid lambda force (lambda_force_kernel.h) to the composed local\nmap g and differentiate the combined MHD-plus-lambda force density by\nforward and reverse Enzyme modes. This proves J_g for the second nonlinear\nforce-density term, not just the MHD force chain.\n\nThe spectral-condensation constraint force also carries a linear Fourier\nbandpass; it is validated end-to-end against the finite-difference HVP in\nthe pybind exact-HVP path rather than in this flat-buffer microtest.\n\n* apply pre-commit formatting (ruff, docformatter, clang-format)\n\n* apply pre-commit formatting (ruff, docformatter, clang-format)\n\n* apply pre-commit formatting (ruff, docformatter, clang-format)\n\n* apply pre-commit formatting (ruff, docformatter, clang-format)\n\n* apply pre-commit formatting (ruff, docformatter, clang-format)\n\n* apply pre-commit formatting (ruff, docformatter, clang-format)\n\n* apply pre-commit formatting (ruff, docformatter, clang-format)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* bazel: declare force-chain kernel headers in ideal_mhd_model (sandbox fix)\n\n* test: docformatter-format test_internal_gradient docstrings\n\nSatisfies the docformatter pre-commit hook (was failing CI).\n\n* test: docformatter-format external/internal optimizer test docstrings\n\nSatisfies the docformatter pre-commit hook (was failing CI).\n\n* ci: re-trigger (transient apt-403 on packages.microsoft.com)\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* ci: skip benchmark result upload on fork PRs (token is read-only)\n\nThe 'Compare benchmark result' step uses github-action-benchmark with\ncomment-on-alert and the GITHUB_TOKEN, which is read-only for pull requests from\nforks -> 'Resource not accessible by integration'. Gate that step on the PR\ncoming from the same repo so fork PRs still run the benchmarks but skip the\nwrite-back instead of failing.\n\n* ci: build VMEC2000 from source so the compat test runs on numpy 2\n\nThe pinned vmec-0.0.6 cp310 wheel was f90wrapped against numpy 1.x. Under\nthe numpy 2.x that the test env now resolves, importing it dies in the\nf90wrap array interface (f90wrap_vmec_input__array__rbc: 0-th dimension\nmust be fixed to 2 but got 4), so test_ensure_vmec2000_input_from_vmecpp_input\ncould never actually run on CI (and is currently red on main too, where the\nwheel's runtime libs are not even installed).\n\nBuild VMEC2000 from upstream source with current f90wrap, which produces\nnumpy-2-compatible bindings. The recipe mirrors SIMSOPT's own CI\n(hiddenSymmetries/VMEC2000, cmake/machines/ubuntu.json). An explicit\n'import vmec' check in the install step surfaces any remaining problem here\nrather than as a confusing test failure.\n\n* test: skip vmecpp-only indata fields in the VMEC2000 compat subset\n\nWith VMEC2000 built from current upstream source, the compatibility test\nruns for the first time and hits vmecpp indata fields that have no\ncounterpart in the legacy VMEC2000 INDATA namelist (e.g.\nfree_boundary_method), which raised AttributeError. The test explicitly\nchecks only the common subset, so guard the lookup with hasattr and skip\nfields VMEC2000 does not have, instead of enumerating them one by one.\n\n* build: pin abseil to the 20260107.1 commit hash\n\nPin the FetchContent abseil dependency to commit 255c84d (the exact\ncommit behind the 20260107.1 LTS tag) instead of the tag itself, so a\nmoved tag cannot change the dependency under us.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash, not the tag.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash for Clang >= 21.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash for Clang >= 21.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash for Clang >= 21.\n\n* ci: sync VMEC2000-from-source build, benchmark fork guard, abseil commit pin\n\nBring this stack branch up to the corrected CI baseline (from #583/#564):\n- tests.yaml: build VMEC2000 from the pinned source commit and cache the\n  wheel; drop the unused FFTW/HDF5 dev packages.\n- benchmarks.yaml: skip the result upload on fork PRs (read-only token).\n- test_simsopt_compat.py: skip vmecpp-only INDATA fields.\n- CMakeLists: pin abseil to the 20260107.1 commit hash for Clang >= 21.\n\n* ci: cache and pin the VMEC2000-from-source build\n\nUse the canonical recipe (cache the built wheel keyed on the pinned\nsource commit 728af8b, drop the unused FFTW/HDF5 dev packages) instead\nof rebuilding VMEC2000 unpinned on every run.\n\n* ideal_mhd_model: mark Jacobian kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* ideal_mhd_model: mark Jacobian  metric kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* ideal_mhd_model: hoist ForcesToFourier scratch out of the inner loop\n\nThe allocation-free rewrite placed tempR_seg/tempZ_seg in a block-scope\nthread_local inside the (jF, m, zeta) inner loop, which emits a\n__tls_get_addr call and an init-guard branch every iteration. Declare\nthe two scratch vectors once at function scope instead: still\nallocation-free in the hot loop and per-thread safe via the stack frame,\nwithout the per-iteration TLS overhead. Same arithmetic; cma and w7x\nwout are bit-for-bit unchanged.\n\n* ideal_mhd_model: mark Jacobian  metric kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* ideal_mhd_model: mark Jacobian  metric kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* ideal_mhd_model: mark Jacobian  metric kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* ideal_mhd_model: hoist ForcesToFourier scratch out of the inner loop\n\nThe allocation-free rewrite placed tempR_seg/tempZ_seg in a block-scope\nthread_local inside the (jF, m, zeta) inner loop, which emits a\n__tls_get_addr call and an init-guard branch every iteration. Declare\nthe two scratch vectors once at function scope instead: still\nallocation-free in the hot loop and per-thread safe via the stack frame,\nwithout the per-iteration TLS overhead. Same arithmetic; cma and w7x\nwout are bit-for-bit unchanged.\n\n* ideal_mhd_model: hoist ForcesToFourier scratch out of the inner loop\n\nThe allocation-free rewrite placed tempR_seg/tempZ_seg in a block-scope\nthread_local inside the (jF, m, zeta) inner loop, which emits a\n__tls_get_addr call and an init-guard branch every iteration. Declare\nthe two scratch vectors once at function scope instead: still\nallocation-free in the hot loop and per-thread safe via the stack frame,\nwithout the per-iteration TLS overhead. Same arithmetic; cma and w7x\nwout are bit-for-bit unchanged.\n\n* ideal_mhd_model: mark Jacobian  metric kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* ideal_mhd_model: mark Jacobian  metric kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* ideal_mhd_model: mark Jacobian  metric kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* ideal_mhd_model: hoist ForcesToFourier scratch out of the inner loop\n\nThe allocation-free rewrite placed tempR_seg/tempZ_seg in a block-scope\nthread_local inside the (jF, m, zeta) inner loop, which emits a\n__tls_get_addr call and an init-guard branch every iteration. Declare\nthe two scratch vectors once at function scope instead: still\nallocation-free in the hot loop and per-thread safe via the stack frame,\nwithout the per-iteration TLS overhead. Same arithmetic; cma and w7x\nwout are bit-for-bit unchanged.\n\n* ideal_mhd_model: hoist ForcesToFourier scratch out of the inner loop\n\nThe allocation-free rewrite placed tempR_seg/tempZ_seg in a block-scope\nthread_local inside the (jF, m, zeta) inner loop, which emits a\n__tls_get_addr call and an init-guard branch every iteration. Declare\nthe two scratch vectors once at function scope instead: still\nallocation-free in the hot loop and per-thread safe via the stack frame,\nwithout the per-iteration TLS overhead. Same arithmetic; cma and w7x\nwout are bit-for-bit unchanged.\n\n* ideal_mhd_model: mark Jacobian  metric kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* ideal_mhd_model: mark Jacobian  metric kernel buffers __restrict\n\nRaw double* kernel params over the same flat layout prevent the compiler\nfrom vectorizing the pointwise loop (assumed aliasing), so on w7x these\nkernels ran ~2x slower than the Eigen-expression code they replaced.\nThe buffers never overlap; mark them __restrict to restore SIMD. Enzyme\nderivatives are unchanged (jacobian_kernel_autodiff + QS GN benchmark).\n\n* output_quantities: compare jcuru/jcurv at a looser opt-in tolerance\n\nThe free-boundary in-memory-vs-disk mgrid golden compares two independent\nsolves. jcuru/jcurv are curl(B) current densities that amplify the rounding\nof the converged state, so under vectorized/optimized builds the two paths\ndiverge by ~1.03e-7 (measured on the CI asan/ubsan runners) while every other\nwout quantity still agrees to 1e-7. The math is unchanged: with vs without the\nkernel __restrict the cth_like wout is bit-for-bit identical on gcc Release, so\nthis is an FP-ordering reproducibility floor, not an accuracy regression.\n\nAdd an opt-in current_density_tolerance to CompareWOut (default 0 = use the\nmain tolerance, so every other caller is unchanged) and have the two\nvmec_in_memory_mgrid_test comparisons pass 2e-7 for jcuru/jcurv only, keeping\n1e-7 for all profiles and geometry.\n\n* output_quantities: compare jcuru/jcurv at a looser opt-in tolerance\n\nThe free-boundary in-memory-vs-disk mgrid golden compares two independent\nsolves. jcuru/jcurv are curl(B) current densities that amplify the rounding\nof the converged state, so under vectorized/optimized builds the two paths\ndiverge by ~1.03e-7 (measured on the CI asan/ubsan runners) while every other\nwout quantity still agrees to 1e-7. The math is unchanged: with vs without the\nkernel __restrict the cth_like wout is bit-for-bit identical on gcc Release, so\nthis is an FP-ordering reproducibility floor, not an accuracy regression.\n\nAdd an opt-in current_density_tolerance to CompareWOut (default 0 = use the\nmain tolerance, so every other caller is unchanged) and have the two\nvmec_in_memory_mgrid_test comparisons pass 2e-7 for jcuru/jcurv only, keeping\n1e-7 for all profiles and geometry.\n\n(cherry picked from commit 27d36d21e1dd8ea6f73127b95bdc81d529f81672)\n\n* output_quantities: compare jcuru/jcurv at a looser opt-in tolerance\n\nThe free-boundary in-memory-vs-disk mgrid golden compares two independent\nsolves. jcuru/jcurv are curl(B) current densities that amplify the rounding\nof the converged state, so under vectorized/optimized builds the two paths\ndiverge by ~1.03e-7 (measured on the CI asan/ubsan runners) while every other\nwout quantity still agrees to 1e-7. The math is unchanged: with vs without the\nkernel __restrict the cth_like wout is bit-for-bit identical on gcc Release, so\nthis is an FP-ordering reproducibility floor, not an accuracy regression.\n\nAdd an opt-in current_density_tolerance to CompareWOut (default 0 = use the\nmain tolerance, so every other caller is unchanged) and have the two\nvmec_in_memory_mgrid_test comparisons pass 2e-7 for jcuru/jcurv only, keeping\n1e-7 for all profiles and geometry.\n\n(cherry picked from commit 27d36d21e1dd8ea6f73127b95bdc81d529f81672)\n\n* output_quantities: compare jcuru/jcurv at a looser opt-in tolerance\n\nThe free-boundary in-memory-vs-disk mgrid golden compares two independent\nsolves. jcuru/jcurv are curl(B) current densities that amplify the rounding\nof the converged state, so under vectorized/optimized builds the two paths\ndiverge by ~1.03e-7 (measured on the CI asan/ubsan runners) while every other\nwout quantity still agrees to 1e-7. The math is unchanged: with vs without the\nkernel __restrict the cth_like wout is bit-for-bit identical on gcc Release, so\nthis is an FP-ordering reproducibility floor, not an accuracy regression.\n\nAdd an opt-in current_density_tolerance to CompareWOut (default 0 = use the\nmain tolerance, so every other caller is unchanged) and have the two\nvmec_in_memory_mgrid_test comparisons pass 2e-7 for jcuru/jcurv only, keeping\n1e-7 for all profiles and geometry.\n\n(cherry picked from commit 27d36d21e1dd8ea6f73127b95bdc81d529f81672)\n\n* output_quantities: compare jcuru/jcurv at a looser opt-in tolerance\n\nThe free-boundary in-memory-vs-disk mgrid golden compares two independent\nsolves. jcuru/jcurv are curl(B) current densities that amplify the rounding\nof the converged state, so under vectorized/optimized builds the two paths\ndiverge by ~1.03e-7 (measured on the CI asan/ubsan runners) while every other\nwout quantity still agrees to 1e-7. The math is unchanged: with vs without the\nkernel __restrict the cth_like wout is bit-for-bit identical on gcc Release, so\nthis is an FP-ordering reproducibility floor, not an accuracy regression.\n\nAdd an opt-in current_density_tolerance to CompareWOut (default 0 = use the\nmain tolerance, so every other caller is unchanged) and have the two\nvmec_in_memory_mgrid_test comparisons pass 2e-7 for jcuru/jcurv only, keeping\n1e-7 for all profiles and geometry.\n\n(cherry picked from commit 27d36d21e1dd8ea6f73127b95bdc81d529f81672)\n\n* output_quantities: compare jcuru/jcurv at a looser opt-in tolerance\n\nThe free-boundary in-memory-vs-disk mgrid golden compares two independent\nsolves. jcuru/jcurv are curl(B) current densities that amplify the rounding\nof the converged state, so under vectorized/optimized builds the two paths\ndiverge by ~1.03e-7 (measured on the CI asan/ubsan runners) while every other\nwout quantity still agrees to 1e-7. The math is unchanged: with vs without the\nkernel __restrict the cth_like wout is bit-for-bit identical on gcc Release, so\nthis is an FP-ordering reproducibility floor, not an accuracy regression.\n\nAdd an opt-in current_density_tolerance to CompareWOut (default 0 = use the\nmain tolerance, so every other caller is unchanged) and have the two\nvmec_in_memory_mgrid_test comparisons pass 2e-7 for jcuru/jcurv only, keeping\n1e-7 for all profiles and geometry.\n\n(cherry picked from commit 27d36d21e1dd8ea6f73127b95bdc81d529f81672)\n\n* output_quantities: compare jcuru/jcurv at a looser opt-in tolerance\n\nThe free-boundary in-memory-vs-disk mgrid golden compares two independent\nsolves. jcuru/jcurv are curl(B) current densities that amplify the rounding\nof the converged state, so under vectorized/optimized builds the two paths\ndiverge by ~1.03e-7 (measured on the CI asan/ubsan runners) while every other\nwout quantity still agrees to 1e-7. The math is unchanged: with vs without the\nkernel __restrict the cth_like wout is bit-for-bit identical on gcc Release, so\nthis is an FP-ordering reproducibility floor, not an accuracy regression.\n\nAdd an opt-in current_density_tolerance to CompareWOut (default 0 = use the\nmain tolerance, so every other caller is unchanged) and have the two\nvmec_in_memory_mgrid_test comparisons pass 2e-7 for jcuru/jcurv only, keeping\n1e-7 for all profiles and geometry.\n\n(cherry picked from commit 27d36d21e1dd8ea6f73127b95bdc81d529f81672)\n\n* output_quantities: compare jcuru/jcurv at a looser opt-in tolerance\n\nThe free-boundary in-memory-vs-disk mgrid golden compares two independent\nsolves. jcuru/jcurv are curl(B) current densities that amplify the rounding\nof the converged state, so under vectorized/optimized builds the two paths\ndiverge by ~1.03e-7 (measured on the CI asan/ubsan runners) while every other\nwout quantity still agrees to 1e-7. The math is unchanged: with vs without the\nkernel __restrict the cth_like wout is bit-for-bit identical on gcc Release, so\nthis is an FP-ordering reproducibility floor, not an accuracy regression.\n\nAdd an opt-in current_density_tolerance to CompareWOut (default 0 = use the\nmain tolerance, so every other caller is unchanged) and have the two\nvmec_in_memory_mgrid_test comparisons pass 2e-7 for jcuru/jcurv only, keeping\n1e-7 for all profiles and geometry.\n\n(cherry picked from commit 27d36d21e1dd8ea6f73127b95bdc81d529f81672)\n\n* output_quantities: compare jcuru/jcurv at a looser opt-in tolerance\n\nThe free-boundary in-memory-vs-disk mgrid golden compares two independent\nsolves. jcuru/jcurv are curl(B) current densities that amplify the rounding\nof the converged state, so under vectorized/optimized builds the two paths\ndiverge by ~1.03e-7 (measured on the CI asan/ubsan runners) while every other\nwout quantity still agrees to 1e-7. The math is unchanged: with vs without the\nkernel __restrict the cth_like wout is bit-for-bit identical on gcc Release, so\nthis is an FP-ordering reproducibility floor, not an accuracy regression.\n\nAdd an opt-in current_density_tolerance to CompareWOut (default 0 = use the\nmain tolerance…",
          "timestamp": "2026-07-09T14:55:50+02:00",
          "tree_id": "5d3bf5206bdeeaa0172c8a07985b9b4acc4e9e33",
          "url": "https://github.com/proximafusion/vmecpp/commit/2449ddc9c59d0a4cbc676fbe30b0dc1dfdc72089"
        },
        "date": 1783604630751,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.3652101238000114,
            "range": "stddev: 0.0053876768972931435",
            "unit": "seconds",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.3631968379999876,
            "range": "stddev: 0.00829215164471065",
            "unit": "seconds",
            "extra": "rounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 3.8679229873333534,
            "range": "stddev: 0.01933744775039798",
            "unit": "seconds",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 1.6569237620000006,
            "range": "stddev: 0.014360794896952477",
            "unit": "seconds",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma_6x8",
            "value": 2.339887266333316,
            "range": "stddev: 0.020977307945756862",
            "unit": "seconds",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 2.3239169386666654,
            "range": "stddev: 0.011272736323205862",
            "unit": "seconds",
            "extra": "rounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 9.51514209033335,
            "range": "stddev: 0.04571966218674155",
            "unit": "seconds",
            "extra": "rounds: 3"
          }
        ]
      }
    ],
    "C++ Microbenchmarks": [
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "365c07eaab0152eb00387f41e821e4686ae3fe6e",
          "message": "Converted ivac counter to an enum with descriptive states (#280)",
          "timestamp": "2025-05-08T16:25:43Z",
          "tree_id": "d76455baa793c81b29b9a6bab932bbaff47f7e26",
          "url": "https://github.com/proximafusion/vmecpp/commit/365c07eaab0152eb00387f41e821e4686ae3fe6e"
        },
        "date": 1783604248037,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002649852207728795,
            "extra": "iterations: 994\ncpu: 0.0002638850573440644 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014145570671154792,
            "extra": "iterations: 1979\ncpu: 0.00014145325113693787 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031692079231563936,
            "extra": "iterations: 885\ncpu: 0.00031688628022598874 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002878943669427302,
            "extra": "iterations: 970\ncpu: 0.00028788988865979397 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005299030945747245,
            "extra": "iterations: 529\ncpu: 0.000529894527410208 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005111099375184552,
            "extra": "iterations: 549\ncpu: 0.0005110622204007283 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0022365379333496097,
            "extra": "iterations: 125\ncpu: 0.0022364240560000005 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024006835415832,
            "extra": "iterations: 117\ncpu: 0.00240064365811966 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "6b8faa83515cd514e01084f1c440611fb5c54b4c",
          "message": "Renamed ivac counter to vacuum_state (#317)",
          "timestamp": "2025-05-08T16:25:44Z",
          "tree_id": "87276ba692180efd37eae86b0e9d5aa9e92799c7",
          "url": "https://github.com/proximafusion/vmecpp/commit/6b8faa83515cd514e01084f1c440611fb5c54b4c"
        },
        "date": 1783604248152,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015635569839428666,
            "extra": "iterations: 1561\ncpu: 0.00015632238500960926 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014223028184943677,
            "extra": "iterations: 1946\ncpu: 0.0001422209743062693 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000316786443137692,
            "extra": "iterations: 886\ncpu: 0.00031678126523702036 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028593644910740294,
            "extra": "iterations: 979\ncpu: 0.0002858923738508682 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005316973635644624,
            "extra": "iterations: 528\ncpu: 0.0005316874507575761 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005135895574794097,
            "extra": "iterations: 544\ncpu: 0.0005135814191176474 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021631717681884766,
            "extra": "iterations: 129\ncpu: 0.002162694387596899 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024174965661147548,
            "extra": "iterations: 116\ncpu: 0.0024174498103448285 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurasic",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d7153695cf9f40f2d270f5c2b3c87b1c7a747477",
          "message": "Pass on verbose flag to ideal mhd",
          "timestamp": "2025-05-07T13:38:22+02:00",
          "tree_id": "6f8ab5405e055f7fd6c9a200d6439c7836b83448",
          "url": "https://github.com/proximafusion/vmecpp/commit/d7153695cf9f40f2d270f5c2b3c87b1c7a747477"
        },
        "date": 1783604248387,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015556570917736693,
            "extra": "iterations: 1778\ncpu: 0.00015555352530933635 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014277459652861223,
            "extra": "iterations: 1957\ncpu: 0.00014274729892692898 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003168882627044367,
            "extra": "iterations: 883\ncpu: 0.00031686759116647794 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028706623957707334,
            "extra": "iterations: 975\ncpu: 0.0002870613928205128 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005290170885481924,
            "extra": "iterations: 530\ncpu: 0.0005289359396226418 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005131020624415312,
            "extra": "iterations: 547\ncpu: 0.0005130709195612433 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021699325058811398,
            "extra": "iterations: 129\ncpu: 0.0021696927286821723 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023853819594423994,
            "extra": "iterations: 117\ncpu: 0.0023848581965811963 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurasic",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "448609b293e4ca1f4d6ad46a66e5f276f6c45825",
          "message": "Clearly show reduction in iteration count for hot-restart",
          "timestamp": "2025-05-11T20:06:21+02:00",
          "tree_id": "71b186ba152c4f5290b4359eedfb38d47d2d4c45",
          "url": "https://github.com/proximafusion/vmecpp/commit/448609b293e4ca1f4d6ad46a66e5f276f6c45825"
        },
        "date": 1783604248074,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015867622282392824,
            "extra": "iterations: 1721\ncpu: 0.00015865352585705986 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014144287940758593,
            "extra": "iterations: 1973\ncpu: 0.00014143312062848455 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000316820306293035,
            "extra": "iterations: 885\ncpu: 0.00031677401129943483 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002921851290624949,
            "extra": "iterations: 943\ncpu: 0.0002921693181336162 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005300465752096738,
            "extra": "iterations: 527\ncpu: 0.0005300368576850098 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005119864958046127,
            "extra": "iterations: 548\ncpu: 0.0005118813832116788 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021809724372202958,
            "extra": "iterations: 127\ncpu: 0.002180933992125982 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002402346709678913,
            "extra": "iterations: 116\ncpu: 0.002402302862068964 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b91ddb712f34c657fc11a06640d7daf1129667e7",
          "message": "Skip gradual vacuum pressure activation on hot-restart (#330)",
          "timestamp": "2025-05-12T17:14:10Z",
          "tree_id": "260ce1451d1cbd83185211304240574f3db9b40c",
          "url": "https://github.com/proximafusion/vmecpp/commit/b91ddb712f34c657fc11a06640d7daf1129667e7"
        },
        "date": 1783604248309,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015589127418594447,
            "extra": "iterations: 1798\ncpu: 0.00015588855116796443 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014388475486700486,
            "extra": "iterations: 1946\ncpu: 0.00014387138797533405 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031666782616221977,
            "extra": "iterations: 885\ncpu: 0.00031666253672316374 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002846756680676991,
            "extra": "iterations: 982\ncpu: 0.00028464691547861506 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005291470072486185,
            "extra": "iterations: 528\ncpu: 0.0005291194753787878 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005116040689231706,
            "extra": "iterations: 548\ncpu: 0.0005115948065693427 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021284981207414107,
            "extra": "iterations: 132\ncpu: 0.002128463553030306 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024172886558201,
            "extra": "iterations: 115\ncpu: 0.002416899286956522 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurasic",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7c32f2ea68f2adfeedfe1d11b41842fe6a304446",
          "message": "Added details to VmecInput documentation",
          "timestamp": "2025-05-12T19:56:36+02:00",
          "tree_id": "55cc0e1d295624a51daed46e426119848e2bacc2",
          "url": "https://github.com/proximafusion/vmecpp/commit/7c32f2ea68f2adfeedfe1d11b41842fe6a304446"
        },
        "date": 1783604248172,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015531880703659268,
            "extra": "iterations: 1805\ncpu: 0.0001553027235457064 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014289289529318682,
            "extra": "iterations: 1948\ncpu: 0.0001428906688911705 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000315326941697214,
            "extra": "iterations: 889\ncpu: 0.00031502797300337454 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002874959576703387,
            "extra": "iterations: 977\ncpu: 0.0002874908618219038 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005329418182373047,
            "extra": "iterations: 525\ncpu: 0.0005329327390476193 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.000511902559827843,
            "extra": "iterations: 547\ncpu: 0.0005117994369287016 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002390233879415398,
            "extra": "iterations: 117\ncpu: 0.002390188743589742 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024201541111387058,
            "extra": "iterations: 116\ncpu: 0.0024201151465517274 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "6b63d4712683803353812b5e26c854915530221f",
          "message": "WOut quantity docstrings (#334)",
          "timestamp": "2025-05-22T09:40:14Z",
          "tree_id": "c005c251d870130f243a57bca36db54746fd587c",
          "url": "https://github.com/proximafusion/vmecpp/commit/6b63d4712683803353812b5e26c854915530221f"
        },
        "date": 1783604248149,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015561995011180855,
            "extra": "iterations: 1801\ncpu: 0.00015559611271515823 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014222873050346782,
            "extra": "iterations: 1969\ncpu: 0.00014222056729304222 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031699344955063734,
            "extra": "iterations: 882\ncpu: 0.00031696770408163244 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028686782110676917,
            "extra": "iterations: 977\ncpu: 0.0002868624483111565 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000541507735732914,
            "extra": "iterations: 516\ncpu: 0.0005414696453488371 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005125283321618161,
            "extra": "iterations: 546\ncpu: 0.000512496514652015 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021593423990102912,
            "extra": "iterations: 130\ncpu: 0.0021590916538461553 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024025093795906785,
            "extra": "iterations: 117\ncpu: 0.002402281965811968 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5408dad503a9b14656cc61219b5435b409a0a139",
          "message": "Disable SZIP support in HDF5: not needed here (#335)",
          "timestamp": "2025-05-27T16:49:44Z",
          "tree_id": "8dd6e0635b417fada61c78c463ab324d7165d71f",
          "url": "https://github.com/proximafusion/vmecpp/commit/5408dad503a9b14656cc61219b5435b409a0a139"
        },
        "date": 1783604248107,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015665700771664613,
            "extra": "iterations: 1788\ncpu: 0.00015664069295302015 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014162616383644843,
            "extra": "iterations: 1984\ncpu: 0.00014162406552419357 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00033583952149604654,
            "extra": "iterations: 889\ncpu: 0.00033583496962879633 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00029014827543244893,
            "extra": "iterations: 969\ncpu: 0.0002901162590299279 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000532235029532429,
            "extra": "iterations: 526\ncpu: 0.0005321660627376427 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005122910450844869,
            "extra": "iterations: 548\ncpu: 0.0005122647226277374 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021611818900475134,
            "extra": "iterations: 130\ncpu: 0.002161072323076923 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002420614505636281,
            "extra": "iterations: 116\ncpu: 0.002420572293103448 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "386a68cfe06f79c38e115e81104c33811b48987b",
          "message": "Add AGENTS.md for working with AI (#338)",
          "timestamp": "2025-06-10T09:20:04Z",
          "tree_id": "7ad8b689d739e4803e475de6c56dfc8cfc2f3bcc",
          "url": "https://github.com/proximafusion/vmecpp/commit/386a68cfe06f79c38e115e81104c33811b48987b"
        },
        "date": 1783604248042,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015640018770358706,
            "extra": "iterations: 1788\ncpu: 0.00015639753579418345 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014217573267812948,
            "extra": "iterations: 1965\ncpu: 0.00014216744936386774 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00032481942241547874,
            "extra": "iterations: 884\ncpu: 0.0003247536380090497 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028718446121841183,
            "extra": "iterations: 976\ncpu: 0.00028717961577868844 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.00053381829550772,
            "extra": "iterations: 528\ncpu: 0.0005338091212121212 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005137009567089295,
            "extra": "iterations: 534\ncpu: 0.0005136260449438197 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0022055107777512917,
            "extra": "iterations: 127\ncpu: 0.00220537631496063 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024192939724838524,
            "extra": "iterations: 114\ncpu: 0.0024192515877193006 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b44fb7f0dc5e817ccba58807586665d1efaafa5f",
          "message": "Add a naming guide for VMEC++ (#339)",
          "timestamp": "2025-06-10T11:08:52Z",
          "tree_id": "8fe6e1412ce33dfa43e5ab3f093e3aee868731a5",
          "url": "https://github.com/proximafusion/vmecpp/commit/b44fb7f0dc5e817ccba58807586665d1efaafa5f"
        },
        "date": 1783604248306,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001558265421125624,
            "extra": "iterations: 1800\ncpu: 0.00015580692777777779 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014163052487023623,
            "extra": "iterations: 1977\ncpu: 0.00014162141527567025 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003155463174607063,
            "extra": "iterations: 887\ncpu: 0.0003155423111612174 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000285670557392643,
            "extra": "iterations: 978\ncpu: 0.0002856541768916157 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005374636622628415,
            "extra": "iterations: 521\ncpu: 0.0005374389404990402 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005153274887394071,
            "extra": "iterations: 543\ncpu: 0.0005153038821362793 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021565311638883844,
            "extra": "iterations: 129\ncpu: 0.0021563320232558143 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002389775382147895,
            "extra": "iterations: 117\ncpu: 0.00238957211111111 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a7797dc5ccbee0541708f452d7b0e63bc6912bf4",
          "message": "Consolidate algorithmic constants into comprehensive constants header (#340)",
          "timestamp": "2025-06-10T17:12:43Z",
          "tree_id": "4eb9cc73e7ac01ca670d4d148027d1a3f38a6784",
          "url": "https://github.com/proximafusion/vmecpp/commit/a7797dc5ccbee0541708f452d7b0e63bc6912bf4"
        },
        "date": 1783604248277,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015586083609702063,
            "extra": "iterations: 1796\ncpu: 0.0001558422488864143 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014303689351420534,
            "extra": "iterations: 1957\ncpu: 0.00014300875523760861 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003154083833856098,
            "extra": "iterations: 885\ncpu: 0.0003154027694915254 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028766167469513725,
            "extra": "iterations: 975\ncpu: 0.0002876452235897437 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005281820135601496,
            "extra": "iterations: 531\ncpu: 0.0005281590207156311 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005149544377030034,
            "extra": "iterations: 546\ncpu: 0.0005149461739926736 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021736323833465576,
            "extra": "iterations: 128\ncpu: 0.0021735942812500003 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024133653476320466,
            "extra": "iterations: 116\ncpu: 0.0024132246637931006 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "39d75227e5aaefdddccfded200ae84e1c926d2aa",
          "message": "migrate m_evn/m_odd to k{Even,Odd}Parity (#341)",
          "timestamp": "2025-06-10T17:12:43Z",
          "tree_id": "4c34af5ab2e3885f2228adc48bed64f9ed7edfaa",
          "url": "https://github.com/proximafusion/vmecpp/commit/39d75227e5aaefdddccfded200ae84e1c926d2aa"
        },
        "date": 1783604248045,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015501130650908665,
            "extra": "iterations: 1805\ncpu: 0.00015500850858725764 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014216448678878405,
            "extra": "iterations: 1971\ncpu: 0.00014216177422628104 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003186906931883653,
            "extra": "iterations: 878\ncpu: 0.00031862022323462424 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028735809679021574,
            "extra": "iterations: 973\ncpu: 0.00028734704933196275 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005299945483850121,
            "extra": "iterations: 527\ncpu: 0.0005299859089184064 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005139980841120449,
            "extra": "iterations: 545\ncpu: 0.0005139030348623858 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002239525318145752,
            "extra": "iterations: 128\ncpu: 0.002239483593749998 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024052226025125254,
            "extra": "iterations: 115\ncpu: 0.002405183217391305 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8438ec1d093aaed11b69b8a276f2190a92554b48",
          "message": "Remove all non-ASCII characters from VMEC++. (#342)",
          "timestamp": "2025-06-10T18:20:24Z",
          "tree_id": "8a8714eb1257b7d6955a0ca6d0ec4021199da203",
          "url": "https://github.com/proximafusion/vmecpp/commit/8438ec1d093aaed11b69b8a276f2190a92554b48"
        },
        "date": 1783604248195,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015579030671472217,
            "extra": "iterations: 1799\ncpu: 0.00015578811506392445 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001427784950439578,
            "extra": "iterations: 1957\ncpu: 0.00014277614665304038 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031606051671867843,
            "extra": "iterations: 887\ncpu: 0.00031605494363021424 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028533168334339396,
            "extra": "iterations: 982\ncpu: 0.0002853270295315682 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005285027797821565,
            "extra": "iterations: 529\ncpu: 0.0005284561928166348 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005126535565587241,
            "extra": "iterations: 547\ncpu: 0.0005126454259597804 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021459910705799363,
            "extra": "iterations: 131\ncpu: 0.00214583838167939 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024207073709239134,
            "extra": "iterations: 115\ncpu: 0.0024206667043478246 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "01bab5d7f1788e7d3f119fd1263517e0d7403761",
          "message": "Remove remains of m_evn and m_odd old constants. (#343)",
          "timestamp": "2025-06-10T20:31:02Z",
          "tree_id": "506a86094396b41dd2fd0697d230ef0e9baaf011",
          "url": "https://github.com/proximafusion/vmecpp/commit/01bab5d7f1788e7d3f119fd1263517e0d7403761"
        },
        "date": 1783604247950,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015526637434959412,
            "extra": "iterations: 1792\ncpu: 0.00015523443638392858 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014164233231809672,
            "extra": "iterations: 1978\ncpu: 0.0001416337886754298 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003158058592751118,
            "extra": "iterations: 886\ncpu: 0.0003158010970654626 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002871324106590035,
            "extra": "iterations: 970\ncpu: 0.00028707472886597957 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005299028360618736,
            "extra": "iterations: 530\ncpu: 0.0005298734603773589 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005152269160791195,
            "extra": "iterations: 546\ncpu: 0.0005151786263736267 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002144917037135871,
            "extra": "iterations: 129\ncpu: 0.0021445817131782955 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002413348904971419,
            "extra": "iterations: 116\ncpu: 0.0024131827931034495 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ef96e726f1cc9bda8bb2a97ab32add94891adb36",
          "message": "Remove an unused converter method. (#344)",
          "timestamp": "2025-06-10T20:31:02Z",
          "tree_id": "39d6d02baa81b2471b79a43bb79f6bf4f05d7fc1",
          "url": "https://github.com/proximafusion/vmecpp/commit/ef96e726f1cc9bda8bb2a97ab32add94891adb36"
        },
        "date": 1783604248420,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015566323877109017,
            "extra": "iterations: 1801\ncpu: 0.00015566028317601333 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014280956941034936,
            "extra": "iterations: 1962\ncpu: 0.00014279051325178385 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003200513205377225,
            "extra": "iterations: 884\ncpu: 0.00031812251131221723 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028668967825851524,
            "extra": "iterations: 977\ncpu: 0.00028668465097236433 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005304547473929104,
            "extra": "iterations: 529\ncpu: 0.0005303500151228732 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.000513189441555149,
            "extra": "iterations: 546\ncpu: 0.0005131810641025639 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021642079720130335,
            "extra": "iterations: 130\ncpu: 0.0021640690846153826 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002423413868608146,
            "extra": "iterations: 116\ncpu: 0.002423128931034484 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "bfbb2c18d447cdca4972d39c3a7c9e1c0084a527",
          "message": "Clean up the naming guide. (#345)",
          "timestamp": "2025-06-12T07:33:18Z",
          "tree_id": "4511a38e480ab867e9eb22825fe0533443e82cf5",
          "url": "https://github.com/proximafusion/vmecpp/commit/bfbb2c18d447cdca4972d39c3a7c9e1c0084a527"
        },
        "date": 1783604248314,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015629522672035772,
            "extra": "iterations: 1801\ncpu: 0.00015599438645197118 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014121193558939044,
            "extra": "iterations: 1984\ncpu: 0.00014118505745967742 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003163011761399003,
            "extra": "iterations: 888\ncpu: 0.0003162954617117117 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028742790222167973,
            "extra": "iterations: 975\ncpu: 0.00028737633230769226 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005316842891685877,
            "extra": "iterations: 527\ncpu: 0.0005316756850094879 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005126029905611582,
            "extra": "iterations: 548\ncpu: 0.0005125947025547443 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021599586193378154,
            "extra": "iterations: 130\ncpu: 0.0021595294846153846 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024106255893049573,
            "extra": "iterations: 116\ncpu: 0.002410586879310345 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c8d833dd5e17808c7ce004e70f1f71a120cd4041",
          "message": "Refactor modified arguments to start with m_` (#346)",
          "timestamp": "2025-06-12T07:33:18Z",
          "tree_id": "2983dbc7003ff5f42bb2f8e16b2517a5cb0e4531",
          "url": "https://github.com/proximafusion/vmecpp/commit/c8d833dd5e17808c7ce004e70f1f71a120cd4041"
        },
        "date": 1783604248341,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022699300986362168,
            "extra": "iterations: 1004\ncpu: 0.00022690539940239048 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014238406576043842,
            "extra": "iterations: 1945\ncpu: 0.00014236449048843193 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003178694226720312,
            "extra": "iterations: 888\ncpu: 0.0003178508434684685 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002863890316981742,
            "extra": "iterations: 977\ncpu: 0.00028638403070624363 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005299995436785578,
            "extra": "iterations: 529\ncpu: 0.0005299660132325145 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005128727970437115,
            "extra": "iterations: 547\ncpu: 0.0005128309104204747 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002716756561427441,
            "extra": "iterations: 103\ncpu: 0.0027166170873786427 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024045614095834587,
            "extra": "iterations: 117\ncpu: 0.0024043970000000034 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "452a07722e61e449783f3f1e0dbfa854e9b9e8ff",
          "message": "Mention the naming guide in AGENTS.md (#347)",
          "timestamp": "2025-06-12T13:43:55Z",
          "tree_id": "e42f219f13776a4867ae6518ec94e34be5228725",
          "url": "https://github.com/proximafusion/vmecpp/commit/452a07722e61e449783f3f1e0dbfa854e9b9e8ff"
        },
        "date": 1783604248076,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015567885504828562,
            "extra": "iterations: 1800\ncpu: 0.00015567617277777778 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014181357238088267,
            "extra": "iterations: 1971\ncpu: 0.00014180751953323187 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003179421603883086,
            "extra": "iterations: 879\ncpu: 0.0003179369374288965 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002861676716780201,
            "extra": "iterations: 981\ncpu: 0.0002861624525993884 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005296600182246801,
            "extra": "iterations: 526\ncpu: 0.0005296322490494296 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005117896705920343,
            "extra": "iterations: 547\ncpu: 0.0005117797531992688 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021649709114661586,
            "extra": "iterations: 130\ncpu: 0.002164930200000001 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002412997443100502,
            "extra": "iterations: 116\ncpu: 0.0024128373706896545 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "fb738c785ce1db3dbe31f7e2298b2c09770a64c5",
          "message": "Document FourierBasis* members (#348)",
          "timestamp": "2025-06-16T10:10:09Z",
          "tree_id": "9ec8ea31405a1465c34127cff17590e31d42aba2",
          "url": "https://github.com/proximafusion/vmecpp/commit/fb738c785ce1db3dbe31f7e2298b2c09770a64c5"
        },
        "date": 1783604248442,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001567249436001889,
            "extra": "iterations: 1798\ncpu: 0.00015671821968854286 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00015252640689306272,
            "extra": "iterations: 1586\ncpu: 0.00015251396595208067 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003193222536082183,
            "extra": "iterations: 786\ncpu: 0.00031930205597964376 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002892236768102353,
            "extra": "iterations: 978\ncpu: 0.00028920825255623726 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000529140966375744,
            "extra": "iterations: 529\ncpu: 0.0005291318468809077 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005190916539123196,
            "extra": "iterations: 539\ncpu: 0.0005190835473098328 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002174658368724261,
            "extra": "iterations: 129\ncpu: 0.002174479472868217 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024167237610652533,
            "extra": "iterations: 116\ncpu: 0.0024165731724137924 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a4295888965cd885de89677ead48b67e00237210",
          "message": "Make FourierBasisFastToroidal consistent with FourierBasisFastPoloidal. (#349)",
          "timestamp": "2025-06-16T18:10:15Z",
          "tree_id": "af19530aa3ea2c8a2ff3db1b432f7b0d0ed2b847",
          "url": "https://github.com/proximafusion/vmecpp/commit/a4295888965cd885de89677ead48b67e00237210"
        },
        "date": 1783604248272,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015587229603705166,
            "extra": "iterations: 1793\ncpu: 0.0001558694640267708 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014223713560152784,
            "extra": "iterations: 1970\ncpu: 0.00014222768934010155 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003231147440468393,
            "extra": "iterations: 820\ncpu: 0.00032308568048780497 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028509990880131966,
            "extra": "iterations: 974\ncpu: 0.0002850956899383983 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005290170885481924,
            "extra": "iterations: 530\ncpu: 0.0005289842735849058 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005148046570950329,
            "extra": "iterations: 542\ncpu: 0.0005147957822878224 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002167304356892904,
            "extra": "iterations: 129\ncpu: 0.0021672725116279085 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002414230642647579,
            "extra": "iterations: 116\ncpu: 0.0024141920517241375 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0249e786dea80cc6d266a9777e22eb53946f8384",
          "message": "no OpenMP -> no compiler warnings (#336)",
          "timestamp": "2025-06-16T18:24:44Z",
          "tree_id": "2f2a2b4038b03c0e0185367a80f78f924debf2e1",
          "url": "https://github.com/proximafusion/vmecpp/commit/0249e786dea80cc6d266a9777e22eb53946f8384"
        },
        "date": 1783604247952,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015523754303535837,
            "extra": "iterations: 1807\ncpu: 0.00015522867072495852 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014400052041122594,
            "extra": "iterations: 1940\ncpu: 0.00014396055927835054 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031737178767732174,
            "extra": "iterations: 878\ncpu: 0.00031734043280182215 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028714907070821966,
            "extra": "iterations: 968\ncpu: 0.0002871317964876033 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005287490714433297,
            "extra": "iterations: 527\ncpu: 0.0005286475294117645 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005113908222743443,
            "extra": "iterations: 546\ncpu: 0.0005113806282051283 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021593900827261117,
            "extra": "iterations: 130\ncpu: 0.0021593493923076925 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023995994502662593,
            "extra": "iterations: 117\ncpu: 0.002398958504273508 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0e9d79a6af6d67cb27bd1fefe19f6e5b3127acd7",
          "message": "Document the converter functions between different Fourier basis representations. (#350)",
          "timestamp": "2025-06-17T07:47:30Z",
          "tree_id": "a3d7736c94ee25044537ec73ac6c7c8ab9a54a88",
          "url": "https://github.com/proximafusion/vmecpp/commit/0e9d79a6af6d67cb27bd1fefe19f6e5b3127acd7"
        },
        "date": 1783604247977,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015711262863937953,
            "extra": "iterations: 1783\ncpu: 0.00015710986371284355 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014169451556628266,
            "extra": "iterations: 1975\ncpu: 0.000141659164556962 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00032061365517702974,
            "extra": "iterations: 880\ncpu: 0.00032058112613636347 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002853795203890951,
            "extra": "iterations: 983\ncpu: 0.00028535922075279745 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005300775142157687,
            "extra": "iterations: 529\ncpu: 0.0005300031965973536 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005148421315585866,
            "extra": "iterations: 544\ncpu: 0.0005148333124999995 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.00244587400685186,
            "extra": "iterations: 115\ncpu: 0.0024458350000000012 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024216709465816107,
            "extra": "iterations: 116\ncpu: 0.0024210038879310335 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Matt Landreman",
            "email": "mattland@umd.edu",
            "username": ""
          },
          "committer": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "70bc3eece71a3a1ae33ac0434199808bf22946f7",
          "message": "In simsopt_compat, ensure boundary surface mpol and ntor matches that of vmec indata",
          "timestamp": "2025-06-28T14:52:47-04:00",
          "tree_id": "e67ec5ec07cf9c1e6a8701d7ebb8bc71d56620b2",
          "url": "https://github.com/proximafusion/vmecpp/commit/70bc3eece71a3a1ae33ac0434199808bf22946f7"
        },
        "date": 1783604248161,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015666942457680472,
            "extra": "iterations: 1787\ncpu: 0.0001566664208170118 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014146630145707117,
            "extra": "iterations: 1982\ncpu: 0.00014146378657921296 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003181095166492571,
            "extra": "iterations: 883\ncpu: 0.00031800212797281996 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002874000255878155,
            "extra": "iterations: 975\ncpu: 0.000287384950769231 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005296556620606853,
            "extra": "iterations: 529\ncpu: 0.0005296281587901702 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005218082589558695,
            "extra": "iterations: 543\ncpu: 0.0005217012891344379 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.00216801221980605,
            "extra": "iterations: 129\ncpu: 0.0021679782015503893 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002411577208288785,
            "extra": "iterations: 116\ncpu: 0.002411441241379306 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "526d29877859bd61fed5605eb43d4b6021bb4bc1",
          "message": "Indata2json update to fix gfortran14 errors (#353)",
          "timestamp": "2025-07-10T10:04:01Z",
          "tree_id": "85ffc0e814db3075f8e6aa8840a17302188ad66f",
          "url": "https://github.com/proximafusion/vmecpp/commit/526d29877859bd61fed5605eb43d4b6021bb4bc1"
        },
        "date": 1783604248102,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015743591568686746,
            "extra": "iterations: 1760\ncpu: 0.0001574334897727273 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014414256927773096,
            "extra": "iterations: 1942\ncpu: 0.0001441399953656025 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031678097396247134,
            "extra": "iterations: 885\ncpu: 0.00031677530169491513 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002875775098800659,
            "extra": "iterations: 976\ncpu: 0.00028757249385245897 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000530497142762849,
            "extra": "iterations: 528\ncpu: 0.0005304611231060608 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005176525680062511,
            "extra": "iterations: 541\ncpu: 0.0005176432273567465 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021733054819033128,
            "extra": "iterations: 129\ncpu: 0.00217326884496124 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023994832976251585,
            "extra": "iterations: 117\ncpu: 0.0023993240940170923 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "60389860bb86f719c04b2687ff61167c71d697fa",
          "message": "More informative FATAL ERROR message (#356)",
          "timestamp": "2025-07-11T10:10:50Z",
          "tree_id": "3e9cf51a3485d10250672b0b5df0f9d763bb2921",
          "url": "https://github.com/proximafusion/vmecpp/commit/60389860bb86f719c04b2687ff61167c71d697fa"
        },
        "date": 1783604248131,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001566606409409467,
            "extra": "iterations: 1785\ncpu: 0.00015665284257703084 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001433232768637235,
            "extra": "iterations: 1952\ncpu: 0.00014332094620901644 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003165506721894487,
            "extra": "iterations: 882\ncpu: 0.00031653311337868483 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028627238322779077,
            "extra": "iterations: 970\ncpu: 0.0002862513030927836 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005348246052580061,
            "extra": "iterations: 530\ncpu: 0.0005347899811320757 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005135063731342281,
            "extra": "iterations: 545\ncpu: 0.0005134769889908254 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002173515036702156,
            "extra": "iterations: 128\ncpu: 0.0021732813203125013 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002398837325919388,
            "extra": "iterations: 117\ncpu: 0.0023987971538461566 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "118964bf45ac8c526e9c5070d1780c280c43eaed",
          "message": "Raise on nzeta mismatch in input vs mgrid (#357)",
          "timestamp": "2025-07-11T10:24:25Z",
          "tree_id": "4dce3cadcc17f0fd24fa7713f45902be4ece7076",
          "url": "https://github.com/proximafusion/vmecpp/commit/118964bf45ac8c526e9c5070d1780c280c43eaed"
        },
        "date": 1783604247990,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001558850192333767,
            "extra": "iterations: 1797\ncpu: 0.0001558775381190874 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014229005132541288,
            "extra": "iterations: 1966\ncpu: 0.000142268517293998 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003168325824715899,
            "extra": "iterations: 881\ncpu: 0.00031680686606129396 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028680875653126206,
            "extra": "iterations: 976\ncpu: 0.0002868036588114754 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005306457337879,
            "extra": "iterations: 525\ncpu: 0.000530544967619048 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005123707641094814,
            "extra": "iterations: 542\ncpu: 0.0005123633745387455 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0023550670593976974,
            "extra": "iterations: 128\ncpu: 0.0023550109062500002 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024200238679584706,
            "extra": "iterations: 114\ncpu: 0.002419590850877195 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "21f1d7e3192a1785e8e3d96992c8b849e8811e37",
          "message": "fsqt tracks force balance without pre-conditioning (#328)",
          "timestamp": "2025-07-12T08:16:47Z",
          "tree_id": "30846c3791dd0b47aefa800cc39b87db5deccffd",
          "url": "https://github.com/proximafusion/vmecpp/commit/21f1d7e3192a1785e8e3d96992c8b849e8811e37"
        },
        "date": 1783604248015,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001566741719592217,
            "extra": "iterations: 1790\ncpu: 0.0001566603994413408 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014229472244130955,
            "extra": "iterations: 1969\ncpu: 0.0001422860279329609 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031660937832386717,
            "extra": "iterations: 886\ncpu: 0.00031659007787810387 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028839851332347903,
            "extra": "iterations: 973\ncpu: 0.000288364273381295 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005288313377106349,
            "extra": "iterations: 529\ncpu: 0.0005288231644612477 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005145157416035811,
            "extra": "iterations: 508\ncpu: 0.0005144506771653546 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021710689251239483,
            "extra": "iterations: 130\ncpu: 0.0021708783384615377 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0025308947814138313,
            "extra": "iterations: 114\ncpu: 0.0025308546929824535 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "df6327195dfddde264d61b3df30c2cc358c45bc6",
          "message": "Execute python -m vmecpp as vmecpp from CLI (#354)",
          "timestamp": "2025-07-12T08:31:23Z",
          "tree_id": "aa03de2dede7d8b89088e2dc12212c06c7f47be2",
          "url": "https://github.com/proximafusion/vmecpp/commit/df6327195dfddde264d61b3df30c2cc358c45bc6"
        },
        "date": 1783604248405,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001555135874300284,
            "extra": "iterations: 1799\ncpu: 0.00015550610672595888 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001413293024647465,
            "extra": "iterations: 1978\ncpu: 0.00014132153589484328 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000317042560900672,
            "extra": "iterations: 885\ncpu: 0.0003170124214689266 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028458028880472714,
            "extra": "iterations: 985\ncpu: 0.0002845632761421321 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005301523298757514,
            "extra": "iterations: 529\ncpu: 0.000530143327032136 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005125716137929634,
            "extra": "iterations: 547\ncpu: 0.0005125082376599637 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021866634488105774,
            "extra": "iterations: 128\ncpu: 0.0021866247109375014 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024571146881371218,
            "extra": "iterations: 114\ncpu: 0.0024570710175438625 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5eabd513987af3d511736cd13f4a2cba6e3ba043",
          "message": "Prepare asymmetric support infrastructure (#360)",
          "timestamp": "2025-07-19T09:00:35+02:00",
          "tree_id": "21ff3ed31041158739af20115a665ca1f2716091",
          "url": "https://github.com/proximafusion/vmecpp/commit/5eabd513987af3d511736cd13f4a2cba6e3ba043"
        },
        "date": 1783604248123,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015589813878116738,
            "extra": "iterations: 1796\ncpu: 0.00015588863697104679 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014263000197082987,
            "extra": "iterations: 1965\ncpu: 0.0001426275872773537 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031733054381150467,
            "extra": "iterations: 884\ncpu: 0.0003172716108597284 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028709647843391166,
            "extra": "iterations: 977\ncpu: 0.0002870914667349029 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005316241190148486,
            "extra": "iterations: 527\ncpu: 0.0005316149108159393 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005133646343826154,
            "extra": "iterations: 545\ncpu: 0.0005132447908256884 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002159685354966384,
            "extra": "iterations: 130\ncpu: 0.002159575876923076 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002401559780805539,
            "extra": "iterations: 117\ncpu: 0.002401520854700852 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "dc46d2c3d26d9498fcf0f01cee4e3037628fcfa1",
          "message": "Fix VmecInput default values (#361)",
          "timestamp": "2025-07-20T09:01:43Z",
          "tree_id": "71f897d5e6db004afa50f825be6cfd8280c4f8c9",
          "url": "https://github.com/proximafusion/vmecpp/commit/dc46d2c3d26d9498fcf0f01cee4e3037628fcfa1"
        },
        "date": 1783604248395,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000156371397876951,
            "extra": "iterations: 1804\ncpu: 0.00015636871286031043 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014337519610949578,
            "extra": "iterations: 1949\ncpu: 0.00014335371164699845 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003178092566403476,
            "extra": "iterations: 880\ncpu: 0.0003178033011363635 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002883951033442474,
            "extra": "iterations: 968\ncpu: 0.0002883905433884299 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005289562739276344,
            "extra": "iterations: 527\ncpu: 0.0005288631252371917 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005128466754878333,
            "extra": "iterations: 545\ncpu: 0.0005128372880733944 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002167990431189537,
            "extra": "iterations: 128\ncpu: 0.002167952796875001 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002402071330858314,
            "extra": "iterations: 115\ncpu: 0.002401669565217391 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "f0f7059ee19cc3f9a676212653961ce10a3079ff",
          "message": "Improved Sphinx docs (Cleaned up docstrings and config) (#358)",
          "timestamp": "2025-07-20T09:32:35Z",
          "tree_id": "57c0e775a4dfac4eb2d43fa52fea713486d2cc48",
          "url": "https://github.com/proximafusion/vmecpp/commit/f0f7059ee19cc3f9a676212653961ce10a3079ff"
        },
        "date": 1783604248425,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00016696856479452116,
            "extra": "iterations: 1584\ncpu: 0.0001669536912878788 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001427129116184793,
            "extra": "iterations: 1961\ncpu: 0.00014270615094339625 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003179890923159849,
            "extra": "iterations: 883\ncpu: 0.00031797075651189116 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028658748041343493,
            "extra": "iterations: 971\ncpu: 0.000286572234809475 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005285600182690107,
            "extra": "iterations: 529\ncpu: 0.0005285158298676744 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005126670683977591,
            "extra": "iterations: 547\ncpu: 0.0005126374241316273 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021546418850238508,
            "extra": "iterations: 130\ncpu: 0.002154533400000001 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002403206295437283,
            "extra": "iterations: 117\ncpu: 0.0024030689658119673 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0a9512f93d40a96b83742f1dfae5b455a2f231ec",
          "message": "Fix makegrid's write to NetCDF file method (#364)",
          "timestamp": "2025-07-23T20:28:46Z",
          "tree_id": "1ef66a63e0ba0e423ba89a2b6d296d44a30b6a27",
          "url": "https://github.com/proximafusion/vmecpp/commit/0a9512f93d40a96b83742f1dfae5b455a2f231ec"
        },
        "date": 1783604247971,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015608686590593175,
            "extra": "iterations: 1795\ncpu: 0.00015607361838440112 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014134453422755093,
            "extra": "iterations: 1974\ncpu: 0.0001413336595744681 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031836683099920103,
            "extra": "iterations: 880\ncpu: 0.00031836136704545455 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000287521348652908,
            "extra": "iterations: 978\ncpu: 0.00028747639979550106 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005278767279858859,
            "extra": "iterations: 530\ncpu: 0.0005278013018867923 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005150003170748369,
            "extra": "iterations: 545\ncpu: 0.0005149917119266059 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0022201897606017097,
            "extra": "iterations: 126\ncpu: 0.0022198698412698413 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002402745444199135,
            "extra": "iterations: 116\ncpu: 0.002402441000000001 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "4f4db0e3e0eb5aeefda6be06f29f0d654e0d8df4",
          "message": "CLI main script terminates gracefully (#365)",
          "timestamp": "2025-07-23T23:32:43+02:00",
          "tree_id": "80d3bc010bf8ce5d8dfc060044cfa428c6b058d4",
          "url": "https://github.com/proximafusion/vmecpp/commit/4f4db0e3e0eb5aeefda6be06f29f0d654e0d8df4"
        },
        "date": 1783604248097,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015649072809240696,
            "extra": "iterations: 1788\ncpu: 0.00015648772483221478 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014427516045457427,
            "extra": "iterations: 1942\ncpu: 0.00014427254994850674 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003197055596571702,
            "extra": "iterations: 884\ncpu: 0.00031968199547511314 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028713658559236595,
            "extra": "iterations: 973\ncpu: 0.0002871159866392602 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005408191127518851,
            "extra": "iterations: 517\ncpu: 0.0005408093500967117 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005135166006035373,
            "extra": "iterations: 541\ncpu: 0.0005135075896487988 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.00220278836786747,
            "extra": "iterations: 128\ncpu: 0.0022027505859374998 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024026817288892023,
            "extra": "iterations: 116\ncpu: 0.0024026362844827612 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c6950322bd162bef45f35a2a781d6db7f26a5b5d",
          "message": "Faster cibuildwheel (#369)",
          "timestamp": "2025-07-24T11:46:42+02:00",
          "tree_id": "cd92c0d69b93349de399978bd972929b4456ecda",
          "url": "https://github.com/proximafusion/vmecpp/commit/c6950322bd162bef45f35a2a781d6db7f26a5b5d"
        },
        "date": 1783604248330,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015613563042923698,
            "extra": "iterations: 1604\ncpu: 0.00015612697630922696 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014248309667415297,
            "extra": "iterations: 1963\ncpu: 0.00014248077330616404 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031843771299811354,
            "extra": "iterations: 879\ncpu: 0.0003184327474402731 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002866005848765617,
            "extra": "iterations: 978\ncpu: 0.00028658589570552153 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005282008715268582,
            "extra": "iterations: 531\ncpu: 0.0005280967627118648 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005425005933664141,
            "extra": "iterations: 548\ncpu: 0.0005424683795620435 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021705072979594387,
            "extra": "iterations: 129\ncpu: 0.0021703937131782956 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024030228965302818,
            "extra": "iterations: 117\ncpu: 0.002402644145299148 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ea303e348e76ff7fbe4b787efc63fb3a98a3875f",
          "message": "Human readable termination reason (#366)",
          "timestamp": "2025-07-24T16:47:27+02:00",
          "tree_id": "fc6b44b44506bf3bbc417899136154ee5162cf46",
          "url": "https://github.com/proximafusion/vmecpp/commit/ea303e348e76ff7fbe4b787efc63fb3a98a3875f"
        },
        "date": 1783604248415,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015563451204612347,
            "extra": "iterations: 1801\ncpu: 0.00015562712992781791 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014248808669655878,
            "extra": "iterations: 1966\ncpu: 0.0001424858128179044 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003180260560950455,
            "extra": "iterations: 882\ncpu: 0.0003179725487528346 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002899416646019357,
            "extra": "iterations: 976\ncpu: 0.00028993674897540993 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000529116597668878,
            "extra": "iterations: 522\ncpu: 0.0005290661417624523 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005135776799753171,
            "extra": "iterations: 545\ncpu: 0.0005134684623853206 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002165101295293764,
            "extra": "iterations: 129\ncpu: 0.002164877480620158 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002401934729682075,
            "extra": "iterations: 117\ncpu: 0.0024018933675213697 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8c4075f7f10593ecfdde3b60b5035eb6550661ef",
          "message": "Added missing lasym terms to VmecWout (#368)",
          "timestamp": "2025-07-24T16:51:49+02:00",
          "tree_id": "8ab59ceaa50e1c6f1f6fafc6f3b4c6241f5ae5e4",
          "url": "https://github.com/proximafusion/vmecpp/commit/8c4075f7f10593ecfdde3b60b5035eb6550661ef"
        },
        "date": 1783604248208,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001637603610213421,
            "extra": "iterations: 1709\ncpu: 0.00016375778525453484 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014332359136357483,
            "extra": "iterations: 1976\ncpu: 0.00014331031275303648 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000320376973257386,
            "extra": "iterations: 861\ncpu: 0.0003203715412311265 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002874889902510438,
            "extra": "iterations: 974\ncpu: 0.00028748447022587285 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005413459596179781,
            "extra": "iterations: 525\ncpu: 0.0005413173847619048 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005153290020143854,
            "extra": "iterations: 542\ncpu: 0.0005153209889298894 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002530009896905573,
            "extra": "iterations: 111\ncpu: 0.002529971216216214 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002421775768543112,
            "extra": "iterations: 116\ncpu: 0.0024213853965517206 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "dd3ad47db769055d6d4ec91b91f46c90022b8122",
          "message": "exact profile evaluation (#370)",
          "timestamp": "2025-07-24T18:23:08+02:00",
          "tree_id": "174125d88811d4405e2a7b3f68491bf20c976a5f",
          "url": "https://github.com/proximafusion/vmecpp/commit/dd3ad47db769055d6d4ec91b91f46c90022b8122"
        },
        "date": 1783604248400,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015663686122241745,
            "extra": "iterations: 1783\ncpu: 0.00015663416657319125 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014126210456065543,
            "extra": "iterations: 1979\ncpu: 0.00014125118797372413 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003162685760670463,
            "extra": "iterations: 885\ncpu: 0.00031624787570621465 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002869172574064727,
            "extra": "iterations: 978\ncpu: 0.00028683589366053185 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000528992466501312,
            "extra": "iterations: 527\ncpu: 0.0005289630436432636 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005118907914519093,
            "extra": "iterations: 547\ncpu: 0.0005118609341864716 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0022445240020751954,
            "extra": "iterations: 125\ncpu: 0.0022440088320000023 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024306219199608115,
            "extra": "iterations: 116\ncpu: 0.0024180685862068944 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "008e83d5c6386352a149b6a114ccc07519ecfddf",
          "message": "Backwards compatibility with very old VMEC files (#371)",
          "timestamp": "2025-07-24T18:26:47+02:00",
          "tree_id": "58d9be2db28e9675875b70c548af705f6b102981",
          "url": "https://github.com/proximafusion/vmecpp/commit/008e83d5c6386352a149b6a114ccc07519ecfddf"
        },
        "date": 1783604247940,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001555584321308773,
            "extra": "iterations: 1796\ncpu: 0.00015555099331848553 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014290400705147666,
            "extra": "iterations: 1959\ncpu: 0.00014290173864216443 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031792266992588413,
            "extra": "iterations: 883\ncpu: 0.0003179062219705549 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002884422657919712,
            "extra": "iterations: 976\ncpu: 0.00028842687602458994 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005301395148941966,
            "extra": "iterations: 528\ncpu: 0.0005301299488636366 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005314041228189956,
            "extra": "iterations: 548\ncpu: 0.0005313584306569343 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021646928417590238,
            "extra": "iterations: 129\ncpu: 0.002164657620155041 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.00240459524351975,
            "extra": "iterations: 116\ncpu: 0.0024044709913793117 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0474522411fd92402908b6438b5c28d320d8f1b3",
          "message": "Add UV to pypi_publish CI (#372)",
          "timestamp": "2025-07-24T18:58:59+02:00",
          "tree_id": "f9c5ae8553a7bc7f151bd3753664a9b285c93292",
          "url": "https://github.com/proximafusion/vmecpp/commit/0474522411fd92402908b6438b5c28d320d8f1b3"
        },
        "date": 1783604247957,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015648396987488818,
            "extra": "iterations: 1790\ncpu: 0.00015648126815642456 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.000142475252481682,
            "extra": "iterations: 1964\ncpu: 0.00014244762881873723 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003171863534307426,
            "extra": "iterations: 883\ncpu: 0.00031716133522083806 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028801750080961156,
            "extra": "iterations: 971\ncpu: 0.00028801281668383127 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005293101754807334,
            "extra": "iterations: 524\ncpu: 0.0005292259064885497 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005155446774819318,
            "extra": "iterations: 544\ncpu: 0.0005155352922794115 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002451601781343159,
            "extra": "iterations: 114\ncpu: 0.002451563464912281 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024167993794316834,
            "extra": "iterations: 115\ncpu: 0.0024163052782608723 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b35ea2d27022d2c64974392ca5f4c68a84495c86",
          "message": "Default factories to silence warnings (#373)",
          "timestamp": "2025-07-25T17:00:14+02:00",
          "tree_id": "29800639176708dc52d9de59671c8069ba9d5fc3",
          "url": "https://github.com/proximafusion/vmecpp/commit/b35ea2d27022d2c64974392ca5f4c68a84495c86"
        },
        "date": 1783604248304,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015614564489425397,
            "extra": "iterations: 1789\ncpu: 0.00015612030128563445 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001416434056232408,
            "extra": "iterations: 1979\ncpu: 0.00014164103335017684 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003180007027337667,
            "extra": "iterations: 883\ncpu: 0.00031798273499433755 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002870889472179726,
            "extra": "iterations: 976\ncpu: 0.0002870456454918034 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005279741269271478,
            "extra": "iterations: 531\ncpu: 0.0005279414500941619 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005127625230025548,
            "extra": "iterations: 547\ncpu: 0.0005127338372943329 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021595622217932416,
            "extra": "iterations: 129\ncpu: 0.0021594390232558126 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002414976728373561,
            "extra": "iterations: 116\ncpu: 0.0024149329827586217 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "4011a876c9c187bf3ab590e897d76c03390fb5c5",
          "message": "Add CLAUDE.md (#375)",
          "timestamp": "2025-07-29T15:34:19Z",
          "tree_id": "07f9754e984ada5e852f8f490e74ba0316b1f53d",
          "url": "https://github.com/proximafusion/vmecpp/commit/4011a876c9c187bf3ab590e897d76c03390fb5c5"
        },
        "date": 1783604248058,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015563061000109383,
            "extra": "iterations: 1799\ncpu: 0.00015562777765425241 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014164712693956163,
            "extra": "iterations: 1980\ncpu: 0.00014161949141414145 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003162700666021977,
            "extra": "iterations: 884\ncpu: 0.00031626452828054303 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002886280884075762,
            "extra": "iterations: 958\ncpu: 0.00028861225782881005 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005327757773743837,
            "extra": "iterations: 526\ncpu: 0.0005326961083650194 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005108522712637525,
            "extra": "iterations: 545\ncpu: 0.0005107979871559634 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0022077901022774836,
            "extra": "iterations: 126\ncpu: 0.0022076771507936487 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002392033226469643,
            "extra": "iterations: 117\ncpu: 0.002391438555555555 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "425dc0812adf267e57d6f0a39eb39da387f0a733",
          "message": "fix AGENTS.md instructions for Python code (#377)",
          "timestamp": "2025-08-04T08:18:41Z",
          "tree_id": "eb1c6f6f679f279d2dd1d33a24fef6d851fd2b82",
          "url": "https://github.com/proximafusion/vmecpp/commit/425dc0812adf267e57d6f0a39eb39da387f0a733"
        },
        "date": 1783604248068,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015639340771096092,
            "extra": "iterations: 1792\ncpu: 0.00015635839341517858 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014102158224065417,
            "extra": "iterations: 1982\ncpu: 0.00014101388042381437 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031591106831960235,
            "extra": "iterations: 887\ncpu: 0.00031588593235625705 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002874801495323883,
            "extra": "iterations: 978\ncpu: 0.00028742596625766856 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005339474389047334,
            "extra": "iterations: 528\ncpu: 0.000533938857954545 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005135159973704487,
            "extra": "iterations: 545\ncpu: 0.0005135098458715596 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021537175545325647,
            "extra": "iterations: 130\ncpu: 0.00215320789230769 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023981974675105168,
            "extra": "iterations: 117\ncpu: 0.0023980647008547016 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "cfe73928d34e994d16ce6aa67277d4172a631d52",
          "message": "Fix a race condition in ijacobian (#379)",
          "timestamp": "2025-08-11T12:53:17Z",
          "tree_id": "722d191a11a54da0ffc6a8230d58eeb7f2f8dfa7",
          "url": "https://github.com/proximafusion/vmecpp/commit/cfe73928d34e994d16ce6aa67277d4172a631d52"
        },
        "date": 1783604248372,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015716356957589013,
            "extra": "iterations: 1778\ncpu: 0.00015716116254218225 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001414531248138226,
            "extra": "iterations: 1967\ncpu: 0.0001414506802236909 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003186695037349578,
            "extra": "iterations: 868\ncpu: 0.00031862969585253446 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028826454721614186,
            "extra": "iterations: 969\ncpu: 0.00028823952734778125 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005271775381905691,
            "extra": "iterations: 532\ncpu: 0.000527168180451128 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.000513366040061502,
            "extra": "iterations: 544\ncpu: 0.0005133582371323535 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021453802402202904,
            "extra": "iterations: 130\ncpu: 0.002145345846153844 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002389229260958158,
            "extra": "iterations: 117\ncpu: 0.0023891894017094017 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5e161a13e42219a05ea4c72e53527e7f856ed49a",
          "message": "Use hot-restart for Fourier resolution increments. (#378)",
          "timestamp": "2025-08-12T18:17:39Z",
          "tree_id": "27e1f878609c5fa81e32f3932bded0787e841577",
          "url": "https://github.com/proximafusion/vmecpp/commit/5e161a13e42219a05ea4c72e53527e7f856ed49a"
        },
        "date": 1783604248120,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015724260220016848,
            "extra": "iterations: 1783\ncpu: 0.0001572399899046551 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001415995316609145,
            "extra": "iterations: 1977\ncpu: 0.00014159264744562468 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003175567877806233,
            "extra": "iterations: 882\ncpu: 0.000317551529478458 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028589616725012635,
            "extra": "iterations: 982\ncpu: 0.0002858805906313646 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005297611250994562,
            "extra": "iterations: 529\ncpu: 0.0005297360113421552 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005154978504497043,
            "extra": "iterations: 543\ncpu: 0.0005154886850828726 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002172776063283285,
            "extra": "iterations: 120\ncpu: 0.0021727394249999995 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024065197559825164,
            "extra": "iterations: 114\ncpu: 0.0024063897017543886 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "f08ba944041900068a6801ed2f12eb182d62d8e9",
          "message": "Remove gcc and cmake from required MacOS packages (#390)",
          "timestamp": "2025-09-15T08:06:41Z",
          "tree_id": "31d4876375df38f56e80c811ace8e5d4e54d23a2",
          "url": "https://github.com/proximafusion/vmecpp/commit/f08ba944041900068a6801ed2f12eb182d62d8e9"
        },
        "date": 1783604248422,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015569608876223558,
            "extra": "iterations: 1797\ncpu: 0.0001556804501947691 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001410271719060557,
            "extra": "iterations: 1985\ncpu: 0.00014102473148614608 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031900052066263,
            "extra": "iterations: 876\ncpu: 0.00031898274086758 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028816056005733534,
            "extra": "iterations: 970\ncpu: 0.0002881420567010309 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005288732525080951,
            "extra": "iterations: 529\ncpu: 0.0005288445595463137 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005119395082014321,
            "extra": "iterations: 548\ncpu: 0.0005119125383211679 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002160939803490272,
            "extra": "iterations: 130\ncpu: 0.0021606607230769226 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002399491448687692,
            "extra": "iterations: 117\ncpu: 0.002399446837606838 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9bf8324096284391ab45054cfe02961c1ba823e3",
          "message": "Fix a race condition on liter_flag (#386)",
          "timestamp": "2025-09-15T08:51:22Z",
          "tree_id": "becdc4d45a8bef943bf5bcb9deef778fd13bc297",
          "url": "https://github.com/proximafusion/vmecpp/commit/9bf8324096284391ab45054cfe02961c1ba823e3"
        },
        "date": 1783604248246,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001553632680652211,
            "extra": "iterations: 1545\ncpu: 0.0001553610737864078 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014249073094268458,
            "extra": "iterations: 1974\ncpu: 0.0001424790268490375 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003451127684518193,
            "extra": "iterations: 890\ncpu: 0.00034509617415730336 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028557344800227765,
            "extra": "iterations: 981\ncpu: 0.0002855525045871562 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005337721482626175,
            "extra": "iterations: 527\ncpu: 0.0005337279127134723 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.000510592278235597,
            "extra": "iterations: 549\ncpu: 0.0005105838961748632 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002163093537092209,
            "extra": "iterations: 128\ncpu: 0.0021630581875000006 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002396332911956005,
            "extra": "iterations: 117\ncpu: 0.0023960990512820485 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "2fb7685a7f7f4f0d9099e813dd153a00fcf7443b",
          "message": "Fix a race condition on iter2. (#387)",
          "timestamp": "2025-09-15T08:51:22Z",
          "tree_id": "4ed9ef7a34bab252f07824813290b146b1c0989e",
          "url": "https://github.com/proximafusion/vmecpp/commit/2fb7685a7f7f4f0d9099e813dd153a00fcf7443b"
        },
        "date": 1783604248031,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015554441309371483,
            "extra": "iterations: 1805\ncpu: 0.0001555305578947369 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014137462369110757,
            "extra": "iterations: 1978\ncpu: 0.00014137221789686558 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031752827801281896,
            "extra": "iterations: 869\ncpu: 0.0003175231967779057 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00029289022655860285,
            "extra": "iterations: 971\ncpu: 0.0002928581977342947 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005356704487520106,
            "extra": "iterations: 527\ncpu: 0.0005356612542694497 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005170331358691674,
            "extra": "iterations: 547\ncpu: 0.0005170052559414994 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021951738744974136,
            "extra": "iterations: 128\ncpu: 0.002194945726562498 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.00239871098445012,
            "extra": "iterations: 117\ncpu: 0.00239866653846154 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d64815371969d22e12e64d825dfe87bef5b6d44f",
          "message": "Fix more race conditions identified by TSan/Archer (#388)",
          "timestamp": "2025-09-15T08:51:23Z",
          "tree_id": "6991fbd785f5347d54a548c1343a5c8e3851fff6",
          "url": "https://github.com/proximafusion/vmecpp/commit/d64815371969d22e12e64d825dfe87bef5b6d44f"
        },
        "date": 1783604248382,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015604814247241673,
            "extra": "iterations: 1798\ncpu: 0.00015603977085650728 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014179048658926276,
            "extra": "iterations: 1975\ncpu: 0.00014178810936708863 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003172188997268677,
            "extra": "iterations: 880\ncpu: 0.0003172135511363637 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028748413441712377,
            "extra": "iterations: 965\ncpu: 0.00028747977098445596 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005327005692826327,
            "extra": "iterations: 529\ncpu: 0.0005326916068052931 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005155044365982984,
            "extra": "iterations: 543\ncpu: 0.0005154957863720069 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0022109323932278544,
            "extra": "iterations: 124\ncpu: 0.002210828379032257 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023914178212483725,
            "extra": "iterations: 117\ncpu: 0.002391378367521367 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9386a845bbc064d2d338375fd6a855da443dd12c",
          "message": "Fix a data race on get_delbsq (#389)",
          "timestamp": "2025-09-15T08:51:23Z",
          "tree_id": "e2eb179b839020d725444f1d35a1467aab620ccf",
          "url": "https://github.com/proximafusion/vmecpp/commit/9386a845bbc064d2d338375fd6a855da443dd12c"
        },
        "date": 1783604248232,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015714948602446369,
            "extra": "iterations: 1733\ncpu: 0.00015714631736872477 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001420568294902552,
            "extra": "iterations: 1971\ncpu: 0.00014204710350076107 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031720150662007307,
            "extra": "iterations: 885\ncpu: 0.0003171752406779661 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028761644912845316,
            "extra": "iterations: 972\ncpu: 0.0002876111965020576 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005281213530920502,
            "extra": "iterations: 532\ncpu: 0.0005280915789473683 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005850455699822841,
            "extra": "iterations: 546\ncpu: 0.0005850641648351651 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002274758999164288,
            "extra": "iterations: 130\ncpu: 0.0022747254692307675 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023971357916155434,
            "extra": "iterations: 117\ncpu: 0.002396994307692307 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "908d753d76b1226c5ba68e2941e697d25695f9df",
          "message": "Get the TSan/Archer setup working again (#384)",
          "timestamp": "2025-09-15T15:31:47Z",
          "tree_id": "ff751500c48609c6a73eaf9eeabb812e3a78e360",
          "url": "https://github.com/proximafusion/vmecpp/commit/908d753d76b1226c5ba68e2941e697d25695f9df"
        },
        "date": 1783604248226,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015597574172481415,
            "extra": "iterations: 1798\ncpu: 0.00015594781646273643 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014118782566967453,
            "extra": "iterations: 1982\ncpu: 0.00014117818617558026 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000316918165975027,
            "extra": "iterations: 884\ncpu: 0.0003168673450226246 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028846614284848627,
            "extra": "iterations: 973\ncpu: 0.0002884158396711201 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005285852572868362,
            "extra": "iterations: 529\ncpu: 0.0005285476351606808 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005138758774642107,
            "extra": "iterations: 546\ncpu: 0.0005138677985347984 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002153019868690549,
            "extra": "iterations: 131\ncpu: 0.0021524764274809148 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024017239439076393,
            "extra": "iterations: 116\ncpu: 0.0024016866034482784 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Jonathan Schilling",
            "email": "jons@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "4f41149a01e8a43affcdb83bebb82c47b2a9a8a7",
          "message": "Migrate VmecINDATA to use Eigen for vectors",
          "timestamp": "2025-11-02T17:49:02+01:00",
          "tree_id": "7da5b2e139139d6aad07a1090dbe510ce17182fc",
          "url": "https://github.com/proximafusion/vmecpp/commit/4f41149a01e8a43affcdb83bebb82c47b2a9a8a7"
        },
        "date": 1783604248094,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015765482240968996,
            "extra": "iterations: 1776\ncpu: 0.0001576519560810811 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014731991707177136,
            "extra": "iterations: 1948\ncpu: 0.0001473036421971253 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00032773252482817973,
            "extra": "iterations: 874\ncpu: 0.00032772760068649896 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028831485834102203,
            "extra": "iterations: 978\ncpu: 0.0002883100081799592 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005423355643375159,
            "extra": "iterations: 529\ncpu: 0.0005423130264650285 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005128594005809112,
            "extra": "iterations: 544\ncpu: 0.000512804034926471 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021657518638196847,
            "extra": "iterations: 129\ncpu: 0.0021657158527131812 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002406340220878864,
            "extra": "iterations: 116\ncpu: 0.0024062146120689665 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9d1944f6d40eadf695ec6e2c99bebafc40369772",
          "message": "Use hot-restart for Fourier resolution increments. (#381)",
          "timestamp": "2025-12-04T17:19:18+01:00",
          "tree_id": "1c0a23dbaef479e8a08ec146673d2d7b678929f3",
          "url": "https://github.com/proximafusion/vmecpp/commit/9d1944f6d40eadf695ec6e2c99bebafc40369772"
        },
        "date": 1783604248251,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015597587432691597,
            "extra": "iterations: 1798\ncpu: 0.0001559612786429366 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001425290071290182,
            "extra": "iterations: 1963\ncpu: 0.00014251923637289863 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000317697221927015,
            "extra": "iterations: 881\ncpu: 0.0003176917491486947 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002865630753186285,
            "extra": "iterations: 980\ncpu: 0.0002865532122448979 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005339499006649948,
            "extra": "iterations: 529\ncpu: 0.0005339419111531193 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005119029956936183,
            "extra": "iterations: 547\ncpu: 0.0005118743729433269 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021533159109262323,
            "extra": "iterations: 130\ncpu: 0.0021531057384615387 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023874258383726464,
            "extra": "iterations: 117\ncpu: 0.0023872989914529926 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jungdaesuh",
            "email": "78460559+jungdaesuh@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "001b7607f1b7da01719e6fa7e66ec43aa43f6580",
          "message": "Refactor HandoverStorage to use Eigen::RowMatrixXd for improved performance (#396)",
          "timestamp": "2025-12-05T19:57:21+09:00",
          "tree_id": "9435d64702e3029a2e0fc2c335dfc7b4db1861ef",
          "url": "https://github.com/proximafusion/vmecpp/commit/001b7607f1b7da01719e6fa7e66ec43aa43f6580"
        },
        "date": 1783604247937,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015554025661673962,
            "extra": "iterations: 1801\ncpu: 0.0001555292842865075 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014149039214948964,
            "extra": "iterations: 1977\ncpu: 0.0001414879752149722 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003159275162691451,
            "extra": "iterations: 885\ncpu: 0.0003159074610169491 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002868077993880508,
            "extra": "iterations: 978\ncpu: 0.00028678215848670766 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005279538311444519,
            "extra": "iterations: 529\ncpu: 0.0005279251228733461 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005109244889586512,
            "extra": "iterations: 548\ncpu: 0.0005109155145985406 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021692874819733378,
            "extra": "iterations: 129\ncpu: 0.0021690864418604672 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023964551778940055,
            "extra": "iterations: 117\ncpu: 0.0023963054188034186 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "04f16f531ead8995b1f4a5a5f92024e82f83f86a",
          "message": "szip requirement in netcdf, limit version (#397)",
          "timestamp": "2025-12-05T11:29:26Z",
          "tree_id": "9abe1815e47962490acc60251edea222b4d2b353",
          "url": "https://github.com/proximafusion/vmecpp/commit/04f16f531ead8995b1f4a5a5f92024e82f83f86a"
        },
        "date": 1783604247960,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015550881707236628,
            "extra": "iterations: 1795\ncpu: 0.00015550597938718668 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014146971377623557,
            "extra": "iterations: 1981\ncpu: 0.00014144992175668853 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00032060556835840066,
            "extra": "iterations: 877\ncpu: 0.00032060077537058137 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028624685751636096,
            "extra": "iterations: 978\ncpu: 0.0002862415715746423 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005291520424608914,
            "extra": "iterations: 530\ncpu: 0.0005291433037735848 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005146833634069172,
            "extra": "iterations: 543\ncpu: 0.0005146483609576426 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002144411893991324,
            "extra": "iterations: 130\ncpu: 0.002144299146153844 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024113942836893015,
            "extra": "iterations: 116\ncpu: 0.002411356637931036 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "54700786bb8602a839e30115576a8387aab486be",
          "message": "Add libaec dpendency for szip support (requirement in newest netcdf version) (#398)",
          "timestamp": "2025-12-05T18:19:53+01:00",
          "tree_id": "e8fbf34e248547275106af4a779d70fa5adc75a2",
          "url": "https://github.com/proximafusion/vmecpp/commit/54700786bb8602a839e30115576a8387aab486be"
        },
        "date": 1783604248110,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015510435516256037,
            "extra": "iterations: 1806\ncpu: 0.00015507518106312295 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001420512577309144,
            "extra": "iterations: 1981\ncpu: 0.00014204415497223627 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003256908369280095,
            "extra": "iterations: 884\ncpu: 0.0003256585882352941 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002880289346845318,
            "extra": "iterations: 964\ncpu: 0.0002879985217842325 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005323515897457695,
            "extra": "iterations: 527\ncpu: 0.0005323218406072107 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.000523079445947221,
            "extra": "iterations: 546\ncpu: 0.0005230438278388279 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002099555867318889,
            "extra": "iterations: 131\ncpu: 0.002099057923664125 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024083462254754426,
            "extra": "iterations: 116\ncpu: 0.0024083113362068998 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jungdaesuh",
            "email": "78460559+jungdaesuh@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cd4ec83601b25018a6e13ecfb9aeb2fc31a0a9ab",
          "message": "Handle bytes in VmecWOut netCDF output (#401)",
          "timestamp": "2026-01-12T19:06:36+09:00",
          "tree_id": "d1c03660c939bb6db95cb73123de8a2cacc5f1e1",
          "url": "https://github.com/proximafusion/vmecpp/commit/cd4ec83601b25018a6e13ecfb9aeb2fc31a0a9ab"
        },
        "date": 1783604248364,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015583417361282256,
            "extra": "iterations: 1799\ncpu: 0.00015582391606448027 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001414255662397905,
            "extra": "iterations: 1980\ncpu: 0.00014142369595959603 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003167720837781658,
            "extra": "iterations: 885\ncpu: 0.0003167079141242939 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002866972387143299,
            "extra": "iterations: 973\ncpu: 0.000286671721479959 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005348973246878921,
            "extra": "iterations: 523\ncpu: 0.0005348508699808792 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005127418608892532,
            "extra": "iterations: 546\ncpu: 0.0005126234230769228 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021495874537978063,
            "extra": "iterations: 129\ncpu: 0.0021495514108527125 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023906964522141675,
            "extra": "iterations: 117\ncpu: 0.0023906568717948725 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "fe8cd5d192100810e746f1036b3853fc5b7c78db",
          "message": "Add lrfp_logical_ flag to wout for Fortran VMEC compatibility (Python-only) (#399)",
          "timestamp": "2026-01-12T11:55:04+01:00",
          "tree_id": "26749dd25d88ec5cc717fecb080b8dd9bc390592",
          "url": "https://github.com/proximafusion/vmecpp/commit/fe8cd5d192100810e746f1036b3853fc5b7c78db"
        },
        "date": 1783604248445,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015562213957309723,
            "extra": "iterations: 1792\ncpu: 0.00015561975111607143 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001411458407000871,
            "extra": "iterations: 1985\ncpu: 0.0001411297607052897 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031689385236319847,
            "extra": "iterations: 885\ncpu: 0.0003168791288135594 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028681608647138584,
            "extra": "iterations: 977\ncpu: 0.0002868107031729788 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005347715079329396,
            "extra": "iterations: 524\ncpu: 0.0005345805095419849 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005111227105388712,
            "extra": "iterations: 546\ncpu: 0.00051111439010989 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021502733230590824,
            "extra": "iterations: 130\ncpu: 0.0021502356384615375 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002402529757247012,
            "extra": "iterations: 117\ncpu: 0.0024021663504273476 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b226b09d694b01edf090e2f2abb26b352c4ad15d",
          "message": "CIbuildwheel update. Minimum supported linux version changed from manylinux2014 to manylinux_2_28 (#403)",
          "timestamp": "2026-01-12T16:18:25+01:00",
          "tree_id": "2192fb8a5dbf6f01bfb1c2566d46d71a59af5073",
          "url": "https://github.com/proximafusion/vmecpp/commit/b226b09d694b01edf090e2f2abb26b352c4ad15d"
        },
        "date": 1783604248296,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015553999185429754,
            "extra": "iterations: 1801\ncpu: 0.0001555075769017213 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014075405301768653,
            "extra": "iterations: 1982\ncpu: 0.00014074107416750758 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031579172463400926,
            "extra": "iterations: 887\ncpu: 0.0003157733179255919 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002878635633187216,
            "extra": "iterations: 976\ncpu: 0.0002878132202868852 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000531720206851051,
            "extra": "iterations: 525\ncpu: 0.0005317114590476195 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005145160110166784,
            "extra": "iterations: 547\ncpu: 0.0005144835630712975 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002160154856168307,
            "extra": "iterations: 130\ncpu: 0.002159757769230767 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002428525427113409,
            "extra": "iterations: 115\ncpu: 0.0024283920695652157 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "4c71e83b9705cebd3c80d05d71016a9cfb8ffaf0",
          "message": "CIbuildwheel MacOS fix (#404)",
          "timestamp": "2026-01-12T16:42:26+01:00",
          "tree_id": "e0c804c36ca6f1d372c72a6768c60a2733bffa01",
          "url": "https://github.com/proximafusion/vmecpp/commit/4c71e83b9705cebd3c80d05d71016a9cfb8ffaf0"
        },
        "date": 1783604248089,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015590225402390242,
            "extra": "iterations: 1796\ncpu: 0.00015589529565701563 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001427362159806855,
            "extra": "iterations: 1960\ncpu: 0.00014272596275510208 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003176915443549722,
            "extra": "iterations: 885\ncpu: 0.00031768641581920906 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00029470378023597544,
            "extra": "iterations: 958\ncpu: 0.0002946873768267224 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005280037608748326,
            "extra": "iterations: 531\ncpu: 0.0005279676629001885 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005126754879514814,
            "extra": "iterations: 546\ncpu: 0.0005126473260073257 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021652694820433623,
            "extra": "iterations: 129\ncpu: 0.0021652387906976735 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002416035224651468,
            "extra": "iterations: 116\ncpu: 0.0024159113275862096 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "c44c00a514c568170f3ec892558ce64441ddd00d",
          "message": "Pin archlinux test python version to 3.13 (#405)",
          "timestamp": "2026-01-12T21:02:11Z",
          "tree_id": "2ad6d163736b4c2de9fffef66bfb61fcec733062",
          "url": "https://github.com/proximafusion/vmecpp/commit/c44c00a514c568170f3ec892558ce64441ddd00d"
        },
        "date": 1783604248325,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015568246291902796,
            "extra": "iterations: 1371\ncpu: 0.00015565363384390958 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001415710005873439,
            "extra": "iterations: 1979\ncpu: 0.00014156231632137445 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031532725191170093,
            "extra": "iterations: 887\ncpu: 0.0003153220631341599 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028676957470403873,
            "extra": "iterations: 979\ncpu: 0.0002867648529111338 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005296254653786441,
            "extra": "iterations: 529\ncpu: 0.0005296174423440454 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005114663651574662,
            "extra": "iterations: 546\ncpu: 0.0005114575567765568 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002166128897851752,
            "extra": "iterations: 129\ncpu: 0.002166095333333335 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002400825763570851,
            "extra": "iterations: 116\ncpu: 0.002400648681034485 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jungdaesuh",
            "email": "78460559+jungdaesuh@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "f604ace31ba368141ef5f10ba755f2a0eb97aeaf",
          "message": "Fix simsopt compatibility with change_resolution (#402)",
          "timestamp": "2026-02-11T09:48:57+09:00",
          "tree_id": "ff8d2a393ec78192a127574988c8227a8f619e76",
          "url": "https://github.com/proximafusion/vmecpp/commit/f604ace31ba368141ef5f10ba755f2a0eb97aeaf"
        },
        "date": 1783604248436,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015662659728867288,
            "extra": "iterations: 1787\ncpu: 0.00015661722663682151 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014234378235888822,
            "extra": "iterations: 1964\ncpu: 0.00014233489103869653 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003187976463934125,
            "extra": "iterations: 879\ncpu: 0.00031878045164960184 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028837851292111096,
            "extra": "iterations: 965\ncpu: 0.0002883738435233163 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005295576368059431,
            "extra": "iterations: 525\ncpu: 0.0005295492038095239 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005146749727018587,
            "extra": "iterations: 546\ncpu: 0.0005146490476190472 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021825768053531647,
            "extra": "iterations: 128\ncpu: 0.0021824525078124983 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002394926853668995,
            "extra": "iterations: 117\ncpu: 0.0023948829743589755 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jungdaesuh",
            "email": "78460559+jungdaesuh@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "80f1ade9b830417e54e53ffb5a8596fcefa5e1ca",
          "message": "Fix #164: rebind spans after copy/move/assignment (#400)",
          "timestamp": "2026-02-11T21:24:14+09:00",
          "tree_id": "fdccc6e689baa4806259788395b97610aab4d060",
          "url": "https://github.com/proximafusion/vmecpp/commit/80f1ade9b830417e54e53ffb5a8596fcefa5e1ca"
        },
        "date": 1783604248186,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015717551262354665,
            "extra": "iterations: 1786\ncpu: 0.0001571729529675252 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014134392593846177,
            "extra": "iterations: 1980\ncpu: 0.00014134148282828288 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003189778060055851,
            "extra": "iterations: 890\ncpu: 0.0003188386707865168 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028707547178053903,
            "extra": "iterations: 978\ncpu: 0.0002870511871165645 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000530893603960673,
            "extra": "iterations: 528\ncpu: 0.0005308847234848482 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005139680135817755,
            "extra": "iterations: 546\ncpu: 0.0005139416428571431 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002097420226362415,
            "extra": "iterations: 133\ncpu: 0.0020972496766917285 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002413153648376465,
            "extra": "iterations: 116\ncpu: 0.0024131103534482785 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "140cd686f1602c795ed0c96f617765249f1fd6fb",
          "message": "Test copy semantics (#416)",
          "timestamp": "2026-02-11T13:36:59+01:00",
          "tree_id": "e5fab3fba1e767ff4ab40bb0fd55973a9322e26e",
          "url": "https://github.com/proximafusion/vmecpp/commit/140cd686f1602c795ed0c96f617765249f1fd6fb"
        },
        "date": 1783604247995,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015592282371861596,
            "extra": "iterations: 1792\ncpu: 0.00015592025279017859 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001430952494157841,
            "extra": "iterations: 1957\ncpu: 0.0001430877991824221 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031713026741711964,
            "extra": "iterations: 881\ncpu: 0.000317125107832009 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028625896301581895,
            "extra": "iterations: 977\ncpu: 0.00028623694370522 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005305232532984368,
            "extra": "iterations: 529\ncpu: 0.0005304927882797736 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005155197985760577,
            "extra": "iterations: 539\ncpu: 0.0005154791651205937 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0023279388745625815,
            "extra": "iterations: 120\ncpu: 0.002327896841666666 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024035995842045187,
            "extra": "iterations: 117\ncpu: 0.002403461948717948 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "2eec4eb7b9c12139c2ec3f61066fd0203bb86576",
          "message": "Remove copyFrom constructor, now that copy semantics are working (#417)",
          "timestamp": "2026-02-11T14:00:19Z",
          "tree_id": "c3149fc1470fe2e4ce5ff8169167bf5bc01a8d85",
          "url": "https://github.com/proximafusion/vmecpp/commit/2eec4eb7b9c12139c2ec3f61066fd0203bb86576"
        },
        "date": 1783604248029,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001625815335051108,
            "extra": "iterations: 1722\ncpu: 0.00016257300580720094 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014160605859973323,
            "extra": "iterations: 1979\ncpu: 0.00014160348408287016 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003170284750235027,
            "extra": "iterations: 884\ncpu: 0.0003169956911764707 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028655030688301466,
            "extra": "iterations: 976\ncpu: 0.0002865454272540983 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005295930152128685,
            "extra": "iterations: 529\ncpu: 0.0005295840472589788 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005160231431911792,
            "extra": "iterations: 542\ncpu: 0.0005159941291512919 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002226195638141935,
            "extra": "iterations: 126\ncpu: 0.002226037309523811 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023961923061273037,
            "extra": "iterations: 117\ncpu: 0.0023961564017094036 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "40393aa062f2c29d6387a2dba06587ca0767f338",
          "message": "More informative error message. (#419)",
          "timestamp": "2026-02-11T17:22:46Z",
          "tree_id": "2d9413cf5c65e06a9a04b16f26438c512be4c417",
          "url": "https://github.com/proximafusion/vmecpp/commit/40393aa062f2c29d6387a2dba06587ca0767f338"
        },
        "date": 1783604248060,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015605816127329186,
            "extra": "iterations: 1797\ncpu: 0.00015605560322760157 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014231896472481444,
            "extra": "iterations: 1986\ncpu: 0.00014231653423967775 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003251667508935321,
            "extra": "iterations: 863\ncpu: 0.0003251338285052143 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00029121378085863824,
            "extra": "iterations: 961\ncpu: 0.00029119938085327764 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005274411648680262,
            "extra": "iterations: 531\ncpu: 0.0005274321431261776 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005128745405068128,
            "extra": "iterations: 547\ncpu: 0.0005128449177330894 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002166751743287079,
            "extra": "iterations: 129\ncpu: 0.002166713457364344 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024145759385207606,
            "extra": "iterations: 116\ncpu: 0.0024144264482758647 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "6268991099391a555d47194b051835e5d16d0a25",
          "message": "GIL handling to allow Ctrl+C and progressive logging in Jupyter (#415)",
          "timestamp": "2026-02-11T22:48:00Z",
          "tree_id": "10344098f8352d60b28be3f6f7022f5a8ec59563",
          "url": "https://github.com/proximafusion/vmecpp/commit/6268991099391a555d47194b051835e5d16d0a25"
        },
        "date": 1783604248133,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015547861644503008,
            "extra": "iterations: 1798\ncpu: 0.0001554667469410456 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014366833030754795,
            "extra": "iterations: 1974\ncpu: 0.0001436606681864235 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003203696462600673,
            "extra": "iterations: 874\ncpu: 0.0003203640583524026 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002865771838179404,
            "extra": "iterations: 979\ncpu: 0.0002865617732379979 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005284084464019199,
            "extra": "iterations: 530\ncpu: 0.0005283818358490568 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005107086940403403,
            "extra": "iterations: 548\ncpu: 0.0005106603357664229 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002160429030425789,
            "extra": "iterations: 129\ncpu: 0.0021603023178294596 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024166555486173713,
            "extra": "iterations: 117\ncpu: 0.0024166105042735036 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "45c302c6ba8d4c19937ea75c58c1b92d5d282fca",
          "message": "Minor ruff lint changes (#421)",
          "timestamp": "2026-02-12T00:40:36+01:00",
          "tree_id": "1d0648d2c43bc08c192491dbf05b5379165b3131",
          "url": "https://github.com/proximafusion/vmecpp/commit/45c302c6ba8d4c19937ea75c58c1b92d5d282fca"
        },
        "date": 1783604248079,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001568222755152792,
            "extra": "iterations: 1748\ncpu: 0.00015681311212814648 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001418086384662992,
            "extra": "iterations: 1972\ncpu: 0.00014180626673427994 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031737220409103206,
            "extra": "iterations: 881\ncpu: 0.0003173552224744608 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028631682288671076,
            "extra": "iterations: 978\ncpu: 0.0002862992443762783 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005281331404200141,
            "extra": "iterations: 530\ncpu: 0.0005281035339622645 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005143484938035318,
            "extra": "iterations: 545\ncpu: 0.0005143148733944957 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002180522307753563,
            "extra": "iterations: 128\ncpu: 0.002180444445312499 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002384544437767094,
            "extra": "iterations: 117\ncpu: 0.002384508094017094 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "010dd486d8e526b57573bb04bc39897ec5e2764b",
          "message": "C++ dependency version updates (#422)",
          "timestamp": "2026-02-12T01:08:19+01:00",
          "tree_id": "8c62b3c7c26410ec37b537291e79af2ac811de3f",
          "url": "https://github.com/proximafusion/vmecpp/commit/010dd486d8e526b57573bb04bc39897ec5e2764b"
        },
        "date": 1783604247942,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015579074422075915,
            "extra": "iterations: 1796\ncpu: 0.00015576963084632518 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001422677721296038,
            "extra": "iterations: 1967\ncpu: 0.00014226021403152014 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003202276714777543,
            "extra": "iterations: 885\ncpu: 0.00032022273333333335 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002892905015211839,
            "extra": "iterations: 975\ncpu: 0.0002892489856410257 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000536094766483415,
            "extra": "iterations: 529\ncpu: 0.0005360633194706998 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005156468462060999,
            "extra": "iterations: 540\ncpu: 0.0005156381333333334 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002112105376738355,
            "extra": "iterations: 133\ncpu: 0.00211158534586466 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002421432528002509,
            "extra": "iterations: 116\ncpu: 0.0024211399396551693 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b17989ca99f311fe32b9353b1873262e1f0bf2b1",
          "message": "add free-boundary method that only uses magnetic field from external coils (#411)",
          "timestamp": "2026-02-12T00:35:02Z",
          "tree_id": "8c586c94a913b70c8f45b12bea605569a6d2ba57",
          "url": "https://github.com/proximafusion/vmecpp/commit/b17989ca99f311fe32b9353b1873262e1f0bf2b1"
        },
        "date": 1783604248293,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015499192257964784,
            "extra": "iterations: 1802\ncpu: 0.00015498416370699224 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014163679586978774,
            "extra": "iterations: 1977\ncpu: 0.0001416344233687405 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003168364455797014,
            "extra": "iterations: 884\ncpu: 0.0003168315113122171 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028836800728315195,
            "extra": "iterations: 972\ncpu: 0.00028835128600823056 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005281200948751198,
            "extra": "iterations: 530\ncpu: 0.0005280943132075471 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005244959385023205,
            "extra": "iterations: 545\ncpu: 0.0005244866972477064 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0022226590958852616,
            "extra": "iterations: 126\ncpu: 0.0022225317936507972 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024258779442828636,
            "extra": "iterations: 115\ncpu: 0.002425839156521741 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0a4025d7c637efa42eed4fc2fee31d838c90def3",
          "message": "Pin archlinux Python version in CI to 3.13 (#423)",
          "timestamp": "2026-02-12T03:37:13+01:00",
          "tree_id": "f56eece3beeaacd72414150f9405c248ead97f2e",
          "url": "https://github.com/proximafusion/vmecpp/commit/0a4025d7c637efa42eed4fc2fee31d838c90def3"
        },
        "date": 1783604247968,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015667597807391158,
            "extra": "iterations: 1793\ncpu: 0.00015666254210819854 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014204710812554171,
            "extra": "iterations: 1974\ncpu: 0.0001420318080040527 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031537119261459506,
            "extra": "iterations: 889\ncpu: 0.0003153078020247468 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002873386304403089,
            "extra": "iterations: 970\ncpu: 0.00028733415670103115 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005300324445410803,
            "extra": "iterations: 529\ncpu: 0.000530024474480151 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005122621490924842,
            "extra": "iterations: 547\ncpu: 0.000512137522851919 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0024549560209291173,
            "extra": "iterations: 113\ncpu: 0.002454787415929205 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002412035547453782,
            "extra": "iterations: 116\ncpu: 0.0024117100086206884 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "19ae9b8083bad3de0b5a22a9fa0c2a9f62366cfe",
          "message": "Usage example for raw profiles (#420)",
          "timestamp": "2026-02-12T08:57:32Z",
          "tree_id": "00afe8998e196144fd56620fc254ba212bf10fc8",
          "url": "https://github.com/proximafusion/vmecpp/commit/19ae9b8083bad3de0b5a22a9fa0c2a9f62366cfe"
        },
        "date": 1783604248008,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015561985519697082,
            "extra": "iterations: 1802\ncpu: 0.00015561693895671476 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014117627764566155,
            "extra": "iterations: 1982\ncpu: 0.00014116577093844607 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003161185205319507,
            "extra": "iterations: 885\ncpu: 0.0003161131898305085 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002874371430989752,
            "extra": "iterations: 967\ncpu: 0.00028742113547052753 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005277773119368644,
            "extra": "iterations: 530\ncpu: 0.0005277712867924529 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005125567153259949,
            "extra": "iterations: 546\ncpu: 0.0005125476520146516 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002175102010369301,
            "extra": "iterations: 128\ncpu: 0.0021749202421874995 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024028333843263807,
            "extra": "iterations: 117\ncpu: 0.0024027105641025665 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "11142d58ecfafc2ce40665f168c20662b124c937",
          "message": "Usage example for raw profiles (#420)",
          "timestamp": "2026-02-12T08:57:32Z",
          "tree_id": "2949bc0e7c52bf2a058636995bca537a244a0d6b",
          "url": "https://github.com/proximafusion/vmecpp/commit/11142d58ecfafc2ce40665f168c20662b124c937"
        },
        "date": 1783604247987,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015594770125591762,
            "extra": "iterations: 1727\ncpu: 0.00015594513144180666 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001433366399896724,
            "extra": "iterations: 1955\ncpu: 0.0001433343289002558 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00032831556656781364,
            "extra": "iterations: 850\ncpu: 0.0003282631882352942 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002856840852831231,
            "extra": "iterations: 976\ncpu: 0.0002856787827868852 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005289338669686947,
            "extra": "iterations: 530\ncpu: 0.0005289240528301882 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005137929128944328,
            "extra": "iterations: 545\ncpu: 0.000513709462385321 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002176279202103615,
            "extra": "iterations: 128\ncpu: 0.0021762419687499993 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024061100236300766,
            "extra": "iterations: 116\ncpu: 0.00240594879310345 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "40442f3ff5701f31ff35ce05373da16d4a8962fb",
          "message": "Analytical torflux polynomial evaluation (#316)",
          "timestamp": "2026-02-15T09:09:46Z",
          "tree_id": "986f63a2d53264e145fe443769403d02302ecf7f",
          "url": "https://github.com/proximafusion/vmecpp/commit/40442f3ff5701f31ff35ce05373da16d4a8962fb"
        },
        "date": 1783604248066,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015544322118171385,
            "extra": "iterations: 1801\ncpu: 0.00015544052359800113 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014244321646549697,
            "extra": "iterations: 1966\ncpu: 0.00014242953865717191 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003166263753717596,
            "extra": "iterations: 880\ncpu: 0.00031660714090909076 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028701257608175766,
            "extra": "iterations: 978\ncpu: 0.0002870076666666669 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005309590787598582,
            "extra": "iterations: 528\ncpu: 0.0005309056098484851 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005175721119431889,
            "extra": "iterations: 544\ncpu: 0.0005175636856617646 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0028389616215482674,
            "extra": "iterations: 94\ncpu: 0.002838753202127661 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024270888032584354,
            "extra": "iterations: 116\ncpu: 0.0024266219137931053 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c798c70845a8ad5b5d848abbe249278f6c97f753",
          "message": "Add continuous benchmarking (#426)",
          "timestamp": "2026-02-25T13:44:39+01:00",
          "tree_id": "dc0f6863d3a544bc3fed16e17e42dcebcbef34b3",
          "url": "https://github.com/proximafusion/vmecpp/commit/c798c70845a8ad5b5d848abbe249278f6c97f753"
        },
        "date": 1783604248333,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001560027844139508,
            "extra": "iterations: 1792\ncpu: 0.0001559858119419643 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001424838254115318,
            "extra": "iterations: 1963\ncpu: 0.00014247469230769234 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031794463215287493,
            "extra": "iterations: 877\ncpu: 0.0003176289999999999 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002890455735085799,
            "extra": "iterations: 971\ncpu: 0.0002890409433573633 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005279684966465213,
            "extra": "iterations: 530\ncpu: 0.0005279395169811322 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005141108075558152,
            "extra": "iterations: 543\ncpu: 0.0005140594180478823 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021504383820753837,
            "extra": "iterations: 130\ncpu: 0.002150322076923078 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024054297085466057,
            "extra": "iterations: 116\ncpu: 0.0024052529310344827 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "eguiraud-pf",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "eguiraud-pf",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c125adfa5f46fe55ce9e0183576bbaeeb11387c3",
          "message": "Add code owners file (#428)",
          "timestamp": "2026-03-09T10:22:20Z",
          "tree_id": "0a89cb0e32c0ae9230b9f02283da421a2a3341e0",
          "url": "https://github.com/proximafusion/vmecpp/commit/c125adfa5f46fe55ce9e0183576bbaeeb11387c3"
        },
        "date": 1783604248320,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015698846609370342,
            "extra": "iterations: 1782\ncpu: 0.00015698597474747477 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014134284331564242,
            "extra": "iterations: 1974\ncpu: 0.00014133697315096253 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003171518747747348,
            "extra": "iterations: 886\ncpu: 0.00031713582054176075 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028563153367037683,
            "extra": "iterations: 981\ncpu: 0.00028562693781855245 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005301775688935768,
            "extra": "iterations: 529\ncpu: 0.0005301439149338374 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005146168963813083,
            "extra": "iterations: 546\ncpu: 0.0005146091904761903 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021545794583106227,
            "extra": "iterations: 129\ncpu: 0.0021545449457364327 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023880086393437833,
            "extra": "iterations: 117\ncpu: 0.0023878802735042724 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "660fa04e463483d97bca326a6ef28174924dee41",
          "message": "hotfix pre-commit on Arch: use Python 3.10 (not 3.14) (#432)",
          "timestamp": "2026-03-11T18:46:04Z",
          "tree_id": "7ff8697b6377fbbfa65b2654714f0818cfbd2ae4",
          "url": "https://github.com/proximafusion/vmecpp/commit/660fa04e463483d97bca326a6ef28174924dee41"
        },
        "date": 1783604248136,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015534205449915328,
            "extra": "iterations: 1805\ncpu: 0.00015532253739612193 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001415406006291373,
            "extra": "iterations: 1978\ncpu: 0.0001415308660262892 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031935411786276197,
            "extra": "iterations: 882\ncpu: 0.0003193361825396827 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002875937001221458,
            "extra": "iterations: 971\ncpu: 0.0002875633975283214 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005290000610931745,
            "extra": "iterations: 526\ncpu: 0.0005289917927756652 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005141844443224986,
            "extra": "iterations: 545\ncpu: 0.0005140873266055045 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021593496780986936,
            "extra": "iterations: 129\ncpu: 0.002159221100775194 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024147510528564455,
            "extra": "iterations: 115\ncpu: 0.002414547321739132 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "07b8a0f980019f295ebfdb15a6681704bc58a30c",
          "message": "Add stand-alone makegrid executable (#431)",
          "timestamp": "2026-03-11T18:57:23Z",
          "tree_id": "0cbc84d805949211ff1cb9f082a3702d6f73341d",
          "url": "https://github.com/proximafusion/vmecpp/commit/07b8a0f980019f295ebfdb15a6681704bc58a30c"
        },
        "date": 1783604247962,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015544256632419275,
            "extra": "iterations: 1803\ncpu: 0.00015543037548530227 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014091653924643485,
            "extra": "iterations: 1987\ncpu: 0.00014091427981882237 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003167100562919712,
            "extra": "iterations: 883\ncpu: 0.00031670454133635314 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002872639722784851,
            "extra": "iterations: 972\ncpu: 0.0002872450154320989 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005311984073073239,
            "extra": "iterations: 526\ncpu: 0.0005311653193916348 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.000512986664378315,
            "extra": "iterations: 545\ncpu: 0.0005129562348623854 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021720360964536667,
            "extra": "iterations: 128\ncpu: 0.002171788640624999 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024015292143210387,
            "extra": "iterations: 117\ncpu: 0.0024014924358974387 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9e57835a6772309ef0d07ba35fdedbbfd915ea59",
          "message": "ignore .cache (#430)",
          "timestamp": "2026-03-11T18:57:23Z",
          "tree_id": "62d5ab9148dafc000a305f8eb9983ec3fc8e4da8",
          "url": "https://github.com/proximafusion/vmecpp/commit/9e57835a6772309ef0d07ba35fdedbbfd915ea59"
        },
        "date": 1783604248256,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015550085287665053,
            "extra": "iterations: 1804\ncpu: 0.000155487227827051 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001415919631102159,
            "extra": "iterations: 1979\ncpu: 0.0001415898610409298 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003178519304803264,
            "extra": "iterations: 871\ncpu: 0.0003178469322617681 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028639388424315913,
            "extra": "iterations: 982\ncpu: 0.00028638954276985766 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005284464740932773,
            "extra": "iterations: 531\ncpu: 0.0005284109378531071 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005107090989965777,
            "extra": "iterations: 549\ncpu: 0.0005107004644808744 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002119333455056855,
            "extra": "iterations: 132\ncpu: 0.002119225378787881 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023958540370321683,
            "extra": "iterations: 117\ncpu: 0.0023958088803418803 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jungdaesuh",
            "email": "78460559+jungdaesuh@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "e755dc32c9ce57a0c85c6c7f7fe6f10cf4ccf84a",
          "message": "Preserve assigned boundary during run (#429)",
          "timestamp": "2026-03-12T07:25:59+09:00",
          "tree_id": "5fe728871e4ab55bd1f1b9748adfb5bd6a9b28cc",
          "url": "https://github.com/proximafusion/vmecpp/commit/e755dc32c9ce57a0c85c6c7f7fe6f10cf4ccf84a"
        },
        "date": 1783604248412,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015631863581282753,
            "extra": "iterations: 1792\ncpu: 0.00015631579464285713 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001411277121767657,
            "extra": "iterations: 1986\ncpu: 0.0001411163952668681 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003252751312786908,
            "extra": "iterations: 862\ncpu: 0.00032526995011600935 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002885348519385168,
            "extra": "iterations: 971\ncpu: 0.00028851342636457245 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005313792609443229,
            "extra": "iterations: 526\ncpu: 0.0005313700342205323 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005130741670319106,
            "extra": "iterations: 547\ncpu: 0.0005130650073126143 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002208229211660532,
            "extra": "iterations: 130\ncpu: 0.002207779107692308 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002386390653430906,
            "extra": "iterations: 117\ncpu: 0.002386344923076922 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "jons-pf",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c8a471ba878f4b3f92db1bdcd6228b5f86698a31",
          "message": "Add only_coils-related cases in indata tests (#433)",
          "timestamp": "2026-03-11T22:50:42Z",
          "tree_id": "fd642d74d332af5a203c80b5f2a6e72b40ef4c4a",
          "url": "https://github.com/proximafusion/vmecpp/commit/c8a471ba878f4b3f92db1bdcd6228b5f86698a31"
        },
        "date": 1783604248338,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00016024967972157422,
            "extra": "iterations: 1759\ncpu: 0.00016024712336554862 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014337683115651722,
            "extra": "iterations: 1948\ncpu: 0.00014336853952772076 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031794321794109367,
            "extra": "iterations: 881\ncpu: 0.0003179199636776389 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002863908812510225,
            "extra": "iterations: 979\ncpu: 0.0002863601767109296 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005442333775897359,
            "extra": "iterations: 516\ncpu: 0.0005441264651162794 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.000512558973685479,
            "extra": "iterations: 547\ncpu: 0.000512524531992687 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002143805263606647,
            "extra": "iterations: 131\ncpu: 0.0021437723358778624 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024029454614362144,
            "extra": "iterations: 117\ncpu: 0.002402902051282054 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cf64eaaeafae5ff91dd8e90a2f538732812be877",
          "message": "Fix ci error in pyright.yaml (#440)",
          "timestamp": "2026-03-18T14:31:35+01:00",
          "tree_id": "922e038285fc3406734ee9fca6b80eb396c0e453",
          "url": "https://github.com/proximafusion/vmecpp/commit/cf64eaaeafae5ff91dd8e90a2f538732812be877"
        },
        "date": 1783604248369,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015591056397963807,
            "extra": "iterations: 1774\ncpu: 0.00015590817812852312 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014249827870287826,
            "extra": "iterations: 1963\ncpu: 0.00014249577840040755 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031842835853463053,
            "extra": "iterations: 873\ncpu: 0.00031840965635738833 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000285900780236891,
            "extra": "iterations: 982\ncpu: 0.0002858697993890021 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005283472672948297,
            "extra": "iterations: 530\ncpu: 0.0005282659264150941 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005139825971965193,
            "extra": "iterations: 543\ncpu: 0.0005139747274401474 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002160493717637173,
            "extra": "iterations: 129\ncpu: 0.002160458984496124 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024031811747057684,
            "extra": "iterations: 116\ncpu: 0.00240294544827586 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jungdaesuh",
            "email": "78460559+jungdaesuh@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "40426da882d554bfa1da117458a424d564a53957",
          "message": "Write back asymmetric boundary coefficients (#439)",
          "timestamp": "2026-03-18T23:48:56+09:00",
          "tree_id": "8f738e16be93c53bc11f563ef75ed996e9a48a7f",
          "url": "https://github.com/proximafusion/vmecpp/commit/40426da882d554bfa1da117458a424d564a53957"
        },
        "date": 1783604248063,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015629924650426946,
            "extra": "iterations: 1788\ncpu: 0.00015628980257270696 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014166887444441647,
            "extra": "iterations: 1977\ncpu: 0.00014166649468892265 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031651426843330683,
            "extra": "iterations: 885\ncpu: 0.00031647795819209036 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028842859307434335,
            "extra": "iterations: 972\ncpu: 0.00028842417283950634 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005301144362853193,
            "extra": "iterations: 527\ncpu: 0.0005301065806451616 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005126483261694202,
            "extra": "iterations: 547\ncpu: 0.0005126223180987206 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002177197803822599,
            "extra": "iterations: 129\ncpu: 0.0021768307209302333 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024089628252489813,
            "extra": "iterations: 116\ncpu: 0.0024089301896551758 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8856974a7b20bccb2a2369c40e44ed5cb610bbd5",
          "message": "Globally prevent 3.14 python (#435)",
          "timestamp": "2026-03-18T16:06:34+01:00",
          "tree_id": "07d365c528c63f38b69a3f0811dd98318c653934",
          "url": "https://github.com/proximafusion/vmecpp/commit/8856974a7b20bccb2a2369c40e44ed5cb610bbd5"
        },
        "date": 1783604248200,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015699204488016252,
            "extra": "iterations: 1791\ncpu: 0.00015698943160245677 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014206299713818459,
            "extra": "iterations: 1964\ncpu: 0.00014203509266802445 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031950801983899495,
            "extra": "iterations: 879\ncpu: 0.00031947990898748566 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028836629027692383,
            "extra": "iterations: 972\ncpu: 0.0002883508744855967 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005303587232317244,
            "extra": "iterations: 525\ncpu: 0.0005302414685714286 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005132540678366637,
            "extra": "iterations: 546\ncpu: 0.0005132459780219782 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002168453941049502,
            "extra": "iterations: 129\ncpu: 0.0021684135736434094 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002405881881713867,
            "extra": "iterations: 115\ncpu: 0.002405415026086957 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "827c4b0ed5d4c5c4305b79e25b000291ebfe0413",
          "message": "DNDEBUG for faster eigen3 performance (#447)",
          "timestamp": "2026-03-26T14:50:12+01:00",
          "tree_id": "bdac70f73639a81791d026c842d96a260943c70e",
          "url": "https://github.com/proximafusion/vmecpp/commit/827c4b0ed5d4c5c4305b79e25b000291ebfe0413"
        },
        "date": 1783604248192,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015601800314745114,
            "extra": "iterations: 1791\ncpu: 0.00015601492071468456 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014365895203285046,
            "extra": "iterations: 1977\ncpu: 0.00014365670055639857 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0003188199737445027,
            "extra": "iterations: 882\ncpu: 0.0003188027619047619 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028630752524431557,
            "extra": "iterations: 977\ncpu: 0.00028628895701125886 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005304002130865374,
            "extra": "iterations: 529\ncpu: 0.0005303921285444235 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005126846578968313,
            "extra": "iterations: 546\ncpu: 0.0005126614285714284 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002181597054004669,
            "extra": "iterations: 128\ncpu: 0.002181562750000001 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024063196675530793,
            "extra": "iterations: 116\ncpu: 0.0024062773793103445 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5c527e1ec36b148acecd0cf59dc2f8418f6bf791",
          "message": "apt get update in docs CI (#448)",
          "timestamp": "2026-03-26T15:32:42+01:00",
          "tree_id": "25ec0c21ffc071b90ef07bd6beb8f7c6f95541ef",
          "url": "https://github.com/proximafusion/vmecpp/commit/5c527e1ec36b148acecd0cf59dc2f8418f6bf791"
        },
        "date": 1783604248118,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015678816690215466,
            "extra": "iterations: 1787\ncpu: 0.00015678154392837157 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014300113297285714,
            "extra": "iterations: 1964\ncpu: 0.00014297835539714868 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031668095771669,
            "extra": "iterations: 886\ncpu: 0.0003166753961625282 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002857072012765067,
            "extra": "iterations: 980\ncpu: 0.00028570193979591857 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005310052260049606,
            "extra": "iterations: 527\ncpu: 0.0005309776660341559 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005140527264102475,
            "extra": "iterations: 546\ncpu: 0.0005139930824175824 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002187758918822281,
            "extra": "iterations: 127\ncpu: 0.0021876792755905513 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0023832119117348886,
            "extra": "iterations: 118\ncpu: 0.0023830197711864373 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a5682f80e0acc586de0b0ae8f5b751dc12f89238",
          "message": "Fix auto benchmarks (#449)",
          "timestamp": "2026-03-26T15:33:28+01:00",
          "tree_id": "26c20171c6d12ed2706e72c307c69cdde94da07c",
          "url": "https://github.com/proximafusion/vmecpp/commit/a5682f80e0acc586de0b0ae8f5b751dc12f89238"
        },
        "date": 1783604248274,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001561404738860181,
            "extra": "iterations: 1791\ncpu: 0.00015612848408710217 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014233546597617014,
            "extra": "iterations: 1680\ncpu: 0.0001423190297619048 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031651223486217904,
            "extra": "iterations: 886\ncpu: 0.0003165064672686229 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000285187541092201,
            "extra": "iterations: 974\ncpu: 0.00028516719301848034 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005280109448621503,
            "extra": "iterations: 531\ncpu: 0.0005280015555555557 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005110576205009963,
            "extra": "iterations: 548\ncpu: 0.0005110479361313868 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0024341240263821787,
            "extra": "iterations: 114\ncpu: 0.002433538657894738 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024016372158996062,
            "extra": "iterations: 117\ncpu: 0.00240154011965812 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "e1d17b9da212722e00f01fef1a5d36af4b8ce119",
          "message": "Ignore git worktrees for Claude (#455)",
          "timestamp": "2026-03-26T17:14:25Z",
          "tree_id": "42a875b6850c90124f283db079a1c159aa7463ff",
          "url": "https://github.com/proximafusion/vmecpp/commit/e1d17b9da212722e00f01fef1a5d36af4b8ce119"
        },
        "date": 1783604248410,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015879726675559552,
            "extra": "iterations: 1795\ncpu: 0.00015879451810584962 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014397684471299216,
            "extra": "iterations: 1943\ncpu: 0.00014397426505404016 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031797109207690237,
            "extra": "iterations: 881\ncpu: 0.00031793747105561857 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0002877328334710537,
            "extra": "iterations: 975\ncpu: 0.00028770818974358973 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005312409301410364,
            "extra": "iterations: 527\ncpu: 0.000531231724857685 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005129353030697331,
            "extra": "iterations: 546\ncpu: 0.000512858091575092 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002171538596929506,
            "extra": "iterations: 129\ncpu: 0.0021715026589147262 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002413776414147739,
            "extra": "iterations: 116\ncpu: 0.0024137418965517236 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "14c9c876477f74fc42a46a1e94a8fcd69afad6b4",
          "message": "high verbosity CI logs for docs were enabled for debugging, no longer needed. (#450)",
          "timestamp": "2026-03-26T17:14:25Z",
          "tree_id": "80c8918132fc1bc43f96c90db3159e62b8e8b1a6",
          "url": "https://github.com/proximafusion/vmecpp/commit/14c9c876477f74fc42a46a1e94a8fcd69afad6b4"
        },
        "date": 1783604247997,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0001569676787012157,
            "extra": "iterations: 1783\ncpu: 0.00015696481996634886 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00014162220611881464,
            "extra": "iterations: 1974\ncpu: 0.00014161935207700102 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031925659787953583,
            "extra": "iterations: 878\ncpu: 0.00031923659225512524 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000289355443999647,
            "extra": "iterations: 971\ncpu: 0.0002893501967044284 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000532549143744059,
            "extra": "iterations: 526\ncpu: 0.0005325219904942966 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005131604684554245,
            "extra": "iterations: 547\ncpu: 0.0005130911407678241 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0024191400279169497,
            "extra": "iterations: 115\ncpu: 0.002419013460869568 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0024004146970551593,
            "extra": "iterations: 116\ncpu: 0.0024003722413793106 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "27a12a3ee727a764f58c759a333579c1557af738",
          "message": "Cleaner benchmark online overview (#451)",
          "timestamp": "2026-03-26T17:37:22Z",
          "tree_id": "022a42e50d70191af020a2783e5aa82dc45cb9ca",
          "url": "https://github.com/proximafusion/vmecpp/commit/27a12a3ee727a764f58c759a333579c1557af738"
        },
        "date": 1783604248021,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00015653699836474939,
            "extra": "iterations: 1788\ncpu: 0.00015652073322147656 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0001417669739991472,
            "extra": "iterations: 1973\ncpu: 0.0001417645904713635 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00031665196785560023,
            "extra": "iterations: 884\ncpu: 0.0003165830316742081 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.00028639843076680606,
            "extra": "iterations: 979\ncpu: 0.000286381188968335 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005314748246484283,
            "extra": "iterations: 527\ncpu: 0.0005314658481973438 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0005140676410920029,
            "extra": "iterations: 545\ncpu: 0.0005139901192660545 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0021658535151518592,
            "extra": "iterations: 129\ncpu: 0.0021656177209302334 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002387059145960315,
            "extra": "iterations: 116\ncpu: 0.0023870208017241373 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "jurasic-pf",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "a304c1df8c7f1e53648449541dce0c291856b82e",
          "message": "Migrate FourierBasisFastPoloidal and FourierBasisFastToroidal to Eigen3 (#410)",
          "timestamp": "2026-03-26T23:59:51Z",
          "tree_id": "7cfcf9d3a35c7021e12bb29c7280017a59f39996",
          "url": "https://github.com/proximafusion/vmecpp/commit/a304c1df8c7f1e53648449541dce0c291856b82e"
        },
        "date": 1783604248269,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00024207681417465213,
            "extra": "iterations: 1152\ncpu: 0.00024207236024305557 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0002393636948023087,
            "extra": "iterations: 1170\ncpu: 0.00023935079401709404 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004697766079998657,
            "extra": "iterations: 596\ncpu: 0.0004697682919463087 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0004950875528288393,
            "extra": "iterations: 566\ncpu: 0.0004950795547703181 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007557232746216808,
            "extra": "iterations: 371\ncpu: 0.0007556782156334229 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0008826940217604773,
            "extra": "iterations: 317\ncpu: 0.0008826804006309149 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002692963068301861,
            "extra": "iterations: 104\ncpu: 0.002692913000000002 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0029289101560910544,
            "extra": "iterations: 96\ncpu: 0.0029287559479166665 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "69142e7a02b5457b323fff1fc4512a1abf0df591",
          "message": "Vmec constructor -> Factory for error handling during initialization. (#244)",
          "timestamp": "2026-03-27T01:38:24+01:00",
          "tree_id": "d1c86ca8ebee63e151d20fc645b5bae901836a2d",
          "url": "https://github.com/proximafusion/vmecpp/commit/69142e7a02b5457b323fff1fc4512a1abf0df591"
        },
        "date": 1783604248144,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00024075713650933628,
            "extra": "iterations: 1160\ncpu: 0.00024074750172413793 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00024221743665140595,
            "extra": "iterations: 1159\ncpu: 0.00024221306643658333 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00047272000763867353,
            "extra": "iterations: 592\ncpu: 0.00047253500675675667 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0004960503198404228,
            "extra": "iterations: 565\ncpu: 0.0004960252690265488 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007547625312907997,
            "extra": "iterations: 371\ncpu: 0.0007547215687331538 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0008770644664764405,
            "extra": "iterations: 320\ncpu: 0.0008769877906250002 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026751550344320443,
            "extra": "iterations: 104\ncpu: 0.0026751090961538463 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0029342726657265116,
            "extra": "iterations: 95\ncpu: 0.0029342215473684217 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "2d8888e1defbd4d5f2949047dfa742495854ca1f",
          "message": "SImsopt keep dependency chain even when resolution is inconsistent (#436)",
          "timestamp": "2026-03-27T01:57:22+01:00",
          "tree_id": "3b8c2e550b838b75d8fc95055d8f597c69aa717d",
          "url": "https://github.com/proximafusion/vmecpp/commit/2d8888e1defbd4d5f2949047dfa742495854ca1f"
        },
        "date": 1783604248024,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00024241048216923515,
            "extra": "iterations: 1149\ncpu: 0.00024240631070496086 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00024171333411985617,
            "extra": "iterations: 1156\ncpu: 0.0002417018079584775 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004701135126320092,
            "extra": "iterations: 592\ncpu: 0.00047010506925675665 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0004885842750122498,
            "extra": "iterations: 572\ncpu: 0.0004885559965034964 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007724312116514962,
            "extra": "iterations: 371\ncpu: 0.0007718379164420488 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0008752748370170594,
            "extra": "iterations: 320\ncpu: 0.0008752369406250002 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026827844289632943,
            "extra": "iterations: 104\ncpu: 0.0026827329038461556 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0029358441630999246,
            "extra": "iterations: 96\ncpu: 0.0029351731666666663 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "1ba362029017a7ee7ffc3dc1c8ed77fb8e209a1a",
          "message": "Migrate ideal MHD core to eigen3 (#452)",
          "timestamp": "2026-03-27T09:50:07+01:00",
          "tree_id": "1595f0d2dadf78d0732658ccb4881db013369167",
          "url": "https://github.com/proximafusion/vmecpp/commit/1ba362029017a7ee7ffc3dc1c8ed77fb8e209a1a"
        },
        "date": 1783604248010,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022436255554882266,
            "extra": "iterations: 1251\ncpu: 0.00022435453157474024 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0002315531133121475,
            "extra": "iterations: 1205\ncpu: 0.00023154940331950215 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00046263529249459265,
            "extra": "iterations: 605\ncpu: 0.00046262756198347117 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0004968715179877002,
            "extra": "iterations: 563\ncpu: 0.0004968181101243337 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007613309397684456,
            "extra": "iterations: 367\ncpu: 0.0007613172316076291 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0009365847278199467,
            "extra": "iterations: 299\ncpu: 0.0009365688628762534 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002852323103924187,
            "extra": "iterations: 98\ncpu: 0.002852177010204081 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0028795163656018447,
            "extra": "iterations: 97\ncpu: 0.0028794624020618575 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "df0547f8fbd75d5c34354b4148de251a5f365491",
          "message": "Simplify a few more ideal mhd expressions with Eigen3 (#453)",
          "timestamp": "2026-03-27T10:07:38+01:00",
          "tree_id": "59491cd9d769f611dc4d46f48ef19e3d7f9b388e",
          "url": "https://github.com/proximafusion/vmecpp/commit/df0547f8fbd75d5c34354b4148de251a5f365491"
        },
        "date": 1783604248402,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00023167770962382472,
            "extra": "iterations: 1204\ncpu: 0.000231665426910299 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00022535790235568318,
            "extra": "iterations: 1248\ncpu: 0.00022535375080128206 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004523673365193029,
            "extra": "iterations: 620\ncpu: 0.00045224667903225805 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0004724178995404925,
            "extra": "iterations: 616\ncpu: 0.00047238745779220806 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007358945477071234,
            "extra": "iterations: 382\ncpu: 0.0007358423507853402 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0008431447518838419,
            "extra": "iterations: 333\ncpu: 0.0008430323513513514 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026972866975344145,
            "extra": "iterations: 104\ncpu: 0.002697148615384619 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027810535808601004,
            "extra": "iterations: 101\ncpu: 0.0027809988712871273 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "159fcbc0380fd0f104d7a272ff18403b5ee8f6ad",
          "message": "Optimize hot loops further, redundant evaluations (#454)",
          "timestamp": "2026-03-27T10:09:00+01:00",
          "tree_id": "95f95268707df1057a1a620917b75aab8bdf025b",
          "url": "https://github.com/proximafusion/vmecpp/commit/159fcbc0380fd0f104d7a272ff18403b5ee8f6ad"
        },
        "date": 1783604248000,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021786853160442223,
            "extra": "iterations: 1284\ncpu: 0.00021784843380062311 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003679483857669508,
            "extra": "iterations: 769\ncpu: 0.00036794163979193763 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048797973583163514,
            "extra": "iterations: 461\ncpu: 0.00048797165292841665 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006737903906748845,
            "extra": "iterations: 416\ncpu: 0.0006737212427884614 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007600901237300696,
            "extra": "iterations: 367\ncpu: 0.0007600376512261583 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001121396064758301,
            "extra": "iterations: 250\ncpu: 0.0011213034559999998 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026644638606480187,
            "extra": "iterations: 105\ncpu: 0.002664187171428572 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037200514475504557,
            "extra": "iterations: 75\ncpu: 0.0037199919333333344 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c9181b56ab9c9f097eb576b58d8a86462659f748",
          "message": "Rename WOutFileContents scalar/1D fields to match Python VmecWOut (#442)",
          "timestamp": "2026-03-27T11:10:05+01:00",
          "tree_id": "350559997c660245c118de00e7f79b90aa1a5712",
          "url": "https://github.com/proximafusion/vmecpp/commit/c9181b56ab9c9f097eb576b58d8a86462659f748"
        },
        "date": 1783604248346,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002192297984030838,
            "extra": "iterations: 1218\ncpu: 0.00021922617405582928 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00037641537950394,
            "extra": "iterations: 752\ncpu: 0.00037636848670212774 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004888599683023376,
            "extra": "iterations: 571\ncpu: 0.0004888290122591945 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006863457315108356,
            "extra": "iterations: 408\ncpu: 0.0006863365367647054 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007580093435339026,
            "extra": "iterations: 370\ncpu: 0.0007579981918918923 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011266809869482813,
            "extra": "iterations: 249\ncpu: 0.0011266074216867469 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.00266459584236145,
            "extra": "iterations: 104\ncpu: 0.0026644935384615392 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003762538368637498,
            "extra": "iterations: 74\ncpu: 0.003762310432432434 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0e531fae6fa57f52c0a52d8400ff9c0e0c76ca0b",
          "message": "Rename and pad half-grid 1D arrays to match Python VmecWOut (#443)",
          "timestamp": "2026-03-27T11:14:09+01:00",
          "tree_id": "a015fbee1637174e551bd2cae6554899ad72e0f6",
          "url": "https://github.com/proximafusion/vmecpp/commit/0e531fae6fa57f52c0a52d8400ff9c0e0c76ca0b"
        },
        "date": 1783604247974,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002191822794580385,
            "extra": "iterations: 1274\ncpu: 0.00021917925667189954 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00039447921205801794,
            "extra": "iterations: 753\ncpu: 0.0003944545351925631 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004910424117642548,
            "extra": "iterations: 573\ncpu: 0.0004910343280977314 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007044911960472807,
            "extra": "iterations: 414\ncpu: 0.00070434222705314 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007616122347132712,
            "extra": "iterations: 367\ncpu: 0.00076154989373297 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.00112735464217815,
            "extra": "iterations: 235\ncpu: 0.0011273376382978718 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026784791396214412,
            "extra": "iterations: 104\ncpu: 0.002678438163461536 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003803153295774718,
            "extra": "iterations: 74\ncpu: 0.0038029384054054046 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ab4e491bed0ce79324ae8112abf1f8424b9342ed",
          "message": "Transpose 2D Fourier arrays to (mnmax, ns) layout (#444)",
          "timestamp": "2026-03-27T11:15:18+01:00",
          "tree_id": "c9712b333516c57c9f32b284646dd9145bc5c50c",
          "url": "https://github.com/proximafusion/vmecpp/commit/ab4e491bed0ce79324ae8112abf1f8424b9342ed"
        },
        "date": 1783604248285,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022399829446658834,
            "extra": "iterations: 1269\ncpu: 0.00022318376516942474 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036290492838304177,
            "extra": "iterations: 771\ncpu: 0.00036279669001297024 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048599847488933144,
            "extra": "iterations: 576\ncpu: 0.00048599035763888904 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006816259244593178,
            "extra": "iterations: 410\ncpu: 0.000681614465853659 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007581988647378234,
            "extra": "iterations: 369\ncpu: 0.0007580339728997288 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001122187419110034,
            "extra": "iterations: 249\ncpu: 0.0011220473212851408 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.00267635981241862,
            "extra": "iterations: 105\ncpu: 0.0026761952666666667 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037428601582845055,
            "extra": "iterations: 75\ncpu: 0.003741859426666666 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "684058866cbb7969ca2e90bcb8c01b6388e3265c",
          "message": "Right-pad profile arrays in C++, add lrfp field, final cleanup (#445)",
          "timestamp": "2026-03-27T11:17:46+01:00",
          "tree_id": "e091a056a071f389319fdb60afead4147c667238",
          "url": "https://github.com/proximafusion/vmecpp/commit/684058866cbb7969ca2e90bcb8c01b6388e3265c"
        },
        "date": 1783604248139,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021911059187051837,
            "extra": "iterations: 1278\ncpu: 0.00021905980281690144 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003702356325869529,
            "extra": "iterations: 755\ncpu: 0.00037021523973509936 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004860121342870925,
            "extra": "iterations: 576\ncpu: 0.0004859805156249999 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000684218768094163,
            "extra": "iterations: 409\ncpu: 0.0006839918801955991 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007583633755468218,
            "extra": "iterations: 367\ncpu: 0.0007583222152588562 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011271961242798343,
            "extra": "iterations: 249\ncpu: 0.0011271770240963862 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026691663832891558,
            "extra": "iterations: 105\ncpu: 0.0026686624857142852 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037497584025065105,
            "extra": "iterations: 75\ncpu: 0.003749532560000001 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "16e7d0e3171a58d62b6fb12fe3e800e56eda11f8",
          "message": "Add HDF5 backwards compatibility for renamed/resized wout fields (#446)",
          "timestamp": "2026-03-27T11:21:40+01:00",
          "tree_id": "53fa7dee60337426e3269ec381c9c2296e887308",
          "url": "https://github.com/proximafusion/vmecpp/commit/16e7d0e3171a58d62b6fb12fe3e800e56eda11f8"
        },
        "date": 1783604248002,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022391460009608984,
            "extra": "iterations: 1091\ncpu: 0.00022391124747937676 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003627185697679396,
            "extra": "iterations: 770\ncpu: 0.0003627127194805195 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004891209161302391,
            "extra": "iterations: 573\ncpu: 0.0004890838097731242 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006779912692397389,
            "extra": "iterations: 402\ncpu: 0.0006779368358208956 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007589412575491722,
            "extra": "iterations: 369\ncpu: 0.0007589295555555554 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011214418411254884,
            "extra": "iterations: 250\ncpu: 0.0011213171079999997 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002668448856898717,
            "extra": "iterations: 105\ncpu: 0.002668400466666667 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037640236519478464,
            "extra": "iterations: 74\ncpu: 0.0037639636891891957 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c81b6dd298d1937ed3eeac79e23d00a1e3234ea9",
          "message": "Pass through outer pydantic serialization contexts (#457)",
          "timestamp": "2026-03-27T21:51:03+01:00",
          "tree_id": "2fb6ae029d04b3512849a84ff0805dddc450076d",
          "url": "https://github.com/proximafusion/vmecpp/commit/c81b6dd298d1937ed3eeac79e23d00a1e3234ea9"
        },
        "date": 1783604248336,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021738023333583002,
            "extra": "iterations: 1281\ncpu: 0.00021736325448868074 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003683849756896732,
            "extra": "iterations: 759\ncpu: 0.00036835583530961814 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048669304107080166,
            "extra": "iterations: 573\ncpu: 0.00048668589005235603 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006872842124864167,
            "extra": "iterations: 408\ncpu: 0.0006872727058823528 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007589425497907934,
            "extra": "iterations: 369\ncpu: 0.0007588442520325207 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011159553527832031,
            "extra": "iterations: 250\ncpu: 0.0011158423879999992 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002662794930594308,
            "extra": "iterations: 105\ncpu: 0.0026626396380952408 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037259546915690105,
            "extra": "iterations: 75\ncpu: 0.003725543773333335 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "3d6cbed0f25c1e5a296d1befb8b7ca15cca07997",
          "message": "[Bugfix] xm, xn should be double but were serialized as int (#459)",
          "timestamp": "2026-03-28T16:13:28+01:00",
          "tree_id": "822a97757f797559878c6a25e283b92420d4c609",
          "url": "https://github.com/proximafusion/vmecpp/commit/3d6cbed0f25c1e5a296d1befb8b7ca15cca07997"
        },
        "date": 1783604248053,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021909318940431605,
            "extra": "iterations: 1273\ncpu: 0.00021902950117831893 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003733729309645681,
            "extra": "iterations: 758\ncpu: 0.0003733665395778365 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048762570403030107,
            "extra": "iterations: 567\ncpu: 0.0004875604197530863 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006817013677889414,
            "extra": "iterations: 411\ncpu: 0.0006816208661800486 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000759909146523411,
            "extra": "iterations: 369\ncpu: 0.0007598087018970193 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011215658187866211,
            "extra": "iterations: 250\ncpu: 0.0011215480919999994 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002675077089896569,
            "extra": "iterations: 104\ncpu: 0.002674769451923076 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037675193838171057,
            "extra": "iterations: 74\ncpu: 0.0037671078108108113 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8f8383a4aef70c8dbde420ecbcc7853b25db06ff",
          "message": "Port calculate_magnetic_field into example",
          "timestamp": "2026-04-06T09:31:46-05:00",
          "tree_id": "a2d3875f78fe4375d9830a215540823e4ab554bd",
          "url": "https://github.com/proximafusion/vmecpp/commit/8f8383a4aef70c8dbde420ecbcc7853b25db06ff"
        },
        "date": 1783604248220,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021779200345792429,
            "extra": "iterations: 1274\ncpu: 0.0002177881978021978 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003673933309520011,
            "extra": "iterations: 762\ncpu: 0.0003673866968503937 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048800566985602976,
            "extra": "iterations: 571\ncpu: 0.0004879814483362521 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006853636436182596,
            "extra": "iterations: 409\ncpu: 0.0006853510195599024 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007607574048249619,
            "extra": "iterations: 368\ncpu: 0.0007607442961956524 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001126308517762456,
            "extra": "iterations: 249\ncpu: 0.0011262413012048198 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002672574633643741,
            "extra": "iterations: 105\ncpu: 0.002672525057142856 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037445831298828125,
            "extra": "iterations: 75\ncpu: 0.003744515906666669 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "1dd86d0442c7be39d04adf32829ca1a4c130c07a",
          "message": "Constructing magnetic field over a flux surface",
          "timestamp": "2026-04-06T09:36:39-05:00",
          "tree_id": "224725e2b7b592bc4ddd6850cc09d8a5ebc29d3a",
          "url": "https://github.com/proximafusion/vmecpp/commit/1dd86d0442c7be39d04adf32829ca1a4c130c07a"
        },
        "date": 1783604248013,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021699502003947864,
            "extra": "iterations: 1279\ncpu: 0.000216969379202502 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036760426442557524,
            "extra": "iterations: 763\ncpu: 0.00036758012581913506 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004881337960007299,
            "extra": "iterations: 574\ncpu: 0.00048812517770034837 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006858194541467078,
            "extra": "iterations: 411\ncpu: 0.0006858090754257907 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007572296503427868,
            "extra": "iterations: 370\ncpu: 0.0007571517459459464 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011180500428839381,
            "extra": "iterations: 249\ncpu: 0.0011179782771084327 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002671078273228237,
            "extra": "iterations: 105\ncpu: 0.0026710338952380948 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037566439310709634,
            "extra": "iterations: 75\ncpu: 0.0037562616133333354 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cce98fe2a193219e631dbf0e78ad8d948c7d74ed",
          "message": "Visualize with pyVista",
          "timestamp": "2026-04-06T09:46:03-05:00",
          "tree_id": "a9c11fa677dc527a7d905a9639d6c593e2fae463",
          "url": "https://github.com/proximafusion/vmecpp/commit/cce98fe2a193219e631dbf0e78ad8d948c7d74ed"
        },
        "date": 1783604248359,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000217675603489283,
            "extra": "iterations: 1279\ncpu: 0.00021767185222830337 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036741052980035504,
            "extra": "iterations: 763\ncpu: 0.00036733516382699867 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048786349737623395,
            "extra": "iterations: 573\ncpu: 0.0004878329650959862 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006809676565775058,
            "extra": "iterations: 410\ncpu: 0.0006808872634146342 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007588539434515912,
            "extra": "iterations: 368\ncpu: 0.0007588409021739133 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011138460075712775,
            "extra": "iterations: 251\ncpu: 0.001113826621513944 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026790820635282076,
            "extra": "iterations: 104\ncpu: 0.0026786173749999997 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037735589345296227,
            "extra": "iterations: 75\ncpu: 0.0037734970266666687 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "52f46df618511ba29fc7e1823f8f3d7c4c77b036",
          "message": "Attempt to construct current density field",
          "timestamp": "2026-04-06T10:15:06-05:00",
          "tree_id": "8f1060b34cdeefd2ad8122f35b042d036d5dc016",
          "url": "https://github.com/proximafusion/vmecpp/commit/52f46df618511ba29fc7e1823f8f3d7c4c77b036"
        },
        "date": 1783604248104,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000218803486165174,
            "extra": "iterations: 1281\ncpu: 0.00021879993598750982 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036995404611819637,
            "extra": "iterations: 756\ncpu: 0.00036993141137566145 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048695873300372934,
            "extra": "iterations: 574\ncpu: 0.000486950630662021 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006854971347053359,
            "extra": "iterations: 409\ncpu: 0.0006854491589242056 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007592287813098773,
            "extra": "iterations: 369\ncpu: 0.0007591281409214088 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011177072487029423,
            "extra": "iterations: 251\ncpu: 0.001117646390438248 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002703537061376479,
            "extra": "iterations: 103\ncpu: 0.0027034914951456304 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037081053382471986,
            "extra": "iterations: 76\ncpu: 0.0037080378421052605 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a079cd5f6ec56873e5ea97a9920fd51947bb3122",
          "message": "Add visualize_magnetic_field.py to index of examples",
          "timestamp": "2026-04-06T10:24:37-05:00",
          "tree_id": "68ab4f0aae4a221c4f477dafcc0de9d9c2d2b035",
          "url": "https://github.com/proximafusion/vmecpp/commit/a079cd5f6ec56873e5ea97a9920fd51947bb3122"
        },
        "date": 1783604248259,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022508055450831964,
            "extra": "iterations: 1244\ncpu: 0.00022506995578778138 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003668145328209709,
            "extra": "iterations: 758\ncpu: 0.00036678942216358844 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004879764888597572,
            "extra": "iterations: 575\ncpu: 0.00048796817739130436 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006794055688728407,
            "extra": "iterations: 412\ncpu: 0.0006793951140776698 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007564738754313589,
            "extra": "iterations: 371\ncpu: 0.000756434970350404 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001118027687072754,
            "extra": "iterations: 250\ncpu: 0.0011179705880000005 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026624157315208803,
            "extra": "iterations: 105\ncpu: 0.0026623721904761887 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003751227060953776,
            "extra": "iterations: 75\ncpu: 0.0037507284666666685 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "93fc5bedd34a265c1afe235e08d01407f3f448fc",
          "message": "Rename suptht -> supu and supzta -> supv for consistency",
          "timestamp": "2026-04-07T21:06:10-05:00",
          "tree_id": "4d41779cd8bb0e4f6aa7a20f4412bf9742c12056",
          "url": "https://github.com/proximafusion/vmecpp/commit/93fc5bedd34a265c1afe235e08d01407f3f448fc"
        },
        "date": 1783604248235,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002175111326535292,
            "extra": "iterations: 1267\ncpu: 0.00021750761247040255 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003758960230032393,
            "extra": "iterations: 757\ncpu: 0.00037588910964332894 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000486109819677141,
            "extra": "iterations: 576\ncpu: 0.0004860792586805556 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006835519098768048,
            "extra": "iterations: 408\ncpu: 0.0006835191691176469 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0008298534335512104,
            "extra": "iterations: 330\ncpu: 0.0008298028909090905 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011162395477294922,
            "extra": "iterations: 250\ncpu: 0.0011161297359999997 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002661809467134022,
            "extra": "iterations: 105\ncpu: 0.002661288971428569 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003682280841626619,
            "extra": "iterations: 76\ncpu: 0.0036820184342105304 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "82727604d701fd1202fef14aa41b6f177a249ecb",
          "message": "Tentative J calculation for when currumnc and currvmnc are available",
          "timestamp": "2026-04-07T21:12:58-05:00",
          "tree_id": "9f75260c54505d58b2787a498f5c970e5e7b6983",
          "url": "https://github.com/proximafusion/vmecpp/commit/82727604d701fd1202fef14aa41b6f177a249ecb"
        },
        "date": 1783604248189,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021972746219275135,
            "extra": "iterations: 1272\ncpu: 0.00021969361871069186 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00037994417203526923,
            "extra": "iterations: 735\ncpu: 0.00037991899591836745 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004875278472900391,
            "extra": "iterations: 575\ncpu: 0.00048752017391304344 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000686351264395365,
            "extra": "iterations: 410\ncpu: 0.0006862085707317076 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007659848270520486,
            "extra": "iterations: 366\ncpu: 0.0007659409699453549 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011000004340344528,
            "extra": "iterations: 254\ncpu: 0.0010997352755905503 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002657670066470192,
            "extra": "iterations: 105\ncpu: 0.0026574931428571405 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037371476491292323,
            "extra": "iterations: 75\ncpu: 0.003737091960000001 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "daed90fa13b4c18db1121abaf2ff08a9374ff53a",
          "message": "Calculate Cartesian coordinates only if requested",
          "timestamp": "2026-04-07T21:17:41-05:00",
          "tree_id": "7c87983e2932e664481d10b0187f1cb7d2e62be9",
          "url": "https://github.com/proximafusion/vmecpp/commit/daed90fa13b4c18db1121abaf2ff08a9374ff53a"
        },
        "date": 1783604248392,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021843805335585004,
            "extra": "iterations: 1273\ncpu: 0.00021842625844461903 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003603283114788772,
            "extra": "iterations: 778\ncpu: 0.00036024319794344484 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004888700402301291,
            "extra": "iterations: 575\ncpu: 0.0004887569113043479 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006836949325189358,
            "extra": "iterations: 410\ncpu: 0.0006836821926829268 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007735037285348644,
            "extra": "iterations: 368\ncpu: 0.0007734565217391311 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011228275299072267,
            "extra": "iterations: 250\ncpu: 0.0011227549239999988 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002665417534964425,
            "extra": "iterations: 105\ncpu: 0.002665371876190479 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037387402852376308,
            "extra": "iterations: 75\ncpu: 0.003737723053333332 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cf0ae1fbc9884abd5b7e4db7a7f3dc7dcd6b93a6",
          "message": "Precompute and reuse sin and cos terms",
          "timestamp": "2026-04-07T21:20:24-05:00",
          "tree_id": "2dea4c1b4d9b05940012201307a99708d64d4fa7",
          "url": "https://github.com/proximafusion/vmecpp/commit/cf0ae1fbc9884abd5b7e4db7a7f3dc7dcd6b93a6"
        },
        "date": 1783604248367,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021693128976762666,
            "extra": "iterations: 1288\ncpu: 0.00021692799223602493 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.000380092520054763,
            "extra": "iterations: 738\ncpu: 0.000380066443089431 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005628660347703084,
            "extra": "iterations: 563\ncpu: 0.0005628182220248668 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006824953883301978,
            "extra": "iterations: 408\ncpu: 0.0006824520955882352 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007622300441648396,
            "extra": "iterations: 367\ncpu: 0.0007622178038147139 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011399780831685882,
            "extra": "iterations: 246\ncpu: 0.0011399599065040648 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002654475257510231,
            "extra": "iterations: 105\ncpu: 0.002654164447619048 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003759081299240525,
            "extra": "iterations: 74\ncpu: 0.0037590344864864914 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7e673720abc3c6c283ee636f91470b540f7b71aa",
          "message": "Remove unnecessary comment",
          "timestamp": "2026-04-07T21:23:05-05:00",
          "tree_id": "0fab048c9a714e1df431ec297d2e69ce4ef99b45",
          "url": "https://github.com/proximafusion/vmecpp/commit/7e673720abc3c6c283ee636f91470b540f7b71aa"
        },
        "date": 1783604248183,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021882098458302857,
            "extra": "iterations: 1271\ncpu: 0.0002188172344610543 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.000368296786358482,
            "extra": "iterations: 760\ncpu: 0.00036829054210526314 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004864761398898231,
            "extra": "iterations: 576\ncpu: 0.00048643135937500003 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006830529864977167,
            "extra": "iterations: 411\ncpu: 0.0006829709708029198 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007573039874151794,
            "extra": "iterations: 369\ncpu: 0.0007571656639566396 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011160763136419168,
            "extra": "iterations: 251\ncpu: 0.0011160143266932272 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002678483724594116,
            "extra": "iterations: 104\ncpu: 0.002678349855769228 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003724652367669183,
            "extra": "iterations: 74\ncpu: 0.0037245808108108103 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "3ec621eb49fd0d908f96157352da9aed133f8c82",
          "message": "Automatic changes by pre-commit run --all-files",
          "timestamp": "2026-04-07T21:28:26-05:00",
          "tree_id": "4da7da2b437c6e1a3277f6bfb356e29f89d93514",
          "url": "https://github.com/proximafusion/vmecpp/commit/3ec621eb49fd0d908f96157352da9aed133f8c82"
        },
        "date": 1783604248055,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002170961799351036,
            "extra": "iterations: 1269\ncpu: 0.00021706797714736014 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003718631020907698,
            "extra": "iterations: 754\ncpu: 0.00037183926127320956 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004875332376231318,
            "extra": "iterations: 575\ncpu: 0.00048752520695652163 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006743726822046133,
            "extra": "iterations: 416\ncpu: 0.0006743609158653851 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007627075309649476,
            "extra": "iterations: 367\ncpu: 0.0007626931689373301 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011216745376586915,
            "extra": "iterations: 250\ncpu: 0.001121561752 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026600315457298647,
            "extra": "iterations: 105\ncpu: 0.002659911790476193 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037632085181571345,
            "extra": "iterations: 74\ncpu: 0.003762751621621622 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "32790ff82a678f7d666fbef11845d30b5ca78fce",
          "message": "Import numpy so example function is callable as a module",
          "timestamp": "2026-04-07T22:34:55-05:00",
          "tree_id": "e4299d9e5d858c85b7679e831d795e0288d5c253",
          "url": "https://github.com/proximafusion/vmecpp/commit/32790ff82a678f7d666fbef11845d30b5ca78fce"
        },
        "date": 1783604248034,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002190054587598117,
            "extra": "iterations: 1272\ncpu: 0.00021899542924528308 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003754010936557847,
            "extra": "iterations: 745\ncpu: 0.0003753420268456376 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004873811575593849,
            "extra": "iterations: 574\ncpu: 0.000487372736933798 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006801221907571788,
            "extra": "iterations: 413\ncpu: 0.0006800667457627118 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007581271453278498,
            "extra": "iterations: 369\ncpu: 0.0007580628997289968 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011146116066738905,
            "extra": "iterations: 251\ncpu: 0.0011145926573705176 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002660596938360305,
            "extra": "iterations: 105\ncpu: 0.00266054843809524 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037654368082682293,
            "extra": "iterations: 75\ncpu: 0.0037652349333333335 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "dcaa3059ffab97b6805c2d9afdb3c90909fa5bc9",
          "message": "Reshape B and J field to be compatible with a volumetric grid",
          "timestamp": "2026-04-07T23:24:05-05:00",
          "tree_id": "4160e98df87f6b374707cba451ad7e585d9d7718",
          "url": "https://github.com/proximafusion/vmecpp/commit/dcaa3059ffab97b6805c2d9afdb3c90909fa5bc9"
        },
        "date": 1783604248397,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002185419691560414,
            "extra": "iterations: 1278\ncpu: 0.00021850740140845072 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003849997008142393,
            "extra": "iterations: 726\ncpu: 0.0003849772699724519 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004911572783143371,
            "extra": "iterations: 572\ncpu: 0.0004910819527972027 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006824970245361329,
            "extra": "iterations: 405\ncpu: 0.0006823363802469136 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007602501174677974,
            "extra": "iterations: 368\ncpu: 0.0007602370760869567 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011208209991455079,
            "extra": "iterations: 250\ncpu: 0.0011207423399999996 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002668056033906483,
            "extra": "iterations: 105\ncpu: 0.0026679238571428576 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037543233235677086,
            "extra": "iterations: 75\ncpu: 0.0037542563466666638 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "EdoAlvarezR",
            "email": "edo.alvarezr@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "279080900a2abedea6a3f478a1218321cd40b385",
          "message": "Automatic changes by pre-commit run --all-files",
          "timestamp": "2026-04-07T23:27:35-05:00",
          "tree_id": "0659fdd49f80b3d0af9f2ee548f1a7880f20869c",
          "url": "https://github.com/proximafusion/vmecpp/commit/279080900a2abedea6a3f478a1218321cd40b385"
        },
        "date": 1783604248018,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022240846151437603,
            "extra": "iterations: 1269\ncpu: 0.00022236698502758082 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00037030536661703125,
            "extra": "iterations: 756\ncpu: 0.0003702989933862435 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004874523826267409,
            "extra": "iterations: 575\ncpu: 0.0004873733826086957 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006801196672384022,
            "extra": "iterations: 412\ncpu: 0.0006801093762135925 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007573043977892077,
            "extra": "iterations: 370\ncpu: 0.00075718592972973 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011107732379247272,
            "extra": "iterations: 252\ncpu: 0.0011107533531746038 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026732626415434343,
            "extra": "iterations: 105\ncpu: 0.002672886066666667 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037448247273763023,
            "extra": "iterations: 75\ncpu: 0.0037445036133333313 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "bb623a411e381b7a45c3905502ce63c2d6886b56",
          "message": "Implement the current density Fourier coefficients. (#474)",
          "timestamp": "2026-04-10T12:26:12+02:00",
          "tree_id": "27605f632f483ab8cb9a4a98f54cbd75d36dee9c",
          "url": "https://github.com/proximafusion/vmecpp/commit/bb623a411e381b7a45c3905502ce63c2d6886b56"
        },
        "date": 1783604248312,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021732094874634552,
            "extra": "iterations: 1284\ncpu: 0.0002173171479750779 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003669368173941716,
            "extra": "iterations: 763\ncpu: 0.00036693006684141555 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048625281968041865,
            "extra": "iterations: 573\ncpu: 0.0004862440279232112 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006791706224089688,
            "extra": "iterations: 412\ncpu: 0.0006791577475728159 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000759789614173455,
            "extra": "iterations: 369\ncpu: 0.0007597772601626012 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011143294938532006,
            "extra": "iterations: 251\ncpu: 0.0011143115617529877 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002657136462983631,
            "extra": "iterations: 105\ncpu: 0.0026569987904761926 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003725681304931641,
            "extra": "iterations: 75\ncpu: 0.0037256193466666627 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d79e972b074dedc10976091e00e12f68551d5fb3",
          "message": "Restrict pydantic version, breaking change (#480)",
          "timestamp": "2026-04-16T11:06:55+02:00",
          "tree_id": "c081f0908ae5d478b1d3e6f9b2b2ae2457d276fb",
          "url": "https://github.com/proximafusion/vmecpp/commit/d79e972b074dedc10976091e00e12f68551d5fb3"
        },
        "date": 1783604248390,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021753434135195817,
            "extra": "iterations: 1202\ncpu: 0.00021747156655574045 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003694573188210554,
            "extra": "iterations: 748\ncpu: 0.00036943836898395725 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004897088437647252,
            "extra": "iterations: 572\ncpu: 0.0004897004020979022 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006823661553598668,
            "extra": "iterations: 411\ncpu: 0.0006822373746958635 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007599909011631796,
            "extra": "iterations: 365\ncpu: 0.0007599767123287665 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011201219558715822,
            "extra": "iterations: 250\ncpu: 0.0011200356119999988 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002670219966343471,
            "extra": "iterations: 105\ncpu: 0.002669821904761903 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037561130523681642,
            "extra": "iterations: 75\ncpu: 0.003756049146666669 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "48dd1a146c28f3a06e53af22dbc2e8d2c32c38cc",
          "message": "Also add .github/workflows/copilot-setup-steps.yml to pre-install vmecpp and its Ubuntu system dependencies in the Copilot cloud agent environment. (#483)",
          "timestamp": "2026-04-16T13:33:19+02:00",
          "tree_id": "1aed1e09f2dffc1c1fdbc2e0601a7f79d30d2d0d",
          "url": "https://github.com/proximafusion/vmecpp/commit/48dd1a146c28f3a06e53af22dbc2e8d2c32c38cc"
        },
        "date": 1783604248087,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022113793143704354,
            "extra": "iterations: 1243\ncpu: 0.00022113396460176992 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036664845431662356,
            "extra": "iterations: 764\ncpu: 0.00036664226308900526 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004880204408065133,
            "extra": "iterations: 575\ncpu: 0.00048799468521739135 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006806548048810262,
            "extra": "iterations: 410\ncpu: 0.0006806436829268294 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007593121308944413,
            "extra": "iterations: 369\ncpu: 0.0007592990189701897 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011151501856952076,
            "extra": "iterations: 251\ncpu: 0.001115080111553786 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026646614074707035,
            "extra": "iterations: 105\ncpu: 0.0026644346761904767 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003736839294433594,
            "extra": "iterations: 75\ncpu: 0.0037367725599999936 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8adda439309a04d379648e25c4f64f2c5f3bdedb",
          "message": "Add a Python 3.13 Nix development shell (#479)",
          "timestamp": "2026-04-16T14:44:28+02:00",
          "tree_id": "e8e3f2cfad15e4222d2e4ee2fbb465245fd99bf3",
          "url": "https://github.com/proximafusion/vmecpp/commit/8adda439309a04d379648e25c4f64f2c5f3bdedb"
        },
        "date": 1783604248206,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021744292313225793,
            "extra": "iterations: 1289\ncpu: 0.00021743891621411955 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00037031413069489453,
            "extra": "iterations: 757\ncpu: 0.0003702446525759577 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000488331861663283,
            "extra": "iterations: 570\ncpu: 0.0004883230210526317 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000679861573339666,
            "extra": "iterations: 412\ncpu: 0.0006798488956310685 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007591208791345117,
            "extra": "iterations: 369\ncpu: 0.0007589750162601623 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001117422286257801,
            "extra": "iterations: 251\ncpu: 0.0011174033227091635 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026735373905726846,
            "extra": "iterations: 105\ncpu: 0.0026734909714285706 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.00381337629782187,
            "extra": "iterations: 74\ncpu: 0.0038124879729729707 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "fa1b951045b2243a0c9340d0b16bec4a60e5136e",
          "message": "Update README.md",
          "timestamp": "2026-04-23T10:12:38+02:00",
          "tree_id": "f3fac97cd72cc9ec7d5fd85ca957778291adbb65",
          "url": "https://github.com/proximafusion/vmecpp/commit/fa1b951045b2243a0c9340d0b16bec4a60e5136e"
        },
        "date": 1783604248438,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022128687344148327,
            "extra": "iterations: 1281\ncpu: 0.00022126399843871975 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036870523502952177,
            "extra": "iterations: 760\ncpu: 0.00036868734210526315 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000488545466051884,
            "extra": "iterations: 573\ncpu: 0.0004885368883071554 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006820662727569068,
            "extra": "iterations: 358\ncpu: 0.0006820242988826814 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007575805271221049,
            "extra": "iterations: 369\ncpu: 0.0007575675067750682 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011182012557983398,
            "extra": "iterations: 250\ncpu: 0.001118179776 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026628925686790835,
            "extra": "iterations: 105\ncpu: 0.002662736638095239 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037560495170387066,
            "extra": "iterations: 74\ncpu: 0.0037559888918918937 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "51527dfb71c83aee7a8d83df6203758e2c78ef18",
          "message": "Miller recurrence relation for numerically stable NESTOR at high mpol/ntor (#484)",
          "timestamp": "2026-04-24T14:28:08+02:00",
          "tree_id": "46ee572b3021f833c339ffbbac9effe65b0f54e2",
          "url": "https://github.com/proximafusion/vmecpp/commit/51527dfb71c83aee7a8d83df6203758e2c78ef18"
        },
        "date": 1783604248100,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021887160575920304,
            "extra": "iterations: 1281\ncpu: 0.0002188563348946136 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036667215089810474,
            "extra": "iterations: 763\ncpu: 0.0003666407391874181 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004886776417285413,
            "extra": "iterations: 572\ncpu: 0.000488649770979021 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006795708838733,
            "extra": "iterations: 413\ncpu: 0.0006795593341404361 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007626152818144506,
            "extra": "iterations: 367\ncpu: 0.0007625680517711174 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011133777192864286,
            "extra": "iterations: 251\ncpu: 0.0011133070039840642 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002684475148765786,
            "extra": "iterations: 103\ncpu: 0.0026842928252427197 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037405395507812502,
            "extra": "iterations: 75\ncpu: 0.0037404716400000026 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "f3f1a1068147acd40e45ed847d5a8cc06015b2a3",
          "message": "fix: write test output files to temp directories instead of CWD (#489)",
          "timestamp": "2026-04-27T10:41:19+02:00",
          "tree_id": "d77249fb049a39855dbb393d8b2771a785f8e3b0",
          "url": "https://github.com/proximafusion/vmecpp/commit/f3f1a1068147acd40e45ed847d5a8cc06015b2a3"
        },
        "date": 1783604248430,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022050860552381643,
            "extra": "iterations: 1268\ncpu: 0.00022050479100946374 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003647596535354039,
            "extra": "iterations: 769\ncpu: 0.0003646992496749025 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000487529155280855,
            "extra": "iterations: 576\ncpu: 0.0004875054861111113 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006827478571239759,
            "extra": "iterations: 411\ncpu: 0.000682737170316302 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007582111410332243,
            "extra": "iterations: 369\ncpu: 0.0007580931626016264 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011203622817993165,
            "extra": "iterations: 250\ncpu: 0.0011203442279999994 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026708716437930154,
            "extra": "iterations: 105\ncpu: 0.0026706635428571414 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037511761983235682,
            "extra": "iterations: 75\ncpu: 0.0037504822266666692 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9c2041a1d23a6d9c69f9d98caff279d7166f4282",
          "message": "Fix normalize_by_currents having no effect on MagneticFieldResponseTable (#487)",
          "timestamp": "2026-04-27T14:21:51Z",
          "tree_id": "b2a4004f87cba6621d18e876a2da25becd03f3da",
          "url": "https://github.com/proximafusion/vmecpp/commit/9c2041a1d23a6d9c69f9d98caff279d7166f4282"
        },
        "date": 1783604248248,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021781221578867527,
            "extra": "iterations: 1281\ncpu: 0.0002178039695550352 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003717946690439389,
            "extra": "iterations: 755\ncpu: 0.00037177043311258285 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048194595218933737,
            "extra": "iterations: 582\ncpu: 0.00048191799828178704 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006822595386458492,
            "extra": "iterations: 409\ncpu: 0.0006821824303178484 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007561142380173143,
            "extra": "iterations: 370\ncpu: 0.0007561025135135134 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011402397620968703,
            "extra": "iterations: 246\ncpu: 0.0011402234715447155 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026580657599107275,
            "extra": "iterations: 106\ncpu: 0.0026578896698113198 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037058607737223306,
            "extra": "iterations: 75\ncpu: 0.003705608733333333 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d6875c57529a77d7e32e6845b9d3ee40afdda5b3",
          "message": "Abseil errors instead of LOG(FATAL) (#493)",
          "timestamp": "2026-04-27T17:58:22+02:00",
          "tree_id": "a4313fa42f5838bf688910bffd3f9f3d68f9827b",
          "url": "https://github.com/proximafusion/vmecpp/commit/d6875c57529a77d7e32e6845b9d3ee40afdda5b3"
        },
        "date": 1783604248385,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021712785539102592,
            "extra": "iterations: 1291\ncpu: 0.0002170956545313711 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003674399993736077,
            "extra": "iterations: 761\ncpu: 0.00036739708278580826 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004889615543225673,
            "extra": "iterations: 573\ncpu: 0.0004887919267015708 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006816671422508222,
            "extra": "iterations: 411\ncpu: 0.0006815942579075429 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007606064495833024,
            "extra": "iterations: 368\ncpu: 0.000760594834239131 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.00111864382527264,
            "extra": "iterations: 251\ncpu: 0.0011184611035856568 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026555781094533093,
            "extra": "iterations: 106\ncpu: 0.002655532773584908 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003740777969360352,
            "extra": "iterations: 75\ncpu: 0.003740538813333328 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "97cb9d20e1c85fdf05f9dbfae0549754daacbf4a",
          "message": "Scientific variable names get flagged incorrectly too often by clang-tidy (#491)",
          "timestamp": "2026-04-27T18:18:23+02:00",
          "tree_id": "b23af3b37dab107f49bb1792b9e40ac499ab3af8",
          "url": "https://github.com/proximafusion/vmecpp/commit/97cb9d20e1c85fdf05f9dbfae0549754daacbf4a"
        },
        "date": 1783604248238,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022548734213332896,
            "extra": "iterations: 1284\ncpu: 0.00022548371728971963 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00037059361341769105,
            "extra": "iterations: 756\ncpu: 0.000370560253968254 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004880859088385895,
            "extra": "iterations: 559\ncpu: 0.0004880542790697675 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006820780313043849,
            "extra": "iterations: 413\ncpu: 0.0006820301937046002 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.00076231159611211,
            "extra": "iterations: 371\ncpu: 0.0007622364609164426 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011254639989400964,
            "extra": "iterations: 249\ncpu: 0.001125384445783133 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002660531089419411,
            "extra": "iterations: 105\ncpu: 0.0026602898190476194 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0039169756571451825,
            "extra": "iterations: 75\ncpu: 0.003912390973333331 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "42e3ec5f8b9b731a5ad717a06ec19adf7131111f",
          "message": "Use std::unreachable instead of LOG(FATAL) for unreachable code (#494)",
          "timestamp": "2026-04-27T23:01:30+02:00",
          "tree_id": "c57ee3789ceb1a61f7bbd68a6322c8d76d8bac12",
          "url": "https://github.com/proximafusion/vmecpp/commit/42e3ec5f8b9b731a5ad717a06ec19adf7131111f"
        },
        "date": 1783604248071,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022039349923686923,
            "extra": "iterations: 1276\ncpu: 0.00022038971394984332 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00039512833011572205,
            "extra": "iterations: 763\ncpu: 0.00039510985190039326 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004856276553466374,
            "extra": "iterations: 577\ncpu: 0.00048559618717504313 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000675943098872541,
            "extra": "iterations: 415\ncpu: 0.0006759337831325304 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007601598413979135,
            "extra": "iterations: 369\ncpu: 0.0007601463468834687 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011196493619849326,
            "extra": "iterations: 247\ncpu: 0.001119594315789473 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002660919371105376,
            "extra": "iterations: 105\ncpu: 0.002660636276190475 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0036991113110592493,
            "extra": "iterations: 76\ncpu: 0.003699049250000004 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b24222e51f3d3ab1f9acbff4139fbca21783a0b7",
          "message": "Terminate on NaN or inf, return NaN instead of terminating in tridiagonal solve (#496)",
          "timestamp": "2026-04-28T09:42:10+02:00",
          "tree_id": "77e85f47c89c700befb46c86153741f0fbeb0185",
          "url": "https://github.com/proximafusion/vmecpp/commit/b24222e51f3d3ab1f9acbff4139fbca21783a0b7"
        },
        "date": 1783604248298,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021781016847847217,
            "extra": "iterations: 1281\ncpu: 0.00021779939578454337 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003701996583398309,
            "extra": "iterations: 759\ncpu: 0.0003701931317523056 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004873105754023013,
            "extra": "iterations: 575\ncpu: 0.0004872620799999999 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006736173468121027,
            "extra": "iterations: 413\ncpu: 0.0006736057094430995 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007629868769840583,
            "extra": "iterations: 367\ncpu: 0.0007629456185286106 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011352163846375513,
            "extra": "iterations: 244\ncpu: 0.0011351963565573771 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026606309981573196,
            "extra": "iterations: 105\ncpu: 0.0026605887809523837 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037339178721110026,
            "extra": "iterations: 75\ncpu: 0.003733627906666666 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "6f846de63275c49dc5f60da16e2dfb052055c50d",
          "message": "TSAN docker setup (#492)",
          "timestamp": "2026-04-28T12:23:06+02:00",
          "tree_id": "6ea1141f29156aa8a27ad98b5fa089076d812218",
          "url": "https://github.com/proximafusion/vmecpp/commit/6f846de63275c49dc5f60da16e2dfb052055c50d"
        },
        "date": 1783604248158,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021930401199544912,
            "extra": "iterations: 1271\ncpu: 0.00021928400944138475 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003700196506797559,
            "extra": "iterations: 757\ncpu: 0.00037000304755614277 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048660569720798073,
            "extra": "iterations: 576\ncpu: 0.00048658180729166677 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006735783356886643,
            "extra": "iterations: 416\ncpu: 0.0006734024543269231 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007824851147955358,
            "extra": "iterations: 358\ncpu: 0.0007824716061452516 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011378090556074933,
            "extra": "iterations: 246\ncpu: 0.001137787577235771 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026528608231317433,
            "extra": "iterations: 105\ncpu: 0.0026523991238095236 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003716096878051758,
            "extra": "iterations: 75\ncpu: 0.0037160251066666685 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c9772aed7885bd7ed62e4fbaae99d8a30915e74c",
          "message": "Niter wording improved (#504)",
          "timestamp": "2026-04-28T16:59:51+02:00",
          "tree_id": "7cc623a63d553fa23218de359440aa6bc4c7d20e",
          "url": "https://github.com/proximafusion/vmecpp/commit/c9772aed7885bd7ed62e4fbaae99d8a30915e74c"
        },
        "date": 1783604248351,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021762792454209438,
            "extra": "iterations: 1290\ncpu: 0.0002175880542635659 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003663884234147179,
            "extra": "iterations: 763\ncpu: 0.00036637216251638274 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004852090327384976,
            "extra": "iterations: 578\ncpu: 0.0004851781349480969 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006726960101759577,
            "extra": "iterations: 415\ncpu: 0.000672683079518072 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007584807665451713,
            "extra": "iterations: 368\ncpu: 0.0007584690978260869 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001132251762668131,
            "extra": "iterations: 247\ncpu: 0.0011320402631578947 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026704370975494385,
            "extra": "iterations: 104\ncpu: 0.0026703854519230775 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037094542854710635,
            "extra": "iterations: 76\ncpu: 0.003709387328947369 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ea76d2124c0a47e294f7369d8fa36781d5f7d2fe",
          "message": "FFTW3 dependencies (#501)",
          "timestamp": "2026-04-28T17:32:26+02:00",
          "tree_id": "e7fb40efbab4d320cc28192ea0b0a916d6b7500d",
          "url": "https://github.com/proximafusion/vmecpp/commit/ea76d2124c0a47e294f7369d8fa36781d5f7d2fe"
        },
        "date": 1783604248417,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021795799760039217,
            "extra": "iterations: 1285\ncpu: 0.00021794673229571986 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00035533112317777534,
            "extra": "iterations: 788\ncpu: 0.000355324673857868 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005201096871528483,
            "extra": "iterations: 538\ncpu: 0.0005200582249070634 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006749175021037964,
            "extra": "iterations: 414\ncpu: 0.0006749060048309181 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007594058184119744,
            "extra": "iterations: 369\ncpu: 0.0007593290650406506 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011224708557128907,
            "extra": "iterations: 250\ncpu: 0.0011224514719999992 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026640960148402626,
            "extra": "iterations: 105\ncpu: 0.0026640548761904764 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003737758000691732,
            "extra": "iterations: 75\ncpu: 0.003737402826666667 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a2b31a8383c130052994620d0585b075b7a13fa6",
          "message": "Refactor LaplaceSolver to use Eigen3 for matrix operations (#499)",
          "timestamp": "2026-04-28T15:52:23Z",
          "tree_id": "7cdf4e474746688ac76d605713e9bb783da32bd4",
          "url": "https://github.com/proximafusion/vmecpp/commit/a2b31a8383c130052994620d0585b075b7a13fa6"
        },
        "date": 1783604248266,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000221659429371357,
            "extra": "iterations: 1280\ncpu: 0.0002215926578125 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003670717376509523,
            "extra": "iterations: 765\ncpu: 0.0003670145176470588 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004840880580750041,
            "extra": "iterations: 577\ncpu: 0.0004840797261698441 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006735949562146113,
            "extra": "iterations: 416\ncpu: 0.0006735601394230767 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007607839677644813,
            "extra": "iterations: 368\ncpu: 0.0007607386603260867 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011261917022337398,
            "extra": "iterations: 249\ncpu: 0.0011261348353413649 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026796023050944012,
            "extra": "iterations: 105\ncpu: 0.0026790256571428577 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0038467694635260594,
            "extra": "iterations: 73\ncpu: 0.0038465623835616386 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "12b7f60d9a4c5417ceddaa4d5ea5ab0e25febf6a",
          "message": "Laplace solver stellarator asym added (#502)",
          "timestamp": "2026-04-28T18:12:25+02:00",
          "tree_id": "2f14fa7d2c8decb9c434dc27157c2c2916187b00",
          "url": "https://github.com/proximafusion/vmecpp/commit/12b7f60d9a4c5417ceddaa4d5ea5ab0e25febf6a"
        },
        "date": 1783604247992,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002198252984829937,
            "extra": "iterations: 1274\ncpu: 0.00021978385086342234 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003624031146888183,
            "extra": "iterations: 763\ncpu: 0.00036238912319790307 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048653022102687675,
            "extra": "iterations: 575\ncpu: 0.00048648139652173903 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006756162068930017,
            "extra": "iterations: 415\ncpu: 0.0006755018120481927 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007588850344465079,
            "extra": "iterations: 366\ncpu: 0.0007588311065573774 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011217412948608398,
            "extra": "iterations: 250\ncpu: 0.001121528144000001 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026657717568533764,
            "extra": "iterations: 105\ncpu: 0.0026656316952380927 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037626640216724292,
            "extra": "iterations: 74\ncpu: 0.0037620715405405443 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c8e567fc623c429024c11a4555e0fd8a58b71679",
          "message": "Revert \"FFTW3 dependencies (#501)\" (#506)",
          "timestamp": "2026-04-29T15:14:21+02:00",
          "tree_id": "4895e3fe13cfbe6572522fbff370fdb42ef68df8",
          "url": "https://github.com/proximafusion/vmecpp/commit/c8e567fc623c429024c11a4555e0fd8a58b71679"
        },
        "date": 1783604248343,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021793671219657624,
            "extra": "iterations: 1282\ncpu: 0.00021792383541341655 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003682990581697923,
            "extra": "iterations: 761\ncpu: 0.0003682932877792379 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048824095975231216,
            "extra": "iterations: 574\ncpu: 0.0004882157108013936 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006730328336286773,
            "extra": "iterations: 418\ncpu: 0.0006729783253588518 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007597110841585243,
            "extra": "iterations: 368\ncpu: 0.0007596998396739133 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.00112684950771102,
            "extra": "iterations: 249\ncpu: 0.0011268321325301194 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026686441330682664,
            "extra": "iterations: 105\ncpu: 0.0026685998095238085 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003703186386509946,
            "extra": "iterations: 76\ncpu: 0.003702878723684209 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "378b4ff6edc6dd41d3afde48fc66f38723726297",
          "message": "Reduce free boundary oversubscription due to Eigen low level parallelism on some platforms (#507)",
          "timestamp": "2026-04-29T15:33:24+02:00",
          "tree_id": "32526f65a9fdcc89d31c91b4fdbf874ceb6c27b9",
          "url": "https://github.com/proximafusion/vmecpp/commit/378b4ff6edc6dd41d3afde48fc66f38723726297"
        },
        "date": 1783604248039,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002179915263173745,
            "extra": "iterations: 1289\ncpu: 0.00021798739410395658 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003804984879851504,
            "extra": "iterations: 733\ncpu: 0.0003804714952251024 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004876132633375085,
            "extra": "iterations: 575\ncpu: 0.00048757524695652177 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006880514158956541,
            "extra": "iterations: 407\ncpu: 0.0006879850221130219 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007573162637105802,
            "extra": "iterations: 369\ncpu: 0.0007572699376693772 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011488902764242204,
            "extra": "iterations: 244\ncpu: 0.0011488372049180326 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0026555488694388913,
            "extra": "iterations: 106\ncpu: 0.002655410330188681 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.003804065085746147,
            "extra": "iterations: 74\ncpu: 0.0038035156351351376 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7bb4582991a98dc91f165f9659e9338fefb7e8bb",
          "message": "Reduce thread spinning on first parallel region enter (#508)",
          "timestamp": "2026-04-30T01:39:45+02:00",
          "tree_id": "c102485b620e8b188776c29b801d5c85342061c5",
          "url": "https://github.com/proximafusion/vmecpp/commit/7bb4582991a98dc91f165f9659e9338fefb7e8bb"
        },
        "date": 1783604248169,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021766412529960908,
            "extra": "iterations: 1228\ncpu: 0.0002176530675895766 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036236451269736276,
            "extra": "iterations: 774\ncpu: 0.0003623432700258398 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004913806915283203,
            "extra": "iterations: 570\ncpu: 0.0004912840789473683 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006916703688194816,
            "extra": "iterations: 409\ncpu: 0.0006916601833740833 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007606676553835336,
            "extra": "iterations: 367\ncpu: 0.0007605952043596734 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011123598568023197,
            "extra": "iterations: 252\ncpu: 0.0011123433690476184 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002658221835181827,
            "extra": "iterations: 105\ncpu: 0.00265818055238095 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0037460772196451823,
            "extra": "iterations: 75\ncpu: 0.0037453264399999967 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "01b1fcb56a26188eb71a74fcf9cdf0c7349e5c04",
          "message": "Diagnose how much force residual is being truncated away (#495)",
          "timestamp": "2026-04-30T21:49:38+02:00",
          "tree_id": "89a1ebe9dc8cda139a3ada16a72c43556f11f0ff",
          "url": "https://github.com/proximafusion/vmecpp/commit/01b1fcb56a26188eb71a74fcf9cdf0c7349e5c04"
        },
        "date": 1783604247947,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021892192804417906,
            "extra": "iterations: 1266\ncpu: 0.00021890973538704587 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00037879155580267354,
            "extra": "iterations: 738\ncpu: 0.00037875413008130095 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00048660941240264154,
            "extra": "iterations: 574\ncpu: 0.00048660075435540096 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006837926061774113,
            "extra": "iterations: 411\ncpu: 0.0006837604136253046 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0007603284474965689,
            "extra": "iterations: 370\ncpu: 0.0007602727108108108 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0011894814321927938,
            "extra": "iterations: 214\ncpu: 0.0011857321775700944 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.002749693275678276,
            "extra": "iterations: 101\ncpu: 0.0027494074059405942 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0038556925455729167,
            "extra": "iterations: 75\ncpu: 0.003854749906666667 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "3ac242a3d87feef9b57e348f50131d9754ca43d1",
          "message": "Batched FFT, 15% expected gain for w7x (#503)",
          "timestamp": "2026-05-05T15:22:26+02:00",
          "tree_id": "b48a2de956f8741c9aecfcbf733013e1a4b4f1da",
          "url": "https://github.com/proximafusion/vmecpp/commit/3ac242a3d87feef9b57e348f50131d9754ca43d1"
        },
        "date": 1783604248048,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00023264945451462678,
            "extra": "iterations: 1261\ncpu: 0.00023263944091990481 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036750333493533954,
            "extra": "iterations: 763\ncpu: 0.00036748658060288346 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004794379781345426,
            "extra": "iterations: 591\ncpu: 0.00047941171404399327 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007246717310074996,
            "extra": "iterations: 387\ncpu: 0.0007245387519379849 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005601500757107708,
            "extra": "iterations: 539\ncpu: 0.0005601401651205934 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010261098980466961,
            "extra": "iterations: 273\ncpu: 0.0010260226483516492 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018407341681028668,
            "extra": "iterations: 152\ncpu: 0.0018403539736842115 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002743749057545382,
            "extra": "iterations: 102\ncpu: 0.002743701843137251 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8ed41a4ebc3aac15f0356bc75d4f8cebbac6ef0c",
          "message": "More realistic wait policy, additional fixed boundary benchmark (#515)",
          "timestamp": "2026-05-05T15:48:58+02:00",
          "tree_id": "217c3b116ed2219ed36cb2ea3d15e5d7f56c8ef8",
          "url": "https://github.com/proximafusion/vmecpp/commit/8ed41a4ebc3aac15f0356bc75d4f8cebbac6ef0c"
        },
        "date": 1783604248217,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022048389210420498,
            "extra": "iterations: 1275\ncpu: 0.00022047992862745105 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00037032444850203217,
            "extra": "iterations: 754\ncpu: 0.00037031757427055705 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00047612716467554584,
            "extra": "iterations: 589\ncpu: 0.00047600578268251276 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007130127918871143,
            "extra": "iterations: 395\ncpu: 0.0007130001113924049 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005190063405919958,
            "extra": "iterations: 540\ncpu: 0.000518997188888889 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001036994648675849,
            "extra": "iterations: 274\ncpu: 0.0010367887992700725 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018431293336968674,
            "extra": "iterations: 152\ncpu: 0.0018430960723684214 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027327373916027596,
            "extra": "iterations: 102\ncpu: 0.0027321197549019603 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "04158aa5f9eae86e399a928db56afef3413dabb1",
          "message": "Update README.md",
          "timestamp": "2026-05-12T21:26:02+02:00",
          "tree_id": "4901ac7979c957946219f2483da33f2eab28d01f",
          "url": "https://github.com/proximafusion/vmecpp/commit/04158aa5f9eae86e399a928db56afef3413dabb1"
        },
        "date": 1783604247955,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000222714905886306,
            "extra": "iterations: 1261\ncpu: 0.00022270351942902465 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003665191675323287,
            "extra": "iterations: 765\ncpu: 0.00036649803790849675 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004756664825698077,
            "extra": "iterations: 590\ncpu: 0.0004756382677966102 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007065561753285083,
            "extra": "iterations: 395\ncpu: 0.0007065137063291143 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005175961779492368,
            "extra": "iterations: 542\ncpu: 0.0005175652029520296 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010170216993852096,
            "extra": "iterations: 275\ncpu: 0.0010170039600000003 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018322561301437079,
            "extra": "iterations: 153\ncpu: 0.001832227647058824 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.00273667492912811,
            "extra": "iterations: 103\ncpu: 0.0027364934660194153 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9b2e1e066106a9100d31aad811677c8f23fb8456",
          "message": "Update copilot-setup-steps.yml",
          "timestamp": "2026-05-13T09:45:01+02:00",
          "tree_id": "213971aa8b812e90eea7f78beeed6a4d98f68681",
          "url": "https://github.com/proximafusion/vmecpp/commit/9b2e1e066106a9100d31aad811677c8f23fb8456"
        },
        "date": 1783604248243,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022036473270461494,
            "extra": "iterations: 1265\ncpu: 0.00022036088063241108 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003709749458662046,
            "extra": "iterations: 765\ncpu: 0.0003709689843137256 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00047750042488059154,
            "extra": "iterations: 587\ncpu: 0.0004774130783645656 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007110473157188668,
            "extra": "iterations: 393\ncpu: 0.0007110351755725191 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005226032756199347,
            "extra": "iterations: 535\ncpu: 0.0005225571102803735 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010263244895374075,
            "extra": "iterations: 272\ncpu: 0.0010260922022058824 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018340076496398527,
            "extra": "iterations: 153\ncpu: 0.0018339782810457513 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002747684422105846,
            "extra": "iterations: 101\ncpu: 0.0027471706633663343 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5f23d714753bb808bf5ca2675374e1385ae1468a",
          "message": "Remove FFTW from container installs, since we settled on FFTX now (#517)",
          "timestamp": "2026-05-13T10:01:17+02:00",
          "tree_id": "3aa86ce29736387d10a101cb18432185447d9955",
          "url": "https://github.com/proximafusion/vmecpp/commit/5f23d714753bb808bf5ca2675374e1385ae1468a"
        },
        "date": 1783604248126,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021987242961493065,
            "extra": "iterations: 1270\ncpu: 0.00021985454724409456 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036458589389492266,
            "extra": "iterations: 766\ncpu: 0.00036457950783289837 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00047454066195730437,
            "extra": "iterations: 590\ncpu: 0.00047433524237288136 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007111687345553172,
            "extra": "iterations: 394\ncpu: 0.0007110930532994926 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005218804772220441,
            "extra": "iterations: 536\ncpu: 0.0005218344925373137 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010202157236363766,
            "extra": "iterations: 274\ncpu: 0.0010200472554744515 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0019279781140779195,
            "extra": "iterations: 152\ncpu: 0.0019278502171052637 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027465105056762697,
            "extra": "iterations: 100\ncpu: 0.0027459105600000024 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "6a5bee6ee5669c5a77face38d1e8056bba120abd",
          "message": "Enable pre-commit for copilot agents (#520)",
          "timestamp": "2026-05-13T13:44:26+02:00",
          "tree_id": "265592f4b5820720cc5c7fbe0372796b1e38086b",
          "url": "https://github.com/proximafusion/vmecpp/commit/6a5bee6ee5669c5a77face38d1e8056bba120abd"
        },
        "date": 1783604248146,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022273667299064102,
            "extra": "iterations: 1256\ncpu: 0.00022268387022292993 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036720362009178285,
            "extra": "iterations: 764\ncpu: 0.0003671970130890053 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004799178201858312,
            "extra": "iterations: 584\ncpu: 0.0004799095136986303 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007108742677712742,
            "extra": "iterations: 395\ncpu: 0.0007107554936708864 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005205281397927851,
            "extra": "iterations: 537\ncpu: 0.0005205195270018627 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010168725794011896,
            "extra": "iterations: 275\ncpu: 0.0010168140436363628 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018389789681685599,
            "extra": "iterations: 152\ncpu: 0.0018388956250000007 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002759482600901387,
            "extra": "iterations: 101\ncpu: 0.0027594332079207947 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Asim Arshad",
            "email": "asim48@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "4667d5dfa147f908c9f8d6235977e5536d70a1d5",
          "message": "Fix force balance spectrum example shape",
          "timestamp": "2026-05-14T02:00:00+01:00",
          "tree_id": "aafb2c13b94ea6c1640ce5ac74433f563d72b95d",
          "url": "https://github.com/proximafusion/vmecpp/commit/4667d5dfa147f908c9f8d6235977e5536d70a1d5"
        },
        "date": 1783604248082,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022178635394888178,
            "extra": "iterations: 1249\ncpu: 0.0002217823923138511 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003652706295446125,
            "extra": "iterations: 767\ncpu: 0.0003652640052151239 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004731811381675102,
            "extra": "iterations: 592\ncpu: 0.00047311141891891905 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007130149657351112,
            "extra": "iterations: 394\ncpu: 0.0007129379949238575 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005221282368275657,
            "extra": "iterations: 536\ncpu: 0.0005220925093283583 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010243789616958562,
            "extra": "iterations: 273\ncpu: 0.0010243002747252743 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018680477142333986,
            "extra": "iterations: 150\ncpu: 0.0018679072933333323 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027300061531437257,
            "extra": "iterations: 103\ncpu: 0.0027295368349514555 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7e06d9300a2b0afadf440c9b557fe4bfab5c3c79",
          "message": "Update README.md (#516)",
          "timestamp": "2026-05-14T23:54:23+02:00",
          "tree_id": "0b15c8f66daa0128501442e828e3d74f0cb119b3",
          "url": "https://github.com/proximafusion/vmecpp/commit/7e06d9300a2b0afadf440c9b557fe4bfab5c3c79"
        },
        "date": 1783604248181,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000220413470831443,
            "extra": "iterations: 1270\ncpu: 0.0002203966905511811 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003649692260782132,
            "extra": "iterations: 764\ncpu: 0.0003649628874345551 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004751047845614159,
            "extra": "iterations: 590\ncpu: 0.00047509690677966106 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007142035328612036,
            "extra": "iterations: 392\ncpu: 0.0007141557091836737 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005190098219324795,
            "extra": "iterations: 539\ncpu: 0.0005190017551020411 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010258011958178353,
            "extra": "iterations: 272\ncpu: 0.0010257402132352945 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018351579967298005,
            "extra": "iterations: 152\ncpu: 0.001834747638157894 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027573816847093037,
            "extra": "iterations: 101\ncpu: 0.0027573359504950534 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "1721a87219664eea872c668e74189a3eea115d94",
          "message": "Add CI coverage for example scripts via dedicated pytest job (#523)",
          "timestamp": "2026-05-16T15:56:22Z",
          "tree_id": "3f864c1d77b0ca3397cff9c28da416dc5109f32c",
          "url": "https://github.com/proximafusion/vmecpp/commit/1721a87219664eea872c668e74189a3eea115d94"
        },
        "date": 1783604248005,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000221949924465969,
            "extra": "iterations: 1261\ncpu: 0.00022194585329103888 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036967907076567607,
            "extra": "iterations: 765\ncpu: 0.0003696729307189543 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00047435695951561843,
            "extra": "iterations: 591\ncpu: 0.0004743357411167514 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007104020433377494,
            "extra": "iterations: 394\ncpu: 0.0007103889111675131 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005278241611918937,
            "extra": "iterations: 531\ncpu: 0.0005277944124293786 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010162021802819295,
            "extra": "iterations: 276\ncpu: 0.0010161071376811593 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018545404860847874,
            "extra": "iterations: 152\ncpu: 0.0018545047039473677 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002746825124703202,
            "extra": "iterations: 102\ncpu: 0.0027466651666666684 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "bfc276c7f512c3abab65c6760cfaf3fb7c231357",
          "message": "README: fix missing word + 'suggest to give' phrasing",
          "timestamp": "2026-05-18T05:17:51-04:00",
          "tree_id": "8bf398f0174c87a944caff9c9b33117e0b9a715c",
          "url": "https://github.com/proximafusion/vmecpp/commit/bfc276c7f512c3abab65c6760cfaf3fb7c231357"
        },
        "date": 1783604248317,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002289596218732918,
            "extra": "iterations: 1272\ncpu: 0.00022893617452830192 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003689556045735136,
            "extra": "iterations: 752\ncpu: 0.00036889342021276586 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004757231888827727,
            "extra": "iterations: 589\ncpu: 0.0004757154329371818 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007117498315320973,
            "extra": "iterations: 393\ncpu: 0.0007117020788804077 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005185368345744575,
            "extra": "iterations: 536\ncpu: 0.0005185285074626867 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010191362554376777,
            "extra": "iterations: 275\ncpu: 0.001019077650909091 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018530397225689416,
            "extra": "iterations: 151\ncpu: 0.001852948516556292 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002792632579803467,
            "extra": "iterations: 100\ncpu: 0.0027925815899999982 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8d14954abd147b871352ae1522eb3973c5ba3081",
          "message": "bazel: make vmecpp usable as a Bazel module (#532)",
          "timestamp": "2026-05-28T17:12:08-04:00",
          "tree_id": "87a632621a184995f103204598cb64adbcc5d063",
          "url": "https://github.com/proximafusion/vmecpp/commit/8d14954abd147b871352ae1522eb3973c5ba3081"
        },
        "date": 1783604248211,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022252179782858344,
            "extra": "iterations: 1259\ncpu: 0.00022250983876092142 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003683379069264396,
            "extra": "iterations: 761\ncpu: 0.0003682564388961894 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004912589725695159,
            "extra": "iterations: 570\ncpu: 0.0004912298789473684 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007181723912556967,
            "extra": "iterations: 390\ncpu: 0.0007181606461538461 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005190866964834708,
            "extra": "iterations: 540\ncpu: 0.0005189926092592595 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010136555934297867,
            "extra": "iterations: 276\ncpu: 0.0010136387318840578 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018580037783953102,
            "extra": "iterations: 153\ncpu: 0.0018578521568627434 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027366133019475656,
            "extra": "iterations: 101\ncpu: 0.0027362606039603972 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ab117cb23c72e537b233ebb59597598a600f5a3a",
          "message": "output_quantities: zero-initialize jPS2 axis/edge entries (#529)",
          "timestamp": "2026-05-28T17:16:42-04:00",
          "tree_id": "d05dd4fd3583aae9c378126982384bdfac009e1e",
          "url": "https://github.com/proximafusion/vmecpp/commit/ab117cb23c72e537b233ebb59597598a600f5a3a"
        },
        "date": 1783604248282,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021971010026477635,
            "extra": "iterations: 1260\ncpu: 0.00021969263571428572 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003679068110462244,
            "extra": "iterations: 761\ncpu: 0.0003678736780551905 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00047473786240917145,
            "extra": "iterations: 590\ncpu: 0.00047472986949152545 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007398971467030709,
            "extra": "iterations: 379\ncpu: 0.0007398544036939318 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005188518595472674,
            "extra": "iterations: 535\ncpu: 0.0005188427869158877 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010008352143423898,
            "extra": "iterations: 280\ncpu: 0.001000782010714286 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018376182107364431,
            "extra": "iterations: 153\ncpu: 0.0018372290522875845 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002733945846557617,
            "extra": "iterations: 102\ncpu: 0.0027338974901960794 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8596e70339522c3c4959a1af1e278c758d6c04bb",
          "message": "tests: cross-check vmecpp/test_data and vmecpp_large_cpp_tests inputs via indata2json (#534)",
          "timestamp": "2026-05-28T17:19:33-04:00",
          "tree_id": "a1625da2614ebf437f5631901fb16e3abf41914f",
          "url": "https://github.com/proximafusion/vmecpp/commit/8596e70339522c3c4959a1af1e278c758d6c04bb"
        },
        "date": 1783604248197,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022121084577785796,
            "extra": "iterations: 1261\ncpu: 0.00022119169785884225 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003690628748529061,
            "extra": "iterations: 761\ncpu: 0.00036901098685939555 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004875234917995278,
            "extra": "iterations: 589\ncpu: 0.00048748788794567065 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007147053066565067,
            "extra": "iterations: 392\ncpu: 0.0007146514821428572 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005178839246803356,
            "extra": "iterations: 535\ncpu: 0.0005178745271028036 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0009941361474652664,
            "extra": "iterations: 282\ncpu: 0.0009940860000000001 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018472577396192048,
            "extra": "iterations: 152\ncpu: 0.0018467456842105252 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002715444101870639,
            "extra": "iterations: 103\ncpu: 0.0027152783009708765 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ca4fc48fa3f627ded8128565b54c00f068a0b80b",
          "message": "Populate full-grid bsubsmns_full in wout (#530)",
          "timestamp": "2026-05-28T17:40:09-04:00",
          "tree_id": "b0eb42db2d5fdc07733bcbb41e64fd47d67a2129",
          "url": "https://github.com/proximafusion/vmecpp/commit/ca4fc48fa3f627ded8128565b54c00f068a0b80b"
        },
        "date": 1783604248356,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002213734468706632,
            "extra": "iterations: 1261\ncpu: 0.0002213581697065821 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003701640307746022,
            "extra": "iterations: 758\ncpu: 0.000370157622691293 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004738635823206248,
            "extra": "iterations: 591\ncpu: 0.0004738378595600676 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007115283995184279,
            "extra": "iterations: 393\ncpu: 0.0007114920381679388 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005193245944692128,
            "extra": "iterations: 536\ncpu: 0.0005193158507462686 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010166603283290445,
            "extra": "iterations: 274\ncpu: 0.0010165958175182477 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001882998272776604,
            "extra": "iterations: 128\ncpu: 0.001882898984374999 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0028009799995807687,
            "extra": "iterations: 99\ncpu: 0.0028007759393939378 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a14f9448e88d417947535cb4ab0453a2fc620b0b",
          "message": "radial_profiles: implement spline and analytic profile evaluators (#531)",
          "timestamp": "2026-05-29T06:53:35-04:00",
          "tree_id": "551fabc736ae3a92dda5670a0e3d44ba063c0663",
          "url": "https://github.com/proximafusion/vmecpp/commit/a14f9448e88d417947535cb4ab0453a2fc620b0b"
        },
        "date": 1783604248261,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021909224608587842,
            "extra": "iterations: 1008\ncpu: 0.00021909005158730161 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036955565408943527,
            "extra": "iterations: 765\ncpu: 0.00036953772418300666 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00044272314798698856,
            "extra": "iterations: 627\ncpu: 0.00044269975917065387 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007003733986302427,
            "extra": "iterations: 399\ncpu: 0.0007003614887218047 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005211581508984557,
            "extra": "iterations: 537\ncpu: 0.0005211332867783985 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010046838856429506,
            "extra": "iterations: 278\ncpu: 0.0010045932194244608 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018423356507953847,
            "extra": "iterations: 152\ncpu: 0.0018423054210526318 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027086734771728516,
            "extra": "iterations: 103\ncpu: 0.0027082969805825204 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "4d8a378fdd27c735c568b187e2bcf5107cb5ea78",
          "message": "Complete the Eigen3 port of the free-boundary code (#543)",
          "timestamp": "2026-05-31T07:00:34-04:00",
          "tree_id": "2a4b84595103ac514d38fbc52f6506f9416de2e7",
          "url": "https://github.com/proximafusion/vmecpp/commit/4d8a378fdd27c735c568b187e2bcf5107cb5ea78"
        },
        "date": 1783604248092,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002195082711491183,
            "extra": "iterations: 1258\ncpu: 0.00021949355087440384 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003670256388814826,
            "extra": "iterations: 760\ncpu: 0.0003669728513157896 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00044379180993515846,
            "extra": "iterations: 626\ncpu: 0.0004437708865814697 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007377174095616807,
            "extra": "iterations: 379\ncpu: 0.0007377050686015832 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005190420944016435,
            "extra": "iterations: 541\ncpu: 0.0005189648539741217 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010184574127197266,
            "extra": "iterations: 275\ncpu: 0.0010183423490909085 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001846923639899806,
            "extra": "iterations: 152\ncpu: 0.0018466550328947377 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027151316114999717,
            "extra": "iterations: 103\ncpu: 0.0027149941844660207 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "10027e5d5ec4f31fcd863663cf6c8d4a41b45b5d",
          "message": "Cache TSAN Docker image in GHCR (#553)",
          "timestamp": "2026-05-31T14:42:53+02:00",
          "tree_id": "b8cc6a49e078feacb3fcf762afc3455a317e1624",
          "url": "https://github.com/proximafusion/vmecpp/commit/10027e5d5ec4f31fcd863663cf6c8d4a41b45b5d"
        },
        "date": 1783604247982,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002177635704671351,
            "extra": "iterations: 1282\ncpu: 0.00021775572308892362 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00037025678062944895,
            "extra": "iterations: 754\ncpu: 0.000370192401856764 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004430291028428905,
            "extra": "iterations: 634\ncpu: 0.00044298099526813905 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007335128584457319,
            "extra": "iterations: 382\ncpu: 0.0007335004999999996 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005231736976409627,
            "extra": "iterations: 535\ncpu: 0.0005230928093457942 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010491757497300197,
            "extra": "iterations: 274\ncpu: 0.0010490575839416053 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001850633431744102,
            "extra": "iterations: 151\ncpu: 0.001850546403973511 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002741652376511518,
            "extra": "iterations: 102\ncpu: 0.002741054205882354 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c945b86dc692a2c60b1d3832095c2d68f719e25f",
          "message": "free_boundary: multi-grid test case and opt-in to persist the Nestor solver across grid steps (#535)",
          "timestamp": "2026-05-31T09:54:14-04:00",
          "tree_id": "d1ce5591064c5a20b3c3b95506696129531a660c",
          "url": "https://github.com/proximafusion/vmecpp/commit/c945b86dc692a2c60b1d3832095c2d68f719e25f"
        },
        "date": 1783604248348,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021776067747377765,
            "extra": "iterations: 1261\ncpu: 0.00021775057494052342 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036838740776152243,
            "extra": "iterations: 761\ncpu: 0.0003683815256241787 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004417381076030551,
            "extra": "iterations: 634\ncpu: 0.0004417307807570979 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007319774079883067,
            "extra": "iterations: 383\ncpu: 0.0007319284360313314 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005199679195991474,
            "extra": "iterations: 539\ncpu: 0.0005199587235621522 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010112399228643426,
            "extra": "iterations: 277\ncpu: 0.0010112240830324904 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018514269276669153,
            "extra": "iterations: 152\ncpu: 0.001851317868421053 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002723369783568151,
            "extra": "iterations: 103\ncpu: 0.0027232232912621375 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0138a54cf5c8b36d5b9229f68685f8cc331a2c4d",
          "message": "free_boundary: implement non-stellarator-symmetric surface geometry (#533)",
          "timestamp": "2026-06-01T18:59:48-04:00",
          "tree_id": "d39ce9ccbe6b761888e0a2bafef155883520004b",
          "url": "https://github.com/proximafusion/vmecpp/commit/0138a54cf5c8b36d5b9229f68685f8cc331a2c4d"
        },
        "date": 1783604247945,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021847897011807252,
            "extra": "iterations: 1269\ncpu: 0.000218475463356974 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003699235001312136,
            "extra": "iterations: 761\ncpu: 0.0003698668948751645 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004439657177947853,
            "extra": "iterations: 629\ncpu: 0.00044395789507154225 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007151833602360317,
            "extra": "iterations: 392\ncpu: 0.0007150880994897962 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005192805219579626,
            "extra": "iterations: 540\ncpu: 0.0005192038870370371 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001037295659383138,
            "extra": "iterations: 270\ncpu: 0.001037243859259259 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001855854956519525,
            "extra": "iterations: 151\ncpu: 0.0018555619933774848 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027086551372821513,
            "extra": "iterations: 104\ncpu: 0.002708402721153845 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8dca9e9711204f7a75df4bad46a1e847447d8c59",
          "message": "Streamlined AGENTS.md (#539)",
          "timestamp": "2026-06-02T01:10:12+02:00",
          "tree_id": "6991bf52fdd0f0ecde43b759bf11bd23f9a74f76",
          "url": "https://github.com/proximafusion/vmecpp/commit/8dca9e9711204f7a75df4bad46a1e847447d8c59"
        },
        "date": 1783604248214,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00021848433294615894,
            "extra": "iterations: 1283\ncpu: 0.0002183688051441933 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036545123084116794,
            "extra": "iterations: 767\ncpu: 0.0003653817288135594 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00044228338668775107,
            "extra": "iterations: 634\ncpu: 0.0004422761782334387 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007145669057927745,
            "extra": "iterations: 397\ncpu: 0.0007142990806045341 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005282571441248844,
            "extra": "iterations: 532\ncpu: 0.0005282482030075187 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010189627208848939,
            "extra": "iterations: 274\ncpu: 0.0010189445583941613 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018711996078491211,
            "extra": "iterations: 150\ncpu: 0.001871086886666668 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.00278963565826416,
            "extra": "iterations: 100\ncpu: 0.0027895846400000004 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a8268b821e9e2570f33ec47323823c90e873313e",
          "message": "free_boundary: implement axisymmetric (nzeta=1) free-boundary support (#536)",
          "timestamp": "2026-06-02T02:18:05-04:00",
          "tree_id": "3ba41eaa4b12a01ed15c5589262cdda109e0830b",
          "url": "https://github.com/proximafusion/vmecpp/commit/a8268b821e9e2570f33ec47323823c90e873313e"
        },
        "date": 1783604248279,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000222164448176589,
            "extra": "iterations: 1070\ncpu: 0.0002221607728971963 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00037262035575395504,
            "extra": "iterations: 761\ncpu: 0.0003725974244415244 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004746174408217608,
            "extra": "iterations: 590\ncpu: 0.0004745520644067798 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007152912855761217,
            "extra": "iterations: 389\ncpu: 0.0007152786735218512 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005182381029482241,
            "extra": "iterations: 540\ncpu: 0.0005182301999999999 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001053642509575177,
            "extra": "iterations: 266\ncpu: 0.0010535348270676698 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001845094541840206,
            "extra": "iterations: 151\ncpu: 0.0018449127549668872 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027078702611830633,
            "extra": "iterations: 103\ncpu: 0.002707391660194173 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5f3407175584ab13b2f0c1b5a5623f207e5ea654",
          "message": "Expose the forward model and drive the equilibrium iteration from Python (#546)",
          "timestamp": "2026-06-02T02:49:24-04:00",
          "tree_id": "b911431cff5a48c2af7038c5e16230a4306b0977",
          "url": "https://github.com/proximafusion/vmecpp/commit/5f3407175584ab13b2f0c1b5a5623f207e5ea654"
        },
        "date": 1783604248128,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002210402413259579,
            "extra": "iterations: 1264\ncpu: 0.00022103674446202537 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036518120517333347,
            "extra": "iterations: 768\ncpu: 0.0003651603854166666 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004737988290256031,
            "extra": "iterations: 593\ncpu: 0.0004737612192242833 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007088987133170984,
            "extra": "iterations: 395\ncpu: 0.0007088866962025322 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005209609617328821,
            "extra": "iterations: 539\ncpu: 0.0005209306846011135 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001033926362041178,
            "extra": "iterations: 271\ncpu: 0.0010339100738007377 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018417472081468595,
            "extra": "iterations: 151\ncpu: 0.0018417171456953645 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0026964682799119214,
            "extra": "iterations: 104\ncpu: 0.0026963319230769232 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "3ca4231d1dfaa8696acf81af115e5206507d1d4d",
          "message": "Expose the full threed1 output quantities in the Python interface (#542)",
          "timestamp": "2026-06-02T03:23:15-04:00",
          "tree_id": "f9315d08af9a48356fc6e9d9f49a1c40b4156a26",
          "url": "https://github.com/proximafusion/vmecpp/commit/3ca4231d1dfaa8696acf81af115e5206507d1d4d"
        },
        "date": 1783604248050,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022592456535963914,
            "extra": "iterations: 1246\ncpu: 0.00022592035313001607 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003624190347755294,
            "extra": "iterations: 774\ncpu: 0.00036237513953488377 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004755862716103898,
            "extra": "iterations: 588\ncpu: 0.00047557771088435375 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007121678517789257,
            "extra": "iterations: 392\ncpu: 0.0007121574005102039 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005187775565789092,
            "extra": "iterations: 538\ncpu: 0.0005187498791821564 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010420637946146573,
            "extra": "iterations: 269\ncpu: 0.0010420457992565062 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018645038730219791,
            "extra": "iterations: 152\ncpu: 0.0018643248026315802 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002707745020206158,
            "extra": "iterations: 104\ncpu: 0.0027076989711538432 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "55c569444c82fc93b55b2c752c7f793d841625fa",
          "message": "Python iteration debug script (#556)",
          "timestamp": "2026-06-03T08:58:09+02:00",
          "tree_id": "10550af4bf37180818ab1e0ca3487939aacef66c",
          "url": "https://github.com/proximafusion/vmecpp/commit/55c569444c82fc93b55b2c752c7f793d841625fa"
        },
        "date": 1783604248115,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000219818550770677,
            "extra": "iterations: 1270\ncpu: 0.00021980618425196852 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036109876386898084,
            "extra": "iterations: 776\ncpu: 0.0003610422925257732 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004740346305059701,
            "extra": "iterations: 591\ncpu: 0.0004740272267343486 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006998144183075637,
            "extra": "iterations: 401\ncpu: 0.0006997571870324187 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005192995513746159,
            "extra": "iterations: 539\ncpu: 0.0005192664341372915 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001039859100624367,
            "extra": "iterations: 270\ncpu: 0.0010398417111111104 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001852340256141511,
            "extra": "iterations: 151\ncpu: 0.0018520362913907287 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027130205654403543,
            "extra": "iterations: 103\ncpu: 0.0027129707961165072 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "0ff3e7c9fb29385e6a91753ee320c983cff5e6e0",
          "message": "Update fourier_basis_implementation.md (#559)",
          "timestamp": "2026-06-05T10:02:40+02:00",
          "tree_id": "4be8128819c0b9b38cd83ea5c1989c69e26f42ec",
          "url": "https://github.com/proximafusion/vmecpp/commit/0ff3e7c9fb29385e6a91753ee320c983cff5e6e0"
        },
        "date": 1783604247979,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002210378076015999,
            "extra": "iterations: 1253\ncpu: 0.00022103401675977656 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003633855688246391,
            "extra": "iterations: 769\ncpu: 0.0003633793198959689 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00047547662624320076,
            "extra": "iterations: 586\ncpu: 0.0004753598037542663 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007193377337504908,
            "extra": "iterations: 388\ncpu: 0.0007193257268041241 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000515153487230139,
            "extra": "iterations: 542\ncpu: 0.0005151448985239853 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010322371972003106,
            "extra": "iterations: 271\ncpu: 0.0010320633542435427 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.00184275901395511,
            "extra": "iterations: 153\ncpu: 0.0018427256405228742 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027074397188945887,
            "extra": "iterations: 103\ncpu: 0.002707210300970871 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a248cd9869c53281ca99b2485c75de3dd5851153",
          "message": "build+ci: abseil commit pin for Clang>=21, VMEC2000-from-source, benchmark fork guard (#564)",
          "timestamp": "2026-06-15T08:57:27+02:00",
          "tree_id": "13ea6b4473301c940018476545b224d738c209bd",
          "url": "https://github.com/proximafusion/vmecpp/commit/a248cd9869c53281ca99b2485c75de3dd5851153"
        },
        "date": 1783604248264,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000244029921312887,
            "extra": "iterations: 911\ncpu: 0.00024401761031833156 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036257243587681183,
            "extra": "iterations: 774\ncpu: 0.00036255204392764855 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00047604997279280327,
            "extra": "iterations: 590\ncpu: 0.00047599174237288137 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007232471058766047,
            "extra": "iterations: 384\ncpu: 0.0007232048776041664 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005232082472907173,
            "extra": "iterations: 540\ncpu: 0.000522467716666667 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010463863838719959,
            "extra": "iterations: 262\ncpu: 0.0010462068549618318 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018726755308624884,
            "extra": "iterations: 149\ncpu: 0.0018724796912751673 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002731657945192777,
            "extra": "iterations: 104\ncpu: 0.002716402125 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c3ec179b9fa09fc043444ed60a46fafa726d620b",
          "message": "Handle jaxtyping anonymous-dimension API drift in wout serialization (#562)",
          "timestamp": "2026-06-15T17:12:05+02:00",
          "tree_id": "5c962a9aab90d2912a1034d8f8fc1c75980d8db6",
          "url": "https://github.com/proximafusion/vmecpp/commit/c3ec179b9fa09fc043444ed60a46fafa726d620b"
        },
        "date": 1783604248322,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022380032877283775,
            "extra": "iterations: 1270\ncpu: 0.00022377011181102364 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00036560399725293994,
            "extra": "iterations: 766\ncpu: 0.00036559805483028714 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00047666888658692245,
            "extra": "iterations: 588\ncpu: 0.00047664359693877566 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007127844137588734,
            "extra": "iterations: 394\ncpu: 0.000712772218274112 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005195686937044453,
            "extra": "iterations: 537\ncpu: 0.0005195606554934828 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010119439892820503,
            "extra": "iterations: 277\ncpu: 0.0010118836859205778 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018470303112307924,
            "extra": "iterations: 151\ncpu: 0.001846998430463577 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002713666378873066,
            "extra": "iterations: 103\ncpu: 0.002713616320388351 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Copilot",
            "email": "198982749+Copilot@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b0876027f1c16d0910a80564118dae42161d0cad",
          "message": "Update abseil-cpp Bazel dependency to 20260107.1 LTS (#586)",
          "timestamp": "2026-06-15T16:31:54Z",
          "tree_id": "3b32e93144a6947920897c3c354a59cf41e9dcc5",
          "url": "https://github.com/proximafusion/vmecpp/commit/b0876027f1c16d0910a80564118dae42161d0cad"
        },
        "date": 1783604248290,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002174692532039424,
            "extra": "iterations: 1135\ncpu: 0.00021746549162995595 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003665882771409403,
            "extra": "iterations: 762\ncpu: 0.0003665676207349082 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004417993584457709,
            "extra": "iterations: 637\ncpu: 0.0004417091051805338 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0007021403252929076,
            "extra": "iterations: 399\ncpu: 0.0007021290000000004 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005183908674452041,
            "extra": "iterations: 540\ncpu: 0.0005183407555555558 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001016808423128995,
            "extra": "iterations: 275\ncpu: 0.0010166029272727268 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018354826851894983,
            "extra": "iterations: 152\ncpu: 0.001835391486842107 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002717254231277022,
            "extra": "iterations: 103\ncpu: 0.0027167130485436894 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "559e230b6dde38c018c3fb62994750e2bbef007c",
          "message": "ideal_mhd_model: make computeMHDForces allocation-free (#566)",
          "timestamp": "2026-06-17T09:52:05+02:00",
          "tree_id": "33bc7ad8a8a303b6a6c44a48d490975675a44ddd",
          "url": "https://github.com/proximafusion/vmecpp/commit/559e230b6dde38c018c3fb62994750e2bbef007c"
        },
        "date": 1783604248112,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00022992313903954262,
            "extra": "iterations: 1257\ncpu: 0.0002298775377883851 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003255202795154957,
            "extra": "iterations: 861\ncpu: 0.00032549366898954713 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.00044081857200658836,
            "extra": "iterations: 631\ncpu: 0.0004408115007923933 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006240545243091202,
            "extra": "iterations: 449\ncpu: 0.0006239247661469937 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000523176532559627,
            "extra": "iterations: 534\ncpu: 0.0005231683670411987 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010353485743204752,
            "extra": "iterations: 270\ncpu: 0.001035330614814814 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018396659901267606,
            "extra": "iterations: 152\ncpu: 0.0018394987565789494 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002749284108479818,
            "extra": "iterations: 102\ncpu: 0.002749239176470594 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "73f541fb66f34267d3dffcd45c8682b01aa9813c",
          "message": "Make the iteration hot loop allocation-free (#595)",
          "timestamp": "2026-06-17T17:42:06-04:00",
          "tree_id": "cda00ec27b0a41c3214f6c10e4a9bf8d0f0e73d2",
          "url": "https://github.com/proximafusion/vmecpp/commit/73f541fb66f34267d3dffcd45c8682b01aa9813c"
        },
        "date": 1783604248164,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026939152541318774,
            "extra": "iterations: 1055\ncpu: 0.0002693780104265403 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00032500375325544405,
            "extra": "iterations: 863\ncpu: 0.000324971527230591 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004976939138680114,
            "extra": "iterations: 563\ncpu: 0.0004976860781527533 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006129715640472012,
            "extra": "iterations: 458\ncpu: 0.0006129382423580786 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005185220686653406,
            "extra": "iterations: 541\ncpu: 0.0005185139279112754 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001004775365193685,
            "extra": "iterations: 279\ncpu: 0.001004716946236558 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018356834587297944,
            "extra": "iterations: 152\ncpu: 0.0018355808355263181 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027182403120022377,
            "extra": "iterations: 103\ncpu: 0.002718198067961166 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7d186f2c1ac2f889f9778d292a58f4158b10b0e3",
          "message": "iteration: fix post-reguess lambda scaling divergence, add Python multigrid driver (#560)",
          "timestamp": "2026-06-23T19:23:01-04:00",
          "tree_id": "93262fcd05de06ef9ce49df381d121b66eccd959",
          "url": "https://github.com/proximafusion/vmecpp/commit/7d186f2c1ac2f889f9778d292a58f4158b10b0e3"
        },
        "date": 1783604248178,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002692670465629792,
            "extra": "iterations: 856\ncpu: 0.0002691643341121496 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003234262147439255,
            "extra": "iterations: 867\ncpu: 0.0003234210276816609 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004952320361727118,
            "extra": "iterations: 566\ncpu: 0.0004952052367491165 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006115462639138787,
            "extra": "iterations: 457\ncpu: 0.0006115082625820568 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005171829782744286,
            "extra": "iterations: 539\ncpu: 0.0005171744619666047 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010192524303089489,
            "extra": "iterations: 275\ncpu: 0.0010191894909090905 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018459699655833997,
            "extra": "iterations: 152\ncpu: 0.00184582057236842 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027214662701475854,
            "extra": "iterations: 102\ncpu: 0.0027212114117647042 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "2e7c3d039e5571c3e8f3a2f17207cfa8f5903e84",
          "message": "enzyme: opt-in Clang/Enzyme build option + AD smoke test (#565)",
          "timestamp": "2026-06-24T08:42:22+02:00",
          "tree_id": "813c275e8dbd0aa8c2f0bc854d6e614292584fab",
          "url": "https://github.com/proximafusion/vmecpp/commit/2e7c3d039e5571c3e8f3a2f17207cfa8f5903e84"
        },
        "date": 1783604248026,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002708332684203869,
            "extra": "iterations: 1048\ncpu: 0.0002708289971374046 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00032265807077082624,
            "extra": "iterations: 868\ncpu: 0.0003226014711981567 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004974791433958882,
            "extra": "iterations: 565\ncpu: 0.0004974007238938053 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006102038466412089,
            "extra": "iterations: 460\ncpu: 0.0006101934804347828 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005239127727037065,
            "extra": "iterations: 534\ncpu: 0.0005238397172284642 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001010473884830406,
            "extra": "iterations: 277\ncpu: 0.0010103636028880873 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001837159457959627,
            "extra": "iterations: 152\ncpu: 0.0018371262302631583 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002707377220820455,
            "extra": "iterations: 103\ncpu: 0.0027069366990291254 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "776d247ba8021d6be42573c4660bd973f346c13a",
          "message": "Migrate large cpp tests to the main repository (#596)",
          "timestamp": "2026-06-25T16:27:28+02:00",
          "tree_id": "cdbb8a7eda60cd8ae4970b3f1f88bcdea908d11e",
          "url": "https://github.com/proximafusion/vmecpp/commit/776d247ba8021d6be42573c4660bd973f346c13a"
        },
        "date": 1783604248166,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026503814892335373,
            "extra": "iterations: 1056\ncpu: 0.00026498485511363643 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003253741848000753,
            "extra": "iterations: 866\ncpu: 0.00032536880369515023 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004933504624800249,
            "extra": "iterations: 550\ncpu: 0.00049332532 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006228626838752202,
            "extra": "iterations: 448\ncpu: 0.0006228514531250001 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005198456622936108,
            "extra": "iterations: 540\ncpu: 0.0005198372314814818 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001013607755034409,
            "extra": "iterations: 277\ncpu: 0.0010133715992779778 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018377414602317556,
            "extra": "iterations: 151\ncpu: 0.0018376386490066216 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002730529285171657,
            "extra": "iterations: 103\ncpu: 0.0027302386504854366 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "b25cfdde81133d5420f7e87744f0b392952d5861",
          "message": "Make the lambda Fourier resolution independent of the geometry (#598)",
          "timestamp": "2026-06-26T16:01:27-04:00",
          "tree_id": "797b573c496ca778c519d25d96e62b0c37626d56",
          "url": "https://github.com/proximafusion/vmecpp/commit/b25cfdde81133d5420f7e87744f0b392952d5861"
        },
        "date": 1783604248301,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026659420558384486,
            "extra": "iterations: 1050\ncpu: 0.0002664117552380953 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003575431111523563,
            "extra": "iterations: 782\ncpu: 0.00035751926598465476 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000520149276959874,
            "extra": "iterations: 539\ncpu: 0.000520140755102041 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006141813891721762,
            "extra": "iterations: 457\ncpu: 0.0006137591094091904 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000520347885386621,
            "extra": "iterations: 539\ncpu: 0.0005203391558441558 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010127435952747771,
            "extra": "iterations: 277\ncpu: 0.0010127258050541513 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018437410655774567,
            "extra": "iterations: 152\ncpu: 0.0018437131710526324 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027130367686447592,
            "extra": "iterations: 103\ncpu: 0.002712991708737864 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "484b8fe6e4637e0102bb201926eccd6ebe7dd2e4",
          "message": "Update bazel lock (#604)",
          "timestamp": "2026-07-06T11:26:00+02:00",
          "tree_id": "ece6e2bcf41bbd1e91c7a4448987c5cf5ac2049c",
          "url": "https://github.com/proximafusion/vmecpp/commit/484b8fe6e4637e0102bb201926eccd6ebe7dd2e4"
        },
        "date": 1783604248084,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026639779408772784,
            "extra": "iterations: 1050\ncpu: 0.0002663933933333333 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003617412036227197,
            "extra": "iterations: 776\ncpu: 0.00036170105412371145 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005252004989398802,
            "extra": "iterations: 542\ncpu: 0.0005251459335793359 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006456312380338971,
            "extra": "iterations: 456\ncpu: 0.0006456207149122807 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005276131180097473,
            "extra": "iterations: 530\ncpu: 0.0005275355792452827 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0009959783521078036,
            "extra": "iterations: 289\ncpu: 0.000995964083044982 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001852303537829169,
            "extra": "iterations: 145\ncpu: 0.001852197593103447 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027229878508928914,
            "extra": "iterations: 103\ncpu: 0.0027229414174757285 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d0e474d4230f3d421a34715808e10e4f356d9f5e",
          "message": "Benchmark CI action remove code duplication (#605)",
          "timestamp": "2026-07-08T00:34:04+02:00",
          "tree_id": "7a02f9ebab9973f3b10d0358b82819547ff89b91",
          "url": "https://github.com/proximafusion/vmecpp/commit/d0e474d4230f3d421a34715808e10e4f356d9f5e"
        },
        "date": 1783604248374,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026990317930125824,
            "extra": "iterations: 1036\ncpu: 0.00026984188803088813 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003562107549012088,
            "extra": "iterations: 783\ncpu: 0.00035617551213282256 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.000518635908762614,
            "extra": "iterations: 540\ncpu: 0.0005186281796296297 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006446365936560565,
            "extra": "iterations: 434\ncpu: 0.0006445175000000002 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.000519242490156241,
            "extra": "iterations: 539\ncpu: 0.0005192347922077925 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010156293348832564,
            "extra": "iterations: 275\ncpu: 0.0010156122945454546 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001846809136240106,
            "extra": "iterations: 152\ncpu: 0.0018466205723684224 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027124858596949907,
            "extra": "iterations: 103\ncpu: 0.002712286378640778 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "93456948ee947ff235d9bc22fc8ad074ec7a8978",
          "message": "Continue to higher multigrid resolutions from hot restart (#603)",
          "timestamp": "2026-07-08T00:34:44+02:00",
          "tree_id": "c25024a7689b4e887598aa33971acd52d1d28f13",
          "url": "https://github.com/proximafusion/vmecpp/commit/93456948ee947ff235d9bc22fc8ad074ec7a8978"
        },
        "date": 1783604248229,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026907390001410485,
            "extra": "iterations: 1042\ncpu: 0.00026906908637236085 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003569824680401262,
            "extra": "iterations: 785\ncpu: 0.0003569438318471338 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005177241849723338,
            "extra": "iterations: 542\ncpu: 0.0005177149833948339 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006139101032616013,
            "extra": "iterations: 457\ncpu: 0.000613814978118162 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005207452472139782,
            "extra": "iterations: 537\ncpu: 0.0005207037467411544 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001003053453233507,
            "extra": "iterations: 279\ncpu: 0.0010030128422939067 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018508418610221464,
            "extra": "iterations: 152\ncpu: 0.0018507480131578944 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027315477723056833,
            "extra": "iterations: 103\ncpu: 0.002731374796116505 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "6d4a597cb2654a8f8ca5689cb2a238c567624d3c",
          "message": "Python-side resolution continuation (#544)",
          "timestamp": "2026-07-07T19:01:18-04:00",
          "tree_id": "aaccd48ea2a365ca0f8e62d108f9b107ddcfca5c",
          "url": "https://github.com/proximafusion/vmecpp/commit/6d4a597cb2654a8f8ca5689cb2a238c567624d3c"
        },
        "date": 1783604248155,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000267468913320149,
            "extra": "iterations: 1045\ncpu: 0.00026740958564593304 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00035771119351289714,
            "extra": "iterations: 784\ncpu: 0.0003577057946428572 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005191858712728922,
            "extra": "iterations: 539\ncpu: 0.0005191782152133581 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006173927878493255,
            "extra": "iterations: 454\ncpu: 0.0006172517048458146 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005157121905574092,
            "extra": "iterations: 540\ncpu: 0.0005156624833333334 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001024891605307331,
            "extra": "iterations: 273\ncpu: 0.0010248746483516478 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018564828236897787,
            "extra": "iterations: 150\ncpu: 0.001856129426666667 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002735224424623976,
            "extra": "iterations: 102\ncpu: 0.0027350744411764705 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "f5b7ac74dc86f226bfb74060f7865784d29cbf78",
          "message": "ideal_mhd_model: share the Jacobian kernel with exact autodiff (fwd vs rev) (#567)",
          "timestamp": "2026-07-08T01:21:52+02:00",
          "tree_id": "1cb265e12f30abd17a9e48f5b70d530ab62ab9bf",
          "url": "https://github.com/proximafusion/vmecpp/commit/f5b7ac74dc86f226bfb74060f7865784d29cbf78"
        },
        "date": 1783604248433,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002624371762782587,
            "extra": "iterations: 1063\ncpu: 0.00026242360301034805 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003395190383448745,
            "extra": "iterations: 825\ncpu: 0.0003395152945454546 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005044434521649336,
            "extra": "iterations: 555\ncpu: 0.0005044350306306305 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006060862901658786,
            "extra": "iterations: 463\ncpu: 0.0006060515161987041 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005169992517281282,
            "extra": "iterations: 542\ncpu: 0.0005169717047970481 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010120180109347678,
            "extra": "iterations: 277\ncpu: 0.0010120009025270762 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001819564150525378,
            "extra": "iterations: 154\ncpu: 0.00181953187012987 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027490316652784164,
            "extra": "iterations: 102\ncpu: 0.0027486724509803925 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "07ba4fdd96acd6e5b44d62e9c8ccdcbeef493273",
          "message": "ideal_mhd_model: share the metric kernel (gsqrt, guu, guv, gvv) (#568)",
          "timestamp": "2026-07-08T02:45:04+02:00",
          "tree_id": "462646ea4dbacc91f75b7d99278e90b96dee1a0e",
          "url": "https://github.com/proximafusion/vmecpp/commit/07ba4fdd96acd6e5b44d62e9c8ccdcbeef493273"
        },
        "date": 1783604247965,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026814112645138355,
            "extra": "iterations: 1052\ncpu: 0.00026809142965779467 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00032406286248620953,
            "extra": "iterations: 848\ncpu: 0.00032404523584905667 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004954539909564629,
            "extra": "iterations: 567\ncpu: 0.0004954459629629631 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006159721759327672,
            "extra": "iterations: 456\ncpu: 0.000615847403508772 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005184152272826466,
            "extra": "iterations: 537\ncpu: 0.0005183495288640596 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0009811424708866573,
            "extra": "iterations: 286\ncpu: 0.0009811279125874128 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001820824363014915,
            "extra": "iterations: 154\ncpu: 0.0018207933506493505 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027847336787803502,
            "extra": "iterations: 102\ncpu: 0.0027846866568627445 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "10938d76de9e7d96d5913d43abef20e5b7049646",
          "message": "ideal_mhd_model: share the contravariant-field kernel (bsupu, bsupv) (#569)",
          "timestamp": "2026-07-08T08:21:57+02:00",
          "tree_id": "5d0e9ccf0341e22da884a9204450edf363d88016",
          "url": "https://github.com/proximafusion/vmecpp/commit/10938d76de9e7d96d5913d43abef20e5b7049646"
        },
        "date": 1783604247985,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002646506270481682,
            "extra": "iterations: 953\ncpu: 0.00026461940398740815 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00033829983881706206,
            "extra": "iterations: 828\ncpu: 0.0003382784649758454 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005375199090866816,
            "extra": "iterations: 525\ncpu: 0.0005375111104761907 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000601293707406649,
            "extra": "iterations: 465\ncpu: 0.0006012564580645164 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005176966379839039,
            "extra": "iterations: 541\ncpu: 0.0005176888964879856 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010093990847361174,
            "extra": "iterations: 278\ncpu: 0.0010093823453237415 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018447982637505784,
            "extra": "iterations: 152\ncpu: 0.0018446706447368415 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027299348045797906,
            "extra": "iterations: 102\ncpu: 0.0027297074509803933 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d1c2328aee3ee8e7613ebcba1b023bcbcf041454",
          "message": "Add C++ Google Benchmark microbenchmarks for critical hot functions (#606)",
          "timestamp": "2026-07-08T08:48:02+02:00",
          "tree_id": "ffb12cfbe7d56b38c4799b2acec380d6a211006f",
          "url": "https://github.com/proximafusion/vmecpp/commit/d1c2328aee3ee8e7613ebcba1b023bcbcf041454"
        },
        "date": 1783604248377,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026401902683268846,
            "extra": "iterations: 1061\ncpu: 0.00026400746088595665 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00033950863462505924,
            "extra": "iterations: 825\ncpu: 0.0003395026096969698 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005041528091156226,
            "extra": "iterations: 556\ncpu: 0.0005040230035971223 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006536320724308631,
            "extra": "iterations: 427\ncpu: 0.0006536214496487119 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005482650557931797,
            "extra": "iterations: 537\ncpu: 0.0005482554003724396 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001011884599816498,
            "extra": "iterations: 277\ncpu: 0.0010117755848375446 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018382539936140471,
            "extra": "iterations: 153\ncpu: 0.001838225366013072 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027397847643085556,
            "extra": "iterations: 102\ncpu: 0.0027394098823529424 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.00003130066962469192,
            "extra": "iterations: 7560\ncpu: 3.129870634920636e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004504500364898884,
            "extra": "iterations: 6187\ncpu: 4.503822999838372e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006871628643843928,
            "extra": "iterations: 406\ncpu: 0.0006871162290640394 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015134542219100461,
            "extra": "iterations: 186\ncpu: 0.0015131566989247307 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007030184348643981,
            "extra": "iterations: 3979\ncpu: 7.031797310882144e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.000539087332212008,
            "extra": "iterations: 520\ncpu: 0.0005391240461538462 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.003208609833114449,
            "extra": "iterations: 87\ncpu: 0.0032087959770115025 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006692264117660146,
            "extra": "iterations: 4247\ncpu: 6.69358179891678e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005213142780775436,
            "extra": "iterations: 534\ncpu: 0.0005213797996254715 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031494746047459293,
            "extra": "iterations: 89\ncpu: 0.003149421415730328 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.0002793248077454168,
            "extra": "iterations: 991\ncpu: 0.0002792849646821391 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0012371909301892845,
            "extra": "iterations: 226\ncpu: 0.0012370831327433631 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004613266616571145,
            "extra": "iterations: 61\ncpu: 0.004612969377049181 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011105899810791017,
            "extra": "iterations: 25\ncpu: 0.0111047266 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c55210cb72406cfe184665b7bdf8832936426543",
          "message": "ideal_mhd_model: share the covariant-field kernel (bsubu, bsubv) (#570)",
          "timestamp": "2026-07-08T08:50:13+02:00",
          "tree_id": "e8171646866b31b130b72d4100fe89b85bfd7838",
          "url": "https://github.com/proximafusion/vmecpp/commit/c55210cb72406cfe184665b7bdf8832936426543"
        },
        "date": 1783604248328,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.000263701089163133,
            "extra": "iterations: 1058\ncpu: 0.00026368794706994333 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00032886809851290916,
            "extra": "iterations: 845\ncpu: 0.0003288622923076923 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0004962627963547993,
            "extra": "iterations: 566\ncpu: 0.0004962547968197881 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006144980633024053,
            "extra": "iterations: 457\ncpu: 0.0006144874770240699 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005172129030580875,
            "extra": "iterations: 540\ncpu: 0.000517204511111111 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010202374771563676,
            "extra": "iterations: 274\ncpu: 0.001020139572992701 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018463782127329845,
            "extra": "iterations: 151\ncpu: 0.001846346264900664 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027454177152763295,
            "extra": "iterations: 103\ncpu: 0.0027453715825242688 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.00003156139732018768,
            "extra": "iterations: 8966\ncpu: 3.156085233102833e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004546376447982553,
            "extra": "iterations: 6161\ncpu: 4.5454186820321395e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006863328756070605,
            "extra": "iterations: 408\ncpu: 0.0006863021715686275 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015062337280601584,
            "extra": "iterations: 186\ncpu: 0.0015061301451612897 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007033318131413292,
            "extra": "iterations: 3980\ncpu: 7.035028266331685e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005380485969160515,
            "extra": "iterations: 518\ncpu: 0.0005380909594594608 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.003218998854187714,
            "extra": "iterations: 87\ncpu: 0.0032191653333333357 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006607128504816511,
            "extra": "iterations: 4242\ncpu: 6.60797553041028e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005213298691447016,
            "extra": "iterations: 539\ncpu: 0.0005213608274582638 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031690597534179688,
            "extra": "iterations: 89\ncpu: 0.0031690825730336814 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00027777652929324913,
            "extra": "iterations: 1010\ncpu: 0.00027777181584158413 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0011928142385279879,
            "extra": "iterations: 235\ncpu: 0.0011926692808510641 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004673107465108236,
            "extra": "iterations: 60\ncpu: 0.004672778566666665 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.01113715171813965,
            "extra": "iterations: 25\ncpu: 0.011130025760000006 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "d2033339109a71943f87d81046b6acac19763ce0",
          "message": "ideal_mhd_model: share the magnetic-pressure kernel (#571)",
          "timestamp": "2026-07-08T09:08:12+02:00",
          "tree_id": "5034e4f34354d10e6f9d9ebbef2cf71276f069d9",
          "url": "https://github.com/proximafusion/vmecpp/commit/d2033339109a71943f87d81046b6acac19763ce0"
        },
        "date": 1783604248380,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002699786945676483,
            "extra": "iterations: 1041\ncpu: 0.0002699469586935639 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003538483035954781,
            "extra": "iterations: 794\ncpu: 0.0003538272065491185 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005199687218976865,
            "extra": "iterations: 537\ncpu: 0.0005199612309124767 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006180851664764203,
            "extra": "iterations: 453\ncpu: 0.0006180767373068431 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005175768229354352,
            "extra": "iterations: 542\ncpu: 0.0005175446752767529 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0009978182375218947,
            "extra": "iterations: 281\ncpu: 0.0009978023380782918 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001878532810487609,
            "extra": "iterations: 138\ncpu: 0.0018785001594202873 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027402522517185586,
            "extra": "iterations: 102\ncpu: 0.00274003060784314 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.000031471623438417194,
            "extra": "iterations: 8997\ncpu: 3.147105868622875e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.000049465306128235894,
            "extra": "iterations: 5661\ncpu: 4.9458439498321864e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006944973474696612,
            "extra": "iterations: 403\ncpu: 0.0006944857493796525 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015124114784034525,
            "extra": "iterations: 185\ncpu: 0.001512300897297297 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007037946412696245,
            "extra": "iterations: 3982\ncpu: 7.03934271722754e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005411936686589168,
            "extra": "iterations: 520\ncpu: 0.0005410560596153831 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.00321197509765625,
            "extra": "iterations: 87\ncpu: 0.00321194304597702 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006612453919037904,
            "extra": "iterations: 4203\ncpu: 6.613745396145544e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005221531155833097,
            "extra": "iterations: 537\ncpu: 0.0005220463463687208 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031469725490955824,
            "extra": "iterations: 89\ncpu: 0.0031470216629213652 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.0002817684448273138,
            "extra": "iterations: 986\ncpu: 0.00028176360344827587 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.001198052341102535,
            "extra": "iterations: 234\ncpu: 0.0011974868504273505 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004618285132236168,
            "extra": "iterations: 61\ncpu: 0.004618040016393442 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011049423217773437,
            "extra": "iterations: 25\ncpu: 0.011044113120000008 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "acd99d524bc9ca92528cc031bc5d5518feb33259",
          "message": "ideal_mhd_model: share the MHD force-density kernel (6th/last) (#572)",
          "timestamp": "2026-07-08T09:30:41+02:00",
          "tree_id": "909aa733d849bf68c7e22d8da94d11507e914ecf",
          "url": "https://github.com/proximafusion/vmecpp/commit/acd99d524bc9ca92528cc031bc5d5518feb33259"
        },
        "date": 1783604248288,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026340327172909143,
            "extra": "iterations: 1060\ncpu: 0.00026337826792452836 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003419278883466534,
            "extra": "iterations: 816\ncpu: 0.00034190829779411774 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005051866709757195,
            "extra": "iterations: 556\ncpu: 0.0005051775233812951 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006046290049737104,
            "extra": "iterations: 466\ncpu: 0.0006045956480686697 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005192459737501181,
            "extra": "iterations: 538\ncpu: 0.000519226405204461 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010493549987347457,
            "extra": "iterations: 274\ncpu: 0.001049337270072994 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018741430020799824,
            "extra": "iterations: 153\ncpu: 0.0018740389542483684 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002742703598324615,
            "extra": "iterations: 101\ncpu: 0.002742668445544557 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.00003241228011429257,
            "extra": "iterations: 8936\ncpu: 3.2402870299910486e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004524900910146196,
            "extra": "iterations: 6168\ncpu: 4.524465596627757e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006855140396901623,
            "extra": "iterations: 409\ncpu: 0.000685502858190709 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015187173276334197,
            "extra": "iterations: 185\ncpu: 0.0015178887081081088 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007056530166721944,
            "extra": "iterations: 3975\ncpu: 7.057958163521984e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005385857362013597,
            "extra": "iterations: 520\ncpu: 0.0005386545365384609 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.0032072231687348468,
            "extra": "iterations: 87\ncpu: 0.003207254298850574 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.0000659809778478523,
            "extra": "iterations: 4239\ncpu: 6.599039065817379e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005224841234852391,
            "extra": "iterations: 538\ncpu: 0.0005225450297397779 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.003147294012348304,
            "extra": "iterations: 89\ncpu: 0.003147397617977538 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00028285393200055384,
            "extra": "iterations: 991\ncpu: 0.00028280332088799183 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.001197181196294279,
            "extra": "iterations: 234\ncpu: 0.0011969354401709397 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004616363843282064,
            "extra": "iterations: 60\ncpu: 0.004615840083333327 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011057014465332032,
            "extra": "iterations: 25\ncpu: 0.011050390840000006 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Wenyin Wei",
            "email": "wenyin.wei.ww@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "99d2e45ea900bcf817edfe841abdac53f92de5fb",
          "message": "Fix current density in magnetic field visualization example (#607)",
          "timestamp": "2026-07-08T15:48:59+08:00",
          "tree_id": "5e3ee6c5b68a715dfc9b315615daaffaafe9e473",
          "url": "https://github.com/proximafusion/vmecpp/commit/99d2e45ea900bcf817edfe841abdac53f92de5fb"
        },
        "date": 1783604248240,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002629406378024261,
            "extra": "iterations: 1061\ncpu: 0.00026293686145146096 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003401103164210464,
            "extra": "iterations: 825\ncpu: 0.0003401041187878788 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005035366085793475,
            "extra": "iterations: 556\ncpu: 0.0005033939676258993 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0005981697995438535,
            "extra": "iterations: 468\ncpu: 0.0005981068846153844 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005181373377604317,
            "extra": "iterations: 541\ncpu: 0.0005181293456561922 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010113690303981521,
            "extra": "iterations: 277\ncpu: 0.0010112145018050535 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001849088229631123,
            "extra": "iterations: 152\ncpu: 0.0018490562302631577 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002732597145379759,
            "extra": "iterations: 102\ncpu: 0.0027322847058823515 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.000031088271648642794,
            "extra": "iterations: 8962\ncpu: 3.108217261771926e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004508023207284967,
            "extra": "iterations: 6272\ncpu: 4.50794661989796e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006840496528439406,
            "extra": "iterations: 410\ncpu: 0.0006840406487804878 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015396901539393834,
            "extra": "iterations: 182\ncpu: 0.0015391161318681315 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007019186919590212,
            "extra": "iterations: 3975\ncpu: 7.020877056603725e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005384897221030528,
            "extra": "iterations: 519\ncpu: 0.0005385364508670508 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.0032057873038358465,
            "extra": "iterations: 86\ncpu: 0.0032057775465116345 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006591447006315875,
            "extra": "iterations: 4225\ncpu: 6.592472142011465e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005258794603401056,
            "extra": "iterations: 537\ncpu: 0.0005258630540037288 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031553118416432586,
            "extra": "iterations: 89\ncpu: 0.0031553111011235827 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00028262128743345875,
            "extra": "iterations: 991\ncpu: 0.0002826166165489405 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0011902088814593377,
            "extra": "iterations: 235\ncpu: 0.0011899874680851058 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004623983727126825,
            "extra": "iterations: 61\ncpu: 0.004623734049180326 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.01110825538635254,
            "extra": "iterations: 25\ncpu: 0.011107120199999994 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ca0156935152b9d594f2223916985897c9d3ce9b",
          "message": "enzyme: exact Hessian of the composed local force map (#573)",
          "timestamp": "2026-07-08T09:49:17+02:00",
          "tree_id": "3f61c23861be10fb2def771ff724e1f0aba592c8",
          "url": "https://github.com/proximafusion/vmecpp/commit/ca0156935152b9d594f2223916985897c9d3ce9b"
        },
        "date": 1783604248354,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002625701540992374,
            "extra": "iterations: 1050\ncpu: 0.00026254961619047624 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00033945263730988956,
            "extra": "iterations: 826\ncpu: 0.00033941765254237295 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005054465254219704,
            "extra": "iterations: 553\ncpu: 0.0005054392567811935 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006034420283567241,
            "extra": "iterations: 466\ncpu: 0.0006033982703862662 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005226932953451282,
            "extra": "iterations: 535\ncpu: 0.0005226478878504675 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010063810587783869,
            "extra": "iterations: 279\ncpu: 0.001006363258064517 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001834208669226154,
            "extra": "iterations: 153\ncpu: 0.001834124320261438 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.00272730484749507,
            "extra": "iterations: 103\ncpu: 0.0027272655922330087 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.00003111825664429965,
            "extra": "iterations: 8956\ncpu: 3.111255270209915e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.0000456023871094623,
            "extra": "iterations: 6153\ncpu: 4.5601675605395744e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006916258070203993,
            "extra": "iterations: 405\ncpu: 0.0006916143037037035 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015058440546835623,
            "extra": "iterations: 186\ncpu: 0.0015056950430107536 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007023640272309897,
            "extra": "iterations: 3986\ncpu: 7.025035198193699e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005398993308727558,
            "extra": "iterations: 520\ncpu: 0.000539966607692307 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.0032104240066703708,
            "extra": "iterations: 87\ncpu: 0.0032105576091954052 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006600786254925937,
            "extra": "iterations: 4246\ncpu: 6.602262293923858e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.000519615378583296,
            "extra": "iterations: 539\ncpu: 0.0005196346326530626 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031453866637154914,
            "extra": "iterations: 89\ncpu: 0.003145401752808977 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00028298021566988246,
            "extra": "iterations: 990\ncpu: 0.00028296179292929284 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0011909150062723362,
            "extra": "iterations: 235\ncpu: 0.0011908961361702124 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004606071065683835,
            "extra": "iterations: 61\ncpu: 0.00460599362295082 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011556282043457032,
            "extra": "iterations: 25\ncpu: 0.011552556200000003 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "888163596accf7515c0a55eb28affd33ba45c048",
          "message": "ideal_mhd_model: share the hybrid lambda-force kernel (#574)",
          "timestamp": "2026-07-08T10:09:55+02:00",
          "tree_id": "94572c488fe96cabd2e5e3e30c1e88ff0ce12d54",
          "url": "https://github.com/proximafusion/vmecpp/commit/888163596accf7515c0a55eb28affd33ba45c048"
        },
        "date": 1783604248203,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002647911508878072,
            "extra": "iterations: 1056\ncpu: 0.00026472366287878796 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003559992816915342,
            "extra": "iterations: 786\ncpu: 0.00035597662595419856 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005954372565555935,
            "extra": "iterations: 526\ncpu: 0.0005954284847908747 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006214798821343317,
            "extra": "iterations: 450\ncpu: 0.000621358737777778 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005191413886485047,
            "extra": "iterations: 538\ncpu: 0.0005191327899628254 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001025213863386776,
            "extra": "iterations: 273\ncpu: 0.001025037102564102 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001826555900324404,
            "extra": "iterations: 153\ncpu: 0.0018264638692810455 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027765236278571707,
            "extra": "iterations: 101\ncpu: 0.002776480881188119 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.00003111699580141482,
            "extra": "iterations: 9017\ncpu: 3.1116416213818346e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004468557366914156,
            "extra": "iterations: 6270\ncpu: 4.468172695374801e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006787076737116843,
            "extra": "iterations: 412\ncpu: 0.0006786775461165051 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015015153474705195,
            "extra": "iterations: 186\ncpu: 0.0015014465322580638 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007044378124522145,
            "extra": "iterations: 3934\ncpu: 7.045344814438262e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005387950911973847,
            "extra": "iterations: 517\ncpu: 0.0005388624255319165 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.003215665041014206,
            "extra": "iterations: 86\ncpu: 0.003213579348837215 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006639111830677427,
            "extra": "iterations: 4187\ncpu: 6.640056364938841e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005218253206850878,
            "extra": "iterations: 536\ncpu: 0.0005218994664179041 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031697642937135164,
            "extra": "iterations: 89\ncpu: 0.0031672247528089976 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00028242623542156257,
            "extra": "iterations: 994\ncpu: 0.0002823857484909456 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0011943874196109609,
            "extra": "iterations: 234\ncpu: 0.0011937132222222216 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004587193004420546,
            "extra": "iterations: 61\ncpu: 0.004586268622950817 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011035861968994141,
            "extra": "iterations: 25\ncpu: 0.011035292400000002 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "f36757d8ec444b5f5ba0d2211442f9af5b5cccfc",
          "message": "ideal_mhd_model: share the constraint-force kernels (#575)",
          "timestamp": "2026-07-08T11:20:18+02:00",
          "tree_id": "8a175395c8e040b020b87da2468d50fa80eeceec",
          "url": "https://github.com/proximafusion/vmecpp/commit/f36757d8ec444b5f5ba0d2211442f9af5b5cccfc"
        },
        "date": 1783604248428,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00025631573574602114,
            "extra": "iterations: 1089\ncpu: 0.0002562725399449036 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003889352083206177,
            "extra": "iterations: 720\ncpu: 0.00038892937777777785 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005402172258658003,
            "extra": "iterations: 516\ncpu: 0.0005401773488372093 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006795672024543559,
            "extra": "iterations: 411\ncpu: 0.0006795285425790754 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005187427556073225,
            "extra": "iterations: 540\ncpu: 0.0005187346703703705 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010168509049849077,
            "extra": "iterations: 275\ncpu: 0.0010166618145454544 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018492060781314674,
            "extra": "iterations: 151\ncpu: 0.0018491713178807956 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027039784651536206,
            "extra": "iterations: 104\ncpu: 0.0027037247115384573 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.00003113216824001736,
            "extra": "iterations: 9000\ncpu: 3.11305898888889e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004501022990396377,
            "extra": "iterations: 6218\ncpu: 4.5009424895464796e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006785733359200614,
            "extra": "iterations: 413\ncpu: 0.0006785608305084743 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015172146462105415,
            "extra": "iterations: 185\ncpu: 0.0015171436540540543 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007032737674483335,
            "extra": "iterations: 3984\ncpu: 7.034150451807208e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005442852689121959,
            "extra": "iterations: 519\ncpu: 0.0005443551560693668 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.0032133343576014727,
            "extra": "iterations: 87\ncpu: 0.003213338954022986 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006592665347498492,
            "extra": "iterations: 4231\ncpu: 6.593601796265816e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005227549752192712,
            "extra": "iterations: 536\ncpu: 0.0005227590429104517 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031494317429789,
            "extra": "iterations: 89\ncpu: 0.0031494910674157283 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00028046872933865075,
            "extra": "iterations: 997\ncpu: 0.0002804509027081244 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0012048897535904596,
            "extra": "iterations: 230\ncpu: 0.0012047954826086946 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004604085547025087,
            "extra": "iterations: 61\ncpu: 0.004603641606557377 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011018466949462891,
            "extra": "iterations: 25\ncpu: 0.011015994359999998 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "9de6072722f45ab6ad10b1d315baa05115137e9a",
          "message": "Merge fast_poloidal and fast_toroidal Fourier bases into one template (#611)",
          "timestamp": "2026-07-08T09:20:31-04:00",
          "tree_id": "a2d28fa6321018bda16919a4bf4a1aea2adffc3e",
          "url": "https://github.com/proximafusion/vmecpp/commit/9de6072722f45ab6ad10b1d315baa05115137e9a"
        },
        "date": 1783604248253,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002671688141654626,
            "extra": "iterations: 1049\ncpu: 0.0002671641448999047 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003621655424641822,
            "extra": "iterations: 772\ncpu: 0.00036214344300518136 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005182235329239457,
            "extra": "iterations: 540\ncpu: 0.0005181712444444446 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006189462101748973,
            "extra": "iterations: 453\ncpu: 0.0006189353399558499 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005171841041438019,
            "extra": "iterations: 541\ncpu: 0.0005170700295748613 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010289795258465937,
            "extra": "iterations: 272\ncpu: 0.0010289633786764697 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018354740018158957,
            "extra": "iterations: 153\ncpu: 0.0018353640718954264 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027446513082466876,
            "extra": "iterations: 102\ncpu: 0.0027443642058823563 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.00003120702580274342,
            "extra": "iterations: 8961\ncpu: 3.120648220064725e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004625679839932878,
            "extra": "iterations: 6103\ncpu: 4.6249126495166315e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006430663100076379,
            "extra": "iterations: 436\ncpu: 0.0006430560825688072 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015355007989065988,
            "extra": "iterations: 182\ncpu: 0.0015354739670329675 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.0000703179695533766,
            "extra": "iterations: 3981\ncpu: 7.033929037930166e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.000539171443049848,
            "extra": "iterations: 519\ncpu: 0.0005392708439306358 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.0032119613954390606,
            "extra": "iterations: 87\ncpu: 0.003211586977011486 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.0000661802741716493,
            "extra": "iterations: 4240\ncpu: 6.619559622641633e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005212968654846878,
            "extra": "iterations: 534\ncpu: 0.0005213010898876396 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.003149704986743713,
            "extra": "iterations: 89\ncpu: 0.003149372483146059 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00028226651138995765,
            "extra": "iterations: 995\ncpu: 0.00028226243417085435 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0011986928948005383,
            "extra": "iterations: 233\ncpu: 0.0011985835879828332 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004690726598103841,
            "extra": "iterations: 60\ncpu: 0.004689305750000002 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.01112051010131836,
            "extra": "iterations: 25\ncpu: 0.011118527320000002 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Christopher Albert",
            "email": "albert@tugraz.at",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "902d93092f4836c542ee9e6042c5acf3432a05d3",
          "message": "pybind: expose the unpreconditioned internal-basis gradient (#577)",
          "timestamp": "2026-07-08T20:12:48+02:00",
          "tree_id": "1d2259537eb1696ba807972e3fa3a69aed39b7ed",
          "url": "https://github.com/proximafusion/vmecpp/commit/902d93092f4836c542ee9e6042c5acf3432a05d3"
        },
        "date": 1783604248223,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002652010081503629,
            "extra": "iterations: 1055\ncpu: 0.0002651964511848342 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.0003560431757751776,
            "extra": "iterations: 784\ncpu: 0.000356029174744898 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005205652527649606,
            "extra": "iterations: 538\ncpu: 0.0005205570576208179 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006131774003285143,
            "extra": "iterations: 454\ncpu: 0.0006131332378854628 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005173170057873558,
            "extra": "iterations: 539\ncpu: 0.0005173082615955474 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010074441150952412,
            "extra": "iterations: 279\ncpu: 0.0010073425268817202 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018627627899772245,
            "extra": "iterations: 152\ncpu: 0.0018627341447368397 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002738781345700755,
            "extra": "iterations: 103\ncpu: 0.00273854062135922 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.00003123629620573643,
            "extra": "iterations: 8883\ncpu: 3.1234739277271195e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004739313727978647,
            "extra": "iterations: 6317\ncpu: 4.7391173658382156e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006723469852851833,
            "extra": "iterations: 434\ncpu: 0.0006723351474654381 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015509181848451412,
            "extra": "iterations: 179\ncpu: 0.0015508438044692728 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007212605086000405,
            "extra": "iterations: 3982\ncpu: 7.214282872928176e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.000589331702503097,
            "extra": "iterations: 391\ncpu: 0.0005894266138107407 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.003241807564921763,
            "extra": "iterations: 87\ncpu: 0.0032420567126436686 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006608523359567697,
            "extra": "iterations: 4219\ncpu: 6.605151836928189e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005224115812956397,
            "extra": "iterations: 536\ncpu: 0.0005224287611940325 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031509747665919617,
            "extra": "iterations: 89\ncpu: 0.0031508562471910216 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.0002794781637625699,
            "extra": "iterations: 989\ncpu: 0.00027946219514661297 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0011975276164519482,
            "extra": "iterations: 234\ncpu: 0.0011974507307692312 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004578836628648102,
            "extra": "iterations: 61\ncpu: 0.0045787635573770495 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011154613494873048,
            "extra": "iterations: 25\ncpu: 0.01112824068 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cd3dfc2d6facd85e77f076120b479ad1c92a5dbc",
          "message": "Clean up benchmark on demandplots (#615)",
          "timestamp": "2026-07-08T23:12:19+02:00",
          "tree_id": "ab8b5c4ded4e43c9df5c47f8b23e03b29fe692e2",
          "url": "https://github.com/proximafusion/vmecpp/commit/cd3dfc2d6facd85e77f076120b479ad1c92a5dbc"
        },
        "date": 1783604248361,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002664251336623286,
            "extra": "iterations: 1047\ncpu: 0.00026642063992359124 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00035755796102799773,
            "extra": "iterations: 781\ncpu: 0.00035753953777208715 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005191022699529475,
            "extra": "iterations: 539\ncpu: 0.0005190951224489797 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006119892586787196,
            "extra": "iterations: 458\ncpu: 0.0006119786986899564 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005839635928471884,
            "extra": "iterations: 480\ncpu: 0.0005839199729166671 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010397690313833732,
            "extra": "iterations: 270\ncpu: 0.0010397509111111111 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.001851739067780344,
            "extra": "iterations: 152\ncpu: 0.0018515995986842094 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0028732601477175343,
            "extra": "iterations: 98\ncpu: 0.0028732125000000036 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.0000365479169402515,
            "extra": "iterations: 7667\ncpu: 3.654513421155602e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004394493013058069,
            "extra": "iterations: 6360\ncpu: 4.394268710691824e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006328918275789977,
            "extra": "iterations: 442\ncpu: 0.0006328539276018102 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.001533761050531773,
            "extra": "iterations: 183\ncpu: 0.001533733978142077 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007030421083078445,
            "extra": "iterations: 3975\ncpu: 7.031651547169817e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005384996099379456,
            "extra": "iterations: 515\ncpu: 0.0005385459417475749 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.003213959178705325,
            "extra": "iterations: 87\ncpu: 0.0032139305057471285 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.0000664502220576764,
            "extra": "iterations: 4239\ncpu: 6.644429440905836e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005222532484266493,
            "extra": "iterations: 540\ncpu: 0.0005223082925925923 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0033475079319693827,
            "extra": "iterations: 88\ncpu: 0.0033477060454545505 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00027929799473700833,
            "extra": "iterations: 1005\ncpu: 0.00027923700398009956 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0011964884854979434,
            "extra": "iterations: 236\ncpu: 0.0011964694491525432 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004643007119496664,
            "extra": "iterations: 60\ncpu: 0.00464238781666666 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011082639694213869,
            "extra": "iterations: 25\ncpu: 0.011080300799999998 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "7c811775db796064e7f36e54a6ecedf0258d9ed0",
          "message": "Report less, more targeted results in FFT bench (#617)",
          "timestamp": "2026-07-09T11:25:07+02:00",
          "tree_id": "64bbb7827ce86119c7c8e7a5fcb32ce919582512",
          "url": "https://github.com/proximafusion/vmecpp/commit/7c811775db796064e7f36e54a6ecedf0258d9ed0"
        },
        "date": 1783604248175,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026670066993808106,
            "extra": "iterations: 1047\ncpu: 0.00026668719579751675 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00035599078840881814,
            "extra": "iterations: 786\ncpu: 0.00035598529134860053 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005194197741101277,
            "extra": "iterations: 541\ncpu: 0.0005194120425138631 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006153630582909835,
            "extra": "iterations: 456\ncpu: 0.0006153313070175442 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005203727459330355,
            "extra": "iterations: 537\ncpu: 0.0005203657411545626 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010210167277943005,
            "extra": "iterations: 275\ncpu: 0.001020957894545455 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018641408284505209,
            "extra": "iterations: 150\ncpu: 0.0018640123866666663 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027432091095868283,
            "extra": "iterations: 102\ncpu: 0.00274316505882353 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.000031318119410067027,
            "extra": "iterations: 8887\ncpu: 3.1317583661528076e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004386756546442575,
            "extra": "iterations: 6399\ncpu: 4.386672964525708e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006323858623590945,
            "extra": "iterations: 442\ncpu: 0.0006322554864253395 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.001534035470750597,
            "extra": "iterations: 180\ncpu: 0.0015339423222222222 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007040791160990705,
            "extra": "iterations: 3972\ncpu: 7.042106898288006e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005395159297928387,
            "extra": "iterations: 518\ncpu: 0.0005396095559845549 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.003217998592332862,
            "extra": "iterations: 87\ncpu: 0.003218173471264367 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006642488661009938,
            "extra": "iterations: 4248\ncpu: 6.64345621468909e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005222244333600466,
            "extra": "iterations: 538\ncpu: 0.0005222960892193311 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031719045205549764,
            "extra": "iterations: 88\ncpu: 0.003171767977272715 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00027834683183639773,
            "extra": "iterations: 1008\ncpu: 0.00027834171825396806 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0012755643237720839,
            "extra": "iterations: 220\ncpu: 0.001275543400000001 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.0046146424090276,
            "extra": "iterations: 61\ncpu: 0.0046143652295081985 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011060447692871095,
            "extra": "iterations: 25\ncpu: 0.011058456000000003 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "CharlesCNorton",
            "email": "machineelv@gmail.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "686bd269a67c9e0ab0bf5e8b30c23ed97a38486d",
          "message": "Honor iteration_style=parvmec in the native solver (#612)",
          "timestamp": "2026-07-09T08:15:48-04:00",
          "tree_id": "cabfc258e275f96602f96983d37d7c8f9b16915d",
          "url": "https://github.com/proximafusion/vmecpp/commit/686bd269a67c9e0ab0bf5e8b30c23ed97a38486d"
        },
        "date": 1783604248141,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.0002662352120887633,
            "extra": "iterations: 1053\ncpu: 0.000266224641025641 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00035521061972504055,
            "extra": "iterations: 786\ncpu: 0.00035515655470737923 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005211612992127145,
            "extra": "iterations: 538\ncpu: 0.0005210733457249072 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.000613164588711194,
            "extra": "iterations: 457\ncpu: 0.0006131366323851205 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005210899725192931,
            "extra": "iterations: 533\ncpu: 0.0005210811651031896 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.001021670599053376,
            "extra": "iterations: 274\ncpu: 0.0010216540875912416 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018482976838162073,
            "extra": "iterations: 152\ncpu: 0.0018478432434210536 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.0027597989186201946,
            "extra": "iterations: 101\ncpu: 0.0027595095742574283 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.000031193593694736233,
            "extra": "iterations: 8968\ncpu: 3.11930153880464e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.00004380563582387822,
            "extra": "iterations: 6421\ncpu: 4.3804881638374085e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006416199404165286,
            "extra": "iterations: 436\ncpu: 0.000641524013761468 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015277266502380371,
            "extra": "iterations: 184\ncpu: 0.0015277013967391305 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007033276180431274,
            "extra": "iterations: 3979\ncpu: 7.032963257099748e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005388930583505502,
            "extra": "iterations: 519\ncpu: 0.0005389350616570294 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.0032072368709520366,
            "extra": "iterations: 87\ncpu: 0.003206981850574713 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006671478791016661,
            "extra": "iterations: 4244\ncpu: 6.671928228086586e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005209179563895284,
            "extra": "iterations: 537\ncpu: 0.0005209969180633128 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.003148566471056992,
            "extra": "iterations: 89\ncpu: 0.003145252191011261 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.0002808468341827393,
            "extra": "iterations: 1000\ncpu: 0.0002807728850000002 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0011957117851744306,
            "extra": "iterations: 235\ncpu: 0.0011956118595744679 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004658408959706624,
            "extra": "iterations: 60\ncpu: 0.00465479686666667 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.01107245445251465,
            "extra": "iterations: 25\ncpu: 0.01107219440000001 seconds\nthreads: 1"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "committer": {
            "name": "GitHub",
            "email": "noreply@github.com",
            "username": ""
          },
          "distinct": true,
          "id": "e0f1985e4ff668a5c80b92561bb610d93b59f861",
          "message": "Abseil status handling for mgrid errors (#613)",
          "timestamp": "2026-07-09T14:23:00+02:00",
          "tree_id": "f772e916641ec1e556c4a112c885018f6c8f57a6",
          "url": "https://github.com/proximafusion/vmecpp/commit/e0f1985e4ff668a5c80b92561bb610d93b59f861"
        },
        "date": 1783604248407,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "ToroidalFourierToReal/4x4",
            "unit": "seconds",
            "value": 0.00026427910774101156,
            "extra": "iterations: 1058\ncpu: 0.000264274309073724 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/4x4",
            "unit": "seconds",
            "value": 0.00035950641092726996,
            "extra": "iterations: 778\ncpu: 0.0003595006323907456 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/6x8",
            "unit": "seconds",
            "value": 0.0005170126719307327,
            "extra": "iterations: 541\ncpu: 0.0005169849316081333 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/6x8",
            "unit": "seconds",
            "value": 0.0006166517997103116,
            "extra": "iterations: 454\ncpu: 0.0006166414383259909 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x12",
            "unit": "seconds",
            "value": 0.0005162460369299789,
            "extra": "iterations: 543\ncpu: 0.0005162363130755067 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x12",
            "unit": "seconds",
            "value": 0.0010126371314560158,
            "extra": "iterations: 276\ncpu: 0.0010125446811594205 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalFourierToReal/12x13",
            "unit": "seconds",
            "value": 0.0018366402701327676,
            "extra": "iterations: 152\ncpu: 0.0018366117763157891 seconds\nthreads: 1"
          },
          {
            "name": "ToroidalForcesToFourier/12x13",
            "unit": "seconds",
            "value": 0.002738017661898744,
            "extra": "iterations: 102\ncpu: 0.002737907852941179 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/4x4",
            "unit": "seconds",
            "value": 0.00003107663498213091,
            "extra": "iterations: 8998\ncpu: 3.107611235830185e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/7x1",
            "unit": "seconds",
            "value": 0.0000438723196208885,
            "extra": "iterations: 6428\ncpu: 4.386449797759802e-05 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/12x12",
            "unit": "seconds",
            "value": 0.0006387380369703428,
            "extra": "iterations: 439\ncpu: 0.000638690341685649 seconds\nthreads: 1"
          },
          {
            "name": "DeAliasConstraintForce/16x18",
            "unit": "seconds",
            "value": 0.0015410705165968418,
            "extra": "iterations: 181\ncpu: 0.0015409914143646407 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/5x4",
            "unit": "seconds",
            "value": 0.00007109418380580393,
            "extra": "iterations: 3987\ncpu: 7.110009681464782e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/8x6",
            "unit": "seconds",
            "value": 0.0005397257076241293,
            "extra": "iterations: 517\ncpu: 0.000539760808510638 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceSolve/12x8",
            "unit": "seconds",
            "value": 0.003213808454316238,
            "extra": "iterations: 87\ncpu: 0.00321345171264367 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/5x4",
            "unit": "seconds",
            "value": 0.00006608049346398779,
            "extra": "iterations: 4243\ncpu: 6.609296606174953e-05 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/8x6",
            "unit": "seconds",
            "value": 0.0005217344401268986,
            "extra": "iterations: 537\ncpu: 0.0005217785940409601 seconds\nthreads: 1"
          },
          {
            "name": "LaplaceDecompose/12x8",
            "unit": "seconds",
            "value": 0.0031522124001149383,
            "extra": "iterations: 89\ncpu: 0.003152182921348332 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/5x4",
            "unit": "seconds",
            "value": 0.00027909738643031945,
            "extra": "iterations: 1006\ncpu: 0.00027908646819085474 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/8x6",
            "unit": "seconds",
            "value": 0.0011960380097739717,
            "extra": "iterations: 234\ncpu: 0.0011957682777777765 seconds\nthreads: 1"
          },
          {
            "name": "TransformGreensFunctionDerivative/12x8",
            "unit": "seconds",
            "value": 0.004611355359437036,
            "extra": "iterations: 61\ncpu: 0.00461128032786885 seconds\nthreads: 1"
          },
          {
            "name": "ComputeOutputQuantities/cma",
            "unit": "seconds",
            "value": 0.011053552627563478,
            "extra": "iterations: 25\ncpu: 0.011043725799999998 seconds\nthreads: 1"
          }
        ]
      }
    ]
  }
}
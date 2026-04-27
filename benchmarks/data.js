window.BENCHMARK_DATA = {
  "lastUpdate": 1777306934438,
  "repoUrl": "https://github.com/proximafusion/vmecpp",
  "entries": {
    "Benchmark": [
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
          "id": "802d098294ea8883b99c4ec2b261debb450aa069",
          "message": "replaced submodules with FetchContent calls",
          "timestamp": "2025-03-07T00:01:03+01:00",
          "tree_id": "7c329fd082e4f19809017a1b913c8669b0844a85",
          "url": "https://github.com/proximafusion/vmecpp/commit/802d098294ea8883b99c4ec2b261debb450aa069"
        },
        "date": 1776338312933,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6293900038523056,
            "unit": "iter/sec",
            "range": "stddev: 0.026581986081059217",
            "extra": "mean: 1.5888 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.6249112462772194,
            "unit": "iter/sec",
            "range": "stddev: 0.017779187307991776",
            "extra": "mean: 1.6002 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7123355737076298,
            "unit": "iter/sec",
            "range": "stddev: 0.003962959449410346",
            "extra": "mean: 1.4038 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27398645386843307,
            "unit": "iter/sec",
            "range": "stddev: 0.007163937595523248",
            "extra": "mean: 3.6498 sec\nrounds: 3"
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
          "id": "6f73b890e5792a5ee97b60a3eab0dcfd69f5099e",
          "message": "Test wheels on all kinds of linux distros",
          "timestamp": "2025-03-08T03:39:59+01:00",
          "tree_id": "be9c75560e51f03b7e4fda8e427186c08aa9b62e",
          "url": "https://github.com/proximafusion/vmecpp/commit/6f73b890e5792a5ee97b60a3eab0dcfd69f5099e"
        },
        "date": 1776338312916,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6298331969815142,
            "unit": "iter/sec",
            "range": "stddev: 0.017513741910844375",
            "extra": "mean: 1.5877 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.6261448129645413,
            "unit": "iter/sec",
            "range": "stddev: 0.02621084852961993",
            "extra": "mean: 1.5971 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7140534517392831,
            "unit": "iter/sec",
            "range": "stddev: 0.003135731202394798",
            "extra": "mean: 1.4005 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26712109359345915,
            "unit": "iter/sec",
            "range": "stddev: 0.09304421917474794",
            "extra": "mean: 3.7436 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c1d280516abda81e480f68acc62068725d8002f0",
          "message": "Add mpi4py to CI setup - needed for VMEC2000.",
          "timestamp": "2025-03-12T08:51:41+01:00",
          "tree_id": "9dce8479a903b4c99e4603a137ea49b2ee7f0e6c",
          "url": "https://github.com/proximafusion/vmecpp/commit/c1d280516abda81e480f68acc62068725d8002f0"
        },
        "date": 1776338313012,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.5955722521107676,
            "unit": "iter/sec",
            "range": "stddev: 0.02486297292497006",
            "extra": "mean: 1.6791 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.6240906946311547,
            "unit": "iter/sec",
            "range": "stddev: 0.018402318995986766",
            "extra": "mean: 1.6023 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7128716600241123,
            "unit": "iter/sec",
            "range": "stddev: 0.0018593044686536523",
            "extra": "mean: 1.4028 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2698587926222305,
            "unit": "iter/sec",
            "range": "stddev: 0.019009898160230087",
            "extra": "mean: 3.7056 sec\nrounds: 3"
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
          "id": "9f2b1cdf7dccb47a614380094e7f57883495c12c",
          "message": "SSH cloning might cause confusion for users that don't have SSH keys configured, default instructions to https",
          "timestamp": "2025-03-12T18:43:48+01:00",
          "tree_id": "a055d6b437a25272460c8d024ac8909fb02520c1",
          "url": "https://github.com/proximafusion/vmecpp/commit/9f2b1cdf7dccb47a614380094e7f57883495c12c"
        },
        "date": 1776338312977,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6252886654503316,
            "unit": "iter/sec",
            "range": "stddev: 0.03578410178999251",
            "extra": "mean: 1.5993 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.633662135491223,
            "unit": "iter/sec",
            "range": "stddev: 0.012540969216665641",
            "extra": "mean: 1.5781 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7149952258386205,
            "unit": "iter/sec",
            "range": "stddev: 0.001269812366014958",
            "extra": "mean: 1.3986 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2726263213071547,
            "unit": "iter/sec",
            "range": "stddev: 0.023851219902231265",
            "extra": "mean: 3.6680 sec\nrounds: 3"
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
          "id": "5d7145b6f42f816d9bcc05477f9efe3f8815180d",
          "message": "MPI finite difference",
          "timestamp": "2025-03-13T02:06:31+01:00",
          "tree_id": "546a186c5724cc8a2735d18c96d5b62a800b9cc1",
          "url": "https://github.com/proximafusion/vmecpp/commit/5d7145b6f42f816d9bcc05477f9efe3f8815180d"
        },
        "date": 1776338312895,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6366267444741006,
            "unit": "iter/sec",
            "range": "stddev: 0.020909562519224097",
            "extra": "mean: 1.5708 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.6251697862675178,
            "unit": "iter/sec",
            "range": "stddev: 0.02020251760789227",
            "extra": "mean: 1.5996 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7133210577801516,
            "unit": "iter/sec",
            "range": "stddev: 0.0022662578894311796",
            "extra": "mean: 1.4019 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27376200288113917,
            "unit": "iter/sec",
            "range": "stddev: 0.017253104450686396",
            "extra": "mean: 3.6528 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "096eb58eb2e6c607f6eff17e3d6c71a89e3eae34",
          "message": "Make a backup of the original input in gradient approx ...",
          "timestamp": "2025-03-13T08:17:34+01:00",
          "tree_id": "6ac7c794056601a992eacae19e5795d2dcee70fd",
          "url": "https://github.com/proximafusion/vmecpp/commit/096eb58eb2e6c607f6eff17e3d6c71a89e3eae34"
        },
        "date": 1776338312778,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.5915719378756197,
            "unit": "iter/sec",
            "range": "stddev: 0.026106406877590527",
            "extra": "mean: 1.6904 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5908905590716983,
            "unit": "iter/sec",
            "range": "stddev: 0.033181528632343166",
            "extra": "mean: 1.6924 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7126416173269606,
            "unit": "iter/sec",
            "range": "stddev: 0.001608708872978342",
            "extra": "mean: 1.4032 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26894033617593643,
            "unit": "iter/sec",
            "range": "stddev: 0.056613799844379274",
            "extra": "mean: 3.7183 sec\nrounds: 3"
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
          "id": "b25bd59a9cbeb0be665051bccb0a294b373dfc46",
          "message": "Add a comparison plot script for checking if a case agrees between PARVMEC and VMEC++.",
          "timestamp": "2025-03-09T19:02:06+01:00",
          "tree_id": "60d72fa4e77d8eaf5d00f9a5af18d58f16f30604",
          "url": "https://github.com/proximafusion/vmecpp/commit/b25bd59a9cbeb0be665051bccb0a294b373dfc46"
        },
        "date": 1776338312993,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.617813281290619,
            "unit": "iter/sec",
            "range": "stddev: 0.03804501968476442",
            "extra": "mean: 1.6186 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5850255621080762,
            "unit": "iter/sec",
            "range": "stddev: 0.02541667824535984",
            "extra": "mean: 1.7093 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7174983044020302,
            "unit": "iter/sec",
            "range": "stddev: 0.0002693603543841597",
            "extra": "mean: 1.3937 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.272653054359866,
            "unit": "iter/sec",
            "range": "stddev: 0.002460786004133302",
            "extra": "mean: 3.6677 sec\nrounds: 3"
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
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8114f2fd34857cfb270f4d813661b0edf56b21f6",
          "message": "Update examples/compare_vmecpp_to_parvmec.py",
          "timestamp": "2025-03-10T18:31:52+01:00",
          "tree_id": "a5c0af49d01256e8a35a00a61860ed0cf231074e",
          "url": "https://github.com/proximafusion/vmecpp/commit/8114f2fd34857cfb270f4d813661b0edf56b21f6"
        },
        "date": 1776338312937,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.596851188959408,
            "unit": "iter/sec",
            "range": "stddev: 0.01981476846109757",
            "extra": "mean: 1.6755 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5974652740797307,
            "unit": "iter/sec",
            "range": "stddev: 0.019254417103083506",
            "extra": "mean: 1.6737 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7156123437099166,
            "unit": "iter/sec",
            "range": "stddev: 0.004254504681059182",
            "extra": "mean: 1.3974 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2716304942298279,
            "unit": "iter/sec",
            "range": "stddev: 0.012792313153557961",
            "extra": "mean: 3.6815 sec\nrounds: 3"
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
          "id": "ff428e88a9a66416be7dfd1bed15c9de84ca329e",
          "message": "Add a simple example for hot-restart run.",
          "timestamp": "2025-03-13T13:53:31+01:00",
          "tree_id": "48817f5c25327f76a2d8dbce35844d2f8a343a23",
          "url": "https://github.com/proximafusion/vmecpp/commit/ff428e88a9a66416be7dfd1bed15c9de84ca329e"
        },
        "date": 1776338313098,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6008658363498727,
            "unit": "iter/sec",
            "range": "stddev: 0.020280315137889054",
            "extra": "mean: 1.6643 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5978575723912535,
            "unit": "iter/sec",
            "range": "stddev: 0.005529349479616177",
            "extra": "mean: 1.6726 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.709206591633545,
            "unit": "iter/sec",
            "range": "stddev: 0.00518634173753915",
            "extra": "mean: 1.4100 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26902829747521556,
            "unit": "iter/sec",
            "range": "stddev: 0.015903846302647465",
            "extra": "mean: 3.7171 sec\nrounds: 3"
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
          "id": "dac6fbb2d0011835523931ff0412633bd31a2e40",
          "message": "Add an example of how scaling the perturbation size affects niter",
          "timestamp": "2025-03-13T14:01:14+01:00",
          "tree_id": "e75b558153ed6e5a4b67f486dad044eb2768f7f3",
          "url": "https://github.com/proximafusion/vmecpp/commit/dac6fbb2d0011835523931ff0412633bd31a2e40"
        },
        "date": 1776338313054,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6069894485373089,
            "unit": "iter/sec",
            "range": "stddev: 0.03554800763043278",
            "extra": "mean: 1.6475 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5914964141923809,
            "unit": "iter/sec",
            "range": "stddev: 0.010495471764395477",
            "extra": "mean: 1.6906 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7124619056148426,
            "unit": "iter/sec",
            "range": "stddev: 0.0027451077139031585",
            "extra": "mean: 1.4036 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.270194663206862,
            "unit": "iter/sec",
            "range": "stddev: 0.019302896122166843",
            "extra": "mean: 3.7010 sec\nrounds: 3"
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
          "id": "c7dbfd122e77e624558e9368251d4a85a0aa9553",
          "message": "Add an example of random perturbations of an initial run.",
          "timestamp": "2025-03-13T14:17:36+01:00",
          "tree_id": "8a3d31de1e5f31caaa453134e9600d1d04541ff0",
          "url": "https://github.com/proximafusion/vmecpp/commit/c7dbfd122e77e624558e9368251d4a85a0aa9553"
        },
        "date": 1776338313026,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.5901090698074493,
            "unit": "iter/sec",
            "range": "stddev: 0.03630279882011335",
            "extra": "mean: 1.6946 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.6165067407389188,
            "unit": "iter/sec",
            "range": "stddev: 0.007388704530665143",
            "extra": "mean: 1.6220 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7114359889983971,
            "unit": "iter/sec",
            "range": "stddev: 0.0013291082131105163",
            "extra": "mean: 1.4056 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2704944538072778,
            "unit": "iter/sec",
            "range": "stddev: 0.008166553075058456",
            "extra": "mean: 3.6969 sec\nrounds: 3"
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
          "id": "058ac31fae6556a5edfea573e38799fd898ac5d1",
          "message": "Add a method to load a VmecWout object from a NetCDF file.",
          "timestamp": "2025-02-06T15:31:33+01:00",
          "tree_id": "091b6dc1587b1e4e71870b7f1dd1461880f3a70c",
          "url": "https://github.com/proximafusion/vmecpp/commit/058ac31fae6556a5edfea573e38799fd898ac5d1"
        },
        "date": 1776338312768,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.576021736033686,
            "unit": "iter/sec",
            "range": "stddev: 0.08639811536336119",
            "extra": "mean: 1.7360 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5521091818901265,
            "unit": "iter/sec",
            "range": "stddev: 0.051303741795929954",
            "extra": "mean: 1.8112 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7142479739329402,
            "unit": "iter/sec",
            "range": "stddev: 0.002219349735431184",
            "extra": "mean: 1.4001 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26744537774500804,
            "unit": "iter/sec",
            "range": "stddev: 0.0668670397796982",
            "extra": "mean: 3.7391 sec\nrounds: 3"
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
            "name": "Jonathan Schilling",
            "email": "130992531+jons-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "54c003f509d297daf2af28798c2c5ed0e0b180c4",
          "message": "make intent clearer, as suggested by eguiraud",
          "timestamp": "2025-03-14T11:01:40+01:00",
          "tree_id": "724c7cdc2f344b787a18a4928b8034f71fc6ebaf",
          "url": "https://github.com/proximafusion/vmecpp/commit/54c003f509d297daf2af28798c2c5ed0e0b180c4"
        },
        "date": 1776338312881,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6144042469383164,
            "unit": "iter/sec",
            "range": "stddev: 0.03684834295849827",
            "extra": "mean: 1.6276 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5837773255480134,
            "unit": "iter/sec",
            "range": "stddev: 0.23943337495898714",
            "extra": "mean: 1.7130 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7150569878621749,
            "unit": "iter/sec",
            "range": "stddev: 0.002191504349272028",
            "extra": "mean: 1.3985 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2723070485641949,
            "unit": "iter/sec",
            "range": "stddev: 0.013442938172383812",
            "extra": "mean: 3.6723 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Enrico Guiraud",
            "email": "eguiraud@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Enrico Guiraud",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "81c6c3591ead1fbfd044ba51226387e56ccc29b8",
          "message": "Fix path comparison in simsopt_compat.py",
          "timestamp": "2025-03-14T15:19:31-06:00",
          "tree_id": "85dd84ceebc5492e06f0a9162cb6f220f6c895ba",
          "url": "https://github.com/proximafusion/vmecpp/commit/81c6c3591ead1fbfd044ba51226387e56ccc29b8"
        },
        "date": 1776338312938,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6107409977844899,
            "unit": "iter/sec",
            "range": "stddev: 0.028225982763145276",
            "extra": "mean: 1.6374 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.6150817470884119,
            "unit": "iter/sec",
            "range": "stddev: 0.03693464179035314",
            "extra": "mean: 1.6258 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7099293313199666,
            "unit": "iter/sec",
            "range": "stddev: 0.004258092686304255",
            "extra": "mean: 1.4086 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2721422103446045,
            "unit": "iter/sec",
            "range": "stddev: 0.006471838391423299",
            "extra": "mean: 3.6745 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Enrico Guiraud",
            "email": "eguiraud@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Enrico Guiraud",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "02e0474a1d0ff523089642b7e292bb07eb7613db",
          "message": "Bump VMEC++ version to v0.2.0",
          "timestamp": "2025-03-14T15:49:54-06:00",
          "tree_id": "a850999c9009fdf87e0605f00d7f40e429756f39",
          "url": "https://github.com/proximafusion/vmecpp/commit/02e0474a1d0ff523089642b7e292bb07eb7613db"
        },
        "date": 1776338312759,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.5252664449024181,
            "unit": "iter/sec",
            "range": "stddev: 0.008189346727117904",
            "extra": "mean: 1.9038 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5813104871044612,
            "unit": "iter/sec",
            "range": "stddev: 0.05996906329380744",
            "extra": "mean: 1.7203 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.706979953791729,
            "unit": "iter/sec",
            "range": "stddev: 0.003050771078269225",
            "extra": "mean: 1.4145 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27040771724617785,
            "unit": "iter/sec",
            "range": "stddev: 0.009682135561293579",
            "extra": "mean: 3.6981 sec\nrounds: 3"
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
          "id": "47e54c08b98e69a106e4242ba047f1cab9ddc6c9",
          "message": "Add Fortran VMEC files for cth_like_fixed_bdy",
          "timestamp": "2025-03-14T15:47:16+01:00",
          "tree_id": "be2b0c9872a7b8dadeb5fe3603d187d246956652",
          "url": "https://github.com/proximafusion/vmecpp/commit/47e54c08b98e69a106e4242ba047f1cab9ddc6c9"
        },
        "date": 1776338312866,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.532836002055425,
            "unit": "iter/sec",
            "range": "stddev: 0.023470396997386005",
            "extra": "mean: 1.8768 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5238589933059847,
            "unit": "iter/sec",
            "range": "stddev: 0.03011268002859596",
            "extra": "mean: 1.9089 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7091611922515184,
            "unit": "iter/sec",
            "range": "stddev: 0.0013162398463913787",
            "extra": "mean: 1.4101 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2719246138346193,
            "unit": "iter/sec",
            "range": "stddev: 0.016145899088693825",
            "extra": "mean: 3.6775 sec\nrounds: 3"
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
          "id": "38b0892a4b580dcc0d0a921d26a5c04b64973806",
          "message": "Fixup VmecWOut.load_wout_file for Fortran VMEC.",
          "timestamp": "2025-03-14T15:41:57+01:00",
          "tree_id": "3e31c86ec738e8c250ad597859ca420cf68bc64b",
          "url": "https://github.com/proximafusion/vmecpp/commit/38b0892a4b580dcc0d0a921d26a5c04b64973806"
        },
        "date": 1776338312842,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.5264906860218392,
            "unit": "iter/sec",
            "range": "stddev: 0.026092921393679736",
            "extra": "mean: 1.8994 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5536779374527474,
            "unit": "iter/sec",
            "range": "stddev: 0.12811894017991488",
            "extra": "mean: 1.8061 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7086018779149847,
            "unit": "iter/sec",
            "range": "stddev: 0.004983012305975064",
            "extra": "mean: 1.4112 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27146987660316996,
            "unit": "iter/sec",
            "range": "stddev: 0.004190435727627033",
            "extra": "mean: 3.6836 sec\nrounds: 3"
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
          "id": "9c6c6c3e11190662fb5a22ac00656001b87c1b0d",
          "message": "Add a README for the examples data.",
          "timestamp": "2025-03-14T23:17:53+01:00",
          "tree_id": "5e5a30d8346eb66cbdc0787490d279b416889c07",
          "url": "https://github.com/proximafusion/vmecpp/commit/9c6c6c3e11190662fb5a22ac00656001b87c1b0d"
        },
        "date": 1776338312971,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.5388507247073079,
            "unit": "iter/sec",
            "range": "stddev: 0.07601820583420897",
            "extra": "mean: 1.8558 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5390652516442758,
            "unit": "iter/sec",
            "range": "stddev: 0.10225878766888935",
            "extra": "mean: 1.8551 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7099196165235713,
            "unit": "iter/sec",
            "range": "stddev: 0.0020969868961329885",
            "extra": "mean: 1.4086 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26833944141046473,
            "unit": "iter/sec",
            "range": "stddev: 0.024443995372596217",
            "extra": "mean: 3.7266 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Enrico Guiraud",
            "email": "eguiraud@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Enrico Guiraud",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "474c1d725b76dc14a6fc20262d05de36f468dbea",
          "message": "In test_simsopt_compat, also test with Fortran inputs",
          "timestamp": "2025-03-14T15:28:48-06:00",
          "tree_id": "53b3e90c48ca4ba3f35c9e5b36384ba879b731c2",
          "url": "https://github.com/proximafusion/vmecpp/commit/474c1d725b76dc14a6fc20262d05de36f468dbea"
        },
        "date": 1776338312864,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.5341180258661248,
            "unit": "iter/sec",
            "range": "stddev: 0.10582507931021831",
            "extra": "mean: 1.8722 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5243650803371391,
            "unit": "iter/sec",
            "range": "stddev: 0.016190743130893543",
            "extra": "mean: 1.9071 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7096377230447023,
            "unit": "iter/sec",
            "range": "stddev: 0.003288628630010634",
            "extra": "mean: 1.4092 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2697802357627719,
            "unit": "iter/sec",
            "range": "stddev: 0.04260359657709518",
            "extra": "mean: 3.7067 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Enrico Guiraud",
            "email": "eguiraud@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Enrico Guiraud",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "f10109d76b27aba75fbad58d83d9d05acc9da9ec",
          "message": "Reuse VMEC++ run in test_simsopt_compat",
          "timestamp": "2025-03-14T15:44:18-06:00",
          "tree_id": "d64d5f77bcecdecdb56a19a23aa5bdcff360d77b",
          "url": "https://github.com/proximafusion/vmecpp/commit/f10109d76b27aba75fbad58d83d9d05acc9da9ec"
        },
        "date": 1776338313084,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6211005406712907,
            "unit": "iter/sec",
            "range": "stddev: 0.019198462148314638",
            "extra": "mean: 1.6100 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.57511996717258,
            "unit": "iter/sec",
            "range": "stddev: 0.24268348202541562",
            "extra": "mean: 1.7388 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7130539236492958,
            "unit": "iter/sec",
            "range": "stddev: 0.003005880548747188",
            "extra": "mean: 1.4024 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2738220680200505,
            "unit": "iter/sec",
            "range": "stddev: 0.005524848193406645",
            "extra": "mean: 3.6520 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Enrico Guiraud",
            "email": "eguiraud@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Enrico Guiraud",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c7223f880cee400dbe8ab4fd9d18dc1e389bc9b3",
          "message": "Bump patch version to release from_wout bugfix",
          "timestamp": "2025-03-14T17:33:52-06:00",
          "tree_id": "8a3e6eeb4145c6c2de2f1be9e69658c6f12ffa2c",
          "url": "https://github.com/proximafusion/vmecpp/commit/c7223f880cee400dbe8ab4fd9d18dc1e389bc9b3"
        },
        "date": 1776338313021,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6277833367489483,
            "unit": "iter/sec",
            "range": "stddev: 0.02116872943636693",
            "extra": "mean: 1.5929 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.6356973696020493,
            "unit": "iter/sec",
            "range": "stddev: 0.01103172711885039",
            "extra": "mean: 1.5731 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7120389242856463,
            "unit": "iter/sec",
            "range": "stddev: 0.006572408340131826",
            "extra": "mean: 1.4044 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27302287427058947,
            "unit": "iter/sec",
            "range": "stddev: 0.02141945364929043",
            "extra": "mean: 3.6627 sec\nrounds: 3"
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
          "id": "8dedd2496e33b80279461dfbe78cca707d8452a0",
          "message": "Explicitly set default behaviors for deprecations warnings in CMake",
          "timestamp": "2025-03-12T14:28:37+01:00",
          "tree_id": "d00acb2ecd8853100c360d42a8359be7cc297f29",
          "url": "https://github.com/proximafusion/vmecpp/commit/8dedd2496e33b80279461dfbe78cca707d8452a0"
        },
        "date": 1776338312958,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.6065040414556304,
            "unit": "iter/sec",
            "range": "stddev: 0.022511898280753725",
            "extra": "mean: 1.6488 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.6209955018126951,
            "unit": "iter/sec",
            "range": "stddev: 0.01160694791619782",
            "extra": "mean: 1.6103 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7127257026436309,
            "unit": "iter/sec",
            "range": "stddev: 0.008083283144685316",
            "extra": "mean: 1.4031 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2730108231332256,
            "unit": "iter/sec",
            "range": "stddev: 0.015182513345970868",
            "extra": "mean: 3.6629 sec\nrounds: 3"
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
          "id": "e622b5aeb6576bede750eaa5fef5f16c9b1589c4",
          "message": "Removed unused field from IdealMhdModel",
          "timestamp": "2025-03-13T13:45:58+01:00",
          "tree_id": "dde7ccea7a2d56f9897f9741ecbf72d4d59c764f",
          "url": "https://github.com/proximafusion/vmecpp/commit/e622b5aeb6576bede750eaa5fef5f16c9b1589c4"
        },
        "date": 1776338313070,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 0.527502774272927,
            "unit": "iter/sec",
            "range": "stddev: 0.02974674802238495",
            "extra": "mean: 1.8957 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 0.5294015184571541,
            "unit": "iter/sec",
            "range": "stddev: 0.01743061802651458",
            "extra": "mean: 1.8889 sec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7111932777675765,
            "unit": "iter/sec",
            "range": "stddev: 0.0008460852780606393",
            "extra": "mean: 1.4061 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2708985324603253,
            "unit": "iter/sec",
            "range": "stddev: 0.020552924918233736",
            "extra": "mean: 3.6914 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "96e65c5bfc80badfca0b6241a3d53e592a8e49a6",
          "message": "Migrate ensure_vmec2000_input et al to _util",
          "timestamp": "2025-03-06T09:52:08+01:00",
          "tree_id": "cad40fdbdfa43932390526ccfe3128409b3e9552",
          "url": "https://github.com/proximafusion/vmecpp/commit/96e65c5bfc80badfca0b6241a3d53e592a8e49a6"
        },
        "date": 1776338312968,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.5110768917341828,
            "unit": "iter/sec",
            "range": "stddev: 0.02497894853333545",
            "extra": "mean: 661.7797 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.4915446877714675,
            "unit": "iter/sec",
            "range": "stddev: 0.014532080514971011",
            "extra": "mean: 670.4459 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7096528170042157,
            "unit": "iter/sec",
            "range": "stddev: 0.0002810959042581285",
            "extra": "mean: 1.4091 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26988542236237995,
            "unit": "iter/sec",
            "range": "stddev: 0.04139434370125922",
            "extra": "mean: 3.7053 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "ce2ce4f9204a6e21088f9062ea692d14f07f0a19",
          "message": "Update clang_tidy.yaml",
          "timestamp": "2025-03-17T18:36:17+01:00",
          "tree_id": "6848cc1af8643f2533a86007c4d8288981f4deea",
          "url": "https://github.com/proximafusion/vmecpp/commit/ce2ce4f9204a6e21088f9062ea692d14f07f0a19"
        },
        "date": 1776338313035,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.6459873127445386,
            "unit": "iter/sec",
            "range": "stddev: 0.013536085812106495",
            "extra": "mean: 607.5381 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.4851480603372995,
            "unit": "iter/sec",
            "range": "stddev: 0.008574895587528874",
            "extra": "mean: 673.3335 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7086264009676533,
            "unit": "iter/sec",
            "range": "stddev: 0.007655770399934506",
            "extra": "mean: 1.4112 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26884720743260987,
            "unit": "iter/sec",
            "range": "stddev: 0.09327507508982255",
            "extra": "mean: 3.7196 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Enrico Guiraud",
            "email": "eguiraud@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Enrico Guiraud",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "1427e0357050acb22d97e208d996d1853b0c2811",
          "message": "Mention the benchmarks repo in the README",
          "timestamp": "2025-03-18T19:15:25-06:00",
          "tree_id": "66d7319f55eba0b0ba096f2d48aa7362f1746eed",
          "url": "https://github.com/proximafusion/vmecpp/commit/1427e0357050acb22d97e208d996d1853b0c2811"
        },
        "date": 1776338312799,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.4951204781598173,
            "unit": "iter/sec",
            "range": "stddev: 0.016666195884652735",
            "extra": "mean: 668.8424 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.5564171075473945,
            "unit": "iter/sec",
            "range": "stddev: 0.04689599446965507",
            "extra": "mean: 642.5013 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7093864103461576,
            "unit": "iter/sec",
            "range": "stddev: 0.004575551760460951",
            "extra": "mean: 1.4097 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.270037821810579,
            "unit": "iter/sec",
            "range": "stddev: 0.05926664716702794",
            "extra": "mean: 3.7032 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Enrico Guiraud",
            "email": "eguiraud@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Enrico Guiraud",
            "email": "148162947+eguiraud-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "da2973af431b1b40febf7aba1b927cca70f9a5de",
          "message": "Rename example {direct_python_api => python_api}.py",
          "timestamp": "2025-03-17T12:20:18-06:00",
          "tree_id": "04ac46570b5d4bc34526a6c172f0f9346f1fb7f6",
          "url": "https://github.com/proximafusion/vmecpp/commit/da2973af431b1b40febf7aba1b927cca70f9a5de"
        },
        "date": 1776338313053,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.482105829211158,
            "unit": "iter/sec",
            "range": "stddev: 0.02367520127350739",
            "extra": "mean: 674.7157 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.4817343180400746,
            "unit": "iter/sec",
            "range": "stddev: 0.010808490600067932",
            "extra": "mean: 674.8848 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7076617586090378,
            "unit": "iter/sec",
            "range": "stddev: 0.010800281618062126",
            "extra": "mean: 1.4131 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2710361902193058,
            "unit": "iter/sec",
            "range": "stddev: 0.037159234677005736",
            "extra": "mean: 3.6895 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Enrico Guiraud",
            "email": "eguiraud@proximafusion.com",
            "username": ""
          },
          "committer": {
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "a490a75577be69c851d66f0e41fc9aaea2fd1ef6",
          "message": "Expand README usage example",
          "timestamp": "2025-03-19T15:28:14-06:00",
          "tree_id": "6b3cdcecf85bb229737d2a46b60170f5aee519c2",
          "url": "https://github.com/proximafusion/vmecpp/commit/a490a75577be69c851d66f0e41fc9aaea2fd1ef6"
        },
        "date": 1776338312985,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.686473765303604,
            "unit": "iter/sec",
            "range": "stddev: 0.004912111780824288",
            "extra": "mean: 592.9532 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.6515541823298379,
            "unit": "iter/sec",
            "range": "stddev: 0.014532970945284248",
            "extra": "mean: 605.4903 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7136940068976875,
            "unit": "iter/sec",
            "range": "stddev: 0.001136500663617945",
            "extra": "mean: 1.4012 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27134370334502755,
            "unit": "iter/sec",
            "range": "stddev: 0.003473032045722538",
            "extra": "mean: 3.6854 sec\nrounds: 3"
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
          "id": "ecb8212a0c21f54ea6ebf5f39e711850af1fe49a",
          "message": "without OpenMP max threads should always be one",
          "timestamp": "2025-03-14T18:07:16+01:00",
          "tree_id": "1f940f5eba27f0382d49c599f760082d02120f00",
          "url": "https://github.com/proximafusion/vmecpp/commit/ecb8212a0c21f54ea6ebf5f39e711850af1fe49a"
        },
        "date": 1776338313077,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.733216529374745,
            "unit": "iter/sec",
            "range": "stddev: 0.003418165124185982",
            "extra": "mean: 576.9620 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.7031568875732899,
            "unit": "iter/sec",
            "range": "stddev: 0.00967463753395908",
            "extra": "mean: 587.1450 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7172590059553652,
            "unit": "iter/sec",
            "range": "stddev: 0.0027050200798731344",
            "extra": "mean: 1.3942 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27117245761654396,
            "unit": "iter/sec",
            "range": "stddev: 0.050951881132611995",
            "extra": "mean: 3.6877 sec\nrounds: 3"
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
          "id": "9f5e1cd85ecf38d9bd4f1fbc284d8d6d71d4f4bf",
          "message": "Remove OpenMPI dep and Eigen3, nlohman::Json (they are fetched by CMake/Bazel)",
          "timestamp": "2025-03-26T17:25:42+01:00",
          "tree_id": "23b1c24f04a832b4e4d0b0daf258c90e013893ed",
          "url": "https://github.com/proximafusion/vmecpp/commit/9f5e1cd85ecf38d9bd4f1fbc284d8d6d71d4f4bf"
        },
        "date": 1776338312978,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.64910739331353,
            "unit": "iter/sec",
            "range": "stddev: 0.0057120213126872505",
            "extra": "mean: 606.3886 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.698979378200545,
            "unit": "iter/sec",
            "range": "stddev: 0.008969976679759645",
            "extra": "mean: 588.5887 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7111493853185062,
            "unit": "iter/sec",
            "range": "stddev: 0.007859316151338152",
            "extra": "mean: 1.4062 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2721834871122188,
            "unit": "iter/sec",
            "range": "stddev: 0.005327480092610018",
            "extra": "mean: 3.6740 sec\nrounds: 3"
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
          "id": "62b8d55d179ad0afb03222e235cb1f5b60e18bb7",
          "message": "Make sure Hdf5 writing doesn't raise an error",
          "timestamp": "2025-03-24T23:36:01+01:00",
          "tree_id": "44569510e653a2683cd5a0290ffd1a4fc411c7b7",
          "url": "https://github.com/proximafusion/vmecpp/commit/62b8d55d179ad0afb03222e235cb1f5b60e18bb7"
        },
        "date": 1776338312907,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8966628667506575,
            "unit": "iter/sec",
            "range": "stddev: 0.004222634921578921",
            "extra": "mean: 527.2418 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.901189878240926,
            "unit": "iter/sec",
            "range": "stddev: 0.0035892370893654576",
            "extra": "mean: 525.9864 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6591520924420299,
            "unit": "iter/sec",
            "range": "stddev: 0.0005346916482048143",
            "extra": "mean: 1.5171 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24287124930388868,
            "unit": "iter/sec",
            "range": "stddev: 0.025056659876389564",
            "extra": "mean: 4.1174 sec\nrounds: 3"
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
          "id": "e55b700adaad479ab3a9e080d95476eaabd28ae9",
          "message": "Support string path in FortranWOutAdapter.save",
          "timestamp": "2025-03-25T12:14:26+01:00",
          "tree_id": "187cee27660018966a7d2dbf6b73759fb20db32c",
          "url": "https://github.com/proximafusion/vmecpp/commit/e55b700adaad479ab3a9e080d95476eaabd28ae9"
        },
        "date": 1776338313068,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.9310704808955357,
            "unit": "iter/sec",
            "range": "stddev: 0.0036190619016997472",
            "extra": "mean: 517.8475 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.9232410140002796,
            "unit": "iter/sec",
            "range": "stddev: 0.0009458410199675161",
            "extra": "mean: 519.9556 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6582089236337819,
            "unit": "iter/sec",
            "range": "stddev: 0.003899655016237776",
            "extra": "mean: 1.5193 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24632325668905403,
            "unit": "iter/sec",
            "range": "stddev: 0.004293848873663388",
            "extra": "mean: 4.0597 sec\nrounds: 3"
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
          "id": "5c96b7f0eb632fbd780edc06b20c6cf9847a8706",
          "message": "Simplify bazel build, asan works with clang",
          "timestamp": "2025-03-27T14:49:55+01:00",
          "tree_id": "3a790185d7de1bed8acf8c3c05c18cb6fb2e01e2",
          "url": "https://github.com/proximafusion/vmecpp/commit/5c96b7f0eb632fbd780edc06b20c6cf9847a8706"
        },
        "date": 1776338312893,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8878148356421969,
            "unit": "iter/sec",
            "range": "stddev: 0.003946999973607131",
            "extra": "mean: 529.7130 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.908556450163632,
            "unit": "iter/sec",
            "range": "stddev: 0.003428586383786549",
            "extra": "mean: 523.9562 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6628937176556327,
            "unit": "iter/sec",
            "range": "stddev: 0.024089416753428603",
            "extra": "mean: 1.5085 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24449302990737545,
            "unit": "iter/sec",
            "range": "stddev: 0.025173928025691295",
            "extra": "mean: 4.0901 sec\nrounds: 3"
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
          "id": "4e349e188c24ffd87f18a7ab5a520dfb876914fe",
          "message": "Print version info with help message",
          "timestamp": "2025-03-24T22:23:03+01:00",
          "tree_id": "20c58d237c1ea69dcea62008f1bf1a484a8c21f1",
          "url": "https://github.com/proximafusion/vmecpp/commit/4e349e188c24ffd87f18a7ab5a520dfb876914fe"
        },
        "date": 1776338312869,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8969809621969673,
            "unit": "iter/sec",
            "range": "stddev: 0.005411477882135665",
            "extra": "mean: 527.1534 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.8882401458862441,
            "unit": "iter/sec",
            "range": "stddev: 0.00600482423169038",
            "extra": "mean: 529.5937 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6572775571077426,
            "unit": "iter/sec",
            "range": "stddev: 0.011254462103569354",
            "extra": "mean: 1.5214 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24413086181187094,
            "unit": "iter/sec",
            "range": "stddev: 0.009354895164671714",
            "extra": "mean: 4.0962 sec\nrounds: 3"
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
          "id": "6c4afb41a13868c9c7efba6221a315375414c154",
          "message": "Update README.md",
          "timestamp": "2025-03-29T03:01:19+01:00",
          "tree_id": "e2c1552503af520d754892ef2b18c650a1a0590c",
          "url": "https://github.com/proximafusion/vmecpp/commit/6c4afb41a13868c9c7efba6221a315375414c154"
        },
        "date": 1776338312915,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.855435733861843,
            "unit": "iter/sec",
            "range": "stddev: 0.004127111392294152",
            "extra": "mean: 538.9570 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.87075878848611,
            "unit": "iter/sec",
            "range": "stddev: 0.0049008160673843955",
            "extra": "mean: 534.5425 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6720447026071463,
            "unit": "iter/sec",
            "range": "stddev: 0.010088188844105967",
            "extra": "mean: 1.4880 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2444794690277307,
            "unit": "iter/sec",
            "range": "stddev: 0.014061624994901213",
            "extra": "mean: 4.0903 sec\nrounds: 3"
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
          "id": "2cff5b0ad8a89df584ffe84313cd22193430bf87",
          "message": "Save method for VmecInput",
          "timestamp": "2025-03-27T16:58:23+01:00",
          "tree_id": "240060c936b2d0480d930275d615ae2752e55803",
          "url": "https://github.com/proximafusion/vmecpp/commit/2cff5b0ad8a89df584ffe84313cd22193430bf87"
        },
        "date": 1776338312827,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.9047879034841209,
            "unit": "iter/sec",
            "range": "stddev: 0.0023325814258065986",
            "extra": "mean: 524.9928 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.901463416976686,
            "unit": "iter/sec",
            "range": "stddev: 0.0052529266465209485",
            "extra": "mean: 525.9107 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6616373135529033,
            "unit": "iter/sec",
            "range": "stddev: 0.011987026634683021",
            "extra": "mean: 1.5114 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24470531555324596,
            "unit": "iter/sec",
            "range": "stddev: 0.007564602175324047",
            "extra": "mean: 4.0865 sec\nrounds: 3"
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
          "id": "c2c4243658245e43862ee998ed9a09c3960609a6",
          "message": "Move from hatchling to scikit-build-core",
          "timestamp": "2025-03-29T02:48:00+01:00",
          "tree_id": "bf015a9872adb095797ad72e28737ef72ff54891",
          "url": "https://github.com/proximafusion/vmecpp/commit/c2c4243658245e43862ee998ed9a09c3960609a6"
        },
        "date": 1776338313013,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8442382441743974,
            "unit": "iter/sec",
            "range": "stddev: 0.0008846956064505964",
            "extra": "mean: 542.2293 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.8367998964503893,
            "unit": "iter/sec",
            "range": "stddev: 0.0029726578632073984",
            "extra": "mean: 544.4251 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6560192703963583,
            "unit": "iter/sec",
            "range": "stddev: 0.015181119679300406",
            "extra": "mean: 1.5243 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2422680586966513,
            "unit": "iter/sec",
            "range": "stddev: 0.012844924612796982",
            "extra": "mean: 4.1277 sec\nrounds: 3"
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
          "id": "59667f21c2cc586c96dc9520e603ec57643dcbb2",
          "message": "Add doctest to CI",
          "timestamp": "2025-03-31T18:02:46+02:00",
          "tree_id": "7e64bb77339cae3978406c8ceb9c22969ab3d950",
          "url": "https://github.com/proximafusion/vmecpp/commit/59667f21c2cc586c96dc9520e603ec57643dcbb2"
        },
        "date": 1776338312888,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8250684585004409,
            "unit": "iter/sec",
            "range": "stddev: 0.00477285762428544",
            "extra": "mean: 547.9247 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.8633230954560103,
            "unit": "iter/sec",
            "range": "stddev: 0.00708449153466442",
            "extra": "mean: 536.6756 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.656705052704888,
            "unit": "iter/sec",
            "range": "stddev: 0.007562182027195963",
            "extra": "mean: 1.5228 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24247087336455292,
            "unit": "iter/sec",
            "range": "stddev: 0.005155569799382448",
            "extra": "mean: 4.1242 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "c45929b9dac1a6d062e2213ca310e2619e49a109",
          "message": "Update tests.yaml",
          "timestamp": "2025-04-02T12:12:24+02:00",
          "tree_id": "2ec3638abb1db32a02f3c852b131381588c85862",
          "url": "https://github.com/proximafusion/vmecpp/commit/c45929b9dac1a6d062e2213ca310e2619e49a109"
        },
        "date": 1776338313016,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.846167087659291,
            "unit": "iter/sec",
            "range": "stddev: 0.008186987835800416",
            "extra": "mean: 541.6628 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.8629857245264367,
            "unit": "iter/sec",
            "range": "stddev: 0.005562324754762539",
            "extra": "mean: 536.7728 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.667629652116216,
            "unit": "iter/sec",
            "range": "stddev: 0.022838152675517226",
            "extra": "mean: 1.4978 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24386183931928213,
            "unit": "iter/sec",
            "range": "stddev: 0.007020213510431317",
            "extra": "mean: 4.1007 sec\nrounds: 3"
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
          "id": "279eedf184233af2ade3ed6006376c33844a4154",
          "message": "Only write rbs, zbc when lasym to be consistent with r_axis, z_axis",
          "timestamp": "2025-03-31T15:21:33+02:00",
          "tree_id": "106c42905a51b6af49d981b70f99887d90dcb12f",
          "url": "https://github.com/proximafusion/vmecpp/commit/279eedf184233af2ade3ed6006376c33844a4154"
        },
        "date": 1776338312819,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8696070865754824,
            "unit": "iter/sec",
            "range": "stddev: 0.0029167978725483097",
            "extra": "mean: 534.8717 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.8641852719219556,
            "unit": "iter/sec",
            "range": "stddev: 0.0029564170821466977",
            "extra": "mean: 536.4274 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6555970297017476,
            "unit": "iter/sec",
            "range": "stddev: 0.005899233460941311",
            "extra": "mean: 1.5253 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24347124011594892,
            "unit": "iter/sec",
            "range": "stddev: 0.012416434722500915",
            "extra": "mean: 4.1073 sec\nrounds: 3"
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
          "id": "64fa9a2086010042699a7af583ae1b34a0b4735f",
          "message": "Typo in ac_aux_f to_json()",
          "timestamp": "2025-03-31T15:27:28+02:00",
          "tree_id": "e1eb736730fe0f9fec9f1f03a579f1ee81e15ef9",
          "url": "https://github.com/proximafusion/vmecpp/commit/64fa9a2086010042699a7af583ae1b34a0b4735f"
        },
        "date": 1776338312908,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8806303489128988,
            "unit": "iter/sec",
            "range": "stddev: 0.002554317327193395",
            "extra": "mean: 531.7366 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.8868937469829603,
            "unit": "iter/sec",
            "range": "stddev: 0.002892240300651544",
            "extra": "mean: 529.9715 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6587447008278992,
            "unit": "iter/sec",
            "range": "stddev: 0.004978226711486665",
            "extra": "mean: 1.5180 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24393441428456114,
            "unit": "iter/sec",
            "range": "stddev: 0.016168054307464798",
            "extra": "mean: 4.0995 sec\nrounds: 3"
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
          "id": "ecac945862e908876e8991f5cb5ec5af50d4d314",
          "message": "simsopt_compat.Vmec should to use Python API instead of CPP API",
          "timestamp": "2025-03-27T00:46:02+01:00",
          "tree_id": "093fbb6f4e66cbc4ee387e778d7c63660d95fda5",
          "url": "https://github.com/proximafusion/vmecpp/commit/ecac945862e908876e8991f5cb5ec5af50d4d314"
        },
        "date": 1776338313075,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.9069393825427403,
            "unit": "iter/sec",
            "range": "stddev: 0.0059769557159950705",
            "extra": "mean: 524.4005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.9215844129428723,
            "unit": "iter/sec",
            "range": "stddev: 0.002576311880328633",
            "extra": "mean: 520.4039 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6611315759995517,
            "unit": "iter/sec",
            "range": "stddev: 0.005345927674084469",
            "extra": "mean: 1.5126 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24504200449556118,
            "unit": "iter/sec",
            "range": "stddev: 0.019686382573188332",
            "extra": "mean: 4.0809 sec\nrounds: 3"
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
          "id": "8048726e4c6a9c29b05d43803f894636fb698fdd",
          "message": "Update module lock",
          "timestamp": "2025-03-26T17:40:44+01:00",
          "tree_id": "5f2e3a3d271ab69568ba4b537afef4a632cb1857",
          "url": "https://github.com/proximafusion/vmecpp/commit/8048726e4c6a9c29b05d43803f894636fb698fdd"
        },
        "date": 1776338312934,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8789401294656898,
            "unit": "iter/sec",
            "range": "stddev: 0.004492048491790052",
            "extra": "mean: 532.2149 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.9040995972656531,
            "unit": "iter/sec",
            "range": "stddev: 0.002793299224165103",
            "extra": "mean: 525.1826 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6626943175859009,
            "unit": "iter/sec",
            "range": "stddev: 0.008493217921899925",
            "extra": "mean: 1.5090 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24470546339876473,
            "unit": "iter/sec",
            "range": "stddev: 0.00872202226373511",
            "extra": "mean: 4.0865 sec\nrounds: 3"
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
          "id": "c5adea504d70f3ae1af8a9c7179fedbdb9169c88",
          "message": "Terminate Fortran namelist files only with / not / and &END",
          "timestamp": "2025-04-02T18:25:09+02:00",
          "tree_id": "d7a7323e9566573d99c9843800a8aab800d4fc45",
          "url": "https://github.com/proximafusion/vmecpp/commit/c5adea504d70f3ae1af8a9c7179fedbdb9169c88"
        },
        "date": 1776338313019,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8775487278061211,
            "unit": "iter/sec",
            "range": "stddev: 0.005486869102996734",
            "extra": "mean: 532.6093 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.869270048711287,
            "unit": "iter/sec",
            "range": "stddev: 0.0037473262296862605",
            "extra": "mean: 534.9682 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6684799373627163,
            "unit": "iter/sec",
            "range": "stddev: 0.02381833794574117",
            "extra": "mean: 1.4959 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2433437640404459,
            "unit": "iter/sec",
            "range": "stddev: 0.00033749600410387605",
            "extra": "mean: 4.1094 sec\nrounds: 3"
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
          "id": "8d7d56470d71a307082a6e7edab7579f5c871ee2",
          "message": "Fully support and test  editable installs",
          "timestamp": "2025-04-03T13:47:51+02:00",
          "tree_id": "fa0a4efcb796e0e3d088535dcf450d2dec5678d6",
          "url": "https://github.com/proximafusion/vmecpp/commit/8d7d56470d71a307082a6e7edab7579f5c871ee2"
        },
        "date": 1776338312956,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8597300028954231,
            "unit": "iter/sec",
            "range": "stddev: 0.008541062984356428",
            "extra": "mean: 537.7125 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.8635668146959765,
            "unit": "iter/sec",
            "range": "stddev: 0.007133840939511654",
            "extra": "mean: 536.6054 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6738304816563289,
            "unit": "iter/sec",
            "range": "stddev: 0.019348620884080136",
            "extra": "mean: 1.4841 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24298983473336655,
            "unit": "iter/sec",
            "range": "stddev: 0.018929971671093525",
            "extra": "mean: 4.1154 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "5c906ae3194c0ec3422cdff9d23de7c5848e3be7",
          "message": "Update _util.py",
          "timestamp": "2025-04-03T17:53:34+02:00",
          "tree_id": "04e8a683a9b03cf98382025f2fcc69f3da5adc10",
          "url": "https://github.com/proximafusion/vmecpp/commit/5c906ae3194c0ec3422cdff9d23de7c5848e3be7"
        },
        "date": 1776338312892,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 1.8863944207599348,
            "unit": "iter/sec",
            "range": "stddev: 0.004117160034834169",
            "extra": "mean: 530.1118 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 1.8826819067488783,
            "unit": "iter/sec",
            "range": "stddev: 0.004422708026094172",
            "extra": "mean: 531.1572 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6698304104053207,
            "unit": "iter/sec",
            "range": "stddev: 0.024666172627315998",
            "extra": "mean: 1.4929 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24422147701624322,
            "unit": "iter/sec",
            "range": "stddev: 0.014001105537128548",
            "extra": "mean: 4.0946 sec\nrounds: 3"
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
          "id": "5d84d8f3f4a15de476ac3f917622f5fe31bffdca",
          "message": "Move from hatchling to scikit-build-core",
          "timestamp": "2025-03-29T02:48:00+01:00",
          "tree_id": "8a9c169dc240b0cd3a2b1f60cee9d58a73a99573",
          "url": "https://github.com/proximafusion/vmecpp/commit/5d84d8f3f4a15de476ac3f917622f5fe31bffdca"
        },
        "date": 1776338312897,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.5600630117767533,
            "unit": "iter/sec",
            "range": "stddev: 0.0022703398727172697",
            "extra": "mean: 280.8939 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.5591071737352893,
            "unit": "iter/sec",
            "range": "stddev: 0.0018194274673787169",
            "extra": "mean: 280.9693 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6619537650369808,
            "unit": "iter/sec",
            "range": "stddev: 0.00739343711057283",
            "extra": "mean: 1.5107 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24385620922023893,
            "unit": "iter/sec",
            "range": "stddev: 0.016252260718114277",
            "extra": "mean: 4.1008 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "521d5ecd35eec78571fa38a6e738773ac8c215e6",
          "message": "Apply suggestions from code review",
          "timestamp": "2025-04-03T23:06:33+02:00",
          "tree_id": "0eef28bdb1b16f224768b797452d7ab36eb6d617",
          "url": "https://github.com/proximafusion/vmecpp/commit/521d5ecd35eec78571fa38a6e738773ac8c215e6"
        },
        "date": 1776338312874,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.5656386085024647,
            "unit": "iter/sec",
            "range": "stddev: 0.000945837755039766",
            "extra": "mean: 280.4547 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.5659602121967353,
            "unit": "iter/sec",
            "range": "stddev: 0.0015495216678276594",
            "extra": "mean: 280.4294 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6596051077855496,
            "unit": "iter/sec",
            "range": "stddev: 0.003508078846895477",
            "extra": "mean: 1.5161 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24567174594674956,
            "unit": "iter/sec",
            "range": "stddev: 0.04465010669301351",
            "extra": "mean: 4.0705 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "032e006fe1406ff700d2a2dc5568e4525077abb4",
          "message": "Apply suggestions from code review",
          "timestamp": "2025-04-07T11:40:53+02:00",
          "tree_id": "c6794f2b61a65d044e4120b36ead1a28e0dd4162",
          "url": "https://github.com/proximafusion/vmecpp/commit/032e006fe1406ff700d2a2dc5568e4525077abb4"
        },
        "date": 1776338312760,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.5311464906851544,
            "unit": "iter/sec",
            "range": "stddev: 0.003282884344681607",
            "extra": "mean: 283.1941 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.582627080623051,
            "unit": "iter/sec",
            "range": "stddev: 0.0011909853007189288",
            "extra": "mean: 279.1248 msec\nrounds: 5"
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
          "id": "a2d41a27d7b4616b5e687324b25db0a5806e74a1",
          "message": "Made lasym specific terms std::optional and Optional in Python",
          "timestamp": "2025-04-07T13:21:23+02:00",
          "tree_id": "8578bb71a6aa8330e956bf855a6cc682682f3c22",
          "url": "https://github.com/proximafusion/vmecpp/commit/a2d41a27d7b4616b5e687324b25db0a5806e74a1"
        },
        "date": 1776338312981,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.5591779267663513,
            "unit": "iter/sec",
            "range": "stddev: 0.0026247182538981628",
            "extra": "mean: 280.9638 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.546935353753217,
            "unit": "iter/sec",
            "range": "stddev: 0.003213815780963389",
            "extra": "mean: 281.9335 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6589345487097921,
            "unit": "iter/sec",
            "range": "stddev: 0.005041147220655135",
            "extra": "mean: 1.5176 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24648744739208964,
            "unit": "iter/sec",
            "range": "stddev: 0.042833927009312836",
            "extra": "mean: 4.0570 sec\nrounds: 3"
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
          "id": "08f5d6c4c83f249a8bf696e56d0c7aa42edd7691",
          "message": "Use the pydantic model validator when converting netcdf to self.wout",
          "timestamp": "2025-03-12T17:20:33+01:00",
          "tree_id": "a6751c960092627c387f141972ad46a83c70cab5",
          "url": "https://github.com/proximafusion/vmecpp/commit/08f5d6c4c83f249a8bf696e56d0c7aa42edd7691"
        },
        "date": 1776338312777,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.510411407735665,
            "unit": "iter/sec",
            "range": "stddev: 0.0016655750122326503",
            "extra": "mean: 284.8669 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.5175272229348566,
            "unit": "iter/sec",
            "range": "stddev: 0.004202337818759986",
            "extra": "mean: 284.2906 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6578528010343038,
            "unit": "iter/sec",
            "range": "stddev: 0.0031432088942601544",
            "extra": "mean: 1.5201 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24354179749941904,
            "unit": "iter/sec",
            "range": "stddev: 0.03618465092137612",
            "extra": "mean: 4.1061 sec\nrounds: 3"
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
          "id": "07cebffca5708cb87d623cf65fb58a1c6544fa60",
          "message": "Made lasym specific terms std::optional and Optional in Python",
          "timestamp": "2025-04-07T13:21:16+02:00",
          "tree_id": "ddc8ece605c97d2753b826a982b5c815c8661850",
          "url": "https://github.com/proximafusion/vmecpp/commit/07cebffca5708cb87d623cf65fb58a1c6544fa60"
        },
        "date": 1776338312774,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.531243823853205,
            "unit": "iter/sec",
            "range": "stddev: 0.001051432865406795",
            "extra": "mean: 283.1863 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.5055037621983227,
            "unit": "iter/sec",
            "range": "stddev: 0.0025501012878478614",
            "extra": "mean: 285.2657 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6629339608716103,
            "unit": "iter/sec",
            "range": "stddev: 0.015687235412151238",
            "extra": "mean: 1.5084 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2440371165815434,
            "unit": "iter/sec",
            "range": "stddev: 0.02988368273735427",
            "extra": "mean: 4.0977 sec\nrounds: 3"
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
          "id": "3a93b557bbe11e735b6fcfdc54804fd6c8ad5906",
          "message": "Remove FortranWoutAdapter, redundant with VmecWOut class",
          "timestamp": "2025-04-01T19:31:08+02:00",
          "tree_id": "1b338c0b8aff0cbfd5508b46078eefa1a6b8b63f",
          "url": "https://github.com/proximafusion/vmecpp/commit/3a93b557bbe11e735b6fcfdc54804fd6c8ad5906"
        },
        "date": 1776338312846,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.499588405709046,
            "unit": "iter/sec",
            "range": "stddev: 0.0018431896149836227",
            "extra": "mean: 285.7479 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.477624987953677,
            "unit": "iter/sec",
            "range": "stddev: 0.0031858564468892382",
            "extra": "mean: 287.5526 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6616127745842818,
            "unit": "iter/sec",
            "range": "stddev: 0.01190935193921059",
            "extra": "mean: 1.5115 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24182991366574055,
            "unit": "iter/sec",
            "range": "stddev: 0.09433422166000133",
            "extra": "mean: 4.1351 sec\nrounds: 3"
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
          "id": "55c573383c72b85725f7befb571f7a3109d56a94",
          "message": "from_wout_file Iterate through present keys instead of expected keys",
          "timestamp": "2025-04-03T16:55:44+02:00",
          "tree_id": "c96be451bdb1e0ea0ba0fa5164eadfa8b5e11c91",
          "url": "https://github.com/proximafusion/vmecpp/commit/55c573383c72b85725f7befb571f7a3109d56a94"
        },
        "date": 1776338312882,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.544384906992734,
            "unit": "iter/sec",
            "range": "stddev: 0.0012347623554989939",
            "extra": "mean: 282.1364 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.456813165463366,
            "unit": "iter/sec",
            "range": "stddev: 0.0058237556800705935",
            "extra": "mean: 289.2838 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.662640875666543,
            "unit": "iter/sec",
            "range": "stddev: 0.028607207424255692",
            "extra": "mean: 1.5091 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24450095412255085,
            "unit": "iter/sec",
            "range": "stddev: 0.029758912379669603",
            "extra": "mean: 4.0900 sec\nrounds: 3"
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
          "id": "1635288511b0144ca159f60ba3aaf1de6d0dadd8",
          "message": "Re-generate reference data with  added to wout files.",
          "timestamp": "2025-04-03T14:51:30+02:00",
          "tree_id": "8ae7dded6e3c4f2b2a4df74d11026ff79a2afa04",
          "url": "https://github.com/proximafusion/vmecpp/commit/1635288511b0144ca159f60ba3aaf1de6d0dadd8"
        },
        "date": 1776338312802,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.559085669950738,
            "unit": "iter/sec",
            "range": "stddev: 0.0008928990096502742",
            "extra": "mean: 280.9710 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.501904523031654,
            "unit": "iter/sec",
            "range": "stddev: 0.008165424470578424",
            "extra": "mean: 285.5589 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6629114307425781,
            "unit": "iter/sec",
            "range": "stddev: 0.003692009314288054",
            "extra": "mean: 1.5085 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2444617107551637,
            "unit": "iter/sec",
            "range": "stddev: 0.021625052918187934",
            "extra": "mean: 4.0906 sec\nrounds: 3"
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
          "id": "d2d6685fb34e4f811aa4e799c2bfeb542e69800d",
          "message": "GCC already includes OpenMP headers on manylinux",
          "timestamp": "2025-04-07T16:21:55+02:00",
          "tree_id": "231167129bf993b647e333c2840ab6a675d94619",
          "url": "https://github.com/proximafusion/vmecpp/commit/d2d6685fb34e4f811aa4e799c2bfeb542e69800d"
        },
        "date": 1776338313044,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.3170773962677838,
            "unit": "iter/sec",
            "range": "stddev: 0.0012016293221431464",
            "extra": "mean: 301.4702 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.337644116544221,
            "unit": "iter/sec",
            "range": "stddev: 0.0009500762218896852",
            "extra": "mean: 299.6125 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6947010565620426,
            "unit": "iter/sec",
            "range": "stddev: 0.006766238678709327",
            "extra": "mean: 1.4395 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2747933154123227,
            "unit": "iter/sec",
            "range": "stddev: 0.025632103568942393",
            "extra": "mean: 3.6391 sec\nrounds: 3"
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
          "id": "bbc6e4a006a1ad8fe74865149c21bda482803ed9",
          "message": "Remove HDF5 unit test (deprecated)",
          "timestamp": "2025-04-07T21:55:09+02:00",
          "tree_id": "7eeadec5ba875b0fd6c83406610860d7b079995b",
          "url": "https://github.com/proximafusion/vmecpp/commit/bbc6e4a006a1ad8fe74865149c21bda482803ed9"
        },
        "date": 1776338313003,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2989934049964136,
            "unit": "iter/sec",
            "range": "stddev: 0.001197162326080925",
            "extra": "mean: 303.1228 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.333251637557831,
            "unit": "iter/sec",
            "range": "stddev: 0.0006560250197653883",
            "extra": "mean: 300.0074 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6964343823651379,
            "unit": "iter/sec",
            "range": "stddev: 0.0010271554097292003",
            "extra": "mean: 1.4359 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27166293820829696,
            "unit": "iter/sec",
            "range": "stddev: 0.04184882410078748",
            "extra": "mean: 3.6810 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "bc42de8fa846810b61af414d742dc902d1366f5b",
          "message": "Update README.md",
          "timestamp": "2025-04-07T22:32:09+02:00",
          "tree_id": "594c0d206c1023e0c7b3ed877b959087b76139a3",
          "url": "https://github.com/proximafusion/vmecpp/commit/bc42de8fa846810b61af414d742dc902d1366f5b"
        },
        "date": 1776338313006,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.3227761005628067,
            "unit": "iter/sec",
            "range": "stddev: 0.0009210399910594976",
            "extra": "mean: 300.9532 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.3370054074540403,
            "unit": "iter/sec",
            "range": "stddev: 0.0009481081181860942",
            "extra": "mean: 299.6699 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6937551007910039,
            "unit": "iter/sec",
            "range": "stddev: 0.0030545304891087767",
            "extra": "mean: 1.4414 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27550676485197234,
            "unit": "iter/sec",
            "range": "stddev: 0.008734981065615674",
            "extra": "mean: 3.6297 sec\nrounds: 3"
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
          "id": "5ffcefe2b42df0ed9ff60a86453e5ab01930772d",
          "message": "Remove mgrid test files from wheels",
          "timestamp": "2025-04-07T17:19:01+02:00",
          "tree_id": "39df3189953669182ca4a013f08f90f8939e4fba",
          "url": "https://github.com/proximafusion/vmecpp/commit/5ffcefe2b42df0ed9ff60a86453e5ab01930772d"
        },
        "date": 1776338312901,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.337350526636081,
            "unit": "iter/sec",
            "range": "stddev: 0.00171094998180755",
            "extra": "mean: 299.6389 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.3465492803374444,
            "unit": "iter/sec",
            "range": "stddev: 0.0008001704615064292",
            "extra": "mean: 298.8153 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6916597572585191,
            "unit": "iter/sec",
            "range": "stddev: 0.010733575467015775",
            "extra": "mean: 1.4458 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2767491844427085,
            "unit": "iter/sec",
            "range": "stddev: 0.01411226503869706",
            "extra": "mean: 3.6134 sec\nrounds: 3"
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
          "id": "31b2a06e8f5043b6c0e9fb8f4ae157773d469e58",
          "message": "Keep isvmec and ensure_vmec methods around in simsopt_compat for backwards compatibility",
          "timestamp": "2025-04-08T02:02:40+02:00",
          "tree_id": "0aa344364d973141639608df523c4978e6b448e1",
          "url": "https://github.com/proximafusion/vmecpp/commit/31b2a06e8f5043b6c0e9fb8f4ae157773d469e58"
        },
        "date": 1776338312834,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2239148954660353,
            "unit": "iter/sec",
            "range": "stddev: 0.0036334266809676034",
            "extra": "mean: 310.1819 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.3469346151473136,
            "unit": "iter/sec",
            "range": "stddev: 0.0006096723044175566",
            "extra": "mean: 298.7809 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6821794386221948,
            "unit": "iter/sec",
            "range": "stddev: 0.030503396990531156",
            "extra": "mean: 1.4659 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27381512445993433,
            "unit": "iter/sec",
            "range": "stddev: 0.052738772423561446",
            "extra": "mean: 3.6521 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "8c59c89d651f255b7920e5fea65b7ddf099742af",
          "message": "Update simsopt_compat.py",
          "timestamp": "2025-04-08T11:39:03+02:00",
          "tree_id": "83a8b0f87977594a449f42e8893815787f66f6d8",
          "url": "https://github.com/proximafusion/vmecpp/commit/8c59c89d651f255b7920e5fea65b7ddf099742af"
        },
        "date": 1776338312955,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2934747164680984,
            "unit": "iter/sec",
            "range": "stddev: 0.007677391544491617",
            "extra": "mean: 303.6307 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.336841247758459,
            "unit": "iter/sec",
            "range": "stddev: 0.0011322099238506482",
            "extra": "mean: 299.6846 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6950666146607324,
            "unit": "iter/sec",
            "range": "stddev: 0.00699262257661283",
            "extra": "mean: 1.4387 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27437653006992613,
            "unit": "iter/sec",
            "range": "stddev: 0.03870042398581384",
            "extra": "mean: 3.6446 sec\nrounds: 3"
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
          "id": "71d3936da50d439c2950e65e802c7943248e4056",
          "message": "clang-tidy: Change enum to Google style",
          "timestamp": "2025-04-07T21:00:21+02:00",
          "tree_id": "bfe4ea6c249f4f8d1b59a856b75c36c89674342a",
          "url": "https://github.com/proximafusion/vmecpp/commit/71d3936da50d439c2950e65e802c7943248e4056"
        },
        "date": 1776338312920,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.3300899367378967,
            "unit": "iter/sec",
            "range": "stddev: 0.0035782763558442152",
            "extra": "mean: 300.2922 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.3448395926818626,
            "unit": "iter/sec",
            "range": "stddev: 0.0012257458814201008",
            "extra": "mean: 298.9680 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6876262019994474,
            "unit": "iter/sec",
            "range": "stddev: 0.019501913119400506",
            "extra": "mean: 1.4543 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2768608211469524,
            "unit": "iter/sec",
            "range": "stddev: 0.018348699981496575",
            "extra": "mean: 3.6119 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "cfd098009c14f160511f26a50e1163b232e8b875",
          "message": "Clang-format pass over the codebase (#224)",
          "timestamp": "2025-04-09T12:15:19Z",
          "tree_id": "9a21eb5d204c4aaea1bf820650ba69d0ff4267ce",
          "url": "https://github.com/proximafusion/vmecpp/commit/cfd098009c14f160511f26a50e1163b232e8b875"
        },
        "date": 1776338313041,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.296183951408838,
            "unit": "iter/sec",
            "range": "stddev: 0.0023015026682928655",
            "extra": "mean: 303.3811 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2190856839813797,
            "unit": "iter/sec",
            "range": "stddev: 0.00657025490325801",
            "extra": "mean: 310.6472 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6936748502538185,
            "unit": "iter/sec",
            "range": "stddev: 0.0031391053493654503",
            "extra": "mean: 1.4416 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2748498006878789,
            "unit": "iter/sec",
            "range": "stddev: 0.026208278004965238",
            "extra": "mean: 3.6384 sec\nrounds: 3"
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
          "id": "b6669bac6f8c4015ab7f9feac2463ecc7709f1ed",
          "message": "No dot after output file name in CLI",
          "timestamp": "2025-04-08T15:44:55+02:00",
          "tree_id": "32d06c7c89d1535e528d367c4f584285218dbf83",
          "url": "https://github.com/proximafusion/vmecpp/commit/b6669bac6f8c4015ab7f9feac2463ecc7709f1ed"
        },
        "date": 1776338313001,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.274910735741499,
            "unit": "iter/sec",
            "range": "stddev: 0.0009440643555916851",
            "extra": "mean: 305.3518 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.1731918456511816,
            "unit": "iter/sec",
            "range": "stddev: 0.004412801253711438",
            "extra": "mean: 315.1401 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6950211337253718,
            "unit": "iter/sec",
            "range": "stddev: 0.004278314302943655",
            "extra": "mean: 1.4388 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2716278161747268,
            "unit": "iter/sec",
            "range": "stddev: 0.0270270345017149",
            "extra": "mean: 3.6815 sec\nrounds: 3"
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
          "id": "0c095ca02c0a0b662bf643f056d8a8c1330470c7",
          "message": "Clang-format as part of pre-commit hooks (#221)",
          "timestamp": "2025-04-09T20:16:46+02:00",
          "tree_id": "631b1815112e74786fad547bb9c5314396d55b8b",
          "url": "https://github.com/proximafusion/vmecpp/commit/0c095ca02c0a0b662bf643f056d8a8c1330470c7"
        },
        "date": 1776338312784,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.3088586106124307,
            "unit": "iter/sec",
            "range": "stddev: 0.002065770773842302",
            "extra": "mean: 302.2190 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.3113918741678234,
            "unit": "iter/sec",
            "range": "stddev: 0.0037531663907920104",
            "extra": "mean: 301.9878 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6964557863462943,
            "unit": "iter/sec",
            "range": "stddev: 0.0012147258490921512",
            "extra": "mean: 1.4358 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27564619477875246,
            "unit": "iter/sec",
            "range": "stddev: 0.011891886821963597",
            "extra": "mean: 3.6278 sec\nrounds: 3"
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
          "id": "8b5db231ffb668965993182bb4d2f6a4de142336",
          "message": "isvmec2000 supports fortran comments and blank lines (#233)",
          "timestamp": "2025-04-10T14:23:27+02:00",
          "tree_id": "4788a366300e930c0b5bfde33f98fbd6cce3f520",
          "url": "https://github.com/proximafusion/vmecpp/commit/8b5db231ffb668965993182bb4d2f6a4de142336"
        },
        "date": 1776338312952,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.321387403329687,
            "unit": "iter/sec",
            "range": "stddev: 0.0015408166579776277",
            "extra": "mean: 301.0790 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.3245905290682454,
            "unit": "iter/sec",
            "range": "stddev: 0.001117424273699094",
            "extra": "mean: 300.7889 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6934752844816956,
            "unit": "iter/sec",
            "range": "stddev: 0.011154481941124329",
            "extra": "mean: 1.4420 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27608246235838446,
            "unit": "iter/sec",
            "range": "stddev: 0.011104438171254147",
            "extra": "mean: 3.6221 sec\nrounds: 3"
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
          "id": "10e68b1fdce8074556a63b7ceeff3fd2803baa97",
          "message": "[WOut] VMEC2000 wout files (#229)",
          "timestamp": "2025-04-11T13:30:39+02:00",
          "tree_id": "bb01666d8510e7c603c4a00880349fa5dc0f2baf",
          "url": "https://github.com/proximafusion/vmecpp/commit/10e68b1fdce8074556a63b7ceeff3fd2803baa97"
        },
        "date": 1776338312790,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.3358283817390104,
            "unit": "iter/sec",
            "range": "stddev: 0.0006457677765066738",
            "extra": "mean: 299.7756 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.326461336569493,
            "unit": "iter/sec",
            "range": "stddev: 0.0019051833031622396",
            "extra": "mean: 300.6198 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6918186470897125,
            "unit": "iter/sec",
            "range": "stddev: 0.01187432473809052",
            "extra": "mean: 1.4455 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2751130860478966,
            "unit": "iter/sec",
            "range": "stddev: 0.01354174322696292",
            "extra": "mean: 3.6349 sec\nrounds: 3"
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
          "id": "ddfb9ed281b22f821261ec0e612671fda6a6b07b",
          "message": "[WOut] Test failed for scalar comparison nan==nan, although it is a valid value (#230)",
          "timestamp": "2025-04-11T13:48:29+02:00",
          "tree_id": "3745b7cdb7df81579a44c6af016392a9dfc4bb7d",
          "url": "https://github.com/proximafusion/vmecpp/commit/ddfb9ed281b22f821261ec0e612671fda6a6b07b"
        },
        "date": 1776338313063,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.318252688939761,
            "unit": "iter/sec",
            "range": "stddev: 0.002216873550229344",
            "extra": "mean: 301.3634 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2991354106278887,
            "unit": "iter/sec",
            "range": "stddev: 0.0027502289734397324",
            "extra": "mean: 303.1097 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6960359541771329,
            "unit": "iter/sec",
            "range": "stddev: 0.0035509863021573828",
            "extra": "mean: 1.4367 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2710359923900256,
            "unit": "iter/sec",
            "range": "stddev: 0.031814449659428444",
            "extra": "mean: 3.6895 sec\nrounds: 3"
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
          "id": "feb0901436af8780c67ddc5edfa89d079b5ba347",
          "message": "Rename loadFromMgrid to LoadFile (#240)",
          "timestamp": "2025-04-14T08:09:38Z",
          "tree_id": "e087985768d96ab53f2a0133d2fc0362ed52b3a7",
          "url": "https://github.com/proximafusion/vmecpp/commit/feb0901436af8780c67ddc5edfa89d079b5ba347"
        },
        "date": 1776338313096,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.327093360863566,
            "unit": "iter/sec",
            "range": "stddev: 0.0009979090473234884",
            "extra": "mean: 300.5627 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.327063046067944,
            "unit": "iter/sec",
            "range": "stddev: 0.0007165937267779317",
            "extra": "mean: 300.5654 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6935464326568291,
            "unit": "iter/sec",
            "range": "stddev: 0.0049359330882443415",
            "extra": "mean: 1.4419 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2756193001405115,
            "unit": "iter/sec",
            "range": "stddev: 0.011886902962851791",
            "extra": "mean: 3.6282 sec\nrounds: 3"
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
          "id": "1603f7782eeca94b74befc34e69fafbf04dbe48d",
          "message": "mgrid load, string to filesystem (#241)",
          "timestamp": "2025-04-14T08:09:38Z",
          "tree_id": "fb834827fe51133d46de575963b3f73b528112b5",
          "url": "https://github.com/proximafusion/vmecpp/commit/1603f7782eeca94b74befc34e69fafbf04dbe48d"
        },
        "date": 1776338312800,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.3206966678951346,
            "unit": "iter/sec",
            "range": "stddev: 0.0006469805793274777",
            "extra": "mean: 301.1416 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.3366408363978204,
            "unit": "iter/sec",
            "range": "stddev: 0.0011777897299494423",
            "extra": "mean: 299.7026 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6963356121786638,
            "unit": "iter/sec",
            "range": "stddev: 0.004423716294299636",
            "extra": "mean: 1.4361 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2755039430240046,
            "unit": "iter/sec",
            "range": "stddev: 0.04881982631028344",
            "extra": "mean: 3.6297 sec\nrounds: 3"
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
          "id": "5a88211c79a035f3f896523eff0f0529ebdaff58",
          "message": "[WOut] Migrate output quantities bdotb, nextcur added (#210)",
          "timestamp": "2025-04-14T08:09:38Z",
          "tree_id": "617a8887d28cc6c3274827efe4eacad9485387e3",
          "url": "https://github.com/proximafusion/vmecpp/commit/5a88211c79a035f3f896523eff0f0529ebdaff58"
        },
        "date": 1776338312889,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2905363997422508,
            "unit": "iter/sec",
            "range": "stddev: 0.0009363478629673582",
            "extra": "mean: 303.9018 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2875591525088295,
            "unit": "iter/sec",
            "range": "stddev: 0.0035259219938787033",
            "extra": "mean: 304.1770 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6950036794769513,
            "unit": "iter/sec",
            "range": "stddev: 0.0034001503311554345",
            "extra": "mean: 1.4388 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27660153999904474,
            "unit": "iter/sec",
            "range": "stddev: 0.002471101179796045",
            "extra": "mean: 3.6153 sec\nrounds: 3"
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
          "id": "01b9ea1c5f2a237830541963cceac7da2fcfb4b1",
          "message": "[WOut] Tracking down subtle compatibility issues (#234)",
          "timestamp": "2025-04-14T08:09:39Z",
          "tree_id": "e7617f10e960281f424c506067d913142a1934c6",
          "url": "https://github.com/proximafusion/vmecpp/commit/01b9ea1c5f2a237830541963cceac7da2fcfb4b1"
        },
        "date": 1776338312755,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2777510018294915,
            "unit": "iter/sec",
            "range": "stddev: 0.0023813725129543406",
            "extra": "mean: 305.0872 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.316181408592299,
            "unit": "iter/sec",
            "range": "stddev: 0.0005035968040642293",
            "extra": "mean: 301.5517 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6973154119271527,
            "unit": "iter/sec",
            "range": "stddev: 0.002217407391649254",
            "extra": "mean: 1.4341 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.276701826006781,
            "unit": "iter/sec",
            "range": "stddev: 0.010459726137727403",
            "extra": "mean: 3.6140 sec\nrounds: 3"
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
          "id": "869a6b4b45cd7e5ded675ade129a87d932b72c56",
          "message": "Move from int return values to absl::status (#219)",
          "timestamp": "2025-04-14T09:30:08Z",
          "tree_id": "a0c75b5beb56836ef1e46f7a8d64bc82ea1d03d0",
          "url": "https://github.com/proximafusion/vmecpp/commit/869a6b4b45cd7e5ded675ade129a87d932b72c56"
        },
        "date": 1776338312946,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.312835440117413,
            "unit": "iter/sec",
            "range": "stddev: 0.0016068308713461869",
            "extra": "mean: 301.8562 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.31615563168561,
            "unit": "iter/sec",
            "range": "stddev: 0.001445863842909379",
            "extra": "mean: 301.5540 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6862531878724883,
            "unit": "iter/sec",
            "range": "stddev: 0.01932765022935081",
            "extra": "mean: 1.4572 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2764658566175658,
            "unit": "iter/sec",
            "range": "stddev: 0.028100553347341916",
            "extra": "mean: 3.6171 sec\nrounds: 3"
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
          "id": "40b8df7476e1f6a5ad9eb42f702884f73efac0c4",
          "message": "Update license format in pyproject.toml (PEP 621 is deprecated) (#242)",
          "timestamp": "2025-04-14T10:01:39Z",
          "tree_id": "0c436ff58f6378d3c5d659da2af0414c8c533a55",
          "url": "https://github.com/proximafusion/vmecpp/commit/40b8df7476e1f6a5ad9eb42f702884f73efac0c4"
        },
        "date": 1776338312856,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.3063057212034264,
            "unit": "iter/sec",
            "range": "stddev: 0.0014767957056156403",
            "extra": "mean: 302.4524 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.31412517419885,
            "unit": "iter/sec",
            "range": "stddev: 0.000895144744917941",
            "extra": "mean: 301.7388 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6937180402407721,
            "unit": "iter/sec",
            "range": "stddev: 0.0072395114150489385",
            "extra": "mean: 1.4415 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2747692821710642,
            "unit": "iter/sec",
            "range": "stddev: 0.04297257238512977",
            "extra": "mean: 3.6394 sec\nrounds: 3"
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
          "id": "4185f511c668e39dda1b62d0997e752cbbefcc70",
          "message": "Add LFS to PyPi workflow to get mgrid for tests",
          "timestamp": "2025-04-14T13:56:36+02:00",
          "tree_id": "0df9de0c0e2afb9aeedf01c2713f552f70280c47",
          "url": "https://github.com/proximafusion/vmecpp/commit/4185f511c668e39dda1b62d0997e752cbbefcc70"
        },
        "date": 1776338312857,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.303657570675611,
            "unit": "iter/sec",
            "range": "stddev: 0.003072736559505037",
            "extra": "mean: 302.6948 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.316269277422924,
            "unit": "iter/sec",
            "range": "stddev: 0.0006594135435681188",
            "extra": "mean: 301.5437 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6928430804025947,
            "unit": "iter/sec",
            "range": "stddev: 0.007706913853231728",
            "extra": "mean: 1.4433 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27503985614473303,
            "unit": "iter/sec",
            "range": "stddev: 0.04681156538146258",
            "extra": "mean: 3.6358 sec\nrounds: 3"
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
          "id": "d4a674dbc4dd9680dfbd2b491cf64f703e654b96",
          "message": "Git-lfs install before checkout",
          "timestamp": "2025-04-14T15:41:21+02:00",
          "tree_id": "90711ef6142c3dd151d36362f74346dc6d3254e8",
          "url": "https://github.com/proximafusion/vmecpp/commit/d4a674dbc4dd9680dfbd2b491cf64f703e654b96"
        },
        "date": 1776338313046,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.3071581532635306,
            "unit": "iter/sec",
            "range": "stddev: 0.0012271783036415722",
            "extra": "mean: 302.3744 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2857414450897613,
            "unit": "iter/sec",
            "range": "stddev: 0.0016257166719416392",
            "extra": "mean: 304.3453 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6963980177240946,
            "unit": "iter/sec",
            "range": "stddev: 0.003662265598864011",
            "extra": "mean: 1.4360 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27527650599639336,
            "unit": "iter/sec",
            "range": "stddev: 0.047901996073974135",
            "extra": "mean: 3.6327 sec\nrounds: 3"
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
          "id": "f8dd1a8c5cc7d9f622bb9e148bc416248f056673",
          "message": "create venv after checkout",
          "timestamp": "2025-04-14T15:52:40+02:00",
          "tree_id": "cd1b6b35c4d8b296a209b7d77003deec77e36379",
          "url": "https://github.com/proximafusion/vmecpp/commit/f8dd1a8c5cc7d9f622bb9e148bc416248f056673"
        },
        "date": 1776338313087,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2890151608758726,
            "unit": "iter/sec",
            "range": "stddev: 0.004596203456852025",
            "extra": "mean: 304.0424 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.301582928153602,
            "unit": "iter/sec",
            "range": "stddev: 0.0010274011806848569",
            "extra": "mean: 302.8850 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.68882430003339,
            "unit": "iter/sec",
            "range": "stddev: 0.026131458952037764",
            "extra": "mean: 1.4517 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.276180914739701,
            "unit": "iter/sec",
            "range": "stddev: 0.04921615192376511",
            "extra": "mean: 3.6208 sec\nrounds: 3"
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
          "id": "e3df5f049e5c4ca37ed6f77f1b9854a2f2608389",
          "message": "Explicit default values in aux arrays for wout, robust when passing empty arrays",
          "timestamp": "2025-04-16T09:42:06+02:00",
          "tree_id": "6928e793240bc06f186a6d33c596e7aaa87fabf7",
          "url": "https://github.com/proximafusion/vmecpp/commit/e3df5f049e5c4ca37ed6f77f1b9854a2f2608389"
        },
        "date": 1776338313067,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2488447454082516,
            "unit": "iter/sec",
            "range": "stddev: 0.010900274946402397",
            "extra": "mean: 307.8017 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.301685443526521,
            "unit": "iter/sec",
            "range": "stddev: 0.0013719664722869687",
            "extra": "mean: 302.8756 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6949200601198458,
            "unit": "iter/sec",
            "range": "stddev: 0.007159561195792852",
            "extra": "mean: 1.4390 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2771753256328699,
            "unit": "iter/sec",
            "range": "stddev: 0.023208583521025086",
            "extra": "mean: 3.6078 sec\nrounds: 3"
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
          "id": "bc6dd26e1cba306bac0483b68c6ea05e7e7de044",
          "message": "Build docker on release (#255)",
          "timestamp": "2025-04-17T07:24:35Z",
          "tree_id": "845ac0a83f081388faa6871022d4830dc9b07fc0",
          "url": "https://github.com/proximafusion/vmecpp/commit/bc6dd26e1cba306bac0483b68c6ea05e7e7de044"
        },
        "date": 1776338313008,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.29253885592989,
            "unit": "iter/sec",
            "range": "stddev: 0.0013807384867741584",
            "extra": "mean: 303.7170 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2844341651108646,
            "unit": "iter/sec",
            "range": "stddev: 0.0015610066654310143",
            "extra": "mean: 304.4664 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6960174656673909,
            "unit": "iter/sec",
            "range": "stddev: 0.001676084768344129",
            "extra": "mean: 1.4367 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2746366108858803,
            "unit": "iter/sec",
            "range": "stddev: 0.04093386703288875",
            "extra": "mean: 3.6412 sec\nrounds: 3"
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
          "id": "7d2374b6672db622472f96dad245fb33e7a99010",
          "message": "Zero padded strings are stripped (#256)",
          "timestamp": "2025-04-17T07:41:09Z",
          "tree_id": "32a94ff456bb4bee04dea3f940af70e698f236b9",
          "url": "https://github.com/proximafusion/vmecpp/commit/7d2374b6672db622472f96dad245fb33e7a99010"
        },
        "date": 1776338312927,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2561434591532405,
            "unit": "iter/sec",
            "range": "stddev: 0.004660498458397488",
            "extra": "mean: 307.1118 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2952034272865594,
            "unit": "iter/sec",
            "range": "stddev: 0.0015619396844207792",
            "extra": "mean: 303.4714 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6975115642534951,
            "unit": "iter/sec",
            "range": "stddev: 0.0012785175302321757",
            "extra": "mean: 1.4337 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2745066888196666,
            "unit": "iter/sec",
            "range": "stddev: 0.009923128000061465",
            "extra": "mean: 3.6429 sec\nrounds: 3"
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
          "id": "26834124aeb5d854c3b3306dc80e965794e08f0f",
          "message": "Wout test variable contents in both directions (#254)",
          "timestamp": "2025-04-17T07:56:53Z",
          "tree_id": "81888bc93cdfeb751bdaa6e4b48ea6f7735f6177",
          "url": "https://github.com/proximafusion/vmecpp/commit/26834124aeb5d854c3b3306dc80e965794e08f0f"
        },
        "date": 1776338312817,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.1200599038524577,
            "unit": "iter/sec",
            "range": "stddev: 0.003196923093019708",
            "extra": "mean: 320.5067 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.1108854983573044,
            "unit": "iter/sec",
            "range": "stddev: 0.005293032851851149",
            "extra": "mean: 321.4519 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7143262210044525,
            "unit": "iter/sec",
            "range": "stddev: 0.002028905115120908",
            "extra": "mean: 1.3999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27047855725424086,
            "unit": "iter/sec",
            "range": "stddev: 0.03934338213660061",
            "extra": "mean: 3.6972 sec\nrounds: 3"
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
          "id": "869310a1bdf1ce7c030e1fbd8b05fac2d5867d4f",
          "message": "Add comment for building with debug info",
          "timestamp": "2025-04-10T13:53:47+02:00",
          "tree_id": "30391dcbf1d6316038bfa19d66dedfc3c380c971",
          "url": "https://github.com/proximafusion/vmecpp/commit/869310a1bdf1ce7c030e1fbd8b05fac2d5867d4f"
        },
        "date": 1776338312945,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.215268591349075,
            "unit": "iter/sec",
            "range": "stddev: 0.0015592900469100176",
            "extra": "mean: 311.0160 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.1812738696189546,
            "unit": "iter/sec",
            "range": "stddev: 0.001334717698960064",
            "extra": "mean: 314.3395 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7105958224767517,
            "unit": "iter/sec",
            "range": "stddev: 0.007519885952452309",
            "extra": "mean: 1.4073 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2702169952612394,
            "unit": "iter/sec",
            "range": "stddev: 0.03899805708044073",
            "extra": "mean: 3.7007 sec\nrounds: 3"
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
            "name": "Philipp Jurasic",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "fa7e02096b8718247c0f820ee1029ca9f2634200",
          "message": "[WOut] Add input_extension (#237)",
          "timestamp": "2025-04-17T08:12:59Z",
          "tree_id": "6440afd70faa0ce821fb38e440119373197ead24",
          "url": "https://github.com/proximafusion/vmecpp/commit/fa7e02096b8718247c0f820ee1029ca9f2634200"
        },
        "date": 1776338313089,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.205433395263809,
            "unit": "iter/sec",
            "range": "stddev: 0.0015741740889657623",
            "extra": "mean: 311.9703 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2339903939633268,
            "unit": "iter/sec",
            "range": "stddev: 0.0017016163919098967",
            "extra": "mean: 309.2155 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7093951872728771,
            "unit": "iter/sec",
            "range": "stddev: 0.004412234709937217",
            "extra": "mean: 1.4097 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2716280803871422,
            "unit": "iter/sec",
            "range": "stddev: 0.04456899264569739",
            "extra": "mean: 3.6815 sec\nrounds: 3"
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
          "id": "1cc5f6769e64be435c863a6dc71f4b4d2d360c93",
          "message": "mgrid file loading with absl::Status (#245)",
          "timestamp": "2025-04-17T11:30:30Z",
          "tree_id": "c2e3f1ec568ec633812a2fd29f3c6d842735c9a9",
          "url": "https://github.com/proximafusion/vmecpp/commit/1cc5f6769e64be435c863a6dc71f4b4d2d360c93"
        },
        "date": 1776338312809,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2272371108092646,
            "unit": "iter/sec",
            "range": "stddev: 0.002845756553240217",
            "extra": "mean: 309.8626 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2307964834823313,
            "unit": "iter/sec",
            "range": "stddev: 0.004704808217492671",
            "extra": "mean: 309.5212 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7132851591331529,
            "unit": "iter/sec",
            "range": "stddev: 0.005929651909487423",
            "extra": "mean: 1.4020 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27544297380600186,
            "unit": "iter/sec",
            "range": "stddev: 0.018242630641216517",
            "extra": "mean: 3.6305 sec\nrounds: 3"
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
          "id": "bc2df9db2435a2073fe7aed14866e31b6f80cf68",
          "message": "Expose FB hot-restart in Python (#194)",
          "timestamp": "2025-04-17T11:47:00Z",
          "tree_id": "8d44407bc9ee81e58c68daef182eb20dae82b0e8",
          "url": "https://github.com/proximafusion/vmecpp/commit/bc2df9db2435a2073fe7aed14866e31b6f80cf68"
        },
        "date": 1776338313005,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.211883256811134,
            "unit": "iter/sec",
            "range": "stddev: 0.002748041182144884",
            "extra": "mean: 311.3438 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.1629889662982285,
            "unit": "iter/sec",
            "range": "stddev: 0.00283659972893383",
            "extra": "mean: 316.1567 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7129063281264066,
            "unit": "iter/sec",
            "range": "stddev: 0.0023977451141362293",
            "extra": "mean: 1.4027 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27503252799394884,
            "unit": "iter/sec",
            "range": "stddev: 0.016674103887253648",
            "extra": "mean: 3.6359 sec\nrounds: 3"
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
          "id": "9f1715b14cc1f54c86aa93d954b60715253e772e",
          "message": "MagneticFieldResponseTable to use Eigen Matrices (#246)",
          "timestamp": "2025-04-17T13:38:30Z",
          "tree_id": "5d03618250e5b1155c3a6489e7981db9648dc682",
          "url": "https://github.com/proximafusion/vmecpp/commit/9f1715b14cc1f54c86aa93d954b60715253e772e"
        },
        "date": 1776338312975,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.237742648907034,
            "unit": "iter/sec",
            "range": "stddev: 0.0025194447091231955",
            "extra": "mean: 308.8572 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2465067335941944,
            "unit": "iter/sec",
            "range": "stddev: 0.0009701526733017742",
            "extra": "mean: 308.0234 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7170692499194676,
            "unit": "iter/sec",
            "range": "stddev: 0.0009518316183508758",
            "extra": "mean: 1.3946 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27408291360332976,
            "unit": "iter/sec",
            "range": "stddev: 0.0022434753082393845",
            "extra": "mean: 3.6485 sec\nrounds: 3"
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
          "id": "c725b56b2382699370c9aaf11ba1597faf75061d",
          "message": "cmake.verbose deprecation warning",
          "timestamp": "2025-04-17T15:57:52+02:00",
          "tree_id": "386a19c395fb1e485159943f7255863f7e13e04f",
          "url": "https://github.com/proximafusion/vmecpp/commit/c725b56b2382699370c9aaf11ba1597faf75061d"
        },
        "date": 1776338313023,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.197941941693896,
            "unit": "iter/sec",
            "range": "stddev: 0.00775784057270339",
            "extra": "mean: 312.7011 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2118891205464535,
            "unit": "iter/sec",
            "range": "stddev: 0.0011423645702992346",
            "extra": "mean: 311.3433 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7116266861000579,
            "unit": "iter/sec",
            "range": "stddev: 0.0020347511398988802",
            "extra": "mean: 1.4052 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27362791702614553,
            "unit": "iter/sec",
            "range": "stddev: 0.014840333210033313",
            "extra": "mean: 3.6546 sec\nrounds: 3"
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
          "id": "3126b39c6e959e732d9044f22d2805184962e1ca",
          "message": "Trigger docker action on release",
          "timestamp": "2025-04-17T16:14:02+02:00",
          "tree_id": "b9c558ffa360c1865241954e5ca134f50ba049bd",
          "url": "https://github.com/proximafusion/vmecpp/commit/3126b39c6e959e732d9044f22d2805184962e1ca"
        },
        "date": 1776338312832,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.1627648053691155,
            "unit": "iter/sec",
            "range": "stddev: 0.002595069808001656",
            "extra": "mean: 316.1791 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.1225116021992863,
            "unit": "iter/sec",
            "range": "stddev: 0.006162825488322819",
            "extra": "mean: 320.2550 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7035536596896972,
            "unit": "iter/sec",
            "range": "stddev: 0.03383379127044711",
            "extra": "mean: 1.4214 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27252590464940457,
            "unit": "iter/sec",
            "range": "stddev: 0.008684045239430236",
            "extra": "mean: 3.6694 sec\nrounds: 3"
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
          "id": "4008fe2d0192131f1e683e905a2b1693f4d9a4ef",
          "message": "The correct default for mgrid_file is NONE",
          "timestamp": "2025-04-20T18:40:51+02:00",
          "tree_id": "d55e3e97f246f567ea1f72f470f00a268302b69f",
          "url": "https://github.com/proximafusion/vmecpp/commit/4008fe2d0192131f1e683e905a2b1693f4d9a4ef"
        },
        "date": 1776338312849,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.098818157051195,
            "unit": "iter/sec",
            "range": "stddev: 0.00765765185882339",
            "extra": "mean: 322.7037 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2081440641421413,
            "unit": "iter/sec",
            "range": "stddev: 0.0027505809366741595",
            "extra": "mean: 311.7067 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7145506460170232,
            "unit": "iter/sec",
            "range": "stddev: 0.004186293318570128",
            "extra": "mean: 1.3995 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2728397760261944,
            "unit": "iter/sec",
            "range": "stddev: 0.042597742494665046",
            "extra": "mean: 3.6652 sec\nrounds: 3"
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
          "id": "cedecccde2a1dac204ee73e0833674a028df0d21",
          "message": "Raise error if mirror != NIL (#253)",
          "timestamp": "2025-04-22T09:08:25Z",
          "tree_id": "ec91eb3a0213695425e06d1db26f45764609cd75",
          "url": "https://github.com/proximafusion/vmecpp/commit/cedecccde2a1dac204ee73e0833674a028df0d21"
        },
        "date": 1776338313037,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.1574395696596227,
            "unit": "iter/sec",
            "range": "stddev: 0.003773144820664977",
            "extra": "mean: 316.7123 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.1623854020070716,
            "unit": "iter/sec",
            "range": "stddev: 0.003057009676897835",
            "extra": "mean: 316.2170 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7140268615276895,
            "unit": "iter/sec",
            "range": "stddev: 0.002587198844473214",
            "extra": "mean: 1.4005 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2714917840823756,
            "unit": "iter/sec",
            "range": "stddev: 0.04046340551505772",
            "extra": "mean: 3.6834 sec\nrounds: 3"
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
          "id": "1bd5de0ef7c071ae55e7c1599467613d2864b55f",
          "message": "Eigen3.4 version issue with hash in Bazel Central Registry",
          "timestamp": "2025-04-22T12:10:12+02:00",
          "tree_id": "1153b4ccd544ded735f63d2e414409891ae58450",
          "url": "https://github.com/proximafusion/vmecpp/commit/1bd5de0ef7c071ae55e7c1599467613d2864b55f"
        },
        "date": 1776338312807,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.1698454458991723,
            "unit": "iter/sec",
            "range": "stddev: 0.004436389888293113",
            "extra": "mean: 315.4728 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.233372840278217,
            "unit": "iter/sec",
            "range": "stddev: 0.001723716653732442",
            "extra": "mean: 309.2746 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7127932335313854,
            "unit": "iter/sec",
            "range": "stddev: 0.005277685745620618",
            "extra": "mean: 1.4029 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2729748966189533,
            "unit": "iter/sec",
            "range": "stddev: 0.041290732148536936",
            "extra": "mean: 3.6633 sec\nrounds: 3"
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
          "id": "002221c0e4117bd9c692b2dcf8b2de44281109a9",
          "message": "More specific array type makes typechecker happy (#257)",
          "timestamp": "2025-04-22T10:37:39Z",
          "tree_id": "61f6f6f005d97da4ad0375dd48ffb8cef712e9f6",
          "url": "https://github.com/proximafusion/vmecpp/commit/002221c0e4117bd9c692b2dcf8b2de44281109a9"
        },
        "date": 1776338312750,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2258949972267565,
            "unit": "iter/sec",
            "range": "stddev: 0.001017918288647962",
            "extra": "mean: 309.9915 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2349288466075508,
            "unit": "iter/sec",
            "range": "stddev: 0.0022239460878357557",
            "extra": "mean: 309.1258 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7136439191867417,
            "unit": "iter/sec",
            "range": "stddev: 0.004753327648478705",
            "extra": "mean: 1.4013 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27272395642876285,
            "unit": "iter/sec",
            "range": "stddev: 0.0453966623673983",
            "extra": "mean: 3.6667 sec\nrounds: 3"
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
          "id": "56025153ef137edddce5f7bb26f7b29716a37b88",
          "message": "Default initializer for VmecInput (reads values from VmecINDATAPyWrapper) (#260)",
          "timestamp": "2025-04-22T10:54:06Z",
          "tree_id": "2e0dc76a0c92b5234067481f90f0daa3c9e92ac2",
          "url": "https://github.com/proximafusion/vmecpp/commit/56025153ef137edddce5f7bb26f7b29716a37b88"
        },
        "date": 1776338312884,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.127754437396766,
            "unit": "iter/sec",
            "range": "stddev: 0.004308061409443277",
            "extra": "mean: 319.7182 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2363468574940466,
            "unit": "iter/sec",
            "range": "stddev: 0.002743209571082685",
            "extra": "mean: 308.9904 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7127734616951642,
            "unit": "iter/sec",
            "range": "stddev: 0.0018435333140258",
            "extra": "mean: 1.4030 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27376994110307407,
            "unit": "iter/sec",
            "range": "stddev: 0.04432018339749572",
            "extra": "mean: 3.6527 sec\nrounds: 3"
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
          "id": "ca715e1637cd7156ed220138ba5c67e9eef73309",
          "message": "test type annotations",
          "timestamp": "2025-04-24T16:46:17+02:00",
          "tree_id": "1a934ba1a389f47068eae74d76583df4410ca5e2",
          "url": "https://github.com/proximafusion/vmecpp/commit/ca715e1637cd7156ed220138ba5c67e9eef73309"
        },
        "date": 1776338313030,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2414726436539083,
            "unit": "iter/sec",
            "range": "stddev: 0.002232278191642686",
            "extra": "mean: 308.5018 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2263489944594204,
            "unit": "iter/sec",
            "range": "stddev: 0.0009805745559064616",
            "extra": "mean: 309.9479 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7142888957624975,
            "unit": "iter/sec",
            "range": "stddev: 0.0030626518045834402",
            "extra": "mean: 1.4000 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27349017049976304,
            "unit": "iter/sec",
            "range": "stddev: 0.03205372340330221",
            "extra": "mean: 3.6564 sec\nrounds: 3"
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
          "id": "0a5df4b4de2388afaa42f0728e7b2a243817e04f",
          "message": "Remove unused C++ methods",
          "timestamp": "2025-04-10T21:06:59+02:00",
          "tree_id": "bdaa959a2c7d2e6d77c64c17058186a34c68136c",
          "url": "https://github.com/proximafusion/vmecpp/commit/0a5df4b4de2388afaa42f0728e7b2a243817e04f"
        },
        "date": 1776338312781,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.1976107677460797,
            "unit": "iter/sec",
            "range": "stddev: 0.0019357685576841595",
            "extra": "mean: 312.7335 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.146813492821879,
            "unit": "iter/sec",
            "range": "stddev: 0.0038639001211231492",
            "extra": "mean: 317.7818 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7115686257443857,
            "unit": "iter/sec",
            "range": "stddev: 0.0025976073760408847",
            "extra": "mean: 1.4053 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27316235440154485,
            "unit": "iter/sec",
            "range": "stddev: 0.0033900957507290753",
            "extra": "mean: 3.6608 sec\nrounds: 3"
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
          "id": "05c0220543ab98fde3cb337e1e6fdf81e164bf37",
          "message": "Update bazel lock",
          "timestamp": "2025-04-24T19:05:23+02:00",
          "tree_id": "c966b706400f721b9183172de719093c9e4ac5a6",
          "url": "https://github.com/proximafusion/vmecpp/commit/05c0220543ab98fde3cb337e1e6fdf81e164bf37"
        },
        "date": 1776338312769,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.112870728389784,
            "unit": "iter/sec",
            "range": "stddev: 0.004344339685619881",
            "extra": "mean: 321.2469 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.096066291054222,
            "unit": "iter/sec",
            "range": "stddev: 0.0042985709601675555",
            "extra": "mean: 322.9905 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7107021344828076,
            "unit": "iter/sec",
            "range": "stddev: 0.0017995749699814575",
            "extra": "mean: 1.4071 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2722931183807088,
            "unit": "iter/sec",
            "range": "stddev: 0.005229097003103431",
            "extra": "mean: 3.6725 sec\nrounds: 3"
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
          "id": "fbdac7f79d5c032dd59e35e2e5df9d7676867655",
          "message": "Reference to nullptr in freeb (#277)",
          "timestamp": "2025-04-25T19:16:11Z",
          "tree_id": "c47e0288e896f2e14aac1369085924b494628e96",
          "url": "https://github.com/proximafusion/vmecpp/commit/fbdac7f79d5c032dd59e35e2e5df9d7676867655"
        },
        "date": 1776338313092,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.190607317630073,
            "unit": "iter/sec",
            "range": "stddev: 0.003196141377058729",
            "extra": "mean: 313.4200 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2177806755076537,
            "unit": "iter/sec",
            "range": "stddev: 0.002635332517930964",
            "extra": "mean: 310.7732 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7144314392477885,
            "unit": "iter/sec",
            "range": "stddev: 0.007020336745316961",
            "extra": "mean: 1.3997 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27360532691754474,
            "unit": "iter/sec",
            "range": "stddev: 0.020970404532607546",
            "extra": "mean: 3.6549 sec\nrounds: 3"
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
          "id": "044d155fdbd5a619fcc0d6aa3008e3eaa0cc0fac",
          "message": "Uninitialized bool was triggering ubsan (#278)",
          "timestamp": "2025-04-25T19:56:27Z",
          "tree_id": "f034c82672124343e31b3691355e3dc49e408f5d",
          "url": "https://github.com/proximafusion/vmecpp/commit/044d155fdbd5a619fcc0d6aa3008e3eaa0cc0fac"
        },
        "date": 1776338312762,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2027510247716755,
            "unit": "iter/sec",
            "range": "stddev: 0.002675757299585355",
            "extra": "mean: 312.2316 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2380118489892253,
            "unit": "iter/sec",
            "range": "stddev: 0.0025971979102209913",
            "extra": "mean: 308.8315 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7158770957747944,
            "unit": "iter/sec",
            "range": "stddev: 0.0030303590073368415",
            "extra": "mean: 1.3969 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27246816899811654,
            "unit": "iter/sec",
            "range": "stddev: 0.03453725265214187",
            "extra": "mean: 3.6702 sec\nrounds: 3"
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
          "id": "d5f9b5f80007124352039c182ce29d46f217f3d8",
          "message": "Add ubsan to the CI (#275)",
          "timestamp": "2025-04-25T20:30:49Z",
          "tree_id": "42b2076461992d4fff22af16ef8a5c6ac7d2f0ec",
          "url": "https://github.com/proximafusion/vmecpp/commit/d5f9b5f80007124352039c182ce29d46f217f3d8"
        },
        "date": 1776338313047,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.207013273138951,
            "unit": "iter/sec",
            "range": "stddev: 0.0032941711537044765",
            "extra": "mean: 311.8166 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.1978853494487693,
            "unit": "iter/sec",
            "range": "stddev: 0.0034169480568906597",
            "extra": "mean: 312.7066 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7093498175827133,
            "unit": "iter/sec",
            "range": "stddev: 0.002526672555769807",
            "extra": "mean: 1.4097 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26573239027309997,
            "unit": "iter/sec",
            "range": "stddev: 0.07283508299755195",
            "extra": "mean: 3.7632 sec\nrounds: 3"
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
          "id": "32c645bc0ada2ade08ef92a84653ba2e86e75a13",
          "message": "[flow control] Convert while to for loop (num_eqsolve_retries_) (#268)",
          "timestamp": "2025-04-25T20:53:31Z",
          "tree_id": "2712bfb025ac73a64e3316e78c9bebdb1536ce19",
          "url": "https://github.com/proximafusion/vmecpp/commit/32c645bc0ada2ade08ef92a84653ba2e86e75a13"
        },
        "date": 1776338312836,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.1281750233546077,
            "unit": "iter/sec",
            "range": "stddev: 0.0031483931115437146",
            "extra": "mean: 319.6752 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.13720272449757,
            "unit": "iter/sec",
            "range": "stddev: 0.0027909084074186894",
            "extra": "mean: 318.7553 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7116218762262897,
            "unit": "iter/sec",
            "range": "stddev: 0.009259532820313693",
            "extra": "mean: 1.4052 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2719635718705456,
            "unit": "iter/sec",
            "range": "stddev: 0.007434718556940079",
            "extra": "mean: 3.6770 sec\nrounds: 3"
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
          "id": "1ec6423ec51a42c34548f55ffe0744ddf2d64e3c",
          "message": "Cast to silence signed/unsigned comparison",
          "timestamp": "2025-04-26T20:58:22+02:00",
          "tree_id": "a9ee582b381435af68c97f4709a10847892a1111",
          "url": "https://github.com/proximafusion/vmecpp/commit/1ec6423ec51a42c34548f55ffe0744ddf2d64e3c"
        },
        "date": 1776338312812,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.138407567431131,
            "unit": "iter/sec",
            "range": "stddev: 0.004840097356607411",
            "extra": "mean: 318.6329 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2256619939817965,
            "unit": "iter/sec",
            "range": "stddev: 0.0010355867594480442",
            "extra": "mean: 310.0139 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7132263013530937,
            "unit": "iter/sec",
            "range": "stddev: 0.004172403750996607",
            "extra": "mean: 1.4021 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.273993989384354,
            "unit": "iter/sec",
            "range": "stddev: 0.007334316720159653",
            "extra": "mean: 3.6497 sec\nrounds: 3"
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
          "id": "7b04cbd0847085921045d52717b6a936dab5820a",
          "message": "[flow control] igrid replace 1 based with 0 based indexing (#270)",
          "timestamp": "2025-04-28T08:28:51Z",
          "tree_id": "190bf0b827b9d5c7bdfc323f7edef339c679d8d7",
          "url": "https://github.com/proximafusion/vmecpp/commit/7b04cbd0847085921045d52717b6a936dab5820a"
        },
        "date": 1776338312922,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.2304446886652545,
            "unit": "iter/sec",
            "range": "stddev: 0.0015530388191016512",
            "extra": "mean: 309.5549 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.1621058296744793,
            "unit": "iter/sec",
            "range": "stddev: 0.003025786836609528",
            "extra": "mean: 316.2449 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7118031124726534,
            "unit": "iter/sec",
            "range": "stddev: 0.008434053587351751",
            "extra": "mean: 1.4049 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27423391828701865,
            "unit": "iter/sec",
            "range": "stddev: 0.016712433902983277",
            "extra": "mean: 3.6465 sec\nrounds: 3"
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
          "id": "cd0654261768485c162c48b99473f9f44f5edad8",
          "message": "[flow control] Comments and refactored if statements (#271)",
          "timestamp": "2025-04-28T08:28:51Z",
          "tree_id": "652ea70661e92be83deef7e3dab896b56767b41f",
          "url": "https://github.com/proximafusion/vmecpp/commit/cd0654261768485c162c48b99473f9f44f5edad8"
        },
        "date": 1776338313033,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0319207161954287,
            "unit": "iter/sec",
            "range": "stddev: 0.007868686101681305",
            "extra": "mean: 329.8239 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.098848144572216,
            "unit": "iter/sec",
            "range": "stddev: 0.005158162696280726",
            "extra": "mean: 322.7005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.708713051421348,
            "unit": "iter/sec",
            "range": "stddev: 0.0024446910473953217",
            "extra": "mean: 1.4110 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2716341461255211,
            "unit": "iter/sec",
            "range": "stddev: 0.014888552603486253",
            "extra": "mean: 3.6814 sec\nrounds: 3"
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
          "id": "7fe779abe475ac983ca17122861ada8eb4c15885",
          "message": "[flow control] Thread local iter2 counter (#273)",
          "timestamp": "2025-04-28T08:28:52Z",
          "tree_id": "5acf3486a96c6e637f6a29adecb6b24725a82b0a",
          "url": "https://github.com/proximafusion/vmecpp/commit/7fe779abe475ac983ca17122861ada8eb4c15885"
        },
        "date": 1776338312931,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.201857271888238,
            "unit": "iter/sec",
            "range": "stddev: 0.0019562316743006107",
            "extra": "mean: 312.3187 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.2192792694313033,
            "unit": "iter/sec",
            "range": "stddev: 0.00233329792415394",
            "extra": "mean: 310.6285 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7152035849401097,
            "unit": "iter/sec",
            "range": "stddev: 0.0038083303638848376",
            "extra": "mean: 1.3982 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2736758149646952,
            "unit": "iter/sec",
            "range": "stddev: 0.011685877470497407",
            "extra": "mean: 3.6540 sec\nrounds: 3"
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
            "name": "Philipp Jurasic",
            "email": "jurasic@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "1fbdab68ee6e974bf077f614e5c14bcb52c284ed",
          "message": "[flow control] while->for loop (#274)",
          "timestamp": "2025-04-28T08:28:52Z",
          "tree_id": "7029c329c197ac574fdd20df330cc8b4829c7be9",
          "url": "https://github.com/proximafusion/vmecpp/commit/1fbdab68ee6e974bf077f614e5c14bcb52c284ed"
        },
        "date": 1776338312814,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.181187370003303,
            "unit": "iter/sec",
            "range": "stddev: 0.0028502598122628737",
            "extra": "mean: 314.3480 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.225770812049092,
            "unit": "iter/sec",
            "range": "stddev: 0.0037702585205535285",
            "extra": "mean: 310.0034 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7147083419187251,
            "unit": "iter/sec",
            "range": "stddev: 0.0038934190735907506",
            "extra": "mean: 1.3992 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27142711002364334,
            "unit": "iter/sec",
            "range": "stddev: 0.08571440537704236",
            "extra": "mean: 3.6842 sec\nrounds: 3"
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
          "id": "2994b344b014b02e930be115aade26fb67688d70",
          "message": "fsqt,wdot output quantities (#282)",
          "timestamp": "2025-04-28T11:09:32Z",
          "tree_id": "e3a785ca4f451719c1f80e30f9a3b6cd41df6523",
          "url": "https://github.com/proximafusion/vmecpp/commit/2994b344b014b02e930be115aade26fb67688d70"
        },
        "date": 1776338312820,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.8958368450893293,
            "unit": "iter/sec",
            "range": "stddev: 0.023215342338677284",
            "extra": "mean: 256.6843 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 4.087334123811321,
            "unit": "iter/sec",
            "range": "stddev: 0.0030755795337931323",
            "extra": "mean: 244.6582 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8741037513036258,
            "unit": "iter/sec",
            "range": "stddev: 0.00303773121841865",
            "extra": "mean: 1.1440 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3244670160784614,
            "unit": "iter/sec",
            "range": "stddev: 0.015197910491398346",
            "extra": "mean: 3.0820 sec\nrounds: 3"
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
          "id": "7e63d9e5d5331d1147c5b5008f8e5786d42e81d4",
          "message": "Add an example script that shows how to plot the plasma boundary geometry for addressing #263 . (#283)",
          "timestamp": "2025-04-28T11:56:35Z",
          "tree_id": "86e02a6549b41c54e909663bf091a32e390a473e",
          "url": "https://github.com/proximafusion/vmecpp/commit/7e63d9e5d5331d1147c5b5008f8e5786d42e81d4"
        },
        "date": 1776338312929,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 4.0642984632468995,
            "unit": "iter/sec",
            "range": "stddev: 0.0008865142792065142",
            "extra": "mean: 246.0449 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 4.069585629810083,
            "unit": "iter/sec",
            "range": "stddev: 0.0013645746710996185",
            "extra": "mean: 245.7253 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8733695198350925,
            "unit": "iter/sec",
            "range": "stddev: 0.001673547408567814",
            "extra": "mean: 1.1450 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3209296496334637,
            "unit": "iter/sec",
            "range": "stddev: 0.05534800487529515",
            "extra": "mean: 3.1159 sec\nrounds: 3"
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
          "id": "ebf4b075f2f2db11ee65b515b852324111f92a89",
          "message": "Adjust tolerance for jdotb. (#284)",
          "timestamp": "2025-04-28T13:50:59Z",
          "tree_id": "69178e3eeb1b079fe0ecefae8d467edd61c2d65b",
          "url": "https://github.com/proximafusion/vmecpp/commit/ebf4b075f2f2db11ee65b515b852324111f92a89"
        },
        "date": 1776338313074,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.994569622272748,
            "unit": "iter/sec",
            "range": "stddev: 0.002082471360806423",
            "extra": "mean: 250.3399 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.9522022306342834,
            "unit": "iter/sec",
            "range": "stddev: 0.004931334028207972",
            "extra": "mean: 253.0235 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8717032244344709,
            "unit": "iter/sec",
            "range": "stddev: 0.006281356806478101",
            "extra": "mean: 1.1472 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32163279187111005,
            "unit": "iter/sec",
            "range": "stddev: 0.049527255186375324",
            "extra": "mean: 3.1091 sec\nrounds: 3"
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
          "id": "86d7a4fb30290b2914a011704b28a9c528805f7d",
          "message": "Return status code instead of terminating when coils file is missing (#285)",
          "timestamp": "2025-04-28T14:57:00Z",
          "tree_id": "c96fbe2120dfc32e0c0d1e513588ff6d61bdc78b",
          "url": "https://github.com/proximafusion/vmecpp/commit/86d7a4fb30290b2914a011704b28a9c528805f7d"
        },
        "date": 1776338312948,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 4.044298583922085,
            "unit": "iter/sec",
            "range": "stddev: 0.0014758856888734454",
            "extra": "mean: 247.2617 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.9941684501562187,
            "unit": "iter/sec",
            "range": "stddev: 0.0018511925592898481",
            "extra": "mean: 250.3650 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8740056593841632,
            "unit": "iter/sec",
            "range": "stddev: 0.0010749460639037634",
            "extra": "mean: 1.1442 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32139847139422195,
            "unit": "iter/sec",
            "range": "stddev: 0.06299097648028851",
            "extra": "mean: 3.1114 sec\nrounds: 3"
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
          "id": "b2bfbe8ac74dfbd2994971915d7af6797afc4b5d",
          "message": "Jdotb tolerance in cpp test (#286)",
          "timestamp": "2025-04-28T16:17:58Z",
          "tree_id": "88c1947c66828c862c69776af31183dc39a119e1",
          "url": "https://github.com/proximafusion/vmecpp/commit/b2bfbe8ac74dfbd2994971915d7af6797afc4b5d"
        },
        "date": 1776338312995,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.607308331358632,
            "unit": "iter/sec",
            "range": "stddev: 0.06480631552816281",
            "extra": "mean: 277.2150 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.9394708342833096,
            "unit": "iter/sec",
            "range": "stddev: 0.005524095359086097",
            "extra": "mean: 253.8412 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.869086612905042,
            "unit": "iter/sec",
            "range": "stddev: 0.006085608314244905",
            "extra": "mean: 1.1506 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3204216680795309,
            "unit": "iter/sec",
            "range": "stddev: 0.060802903066527754",
            "extra": "mean: 3.1209 sec\nrounds: 3"
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
          "id": "7b6b6d127ee9dcfdfe81bda34a94defca1139c07",
          "message": "BaseModelWithNumpy for numpy+pydantic serialization (#264)",
          "timestamp": "2025-04-28T16:41:56Z",
          "tree_id": "0a72155e972bed6b3c23c20bb50a0f4c5f0677dd",
          "url": "https://github.com/proximafusion/vmecpp/commit/7b6b6d127ee9dcfdfe81bda34a94defca1139c07"
        },
        "date": 1776338312925,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 4.024515915006898,
            "unit": "iter/sec",
            "range": "stddev: 0.0019944408167769725",
            "extra": "mean: 248.4771 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 4.009524532506138,
            "unit": "iter/sec",
            "range": "stddev: 0.0026270819927495313",
            "extra": "mean: 249.4061 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8718423242351601,
            "unit": "iter/sec",
            "range": "stddev: 0.0022551098764314968",
            "extra": "mean: 1.1470 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3202355556078609,
            "unit": "iter/sec",
            "range": "stddev: 0.04869618000684561",
            "extra": "mean: 3.1227 sec\nrounds: 3"
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
          "id": "6c4964354e54e162c36c7a609db4ffc48f31512a",
          "message": "BaseModelWithNumpy replaces numpydantic (#265)",
          "timestamp": "2025-04-28T16:41:56Z",
          "tree_id": "a2bfde9f8b88b960a491daf85c346d25a78cbbec",
          "url": "https://github.com/proximafusion/vmecpp/commit/6c4964354e54e162c36c7a609db4ffc48f31512a"
        },
        "date": 1776338312913,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.29131636096063,
            "unit": "iter/sec",
            "range": "stddev: 0.0024056858426791936",
            "extra": "mean: 436.4303 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.26920349160677,
            "unit": "iter/sec",
            "range": "stddev: 0.007766896437301568",
            "extra": "mean: 440.6833 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.874709226425467,
            "unit": "iter/sec",
            "range": "stddev: 0.00037245777020010435",
            "extra": "mean: 1.1432 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3246253573495843,
            "unit": "iter/sec",
            "range": "stddev: 0.03615447289094832",
            "extra": "mean: 3.0805 sec\nrounds: 3"
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
          "id": "062c4d3174e2ecae317da6e28567d1a920fa30f6",
          "message": "Pyright type checking",
          "timestamp": "2025-03-10T19:04:45+01:00",
          "tree_id": "95e738698081f41e3a3d9d863a4c4a4a2ab2dcb0",
          "url": "https://github.com/proximafusion/vmecpp/commit/062c4d3174e2ecae317da6e28567d1a920fa30f6"
        },
        "date": 1776338312771,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.2502494440886927,
            "unit": "iter/sec",
            "range": "stddev: 0.005292874222762814",
            "extra": "mean: 444.3952 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.277174974291212,
            "unit": "iter/sec",
            "range": "stddev: 0.004724986274117291",
            "extra": "mean: 439.1406 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8742477608724102,
            "unit": "iter/sec",
            "range": "stddev: 0.0019300542152444048",
            "extra": "mean: 1.1438 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32448549504192886,
            "unit": "iter/sec",
            "range": "stddev: 0.038097419935980176",
            "extra": "mean: 3.0818 sec\nrounds: 3"
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
          "id": "a46fddd7bec038b1e8f79da727be8dd158e885c3",
          "message": "Add Python types to interact with free-boundary (#220)",
          "timestamp": "2025-04-28T18:10:35Z",
          "tree_id": "e12e29e1c76dcb603cdad882fe73413bab7b99ac",
          "url": "https://github.com/proximafusion/vmecpp/commit/a46fddd7bec038b1e8f79da727be8dd158e885c3"
        },
        "date": 1776338312984,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.260369055274704,
            "unit": "iter/sec",
            "range": "stddev: 0.0026001718950910894",
            "extra": "mean: 442.4056 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.259282632994328,
            "unit": "iter/sec",
            "range": "stddev: 0.00348708759181772",
            "extra": "mean: 442.6184 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8756891422883852,
            "unit": "iter/sec",
            "range": "stddev: 0.001560148142803835",
            "extra": "mean: 1.1420 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32420504581809667,
            "unit": "iter/sec",
            "range": "stddev: 0.056147928155514906",
            "extra": "mean: 3.0845 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.735840991245047,
            "unit": "iter/sec",
            "range": "stddev: 0.01583178690926345",
            "extra": "mean: 1.3590 sec\nrounds: 3"
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
          "id": "1763e4337598df0372cbc651375b2bc278c0d8ee",
          "message": "Build NetCDF4 from source as well. (#288)",
          "timestamp": "2025-04-28T18:55:08Z",
          "tree_id": "c0297c4270e7bfaa66036215112ae33a6f508b24",
          "url": "https://github.com/proximafusion/vmecpp/commit/1763e4337598df0372cbc651375b2bc278c0d8ee"
        },
        "date": 1776338312803,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.0674048378917185,
            "unit": "iter/sec",
            "range": "stddev: 0.005020334404156392",
            "extra": "mean: 483.6982 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.091426938600081,
            "unit": "iter/sec",
            "range": "stddev: 0.018398800608518787",
            "extra": "mean: 478.1424 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8707346839830662,
            "unit": "iter/sec",
            "range": "stddev: 0.006101197747902535",
            "extra": "mean: 1.1485 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32096988001025023,
            "unit": "iter/sec",
            "range": "stddev: 0.05061358030990874",
            "extra": "mean: 3.1156 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7380446614058371,
            "unit": "iter/sec",
            "range": "stddev: 0.013042861547097362",
            "extra": "mean: 1.3549 sec\nrounds: 3"
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
          "id": "f9b89236f3e92d36e1206322e18517a7479c7f4c",
          "message": "jaxtyping annotations typo broke docs CI (#289)",
          "timestamp": "2025-04-28T20:16:28Z",
          "tree_id": "1e27205670b6bc5039506749abbff32c3f16090e",
          "url": "https://github.com/proximafusion/vmecpp/commit/f9b89236f3e92d36e1206322e18517a7479c7f4c"
        },
        "date": 1776338313088,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.1406117255582386,
            "unit": "iter/sec",
            "range": "stddev: 0.011090124326245473",
            "extra": "mean: 467.1562 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.175832944545633,
            "unit": "iter/sec",
            "range": "stddev: 0.007555397904903559",
            "extra": "mean: 459.5941 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8754877430130071,
            "unit": "iter/sec",
            "range": "stddev: 0.0021919035101476587",
            "extra": "mean: 1.1422 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3244251339808591,
            "unit": "iter/sec",
            "range": "stddev: 0.008010238704051511",
            "extra": "mean: 3.0824 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7393351884233819,
            "unit": "iter/sec",
            "range": "stddev: 0.015548799625327422",
            "extra": "mean: 1.3526 sec\nrounds: 3"
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
            "name": "Jonathan Schilling",
            "email": "jons@proximafusion.com",
            "username": ""
          },
          "distinct": true,
          "id": "56e90ffd9096cdc6cb16119194a38e3dca885d01",
          "message": "Add currH to OutputQuantities. (#88)",
          "timestamp": "2025-04-29T07:45:49Z",
          "tree_id": "f90155ccb32ec6d3f1689cdc39ed5c605b8d2c5a",
          "url": "https://github.com/proximafusion/vmecpp/commit/56e90ffd9096cdc6cb16119194a38e3dca885d01"
        },
        "date": 1776338312885,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.1741561264218454,
            "unit": "iter/sec",
            "range": "stddev: 0.009819336079601684",
            "extra": "mean: 459.9486 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.1751579639301895,
            "unit": "iter/sec",
            "range": "stddev: 0.01665256831256644",
            "extra": "mean: 459.7367 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.877199842312816,
            "unit": "iter/sec",
            "range": "stddev: 0.0029571357211382334",
            "extra": "mean: 1.1400 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3262711936370756,
            "unit": "iter/sec",
            "range": "stddev: 0.0150936968757038",
            "extra": "mean: 3.0649 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7418517355185983,
            "unit": "iter/sec",
            "range": "stddev: 0.009979524070850607",
            "extra": "mean: 1.3480 sec\nrounds: 3"
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
          "id": "acb74ee9e43071f29c7d24966ea52f80664d624a",
          "message": "Add chipH to OutputQuantities. (#89)",
          "timestamp": "2025-04-29T08:37:38Z",
          "tree_id": "27f42beaedb46746f305be8e93ebf69577b605eb",
          "url": "https://github.com/proximafusion/vmecpp/commit/acb74ee9e43071f29c7d24966ea52f80664d624a"
        },
        "date": 1776338312989,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.191381949577306,
            "unit": "iter/sec",
            "range": "stddev: 0.0027864559270255602",
            "extra": "mean: 456.3330 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.140390495085099,
            "unit": "iter/sec",
            "range": "stddev: 0.017424403374088508",
            "extra": "mean: 467.2045 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8739506164695318,
            "unit": "iter/sec",
            "range": "stddev: 0.001483041742540024",
            "extra": "mean: 1.1442 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32383088545472394,
            "unit": "iter/sec",
            "range": "stddev: 0.025835063256850376",
            "extra": "mean: 3.0880 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7356965667497576,
            "unit": "iter/sec",
            "range": "stddev: 0.020419521633652942",
            "extra": "mean: 1.3593 sec\nrounds: 3"
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
          "id": "e227066ba50ebc58031e184485e98b5a3819b18b",
          "message": "One more jdotb tolerance increased",
          "timestamp": "2025-04-29T13:03:42+02:00",
          "tree_id": "7220a9f5b3afbd0632121109aeff8780a5eee867",
          "url": "https://github.com/proximafusion/vmecpp/commit/e227066ba50ebc58031e184485e98b5a3819b18b"
        },
        "date": 1776338313066,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.15405088120479,
            "unit": "iter/sec",
            "range": "stddev: 0.008316850843297516",
            "extra": "mean: 464.2416 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.1414884181831013,
            "unit": "iter/sec",
            "range": "stddev: 0.006440110401630361",
            "extra": "mean: 466.9649 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8727244523356508,
            "unit": "iter/sec",
            "range": "stddev: 0.002851180117466034",
            "extra": "mean: 1.1458 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3206559371712017,
            "unit": "iter/sec",
            "range": "stddev: 0.041070088655615365",
            "extra": "mean: 3.1186 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7413499843544842,
            "unit": "iter/sec",
            "range": "stddev: 0.007932916266057854",
            "extra": "mean: 1.3489 sec\nrounds: 3"
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
          "id": "fe1e4912489781c5abfe294c49cb02a46f7bdfcc",
          "message": "Migrate some wout docs from repo/13962 (#290)",
          "timestamp": "2025-04-29T11:12:07Z",
          "tree_id": "fadb1d96a9034afe64beb78dcaf43c61d1a2dba2",
          "url": "https://github.com/proximafusion/vmecpp/commit/fe1e4912489781c5abfe294c49cb02a46f7bdfcc"
        },
        "date": 1776338313094,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.2973100102382835,
            "unit": "iter/sec",
            "range": "stddev: 0.0014108525652301116",
            "extra": "mean: 435.2917 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.2914547354608414,
            "unit": "iter/sec",
            "range": "stddev: 0.0018397792610094833",
            "extra": "mean: 436.4040 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8719879976246576,
            "unit": "iter/sec",
            "range": "stddev: 0.005498833454025476",
            "extra": "mean: 1.1468 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3309603844809465,
            "unit": "iter/sec",
            "range": "stddev: 0.0093483655220772",
            "extra": "mean: 3.0215 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7416252652920001,
            "unit": "iter/sec",
            "range": "stddev: 0.012803538956893038",
            "extra": "mean: 1.3484 sec\nrounds: 3"
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
          "id": "dbeefde7fed6863215ae626c0c36f8ab5b10ffe6",
          "message": "Don't cancel CI on main (#292)",
          "timestamp": "2025-04-29T11:28:37Z",
          "tree_id": "a3adabf603aca0083b1b549843aeeec19612c496",
          "url": "https://github.com/proximafusion/vmecpp/commit/dbeefde7fed6863215ae626c0c36f8ab5b10ffe6"
        },
        "date": 1776338313057,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.258353448412177,
            "unit": "iter/sec",
            "range": "stddev: 0.005613427543405777",
            "extra": "mean: 442.8005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.2674908345234006,
            "unit": "iter/sec",
            "range": "stddev: 0.0059506561431228455",
            "extra": "mean: 441.0161 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8694866818259359,
            "unit": "iter/sec",
            "range": "stddev: 0.010621755384868052",
            "extra": "mean: 1.1501 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32398225111446105,
            "unit": "iter/sec",
            "range": "stddev: 0.08749766717468967",
            "extra": "mean: 3.0866 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7409127603772381,
            "unit": "iter/sec",
            "range": "stddev: 0.008933040683438942",
            "extra": "mean: 1.3497 sec\nrounds: 3"
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
          "id": "139337b03e77e0cf4f976eb5a1992d36992a8843",
          "message": "Fixed Py3.13 pydantic serialization (#293)",
          "timestamp": "2025-04-29T11:44:56Z",
          "tree_id": "aa5cfc7db09e143e1fab463faef8bfb96157f6c5",
          "url": "https://github.com/proximafusion/vmecpp/commit/139337b03e77e0cf4f976eb5a1992d36992a8843"
        },
        "date": 1776338312796,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.2787632847395094,
            "unit": "iter/sec",
            "range": "stddev: 0.003223369633055033",
            "extra": "mean: 438.8345 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.2859743277034412,
            "unit": "iter/sec",
            "range": "stddev: 0.0036550852293408143",
            "extra": "mean: 437.4502 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8749354915530809,
            "unit": "iter/sec",
            "range": "stddev: 0.0015546945793092266",
            "extra": "mean: 1.1429 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3270218736333772,
            "unit": "iter/sec",
            "range": "stddev: 0.033405325702256704",
            "extra": "mean: 3.0579 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7375002518594738,
            "unit": "iter/sec",
            "range": "stddev: 0.009870238600077193",
            "extra": "mean: 1.3559 sec\nrounds: 3"
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
          "id": "36f313b649b443add37b2b8ac880c11bdc03d89b",
          "message": "Python version matrix added to tests (#291)",
          "timestamp": "2025-04-29T12:18:41Z",
          "tree_id": "0558e47c6d650c33454c6d631187f0b1a4fd3612",
          "url": "https://github.com/proximafusion/vmecpp/commit/36f313b649b443add37b2b8ac880c11bdc03d89b"
        },
        "date": 1776338312839,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.2789099585523083,
            "unit": "iter/sec",
            "range": "stddev: 0.0037268008656685817",
            "extra": "mean: 438.8063 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.246674138859166,
            "unit": "iter/sec",
            "range": "stddev: 0.008199649374613297",
            "extra": "mean: 445.1024 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.873539887177847,
            "unit": "iter/sec",
            "range": "stddev: 0.004506156122099967",
            "extra": "mean: 1.1448 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3284865321078595,
            "unit": "iter/sec",
            "range": "stddev: 0.050216924054087154",
            "extra": "mean: 3.0443 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7439918338682858,
            "unit": "iter/sec",
            "range": "stddev: 0.006623349983815891",
            "extra": "mean: 1.3441 sec\nrounds: 3"
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
          "id": "b504c3ffe9b867ec1e0936e470ab19a19b034186",
          "message": "pip cache didnt help much",
          "timestamp": "2025-04-29T14:31:17+02:00",
          "tree_id": "31bd759b9ec74de0d8e0803fbac7a7deef970f97",
          "url": "https://github.com/proximafusion/vmecpp/commit/b504c3ffe9b867ec1e0936e470ab19a19b034186"
        },
        "date": 1776338312999,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.271577482807989,
            "unit": "iter/sec",
            "range": "stddev: 0.0038724919941883",
            "extra": "mean: 440.2227 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.2334520568847878,
            "unit": "iter/sec",
            "range": "stddev: 0.0117248537833363",
            "extra": "mean: 447.7374 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8777038861891696,
            "unit": "iter/sec",
            "range": "stddev: 0.005952936261242418",
            "extra": "mean: 1.1393 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32580884276464245,
            "unit": "iter/sec",
            "range": "stddev: 0.04760091080875552",
            "extra": "mean: 3.0693 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.740205089408724,
            "unit": "iter/sec",
            "range": "stddev: 0.011853672709856272",
            "extra": "mean: 1.3510 sec\nrounds: 3"
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
          "id": "0c4edb54e875192878e8ff3253d35e3c31f38ee3",
          "message": "Cache and unify bazel CI to avoid rate limits (#296)",
          "timestamp": "2025-04-29T13:01:34Z",
          "tree_id": "267978528490b0d9d4a5e704d5b62d04f24effa2",
          "url": "https://github.com/proximafusion/vmecpp/commit/0c4edb54e875192878e8ff3253d35e3c31f38ee3"
        },
        "date": 1776338312785,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.2912260958704422,
            "unit": "iter/sec",
            "range": "stddev: 0.003958511806132779",
            "extra": "mean: 436.4475 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.2911974800200303,
            "unit": "iter/sec",
            "range": "stddev: 0.002339133478538285",
            "extra": "mean: 436.4530 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.872980432989454,
            "unit": "iter/sec",
            "range": "stddev: 0.001640436506066121",
            "extra": "mean: 1.1455 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32774157575528956,
            "unit": "iter/sec",
            "range": "stddev: 0.040600059724970376",
            "extra": "mean: 3.0512 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7409619019515475,
            "unit": "iter/sec",
            "range": "stddev: 0.013476090560792586",
            "extra": "mean: 1.3496 sec\nrounds: 3"
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
          "id": "2e20fd35ed612b5c8b98074c7bb0b766f299554f",
          "message": "CI concurrency fix",
          "timestamp": "2025-04-29T14:47:14+02:00",
          "tree_id": "8cdec1c32a46b7f227091c1756144f51801ee5b9",
          "url": "https://github.com/proximafusion/vmecpp/commit/2e20fd35ed612b5c8b98074c7bb0b766f299554f"
        },
        "date": 1776338312828,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.2912098533605287,
            "unit": "iter/sec",
            "range": "stddev: 0.0030192582551668017",
            "extra": "mean: 436.4506 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.283772997233867,
            "unit": "iter/sec",
            "range": "stddev: 0.0026709215183598465",
            "extra": "mean: 437.8719 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8708437856939508,
            "unit": "iter/sec",
            "range": "stddev: 0.005171572286032638",
            "extra": "mean: 1.1483 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.32588753513511226,
            "unit": "iter/sec",
            "range": "stddev: 0.01773728460346612",
            "extra": "mean: 3.0685 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7424502127401954,
            "unit": "iter/sec",
            "range": "stddev: 0.007761909257496457",
            "extra": "mean: 1.3469 sec\nrounds: 3"
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
          "id": "11c0fe4718d506e01736215e5f2c3c2e7d7ed9cb",
          "message": "HDF5 compatibility with old versions",
          "timestamp": "2025-04-29T16:54:26+02:00",
          "tree_id": "eb3df5250d58c02f859607740d7b3c9e8d293bf0",
          "url": "https://github.com/proximafusion/vmecpp/commit/11c0fe4718d506e01736215e5f2c3c2e7d7ed9cb"
        },
        "date": 1776338312794,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.002926432547398,
            "unit": "iter/sec",
            "range": "stddev: 0.013271650787263082",
            "extra": "mean: 499.2695 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.167705418458548,
            "unit": "iter/sec",
            "range": "stddev: 0.010633945040267814",
            "extra": "mean: 461.3173 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.874769445983467,
            "unit": "iter/sec",
            "range": "stddev: 0.0013523326961458024",
            "extra": "mean: 1.1432 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.322762858850906,
            "unit": "iter/sec",
            "range": "stddev: 0.06580329421319472",
            "extra": "mean: 3.0982 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7434151816291945,
            "unit": "iter/sec",
            "range": "stddev: 0.007043880282507367",
            "extra": "mean: 1.3451 sec\nrounds: 3"
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
          "id": "600c5acf35724520c9200cee2e5d9102aa380542",
          "message": "Pass Magnetic response table to Python layer by reference",
          "timestamp": "2025-04-30T01:14:09+02:00",
          "tree_id": "bbc11a1f8f70202c9e5dfffe0482b14060f16b20",
          "url": "https://github.com/proximafusion/vmecpp/commit/600c5acf35724520c9200cee2e5d9102aa380542"
        },
        "date": 1776338312902,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.25576817076372,
            "unit": "iter/sec",
            "range": "stddev: 0.007754325378861364",
            "extra": "mean: 443.3080 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.2972098831212846,
            "unit": "iter/sec",
            "range": "stddev: 0.0023520084506412626",
            "extra": "mean: 435.3107 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8644010552309773,
            "unit": "iter/sec",
            "range": "stddev: 0.026663034564905347",
            "extra": "mean: 1.1569 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.329429408097503,
            "unit": "iter/sec",
            "range": "stddev: 0.008607201648504237",
            "extra": "mean: 3.0356 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.12712553274724156,
            "unit": "iter/sec",
            "range": "stddev: 0.024615523461461033",
            "extra": "mean: 7.8662 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7411914735505554,
            "unit": "iter/sec",
            "range": "stddev: 0.013043557888439528",
            "extra": "mean: 1.3492 sec\nrounds: 3"
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
          "id": "39893fb3de2709683c09ea4815b89d12720a68dd",
          "message": "Example using threed1 quantities added",
          "timestamp": "2025-05-02T17:20:28+02:00",
          "tree_id": "b48d9083c5fbd8d1248e24ffe05e064b1cfbb6ac",
          "url": "https://github.com/proximafusion/vmecpp/commit/39893fb3de2709683c09ea4815b89d12720a68dd"
        },
        "date": 1776338312843,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.148483656311099,
            "unit": "iter/sec",
            "range": "stddev: 0.007097908056413848",
            "extra": "mean: 465.4445 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.1583588119313886,
            "unit": "iter/sec",
            "range": "stddev: 0.009557924396748239",
            "extra": "mean: 463.3150 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8671156771810622,
            "unit": "iter/sec",
            "range": "stddev: 0.012342869198368503",
            "extra": "mean: 1.1532 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3208858520299375,
            "unit": "iter/sec",
            "range": "stddev: 0.023919602669903436",
            "extra": "mean: 3.1164 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.12715690961212894,
            "unit": "iter/sec",
            "range": "stddev: 0.022812019266997545",
            "extra": "mean: 7.8643 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7420214756706399,
            "unit": "iter/sec",
            "range": "stddev: 0.01007624394732326",
            "extra": "mean: 1.3477 sec\nrounds: 3"
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
          "id": "6fd8f2aa1ff4815eead3c2b29633e87336848140",
          "message": "Removed jax import requirement, to improve vmecpp import time",
          "timestamp": "2025-05-02T15:55:02+02:00",
          "tree_id": "8fb760e141e1f1a77c22d26c4aa3ee2cc5d711fd",
          "url": "https://github.com/proximafusion/vmecpp/commit/6fd8f2aa1ff4815eead3c2b29633e87336848140"
        },
        "date": 1776338312917,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.8930657283968673,
            "unit": "iter/sec",
            "range": "stddev: 0.0035166711269966388",
            "extra": "mean: 256.8670 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.923969915458568,
            "unit": "iter/sec",
            "range": "stddev: 0.0031375658879194235",
            "extra": "mean: 254.8440 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.8713988595782176,
            "unit": "iter/sec",
            "range": "stddev: 0.0017142893126303764",
            "extra": "mean: 1.1476 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.3249429141100837,
            "unit": "iter/sec",
            "range": "stddev: 0.013632410267707982",
            "extra": "mean: 3.0775 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.12646667816135262,
            "unit": "iter/sec",
            "range": "stddev: 0.029731026311907743",
            "extra": "mean: 7.9072 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.7429702666466539,
            "unit": "iter/sec",
            "range": "stddev: 0.00922769146015191",
            "extra": "mean: 1.3459 sec\nrounds: 3"
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
          "id": "2b58bab226a26606929367873d51ef00b6f5b949",
          "message": "Improved backwards compatibility when padding wout using VMEC2000 convention",
          "timestamp": "2025-04-30T11:41:06+02:00",
          "tree_id": "7e92c08b828ab3467ba737aecd48cf333dea9012",
          "url": "https://github.com/proximafusion/vmecpp/commit/2b58bab226a26606929367873d51ef00b6f5b949"
        },
        "date": 1776338312824,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.964179919104997,
            "unit": "iter/sec",
            "range": "stddev: 0.003909411698195258",
            "extra": "mean: 337.3614 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.885575202951637,
            "unit": "iter/sec",
            "range": "stddev: 0.019683250333034624",
            "extra": "mean: 346.5514 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7072407629216485,
            "unit": "iter/sec",
            "range": "stddev: 0.005916287664319853",
            "extra": "mean: 1.4139 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2715349565617608,
            "unit": "iter/sec",
            "range": "stddev: 0.0160544908205956",
            "extra": "mean: 3.6828 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10442740483644126,
            "unit": "iter/sec",
            "range": "stddev: 0.01462709991698457",
            "extra": "mean: 9.5760 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6429364278945446,
            "unit": "iter/sec",
            "range": "stddev: 0.011977685522069903",
            "extra": "mean: 1.5554 sec\nrounds: 3"
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
          "id": "872bf02fd64defb5011e92c7180a136e4d0244c1",
          "message": "Improved backwards compatibility when padding wout using VMEC2000 convention",
          "timestamp": "2025-04-30T11:41:06+02:00",
          "tree_id": "4e69d8194209fa7cedba2193ca855e3f366ef135",
          "url": "https://github.com/proximafusion/vmecpp/commit/872bf02fd64defb5011e92c7180a136e4d0244c1"
        },
        "date": 1776338312949,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8890599416535627,
            "unit": "iter/sec",
            "range": "stddev: 0.006017906203491019",
            "extra": "mean: 346.1334 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9489711825443647,
            "unit": "iter/sec",
            "range": "stddev: 0.00931120689633498",
            "extra": "mean: 339.1013 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.713897048845189,
            "unit": "iter/sec",
            "range": "stddev: 0.002245607555808718",
            "extra": "mean: 1.4008 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2720398217172224,
            "unit": "iter/sec",
            "range": "stddev: 0.028634761218166604",
            "extra": "mean: 3.6759 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10388372029106958,
            "unit": "iter/sec",
            "range": "stddev: 0.05244040508936992",
            "extra": "mean: 9.6261 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6433581210467633,
            "unit": "iter/sec",
            "range": "stddev: 0.010205555566337832",
            "extra": "mean: 1.5543 sec\nrounds: 3"
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
          "id": "2c0ae110106758411658177c385515c0d0fd961d",
          "message": "VmecInput only dump non-default values",
          "timestamp": "2025-04-30T16:03:52+02:00",
          "tree_id": "03b769dfbbed27aa7135d22a9bccc76d8ca39223",
          "url": "https://github.com/proximafusion/vmecpp/commit/2c0ae110106758411658177c385515c0d0fd961d"
        },
        "date": 1776338312825,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9806564665167388,
            "unit": "iter/sec",
            "range": "stddev: 0.001623736985793535",
            "extra": "mean: 335.4966 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.979375293598083,
            "unit": "iter/sec",
            "range": "stddev: 0.002596032541750628",
            "extra": "mean: 335.6408 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7124913768504403,
            "unit": "iter/sec",
            "range": "stddev: 0.0028103559333023283",
            "extra": "mean: 1.4035 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27003724763635345,
            "unit": "iter/sec",
            "range": "stddev: 0.06251470988419389",
            "extra": "mean: 3.7032 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10454066922689056,
            "unit": "iter/sec",
            "range": "stddev: 0.013199716943194321",
            "extra": "mean: 9.5657 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6438633331525507,
            "unit": "iter/sec",
            "range": "stddev: 0.009995030320942387",
            "extra": "mean: 1.5531 sec\nrounds: 3"
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
          "id": "574f69c98da4d24f2fcabf71363fe53daada4385",
          "message": "Added more missing quantities, improving compatibility",
          "timestamp": "2025-05-02T15:17:59+02:00",
          "tree_id": "7fa277f85a390e80a783c71dfd1108c4f44f0eef",
          "url": "https://github.com/proximafusion/vmecpp/commit/574f69c98da4d24f2fcabf71363fe53daada4385"
        },
        "date": 1776338312887,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9719848363696,
            "unit": "iter/sec",
            "range": "stddev: 0.004474999341698485",
            "extra": "mean: 336.4755 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.003440130714801,
            "unit": "iter/sec",
            "range": "stddev: 0.001611645635333826",
            "extra": "mean: 332.9515 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7094338430257269,
            "unit": "iter/sec",
            "range": "stddev: 0.0009737462541292439",
            "extra": "mean: 1.4096 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2679437611803571,
            "unit": "iter/sec",
            "range": "stddev: 0.07067343794151769",
            "extra": "mean: 3.7321 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1042847529127838,
            "unit": "iter/sec",
            "range": "stddev: 0.02338905688867308",
            "extra": "mean: 9.5891 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6411440195010626,
            "unit": "iter/sec",
            "range": "stddev: 0.01648331202006443",
            "extra": "mean: 1.5597 sec\nrounds: 3"
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
          "id": "96e1e648035dab1580cb639408ebb158701884db",
          "message": "Optional parameter to format VmecInput json for human readability (#309)",
          "timestamp": "2025-05-05T11:18:51Z",
          "tree_id": "66d28f48e14469982830682182cfcec4ec1616aa",
          "url": "https://github.com/proximafusion/vmecpp/commit/96e1e648035dab1580cb639408ebb158701884db"
        },
        "date": 1776338312967,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9609103914496178,
            "unit": "iter/sec",
            "range": "stddev: 0.0029367649307621337",
            "extra": "mean: 337.7340 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.001103691896064,
            "unit": "iter/sec",
            "range": "stddev: 0.0018671898031066994",
            "extra": "mean: 333.2107 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7070842670899747,
            "unit": "iter/sec",
            "range": "stddev: 0.007896511635651877",
            "extra": "mean: 1.4143 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2718919042792898,
            "unit": "iter/sec",
            "range": "stddev: 0.011267758418040435",
            "extra": "mean: 3.6779 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10467779827677905,
            "unit": "iter/sec",
            "range": "stddev: 0.01744301138583181",
            "extra": "mean: 9.5531 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6427457692016405,
            "unit": "iter/sec",
            "range": "stddev: 0.011323550818872296",
            "extra": "mean: 1.5558 sec\nrounds: 3"
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
          "id": "0454686c33c42b4d91f3a595df49edec4dfcf1b2",
          "message": "Renamed dimension names to be closer to wout convention (#310)",
          "timestamp": "2025-05-05T11:48:07Z",
          "tree_id": "0cf6832aaab95ac244cf9c14fe7872962c46db07",
          "url": "https://github.com/proximafusion/vmecpp/commit/0454686c33c42b4d91f3a595df49edec4dfcf1b2"
        },
        "date": 1776338312763,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.942374432376488,
            "unit": "iter/sec",
            "range": "stddev: 0.014276365441744496",
            "extra": "mean: 339.8616 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9960542265440875,
            "unit": "iter/sec",
            "range": "stddev: 0.004726828944846822",
            "extra": "mean: 333.7723 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7088633193456653,
            "unit": "iter/sec",
            "range": "stddev: 0.0036884604144178834",
            "extra": "mean: 1.4107 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2713253839728362,
            "unit": "iter/sec",
            "range": "stddev: 0.055195222859631604",
            "extra": "mean: 3.6856 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10435560703702407,
            "unit": "iter/sec",
            "range": "stddev: 0.005180776282875042",
            "extra": "mean: 9.5826 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6418197392178897,
            "unit": "iter/sec",
            "range": "stddev: 0.01190719909503164",
            "extra": "mean: 1.5581 sec\nrounds: 3"
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
          "id": "2b0718148ac714f524ccc8847f4c18559e5224a6",
          "message": "Generalized wout handling, supports arbitrary extra fields now (#308)",
          "timestamp": "2025-05-05T12:12:52Z",
          "tree_id": "1beaab4902a49451ad43bed0de80c2e4de9742e8",
          "url": "https://github.com/proximafusion/vmecpp/commit/2b0718148ac714f524ccc8847f4c18559e5224a6"
        },
        "date": 1776338312823,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.955885509809624,
            "unit": "iter/sec",
            "range": "stddev: 0.0038004305398411514",
            "extra": "mean: 338.3081 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9691720656753464,
            "unit": "iter/sec",
            "range": "stddev: 0.002191934134158628",
            "extra": "mean: 336.7942 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7120540142605049,
            "unit": "iter/sec",
            "range": "stddev: 0.0020203256588498014",
            "extra": "mean: 1.4044 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27244500132824323,
            "unit": "iter/sec",
            "range": "stddev: 0.01029548005841246",
            "extra": "mean: 3.6705 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10446151434713954,
            "unit": "iter/sec",
            "range": "stddev: 0.00561926047675551",
            "extra": "mean: 9.5729 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6442039277823942,
            "unit": "iter/sec",
            "range": "stddev: 0.011218328529120513",
            "extra": "mean: 1.5523 sec\nrounds: 3"
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
          "id": "7b352ae15c3f7fab8fbce5f76af0ef74d3c812d8",
          "message": "FORTRAN_MISSING_VARIABLES moved to test cases (#311)",
          "timestamp": "2025-05-05T12:12:52Z",
          "tree_id": "ed17182c155b583cddc69161ac92436d65c553ca",
          "url": "https://github.com/proximafusion/vmecpp/commit/7b352ae15c3f7fab8fbce5f76af0ef74d3c812d8"
        },
        "date": 1776338312923,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.91451196238678,
            "unit": "iter/sec",
            "range": "stddev: 0.014003296042083202",
            "extra": "mean: 343.1106 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9815978683638225,
            "unit": "iter/sec",
            "range": "stddev: 0.003127747373795528",
            "extra": "mean: 335.3906 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7117763542056792,
            "unit": "iter/sec",
            "range": "stddev: 0.0031734190911734926",
            "extra": "mean: 1.4049 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2704816186613637,
            "unit": "iter/sec",
            "range": "stddev: 0.06054248401644706",
            "extra": "mean: 3.6971 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10493427022090258,
            "unit": "iter/sec",
            "range": "stddev: 0.016264051642714272",
            "extra": "mean: 9.5298 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6435482263135305,
            "unit": "iter/sec",
            "range": "stddev: 0.012956189965313558",
            "extra": "mean: 1.5539 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "email": "166746189+jurasic-pf@users.noreply.github.com",
            "username": ""
          },
          "distinct": true,
          "id": "199b0f358221301bb9b82574bbaf161c1b43e905",
          "message": "Added PyPi badge",
          "timestamp": "2025-05-05T14:46:58+02:00",
          "tree_id": "e3df0cb8d105050c9e43283693ca0ded8b147b9c",
          "url": "https://github.com/proximafusion/vmecpp/commit/199b0f358221301bb9b82574bbaf161c1b43e905"
        },
        "date": 1776338312805,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.943308312260781,
            "unit": "iter/sec",
            "range": "stddev: 0.003818959172922271",
            "extra": "mean: 339.7537 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9066725524538097,
            "unit": "iter/sec",
            "range": "stddev: 0.0032997184488068932",
            "extra": "mean: 344.0360 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7061861457468354,
            "unit": "iter/sec",
            "range": "stddev: 0.005236631164097867",
            "extra": "mean: 1.4161 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26973714260841436,
            "unit": "iter/sec",
            "range": "stddev: 0.04801173971448981",
            "extra": "mean: 3.7073 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10442860085783486,
            "unit": "iter/sec",
            "range": "stddev: 0.017597043095577844",
            "extra": "mean: 9.5759 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6429881131518812,
            "unit": "iter/sec",
            "range": "stddev: 0.007933509517622998",
            "extra": "mean: 1.5552 sec\nrounds: 3"
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
          "id": "8f1c3f31a5c688088fc7d0211a0ff49da9fef71d",
          "message": "MODULE.bazel version bump",
          "timestamp": "2025-05-02T17:23:35+02:00",
          "tree_id": "ce11f4055a0b6f199763b87a9864866d86b7761a",
          "url": "https://github.com/proximafusion/vmecpp/commit/8f1c3f31a5c688088fc7d0211a0ff49da9fef71d"
        },
        "date": 1776338312959,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.944454368825961,
            "unit": "iter/sec",
            "range": "stddev: 0.002839304971836695",
            "extra": "mean: 339.6215 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.952927260742005,
            "unit": "iter/sec",
            "range": "stddev: 0.005413307200246998",
            "extra": "mean: 338.6470 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7096387047024858,
            "unit": "iter/sec",
            "range": "stddev: 0.004763646153984046",
            "extra": "mean: 1.4092 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2728610109379583,
            "unit": "iter/sec",
            "range": "stddev: 0.01702170325329811",
            "extra": "mean: 3.6649 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10488247043366876,
            "unit": "iter/sec",
            "range": "stddev: 0.031664833597358076",
            "extra": "mean: 9.5345 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6432565160625761,
            "unit": "iter/sec",
            "range": "stddev: 0.007621588509455559",
            "extra": "mean: 1.5546 sec\nrounds: 3"
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
          "id": "1f71b53d523881adba9c3a9166d08e32d959bb08",
          "message": "Restricted versions of libraries we depend on to fix compatibility issues",
          "timestamp": "2025-05-06T14:45:42+02:00",
          "tree_id": "867bdc7960d7f5ecf3f67195ad32cc00f5d05bc4",
          "url": "https://github.com/proximafusion/vmecpp/commit/1f71b53d523881adba9c3a9166d08e32d959bb08"
        },
        "date": 1776338312813,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8698197957614235,
            "unit": "iter/sec",
            "range": "stddev: 0.01742949472342378",
            "extra": "mean: 348.4539 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9451880818233445,
            "unit": "iter/sec",
            "range": "stddev: 0.006039352985659919",
            "extra": "mean: 339.5369 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7091942857500885,
            "unit": "iter/sec",
            "range": "stddev: 0.008804692327367488",
            "extra": "mean: 1.4101 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2680276100683862,
            "unit": "iter/sec",
            "range": "stddev: 0.11504993552637607",
            "extra": "mean: 3.7310 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10414667772610209,
            "unit": "iter/sec",
            "range": "stddev: 0.05425942261306943",
            "extra": "mean: 9.6018 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6390856724329537,
            "unit": "iter/sec",
            "range": "stddev: 0.022740285169232554",
            "extra": "mean: 1.5647 sec\nrounds: 3"
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
          "id": "c5833daf1a02a8c1b0be25cdbbd31f17483b77df",
          "message": "Output serialization fix",
          "timestamp": "2025-05-06T22:52:01+02:00",
          "tree_id": "a3a6f2857b207a90ccdc507985f32f6e5aeda9be",
          "url": "https://github.com/proximafusion/vmecpp/commit/c5833daf1a02a8c1b0be25cdbbd31f17483b77df"
        },
        "date": 1776338313017,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8200869262122303,
            "unit": "iter/sec",
            "range": "stddev: 0.010888110576866535",
            "extra": "mean: 354.5990 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.962874252312717,
            "unit": "iter/sec",
            "range": "stddev: 0.0045616453293505456",
            "extra": "mean: 337.5101 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.708671441954601,
            "unit": "iter/sec",
            "range": "stddev: 0.003761058548518667",
            "extra": "mean: 1.4111 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2716992016283992,
            "unit": "iter/sec",
            "range": "stddev: 0.014048991253977166",
            "extra": "mean: 3.6805 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10463461228268438,
            "unit": "iter/sec",
            "range": "stddev: 0.018434642309751927",
            "extra": "mean: 9.5571 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6442308036661457,
            "unit": "iter/sec",
            "range": "stddev: 0.011119182704632712",
            "extra": "mean: 1.5522 sec\nrounds: 3"
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
          "id": "50bec9ae1eb82b8ac41a5a27cf23199d9bcb9a64",
          "message": "Enable VmecOutput model serialization",
          "timestamp": "2025-05-07T00:03:37+02:00",
          "tree_id": "60041c1c1600f71a45a503ca03dad36c4d1561ed",
          "url": "https://github.com/proximafusion/vmecpp/commit/50bec9ae1eb82b8ac41a5a27cf23199d9bcb9a64"
        },
        "date": 1776338312873,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9456883988365234,
            "unit": "iter/sec",
            "range": "stddev: 0.006501962185279586",
            "extra": "mean: 339.4792 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.962786610881286,
            "unit": "iter/sec",
            "range": "stddev: 0.003822100840613719",
            "extra": "mean: 337.5201 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7120266368595399,
            "unit": "iter/sec",
            "range": "stddev: 0.0009624175534517624",
            "extra": "mean: 1.4044 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26872107865373906,
            "unit": "iter/sec",
            "range": "stddev: 0.03702553859712977",
            "extra": "mean: 3.7213 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10443062381805736,
            "unit": "iter/sec",
            "range": "stddev: 0.02123881824298768",
            "extra": "mean: 9.5757 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6357192780844205,
            "unit": "iter/sec",
            "range": "stddev: 0.04344994454384183",
            "extra": "mean: 1.5730 sec\nrounds: 3"
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
          "id": "d8eaf5193e640661737c73f2d26628306f75e14b",
          "message": "Update README.md",
          "timestamp": "2025-05-07T13:59:33+02:00",
          "tree_id": "6ea027adb4d0301646789242cdb9e8402eff2255",
          "url": "https://github.com/proximafusion/vmecpp/commit/d8eaf5193e640661737c73f2d26628306f75e14b"
        },
        "date": 1776338313051,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.89460111943833,
            "unit": "iter/sec",
            "range": "stddev: 0.014045215327120676",
            "extra": "mean: 345.4707 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.923555209333516,
            "unit": "iter/sec",
            "range": "stddev: 0.014241172273225898",
            "extra": "mean: 342.0493 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7122590280270881,
            "unit": "iter/sec",
            "range": "stddev: 0.00357051968014006",
            "extra": "mean: 1.4040 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2677455676612222,
            "unit": "iter/sec",
            "range": "stddev: 0.034942776168903913",
            "extra": "mean: 3.7349 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10135142089006766,
            "unit": "iter/sec",
            "range": "stddev: 0.46017195939015765",
            "extra": "mean: 9.8667 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6434335262898,
            "unit": "iter/sec",
            "range": "stddev: 0.015588329239782279",
            "extra": "mean: 1.5542 sec\nrounds: 3"
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
          "id": "29d23f801ca3199fdb159c6d17aa7f59594cd6b5",
          "message": "Remove internal references (#323)",
          "timestamp": "2025-05-07T13:21:31Z",
          "tree_id": "4f0893027461406316f54f8100fd927cd9923d05",
          "url": "https://github.com/proximafusion/vmecpp/commit/29d23f801ca3199fdb159c6d17aa7f59594cd6b5"
        },
        "date": 1776338312821,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.94066986755185,
            "unit": "iter/sec",
            "range": "stddev: 0.003351955250748748",
            "extra": "mean: 340.0586 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9797517282422517,
            "unit": "iter/sec",
            "range": "stddev: 0.003263893483558816",
            "extra": "mean: 335.5984 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7053044984591526,
            "unit": "iter/sec",
            "range": "stddev: 0.015130743502490659",
            "extra": "mean: 1.4178 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2716107308164167,
            "unit": "iter/sec",
            "range": "stddev: 0.008228962482729904",
            "extra": "mean: 3.6817 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10436185956268673,
            "unit": "iter/sec",
            "range": "stddev: 0.022171418747572776",
            "extra": "mean: 9.5820 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6231243497518291,
            "unit": "iter/sec",
            "range": "stddev: 0.06281455366244813",
            "extra": "mean: 1.6048 sec\nrounds: 3"
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
          "id": "ecd2e236cebffdfcf0263085d4ff56f0ecad0d42",
          "message": "Show constructing VmecInput completely in Python",
          "timestamp": "2025-05-07T01:07:17+02:00",
          "tree_id": "91b7e895b513520721a3e18fad14472d69e7cd6a",
          "url": "https://github.com/proximafusion/vmecpp/commit/ecd2e236cebffdfcf0263085d4ff56f0ecad0d42"
        },
        "date": 1776338313078,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8948365587134552,
            "unit": "iter/sec",
            "range": "stddev: 0.005206503229346773",
            "extra": "mean: 345.4426 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9022385613226827,
            "unit": "iter/sec",
            "range": "stddev: 0.0024719045290696206",
            "extra": "mean: 344.5616 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7128704852694628,
            "unit": "iter/sec",
            "range": "stddev: 0.005924780313620389",
            "extra": "mean: 1.4028 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2698710493947737,
            "unit": "iter/sec",
            "range": "stddev: 0.021441540908742732",
            "extra": "mean: 3.7055 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10411630672256986,
            "unit": "iter/sec",
            "range": "stddev: 0.01985295884782617",
            "extra": "mean: 9.6046 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6392425487873135,
            "unit": "iter/sec",
            "range": "stddev: 0.01622259865739257",
            "extra": "mean: 1.5644 sec\nrounds: 3"
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
          "id": "07eba36ff16696b838fbfbed50f2cea6209bcf84",
          "message": "Resize sparse coefficients when too small",
          "timestamp": "2025-05-07T01:08:50+02:00",
          "tree_id": "8ef02376e936f5f13e4f1c8cdae7217e3f79cf4d",
          "url": "https://github.com/proximafusion/vmecpp/commit/07eba36ff16696b838fbfbed50f2cea6209bcf84"
        },
        "date": 1776338312775,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.847255771239944,
            "unit": "iter/sec",
            "range": "stddev: 0.004857796604435802",
            "extra": "mean: 351.2154 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.938567173323002,
            "unit": "iter/sec",
            "range": "stddev: 0.0052455799700204244",
            "extra": "mean: 340.3019 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.70919251501225,
            "unit": "iter/sec",
            "range": "stddev: 0.005128852062480883",
            "extra": "mean: 1.4101 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2700353710016249,
            "unit": "iter/sec",
            "range": "stddev: 0.04606577913461539",
            "extra": "mean: 3.7032 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10485461174152091,
            "unit": "iter/sec",
            "range": "stddev: 0.01079133423771918",
            "extra": "mean: 9.5370 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6430976914815774,
            "unit": "iter/sec",
            "range": "stddev: 0.013079238633017569",
            "extra": "mean: 1.5550 sec\nrounds: 3"
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
          "id": "365c07eaab0152eb00387f41e821e4686ae3fe6e",
          "message": "Converted ivac counter to an enum with descriptive states (#280)",
          "timestamp": "2025-05-08T16:25:43Z",
          "tree_id": "d76455baa793c81b29b9a6bab932bbaff47f7e26",
          "url": "https://github.com/proximafusion/vmecpp/commit/365c07eaab0152eb00387f41e821e4686ae3fe6e"
        },
        "date": 1776338312838,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.956647168638955,
            "unit": "iter/sec",
            "range": "stddev: 0.0028220848329787588",
            "extra": "mean: 338.2209 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.96374090850841,
            "unit": "iter/sec",
            "range": "stddev: 0.0049738820678254",
            "extra": "mean: 337.4114 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7129543748598125,
            "unit": "iter/sec",
            "range": "stddev: 0.006462072987358476",
            "extra": "mean: 1.4026 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2624102912697823,
            "unit": "iter/sec",
            "range": "stddev: 0.2986457646028822",
            "extra": "mean: 3.8108 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10452507781610547,
            "unit": "iter/sec",
            "range": "stddev: 0.01818566669856406",
            "extra": "mean: 9.5671 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6422098017990502,
            "unit": "iter/sec",
            "range": "stddev: 0.012403707081243245",
            "extra": "mean: 1.5571 sec\nrounds: 3"
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
        "date": 1776338312912,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.918356602586808,
            "unit": "iter/sec",
            "range": "stddev: 0.003405136678613861",
            "extra": "mean: 342.6586 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9888683376695306,
            "unit": "iter/sec",
            "range": "stddev: 0.0011646960016879508",
            "extra": "mean: 334.5748 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7134415503688893,
            "unit": "iter/sec",
            "range": "stddev: 0.00303533069079234",
            "extra": "mean: 1.4017 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27140903868095506,
            "unit": "iter/sec",
            "range": "stddev: 0.033360789863743426",
            "extra": "mean: 3.6845 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10487540087545101,
            "unit": "iter/sec",
            "range": "stddev: 0.032152628246928085",
            "extra": "mean: 9.5351 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6415048426757498,
            "unit": "iter/sec",
            "range": "stddev: 0.007205265279539715",
            "extra": "mean: 1.5588 sec\nrounds: 3"
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
        "date": 1776338313050,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.951779420274917,
            "unit": "iter/sec",
            "range": "stddev: 0.004087985696758721",
            "extra": "mean: 338.7787 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9621983769942717,
            "unit": "iter/sec",
            "range": "stddev: 0.0050057755011435395",
            "extra": "mean: 337.5871 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7112153594780515,
            "unit": "iter/sec",
            "range": "stddev: 0.0019478000769872462",
            "extra": "mean: 1.4060 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27319576974513277,
            "unit": "iter/sec",
            "range": "stddev: 0.029362521505824093",
            "extra": "mean: 3.6604 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10431927370943238,
            "unit": "iter/sec",
            "range": "stddev: 0.011800779078727818",
            "extra": "mean: 9.5860 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6438977709728173,
            "unit": "iter/sec",
            "range": "stddev: 0.00836504970600146",
            "extra": "mean: 1.5530 sec\nrounds: 3"
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
        "date": 1776338312860,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9301294035734053,
            "unit": "iter/sec",
            "range": "stddev: 0.0034874829258953665",
            "extra": "mean: 341.2819 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.940038336783392,
            "unit": "iter/sec",
            "range": "stddev: 0.007527959874002897",
            "extra": "mean: 340.1316 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7100704332307403,
            "unit": "iter/sec",
            "range": "stddev: 0.004966047952116398",
            "extra": "mean: 1.4083 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.272491384020913,
            "unit": "iter/sec",
            "range": "stddev: 0.03286745829195546",
            "extra": "mean: 3.6698 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10419235422073257,
            "unit": "iter/sec",
            "range": "stddev: 0.008944400035794659",
            "extra": "mean: 9.5976 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6426683152462975,
            "unit": "iter/sec",
            "range": "stddev: 0.01110280468000244",
            "extra": "mean: 1.5560 sec\nrounds: 3"
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
        "date": 1776338313002,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.966148729738619,
            "unit": "iter/sec",
            "range": "stddev: 0.0035222516103644696",
            "extra": "mean: 337.1375 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9778823831086254,
            "unit": "iter/sec",
            "range": "stddev: 0.002472648887385487",
            "extra": "mean: 335.8091 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7113421049483898,
            "unit": "iter/sec",
            "range": "stddev: 0.003069421952964061",
            "extra": "mean: 1.4058 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27225271150086844,
            "unit": "iter/sec",
            "range": "stddev: 0.02249682172918081",
            "extra": "mean: 3.6731 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10499886092930714,
            "unit": "iter/sec",
            "range": "stddev: 0.011983154123792182",
            "extra": "mean: 9.5239 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6439016937066925,
            "unit": "iter/sec",
            "range": "stddev: 0.008929978470875512",
            "extra": "mean: 1.5530 sec\nrounds: 3"
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
        "date": 1776338312926,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9467154653825767,
            "unit": "iter/sec",
            "range": "stddev: 0.0030259684132426037",
            "extra": "mean: 339.3609 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.000025462416411,
            "unit": "iter/sec",
            "range": "stddev: 0.0017756840485153309",
            "extra": "mean: 333.3305 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7143732173507996,
            "unit": "iter/sec",
            "range": "stddev: 0.0009912081526400383",
            "extra": "mean: 1.3998 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27050132270358085,
            "unit": "iter/sec",
            "range": "stddev: 0.023720845903519572",
            "extra": "mean: 3.6968 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10499787630613056,
            "unit": "iter/sec",
            "range": "stddev: 0.00670229510575928",
            "extra": "mean: 9.5240 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6427222146675942,
            "unit": "iter/sec",
            "range": "stddev: 0.006094993344638243",
            "extra": "mean: 1.5559 sec\nrounds: 3"
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
        "date": 1776338312911,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.946553959295969,
            "unit": "iter/sec",
            "range": "stddev: 0.004126569146288722",
            "extra": "mean: 339.3795 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.915220326260054,
            "unit": "iter/sec",
            "range": "stddev: 0.004804803960084789",
            "extra": "mean: 343.0272 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7087729523459598,
            "unit": "iter/sec",
            "range": "stddev: 0.0001293326867474969",
            "extra": "mean: 1.4109 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27239787354332995,
            "unit": "iter/sec",
            "range": "stddev: 0.04182530986025014",
            "extra": "mean: 3.6711 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1048367326344706,
            "unit": "iter/sec",
            "range": "stddev: 0.006628003806832814",
            "extra": "mean: 9.5386 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6398223112093244,
            "unit": "iter/sec",
            "range": "stddev: 0.014634226697673914",
            "extra": "mean: 1.5629 sec\nrounds: 3"
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
        "date": 1776338312878,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9743397499355755,
            "unit": "iter/sec",
            "range": "stddev: 0.005339206651143382",
            "extra": "mean: 336.2091 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9842207780525096,
            "unit": "iter/sec",
            "range": "stddev: 0.0030713793199821727",
            "extra": "mean: 335.0959 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7074771903056362,
            "unit": "iter/sec",
            "range": "stddev: 0.006090103125901026",
            "extra": "mean: 1.4135 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2712252336216545,
            "unit": "iter/sec",
            "range": "stddev: 0.03610923365422061",
            "extra": "mean: 3.6870 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10467509873232814,
            "unit": "iter/sec",
            "range": "stddev: 0.026153692337962893",
            "extra": "mean: 9.5534 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6428738646492654,
            "unit": "iter/sec",
            "range": "stddev: 0.009764794686083662",
            "extra": "mean: 1.5555 sec\nrounds: 3"
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
        "date": 1776338312840,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.95964654878058,
            "unit": "iter/sec",
            "range": "stddev: 0.0013967166576528555",
            "extra": "mean: 337.8782 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9838023406730763,
            "unit": "iter/sec",
            "range": "stddev: 0.0018716676518944846",
            "extra": "mean: 335.1428 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7120884916621951,
            "unit": "iter/sec",
            "range": "stddev: 0.005045131957996551",
            "extra": "mean: 1.4043 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2715479400483286,
            "unit": "iter/sec",
            "range": "stddev: 0.025830452019362113",
            "extra": "mean: 3.6826 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1047340476049102,
            "unit": "iter/sec",
            "range": "stddev: 0.016058497513198076",
            "extra": "mean: 9.5480 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6427576042667552,
            "unit": "iter/sec",
            "range": "stddev: 0.009939807665091185",
            "extra": "mean: 1.5558 sec\nrounds: 3"
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
        "date": 1776338312998,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.836109801731378,
            "unit": "iter/sec",
            "range": "stddev: 0.0037616584264076017",
            "extra": "mean: 352.5957 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9180411129080013,
            "unit": "iter/sec",
            "range": "stddev: 0.004132173891202657",
            "extra": "mean: 342.6957 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7135940820809242,
            "unit": "iter/sec",
            "range": "stddev: 0.001150871270628037",
            "extra": "mean: 1.4014 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.271947690271468,
            "unit": "iter/sec",
            "range": "stddev: 0.012380370860106767",
            "extra": "mean: 3.6772 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10503243068874474,
            "unit": "iter/sec",
            "range": "stddev: 0.008648399770789248",
            "extra": "mean: 9.5209 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6372292280784796,
            "unit": "iter/sec",
            "range": "stddev: 0.004765439187284351",
            "extra": "mean: 1.5693 sec\nrounds: 3"
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
        "date": 1776338312986,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.826186996291621,
            "unit": "iter/sec",
            "range": "stddev: 0.002913920688069948",
            "extra": "mean: 353.8336 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.839103603300548,
            "unit": "iter/sec",
            "range": "stddev: 0.006406639110717285",
            "extra": "mean: 352.2238 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7075604584308733,
            "unit": "iter/sec",
            "range": "stddev: 0.010660949925888662",
            "extra": "mean: 1.4133 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.269668460399095,
            "unit": "iter/sec",
            "range": "stddev: 0.01629500689187062",
            "extra": "mean: 3.7083 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10257325390083474,
            "unit": "iter/sec",
            "range": "stddev: 0.09976425448243925",
            "extra": "mean: 9.7491 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6413420797559384,
            "unit": "iter/sec",
            "range": "stddev: 0.0075934089127890885",
            "extra": "mean: 1.5592 sec\nrounds: 3"
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
        "date": 1776338312844,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8594457076584527,
            "unit": "iter/sec",
            "range": "stddev: 0.005211922517355227",
            "extra": "mean: 349.7181 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.938590084354111,
            "unit": "iter/sec",
            "range": "stddev: 0.002271858958283518",
            "extra": "mean: 340.2992 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7117586524726984,
            "unit": "iter/sec",
            "range": "stddev: 0.005500657204084884",
            "extra": "mean: 1.4050 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27121101147624876,
            "unit": "iter/sec",
            "range": "stddev: 0.054039549682061656",
            "extra": "mean: 3.6872 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10352763833609932,
            "unit": "iter/sec",
            "range": "stddev: 0.1286392008109983",
            "extra": "mean: 9.6593 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6426487904731514,
            "unit": "iter/sec",
            "range": "stddev: 0.017552225353002156",
            "extra": "mean: 1.5561 sec\nrounds: 3"
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
        "date": 1776338312943,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.806431312065947,
            "unit": "iter/sec",
            "range": "stddev: 0.00410504107160283",
            "extra": "mean: 356.3244 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.903737184731057,
            "unit": "iter/sec",
            "range": "stddev: 0.003201749249566183",
            "extra": "mean: 344.3838 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7111114258121428,
            "unit": "iter/sec",
            "range": "stddev: 0.00261893527893214",
            "extra": "mean: 1.4062 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2635957309455334,
            "unit": "iter/sec",
            "range": "stddev: 0.14078550965837586",
            "extra": "mean: 3.7937 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.104419975855601,
            "unit": "iter/sec",
            "range": "stddev: 0.015292098029469146",
            "extra": "mean: 9.5767 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6429271621255949,
            "unit": "iter/sec",
            "range": "stddev: 0.011219773579172167",
            "extra": "mean: 1.5554 sec\nrounds: 3"
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
        "date": 1776338312756,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8785254819998918,
            "unit": "iter/sec",
            "range": "stddev: 0.005620179989804964",
            "extra": "mean: 347.4001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9274788829969145,
            "unit": "iter/sec",
            "range": "stddev: 0.005861938226473355",
            "extra": "mean: 341.5909 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7103214481673399,
            "unit": "iter/sec",
            "range": "stddev: 0.0018925004902759696",
            "extra": "mean: 1.4078 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27174523928745364,
            "unit": "iter/sec",
            "range": "stddev: 0.02226298919795643",
            "extra": "mean: 3.6799 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10480779913724025,
            "unit": "iter/sec",
            "range": "stddev: 0.01062853957447791",
            "extra": "mean: 9.5413 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6413204524747562,
            "unit": "iter/sec",
            "range": "stddev: 0.006419516168932661",
            "extra": "mean: 1.5593 sec\nrounds: 3"
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
        "date": 1776338313080,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.920724164909778,
            "unit": "iter/sec",
            "range": "stddev: 0.0031551154553566578",
            "extra": "mean: 342.3808 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.947570008282624,
            "unit": "iter/sec",
            "range": "stddev: 0.0016609500206462846",
            "extra": "mean: 339.2625 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7171205652998133,
            "unit": "iter/sec",
            "range": "stddev: 0.002723134922385687",
            "extra": "mean: 1.3945 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2724842502420776,
            "unit": "iter/sec",
            "range": "stddev: 0.016095519369774134",
            "extra": "mean: 3.6699 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10427732211118451,
            "unit": "iter/sec",
            "range": "stddev: 0.03319451372577344",
            "extra": "mean: 9.5898 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6453563396305925,
            "unit": "iter/sec",
            "range": "stddev: 0.012590769914050802",
            "extra": "mean: 1.5495 sec\nrounds: 3"
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
        "date": 1776338313009,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9058936314660575,
            "unit": "iter/sec",
            "range": "stddev: 0.013289613790969917",
            "extra": "mean: 344.1282 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.931737006911877,
            "unit": "iter/sec",
            "range": "stddev: 0.0022030966466086385",
            "extra": "mean: 341.0947 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7135809234899038,
            "unit": "iter/sec",
            "range": "stddev: 0.0004950301233297558",
            "extra": "mean: 1.4014 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2720592686630445,
            "unit": "iter/sec",
            "range": "stddev: 0.034402447142781734",
            "extra": "mean: 3.6757 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09245323617183028,
            "unit": "iter/sec",
            "range": "stddev: 2.1769194250710937",
            "extra": "mean: 10.8163 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6403057974817947,
            "unit": "iter/sec",
            "range": "stddev: 0.016955594887157012",
            "extra": "mean: 1.5618 sec\nrounds: 3"
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
        "date": 1776338313028,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8672186238376374,
            "unit": "iter/sec",
            "range": "stddev: 0.003625934990784146",
            "extra": "mean: 348.7701 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.8763661876484226,
            "unit": "iter/sec",
            "range": "stddev: 0.0031743031541265057",
            "extra": "mean: 347.6609 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7130302376365237,
            "unit": "iter/sec",
            "range": "stddev: 0.0024277700666691816",
            "extra": "mean: 1.4025 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2717975618882417,
            "unit": "iter/sec",
            "range": "stddev: 0.02673156353437678",
            "extra": "mean: 3.6792 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1040041322120281,
            "unit": "iter/sec",
            "range": "stddev: 0.015603251241545718",
            "extra": "mean: 9.6150 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.643822984208036,
            "unit": "iter/sec",
            "range": "stddev: 0.0115472745190672",
            "extra": "mean: 1.5532 sec\nrounds: 3"
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
        "date": 1776338312862,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.909468142055124,
            "unit": "iter/sec",
            "range": "stddev: 0.007475945251330393",
            "extra": "mean: 343.7054 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9309985848979734,
            "unit": "iter/sec",
            "range": "stddev: 0.0017449013596332525",
            "extra": "mean: 341.1806 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7133808116180544,
            "unit": "iter/sec",
            "range": "stddev: 0.000576369711698226",
            "extra": "mean: 1.4018 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2703594370046996,
            "unit": "iter/sec",
            "range": "stddev: 0.06487111427901819",
            "extra": "mean: 3.6988 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10521345204756277,
            "unit": "iter/sec",
            "range": "stddev: 0.04382804007875021",
            "extra": "mean: 9.5045 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6413314140031204,
            "unit": "iter/sec",
            "range": "stddev: 0.012311314734466993",
            "extra": "mean: 1.5593 sec\nrounds: 3"
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
        "date": 1776338313091,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9449779902294617,
            "unit": "iter/sec",
            "range": "stddev: 0.00395666804542043",
            "extra": "mean: 339.5611 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9694781458291533,
            "unit": "iter/sec",
            "range": "stddev: 0.001053611616635969",
            "extra": "mean: 336.7595 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.71502638338788,
            "unit": "iter/sec",
            "range": "stddev: 0.003050704997649719",
            "extra": "mean: 1.3985 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27247868311727974,
            "unit": "iter/sec",
            "range": "stddev: 0.01151106472736913",
            "extra": "mean: 3.6700 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1049106299589574,
            "unit": "iter/sec",
            "range": "stddev: 0.04285335414990936",
            "extra": "mean: 9.5319 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6432882593014979,
            "unit": "iter/sec",
            "range": "stddev: 0.009442862624343216",
            "extra": "mean: 1.5545 sec\nrounds: 3"
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
        "date": 1776338312982,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.944194751671635,
            "unit": "iter/sec",
            "range": "stddev: 0.004451792821031945",
            "extra": "mean: 339.6514 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9776734834271212,
            "unit": "iter/sec",
            "range": "stddev: 0.0017878396852806459",
            "extra": "mean: 335.8327 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7094471989175773,
            "unit": "iter/sec",
            "range": "stddev: 0.01951957836826113",
            "extra": "mean: 1.4095 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2728121994873843,
            "unit": "iter/sec",
            "range": "stddev: 0.01933998028297991",
            "extra": "mean: 3.6655 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10466437494472265,
            "unit": "iter/sec",
            "range": "stddev: 0.03785568866376054",
            "extra": "mean: 9.5543 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6415456469464431,
            "unit": "iter/sec",
            "range": "stddev: 0.012002382651120804",
            "extra": "mean: 1.5587 sec\nrounds: 3"
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
        "date": 1776338312758,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0767716789291284,
            "unit": "iter/sec",
            "range": "stddev: 0.00043652894405203944",
            "extra": "mean: 325.0160 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0953849279170225,
            "unit": "iter/sec",
            "range": "stddev: 0.0012349976322905327",
            "extra": "mean: 323.0616 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7155465207143547,
            "unit": "iter/sec",
            "range": "stddev: 0.0034972643304026847",
            "extra": "mean: 1.3975 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27135010527627124,
            "unit": "iter/sec",
            "range": "stddev: 0.07812153850267105",
            "extra": "mean: 3.6853 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10529259720246571,
            "unit": "iter/sec",
            "range": "stddev: 0.007738769976308883",
            "extra": "mean: 9.4973 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6362312601404422,
            "unit": "iter/sec",
            "range": "stddev: 0.023444912996834478",
            "extra": "mean: 1.5718 sec\nrounds: 3"
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
        "date": 1776338312788,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0745937741407596,
            "unit": "iter/sec",
            "range": "stddev: 0.0014970057055340587",
            "extra": "mean: 325.2462 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.081528948165563,
            "unit": "iter/sec",
            "range": "stddev: 0.0013215286357222465",
            "extra": "mean: 324.5142 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7163498685862938,
            "unit": "iter/sec",
            "range": "stddev: 0.001088659184387456",
            "extra": "mean: 1.3960 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27535200238405616,
            "unit": "iter/sec",
            "range": "stddev: 0.00811850468225135",
            "extra": "mean: 3.6317 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1060718001149946,
            "unit": "iter/sec",
            "range": "stddev: 0.06267389155950243",
            "extra": "mean: 9.4276 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6458959540495252,
            "unit": "iter/sec",
            "range": "stddev: 0.009624635931222276",
            "extra": "mean: 1.5482 sec\nrounds: 3"
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
        "date": 1776338312919,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.049660517555563,
            "unit": "iter/sec",
            "range": "stddev: 0.005786604992355761",
            "extra": "mean: 327.9054 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0814475958195837,
            "unit": "iter/sec",
            "range": "stddev: 0.001488873360800749",
            "extra": "mean: 324.5228 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7134774209816102,
            "unit": "iter/sec",
            "range": "stddev: 0.0007191517973077475",
            "extra": "mean: 1.4016 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2755558151683193,
            "unit": "iter/sec",
            "range": "stddev: 0.01035725163477256",
            "extra": "mean: 3.6290 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10555593507331755,
            "unit": "iter/sec",
            "range": "stddev: 0.00731958847020148",
            "extra": "mean: 9.4737 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6465686331617663,
            "unit": "iter/sec",
            "range": "stddev: 0.010001337513870853",
            "extra": "mean: 1.5466 sec\nrounds: 3"
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
        "date": 1776338312876,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0789983369604808,
            "unit": "iter/sec",
            "range": "stddev: 0.0022807464752766507",
            "extra": "mean: 324.7809 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0754806878348107,
            "unit": "iter/sec",
            "range": "stddev: 0.0023708363920893444",
            "extra": "mean: 325.1524 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7143559062168062,
            "unit": "iter/sec",
            "range": "stddev: 0.008848508129172655",
            "extra": "mean: 1.3999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27462946844960207,
            "unit": "iter/sec",
            "range": "stddev: 0.015070465877304736",
            "extra": "mean: 3.6413 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10525027704823503,
            "unit": "iter/sec",
            "range": "stddev: 0.00935418127242801",
            "extra": "mean: 9.5012 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6363938484287646,
            "unit": "iter/sec",
            "range": "stddev: 0.025790330694460375",
            "extra": "mean: 1.5714 sec\nrounds: 3"
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
        "date": 1776338312904,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0555231890763483,
            "unit": "iter/sec",
            "range": "stddev: 0.004575273523010026",
            "extra": "mean: 327.2762 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0858210466591336,
            "unit": "iter/sec",
            "range": "stddev: 0.0012865379748534542",
            "extra": "mean: 324.0629 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7182082973094561,
            "unit": "iter/sec",
            "range": "stddev: 0.002956522729010991",
            "extra": "mean: 1.3924 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2708842593634173,
            "unit": "iter/sec",
            "range": "stddev: 0.09681172579630609",
            "extra": "mean: 3.6916 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10534463554615649,
            "unit": "iter/sec",
            "range": "stddev: 0.017184570462413704",
            "extra": "mean: 9.4927 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6227167841319913,
            "unit": "iter/sec",
            "range": "stddev: 0.03410028246634357",
            "extra": "mean: 1.6059 sec\nrounds: 3"
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
        "date": 1776338312793,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0856825892305015,
            "unit": "iter/sec",
            "range": "stddev: 0.0013259328990669445",
            "extra": "mean: 324.0774 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0957445580167695,
            "unit": "iter/sec",
            "range": "stddev: 0.0010376580473286262",
            "extra": "mean: 323.0241 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7171094214174919,
            "unit": "iter/sec",
            "range": "stddev: 0.002520055658987017",
            "extra": "mean: 1.3945 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.276339015342507,
            "unit": "iter/sec",
            "range": "stddev: 0.016729748841631226",
            "extra": "mean: 3.6187 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10592991697186872,
            "unit": "iter/sec",
            "range": "stddev: 0.005505506318025701",
            "extra": "mean: 9.4402 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6445338291550339,
            "unit": "iter/sec",
            "range": "stddev: 0.014108520772556985",
            "extra": "mean: 1.5515 sec\nrounds: 3"
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
        "date": 1776338312816,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.065323977763741,
            "unit": "iter/sec",
            "range": "stddev: 0.0022902286070712943",
            "extra": "mean: 326.2298 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.072839755088327,
            "unit": "iter/sec",
            "range": "stddev: 0.0018428925638950481",
            "extra": "mean: 325.4319 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7119127509404408,
            "unit": "iter/sec",
            "range": "stddev: 0.017185352822473163",
            "extra": "mean: 1.4047 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.275434485635077,
            "unit": "iter/sec",
            "range": "stddev: 0.004247006399065602",
            "extra": "mean: 3.6306 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10542904634117893,
            "unit": "iter/sec",
            "range": "stddev: 0.04338018858851916",
            "extra": "mean: 9.4851 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6427599878025767,
            "unit": "iter/sec",
            "range": "stddev: 0.019148761182206184",
            "extra": "mean: 1.5558 sec\nrounds: 3"
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
        "date": 1776338313064,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0539310729101006,
            "unit": "iter/sec",
            "range": "stddev: 0.004213404855853626",
            "extra": "mean: 327.4468 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0678013970732887,
            "unit": "iter/sec",
            "range": "stddev: 0.0017510385764559172",
            "extra": "mean: 325.9663 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7167798768395186,
            "unit": "iter/sec",
            "range": "stddev: 0.0006110750758291595",
            "extra": "mean: 1.3951 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2747943557363398,
            "unit": "iter/sec",
            "range": "stddev: 0.04662341456052513",
            "extra": "mean: 3.6391 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10578631388031082,
            "unit": "iter/sec",
            "range": "stddev: 0.03461558230901231",
            "extra": "mean: 9.4530 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6392394181440985,
            "unit": "iter/sec",
            "range": "stddev: 0.020007176339876368",
            "extra": "mean: 1.5644 sec\nrounds: 3"
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
        "date": 1776338312899,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0711550395315914,
            "unit": "iter/sec",
            "range": "stddev: 0.0012570179195196321",
            "extra": "mean: 325.6104 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0753289210361876,
            "unit": "iter/sec",
            "range": "stddev: 0.0011680142773875639",
            "extra": "mean: 325.1685 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7064220013405893,
            "unit": "iter/sec",
            "range": "stddev: 0.01652653389461303",
            "extra": "mean: 1.4156 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2707636514051079,
            "unit": "iter/sec",
            "range": "stddev: 0.00620425797548146",
            "extra": "mean: 3.6933 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10572854818298273,
            "unit": "iter/sec",
            "range": "stddev: 0.046069141877628575",
            "extra": "mean: 9.4582 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6424584903885829,
            "unit": "iter/sec",
            "range": "stddev: 0.019174763677202576",
            "extra": "mean: 1.5565 sec\nrounds: 3"
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
        "date": 1776338313059,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.033912039676161,
            "unit": "iter/sec",
            "range": "stddev: 0.0033263836908126454",
            "extra": "mean: 329.6074 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0605935034725444,
            "unit": "iter/sec",
            "range": "stddev: 0.0002739663718261359",
            "extra": "mean: 326.7340 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7157991331648855,
            "unit": "iter/sec",
            "range": "stddev: 0.0023952311485414533",
            "extra": "mean: 1.3970 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.269224194665483,
            "unit": "iter/sec",
            "range": "stddev: 0.02923089652258981",
            "extra": "mean: 3.7144 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10526620066043625,
            "unit": "iter/sec",
            "range": "stddev: 0.007860855803060318",
            "extra": "mean: 9.4997 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.642576181745964,
            "unit": "iter/sec",
            "range": "stddev: 0.01628780026486922",
            "extra": "mean: 1.5562 sec\nrounds: 3"
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
        "date": 1776338313082,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0337117271868066,
            "unit": "iter/sec",
            "range": "stddev: 0.0016720121464923817",
            "extra": "mean: 329.6292 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0609112036610138,
            "unit": "iter/sec",
            "range": "stddev: 0.0013124017862430363",
            "extra": "mean: 326.7001 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7143472413538354,
            "unit": "iter/sec",
            "range": "stddev: 0.003075329101708675",
            "extra": "mean: 1.3999 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2691126753572749,
            "unit": "iter/sec",
            "range": "stddev: 0.05666816529274826",
            "extra": "mean: 3.7159 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10542155277905599,
            "unit": "iter/sec",
            "range": "stddev: 0.02783255126350752",
            "extra": "mean: 9.4857 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6447065133775013,
            "unit": "iter/sec",
            "range": "stddev: 0.010870039724056644",
            "extra": "mean: 1.5511 sec\nrounds: 3"
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
        "date": 1776338312783,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0318571952136293,
            "unit": "iter/sec",
            "range": "stddev: 0.0028229557081374027",
            "extra": "mean: 329.8308 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.014116813822637,
            "unit": "iter/sec",
            "range": "stddev: 0.008396327838035164",
            "extra": "mean: 331.7721 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7199862631228191,
            "unit": "iter/sec",
            "range": "stddev: 0.0034252620108960598",
            "extra": "mean: 1.3889 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2714571181099019,
            "unit": "iter/sec",
            "range": "stddev: 0.029645725351085037",
            "extra": "mean: 3.6838 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10541054628629344,
            "unit": "iter/sec",
            "range": "stddev: 0.00879732709977966",
            "extra": "mean: 9.4867 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6436531390393939,
            "unit": "iter/sec",
            "range": "stddev: 0.011382028119798769",
            "extra": "mean: 1.5536 sec\nrounds: 3"
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
        "date": 1776338312871,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0442544239790426,
            "unit": "iter/sec",
            "range": "stddev: 0.001040021889052509",
            "extra": "mean: 328.4877 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.046803574878544,
            "unit": "iter/sec",
            "range": "stddev: 0.0017806119833106337",
            "extra": "mean: 328.2128 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7165115993656955,
            "unit": "iter/sec",
            "range": "stddev: 0.000655185998153184",
            "extra": "mean: 1.3957 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2705033004316713,
            "unit": "iter/sec",
            "range": "stddev: 0.04154730545553102",
            "extra": "mean: 3.6968 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10491234701190563,
            "unit": "iter/sec",
            "range": "stddev: 0.015693846632414913",
            "extra": "mean: 9.5318 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6451438853118155,
            "unit": "iter/sec",
            "range": "stddev: 0.009474995289222037",
            "extra": "mean: 1.5500 sec\nrounds: 3"
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
        "date": 1776338313020,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.030680517174072,
            "unit": "iter/sec",
            "range": "stddev: 0.0011499727196899472",
            "extra": "mean: 329.9589 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.040277225416772,
            "unit": "iter/sec",
            "range": "stddev: 0.0007860149521735401",
            "extra": "mean: 328.9174 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7154951948587694,
            "unit": "iter/sec",
            "range": "stddev: 0.0013095378771519307",
            "extra": "mean: 1.3976 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2706914703682759,
            "unit": "iter/sec",
            "range": "stddev: 0.024913527167646495",
            "extra": "mean: 3.6942 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10482622715852564,
            "unit": "iter/sec",
            "range": "stddev: 0.050903427171698816",
            "extra": "mean: 9.5396 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6410194073858476,
            "unit": "iter/sec",
            "range": "stddev: 0.025170731991037493",
            "extra": "mean: 1.5600 sec\nrounds: 3"
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
        "date": 1776338313073,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.001795778096658,
            "unit": "iter/sec",
            "range": "stddev: 0.008901814578686825",
            "extra": "mean: 333.1339 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.028583707003759,
            "unit": "iter/sec",
            "range": "stddev: 0.0032650373322018404",
            "extra": "mean: 330.1873 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7159890621520402,
            "unit": "iter/sec",
            "range": "stddev: 0.004652479024062491",
            "extra": "mean: 1.3967 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2697660341468529,
            "unit": "iter/sec",
            "range": "stddev: 0.0210839643200361",
            "extra": "mean: 3.7069 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10520671017202587,
            "unit": "iter/sec",
            "range": "stddev: 0.012610303563481497",
            "extra": "mean: 9.5051 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6454187731327238,
            "unit": "iter/sec",
            "range": "stddev: 0.00822030010410378",
            "extra": "mean: 1.5494 sec\nrounds: 3"
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
        "date": 1776338312954,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.012716818081804,
            "unit": "iter/sec",
            "range": "stddev: 0.001987992217641316",
            "extra": "mean: 331.9263 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0123652823253777,
            "unit": "iter/sec",
            "range": "stddev: 0.003959416075225854",
            "extra": "mean: 331.9651 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7153457645654432,
            "unit": "iter/sec",
            "range": "stddev: 0.0030328639067916745",
            "extra": "mean: 1.3979 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26906619458928693,
            "unit": "iter/sec",
            "range": "stddev: 0.04096745602496689",
            "extra": "mean: 3.7166 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10479274646215839,
            "unit": "iter/sec",
            "range": "stddev: 0.029520226712985934",
            "extra": "mean: 9.5426 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6439505198758034,
            "unit": "iter/sec",
            "range": "stddev: 0.01489833807010207",
            "extra": "mean: 1.5529 sec\nrounds: 3"
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
        "date": 1776338313061,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.975290080800733,
            "unit": "iter/sec",
            "range": "stddev: 0.002345220330055264",
            "extra": "mean: 336.1017 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.982523616546092,
            "unit": "iter/sec",
            "range": "stddev: 0.0026285340135184456",
            "extra": "mean: 335.2865 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7168361947668684,
            "unit": "iter/sec",
            "range": "stddev: 0.00347159536745071",
            "extra": "mean: 1.3950 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2708019249061557,
            "unit": "iter/sec",
            "range": "stddev: 0.026173094382898145",
            "extra": "mean: 3.6927 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10491990500014471,
            "unit": "iter/sec",
            "range": "stddev: 0.013492053975047758",
            "extra": "mean: 9.5311 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6426702075835873,
            "unit": "iter/sec",
            "range": "stddev: 0.008899913695651379",
            "extra": "mean: 1.5560 sec\nrounds: 3"
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
        "date": 1776338312752,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9920948853128246,
            "unit": "iter/sec",
            "range": "stddev: 0.0017565898333054613",
            "extra": "mean: 334.2140 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9977956572997835,
            "unit": "iter/sec",
            "range": "stddev: 0.00121835610114361",
            "extra": "mean: 333.5784 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7120269652149556,
            "unit": "iter/sec",
            "range": "stddev: 0.007649654741966088",
            "extra": "mean: 1.4044 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2702050665744579,
            "unit": "iter/sec",
            "range": "stddev: 0.004624974393849856",
            "extra": "mean: 3.7009 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10521962550091124,
            "unit": "iter/sec",
            "range": "stddev: 0.036486284931807555",
            "extra": "mean: 9.5039 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6445823885229084,
            "unit": "iter/sec",
            "range": "stddev: 0.008752771709168834",
            "extra": "mean: 1.5514 sec\nrounds: 3"
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
        "date": 1776338312765,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.997060476491324,
            "unit": "iter/sec",
            "range": "stddev: 0.002024070865505867",
            "extra": "mean: 333.6603 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9897182483243068,
            "unit": "iter/sec",
            "range": "stddev: 0.0035236622689283083",
            "extra": "mean: 334.4797 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7144895790869725,
            "unit": "iter/sec",
            "range": "stddev: 0.005479592666711879",
            "extra": "mean: 1.3996 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2705258050218521,
            "unit": "iter/sec",
            "range": "stddev: 0.010539287026758713",
            "extra": "mean: 3.6965 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10488955935842878,
            "unit": "iter/sec",
            "range": "stddev: 0.03255196325510589",
            "extra": "mean: 9.5338 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6435162775505237,
            "unit": "iter/sec",
            "range": "stddev: 0.00688741588879767",
            "extra": "mean: 1.5540 sec\nrounds: 3"
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
        "date": 1776338312996,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9826566343205037,
            "unit": "iter/sec",
            "range": "stddev: 0.004515522436441058",
            "extra": "mean: 335.2716 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9881788933871007,
            "unit": "iter/sec",
            "range": "stddev: 0.0023779773722833572",
            "extra": "mean: 334.6520 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7139354023323278,
            "unit": "iter/sec",
            "range": "stddev: 0.0045469263435278864",
            "extra": "mean: 1.4007 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26620820651384225,
            "unit": "iter/sec",
            "range": "stddev: 0.047155070666349816",
            "extra": "mean: 3.7565 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10473167035816429,
            "unit": "iter/sec",
            "range": "stddev: 0.02063227467310981",
            "extra": "mean: 9.5482 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6429084528296971,
            "unit": "iter/sec",
            "range": "stddev: 0.008161856835037902",
            "extra": "mean: 1.5554 sec\nrounds: 3"
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
        "date": 1776338312850,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.989076809770109,
            "unit": "iter/sec",
            "range": "stddev: 0.0026332909690051486",
            "extra": "mean: 334.5515 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.008509750339181,
            "unit": "iter/sec",
            "range": "stddev: 0.001224666268192613",
            "extra": "mean: 332.3905 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7139229119204769,
            "unit": "iter/sec",
            "range": "stddev: 0.0020721991926466055",
            "extra": "mean: 1.4007 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2691268407119111,
            "unit": "iter/sec",
            "range": "stddev: 0.022519668379544003",
            "extra": "mean: 3.7157 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10485709250156083,
            "unit": "iter/sec",
            "range": "stddev: 0.012358171374223964",
            "extra": "mean: 9.5368 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6434106054872538,
            "unit": "iter/sec",
            "range": "stddev: 0.007817184707497532",
            "extra": "mean: 1.5542 sec\nrounds: 3"
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
        "date": 1776338312859,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.956299226250825,
            "unit": "iter/sec",
            "range": "stddev: 0.01073758687092046",
            "extra": "mean: 338.2608 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.902497554272849,
            "unit": "iter/sec",
            "range": "stddev: 0.007560111531183339",
            "extra": "mean: 344.5309 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7101553207451052,
            "unit": "iter/sec",
            "range": "stddev: 0.0022890706094459883",
            "extra": "mean: 1.4081 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2670451583625285,
            "unit": "iter/sec",
            "range": "stddev: 0.06558000123876896",
            "extra": "mean: 3.7447 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1047878785732855,
            "unit": "iter/sec",
            "range": "stddev: 0.054578835584119974",
            "extra": "mean: 9.5431 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6388679697073548,
            "unit": "iter/sec",
            "range": "stddev: 0.022062014731146707",
            "extra": "mean: 1.5653 sec\nrounds: 3"
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
        "date": 1776338313043,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9764117901244718,
            "unit": "iter/sec",
            "range": "stddev: 0.0021983469574216735",
            "extra": "mean: 335.9750 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9227846863366924,
            "unit": "iter/sec",
            "range": "stddev: 0.001841613872536106",
            "extra": "mean: 342.1395 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7029116411109118,
            "unit": "iter/sec",
            "range": "stddev: 0.014383852724071713",
            "extra": "mean: 1.4227 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23106841240845397,
            "unit": "iter/sec",
            "range": "stddev: 0.1965399859560155",
            "extra": "mean: 4.3277 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10469536116575712,
            "unit": "iter/sec",
            "range": "stddev: 0.013846936940926546",
            "extra": "mean: 9.5515 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6073007865293532,
            "unit": "iter/sec",
            "range": "stddev: 0.04260405363734865",
            "extra": "mean: 1.6466 sec\nrounds: 3"
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
        "date": 1776338312898,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9429674376581456,
            "unit": "iter/sec",
            "range": "stddev: 0.00595247337342857",
            "extra": "mean: 339.7931 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9946865946556804,
            "unit": "iter/sec",
            "range": "stddev: 0.003987073068070185",
            "extra": "mean: 333.9248 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.6933135260253038,
            "unit": "iter/sec",
            "range": "stddev: 0.03836803883542349",
            "extra": "mean: 1.4423 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26915387494319587,
            "unit": "iter/sec",
            "range": "stddev: 0.005643015993855408",
            "extra": "mean: 3.7153 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10488334710883658,
            "unit": "iter/sec",
            "range": "stddev: 0.003141775874841421",
            "extra": "mean: 9.5344 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6453404665356072,
            "unit": "iter/sec",
            "range": "stddev: 0.011196864015974783",
            "extra": "mean: 1.5496 sec\nrounds: 3"
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
        "date": 1776338313081,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.990245402060163,
            "unit": "iter/sec",
            "range": "stddev: 0.00577838820262589",
            "extra": "mean: 334.4207 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0184650202728016,
            "unit": "iter/sec",
            "range": "stddev: 0.0018699729802360118",
            "extra": "mean: 331.2942 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7161091954743375,
            "unit": "iter/sec",
            "range": "stddev: 0.002454016650404386",
            "extra": "mean: 1.3964 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2684103756327275,
            "unit": "iter/sec",
            "range": "stddev: 0.07204250257456565",
            "extra": "mean: 3.7256 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1047229983448418,
            "unit": "iter/sec",
            "range": "stddev: 0.019102674506002747",
            "extra": "mean: 9.5490 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6381844797220697,
            "unit": "iter/sec",
            "range": "stddev: 0.026044463262016742",
            "extra": "mean: 1.5669 sec\nrounds: 3"
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
        "date": 1776338312970,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0147571209435178,
            "unit": "iter/sec",
            "range": "stddev: 0.0015141483028631192",
            "extra": "mean: 331.7017 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0139678760627873,
            "unit": "iter/sec",
            "range": "stddev: 0.001525998357968635",
            "extra": "mean: 331.7885 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7130667333264291,
            "unit": "iter/sec",
            "range": "stddev: 0.004141884553966005",
            "extra": "mean: 1.4024 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26968321541630996,
            "unit": "iter/sec",
            "range": "stddev: 0.02032677301267628",
            "extra": "mean: 3.7081 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10425810428218747,
            "unit": "iter/sec",
            "range": "stddev: 0.0336261022163776",
            "extra": "mean: 9.5916 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6453255961869148,
            "unit": "iter/sec",
            "range": "stddev: 0.008396451902798092",
            "extra": "mean: 1.5496 sec\nrounds: 3"
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
        "date": 1776338312831,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9481092164744602,
            "unit": "iter/sec",
            "range": "stddev: 0.0033315949702561543",
            "extra": "mean: 339.2005 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9048391456790355,
            "unit": "iter/sec",
            "range": "stddev: 0.0036062842655265677",
            "extra": "mean: 344.2531 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7141674738280764,
            "unit": "iter/sec",
            "range": "stddev: 0.002448137101222051",
            "extra": "mean: 1.4002 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2687353368017909,
            "unit": "iter/sec",
            "range": "stddev: 0.025705524194685702",
            "extra": "mean: 3.7211 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1043324566822413,
            "unit": "iter/sec",
            "range": "stddev: 0.040975776577479965",
            "extra": "mean: 9.5847 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6440233676502436,
            "unit": "iter/sec",
            "range": "stddev: 0.013384866093861145",
            "extra": "mean: 1.5527 sec\nrounds: 3"
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
        "date": 1776338313049,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.902066599352329,
            "unit": "iter/sec",
            "range": "stddev: 0.003624513626590629",
            "extra": "mean: 344.5820 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9083441410743265,
            "unit": "iter/sec",
            "range": "stddev: 0.003893320377572175",
            "extra": "mean: 343.8383 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7117714850605564,
            "unit": "iter/sec",
            "range": "stddev: 0.005619512076270999",
            "extra": "mean: 1.4049 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2680345951848154,
            "unit": "iter/sec",
            "range": "stddev: 0.03764921427879965",
            "extra": "mean: 3.7309 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10481313037359415,
            "unit": "iter/sec",
            "range": "stddev: 0.01578167188976067",
            "extra": "mean: 9.5408 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6405558770548094,
            "unit": "iter/sec",
            "range": "stddev: 0.01827223451477339",
            "extra": "mean: 1.5611 sec\nrounds: 3"
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
        "date": 1776338312964,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9601243309049896,
            "unit": "iter/sec",
            "range": "stddev: 0.002306203214177996",
            "extra": "mean: 337.8236 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.963944670616855,
            "unit": "iter/sec",
            "range": "stddev: 0.0013641734071558605",
            "extra": "mean: 337.3882 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7140401810288985,
            "unit": "iter/sec",
            "range": "stddev: 0.0030804872356505037",
            "extra": "mean: 1.4005 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26852035896617105,
            "unit": "iter/sec",
            "range": "stddev: 0.006432762323783764",
            "extra": "mean: 3.7241 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10460680762255922,
            "unit": "iter/sec",
            "range": "stddev: 0.013363918322967556",
            "extra": "mean: 9.5596 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6386272781928157,
            "unit": "iter/sec",
            "range": "stddev: 0.024712527599266485",
            "extra": "mean: 1.5659 sec\nrounds: 3"
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
        "date": 1776338312962,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.922600577760823,
            "unit": "iter/sec",
            "range": "stddev: 0.002853346378841653",
            "extra": "mean: 342.1610 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9246682819739562,
            "unit": "iter/sec",
            "range": "stddev: 0.0005624293557724635",
            "extra": "mean: 341.9191 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7080826086869095,
            "unit": "iter/sec",
            "range": "stddev: 0.004419675821321209",
            "extra": "mean: 1.4123 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26632783194655857,
            "unit": "iter/sec",
            "range": "stddev: 0.01902320830139693",
            "extra": "mean: 3.7548 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10440896644153161,
            "unit": "iter/sec",
            "range": "stddev: 0.01693780176897076",
            "extra": "mean: 9.5777 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6417563664906633,
            "unit": "iter/sec",
            "range": "stddev: 0.010606748683971021",
            "extra": "mean: 1.5582 sec\nrounds: 3"
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
        "date": 1776338312870,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9212041906635973,
            "unit": "iter/sec",
            "range": "stddev: 0.009162997598954501",
            "extra": "mean: 342.3246 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.892829024105913,
            "unit": "iter/sec",
            "range": "stddev: 0.011408236107369067",
            "extra": "mean: 345.6824 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7084488965997814,
            "unit": "iter/sec",
            "range": "stddev: 0.017457064192974905",
            "extra": "mean: 1.4115 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2697835471619102,
            "unit": "iter/sec",
            "range": "stddev: 0.01703492433845774",
            "extra": "mean: 3.7067 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10506274863732036,
            "unit": "iter/sec",
            "range": "stddev: 0.010978555767917546",
            "extra": "mean: 9.5181 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6446663552841239,
            "unit": "iter/sec",
            "range": "stddev: 0.014228641242276206",
            "extra": "mean: 1.5512 sec\nrounds: 3"
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
        "date": 1776338312972,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.989806610745926,
            "unit": "iter/sec",
            "range": "stddev: 0.0018511782983492147",
            "extra": "mean: 334.4698 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9988324681533043,
            "unit": "iter/sec",
            "range": "stddev: 0.002526688941914047",
            "extra": "mean: 333.4631 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.715399036491999,
            "unit": "iter/sec",
            "range": "stddev: 0.0009946283206171546",
            "extra": "mean: 1.3978 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2689082164797145,
            "unit": "iter/sec",
            "range": "stddev: 0.04249462070718251",
            "extra": "mean: 3.7187 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10487931785877357,
            "unit": "iter/sec",
            "range": "stddev: 0.021455963082141954",
            "extra": "mean: 9.5348 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6464711252110984,
            "unit": "iter/sec",
            "range": "stddev: 0.009394929525637122",
            "extra": "mean: 1.5469 sec\nrounds: 3"
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
        "date": 1776338312749,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0094937109416393,
            "unit": "iter/sec",
            "range": "stddev: 0.0016543324434266917",
            "extra": "mean: 332.2818 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9984576941126106,
            "unit": "iter/sec",
            "range": "stddev: 0.0013450916581753437",
            "extra": "mean: 333.5048 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7180275143126988,
            "unit": "iter/sec",
            "range": "stddev: 0.0006484368762425071",
            "extra": "mean: 1.3927 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2693479815300138,
            "unit": "iter/sec",
            "range": "stddev: 0.03073075814448573",
            "extra": "mean: 3.7127 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10450933134700005,
            "unit": "iter/sec",
            "range": "stddev: 0.0049229833506527505",
            "extra": "mean: 9.5685 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6455644000314769,
            "unit": "iter/sec",
            "range": "stddev: 0.010153154972691066",
            "extra": "mean: 1.5490 sec\nrounds: 3"
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
        "date": 1776338312766,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.991605396105822,
            "unit": "iter/sec",
            "range": "stddev: 0.0016996981586130263",
            "extra": "mean: 334.2687 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.995698545359665,
            "unit": "iter/sec",
            "range": "stddev: 0.0019594689416976958",
            "extra": "mean: 333.8120 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7165617936602507,
            "unit": "iter/sec",
            "range": "stddev: 0.00291266637891172",
            "extra": "mean: 1.3956 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2685862671620392,
            "unit": "iter/sec",
            "range": "stddev: 0.045797832282684886",
            "extra": "mean: 3.7232 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10478737939483627,
            "unit": "iter/sec",
            "range": "stddev: 0.05364605563138238",
            "extra": "mean: 9.5431 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6437537754389466,
            "unit": "iter/sec",
            "range": "stddev: 0.011308088272952694",
            "extra": "mean: 1.5534 sec\nrounds: 3"
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
        "date": 1776338312880,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.936832939587035,
            "unit": "iter/sec",
            "range": "stddev: 0.010730509624660736",
            "extra": "mean: 340.5029 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9405304412410755,
            "unit": "iter/sec",
            "range": "stddev: 0.004800216729871957",
            "extra": "mean: 340.0747 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7157048069607721,
            "unit": "iter/sec",
            "range": "stddev: 0.00209434314575989",
            "extra": "mean: 1.3972 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2668517954437018,
            "unit": "iter/sec",
            "range": "stddev: 0.01660402652321184",
            "extra": "mean: 3.7474 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10405713824022748,
            "unit": "iter/sec",
            "range": "stddev: 0.02112993027387036",
            "extra": "mean: 9.6101 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6409384191673375,
            "unit": "iter/sec",
            "range": "stddev: 0.016689039272234096",
            "extra": "mean: 1.5602 sec\nrounds: 3"
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
        "date": 1776338313034,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9985821743932752,
            "unit": "iter/sec",
            "range": "stddev: 0.0021656824333325104",
            "extra": "mean: 333.4909 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.985281817899716,
            "unit": "iter/sec",
            "range": "stddev: 0.002231449096969231",
            "extra": "mean: 334.9767 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7172177425966156,
            "unit": "iter/sec",
            "range": "stddev: 0.001655994716925325",
            "extra": "mean: 1.3943 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2609843476313013,
            "unit": "iter/sec",
            "range": "stddev: 0.01600606685183157",
            "extra": "mean: 3.8316 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10404955004304356,
            "unit": "iter/sec",
            "range": "stddev: 0.025710199839096125",
            "extra": "mean: 9.6108 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6394292847750871,
            "unit": "iter/sec",
            "range": "stddev: 0.013595760765346531",
            "extra": "mean: 1.5639 sec\nrounds: 3"
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
        "date": 1776338313095,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9026001395681207,
            "unit": "iter/sec",
            "range": "stddev: 0.005494857008636822",
            "extra": "mean: 344.5187 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.895632408935775,
            "unit": "iter/sec",
            "range": "stddev: 0.007675248278706192",
            "extra": "mean: 345.3477 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7142291855964847,
            "unit": "iter/sec",
            "range": "stddev: 0.002967074119603573",
            "extra": "mean: 1.4001 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26714105701425633,
            "unit": "iter/sec",
            "range": "stddev: 0.0480124492278923",
            "extra": "mean: 3.7433 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10414988648537997,
            "unit": "iter/sec",
            "range": "stddev: 0.013510539465313623",
            "extra": "mean: 9.6015 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6425951995727025,
            "unit": "iter/sec",
            "range": "stddev: 0.015468538541505384",
            "extra": "mean: 1.5562 sec\nrounds: 3"
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
        "date": 1776338312992,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.928709900067996,
            "unit": "iter/sec",
            "range": "stddev: 0.004518225779602481",
            "extra": "mean: 341.4473 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9636307572238745,
            "unit": "iter/sec",
            "range": "stddev: 0.004519724084143862",
            "extra": "mean: 337.4240 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.715402870033498,
            "unit": "iter/sec",
            "range": "stddev: 0.0009594678819778511",
            "extra": "mean: 1.3978 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2687577362334199,
            "unit": "iter/sec",
            "range": "stddev: 0.0195078393432765",
            "extra": "mean: 3.7208 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10410602270703269,
            "unit": "iter/sec",
            "range": "stddev: 0.012655987827719229",
            "extra": "mean: 9.6056 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6453044275874067,
            "unit": "iter/sec",
            "range": "stddev: 0.00976740198439147",
            "extra": "mean: 1.5497 sec\nrounds: 3"
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
        "date": 1776338312867,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9805714247454094,
            "unit": "iter/sec",
            "range": "stddev: 0.003140803464713707",
            "extra": "mean: 335.5061 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.932694146153237,
            "unit": "iter/sec",
            "range": "stddev: 0.011683255822707728",
            "extra": "mean: 340.9834 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.714856909976768,
            "unit": "iter/sec",
            "range": "stddev: 0.008758202453714918",
            "extra": "mean: 1.3989 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2696969638202984,
            "unit": "iter/sec",
            "range": "stddev: 0.03224009815571433",
            "extra": "mean: 3.7079 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10488318050994538,
            "unit": "iter/sec",
            "range": "stddev: 0.008456956833075856",
            "extra": "mean: 9.5344 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6453587680237981,
            "unit": "iter/sec",
            "range": "stddev: 0.008815796915885948",
            "extra": "mean: 1.5495 sec\nrounds: 3"
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
        "date": 1776338313014,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.005557914965265,
            "unit": "iter/sec",
            "range": "stddev: 0.0007037514173956469",
            "extra": "mean: 332.7169 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.996605469895763,
            "unit": "iter/sec",
            "range": "stddev: 0.002328007290271395",
            "extra": "mean: 333.7109 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7194355509533648,
            "unit": "iter/sec",
            "range": "stddev: 0.0014421143633452569",
            "extra": "mean: 1.3900 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2679328640815381,
            "unit": "iter/sec",
            "range": "stddev: 0.05264942679878461",
            "extra": "mean: 3.7323 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10496117468829672,
            "unit": "iter/sec",
            "range": "stddev: 0.021547980820656405",
            "extra": "mean: 9.5273 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6352003372222684,
            "unit": "iter/sec",
            "range": "stddev: 0.045886699281193405",
            "extra": "mean: 1.5743 sec\nrounds: 3"
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
        "date": 1776338313085,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.99595016384757,
            "unit": "iter/sec",
            "range": "stddev: 0.0010939534108658161",
            "extra": "mean: 333.7839 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.999197599472685,
            "unit": "iter/sec",
            "range": "stddev: 0.0007250032349344339",
            "extra": "mean: 333.4225 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7142605401729678,
            "unit": "iter/sec",
            "range": "stddev: 0.014872168189478074",
            "extra": "mean: 1.4000 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2695682501177198,
            "unit": "iter/sec",
            "range": "stddev: 0.0013527280222031708",
            "extra": "mean: 3.7096 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1043247682194275,
            "unit": "iter/sec",
            "range": "stddev: 0.029238647444457172",
            "extra": "mean: 9.5855 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.63496969261189,
            "unit": "iter/sec",
            "range": "stddev: 0.04432045987140568",
            "extra": "mean: 1.5749 sec\nrounds: 3"
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
        "date": 1776338312936,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9947471900694804,
            "unit": "iter/sec",
            "range": "stddev: 0.0012977197296053218",
            "extra": "mean: 333.9180 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9650276008520167,
            "unit": "iter/sec",
            "range": "stddev: 0.007213137481819535",
            "extra": "mean: 337.2650 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7128822024986806,
            "unit": "iter/sec",
            "range": "stddev: 0.009930745494983958",
            "extra": "mean: 1.4028 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27015306672306977,
            "unit": "iter/sec",
            "range": "stddev: 0.010644754792305975",
            "extra": "mean: 3.7016 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1053977075208979,
            "unit": "iter/sec",
            "range": "stddev: 0.01177696415484824",
            "extra": "mean: 9.4879 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6388948750144144,
            "unit": "iter/sec",
            "range": "stddev: 0.022839041197877077",
            "extra": "mean: 1.5652 sec\nrounds: 3"
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
        "date": 1776338312797,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9705627027137833,
            "unit": "iter/sec",
            "range": "stddev: 0.007316361141564697",
            "extra": "mean: 336.6366 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9776184764333284,
            "unit": "iter/sec",
            "range": "stddev: 0.005901834690051562",
            "extra": "mean: 335.8389 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7158053081113834,
            "unit": "iter/sec",
            "range": "stddev: 0.006939545883307542",
            "extra": "mean: 1.3970 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26837089095658095,
            "unit": "iter/sec",
            "range": "stddev: 0.053426975626714196",
            "extra": "mean: 3.7262 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10515447550502634,
            "unit": "iter/sec",
            "range": "stddev: 0.01062198937172786",
            "extra": "mean: 9.5098 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6339436448481189,
            "unit": "iter/sec",
            "range": "stddev: 0.04622416002694169",
            "extra": "mean: 1.5774 sec\nrounds: 3"
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
        "date": 1776338312830,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.988092037884518,
            "unit": "iter/sec",
            "range": "stddev: 0.0015010751349438108",
            "extra": "mean: 334.6617 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.011014471598105,
            "unit": "iter/sec",
            "range": "stddev: 0.0006915175294755253",
            "extra": "mean: 332.1140 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7162478645669627,
            "unit": "iter/sec",
            "range": "stddev: 0.0007544504892391433",
            "extra": "mean: 1.3962 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27019514740225986,
            "unit": "iter/sec",
            "range": "stddev: 0.015392150336874166",
            "extra": "mean: 3.7010 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10499203196780851,
            "unit": "iter/sec",
            "range": "stddev: 0.025006554922449718",
            "extra": "mean: 9.5245 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6438530538870857,
            "unit": "iter/sec",
            "range": "stddev: 0.007250132723380405",
            "extra": "mean: 1.5531 sec\nrounds: 3"
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
        "date": 1776338312851,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.006620294063077,
            "unit": "iter/sec",
            "range": "stddev: 0.0010191203302845335",
            "extra": "mean: 332.5994 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9995177777256665,
            "unit": "iter/sec",
            "range": "stddev: 0.0009107660976519987",
            "extra": "mean: 333.3869 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.719748583413149,
            "unit": "iter/sec",
            "range": "stddev: 0.004757997138365604",
            "extra": "mean: 1.3894 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2702648354392166,
            "unit": "iter/sec",
            "range": "stddev: 0.01602583689672789",
            "extra": "mean: 3.7001 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10508642353657116,
            "unit": "iter/sec",
            "range": "stddev: 0.02607870662672013",
            "extra": "mean: 9.5160 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6443695303387631,
            "unit": "iter/sec",
            "range": "stddev: 0.012515090010591983",
            "extra": "mean: 1.5519 sec\nrounds: 3"
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
        "date": 1776338312905,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.988625652986045,
            "unit": "iter/sec",
            "range": "stddev: 0.0018758903578102901",
            "extra": "mean: 334.6020 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9205632605085885,
            "unit": "iter/sec",
            "range": "stddev: 0.017064300455509655",
            "extra": "mean: 342.3997 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7175813525554812,
            "unit": "iter/sec",
            "range": "stddev: 0.0005164668426811547",
            "extra": "mean: 1.3936 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2718138048762654,
            "unit": "iter/sec",
            "range": "stddev: 0.037745234932373674",
            "extra": "mean: 3.6790 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10540050828528007,
            "unit": "iter/sec",
            "range": "stddev: 0.03486076412420624",
            "extra": "mean: 9.4876 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6457838210653667,
            "unit": "iter/sec",
            "range": "stddev: 0.010671474353672246",
            "extra": "mean: 1.5485 sec\nrounds: 3"
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
        "date": 1776338312863,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.995703956818248,
            "unit": "iter/sec",
            "range": "stddev: 0.0016387758670662108",
            "extra": "mean: 333.8114 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9966891876633217,
            "unit": "iter/sec",
            "range": "stddev: 0.0008760922902120077",
            "extra": "mean: 333.7016 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7136992698048051,
            "unit": "iter/sec",
            "range": "stddev: 0.0063954535768036055",
            "extra": "mean: 1.4012 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26944015880885636,
            "unit": "iter/sec",
            "range": "stddev: 0.013206448181007623",
            "extra": "mean: 3.7114 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10500030164276665,
            "unit": "iter/sec",
            "range": "stddev: 0.014660158041366094",
            "extra": "mean: 9.5238 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6452276552477652,
            "unit": "iter/sec",
            "range": "stddev: 0.009307702626082028",
            "extra": "mean: 1.5498 sec\nrounds: 3"
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
        "date": 1776338312753,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9903394077148255,
            "unit": "iter/sec",
            "range": "stddev: 0.0023540935154916683",
            "extra": "mean: 334.4102 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.983224354536544,
            "unit": "iter/sec",
            "range": "stddev: 0.0021881546827563804",
            "extra": "mean: 335.2078 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7177226688999059,
            "unit": "iter/sec",
            "range": "stddev: 0.0037275345552366386",
            "extra": "mean: 1.3933 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26530759842553747,
            "unit": "iter/sec",
            "range": "stddev: 0.13374590360609623",
            "extra": "mean: 3.7692 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10493862742764078,
            "unit": "iter/sec",
            "range": "stddev: 0.018540803685200495",
            "extra": "mean: 9.5294 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6444754229458507,
            "unit": "iter/sec",
            "range": "stddev: 0.010823023927016362",
            "extra": "mean: 1.5516 sec\nrounds: 3"
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
        "date": 1776338312990,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.976588770344259,
            "unit": "iter/sec",
            "range": "stddev: 0.002212993679872071",
            "extra": "mean: 335.9550 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.983368022803249,
            "unit": "iter/sec",
            "range": "stddev: 0.0017322306938028853",
            "extra": "mean: 335.1916 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.714888825307007,
            "unit": "iter/sec",
            "range": "stddev: 0.002580735644290009",
            "extra": "mean: 1.3988 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27181569313531156,
            "unit": "iter/sec",
            "range": "stddev: 0.0065153239917705415",
            "extra": "mean: 3.6790 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10467293964769318,
            "unit": "iter/sec",
            "range": "stddev: 0.0038931600075323264",
            "extra": "mean: 9.5536 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6447758355630505,
            "unit": "iter/sec",
            "range": "stddev: 0.009484900393530435",
            "extra": "mean: 1.5509 sec\nrounds: 3"
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
        "date": 1776338312780,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9945296602935247,
            "unit": "iter/sec",
            "range": "stddev: 0.003053399621998878",
            "extra": "mean: 333.9423 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9833413482889126,
            "unit": "iter/sec",
            "range": "stddev: 0.002017444940019315",
            "extra": "mean: 335.1946 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7172423637164173,
            "unit": "iter/sec",
            "range": "stddev: 0.0007626361265598712",
            "extra": "mean: 1.3942 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2682558259590622,
            "unit": "iter/sec",
            "range": "stddev: 0.07083093029863093",
            "extra": "mean: 3.7278 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10481168064258053,
            "unit": "iter/sec",
            "range": "stddev: 0.00912268775715125",
            "extra": "mean: 9.5409 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6443542121448164,
            "unit": "iter/sec",
            "range": "stddev: 0.009735582838498883",
            "extra": "mean: 1.5519 sec\nrounds: 3"
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
        "date": 1776338312806,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9919091351063956,
            "unit": "iter/sec",
            "range": "stddev: 0.0013820128932934364",
            "extra": "mean: 334.2347 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.997329831716725,
            "unit": "iter/sec",
            "range": "stddev: 0.002531947277227127",
            "extra": "mean: 333.6303 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7174439939650302,
            "unit": "iter/sec",
            "range": "stddev: 0.0017166380115363567",
            "extra": "mean: 1.3938 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2712651572288815,
            "unit": "iter/sec",
            "range": "stddev: 0.023251095557744687",
            "extra": "mean: 3.6864 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10483769173555929,
            "unit": "iter/sec",
            "range": "stddev: 0.034733658609311414",
            "extra": "mean: 9.5386 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6426917073551902,
            "unit": "iter/sec",
            "range": "stddev: 0.018661769263147464",
            "extra": "mean: 1.5560 sec\nrounds: 3"
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
        "date": 1776338312791,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.97958210257347,
            "unit": "iter/sec",
            "range": "stddev: 0.0014341555527862455",
            "extra": "mean: 335.6175 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.992320385082628,
            "unit": "iter/sec",
            "range": "stddev: 0.001307735618836057",
            "extra": "mean: 334.1888 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.715640030328146,
            "unit": "iter/sec",
            "range": "stddev: 0.002311750194991713",
            "extra": "mean: 1.3974 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27168679585518907,
            "unit": "iter/sec",
            "range": "stddev: 0.012438116203160734",
            "extra": "mean: 3.6807 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10426097626942663,
            "unit": "iter/sec",
            "range": "stddev: 0.04095864780241601",
            "extra": "mean: 9.5913 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6437835809989402,
            "unit": "iter/sec",
            "range": "stddev: 0.015584400521500263",
            "extra": "mean: 1.5533 sec\nrounds: 3"
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
        "date": 1776338312854,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.971028831757278,
            "unit": "iter/sec",
            "range": "stddev: 0.0015677149656565712",
            "extra": "mean: 336.5837 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.982257312418112,
            "unit": "iter/sec",
            "range": "stddev: 0.0025199254331199823",
            "extra": "mean: 335.3165 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.715845897692686,
            "unit": "iter/sec",
            "range": "stddev: 0.0026837635120815955",
            "extra": "mean: 1.3969 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27086582474051374,
            "unit": "iter/sec",
            "range": "stddev: 0.012438723377790025",
            "extra": "mean: 3.6919 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10465227392648123,
            "unit": "iter/sec",
            "range": "stddev: 0.02357321030498421",
            "extra": "mean: 9.5555 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6452656705789555,
            "unit": "iter/sec",
            "range": "stddev: 0.009482403065858505",
            "extra": "mean: 1.5497 sec\nrounds: 3"
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
        "date": 1776338313024,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9613205137017435,
            "unit": "iter/sec",
            "range": "stddev: 0.004267328976467025",
            "extra": "mean: 337.6872 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9930779744162876,
            "unit": "iter/sec",
            "range": "stddev: 0.0030477589584033925",
            "extra": "mean: 334.1042 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7161309887833576,
            "unit": "iter/sec",
            "range": "stddev: 0.0026719492692791886",
            "extra": "mean: 1.3964 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26721630859482254,
            "unit": "iter/sec",
            "range": "stddev: 0.060093982671798885",
            "extra": "mean: 3.7423 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1047228127010361,
            "unit": "iter/sec",
            "range": "stddev: 0.02918949777139112",
            "extra": "mean: 9.5490 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6432228153952693,
            "unit": "iter/sec",
            "range": "stddev: 0.01311672227077597",
            "extra": "mean: 1.5547 sec\nrounds: 3"
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
        "date": 1776338313010,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9808942510226677,
            "unit": "iter/sec",
            "range": "stddev: 0.0018234680553929573",
            "extra": "mean: 335.4698 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9930051685380334,
            "unit": "iter/sec",
            "range": "stddev: 0.0014132865932088453",
            "extra": "mean: 334.1124 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7168895217194291,
            "unit": "iter/sec",
            "range": "stddev: 0.0013650493788994614",
            "extra": "mean: 1.3949 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.26948379032543995,
            "unit": "iter/sec",
            "range": "stddev: 0.038830574205232525",
            "extra": "mean: 3.7108 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10439520723340785,
            "unit": "iter/sec",
            "range": "stddev: 0.0381223422782303",
            "extra": "mean: 9.5790 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6432474068480847,
            "unit": "iter/sec",
            "range": "stddev: 0.009967605866212972",
            "extra": "mean: 1.5546 sec\nrounds: 3"
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
        "date": 1776338312909,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9832575736566262,
            "unit": "iter/sec",
            "range": "stddev: 0.0011182158540103907",
            "extra": "mean: 335.2040 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9832731893726105,
            "unit": "iter/sec",
            "range": "stddev: 0.0021680446085067524",
            "extra": "mean: 335.2023 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7159545357817184,
            "unit": "iter/sec",
            "range": "stddev: 0.002121614761482511",
            "extra": "mean: 1.3967 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27073004596101236,
            "unit": "iter/sec",
            "range": "stddev: 0.017395855261328134",
            "extra": "mean: 3.6937 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10498783865601871,
            "unit": "iter/sec",
            "range": "stddev: 0.08564623983871944",
            "extra": "mean: 9.5249 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6454968913748335,
            "unit": "iter/sec",
            "range": "stddev: 0.008124656446611504",
            "extra": "mean: 1.5492 sec\nrounds: 3"
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
        "date": 1776338312772,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.996640392138297,
            "unit": "iter/sec",
            "range": "stddev: 0.0009164839825660909",
            "extra": "mean: 333.7070 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.987309012730798,
            "unit": "iter/sec",
            "range": "stddev: 0.0009668714770873956",
            "extra": "mean: 334.7494 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7193034754845127,
            "unit": "iter/sec",
            "range": "stddev: 0.0015929561475829595",
            "extra": "mean: 1.3902 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27021005870562576,
            "unit": "iter/sec",
            "range": "stddev: 0.06456902169101329",
            "extra": "mean: 3.7008 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10555263403436987,
            "unit": "iter/sec",
            "range": "stddev: 0.0067374211377768875",
            "extra": "mean: 9.4739 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6428854979190374,
            "unit": "iter/sec",
            "range": "stddev: 0.019132032391055365",
            "extra": "mean: 1.5555 sec\nrounds: 3"
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
        "date": 1776338312974,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9857640692012195,
            "unit": "iter/sec",
            "range": "stddev: 0.0017076407200009459",
            "extra": "mean: 334.9226 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9571844580529842,
            "unit": "iter/sec",
            "range": "stddev: 0.0029491712578443557",
            "extra": "mean: 338.1595 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7166709167600351,
            "unit": "iter/sec",
            "range": "stddev: 0.002486096246619123",
            "extra": "mean: 1.3953 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.272989978403844,
            "unit": "iter/sec",
            "range": "stddev: 0.0033072975106675307",
            "extra": "mean: 3.6631 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10543745497554603,
            "unit": "iter/sec",
            "range": "stddev: 0.012369005078387153",
            "extra": "mean: 9.4843 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6441190276104842,
            "unit": "iter/sec",
            "range": "stddev: 0.007495807974940525",
            "extra": "mean: 1.5525 sec\nrounds: 3"
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
        "date": 1776338313071,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.987072600369962,
            "unit": "iter/sec",
            "range": "stddev: 0.001413365986678515",
            "extra": "mean: 334.7759 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.991072117532553,
            "unit": "iter/sec",
            "range": "stddev: 0.0009397507049121585",
            "extra": "mean: 334.3283 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7183462944783189,
            "unit": "iter/sec",
            "range": "stddev: 0.0034437378242338315",
            "extra": "mean: 1.3921 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27235339326825303,
            "unit": "iter/sec",
            "range": "stddev: 0.02600986520758512",
            "extra": "mean: 3.6717 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10528122641396556,
            "unit": "iter/sec",
            "range": "stddev: 0.028369318000690896",
            "extra": "mean: 9.4984 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6447569240154174,
            "unit": "iter/sec",
            "range": "stddev: 0.010188188936868733",
            "extra": "mean: 1.5510 sec\nrounds: 3"
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
        "date": 1776338313027,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9524619332944475,
            "unit": "iter/sec",
            "range": "stddev: 0.00681119252022579",
            "extra": "mean: 338.7004 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9848939953363542,
            "unit": "iter/sec",
            "range": "stddev: 0.002349110479795015",
            "extra": "mean: 335.0203 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7183310542816141,
            "unit": "iter/sec",
            "range": "stddev: 0.0007162509148830164",
            "extra": "mean: 1.3921 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27130708709530393,
            "unit": "iter/sec",
            "range": "stddev: 0.032905019708879435",
            "extra": "mean: 3.6859 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10536819129986504,
            "unit": "iter/sec",
            "range": "stddev: 0.04852719616608041",
            "extra": "mean: 9.4905 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6429686342613445,
            "unit": "iter/sec",
            "range": "stddev: 0.016392114265102405",
            "extra": "mean: 1.5553 sec\nrounds: 3"
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
        "date": 1776338313040,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9744752645044734,
            "unit": "iter/sec",
            "range": "stddev: 0.003811677319958553",
            "extra": "mean: 336.1938 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9844722623660194,
            "unit": "iter/sec",
            "range": "stddev: 0.002622458642476366",
            "extra": "mean: 335.0676 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7105207016288699,
            "unit": "iter/sec",
            "range": "stddev: 0.027342199500098995",
            "extra": "mean: 1.4074 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2718597720993924,
            "unit": "iter/sec",
            "range": "stddev: 0.019688635088281666",
            "extra": "mean: 3.6784 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10534462829579803,
            "unit": "iter/sec",
            "range": "stddev: 0.030638860212360387",
            "extra": "mean: 9.4927 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6419211812879362,
            "unit": "iter/sec",
            "range": "stddev: 0.014874594345414426",
            "extra": "mean: 1.5578 sec\nrounds: 3"
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
        "date": 1776338312853,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9718459240203745,
            "unit": "iter/sec",
            "range": "stddev: 0.0013215061069819201",
            "extra": "mean: 336.4912 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9818165908076337,
            "unit": "iter/sec",
            "range": "stddev: 0.0033225610278621244",
            "extra": "mean: 335.3660 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7199552689487267,
            "unit": "iter/sec",
            "range": "stddev: 0.0009424115154686969",
            "extra": "mean: 1.3890 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2716106198382143,
            "unit": "iter/sec",
            "range": "stddev: 0.024106660733356034",
            "extra": "mean: 3.6817 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10537154428528107,
            "unit": "iter/sec",
            "range": "stddev: 0.008148078443357994",
            "extra": "mean: 9.4902 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6446653372171557,
            "unit": "iter/sec",
            "range": "stddev: 0.00991822601500285",
            "extra": "mean: 1.5512 sec\nrounds: 3"
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
        "date": 1776338312951,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9551344100109835,
            "unit": "iter/sec",
            "range": "stddev: 0.008020435288757628",
            "extra": "mean: 338.3941 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9941613273246843,
            "unit": "iter/sec",
            "range": "stddev: 0.00032024469940195336",
            "extra": "mean: 333.9833 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7078925452415252,
            "unit": "iter/sec",
            "range": "stddev: 0.032926126356134774",
            "extra": "mean: 1.4126 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.27085062544355226,
            "unit": "iter/sec",
            "range": "stddev: 0.025616822388729144",
            "extra": "mean: 3.6921 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10540643962718087,
            "unit": "iter/sec",
            "range": "stddev: 0.017997444979481518",
            "extra": "mean: 9.4871 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6458644650500732,
            "unit": "iter/sec",
            "range": "stddev: 0.010502875651155987",
            "extra": "mean: 1.5483 sec\nrounds: 3"
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
        "date": 1776338312942,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.809158954888633,
            "unit": "iter/sec",
            "range": "stddev: 0.0021376778008524114",
            "extra": "mean: 355.9784 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.816056074425938,
            "unit": "iter/sec",
            "range": "stddev: 0.005283053572898488",
            "extra": "mean: 355.1066 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.725021033579168,
            "unit": "iter/sec",
            "range": "stddev: 0.003366145127158722",
            "extra": "mean: 1.3793 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2816813736247242,
            "unit": "iter/sec",
            "range": "stddev: 0.04824808923183275",
            "extra": "mean: 3.5501 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10556974536293408,
            "unit": "iter/sec",
            "range": "stddev: 0.017331925661706653",
            "extra": "mean: 9.4724 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6425831014057666,
            "unit": "iter/sec",
            "range": "stddev: 0.009093252776423722",
            "extra": "mean: 1.5562 sec\nrounds: 3"
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
        "date": 1776338312891,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.877809190996277,
            "unit": "iter/sec",
            "range": "stddev: 0.002821657871114922",
            "extra": "mean: 347.4866 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.897426466060102,
            "unit": "iter/sec",
            "range": "stddev: 0.002374634352350449",
            "extra": "mean: 345.1339 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7269591287359407,
            "unit": "iter/sec",
            "range": "stddev: 0.0030805496584574974",
            "extra": "mean: 1.3756 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2817723752264539,
            "unit": "iter/sec",
            "range": "stddev: 0.04337829585159578",
            "extra": "mean: 3.5490 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10541035858573526,
            "unit": "iter/sec",
            "range": "stddev: 0.01534485204636709",
            "extra": "mean: 9.4867 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6428528317942245,
            "unit": "iter/sec",
            "range": "stddev: 0.008864573942876205",
            "extra": "mean: 1.5556 sec\nrounds: 3"
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
        "date": 1776338312787,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.870098361422562,
            "unit": "iter/sec",
            "range": "stddev: 0.0046876382673959735",
            "extra": "mean: 348.4201 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9145630263976345,
            "unit": "iter/sec",
            "range": "stddev: 0.005180511981004363",
            "extra": "mean: 343.1046 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5694923698322695,
            "unit": "iter/sec",
            "range": "stddev: 0.008562798030563904",
            "extra": "mean: 1.7559 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23467914255533212,
            "unit": "iter/sec",
            "range": "stddev: 0.021023108482476842",
            "extra": "mean: 4.2611 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10070039885588188,
            "unit": "iter/sec",
            "range": "stddev: 0.014072107950659904",
            "extra": "mean: 9.9304 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6422306248632267,
            "unit": "iter/sec",
            "range": "stddev: 0.014833074142473136",
            "extra": "mean: 1.5571 sec\nrounds: 3"
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
        "date": 1776338312988,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.885687152686032,
            "unit": "iter/sec",
            "range": "stddev: 0.00452691885528555",
            "extra": "mean: 346.5379 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.8974279788547483,
            "unit": "iter/sec",
            "range": "stddev: 0.007217264211983272",
            "extra": "mean: 345.1337 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5762446236796511,
            "unit": "iter/sec",
            "range": "stddev: 0.004406736963774789",
            "extra": "mean: 1.7354 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23359190683825873,
            "unit": "iter/sec",
            "range": "stddev: 0.03218743350503028",
            "extra": "mean: 4.2810 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1011254775359737,
            "unit": "iter/sec",
            "range": "stddev: 0.02107376339496966",
            "extra": "mean: 9.8887 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.642621722520181,
            "unit": "iter/sec",
            "range": "stddev: 0.012898956076447938",
            "extra": "mean: 1.5561 sec\nrounds: 3"
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
        "date": 1776338312961,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.891803441440522,
            "unit": "iter/sec",
            "range": "stddev: 0.0023151799635508503",
            "extra": "mean: 345.8050 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9028260135219828,
            "unit": "iter/sec",
            "range": "stddev: 0.0025500463408348363",
            "extra": "mean: 344.4919 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5690618059819623,
            "unit": "iter/sec",
            "range": "stddev: 0.008500989492839254",
            "extra": "mean: 1.7573 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.22958632672727708,
            "unit": "iter/sec",
            "range": "stddev: 0.052483163396934256",
            "extra": "mean: 4.3557 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10067450048146245,
            "unit": "iter/sec",
            "range": "stddev: 0.012830749107076914",
            "extra": "mean: 9.9330 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.645142619062772,
            "unit": "iter/sec",
            "range": "stddev: 0.010074868396225432",
            "extra": "mean: 1.5500 sec\nrounds: 3"
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
        "date": 1776338312810,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8863913084726076,
            "unit": "iter/sec",
            "range": "stddev: 0.002475333538185564",
            "extra": "mean: 346.4534 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.8924176088048967,
            "unit": "iter/sec",
            "range": "stddev: 0.002015591894615856",
            "extra": "mean: 345.7315 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5694376107001675,
            "unit": "iter/sec",
            "range": "stddev: 0.005597342649637997",
            "extra": "mean: 1.7561 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.229086340956825,
            "unit": "iter/sec",
            "range": "stddev: 0.053486932091405454",
            "extra": "mean: 4.3652 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10163159217831574,
            "unit": "iter/sec",
            "range": "stddev: 0.06885280381354371",
            "extra": "mean: 9.8395 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6444883397146379,
            "unit": "iter/sec",
            "range": "stddev: 0.008045694897445377",
            "extra": "mean: 1.5516 sec\nrounds: 3"
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
        "date": 1776338313031,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8925635122271274,
            "unit": "iter/sec",
            "range": "stddev: 0.004343559875071556",
            "extra": "mean: 345.7141 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9215062064581563,
            "unit": "iter/sec",
            "range": "stddev: 0.002079858664865692",
            "extra": "mean: 342.2892 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5686211511448482,
            "unit": "iter/sec",
            "range": "stddev: 0.011363881356843188",
            "extra": "mean: 1.7586 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23052244039568137,
            "unit": "iter/sec",
            "range": "stddev: 0.03370182329723057",
            "extra": "mean: 4.3380 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10081316834783403,
            "unit": "iter/sec",
            "range": "stddev: 0.02858030038957976",
            "extra": "mean: 9.9193 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6444277816948347,
            "unit": "iter/sec",
            "range": "stddev: 0.009999723183750921",
            "extra": "mean: 1.5518 sec\nrounds: 3"
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
        "date": 1776338312877,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.805114393624862,
            "unit": "iter/sec",
            "range": "stddev: 0.0013806848691246953",
            "extra": "mean: 356.4917 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.881086108429947,
            "unit": "iter/sec",
            "range": "stddev: 0.005951695450360424",
            "extra": "mean: 347.0913 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.570484512576437,
            "unit": "iter/sec",
            "range": "stddev: 0.005927839926108163",
            "extra": "mean: 1.7529 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23028841836517372,
            "unit": "iter/sec",
            "range": "stddev: 0.05227843008135557",
            "extra": "mean: 4.3424 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10105580639411346,
            "unit": "iter/sec",
            "range": "stddev: 0.006287192768568152",
            "extra": "mean: 9.8955 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6435055408787395,
            "unit": "iter/sec",
            "range": "stddev: 0.009555650322266874",
            "extra": "mean: 1.5540 sec\nrounds: 3"
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
        "date": 1776338312979,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.905737887905054,
            "unit": "iter/sec",
            "range": "stddev: 0.0011199435647329579",
            "extra": "mean: 344.1467 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.877868511337958,
            "unit": "iter/sec",
            "range": "stddev: 0.0032829994500922664",
            "extra": "mean: 347.4794 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.568523821223675,
            "unit": "iter/sec",
            "range": "stddev: 0.00552056383786554",
            "extra": "mean: 1.7589 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23180797079191254,
            "unit": "iter/sec",
            "range": "stddev: 0.010641393676449137",
            "extra": "mean: 4.3139 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10092120613895514,
            "unit": "iter/sec",
            "range": "stddev: 0.032671860935595765",
            "extra": "mean: 9.9087 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6447159755701823,
            "unit": "iter/sec",
            "range": "stddev: 0.00975899820783999",
            "extra": "mean: 1.5511 sec\nrounds: 3"
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
        "date": 1776338312965,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.913253611622944,
            "unit": "iter/sec",
            "range": "stddev: 0.0017143549263350377",
            "extra": "mean: 343.2588 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9068150863970024,
            "unit": "iter/sec",
            "range": "stddev: 0.0018969969373823518",
            "extra": "mean: 344.0191 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5666821497893016,
            "unit": "iter/sec",
            "range": "stddev: 0.004198950244335243",
            "extra": "mean: 1.7647 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23116122868190278,
            "unit": "iter/sec",
            "range": "stddev: 0.019405540872141747",
            "extra": "mean: 4.3260 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10080151636930851,
            "unit": "iter/sec",
            "range": "stddev: 0.02302633088713974",
            "extra": "mean: 9.9205 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.6449156249020314,
            "unit": "iter/sec",
            "range": "stddev: 0.010333080780598601",
            "extra": "mean: 1.5506 sec\nrounds: 3"
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
        "date": 1776338312940,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8267122706263215,
            "unit": "iter/sec",
            "range": "stddev: 0.005542657877158118",
            "extra": "mean: 353.7679 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.8481646268502,
            "unit": "iter/sec",
            "range": "stddev: 0.0023921532232615843",
            "extra": "mean: 351.1033 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.555866731604795,
            "unit": "iter/sec",
            "range": "stddev: 0.00392592701417723",
            "extra": "mean: 1.7990 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23579232825066884,
            "unit": "iter/sec",
            "range": "stddev: 0.04279310800701651",
            "extra": "mean: 4.2410 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09576396142914964,
            "unit": "iter/sec",
            "range": "stddev: 0.0183006319456797",
            "extra": "mean: 10.4423 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.5965383903255422,
            "unit": "iter/sec",
            "range": "stddev: 0.012167893592068558",
            "extra": "mean: 1.6763 sec\nrounds: 3"
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
        "date": 1776338313056,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.7780891923160156,
            "unit": "iter/sec",
            "range": "stddev: 0.0044997053500403774",
            "extra": "mean: 359.9596 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.8654684281760456,
            "unit": "iter/sec",
            "range": "stddev: 0.004293043026343432",
            "extra": "mean: 348.9831 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5590463114331506,
            "unit": "iter/sec",
            "range": "stddev: 0.004533000364270082",
            "extra": "mean: 1.7888 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23322836911053846,
            "unit": "iter/sec",
            "range": "stddev: 0.05587226116096153",
            "extra": "mean: 4.2876 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.0951998484118853,
            "unit": "iter/sec",
            "range": "stddev: 0.06677726218365365",
            "extra": "mean: 10.5042 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.597337020430868,
            "unit": "iter/sec",
            "range": "stddev: 0.011263170003225517",
            "extra": "mean: 1.6741 sec\nrounds: 3"
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
        "date": 1776338313038,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8131026960002536,
            "unit": "iter/sec",
            "range": "stddev: 0.004648208709649436",
            "extra": "mean: 355.4794 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.7936846238059307,
            "unit": "iter/sec",
            "range": "stddev: 0.00527024635857096",
            "extra": "mean: 357.9502 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5565841118625957,
            "unit": "iter/sec",
            "range": "stddev: 0.005732142640189796",
            "extra": "mean: 1.7967 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2356531821604418,
            "unit": "iter/sec",
            "range": "stddev: 0.01170601094980567",
            "extra": "mean: 4.2435 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09551002091812817,
            "unit": "iter/sec",
            "range": "stddev: 0.06065546800036372",
            "extra": "mean: 10.4701 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.5967988885698094,
            "unit": "iter/sec",
            "range": "stddev: 0.00970881239476316",
            "extra": "mean: 1.6756 sec\nrounds: 3"
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
        "date": 1776338312930,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.836935210391988,
            "unit": "iter/sec",
            "range": "stddev: 0.002365823397165913",
            "extra": "mean: 352.4931 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.825604911518068,
            "unit": "iter/sec",
            "range": "stddev: 0.0066489383244674035",
            "extra": "mean: 353.9065 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.553599517255314,
            "unit": "iter/sec",
            "range": "stddev: 0.00889282396144197",
            "extra": "mean: 1.8064 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23533569144120772,
            "unit": "iter/sec",
            "range": "stddev: 0.02935413334078881",
            "extra": "mean: 4.2492 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09626486456847276,
            "unit": "iter/sec",
            "range": "stddev: 0.018207601790541355",
            "extra": "mean: 10.3880 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.5975609456466876,
            "unit": "iter/sec",
            "range": "stddev: 0.009787020548755066",
            "extra": "mean: 1.6735 sec\nrounds: 3"
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
        "date": 1776338312847,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.059208366595248,
            "unit": "iter/sec",
            "range": "stddev: 0.001120776374953571",
            "extra": "mean: 326.8820 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0482728630141755,
            "unit": "iter/sec",
            "range": "stddev: 0.0010518120523679928",
            "extra": "mean: 328.0546 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5613280819292606,
            "unit": "iter/sec",
            "range": "stddev: 0.00653723565675367",
            "extra": "mean: 1.7815 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2411512100968555,
            "unit": "iter/sec",
            "range": "stddev: 0.0813105164682674",
            "extra": "mean: 4.1468 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09725970034044164,
            "unit": "iter/sec",
            "range": "stddev: 0.05344137202068646",
            "extra": "mean: 10.2818 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.5976648917113827,
            "unit": "iter/sec",
            "range": "stddev: 0.015653490994294913",
            "extra": "mean: 1.6732 sec\nrounds: 3"
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
        "date": 1776338312835,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0000799953328974,
            "unit": "iter/sec",
            "range": "stddev: 0.006201714450302475",
            "extra": "mean: 333.3244 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.003026754870535,
            "unit": "iter/sec",
            "range": "stddev: 0.009206995657371883",
            "extra": "mean: 332.9974 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5592320672973023,
            "unit": "iter/sec",
            "range": "stddev: 0.01736340529161756",
            "extra": "mean: 1.7882 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24240983881359593,
            "unit": "iter/sec",
            "range": "stddev: 0.028151578169771704",
            "extra": "mean: 4.1252 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09750080643209506,
            "unit": "iter/sec",
            "range": "stddev: 0.018350655619710735",
            "extra": "mean: 10.2563 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.5988814837852113,
            "unit": "iter/sec",
            "range": "stddev: 0.00821230224197401",
            "extra": "mean: 1.6698 sec\nrounds: 3"
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
        "date": 1776338313060,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.0629675727585925,
            "unit": "iter/sec",
            "range": "stddev: 0.0005003262688590382",
            "extra": "mean: 326.4808 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.062654281821982,
            "unit": "iter/sec",
            "range": "stddev: 0.0009706736661577302",
            "extra": "mean: 326.5142 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5622346685285404,
            "unit": "iter/sec",
            "range": "stddev: 0.0014758760795418522",
            "extra": "mean: 1.7786 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2424370436486908,
            "unit": "iter/sec",
            "range": "stddev: 0.03618248877905782",
            "extra": "mean: 4.1248 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09700564185019918,
            "unit": "iter/sec",
            "range": "stddev: 0.06898342232068558",
            "extra": "mean: 10.3087 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.5968858826839266,
            "unit": "iter/sec",
            "range": "stddev: 0.005539097402383168",
            "extra": "mean: 1.6754 sec\nrounds: 3"
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
            "name": "Philipp Jurašić",
            "username": "jurasic-pf",
            "email": "166746189+jurasic-pf@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "d79e972b074dedc10976091e00e12f68551d5fb3",
          "message": "Restrict pydantic version, breaking change (#480)\n\npydantic 2.13 is causing validation failiures, restrict it until we understand the issue",
          "timestamp": "2026-04-16T09:06:55Z",
          "url": "https://github.com/proximafusion/vmecpp/commit/d79e972b074dedc10976091e00e12f68551d5fb3"
        },
        "date": 1776338592851,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.0420765598142125,
            "unit": "iter/sec",
            "range": "stddev: 0.002014551686561701",
            "extra": "mean: 328.72282480000194 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.071681825583406,
            "unit": "iter/sec",
            "range": "stddev: 0.0010886654611289134",
            "extra": "mean: 325.55455179999626 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.24217181239000385,
            "unit": "iter/sec",
            "range": "stddev: 0.08927802373469163",
            "extra": "mean: 4.129299732000011 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5559724259456045,
            "unit": "iter/sec",
            "range": "stddev: 0.00169973232673371",
            "extra": "mean: 1.79865035266666 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.24780756895627773,
            "unit": "iter/sec",
            "range": "stddev: 0.0031220087120115755",
            "extra": "mean: 4.035389250666658 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09666303317435072,
            "unit": "iter/sec",
            "range": "stddev: 0.09915692862364608",
            "extra": "mean: 10.345216440666661 sec\nrounds: 3"
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
          "id": "48dd1a146c28f3a06e53af22dbc2e8d2c32c38cc",
          "message": "Also add .github/workflows/copilot-setup-steps.yml to pre-install vmecpp and its Ubuntu system dependencies in the Copilot cloud agent environment. (#483)\n\nAlso add .github/workflows/copilot-setup-steps.yml to pre-install vmecpp and\nits Ubuntu system dependencies in the Copilot cloud agent environment.",
          "timestamp": "2026-04-16T13:33:19+02:00",
          "tree_id": "1aed1e09f2dffc1c1fdbc2e0601a7f79d30d2d0d",
          "url": "https://github.com/proximafusion/vmecpp/commit/48dd1a146c28f3a06e53af22dbc2e8d2c32c38cc"
        },
        "date": 1776339448637,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9749995431442224,
            "unit": "iter/sec",
            "range": "stddev: 0.0010899525310423717",
            "extra": "mean: 336.1345053999969 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9437706078293844,
            "unit": "iter/sec",
            "range": "stddev: 0.00037705187844342823",
            "extra": "mean: 339.7003819999952 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23619984761903837,
            "unit": "iter/sec",
            "range": "stddev: 0.030243292864653184",
            "extra": "mean: 4.233702985333328 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5670677980271094,
            "unit": "iter/sec",
            "range": "stddev: 0.0006167950069048943",
            "extra": "mean: 1.7634575679999973 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.27916840285839034,
            "unit": "iter/sec",
            "range": "stddev: 0.0019631419982103263",
            "extra": "mean: 3.5820672746666653 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1013920529199509,
            "unit": "iter/sec",
            "range": "stddev: 0.004145177924291285",
            "extra": "mean: 9.862705914333352 sec\nrounds: 3"
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
          "id": "8adda439309a04d379648e25c4f64f2c5f3bdedb",
          "message": "Add a Python 3.13 Nix development shell (#479)\n\n* Add a Python 3.13 Nix development shell\n\n* Update README.md\n\nCo-authored-by: Philipp Jurašić <166746189+jurasic-pf@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Philipp Jurašić <166746189+jurasic-pf@users.noreply.github.com>",
          "timestamp": "2026-04-16T14:44:28+02:00",
          "tree_id": "e8e3f2cfad15e4222d2e4ee2fbb465245fd99bf3",
          "url": "https://github.com/proximafusion/vmecpp/commit/8adda439309a04d379648e25c4f64f2c5f3bdedb"
        },
        "date": 1776343718838,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9609476989705588,
            "unit": "iter/sec",
            "range": "stddev: 0.0007313969007659451",
            "extra": "mean: 337.7297073999898 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9451254767322927,
            "unit": "iter/sec",
            "range": "stddev: 0.0038709303114931146",
            "extra": "mean: 339.5441069999947 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23418334402193228,
            "unit": "iter/sec",
            "range": "stddev: 0.026700440493864684",
            "extra": "mean: 4.270158512666664 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5704369339674493,
            "unit": "iter/sec",
            "range": "stddev: 0.0032514267888857896",
            "extra": "mean: 1.7530421689999873 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2784592097744244,
            "unit": "iter/sec",
            "range": "stddev: 0.012022105711454881",
            "extra": "mean: 3.591190253000017 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.1011718573075453,
            "unit": "iter/sec",
            "range": "stddev: 0.016143795089124283",
            "extra": "mean: 9.884171612666648 sec\nrounds: 3"
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
          "id": "fa1b951045b2243a0c9340d0b16bec4a60e5136e",
          "message": "Update README.md",
          "timestamp": "2026-04-23T10:12:38+02:00",
          "tree_id": "f3fac97cd72cc9ec7d5fd85ca957778291adbb65",
          "url": "https://github.com/proximafusion/vmecpp/commit/fa1b951045b2243a0c9340d0b16bec4a60e5136e"
        },
        "date": 1776932211690,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.898830803788288,
            "unit": "iter/sec",
            "range": "stddev: 0.0039863085112753355",
            "extra": "mean: 344.96666679999635 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.9038694620449017,
            "unit": "iter/sec",
            "range": "stddev: 0.0019802123097922655",
            "extra": "mean: 344.3680967999853 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23531860185546885,
            "unit": "iter/sec",
            "range": "stddev: 0.00997675780302466",
            "extra": "mean: 4.249557799999991 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5681926696932987,
            "unit": "iter/sec",
            "range": "stddev: 0.005666669198639517",
            "extra": "mean: 1.759966387000001 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.2785109390511287,
            "unit": "iter/sec",
            "range": "stddev: 0.015564478277814223",
            "extra": "mean: 3.590523242666677 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10146232267148701,
            "unit": "iter/sec",
            "range": "stddev: 0.013614098861036131",
            "extra": "mean: 9.855875300999989 sec\nrounds: 3"
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
          "id": "51527dfb71c83aee7a8d83df6203758e2c78ef18",
          "message": "Miller recurrence relation for numerically stable NESTOR at high mpol/ntor (#484)\n\nA stable recurrence relation that will help free boundary convergence of any free boundary computation >=12 mpol or ntor. \r\n\r\n![image.png](https://app.graphite.com/user-attachments/assets/54557a41-e8c4-40d7-8184-68fc73a44ce9.png)\r\n\r\nSpuriously growing error at high mode numbers",
          "timestamp": "2026-04-24T12:28:08Z",
          "tree_id": "46ee572b3021f833c339ffbbac9effe65b0f54e2",
          "url": "https://github.com/proximafusion/vmecpp/commit/51527dfb71c83aee7a8d83df6203758e2c78ef18"
        },
        "date": 1777033939855,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.9422803085841718,
            "unit": "iter/sec",
            "range": "stddev: 0.0012329257788555586",
            "extra": "mean: 339.8724441999889 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.928693501997443,
            "unit": "iter/sec",
            "range": "stddev: 0.00237653330579315",
            "extra": "mean: 341.44918179999877 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23291252961650874,
            "unit": "iter/sec",
            "range": "stddev: 0.032161959000972094",
            "extra": "mean: 4.293457297666653 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5679117544156642,
            "unit": "iter/sec",
            "range": "stddev: 0.008313450350129358",
            "extra": "mean: 1.7608369473333407 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.27916794504298825,
            "unit": "iter/sec",
            "range": "stddev: 0.009364344750837604",
            "extra": "mean: 3.582073149000015 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10004713222066393,
            "unit": "iter/sec",
            "range": "stddev: 0.020810049308553808",
            "extra": "mean: 9.995288998333308 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "198982749+Copilot@users.noreply.github.com",
            "name": "Copilot",
            "username": "Copilot"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f3f1a1068147acd40e45ed847d5a8cc06015b2a3",
          "message": "fix: write test output files to temp directories instead of CWD (#489)\n\nfix: create cma.json and input.test in temp directories during tests\n\nAgent-Logs-Url: https://github.com/proximafusion/vmecpp/sessions/ac967efc-3dbd-45b3-8844-7369cb137209\n\nCo-authored-by: copilot-swe-agent[bot] <198982749+Copilot@users.noreply.github.com>\nCo-authored-by: jurasic-pf <166746189+jurasic-pf@users.noreply.github.com>",
          "timestamp": "2026-04-27T10:41:19+02:00",
          "tree_id": "d77249fb049a39855dbb393d8b2771a785f8e3b0",
          "url": "https://github.com/proximafusion/vmecpp/commit/f3f1a1068147acd40e45ed847d5a8cc06015b2a3"
        },
        "date": 1777279533929,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.133377765005985,
            "unit": "iter/sec",
            "range": "stddev: 0.00258390138788545",
            "extra": "mean: 319.14441060000627 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.145997092444199,
            "unit": "iter/sec",
            "range": "stddev: 0.0006765663096218686",
            "extra": "mean: 317.86424800001214 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.20094960150214625,
            "unit": "iter/sec",
            "range": "stddev: 0.009964249369552316",
            "extra": "mean: 4.976372147666685 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5642366774702058,
            "unit": "iter/sec",
            "range": "stddev: 0.005611477276268905",
            "extra": "mean: 1.772305913333336 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.30293514672498484,
            "unit": "iter/sec",
            "range": "stddev: 0.02767194728609154",
            "extra": "mean: 3.301036577666688 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10630145881855992,
            "unit": "iter/sec",
            "range": "stddev: 0.048674544790800155",
            "extra": "mean: 9.407208622666644 sec\nrounds: 3"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "198982749+Copilot@users.noreply.github.com",
            "name": "Copilot",
            "username": "Copilot"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9c2041a1d23a6d9c69f9d98caff279d7166f4282",
          "message": "Fix normalize_by_currents having no effect on MagneticFieldResponseTable (#487)\n\n`normalize_by_currents` was silently ignored in `ComputeMagneticFieldResponseTable` and `ComputeVectorPotentialCache`. The coils file parser always sets `serial_circuit.current = 1.0`, so normalizing by setting the current to `1.0` was a no-op — the actual current information lives in `coil.num_windings`, which was never factored in. Normalized and raw results were always identical.\n\n## Changes\n\n- **`makegrid_lib.cc`** — When `normalize_by_currents = true`, call `NumWindingsToCircuitCurrents()` internally before computing. This migrates `num_windings` into `serial_circuit.current` (making `current = original_current × num_windings`, `num_windings = ±1`) so that subsequently setting `current = 1.0` correctly yields field per unit current-turn. A `MagneticConfiguration` copy is only created when normalization is requested. Applied identically to both `ComputeMagneticFieldResponseTable` and `ComputeVectorPotentialCache`.\n\n- **`makegrid.cc`** — Removed the now-redundant explicit `NumWindingsToCircuitCurrents` call; the migration is now encapsulated inside the library functions.\n\n- **`makegrid_lib_test.cc`** — Added `CheckNormalizeByCurrentsScalesMagneticFieldResponseTable`: builds a single-circuit config with `current = 5.0`, `num_windings = 7`, and asserts that the raw response table equals the normalized table scaled by `current × num_windings = 35` at every grid point.\n\n- **`tests/test_free_boundary.py`** — Changed `test_magnetic_field_response_table_loading` to use `normalize_by_currents=False`. The test coils file has non-uniform `num_windings` across coils within a circuit (e.g. 96 and −16), which is incompatible with `NumWindingsToCircuitCurrents` and makes the normalization semantics undefined for that configuration.\n\n## Before / After\n\n```python\n# Before fix: normalized == raw (normalize_by_currents had no effect)\nraw   = MagneticFieldResponseTable.from_coils_file(coils, params_raw)\nnormd = MagneticFieldResponseTable.from_coils_file(coils, params_normalized)\nassert np.allclose(raw.b_r, normd.b_r)  # True — bug\n\n# After fix: raw == normalized * current * num_windings\nassert np.allclose(raw.b_r, normd.b_r * current * num_windings)  # True — correct\n```",
          "timestamp": "2026-04-27T14:21:51Z",
          "tree_id": "b2a4004f87cba6621d18e876a2da25becd03f3da",
          "url": "https://github.com/proximafusion/vmecpp/commit/9c2041a1d23a6d9c69f9d98caff279d7166f4282"
        },
        "date": 1777299975041,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.049966320289393,
            "unit": "iter/sec",
            "range": "stddev: 0.002430211213190244",
            "extra": "mean: 327.8724730000022 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.052623802597628,
            "unit": "iter/sec",
            "range": "stddev: 0.0011701277589351803",
            "extra": "mean: 327.5870413999428 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2440246791997162,
            "unit": "iter/sec",
            "range": "stddev: 0.011949913172248972",
            "extra": "mean: 4.097946171999979 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5570816407863026,
            "unit": "iter/sec",
            "range": "stddev: 0.006943725269760385",
            "extra": "mean: 1.7950690290000086 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.24765920692432888,
            "unit": "iter/sec",
            "range": "stddev: 0.010785219993034572",
            "extra": "mean: 4.037806679666649 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.09715062746895459,
            "unit": "iter/sec",
            "range": "stddev: 0.005283342459474938",
            "extra": "mean: 10.293294300333363 sec\nrounds: 3"
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
          "id": "d6875c57529a77d7e32e6845b9d3ee40afdda5b3",
          "message": "Abseil errors instead of LOG(FATAL) (#493)\n\n`LOG(FATAL) <<`  -> `return absl::InvalidArgumentError(` in a few places.\nclang-tidy is enabled in pre-commit, so whitespace formatting changed.",
          "timestamp": "2026-04-27T15:58:22Z",
          "tree_id": "a4313fa42f5838bf688910bffd3f9f3d68f9827b",
          "url": "https://github.com/proximafusion/vmecpp/commit/d6875c57529a77d7e32e6845b9d3ee40afdda5b3"
        },
        "date": 1777305780720,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 2.8508630558275074,
            "unit": "iter/sec",
            "range": "stddev: 0.004830365187741993",
            "extra": "mean: 350.770970200017 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 2.8708363495344122,
            "unit": "iter/sec",
            "range": "stddev: 0.001227015915010409",
            "extra": "mean: 348.3305484000084 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.23166512118964622,
            "unit": "iter/sec",
            "range": "stddev: 0.06057754256214197",
            "extra": "mean: 4.316575559 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.5627987472932756,
            "unit": "iter/sec",
            "range": "stddev: 0.0028650724069687206",
            "extra": "mean: 1.7768340900000226 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.277522072743032,
            "unit": "iter/sec",
            "range": "stddev: 0.009963664380678233",
            "extra": "mean: 3.603316990666675 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.10015096184125961,
            "unit": "iter/sec",
            "range": "stddev: 0.05064998514239424",
            "extra": "mean: 9.984926570999997 sec\nrounds: 3"
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
          "id": "97cb9d20e1c85fdf05f9dbfae0549754daacbf4a",
          "message": "Scientific variable names get flagged incorrectly too often by clang-tidy (#491)",
          "timestamp": "2026-04-27T16:18:23Z",
          "tree_id": "b23af3b37dab107f49bb1792b9e40ac499ab3af8",
          "url": "https://github.com/proximafusion/vmecpp/commit/97cb9d20e1c85fdf05f9dbfae0549754daacbf4a"
        },
        "date": 1777306932866,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
            "value": 3.7056034073670645,
            "unit": "iter/sec",
            "range": "stddev: 0.006248319428713388",
            "extra": "mean: 269.8615825999923 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
            "value": 3.6111423206092796,
            "unit": "iter/sec",
            "range": "stddev: 0.009300994299441805",
            "extra": "mean: 276.92068359999666 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
            "value": 0.2894082849600834,
            "unit": "iter/sec",
            "range": "stddev: 0.01597848474763014",
            "extra": "mean: 3.45532609800001 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
            "value": 0.7076386474609254,
            "unit": "iter/sec",
            "range": "stddev: 0.002258206565710031",
            "extra": "mean: 1.4131506293333398 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
            "value": 0.3199383488869086,
            "unit": "iter/sec",
            "range": "stddev: 0.02556928663074986",
            "extra": "mean: 3.125602177666669 sec\nrounds: 3"
          },
          {
            "name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
            "value": 0.12222517154865759,
            "unit": "iter/sec",
            "range": "stddev: 0.030543208231392685",
            "extra": "mean: 8.181620752333345 sec\nrounds: 3"
          }
        ]
      }
    ]
  }
}
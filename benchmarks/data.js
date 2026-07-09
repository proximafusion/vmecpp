window.BENCHMARK_DATA = {
"lastUpdate": 1783589338352,
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338166,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6049421236000171,
"range": "stddev: 0.010845948938617215",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5957320195999956,
"range": "stddev: 0.00841539379024821",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4113570186666493,
"range": "stddev: 0.0022298069127498618",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8165107519999992,
"range": "stddev: 0.008412239087879626",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338081,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6092223206000199,
"range": "stddev: 0.004369283394875951",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5989546366000014,
"range": "stddev: 0.009460458931711525",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4177695260000291,
"range": "stddev: 0.009562814166423293",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8157433320000487,
"range": "stddev: 0.010753652348835312",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
"email": "166746189+jurasic-pf@users.noreply.github.com",
"username": ""
},
"committer": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338169,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5993044305999774,
"range": "stddev: 0.005182845236499909",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6136232127999847,
"range": "stddev: 0.018749737102218677",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4097477329999795,
"range": "stddev: 0.001971511931099196",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.816471428,
"range": "stddev: 0.03278713022914895",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338033,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5929190311999946,
"range": "stddev: 0.006406701733503289",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5962337231999981,
"range": "stddev: 0.010391611188430858",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4198665816666864,
"range": "stddev: 0.002641869183869476",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7969782180000116,
"range": "stddev: 0.0188988197565358",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338095,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6478068672000064,
"range": "stddev: 0.01037409347789336",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6397786657999405,
"range": "stddev: 0.012102449493484883",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4346189626667183,
"range": "stddev: 0.00861406443421678",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.840563995333317,
"range": "stddev: 0.01406591429136527",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338209,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.59090095319998,
"range": "stddev: 0.0025784571692820434",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5937837668000157,
"range": "stddev: 0.004191723575777942",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4213250103334,
"range": "stddev: 0.008404695792002209",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7875861269999405,
"range": "stddev: 0.019614670600219826",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338118,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5964387207999607,
"range": "stddev: 0.008528108174528764",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5978986459998851,
"range": "stddev: 0.005916834922508908",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.416996119333362,
"range": "stddev: 0.004429933299541528",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7962016730000414,
"range": "stddev: 0.016710635088990188",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338173,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.595066503399994,
"range": "stddev: 0.004269539037729011",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5945102719999795,
"range": "stddev: 0.005528069122642723",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4241225963333666,
"range": "stddev: 0.009877786624307019",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7918241263332675,
"range": "stddev: 0.008495938576758425",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338134,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6354506312000012,
"range": "stddev: 0.013209019211201066",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6144302298000184,
"range": "stddev: 0.011671320729814201",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.424058371000001,
"range": "stddev: 0.005944267897708213",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8573344150000444,
"range": "stddev: 0.03448994801951164",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
"email": "166746189+jurasic-pf@users.noreply.github.com",
"username": ""
},
"committer": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338086,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6029726511998887,
"range": "stddev: 0.008167525318611077",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5956084628000099,
"range": "stddev: 0.004669099880459024",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.42284340233338,
"range": "stddev: 0.004656952433906125",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.803613297999997,
"range": "stddev: 0.01738862360447161",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338088,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30621746340002576,
"range": "stddev: 0.004327603739117211",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30651056699998663,
"range": "stddev: 0.0015638066742787377",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.414923963000092,
"range": "stddev: 0.008127902908606986",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.827333763999983,
"range": "stddev: 0.01708589690561527",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
"email": "166746189+jurasic-pf@users.noreply.github.com",
"username": ""
},
"committer": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338070,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3071541694000189,
"range": "stddev: 0.0021154886173705796",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3195858538000266,
"range": "stddev: 0.0016198664094469013",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4239939996666635,
"range": "stddev: 0.012694011416235702",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8037987696666278,
"range": "stddev: 0.017203883337843753",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
"email": "166746189+jurasic-pf@users.noreply.github.com",
"username": ""
},
"committer": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589337980,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3078090315999361,
"range": "stddev: 0.004311112260740319",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30743307359994104,
"range": "stddev: 0.004693532119943324",
"extra": "rounds: 5"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338144,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3150973774000249,
"range": "stddev: 0.006161223355724151",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3162427690000641,
"range": "stddev: 0.0009626538920120508",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4240375609999774,
"range": "stddev: 0.005020758428742842",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8460015086667076,
"range": "stddev: 0.02225192508382105",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589337993,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.32088386739997077,
"range": "stddev: 0.004080117240516881",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31850437840007545,
"range": "stddev: 0.0022858153412821814",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4298894010000065,
"range": "stddev: 0.00912383484544472",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8493693433332132,
"range": "stddev: 0.01230027520559314",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589337991,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31072924600002805,
"range": "stddev: 0.0027324366403574077",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3131998624000062,
"range": "stddev: 0.0030235065828626956",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4209814396667753,
"range": "stddev: 0.005709752921093568",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.837881664333357,
"range": "stddev: 0.009502679245131576",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338058,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3125997439999992,
"range": "stddev: 0.002734017524866199",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3098371252000106,
"range": "stddev: 0.0010193487175929052",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4212942943332412,
"range": "stddev: 0.0036776114221870157",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.833002047666772,
"range": "stddev: 0.016503138591124223",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338072,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3257973403999131,
"range": "stddev: 0.004322383534361901",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3288615863999894,
"range": "stddev: 0.008527214135263219",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4207764539999819,
"range": "stddev: 0.0015210963954447546",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8524892529999306,
"range": "stddev: 0.022456079605781755",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338012,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3151674004000597,
"range": "stddev: 0.001523290460540358",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3154187255999659,
"range": "stddev: 0.0018600628982022773",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4302037683332856,
"range": "stddev: 0.008139413542864718",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8483102716666813,
"range": "stddev: 0.03005803643131687",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338187,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3175609922000149,
"range": "stddev: 0.003052679211796122",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3119036820000474,
"range": "stddev: 0.003627650883327093",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4171556833334762,
"range": "stddev: 0.0021857275117470555",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8453585519999556,
"range": "stddev: 0.025700479832465147",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338157,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3212431431999903,
"range": "stddev: 0.007748995948822286",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31785965039989605,
"range": "stddev: 0.005933095809656677",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.419125198999912,
"range": "stddev: 0.0031234463101822074",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.858827152666587,
"range": "stddev: 0.03756640007784335",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
"email": "166746189+jurasic-pf@users.noreply.github.com",
"username": ""
},
"committer": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338162,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3238717967999946,
"range": "stddev: 0.004356363379231423",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.317312851600127,
"range": "stddev: 0.005221849861425966",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4170926003334898,
"range": "stddev: 0.003054881729842562",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8260325419998176,
"range": "stddev: 0.01909457809709606",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338091,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3201672562000567,
"range": "stddev: 0.005718165375399082",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3163395098001274,
"range": "stddev: 0.0018600821376790857",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.428422133666724,
"range": "stddev: 0.0060955745992656955",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.83492308766669,
"range": "stddev: 0.025198803275664147",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338049,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3220439606000582,
"range": "stddev: 0.00481106652746372",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3199220260000402,
"range": "stddev: 0.003217229340021404",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4242278136666755,
"range": "stddev: 0.0017396749807886375",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8654319076666375,
"range": "stddev: 0.02044397381406482",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
"email": "166746189+jurasic-pf@users.noreply.github.com",
"username": ""
},
"committer": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338132,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2934741330000065,
"range": "stddev: 0.0008596217621153431",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2944824927999946,
"range": "stddev: 0.0031831530906799455",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5800904443333177,
"range": "stddev: 0.019505018591436895",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.4122064909999965,
"range": "stddev: 0.005824925311992268",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338102,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3017025363999892,
"range": "stddev: 0.005140501003339664",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2911831047999954,
"range": "stddev: 0.004373965930081991",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5735266366666565,
"range": "stddev: 0.004491795783796313",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.369496150000013,
"range": "stddev: 0.06791436021553966",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338185,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.29297739260000527,
"range": "stddev: 0.00276441345941973",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2825789048000047,
"range": "stddev: 0.0022886155639640546",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5616181616666533,
"range": "stddev: 0.00917876868155324",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.342874057999988,
"range": "stddev: 0.027771316297814756",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338155,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2821153525999989,
"range": "stddev: 0.0021288753659145527",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28499360980000576,
"range": "stddev: 0.002295520179448596",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5615489203333368,
"range": "stddev: 0.008646275502279225",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.351291280333347,
"range": "stddev: 0.04522379259622371",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589337998,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28486820739997254,
"range": "stddev: 0.004151314673780135",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2926459914000361,
"range": "stddev: 0.001854527028951082",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5641999759999787,
"range": "stddev: 0.00799566050645155",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.353205273333326,
"range": "stddev: 0.041800961625088735",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338130,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2808060225999952,
"range": "stddev: 0.0022437774680822",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27836452660001215,
"range": "stddev: 0.0035687407216328823",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.558039264666642,
"range": "stddev: 0.011624552153226957",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.336771435333351,
"range": "stddev: 0.017785345014119297",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338002,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28503323980003187,
"range": "stddev: 0.001755114142317753",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28356582559999877,
"range": "stddev: 0.0025654990529975887",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5733034326666484,
"range": "stddev: 0.008769666470988046",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.402866815666698,
"range": "stddev: 0.005985979915249805",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338199,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2823519122000107,
"range": "stddev: 0.001694592263242132",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2829723600000079,
"range": "stddev: 0.002932688258764294",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5580496220000366,
"range": "stddev: 0.018471161173584386",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.364552454000015,
"range": "stddev: 0.027988122522106022",
"extra": "rounds: 3"
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
"date": 1783589338223,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2901002035999909,
"range": "stddev: 0.0017810672324870506",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.29324878579998315,
"range": "stddev: 0.002410951848039104",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5576831986666473,
"range": "stddev: 0.003381991527832695",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.410717685666593,
"range": "stddev: 0.010124386264485067",
"extra": "rounds: 3"
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
"date": 1783589338009,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2868924730000117,
"range": "stddev: 0.012212904404089401",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2771713171999636,
"range": "stddev: 0.002206486120581934",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5661855773333475,
"range": "stddev: 0.004245447147901006",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.366812533666651,
"range": "stddev: 0.02842130597240725",
"extra": "rounds: 3"
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
"date": 1783589338084,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2888534150000396,
"range": "stddev: 0.0013426440533053645",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2867062923999811,
"range": "stddev: 0.002469977613942284",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5463820276666336,
"range": "stddev: 0.010504410784925332",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.365379170999934,
"range": "stddev: 0.01960535178000235",
"extra": "rounds: 3"
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
"date": 1783589337977,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2860231082000155,
"range": "stddev: 0.005977851287277419",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2850962437999442,
"range": "stddev: 0.0026841502292095963",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5607465686667485,
"range": "stddev: 0.005571673655487649",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.35717724066664,
"range": "stddev: 0.03353294116511723",
"extra": "rounds: 3"
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
"date": 1783589338123,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2829002676000073,
"range": "stddev: 0.0016717443025640539",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28594322599997213,
"range": "stddev: 0.0020691118791220887",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5488633799999814,
"range": "stddev: 0.004152838048723596",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.292045946666728,
"range": "stddev: 0.02328555629254834",
"extra": "rounds: 3"
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
"date": 1783589338063,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28381782639994524,
"range": "stddev: 0.0023578901273772846",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2855379781999545,
"range": "stddev: 0.0019197815203278806",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5411531520000306,
"range": "stddev: 0.020383067533926554",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.357765968333297,
"range": "stddev: 0.016092741725586033",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338065,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2780591039999763,
"range": "stddev: 0.002258808519452905",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2793612001999463,
"range": "stddev: 0.00288116383670166",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5700973150000361,
"range": "stddev: 0.00441164599389577",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.305260649666555,
"range": "stddev: 0.01793321453777678",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338189,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28100672840000696,
"range": "stddev: 0.0023327434356579186",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28390001820007454,
"range": "stddev: 0.002528079461474811",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5687794926666356,
"range": "stddev: 0.006793154834203884",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.323770445333291,
"range": "stddev: 0.012408790143471087",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338211,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2818809976000466,
"range": "stddev: 0.001637308703174723",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2807617367999683,
"range": "stddev: 0.0036645882814804356",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5505632809999952,
"range": "stddev: 0.005733970061783101",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.332636253666654,
"range": "stddev: 0.018051193208836594",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338204,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.280381002599961,
"range": "stddev: 0.0023751595634620187",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2810116804000245,
"range": "stddev: 0.0011902632971256318",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5493050256666265,
"range": "stddev: 0.005045962659465629",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.37288837866663,
"range": "stddev: 0.05490318756222207",
"extra": "rounds: 3"
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
"date": 1783589338164,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2805900783999277,
"range": "stddev: 0.0012924369178181556",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27700172279996876,
"range": "stddev: 0.004470082637790039",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5682488229999005,
"range": "stddev: 0.04127791577925546",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.346455534333245,
"range": "stddev: 0.043995604988952115",
"extra": "rounds: 3"
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
"date": 1783589338112,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2826688736000051,
"range": "stddev: 0.00265225335012841",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28192941500005875,
"range": "stddev: 0.0015990547514341319",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5650796093333763,
"range": "stddev: 0.001805385586705998",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.358801100666521,
"range": "stddev: 0.012174765926070901",
"extra": "rounds: 3"
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
"date": 1783589338030,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2814107225999578,
"range": "stddev: 0.0023107298530216764",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2804305780000959,
"range": "stddev: 0.001929580975421554",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.561963526999989,
"range": "stddev: 0.007995714736870682",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.328612352333266,
"range": "stddev: 0.015511700869374239",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338121,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2870866887999455,
"range": "stddev: 0.008165280517956362",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2828427474000364,
"range": "stddev: 0.0018510167578393067",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5462527673333473,
"range": "stddev: 0.01134309780179137",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.352814982666511,
"range": "stddev: 0.054816478356183994",
"extra": "rounds: 3"
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
"date": 1783589338216,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27886264160006247,
"range": "stddev: 0.002705174026838571",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2776933687999644,
"range": "stddev: 0.0022703367783021805",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5405663459999535,
"range": "stddev: 0.015680894843095365",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.260002813333358,
"range": "stddev: 0.011839666105569534",
"extra": "rounds: 3"
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
"date": 1783589338021,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27979864500002805,
"range": "stddev: 0.0021999184283789373",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.274665406800068,
"range": "stddev: 0.0008024340909043755",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5125455760000175,
"range": "stddev: 0.01079968837661348",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.231734278666636,
"range": "stddev: 0.023288461644791966",
"extra": "rounds: 3"
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
"date": 1783589338160,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27711905520000074,
"range": "stddev: 0.0012289249671118415",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27580524319996585,
"range": "stddev: 0.0013098670307416585",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.523186347000016,
"range": "stddev: 0.008226403230563496",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.257972518666823,
"range": "stddev: 0.012531564971826262",
"extra": "rounds: 3"
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
"date": 1783589338141,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2758301909999773,
"range": "stddev: 0.0018555301574459605",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.274480182599973,
"range": "stddev: 0.0007432306513511377",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.525216029666808,
"range": "stddev: 0.0094391776346834",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.247645221666592,
"range": "stddev: 0.01581443588482965",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338176,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27576535979997063,
"range": "stddev: 0.0002752371813540678",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27603983779990815,
"range": "stddev: 0.0013876088087232243",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5361874690000452,
"range": "stddev: 0.0007453473511488262",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.291485204666515,
"range": "stddev: 0.027198030617863727",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338047,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2777800904000287,
"range": "stddev: 0.0013284556612147213",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2773135228000683,
"range": "stddev: 0.0016033543765905675",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.5135748993334346,
"range": "stddev: 0.01870106663607315",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.295626636000027,
"range": "stddev: 0.009619736565140256",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338061,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31102273660001173,
"range": "stddev: 0.0024549141001410715",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.32506448619999445,
"range": "stddev: 0.002308566655711128",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.416527609666673,
"range": "stddev: 0.0010917777183597139",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8280593213333227,
"range": "stddev: 0.02634663548602493",
"extra": "rounds: 3"
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
"date": 1783589338182,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.32155654159998903,
"range": "stddev: 0.0030843186618982952",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3229244335999965,
"range": "stddev: 0.0039801647473097276",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4202110349999657,
"range": "stddev: 0.003386827328211666",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.79683744066665,
"range": "stddev: 0.0035906056868271375",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338019,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.32437513559998477,
"range": "stddev: 0.0073902451852540735",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.32148030779999315,
"range": "stddev: 0.004291222783407103",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4166622396666828,
"range": "stddev: 0.0031876882853772025",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8169540100000177,
"range": "stddev: 0.01375449374733834",
"extra": "rounds: 3"
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
"date": 1783589337975,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3242117334000113,
"range": "stddev: 0.0038009879459619447",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3189731268000287,
"range": "stddev: 0.0033068075690301734",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4206384713333666,
"range": "stddev: 0.0016740334228135768",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8064742849999598,
"range": "stddev: 0.004839751415327457",
"extra": "rounds: 3"
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
"date": 1783589338074,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3245883873999901,
"range": "stddev: 0.011916004023608952",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3169508420000057,
"range": "stddev: 0.003640649636442087",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4194592366667014,
"range": "stddev: 0.0015828296977195115",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.815315364333363,
"range": "stddev: 0.014537578189307397",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338178,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31908211900004063,
"range": "stddev: 0.004063216977118901",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31653821539998717,
"range": "stddev: 0.002496446777670363",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4281618123333146,
"range": "stddev: 0.01286501513068295",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8023468280000166,
"range": "stddev: 0.021006444988314384",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589337996,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3112013093999394,
"range": "stddev: 0.0025949122254110045",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31807667860002764,
"range": "stddev: 0.003455743863272943",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4290748126666888,
"range": "stddev: 0.016679800757006296",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.792464584333402,
"range": "stddev: 0.028242174354183452",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589337986,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.308940756200036,
"range": "stddev: 0.001628363500854415",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3059274032000303,
"range": "stddev: 0.0013775325093208998",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4173285786667595,
"range": "stddev: 0.0032868127660577572",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7812812979999912,
"range": "stddev: 0.003761592294624126",
"extra": "rounds: 3"
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
"date": 1783589338219,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3146996238000156,
"range": "stddev: 0.0027542937864274944",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31078998379994116,
"range": "stddev: 0.003665627048373407",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4242061790000662,
"range": "stddev: 0.003222647506738329",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.807730306666675,
"range": "stddev: 0.017840809909404665",
"extra": "rounds: 3"
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
"date": 1783589337982,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31639782800002647,
"range": "stddev: 0.001957517720863795",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31207602260001294,
"range": "stddev: 0.0024076890809667222",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4237228243332538,
"range": "stddev: 0.007900279567410705",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.783103078666651,
"range": "stddev: 0.007917230466391163",
"extra": "rounds: 3"
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
"date": 1783589338192,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3134248872000171,
"range": "stddev: 0.005079021397861176",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3150751990000572,
"range": "stddev: 0.0036328725756621027",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4158758863332725,
"range": "stddev: 0.002006942520644437",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7927112919999977,
"range": "stddev: 0.010161640653889116",
"extra": "rounds: 3"
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
"date": 1783589338052,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30813129939992906,
"range": "stddev: 0.0013561641026282193",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3062388925999585,
"range": "stddev: 0.0030855237905940843",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.417579356333287,
"range": "stddev: 0.00437962188465705",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.795452488333391,
"range": "stddev: 0.01576034530895797",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338023,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31048360220001997,
"range": "stddev: 0.008120066199443367",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31273727819998387,
"range": "stddev: 0.01614523475764789",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4109123913333406,
"range": "stddev: 0.0018713809668138522",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7756073396665975,
"range": "stddev: 0.01622372801070839",
"extra": "rounds: 3"
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
"date": 1783589338105,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30763931979990955,
"range": "stddev: 0.0024876409912233527",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3122554503999709,
"range": "stddev: 0.00162765850067496",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4152370330000394,
"range": "stddev: 0.0031456125603791414",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.795918428333304,
"range": "stddev: 0.036770932330928985",
"extra": "rounds: 3"
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
"date": 1783589338180,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3115370200000143,
"range": "stddev: 0.0031570260216047708",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31070905920005315,
"range": "stddev: 0.002400452637213988",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4206216886667182,
"range": "stddev: 0.00624832075803926",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.78918859766668,
"range": "stddev: 0.020378995627268245",
"extra": "rounds: 3"
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
"date": 1783589338116,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30933565460004503,
"range": "stddev: 0.004109817848802819",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3088521482000033,
"range": "stddev: 0.002198726645391835",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4148273609999553,
"range": "stddev: 0.0069339678993157955",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.774383427333381,
"range": "stddev: 0.012577286519336775",
"extra": "rounds: 3"
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
"date": 1783589338028,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31340940380000576,
"range": "stddev: 0.0022192733641647395",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3065151987999798,
"range": "stddev: 0.0011551828300670115",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4167509336666626,
"range": "stddev: 0.010056162528043871",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7728180616666123,
"range": "stddev: 0.031093948839299685",
"extra": "rounds: 3"
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
"date": 1783589338035,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3165774371999305,
"range": "stddev: 0.0017214143117909098",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.32073603699996056,
"range": "stddev: 0.006143655145810128",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4088086616667017,
"range": "stddev: 0.0009909168709376086",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7646386776668046,
"range": "stddev: 0.02473172526911718",
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
"id": "7e63d9e5d5331d1147c5b5008f8e5786d42e81d4",
"message": "Add an example script that shows how to plot the plasma boundary geometry for addressing #263 . (#283)",
"timestamp": "2025-04-28T11:56:35Z",
"tree_id": "86e02a6549b41c54e909663bf091a32e390a473e",
"url": "https://github.com/proximafusion/vmecpp/commit/7e63d9e5d5331d1147c5b5008f8e5786d42e81d4"
},
"date": 1783589338114,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.314857082799972,
"range": "stddev: 0.00130531113876103",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3135300171999916,
"range": "stddev: 0.002387706550011013",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4214644429999528,
"range": "stddev: 0.0061851252532474695",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7772777626666993,
"range": "stddev: 0.04318300595215523",
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
"id": "ebf4b075f2f2db11ee65b515b852324111f92a89",
"message": "Adjust tolerance for jdotb. (#284)",
"timestamp": "2025-04-28T13:50:59Z",
"tree_id": "69178e3eeb1b079fe0ecefae8d467edd61c2d65b",
"url": "https://github.com/proximafusion/vmecpp/commit/ebf4b075f2f2db11ee65b515b852324111f92a89"
},
"date": 1783589338207,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31972274479994667,
"range": "stddev: 0.0016141073136555798",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3143801409999469,
"range": "stddev: 0.0032422274288954446",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4071506296666787,
"range": "stddev: 0.0054871903817888985",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7720275893333715,
"range": "stddev: 0.01988689173050271",
"extra": "rounds: 3"
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
"date": 1783589338125,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31830912980012727,
"range": "stddev: 0.0022679120305743325",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.32123651119991337,
"range": "stddev: 0.0032214401643616834",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4216241186665382,
"range": "stddev: 0.006085055722378528",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8109981376666533,
"range": "stddev: 0.016508884620663154",
"extra": "rounds: 3"
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
"date": 1783589338151,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3274745960000473,
"range": "stddev: 0.0024704374155281703",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3178640357999939,
"range": "stddev: 0.0031381251793049735",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.425447130000066,
"range": "stddev: 0.0011350066719054176",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8040298103333043,
"range": "stddev: 0.02018026378074974",
"extra": "rounds: 3"
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
"date": 1783589338109,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3172199520000504,
"range": "stddev: 0.003628672072576775",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3241607575999296,
"range": "stddev: 0.0047113248935812145",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4329593299999033,
"range": "stddev: 0.020210650257341194",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.809520690666583,
"range": "stddev: 0.015413589974690172",
"extra": "rounds: 3"
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
"date": 1783589338098,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6043303412000569,
"range": "stddev: 0.0112447647278991",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5897809180000877,
"range": "stddev: 0.010377674588400818",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4160840780000399,
"range": "stddev: 0.004116933340526411",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.839425157666483,
"range": "stddev: 0.06135823600137774",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589337989,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5939464037999642,
"range": "stddev: 0.01737159443264963",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6136203468000531,
"range": "stddev: 0.0191334195327059",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4282398269998946,
"range": "stddev: 0.010803345864889046",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.792474323333105,
"range": "stddev: 0.02555316735573695",
"extra": "rounds: 3"
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
"date": 1783589338146,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6057931442000154,
"range": "stddev: 0.0053946089373611466",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6218443251999816,
"range": "stddev: 0.010819411889033615",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4197432739999083,
"range": "stddev: 0.0033117416845851785",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.799993938666679,
"range": "stddev: 0.013660988426277769",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.555066268666754,
"range": "stddev: 0.012891324963186493",
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
"id": "1763e4337598df0372cbc651375b2bc278c0d8ee",
"message": "Build NetCDF4 from source as well. (#288)",
"timestamp": "2025-04-28T18:55:08Z",
"tree_id": "c0297c4270e7bfaa66036215112ae33a6f508b24",
"url": "https://github.com/proximafusion/vmecpp/commit/1763e4337598df0372cbc651375b2bc278c0d8ee"
},
"date": 1783589338014,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6001449833999686,
"range": "stddev: 0.006563135469902521",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5880505796000761,
"range": "stddev: 0.005915672667193014",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4169142013335356,
"range": "stddev: 0.003773853722265851",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8078762819999006,
"range": "stddev: 0.024601058666097735",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.552368905333272,
"range": "stddev: 0.010002820807719002",
"extra": "rounds: 3"
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
"date": 1783589338214,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.593850991599993,
"range": "stddev: 0.007010098372668359",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5925408777999109,
"range": "stddev: 0.006643245948423371",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4058997240000888,
"range": "stddev: 0.0006194429945188875",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7845157733333203,
"range": "stddev: 0.0258505744109185",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5480481463335611,
"range": "stddev: 0.01190949877101766",
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
"date": 1783589338077,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6387748704000046,
"range": "stddev: 0.014192196568294324",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.623669515399979,
"range": "stddev: 0.0205773228291538",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4157255099999968,
"range": "stddev: 0.0015610655899763947",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8155114760000024,
"range": "stddev: 0.018398923634540727",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.58261169266666,
"range": "stddev: 0.03104535102135367",
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
"id": "acb74ee9e43071f29c7d24966ea52f80664d624a",
"message": "Add chipH to OutputQuantities. (#89)",
"timestamp": "2025-04-29T08:37:38Z",
"tree_id": "27f42beaedb46746f305be8e93ebf69577b605eb",
"url": "https://github.com/proximafusion/vmecpp/commit/acb74ee9e43071f29c7d24966ea52f80664d624a"
},
"date": 1783589338148,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6167908854000075,
"range": "stddev: 0.01597252828625898",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6521034755999949,
"range": "stddev: 0.017989691887013208",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4257062639999845,
"range": "stddev: 0.006888881416919802",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.747624119333333,
"range": "stddev: 0.025342274969776958",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5500817320000049,
"range": "stddev: 0.008266935348651359",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338202,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5819898899999998,
"range": "stddev: 0.006270436653545086",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5854582356000038,
"range": "stddev: 0.00802048763420328",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4269324890000046,
"range": "stddev: 0.007800687493928258",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7967078683333475,
"range": "stddev: 0.030258464747559045",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5559505919999879,
"range": "stddev: 0.010964465434355326",
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
"id": "fe1e4912489781c5abfe294c49cb02a46f7bdfcc",
"message": "Migrate some wout docs from repo/13962 (#290)",
"timestamp": "2025-04-29T11:12:07Z",
"tree_id": "fadb1d96a9034afe64beb78dcaf43c61d1a2dba2",
"url": "https://github.com/proximafusion/vmecpp/commit/fe1e4912489781c5abfe294c49cb02a46f7bdfcc"
},
"date": 1783589338221,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5825482799999918,
"range": "stddev: 0.012603654430814829",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5744715560000031,
"range": "stddev: 0.007419843990978412",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.42296949366668,
"range": "stddev: 0.004687319511711772",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.76220582066666,
"range": "stddev: 0.019330975015810433",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5556849023333068,
"range": "stddev: 0.016608723109621307",
"extra": "rounds: 3"
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
"date": 1783589338196,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6105973424000013,
"range": "stddev: 0.012275792311527046",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5932469692000268,
"range": "stddev: 0.012286798319573208",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4206849140000106,
"range": "stddev: 0.011163770727559497",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.787582438666694,
"range": "stddev: 0.018434981669663033",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5535392753333401,
"range": "stddev: 0.011943808753604852",
"extra": "rounds: 3"
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
"date": 1783589338007,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5925213418000339,
"range": "stddev: 0.006343129387576247",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5925857906000147,
"range": "stddev: 0.01448248331402347",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4169600210000226,
"range": "stddev: 0.005192023954300058",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8042009620000194,
"range": "stddev: 0.04006132854364045",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5603403556666535,
"range": "stddev: 0.006971667982997855",
"extra": "rounds: 3"
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
"date": 1783589338054,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6168599660000155,
"range": "stddev: 0.00646083261013123",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6193552984000007,
"range": "stddev: 0.0036449661750852114",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4221905566666162,
"range": "stddev: 0.004671092841413254",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.775744596000019,
"range": "stddev: 0.0054831591436919624",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.556249251333308,
"range": "stddev: 0.013040560064326015",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338153,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5897120233999885,
"range": "stddev: 0.008720318082581934",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5839180157999635,
"range": "stddev: 0.009388372553963666",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4264919063333537,
"range": "stddev: 0.01679507689778383",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7731063173333346,
"range": "stddev: 0.0301397634335558",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5538906123333238,
"range": "stddev: 0.011508160153329658",
"extra": "rounds: 3"
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
"date": 1783589338000,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5921168182000202,
"range": "stddev: 0.008524082605411626",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5941258554000342,
"range": "stddev: 0.02359815973629605",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.420421044999974,
"range": "stddev: 0.001867480065479563",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7658265643333757,
"range": "stddev: 0.010857235172003144",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5563150370000283,
"range": "stddev: 0.009435341463403097",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338045,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6035458611999729,
"range": "stddev: 0.010760877365939947",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5983025842000188,
"range": "stddev: 0.014859298562229074",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4293662146666672,
"range": "stddev: 0.007811861694218162",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.775495556333302,
"range": "stddev: 0.014366236649690103",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5531005403333893,
"range": "stddev: 0.00992474612073622",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338005,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6116560178000328,
"range": "stddev: 0.014622724848170775",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6058452576000946,
"range": "stddev: 0.009620237832072146",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4158142213333729,
"range": "stddev: 0.002105967630708285",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.788327764999925,
"range": "stddev: 0.009609978024881331",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5543761030000194,
"range": "stddev: 0.008266786949311536",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338093,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6109824783999557,
"range": "stddev: 0.012966588342953535",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5846393053999691,
"range": "stddev: 0.007382057574235302",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4242826569999731,
"range": "stddev: 0.012371749615996635",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.767777306000047,
"range": "stddev: 0.017876767916645726",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.60630954766672,
"range": "stddev: 0.022066795428560725",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5570195540000593,
"range": "stddev: 0.009509542498038143",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338056,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5786229840000032,
"range": "stddev: 0.010890852699242523",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5745749384000192,
"range": "stddev: 0.014474949750349622",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4134117116666403,
"range": "stddev: 0.001917673640447255",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7751463359999584,
"range": "stddev: 0.019218846349634743",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.613464168333318,
"range": "stddev: 0.03761302184513803",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.553965964666683,
"range": "stddev: 0.014727758022618843",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338100,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3383303619999879,
"range": "stddev: 0.005292062916626205",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3279743025999323,
"range": "stddev: 0.0030385625693145663",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4168462866666687,
"range": "stddev: 0.0004578342382818995",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7572903940000892,
"range": "stddev: 0.00434425632374993",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.618414991000085,
"range": "stddev: 0.006660496078628539",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5585452310000012,
"range": "stddev: 0.01828849013890208",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338040,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3358315229999789,
"range": "stddev: 0.009717394175345477",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33302043359999517,
"range": "stddev: 0.006391573241142689",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4167015696666567,
"range": "stddev: 0.0014086952320577185",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.784629928666694,
"range": "stddev: 0.02207864026502808",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.602862601333376,
"range": "stddev: 0.026522839013145626",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5557916283333573,
"range": "stddev: 0.01624626631119469",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338128,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3477195622000181,
"range": "stddev: 0.002493428349568313",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3504288173999612,
"range": "stddev: 0.003924621335234931",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4483814660000764,
"range": "stddev: 0.04705084274152694",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8678907256666357,
"range": "stddev: 0.10645440220615913",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.675972962666643,
"range": "stddev: 0.022739814801775615",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.556196777000044,
"range": "stddev: 0.011108810932324546",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338042,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36281408879999616,
"range": "stddev: 0.005631502067982624",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3637063217999639,
"range": "stddev: 0.0033157140369235306",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4214297796666717,
"range": "stddev: 0.001361509068046907",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8275995960001032,
"range": "stddev: 0.045502941782567",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.563882819666711,
"range": "stddev: 0.030033777067694903",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5598288570000325,
"range": "stddev: 0.011801228683748928",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338079,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33907174439996196,
"range": "stddev: 0.003068502168881712",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3441904744000112,
"range": "stddev: 0.0035624064792657466",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4176079923333873,
"range": "stddev: 0.0015337462428516612",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.791234864333319,
"range": "stddev: 0.040538080459767784",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.585671173666697,
"range": "stddev: 0.006732217021747009",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5518441563333454,
"range": "stddev: 0.0125218787136634",
"extra": "rounds: 3"
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
"date": 1783589338139,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35756976499997106,
"range": "stddev: 0.00426332343730856",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35689051939998534,
"range": "stddev: 0.0042019650189996415",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4226922289999493,
"range": "stddev: 0.004736128782461913",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7957238170000287,
"range": "stddev: 0.003205362444229155",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.573202756333407,
"range": "stddev: 0.01239806070292339",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5635717146666746,
"range": "stddev: 0.02110152709971802",
"extra": "rounds: 3"
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
"date": 1783589337984,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3268821924000804,
"range": "stddev: 0.001592755828128443",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33318193719996997,
"range": "stddev: 0.0021260960926095546",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4080617463334117,
"range": "stddev: 0.0016264435502186857",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.75583679533338,
"range": "stddev: 0.02754586276010434",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.519523307666683,
"range": "stddev: 0.01510359429541",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5463653029998266,
"range": "stddev: 0.010700028645504287",
"extra": "rounds: 3"
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
"date": 1783589338037,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34678702700011854,
"range": "stddev: 0.015150274149097186",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3392419242000869,
"range": "stddev: 0.0039772573824595585",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4194442366667015,
"range": "stddev: 0.002174217562352858",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7722403310000723,
"range": "stddev: 0.005509175768481301",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.485293617000025,
"range": "stddev: 0.06984545322700478",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5532433049997962,
"range": "stddev: 0.019290045594749245",
"extra": "rounds: 3"
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
"date": 1783589338107,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3532757677999143,
"range": "stddev: 0.01600675443784224",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3386349244000485,
"range": "stddev: 0.0018212339913726632",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4147373353334842,
"range": "stddev: 0.003826423247850695",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7893756936665945,
"range": "stddev: 0.0358076667313918",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.591270436999972,
"range": "stddev: 0.011973865954237177",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5523389349999281,
"range": "stddev: 0.01043346045987082",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
"email": "166746189+jurasic-pf@users.noreply.github.com",
"username": ""
},
"committer": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338017,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3429656704000081,
"range": "stddev: 0.0013569101921268758",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3423950975999105,
"range": "stddev: 0.002631874749806958",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4169255200001014,
"range": "stddev: 0.0044275531822004464",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.776547807666551,
"range": "stddev: 0.015093457440075751",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.665967802000068,
"range": "stddev: 0.03377067174489951",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5571696183331671,
"range": "stddev: 0.012164378102547887",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338137,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.364085470200007,
"range": "stddev: 0.0017352021911934015",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3680460200000198,
"range": "stddev: 0.003557566925823504",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4215749246667049,
"range": "stddev: 0.0082329669330634",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.819504626333431,
"range": "stddev: 0.03237504886023343",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.618992233666631,
"range": "stddev: 0.011854954025662688",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5549991716667744,
"range": "stddev: 0.012920632039170363",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338026,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33656180599991786,
"range": "stddev: 0.0037916693557854173",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.351204370799951,
"range": "stddev: 0.0015207137790642135",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4112568719998915,
"range": "stddev: 0.0012077470449727084",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7718531243332714,
"range": "stddev: 0.016849217037873284",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.63814516066653,
"range": "stddev: 0.009076247168831274",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5533612486665334,
"range": "stddev: 0.008330179485901864",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338171,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34585505639997793,
"range": "stddev: 0.0038372310408277314",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3465442090000579,
"range": "stddev: 0.0037698944138313496",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4162715226667995,
"range": "stddev: 0.0029703717509771653",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.777059641666559,
"range": "stddev: 0.003051735444270479",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.552333451333348,
"range": "stddev: 0.042442476558363675",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5499737476666269,
"range": "stddev: 0.016986727791716796",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338068,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3357732485999804,
"range": "stddev: 0.009907338993089864",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34471709680010465,
"range": "stddev: 0.007152127629615611",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4114566229998975,
"range": "stddev: 0.009534649789597508",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7456231319999156,
"range": "stddev: 0.013707765443564043",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.555114883333317,
"range": "stddev: 0.029200749247858558",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.551900076666546,
"range": "stddev: 0.011761485988183794",
"extra": "rounds: 3"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338194,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3392763127998478,
"range": "stddev: 0.0030169431156373038",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3368834373998652,
"range": "stddev: 0.0018468604393917243",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.411417907333392,
"range": "stddev: 0.001878723501796193",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.756468059000023,
"range": "stddev: 0.009829341657770675",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.529424944999997,
"range": "stddev: 0.014511093466960804",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5475355863335,
"range": "stddev: 0.007754737480068744",
"extra": "rounds: 3"
}
]
}
],
"C++ Microbenchmarks": [
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338246,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0003160501742473526,
"extra": "iterations: 647\ncpu: 0.0003156380664605874 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002022132846071533,
"extra": "iterations: 1389\ncpu: 0.00020221567458603314 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026282778492680305,
"extra": "iterations: 108\ncpu: 0.00262831363888889 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017787705427445707,
"extra": "iterations: 159\ncpu: 0.0017787966792452829 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00582846999168396,
"extra": "iterations: 48\ncpu: 0.005828562687500001 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00387058192736482,
"extra": "iterations: 73\ncpu: 0.0038703721780821952 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026135577095879447,
"extra": "iterations: 108\ncpu: 0.002613598037037037 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017711422111414657,
"extra": "iterations: 158\ncpu: 0.0017711108797468364 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005237102508544922,
"extra": "iterations: 54\ncpu: 0.005236703611111106 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008185945238385882,
"extra": "iterations: 35\ncpu: 0.008186054371428561 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011056556701660157,
"extra": "iterations: 25\ncpu: 0.011056152319999983 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002576230862818727,
"extra": "iterations: 109\ncpu: 0.0025761731467889896 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017630814006493528,
"extra": "iterations: 159\ncpu: 0.001762966842767296 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003655529641485834,
"extra": "iterations: 77\ncpu: 0.0036555844935064904 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028435894937226267,
"extra": "iterations: 99\ncpu: 0.0028436358686868683 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004521116133659117,
"extra": "iterations: 62\ncpu: 0.004520968467741945 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036034431212987653,
"extra": "iterations: 78\ncpu: 0.003603509089743592 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0057253291209538775,
"extra": "iterations: 48\ncpu: 0.005725416729166673 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004821875999713766,
"extra": "iterations: 58\ncpu: 0.004821497017241389 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010939453964802757,
"extra": "iterations: 536\ncpu: 0.0005168236697761198 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.002008401155471802,
"extra": "iterations: 400\ncpu: 0.0008161281850000002 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338293,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0003295040130615235,
"extra": "iterations: 625\ncpu: 0.0003274312352000001 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020388537558956426,
"extra": "iterations: 1380\ncpu: 0.0002038884739130435 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025943628063908332,
"extra": "iterations: 108\ncpu: 0.0025942769907407407 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017705054222782956,
"extra": "iterations: 158\ncpu: 0.0017705325189873423 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005833372473716736,
"extra": "iterations: 48\ncpu: 0.0058329963750000016 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0038290958146791204,
"extra": "iterations: 74\ncpu: 0.003828265391891889 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025862800740750038,
"extra": "iterations: 107\ncpu: 0.002586174000000001 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017670580426102165,
"extra": "iterations: 159\ncpu: 0.0017670169119496871 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005248177726313754,
"extra": "iterations: 53\ncpu: 0.005247621339622644 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008026879174368723,
"extra": "iterations: 35\ncpu: 0.008026605228571437 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011522951126098633,
"extra": "iterations: 25\ncpu: 0.011520673240000008 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026538584491994123,
"extra": "iterations: 101\ncpu: 0.0026537874554455397 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018111046622781193,
"extra": "iterations: 136\ncpu: 0.0018107954191176482 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003651182372848709,
"extra": "iterations: 77\ncpu: 0.003650903610389602 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002844268625432795,
"extra": "iterations: 99\ncpu: 0.002844312585858586 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004694113965894356,
"extra": "iterations: 61\ncpu: 0.004693413639344263 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036955220358712336,
"extra": "iterations: 77\ncpu: 0.00369495536363636 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.006095652884625375,
"extra": "iterations: 47\ncpu: 0.0060957440425532 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004834725938994309,
"extra": "iterations: 58\ncpu: 0.004834630655172419 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010436304553064282,
"extra": "iterations: 501\ncpu: 0.0005011647524950097 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015340667021901985,
"extra": "iterations: 380\ncpu: 0.0007086615684210523 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338229,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021918913195017545,
"extra": "iterations: 1278\ncpu: 0.00021912433724569641 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002016714665905414,
"extra": "iterations: 1369\ncpu: 0.00020163599634769906 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025727005179868928,
"extra": "iterations: 109\ncpu: 0.0025719998807339447 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017769351695318641,
"extra": "iterations: 159\ncpu: 0.0017769579622641514 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005821451544761658,
"extra": "iterations: 48\ncpu: 0.005821525791666671 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037688371297475457,
"extra": "iterations: 74\ncpu: 0.003768894054054051 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002595997731620019,
"extra": "iterations: 109\ncpu: 0.0025958477522935765 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017721939690505401,
"extra": "iterations: 158\ncpu: 0.0017719910189873407 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0052371789824287845,
"extra": "iterations: 53\ncpu: 0.005237259094339617 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008246019908360073,
"extra": "iterations: 35\ncpu: 0.00824452882857143 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010801883844228892,
"extra": "iterations: 26\ncpu: 0.010802044115384607 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025834556019634284,
"extra": "iterations: 109\ncpu: 0.0025830386238532106 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017506331205368044,
"extra": "iterations: 160\ncpu: 0.0017502897187499978 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036834927348347456,
"extra": "iterations: 77\ncpu: 0.00368355355844156 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002884792559074633,
"extra": "iterations: 99\ncpu: 0.0028847368787878816 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004518124365037488,
"extra": "iterations: 62\ncpu: 0.004517785967741926 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003615202047885993,
"extra": "iterations: 78\ncpu: 0.0036151332179487103 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005791167823635802,
"extra": "iterations: 49\ncpu: 0.005790950877551013 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00489815912748638,
"extra": "iterations: 57\ncpu: 0.00489799308771929 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010830929101832676,
"extra": "iterations: 497\ncpu: 0.0005046543480885323 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015572422179416638,
"extra": "iterations: 402\ncpu: 0.0007343331741293527 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338303,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002174105231217512,
"extra": "iterations: 1270\ncpu: 0.00021739398818897636 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020129603724325856,
"extra": "iterations: 1395\ncpu: 0.00020128681003584226 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025841599210686644,
"extra": "iterations: 109\ncpu: 0.0025838957064220187 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017597277959187825,
"extra": "iterations: 150\ncpu: 0.0017594701133333325 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005897000432014465,
"extra": "iterations: 48\ncpu: 0.005897085708333334 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0038224200944642767,
"extra": "iterations: 74\ncpu: 0.003822342391891888 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002595213224303048,
"extra": "iterations: 106\ncpu: 0.0025950127924528324 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017795259026205463,
"extra": "iterations: 157\ncpu: 0.0017793189044586005 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005248299184835182,
"extra": "iterations: 53\ncpu: 0.005247552679245286 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00804126603262765,
"extra": "iterations: 35\ncpu: 0.00804108391428572 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010839737378633939,
"extra": "iterations: 26\ncpu: 0.010839536384615399 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025822895544546625,
"extra": "iterations: 108\ncpu: 0.002582330629629636 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017592532079924578,
"extra": "iterations: 159\ncpu: 0.0017590210000000008 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003643571556388558,
"extra": "iterations: 77\ncpu: 0.0036430769220779273 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002840947131721341,
"extra": "iterations: 98\ncpu: 0.002840994316326525 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004604801054923766,
"extra": "iterations: 62\ncpu: 0.0046045404838709735 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035997211159049694,
"extra": "iterations: 77\ncpu: 0.0035994714415584442 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005843162536621094,
"extra": "iterations: 49\ncpu: 0.005842938469387774 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004908105783295214,
"extra": "iterations: 57\ncpu: 0.004907486456140351 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010354624751912121,
"extra": "iterations: 518\ncpu: 0.0005024241660231656 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016553597475485828,
"extra": "iterations: 378\ncpu: 0.0007711406507936512 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338263,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021807426802391403,
"extra": "iterations: 1287\ncpu: 0.00021807759751359753 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020319771492618253,
"extra": "iterations: 1392\ncpu: 0.0002031848412356322 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026045001555826063,
"extra": "iterations: 107\ncpu: 0.00260444585046729 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017725878123995625,
"extra": "iterations: 158\ncpu: 0.001772618246835444 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005901909889058864,
"extra": "iterations: 47\ncpu: 0.0059016485957446825 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037908425202240817,
"extra": "iterations: 74\ncpu: 0.0037909110675675675 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026184264744553614,
"extra": "iterations: 107\ncpu: 0.0026184613364485993 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017820118339198412,
"extra": "iterations: 157\ncpu: 0.001782035955414013 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005284417350337191,
"extra": "iterations: 53\ncpu: 0.005284157905660379 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008173353531781365,
"extra": "iterations: 34\ncpu: 0.008172803882352937 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010916361441979041,
"extra": "iterations: 26\ncpu: 0.010916532269230771 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002612946210084138,
"extra": "iterations: 108\ncpu: 0.0026128653703703757 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017602953520960778,
"extra": "iterations: 159\ncpu: 0.0017601772327043986 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036718195134943185,
"extra": "iterations: 77\ncpu: 0.003671749974025982 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002849210392345082,
"extra": "iterations: 99\ncpu: 0.0028491078888888896 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004520283370721536,
"extra": "iterations: 61\ncpu: 0.004520346262295088 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036155713068974486,
"extra": "iterations: 77\ncpu: 0.0036156228181818176 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00578555961449941,
"extra": "iterations: 48\ncpu: 0.005784373812499985 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004914831696895132,
"extra": "iterations: 57\ncpu: 0.004914905701754369 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001075602802834492,
"extra": "iterations: 506\ncpu: 0.000503344656126481 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015820747564646823,
"extra": "iterations: 363\ncpu: 0.0007179287410468316 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338269,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021983134614724496,
"extra": "iterations: 1266\ncpu: 0.00021982610979462877 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002039913506200571,
"extra": "iterations: 1381\ncpu: 0.00020399437871107903 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025875546314098217,
"extra": "iterations: 108\ncpu: 0.002587592203703705 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017761819681544214,
"extra": "iterations: 157\ncpu: 0.0017761184140127398 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005826403697331746,
"extra": "iterations: 48\ncpu: 0.005826498208333336 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037807000649941937,
"extra": "iterations: 74\ncpu: 0.0037807591621621577 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025998309806541165,
"extra": "iterations: 108\ncpu: 0.0025996598055555557 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001784486648363945,
"extra": "iterations: 156\ncpu: 0.0017845089743589748 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00527101642680618,
"extra": "iterations: 53\ncpu: 0.00527109145283019 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008584356307983399,
"extra": "iterations: 35\ncpu: 0.00858423654285715 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011055345535278321,
"extra": "iterations: 25\ncpu: 0.011055488680000015 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025788876745435926,
"extra": "iterations: 108\ncpu: 0.0025787108333333326 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017679839194575445,
"extra": "iterations: 158\ncpu: 0.0017680067974683547 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036931571207548444,
"extra": "iterations: 76\ncpu: 0.003692967342105266 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028458444439634985,
"extra": "iterations: 98\ncpu: 0.0028458888673469377 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004496197546682051,
"extra": "iterations: 62\ncpu: 0.004496263193548387 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036374253112000307,
"extra": "iterations: 77\ncpu: 0.003637068909090912 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005800242009370224,
"extra": "iterations: 46\ncpu: 0.005800325478260863 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004905934991507695,
"extra": "iterations: 58\ncpu: 0.004905690775862074 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010862580385348083,
"extra": "iterations: 477\ncpu: 0.0005108979811320747 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016803938023587492,
"extra": "iterations: 376\ncpu: 0.0007674864015957429 seconds\nthreads: 1"
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
"date": 1783589338251,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002195244853936353,
"extra": "iterations: 1236\ncpu: 0.00021949859142394822 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020144249137539092,
"extra": "iterations: 1394\ncpu: 0.0002014454705882353 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025929035963835542,
"extra": "iterations: 108\ncpu: 0.0025928579537037033 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001773129602906051,
"extra": "iterations: 157\ncpu: 0.0017731551210191077 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00582545002301534,
"extra": "iterations: 48\ncpu: 0.005825548937500004 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037839702657751135,
"extra": "iterations: 74\ncpu: 0.003783477067567566 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025973698803197563,
"extra": "iterations: 107\ncpu: 0.002597329299065424 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017955591724177077,
"extra": "iterations: 157\ncpu: 0.0017955009426751597 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005274444256188735,
"extra": "iterations: 53\ncpu: 0.005273826150943392 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008067628315516882,
"extra": "iterations: 35\ncpu: 0.008067732200000005 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010947034909174992,
"extra": "iterations: 26\ncpu: 0.010945670500000018 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002697940184691242,
"extra": "iterations: 107\ncpu: 0.0026978502616822404 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001766269720053371,
"extra": "iterations: 158\ncpu: 0.0017662052721519027 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003684926677394558,
"extra": "iterations: 74\ncpu: 0.003684366662162173 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002840732790760158,
"extra": "iterations: 97\ncpu: 0.0028407763402061867 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004553783324456984,
"extra": "iterations: 62\ncpu: 0.004553323661290329 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0037089554887068904,
"extra": "iterations: 76\ncpu: 0.003708180842105266 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005829100913189828,
"extra": "iterations: 47\ncpu: 0.005829190234042557 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004910991109650711,
"extra": "iterations: 58\ncpu: 0.004910435844827579 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010995333508927692,
"extra": "iterations: 516\ncpu: 0.0005112829321705437 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016422836246058083,
"extra": "iterations: 397\ncpu: 0.0007624764181360204 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338281,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002190459306073037,
"extra": "iterations: 1256\ncpu: 0.00021904185191082805 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002016429433230886,
"extra": "iterations: 1386\ncpu: 0.00020162102092352095 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002589838220438826,
"extra": "iterations: 109\ncpu: 0.002589464944954128 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001789501518200917,
"extra": "iterations: 157\ncpu: 0.0017894693757961777 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005831196904182434,
"extra": "iterations: 48\ncpu: 0.005830749124999998 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037918972642454387,
"extra": "iterations: 73\ncpu: 0.0037919456027397237 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026029738310341523,
"extra": "iterations: 107\ncpu: 0.0026028525046729 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017811705352394445,
"extra": "iterations: 157\ncpu: 0.0017810113885350295 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005326986312866211,
"extra": "iterations: 52\ncpu: 0.005326901096153841 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008018950053623745,
"extra": "iterations: 35\ncpu: 0.008017265942857144 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010831080950223483,
"extra": "iterations: 26\ncpu: 0.010831228153846144 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025830028253957765,
"extra": "iterations: 109\ncpu: 0.002582829513761473 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017626587348648266,
"extra": "iterations: 158\ncpu: 0.0017625906835443044 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003697497049967448,
"extra": "iterations: 75\ncpu: 0.00369737954666667 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002883463790736248,
"extra": "iterations: 97\ncpu: 0.0028835042886597964 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004522596636126119,
"extra": "iterations: 62\ncpu: 0.004522663241935488 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036411957863049633,
"extra": "iterations: 78\ncpu: 0.003641099192307695 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005737359325091044,
"extra": "iterations: 48\ncpu: 0.005736850083333337 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004955690482567096,
"extra": "iterations: 58\ncpu: 0.004955498862068975 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010439774085735454,
"extra": "iterations: 493\ncpu: 0.0004999784381338741 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016693185115682668,
"extra": "iterations: 348\ncpu: 0.0007883652988505742 seconds\nthreads: 1"
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
"date": 1783589338241,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.000222591715534841,
"extra": "iterations: 1270\ncpu: 0.0002225469283464567 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.000201576309697736,
"extra": "iterations: 1391\ncpu: 0.00020157894751977 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025868150922987196,
"extra": "iterations: 108\ncpu: 0.002586850546296296 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017685120618796046,
"extra": "iterations: 158\ncpu: 0.001768537411392406 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005761062105496724,
"extra": "iterations: 48\ncpu: 0.005761145270833333 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037602583567301435,
"extra": "iterations: 75\ncpu: 0.003760064733333331 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002597557173834907,
"extra": "iterations: 108\ncpu: 0.002597293657407405 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017776881592183175,
"extra": "iterations: 158\ncpu: 0.001777717379746835 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00520014762878418,
"extra": "iterations: 54\ncpu: 0.005199945425925921 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007965489796229772,
"extra": "iterations: 35\ncpu: 0.007965160628571423 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010799041161170373,
"extra": "iterations: 26\ncpu: 0.010798499461538472 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025792690592074613,
"extra": "iterations: 109\ncpu: 0.0025793041284403696 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017556577920913698,
"extra": "iterations: 160\ncpu: 0.001755672493749999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036738420787610507,
"extra": "iterations: 76\ncpu: 0.003673672118421062 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028359191586272885,
"extra": "iterations: 99\ncpu: 0.002835956898989899 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00451440195883474,
"extra": "iterations: 62\ncpu: 0.004514163048387087 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003599519971050794,
"extra": "iterations: 79\ncpu: 0.0035995688481012633 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005734876710541395,
"extra": "iterations: 49\ncpu: 0.00573494600000001 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00485058488516972,
"extra": "iterations: 58\ncpu: 0.004850419086206901 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001098676161332564,
"extra": "iterations: 528\ncpu: 0.0005141372367424241 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016565120618570127,
"extra": "iterations: 389\ncpu: 0.000767847357326478 seconds\nthreads: 1"
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
"date": 1783589338325,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002163868944791821,
"extra": "iterations: 1266\ncpu: 0.00021638013981042654 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020135381798562778,
"extra": "iterations: 1393\ncpu: 0.00020135656066044515 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025780572803742294,
"extra": "iterations: 109\ncpu: 0.002577840724770642 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017635807301263392,
"extra": "iterations: 159\ncpu: 0.0017636079748427674 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005772794286410014,
"extra": "iterations: 48\ncpu: 0.0057724311041666655 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037627091278900973,
"extra": "iterations: 74\ncpu: 0.0037626362972972946 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025834374957614476,
"extra": "iterations: 108\ncpu: 0.0025834736018518533 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017634856549999382,
"extra": "iterations: 158\ncpu: 0.0017634169683544304 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005230165877432194,
"extra": "iterations: 53\ncpu: 0.005230236566037728 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007970142364501954,
"extra": "iterations: 35\ncpu: 0.00796985794285714 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010622116235586314,
"extra": "iterations: 26\ncpu: 0.010621972076923085 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025815591899626847,
"extra": "iterations: 109\ncpu: 0.0025815989633027558 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017468973994255066,
"extra": "iterations: 160\ncpu: 0.0017468560500000009 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003648603117311156,
"extra": "iterations: 77\ncpu: 0.0036486553376623393 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028330087661743164,
"extra": "iterations: 98\ncpu: 0.0028330446938775503 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004494993917403683,
"extra": "iterations: 62\ncpu: 0.00449461527419355 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035951482622246997,
"extra": "iterations: 76\ncpu: 0.0035950208289473773 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005729928612709045,
"extra": "iterations: 48\ncpu: 0.00572930443749999 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0049933550650613345,
"extra": "iterations: 57\ncpu: 0.004993429350877197 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001102748824043508,
"extra": "iterations: 489\ncpu: 0.000521697118609407 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016517221769540317,
"extra": "iterations: 377\ncpu: 0.0007604397877984082 seconds\nthreads: 1"
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
"date": 1783589338289,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021807813681287537,
"extra": "iterations: 1293\ncpu: 0.00021808219644238209 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002009725918735031,
"extra": "iterations: 1370\ncpu: 0.00020077137591240882 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025854199020950886,
"extra": "iterations: 108\ncpu: 0.0025850004351851844 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017710486544838435,
"extra": "iterations: 158\ncpu: 0.0017710712531645578 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005776273960969886,
"extra": "iterations: 49\ncpu: 0.005776036285714282 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037261486053466798,
"extra": "iterations: 75\ncpu: 0.0037262006400000007 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025751700095080457,
"extra": "iterations: 109\ncpu: 0.0025752102568807342 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017644864208293412,
"extra": "iterations: 159\ncpu: 0.0017642654905660386 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005169444613986545,
"extra": "iterations: 54\ncpu: 0.005169089351851856 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007855965031517876,
"extra": "iterations: 36\ncpu: 0.007855616472222228 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010680913925170898,
"extra": "iterations: 26\ncpu: 0.01068112353846152 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002576002368220577,
"extra": "iterations: 108\ncpu: 0.0025759474351851885 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017656260316476883,
"extra": "iterations: 159\ncpu: 0.001765307679245281 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036372395304890425,
"extra": "iterations: 77\ncpu: 0.003637287766233769 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028170204162597657,
"extra": "iterations: 100\ncpu: 0.002816989450000005 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00445035147288489,
"extra": "iterations: 63\ncpu: 0.004449844603174609 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036271535433255713,
"extra": "iterations: 78\ncpu: 0.0036270440128205175 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0056587481985286795,
"extra": "iterations: 49\ncpu: 0.0056584935918367225 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004847392701266105,
"extra": "iterations: 57\ncpu: 0.004847316298245617 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001201262651396192,
"extra": "iterations: 484\ncpu: 0.000532476311983472 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.002054248322447768,
"extra": "iterations: 317\ncpu: 0.0008581169432176643 seconds\nthreads: 1"
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
"date": 1783589338316,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022226101726341557,
"extra": "iterations: 1242\ncpu: 0.000222255116747182 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020041301803807476,
"extra": "iterations: 1396\ncpu: 0.00020041556303724932 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025647922798439308,
"extra": "iterations: 108\ncpu: 0.002564827240740741 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017637135107305987,
"extra": "iterations: 158\ncpu: 0.001763623379746835 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005753901540016642,
"extra": "iterations: 49\ncpu: 0.005753825306122452 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004124460993586361,
"extra": "iterations: 74\ncpu: 0.004124199918918917 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002675365518640589,
"extra": "iterations: 108\ncpu: 0.0026753274351851877 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018158593735137544,
"extra": "iterations: 154\ncpu: 0.001815787720779221 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005237252623946579,
"extra": "iterations: 54\ncpu: 0.005237322166666671 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007886654990059989,
"extra": "iterations: 35\ncpu: 0.007886019257142848 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010630192580046478,
"extra": "iterations: 27\ncpu: 0.010628266407407426 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025814002919419903,
"extra": "iterations: 107\ncpu: 0.0025811578224299086 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001773396874688993,
"extra": "iterations: 157\ncpu: 0.001773145439490443 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003662301348401355,
"extra": "iterations: 77\ncpu: 0.003661979000000004 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002878651327016402,
"extra": "iterations: 98\ncpu: 0.002878697612244905 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004469508216494606,
"extra": "iterations: 63\ncpu: 0.0044692123333333325 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003605207839569488,
"extra": "iterations: 77\ncpu: 0.0036047449740259727 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005676784515380859,
"extra": "iterations: 50\ncpu: 0.005676672939999997 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00487001188870134,
"extra": "iterations: 58\ncpu: 0.004869886775862064 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010792397339520288,
"extra": "iterations: 514\ncpu: 0.0005278100505836581 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015250767848288367,
"extra": "iterations: 387\ncpu: 0.000707459860465117 seconds\nthreads: 1"
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
"date": 1783589338284,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021910198460780761,
"extra": "iterations: 1271\ncpu: 0.00021908331549960663 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020379764631466102,
"extra": "iterations: 1381\ncpu: 0.00020379489065894282 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025889586519311975,
"extra": "iterations: 108\ncpu: 0.002588999462962964 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017799003214775762,
"extra": "iterations: 158\ncpu: 0.001779926879746835 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005778864026069641,
"extra": "iterations: 48\ncpu: 0.005778481875000001 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037539577484130863,
"extra": "iterations: 75\ncpu: 0.003753778880000001 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002589382507182934,
"extra": "iterations: 108\ncpu: 0.0025891097685185193 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.00176973131638539,
"extra": "iterations: 158\ncpu: 0.0017697036329113929 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005219106320981627,
"extra": "iterations: 54\ncpu: 0.005218989851851853 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007889842987060547,
"extra": "iterations: 35\ncpu: 0.007889950628571427 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011058779863210825,
"extra": "iterations: 26\ncpu: 0.011057555307692318 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025747587921422558,
"extra": "iterations: 109\ncpu: 0.002574501357798159 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017530678203270872,
"extra": "iterations: 159\ncpu: 0.0017530972955974848 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036528853626994345,
"extra": "iterations: 77\ncpu: 0.0036527019740259757 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.00283745356968471,
"extra": "iterations: 98\ncpu: 0.002837491765306115 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004432080283997551,
"extra": "iterations: 63\ncpu: 0.004432009904761905 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003595355229500013,
"extra": "iterations: 78\ncpu: 0.0035950440512820497 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005673578807285854,
"extra": "iterations: 49\ncpu: 0.005673518367346942 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004827043105816019,
"extra": "iterations: 58\ncpu: 0.0048267352413793015 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001108510971069336,
"extra": "iterations: 500\ncpu: 0.0005241408439999998 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00166646149458898,
"extra": "iterations: 373\ncpu: 0.0007889964986595165 seconds\nthreads: 1"
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
"date": 1783589338337,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022050465695113417,
"extra": "iterations: 1268\ncpu: 0.0002204968619873817 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020293787218463792,
"extra": "iterations: 1387\ncpu: 0.00020291185219899064 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002603714112882261,
"extra": "iterations: 108\ncpu: 0.0026036699722222223 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001796372425861848,
"extra": "iterations: 156\ncpu: 0.0017963993653846155 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005871574929419985,
"extra": "iterations: 47\ncpu: 0.005871658829787234 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0038908322652180995,
"extra": "iterations: 72\ncpu: 0.0038906480555555506 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002581917911494544,
"extra": "iterations: 109\ncpu: 0.0025817889449541286 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018524934124472916,
"extra": "iterations: 151\ncpu: 0.0018524589735099356 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005234817288956552,
"extra": "iterations: 53\ncpu: 0.005234687132075472 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008049992152622767,
"extra": "iterations: 35\ncpu: 0.008048847285714294 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01062472967001108,
"extra": "iterations: 26\ncpu: 0.010623953461538474 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025814695095797203,
"extra": "iterations: 109\ncpu: 0.002581098211009172 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001759559103527909,
"extra": "iterations: 159\ncpu: 0.001759584050314466 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003654947528591404,
"extra": "iterations: 77\ncpu: 0.0036547640000000035 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002847827210718272,
"extra": "iterations: 98\ncpu: 0.0028475877346938734 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0044587718115912545,
"extra": "iterations: 63\ncpu: 0.004458658047619044 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003631287499477989,
"extra": "iterations: 76\ncpu: 0.0036308620000000094 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00579624871412913,
"extra": "iterations: 48\ncpu: 0.005796135937500003 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004811574672830516,
"extra": "iterations: 58\ncpu: 0.004811645017241379 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010746818160261293,
"extra": "iterations: 509\ncpu: 0.0004972900510805507 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015508642002027863,
"extra": "iterations: 392\ncpu: 0.0007008224974489812 seconds\nthreads: 1"
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
"date": 1783589338308,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021820439774628046,
"extra": "iterations: 1271\ncpu: 0.00021820785837922896 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020188162292259328,
"extra": "iterations: 1380\ncpu: 0.00020188458333333335 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025827884674072266,
"extra": "iterations: 108\ncpu: 0.0025826528981481474 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017585274558397212,
"extra": "iterations: 159\ncpu: 0.0017585616981132073 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005787596106529236,
"extra": "iterations: 48\ncpu: 0.0057874528750000015 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037498313027459224,
"extra": "iterations: 74\ncpu: 0.0037498942972972967 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025935393792611582,
"extra": "iterations: 108\ncpu: 0.0025935786481481485 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017778073685078683,
"extra": "iterations: 158\ncpu: 0.001777661322784809 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005196454390039984,
"extra": "iterations: 53\ncpu: 0.005196529528301886 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008202287128993443,
"extra": "iterations: 35\ncpu: 0.008201806000000011 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01053899985093337,
"extra": "iterations: 26\ncpu: 0.010539170730769222 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025898340706513308,
"extra": "iterations: 107\ncpu: 0.002589870532710282 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001762266400494153,
"extra": "iterations: 158\ncpu: 0.0017621707278481003 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003669628971501401,
"extra": "iterations: 76\ncpu: 0.0036696831184210503 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002822399139404297,
"extra": "iterations: 98\ncpu: 0.0028224421632653063 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004475987146771144,
"extra": "iterations: 63\ncpu: 0.004475848190476179 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003569566286527194,
"extra": "iterations: 78\ncpu: 0.003569619974358969 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005945642789204915,
"extra": "iterations: 48\ncpu: 0.005945109166666684 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004850083384020576,
"extra": "iterations: 58\ncpu: 0.004849951499999998 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0013852620841095888,
"extra": "iterations: 466\ncpu: 0.0006077576974248934 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0017967186515639635,
"extra": "iterations: 317\ncpu: 0.0008047507507886448 seconds\nthreads: 1"
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
"date": 1783589338256,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021789660157880161,
"extra": "iterations: 1273\ncpu: 0.00021789994579732918 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020160427876219123,
"extra": "iterations: 1389\ncpu: 0.00020159803599712023 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.00261769116481888,
"extra": "iterations: 107\ncpu: 0.00261727538317757 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017857566857949283,
"extra": "iterations: 156\ncpu: 0.001785786807692308 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005822628102404006,
"extra": "iterations: 47\ncpu: 0.005822208957446805 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004053070264704087,
"extra": "iterations: 68\ncpu: 0.004053133132352939 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002609665747042056,
"extra": "iterations: 108\ncpu: 0.0026097036388888885 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017669755707746782,
"extra": "iterations: 159\ncpu: 0.0017667908930817607 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005221764246622722,
"extra": "iterations: 54\ncpu: 0.005221840685185181 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008052492141723632,
"extra": "iterations: 35\ncpu: 0.008051748714285711 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010899219512939453,
"extra": "iterations: 25\ncpu: 0.010899400319999995 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002609052039958813,
"extra": "iterations: 108\ncpu: 0.002608987388888883 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001770529777381071,
"extra": "iterations: 157\ncpu: 0.0017702632229299367 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003687060674031576,
"extra": "iterations: 75\ncpu: 0.0036871254800000014 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002875515392848424,
"extra": "iterations: 98\ncpu: 0.002875249142857145 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004571696122487386,
"extra": "iterations: 60\ncpu: 0.00457104540000001 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036222934722900395,
"extra": "iterations: 77\ncpu: 0.003622350441558442 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005791013439496358,
"extra": "iterations: 48\ncpu: 0.005790766979166658 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004918771877623441,
"extra": "iterations: 57\ncpu: 0.004918855877192972 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0014096074582083053,
"extra": "iterations: 509\ncpu: 0.0005970687603143431 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0020414310128681525,
"extra": "iterations: 327\ncpu: 0.0008690318746177355 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338236,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002183053698287976,
"extra": "iterations: 1269\ncpu: 0.00021830897005516156 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020130435253272577,
"extra": "iterations: 1387\ncpu: 0.00020129317664023067 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002592696333831211,
"extra": "iterations: 106\ncpu: 0.0025927292547169812 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017642144915423815,
"extra": "iterations: 158\ncpu: 0.0017640924113924042 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005825782815615336,
"extra": "iterations: 48\ncpu: 0.005825646812500002 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037955029370033582,
"extra": "iterations: 73\ncpu: 0.0037955612602739693 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025927446506641533,
"extra": "iterations: 108\ncpu: 0.00259269737037037 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001772349393820461,
"extra": "iterations: 158\ncpu: 0.001772379069620253 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005308380666768776,
"extra": "iterations: 53\ncpu: 0.005308462660377358 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008109807968139648,
"extra": "iterations: 35\ncpu: 0.008109232799999994 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010908145170945387,
"extra": "iterations: 26\ncpu: 0.010907765576923086 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002601048656713183,
"extra": "iterations: 107\ncpu: 0.0026009925794392512 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017481824886873833,
"extra": "iterations: 159\ncpu: 0.0017482079245282997 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036607949357283748,
"extra": "iterations: 76\ncpu: 0.003660461039473683 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002842726166715327,
"extra": "iterations: 97\ncpu: 0.0028426724639175264 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004495651491226689,
"extra": "iterations: 62\ncpu: 0.004495732693548387 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003625584887219714,
"extra": "iterations: 77\ncpu: 0.0036254985064935 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005912721157073975,
"extra": "iterations: 48\ncpu: 0.005912813791666676 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004915818833468253,
"extra": "iterations: 57\ncpu: 0.004915913894736833 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010743310094362328,
"extra": "iterations: 494\ncpu: 0.0005049402408906892 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016611043144674864,
"extra": "iterations: 391\ncpu: 0.0007819266572890029 seconds\nthreads: 1"
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
"date": 1783589338333,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002198836855058586,
"extra": "iterations: 1247\ncpu: 0.00021988739935846033 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002012369109363091,
"extra": "iterations: 1394\ncpu: 0.00020123219655667144 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026069244491719753,
"extra": "iterations: 107\ncpu: 0.0026069710186915887 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017767058815925745,
"extra": "iterations: 157\ncpu: 0.0017767372738853495 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005860733985900879,
"extra": "iterations: 40\ncpu: 0.005860258450000001 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003765521822748958,
"extra": "iterations: 74\ncpu: 0.0037654810675675663 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026117663517176547,
"extra": "iterations: 107\ncpu: 0.00261181127102804 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017775810217555566,
"extra": "iterations: 158\ncpu: 0.0017776088227848103 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005262707764247678,
"extra": "iterations: 53\ncpu: 0.005262791641509435 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00814645430620979,
"extra": "iterations: 34\ncpu: 0.008146189852941172 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010825624832740197,
"extra": "iterations: 26\ncpu: 0.010825798423076917 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002604159239296601,
"extra": "iterations: 107\ncpu: 0.0026036132710280336 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017610286016884092,
"extra": "iterations: 159\ncpu: 0.001761053710691822 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.00368683080415468,
"extra": "iterations: 74\ncpu: 0.003686889621621613 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0029790863698842575,
"extra": "iterations: 98\ncpu: 0.00297901708163265 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004485257210270051,
"extra": "iterations: 62\ncpu: 0.004484848838709671 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035862189072829033,
"extra": "iterations: 78\ncpu: 0.003586282205128205 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005734796325365703,
"extra": "iterations: 48\ncpu: 0.0057345347708333185 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004855772544597757,
"extra": "iterations: 58\ncpu: 0.004855852310344838 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001116837115006146,
"extra": "iterations: 491\ncpu: 0.0005221251608961316 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015620756894350052,
"extra": "iterations: 384\ncpu: 0.0007087171614583351 seconds\nthreads: 1"
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
"date": 1783589338226,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002200780091462312,
"extra": "iterations: 1269\ncpu: 0.0002200814885736801 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002014942746341057,
"extra": "iterations: 1388\ncpu: 0.0002014919963976946 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002590197103994864,
"extra": "iterations: 108\ncpu: 0.0025899631111111114 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017725221670357285,
"extra": "iterations: 157\ncpu: 0.001772550356687898 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005960631877818007,
"extra": "iterations: 47\ncpu: 0.00596048770212766 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003821441572006435,
"extra": "iterations: 73\ncpu: 0.0038215021095890406 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025978662349559643,
"extra": "iterations: 108\ncpu: 0.0025979112685185182 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001777257129644892,
"extra": "iterations: 157\ncpu: 0.0017771202229299365 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0053203465803614205,
"extra": "iterations: 53\ncpu: 0.005320430471698107 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00798694065638951,
"extra": "iterations: 35\ncpu: 0.007986556485714289 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01064764536344088,
"extra": "iterations: 26\ncpu: 0.010647827269230765 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026033905064948254,
"extra": "iterations: 107\ncpu: 0.0026034361121495295 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017739519288268273,
"extra": "iterations: 158\ncpu: 0.0017738908734177218 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003700396219889323,
"extra": "iterations: 75\ncpu: 0.0037003269066666774 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028552886137028334,
"extra": "iterations: 97\ncpu: 0.0028552399175257722 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004618081890168738,
"extra": "iterations: 61\ncpu: 0.004618157065573772 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003586353399814704,
"extra": "iterations: 78\ncpu: 0.003586409628205126 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0056963161546356835,
"extra": "iterations: 49\ncpu: 0.005695973306122457 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004923669915450247,
"extra": "iterations: 57\ncpu: 0.004923538982456147 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001071612658667657,
"extra": "iterations: 514\ncpu: 0.0004982473190661482 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015267168947041328,
"extra": "iterations: 406\ncpu: 0.0006926227216748782 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338291,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.000220495884812723,
"extra": "iterations: 1270\ncpu: 0.0002204696299212598 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020229842221040735,
"extra": "iterations: 1387\ncpu: 0.00020230155010814713 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0027991789523686204,
"extra": "iterations: 107\ncpu: 0.0027990766448598133 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017773193918216002,
"extra": "iterations: 157\ncpu: 0.0017773487388535033 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005806833505630493,
"extra": "iterations: 48\ncpu: 0.005806732166666662 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003769710257246688,
"extra": "iterations: 74\ncpu: 0.0037692855945945934 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002586817299878156,
"extra": "iterations: 108\ncpu: 0.0025868582500000008 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017740545393545417,
"extra": "iterations: 158\ncpu: 0.0017739610000000012 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005328155913442936,
"extra": "iterations: 53\ncpu: 0.005327025037735846 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008069669499116786,
"extra": "iterations: 34\ncpu: 0.008069501529411766 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01078191170325646,
"extra": "iterations: 26\ncpu: 0.010779594576923056 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002692779647969754,
"extra": "iterations: 107\ncpu: 0.002692821065420565 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017589758027274653,
"extra": "iterations: 159\ncpu: 0.0017588780125786122 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036596530362179407,
"extra": "iterations: 76\ncpu: 0.003659091368421049 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028431950783242986,
"extra": "iterations: 98\ncpu: 0.0028432390204081656 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00447739116729252,
"extra": "iterations: 63\ncpu: 0.004477206269841265 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036040758475279202,
"extra": "iterations: 78\ncpu: 0.0036036709230769246 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0057514541003168855,
"extra": "iterations: 49\ncpu: 0.005751344571428558 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004906817486411647,
"extra": "iterations: 57\ncpu: 0.00490579398245614 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011140316482481918,
"extra": "iterations: 492\ncpu: 0.0005253576829268295 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001573377322656503,
"extra": "iterations: 386\ncpu: 0.0007410654559585501 seconds\nthreads: 1"
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
"date": 1783589338313,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002183320006606721,
"extra": "iterations: 1276\ncpu: 0.0002183060007836991 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020073595826902897,
"extra": "iterations: 1394\ncpu: 0.00020073921018651366 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025717439474882905,
"extra": "iterations: 108\ncpu: 0.0025716908888888898 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017864447605760792,
"extra": "iterations: 158\ncpu: 0.0017862582468354431 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005777938025338309,
"extra": "iterations: 49\ncpu: 0.005778028204081635 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003754037221272787,
"extra": "iterations: 75\ncpu: 0.00375409053333333 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025835324216772013,
"extra": "iterations: 108\ncpu: 0.0025835640462962975 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017678632011896447,
"extra": "iterations: 158\ncpu: 0.0017677532341772175 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00527192928172924,
"extra": "iterations: 54\ncpu: 0.005270891074074068 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008126118603874657,
"extra": "iterations: 34\ncpu: 0.008125917882352939 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011613426208496095,
"extra": "iterations: 25\ncpu: 0.011611052960000024 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.00259486314292266,
"extra": "iterations: 107\ncpu: 0.00259477959813084 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017589496660836137,
"extra": "iterations: 158\ncpu: 0.001758906721518989 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003675319646534167,
"extra": "iterations: 76\ncpu: 0.0036746265394736843 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002855707188041843,
"extra": "iterations: 98\ncpu: 0.0028555247857142874 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004454419726417178,
"extra": "iterations: 63\ncpu: 0.004454309904761903 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036181010209120714,
"extra": "iterations: 77\ncpu: 0.003617641116883111 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0057481551656917654,
"extra": "iterations: 49\ncpu: 0.005748249387755115 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0051834708765933395,
"extra": "iterations: 57\ncpu: 0.005182345789473684 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0013947841635939852,
"extra": "iterations: 457\ncpu: 0.0006024239102844648 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0019779435570737654,
"extra": "iterations: 321\ncpu: 0.0008889879190031153 seconds\nthreads: 1"
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
"date": 1783589338266,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021799411653910928,
"extra": "iterations: 1274\ncpu: 0.00021799791287284144 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020238143810327502,
"extra": "iterations: 1380\ncpu: 0.00020237190652173912 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002583586841548255,
"extra": "iterations: 109\ncpu: 0.002583534229357798 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017720672148692457,
"extra": "iterations: 158\ncpu: 0.0017720906835443039 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005783533056577046,
"extra": "iterations: 48\ncpu: 0.00578349714583333 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037751745533298807,
"extra": "iterations: 74\ncpu: 0.003775012500000002 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002586417728000217,
"extra": "iterations: 108\ncpu: 0.0025864602870370344 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017927376328000597,
"extra": "iterations: 157\ncpu: 0.0017927159999999989 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005239342743495725,
"extra": "iterations: 53\ncpu: 0.0052394314339622595 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00798372541155134,
"extra": "iterations: 35\ncpu: 0.007983601028571423 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010845101796663724,
"extra": "iterations: 26\ncpu: 0.010841749923076936 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025858349270290798,
"extra": "iterations: 108\ncpu: 0.002585875018518517 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001765189680663295,
"extra": "iterations: 159\ncpu: 0.0017652153081760974 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036490180275656962,
"extra": "iterations: 77\ncpu: 0.0036489012467532476 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002844856709850078,
"extra": "iterations: 98\ncpu: 0.0028448112346938837 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004460959207443964,
"extra": "iterations: 63\ncpu: 0.004460873111111108 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035895000804554334,
"extra": "iterations: 77\ncpu: 0.003589561090909093 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005785747450225208,
"extra": "iterations: 49\ncpu: 0.005785373775510218 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004865784394113642,
"extra": "iterations: 57\ncpu: 0.0048658686140350795 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001106458297674207,
"extra": "iterations: 552\ncpu: 0.0005205187373188409 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016227387093208931,
"extra": "iterations: 370\ncpu: 0.0007395407459459456 seconds\nthreads: 1"
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
"date": 1783589338248,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022461048409908632,
"extra": "iterations: 1264\ncpu: 0.00022461371281645576 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020132677553372683,
"extra": "iterations: 1393\ncpu: 0.00020132983417085428 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002620406107071343,
"extra": "iterations: 109\ncpu: 0.0026202299174311935 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017606300912844908,
"extra": "iterations: 157\ncpu: 0.0017606629044585992 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0057733505964279175,
"extra": "iterations: 48\ncpu: 0.005773232187500005 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037704944610595704,
"extra": "iterations: 75\ncpu: 0.0037705592533333328 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025940453894784518,
"extra": "iterations: 107\ncpu: 0.002594088579439252 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017974422823998238,
"extra": "iterations: 155\ncpu: 0.0017972605677419363 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005235102441575792,
"extra": "iterations: 54\ncpu: 0.005234489111111113 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008054842267717634,
"extra": "iterations: 35\ncpu: 0.008054698771428573 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010803983761714054,
"extra": "iterations: 26\ncpu: 0.01080273811538462 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025612367402522936,
"extra": "iterations: 109\ncpu: 0.0025612825321100854 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017630514108909752,
"extra": "iterations: 159\ncpu: 0.0017629772955974838 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003720862070719401,
"extra": "iterations: 75\ncpu: 0.0037209194533333327 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028432210286458335,
"extra": "iterations: 99\ncpu: 0.0028431967676767703 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004428046090262277,
"extra": "iterations: 63\ncpu: 0.004427533952380954 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035787383212318906,
"extra": "iterations: 79\ncpu: 0.00357879644303797 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005665122246255681,
"extra": "iterations: 49\ncpu: 0.005664853020408169 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00485415294252593,
"extra": "iterations: 58\ncpu: 0.004853230534482757 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010755426215327433,
"extra": "iterations: 478\ncpu: 0.0005066387991631789 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001597946286201477,
"extra": "iterations: 400\ncpu: 0.0007263728524999992 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338261,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021896284256516635,
"extra": "iterations: 1283\ncpu: 0.00021896618628215118 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020284769852037376,
"extra": "iterations: 1384\ncpu: 0.00020282426156069364 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002755967848891512,
"extra": "iterations: 109\ncpu: 0.0027560149449541286 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017667476486110088,
"extra": "iterations: 159\ncpu: 0.0017665157232704396 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0057457807112713255,
"extra": "iterations: 49\ncpu: 0.0057458823265306115 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003756883982065562,
"extra": "iterations: 74\ncpu: 0.003756808418918915 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025884950602496114,
"extra": "iterations: 108\ncpu: 0.002588132472222222 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017680986872259177,
"extra": "iterations: 159\ncpu: 0.0017681231823899366 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005255586696120929,
"extra": "iterations: 53\ncpu: 0.005255301433962262 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007959236417497908,
"extra": "iterations: 35\ncpu: 0.00795913025714286 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01074654322404128,
"extra": "iterations: 26\ncpu: 0.01074633749999999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002650577997423939,
"extra": "iterations: 97\ncpu: 0.0026500803092783525 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001758563443549774,
"extra": "iterations: 159\ncpu: 0.001758589867924524 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036587560331666625,
"extra": "iterations: 77\ncpu: 0.0036585558701298683 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028269098262594205,
"extra": "iterations: 99\ncpu: 0.0028268410101010104 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004448247334313771,
"extra": "iterations: 63\ncpu: 0.0044481659206349255 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003590262853182279,
"extra": "iterations: 78\ncpu: 0.003590209538461539 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005740058665372888,
"extra": "iterations: 49\ncpu: 0.005740151836734706 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0048847491281074394,
"extra": "iterations: 57\ncpu: 0.0048843900000000075 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010755006774065365,
"extra": "iterations: 474\ncpu: 0.0005061927088607596 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015586027827287584,
"extra": "iterations: 386\ncpu: 0.000731820606217617 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338234,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021774239010281037,
"extra": "iterations: 1278\ncpu: 0.00021773867683881065 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002036089459881512,
"extra": "iterations: 1374\ncpu: 0.00020360509170305678 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025917424096001517,
"extra": "iterations: 108\ncpu: 0.002591378712962964 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017561042833628146,
"extra": "iterations: 159\ncpu: 0.0017560572138364777 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005819087227185567,
"extra": "iterations: 48\ncpu: 0.005817656104166666 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003771177709919133,
"extra": "iterations: 73\ncpu: 0.003771235739726026 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025993784268697104,
"extra": "iterations: 108\ncpu: 0.0025992370833333336 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001772749273082878,
"extra": "iterations: 158\ncpu: 0.0017725040443037972 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005263040650565669,
"extra": "iterations: 53\ncpu: 0.005263119962264152 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008109406062534878,
"extra": "iterations: 35\ncpu: 0.008108461028571438 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011031885147094727,
"extra": "iterations: 25\ncpu: 0.011031527999999983 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025929300873367878,
"extra": "iterations: 108\ncpu: 0.0025928634259259273 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017895895964021137,
"extra": "iterations: 157\ncpu: 0.0017892122738853466 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036763078288028117,
"extra": "iterations: 76\ncpu: 0.0036763613552631517 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002859881946018764,
"extra": "iterations: 98\ncpu: 0.0028595525102040817 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004601603648701652,
"extra": "iterations: 61\ncpu: 0.004600606983606557 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003646438772028143,
"extra": "iterations: 77\ncpu: 0.003646333350649353 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005967089470396651,
"extra": "iterations: 47\ncpu: 0.0059662365106383 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0050577521324157715,
"extra": "iterations: 56\ncpu: 0.005057826892857133 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010759280662616594,
"extra": "iterations: 477\ncpu: 0.0005158797085953879 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015937092968874881,
"extra": "iterations: 391\ncpu: 0.000730387235294117 seconds\nthreads: 1"
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
"date": 1783589338310,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002210757155327281,
"extra": "iterations: 1256\ncpu: 0.00022104378742038218 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020140002977052184,
"extra": "iterations: 1393\ncpu: 0.00020138805599425702 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026134441946154443,
"extra": "iterations: 107\ncpu: 0.0026133629345794387 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001777234350799755,
"extra": "iterations: 157\ncpu: 0.0017771991019108278 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005868643522262573,
"extra": "iterations: 48\ncpu: 0.005868446020833335 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037903721268112596,
"extra": "iterations: 74\ncpu: 0.003789668486486485 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026290041429025157,
"extra": "iterations: 108\ncpu: 0.0026290461018518515 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017864533833095006,
"extra": "iterations: 154\ncpu: 0.0017863811038961051 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005257939392665647,
"extra": "iterations: 53\ncpu: 0.005256835207547173 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008054656140944538,
"extra": "iterations: 34\ncpu: 0.008054791999999993 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011079940795898437,
"extra": "iterations: 25\ncpu: 0.011076489399999988 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.00260492828157213,
"extra": "iterations: 108\ncpu: 0.0026047607222222293 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001779694466074561,
"extra": "iterations: 157\ncpu: 0.0017796591528662427 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003709192276000977,
"extra": "iterations: 75\ncpu: 0.003708775199999994 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002855344694487903,
"extra": "iterations: 98\ncpu: 0.0028553892346938844 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004556386701522335,
"extra": "iterations: 62\ncpu: 0.004555709870967737 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003745125300848662,
"extra": "iterations: 67\ncpu: 0.003745224507462677 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0059943503521858385,
"extra": "iterations: 47\ncpu: 0.005994128957446814 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004997093027288264,
"extra": "iterations: 55\ncpu: 0.004996338236363634 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011199124777109526,
"extra": "iterations: 517\ncpu: 0.0005198984468085105 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015942522684733074,
"extra": "iterations: 375\ncpu: 0.000758181933333333 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338328,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022035827874765384,
"extra": "iterations: 1282\ncpu: 0.0002203412145085804 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002015803357680067,
"extra": "iterations: 1390\ncpu: 0.00020153182517985616 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026184342525623467,
"extra": "iterations: 108\ncpu: 0.002618371111111111 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017750006687792042,
"extra": "iterations: 158\ncpu: 0.0017750259683544309 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005799834926923116,
"extra": "iterations: 48\ncpu: 0.005799944083333337 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037599754333496097,
"extra": "iterations: 75\ncpu: 0.003760031599999998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025942303309930822,
"extra": "iterations: 107\ncpu: 0.00259393990654206 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017910711226924775,
"extra": "iterations: 155\ncpu: 0.0017910972645161281 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005290053925424252,
"extra": "iterations: 53\ncpu: 0.005289921283018866 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007903262547084264,
"extra": "iterations: 35\ncpu: 0.007902710085714292 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010663206760699932,
"extra": "iterations: 26\ncpu: 0.010663038692307685 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026010130053368686,
"extra": "iterations: 107\ncpu: 0.0026002961121495377 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017867490842744902,
"extra": "iterations: 154\ncpu: 0.0017866857727272705 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036593133752996273,
"extra": "iterations: 77\ncpu: 0.003659199571428573 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028452946215259786,
"extra": "iterations: 98\ncpu: 0.002845101061224487 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004461220332554409,
"extra": "iterations: 63\ncpu: 0.004461285158730159 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003588324938064967,
"extra": "iterations: 78\ncpu: 0.003588108038461538 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005772228042284648,
"extra": "iterations: 48\ncpu: 0.005771972104166671 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004827273303064807,
"extra": "iterations: 58\ncpu: 0.004827337224137931 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00111902767173902,
"extra": "iterations: 509\ncpu: 0.0005316380039292727 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015852974060384987,
"extra": "iterations: 374\ncpu: 0.0007230305401069514 seconds\nthreads: 1"
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
"date": 1783589338321,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002160782549323469,
"extra": "iterations: 1279\ncpu: 0.0002160644964816263 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002022758395621851,
"extra": "iterations: 1383\ncpu: 0.00020226770354302243 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025873404962045176,
"extra": "iterations: 108\ncpu: 0.0025872459259259272 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017360930117020696,
"extra": "iterations: 161\ncpu: 0.0017360696645962737 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005754514616362903,
"extra": "iterations: 49\ncpu: 0.005754593673469392 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003694267649399607,
"extra": "iterations: 76\ncpu: 0.0036943231578947344 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025726367380017436,
"extra": "iterations: 107\ncpu: 0.0025726149719626163 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017324486134215173,
"extra": "iterations: 161\ncpu: 0.001732475881987578 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005173970151830603,
"extra": "iterations: 54\ncpu: 0.005173915203703698 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00780585077073839,
"extra": "iterations: 36\ncpu: 0.007805966611111116 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010666278692392202,
"extra": "iterations: 26\ncpu: 0.010666435807692323 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0025689580024929226,
"extra": "iterations: 109\ncpu: 0.0025688250366972475 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017381682991981506,
"extra": "iterations: 160\ncpu: 0.0017379519000000011 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.00365741629349558,
"extra": "iterations: 76\ncpu: 0.0036572738947368423 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002785400910810991,
"extra": "iterations: 99\ncpu: 0.0027854395353535412 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004431452602148056,
"extra": "iterations: 64\ncpu: 0.004431520515624995 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035355916390052214,
"extra": "iterations: 78\ncpu: 0.00353551983333334 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005751001591585121,
"extra": "iterations: 49\ncpu: 0.005750668224489795 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004775992306795988,
"extra": "iterations: 55\ncpu: 0.004776067545454539 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010247401158168653,
"extra": "iterations: 517\ncpu: 0.000487170864603482 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001525509386457086,
"extra": "iterations: 411\ncpu: 0.000686196751824817 seconds\nthreads: 1"
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
"date": 1783589338258,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002098737166367269,
"extra": "iterations: 1324\ncpu: 0.00020987693731117825 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00019235775957470534,
"extra": "iterations: 1445\ncpu: 0.0001923489349480969 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0025869106577935622,
"extra": "iterations: 107\ncpu: 0.002586580261682243 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017549455165863038,
"extra": "iterations: 160\ncpu: 0.0017547427625 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005851248900095622,
"extra": "iterations: 48\ncpu: 0.005851326375000003 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037634340922037763,
"extra": "iterations: 75\ncpu: 0.003763493666666662 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0025988338149596597,
"extra": "iterations: 107\ncpu: 0.002598672551401871 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017737768873383728,
"extra": "iterations: 158\ncpu: 0.0017737456012658218 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0052063730027940534,
"extra": "iterations: 54\ncpu: 0.005206458481481483 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007915762492588588,
"extra": "iterations: 35\ncpu: 0.007915885542857142 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010735429250277005,
"extra": "iterations: 26\ncpu: 0.010735621346153855 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002608978859732084,
"extra": "iterations: 107\ncpu: 0.0026088211121495283 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017692801318591155,
"extra": "iterations: 158\ncpu: 0.001769308905063295 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003647479144009677,
"extra": "iterations: 77\ncpu: 0.003647407818181813 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002836798628171285,
"extra": "iterations: 96\ncpu: 0.0028367687291666705 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004416518741183811,
"extra": "iterations: 63\ncpu: 0.004416610539682532 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035647048225885707,
"extra": "iterations: 79\ncpu: 0.003564659556962027 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005655142725730429,
"extra": "iterations: 49\ncpu: 0.005654715530612256 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004926821281170023,
"extra": "iterations: 58\ncpu: 0.004926733655172408 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0014080256223678589,
"extra": "iterations: 432\ncpu: 0.0006020018449074076 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001987669203016493,
"extra": "iterations: 324\ncpu: 0.000858509345679012 seconds\nthreads: 1"
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
"date": 1783589338274,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022828753685718252,
"extra": "iterations: 1228\ncpu: 0.0002282427687296417 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021154514005013617,
"extra": "iterations: 1314\ncpu: 0.00021153174809741246 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026658103579566593,
"extra": "iterations: 105\ncpu: 0.0026654507428571436 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017678888538215734,
"extra": "iterations: 158\ncpu: 0.001767915367088607 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005988866724866503,
"extra": "iterations: 47\ncpu: 0.0059887640851063835 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003750782012939453,
"extra": "iterations: 75\ncpu: 0.0037503920400000015 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002664777210780553,
"extra": "iterations: 105\ncpu: 0.002664825 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017639316102993565,
"extra": "iterations: 159\ncpu: 0.0017639128176100624 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005301682454235149,
"extra": "iterations: 53\ncpu: 0.005301146735849052 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00808178356715611,
"extra": "iterations: 35\ncpu: 0.008081913228571429 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010918488869300256,
"extra": "iterations: 26\ncpu: 0.010917966538461537 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002641348611740839,
"extra": "iterations: 105\ncpu: 0.0026413865142857133 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017497673630714418,
"extra": "iterations: 160\ncpu: 0.001749714631249999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037257893880208335,
"extra": "iterations: 75\ncpu: 0.003725848520000004 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028279815057311395,
"extra": "iterations: 99\ncpu: 0.002828023212121221 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004491363252912248,
"extra": "iterations: 63\ncpu: 0.004491221460317469 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003591522192343687,
"extra": "iterations: 78\ncpu: 0.0035915724487179403 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005832955241203308,
"extra": "iterations: 48\ncpu: 0.005832527562500004 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005021280255810968,
"extra": "iterations: 58\ncpu: 0.005020806568965527 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0013747374559270925,
"extra": "iterations: 464\ncpu: 0.000613291375000001 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0020280917988547797,
"extra": "iterations: 316\ncpu: 0.0008819873797468357 seconds\nthreads: 1"
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
"date": 1783589338279,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022621910746504622,
"extra": "iterations: 1230\ncpu: 0.00022622317886178864 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002108933607617445,
"extra": "iterations: 1327\ncpu: 0.00021088988470233612 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002645556131998698,
"extra": "iterations: 105\ncpu: 0.0026455914095238095 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017564356327056886,
"extra": "iterations: 160\ncpu: 0.0017564632812499996 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005977792942777593,
"extra": "iterations: 47\ncpu: 0.005977318255319147 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037738068898518886,
"extra": "iterations: 75\ncpu: 0.0037737585600000016 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002646032969156901,
"extra": "iterations: 105\ncpu: 0.0026460729809523824 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017557459057502029,
"extra": "iterations: 159\ncpu: 0.0017556442515723279 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005305335206805535,
"extra": "iterations: 53\ncpu: 0.005305273698113207 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008098377900965074,
"extra": "iterations: 34\ncpu: 0.008098095647058826 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01091624223268949,
"extra": "iterations: 26\ncpu: 0.01091504673076921 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026661697423683026,
"extra": "iterations: 106\ncpu: 0.0026659585 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017751629944819556,
"extra": "iterations: 157\ncpu: 0.0017750453949044568 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037373701731363935,
"extra": "iterations: 75\ncpu: 0.0037374297199999993 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002849498573614627,
"extra": "iterations: 98\ncpu: 0.0028494370102040757 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004515909379528415,
"extra": "iterations: 62\ncpu: 0.004515140774193554 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036257706679307024,
"extra": "iterations: 77\ncpu: 0.003625696675324675 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005841006835301717,
"extra": "iterations: 48\ncpu: 0.005840607229166683 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004872248090546707,
"extra": "iterations: 58\ncpu: 0.0048719132758620705 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011150213388296277,
"extra": "iterations: 520\ncpu: 0.0005247168673076927 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016235748715995163,
"extra": "iterations: 377\ncpu: 0.0007399113740053032 seconds\nthreads: 1"
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
"date": 1783589338243,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002283043853559322,
"extra": "iterations: 1218\ncpu: 0.00022828761658456493 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002129499184113827,
"extra": "iterations: 1313\ncpu: 0.00021294428255902513 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026826742783333493,
"extra": "iterations: 103\ncpu: 0.0026827156310679614 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001776121839692321,
"extra": "iterations: 158\ncpu: 0.0017761471075949367 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006006012571618913,
"extra": "iterations: 47\ncpu: 0.006005658978723407 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037772848799421986,
"extra": "iterations: 74\ncpu: 0.003777226216216219 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026765580360706034,
"extra": "iterations: 104\ncpu: 0.0026766062499999996 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017786921968885288,
"extra": "iterations: 157\ncpu: 0.0017786440191082812 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005445405548694088,
"extra": "iterations: 51\ncpu: 0.005445508509803922 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00835727944093592,
"extra": "iterations: 34\ncpu: 0.0083568860882353 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011242094039916993,
"extra": "iterations: 25\ncpu: 0.011241031120000003 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026755855196998234,
"extra": "iterations: 105\ncpu: 0.002675518380952377 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017840541326082672,
"extra": "iterations: 156\ncpu: 0.0017840831858974375 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037868409543424042,
"extra": "iterations: 74\ncpu: 0.0037869084594594677 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0029363556111112557,
"extra": "iterations: 94\ncpu: 0.002936185510638305 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0046073569626104635,
"extra": "iterations: 61\ncpu: 0.004607351163934432 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036586848172274504,
"extra": "iterations: 77\ncpu: 0.0036585498961038956 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005850702524185181,
"extra": "iterations: 48\ncpu: 0.005850525729166668 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004856833240442109,
"extra": "iterations: 57\ncpu: 0.004856752666666668 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011195894330739975,
"extra": "iterations: 512\ncpu: 0.0005267790429687499 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016007920913903048,
"extra": "iterations: 369\ncpu: 0.0007352562981029811 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338276,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002283465150722401,
"extra": "iterations: 1223\ncpu: 0.00022834994930498776 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021275703973893057,
"extra": "iterations: 1319\ncpu: 0.0002127530894617135 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002662002472650437,
"extra": "iterations: 105\ncpu: 0.002661614142857143 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.00177763685395446,
"extra": "iterations: 158\ncpu: 0.0017775315189873413 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006174272961086697,
"extra": "iterations: 45\ncpu: 0.006173625933333337 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004067534985749618,
"extra": "iterations: 69\ncpu: 0.004067594420289854 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002644784045669268,
"extra": "iterations: 106\ncpu: 0.002644734094339621 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017643874546266953,
"extra": "iterations: 159\ncpu: 0.001764113559748426 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0053279714764289145,
"extra": "iterations: 53\ncpu: 0.00532786849056604 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008169099262782507,
"extra": "iterations: 35\ncpu: 0.008169212942857156 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010834035873413087,
"extra": "iterations: 25\ncpu: 0.010834213519999983 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026720120356633114,
"extra": "iterations: 104\ncpu: 0.0026719655673076953 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017457248279883427,
"extra": "iterations: 159\ncpu: 0.0017456975157232673 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037356789906819667,
"extra": "iterations: 75\ncpu: 0.0037355037200000043 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.00285782764867409,
"extra": "iterations: 97\ncpu: 0.0028577482061855623 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0045411938526591315,
"extra": "iterations: 61\ncpu: 0.004541266918032774 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036824063250893045,
"extra": "iterations: 76\ncpu: 0.003682469065789472 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005758146444956462,
"extra": "iterations: 48\ncpu: 0.005756927374999999 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00485055787222726,
"extra": "iterations: 56\ncpu: 0.004850301285714287 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010736504597450372,
"extra": "iterations: 536\ncpu: 0.0005107394253731352 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016642171127789498,
"extra": "iterations: 357\ncpu: 0.0007778927871148458 seconds\nthreads: 1"
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
"date": 1783589338323,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00024312848561224302,
"extra": "iterations: 1203\ncpu: 0.0002431323782211139 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002089602811200483,
"extra": "iterations: 1332\ncpu: 0.0002089635495495496 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002728541692097982,
"extra": "iterations: 102\ncpu: 0.00272824105882353 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017657085034832266,
"extra": "iterations: 159\ncpu: 0.001765642018867925 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005963249409452398,
"extra": "iterations: 47\ncpu: 0.005962820340425533 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00376438772356188,
"extra": "iterations: 74\ncpu: 0.003764444540540542 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026958217987647424,
"extra": "iterations: 104\ncpu: 0.0026955727884615407 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017569874817470335,
"extra": "iterations: 159\ncpu: 0.0017567964528301893 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005429465037125807,
"extra": "iterations: 52\ncpu: 0.005429327750000004 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008190042832318474,
"extra": "iterations: 34\ncpu: 0.008189484970588239 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010935444098252516,
"extra": "iterations: 26\ncpu: 0.010935114269230781 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002681297414443072,
"extra": "iterations: 102\ncpu: 0.002681260382352937 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001762284508234338,
"extra": "iterations: 158\ncpu: 0.0017621553734177206 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003826758632921193,
"extra": "iterations: 73\ncpu: 0.0038266852054794466 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028478042366578407,
"extra": "iterations: 97\ncpu: 0.002847591938144328 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004718875062876734,
"extra": "iterations: 58\ncpu: 0.004718955051724143 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036225256166960065,
"extra": "iterations: 76\ncpu: 0.0036224775657894776 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0059274764771157125,
"extra": "iterations: 47\ncpu: 0.005926849 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00483607423716578,
"extra": "iterations: 58\ncpu: 0.00483597987931035 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010755290036903514,
"extra": "iterations: 523\ncpu: 0.0005142374608030585 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016543655197855105,
"extra": "iterations: 386\ncpu: 0.0007730381709844564 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338253,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022684861773904035,
"extra": "iterations: 1235\ncpu: 0.00022682454736842105 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020798604984931438,
"extra": "iterations: 1354\ncpu: 0.0002079723537666174 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026837541506840633,
"extra": "iterations: 104\ncpu: 0.002683723846153847 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0019107338804635944,
"extra": "iterations: 161\ncpu: 0.0019105749875776403 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00596030722273157,
"extra": "iterations: 47\ncpu: 0.005960399042553198 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037434959411621095,
"extra": "iterations: 75\ncpu: 0.003743454413333331 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028708485456613395,
"extra": "iterations: 104\ncpu: 0.0028707789134615387 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017483368515968324,
"extra": "iterations: 160\ncpu: 0.0017483152062500009 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005369952091803918,
"extra": "iterations: 52\ncpu: 0.0053696784230769295 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00836181640625,
"extra": "iterations: 33\ncpu: 0.008361946090909107 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011043100357055665,
"extra": "iterations: 25\ncpu: 0.011042057639999995 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002682385536340567,
"extra": "iterations: 104\ncpu: 0.002682424413461543 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001735719633691105,
"extra": "iterations: 162\ncpu: 0.001735695314814815 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037664626095746017,
"extra": "iterations: 74\ncpu: 0.0037663092432432435 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0027968382835388186,
"extra": "iterations: 100\ncpu: 0.002796880540000002 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00466250969191729,
"extra": "iterations: 59\ncpu: 0.004662435322033902 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035620774978246442,
"extra": "iterations: 78\ncpu: 0.0035620281153846156 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005767395099004109,
"extra": "iterations: 48\ncpu: 0.005767490437499985 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004743175991510941,
"extra": "iterations: 59\ncpu: 0.004743092745762718 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010672459691207603,
"extra": "iterations: 483\ncpu: 0.0005092719875776407 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00172314148444634,
"extra": "iterations: 385\ncpu: 0.0007874448597402583 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338342,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002258098211841307,
"extra": "iterations: 1104\ncpu: 0.00022580271376811595 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020934030290292204,
"extra": "iterations: 1329\ncpu: 0.00020933323175319793 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002688302443577693,
"extra": "iterations: 104\ncpu: 0.002688341317307693 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017608021790126585,
"extra": "iterations: 159\ncpu: 0.001760775264150944 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00591161847114563,
"extra": "iterations: 48\ncpu: 0.005911176354166668 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003740152797183475,
"extra": "iterations: 74\ncpu: 0.003740053108108108 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026861062416663538,
"extra": "iterations: 104\ncpu: 0.0026860603269230772 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017720773816108704,
"extra": "iterations: 160\ncpu: 0.0017721036375 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0054184931975144604,
"extra": "iterations: 52\ncpu: 0.005418595807692303 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008220735718222226,
"extra": "iterations: 34\ncpu: 0.008220580058823534 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011109123229980469,
"extra": "iterations: 25\ncpu: 0.011109328319999995 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0029031038284301758,
"extra": "iterations: 104\ncpu: 0.0029031452499999996 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017444442653056212,
"extra": "iterations: 159\ncpu: 0.0017444697547169797 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037894344329833986,
"extra": "iterations: 75\ncpu: 0.0037893567066666733 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028125709957546657,
"extra": "iterations: 99\ncpu: 0.002812403979797983 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0046614050865173345,
"extra": "iterations: 60\ncpu: 0.004661476916666659 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036234546017337153,
"extra": "iterations: 77\ncpu: 0.0036233821038960945 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005809153119723002,
"extra": "iterations: 48\ncpu: 0.005808999729166672 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004783856457677382,
"extra": "iterations: 58\ncpu: 0.004783936948275858 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00111211909240006,
"extra": "iterations: 511\ncpu: 0.0005209476418786709 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016940450668334961,
"extra": "iterations: 350\ncpu: 0.0007795561028571437 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338298,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022569106471153998,
"extra": "iterations: 1240\ncpu: 0.00022569465080645164 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021083854888985322,
"extra": "iterations: 1321\ncpu: 0.000210836044663134 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002695872945692933,
"extra": "iterations: 103\ncpu: 0.0026959139514563107 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017495185136795046,
"extra": "iterations: 160\ncpu: 0.001749396225000001 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005899543563524882,
"extra": "iterations: 48\ncpu: 0.005899640458333337 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037324428558349614,
"extra": "iterations: 74\ncpu: 0.0037325002297297254 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028984340337606575,
"extra": "iterations: 104\ncpu: 0.0028983997692307736 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017649372921714301,
"extra": "iterations: 158\ncpu: 0.0017649259873417727 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00547638593935499,
"extra": "iterations: 51\ncpu: 0.005476470980392158 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008213723407072179,
"extra": "iterations: 34\ncpu: 0.00821383279411764 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011267738342285156,
"extra": "iterations: 25\ncpu: 0.011267898079999982 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026836140641888374,
"extra": "iterations: 103\ncpu: 0.002683376970873785 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017639211138839243,
"extra": "iterations: 159\ncpu: 0.0017639470314465442 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037832227913109032,
"extra": "iterations: 74\ncpu: 0.0037831422702702695 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028772305469123684,
"extra": "iterations: 98\ncpu: 0.002877159438775512 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004696313540140789,
"extra": "iterations: 60\ncpu: 0.00469639153333333 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035414575021478194,
"extra": "iterations: 79\ncpu: 0.0035412273797468313 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005959898233413696,
"extra": "iterations: 48\ncpu: 0.005959621291666662 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004759695570347673,
"extra": "iterations: 59\ncpu: 0.004759645830508473 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010860702078901456,
"extra": "iterations: 499\ncpu: 0.0005117970040080166 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00163629516682511,
"extra": "iterations: 377\ncpu: 0.0007715309734748035 seconds\nthreads: 1"
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
"date": 1783589338271,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022620629259601017,
"extra": "iterations: 1234\ncpu: 0.00022619904376012967 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020997470482847744,
"extra": "iterations: 1330\ncpu: 0.0002099612503759399 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026824199236356295,
"extra": "iterations: 104\ncpu: 0.0026824571634615386 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017561987042427065,
"extra": "iterations: 160\ncpu: 0.0017559723 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0060356855392456055,
"extra": "iterations: 46\ncpu: 0.006035771391304344 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0038632154464721684,
"extra": "iterations: 72\ncpu: 0.0038631664166666655 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026968717575073242,
"extra": "iterations: 102\ncpu: 0.0026966967156862734 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017718227603767491,
"extra": "iterations: 158\ncpu: 0.001771849537974683 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005380694682781513,
"extra": "iterations: 52\ncpu: 0.005380628826923079 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008151545244104722,
"extra": "iterations: 34\ncpu: 0.008150965029411766 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010942321557265062,
"extra": "iterations: 26\ncpu: 0.010942154692307707 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.00267830491065979,
"extra": "iterations: 104\ncpu: 0.002678133990384618 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017472172375791563,
"extra": "iterations: 161\ncpu: 0.0017469748447204992 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037675580462893923,
"extra": "iterations: 74\ncpu: 0.003767510540540533 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002829296420318912,
"extra": "iterations: 99\ncpu: 0.0028292062222222253 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0046847820281982425,
"extra": "iterations: 60\ncpu: 0.004684548616666673 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003600679911099948,
"extra": "iterations: 78\ncpu: 0.0036002961538461557 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005830044442034782,
"extra": "iterations: 47\ncpu: 0.005829948276595748 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004779396385982119,
"extra": "iterations: 58\ncpu: 0.004778982206896547 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001068499536797552,
"extra": "iterations: 505\ncpu: 0.0005011321504950493 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016026466163163334,
"extra": "iterations: 388\ncpu: 0.0007503115902061853 seconds\nthreads: 1"
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
"date": 1783589338352,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002263717248406209,
"extra": "iterations: 994\ncpu: 0.00022619621126760564 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021042401050822706,
"extra": "iterations: 1331\ncpu: 0.00021042198347107438 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0027316060720705524,
"extra": "iterations: 102\ncpu: 0.0027315294607843134 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017346423349262757,
"extra": "iterations: 162\ncpu: 0.0017346214074074068 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0060873879326714415,
"extra": "iterations: 45\ncpu: 0.006087472111111117 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003807616560426477,
"extra": "iterations: 73\ncpu: 0.0038076656164383574 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0027333895365397134,
"extra": "iterations: 102\ncpu: 0.002733354421568626 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017434686422348024,
"extra": "iterations: 160\ncpu: 0.0017433681000000008 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005473688536999272,
"extra": "iterations: 51\ncpu: 0.005473766529411766 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00822274123921114,
"extra": "iterations: 34\ncpu: 0.00822286138235294 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011277532577514649,
"extra": "iterations: 25\ncpu: 0.01127770715999999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002736596500172335,
"extra": "iterations: 102\ncpu: 0.002736488598039215 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017608801523844404,
"extra": "iterations: 159\ncpu: 0.0017608447044025152 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003967016083853586,
"extra": "iterations: 70\ncpu: 0.003966590514285721 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028090284328268034,
"extra": "iterations: 99\ncpu: 0.002809072818181813 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004741981111723801,
"extra": "iterations: 58\ncpu: 0.0047420655689655224 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0037855668501420455,
"extra": "iterations: 77\ncpu: 0.00378510232467533 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005975941394237762,
"extra": "iterations: 47\ncpu: 0.0059755611489361714 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004851977030436198,
"extra": "iterations: 57\ncpu: 0.004852057245614028 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0014207242873677038,
"extra": "iterations: 456\ncpu: 0.0005928417521929833 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001727258857292465,
"extra": "iterations: 316\ncpu: 0.000793427990506328 seconds\nthreads: 1"
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
"date": 1783589338231,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022359798031468546,
"extra": "iterations: 1240\ncpu: 0.00022360085806451613 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021002225122579698,
"extra": "iterations: 1342\ncpu: 0.00021001773174366624 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026718139648437503,
"extra": "iterations: 105\ncpu: 0.002671730838095239 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017603456974029541,
"extra": "iterations: 160\ncpu: 0.0017602963 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005895447223744494,
"extra": "iterations: 47\ncpu: 0.005895297617021277 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0036915289728265063,
"extra": "iterations: 76\ncpu: 0.003690994855263156 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026864913793710563,
"extra": "iterations: 104\ncpu: 0.0026865306249999996 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017395241660360964,
"extra": "iterations: 161\ncpu: 0.0017394793788819871 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005398654020749605,
"extra": "iterations: 52\ncpu: 0.005398742423076928 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00848382360794965,
"extra": "iterations: 34\ncpu: 0.008483738323529406 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011100397109985352,
"extra": "iterations: 25\ncpu: 0.011098204759999997 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002679659071422759,
"extra": "iterations: 105\ncpu: 0.002679597419047617 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017420786508121847,
"extra": "iterations: 161\ncpu: 0.001742103347826087 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037630313151591536,
"extra": "iterations: 74\ncpu: 0.003763100959459462 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002784441013147335,
"extra": "iterations: 101\ncpu: 0.0027843651782178177 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004643440246582031,
"extra": "iterations: 60\ncpu: 0.004643352766666676 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00357315208338484,
"extra": "iterations: 79\ncpu: 0.003573207278481008 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005842640995979309,
"extra": "iterations: 48\ncpu: 0.005842543958333335 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004767422018380001,
"extra": "iterations: 58\ncpu: 0.004767183482758631 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001108402159155869,
"extra": "iterations: 492\ncpu: 0.0005168244227642285 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015887113717886119,
"extra": "iterations: 338\ncpu: 0.0007398036360946744 seconds\nthreads: 1"
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
"date": 1783589338239,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002251468053678187,
"extra": "iterations: 1230\ncpu: 0.00022515072357723578 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002108815007006489,
"extra": "iterations: 1337\ncpu: 0.00021088471054599853 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002668582476102389,
"extra": "iterations: 104\ncpu: 0.002668564701923078 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017621830105781556,
"extra": "iterations: 160\ncpu: 0.0017620688562499987 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00599885494151014,
"extra": "iterations: 47\ncpu: 0.005998960106382977 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0036882845560709636,
"extra": "iterations: 75\ncpu: 0.003688098240000001 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026750240511107214,
"extra": "iterations: 103\ncpu: 0.0026750604854368968 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017620692463041104,
"extra": "iterations: 159\ncpu: 0.001762096660377358 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005521087085499483,
"extra": "iterations: 51\ncpu: 0.00552101603921569 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00816952480989344,
"extra": "iterations: 34\ncpu: 0.008169646235294117 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010975901897136982,
"extra": "iterations: 26\ncpu: 0.010975202923076915 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002683703715984638,
"extra": "iterations: 104\ncpu: 0.0026836535384615443 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017542317509651186,
"extra": "iterations: 160\ncpu: 0.0017541199062500012 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003801432815757958,
"extra": "iterations: 74\ncpu: 0.0038012226081081124 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002832937722254281,
"extra": "iterations: 99\ncpu: 0.002832989343434348 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004687440395355225,
"extra": "iterations: 60\ncpu: 0.004686956283333332 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036005484752165964,
"extra": "iterations: 78\ncpu: 0.0036006026410256428 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005892495314280192,
"extra": "iterations: 48\ncpu: 0.005892383583333328 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0048633485004819675,
"extra": "iterations: 58\ncpu: 0.004863133120689668 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010286730949324791,
"extra": "iterations: 495\ncpu: 0.0004904721757575757 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016556897511084874,
"extra": "iterations: 384\ncpu: 0.0007610780026041676 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338345,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00023002465436670043,
"extra": "iterations: 1229\ncpu: 0.00023000948657445077 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020856480078396895,
"extra": "iterations: 1302\ncpu: 0.0002085674831029186 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026924976935753455,
"extra": "iterations: 104\ncpu: 0.002692446269230769 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017539173364639284,
"extra": "iterations: 160\ncpu: 0.0017539419562500004 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006007224955457323,
"extra": "iterations: 47\ncpu: 0.006006866787234044 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003726123174031576,
"extra": "iterations: 75\ncpu: 0.0037260017333333316 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026882451314192554,
"extra": "iterations: 104\ncpu: 0.0026882868846153864 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017585094619846945,
"extra": "iterations: 159\ncpu: 0.001758466805031446 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005457368551516066,
"extra": "iterations: 51\ncpu: 0.005457451000000001 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008132093092974494,
"extra": "iterations: 34\ncpu: 0.00813144761764706 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010921203173123874,
"extra": "iterations: 26\ncpu: 0.010919800423076918 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026673067183721633,
"extra": "iterations: 105\ncpu: 0.0026673471333333357 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001770273039612589,
"extra": "iterations: 158\ncpu: 0.0017702230886075972 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037613201141357424,
"extra": "iterations: 75\ncpu: 0.0037613767866666713 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002808995246887207,
"extra": "iterations: 100\ncpu: 0.0028090443199999984 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004668529828389486,
"extra": "iterations: 60\ncpu: 0.0046684240333333335 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035673183730885955,
"extra": "iterations: 79\ncpu: 0.0035672637848101238 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005826637148857117,
"extra": "iterations: 48\ncpu: 0.005826155895833333 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0048398026104631104,
"extra": "iterations: 58\ncpu: 0.004839879758620689 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010960760696158225,
"extra": "iterations: 543\ncpu: 0.0005277400313075505 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016346516147736582,
"extra": "iterations: 341\ncpu: 0.0007555413988269787 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.410754633374175e-05,
"extra": "iterations: 8203\ncpu: 3.4107944532488116e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.2167764823912425e-05,
"extra": "iterations: 6631\ncpu: 4.216715412456644e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005868997533949848,
"extra": "iterations: 478\ncpu: 0.0005869121548117154 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014424705013786395,
"extra": "iterations: 194\ncpu: 0.001442441005154639 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.440134059461619e-05,
"extra": "iterations: 4335\ncpu: 6.44194039215687e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0004940621594552672,
"extra": "iterations: 563\ncpu: 0.0004940493783303734 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0029936862248246387,
"extra": "iterations: 93\ncpu: 0.0029935816989247357 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.01827084431183e-05,
"extra": "iterations: 4468\ncpu: 6.019554521038498e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0004772003069136605,
"extra": "iterations: 583\ncpu: 0.000477131207547165 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.002931930124759674,
"extra": "iterations: 96\ncpu: 0.0029317209895833377 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002861998186572115,
"extra": "iterations: 973\ncpu: 0.0002861956824254881 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0012070598273441712,
"extra": "iterations: 232\ncpu: 0.0012070316594827598 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.005236436735908941,
"extra": "iterations: 53\ncpu: 0.005236506943396228 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010611836726848897,
"extra": "iterations: 26\ncpu: 0.010609882961538463 seconds\nthreads: 1"
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
"date": 1783589338330,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022654201163620246,
"extra": "iterations: 1220\ncpu: 0.00022653890819672134 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021169116584266107,
"extra": "iterations: 1312\ncpu: 0.0002116947004573171 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026887976206265963,
"extra": "iterations: 104\ncpu: 0.0026886968653846155 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017661595646339128,
"extra": "iterations: 158\ncpu: 0.0017659417594936716 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0060253701311476695,
"extra": "iterations: 47\ncpu: 0.006024915319148935 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004018759381943854,
"extra": "iterations: 69\ncpu: 0.0040186834782608705 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028265269115717723,
"extra": "iterations: 99\ncpu: 0.0028265761717171716 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017775544580423608,
"extra": "iterations: 159\ncpu: 0.0017774947484276743 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005433486058161809,
"extra": "iterations: 52\ncpu: 0.005433357749999996 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008106336874120377,
"extra": "iterations: 34\ncpu: 0.008106189647058828 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010823396536020132,
"extra": "iterations: 26\ncpu: 0.010823195884615394 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0029914424969599797,
"extra": "iterations: 104\ncpu: 0.002991373096153844 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001741379499435425,
"extra": "iterations: 160\ncpu: 0.001740556106250002 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037888239507805816,
"extra": "iterations: 73\ncpu: 0.0037886833424657577 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002827413082122803,
"extra": "iterations: 100\ncpu: 0.0028273666000000032 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0046400864919026695,
"extra": "iterations: 60\ncpu: 0.004639737366666675 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035369788544087474,
"extra": "iterations: 79\ncpu: 0.003537032075949362 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005812600255012512,
"extra": "iterations: 48\ncpu: 0.005812685354166662 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004753096628997286,
"extra": "iterations: 59\ncpu: 0.0047530251694915225 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011068177499366084,
"extra": "iterations: 518\ncpu: 0.000518934216216216 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016684343940333317,
"extra": "iterations: 342\ncpu: 0.0007903585175438574 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.267023540232159e-05,
"extra": "iterations: 8547\ncpu: 3.2669814788814796e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.187259860623851e-05,
"extra": "iterations: 6748\ncpu: 4.186919546532306e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005859890661060437,
"extra": "iterations: 479\ncpu: 0.000585980006263048 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014415340325267046,
"extra": "iterations: 194\ncpu: 0.0014415531649484533 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.511480568688227e-05,
"extra": "iterations: 4349\ncpu: 6.513898597378737e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0004938086084167373,
"extra": "iterations: 567\ncpu: 0.0004938363897707201 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.002983654699017925,
"extra": "iterations: 93\ncpu: 0.0029836436129032345 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.142284792373366e-05,
"extra": "iterations: 4445\ncpu: 6.143450663666907e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0004769732233206702,
"extra": "iterations: 587\ncpu: 0.00047688357240204275 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0029191870438425166,
"extra": "iterations: 95\ncpu: 0.002919202715789464 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002893920247438175,
"extra": "iterations: 961\ncpu: 0.0002893749229968784 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.001185150589118024,
"extra": "iterations: 237\ncpu: 0.0011849226877637144 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.0052252522221318005,
"extra": "iterations: 54\ncpu: 0.005225340388888888 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010625820893507738,
"extra": "iterations: 26\ncpu: 0.010623295307692307 seconds\nthreads: 1"
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
"date": 1783589338347,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002353241632599855,
"extra": "iterations: 1186\ncpu: 0.0002353280986509275 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020723781164954692,
"extra": "iterations: 1360\ncpu: 0.0002072407588235294 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026690409733698918,
"extra": "iterations: 104\ncpu: 0.002668772307692307 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017454355955123903,
"extra": "iterations: 160\ncpu: 0.0017454650187500003 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005869761109352112,
"extra": "iterations: 48\ncpu: 0.005869689583333337 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037170632680257167,
"extra": "iterations: 75\ncpu: 0.003716658973333331 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026788893200102313,
"extra": "iterations: 105\ncpu: 0.0026789116571428566 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017535627258490332,
"extra": "iterations: 161\ncpu: 0.001753528254658386 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005363056292900672,
"extra": "iterations: 52\ncpu: 0.005363094384615387 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.009303780163035675,
"extra": "iterations: 34\ncpu: 0.009303623823529407 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010997790556687575,
"extra": "iterations: 26\ncpu: 0.01099646330769232 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026568367367699034,
"extra": "iterations: 105\ncpu: 0.002656630495238097 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001742246747016907,
"extra": "iterations: 160\ncpu: 0.0017419261625000015 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003775090784639926,
"extra": "iterations: 74\ncpu: 0.003775145040540539 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028160894759977706,
"extra": "iterations: 99\ncpu: 0.002816022333333341 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004696059226989746,
"extra": "iterations: 60\ncpu: 0.004695463300000007 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003584050512933112,
"extra": "iterations: 77\ncpu: 0.003584113909090907 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005783607562383016,
"extra": "iterations: 48\ncpu: 0.005783530541666656 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004759730963871397,
"extra": "iterations: 58\ncpu: 0.004759255051724128 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0013589879043963778,
"extra": "iterations: 476\ncpu: 0.0005885993025210093 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0020596620036224464,
"extra": "iterations: 317\ncpu: 0.0009064451955835955 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.311136733264237e-05,
"extra": "iterations: 8423\ncpu: 3.311172432624956e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.167640393469195e-05,
"extra": "iterations: 6728\ncpu: 4.1676821492271105e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005861241448374474,
"extra": "iterations: 478\ncpu: 0.0005858732092050209 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014683663532996053,
"extra": "iterations: 191\ncpu: 0.0014683855340314133 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.590083438587329e-05,
"extra": "iterations: 4082\ncpu: 6.590322317491397e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0004945155167326911,
"extra": "iterations: 566\ncpu: 0.0004945354858657235 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.002983019707050729,
"extra": "iterations: 94\ncpu: 0.002983049372340435 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.282995300559266e-05,
"extra": "iterations: 4584\ncpu: 6.283942888307402e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.00047840193270009537,
"extra": "iterations: 586\ncpu: 0.00047835034812287357 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.002921337882677714,
"extra": "iterations: 96\ncpu: 0.0029213094479166664 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00028682118878602255,
"extra": "iterations: 983\ncpu: 0.00028682443540183116 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011767780079561122,
"extra": "iterations: 238\ncpu: 0.0011767915084033617 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.005231451105188441,
"extra": "iterations: 54\ncpu: 0.005231506148148145 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010628553537222056,
"extra": "iterations: 26\ncpu: 0.010625362230769231 seconds\nthreads: 1"
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
"date": 1783589338318,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002279112715985496,
"extra": "iterations: 974\ncpu: 0.00022790070944558524 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021018967731444585,
"extra": "iterations: 1343\ncpu: 0.00021017871854058081 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026846348660663493,
"extra": "iterations: 103\ncpu: 0.0026844370485436895 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017667933355403854,
"extra": "iterations: 158\ncpu: 0.0017667713924050636 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005916986059635244,
"extra": "iterations: 47\ncpu: 0.005916128978723403 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003717953364054362,
"extra": "iterations: 75\ncpu: 0.0037180072133333295 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026973669345562276,
"extra": "iterations: 104\ncpu: 0.0026972458076923085 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017478675962244192,
"extra": "iterations: 159\ncpu: 0.0017477516540880492 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005418690351339487,
"extra": "iterations: 52\ncpu: 0.00541877615384615 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008173690122716569,
"extra": "iterations: 34\ncpu: 0.008173363147058816 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010875408466045674,
"extra": "iterations: 26\ncpu: 0.010875573192307707 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026897421249976526,
"extra": "iterations: 104\ncpu: 0.002689782942307689 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018010139465332033,
"extra": "iterations: 155\ncpu: 0.0018010386516128997 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037735366821289064,
"extra": "iterations: 75\ncpu: 0.003773458919999999 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002825228854863331,
"extra": "iterations: 99\ncpu: 0.0028251728383838397 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004697652657826742,
"extra": "iterations: 60\ncpu: 0.004697735049999988 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035981229833654455,
"extra": "iterations: 74\ncpu: 0.003597676310810802 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0058694951077725025,
"extra": "iterations: 47\ncpu: 0.005868896595744669 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004859648901840736,
"extra": "iterations: 58\ncpu: 0.004859572172413786 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010843222566660014,
"extra": "iterations: 482\ncpu: 0.0005089470165975108 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016278801017633366,
"extra": "iterations: 358\ncpu: 0.000756083541899443 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.262282000421094e-05,
"extra": "iterations: 6548\ncpu: 3.2623367898595e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.168912559738819e-05,
"extra": "iterations: 6714\ncpu: 4.1689698838248446e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005831648906071982,
"extra": "iterations: 480\ncpu: 0.0005831609812499999 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014526561363456176,
"extra": "iterations: 194\ncpu: 0.0014526367783505149 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.447193587398177e-05,
"extra": "iterations: 4336\ncpu: 6.449615313653184e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0004948226305154655,
"extra": "iterations: 520\ncpu: 0.0004947872288461544 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0029842443363640903,
"extra": "iterations: 93\ncpu: 0.00298424889247311 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.028794064301554e-05,
"extra": "iterations: 4655\ncpu: 6.029376519871129e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.00047822225661504843,
"extra": "iterations: 588\ncpu: 0.0004781407091836656 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0029202202955881753,
"extra": "iterations: 96\ncpu: 0.002920108083333343 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002874913210756581,
"extra": "iterations: 977\ncpu: 0.0002874793469805529 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011756360029973906,
"extra": "iterations: 238\ncpu: 0.0011754397941176457 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.005207105919166848,
"extra": "iterations: 54\ncpu: 0.005207184833333333 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010628599386948805,
"extra": "iterations: 26\ncpu: 0.010627621423076921 seconds\nthreads: 1"
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
"date": 1783589338301,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002287206790256032,
"extra": "iterations: 1222\ncpu: 0.00022871750327332243 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021016428086641578,
"extra": "iterations: 1335\ncpu: 0.00021013008389513107 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002704418978644807,
"extra": "iterations: 103\ncpu: 0.0027044564854368943 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017567857911315148,
"extra": "iterations: 158\ncpu: 0.0017567669050632905 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005981024275434779,
"extra": "iterations: 47\ncpu: 0.005980620063829788 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037323347727457683,
"extra": "iterations: 75\ncpu: 0.0037323963333333317 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0027196476760419828,
"extra": "iterations: 103\ncpu: 0.002719534029126217 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0019042854067645495,
"extra": "iterations: 158\ncpu: 0.0019043145379746861 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005403761680309589,
"extra": "iterations: 52\ncpu: 0.005403662769230767 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008109239970936495,
"extra": "iterations: 34\ncpu: 0.008108693088235292 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011121575648968037,
"extra": "iterations: 26\ncpu: 0.011121237153846157 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002701410880455604,
"extra": "iterations: 104\ncpu: 0.002701067932692314 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017651312006344588,
"extra": "iterations: 159\ncpu: 0.0017648454968553445 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038123783999926427,
"extra": "iterations: 73\ncpu: 0.003812314410958906 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028308088129216976,
"extra": "iterations: 99\ncpu: 0.0028302393333333366 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004727808095641056,
"extra": "iterations: 59\ncpu: 0.004727549169491517 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00359807386026754,
"extra": "iterations: 77\ncpu: 0.003597776948051953 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005972195755351674,
"extra": "iterations: 44\ncpu: 0.005971815613636356 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0048687252505072234,
"extra": "iterations: 58\ncpu: 0.004868354224137932 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011045987789447492,
"extra": "iterations: 520\ncpu: 0.000519727846153846 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015687948828983053,
"extra": "iterations: 377\ncpu: 0.0007517592015915116 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.269000324993686e-05,
"extra": "iterations: 8568\ncpu: 3.2690396358543426e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.163309810320207e-05,
"extra": "iterations: 6749\ncpu: 4.1632523336790635e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005831406171009595,
"extra": "iterations: 481\ncpu: 0.0005831344594594593 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014370569248789366,
"extra": "iterations: 194\ncpu: 0.0014369754020618567 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.56656671237086e-05,
"extra": "iterations: 4327\ncpu: 6.568893459671764e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0004940679375554475,
"extra": "iterations: 568\ncpu: 0.0004940781531690153 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0029909839021398667,
"extra": "iterations: 94\ncpu: 0.0029909925638297937 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.015354513811838e-05,
"extra": "iterations: 4612\ncpu: 6.0167915437984885e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0004766932150133613,
"extra": "iterations: 588\ncpu: 0.00047665655612244193 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.002927881975968679,
"extra": "iterations: 96\ncpu: 0.0029279254583333323 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002914224138837269,
"extra": "iterations: 958\ncpu: 0.0002914114415448852 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0012433408689098199,
"extra": "iterations: 238\ncpu: 0.0012432649915966386 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.005276913912791127,
"extra": "iterations: 53\ncpu: 0.0052767111886792475 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010839178011967586,
"extra": "iterations: 26\ncpu: 0.010837381192307698 seconds\nthreads: 1"
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
"date": 1783589338335,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002264549221032575,
"extra": "iterations: 1222\ncpu: 0.0002264393739770868 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020954349960727194,
"extra": "iterations: 1335\ncpu: 0.0002095408734082398 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002691818218605191,
"extra": "iterations: 102\ncpu: 0.002691694950980393 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017633648038660207,
"extra": "iterations: 159\ncpu: 0.0017631067169811315 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0058924804131189985,
"extra": "iterations: 48\ncpu: 0.005892353749999996 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037447230021158853,
"extra": "iterations: 75\ncpu: 0.0037446446399999993 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0027219267452464383,
"extra": "iterations: 102\ncpu: 0.0027216734313725488 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017838554017862698,
"extra": "iterations: 157\ncpu: 0.0017838097770700638 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005522396050247492,
"extra": "iterations: 51\ncpu: 0.0055217260196078494 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00821220874786377,
"extra": "iterations: 34\ncpu: 0.008212297852941178 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01110860824584961,
"extra": "iterations: 25\ncpu: 0.011108722799999988 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002675746736072359,
"extra": "iterations: 105\ncpu: 0.002675547657142857 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001758447176293482,
"extra": "iterations: 158\ncpu: 0.0017583568354430413 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003790568661045384,
"extra": "iterations: 74\ncpu: 0.003789755013513511 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028052759170532227,
"extra": "iterations: 100\ncpu: 0.0028053030500000014 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004663785298665364,
"extra": "iterations: 60\ncpu: 0.004663437416666675 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003580654723734795,
"extra": "iterations: 79\ncpu: 0.0035802854050632966 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005971599132456679,
"extra": "iterations: 47\ncpu: 0.00597007678723403 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004833036455614814,
"extra": "iterations: 58\ncpu: 0.004832718310344838 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011149800959087556,
"extra": "iterations: 504\ncpu: 0.0005255384900793642 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016323959858548238,
"extra": "iterations: 383\ncpu: 0.0007748701644908609 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.279203528684917e-05,
"extra": "iterations: 7142\ncpu: 3.279254312517502e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.152248525952464e-05,
"extra": "iterations: 6731\ncpu: 4.152310934482246e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005882176213294455,
"extra": "iterations: 477\ncpu: 0.0005881853438155137 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014413963888109346,
"extra": "iterations: 194\ncpu: 0.001441309969072165 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.417258514936011e-05,
"extra": "iterations: 4362\ncpu: 6.419503668042168e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0004936083101890457,
"extra": "iterations: 568\ncpu: 0.0004936550035211271 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0029953008002423227,
"extra": "iterations: 94\ncpu: 0.002995216755319142 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.403173235794805e-05,
"extra": "iterations: 4369\ncpu: 6.403793064774518e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0004763237662372232,
"extra": "iterations: 587\ncpu: 0.0004762763611584246 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.002946945528189341,
"extra": "iterations: 96\ncpu: 0.002946975218750013 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00030981943026598535,
"extra": "iterations: 973\ncpu: 0.00030980664131551944 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011869643810932143,
"extra": "iterations: 237\ncpu: 0.0011869812995780591 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.005222117459332502,
"extra": "iterations: 54\ncpu: 0.005222044037037032 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010675466977632962,
"extra": "iterations: 26\ncpu: 0.010675574 seconds\nthreads: 1"
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
"date": 1783589338286,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00023382385116552175,
"extra": "iterations: 1222\ncpu: 0.00023380627577741407 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020992889832914546,
"extra": "iterations: 1335\ncpu: 0.00020993121123595509 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026729863423567554,
"extra": "iterations: 104\ncpu: 0.0026729239903846166 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017745343944694425,
"extra": "iterations: 158\ncpu: 0.00177455267721519 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00595522941427028,
"extra": "iterations: 47\ncpu: 0.005955289085106383 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0039008480228789866,
"extra": "iterations: 73\ncpu: 0.0039007478767123277 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026932977713071383,
"extra": "iterations: 104\ncpu: 0.0026933249423076906 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017787286430407483,
"extra": "iterations: 157\ncpu: 0.0017786637515923552 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00578706959883372,
"extra": "iterations: 48\ncpu: 0.005787142291666669 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00827499698190128,
"extra": "iterations: 34\ncpu: 0.008274605558823528 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011238803863525392,
"extra": "iterations: 25\ncpu: 0.011237135079999983 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002853236152130423,
"extra": "iterations: 103\ncpu: 0.0028532732038835024 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017719906606491964,
"extra": "iterations: 157\ncpu: 0.001771949178343951 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003780449906440631,
"extra": "iterations: 73\ncpu: 0.0037799218082191733 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028456497192382814,
"extra": "iterations: 100\ncpu: 0.002845680669999995 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004665923118591309,
"extra": "iterations: 60\ncpu: 0.0046658301999999985 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035859285256801513,
"extra": "iterations: 78\ncpu: 0.0035853728205128216 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005881304542223613,
"extra": "iterations: 48\ncpu: 0.005881378062499998 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004830779700443663,
"extra": "iterations: 58\ncpu: 0.0048308347758620656 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010899550701625133,
"extra": "iterations: 481\ncpu: 0.0005127805530145528 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0017143148894703717,
"extra": "iterations: 327\ncpu: 0.0008153284373088699 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.337815501513718e-05,
"extra": "iterations: 8552\ncpu: 3.337798748830683e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.1911280847456245e-05,
"extra": "iterations: 6733\ncpu: 4.1910899153423446e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005835869615909203,
"extra": "iterations: 479\ncpu: 0.000583592755741127 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014367947211632363,
"extra": "iterations: 195\ncpu: 0.0014368139435897438 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.589336502601618e-05,
"extra": "iterations: 4087\ncpu: 6.591005456324913e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0004944936364097933,
"extra": "iterations: 565\ncpu: 0.0004945160247787611 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0029892972720566616,
"extra": "iterations: 93\ncpu: 0.0029891853870967658 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.078391498387199e-05,
"extra": "iterations: 4574\ncpu: 6.080151792741608e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0004770897925407761,
"extra": "iterations: 587\ncpu: 0.0004770066098807558 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0029256045818328857,
"extra": "iterations: 96\ncpu: 0.0029254868333333287 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002880130881987612,
"extra": "iterations: 969\ncpu: 0.0002879946480908154 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011751279549256659,
"extra": "iterations: 237\ncpu: 0.0011751435654008438 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.005234872853314435,
"extra": "iterations: 54\ncpu: 0.005234946814814816 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.01063382625579834,
"extra": "iterations: 26\ncpu: 0.010633904692307699 seconds\nthreads: 1"
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
"date": 1783589338350,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022766049754609918,
"extra": "iterations: 1231\ncpu: 0.00022766406173842406 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020910633934868706,
"extra": "iterations: 1332\ncpu: 0.00020910955780780783 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026856913016392635,
"extra": "iterations: 104\ncpu: 0.0026855890769230772 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017603268413423742,
"extra": "iterations: 159\ncpu: 0.001760103056603774 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0059262285841272236,
"extra": "iterations: 47\ncpu: 0.005926080553191487 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037676836993243245,
"extra": "iterations: 74\ncpu: 0.0037675607297297307 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002686984263933622,
"extra": "iterations: 104\ncpu: 0.0026867968942307662 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001769437065607385,
"extra": "iterations: 158\ncpu: 0.0017694010506329111 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005428231679476225,
"extra": "iterations: 52\ncpu: 0.005427334192307692 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008258272619808422,
"extra": "iterations: 34\ncpu: 0.008258362588235287 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011190023422241211,
"extra": "iterations: 25\ncpu: 0.011190160280000007 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026798248291015625,
"extra": "iterations: 104\ncpu: 0.0026798603942307713 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001761734485626221,
"extra": "iterations: 160\ncpu: 0.001761707625000003 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037792695535195844,
"extra": "iterations: 74\ncpu: 0.003779318108108102 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002794017791748047,
"extra": "iterations: 100\ncpu: 0.002794050559999999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0046522219975789385,
"extra": "iterations: 60\ncpu: 0.004652127149999992 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003573646912207971,
"extra": "iterations: 78\ncpu: 0.003573693205128203 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005910817612992956,
"extra": "iterations: 47\ncpu: 0.005910889659574461 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004917688537062261,
"extra": "iterations: 57\ncpu: 0.004917492456140343 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0013659635840702667,
"extra": "iterations: 469\ncpu: 0.000605023121535181 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0021142514446113683,
"extra": "iterations: 316\ncpu: 0.0008923830094936732 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.271630568849643e-05,
"extra": "iterations: 8603\ncpu: 3.271556596536092e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.186379543017194e-05,
"extra": "iterations: 6733\ncpu: 4.1861985593346206e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005947574656060402,
"extra": "iterations: 470\ncpu: 0.0005947651212765957 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014409199739113833,
"extra": "iterations: 195\ncpu: 0.0014409412871794877 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.659456453579854e-05,
"extra": "iterations: 4090\ncpu: 6.661577359413199e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.000494274991982396,
"extra": "iterations: 566\ncpu: 0.0004943318833922265 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0029894341813757066,
"extra": "iterations: 94\ncpu: 0.002989095382978731 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.029242820172841e-05,
"extra": "iterations: 4643\ncpu: 6.0297363988799885e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.00047832006490210253,
"extra": "iterations: 587\ncpu: 0.00047829945315161824 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0029262006282806396,
"extra": "iterations: 96\ncpu: 0.0029235542291666645 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00029035913225518946,
"extra": "iterations: 962\ncpu: 0.000290319362785863 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011876366906246898,
"extra": "iterations: 236\ncpu: 0.0011876188305084752 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.00524441095498892,
"extra": "iterations: 52\ncpu: 0.0052432709038461485 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010612258544334998,
"extra": "iterations: 26\ncpu: 0.010612060884615391 seconds\nthreads: 1"
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
"date": 1783589338305,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00023188973109988816,
"extra": "iterations: 1198\ncpu: 0.00023188101752921533 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00022141453581264395,
"extra": "iterations: 1262\ncpu: 0.00022141717908082416 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002682559765302218,
"extra": "iterations: 104\ncpu: 0.0026825946346153845 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001783744544739936,
"extra": "iterations: 157\ncpu: 0.0017836355031847142 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005980895913165549,
"extra": "iterations: 46\ncpu: 0.005980972652173914 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003865921334044574,
"extra": "iterations: 73\ncpu: 0.003865734616438358 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0027063007538135233,
"extra": "iterations: 104\ncpu: 0.0027062403653846166 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017483924169960263,
"extra": "iterations: 159\ncpu: 0.0017484144716981133 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00535433108990009,
"extra": "iterations: 52\ncpu: 0.0053541796923076874 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008202749140122359,
"extra": "iterations: 34\ncpu: 0.00820285820588235 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010862139555124136,
"extra": "iterations: 26\ncpu: 0.010861433384615385 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.00265909140964724,
"extra": "iterations: 106\ncpu: 0.002659122528301887 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001761793340527037,
"extra": "iterations: 159\ncpu: 0.0017617509937106913 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0037774878579217037,
"extra": "iterations: 74\ncpu: 0.003777276040540543 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028473333436615615,
"extra": "iterations: 98\ncpu: 0.00284729045918367 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004738452070850437,
"extra": "iterations: 59\ncpu: 0.004738370033898306 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0036841522563587537,
"extra": "iterations: 77\ncpu: 0.0036840205194805164 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005965983614008477,
"extra": "iterations: 47\ncpu: 0.005965876489361699 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004923067594829358,
"extra": "iterations: 57\ncpu: 0.004922957736842105 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0013838320842628317,
"extra": "iterations: 466\ncpu: 0.000605079133047211 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0017074623415547032,
"extra": "iterations: 310\ncpu: 0.0007842632419354833 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.2950990424319044e-05,
"extra": "iterations: 8345\ncpu: 3.292609778310366e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.1460585611798724e-05,
"extra": "iterations: 6765\ncpu: 4.1461188470066514e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005792537011390876,
"extra": "iterations: 484\ncpu: 0.0005792613037190082 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014073609107702822,
"extra": "iterations: 199\ncpu: 0.0014062929597989952 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.451822693378781e-05,
"extra": "iterations: 4328\ncpu: 6.453909981515714e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0004946986127154258,
"extra": "iterations: 562\ncpu: 0.0004941493078291812 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.002986179885043893,
"extra": "iterations: 93\ncpu: 0.0029858909247311802 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.029565281406659e-05,
"extra": "iterations: 4641\ncpu: 6.0313655031242284e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.00047787601112300515,
"extra": "iterations: 585\ncpu: 0.0004774874615384623 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.002924786115947523,
"extra": "iterations: 95\ncpu: 0.002924152831578947 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00028662745027727577,
"extra": "iterations: 977\ncpu: 0.00028659533879222106 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.001385046177155982,
"extra": "iterations: 233\ncpu: 0.0013840826866952797 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.005262019499292914,
"extra": "iterations: 53\ncpu: 0.0052614580188679256 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010611277360182542,
"extra": "iterations: 26\ncpu: 0.010610680230769226 seconds\nthreads: 1"
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
"date": 1783589338296,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002318198398008185,
"extra": "iterations: 1180\ncpu: 0.00023182277627118645 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00022195892333984376,
"extra": "iterations: 1250\ncpu: 0.00022196168240000002 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026682717459542414,
"extra": "iterations: 105\ncpu: 0.00266821960952381 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0017684894272043737,
"extra": "iterations: 158\ncpu: 0.0017685105822784811 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005929566444234645,
"extra": "iterations: 47\ncpu: 0.005929244978723405 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037916157696698165,
"extra": "iterations: 74\ncpu: 0.0037916600405405398 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002688336830872756,
"extra": "iterations: 104\ncpu: 0.002688376278846157 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017699048488954957,
"extra": "iterations: 158\ncpu: 0.001769843069620253 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00536486735710731,
"extra": "iterations: 52\ncpu: 0.0053647621153846145 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008179299971636604,
"extra": "iterations: 34\ncpu: 0.008179087382352939 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011339216232299805,
"extra": "iterations: 25\ncpu: 0.011339366599999998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002665324438185919,
"extra": "iterations: 105\ncpu: 0.002665355514285716 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017672874642618048,
"extra": "iterations: 159\ncpu: 0.0017672568553459165 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003745610649521287,
"extra": "iterations: 74\ncpu: 0.0037454057837837834 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002825968193285393,
"extra": "iterations: 99\ncpu: 0.00282600824242424 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004717619692692992,
"extra": "iterations: 61\ncpu: 0.004714041016393446 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035726871245946638,
"extra": "iterations: 78\ncpu: 0.003572730897435891 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005797887335018237,
"extra": "iterations: 49\ncpu: 0.005797826142857147 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004896598949766995,
"extra": "iterations: 57\ncpu: 0.004896534947368418 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010818823668402185,
"extra": "iterations: 503\ncpu: 0.0005058643578528828 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016113964177794375,
"extra": "iterations: 354\ncpu: 0.0007555835536723144 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.3387992802250896e-05,
"extra": "iterations: 8352\ncpu: 3.337087595785441e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.1882185191695065e-05,
"extra": "iterations: 6695\ncpu: 4.187664301717699e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005821741468184222,
"extra": "iterations: 482\ncpu: 0.0005821542780082985 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.00142226506717241,
"extra": "iterations: 199\ncpu: 0.0014222842010050256 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.490460965787848e-05,
"extra": "iterations: 4106\ncpu: 6.492736166585511e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0004953177629318913,
"extra": "iterations: 565\ncpu: 0.0004953333628318606 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.002986989122755984,
"extra": "iterations: 94\ncpu: 0.00298679845744681 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.0262890741993965e-05,
"extra": "iterations: 4617\ncpu: 6.0276732943470704e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.00047769797902528935,
"extra": "iterations: 588\ncpu: 0.00047762906802720543 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0029459968209266663,
"extra": "iterations: 96\ncpu: 0.0029458095312500034 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00028846427386468364,
"extra": "iterations: 977\ncpu: 0.00028846749641760487 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0012809819534045307,
"extra": "iterations: 238\ncpu: 0.0012809308445378143 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.005268078929973098,
"extra": "iterations: 53\ncpu: 0.005268155792452826 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010673743027907152,
"extra": "iterations: 26\ncpu: 0.010672846192307702 seconds\nthreads: 1"
}
]
},
{
"commit": {
"author": {
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783589338340,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00023305713140291234,
"extra": "iterations: 1204\ncpu: 0.00023306001827242532 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00022398105180595977,
"extra": "iterations: 1255\ncpu: 0.00022397099442231078 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026710555666968937,
"extra": "iterations: 105\ncpu: 0.0026708944190476196 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001771725948501683,
"extra": "iterations: 159\ncpu: 0.0017716885786163525 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.005946788382022939,
"extra": "iterations: 47\ncpu: 0.00594665019148936 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0037770400176177156,
"extra": "iterations: 74\ncpu: 0.003776665445945943 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002703145869727274,
"extra": "iterations: 103\ncpu: 0.0027030213689320385 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0017983684172997107,
"extra": "iterations: 156\ncpu: 0.0017982488846153845 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005419773214003619,
"extra": "iterations: 51\ncpu: 0.005419849490196083 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008171207764569451,
"extra": "iterations: 34\ncpu: 0.00817094544117647 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011060791015625,
"extra": "iterations: 25\ncpu: 0.011060122119999996 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026694910866873604,
"extra": "iterations: 105\ncpu: 0.0026693598476190536 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0017673789330248564,
"extra": "iterations: 159\ncpu: 0.001767200503144655 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003765612035184293,
"extra": "iterations: 74\ncpu: 0.003765541189189193 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028395555457290337,
"extra": "iterations: 98\ncpu: 0.0028395039897959238 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00461543583479084,
"extra": "iterations: 61\ncpu: 0.004614752016393437 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003602049289605557,
"extra": "iterations: 78\ncpu: 0.0036020925897435963 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005869013197878575,
"extra": "iterations: 47\ncpu: 0.005868842957446799 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004872221695749384,
"extra": "iterations: 57\ncpu: 0.004871855736842104 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010939271306892853,
"extra": "iterations: 483\ncpu: 0.0005185679006211184 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015906095504760742,
"extra": "iterations: 366\ncpu: 0.0007214126803278686 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.3077445209864054e-05,
"extra": "iterations: 8467\ncpu: 3.3064927719381135e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.199299524844338e-05,
"extra": "iterations: 6751\ncpu: 4.1988775144423046e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0005988901866270281,
"extra": "iterations: 469\ncpu: 0.0005988983859275053 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0014037692546844483,
"extra": "iterations: 200\ncpu: 0.0014032865100000015 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 6.508692872693643e-05,
"extra": "iterations: 4082\ncpu: 6.510858035276822e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.000494673399798638,
"extra": "iterations: 565\ncpu: 0.0004947112407079655 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0029967363844526575,
"extra": "iterations: 94\ncpu: 0.002996764680851056 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.0297407222741214e-05,
"extra": "iterations: 4632\ncpu: 6.030967918825322e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0004775662471003093,
"extra": "iterations: 586\ncpu: 0.0004775277901023963 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.003034206728140513,
"extra": "iterations: 96\ncpu: 0.0030341521770833398 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002850334132554451,
"extra": "iterations: 983\ncpu: 0.0002850154557477116 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011897397642376044,
"extra": "iterations: 238\ncpu: 0.0011897119495798323 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.005340400731788491,
"extra": "iterations: 53\ncpu: 0.00534024309433962 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.010568820513211764,
"extra": "iterations: 26\ncpu: 0.010568953961538462 seconds\nthreads: 1"
}
]
}
]
}
}
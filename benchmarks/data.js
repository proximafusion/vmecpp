window.BENCHMARK_DATA = {
"lastUpdate": 1783553392960,
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
"date": 1783553392662,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5955334559999983,
"range": "stddev: 0.002467778352951861",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5955570434000265,
"range": "stddev: 0.0032973091183379232",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4053806199999979,
"range": "stddev: 0.005165429287339281",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.783518106000012,
"range": "stddev: 0.002462073172402523",
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
"date": 1783553392392,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6007059292000122,
"range": "stddev: 0.009144849579463505",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5980248600000095,
"range": "stddev: 0.006383969197309972",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4114896173333211,
"range": "stddev: 0.003583441068508177",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.82013960200004,
"range": "stddev: 0.012015044180420863",
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
"date": 1783553392669,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5926399901999957,
"range": "stddev: 0.004543472198358731",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5925204928000539,
"range": "stddev: 0.002724335324763933",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4204178976666906,
"range": "stddev: 0.005752764639076997",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7776624603333175,
"range": "stddev: 0.009953899055891568",
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
"date": 1783553392251,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5965447476000009,
"range": "stddev: 0.005049007569484721",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.59536140419998,
"range": "stddev: 0.004113720618664629",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4101687416665907,
"range": "stddev: 0.005718519444889695",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.779811729666676,
"range": "stddev: 0.012349684359496068",
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
"date": 1783553392423,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.588842707400022,
"range": "stddev: 0.005802047135744484",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5946878736000144,
"range": "stddev: 0.013634541532950495",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4080970723333242,
"range": "stddev: 0.0028195975866397295",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7909375229999873,
"range": "stddev: 0.007535185780691194",
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
"date": 1783553392792,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5983088852000492,
"range": "stddev: 0.002596333608202581",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5983222704000127,
"range": "stddev: 0.006501082481759171",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4277299069999572,
"range": "stddev: 0.001503280923072499",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8188314893333577,
"range": "stddev: 0.035056600353640686",
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
"date": 1783553392486,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6021832785999777,
"range": "stddev: 0.012535810565921872",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.59118772479992,
"range": "stddev: 0.004843003765133755",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4097448500000003,
"range": "stddev: 0.005205554453065162",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7786706029999855,
"range": "stddev: 0.011211671058326184",
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
"date": 1783553392676,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5982431749999705,
"range": "stddev: 0.005424548924394863",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5938169256000038,
"range": "stddev: 0.00689517714518735",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4164605876667338,
"range": "stddev: 0.010894604704326332",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7694318460000127,
"range": "stddev: 0.024018005228365058",
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
"date": 1783553392527,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6007614940000167,
"range": "stddev: 0.002842165768005349",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5895349656000235,
"range": "stddev: 0.0014008251210604619",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4256151756665834,
"range": "stddev: 0.006029886316513779",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.781423382666617,
"range": "stddev: 0.019023290723303478",
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
"date": 1783553392399,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6038574965999942,
"range": "stddev: 0.0110722618230169",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5989751913999953,
"range": "stddev: 0.005861357923181966",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4271016130000287,
"range": "stddev: 0.003713094054633503",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7958718029999545,
"range": "stddev: 0.02605361842489053",
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
"date": 1783553392402,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30292477580001104,
"range": "stddev: 0.0036152576334740993",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30511387299998205,
"range": "stddev: 0.004031890199339803",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4112488666666347,
"range": "stddev: 0.0008234476906386802",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.790948099333491,
"range": "stddev: 0.015780333474129906",
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
"date": 1783553392365,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3041239432000111,
"range": "stddev: 0.0011625870068675554",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31454628940005025,
"range": "stddev: 0.022939767805111425",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4118174819999847,
"range": "stddev: 0.00233326418819139",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7820249493333145,
"range": "stddev: 0.0174665691732219",
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
"date": 1783553392138,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3014299309999842,
"range": "stddev: 0.002127041358881683",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3011680143999911,
"range": "stddev: 0.002432915549523888",
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
"date": 1783553392586,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30330037280004946,
"range": "stddev: 0.001768394877149699",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30389838440005407,
"range": "stddev: 0.0029969804433816346",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4248746516667172,
"range": "stddev: 0.006832597043022172",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.780428430333435,
"range": "stddev: 0.018281788372319953",
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
"date": 1783553392166,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30413395320001657,
"range": "stddev: 0.0036211467693812453",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31384609999995516,
"range": "stddev: 0.027234562680511245",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4107378746667412,
"range": "stddev: 0.004501244825501964",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7754596973333414,
"range": "stddev: 0.0031393287638620637",
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
"date": 1783553392161,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30144259079997937,
"range": "stddev: 0.0011848324676030273",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3008272598000076,
"range": "stddev: 0.002239343674928376",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4098701386665955,
"range": "stddev: 0.0046174846495111764",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.79082850800008,
"range": "stddev: 0.021907479208351704",
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
"date": 1783553392303,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30533205919996365,
"range": "stddev: 0.002756904870015573",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30755019240000364,
"range": "stddev: 0.008666757143510045",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4070941073333263,
"range": "stddev: 0.00451695585840514",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.827658884333308,
"range": "stddev: 0.022695029627978643",
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
"date": 1783553392382,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.303621591800038,
"range": "stddev: 0.0030525164552613502",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30345843059994876,
"range": "stddev: 0.0020662245922723516",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4232451003333608,
"range": "stddev: 0.0030652494586757267",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8027982730000836,
"range": "stddev: 0.04661990145473989",
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
"date": 1783553392214,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30392626000002565,
"range": "stddev: 0.003354397171054436",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3051453003999541,
"range": "stddev: 0.00435598839083492",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4054621343333717,
"range": "stddev: 0.002021286525820771",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.807107321666687,
"range": "stddev: 0.008260439017407197",
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
"date": 1783553392737,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30564653460000957,
"range": "stddev: 0.0025779635869468917",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3031244698000137,
"range": "stddev: 0.0016335084739382106",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4080674719999327,
"range": "stddev: 0.0005712208609134947",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7930663223332886,
"range": "stddev: 0.027137784685717297",
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
"date": 1783553392642,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3019419734000621,
"range": "stddev: 0.003912896468224348",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3001450316000046,
"range": "stddev: 0.0021781303339447095",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4210914933332788,
"range": "stddev: 0.003518820248744692",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7967752116664997,
"range": "stddev: 0.01430651316009048",
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
"date": 1783553392647,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3024746435999077,
"range": "stddev: 0.0021740019962537293",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30349245339994013,
"range": "stddev: 0.001585663697942837",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4067745726667151,
"range": "stddev: 0.001496000776228071",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7897172509998804,
"range": "stddev: 0.010105037764363401",
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
"date": 1783553392414,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3045258233999448,
"range": "stddev: 0.0027702763196643266",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30341997819996325,
"range": "stddev: 0.0015451975579304738",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4117206986669164,
"range": "stddev: 0.0058559515919392215",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.80251430499978,
"range": "stddev: 0.0410673449781537",
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
"date": 1783553392281,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30306582040002467,
"range": "stddev: 0.0022829474706318837",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30132201839987827,
"range": "stddev: 0.002469542608356663",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4163464240000394,
"range": "stddev: 0.002184220518327241",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.799874423333222,
"range": "stddev: 0.03223157329144088",
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
"date": 1783553392523,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30957074560000136,
"range": "stddev: 0.002351719498171554",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30919456020001235,
"range": "stddev: 0.0014039094074226866",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4206964029999842,
"range": "stddev: 0.004337194577164319",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8108909210000284,
"range": "stddev: 0.018189914572851566",
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
"date": 1783553392452,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3135177669999962,
"range": "stddev: 0.002875618353574792",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3177589174000218,
"range": "stddev: 0.0024079666210274028",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4268185070000072,
"range": "stddev: 0.018523272154411456",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8257618209999955,
"range": "stddev: 0.022252763178965747",
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
"date": 1783553392725,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3116852260000087,
"range": "stddev: 0.00263302593256868",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3118600411999978,
"range": "stddev: 0.004741995716308351",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4147193716666493,
"range": "stddev: 0.0022940316625410617",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7849371010000064,
"range": "stddev: 0.004457815314598277",
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
"date": 1783553392634,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31835106099997573,
"range": "stddev: 0.001954714197454268",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3084868806000259,
"range": "stddev: 0.001935182400439922",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4136379310000013,
"range": "stddev: 0.005544654367020443",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8085508579999896,
"range": "stddev: 0.027951533234740575",
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
"date": 1783553392175,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3131296989999782,
"range": "stddev: 0.0035778754193912226",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31357387340001425,
"range": "stddev: 0.001892217512307397",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4210373093333146,
"range": "stddev: 0.014102113095779259",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.785449767000008,
"range": "stddev: 0.014173720700725062",
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
"date": 1783553392518,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30940187020003124,
"range": "stddev: 0.0022995911939350418",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31211833659995136,
"range": "stddev: 0.0030250628888128595",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4139110476666776,
"range": "stddev: 0.0042436774755258",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.806815916000005,
"range": "stddev: 0.041658818657097176",
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
"date": 1783553392191,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31306771180002213,
"range": "stddev: 0.002065587515846511",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3115439870000046,
"range": "stddev: 0.003049672053255257",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4122335476667256,
"range": "stddev: 0.005072168598388703",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7868846146666706,
"range": "stddev: 0.014851613025513246",
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
"date": 1783553392768,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31040201900000286,
"range": "stddev: 0.004431741051091089",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3089965886000073,
"range": "stddev: 0.0020205740610435203",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4127005493333324,
"range": "stddev: 0.0014776804544821007",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.796968252666602,
"range": "stddev: 0.03574836562747892",
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
"date": 1783553392833,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31733263320004423,
"range": "stddev: 0.0029683949981463053",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3100872930000378,
"range": "stddev: 0.002553897296399757",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4132420126666527,
"range": "stddev: 0.0030907484800829802",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.789983141333399,
"range": "stddev: 0.003915972434821837",
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
"date": 1783553392212,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3086250467999889,
"range": "stddev: 0.0014005668933848013",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31050505220002833,
"range": "stddev: 0.0027924256290647667",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.416571855666651,
"range": "stddev: 0.001076678606501022",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7964049186666293,
"range": "stddev: 0.014861636034666089",
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
"date": 1783553392394,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3155435748000855,
"range": "stddev: 0.006061254667811491",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31512143420000027,
"range": "stddev: 0.0020449873496685787",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4136353996667215,
"range": "stddev: 0.0011732406513050358",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8034964786666023,
"range": "stddev: 0.0181829836035499",
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
"date": 1783553392131,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3240574795999237,
"range": "stddev: 0.004302005376685652",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3251825032000397,
"range": "stddev: 0.0035898964312495963",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4149587390000609,
"range": "stddev: 0.0006259972830368471",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.808909495999918,
"range": "stddev: 0.004545219222611179",
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
"date": 1783553392503,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3207096752000325,
"range": "stddev: 0.007109844090820602",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31869933059992944,
"range": "stddev: 0.0017285807831857415",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4169169319999735,
"range": "stddev: 0.002740468838479243",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.818809866666546,
"range": "stddev: 0.023530911369955793",
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
"date": 1783553392327,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3187296027999764,
"range": "stddev: 0.008478740534512488",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.314141985400056,
"range": "stddev: 0.004631325045777997",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4201679823333204,
"range": "stddev: 0.008241354678556587",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8212845836667384,
"range": "stddev: 0.026698072858920226",
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
"date": 1783553392329,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31214600180001073,
"range": "stddev: 0.0022772201455790596",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3117530077999618,
"range": "stddev: 0.0016966043660479268",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4218084066666659,
"range": "stddev: 0.009514425473718979",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8326549153332508,
"range": "stddev: 0.043626982465932135",
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
"date": 1783553392739,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3124403407999125,
"range": "stddev: 0.0011514100410450433",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3142977692000386,
"range": "stddev: 0.00475928158297283",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4229058556666132,
"range": "stddev: 0.0038563537112291826",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8291175373333317,
"range": "stddev: 0.02756440949369795",
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
"date": 1783553392814,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.32274715299995477,
"range": "stddev: 0.009762475908691576",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.32020782539993886,
"range": "stddev: 0.016035483794404913",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4163558426666896,
"range": "stddev: 0.002189216127159589",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.805319756333271,
"range": "stddev: 0.013850839525410892",
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
"date": 1783553392780,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3142687816000489,
"range": "stddev: 0.0014204783363604937",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31025208099999874,
"range": "stddev: 0.0011844842626141663",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4281336899999435,
"range": "stddev: 0.018552901647356755",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8135770330000014,
"range": "stddev: 0.006562434420406616",
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
"date": 1783553392651,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31396328719997657,
"range": "stddev: 0.002830863052483963",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3161311352000212,
"range": "stddev: 0.0026132316742725815",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4172329630000604,
"range": "stddev: 0.0032468784867730562",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.813721443666585,
"range": "stddev: 0.018670246623320834",
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
"date": 1783553392474,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31212573060001886,
"range": "stddev: 0.0016383349195354376",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3112414649999664,
"range": "stddev: 0.001948197563709443",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4133201250000031,
"range": "stddev: 0.003235243992779283",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8165709260000162,
"range": "stddev: 0.043203917965136264",
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
"date": 1783553392247,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3139220248000584,
"range": "stddev: 0.0030650438037475945",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3113018661999831,
"range": "stddev: 0.0014295404584940895",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.417236043999992,
"range": "stddev: 0.003529741693383327",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8187225289999938,
"range": "stddev: 0.015508308371235652",
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
"date": 1783553392501,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3124392043999705,
"range": "stddev: 0.002671115044543108",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.313519412200003,
"range": "stddev: 0.004045582209761769",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4137412623332086,
"range": "stddev: 0.0020715971500834007",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8017801546666306,
"range": "stddev: 0.021150634184758297",
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
"date": 1783553392821,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31385309700003744,
"range": "stddev: 0.004946874041659056",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3128785124000387,
"range": "stddev: 0.0028969814755306746",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4148486306665593,
"range": "stddev: 0.0031914711669422703",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.786517218999961,
"range": "stddev: 0.031400497758666136",
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
"date": 1783553392233,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31385361019997615,
"range": "stddev: 0.0048726299335077414",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31202798740000615,
"range": "stddev: 0.0014988206630703765",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.414741318000021,
"range": "stddev: 0.001299464378625832",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.784630433666507,
"range": "stddev: 0.0210322222052363",
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
"date": 1783553392644,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31289044439999997,
"range": "stddev: 0.0025488500856028864",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31222466119988895,
"range": "stddev: 0.003456811087534411",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4260207140000603,
"range": "stddev: 0.01498067416267911",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.783066622333384,
"range": "stddev: 0.008622231342087516",
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
"date": 1783553392573,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31070094339993376,
"range": "stddev: 0.001895937168579496",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30942705160005063,
"range": "stddev: 0.0010349305135758423",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.411502003333074,
"range": "stddev: 0.0001426157839573255",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8043049516666847,
"range": "stddev: 0.015512110904601301",
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
"date": 1783553392680,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3119962142000986,
"range": "stddev: 0.0017483975377427911",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3131781454000702,
"range": "stddev: 0.0015076581896586807",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4150065006663983,
"range": "stddev: 0.0006958477852404191",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.799803754333425,
"range": "stddev: 0.012429272253083",
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
"date": 1783553392279,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3109327469999698,
"range": "stddev: 0.001220013381256904",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31508426219998,
"range": "stddev: 0.008007626412746289",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.412906670333238,
"range": "stddev: 0.0021445593405771536",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7920518066666773,
"range": "stddev: 0.029997498542695816",
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
"date": 1783553392315,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.301418766200004,
"range": "stddev: 0.0017570386658535097",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30056372799999165,
"range": "stddev: 0.001627531463252853",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4113291050000025,
"range": "stddev: 0.010278912442005789",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.796532150666659,
"range": "stddev: 0.02057303543484771",
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
"date": 1783553392718,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3091485739999996,
"range": "stddev: 0.013305152422223071",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30897537920000673,
"range": "stddev: 0.0011973726286027087",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.429734131000032,
"range": "stddev: 0.01797873270625023",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8050967213333515,
"range": "stddev: 0.07543632929000049",
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
"date": 1783553392231,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3076484810000011,
"range": "stddev: 0.008607688758824276",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3056659939999918,
"range": "stddev: 0.0032391661804433247",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4084145666666548,
"range": "stddev: 0.0022195616538553163",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7974350593333193,
"range": "stddev: 0.017329011643925113",
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
"date": 1783553392119,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30128672439999493,
"range": "stddev: 0.0011299439715777438",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3024200405999977,
"range": "stddev: 0.0018353923088097659",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4253425026666566,
"range": "stddev: 0.002686782692919549",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7581493830000077,
"range": "stddev: 0.021377570373819644",
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
"date": 1783553392385,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3050858940000012,
"range": "stddev: 0.00304715562253165",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30256876420000933,
"range": "stddev: 0.0009339228572794982",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4157163676666566,
"range": "stddev: 0.0012085065157768787",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.796248235666667,
"range": "stddev: 0.01691201692456241",
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
"date": 1783553392706,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3113339953999912,
"range": "stddev: 0.0017665429049980687",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3088420853999992,
"range": "stddev: 0.0017620754432307718",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.414480744333294,
"range": "stddev: 0.0028185796534732246",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.805290853000012,
"range": "stddev: 0.0014156937982440735",
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
"date": 1783553392170,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30915228439998826,
"range": "stddev: 0.0031405105404361815",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31142277460000967,
"range": "stddev: 0.0024006901072496376",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4120056210000105,
"range": "stddev: 0.00240199775535226",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.791025715999998,
"range": "stddev: 0.01860089262294506",
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
"date": 1783553392152,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3147863147999942,
"range": "stddev: 0.0016001901779372696",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3136292785999785,
"range": "stddev: 0.0024806102001013867",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4129050179999467,
"range": "stddev: 0.0018616946957797564",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.78932066666664,
"range": "stddev: 0.009604555392057092",
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
"date": 1783553392826,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.31317693540001984,
"range": "stddev: 0.0029124281162835833",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3131931951999832,
"range": "stddev: 0.0019241181307287027",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4201879119999603,
"range": "stddev: 0.0024949899158932375",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8049989316666597,
"range": "stddev: 0.004418554169633069",
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
"date": 1783553392142,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.306167887599986,
"range": "stddev: 0.00455298943763762",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30515107579999495,
"range": "stddev: 0.002889799317099573",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4117053576666194,
"range": "stddev: 0.0038393725862082644",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.775403970666692,
"range": "stddev: 0.01534737716526181",
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
"date": 1783553392742,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3126837941999838,
"range": "stddev: 0.008795332524901294",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.31165037960001885,
"range": "stddev: 0.007638713558871129",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.410722074333383,
"range": "stddev: 0.001422949449134303",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7885838390000117,
"range": "stddev: 0.03155648963137366",
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
"date": 1783553392286,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30503971319999434,
"range": "stddev: 0.002652328706337517",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30582071819997053,
"range": "stddev: 0.005678290598669849",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4105593430000454,
"range": "stddev: 0.0011225296643589977",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7976482856666585,
"range": "stddev: 0.026076973153578976",
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
"date": 1783553392237,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3019864971999141,
"range": "stddev: 0.001292616218359207",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3043203237999933,
"range": "stddev: 0.005064235721847563",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4099938866667496,
"range": "stddev: 0.0022652148828757406",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8026925666666407,
"range": "stddev: 0.014264597048380363",
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
"date": 1783553392460,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30715778100002356,
"range": "stddev: 0.002832817193508556",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30338760460003866,
"range": "stddev: 0.0039993101029437055",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4195418113333744,
"range": "stddev: 0.0012521444657676848",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7791071029999634,
"range": "stddev: 0.01950112002864633",
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
"date": 1783553392711,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30213561219993607,
"range": "stddev: 0.002006353948669093",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30137821339999393,
"range": "stddev: 0.0012408257517941513",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4068169933332986,
"range": "stddev: 0.0012879836459388505",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7878588106667244,
"range": "stddev: 0.033163138536804505",
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
"date": 1783553392484,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30745732439995666,
"range": "stddev: 0.002829384529889521",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30706101640003,
"range": "stddev: 0.005725704865437918",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4141608876667913,
"range": "stddev: 0.000345345661248729",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.774712104999935,
"range": "stddev: 0.026986363537571386",
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
"date": 1783553392242,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30365064780007744,
"range": "stddev: 0.002983426391991283",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30275209279998305,
"range": "stddev: 0.0004313681472679653",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4141174389999378,
"range": "stddev: 0.005766674136105228",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.745495688333449,
"range": "stddev: 0.016866169322894475",
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
"date": 1783553392256,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3031498883999575,
"range": "stddev: 0.001652950345437423",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3046400500000345,
"range": "stddev: 0.0053643771773260645",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4085017130000779,
"range": "stddev: 0.003196862673310079",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7237846156666214,
"range": "stddev: 0.01538416390920765",
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
"date": 1783553392479,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30757613860000677,
"range": "stddev: 0.003474696709356421",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3047925930000474,
"range": "stddev: 0.0013706061445264432",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.414527924666648,
"range": "stddev: 0.0014900673406336561",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7477424073333623,
"range": "stddev: 0.028071168748282017",
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
"date": 1783553392789,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3094834355999865,
"range": "stddev: 0.010567619773520229",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.30554812839995976,
"range": "stddev: 0.0015743448958273991",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4134865673332417,
"range": "stddev: 0.001579057234628329",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.767139116333207,
"range": "stddev: 0.06379793264182679",
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
"date": 1783553392506,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3072489172000587,
"range": "stddev: 0.0028601236289136378",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3044780918000015,
"range": "stddev: 0.0024667058026712958",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4104888669999884,
"range": "stddev: 0.0026137054477130897",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7608700943332374,
"range": "stddev: 0.027599195364241925",
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
"date": 1783553392624,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.30629910340003336,
"range": "stddev: 0.004411881984894643",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3161118059999808,
"range": "stddev: 0.0016706504718561551",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4158991133334287,
"range": "stddev: 0.009456166036432652",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.769645581333331,
"range": "stddev: 0.02862312625169798",
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
"date": 1783553392465,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3285165920000054,
"range": "stddev: 0.006194557112409858",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.330671798000094,
"range": "stddev: 0.00619737909395474",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.421084765000008,
"range": "stddev: 0.005525140916888139",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8501889816667094,
"range": "stddev: 0.014971276661915247",
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
"date": 1783553392440,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6329001755999343,
"range": "stddev: 0.01330278867627056",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6356325178000134,
"range": "stddev: 0.007224392776020576",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4170736060000308,
"range": "stddev: 0.0024228241853721673",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.853807517000026,
"range": "stddev: 0.01686513819948185",
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
"date": 1783553392154,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5700348609999765,
"range": "stddev: 0.009042135489652109",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.565423144999977,
"range": "stddev: 0.010595558973453304",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4109686550000333,
"range": "stddev: 0.00531684995176376",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.768524811999972,
"range": "stddev: 0.02277053945192315",
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
"date": 1783553392593,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5667530649999662,
"range": "stddev: 0.004414297969918906",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5677657423999335,
"range": "stddev: 0.009034044642442526",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4079927906666398,
"range": "stddev: 0.005749592435408591",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7731468196666356,
"range": "stddev: 0.03618702379148757",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5467249173333737,
"range": "stddev: 0.011419499581154364",
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
"date": 1783553392221,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5708970671998941,
"range": "stddev: 0.005015118451940798",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5734161248002237,
"range": "stddev: 0.011992927456932007",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.418571732333324,
"range": "stddev: 0.004770141918348353",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7651859223333304,
"range": "stddev: 0.01127394765726185",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.551440102000015,
"range": "stddev: 0.010625427429618764",
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
"date": 1783553392816,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5624896773999353,
"range": "stddev: 0.002760452147034357",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5980802501999278,
"range": "stddev: 0.026681960696985592",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4088060146667278,
"range": "stddev: 0.005506562952545363",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7595622140000464,
"range": "stddev: 0.012054073275839324",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.553554569666782,
"range": "stddev: 0.005169503728289764",
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
"date": 1783553392387,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5891772399999808,
"range": "stddev: 0.010562193480367089",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6088884905999976,
"range": "stddev: 0.02649432695499708",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4349760636666435,
"range": "stddev: 0.009863081445762376",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.821949775666667,
"range": "stddev: 0.023942833161925532",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5729459923333404,
"range": "stddev: 0.010467763055539545",
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
"date": 1783553392607,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6175673028000119,
"range": "stddev: 0.00991955344297594",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6058729687999858,
"range": "stddev: 0.007957425651053954",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4394888263333694,
"range": "stddev: 0.005393256682182019",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8447311603333296,
"range": "stddev: 0.05625249085359453",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.597412863000007,
"range": "stddev: 0.04604266469240038",
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
"date": 1783553392777,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.60640154299997,
"range": "stddev: 0.012407720245172128",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5955297819999714,
"range": "stddev: 0.006020828397974909",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4336577909999733,
"range": "stddev: 0.0016654912191752893",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.821519858666685,
"range": "stddev: 0.014590288068359231",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5716396966666555,
"range": "stddev: 0.011534658529837186",
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
"date": 1783553392828,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5914383082000313,
"range": "stddev: 0.0030157915786366077",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5890695910000204,
"range": "stddev: 0.005908993916782701",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.443596988333373,
"range": "stddev: 0.005698879476138808",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8998696990000403,
"range": "stddev: 0.05983969043087384",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5810087176666912,
"range": "stddev: 0.0074368406392403795",
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
"date": 1783553392758,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6026530127999876,
"range": "stddev: 0.002360778767488692",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5943670299999895,
"range": "stddev: 0.0038865611432050157",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4426864383333395,
"range": "stddev: 0.010159654853139842",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8810958206666633,
"range": "stddev: 0.04230970843562854",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5793082653333386,
"range": "stddev: 0.011803477459975694",
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
"date": 1783553392203,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6412109962000159,
"range": "stddev: 0.028498679301615315",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6226292293999904,
"range": "stddev: 0.009095303909825303",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4296197589999338,
"range": "stddev: 0.0063431525542634115",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.916873677666672,
"range": "stddev: 0.036263454153726435",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5706901766667063,
"range": "stddev: 0.008828803764516427",
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
"date": 1783553392291,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6235061478000261,
"range": "stddev: 0.00728546740024864",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6455374982000194,
"range": "stddev: 0.01964006178130404",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.429570126999958,
"range": "stddev: 0.004998520631735339",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.912878138999986,
"range": "stddev: 0.04583659058552584",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5705359926666158,
"range": "stddev: 0.018717040410616635",
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
"date": 1783553392632,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6193607618000442,
"range": "stddev: 0.009961264670768539",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6200322289999803,
"range": "stddev: 0.008139199835554126",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4180836923333118,
"range": "stddev: 0.003957213694232848",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8482090273332688,
"range": "stddev: 0.02104357919522929",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5644371179999628,
"range": "stddev: 0.014078189434823578",
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
"date": 1783553392177,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6023549617999834,
"range": "stddev: 0.01781220860342687",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5875220963999255,
"range": "stddev: 0.010677851285992846",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.426242428000099,
"range": "stddev: 0.005830225380866323",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8285241940000865,
"range": "stddev: 0.018773836887731966",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5616439333333194,
"range": "stddev: 0.011485987799487092",
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
"date": 1783553392270,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.5846687828000541,
"range": "stddev: 0.011152686912630588",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.5815607759999238,
"range": "stddev: 0.0049867964426827975",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4306722343333906,
"range": "stddev: 0.007464406822972894",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8423685269999623,
"range": "stddev: 0.015745860129914482",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5662621163333672,
"range": "stddev: 0.009309699848136517",
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
"date": 1783553392198,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6214755545999651,
"range": "stddev: 0.014837905331786229",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.602525949999972,
"range": "stddev: 0.013484082767946553",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4198864903332833,
"range": "stddev: 0.004279573456964553",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8875181619999544,
"range": "stddev: 0.03736257647141285",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5697311733333663,
"range": "stddev: 0.011698857748216959",
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
"date": 1783553392416,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6047249660000489,
"range": "stddev: 0.004556324875883691",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.6246803086,
"range": "stddev: 0.014504245258727571",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4234738749999754,
"range": "stddev: 0.006830509336096651",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9069380560000204,
"range": "stddev: 0.0300809988654104",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.878478783666651,
"range": "stddev: 0.012271728475475571",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5668546166666601,
"range": "stddev: 0.008062853445029813",
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
"date": 1783553392298,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.6262540667999928,
"range": "stddev: 0.01215876007060517",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.622595293599943,
"range": "stddev: 0.01335113086951019",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4342628753333884,
"range": "stddev: 0.006271702592982679",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9243097013334136,
"range": "stddev: 0.0287828259161793",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.948323604999965,
"range": "stddev: 0.03205637908754287",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5847857329999897,
"range": "stddev: 0.012255891749062357",
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
"date": 1783553392448,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34060620799991737,
"range": "stddev: 0.0040050912043847675",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3393297787999927,
"range": "stddev: 0.005636350416097226",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4182477679999768,
"range": "stddev: 0.004398408927216444",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8538911493333217,
"range": "stddev: 0.01331941318145995",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.888609829666771,
"range": "stddev: 0.027433402585776236",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.568361901000041,
"range": "stddev: 0.02113319191568086",
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
"date": 1783553392263,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34522203180003996,
"range": "stddev: 0.001659220586664375",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3395976769999834,
"range": "stddev: 0.0030798392475948418",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.423309169999963,
"range": "stddev: 0.004690255428855229",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8869586753332896,
"range": "stddev: 0.030287991071175066",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.924762338333494,
"range": "stddev: 0.01963602066950526",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5730744256666942,
"range": "stddev: 0.01631961964519928",
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
"date": 1783553392508,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33804701079989175,
"range": "stddev: 0.0027309644119004033",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33780852799991407,
"range": "stddev: 0.0011998687155999018",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4394226759998976,
"range": "stddev: 0.003663165515241024",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.824530814000051,
"range": "stddev: 0.035947842941933664",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.906055525999818,
"range": "stddev: 0.014310181337049406",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.575676698666636,
"range": "stddev: 0.009560616186140921",
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
"date": 1783553392265,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3436725282000225,
"range": "stddev: 0.006168966587442836",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3426130581999132,
"range": "stddev: 0.004850940636147113",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.443082359333251,
"range": "stddev: 0.009542227473895725",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8979722456665513,
"range": "stddev: 0.025372950235203116",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.944087404999815,
"range": "stddev: 0.015841496320911613",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5921388443334763,
"range": "stddev: 0.017509461667871463",
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
"date": 1783553392389,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33673651220005923,
"range": "stddev: 0.003507962896446781",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3342051189999438,
"range": "stddev: 0.002387541143308862",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4222246520001438,
"range": "stddev: 0.0031499149299847496",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7947063483334205,
"range": "stddev: 0.01649188580960879",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.863868758999919,
"range": "stddev: 0.04378097124578855",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5668368343334198,
"range": "stddev: 0.016355297482589896",
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
"date": 1783553392552,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36074770539999007,
"range": "stddev: 0.016374148762407698",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34459709160000784,
"range": "stddev: 0.0075348239679697915",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.424084490666549,
"range": "stddev: 0.004154042010354692",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.861928758333306,
"range": "stddev: 0.0141538324004974",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.879073959666584,
"range": "stddev: 0.03640328183455752",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5778913256666176,
"range": "stddev: 0.019688725586713727",
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
"date": 1783553392145,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3395793468000193,
"range": "stddev: 0.0029900298351515733",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3366028686000391,
"range": "stddev: 0.0030056283852923442",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4149005430000823,
"range": "stddev: 0.003114906703411826",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8013052390001576,
"range": "stddev: 0.028457261291447985",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.795330153999961,
"range": "stddev: 0.042287329634620986",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5610800320000635,
"range": "stddev: 0.011765868182013828",
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
"date": 1783553392260,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34495745119993443,
"range": "stddev: 0.008319507484351438",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3476818107998952,
"range": "stddev: 0.0039444167056086",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4164297116665996,
"range": "stddev: 0.006784646876115614",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.78338207333339,
"range": "stddev: 0.022559386962997415",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.868670310666706,
"range": "stddev: 0.12673717021518693",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5643523043333214,
"range": "stddev: 0.01301122885600625",
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
"date": 1783553392462,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33891625380001644,
"range": "stddev: 0.0013045502550424958",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3490630500001316,
"range": "stddev: 0.02392821603750973",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4180103333333136,
"range": "stddev: 0.0018689315863043658",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.800430548000198,
"range": "stddev: 0.026949297089621037",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.808882936333399,
"range": "stddev: 0.041974722621399306",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5601807583334448,
"range": "stddev: 0.01297553531056275",
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
"date": 1783553392224,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3382436768000844,
"range": "stddev: 0.0018076504614142812",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33822402100004184,
"range": "stddev: 0.0008787102911167987",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4189725846667898,
"range": "stddev: 0.003099211657549699",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.788911439999841,
"range": "stddev: 0.014338120576515805",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.799765777666684,
"range": "stddev: 0.01771414696804202",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.583459494000029,
"range": "stddev: 0.025980561751403522",
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
"date": 1783553392535,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3390306530000089,
"range": "stddev: 0.0019282828099897692",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33891404980004153,
"range": "stddev: 0.0016274820103773693",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4097144356666529,
"range": "stddev: 0.0021838156170746054",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7808688356666003,
"range": "stddev: 0.015575259086716264",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.742357821000041,
"range": "stddev: 0.03887902426440161",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5555419509999713,
"range": "stddev: 0.011739232626105591",
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
"date": 1783553392240,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3695095186000799,
"range": "stddev: 0.01002225669703888",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3721236479999789,
"range": "stddev: 0.00278849887844936",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4257225633333899,
"range": "stddev: 0.014691654218166518",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.897700365666575,
"range": "stddev: 0.06588174987655744",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.86285985633352,
"range": "stddev: 0.02538948765793063",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5636166553332866,
"range": "stddev: 0.014000717421702124",
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
"date": 1783553392673,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3635456560000421,
"range": "stddev: 0.005911916412095085",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35979102719993533,
"range": "stddev: 0.005258608685485898",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4216185976665656,
"range": "stddev: 0.006974652526648527",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.865424605333525,
"range": "stddev: 0.036742363770596725",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.763761175666636,
"range": "stddev: 0.07129321428380692",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5741687073332287,
"range": "stddev: 0.020237039230782138",
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
"date": 1783553392361,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3430023793998771,
"range": "stddev: 0.003179925373286144",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34176774899997325,
"range": "stddev: 0.0014664570991170416",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4131686446667118,
"range": "stddev: 0.0028469087465425127",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8178982806668196,
"range": "stddev: 0.04047476866797848",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.771531115666676,
"range": "stddev: 0.014378590724256679",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5588329176666775,
"range": "stddev: 0.010501528191859528",
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
"date": 1783553392753,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3483257142001094,
"range": "stddev: 0.008557563644960593",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.344470939599978,
"range": "stddev: 0.00932587368647337",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4165493343333158,
"range": "stddev: 0.0036144798327394038",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8245750026664305,
"range": "stddev: 0.043849412184417354",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.76288768100009,
"range": "stddev: 0.01594058191018637",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5640242249999876,
"range": "stddev: 0.013848354992835227",
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
"id": "29d23f801ca3199fdb159c6d17aa7f59594cd6b5",
"message": "Remove internal references (#323)",
"timestamp": "2025-05-07T13:21:31Z",
"tree_id": "4f0893027461406316f54f8100fd927cd9923d05",
"url": "https://github.com/proximafusion/vmecpp/commit/29d23f801ca3199fdb159c6d17aa7f59594cd6b5"
},
"date": 1783553392258,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33883518699999515,
"range": "stddev: 0.0013015120275256596",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34465940100001263,
"range": "stddev: 0.0018649253236387745",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.417348853666681,
"range": "stddev: 0.004281730299705812",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.792393073000047,
"range": "stddev: 0.018499680996493458",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.604689122333374,
"range": "stddev: 0.01828906667394499",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5636812633333268,
"range": "stddev: 0.014589179624990439",
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
"id": "ecd2e236cebffdfcf0263085d4ff56f0ecad0d42",
"message": "Show constructing VmecInput completely in Python",
"timestamp": "2025-05-07T01:07:17+02:00",
"tree_id": "91b7e895b513520721a3e18fad14472d69e7cd6a",
"url": "https://github.com/proximafusion/vmecpp/commit/ecd2e236cebffdfcf0263085d4ff56f0ecad0d42"
},
"date": 1783553392794,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33880881180000416,
"range": "stddev: 0.0016384844955760992",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34561950859999796,
"range": "stddev: 0.005115596101423967",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.422730935333334,
"range": "stddev: 0.007285831691560229",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.788501412666657,
"range": "stddev: 0.014529777843146642",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.581904269000082,
"range": "stddev: 0.02697921646269289",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5534677373333352,
"range": "stddev: 0.002876898164253813",
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
"id": "07eba36ff16696b838fbfbed50f2cea6209bcf84",
"message": "Resize sparse coefficients when too small",
"timestamp": "2025-05-07T01:08:50+02:00",
"tree_id": "8ef02376e936f5f13e4f1c8cdae7217e3f79cf4d",
"url": "https://github.com/proximafusion/vmecpp/commit/07eba36ff16696b838fbfbed50f2cea6209bcf84"
},
"date": 1783553392164,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3423837977999938,
"range": "stddev: 0.009125949925706714",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3359881606000272,
"range": "stddev: 0.002139455870721161",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.416198222333378,
"range": "stddev: 0.002275248761807112",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7708146573334034,
"range": "stddev: 0.008091269426711845",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.565072817333354,
"range": "stddev: 0.04747853031942143",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5502503496666122,
"range": "stddev: 0.011818614145670701",
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
"id": "365c07eaab0152eb00387f41e821e4686ae3fe6e",
"message": "Converted ivac counter to an enum with descriptive states (#280)",
"timestamp": "2025-05-08T16:25:43Z",
"tree_id": "d76455baa793c81b29b9a6bab932bbaff47f7e26",
"url": "https://github.com/proximafusion/vmecpp/commit/365c07eaab0152eb00387f41e821e4686ae3fe6e"
},
"date": 1783553392288,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33995300220008173,
"range": "stddev: 0.00495066354274584",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3381272910000462,
"range": "stddev: 0.0035569583514084373",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4068175689999407,
"range": "stddev: 0.002349925572176297",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.758656698666679,
"range": "stddev: 0.020457134425073294",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.486189932666624,
"range": "stddev: 0.04159271461376871",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5521270530000493,
"range": "stddev: 0.0060625458571508415",
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
"id": "6b8faa83515cd514e01084f1c440611fb5c54b4c",
"message": "Renamed ivac counter to vacuum_state (#317)",
"timestamp": "2025-05-08T16:25:44Z",
"tree_id": "87276ba692180efd37eae86b0e9d5aa9e92799c7",
"url": "https://github.com/proximafusion/vmecpp/commit/6b8faa83515cd514e01084f1c440611fb5c54b4c"
},
"date": 1783553392438,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3380657168000198,
"range": "stddev: 0.0017111132409434783",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33783098359999714,
"range": "stddev: 0.0014445745204142951",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4173213293333145,
"range": "stddev: 0.004017757102698095",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7892515586667437,
"range": "stddev: 0.05117623126664839",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.549771396666756,
"range": "stddev: 0.028767820593987435",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5600128853333597,
"range": "stddev: 0.018970517130075172",
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
"id": "d7153695cf9f40f2d270f5c2b3c87b1c7a747477",
"message": "Pass on verbose flag to ideal mhd",
"timestamp": "2025-05-07T13:38:22+02:00",
"tree_id": "6f8ab5405e055f7fd6c9a200d6439c7836b83448",
"url": "https://github.com/proximafusion/vmecpp/commit/d7153695cf9f40f2d270f5c2b3c87b1c7a747477"
},
"date": 1783553392748,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3402048489999288,
"range": "stddev: 0.004810016204501639",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3350621556000533,
"range": "stddev: 0.0005263414376584667",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4117387710000457,
"range": "stddev: 0.004197120143773742",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.774382192666735,
"range": "stddev: 0.010462238060625136",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.570426352333394,
"range": "stddev: 0.01369452116632152",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5617542900000292,
"range": "stddev: 0.014300284536615858",
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
"id": "448609b293e4ca1f4d6ad46a66e5f276f6c45825",
"message": "Clearly show reduction in iteration count for hot-restart",
"timestamp": "2025-05-11T20:06:21+02:00",
"tree_id": "71b186ba152c4f5290b4359eedfb38d47d2d4c45",
"url": "https://github.com/proximafusion/vmecpp/commit/448609b293e4ca1f4d6ad46a66e5f276f6c45825"
},
"date": 1783553392336,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3360891975999493,
"range": "stddev: 0.0013559664985734459",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3379222005999509,
"range": "stddev: 0.0016605468356952438",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4141934936665923,
"range": "stddev: 0.0046264999807479204",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7702105289999204,
"range": "stddev: 0.010881375568029093",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.564596189666645,
"range": "stddev: 0.05284884320416316",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5540117043332582,
"range": "stddev: 0.007041457349034013",
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
"id": "b91ddb712f34c657fc11a06640d7daf1129667e7",
"message": "Skip gradual vacuum pressure activation on hot-restart (#330)",
"timestamp": "2025-05-12T17:14:10Z",
"tree_id": "260ce1451d1cbd83185211304240574f3db9b40c",
"url": "https://github.com/proximafusion/vmecpp/commit/b91ddb712f34c657fc11a06640d7daf1129667e7"
},
"date": 1783553392637,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3372083009999642,
"range": "stddev: 0.0018179822303116399",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34055567780001184,
"range": "stddev: 0.001698905975630369",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4111179116666033,
"range": "stddev: 0.0013182917201774546",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7551379279999915,
"range": "stddev: 0.017926037939635424",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.556055234666625,
"range": "stddev: 0.03757675690461971",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5589323606666312,
"range": "stddev: 0.02247483815997383",
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
"id": "7c32f2ea68f2adfeedfe1d11b41842fe6a304446",
"message": "Added details to VmecInput documentation",
"timestamp": "2025-05-12T19:56:36+02:00",
"tree_id": "55cc0e1d295624a51daed46e426119848e2bacc2",
"url": "https://github.com/proximafusion/vmecpp/commit/7c32f2ea68f2adfeedfe1d11b41842fe6a304446"
},
"date": 1783553392469,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34242858980001073,
"range": "stddev: 0.005845434825554943",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33509908460005133,
"range": "stddev: 0.0008983718968895703",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4187319623333678,
"range": "stddev: 0.004799572438630878",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7849960346666953,
"range": "stddev: 0.026698672347836724",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.593195432333383,
"range": "stddev: 0.04318427627171818",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.552550774999948,
"range": "stddev: 0.00947424608157436",
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
"id": "6b63d4712683803353812b5e26c854915530221f",
"message": "WOut quantity docstrings (#334)",
"timestamp": "2025-05-22T09:40:14Z",
"tree_id": "c005c251d870130f243a57bca36db54746fd587c",
"url": "https://github.com/proximafusion/vmecpp/commit/6b63d4712683803353812b5e26c854915530221f"
},
"date": 1783553392435,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34385626360003696,
"range": "stddev: 0.0034292544273101942",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3396632787999806,
"range": "stddev: 0.0028403593889044293",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4229982786666067,
"range": "stddev: 0.004851247234257529",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7714437589999457,
"range": "stddev: 0.03363795606790146",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.564568229666293,
"range": "stddev: 0.015101307359016302",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5486439376666112,
"range": "stddev: 0.011944312354705982",
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
"id": "5408dad503a9b14656cc61219b5435b409a0a139",
"message": "Disable SZIP support in HDF5: not needed here (#335)",
"timestamp": "2025-05-27T16:49:44Z",
"tree_id": "8dd6e0635b417fada61c78c463ab324d7165d71f",
"url": "https://github.com/proximafusion/vmecpp/commit/5408dad503a9b14656cc61219b5435b409a0a139"
},
"date": 1783553392373,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3401174278000326,
"range": "stddev: 0.0024016244876457612",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34154144839994843,
"range": "stddev: 0.004283876791568231",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4148220193333145,
"range": "stddev: 0.003367827463209213",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.790383854666743,
"range": "stddev: 0.05843278788234722",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.550807667999985,
"range": "stddev: 0.01722606685146477",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.550230074666615,
"range": "stddev: 0.011695687645715408",
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
"id": "386a68cfe06f79c38e115e81104c33811b48987b",
"message": "Add AGENTS.md for working with AI (#338)",
"timestamp": "2025-06-10T09:20:04Z",
"tree_id": "7ad8b689d739e4803e475de6c56dfc8cfc2f3bcc",
"url": "https://github.com/proximafusion/vmecpp/commit/386a68cfe06f79c38e115e81104c33811b48987b"
},
"date": 1783553392295,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3452902549999635,
"range": "stddev: 0.002219246720760741",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.344678001200009,
"range": "stddev: 0.0022824108797087237",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4140870603332587,
"range": "stddev: 0.001843742920423385",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7976938556666937,
"range": "stddev: 0.039956935523025136",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.634788109333385,
"range": "stddev: 0.04506044389991422",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5510223546666566,
"range": "stddev: 0.009811330153779862",
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
"id": "b44fb7f0dc5e817ccba58807586665d1efaafa5f",
"message": "Add a naming guide for VMEC++ (#339)",
"timestamp": "2025-06-10T11:08:52Z",
"tree_id": "8fe6e1412ce33dfa43e5ab3f093e3aee868731a5",
"url": "https://github.com/proximafusion/vmecpp/commit/b44fb7f0dc5e817ccba58807586665d1efaafa5f"
},
"date": 1783553392630,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3474633572001039,
"range": "stddev: 0.003868164490610219",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3476154616001622,
"range": "stddev: 0.0026682269529664975",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4146065330000965,
"range": "stddev: 0.002049427465675506",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.782864890666739,
"range": "stddev: 0.03968326824428616",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.547569373333166,
"range": "stddev: 0.005314302228354421",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.557917226333302,
"range": "stddev: 0.018498029126356615",
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
"id": "a7797dc5ccbee0541708f452d7b0e63bc6912bf4",
"message": "Consolidate algorithmic constants into comprehensive constants header (#340)",
"timestamp": "2025-06-10T17:12:43Z",
"tree_id": "4eb9cc73e7ac01ca670d4d148027d1a3f38a6784",
"url": "https://github.com/proximafusion/vmecpp/commit/a7797dc5ccbee0541708f452d7b0e63bc6912bf4"
},
"date": 1783553392598,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34265308340000045,
"range": "stddev: 0.004385650527752882",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3393404674001431,
"range": "stddev: 0.0033515281982882126",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4113109479999366,
"range": "stddev: 0.004716549334822022",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7878208619999896,
"range": "stddev: 0.011046981123783448",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.508369406333259,
"range": "stddev: 0.022601418035295404",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.550764703666573,
"range": "stddev: 0.005959172306139414",
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
"id": "39d75227e5aaefdddccfded200ae84e1c926d2aa",
"message": "migrate m_evn/m_odd to k{Even,Odd}Parity (#341)",
"timestamp": "2025-06-10T17:12:43Z",
"tree_id": "4c34af5ab2e3885f2228adc48bed64f9ed7edfaa",
"url": "https://github.com/proximafusion/vmecpp/commit/39d75227e5aaefdddccfded200ae84e1c926d2aa"
},
"date": 1783553392300,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33746708380012935,
"range": "stddev: 0.0014504011053215156",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3382082569998602,
"range": "stddev: 0.002326757721492698",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4114527336666167,
"range": "stddev: 0.001954842588939225",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7496037866667393,
"range": "stddev: 0.016777334075991315",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.50318734700007,
"range": "stddev: 0.03190664923053746",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5457572710000325,
"range": "stddev: 0.011170958390664851",
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
"id": "8438ec1d093aaed11b69b8a276f2190a92554b48",
"message": "Remove all non-ASCII characters from VMEC++. (#342)",
"timestamp": "2025-06-10T18:20:24Z",
"tree_id": "8a8714eb1257b7d6955a0ca6d0ec4021199da203",
"url": "https://github.com/proximafusion/vmecpp/commit/8438ec1d093aaed11b69b8a276f2190a92554b48"
},
"date": 1783553392496,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3367880382000294,
"range": "stddev: 0.0021498549787527274",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33392491760005216,
"range": "stddev: 0.0011814127751991821",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4118071003332868,
"range": "stddev: 0.00403022710501827",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7554334326666017,
"range": "stddev: 0.023075710973790346",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.543558933666645,
"range": "stddev: 0.06837599076475596",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5481607289999981,
"range": "stddev: 0.008522336387632853",
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
"id": "01bab5d7f1788e7d3f119fd1263517e0d7403761",
"message": "Remove remains of m_evn and m_odd old constants. (#343)",
"timestamp": "2025-06-10T20:31:02Z",
"tree_id": "506a86094396b41dd2fd0697d230ef0e9baaf011",
"url": "https://github.com/proximafusion/vmecpp/commit/01bab5d7f1788e7d3f119fd1263517e0d7403761"
},
"date": 1783553392133,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33645688479991803,
"range": "stddev: 0.0020240078221178013",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33818979159987067,
"range": "stddev: 0.006584737839976348",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4131235913332603,
"range": "stddev: 0.005387342839731023",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7653310223331573,
"range": "stddev: 0.030850796801508797",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.506940991333218,
"range": "stddev: 0.013709820513062085",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5509805279998545,
"range": "stddev: 0.014630137926822311",
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
"id": "ef96e726f1cc9bda8bb2a97ab32add94891adb36",
"message": "Remove an unused converter method. (#344)",
"timestamp": "2025-06-10T20:31:02Z",
"tree_id": "39d6d02baa81b2471b79a43bb79f6bf4f05d7fc1",
"url": "https://github.com/proximafusion/vmecpp/commit/ef96e726f1cc9bda8bb2a97ab32add94891adb36"
},
"date": 1783553392797,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34023533299996417,
"range": "stddev: 0.004442581968640934",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3391638655999486,
"range": "stddev: 0.002711781061206084",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4123702829997455,
"range": "stddev: 0.0017265164365188506",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7798156926664888,
"range": "stddev: 0.015989608391942578",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.602563885333288,
"range": "stddev: 0.00811940632184371",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5540301289999359,
"range": "stddev: 0.01432827675203856",
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
"id": "bfbb2c18d447cdca4972d39c3a7c9e1c0084a527",
"message": "Clean up the naming guide. (#345)",
"timestamp": "2025-06-12T07:33:18Z",
"tree_id": "4511a38e480ab867e9eb22825fe0533443e82cf5",
"url": "https://github.com/proximafusion/vmecpp/commit/bfbb2c18d447cdca4972d39c3a7c9e1c0084a527"
},
"date": 1783553392654,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34583427859997756,
"range": "stddev: 0.007322342570193779",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34056714680009464,
"range": "stddev: 0.0035477676864306327",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4214652003333867,
"range": "stddev: 0.005581258793607768",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7807007306666187,
"range": "stddev: 0.04323258202390185",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.556862052333296,
"range": "stddev: 0.018985989504618453",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5528938343333418,
"range": "stddev: 0.012467580775829778",
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
"id": "c8d833dd5e17808c7ce004e70f1f71a120cd4041",
"message": "Refactor modified arguments to start with m_` (#346)",
"timestamp": "2025-06-12T07:33:18Z",
"tree_id": "2983dbc7003ff5f42bb2f8e16b2517a5cb0e4531",
"url": "https://github.com/proximafusion/vmecpp/commit/c8d833dd5e17808c7ce004e70f1f71a120cd4041"
},
"date": 1783553392690,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3477128964000258,
"range": "stddev: 0.00318366869168025",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3480082830000356,
"range": "stddev: 0.005097722397693624",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4205051833334135,
"range": "stddev: 0.0018491557492104256",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7699575126666787,
"range": "stddev: 0.02707422543514465",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.601420679666413,
"range": "stddev: 0.11252482284204453",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5750652153333249,
"range": "stddev: 0.023184496669187853",
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
"id": "452a07722e61e449783f3f1e0dbfa854e9b9e8ff",
"message": "Mention the naming guide in AGENTS.md (#347)",
"timestamp": "2025-06-12T13:43:55Z",
"tree_id": "e42f219f13776a4867ae6518ec94e34be5228725",
"url": "https://github.com/proximafusion/vmecpp/commit/452a07722e61e449783f3f1e0dbfa854e9b9e8ff"
},
"date": 1783553392339,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33940663040011715,
"range": "stddev: 0.00258284880728997",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3408381446000021,
"range": "stddev: 0.004539513767057104",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4086801449998347,
"range": "stddev: 0.00448588820592262",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.741593398333407,
"range": "stddev: 0.025008752790750575",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.55409093933334,
"range": "stddev: 0.040156358418798443",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5471220063333628,
"range": "stddev: 0.012650046356823343",
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
"id": "fb738c785ce1db3dbe31f7e2298b2c09770a64c5",
"message": "Document FourierBasis* members (#348)",
"timestamp": "2025-06-16T10:10:09Z",
"tree_id": "9ec8ea31405a1465c34127cff17590e31d42aba2",
"url": "https://github.com/proximafusion/vmecpp/commit/fb738c785ce1db3dbe31f7e2298b2c09770a64c5"
},
"date": 1783553392823,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35392486700002335,
"range": "stddev: 0.001500192561175626",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3480911816000116,
"range": "stddev: 0.007872921936614135",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4163615013333886,
"range": "stddev: 0.014354855181493433",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7474064409999905,
"range": "stddev: 0.014201309795226776",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.49096055100002,
"range": "stddev: 0.051994463965048816",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5633162893332155,
"range": "stddev: 0.0128904763707703",
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
"id": "a4295888965cd885de89677ead48b67e00237210",
"message": "Make FourierBasisFastToroidal consistent with FourierBasisFastPoloidal. (#349)",
"timestamp": "2025-06-16T18:10:15Z",
"tree_id": "af19530aa3ea2c8a2ff3db1b432f7b0d0ed2b847",
"url": "https://github.com/proximafusion/vmecpp/commit/a4295888965cd885de89677ead48b67e00237210"
},
"date": 1783553392590,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3413902743999643,
"range": "stddev: 0.002208258736146934",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34127778539996145,
"range": "stddev: 0.0016133656544847987",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4123993743332903,
"range": "stddev: 0.0007513487281439006",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.76125520366683,
"range": "stddev: 0.010771206134942299",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.596486174333373,
"range": "stddev: 0.06986401702317885",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5511662746665327,
"range": "stddev: 0.010477028320001617",
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
"id": "0249e786dea80cc6d266a9777e22eb53946f8384",
"message": "no OpenMP -> no compiler warnings (#336)",
"timestamp": "2025-06-16T18:24:44Z",
"tree_id": "2f2a2b4038b03c0e0185367a80f78f924debf2e1",
"url": "https://github.com/proximafusion/vmecpp/commit/0249e786dea80cc6d266a9777e22eb53946f8384"
},
"date": 1783553392135,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3537458560000232,
"range": "stddev: 0.0033967897588221808",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36276178300013273,
"range": "stddev: 0.01266803105691084",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4167303476665438,
"range": "stddev: 0.002809478837182614",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7680354013333877,
"range": "stddev: 0.03049929942464293",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.580566174333399,
"range": "stddev: 0.017086427888090488",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.56256071100006,
"range": "stddev: 0.027221983140208912",
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
"id": "0e9d79a6af6d67cb27bd1fefe19f6e5b3127acd7",
"message": "Document the converter functions between different Fourier basis representations. (#350)",
"timestamp": "2025-06-17T07:47:30Z",
"tree_id": "a3d7736c94ee25044537ec73ac6c7c8ab9a54a88",
"url": "https://github.com/proximafusion/vmecpp/commit/0e9d79a6af6d67cb27bd1fefe19f6e5b3127acd7"
},
"date": 1783553392182,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34585086480001337,
"range": "stddev: 0.0011444543535395282",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3405967629999395,
"range": "stddev: 0.0035308120498384493",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.415169801333377,
"range": "stddev: 0.006221537377485449",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7551928536666614,
"range": "stddev: 0.001976748303605004",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.550456021666681,
"range": "stddev: 0.01531442776216108",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5488699050000225,
"range": "stddev: 0.010390232967147719",
"extra": "rounds: 3"
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783553392450,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35083426060009515,
"range": "stddev: 0.002428835706519022",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3467605262001598,
"range": "stddev: 0.0029973120350354353",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.41459625966642,
"range": "stddev: 0.00512809704213416",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.75618602433345,
"range": "stddev: 0.010201043312898835",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.505395712666541,
"range": "stddev: 0.060368678302396415",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5517731836668343,
"range": "stddev: 0.00748841033058711",
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
"id": "526d29877859bd61fed5605eb43d4b6021bb4bc1",
"message": "Indata2json update to fix gfortran14 errors (#353)",
"timestamp": "2025-07-10T10:04:01Z",
"tree_id": "85ffc0e814db3075f8e6aa8840a17302188ad66f",
"url": "https://github.com/proximafusion/vmecpp/commit/526d29877859bd61fed5605eb43d4b6021bb4bc1"
},
"date": 1783553392368,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3436046486000123,
"range": "stddev: 0.006123175468065095",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3479333347999273,
"range": "stddev: 0.009123525838585025",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4191716686667253,
"range": "stddev: 0.0025038453299180276",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.770054761333389,
"range": "stddev: 0.020017340459865647",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.613789144332864,
"range": "stddev: 0.015460631503117266",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.6140959416667708,
"range": "stddev: 0.018811811131117493",
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
"id": "60389860bb86f719c04b2687ff61167c71d697fa",
"message": "More informative FATAL ERROR message (#356)",
"timestamp": "2025-07-11T10:10:50Z",
"tree_id": "3e9cf51a3485d10250672b0b5df0f9d763bb2921",
"url": "https://github.com/proximafusion/vmecpp/commit/60389860bb86f719c04b2687ff61167c71d697fa"
},
"date": 1783553392419,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3496832720000384,
"range": "stddev: 0.002575771891137425",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3528142167999249,
"range": "stddev: 0.0037826063981656203",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4123850066665302,
"range": "stddev: 0.0018201376021915712",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.876256701000178,
"range": "stddev: 0.013505374454089041",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.625500979999742,
"range": "stddev: 0.004946184471776683",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5562508506667048,
"range": "stddev: 0.011391350047046098",
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
"id": "118964bf45ac8c526e9c5070d1780c280c43eaed",
"message": "Raise on nzeta mismatch in input vs mgrid (#357)",
"timestamp": "2025-07-11T10:24:25Z",
"tree_id": "4dce3cadcc17f0fd24fa7713f45902be4ece7076",
"url": "https://github.com/proximafusion/vmecpp/commit/118964bf45ac8c526e9c5070d1780c280c43eaed"
},
"date": 1783553392196,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34196571640004547,
"range": "stddev: 0.001791129628834309",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34115662940007496,
"range": "stddev: 0.002393329746179079",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.404289327000015,
"range": "stddev: 0.0030093931112634194",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7278404589999354,
"range": "stddev: 0.027302753331594504",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.474243918666692,
"range": "stddev: 0.0007298863492875398",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.542503016333285,
"range": "stddev: 0.010022362674946664",
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
"id": "21f1d7e3192a1785e8e3d96992c8b849e8811e37",
"message": "fsqt tracks force balance without pre-conditioning (#328)",
"timestamp": "2025-07-12T08:16:47Z",
"tree_id": "30846c3791dd0b47aefa800cc39b87db5deccffd",
"url": "https://github.com/proximafusion/vmecpp/commit/21f1d7e3192a1785e8e3d96992c8b849e8811e37"
},
"date": 1783553392244,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3332063607999771,
"range": "stddev: 0.00196727910298499",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3353339841999968,
"range": "stddev: 0.0012855706117301327",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4161565263333007,
"range": "stddev: 0.0017016141793929115",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.750206621333367,
"range": "stddev: 0.028657419521694812",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.570253752333352,
"range": "stddev: 0.04475217281619132",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5497055776667519,
"range": "stddev: 0.014622959238812513",
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
"id": "df6327195dfddde264d61b3df30c2cc358c45bc6",
"message": "Execute python -m vmecpp as vmecpp from CLI (#354)",
"timestamp": "2025-07-12T08:31:23Z",
"tree_id": "aa03de2dede7d8b89088e2dc12212c06c7f47be2",
"url": "https://github.com/proximafusion/vmecpp/commit/df6327195dfddde264d61b3df30c2cc358c45bc6"
},
"date": 1783553392773,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33352148500002843,
"range": "stddev: 0.002281428833087862",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.333872577000011,
"range": "stddev: 0.0021602178483926255",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4141152403333308,
"range": "stddev: 0.0010386944470298302",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.728035511333322,
"range": "stddev: 0.02564559717685629",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.528775290333366,
"range": "stddev: 0.017453031862658775",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.555803080666692,
"range": "stddev: 0.012523497674011563",
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
"id": "5eabd513987af3d511736cd13f4a2cba6e3ba043",
"message": "Prepare asymmetric support infrastructure (#360)",
"timestamp": "2025-07-19T09:00:35+02:00",
"tree_id": "21ff3ed31041158739af20115a665ca1f2716091",
"url": "https://github.com/proximafusion/vmecpp/commit/5eabd513987af3d511736cd13f4a2cba6e3ba043"
},
"date": 1783553392406,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33514912640002875,
"range": "stddev: 0.0010920671811077822",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3320568947999618,
"range": "stddev: 0.0005412864381938483",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4206349556666282,
"range": "stddev: 0.008904852053442465",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8073746680000418,
"range": "stddev: 0.03648254241010867",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.521676830000084,
"range": "stddev: 0.00942138730644891",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5499754916666006,
"range": "stddev: 0.009830397117034205",
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
"id": "dc46d2c3d26d9498fcf0f01cee4e3037628fcfa1",
"message": "Fix VmecInput default values (#361)",
"timestamp": "2025-07-20T09:01:43Z",
"tree_id": "71f897d5e6db004afa50f825be6cfd8280c4f8c9",
"url": "https://github.com/proximafusion/vmecpp/commit/dc46d2c3d26d9498fcf0f01cee4e3037628fcfa1"
},
"date": 1783553392761,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34657972519994473,
"range": "stddev: 0.006438110443701614",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.340486259599993,
"range": "stddev: 0.005420203695828629",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4120430396665142,
"range": "stddev: 0.002292698106032353",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.799032860666633,
"range": "stddev: 0.026512409063907814",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.566856817333397,
"range": "stddev: 0.026170861717890702",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5530637813333972,
"range": "stddev: 0.020189643823819117",
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
"id": "f0f7059ee19cc3f9a676212653961ce10a3079ff",
"message": "Improved Sphinx docs (Cleaned up docstrings and config) (#358)",
"timestamp": "2025-07-20T09:32:35Z",
"tree_id": "57c0e775a4dfac4eb2d43fa52fea713486d2cc48",
"url": "https://github.com/proximafusion/vmecpp/commit/f0f7059ee19cc3f9a676212653961ce10a3079ff"
},
"date": 1783553392801,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3467083128000013,
"range": "stddev: 0.0020452486516714056",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34525434919996767,
"range": "stddev: 0.0013562703342518778",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4165346559999914,
"range": "stddev: 0.008582115515406812",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8244921356667114,
"range": "stddev: 0.015437263912675449",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.585502483666687,
"range": "stddev: 0.002165603648520968",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5539570656667365,
"range": "stddev: 0.010835200129628414",
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
"id": "0a9512f93d40a96b83742f1dfae5b455a2f231ec",
"message": "Fix makegrid's write to NetCDF file method (#364)",
"timestamp": "2025-07-23T20:28:46Z",
"tree_id": "1ef66a63e0ba0e423ba89a2b6d296d44a30b6a27",
"url": "https://github.com/proximafusion/vmecpp/commit/0a9512f93d40a96b83742f1dfae5b455a2f231ec"
},
"date": 1783553392173,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35731801639990407,
"range": "stddev: 0.007597989734724486",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3609371724000539,
"range": "stddev: 0.010757246004130225",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4222874109999186,
"range": "stddev: 0.005853639464807429",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.818138796666593,
"range": "stddev: 0.01819905994344722",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.71033245566673,
"range": "stddev: 0.08177712889258597",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5593367723332296,
"range": "stddev: 0.010902422264619698",
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
"id": "4f4db0e3e0eb5aeefda6be06f29f0d654e0d8df4",
"message": "CLI main script terminates gracefully (#365)",
"timestamp": "2025-07-23T23:32:43+02:00",
"tree_id": "80d3bc010bf8ce5d8dfc060044cfa428c6b058d4",
"url": "https://github.com/proximafusion/vmecpp/commit/4f4db0e3e0eb5aeefda6be06f29f0d654e0d8df4"
},
"date": 1783553392358,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35701384859994506,
"range": "stddev: 0.003195913484513716",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3574131548000423,
"range": "stddev: 0.005716820982014248",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4151588926667198,
"range": "stddev: 0.0013711445475913938",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.846536513333376,
"range": "stddev: 0.0364843554541366",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.688440685999922,
"range": "stddev: 0.04698637375325315",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.557026696666829,
"range": "stddev: 0.01029749785154731",
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
"id": "c6950322bd162bef45f35a2a781d6db7f26a5b5d",
"message": "Faster cibuildwheel (#369)",
"timestamp": "2025-07-24T11:46:42+02:00",
"tree_id": "cd92c0d69b93349de399978bd972929b4456ecda",
"url": "https://github.com/proximafusion/vmecpp/commit/c6950322bd162bef45f35a2a781d6db7f26a5b5d"
},
"date": 1783553392678,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3599458562000109,
"range": "stddev: 0.005751105840531266",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3600571415999184,
"range": "stddev: 0.008935431342492842",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.420066294999894,
"range": "stddev: 0.0018163780997798513",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8505623126667765,
"range": "stddev: 0.013154830873685257",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.719256166666733,
"range": "stddev: 0.04030689708265097",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5609254063333537,
"range": "stddev: 0.011294619770950507",
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
"id": "ea303e348e76ff7fbe4b787efc63fb3a98a3875f",
"message": "Human readable termination reason (#366)",
"timestamp": "2025-07-24T16:47:27+02:00",
"tree_id": "fc6b44b44506bf3bbc417899136154ee5162cf46",
"url": "https://github.com/proximafusion/vmecpp/commit/ea303e348e76ff7fbe4b787efc63fb3a98a3875f"
},
"date": 1783553392785,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36755882020006536,
"range": "stddev: 0.009116931251936914",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3679238751999037,
"range": "stddev: 0.00496119042476122",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4272744586666402,
"range": "stddev: 0.002241232901733601",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8907295273334057,
"range": "stddev: 0.035907273777146656",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.668412071999986,
"range": "stddev: 0.02996002442559058",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.557568890666668,
"range": "stddev: 0.011657994435233193",
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
"id": "8c4075f7f10593ecfdde3b60b5035eb6550661ef",
"message": "Added missing lasym terms to VmecWout (#368)",
"timestamp": "2025-07-24T16:51:49+02:00",
"tree_id": "8ab59ceaa50e1c6f1f6fafc6f3b4c6241f5ae5e4",
"url": "https://github.com/proximafusion/vmecpp/commit/8c4075f7f10593ecfdde3b60b5035eb6550661ef"
},
"date": 1783553392520,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.37585152120009296,
"range": "stddev: 0.011861277439707928",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.37950791140010554,
"range": "stddev: 0.00931820584798359",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4212967683332256,
"range": "stddev: 0.0018927474536362136",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8970052326665914,
"range": "stddev: 0.01747401053881371",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.695537932666562,
"range": "stddev: 0.023642116297595647",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.558122482999958,
"range": "stddev: 0.008992781789491004",
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
"id": "dd3ad47db769055d6d4ec91b91f46c90022b8122",
"message": "exact profile evaluation (#370)",
"timestamp": "2025-07-24T18:23:08+02:00",
"tree_id": "174125d88811d4405e2a7b3f68491bf20c976a5f",
"url": "https://github.com/proximafusion/vmecpp/commit/dd3ad47db769055d6d4ec91b91f46c90022b8122"
},
"date": 1783553392765,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36440476479992867,
"range": "stddev: 0.01155085248999305",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36945414520005215,
"range": "stddev: 0.015663309169646613",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4176588199999667,
"range": "stddev: 0.004942239311913939",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8617481096668294,
"range": "stddev: 0.0519071072903602",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.549864978333213,
"range": "stddev: 0.02453984766626418",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5540804519999558,
"range": "stddev: 0.015857410765218207",
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
"id": "008e83d5c6386352a149b6a114ccc07519ecfddf",
"message": "Backwards compatibility with very old VMEC files (#371)",
"timestamp": "2025-07-24T18:26:47+02:00",
"tree_id": "58d9be2db28e9675875b70c548af705f6b102981",
"url": "https://github.com/proximafusion/vmecpp/commit/008e83d5c6386352a149b6a114ccc07519ecfddf"
},
"date": 1783553392121,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34274925880008594,
"range": "stddev: 0.00918608817198301",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3404729255999882,
"range": "stddev: 0.006015880094136578",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.410462654666541,
"range": "stddev: 0.0009511606368635533",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7952182129999223,
"range": "stddev: 0.01784842966872587",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.506012157333316,
"range": "stddev: 0.010411691393929773",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.557066300000012,
"range": "stddev: 0.02229303083742233",
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
"id": "0474522411fd92402908b6438b5c28d320d8f1b3",
"message": "Add UV to pypi_publish CI (#372)",
"timestamp": "2025-07-24T18:58:59+02:00",
"tree_id": "f9c5ae8553a7bc7f151bd3753664a9b285c93292",
"url": "https://github.com/proximafusion/vmecpp/commit/0474522411fd92402908b6438b5c28d320d8f1b3"
},
"date": 1783553392147,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.342828780399941,
"range": "stddev: 0.002429154914257731",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33879493039994485,
"range": "stddev: 0.003087404956002712",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4156190659999386,
"range": "stddev: 0.005036971818531349",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7906579533334175,
"range": "stddev: 0.019162234311758876",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.553722028333445,
"range": "stddev: 0.0144987262237043",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5557370309999594,
"range": "stddev: 0.022088179109130433",
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
"id": "b35ea2d27022d2c64974392ca5f4c68a84495c86",
"message": "Default factories to silence warnings (#373)",
"timestamp": "2025-07-25T17:00:14+02:00",
"tree_id": "29800639176708dc52d9de59671c8069ba9d5fc3",
"url": "https://github.com/proximafusion/vmecpp/commit/b35ea2d27022d2c64974392ca5f4c68a84495c86"
},
"date": 1783553392627,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3428676975999224,
"range": "stddev: 0.004341969727483269",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3504215650001242,
"range": "stddev: 0.006488542271298017",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4127814273332053,
"range": "stddev: 0.003681035449113779",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.786638866333457,
"range": "stddev: 0.010845670197492475",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.56796384699995,
"range": "stddev: 0.04103217162947515",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5579187980000218,
"range": "stddev: 0.01639788372106129",
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
"id": "4011a876c9c187bf3ab590e897d76c03390fb5c5",
"message": "Add CLAUDE.md (#375)",
"timestamp": "2025-07-29T15:34:19Z",
"tree_id": "07f9754e984ada5e852f8f490e74ba0316b1f53d",
"url": "https://github.com/proximafusion/vmecpp/commit/4011a876c9c187bf3ab590e897d76c03390fb5c5"
},
"date": 1783553392317,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34593397240005286,
"range": "stddev: 0.0022935462269762317",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3427719418001288,
"range": "stddev: 0.0011226646761290687",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.409263863666638,
"range": "stddev: 0.007611717092343739",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8033713849999913,
"range": "stddev: 0.02823709351783905",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.540962794666635,
"range": "stddev: 0.011188681879386767",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5538800726665916,
"range": "stddev: 0.00865164973857695",
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
"id": "425dc0812adf267e57d6f0a39eb39da387f0a733",
"message": "fix AGENTS.md instructions for Python code (#377)",
"timestamp": "2025-08-04T08:18:41Z",
"tree_id": "eb1c6f6f679f279d2dd1d33a24fef6d851fd2b82",
"url": "https://github.com/proximafusion/vmecpp/commit/425dc0812adf267e57d6f0a39eb39da387f0a733"
},
"date": 1783553392332,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3430050834001122,
"range": "stddev: 0.0026293361358593332",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34267434139992475,
"range": "stddev: 0.004032006780489654",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4118567850000545,
"range": "stddev: 0.007408927018712915",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.7910403383333082,
"range": "stddev: 0.0419654535302629",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.559554604333243,
"range": "stddev: 0.019337969061462772",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5538506023335685,
"range": "stddev: 0.012246737713131476",
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
"id": "cfe73928d34e994d16ce6aa67277d4172a631d52",
"message": "Fix a race condition in ijacobian (#379)",
"timestamp": "2025-08-11T12:53:17Z",
"tree_id": "722d191a11a54da0ffc6a8230d58eeb7f2f8dfa7",
"url": "https://github.com/proximafusion/vmecpp/commit/cfe73928d34e994d16ce6aa67277d4172a631d52"
},
"date": 1783553392727,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34071177639998496,
"range": "stddev: 0.004095398727541133",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34773964479991265,
"range": "stddev: 0.003179227487772644",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4144625410000724,
"range": "stddev: 0.0013513693952363387",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8293856516667497,
"range": "stddev: 0.01886779663625272",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.610448479333476,
"range": "stddev: 0.03238146409074602",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.551482203333156,
"range": "stddev: 0.008690309949604106",
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
"id": "5e161a13e42219a05ea4c72e53527e7f856ed49a",
"message": "Use hot-restart for Fourier resolution increments. (#378)",
"timestamp": "2025-08-12T18:17:39Z",
"tree_id": "27e1f878609c5fa81e32f3932bded0787e841577",
"url": "https://github.com/proximafusion/vmecpp/commit/5e161a13e42219a05ea4c72e53527e7f856ed49a"
},
"date": 1783553392404,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34739159279988596,
"range": "stddev: 0.0011857958243616216",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.347958168400055,
"range": "stddev: 0.0031436947543203035",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4182228693333248,
"range": "stddev: 0.009894727375995007",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.812815206666528,
"range": "stddev: 0.013387580765769302",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.57493460533336,
"range": "stddev: 0.024025971184381725",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5474284443333393,
"range": "stddev: 0.013410343917204552",
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
"id": "f08ba944041900068a6801ed2f12eb182d62d8e9",
"message": "Remove gcc and cmake from required MacOS packages (#390)",
"timestamp": "2025-09-15T08:06:41Z",
"tree_id": "31d4876375df38f56e80c811ace8e5d4e54d23a2",
"url": "https://github.com/proximafusion/vmecpp/commit/f08ba944041900068a6801ed2f12eb182d62d8e9"
},
"date": 1783553392799,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3507104786001037,
"range": "stddev: 0.006496468539487251",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3469542123998508,
"range": "stddev: 0.002919218684848835",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4204121626667074,
"range": "stddev: 0.01396823793094762",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.808957937666643,
"range": "stddev: 0.011592091870095394",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.602808108000014,
"range": "stddev: 0.0033545055109236994",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.583726922333426,
"range": "stddev: 0.050005532859517976",
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
"id": "9bf8324096284391ab45054cfe02961c1ba823e3",
"message": "Fix a race condition on liter_flag (#386)",
"timestamp": "2025-09-15T08:51:22Z",
"tree_id": "becdc4d45a8bef943bf5bcb9deef778fd13bc297",
"url": "https://github.com/proximafusion/vmecpp/commit/9bf8324096284391ab45054cfe02961c1ba823e3"
},
"date": 1783553392561,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3451751175998652,
"range": "stddev: 0.0012503769315781524",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3466337951999776,
"range": "stddev: 0.0016876827241104168",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.417555214999993,
"range": "stddev: 0.0032957392278242146",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8478953396668962,
"range": "stddev: 0.029460959528954854",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.66785831766659,
"range": "stddev: 0.01870830133397579",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5551288603334494,
"range": "stddev: 0.009287426277688119",
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
"id": "2fb7685a7f7f4f0d9099e813dd153a00fcf7443b",
"message": "Fix a race condition on iter2. (#387)",
"timestamp": "2025-09-15T08:51:22Z",
"tree_id": "4ed9ef7a34bab252f07824813290b146b1c0989e",
"url": "https://github.com/proximafusion/vmecpp/commit/2fb7685a7f7f4f0d9099e813dd153a00fcf7443b"
},
"date": 1783553392277,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3399288045999128,
"range": "stddev: 0.0014840951228959562",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.345993228000043,
"range": "stddev: 0.003434541431662712",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4105595536666442,
"range": "stddev: 0.002520320996299611",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8225739530001497,
"range": "stddev: 0.028541304178355167",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.59796778166689,
"range": "stddev: 0.1092759605538792",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5509324500000428,
"range": "stddev: 0.010967195391198201",
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
"id": "d64815371969d22e12e64d825dfe87bef5b6d44f",
"message": "Fix more race conditions identified by TSan/Archer (#388)",
"timestamp": "2025-09-15T08:51:23Z",
"tree_id": "6991fbd785f5347d54a548c1343a5c8e3851fff6",
"url": "https://github.com/proximafusion/vmecpp/commit/d64815371969d22e12e64d825dfe87bef5b6d44f"
},
"date": 1783553392744,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35113158979993386,
"range": "stddev: 0.0017042211438475356",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35092557760008275,
"range": "stddev: 0.0026448953404990413",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4181554540000434,
"range": "stddev: 0.0025688299604891877",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.838717045333245,
"range": "stddev: 0.013697383620695738",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.558750350333261,
"range": "stddev: 0.041113756818084615",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5500271170000512,
"range": "stddev: 0.006054847116577298",
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
"id": "9386a845bbc064d2d338375fd6a855da443dd12c",
"message": "Fix a data race on get_delbsq (#389)",
"timestamp": "2025-09-15T08:51:23Z",
"tree_id": "e2eb179b839020d725444f1d35a1467aab620ccf",
"url": "https://github.com/proximafusion/vmecpp/commit/9386a845bbc064d2d338375fd6a855da443dd12c"
},
"date": 1783553392547,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3414172060000055,
"range": "stddev: 0.004063427360872506",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34120627920001423,
"range": "stddev: 0.0021301778011875885",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4226373853333218,
"range": "stddev: 0.0010098095598572202",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8435746060002507,
"range": "stddev: 0.015428224851364454",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.54421204166662,
"range": "stddev: 0.025127040546090263",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5506812346666266,
"range": "stddev: 0.012518561596244397",
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
"id": "908d753d76b1226c5ba68e2941e697d25695f9df",
"message": "Get the TSan/Archer setup working again (#384)",
"timestamp": "2025-09-15T15:31:47Z",
"tree_id": "ff751500c48609c6a73eaf9eeabb812e3a78e360",
"url": "https://github.com/proximafusion/vmecpp/commit/908d753d76b1226c5ba68e2941e697d25695f9df"
},
"date": 1783553392542,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3385128819999409,
"range": "stddev: 0.0012896785857968226",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34022272100010015,
"range": "stddev: 0.0020601306554040564",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4145738896665232,
"range": "stddev: 0.0009987236802956295",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8300698883332793,
"range": "stddev: 0.01474872437408914",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.490945588332883,
"range": "stddev: 0.02401556843813087",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5442900379997202,
"range": "stddev: 0.013813624797403463",
"extra": "rounds: 3"
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
"date": 1783553392356,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3427526969997416,
"range": "stddev: 0.0037137129174078264",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3395528545999696,
"range": "stddev: 0.0015751044753622297",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4173021240003436,
"range": "stddev: 0.004977363751595398",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8087441693329915,
"range": "stddev: 0.022114692007271664",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.528337080333282,
"range": "stddev: 0.005225098777489861",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5455334833335048,
"range": "stddev: 0.008704098733522403",
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
"id": "9d1944f6d40eadf695ec6e2c99bebafc40369772",
"message": "Use hot-restart for Fourier resolution increments. (#381)",
"timestamp": "2025-12-04T17:19:18+01:00",
"tree_id": "1c0a23dbaef479e8a08ec146673d2d7b678929f3",
"url": "https://github.com/proximafusion/vmecpp/commit/9d1944f6d40eadf695ec6e2c99bebafc40369772"
},
"date": 1783553392566,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.33787450380004885,
"range": "stddev: 0.004845144727921016",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.33782653260004736,
"range": "stddev: 0.0021013232169023524",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.421688858666433,
"range": "stddev: 0.00941905881526709",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.809899442000036,
"range": "stddev: 0.02415611501766379",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.58089396666704,
"range": "stddev: 0.01281385762411331",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5514049553333582,
"range": "stddev: 0.022238870550150443",
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
"id": "001b7607f1b7da01719e6fa7e66ec43aa43f6580",
"message": "Refactor HandoverStorage to use Eigen::RowMatrixXd for improved performance (#396)",
"timestamp": "2025-12-05T19:57:21+09:00",
"tree_id": "9435d64702e3029a2e0fc2c335dfc7b4db1861ef",
"url": "https://github.com/proximafusion/vmecpp/commit/001b7607f1b7da01719e6fa7e66ec43aa43f6580"
},
"date": 1783553392116,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.350812556800156,
"range": "stddev: 0.008011682743769172",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3530551325999113,
"range": "stddev: 0.0025375065340415104",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4234099256664194,
"range": "stddev: 0.004800459672788693",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8796167776666457,
"range": "stddev: 0.0054702395360923944",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.691937250999823,
"range": "stddev: 0.022061008130690497",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5571075606664333,
"range": "stddev: 0.01393341798271525",
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
"id": "04f16f531ead8995b1f4a5a5f92024e82f83f86a",
"message": "szip requirement in netcdf, limit version (#397)",
"timestamp": "2025-12-05T11:29:26Z",
"tree_id": "9abe1815e47962490acc60251edea222b4d2b353",
"url": "https://github.com/proximafusion/vmecpp/commit/04f16f531ead8995b1f4a5a5f92024e82f83f86a"
},
"date": 1783553392150,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27688863120000634,
"range": "stddev: 0.01083082442771004",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.26931468579999773,
"range": "stddev: 0.004362370703343935",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1588312969999872,
"range": "stddev: 0.002995775250023178",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.248269462333326,
"range": "stddev: 0.022588489814022176",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.960781912000016,
"range": "stddev: 0.00722646533160708",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.353828257000013,
"range": "stddev: 0.009586286609740466",
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
"id": "54700786bb8602a839e30115576a8387aab486be",
"message": "Add libaec dpendency for szip support (requirement in newest netcdf version) (#398)",
"timestamp": "2025-12-05T18:19:53+01:00",
"tree_id": "e8fbf34e248547275106af4a779d70fa5adc75a2",
"url": "https://github.com/proximafusion/vmecpp/commit/54700786bb8602a839e30115576a8387aab486be"
},
"date": 1783553392375,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28130397459999584,
"range": "stddev: 0.001746094843156711",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2800250681999955,
"range": "stddev: 0.0027315202591704363",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1558906199999985,
"range": "stddev: 0.003187730823460502",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2790987953333492,
"range": "stddev: 0.01876727263910286",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.9493063546666844,
"range": "stddev: 0.030530375803935694",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3463073360000142,
"range": "stddev: 0.012126940621015811",
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
"id": "cd4ec83601b25018a6e13ecfb9aeb2fc31a0a9ab",
"message": "Handle bytes in VmecWOut netCDF output (#401)",
"timestamp": "2026-01-12T19:06:36+09:00",
"tree_id": "d1c03660c939bb6db95cb73123de8a2cacc5f1e1",
"url": "https://github.com/proximafusion/vmecpp/commit/cd4ec83601b25018a6e13ecfb9aeb2fc31a0a9ab"
},
"date": 1783553392716,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27803331040000784,
"range": "stddev: 0.0017392842963929007",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2734988232000092,
"range": "stddev: 0.0005957125142255616",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.159296937000022,
"range": "stddev: 0.0009806563110383516",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2659553966666786,
"range": "stddev: 0.0362151269585526",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.919466303333347,
"range": "stddev: 0.021800237125337303",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3437173296666742,
"range": "stddev: 0.008500413332541675",
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
"id": "fe8cd5d192100810e746f1036b3853fc5b7c78db",
"message": "Add lrfp_logical_ flag to wout for Fortran VMEC compatibility (Python-only) (#399)",
"timestamp": "2026-01-12T11:55:04+01:00",
"tree_id": "26749dd25d88ec5cc717fecb080b8dd9bc390592",
"url": "https://github.com/proximafusion/vmecpp/commit/fe8cd5d192100810e746f1036b3853fc5b7c78db"
},
"date": 1783553392830,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27804956799996033,
"range": "stddev: 0.005417717746412694",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2703120886000079,
"range": "stddev: 0.002281156300185186",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1570192140000017,
"range": "stddev: 0.001935144955896926",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2474143866666814,
"range": "stddev: 0.028597603179161952",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.910332206333313,
"range": "stddev: 0.015958960237689983",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3423627636666804,
"range": "stddev: 0.007677135362398083",
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
"id": "b226b09d694b01edf090e2f2abb26b352c4ad15d",
"message": "CIbuildwheel update. Minimum supported linux version changed from manylinux2014 to manylinux_2_28 (#403)",
"timestamp": "2026-01-12T16:18:25+01:00",
"tree_id": "2192fb8a5dbf6f01bfb1c2566d46d71a59af5073",
"url": "https://github.com/proximafusion/vmecpp/commit/b226b09d694b01edf090e2f2abb26b352c4ad15d"
},
"date": 1783553392617,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2746804284000291,
"range": "stddev: 0.003426381992352945",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27608547780000664,
"range": "stddev: 0.002885884600494845",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.16683716,
"range": "stddev: 0.007147407741377675",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.248114166666634,
"range": "stddev: 0.050624946230686745",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.954364475666655,
"range": "stddev: 0.029073728579629553",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3476149483333302,
"range": "stddev: 0.013112326655621212",
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
"id": "4c71e83b9705cebd3c80d05d71016a9cfb8ffaf0",
"message": "CIbuildwheel MacOS fix (#404)",
"timestamp": "2026-01-12T16:42:26+01:00",
"tree_id": "e0c804c36ca6f1d372c72a6768c60a2733bffa01",
"url": "https://github.com/proximafusion/vmecpp/commit/4c71e83b9705cebd3c80d05d71016a9cfb8ffaf0"
},
"date": 1783553392351,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.29327007199995025,
"range": "stddev: 0.003338322045764664",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.29125489939999627,
"range": "stddev: 0.004682663611679869",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1590332793333953,
"range": "stddev: 0.0013804619004096441",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.309928758666615,
"range": "stddev: 0.021130810829281133",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 8.00390711033333,
"range": "stddev: 0.04347208700510415",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3438519713333033,
"range": "stddev: 0.011015337717961562",
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
"id": "c44c00a514c568170f3ec892558ce64441ddd00d",
"message": "Pin archlinux test python version to 3.13 (#405)",
"timestamp": "2026-01-12T21:02:11Z",
"tree_id": "2ad6d163736b4c2de9fffef66bfb61fcec733062",
"url": "https://github.com/proximafusion/vmecpp/commit/c44c00a514c568170f3ec892558ce64441ddd00d"
},
"date": 1783553392666,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2796710110000049,
"range": "stddev: 0.0007441242442201047",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27098329880000166,
"range": "stddev: 0.0015917979689926184",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1612890676666818,
"range": "stddev: 0.002279571145088101",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.289737495333346,
"range": "stddev: 0.04923361800516518",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.957872753666682,
"range": "stddev: 0.007645190463782685",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3425215983332919,
"range": "stddev: 0.007991551952196526",
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
"id": "f604ace31ba368141ef5f10ba755f2a0eb97aeaf",
"message": "Fix simsopt compatibility with change_resolution (#402)",
"timestamp": "2026-02-11T09:48:57+09:00",
"tree_id": "ff8d2a393ec78192a127574988c8227a8f619e76",
"url": "https://github.com/proximafusion/vmecpp/commit/f604ace31ba368141ef5f10ba755f2a0eb97aeaf"
},
"date": 1783553392811,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2767413949999991,
"range": "stddev: 0.0018574497540195672",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2783325033999972,
"range": "stddev: 0.0031702085722184114",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1605488773333643,
"range": "stddev: 0.0042916258140965596",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2601703040000225,
"range": "stddev: 0.030778189020520548",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 8.049534608000007,
"range": "stddev: 0.0771346275994518",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.358553148666677,
"range": "stddev: 0.011957609669395965",
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
"id": "80f1ade9b830417e54e53ffb5a8596fcefa5e1ca",
"message": "Fix #164: rebind spans after copy/move/assignment (#400)",
"timestamp": "2026-02-11T21:24:14+09:00",
"tree_id": "fdccc6e689baa4806259788395b97610aab4d060",
"url": "https://github.com/proximafusion/vmecpp/commit/80f1ade9b830417e54e53ffb5a8596fcefa5e1ca"
},
"date": 1783553392489,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27723312359999,
"range": "stddev: 0.004372732875024623",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2785346352000033,
"range": "stddev: 0.0028534699533128635",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1607165900000684,
"range": "stddev: 0.0034840318228996354",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.256680827000082,
"range": "stddev: 0.01652049024451391",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.928129236000056,
"range": "stddev: 0.017703918155895535",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.349184305666692,
"range": "stddev: 0.010987415359252684",
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
"id": "140cd686f1602c795ed0c96f617765249f1fd6fb",
"message": "Test copy semantics (#416)",
"timestamp": "2026-02-11T13:36:59+01:00",
"tree_id": "e5fab3fba1e767ff4ab40bb0fd55973a9322e26e",
"url": "https://github.com/proximafusion/vmecpp/commit/140cd686f1602c795ed0c96f617765249f1fd6fb"
},
"date": 1783553392205,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27580572140000187,
"range": "stddev: 0.0017414535927514436",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27683344020001643,
"range": "stddev: 0.0019836392287032995",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1647635996666093,
"range": "stddev: 0.0012189475815599364",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2593232246666353,
"range": "stddev: 0.041398968964105255",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.927837546333346,
"range": "stddev: 0.06208962499552816",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3550118253334251,
"range": "stddev: 0.011685763861410856",
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
"id": "2eec4eb7b9c12139c2ec3f61066fd0203bb86576",
"message": "Remove copyFrom constructor, now that copy semantics are working (#417)",
"timestamp": "2026-02-11T14:00:19Z",
"tree_id": "c3149fc1470fe2e4ce5ff8169167bf5bc01a8d85",
"url": "https://github.com/proximafusion/vmecpp/commit/2eec4eb7b9c12139c2ec3f61066fd0203bb86576"
},
"date": 1783553392275,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2683991682000396,
"range": "stddev: 0.0029384243702155876",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.26608306059997633,
"range": "stddev: 0.0016944782874944535",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1583572983333852,
"range": "stddev: 0.003269422354655004",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2362429423333956,
"range": "stddev: 0.03762257997702585",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.9755722649999825,
"range": "stddev: 0.032720906953505796",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3418111053333632,
"range": "stddev: 0.009432601427197408",
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
"id": "40393aa062f2c29d6387a2dba06587ca0767f338",
"message": "More informative error message. (#419)",
"timestamp": "2026-02-11T17:22:46Z",
"tree_id": "2d9413cf5c65e06a9a04b16f26438c512be4c417",
"url": "https://github.com/proximafusion/vmecpp/commit/40393aa062f2c29d6387a2dba06587ca0767f338"
},
"date": 1783553392320,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27529215839999777,
"range": "stddev: 0.013033395174386663",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2783717115999934,
"range": "stddev: 0.006519957571919242",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1563503149999785,
"range": "stddev: 0.0014801939895535594",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2356401779999637,
"range": "stddev: 0.039398979846479686",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.903118114666692,
"range": "stddev: 0.02519576838295603",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3466121893333518,
"range": "stddev: 0.004478339642139037",
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
"id": "6268991099391a555d47194b051835e5d16d0a25",
"message": "GIL handling to allow Ctrl+C and progressive logging in Jupyter (#415)",
"timestamp": "2026-02-11T22:48:00Z",
"tree_id": "10344098f8352d60b28be3f6f7022f5a8ec59563",
"url": "https://github.com/proximafusion/vmecpp/commit/6268991099391a555d47194b051835e5d16d0a25"
},
"date": 1783553392421,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2732533014000637,
"range": "stddev: 0.0059944258676235904",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.278225134000013,
"range": "stddev: 0.0037034856530886624",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1576291326665948,
"range": "stddev: 0.0012862672488126309",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2044978246666838,
"range": "stddev: 0.03141523950570598",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.887169357999937,
"range": "stddev: 0.04622097790308616",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3411618093333952,
"range": "stddev: 0.009477053004208832",
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
"id": "45c302c6ba8d4c19937ea75c58c1b92d5d282fca",
"message": "Minor ruff lint changes (#421)",
"timestamp": "2026-02-12T00:40:36+01:00",
"tree_id": "1d0648d2c43bc08c192491dbf05b5379165b3131",
"url": "https://github.com/proximafusion/vmecpp/commit/45c302c6ba8d4c19937ea75c58c1b92d5d282fca"
},
"date": 1783553392341,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27520191320004417,
"range": "stddev: 0.0017967378968107256",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27409972140003447,
"range": "stddev: 0.006403545210019309",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1659700436666753,
"range": "stddev: 0.00831729639958611",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2122760380000273,
"range": "stddev: 0.05312095941452526",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.90921546166669,
"range": "stddev: 0.010569011769404326",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.342154978999967,
"range": "stddev: 0.010634551080723044",
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
"id": "010dd486d8e526b57573bb04bc39897ec5e2764b",
"message": "C++ dependency version updates (#422)",
"timestamp": "2026-02-12T01:08:19+01:00",
"tree_id": "8c62b3c7c26410ec37b537291e79af2ac811de3f",
"url": "https://github.com/proximafusion/vmecpp/commit/010dd486d8e526b57573bb04bc39897ec5e2764b"
},
"date": 1783553392124,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2741548773999966,
"range": "stddev: 0.0037858722808162284",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28190988279998236,
"range": "stddev: 0.0033199997824174153",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1577392693332815,
"range": "stddev: 0.0006613928690847962",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2334142540000053,
"range": "stddev: 0.030489337794813096",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.881434702000054,
"range": "stddev: 0.013218081984126901",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.351281074000023,
"range": "stddev: 0.018932507305046094",
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
"id": "b17989ca99f311fe32b9353b1873262e1f0bf2b1",
"message": "add free-boundary method that only uses magnetic field from external coils (#411)",
"timestamp": "2026-02-12T00:35:02Z",
"tree_id": "8c586c94a913b70c8f45b12bea605569a6d2ba57",
"url": "https://github.com/proximafusion/vmecpp/commit/b17989ca99f311fe32b9353b1873262e1f0bf2b1"
},
"date": 1783553392615,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2953673260000869,
"range": "stddev: 0.0016121558479097258",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2784470975999739,
"range": "stddev: 0.005183207941096885",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1647446120000495,
"range": "stddev: 0.003401835796943922",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.3256278806667674,
"range": "stddev: 0.04663248434254203",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.992556352666649,
"range": "stddev: 0.013974296617221521",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.34746965933338,
"range": "stddev: 0.017240550661407894",
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
"id": "0a4025d7c637efa42eed4fc2fee31d838c90def3",
"message": "Pin archlinux Python version in CI to 3.13 (#423)",
"timestamp": "2026-02-12T03:37:13+01:00",
"tree_id": "f56eece3beeaacd72414150f9405c248ead97f2e",
"url": "https://github.com/proximafusion/vmecpp/commit/0a4025d7c637efa42eed4fc2fee31d838c90def3"
},
"date": 1783553392168,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2820518722000088,
"range": "stddev: 0.0032607362205038177",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27944240520000674,
"range": "stddev: 0.001619999307861082",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1570850876666252,
"range": "stddev: 0.004396992236416616",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2728308083332345,
"range": "stddev: 0.01637886396516051",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.928370544666753,
"range": "stddev: 0.074233461171715",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3458105696666582,
"range": "stddev: 0.00735891465263236",
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
"id": "19ae9b8083bad3de0b5a22a9fa0c2a9f62366cfe",
"message": "Usage example for raw profiles (#420)",
"timestamp": "2026-02-12T08:57:32Z",
"tree_id": "00afe8998e196144fd56620fc254ba212bf10fc8",
"url": "https://github.com/proximafusion/vmecpp/commit/19ae9b8083bad3de0b5a22a9fa0c2a9f62366cfe"
},
"date": 1783553392226,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27178892840001934,
"range": "stddev: 0.004145942076789122",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27693006459985553,
"range": "stddev: 0.004400714274552644",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.158302645333303,
"range": "stddev: 0.0005746249933752123",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.195403388999997,
"range": "stddev: 0.036656506107653926",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.94771076666666,
"range": "stddev: 0.0390754819810528",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3414310993334766,
"range": "stddev: 0.010079127239738805",
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
"name": "Philipp Jura\u0161i\u0107",
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
"date": 1783553392194,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27719946220004205,
"range": "stddev: 0.0026231219273154392",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2844899837999947,
"range": "stddev: 0.004772236860583476",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1620570283333411,
"range": "stddev: 0.006731318815146625",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.207486017333546,
"range": "stddev: 0.015818395812033256",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.936476137000075,
"range": "stddev: 0.03606996654703823",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3464736599999014,
"range": "stddev: 0.010550381755499142",
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
"id": "40442f3ff5701f31ff35ce05373da16d4a8962fb",
"message": "Analytical torflux polynomial evaluation (#316)",
"timestamp": "2026-02-15T09:09:46Z",
"tree_id": "986f63a2d53264e145fe443769403d02302ecf7f",
"url": "https://github.com/proximafusion/vmecpp/commit/40442f3ff5701f31ff35ce05373da16d4a8962fb"
},
"date": 1783553392324,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27840902219986674,
"range": "stddev: 0.00481741925178964",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2721865425999567,
"range": "stddev: 0.006922229961910458",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.166360409000087,
"range": "stddev: 0.011218856031555176",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.234517927999908,
"range": "stddev: 0.04984567160880019",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.947596457666653,
"range": "stddev: 0.0287309514825989",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3544781596666933,
"range": "stddev: 0.018964350174844352",
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
"id": "c798c70845a8ad5b5d848abbe249278f6c97f753",
"message": "Add continuous benchmarking (#426)",
"timestamp": "2026-02-25T13:44:39+01:00",
"tree_id": "dc0f6863d3a544bc3fed16e17e42dcebcbef34b3",
"url": "https://github.com/proximafusion/vmecpp/commit/c798c70845a8ad5b5d848abbe249278f6c97f753"
},
"date": 1783553392683,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28133422219998466,
"range": "stddev: 0.007524271585955444",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27105404939993605,
"range": "stddev: 0.0031535559620335957",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1581226996666676,
"range": "stddev: 0.0006217839768822402",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.242762689666506,
"range": "stddev: 0.03964308006051353",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.951556550666585,
"range": "stddev: 0.017024448801089036",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3456289376666366,
"range": "stddev: 0.009553754740499864",
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
"date": 1783553392660,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28178677959995185,
"range": "stddev: 0.010494867478288088",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27317603380006406,
"range": "stddev: 0.0029356147715000713",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1642041086665813,
"range": "stddev: 0.009901787492971097",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2542553066668916,
"range": "stddev: 0.023285341136733195",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.979770196999955,
"range": "stddev: 0.01289285580981142",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3437158919999395,
"range": "stddev: 0.009548226815489407",
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
"date": 1783553392426,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2852689748001467,
"range": "stddev: 0.01700524137882696",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2748945723998986,
"range": "stddev: 0.0018288995651134425",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1590869163333082,
"range": "stddev: 0.0013876142946774067",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.235123223333327,
"range": "stddev: 0.0652292395655371",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.891921270000087,
"range": "stddev: 0.006191920537403139",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3415866239999257,
"range": "stddev: 0.009425391988429063",
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
"date": 1783553392156,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.26753570239998226,
"range": "stddev: 0.004912983956190901",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2770538242000839,
"range": "stddev: 0.006433292859688084",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1586811613333339,
"range": "stddev: 0.0019446339943095525",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.199969920666566,
"range": "stddev: 0.03169643124266902",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.885121854666674,
"range": "stddev: 0.005845269717738095",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.339222019333344,
"range": "stddev: 0.006824510248321884",
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
"date": 1783553392571,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27134490279986495,
"range": "stddev: 0.003885414112380189",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.273322244999963,
"range": "stddev: 0.005074055352056567",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.156832333666595,
"range": "stddev: 0.0012693729529345923",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.225806223666647,
"range": "stddev: 0.06072255921225745",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.9029512453331945,
"range": "stddev: 0.023253189294549382",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3483909403333503,
"range": "stddev: 0.012147756262191244",
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
"date": 1783553392782,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28026416259999676,
"range": "stddev: 0.0018385120206082543",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27106123080002364,
"range": "stddev: 0.00288676113140238",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1559308706667555,
"range": "stddev: 0.0027903891297270784",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.2513030759999615,
"range": "stddev: 0.03450655928141371",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.934769901999971,
"range": "stddev: 0.020359936545686227",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3394920366667975,
"range": "stddev: 0.006192596031949716",
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
"date": 1783553392687,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2683870023999589,
"range": "stddev: 0.00219769235549786",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2678268227999979,
"range": "stddev: 0.0023181968834010062",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1586814429997503,
"range": "stddev: 0.005478930563955616",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.209478608999992,
"range": "stddev: 0.02241324011516474",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.906574517666741,
"range": "stddev: 0.04938083574464899",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3433714053332249,
"range": "stddev: 0.010868981383144967",
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
"id": "cf64eaaeafae5ff91dd8e90a2f538732812be877",
"message": "Fix ci error in pyright.yaml (#440)",
"timestamp": "2026-03-18T14:31:35+01:00",
"tree_id": "922e038285fc3406734ee9fca6b80eb396c0e453",
"url": "https://github.com/proximafusion/vmecpp/commit/cf64eaaeafae5ff91dd8e90a2f538732812be877"
},
"date": 1783553392723,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27812146579999536,
"range": "stddev: 0.004391526502726901",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2951268424000773,
"range": "stddev: 0.0021643862372655257",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.1554811223331853,
"range": "stddev: 0.005219512891470108",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.234922861333265,
"range": "stddev: 0.026707935592387252",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.8846613876668625,
"range": "stddev: 0.014763370759908986",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.341855601333312,
"range": "stddev: 0.01013168250350534",
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
"date": 1783553392322,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3680927278000013,
"range": "stddev: 0.00910574800055526",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3677362440000138,
"range": "stddev: 0.0037924407558135465",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4198940343333295,
"range": "stddev: 0.005134923051484076",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9379009379999843,
"range": "stddev: 0.04320044221137374",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.667200881666664,
"range": "stddev: 0.020333470717291628",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5571799626666614,
"range": "stddev: 0.012118684454494853",
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
"id": "8856974a7b20bccb2a2369c40e44ed5cb610bbd5",
"message": "Globally prevent 3.14 python (#435)",
"timestamp": "2026-03-18T16:06:34+01:00",
"tree_id": "07d365c528c63f38b69a3f0811dd98318c653934",
"url": "https://github.com/proximafusion/vmecpp/commit/8856974a7b20bccb2a2369c40e44ed5cb610bbd5"
},
"date": 1783553392510,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3768895681999879,
"range": "stddev: 0.004500765370909089",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.37075179080001136,
"range": "stddev: 0.015093948340172761",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4174526863333388,
"range": "stddev: 0.007073161186697202",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.90259325400001,
"range": "stddev: 0.04302097187227178",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.690260810666643,
"range": "stddev: 0.006122766751949489",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5566330776666557,
"range": "stddev: 0.018052251365080766",
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
"id": "827c4b0ed5d4c5c4305b79e25b000291ebfe0413",
"message": "DNDEBUG for faster eigen3 performance (#447)",
"timestamp": "2026-03-26T14:50:12+01:00",
"tree_id": "bdac70f73639a81791d026c842d96a260943c70e",
"url": "https://github.com/proximafusion/vmecpp/commit/827c4b0ed5d4c5c4305b79e25b000291ebfe0413"
},
"date": 1783553392493,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3600071946000071,
"range": "stddev: 0.005221703515565845",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35820998940002935,
"range": "stddev: 0.0031171320493722284",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.4023710230000006,
"range": "stddev: 0.008479829754995529",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.6411587793333333,
"range": "stddev: 0.01724690441001684",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.557635535999983,
"range": "stddev: 0.023751746519399576",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5590321593333176,
"range": "stddev: 0.00799037654635681",
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
"id": "5c527e1ec36b148acecd0cf59dc2f8418f6bf791",
"message": "apt get update in docs CI (#448)",
"timestamp": "2026-03-26T15:32:42+01:00",
"tree_id": "25ec0c21ffc071b90ef07bd6beb8f7c6f95541ef",
"url": "https://github.com/proximafusion/vmecpp/commit/5c527e1ec36b148acecd0cf59dc2f8418f6bf791"
},
"date": 1783553392397,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36329334560002735,
"range": "stddev: 0.0032410480906690086",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.370677671000044,
"range": "stddev: 0.011109139687585852",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3959886626666578,
"range": "stddev: 0.0025318701266961396",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.6465916573332984,
"range": "stddev: 0.013802532594176702",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.595521433999997,
"range": "stddev: 0.012795108000890995",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5656103533333408,
"range": "stddev: 0.02430497411498743",
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
"id": "a5682f80e0acc586de0b0ae8f5b751dc12f89238",
"message": "Fix auto benchmarks (#449)",
"timestamp": "2026-03-26T15:33:28+01:00",
"tree_id": "26c20171c6d12ed2706e72c307c69cdde94da07c",
"url": "https://github.com/proximafusion/vmecpp/commit/a5682f80e0acc586de0b0ae8f5b751dc12f89238"
},
"date": 1783553392595,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36341606880000654,
"range": "stddev: 0.0020200740255867803",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3577680121999947,
"range": "stddev: 0.003954409992742994",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3966188613333845,
"range": "stddev: 0.013684571862793636",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.6521331973333417,
"range": "stddev: 0.013158139961297449",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.546085470333347,
"range": "stddev: 0.0038139803758735712",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.566346574000022,
"range": "stddev: 0.005168614428866018",
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
"date": 1783553392775,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3628868627999964,
"range": "stddev: 0.006498003826997726",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3652166591999958,
"range": "stddev: 0.003742172201663978",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3961089986666668,
"range": "stddev: 0.003918035739631659",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.662711923999988,
"range": "stddev: 0.018740300100825703",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.564071297333308,
"range": "stddev: 0.01948009047501317",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5646237010000352,
"range": "stddev: 0.012824519440726125",
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
"date": 1783553392207,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3678549578000002,
"range": "stddev: 0.004739001691291724",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3615809799999624,
"range": "stddev: 0.00391144548508635",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3922724049999335,
"range": "stddev: 0.0021361750655160385",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.6612973343333883,
"range": "stddev: 0.006107993415648715",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.590440824333276,
"range": "stddev: 0.01706753719427413",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5585502126666597,
"range": "stddev: 0.00970350821662228",
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
"date": 1783553392254,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3640405417999773,
"range": "stddev: 0.0029330458885039865",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3694184727999527,
"range": "stddev: 0.0021089500598226183",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3927307933333093,
"range": "stddev: 0.0019805212796062447",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.6371191089999684,
"range": "stddev: 0.020748330408703042",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.548454240666615,
"range": "stddev: 0.005168024712897979",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5638627543334376,
"range": "stddev: 0.014397207343388431",
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
"date": 1783553392588,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3629604495999956,
"range": "stddev: 0.006161582503823029",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36266844480001054,
"range": "stddev: 0.005492676777858237",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.710755117333368,
"range": "stddev: 0.009062262017898033",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.2262225909999716,
"range": "stddev: 0.01929115815292308",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.972140146666712,
"range": "stddev: 0.007517078325618077",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.558899429333375,
"range": "stddev: 0.012709783032932236",
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
"id": "69142e7a02b5457b323fff1fc4512a1abf0df591",
"message": "Vmec constructor -> Factory for error handling during initialization. (#244)",
"timestamp": "2026-03-27T01:38:24+01:00",
"tree_id": "d1c86ca8ebee63e151d20fc645b5bae901836a2d",
"url": "https://github.com/proximafusion/vmecpp/commit/69142e7a02b5457b323fff1fc4512a1abf0df591"
},
"date": 1783553392431,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.365291108800011,
"range": "stddev: 0.006532185377147334",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3696483333999822,
"range": "stddev: 0.007519146132341692",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.691662826666743,
"range": "stddev: 0.005400632360161222",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.251531475000017,
"range": "stddev: 0.037503847480775315",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.920654369000053,
"range": "stddev: 0.024418294096987463",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5581823483333512,
"range": "stddev: 0.01062882887326883",
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
"id": "2d8888e1defbd4d5f2949047dfa742495854ca1f",
"message": "SImsopt keep dependency chain even when resolution is inconsistent (#436)",
"timestamp": "2026-03-27T01:57:22+01:00",
"tree_id": "3b8c2e550b838b75d8fc95055d8f597c69aa717d",
"url": "https://github.com/proximafusion/vmecpp/commit/2d8888e1defbd4d5f2949047dfa742495854ca1f"
},
"date": 1783553392268,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3622441071999674,
"range": "stddev: 0.004597655795232092",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3680778310000278,
"range": "stddev: 0.011710700214663564",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.6979073526665616,
"range": "stddev: 0.0026837629739815353",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.250365513333311,
"range": "stddev: 0.04255766835573907",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.887606565000018,
"range": "stddev: 0.004101650698010256",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.553699665999981,
"range": "stddev: 0.011598610478181802",
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
"id": "1ba362029017a7ee7ffc3dc1c8ed77fb8e209a1a",
"message": "Migrate ideal MHD core to eigen3 (#452)",
"timestamp": "2026-03-27T09:50:07+01:00",
"tree_id": "1595f0d2dadf78d0732658ccb4881db013369167",
"url": "https://github.com/proximafusion/vmecpp/commit/1ba362029017a7ee7ffc3dc1c8ed77fb8e209a1a"
},
"date": 1783553392228,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3658859138000025,
"range": "stddev: 0.0035067782148146486",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3577943278000021,
"range": "stddev: 0.003115703706501593",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.6970449303333528,
"range": "stddev: 0.0037958787237123235",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.304336221333339,
"range": "stddev: 0.025260630661529206",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.891939375000069,
"range": "stddev: 0.0036040164250360617",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5596292026667318,
"range": "stddev: 0.010944477334609034",
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
"id": "df0547f8fbd75d5c34354b4148de251a5f365491",
"message": "Simplify a few more ideal mhd expressions with Eigen3 (#453)",
"timestamp": "2026-03-27T10:07:38+01:00",
"tree_id": "59491cd9d769f611dc4d46f48ef19e3d7f9b388e",
"url": "https://github.com/proximafusion/vmecpp/commit/df0547f8fbd75d5c34354b4148de251a5f365491"
},
"date": 1783553392770,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36190707000000655,
"range": "stddev: 0.004470655601024206",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35930766720002794,
"range": "stddev: 0.004218857262051831",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.6432259333332695,
"range": "stddev: 0.004882324255067332",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.254965945999932,
"range": "stddev: 0.00935984407178577",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.840540434666536,
"range": "stddev: 0.011119136665482327",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5840249420000418,
"range": "stddev: 0.04566128855766495",
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
"id": "159fcbc0380fd0f104d7a272ff18403b5ee8f6ad",
"message": "Optimize hot loops further, redundant evaluations (#454)",
"timestamp": "2026-03-27T10:09:00+01:00",
"tree_id": "95f95268707df1057a1a620917b75aab8bdf025b",
"url": "https://github.com/proximafusion/vmecpp/commit/159fcbc0380fd0f104d7a272ff18403b5ee8f6ad"
},
"date": 1783553392210,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3720489357999668,
"range": "stddev: 0.007571916872247513",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36251589839998816,
"range": "stddev: 0.0015437954778875654",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.801180936666545,
"range": "stddev: 0.007167767307143907",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.5030485806667,
"range": "stddev: 0.018203348897738583",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.122981248000087,
"range": "stddev: 0.015036200033515114",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5578694593333846,
"range": "stddev: 0.010902726450969263",
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
"id": "c9181b56ab9c9f097eb576b58d8a86462659f748",
"message": "Rename WOutFileContents scalar/1D fields to match Python VmecWOut (#442)",
"timestamp": "2026-03-27T11:10:05+01:00",
"tree_id": "350559997c660245c118de00e7f79b90aa1a5712",
"url": "https://github.com/proximafusion/vmecpp/commit/c9181b56ab9c9f097eb576b58d8a86462659f748"
},
"date": 1783553392694,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.37720278860006146,
"range": "stddev: 0.020159999509052357",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36167785099996763,
"range": "stddev: 0.0030108083291360264",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7690692270001211,
"range": "stddev: 0.011846346768415809",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.508628917666708,
"range": "stddev: 0.023635594229436122",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.05066983700029,
"range": "stddev: 0.023631423817264872",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5576157700000597,
"range": "stddev: 0.011387794287967686",
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
"id": "0e531fae6fa57f52c0a52d8400ff9c0e0c76ca0b",
"message": "Rename and pad half-grid 1D arrays to match Python VmecWOut (#443)",
"timestamp": "2026-03-27T11:14:09+01:00",
"tree_id": "a015fbee1637174e551bd2cae6554899ad72e0f6",
"url": "https://github.com/proximafusion/vmecpp/commit/0e531fae6fa57f52c0a52d8400ff9c0e0c76ca0b"
},
"date": 1783553392180,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.362368743999923,
"range": "stddev: 0.002499267804900068",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36067059759989206,
"range": "stddev: 0.005077509179296317",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.8232914016668171,
"range": "stddev: 0.08148709619467454",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.52779504433344,
"range": "stddev: 0.01334988465197803",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.032178187000227,
"range": "stddev: 0.021507415758270716",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5624964676665816,
"range": "stddev: 0.009508663979110368",
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
"id": "ab4e491bed0ce79324ae8112abf1f8424b9342ed",
"message": "Transpose 2D Fourier arrays to (mnmax, ns) layout (#444)",
"timestamp": "2026-03-27T11:15:18+01:00",
"tree_id": "c9712b333516c57c9f32b284646dd9145bc5c50c",
"url": "https://github.com/proximafusion/vmecpp/commit/ab4e491bed0ce79324ae8112abf1f8424b9342ed"
},
"date": 1783553392605,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36362160700000457,
"range": "stddev: 0.005807353980464317",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3633268868000414,
"range": "stddev: 0.009938593177544055",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.760455840999839,
"range": "stddev: 0.005302502576015599",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.471618129333213,
"range": "stddev: 0.0249694582283376",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.027511172333258,
"range": "stddev: 0.015353690208745246",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5544155013332481,
"range": "stddev: 0.011419511795948535",
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
"id": "684058866cbb7969ca2e90bcb8c01b6388e3265c",
"message": "Right-pad profile arrays in C++, add lrfp field, final cleanup (#445)",
"timestamp": "2026-03-27T11:17:46+01:00",
"tree_id": "e091a056a071f389319fdb60afead4147c667238",
"url": "https://github.com/proximafusion/vmecpp/commit/684058866cbb7969ca2e90bcb8c01b6388e3265c"
},
"date": 1783553392428,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36276741799993034,
"range": "stddev: 0.0037473822030765058",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3596195989998705,
"range": "stddev: 0.0020876278648213615",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7690015836666741,
"range": "stddev: 0.005953391898537618",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.495725563333356,
"range": "stddev: 0.032616824551012794",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.987064405333513,
"range": "stddev: 0.02097083211748228",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.557086784333175,
"range": "stddev: 0.010075215634465062",
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
"id": "16e7d0e3171a58d62b6fb12fe3e800e56eda11f8",
"message": "Add HDF5 backwards compatibility for renamed/resized wout fields (#446)",
"timestamp": "2026-03-27T11:21:40+01:00",
"tree_id": "53fa7dee60337426e3269ec381c9c2296e887308",
"url": "https://github.com/proximafusion/vmecpp/commit/16e7d0e3171a58d62b6fb12fe3e800e56eda11f8"
},
"date": 1783553392217,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3602628676000677,
"range": "stddev: 0.0019418930658577172",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36198702960000445,
"range": "stddev: 0.004725385176986749",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7898489039998822,
"range": "stddev: 0.008450186942285848",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.586648616666632,
"range": "stddev: 0.03206484256979066",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.011411874333438,
"range": "stddev: 0.018593667676700364",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5561524516665486,
"range": "stddev: 0.009935437886908855",
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
"id": "c81b6dd298d1937ed3eeac79e23d00a1e3234ea9",
"message": "Pass through outer pydantic serialization contexts (#457)",
"timestamp": "2026-03-27T21:51:03+01:00",
"tree_id": "2fb6ae029d04b3512849a84ff0805dddc450076d",
"url": "https://github.com/proximafusion/vmecpp/commit/c81b6dd298d1937ed3eeac79e23d00a1e3234ea9"
},
"date": 1783553392685,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36494430940010714,
"range": "stddev: 0.006168861061951783",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3638922937998359,
"range": "stddev: 0.005886675375115806",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7767130499999741,
"range": "stddev: 0.005713203437978654",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.567288872000063,
"range": "stddev: 0.035525268380556564",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.989513601999988,
"range": "stddev: 0.028535008255074237",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5615080396666297,
"range": "stddev: 0.006125687301829053",
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
"id": "3d6cbed0f25c1e5a296d1befb8b7ca15cca07997",
"message": "[Bugfix] xm, xn should be double but were serialized as int (#459)",
"timestamp": "2026-03-28T16:13:28+01:00",
"tree_id": "822a97757f797559878c6a25e283b92420d4c609",
"url": "https://github.com/proximafusion/vmecpp/commit/3d6cbed0f25c1e5a296d1befb8b7ca15cca07997"
},
"date": 1783553392310,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3638762475999101,
"range": "stddev: 0.004608729829151916",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.37532848279997777,
"range": "stddev: 0.019360628382591564",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7763138773331473,
"range": "stddev: 0.004397906352414304",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.581428189333565,
"range": "stddev: 0.04399490122019728",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.04388987066674,
"range": "stddev: 0.0630699496539737",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5594707256667182,
"range": "stddev: 0.015234578702639388",
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
"date": 1783553392537,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3632232356001623,
"range": "stddev: 0.00451809258723205",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.37148156040002506,
"range": "stddev: 0.005874809420862101",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7888749616665034,
"range": "stddev: 0.009757261062269285",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.565332631000122,
"range": "stddev: 0.0052135975458487875",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.993593687000157,
"range": "stddev: 0.029530415338529466",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5556984449999618,
"range": "stddev: 0.01324204894067022",
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
"date": 1783553392235,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36819353440005215,
"range": "stddev: 0.004319074617681581",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36124062140015667,
"range": "stddev: 0.0028927342721529123",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7751784483333115,
"range": "stddev: 0.009063390334960548",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.5695300190000125,
"range": "stddev: 0.01059945164111835",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.01984753033336,
"range": "stddev: 0.009513120859074624",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.556303151333547,
"range": "stddev: 0.009795164337249238",
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
"date": 1783553392709,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36846831959992415,
"range": "stddev: 0.006605625918468756",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3610241985999892,
"range": "stddev: 0.004213805318679696",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7827602886666984,
"range": "stddev: 0.008009948730926687",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.563317082333318,
"range": "stddev: 0.006229609343426338",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.007876729666654,
"range": "stddev: 0.0042138484026119334",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5571077163332727,
"range": "stddev: 0.008989154643480315",
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
"date": 1783553392370,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3638885580000533,
"range": "stddev: 0.006279653815753391",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3664057003999005,
"range": "stddev: 0.0068450635154469605",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7799434126665499,
"range": "stddev: 0.004249403685202336",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.555911282333303,
"range": "stddev: 0.0028622958125013786",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.000766271666635,
"range": "stddev: 0.008704614409309539",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5554807230000733,
"range": "stddev: 0.01351309456062097",
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
"date": 1783553392576,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36428052000001115,
"range": "stddev: 0.0018918926425511829",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36679140579999514,
"range": "stddev: 0.003521864737110876",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7761655483332106,
"range": "stddev: 0.006938436821717316",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.573948099666571,
"range": "stddev: 0.015330568069131955",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.033169977666603,
"range": "stddev: 0.0035042407419160673",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5603692376665397,
"range": "stddev: 0.009327468913755903",
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
"date": 1783553392549,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.362704477600073,
"range": "stddev: 0.003277364322239117",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3634528966000289,
"range": "stddev: 0.0055549496077283295",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7796477933334245,
"range": "stddev: 0.004127483958723376",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.558854697666599,
"range": "stddev: 0.020175399024795735",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.948965989333374,
"range": "stddev: 0.05736258640534283",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5566731860000498,
"range": "stddev: 0.013164643021191422",
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
"date": 1783553392491,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3631606448000639,
"range": "stddev: 0.003080463169724275",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3618280054000934,
"range": "stddev: 0.003379703736547258",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7826724163336014,
"range": "stddev: 0.006795855139145497",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.569374548333296,
"range": "stddev: 0.01403251660919287",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.035515273000025,
"range": "stddev: 0.018101322888331085",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.555888694666616,
"range": "stddev: 0.012654627766278882",
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
"date": 1783553392755,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34517036799999234,
"range": "stddev: 0.0008147503532360875",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34384867859998847,
"range": "stddev: 0.005239667798036131",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.777211687666674,
"range": "stddev: 0.013817645754290417",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.502430922999982,
"range": "stddev: 0.023910070341742206",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.87286579966667,
"range": "stddev: 0.0388454649004038",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.54684315066667,
"range": "stddev: 0.010841389728054074",
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
"date": 1783553392720,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35034681800002543,
"range": "stddev: 0.0020910558964370406",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35320917300000476,
"range": "stddev: 0.01856790755303427",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7830946330000188,
"range": "stddev: 0.01625035255375262",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.533187266666687,
"range": "stddev: 0.038186443665520986",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.935349064333272,
"range": "stddev: 0.03164041730815196",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.547716112333319,
"range": "stddev: 0.010908397348058298",
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
"date": 1783553392481,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3457032001999778,
"range": "stddev: 0.004640428813485307",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34302009419998286,
"range": "stddev: 0.003065086512069681",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.773409238666659,
"range": "stddev: 0.0064051925064185335",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.5087238486666665,
"range": "stddev: 0.026760492743439526",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.885929452333357,
"range": "stddev: 0.0233862153136226",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5483491123333504,
"range": "stddev: 0.01321975313618058",
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
"date": 1783553392312,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35912380360000495,
"range": "stddev: 0.00446760384110545",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3583180811999682,
"range": "stddev: 0.0021721100301770706",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7748358073333748,
"range": "stddev: 0.004056433134627016",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.518806514666646,
"range": "stddev: 0.018269322273163864",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.942960857666662,
"range": "stddev: 0.03439725242804937",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.55095010466664,
"range": "stddev: 0.013680119654877235",
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
"date": 1783553392284,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3441050189999942,
"range": "stddev: 0.0032344248238760794",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34768696540004385,
"range": "stddev: 0.003900163064278228",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.777829537666662,
"range": "stddev: 0.006976336181221847",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.523941027666676,
"range": "stddev: 0.01011108737265793",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.902613814333373,
"range": "stddev: 0.007115462611256159",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5488537096667112,
"range": "stddev: 0.00999914237185793",
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
"date": 1783553392763,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34322435939998286,
"range": "stddev: 0.002057695227816385",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34817523059996347,
"range": "stddev: 0.00852868833401893",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.790444783000036,
"range": "stddev: 0.028002067035149056",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.532106318000008,
"range": "stddev: 0.027083896576072408",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.858756178666605,
"range": "stddev: 0.051041699266732396",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5518940106667287,
"range": "stddev: 0.013888874829984325",
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
"date": 1783553392249,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35698134940003,
"range": "stddev: 0.0026941949034152823",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3512882303999959,
"range": "stddev: 0.0042390796535707554",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.8036773166666837,
"range": "stddev: 0.01330148141511326",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.545724676999877,
"range": "stddev: 0.02300432695360742",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.006095517333279,
"range": "stddev: 0.06621036540192356",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5471356713333837,
"range": "stddev: 0.011768697140007642",
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
"date": 1783553392639,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.342018729000074,
"range": "stddev: 0.0027046305592558465",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.345677469400016,
"range": "stddev: 0.003446906185612428",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7802363680000326,
"range": "stddev: 0.010620311846788055",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.465032754333454,
"range": "stddev: 0.004536043447396093",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.940954187666648,
"range": "stddev: 0.026073357664757463",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5514798046666176,
"range": "stddev: 0.008793572747705479",
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
"id": "d79e972b074dedc10976091e00e12f68551d5fb3",
"message": "Restrict pydantic version, breaking change (#480)",
"timestamp": "2026-04-16T11:06:55+02:00",
"tree_id": "c081f0908ae5d478b1d3e6f9b2b2ae2457d276fb",
"url": "https://github.com/proximafusion/vmecpp/commit/d79e972b074dedc10976091e00e12f68551d5fb3"
},
"date": 1783553392751,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35262016779997796,
"range": "stddev: 0.005454192356924687",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3518791914000303,
"range": "stddev: 0.0021809854625786844",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7778480366666827,
"range": "stddev: 0.0019225948768637302",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.491821212666612,
"range": "stddev: 0.034126505741335286",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.94771756299997,
"range": "stddev: 0.03335421468572456",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5546871669998836,
"range": "stddev: 0.010427600128051983",
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
"id": "48dd1a146c28f3a06e53af22dbc2e8d2c32c38cc",
"message": "Also add .github/workflows/copilot-setup-steps.yml to pre-install vmecpp and its Ubuntu system dependencies in the Copilot cloud agent environment. (#483)",
"timestamp": "2026-04-16T13:33:19+02:00",
"tree_id": "1aed1e09f2dffc1c1fdbc2e0601a7f79d30d2d0d",
"url": "https://github.com/proximafusion/vmecpp/commit/48dd1a146c28f3a06e53af22dbc2e8d2c32c38cc"
},
"date": 1783553392348,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34451257519995127,
"range": "stddev: 0.004467008797202549",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3422564720000537,
"range": "stddev: 0.002656051196063845",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7789396026666964,
"range": "stddev: 0.006237836680544813",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.460095476333436,
"range": "stddev: 0.02720735597072187",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.891079879666677,
"range": "stddev: 0.04185234208913002",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.550767740333337,
"range": "stddev: 0.011241535573821468",
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
"date": 1783553392515,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34277832580000905,
"range": "stddev: 0.002348283280767371",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34523140160004007,
"range": "stddev: 0.002181104942985867",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7749834989999727,
"range": "stddev: 0.0019448422418990128",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.488151312666787,
"range": "stddev: 0.04102086110294463",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.916105489333328,
"range": "stddev: 0.014009976399933299",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5504767719999109,
"range": "stddev: 0.010528508152830924",
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
"id": "fa1b951045b2243a0c9340d0b16bec4a60e5136e",
"message": "Update README.md",
"timestamp": "2026-04-23T10:12:38+02:00",
"tree_id": "f3fac97cd72cc9ec7d5fd85ca957778291adbb65",
"url": "https://github.com/proximafusion/vmecpp/commit/fa1b951045b2243a0c9340d0b16bec4a60e5136e"
},
"date": 1783553392819,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34847508580000974,
"range": "stddev: 0.0018802891496443647",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35306991960001144,
"range": "stddev: 0.005463623529089308",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7729936516666385,
"range": "stddev: 0.003691340440891677",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.5027503450000195,
"range": "stddev: 0.02720733580050879",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.95040120766665,
"range": "stddev: 0.021446739798268617",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5508613026667415,
"range": "stddev: 0.010439899857872886",
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
"id": "51527dfb71c83aee7a8d83df6203758e2c78ef18",
"message": "Miller recurrence relation for numerically stable NESTOR at high mpol/ntor (#484)",
"timestamp": "2026-04-24T14:28:08+02:00",
"tree_id": "46ee572b3021f833c339ffbbac9effe65b0f54e2",
"url": "https://github.com/proximafusion/vmecpp/commit/51527dfb71c83aee7a8d83df6203758e2c78ef18"
},
"date": 1783553392363,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3511303299999781,
"range": "stddev: 0.005136820414084634",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34564149040002123,
"range": "stddev: 0.0027020738794916134",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.776432740666678,
"range": "stddev: 0.001834751118377355",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.504206036999979,
"range": "stddev: 0.006855190168965661",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.024049526666658,
"range": "stddev: 0.03806096215526977",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.548270001999981,
"range": "stddev: 0.012696322974539837",
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
"date": 1783553392806,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34339680059983946,
"range": "stddev: 0.0029197954191669886",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3443806921999567,
"range": "stddev: 0.005110308765257899",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7717318649999168,
"range": "stddev: 0.007852027503157555",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.498725679333347,
"range": "stddev: 0.019775385777682803",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 10.02521363633332,
"range": "stddev: 0.021873716740249655",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5453950690001268,
"range": "stddev: 0.01051881982882626",
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
"date": 1783553392564,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34536150140002064,
"range": "stddev: 0.0030366416804893997",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3477835082000638,
"range": "stddev: 0.006233536031216941",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.8024989526666104,
"range": "stddev: 0.012736305074306009",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.524362286333296,
"range": "stddev: 0.026766374203635188",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.992239122333407,
"range": "stddev: 0.004413806429907347",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.558312188666605,
"range": "stddev: 0.013015214426924373",
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
"id": "d6875c57529a77d7e32e6845b9d3ee40afdda5b3",
"message": "Abseil errors instead of LOG(FATAL) (#493)",
"timestamp": "2026-04-27T17:58:22+02:00",
"tree_id": "a4313fa42f5838bf688910bffd3f9f3d68f9827b",
"url": "https://github.com/proximafusion/vmecpp/commit/d6875c57529a77d7e32e6845b9d3ee40afdda5b3"
},
"date": 1783553392746,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34642217319997143,
"range": "stddev: 0.004841698813178148",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3430798636001782,
"range": "stddev: 0.002425282200583125",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7878879749999517,
"range": "stddev: 0.00922379703835069",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.492036943666487,
"range": "stddev: 0.03333532586279794",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.934790448666414,
"range": "stddev: 0.009211931029028179",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.551284182333499,
"range": "stddev: 0.010832281712528896",
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
"id": "97cb9d20e1c85fdf05f9dbfae0549754daacbf4a",
"message": "Scientific variable names get flagged incorrectly too often by clang-tidy (#491)",
"timestamp": "2026-04-27T18:18:23+02:00",
"tree_id": "b23af3b37dab107f49bb1792b9e40ac499ab3af8",
"url": "https://github.com/proximafusion/vmecpp/commit/97cb9d20e1c85fdf05f9dbfae0549754daacbf4a"
},
"date": 1783553392554,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34758989999991174,
"range": "stddev: 0.004891685630710558",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3461604048001391,
"range": "stddev: 0.008113611281010832",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7776719886668009,
"range": "stddev: 0.0022286153392757943",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.500222951666577,
"range": "stddev: 0.028623063574438654",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.949540905333379,
"range": "stddev: 0.007738699615367019",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5517816333332728,
"range": "stddev: 0.011504119080594111",
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
"id": "42e3ec5f8b9b731a5ad717a06ec19adf7131111f",
"message": "Use std::unreachable instead of LOG(FATAL) for unreachable code (#494)",
"timestamp": "2026-04-27T23:01:30+02:00",
"tree_id": "c57ee3789ceb1a61f7bbd68a6322c8d76d8bac12",
"url": "https://github.com/proximafusion/vmecpp/commit/42e3ec5f8b9b731a5ad717a06ec19adf7131111f"
},
"date": 1783553392334,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3451990399999886,
"range": "stddev: 0.0019271073728224597",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3479262271999687,
"range": "stddev: 0.007632246273086367",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7495062456664527,
"range": "stddev: 0.0014482534540879578",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.4552906686665965,
"range": "stddev: 0.027211819184824692",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.91408665066668,
"range": "stddev: 0.010615253329992882",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5516661859998446,
"range": "stddev: 0.005632213261952116",
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
"id": "b24222e51f3d3ab1f9acbff4139fbca21783a0b7",
"message": "Terminate on NaN or inf, return NaN instead of terminating in tridiagonal solve (#496)",
"timestamp": "2026-04-28T09:42:10+02:00",
"tree_id": "77e85f47c89c700befb46c86153741f0fbeb0185",
"url": "https://github.com/proximafusion/vmecpp/commit/b24222e51f3d3ab1f9acbff4139fbca21783a0b7"
},
"date": 1783553392620,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.348884972399901,
"range": "stddev: 0.006361132618886105",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3467055789999904,
"range": "stddev: 0.0043959007945461965",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7201198579999375,
"range": "stddev: 0.00373211046069717",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.433478528666758,
"range": "stddev: 0.04423110067394176",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.9513072243332,
"range": "stddev: 0.015484060131184961",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5496132079999068,
"range": "stddev: 0.011477821350094482",
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
"id": "6f846de63275c49dc5f60da16e2dfb052055c50d",
"message": "TSAN docker setup (#492)",
"timestamp": "2026-04-28T12:23:06+02:00",
"tree_id": "6ea1141f29156aa8a27ad98b5fa089076d812218",
"url": "https://github.com/proximafusion/vmecpp/commit/6f846de63275c49dc5f60da16e2dfb052055c50d"
},
"date": 1783553392445,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3438983893999648,
"range": "stddev: 0.0011760658760265274",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34760936999991826,
"range": "stddev: 0.005930949027240094",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7378572233334733,
"range": "stddev: 0.010261813527758287",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.445581897333341,
"range": "stddev: 0.033807609264121484",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.926205160000032,
"range": "stddev: 0.020829452392314554",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5509389366666255,
"range": "stddev: 0.009474830168974768",
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
"id": "c9772aed7885bd7ed62e4fbaae99d8a30915e74c",
"message": "Niter wording improved (#504)",
"timestamp": "2026-04-28T16:59:51+02:00",
"tree_id": "7cc623a63d553fa23218de359440aa6bc4c7d20e",
"url": "https://github.com/proximafusion/vmecpp/commit/c9772aed7885bd7ed62e4fbaae99d8a30915e74c"
},
"date": 1783553392699,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35302328919997306,
"range": "stddev: 0.010120578580167303",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36300846919994,
"range": "stddev: 0.006187736478485899",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7358831373332275,
"range": "stddev: 0.006340338186677587",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.441239525333306,
"range": "stddev: 0.03377421051592537",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.939306562666692,
"range": "stddev: 0.040915777532389154",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5561658243332204,
"range": "stddev: 0.01198412859739455",
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
"id": "ea76d2124c0a47e294f7369d8fa36781d5f7d2fe",
"message": "FFTW3 dependencies (#501)",
"timestamp": "2026-04-28T17:32:26+02:00",
"tree_id": "e7fb40efbab4d320cc28192ea0b0a916d6b7500d",
"url": "https://github.com/proximafusion/vmecpp/commit/ea76d2124c0a47e294f7369d8fa36781d5f7d2fe"
},
"date": 1783553392787,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35183697340007714,
"range": "stddev: 0.0024441193719751298",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35181442540006175,
"range": "stddev: 0.006364037529202355",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7325306066666901,
"range": "stddev: 0.004369692478345491",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.462623759666712,
"range": "stddev: 0.04348812939342005",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.954277271333163,
"range": "stddev: 0.027133269813397778",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5578578106666707,
"range": "stddev: 0.006928597660377143",
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
"date": 1783553392583,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34934660940007234,
"range": "stddev: 0.0015971629790605115",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3493250407999767,
"range": "stddev: 0.0011660440748327327",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7530311576668585,
"range": "stddev: 0.009624645237247213",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.435249751333383,
"range": "stddev: 0.014933283560204676",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.237944840333208,
"range": "stddev: 0.01659666889040282",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5571918456666936,
"range": "stddev: 0.006729967478367393",
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
"id": "12b7f60d9a4c5417ceddaa4d5ea5ab0e25febf6a",
"message": "Laplace solver stellarator asym added (#502)",
"timestamp": "2026-04-28T18:12:25+02:00",
"tree_id": "2f14fa7d2c8decb9c434dc27157c2c2916187b00",
"url": "https://github.com/proximafusion/vmecpp/commit/12b7f60d9a4c5417ceddaa4d5ea5ab0e25febf6a"
},
"date": 1783553392201,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3453107166001246,
"range": "stddev: 0.0026761639959401095",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3503566423999473,
"range": "stddev: 0.002208387455600457",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7432632959998955,
"range": "stddev: 0.007328123955144602",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.40903665700004,
"range": "stddev: 0.008336147340205541",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.235454147999917,
"range": "stddev: 0.021836147750840623",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.551434275333122,
"range": "stddev: 0.009026613253907856",
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
"id": "c8e567fc623c429024c11a4555e0fd8a58b71679",
"message": "Revert \"FFTW3 dependencies (#501)\" (#506)",
"timestamp": "2026-04-29T15:14:21+02:00",
"tree_id": "4895e3fe13cfbe6572522fbff370fdb42ef68df8",
"url": "https://github.com/proximafusion/vmecpp/commit/c8e567fc623c429024c11a4555e0fd8a58b71679"
},
"date": 1783553392692,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35155421959998423,
"range": "stddev: 0.007302634031649402",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34972711639993576,
"range": "stddev: 0.008602563263165115",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7526579833333926,
"range": "stddev: 0.03508444590664738",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.422760603999905,
"range": "stddev: 0.022290360210001726",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.277585911666696,
"range": "stddev: 0.010896296581786936",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.551588023000022,
"range": "stddev: 0.010907703803810483",
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
"id": "378b4ff6edc6dd41d3afde48fc66f38723726297",
"message": "Reduce free boundary oversubscription due to Eigen low level parallelism on some platforms (#507)",
"timestamp": "2026-04-29T15:33:24+02:00",
"tree_id": "32526f65a9fdcc89d31c91b4fdbf874ceb6c27b9",
"url": "https://github.com/proximafusion/vmecpp/commit/378b4ff6edc6dd41d3afde48fc66f38723726297"
},
"date": 1783553392293,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35438873979992425,
"range": "stddev: 0.003537795006282235",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3522703055999955,
"range": "stddev: 0.0017886546467305033",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7831863289999699,
"range": "stddev: 0.012342275139900091",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.512410203333123,
"range": "stddev: 0.04784050973163939",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.328786808333158,
"range": "stddev: 0.030944314254119906",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5528325436666819,
"range": "stddev: 0.012039885331632747",
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
"id": "7bb4582991a98dc91f165f9659e9338fefb7e8bb",
"message": "Reduce thread spinning on first parallel region enter (#508)",
"timestamp": "2026-04-30T01:39:45+02:00",
"tree_id": "c102485b620e8b188776c29b801d5c85342061c5",
"url": "https://github.com/proximafusion/vmecpp/commit/7bb4582991a98dc91f165f9659e9338fefb7e8bb"
},
"date": 1783553392467,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3619877117999749,
"range": "stddev: 0.003350314167531666",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35357737620006446,
"range": "stddev: 0.0036840979563194888",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.759528507666649,
"range": "stddev: 0.0040116499586297415",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.522006012333198,
"range": "stddev: 0.08479695295479135",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.317126557000089,
"range": "stddev: 0.01998794350796272",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5512611556669071,
"range": "stddev: 0.007768690047231613",
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
"id": "01b1fcb56a26188eb71a74fcf9cdf0c7349e5c04",
"message": "Diagnose how much force residual is being truncated away (#495)",
"timestamp": "2026-04-30T21:49:38+02:00",
"tree_id": "89a1ebe9dc8cda139a3ada16a72c43556f11f0ff",
"url": "https://github.com/proximafusion/vmecpp/commit/01b1fcb56a26188eb71a74fcf9cdf0c7349e5c04"
},
"date": 1783553392128,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3480808329999491,
"range": "stddev: 0.0036356845876628786",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34533674280000926,
"range": "stddev: 0.0022862282356941447",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7719286130001517,
"range": "stddev: 0.01617208873690742",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 4.481708891666737,
"range": "stddev: 0.052605974589593985",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.29104033733347,
"range": "stddev: 0.004839282344399896",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5653440593332562,
"range": "stddev: 0.0161990491881609",
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
"id": "3ac242a3d87feef9b57e348f50131d9754ca43d1",
"message": "Batched FFT, 15% expected gain for w7x (#503)",
"timestamp": "2026-05-05T15:22:26+02:00",
"tree_id": "b48a2de956f8741c9aecfcbf733013e1a4b4f1da",
"url": "https://github.com/proximafusion/vmecpp/commit/3ac242a3d87feef9b57e348f50131d9754ca43d1"
},
"date": 1783553392305,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.356959029199993,
"range": "stddev: 0.0044172926635082385",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35840237879999676,
"range": "stddev: 0.0025977150770284153",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7155785740000056,
"range": "stddev: 0.005960081958314871",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9143162216666574,
"range": "stddev: 0.01070985306304492",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.322276159666691,
"range": "stddev: 0.013824594726140955",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5584841713333237,
"range": "stddev: 0.01032989312565183",
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
"id": "8ed41a4ebc3aac15f0356bc75d4f8cebbac6ef0c",
"message": "More realistic wait policy, additional fixed boundary benchmark (#515)",
"timestamp": "2026-05-05T15:48:58+02:00",
"tree_id": "217c3b116ed2219ed36cb2ea3d15e5d7f56c8ef8",
"url": "https://github.com/proximafusion/vmecpp/commit/8ed41a4ebc3aac15f0356bc75d4f8cebbac6ef0c"
},
"date": 1783553392532,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35217137760000694,
"range": "stddev: 0.0030072636520090254",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36393917799999825,
"range": "stddev: 0.024380711158052792",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7150461333333358,
"range": "stddev: 0.0023378542169268856",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.890326297666661,
"range": "stddev: 0.019178615307740923",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.285734117333297,
"range": "stddev: 0.03175710042729687",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.553327306666669,
"range": "stddev: 0.00692735442247731",
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
"id": "04158aa5f9eae86e399a928db56afef3413dabb1",
"message": "Update README.md",
"timestamp": "2026-05-12T21:26:02+02:00",
"tree_id": "4901ac7979c957946219f2483da33f2eab28d01f",
"url": "https://github.com/proximafusion/vmecpp/commit/04158aa5f9eae86e399a928db56afef3413dabb1"
},
"date": 1783553392140,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34979624039999635,
"range": "stddev: 0.0017242262954735444",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35486741680003886,
"range": "stddev: 0.00193448479431415",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7185626423333435,
"range": "stddev: 0.008881156362737131",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9029573613333164,
"range": "stddev: 0.014630491142389042",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.254388958333342,
"range": "stddev: 0.02363740828640547",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5532521346666879,
"range": "stddev: 0.010623602500516142",
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
"id": "9b2e1e066106a9100d31aad811677c8f23fb8456",
"message": "Update copilot-setup-steps.yml",
"timestamp": "2026-05-13T09:45:01+02:00",
"tree_id": "213971aa8b812e90eea7f78beeed6a4d98f68681",
"url": "https://github.com/proximafusion/vmecpp/commit/9b2e1e066106a9100d31aad811677c8f23fb8456"
},
"date": 1783553392559,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35045058620000874,
"range": "stddev: 0.0012325520108032214",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3530154844000208,
"range": "stddev: 0.005806538221281914",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7122652993333152,
"range": "stddev: 0.0035282397050651975",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.912108150999984,
"range": "stddev: 0.033366728728396605",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.29814643366664,
"range": "stddev: 0.010494759490875593",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.555521704333349,
"range": "stddev: 0.013378710689877144",
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
"id": "5f23d714753bb808bf5ca2675374e1385ae1468a",
"message": "Remove FFTW from container installs, since we settled on FFTX now (#517)",
"timestamp": "2026-05-13T10:01:17+02:00",
"tree_id": "3aa86ce29736387d10a101cb18432185447d9955",
"url": "https://github.com/proximafusion/vmecpp/commit/5f23d714753bb808bf5ca2675374e1385ae1468a"
},
"date": 1783553392409,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3527917964000153,
"range": "stddev: 0.003945972100493588",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35085257360001376,
"range": "stddev: 0.0037126655849223436",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7115599259999879,
"range": "stddev: 0.001732492814463827",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.874414351666663,
"range": "stddev: 0.022115337604262954",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.289042101333356,
"range": "stddev: 0.023713584605782",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5546541673332968,
"range": "stddev: 0.010354187044502954",
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
"id": "6a5bee6ee5669c5a77face38d1e8056bba120abd",
"message": "Enable pre-commit for copilot agents (#520)",
"timestamp": "2026-05-13T13:44:26+02:00",
"tree_id": "265592f4b5820720cc5c7fbe0372796b1e38086b",
"url": "https://github.com/proximafusion/vmecpp/commit/6a5bee6ee5669c5a77face38d1e8056bba120abd"
},
"date": 1783553392433,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.354516695999996,
"range": "stddev: 0.005526846008425477",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35358764099999007,
"range": "stddev: 0.002336199206685898",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.716155348666651,
"range": "stddev: 0.0012163272828635769",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.875208635999987,
"range": "stddev: 0.011270589723841035",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.309572857333327,
"range": "stddev: 0.01620974284218428",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5618352629999965,
"range": "stddev: 0.018265686707983324",
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
"date": 1783553392344,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3533245597999667,
"range": "stddev: 0.004633895520213229",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3581675933999577,
"range": "stddev: 0.009049601305723704",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7158135129999816,
"range": "stddev: 0.005168741384733064",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9264295619999907,
"range": "stddev: 0.0189205176903495",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.282165908666684,
"range": "stddev: 0.04570242889015352",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5609967653334327,
"range": "stddev: 0.013348284342513757",
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
"id": "7e06d9300a2b0afadf440c9b557fe4bfab5c3c79",
"message": "Update README.md (#516)",
"timestamp": "2026-05-14T23:54:23+02:00",
"tree_id": "0b15c8f66daa0128501442e828e3d74f0cb119b3",
"url": "https://github.com/proximafusion/vmecpp/commit/7e06d9300a2b0afadf440c9b557fe4bfab5c3c79"
},
"date": 1783553392477,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3514538739999352,
"range": "stddev: 0.0019674210295906776",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3542102354000235,
"range": "stddev: 0.008283502627555402",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7116490646666687,
"range": "stddev: 0.0021770342751154764",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.864219950000006,
"range": "stddev: 0.009650485954233779",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.252160298000035,
"range": "stddev: 0.011382606934202212",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.558886971999982,
"range": "stddev: 0.010973251225739448",
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
"date": 1783553392219,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3504078288000073,
"range": "stddev: 0.002001432715424629",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34916389900008654,
"range": "stddev: 0.0033952611909593665",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7164815196666343,
"range": "stddev: 0.008986695124551036",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8561020729999504,
"range": "stddev: 0.01070218880972155",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.266841130333356,
"range": "stddev: 0.012120661397037265",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5569546236666458,
"range": "stddev: 0.007920121363707292",
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
"date": 1783553392657,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3525328403999538,
"range": "stddev: 0.0037577758094791084",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35184747880007305,
"range": "stddev: 0.0026288327069431887",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7124666659999548,
"range": "stddev: 0.003120133718122707",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.908270719666613,
"range": "stddev: 0.018915468688059156",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.245932390000007,
"range": "stddev: 0.001883715597870716",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.6138020126666863,
"range": "stddev: 0.09348970010321855",
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
"date": 1783553392525,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3468460827999479,
"range": "stddev: 0.0010751030341823555",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34958429040007105,
"range": "stddev: 0.0029706425241803913",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7133604490000682,
"range": "stddev: 0.00978309835048393",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8519780229999774,
"range": "stddev: 0.016886177559578806",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.285016479333308,
"range": "stddev: 0.03596008644620061",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.554762323000053,
"range": "stddev: 0.007141613036039124",
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
"date": 1783553392603,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34991129920003916,
"range": "stddev: 0.004436536975034926",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34825226459997793,
"range": "stddev: 0.0029682389422468034",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7250042060000699,
"range": "stddev: 0.0012962325486732425",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.863927180000019,
"range": "stddev: 0.033519233019302215",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.245633640666634,
"range": "stddev: 0.017207851774469953",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5511743236666007,
"range": "stddev: 0.014955583842413743",
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
"date": 1783553392498,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.35141941819993006,
"range": "stddev: 0.004251889627341669",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35356142659998113,
"range": "stddev: 0.000770081514538483",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7228915939999752,
"range": "stddev: 0.004301022981349016",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8864677663333773,
"range": "stddev: 0.02060585773229421",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.261435095666608,
"range": "stddev: 0.02096272005304126",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5506709776666412,
"range": "stddev: 0.010803668050759134",
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
"date": 1783553392704,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36162774359995636,
"range": "stddev: 0.010596487580494663",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35578752119995444,
"range": "stddev: 0.00512507552834096",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7363634126666057,
"range": "stddev: 0.004657011026099146",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.964049409666662,
"range": "stddev: 0.012283911488704934",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.275264874666618,
"range": "stddev: 0.013191417921631848",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5573691630000515,
"range": "stddev: 0.013768267646486391",
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
"date": 1783553392578,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3513579415999629,
"range": "stddev: 0.0021987498186006346",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.34941556639996635,
"range": "stddev: 0.0036611713276919343",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7346262503331975,
"range": "stddev: 0.005821800556153952",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.8960159806665615,
"range": "stddev: 0.0029422483348758182",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.280866049333477,
"range": "stddev: 0.00850008368490833",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5580614186665116,
"range": "stddev: 0.012178724629995325",
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
"date": 1783553392353,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.34645405780001964,
"range": "stddev: 0.0021556095788802508",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3485088724000889,
"range": "stddev: 0.0036175124422121435",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.6992882210000364,
"range": "stddev: 0.00542127214482831",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9079235416664537,
"range": "stddev: 0.011809592778046924",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.268727334666588,
"range": "stddev: 0.008701536200664273",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5517364133332496,
"range": "stddev: 0.009410661925909091",
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
"id": "10027e5d5ec4f31fcd863663cf6c8d4a41b45b5d",
"message": "Cache TSAN Docker image in GHCR (#553)",
"timestamp": "2026-05-31T14:42:53+02:00",
"tree_id": "b8cc6a49e078feacb3fcf762afc3455a317e1624",
"url": "https://github.com/proximafusion/vmecpp/commit/10027e5d5ec4f31fcd863663cf6c8d4a41b45b5d"
},
"date": 1783553392186,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3483424705998914,
"range": "stddev: 0.003128099020819941",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.35064930740009004,
"range": "stddev: 0.0015875732884937648",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7100267083333165,
"range": "stddev: 0.012324028748581706",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.906336803666818,
"range": "stddev: 0.011302060152452541",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.367783182999725,
"range": "stddev: 0.026259556007874274",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.55577838700007,
"range": "stddev: 0.012443348483458423",
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
"date": 1783553392697,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3612009020000187,
"range": "stddev: 0.004323750961799242",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36448504359987055,
"range": "stddev: 0.002513806710970922",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7122313363333888,
"range": "stddev: 0.012114239362757769",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9683381896666106,
"range": "stddev: 0.0145683142322452",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.363141982000343,
"range": "stddev: 0.013995146483920647",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5594856283332774,
"range": "stddev: 0.010377580025473917",
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
"date": 1783553392126,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3652574387999266,
"range": "stddev: 0.008303883350599457",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3636217606001992,
"range": "stddev: 0.002723426493522998",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7409486563333303,
"range": "stddev: 0.003278778038480437",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9838764180000603,
"range": "stddev: 0.05086018460633923",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.327479152666607,
"range": "stddev: 0.018578250361763263",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5600055800000519,
"range": "stddev: 0.012570898160260757",
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
"id": "8dca9e9711204f7a75df4bad46a1e847447d8c59",
"message": "Streamlined AGENTS.md (#539)",
"timestamp": "2026-06-02T01:10:12+02:00",
"tree_id": "6991bf52fdd0f0ecde43b759bf11bd23f9a74f76",
"url": "https://github.com/proximafusion/vmecpp/commit/8dca9e9711204f7a75df4bad46a1e847447d8c59"
},
"date": 1783553392530,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3509875819999252,
"range": "stddev: 0.002439400105248607",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.350421355600065,
"range": "stddev: 0.0018888952736940516",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.734817696666596,
"range": "stddev: 0.005266166139096719",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9436586943334078,
"range": "stddev: 0.01199826947516689",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.421871897333554,
"range": "stddev: 0.07201669302756462",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5550631443332652,
"range": "stddev: 0.011972744894503874",
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
"date": 1783553392600,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3494882548001442,
"range": "stddev: 0.0025619915433517985",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3475344828000743,
"range": "stddev: 0.003975098983013495",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.7460536836667113,
"range": "stddev: 0.011954211973056357",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.88695376133334,
"range": "stddev: 0.009166922979449158",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.305928401999912,
"range": "stddev: 0.0030042567984161044",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.551223686999947,
"range": "stddev: 0.007070941811042476",
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
"date": 1783553392411,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3492120369999611,
"range": "stddev: 0.0013557434820585144",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3499714261999543,
"range": "stddev: 0.001776627874401998",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.715136110000003,
"range": "stddev: 0.003435028873680442",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.889890252666646,
"range": "stddev: 0.027961702186091987",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.236206020999967,
"range": "stddev: 0.011842402881982237",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5607807376666945,
"range": "stddev: 0.00859180492543716",
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
"date": 1783553392308,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3677901691999068,
"range": "stddev: 0.0035859877226130116",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36702681539991316,
"range": "stddev: 0.0018221428892851996",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.727455250666632,
"range": "stddev: 0.001594654961944322",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.876944134999879,
"range": "stddev: 0.02414190133124414",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.245383181999992,
"range": "stddev: 0.016871299988772872",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5557335133333556,
"range": "stddev: 0.013127266530148583",
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
"id": "55c569444c82fc93b55b2c752c7f793d841625fa",
"message": "Python iteration debug script (#556)",
"timestamp": "2026-06-03T08:58:09+02:00",
"tree_id": "10550af4bf37180818ab1e0ca3487939aacef66c",
"url": "https://github.com/proximafusion/vmecpp/commit/55c569444c82fc93b55b2c752c7f793d841625fa"
},
"date": 1783553392380,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3710843834000116,
"range": "stddev: 0.003449381436758076",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.36807705980008903,
"range": "stddev: 0.001916401339009874",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.6936951729999237,
"range": "stddev: 0.004830284907414073",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.915554357333349,
"range": "stddev: 0.03247621222690743",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.29506335699989,
"range": "stddev: 0.005073363982036226",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5540504553334056,
"range": "stddev: 0.011888551326808635",
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
"id": "0ff3e7c9fb29385e6a91753ee320c983cff5e6e0",
"message": "Update fourier_basis_implementation.md (#559)",
"timestamp": "2026-06-05T10:02:40+02:00",
"tree_id": "4be8128819c0b9b38cd83ea5c1989c69e26f42ec",
"url": "https://github.com/proximafusion/vmecpp/commit/0ff3e7c9fb29385e6a91753ee320c983cff5e6e0"
},
"date": 1783553392184,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3703885143999287,
"range": "stddev: 0.0017638328927103066",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.37674988660000963,
"range": "stddev: 0.014183979286647561",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.695139443333422,
"range": "stddev: 0.01280109553463879",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.913007991000086,
"range": "stddev: 0.0143693491692496",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.287266438333214,
"range": "stddev: 0.027056411582925512",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5500477116665934,
"range": "stddev: 0.010079603663071219",
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
"date": 1783553392581,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36938992740015236,
"range": "stddev: 0.0028344982966102285",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3704423843999393,
"range": "stddev: 0.0037273421611067965",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.6916222173335882,
"range": "stddev: 0.004445339498858695",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9011507460001362,
"range": "stddev: 0.021759262596307248",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.272896194666677,
"range": "stddev: 0.03651641192409381",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.555857250000069,
"range": "stddev: 0.011226665741089588",
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
"id": "c3ec179b9fa09fc043444ed60a46fafa726d620b",
"message": "Handle jaxtyping anonymous-dimension API drift in wout serialization (#562)",
"timestamp": "2026-06-15T17:12:05+02:00",
"tree_id": "5c962a9aab90d2912a1034d8f8fc1c75980d8db6",
"url": "https://github.com/proximafusion/vmecpp/commit/c3ec179b9fa09fc043444ed60a46fafa726d620b"
},
"date": 1783553392664,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.36941841400002884,
"range": "stddev: 0.004266202549818866",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.3669961809999222,
"range": "stddev: 0.0010189376864621774",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.6849533303332767,
"range": "stddev: 0.0004694974818672784",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9057657330001043,
"range": "stddev: 0.02284420822651704",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.275369261333404,
"range": "stddev: 0.009231367710396628",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5539491770000495,
"range": "stddev: 0.011851710693137205",
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
"date": 1783553392612,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.3692837564000911,
"range": "stddev: 0.0029004741290754033",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.367549109400079,
"range": "stddev: 0.0013694545112526327",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.6986744066666688,
"range": "stddev: 0.005642005145709742",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.9030257653334957,
"range": "stddev: 0.017301192097413674",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 9.28589130700008,
"range": "stddev: 0.01291126527267809",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.5551682833332354,
"range": "stddev: 0.018135632890617055",
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
"date": 1783553392378,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28697266739999916,
"range": "stddev: 0.003991216319555862",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.284858653200007,
"range": "stddev: 0.006631342910169923",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.2752110609999932,
"range": "stddev: 0.0020277310581043252",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0957612109999957,
"range": "stddev: 0.02836649030958463",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.523995940333322,
"range": "stddev: 0.013066555133878013",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3407323379999998,
"range": "stddev: 0.009304450614484906",
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
"date": 1783553392455,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28563735220001263,
"range": "stddev: 0.003034960648234722",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2831296097999939,
"range": "stddev: 0.0014540647179839422",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3398339553333283,
"range": "stddev: 0.008986375335878056",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.1302843116666472,
"range": "stddev: 0.04392881545128186",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.655145528666689,
"range": "stddev: 0.010695802553468388",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3466840836666734,
"range": "stddev: 0.005514681657698435",
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
"date": 1783553392472,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28008635779999624,
"range": "stddev: 0.001109677902033244",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28098635979999925,
"range": "stddev: 0.0026884592733718094",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3225078086666902,
"range": "stddev: 0.0035181586341272325",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.077557576999993,
"range": "stddev: 0.03478840210963845",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.560586115333308,
"range": "stddev: 0.017012844072514844",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3425796056666666,
"range": "stddev: 0.01019099221239607",
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
"date": 1783553392272,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27953168960002583,
"range": "stddev: 0.0017902352682322997",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27996123119999083,
"range": "stddev: 0.0009286091608619576",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.321713665666645,
"range": "stddev: 0.0034113283625353315",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0518220573333488,
"range": "stddev: 0.037055814514424745",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.553270122666656,
"range": "stddev: 0.008015357371937114",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3404237223333364,
"range": "stddev: 0.009293578303269118",
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
"id": "776d247ba8021d6be42573c4660bd973f346c13a",
"message": "Migrate large cpp tests to the main repository (#596)",
"timestamp": "2026-06-25T16:27:28+02:00",
"tree_id": "cdbb8a7eda60cd8ae4970b3f1f88bcdea908d11e",
"url": "https://github.com/proximafusion/vmecpp/commit/776d247ba8021d6be42573c4660bd973f346c13a"
},
"date": 1783553392457,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2793751736000104,
"range": "stddev: 0.0023815274710671692",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2796333762000131,
"range": "stddev: 0.0011958232482543742",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3203990686666505,
"range": "stddev: 0.0003232679225794452",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0718435553333165,
"range": "stddev: 0.045331950899040616",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.5832636740000225,
"range": "stddev: 0.0149717580713956",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3591530823333262,
"range": "stddev: 0.012421835612794219",
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
"date": 1783553392622,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2841416169999775,
"range": "stddev: 0.003162844928989085",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27980982980000135,
"range": "stddev: 0.002322788571919973",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3213811646666425,
"range": "stddev: 0.008073289420250703",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.068037800666692,
"range": "stddev: 0.010499364843465031",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.568438375666649,
"range": "stddev: 0.03625006190727495",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3415198839999978,
"range": "stddev: 0.009780745904691728",
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
"id": "484b8fe6e4637e0102bb201926eccd6ebe7dd2e4",
"message": "Update bazel lock (#604)",
"timestamp": "2026-07-06T11:26:00+02:00",
"tree_id": "ece6e2bcf41bbd1e91c7a4448987c5cf5ac2049c",
"url": "https://github.com/proximafusion/vmecpp/commit/484b8fe6e4637e0102bb201926eccd6ebe7dd2e4"
},
"date": 1783553392346,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28518629359998615,
"range": "stddev: 0.004974895346209437",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28351438559998315,
"range": "stddev: 0.004296175976609062",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3236757673333461,
"range": "stddev: 0.008818400794789403",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.1067956486666994,
"range": "stddev: 0.025554267181343904",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.594508113333366,
"range": "stddev: 0.02775061083897218",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3701299446666628,
"range": "stddev: 0.02349894986300059",
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
"id": "d0e474d4230f3d421a34715808e10e4f356d9f5e",
"message": "Benchmark CI action remove code duplication (#605)",
"timestamp": "2026-07-08T00:34:04+02:00",
"tree_id": "7a02f9ebab9973f3b10d0358b82819547ff89b91",
"url": "https://github.com/proximafusion/vmecpp/commit/d0e474d4230f3d421a34715808e10e4f356d9f5e"
},
"date": 1783553392730,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28165844600002854,
"range": "stddev: 0.0026840042228261645",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28511147999993225,
"range": "stddev: 0.0031113083617361147",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.31770938400003,
"range": "stddev: 0.003481670303916837",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0830849943333,
"range": "stddev: 0.01892453317541773",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.607333394000004,
"range": "stddev: 0.006856028029292506",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3645468933333025,
"range": "stddev: 0.010932864522134692",
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
"id": "93456948ee947ff235d9bc22fc8ad074ec7a8978",
"message": "Continue to higher multigrid resolutions from hot restart (#603)",
"timestamp": "2026-07-08T00:34:44+02:00",
"tree_id": "c25024a7689b4e887598aa33971acd52d1d28f13",
"url": "https://github.com/proximafusion/vmecpp/commit/93456948ee947ff235d9bc22fc8ad074ec7a8978"
},
"date": 1783553392544,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28144066780000687,
"range": "stddev: 0.0025218469826082533",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27957241500002966,
"range": "stddev: 0.0011635740062133585",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3195015806666863,
"range": "stddev: 0.0058054975683048135",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0684557743333394,
"range": "stddev: 0.013454690766885758",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.592149836333344,
"range": "stddev: 0.03858382088347015",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3623724530000345,
"range": "stddev: 0.004435403968652209",
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
"date": 1783553392443,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28069455720001313,
"range": "stddev: 0.0031097358078186954",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.286297012399973,
"range": "stddev: 0.008604657602336235",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3214441729999937,
"range": "stddev: 0.0007481714279794088",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0949548493332864,
"range": "stddev: 0.032293438969919486",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.615949396666565,
"range": "stddev: 0.03427790455828255",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3414353016666307,
"range": "stddev: 0.01025299455940784",
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
"date": 1783553392809,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28791232560001845,
"range": "stddev: 0.00442850637744308",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28365696800005935,
"range": "stddev: 0.0025429605899853847",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.296660478666657,
"range": "stddev: 0.0021565322470058173",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.064991518666602,
"range": "stddev: 0.007151568033377335",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.587789011000041,
"range": "stddev: 0.004143926816315925",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3497207003333642,
"range": "stddev: 0.01230913117184297",
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
"date": 1783553392159,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2799683881999499,
"range": "stddev: 0.0029174411972602747",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.27758435259997893,
"range": "stddev: 0.0010970024797216078",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.311374723666707,
"range": "stddev: 0.003957537415919574",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.013882793999983,
"range": "stddev: 0.012225583555950069",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.540185622333411,
"range": "stddev: 0.02580354952796777",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.337514241666592,
"range": "stddev: 0.009955257131195673",
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
"date": 1783553392189,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2819110708000608,
"range": "stddev: 0.001208167324397838",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.278760273800026,
"range": "stddev: 0.001918448406030771",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.2970207866666594,
"range": "stddev: 0.0018861490626206345",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0230717999999492,
"range": "stddev: 0.025319269294325045",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.523974097333394,
"range": "stddev: 0.010086576478780024",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.351096615000036,
"range": "stddev: 0.009200846619585402",
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
"id": "d1c2328aee3ee8e7613ebcba1b023bcbcf041454",
"message": "Add C++ Google Benchmark microbenchmarks for critical hot functions (#606)",
"timestamp": "2026-07-08T08:48:02+02:00",
"tree_id": "ffb12cfbe7d56b38c4799b2acec380d6a211006f",
"url": "https://github.com/proximafusion/vmecpp/commit/d1c2328aee3ee8e7613ebcba1b023bcbcf041454"
},
"date": 1783553392732,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2890288378000605,
"range": "stddev: 0.004190789511567264",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.285834769199937,
"range": "stddev: 0.005612568325327619",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.2988358606667134,
"range": "stddev: 0.0015940859997805176",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.046671810333313,
"range": "stddev: 0.024147027674415456",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.520533105999978,
"range": "stddev: 0.020141530938896514",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3470304256666168,
"range": "stddev: 0.0027698712887285876",
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
"date": 1783553392671,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28413947379999627,
"range": "stddev: 0.002915693972255673",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2803744700000152,
"range": "stddev: 0.003015955873790509",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.298991806999993,
"range": "stddev: 0.002801138633813666",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0219007156667885,
"range": "stddev: 0.0066933185382540014",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.587669857666621,
"range": "stddev: 0.03815392002464734",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3688665356666359,
"range": "stddev: 0.013791047633860892",
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
"date": 1783553392735,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2802125940000678,
"range": "stddev: 0.001171329806691209",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2804909933999625,
"range": "stddev: 0.0018382821695520025",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3029131383334516,
"range": "stddev: 0.010905776519764125",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.04217204799996,
"range": "stddev: 0.03106737537433456",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.520039020666597,
"range": "stddev: 0.017633710370940445",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3539618406666705,
"range": "stddev: 0.015475121702652308",
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
"date": 1783553392610,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28100019180001257,
"range": "stddev: 0.0017437998420181703",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28054359059997297,
"range": "stddev: 0.0030784033069116506",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3077548646667008,
"range": "stddev: 0.012233431695524601",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.044243775999803,
"range": "stddev: 0.02153838300708066",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.522542826666722,
"range": "stddev: 0.007842625063467907",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3467143106666601,
"range": "stddev: 0.006897820864638349",
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
"date": 1783553392557,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2845619566000096,
"range": "stddev: 0.003920227062418815",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2810742392000975,
"range": "stddev: 0.0020974251416246105",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3014912629996616,
"range": "stddev: 0.003230054244477452",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0359242296667617,
"range": "stddev: 0.005045639277321064",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.584349277666964,
"range": "stddev: 0.0065621308947100285",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3575787719999728,
"range": "stddev: 0.01400081040164635",
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
"date": 1783553392702,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28106368680009836,
"range": "stddev: 0.0015911816758163444",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2775115551999988,
"range": "stddev: 0.0005192392674611893",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3035371746665685,
"range": "stddev: 0.003173421066580797",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0452557266668614,
"range": "stddev: 0.05486762642563357",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.560037959333537,
"range": "stddev: 0.010278346815424626",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.342042615333412,
"range": "stddev: 0.009762684517761563",
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
"date": 1783553392513,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.2841816635999749,
"range": "stddev: 0.004264196000273627",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2909963905999575,
"range": "stddev: 0.010428561475854903",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3135292713333608,
"range": "stddev: 0.006893477427830985",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.031756576333161,
"range": "stddev: 0.017814213192023278",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.5725351550001205,
"range": "stddev: 0.04820073179880365",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3697239963333534,
"range": "stddev: 0.01336121099038463",
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
"date": 1783553392804,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.28577002700003507,
"range": "stddev: 0.0010600383994597228",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2828574765998383,
"range": "stddev: 0.0038256435822600562",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.2868668540001333,
"range": "stddev: 0.0008164824823941988",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.007909845666594,
"range": "stddev: 0.019058722753096433",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.5663008380000365,
"range": "stddev: 0.11506891445424024",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3408825649999017,
"range": "stddev: 0.009366360049557801",
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
"date": 1783553392569,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.279822834999959,
"range": "stddev: 0.0009820498102702938",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2774531398000363,
"range": "stddev: 0.0015984425366487403",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.3047060610000092,
"range": "stddev: 0.0019327803223393673",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 2.9890507216668993,
"range": "stddev: 0.014388281184520697",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.532884990000032,
"range": "stddev: 0.019855539935493965",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.368862409999868,
"range": "stddev: 0.01653861187151445",
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
"date": 1783553392540,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27898371280016365,
"range": "stddev: 0.0020683058404098852",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.2802924953999536,
"range": "stddev: 0.003982559353144713",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.2937160103332037,
"range": "stddev: 0.002212921663231788",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.0052726216664873,
"range": "stddev: 0.013177361149791813",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.53585685066643,
"range": "stddev: 0.012634837420115438",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3564165783333617,
"range": "stddev: 0.0076764944541545515",
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
"id": "cd3dfc2d6facd85e77f076120b479ad1c92a5dbc",
"message": "Clean up benchmark on demandplots (#615)",
"timestamp": "2026-07-08T23:12:19+02:00",
"tree_id": "ab8b5c4ded4e43c9df5c47f8b23e03b29fe692e2",
"url": "https://github.com/proximafusion/vmecpp/commit/cd3dfc2d6facd85e77f076120b479ad1c92a5dbc"
},
"date": 1783553392713,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_invalid_input",
"unit": "seconds",
"value": 0.27882022740004686,
"range": "stddev: 0.0015046389828468154",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_cli_startup",
"unit": "seconds",
"value": 0.28084071559987933,
"range": "stddev: 0.0017096070452415982",
"extra": "rounds: 5"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_cma",
"unit": "seconds",
"value": 1.2948811670000093,
"range": "stddev: 0.007588260474656938",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_fixed_boundary_w7x",
"unit": "seconds",
"value": 3.057369982999868,
"range": "stddev: 0.05212806690296454",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_free_boundary",
"unit": "seconds",
"value": 7.522371202000007,
"range": "stddev: 0.00889086439846395",
"extra": "rounds: 3"
},
{
"name": "benchmarks/test_benchmarks.py::test_bench_response_table_from_coils",
"unit": "seconds",
"value": 1.3524326633332748,
"range": "stddev: 0.008714217744956008",
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
"date": 1783553392855,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022134455767544836,
"extra": "iterations: 1056\ncpu: 0.00022133757859848488 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020136380041134796,
"extra": "iterations: 1389\ncpu: 0.0002013663650107991 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026301752846196015,
"extra": "iterations: 106\ncpu: 0.0026301095754716975 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001854067606641757,
"extra": "iterations: 151\ncpu: 0.0018537406026490075 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006203672620985244,
"extra": "iterations: 45\ncpu: 0.006203769911111116 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0039517879486083984,
"extra": "iterations: 70\ncpu: 0.0039516595857142876 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002633270227684165,
"extra": "iterations: 106\ncpu: 0.002633157301886794 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018733437856038413,
"extra": "iterations: 150\ncpu: 0.0018730189133333338 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0053082429445706885,
"extra": "iterations: 52\ncpu: 0.005308111673076928 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007980523790631976,
"extra": "iterations: 35\ncpu: 0.007980650685714291 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010742444258469801,
"extra": "iterations: 26\ncpu: 0.010741521153846157 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026658940538067683,
"extra": "iterations: 107\ncpu: 0.002665467186915883 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018479526042938235,
"extra": "iterations: 152\ncpu: 0.0018478687763157906 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036794668749759074,
"extra": "iterations: 76\ncpu: 0.0036793058684210585 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002989121157713611,
"extra": "iterations: 99\ncpu: 0.0029890041717171786 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004356974647158668,
"extra": "iterations: 63\ncpu: 0.004356770492063498 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0038090929200377648,
"extra": "iterations: 79\ncpu: 0.003809146746835445 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.006015376841768306,
"extra": "iterations: 47\ncpu: 0.006014721744680845 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004892750790244655,
"extra": "iterations: 57\ncpu: 0.004892362175438589 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001152810123231676,
"extra": "iterations: 504\ncpu: 0.0005316596051587301 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015618047248428463,
"extra": "iterations: 389\ncpu: 0.0007214402030848336 seconds\nthreads: 1"
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
"date": 1783553392901,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021899186454464557,
"extra": "iterations: 1138\ncpu: 0.00021898507293497364 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002110771925257071,
"extra": "iterations: 1401\ncpu: 0.0002110653019271949 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026393649734069253,
"extra": "iterations: 107\ncpu: 0.002638810168224299 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018436610698699953,
"extra": "iterations: 152\ncpu: 0.0018436890065789472 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006367206573486328,
"extra": "iterations: 45\ncpu: 0.006366758000000004 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0040588043105434365,
"extra": "iterations: 71\ncpu: 0.004058732042253518 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002655340495862459,
"extra": "iterations: 95\ncpu: 0.0026552669473684227 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018735152996139021,
"extra": "iterations: 151\ncpu: 0.0018721554172185433 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005285996311115769,
"extra": "iterations: 53\ncpu: 0.005285884660377362 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007957560675484794,
"extra": "iterations: 35\ncpu: 0.007955745114285711 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010555322353656475,
"extra": "iterations: 26\ncpu: 0.010555509269230776 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026079048620206175,
"extra": "iterations: 107\ncpu: 0.0026075205700934567 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018258577857921327,
"extra": "iterations: 153\ncpu: 0.0018258882418300682 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003666319345173083,
"extra": "iterations: 76\ncpu: 0.0036659369210526303 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002810398737589518,
"extra": "iterations: 99\ncpu: 0.0028097758282828286 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004325855523347855,
"extra": "iterations: 64\ncpu: 0.00432555635937501 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003474865430667077,
"extra": "iterations: 81\ncpu: 0.003474914481481483 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0054639507742489084,
"extra": "iterations: 51\ncpu: 0.005463185764705876 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004672932624816895,
"extra": "iterations: 60\ncpu: 0.004672663249999999 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011616750449985845,
"extra": "iterations: 514\ncpu: 0.0005436165330739306 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001489596379140349,
"extra": "iterations: 389\ncpu: 0.000702279303341901 seconds\nthreads: 1"
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
"date": 1783553392838,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00029844592085005706,
"extra": "iterations: 709\ncpu: 0.00029782563469675597 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002009330477033343,
"extra": "iterations: 1400\ncpu: 0.00020092955357142852 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002618074417114258,
"extra": "iterations: 107\ncpu: 0.002617736411214954 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018523323614865741,
"extra": "iterations: 151\ncpu: 0.0018522519205298015 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006164111031426324,
"extra": "iterations: 45\ncpu: 0.006163908466666667 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003950718470982143,
"extra": "iterations: 70\ncpu: 0.003950005114285712 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002646012126274829,
"extra": "iterations: 106\ncpu: 0.0026459415094339635 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018547773361206057,
"extra": "iterations: 152\ncpu: 0.0018544503355263158 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005269590413795327,
"extra": "iterations: 53\ncpu: 0.005269501641509436 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008014699390956335,
"extra": "iterations: 35\ncpu: 0.008014415171428573 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01078493778522198,
"extra": "iterations: 26\ncpu: 0.010784257000000009 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026125194870422934,
"extra": "iterations: 107\ncpu: 0.002612128542056078 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001829627292608124,
"extra": "iterations: 153\ncpu: 0.0018291703071895427 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036713240982650165,
"extra": "iterations: 77\ncpu: 0.003671244194805203 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028201401835740215,
"extra": "iterations: 99\ncpu: 0.0028200772121212154 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004548402932974009,
"extra": "iterations: 65\ncpu: 0.0045476336153846145 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00349885989458133,
"extra": "iterations: 78\ncpu: 0.003498931282051283 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.006311755180358887,
"extra": "iterations: 50\ncpu: 0.006311863819999992 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004662295867656839,
"extra": "iterations: 58\ncpu: 0.004661959672413792 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011449615236893726,
"extra": "iterations: 521\ncpu: 0.0005246642495201528 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00159507922150872,
"extra": "iterations: 352\ncpu: 0.0007209784232954553 seconds\nthreads: 1"
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
"date": 1783553392911,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002204710100664582,
"extra": "iterations: 1279\ncpu: 0.00022042058092259584 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020127814897310752,
"extra": "iterations: 1398\ncpu: 0.0002012818240343348 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002620119914830288,
"extra": "iterations: 107\ncpu: 0.0026201593551401867 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018476388956371107,
"extra": "iterations: 152\ncpu: 0.0018475355789473677 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0061508708530002175,
"extra": "iterations: 45\ncpu: 0.006150961866666666 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003964625613790163,
"extra": "iterations: 71\ncpu: 0.003963952422535216 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026240994997113664,
"extra": "iterations: 107\ncpu: 0.002623931140186914 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018610796391569228,
"extra": "iterations: 151\ncpu: 0.0018611134370860932 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005394353316380427,
"extra": "iterations: 52\ncpu: 0.005392658134615391 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008099927621729234,
"extra": "iterations: 34\ncpu: 0.008100071941176464 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010817977098318247,
"extra": "iterations: 26\ncpu: 0.01081533199999999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026412411270854627,
"extra": "iterations: 107\ncpu: 0.0026410250560747676 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018311887005575342,
"extra": "iterations: 153\ncpu: 0.0018307473594771236 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036935520172119143,
"extra": "iterations: 75\ncpu: 0.0036933894266666678 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002838842722834373,
"extra": "iterations: 98\ncpu: 0.0028386953367346944 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004425469785928726,
"extra": "iterations: 64\ncpu: 0.004425551718750004 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034984576551220087,
"extra": "iterations: 79\ncpu: 0.0034980703924050597 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00574948836346062,
"extra": "iterations: 49\ncpu: 0.0057495943469387766 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0048294766195889184,
"extra": "iterations: 58\ncpu: 0.004829404948275866 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011620154747596155,
"extra": "iterations: 520\ncpu: 0.0005468594403846155 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016155668426167864,
"extra": "iterations: 353\ncpu: 0.0007269609603399428 seconds\nthreads: 1"
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
"date": 1783553392871,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002188540439979703,
"extra": "iterations: 1275\ncpu: 0.00021885685254901963 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002011309031853099,
"extra": "iterations: 1397\ncpu: 0.00020109230923407302 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026317021557103813,
"extra": "iterations: 107\ncpu: 0.0026317390093457947 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018465048388430947,
"extra": "iterations: 152\ncpu: 0.001846274032894737 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00622557004292806,
"extra": "iterations: 45\ncpu: 0.0062253904888888885 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003952439402190732,
"extra": "iterations: 71\ncpu: 0.0039524979295774655 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002630874795733758,
"extra": "iterations: 106\ncpu: 0.002630539547169813 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018672816800755381,
"extra": "iterations: 151\ncpu: 0.0018673074768211933 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005374087737156794,
"extra": "iterations: 52\ncpu: 0.005374150615384615 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008069004331316267,
"extra": "iterations: 35\ncpu: 0.008063026028571433 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010618833395150991,
"extra": "iterations: 26\ncpu: 0.010619009384615363 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026111050888344093,
"extra": "iterations: 108\ncpu: 0.0026111456388888924 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0019617158603044897,
"extra": "iterations: 153\ncpu: 0.0019614004444444416 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036524085255412314,
"extra": "iterations: 77\ncpu: 0.003652465766233763 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028229819403754342,
"extra": "iterations: 99\ncpu: 0.0028229011010101 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004364171394935021,
"extra": "iterations: 65\ncpu: 0.004363705276923073 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034831494092941286,
"extra": "iterations: 80\ncpu: 0.003482663850000001 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005466821146946328,
"extra": "iterations: 51\ncpu: 0.0054666210588235255 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004664413134257,
"extra": "iterations: 60\ncpu: 0.0046644772500000105 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011908715625978867,
"extra": "iterations: 530\ncpu: 0.0005548054528301882 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016197522069506078,
"extra": "iterations: 386\ncpu: 0.0007488911424870471 seconds\nthreads: 1"
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
"date": 1783553392876,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021910297627351723,
"extra": "iterations: 1225\ncpu: 0.00021909603755102043 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020239042497693935,
"extra": "iterations: 1389\ncpu: 0.0002023822886969043 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002626552760043991,
"extra": "iterations: 107\ncpu: 0.0026265953738317765 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018534455078327106,
"extra": "iterations: 151\ncpu: 0.0018534712185430454 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0062063958909776475,
"extra": "iterations: 45\ncpu: 0.006206493755555555 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003931421629140075,
"extra": "iterations: 71\ncpu: 0.003931071281690139 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002643322044948362,
"extra": "iterations: 106\ncpu: 0.002643284056603775 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018400063640192936,
"extra": "iterations: 152\ncpu: 0.0018397576381578953 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005356243023505578,
"extra": "iterations: 52\ncpu: 0.005356313134615385 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008040387289864677,
"extra": "iterations: 35\ncpu: 0.008038071199999993 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01080017823439378,
"extra": "iterations: 26\ncpu: 0.010799508461538447 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026150039423291933,
"extra": "iterations: 107\ncpu: 0.0026150432523364465 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018228091202773058,
"extra": "iterations: 154\ncpu: 0.0018227165194805149 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036519905189415077,
"extra": "iterations: 77\ncpu: 0.003652035259740256 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028224497130422883,
"extra": "iterations: 99\ncpu: 0.002822227070707067 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004326699330256536,
"extra": "iterations: 65\ncpu: 0.004326603046153848 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003562435691739306,
"extra": "iterations: 81\ncpu: 0.0035623893333333254 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005686720212300618,
"extra": "iterations: 48\ncpu: 0.0056850297708333315 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004671017328898112,
"extra": "iterations: 60\ncpu: 0.004671085916666672 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011247025893422102,
"extra": "iterations: 534\ncpu: 0.0005327354943820234 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0014979497823538136,
"extra": "iterations: 377\ncpu: 0.0007150842944297081 seconds\nthreads: 1"
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
"date": 1783553392860,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022087590188234928,
"extra": "iterations: 1267\ncpu: 0.00022085967403314918 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002078496222126528,
"extra": "iterations: 1355\ncpu: 0.00020784652915129153 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026366354149078655,
"extra": "iterations: 107\ncpu: 0.0026365949158878503 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018452761189037602,
"extra": "iterations: 151\ncpu: 0.0018453041059602647 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006211328506469727,
"extra": "iterations: 45\ncpu: 0.006211409511111112 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00393673064003528,
"extra": "iterations: 71\ncpu: 0.003936775971830984 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026408523883459704,
"extra": "iterations: 106\ncpu: 0.0026408905943396226 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001838418998216328,
"extra": "iterations: 152\ncpu: 0.0018384410065789487 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0053168017909211935,
"extra": "iterations: 53\ncpu: 0.00531688184905661 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00805443355015346,
"extra": "iterations: 35\ncpu: 0.008054046371428583 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010730119851919321,
"extra": "iterations: 26\ncpu: 0.010730289076923057 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002621815583416235,
"extra": "iterations: 107\ncpu: 0.002621754065420561 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001833269019532048,
"extra": "iterations: 153\ncpu: 0.0018331517843137276 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036559971896084876,
"extra": "iterations: 77\ncpu: 0.003655878636363633 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028132019620953183,
"extra": "iterations: 99\ncpu: 0.002813245939393942 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004329569637775421,
"extra": "iterations: 64\ncpu: 0.004329629906249994 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00347227156162262,
"extra": "iterations: 80\ncpu: 0.003472145725000009 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005552427441466089,
"extra": "iterations: 51\ncpu: 0.005552515215686277 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004664103190104167,
"extra": "iterations: 60\ncpu: 0.004664163616666676 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011328859254717827,
"extra": "iterations: 512\ncpu: 0.0005318404707031246 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001699251041077731,
"extra": "iterations: 399\ncpu: 0.0007769992280701743 seconds\nthreads: 1"
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
"date": 1783553392888,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002583800529947086,
"extra": "iterations: 784\ncpu: 0.00025824943367346937 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020108730307934872,
"extra": "iterations: 1391\ncpu: 0.00020109017181883538 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002640613969766869,
"extra": "iterations: 106\ncpu: 0.002640438037735849 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018461174086520547,
"extra": "iterations: 152\ncpu: 0.0018461439671052632 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00617256694369846,
"extra": "iterations: 45\ncpu: 0.006171898111111115 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003937835424718723,
"extra": "iterations: 71\ncpu: 0.00393790836619718 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002632161356368155,
"extra": "iterations: 106\ncpu: 0.0026322050377358473 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018487839322341116,
"extra": "iterations: 152\ncpu: 0.001848745532894737 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005314889943824624,
"extra": "iterations: 53\ncpu: 0.005314816528301882 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008030271530151368,
"extra": "iterations: 35\ncpu: 0.008029173685714281 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010719537734985352,
"extra": "iterations: 26\ncpu: 0.010719692923076927 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026232973437442957,
"extra": "iterations: 107\ncpu: 0.002623109261682238 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018290133258096536,
"extra": "iterations: 153\ncpu: 0.0018289131960784338 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003657567036616338,
"extra": "iterations: 77\ncpu: 0.0036574617532467595 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028175272122778075,
"extra": "iterations: 99\ncpu: 0.0028172178989898998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0043160365178034855,
"extra": "iterations: 65\ncpu: 0.004315954599999992 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034970512873009796,
"extra": "iterations: 79\ncpu: 0.003497099202531649 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005506445379818188,
"extra": "iterations: 51\ncpu: 0.005505271843137266 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004668267567952474,
"extra": "iterations: 60\ncpu: 0.004667958266666676 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010681333597640545,
"extra": "iterations: 513\ncpu: 0.0004930528382066276 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015305814230838395,
"extra": "iterations: 391\ncpu: 0.0007120781023017882 seconds\nthreads: 1"
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
"date": 1783553392850,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021920877344468062,
"extra": "iterations: 1275\ncpu: 0.00021919972313725496 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002011697586268595,
"extra": "iterations: 1387\ncpu: 0.00020116332588320117 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002637595500586168,
"extra": "iterations: 106\ncpu: 0.0026375000849056603 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018426415167356793,
"extra": "iterations: 152\ncpu: 0.0018426681447368432 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006268501281738281,
"extra": "iterations: 45\ncpu: 0.006268121777777782 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003943171299679179,
"extra": "iterations: 71\ncpu: 0.003943226690140849 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026440350514537886,
"extra": "iterations: 106\ncpu: 0.0026437948207547195 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.00231534481048584,
"extra": "iterations: 100\ncpu: 0.0023151781000000015 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005652776131263146,
"extra": "iterations: 52\ncpu: 0.005652627653846154 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008021899632045202,
"extra": "iterations: 35\ncpu: 0.008021167800000003 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010935737536503719,
"extra": "iterations: 26\ncpu: 0.010935912807692308 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002616022234765169,
"extra": "iterations: 107\ncpu: 0.0026159604018691648 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018386622659521167,
"extra": "iterations: 153\ncpu: 0.001838508640522876 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036629268101283485,
"extra": "iterations: 77\ncpu: 0.003662980350649341 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028210456925209125,
"extra": "iterations: 99\ncpu: 0.0028207872424242395 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004326332532442534,
"extra": "iterations: 65\ncpu: 0.004326094600000005 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034863203763961796,
"extra": "iterations: 80\ncpu: 0.00348625106249999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0054863153719434555,
"extra": "iterations: 51\ncpu: 0.005485998078431364 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00467065175374349,
"extra": "iterations: 60\ncpu: 0.004670438466666675 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011282766234462162,
"extra": "iterations: 549\ncpu: 0.0005135877905282324 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001445535976107758,
"extra": "iterations: 404\ncpu: 0.0006747662128712866 seconds\nthreads: 1"
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
"date": 1783553392933,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00028015528353030403,
"extra": "iterations: 957\ncpu: 0.0002801451912225706 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020086191920864438,
"extra": "iterations: 1393\ncpu: 0.0002008467063890883 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002625321442226194,
"extra": "iterations: 106\ncpu: 0.0026253597735849056 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018409725866819684,
"extra": "iterations: 152\ncpu: 0.0018408526118421046 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006187317106458877,
"extra": "iterations: 45\ncpu: 0.006187416933333337 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003950686521933112,
"extra": "iterations: 71\ncpu: 0.003950742788732395 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002626116030684142,
"extra": "iterations: 107\ncpu: 0.0026258818785046727 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018426258313028437,
"extra": "iterations: 152\ncpu: 0.0018425593026315788 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005327031297503777,
"extra": "iterations: 53\ncpu: 0.005326722339622645 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008047328676496233,
"extra": "iterations: 35\ncpu: 0.00804583331428572 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010778427124023438,
"extra": "iterations: 26\ncpu: 0.010775738884615384 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026235469033784957,
"extra": "iterations: 107\ncpu: 0.00262350106542056 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018507913248428447,
"extra": "iterations: 151\ncpu: 0.0018505003112582767 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003650696246654957,
"extra": "iterations: 77\ncpu: 0.003650413467532461 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028318395518293284,
"extra": "iterations: 99\ncpu: 0.0028318845555555597 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004333160817623138,
"extra": "iterations: 64\ncpu: 0.004332439703124996 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034824388998526117,
"extra": "iterations: 81\ncpu: 0.003482317950617293 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005542413861143823,
"extra": "iterations: 51\ncpu: 0.005542500117647061 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00470723311106364,
"extra": "iterations: 60\ncpu: 0.00470621961666667 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010505816676648582,
"extra": "iterations: 523\ncpu: 0.0004988258279158711 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0014995599869028124,
"extra": "iterations: 421\ncpu: 0.0006699091425178162 seconds\nthreads: 1"
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
"date": 1783553392897,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00031680096325476404,
"extra": "iterations: 707\ncpu: 0.00031676269024045263 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020147506080379557,
"extra": "iterations: 1385\ncpu: 0.00020146760505415163 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002609449012257229,
"extra": "iterations: 107\ncpu: 0.0026093908224299072 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018506034320553407,
"extra": "iterations: 151\ncpu: 0.0018505189403973514 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006141154662422512,
"extra": "iterations: 46\ncpu: 0.006141244195652173 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004053616869276849,
"extra": "iterations: 69\ncpu: 0.004053519043478259 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026423121398349977,
"extra": "iterations: 106\ncpu: 0.002642112698113206 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018467824710042854,
"extra": "iterations: 152\ncpu: 0.0018468147697368422 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0053438452573922966,
"extra": "iterations: 52\ncpu: 0.005343698403846154 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007955258233206614,
"extra": "iterations: 35\ncpu: 0.007955402314285707 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010521685635602034,
"extra": "iterations: 27\ncpu: 0.01052185225925926 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026005718195549797,
"extra": "iterations: 107\ncpu: 0.0026004055981308487 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018287920484355853,
"extra": "iterations: 153\ncpu: 0.0018288212287581678 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003669211738987973,
"extra": "iterations: 76\ncpu: 0.003668813710526321 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002823853733563664,
"extra": "iterations: 99\ncpu: 0.0028239025858585874 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004320275038480759,
"extra": "iterations: 64\ncpu: 0.004320348140624999 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034886002540588383,
"extra": "iterations: 80\ncpu: 0.0034884792624999997 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00547926098692651,
"extra": "iterations: 51\ncpu: 0.005479351137254897 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004681102434794109,
"extra": "iterations: 60\ncpu: 0.004681181750000009 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011120446708932025,
"extra": "iterations: 513\ncpu: 0.0005289539922027284 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0019256859290890578,
"extra": "iterations: 410\ncpu: 0.0008107885195121966 seconds\nthreads: 1"
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
"date": 1783553392923,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022004627975115714,
"extra": "iterations: 1271\ncpu: 0.00022004906215578287 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020051976685882896,
"extra": "iterations: 1395\ncpu: 0.00020048830465949825 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026930524752690243,
"extra": "iterations: 104\ncpu: 0.00269290251923077 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001836779849981171,
"extra": "iterations: 153\ncpu: 0.0018367473202614386 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006129575812298319,
"extra": "iterations: 46\ncpu: 0.006129176065217394 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003940652793561908,
"extra": "iterations: 71\ncpu: 0.003940427014084509 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0027051934829125037,
"extra": "iterations: 104\ncpu: 0.002704982692307691 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018497160728404064,
"extra": "iterations: 151\ncpu: 0.0018497458145695363 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005291812824753096,
"extra": "iterations: 53\ncpu: 0.005291657358490565 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008082818984985352,
"extra": "iterations: 35\ncpu: 0.00808242142857143 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010555038085350623,
"extra": "iterations: 26\ncpu: 0.010555189923076915 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002732515335083008,
"extra": "iterations: 107\ncpu: 0.002732082177570093 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018360244261251915,
"extra": "iterations: 148\ncpu: 0.0018360520675675637 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036678654806954524,
"extra": "iterations: 77\ncpu: 0.0036675367402597366 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002814040039524888,
"extra": "iterations: 99\ncpu: 0.0028138401515151474 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004310710613544171,
"extra": "iterations: 65\ncpu: 0.004310432230769233 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034893324345718194,
"extra": "iterations: 81\ncpu: 0.00348938066666667 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005511195051903818,
"extra": "iterations: 51\ncpu: 0.005510886058823529 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004647552967071533,
"extra": "iterations: 60\ncpu: 0.004647622700000002 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010972076387547735,
"extra": "iterations: 536\ncpu: 0.0005141370708955224 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015076104469579125,
"extra": "iterations: 409\ncpu: 0.0007005461198044003 seconds\nthreads: 1"
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
"date": 1783553392890,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021881672583619598,
"extra": "iterations: 1277\ncpu: 0.00021882031245105718 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020115755209792807,
"extra": "iterations: 1394\ncpu: 0.00020112551578192254 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002634015038748768,
"extra": "iterations: 107\ncpu: 0.0026339600654205616 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018540328701600335,
"extra": "iterations: 151\ncpu: 0.001853990529801325 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006130617597828741,
"extra": "iterations: 46\ncpu: 0.00613050410869565 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003986624308994839,
"extra": "iterations: 70\ncpu: 0.003986549871428575 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026602722349621005,
"extra": "iterations: 105\ncpu: 0.0026599221714285744 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018483281925024578,
"extra": "iterations: 151\ncpu: 0.001848360569536424 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005272919276975236,
"extra": "iterations: 53\ncpu: 0.005273024415094336 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008075068978702322,
"extra": "iterations: 34\ncpu: 0.00807466858823529 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010521977036087602,
"extra": "iterations: 27\ncpu: 0.010521652333333338 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002612220906765662,
"extra": "iterations: 107\ncpu: 0.0026122615420560756 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018323870266185088,
"extra": "iterations: 153\ncpu: 0.001832412777777779 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036533914114299576,
"extra": "iterations: 76\ncpu: 0.003653437684210524 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028156824786253654,
"extra": "iterations: 99\ncpu: 0.002815250181818179 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0043124125554011425,
"extra": "iterations: 65\ncpu: 0.00431247310769231 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003489208221435547,
"extra": "iterations: 80\ncpu: 0.0034892579124999996 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005489237168255975,
"extra": "iterations: 51\ncpu: 0.005489006470588225 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004678205649058025,
"extra": "iterations: 60\ncpu: 0.004678279933333334 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001051373915238814,
"extra": "iterations: 550\ncpu: 0.0004999426618181828 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015046957062511911,
"extra": "iterations: 410\ncpu: 0.0006843680902439015 seconds\nthreads: 1"
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
"date": 1783553392945,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022883494318379048,
"extra": "iterations: 1266\ncpu: 0.000228837560821485 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002040108855889768,
"extra": "iterations: 1372\ncpu: 0.00020397776822157434 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026209443529075552,
"extra": "iterations: 107\ncpu: 0.0026209795420560757 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001861280163392326,
"extra": "iterations: 151\ncpu: 0.001861303986754967 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0061176859814187756,
"extra": "iterations: 46\ncpu: 0.006117353456521738 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00396798697995468,
"extra": "iterations: 71\ncpu: 0.003967565535211266 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.00264878992764455,
"extra": "iterations: 106\ncpu: 0.0026488272075471694 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018707393799852205,
"extra": "iterations: 149\ncpu: 0.0018706849060402687 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00668774201319768,
"extra": "iterations: 52\ncpu: 0.0066876156153846174 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008104351588657924,
"extra": "iterations: 35\ncpu: 0.008103678685714273 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010543885983918843,
"extra": "iterations: 19\ncpu: 0.010543300368421087 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026108371877224647,
"extra": "iterations: 107\ncpu: 0.0026108742429906523 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018296226278527992,
"extra": "iterations: 154\ncpu: 0.0018295226363636327 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036615133285522465,
"extra": "iterations: 76\ncpu: 0.003661565302631579 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002823121619947029,
"extra": "iterations: 99\ncpu: 0.002822810313131318 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004314228204580455,
"extra": "iterations: 65\ncpu: 0.004313982599999993 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034894227981567385,
"extra": "iterations: 80\ncpu: 0.0034894797500000024 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005490803718566895,
"extra": "iterations: 50\ncpu: 0.005490665280000009 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004672400156656902,
"extra": "iterations: 60\ncpu: 0.004672182033333336 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010878564654916959,
"extra": "iterations: 552\ncpu: 0.0005126025869565217 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015177853821497884,
"extra": "iterations: 394\ncpu: 0.0007103535329949227 seconds\nthreads: 1"
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
"date": 1783553392916,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002178407051193882,
"extra": "iterations: 1278\ncpu: 0.00021779589123630674 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020421786850226446,
"extra": "iterations: 1399\ncpu: 0.00020421455897069336 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002653101705155283,
"extra": "iterations: 106\ncpu: 0.002653133235849058 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.00186097700864274,
"extra": "iterations: 151\ncpu: 0.001860892119205299 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006257465150621202,
"extra": "iterations: 45\ncpu: 0.006257572333333335 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0040060247693743025,
"extra": "iterations: 70\ncpu: 0.004005479599999999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.00265056682082842,
"extra": "iterations: 106\ncpu: 0.0026505123490566016 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018389334804133366,
"extra": "iterations: 152\ncpu: 0.001838878434210527 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005406228395608755,
"extra": "iterations: 52\ncpu: 0.005406020480769231 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008040280903086942,
"extra": "iterations: 34\ncpu: 0.008040156088235285 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010692504736093374,
"extra": "iterations: 26\ncpu: 0.010691210192307702 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002739940370832171,
"extra": "iterations: 105\ncpu: 0.002739981180952381 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018828448496366803,
"extra": "iterations: 152\ncpu: 0.0018827048289473682 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003664556302522358,
"extra": "iterations: 76\ncpu: 0.0036643989868420973 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028184519873725045,
"extra": "iterations: 99\ncpu: 0.0028184976767676763 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00438443198800087,
"extra": "iterations: 64\ncpu: 0.004383899343750011 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034836083650588994,
"extra": "iterations: 80\ncpu: 0.0034835344125000002 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0055083293540804995,
"extra": "iterations: 51\ncpu: 0.005508403745098029 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00466233491897583,
"extra": "iterations: 60\ncpu: 0.004661769400000004 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010096533108601528,
"extra": "iterations: 565\ncpu: 0.00047238537876106204 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0014880459483076886,
"extra": "iterations: 410\ncpu: 0.0006957717658536598 seconds\nthreads: 1"
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
"date": 1783553392864,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022098763658123997,
"extra": "iterations: 1265\ncpu: 0.0002209238308300396 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020442238354474395,
"extra": "iterations: 1372\ncpu: 0.00020442516107871725 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026642118181501116,
"extra": "iterations: 105\ncpu: 0.0026637428095238097 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018475734634904675,
"extra": "iterations: 151\ncpu: 0.0018475960596026496 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006274551815456815,
"extra": "iterations: 45\ncpu: 0.006274641933333334 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004030735596366551,
"extra": "iterations: 69\ncpu: 0.004029891028985508 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026134230472423413,
"extra": "iterations: 108\ncpu: 0.002613255731481482 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018310998779496337,
"extra": "iterations: 153\ncpu: 0.0018308667516339875 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0053286237536736255,
"extra": "iterations: 53\ncpu: 0.0053279770566037775 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007923187528337752,
"extra": "iterations: 35\ncpu: 0.007923317171428579 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010638071940495418,
"extra": "iterations: 26\ncpu: 0.010637661769230755 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002625289952979898,
"extra": "iterations: 106\ncpu: 0.0026251985943396223 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018417289833617368,
"extra": "iterations: 153\ncpu: 0.0018414730326797371 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036692274244208086,
"extra": "iterations: 76\ncpu: 0.003669272934210523 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002822302808665266,
"extra": "iterations: 99\ncpu: 0.0028223468383838397 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004327733700092023,
"extra": "iterations: 65\ncpu: 0.004326991153846157 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003483420610427857,
"extra": "iterations: 80\ncpu: 0.003483205549999991 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005473721261117973,
"extra": "iterations: 51\ncpu: 0.005473802117647061 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004666622479756673,
"extra": "iterations: 60\ncpu: 0.0046665282999999984 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010264402180048774,
"extra": "iterations: 519\ncpu: 0.0004926423005780347 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0014616674412795104,
"extra": "iterations: 366\ncpu: 0.0006784788224043723 seconds\nthreads: 1"
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
"date": 1783553392845,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022055388059698627,
"extra": "iterations: 1271\ncpu: 0.0002203792077104642 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020102531679214972,
"extra": "iterations: 1395\ncpu: 0.0002010173784946237 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026204897978595485,
"extra": "iterations: 107\ncpu: 0.002620424504672897 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018625322977701825,
"extra": "iterations: 150\ncpu: 0.0018624767066666664 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006180699666341146,
"extra": "iterations: 45\ncpu: 0.0061807875333333324 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003930672793321207,
"extra": "iterations: 71\ncpu: 0.003929923380281692 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026835113201501234,
"extra": "iterations: 106\ncpu: 0.0026833549528301882 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001858373351444472,
"extra": "iterations: 151\ncpu: 0.0018584012582781458 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005324029005490816,
"extra": "iterations: 52\ncpu: 0.005322810461538459 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.0079923152923584,
"extra": "iterations: 35\ncpu: 0.007992435371428564 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010732384828420786,
"extra": "iterations: 26\ncpu: 0.010730422346153838 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002647147988373379,
"extra": "iterations: 106\ncpu: 0.0026470296415094306 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018714920679728192,
"extra": "iterations: 150\ncpu: 0.001871263906666663 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036882734917975096,
"extra": "iterations: 77\ncpu: 0.00368802174025974 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028483771314524644,
"extra": "iterations: 99\ncpu: 0.002848422979797975 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004385806620121002,
"extra": "iterations: 64\ncpu: 0.0043849898593750075 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003485396504402161,
"extra": "iterations: 80\ncpu: 0.0034850474000000077 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005493954116222905,
"extra": "iterations: 51\ncpu: 0.005493761588235297 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004721403121948242,
"extra": "iterations: 60\ncpu: 0.004720805833333334 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00103344236101423,
"extra": "iterations: 525\ncpu: 0.0004890089866666665 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015124765950180878,
"extra": "iterations: 389\ncpu: 0.0006834202853470441 seconds\nthreads: 1"
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
"date": 1783553392940,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022047604664717573,
"extra": "iterations: 1010\ncpu: 0.00022045401188118815 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002037711527155734,
"extra": "iterations: 1392\ncpu: 0.0002037480150862069 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026200177534571236,
"extra": "iterations: 106\ncpu: 0.0026200527358490574 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018494334441936568,
"extra": "iterations: 151\ncpu: 0.0018491927483443714 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006168497933281792,
"extra": "iterations: 45\ncpu: 0.006168591400000001 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00397993837084089,
"extra": "iterations: 70\ncpu: 0.003979988842857142 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026238165169118724,
"extra": "iterations: 107\ncpu: 0.002623482149532709 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018564991603623954,
"extra": "iterations: 151\ncpu: 0.0018564734238410605 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005261083818831534,
"extra": "iterations: 53\ncpu: 0.005260617849056601 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00801997184753418,
"extra": "iterations: 35\ncpu: 0.008020081285714286 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010644032404972957,
"extra": "iterations: 26\ncpu: 0.010644199923076926 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002665746779668899,
"extra": "iterations: 105\ncpu: 0.0026655476857142856 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001827078896599847,
"extra": "iterations: 148\ncpu: 0.001827107006756758 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003657387448595716,
"extra": "iterations: 77\ncpu: 0.0036568913116883086 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028139209747314454,
"extra": "iterations: 100\ncpu: 0.002813970380000006 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004308239106209048,
"extra": "iterations: 62\ncpu: 0.004308300903225798 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003474545478820801,
"extra": "iterations: 80\ncpu: 0.0034742756374999905 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0055018266042073565,
"extra": "iterations: 51\ncpu: 0.005501537666666675 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0046697219212849935,
"extra": "iterations: 60\ncpu: 0.00466979133333334 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010028395033974684,
"extra": "iterations: 524\ncpu: 0.000484143944656488 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015086074068088725,
"extra": "iterations: 396\ncpu: 0.0006920087323232305 seconds\nthreads: 1"
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
"date": 1783553392836,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022189559006109471,
"extra": "iterations: 1230\ncpu: 0.00022184691056910576 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020053765292843013,
"extra": "iterations: 1398\ncpu: 0.000200500147353362 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026788802373976938,
"extra": "iterations: 105\ncpu: 0.002678918980952381 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018807241580630308,
"extra": "iterations: 149\ncpu: 0.0018805859597315442 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006239345338609484,
"extra": "iterations: 45\ncpu: 0.00623907126666667 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004385797794048603,
"extra": "iterations: 65\ncpu: 0.004385729553846153 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002876187548225309,
"extra": "iterations: 81\ncpu: 0.0028759355555555547 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002002415657043457,
"extra": "iterations: 100\ncpu: 0.002002103400000004 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005297912741607091,
"extra": "iterations: 53\ncpu: 0.0052977615471698014 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007944093431745257,
"extra": "iterations: 35\ncpu: 0.007943477114285712 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010599585679861216,
"extra": "iterations: 26\ncpu: 0.010599776192307703 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002644379183931171,
"extra": "iterations: 106\ncpu: 0.0026444213207547198 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001832323136672475,
"extra": "iterations: 153\ncpu: 0.0018321692091503305 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003760062254868545,
"extra": "iterations: 77\ncpu: 0.0037601140909090894 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002833498848809136,
"extra": "iterations: 99\ncpu: 0.0028331253838383853 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004344709102924054,
"extra": "iterations: 65\ncpu: 0.004344554200000007 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034906536340713504,
"extra": "iterations: 80\ncpu: 0.003490561849999996 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00548388911228554,
"extra": "iterations: 51\ncpu: 0.005483054137254898 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0046902656555175785,
"extra": "iterations: 60\ncpu: 0.004690109533333332 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001107633113861084,
"extra": "iterations: 532\ncpu: 0.0005210523477443618 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015672496088166585,
"extra": "iterations: 399\ncpu: 0.000739919521303257 seconds\nthreads: 1"
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
"date": 1783553392899,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021914028057196968,
"extra": "iterations: 1277\ncpu: 0.00021913785591229452 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020321039682805796,
"extra": "iterations: 1398\ncpu: 0.000203195839055794 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002620344964143272,
"extra": "iterations: 107\ncpu: 0.002620056485981308 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001844867279655055,
"extra": "iterations: 152\ncpu: 0.0018448916447368433 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006170553631252713,
"extra": "iterations: 45\ncpu: 0.006170654244444442 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003932697672239492,
"extra": "iterations: 71\ncpu: 0.003932517943661969 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002687257069807786,
"extra": "iterations: 104\ncpu: 0.002687192211538463 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001856069691133815,
"extra": "iterations: 151\ncpu: 0.0018557507615894046 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005290679211886424,
"extra": "iterations: 53\ncpu: 0.005290304622641503 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00803565297807966,
"extra": "iterations: 35\ncpu: 0.008035764057142852 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010698428520789513,
"extra": "iterations: 26\ncpu: 0.010697964038461545 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0029284189332206296,
"extra": "iterations: 106\ncpu: 0.002928471216981129 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001832645702985377,
"extra": "iterations: 153\ncpu: 0.0018325893137254926 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003657498917022309,
"extra": "iterations: 77\ncpu: 0.003657552155844156 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002824398002239189,
"extra": "iterations: 99\ncpu: 0.002824448343434347 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004330992698669434,
"extra": "iterations: 64\ncpu: 0.00433083857812501 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003477790951728821,
"extra": "iterations: 80\ncpu: 0.0034778428249999994 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005529571981991039,
"extra": "iterations: 51\ncpu: 0.0055296406078431245 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004741748174031575,
"extra": "iterations: 60\ncpu: 0.00474156133333333 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011000495861010924,
"extra": "iterations: 538\ncpu: 0.0005227549832713749 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015875925374096748,
"extra": "iterations: 363\ncpu: 0.0007429670743801662 seconds\nthreads: 1"
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
"date": 1783553392921,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021974009032533462,
"extra": "iterations: 1276\ncpu: 0.00021970452194357368 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020471882820129395,
"extra": "iterations: 1000\ncpu: 0.00020472221699999998 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002649298338132484,
"extra": "iterations: 107\ncpu: 0.002649336560747664 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001848435401916504,
"extra": "iterations: 150\ncpu: 0.001848117906666668 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006307946311102973,
"extra": "iterations: 45\ncpu: 0.006306941288888891 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0039712258747645795,
"extra": "iterations: 70\ncpu: 0.003971285185714286 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026449842273064384,
"extra": "iterations: 106\ncpu: 0.002644909773584906 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018723154067993165,
"extra": "iterations: 150\ncpu: 0.0018723444399999984 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005377058712941296,
"extra": "iterations: 53\ncpu: 0.0053772009622641585 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00845223484617291,
"extra": "iterations: 33\ncpu: 0.008452022757575749 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010553457118846752,
"extra": "iterations: 27\ncpu: 0.010553655814814827 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.00263371782482795,
"extra": "iterations: 106\ncpu: 0.002633632452830194 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018749629196367769,
"extra": "iterations: 152\ncpu: 0.0018748431447368431 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036694250608745376,
"extra": "iterations: 76\ncpu: 0.0036694015526315834 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028546999911872708,
"extra": "iterations: 98\ncpu: 0.0028546499693877523 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00430754514840933,
"extra": "iterations: 65\ncpu: 0.004307605876923076 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003500911593437195,
"extra": "iterations: 80\ncpu: 0.0035006537250000026 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005462997099932502,
"extra": "iterations: 51\ncpu: 0.005463084999999998 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00467835267384847,
"extra": "iterations: 60\ncpu: 0.004678417816666662 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011295214505262778,
"extra": "iterations: 568\ncpu: 0.0005337298169014097 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001476462341185869,
"extra": "iterations: 373\ncpu: 0.000681129249329758 seconds\nthreads: 1"
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
"date": 1783553392874,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002213654063996815,
"extra": "iterations: 1260\ncpu: 0.00022133110476190478 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020604198738279146,
"extra": "iterations: 1391\ncpu: 0.000206045294751977 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026278206121141665,
"extra": "iterations: 107\ncpu: 0.0026278641775700938 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018595441182454428,
"extra": "iterations: 150\ncpu: 0.001859501886666668 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006152015262179905,
"extra": "iterations: 45\ncpu: 0.006151911711111114 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0040066135463429926,
"extra": "iterations: 67\ncpu: 0.004006053358208956 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026382050424251916,
"extra": "iterations: 106\ncpu: 0.0026381545471698105 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018454159007352943,
"extra": "iterations: 153\ncpu: 0.001845381601307189 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005417819206531231,
"extra": "iterations: 52\ncpu: 0.005417698615384608 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007986123221261162,
"extra": "iterations: 35\ncpu: 0.007985599771428565 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010506629943847656,
"extra": "iterations: 27\ncpu: 0.010505309111111121 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026077628135681152,
"extra": "iterations: 108\ncpu: 0.0026078053981481502 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001835935256060432,
"extra": "iterations: 153\ncpu: 0.0018355738823529395 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036779297025580156,
"extra": "iterations: 76\ncpu: 0.0036778456710526357 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028389191627502443,
"extra": "iterations: 100\ncpu: 0.00283880108 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004322657218346229,
"extra": "iterations: 65\ncpu: 0.004322717738461534 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003479698557912568,
"extra": "iterations: 81\ncpu: 0.0034794194567901292 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005461692810058594,
"extra": "iterations: 51\ncpu: 0.005461773843137246 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004746329784393311,
"extra": "iterations: 60\ncpu: 0.0047463830166666705 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010374917433812068,
"extra": "iterations: 520\ncpu: 0.0004980138634615379 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0014739967974163684,
"extra": "iterations: 407\ncpu: 0.0006945073562653575 seconds\nthreads: 1"
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
"date": 1783553392857,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022168994984856345,
"extra": "iterations: 1267\ncpu: 0.00022169339936858724 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020155804131620676,
"extra": "iterations: 1395\ncpu: 0.00020152435483870973 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002624624180343916,
"extra": "iterations: 106\ncpu: 0.00262466412264151 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018523607822443476,
"extra": "iterations: 151\ncpu: 0.0018522821589403978 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006171454323662652,
"extra": "iterations: 45\ncpu: 0.006171093577777785 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003989379746573312,
"extra": "iterations: 70\ncpu: 0.0039892772 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026568912324451267,
"extra": "iterations: 105\ncpu: 0.0026568060190476205 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018684848149617514,
"extra": "iterations: 150\ncpu: 0.001868512546666666 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005361038904923659,
"extra": "iterations: 52\ncpu: 0.005360945788461546 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008156012086307302,
"extra": "iterations: 34\ncpu: 0.008155777088235294 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010536697175767686,
"extra": "iterations: 27\ncpu: 0.010536860851851864 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002622622195805345,
"extra": "iterations: 107\ncpu: 0.0026222913177570114 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018470632402520432,
"extra": "iterations: 152\ncpu: 0.0018470940657894757 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003667646332790977,
"extra": "iterations: 76\ncpu: 0.0036674867236842156 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028324079031896115,
"extra": "iterations: 99\ncpu: 0.0028323622424242423 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004326802033644456,
"extra": "iterations: 65\ncpu: 0.004326672230769232 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003483209013938904,
"extra": "iterations: 80\ncpu: 0.0034830824749999996 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005525832082711015,
"extra": "iterations: 51\ncpu: 0.005525913509803935 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004707153638203939,
"extra": "iterations: 60\ncpu: 0.004707028016666663 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011194986956460137,
"extra": "iterations: 560\ncpu: 0.000530075187500001 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001531876580563584,
"extra": "iterations: 346\ncpu: 0.000703143066473989 seconds\nthreads: 1"
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
"date": 1783553392869,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021954589586242965,
"extra": "iterations: 1274\ncpu: 0.00021954291365777084 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020186479341252188,
"extra": "iterations: 1385\ncpu: 0.00020184815667870036 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026190637428069787,
"extra": "iterations: 107\ncpu: 0.0026190990934579434 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018522881514189261,
"extra": "iterations: 151\ncpu: 0.001852241509933775 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006178945965237088,
"extra": "iterations: 45\ncpu: 0.006178377511111115 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0039492929485482236,
"extra": "iterations: 71\ncpu: 0.003949336746478876 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026549100875854492,
"extra": "iterations: 106\ncpu: 0.0026549475283018887 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018457738976729544,
"extra": "iterations: 152\ncpu: 0.0018457377105263156 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005296072869930627,
"extra": "iterations: 53\ncpu: 0.005295928622641511 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008226633071899414,
"extra": "iterations: 35\ncpu: 0.008225965914285715 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010587233763474684,
"extra": "iterations: 26\ncpu: 0.010586991461538443 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002623179248560255,
"extra": "iterations: 107\ncpu: 0.0026232189532710264 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018427058270103055,
"extra": "iterations: 152\ncpu: 0.0018427292894736872 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003663654451246386,
"extra": "iterations: 77\ncpu: 0.003663704155844157 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028141484116062975,
"extra": "iterations: 99\ncpu: 0.002814036505050509 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004348057966965895,
"extra": "iterations: 65\ncpu: 0.004347943215384626 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034790480578387227,
"extra": "iterations: 81\ncpu: 0.003479102444444449 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005496473873362822,
"extra": "iterations: 51\ncpu: 0.005495893960784316 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004644612471262614,
"extra": "iterations: 60\ncpu: 0.004644674816666668 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010482923666328711,
"extra": "iterations: 517\ncpu: 0.000494258926499033 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015326663648724884,
"extra": "iterations: 367\ncpu: 0.0007188750136239785 seconds\nthreads: 1"
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
"date": 1783553392843,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00021906558703999274,
"extra": "iterations: 1265\ncpu: 0.00021906896600790515 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002003150948127452,
"extra": "iterations: 1398\ncpu: 0.00020031764377682405 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026162940765095647,
"extra": "iterations: 107\ncpu: 0.002616134869158879 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.00183701985760739,
"extra": "iterations: 152\ncpu: 0.001837044519736841 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00614654752943251,
"extra": "iterations: 45\ncpu: 0.006146644777777777 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003986103194100516,
"extra": "iterations: 70\ncpu: 0.003985959699999997 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002653744107200986,
"extra": "iterations: 105\ncpu: 0.0026537781809523836 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018643299738566083,
"extra": "iterations: 150\ncpu: 0.001864227646666666 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0052951506848605175,
"extra": "iterations: 53\ncpu: 0.005295218339622645 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.0081453732081822,
"extra": "iterations: 35\ncpu: 0.00814548457142857 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01060635309952956,
"extra": "iterations: 26\ncpu: 0.010606530846153853 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026168958196100198,
"extra": "iterations: 106\ncpu: 0.0026169374433962322 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018330087848738128,
"extra": "iterations: 153\ncpu: 0.0018328896993464042 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.00365894405465377,
"extra": "iterations: 76\ncpu: 0.0036589995131578873 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028236842155456546,
"extra": "iterations: 100\ncpu: 0.002823361840000001 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004323031352116511,
"extra": "iterations: 65\ncpu: 0.0043229110923077 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003492234665670513,
"extra": "iterations: 81\ncpu: 0.003492198691358029 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005540095123590208,
"extra": "iterations: 51\ncpu: 0.005539900960784322 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004663296540578206,
"extra": "iterations: 60\ncpu: 0.004663384466666661 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011327767913991756,
"extra": "iterations: 528\ncpu: 0.0005331367272727284 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015272784466836967,
"extra": "iterations: 408\ncpu: 0.000695599058823531 seconds\nthreads: 1"
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
"date": 1783553392919,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002191385254263878,
"extra": "iterations: 1280\ncpu: 0.0002191115421875 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002004384994506836,
"extra": "iterations: 1400\ncpu: 0.00020044132071428576 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0026276356705995365,
"extra": "iterations: 107\ncpu: 0.0026276738878504673 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018487807951475446,
"extra": "iterations: 152\ncpu: 0.0018487015921052632 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006137272585993228,
"extra": "iterations: 46\ncpu: 0.006137371695652171 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0039684057235717775,
"extra": "iterations: 70\ncpu: 0.003968299885714286 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026246565524662767,
"extra": "iterations: 107\ncpu: 0.0026246048504672873 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018667078018188479,
"extra": "iterations: 150\ncpu: 0.0018667338333333327 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005321552168648199,
"extra": "iterations: 53\ncpu: 0.005321429377358492 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008012158530099052,
"extra": "iterations: 35\ncpu: 0.008012270828571439 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010609782659090482,
"extra": "iterations: 26\ncpu: 0.010609351961538466 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0026252626258636193,
"extra": "iterations: 107\ncpu: 0.002625122252336454 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018429787535416454,
"extra": "iterations: 152\ncpu: 0.0018427366644736835 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038602537923044973,
"extra": "iterations: 77\ncpu: 0.0038603172077922156 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002845513577363929,
"extra": "iterations: 98\ncpu: 0.00284555206122449 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004329275339841843,
"extra": "iterations: 64\ncpu: 0.004328568500000005 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00347478985786438,
"extra": "iterations: 80\ncpu: 0.0034746985249999975 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0054801258386350145,
"extra": "iterations: 51\ncpu: 0.005479405333333319 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004657482696791827,
"extra": "iterations: 59\ncpu: 0.004657389491525422 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010247825457946893,
"extra": "iterations: 533\ncpu: 0.0004759881744840532 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001462177796797319,
"extra": "iterations: 374\ncpu: 0.0007207282326203196 seconds\nthreads: 1"
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
"date": 1783553392935,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002194588664192824,
"extra": "iterations: 1272\ncpu: 0.0002194561352201258 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020122955656120048,
"extra": "iterations: 1394\ncpu: 0.00020122526972740317 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002637710211411962,
"extra": "iterations: 106\ncpu: 0.002637744981132076 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018574810028076172,
"extra": "iterations: 150\ncpu: 0.0018573170333333348 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006211174858940972,
"extra": "iterations: 45\ncpu: 0.006211278200000002 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003977469035557338,
"extra": "iterations: 70\ncpu: 0.003977524885714289 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026376382359918557,
"extra": "iterations: 106\ncpu: 0.002637568962264154 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018415184397446484,
"extra": "iterations: 152\ncpu: 0.0018414437565789465 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0053672377879802995,
"extra": "iterations: 52\ncpu: 0.00536676617307693 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008122963063857135,
"extra": "iterations: 34\ncpu: 0.008123098529411751 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010849549220158504,
"extra": "iterations: 26\ncpu: 0.01084829369230769 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002622460419276975,
"extra": "iterations: 106\ncpu: 0.002622368622641517 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001838713689567217,
"extra": "iterations: 153\ncpu: 0.0018387405490196105 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036645939475611635,
"extra": "iterations: 76\ncpu: 0.003663959513157905 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002826481154470733,
"extra": "iterations: 99\ncpu: 0.00282652152525253 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004317492705125075,
"extra": "iterations: 65\ncpu: 0.0043175563692307625 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003484040498733521,
"extra": "iterations: 80\ncpu: 0.003483939012500004 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005526318842050981,
"extra": "iterations: 49\ncpu: 0.005526402959183669 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004686379432678223,
"extra": "iterations: 60\ncpu: 0.0046864397333333295 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011220419848406759,
"extra": "iterations: 540\ncpu: 0.0005307113 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0014835625591844617,
"extra": "iterations: 404\ncpu: 0.0006830520272277231 seconds\nthreads: 1"
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
"date": 1783553392928,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002201184080233278,
"extra": "iterations: 1273\ncpu: 0.0002200873864886096 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020339611736737447,
"extra": "iterations: 1379\ncpu: 0.00020338632994923865 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002690989237565261,
"extra": "iterations: 104\ncpu: 0.002690903211538462 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018406463296789873,
"extra": "iterations: 152\ncpu: 0.0018405513750000006 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006225501166449653,
"extra": "iterations: 45\ncpu: 0.006225594822222223 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003945730101894325,
"extra": "iterations: 71\ncpu: 0.003945247704225353 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0026628521253477854,
"extra": "iterations: 106\ncpu: 0.00266267241509434 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018480439849247207,
"extra": "iterations: 151\ncpu: 0.001848070006622515 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005277228805254091,
"extra": "iterations: 53\ncpu: 0.005276090660377357 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007936743327549525,
"extra": "iterations: 35\ncpu: 0.00793686502857142 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.010570481971458153,
"extra": "iterations: 27\ncpu: 0.010570634777777782 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002619540579965182,
"extra": "iterations: 107\ncpu: 0.002619334495327096 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018429269916132877,
"extra": "iterations: 152\ncpu: 0.0018428325986842122 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0036719943347730136,
"extra": "iterations: 76\ncpu: 0.003671684815789482 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028199523386329113,
"extra": "iterations: 99\ncpu: 0.002819997080808084 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004338432103395462,
"extra": "iterations: 64\ncpu: 0.004338533625000002 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003492921590805054,
"extra": "iterations: 80\ncpu: 0.003492838862500003 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005528736114501953,
"extra": "iterations: 50\ncpu: 0.005528808380000002 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004673890272776287,
"extra": "iterations: 60\ncpu: 0.004673759866666657 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010584724283664027,
"extra": "iterations: 535\ncpu: 0.0005017228710280367 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015014570492964524,
"extra": "iterations: 416\ncpu: 0.0006977294591346166 seconds\nthreads: 1"
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
"date": 1783553392867,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00022232058696413268,
"extra": "iterations: 1258\ncpu: 0.00022231772496025438 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002019705695220067,
"extra": "iterations: 1421\ncpu: 0.00020196524700914854 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002613974508838119,
"extra": "iterations: 107\ncpu: 0.002613903607476635 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018318618824279386,
"extra": "iterations: 153\ncpu: 0.0018316228431372562 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006104521129442298,
"extra": "iterations: 46\ncpu: 0.006104604065217387 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00393344650805836,
"extra": "iterations: 71\ncpu: 0.003933245366197184 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002614965978658424,
"extra": "iterations: 106\ncpu: 0.0026145149622641504 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018519508211236253,
"extra": "iterations: 152\ncpu: 0.001851855881578949 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005245098361262569,
"extra": "iterations: 54\ncpu: 0.00524444625925926 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.007875800132751465,
"extra": "iterations: 36\ncpu: 0.007874848444444454 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.0104742756596318,
"extra": "iterations: 27\ncpu: 0.01047400444444444 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002621115925156068,
"extra": "iterations: 107\ncpu: 0.002620661981308414 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018469450489574711,
"extra": "iterations: 151\ncpu: 0.0018468857880794712 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003668014105264243,
"extra": "iterations: 77\ncpu: 0.0036675069480519473 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028301609887017142,
"extra": "iterations: 99\ncpu: 0.002829990222222221 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004280992654653697,
"extra": "iterations: 65\ncpu: 0.004280860123076913 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003482848405838013,
"extra": "iterations: 80\ncpu: 0.0034823845624999965 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005467531727809532,
"extra": "iterations: 51\ncpu: 0.005467456019607838 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004679465293884278,
"extra": "iterations: 60\ncpu: 0.004679348366666666 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00128179931640625,
"extra": "iterations: 500\ncpu: 0.0005576685739999992 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001954247224006768,
"extra": "iterations: 331\ncpu: 0.0008165022477341403 seconds\nthreads: 1"
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
"date": 1783553392881,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00025595520270589936,
"extra": "iterations: 1094\ncpu: 0.00025593089945155395 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021341085616097914,
"extra": "iterations: 1309\ncpu: 0.00021342113903743322 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0030098499790314707,
"extra": "iterations: 93\ncpu: 0.0030079142795698933 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018491255526511085,
"extra": "iterations: 151\ncpu: 0.0018491524105960274 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006300120883517796,
"extra": "iterations: 45\ncpu: 0.006299715644444447 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003995210783822196,
"extra": "iterations: 70\ncpu: 0.003994845200000004 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0030114855817569204,
"extra": "iterations: 93\ncpu: 0.003011397516129031 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018520621876967581,
"extra": "iterations: 152\ncpu: 0.001852087210526318 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.006116545718649159,
"extra": "iterations: 46\ncpu: 0.006116448826086955 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.009116272772512129,
"extra": "iterations: 31\ncpu: 0.00911638896774194 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.012057439140651539,
"extra": "iterations: 23\ncpu: 0.012057214391304328 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002985969502875146,
"extra": "iterations: 94\ncpu: 0.00298601840425532 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001828114191691081,
"extra": "iterations: 153\ncpu: 0.0018280385620915046 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003955898150591783,
"extra": "iterations: 71\ncpu: 0.003955779436619722 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028294938983339257,
"extra": "iterations: 99\ncpu: 0.002829439202020202 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0046038783964563595,
"extra": "iterations: 61\ncpu: 0.004603599000000006 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003511803059638301,
"extra": "iterations: 79\ncpu: 0.003511855063291128 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005765835444132487,
"extra": "iterations: 48\ncpu: 0.00576554443750001 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004711577525505653,
"extra": "iterations: 52\ncpu: 0.004711658673076914 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0012933861036769682,
"extra": "iterations: 488\ncpu: 0.0005652137868852453 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.002007247502930068,
"extra": "iterations: 321\ncpu: 0.000851267348909658 seconds\nthreads: 1"
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
"date": 1783553392886,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002608844900832457,
"extra": "iterations: 1088\ncpu: 0.00026085429503676475 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021360525173352862,
"extra": "iterations: 1313\ncpu: 0.0002135922734196497 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0030046355339788623,
"extra": "iterations: 93\ncpu: 0.0030045188172043006 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018509042029287302,
"extra": "iterations: 153\ncpu: 0.0018504912156862756 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006282705730862088,
"extra": "iterations: 45\ncpu: 0.0062826351555555614 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003961161204746791,
"extra": "iterations: 70\ncpu: 0.003961221357142855 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0030047790978544506,
"extra": "iterations: 93\ncpu: 0.0030042516344086024 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.00184909136671769,
"extra": "iterations: 152\ncpu: 0.0018491176644736833 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.006043397861978283,
"extra": "iterations: 46\ncpu: 0.006042405521739132 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.009170659383138022,
"extra": "iterations: 30\ncpu: 0.00917078806666667 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.012163732362830122,
"extra": "iterations: 23\ncpu: 0.012161777173913034 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.003001382274012412,
"extra": "iterations: 93\ncpu: 0.0030010809032258107 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001854677074002904,
"extra": "iterations: 151\ncpu: 0.0018546184834437095 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003975725173950196,
"extra": "iterations: 70\ncpu: 0.003974549042857146 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028386597681527184,
"extra": "iterations: 99\ncpu: 0.002838630747474746 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004604308331599001,
"extra": "iterations: 61\ncpu: 0.004604154081967209 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003514108778555182,
"extra": "iterations: 79\ncpu: 0.0035133902531645474 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0058188339074452715,
"extra": "iterations: 48\ncpu: 0.005818914625000005 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004691143830617269,
"extra": "iterations: 60\ncpu: 0.0046903590500000075 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010584796347269199,
"extra": "iterations: 533\ncpu: 0.0005013018911819893 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016678873697916667,
"extra": "iterations: 375\ncpu: 0.0007667125119999982 seconds\nthreads: 1"
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
"date": 1783553392853,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00025487685051416416,
"extra": "iterations: 1097\ncpu: 0.00025482651139471286 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021260011332116651,
"extra": "iterations: 1315\ncpu: 0.00021260288669201524 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002997354320857836,
"extra": "iterations: 92\ncpu: 0.002997402826086957 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018417203663200733,
"extra": "iterations: 151\ncpu: 0.0018417463046357628 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006282903931357644,
"extra": "iterations: 44\ncpu: 0.006282985363636366 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003960691179547992,
"extra": "iterations: 70\ncpu: 0.003960569528571426 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002991070138647201,
"extra": "iterations: 94\ncpu: 0.002991111617021275 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018456342973207174,
"extra": "iterations: 152\ncpu: 0.0018456559407894742 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.006047870801842731,
"extra": "iterations: 46\ncpu: 0.006047658543478263 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.009158965080015121,
"extra": "iterations: 31\ncpu: 0.00915862980645163 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.012231681657874067,
"extra": "iterations: 23\ncpu: 0.012230851652173916 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0030055635718889136,
"extra": "iterations: 93\ncpu: 0.0030056076236559077 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001842345062055086,
"extra": "iterations: 152\ncpu: 0.0018422837565789459 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0039704416839169785,
"extra": "iterations: 71\ncpu: 0.003969965436619718 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002840947131721341,
"extra": "iterations: 98\ncpu: 0.0028408509897959273 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00460504312984279,
"extra": "iterations: 61\ncpu: 0.004604800950819667 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003514736890792847,
"extra": "iterations: 80\ncpu: 0.0035145462500000063 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005748972600820113,
"extra": "iterations: 49\ncpu: 0.005749052510204076 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004695284366607666,
"extra": "iterations: 60\ncpu: 0.004695111116666666 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010627796215638796,
"extra": "iterations: 538\ncpu: 0.0005015337267658001 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001619150729778686,
"extra": "iterations: 366\ncpu: 0.0007500459699453553 seconds\nthreads: 1"
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
"date": 1783553392883,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002563830147523819,
"extra": "iterations: 1091\ncpu: 0.0002563865637030248 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021304462741060954,
"extra": "iterations: 1312\ncpu: 0.00021303688567073177 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.00300257436690792,
"extra": "iterations: 93\ncpu: 0.003002619860215053 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018327236175537112,
"extra": "iterations: 153\ncpu: 0.001832595718954248 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00670601072765532,
"extra": "iterations: 42\ncpu: 0.006705596214285713 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004451282440669953,
"extra": "iterations: 63\ncpu: 0.004451350428571427 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0029819747234912635,
"extra": "iterations: 94\ncpu: 0.00298170273404255 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018425740693744862,
"extra": "iterations: 152\ncpu: 0.0018425982368421069 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.006067856498386549,
"extra": "iterations: 46\ncpu: 0.006067684630434785 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.009314242998758953,
"extra": "iterations: 30\ncpu: 0.009314385200000016 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.0120914189711861,
"extra": "iterations: 23\ncpu: 0.012091590695652207 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0029897994183479474,
"extra": "iterations: 94\ncpu: 0.002989598085106389 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001821251476512236,
"extra": "iterations: 153\ncpu: 0.0018212763529411758 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003964017814313861,
"extra": "iterations: 71\ncpu: 0.0039634946901408495 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002939133299994715,
"extra": "iterations: 97\ncpu: 0.0029390703814433008 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004601244066582352,
"extra": "iterations: 61\ncpu: 0.004601257819672137 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003506329655647278,
"extra": "iterations: 80\ncpu: 0.003506191462499997 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005864729483922322,
"extra": "iterations: 48\ncpu: 0.005864831833333325 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00468405222488662,
"extra": "iterations: 59\ncpu: 0.004683922491525421 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001098452019075626,
"extra": "iterations: 542\ncpu: 0.0005151344114391144 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016628032202249047,
"extra": "iterations: 364\ncpu: 0.0007714023021978032 seconds\nthreads: 1"
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
"date": 1783553392931,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026480719121659474,
"extra": "iterations: 1053\ncpu: 0.0002647823076923077 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020856260365061186,
"extra": "iterations: 1342\ncpu: 0.0002085657935916543 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002860370947390187,
"extra": "iterations: 98\ncpu: 0.0028602763979591844 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018578554620805956,
"extra": "iterations: 151\ncpu: 0.0018578210662251644 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006348702040585605,
"extra": "iterations: 44\ncpu: 0.006348355931818181 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003968601495447293,
"extra": "iterations: 71\ncpu: 0.00396866167605634 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002868221730602031,
"extra": "iterations: 98\ncpu: 0.0028682635816326537 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018791556358337405,
"extra": "iterations: 152\ncpu: 0.0018723298552631587 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005740642547607422,
"extra": "iterations: 49\ncpu: 0.005740721755102045 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00879497081041336,
"extra": "iterations: 32\ncpu: 0.008794289937500007 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011595795551935831,
"extra": "iterations: 24\ncpu: 0.011595965749999992 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002854519960831623,
"extra": "iterations: 98\ncpu: 0.0028544432448979634 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001847968289726659,
"extra": "iterations: 152\ncpu: 0.0018477924671052614 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038611061043209503,
"extra": "iterations: 72\ncpu: 0.0038611635138888893 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002832612606010052,
"extra": "iterations: 99\ncpu: 0.002832521555555558 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004502581011864447,
"extra": "iterations: 62\ncpu: 0.004501984403225812 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034851431846618657,
"extra": "iterations: 80\ncpu: 0.00348519462500001 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0056523400910046635,
"extra": "iterations: 49\ncpu: 0.005652182367346928 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004676485061645508,
"extra": "iterations: 60\ncpu: 0.004676569733333333 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010278659239134381,
"extra": "iterations: 538\ncpu: 0.0004821253513011146 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016860081212187192,
"extra": "iterations: 379\ncpu: 0.0007767808469656993 seconds\nthreads: 1"
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
"date": 1783553392862,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026709812953428955,
"extra": "iterations: 1049\ncpu: 0.00026708336796949487 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020857590078511058,
"extra": "iterations: 1347\ncpu: 0.0002085787616926504 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0028560137262149733,
"extra": "iterations: 98\ncpu: 0.002856053285714286 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018561754795099726,
"extra": "iterations: 151\ncpu: 0.0018561975165562914 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006388637152585116,
"extra": "iterations: 44\ncpu: 0.006388412068181819 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003974403653826033,
"extra": "iterations: 70\ncpu: 0.003973640828571425 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002862575102825554,
"extra": "iterations: 98\ncpu: 0.0028624755816326532 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018463181821923508,
"extra": "iterations: 152\ncpu: 0.0018460112039473688 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005746890087516941,
"extra": "iterations: 49\ncpu: 0.005746383857142858 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008669272065162659,
"extra": "iterations: 32\ncpu: 0.008669078156249998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011577536662419638,
"extra": "iterations: 24\ncpu: 0.011576795208333324 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028928678060315322,
"extra": "iterations: 97\ncpu: 0.002892907742268045 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001856753368251371,
"extra": "iterations: 151\ncpu: 0.0018567889271523192 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003940703163684254,
"extra": "iterations: 71\ncpu: 0.003940616746478881 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002814605236053467,
"extra": "iterations: 100\ncpu: 0.0028145564400000023 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004527103516363329,
"extra": "iterations: 62\ncpu: 0.004526938822580645 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003476524353027344,
"extra": "iterations: 80\ncpu: 0.0034765809374999956 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005660300352135483,
"extra": "iterations: 49\ncpu: 0.005660383775510213 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00466223955154419,
"extra": "iterations: 60\ncpu: 0.004662034449999998 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010724312003846726,
"extra": "iterations: 566\ncpu: 0.0005102326554770325 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016870146792656896,
"extra": "iterations: 393\ncpu: 0.0007769216768447837 seconds\nthreads: 1"
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
"date": 1783553392950,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026411996659176913,
"extra": "iterations: 1057\ncpu: 0.00026412399243140974 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020791248038963038,
"extra": "iterations: 1350\ncpu: 0.00020788869481481483 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002855254679310079,
"extra": "iterations: 98\ncpu: 0.002855300326530613 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018485502192848607,
"extra": "iterations: 152\ncpu: 0.0018483479671052632 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006386859850449996,
"extra": "iterations: 44\ncpu: 0.006386743681818186 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003949492318289621,
"extra": "iterations: 70\ncpu: 0.003949548442857141 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002875218586045869,
"extra": "iterations: 98\ncpu: 0.002874847295918368 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0021430182456970218,
"extra": "iterations: 100\ncpu: 0.002142776209999999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.006413138642603037,
"extra": "iterations: 49\ncpu: 0.006413230857142851 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008720248937606812,
"extra": "iterations: 32\ncpu: 0.00871888471875 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011663585901260376,
"extra": "iterations: 24\ncpu: 0.01166333004166666 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028463237139643455,
"extra": "iterations: 98\ncpu: 0.0028461018673469427 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018655538558959963,
"extra": "iterations: 150\ncpu: 0.0018654370399999998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038595656826071545,
"extra": "iterations: 73\ncpu: 0.003859422246575335 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002822196844852332,
"extra": "iterations: 99\ncpu: 0.002822177292929288 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004526057550984045,
"extra": "iterations: 62\ncpu: 0.004525738177419361 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035087823867797855,
"extra": "iterations: 80\ncpu: 0.0035086945500000023 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.006016005860998276,
"extra": "iterations: 47\ncpu: 0.006015874531914894 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0047707800137794625,
"extra": "iterations: 59\ncpu: 0.004770861389830506 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00104071022562193,
"extra": "iterations: 563\ncpu: 0.00047782740142095876 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015787317092160146,
"extra": "iterations: 332\ncpu: 0.0007193461234939756 seconds\nthreads: 1"
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
"date": 1783553392906,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002641572042722765,
"extra": "iterations: 1059\ncpu: 0.00026413761756373936 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002085632152771682,
"extra": "iterations: 1335\ncpu: 0.00020855847191011238 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0028940648147740317,
"extra": "iterations: 97\ncpu: 0.0028939346391752578 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018469173657266718,
"extra": "iterations: 152\ncpu: 0.0018467046842105264 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006387439641085539,
"extra": "iterations: 44\ncpu: 0.006387529613636367 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003946156568930183,
"extra": "iterations: 71\ncpu: 0.003946062887323948 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028538922874294984,
"extra": "iterations: 98\ncpu: 0.002853828857142861 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018406243700730175,
"extra": "iterations: 152\ncpu: 0.0018406517894736856 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00576189657052358,
"extra": "iterations: 48\ncpu: 0.005761213458333332 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008679516613483429,
"extra": "iterations: 32\ncpu: 0.008679646375000002 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01164013147354126,
"extra": "iterations: 24\ncpu: 0.011638636083333322 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028604852909944497,
"extra": "iterations: 98\ncpu: 0.0028601638775510156 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018657159805297852,
"extra": "iterations: 150\ncpu: 0.0018656514266666652 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003850100791617616,
"extra": "iterations: 73\ncpu: 0.0038491279178082187 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002823164968779593,
"extra": "iterations: 99\ncpu: 0.002822945303030299 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004580801533114525,
"extra": "iterations: 62\ncpu: 0.004580549645161278 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034683898643211087,
"extra": "iterations: 81\ncpu: 0.00346811920987654 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00578756721652284,
"extra": "iterations: 49\ncpu: 0.005787250938775501 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004731061094898289,
"extra": "iterations: 59\ncpu: 0.0047303521016949035 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001093681007341987,
"extra": "iterations: 529\ncpu: 0.0005006540737240074 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001713258976286108,
"extra": "iterations: 352\ncpu: 0.0007810055056818184 seconds\nthreads: 1"
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
"date": 1783553392879,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026493090571779197,
"extra": "iterations: 1056\ncpu: 0.00026490427272727273 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020793322932818116,
"extra": "iterations: 1339\ncpu: 0.00020793617027632563 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002870800543804558,
"extra": "iterations: 98\ncpu: 0.0028705425816326533 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018430352210998537,
"extra": "iterations: 152\ncpu: 0.001843068546052632 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006356521086259322,
"extra": "iterations: 44\ncpu: 0.0063557287954545405 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003955085512617944,
"extra": "iterations: 71\ncpu: 0.003955149760563377 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028597177918424314,
"extra": "iterations: 97\ncpu: 0.002859761350515462 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018423074170162805,
"extra": "iterations: 152\ncpu: 0.001842135578947369 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005787089467048645,
"extra": "iterations: 48\ncpu: 0.0057866458125 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00877043604850769,
"extra": "iterations: 32\ncpu: 0.008769909000000006 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011968778527301292,
"extra": "iterations: 23\ncpu: 0.011968969304347838 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002859364057842054,
"extra": "iterations: 95\ncpu: 0.002859419157894734 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018575412548140974,
"extra": "iterations: 151\ncpu: 0.0018573305298013234 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003872032034887027,
"extra": "iterations: 73\ncpu: 0.0038721072602739687 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002899904443760111,
"extra": "iterations: 99\ncpu: 0.002899791161616167 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00455233653386434,
"extra": "iterations: 60\ncpu: 0.004552136766666667 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003503614664077759,
"extra": "iterations: 80\ncpu: 0.003503675962499997 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005730069413477061,
"extra": "iterations: 49\ncpu: 0.005729539224489799 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004801244571291168,
"extra": "iterations: 58\ncpu: 0.004800795189655173 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010137943728857803,
"extra": "iterations: 513\ncpu: 0.00047222528460038937 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0017084105765255051,
"extra": "iterations: 359\ncpu: 0.000790038206128134 seconds\nthreads: 1"
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
"date": 1783553392960,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026691074115563174,
"extra": "iterations: 1044\ncpu: 0.00026691461398467434 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002093887543089816,
"extra": "iterations: 1337\ncpu: 0.0002093851555721765 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0028579818959138833,
"extra": "iterations: 98\ncpu: 0.0028580216224489796 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018591991323508963,
"extra": "iterations: 151\ncpu: 0.0018592245364238405 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.00639265233820135,
"extra": "iterations: 44\ncpu: 0.006392731227272726 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003958168164105482,
"extra": "iterations: 71\ncpu: 0.003958222492957745 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002848273575908006,
"extra": "iterations: 99\ncpu: 0.002848321808080807 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018493213400935496,
"extra": "iterations: 151\ncpu: 0.0018493486622516572 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.0056830425651706,
"extra": "iterations: 49\ncpu: 0.0056831353469387735 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008599355816841125,
"extra": "iterations: 32\ncpu: 0.0085992061875 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011338071823120117,
"extra": "iterations: 25\ncpu: 0.011337713120000006 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028246797696508544,
"extra": "iterations: 99\ncpu: 0.0028246204343434305 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001833004109999713,
"extra": "iterations: 153\ncpu: 0.0018330312745098078 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038497578607846615,
"extra": "iterations: 73\ncpu: 0.0038498102876712258 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0029587546984354654,
"extra": "iterations: 96\ncpu: 0.002958646572916668 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004502957867037865,
"extra": "iterations: 62\ncpu: 0.004503028596774185 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003491824865341187,
"extra": "iterations: 80\ncpu: 0.003491555950000003 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005668922346465442,
"extra": "iterations: 49\ncpu: 0.005669023959183663 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00469555289058362,
"extra": "iterations: 59\ncpu: 0.004695621745762715 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011320603312494134,
"extra": "iterations: 541\ncpu: 0.0005342600665434379 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0018424602470012627,
"extra": "iterations: 396\ncpu: 0.0008108276010100997 seconds\nthreads: 1"
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
"date": 1783553392840,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026810479003495046,
"extra": "iterations: 1039\ncpu: 0.00026803960635226183 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021103273396029163,
"extra": "iterations: 1329\ncpu: 0.00021102164559819414 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002856347025657187,
"extra": "iterations: 98\ncpu: 0.002856394612244898 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018494066439176862,
"extra": "iterations: 152\ncpu: 0.0018490237039473693 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006403895941647616,
"extra": "iterations: 44\ncpu: 0.006404003204545454 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003967822437555018,
"extra": "iterations: 71\ncpu: 0.0039678775633802815 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028599414628805576,
"extra": "iterations: 97\ncpu: 0.0028598367216494877 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001854647074314143,
"extra": "iterations: 151\ncpu: 0.001854675986754967 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00579477846622467,
"extra": "iterations: 48\ncpu: 0.005794286958333329 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008692048490047455,
"extra": "iterations: 32\ncpu: 0.00869218628125 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011519312858581543,
"extra": "iterations: 24\ncpu: 0.011518637458333335 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002831702280526209,
"extra": "iterations: 99\ncpu: 0.0028316601818181835 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018208478790482665,
"extra": "iterations: 153\ncpu: 0.00182080205228758 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003854225759636866,
"extra": "iterations: 73\ncpu: 0.0038540998082191805 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002819962501525879,
"extra": "iterations: 100\ncpu: 0.002820003839999998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004490034920828684,
"extra": "iterations: 63\ncpu: 0.004489907587301581 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003995855649312338,
"extra": "iterations: 72\ncpu: 0.003995934833333332 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.006480109123956589,
"extra": "iterations: 42\ncpu: 0.0064802090238095305 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004674665133158367,
"extra": "iterations: 60\ncpu: 0.004673781416666664 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010792198438405532,
"extra": "iterations: 519\ncpu: 0.0005053809826589603 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016150947498238606,
"extra": "iterations: 368\ncpu: 0.0007579322581521736 seconds\nthreads: 1"
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
"date": 1783553392848,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002639356768198861,
"extra": "iterations: 1058\ncpu: 0.0002639195765595463 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020896679933499798,
"extra": "iterations: 1346\ncpu: 0.00020895681872213967 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002869679755771283,
"extra": "iterations: 97\ncpu: 0.0028689804329896903 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018383421395954335,
"extra": "iterations: 152\ncpu: 0.0018383387500000002 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006434085757233376,
"extra": "iterations: 43\ncpu: 0.006434191883720929 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004079483557438505,
"extra": "iterations: 69\ncpu: 0.004079065999999998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002928758404918553,
"extra": "iterations: 97\ncpu: 0.0029287875670103093 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018689568837483724,
"extra": "iterations: 150\ncpu: 0.0018684301666666682 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005850185950597127,
"extra": "iterations: 48\ncpu: 0.005850265395833341 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008758842945098877,
"extra": "iterations: 32\ncpu: 0.008758969593749999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011746476093928019,
"extra": "iterations: 24\ncpu: 0.011744770708333335 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028395989928582705,
"extra": "iterations: 99\ncpu: 0.0028396492323232315 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018368998384164052,
"extra": "iterations: 153\ncpu: 0.00183673756862745 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038692255814870204,
"extra": "iterations: 72\ncpu: 0.0038691706944444424 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028390089670817056,
"extra": "iterations: 99\ncpu: 0.0028388937676767727 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004529014710457095,
"extra": "iterations: 62\ncpu: 0.004528352225806458 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034961909055709842,
"extra": "iterations: 80\ncpu: 0.0034962450500000022 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005685183466697226,
"extra": "iterations: 49\ncpu: 0.005683906693877547 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004690812805951653,
"extra": "iterations: 59\ncpu: 0.004690080457627115 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010176802770953187,
"extra": "iterations: 569\ncpu: 0.0004895617644991223 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015812925697954012,
"extra": "iterations: 377\ncpu: 0.0007342940344827572 seconds\nthreads: 1"
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
"date": 1783553392953,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026435146295594535,
"extra": "iterations: 1054\ncpu: 0.00026433023339658446 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002087267304596241,
"extra": "iterations: 1351\ncpu: 0.0002087079348630644 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0028738926366432428,
"extra": "iterations: 97\ncpu: 0.0028739372680412372 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018499387021096339,
"extra": "iterations: 151\ncpu: 0.0018497174834437094 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006371330131183971,
"extra": "iterations: 44\ncpu: 0.0063711596136363634 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003961185046604702,
"extra": "iterations: 70\ncpu: 0.003961249685714284 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002863871807954749,
"extra": "iterations: 98\ncpu: 0.002863523785714286 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018481596520072538,
"extra": "iterations: 152\ncpu: 0.0018481860855263175 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005757229668753488,
"extra": "iterations: 49\ncpu: 0.005756723102040812 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008779346942901611,
"extra": "iterations: 32\ncpu: 0.008778692937499999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011768728494644165,
"extra": "iterations: 24\ncpu: 0.01176891574999998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028748512268066406,
"extra": "iterations: 98\ncpu: 0.0028744928469387815 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018527413046123178,
"extra": "iterations: 151\ncpu: 0.0018527686490066222 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038624604543050136,
"extra": "iterations: 72\ncpu: 0.0038617144722222284 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002825655118383543,
"extra": "iterations: 99\ncpu: 0.0028255593030302975 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004517089936041063,
"extra": "iterations: 62\ncpu: 0.004517155403225811 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.00349729061126709,
"extra": "iterations: 80\ncpu: 0.0034964077875000005 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005771510455073143,
"extra": "iterations: 49\ncpu: 0.0057712900816326495 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004714828426555052,
"extra": "iterations: 59\ncpu: 0.004714911762711859 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0009845341263661966,
"extra": "iterations: 542\ncpu: 0.0004635089557195567 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015790112274514754,
"extra": "iterations: 354\ncpu: 0.0007550708926553692 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.109671978258386e-05,
"extra": "iterations: 8984\ncpu: 3.108127359750668e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.471547708263844e-05,
"extra": "iterations: 6242\ncpu: 4.4709252643383524e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0006846596555012028,
"extra": "iterations: 410\ncpu: 0.0006846698853658537 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015133625752217063,
"extra": "iterations: 185\ncpu: 0.0015120672648648661 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.037632915389586e-05,
"extra": "iterations: 3984\ncpu: 7.033984061244957e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0005398465309659563,
"extra": "iterations: 517\ncpu: 0.0005400261721470015 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0032225038813448502,
"extra": "iterations: 87\ncpu: 0.003222311609195408 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.603261869863439e-05,
"extra": "iterations: 4243\ncpu: 6.605003275983845e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005215204405829006,
"extra": "iterations: 537\ncpu: 0.0005216359515828627 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.003157117393579376,
"extra": "iterations: 89\ncpu: 0.0031565806067415706 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00027826357455480673,
"extra": "iterations: 1008\ncpu: 0.0002779921339285713 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0012080116844995852,
"extra": "iterations: 233\ncpu: 0.0012080295579399152 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.004611257712046306,
"extra": "iterations: 60\ncpu: 0.0046092690333333335 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.011357178290685018,
"extra": "iterations: 24\ncpu: 0.011353199916666668 seconds\nthreads: 1"
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
"date": 1783553392938,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.000265050002441987,
"extra": "iterations: 1051\ncpu: 0.000265034039961941 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002081325114393305,
"extra": "iterations: 1346\ncpu: 0.00020811741381872214 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0028633852394259708,
"extra": "iterations: 98\ncpu: 0.0028626044591836745 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018490387114468002,
"extra": "iterations: 151\ncpu: 0.0018489618675496676 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006373118270527233,
"extra": "iterations: 44\ncpu: 0.006372426045454546 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0039775916508265905,
"extra": "iterations: 70\ncpu: 0.003977448914285718 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028735116585013793,
"extra": "iterations: 97\ncpu: 0.0028735558762886626 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018739891052246096,
"extra": "iterations: 150\ncpu: 0.0018738271866666672 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005792021751403809,
"extra": "iterations: 48\ncpu: 0.0057921223125 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00864923745393753,
"extra": "iterations: 32\ncpu: 0.008648337156250007 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.01151124636332194,
"extra": "iterations: 24\ncpu: 0.011510189916666672 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002839230527781477,
"extra": "iterations: 99\ncpu: 0.0028390362020202036 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018344143636865554,
"extra": "iterations: 153\ncpu: 0.0018343332091503284 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038534811098281654,
"extra": "iterations: 73\ncpu: 0.003853359383561644 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002858125246488131,
"extra": "iterations: 78\ncpu: 0.002857790910256417 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004495701482219081,
"extra": "iterations: 62\ncpu: 0.004495592290322582 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003490734100341797,
"extra": "iterations: 80\ncpu: 0.003490785050000001 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005683198267099809,
"extra": "iterations: 49\ncpu: 0.005683141673469384 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004674454530080159,
"extra": "iterations: 60\ncpu: 0.00467452603333333 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011467383099162184,
"extra": "iterations: 511\ncpu: 0.000546397377690803 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0015795886355911626,
"extra": "iterations: 371\ncpu: 0.0007336685471698094 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.1089909617613735e-05,
"extra": "iterations: 9018\ncpu: 3.108926591261921e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.447784351132608e-05,
"extra": "iterations: 6297\ncpu: 4.447607876766716e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0006876726313733239,
"extra": "iterations: 409\ncpu: 0.0006875397408312959 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015104239986788843,
"extra": "iterations: 186\ncpu: 0.0015102925806451627 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.026644633181026e-05,
"extra": "iterations: 3955\ncpu: 7.028107838179509e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0005396864088054796,
"extra": "iterations: 519\ncpu: 0.0005398246570327566 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0032421967078899517,
"extra": "iterations: 87\ncpu: 0.00324318741379309 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.60257033726226e-05,
"extra": "iterations: 4242\ncpu: 6.604085407826671e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005221588873552433,
"extra": "iterations: 537\ncpu: 0.0005222742271880816 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0031611377542669124,
"extra": "iterations: 88\ncpu: 0.0031610976363636305 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00027954315613649175,
"extra": "iterations: 998\ncpu: 0.00027952196292585176 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011985373599334849,
"extra": "iterations: 233\ncpu: 0.0011985588712446356 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.0048957205655282,
"extra": "iterations: 57\ncpu: 0.004894779982456141 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.011403550704320272,
"extra": "iterations: 24\ncpu: 0.011403697416666666 seconds\nthreads: 1"
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
"date": 1783553392955,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002646220879653902,
"extra": "iterations: 1058\ncpu: 0.0002646054810964083 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020939705523658782,
"extra": "iterations: 1335\ncpu: 0.00020939215355805245 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002872766907682124,
"extra": "iterations: 97\ncpu: 0.002872812711340206 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018370930665458731,
"extra": "iterations: 153\ncpu: 0.001837125013071896 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006371909921819514,
"extra": "iterations: 44\ncpu: 0.006371731590909091 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003954652377537319,
"extra": "iterations: 70\ncpu: 0.003954712828571427 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002879872764508749,
"extra": "iterations: 97\ncpu: 0.0028798301546391733 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018294917212592233,
"extra": "iterations: 153\ncpu: 0.001829516901960785 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005772161483764649,
"extra": "iterations: 45\ncpu: 0.005771838511111109 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008630894124507904,
"extra": "iterations: 32\ncpu: 0.008631018781250001 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011507381995519003,
"extra": "iterations: 24\ncpu: 0.011507145416666642 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028654774840997195,
"extra": "iterations: 98\ncpu: 0.0028652444999999966 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018475339902157817,
"extra": "iterations: 151\ncpu: 0.0018475597748344368 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003853486643897163,
"extra": "iterations: 72\ncpu: 0.003853185708333336 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002824715893677991,
"extra": "iterations: 99\ncpu: 0.0028247535454545476 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004497231975678475,
"extra": "iterations: 62\ncpu: 0.004497088322580642 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034938961267471317,
"extra": "iterations: 80\ncpu: 0.0034938524875000048 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.0056749947217046,
"extra": "iterations: 49\ncpu: 0.005675081938775517 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004657602310180664,
"extra": "iterations: 60\ncpu: 0.004657392549999987 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001040177267105853,
"extra": "iterations: 549\ncpu: 0.00048795978870674016 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0017483623782594002,
"extra": "iterations: 398\ncpu: 0.000774214208542714 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.108700966452533e-05,
"extra": "iterations: 8976\ncpu: 3.108752852049911e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.48734990076266e-05,
"extra": "iterations: 6246\ncpu: 4.487415834133846e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.000682822689232745,
"extra": "iterations: 411\ncpu: 0.0006825709902676398 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015799042868748898,
"extra": "iterations: 177\ncpu: 0.0015799321355932205 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.027657187763771e-05,
"extra": "iterations: 3980\ncpu: 7.030514346733634e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0005392448322193044,
"extra": "iterations: 518\ncpu: 0.000539394032818534 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0032197579570200255,
"extra": "iterations: 87\ncpu: 0.003218387816091949 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.601782339923787e-05,
"extra": "iterations: 4240\ncpu: 6.603848820754491e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005210842659224325,
"extra": "iterations: 536\ncpu: 0.0005211737126865757 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0031567664628618223,
"extra": "iterations: 89\ncpu: 0.003154303426966271 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002822621471332039,
"extra": "iterations: 993\ncpu: 0.00028226761732124864 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011976718902587892,
"extra": "iterations: 235\ncpu: 0.0011976920340425535 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.004620790481567383,
"extra": "iterations: 60\ncpu: 0.0046117268499999975 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.011146207650502523,
"extra": "iterations: 24\ncpu: 0.01112392741666667 seconds\nthreads: 1"
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
"date": 1783553392926,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026460195129567927,
"extra": "iterations: 1056\ncpu: 0.00026458357859848486 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020885892960218695,
"extra": "iterations: 1345\ncpu: 0.00020885192416356877 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0028777481407247564,
"extra": "iterations: 93\ncpu: 0.002877796548387097 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018656174341837566,
"extra": "iterations: 150\ncpu: 0.001865646953333333 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0064650611443953085,
"extra": "iterations: 44\ncpu: 0.006465175272727276 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003981399536132813,
"extra": "iterations: 70\ncpu: 0.0039814623142857145 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028781350125971533,
"extra": "iterations: 97\ncpu: 0.0028780609278350527 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001852509201757166,
"extra": "iterations: 151\ncpu: 0.0018524608344370858 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005766779184341431,
"extra": "iterations: 48\ncpu: 0.005766873312500004 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008684918284416199,
"extra": "iterations: 32\ncpu: 0.008684388749999996 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011560936768849691,
"extra": "iterations: 24\ncpu: 0.011559691416666667 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028528388665646927,
"extra": "iterations: 98\ncpu: 0.002852709948979596 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018245678443413278,
"extra": "iterations: 154\ncpu: 0.0018245960974025979 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038590729236602788,
"extra": "iterations: 72\ncpu: 0.0038583977222222217 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002822998798254765,
"extra": "iterations: 99\ncpu: 0.002822863777777775 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004513790530543174,
"extra": "iterations: 62\ncpu: 0.0045138584032258025 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003479215502738953,
"extra": "iterations: 80\ncpu: 0.003478592812500003 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005701196436979333,
"extra": "iterations: 49\ncpu: 0.005701096530612254 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004686395327250163,
"extra": "iterations: 60\ncpu: 0.0046864712833333344 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010221448687999421,
"extra": "iterations: 526\ncpu: 0.00048276787642585625 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016136157300332727,
"extra": "iterations: 393\ncpu: 0.0007410320152671757 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.114050950908767e-05,
"extra": "iterations: 9010\ncpu: 3.11368166481687e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.4867382067847686e-05,
"extra": "iterations: 6258\ncpu: 4.48653071268776e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0006867945194244385,
"extra": "iterations: 408\ncpu: 0.000686780556372549 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015087565860232794,
"extra": "iterations: 185\ncpu: 0.0015086886702702698 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.034592839129958e-05,
"extra": "iterations: 3984\ncpu: 7.036707680722903e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0005394544215560649,
"extra": "iterations: 519\ncpu: 0.0005395648246628142 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.003217962966568169,
"extra": "iterations: 87\ncpu: 0.003218373275862073 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.645219718794652e-05,
"extra": "iterations: 4243\ncpu: 6.646597171812485e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005209711847695277,
"extra": "iterations: 538\ncpu: 0.000521067513011146 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.003152442782112722,
"extra": "iterations: 89\ncpu: 0.0031526884719101254 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002792441018737189,
"extra": "iterations: 1010\ncpu: 0.00027924917326732667 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0012012231044280224,
"extra": "iterations: 234\ncpu: 0.0012012461282051277 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.0048834691967880525,
"extra": "iterations: 57\ncpu: 0.004883546771929817 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.011327117681503296,
"extra": "iterations: 24\ncpu: 0.011325895541666667 seconds\nthreads: 1"
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
"date": 1783553392909,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002662910947790609,
"extra": "iterations: 1051\ncpu: 0.00026616127212178875 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020803782173502242,
"extra": "iterations: 1347\ncpu: 0.00020802182999257615 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0028703114421097277,
"extra": "iterations: 97\ncpu: 0.002870354103092784 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.00185442594141742,
"extra": "iterations: 153\ncpu: 0.0018543383006535956 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006393513896248558,
"extra": "iterations: 44\ncpu: 0.0063936309545454565 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003934420330423705,
"extra": "iterations: 71\ncpu: 0.003933923000000001 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028678227444084324,
"extra": "iterations: 98\ncpu: 0.0028675981020408175 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018471345206759625,
"extra": "iterations: 151\ncpu: 0.0018471602450331115 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005752041935920715,
"extra": "iterations: 48\ncpu: 0.005751628145833336 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00865127146244049,
"extra": "iterations: 32\ncpu: 0.008651380312499998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011638293663660685,
"extra": "iterations: 24\ncpu: 0.01163682275 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002876410678941377,
"extra": "iterations: 98\ncpu: 0.002876254642857142 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001861855189005534,
"extra": "iterations: 150\ncpu: 0.0018617031666666682 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038707190089755593,
"extra": "iterations: 72\ncpu: 0.0038705407916666736 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002836577746332908,
"extra": "iterations: 98\ncpu: 0.0028366161224489803 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0044780623528265185,
"extra": "iterations: 62\ncpu: 0.004477479306451612 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034874975681304936,
"extra": "iterations: 80\ncpu: 0.0034875453625 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00566593481569874,
"extra": "iterations: 49\ncpu: 0.005666011122448975 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004675086339314779,
"extra": "iterations: 60\ncpu: 0.004675048299999999 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010371243060179995,
"extra": "iterations: 547\ncpu: 0.0004914014990859245 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.001628065615180949,
"extra": "iterations: 377\ncpu: 0.0007596224933687008 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.123944105949065e-05,
"extra": "iterations: 8949\ncpu: 3.123989216672254e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.547129562445127e-05,
"extra": "iterations: 6131\ncpu: 4.546977002120371e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0006882139280730603,
"extra": "iterations: 408\ncpu: 0.0006882250343137256 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015201827754145083,
"extra": "iterations: 184\ncpu: 0.0015201102989130437 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.0442016455729e-05,
"extra": "iterations: 3976\ncpu: 7.045725352112668e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0005446972205624943,
"extra": "iterations: 513\ncpu: 0.0005447786432748544 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0032275983656959973,
"extra": "iterations: 87\ncpu: 0.0032282856781609233 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.60932738081005e-05,
"extra": "iterations: 4227\ncpu: 6.610696829902792e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005259854452950613,
"extra": "iterations: 532\ncpu: 0.0005260575075187914 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.003168071253915851,
"extra": "iterations: 89\ncpu: 0.003168532258426961 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002807530796962229,
"extra": "iterations: 988\ncpu: 0.0002807399109311741 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0012043611756686508,
"extra": "iterations: 232\ncpu: 0.0012043045560344839 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.004882532253600003,
"extra": "iterations: 57\ncpu: 0.004882606070175444 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.011447479327519735,
"extra": "iterations: 24\ncpu: 0.011447576416666671 seconds\nthreads: 1"
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
"date": 1783553392943,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002654959560006659,
"extra": "iterations: 1053\ncpu: 0.0002654990237416904 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.000209306966707979,
"extra": "iterations: 1321\ncpu: 0.00020929105147615443 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002868903051946581,
"extra": "iterations: 97\ncpu: 0.0028688207010309275 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.00184642400173162,
"extra": "iterations: 151\ncpu: 0.0018464525033112579 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006382113153284246,
"extra": "iterations: 44\ncpu: 0.006381347477272723 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004135186331612723,
"extra": "iterations: 70\ncpu: 0.004135083099999998 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002881753075983107,
"extra": "iterations: 97\ncpu: 0.0028817928659793804 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018500560208370813,
"extra": "iterations: 152\ncpu: 0.0018488736842105242 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005805803804981466,
"extra": "iterations: 49\ncpu: 0.005805889653061231 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008680619299411774,
"extra": "iterations: 32\ncpu: 0.0086805176875 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011598954598108927,
"extra": "iterations: 24\ncpu: 0.011598005874999984 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002907236417134603,
"extra": "iterations: 96\ncpu: 0.0029071719270833316 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018489376598636049,
"extra": "iterations: 151\ncpu: 0.0018489635827814567 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038879668875916366,
"extra": "iterations: 73\ncpu: 0.003888023534246577 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028225508603182707,
"extra": "iterations: 99\ncpu: 0.0028224574040404114 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004502161856620543,
"extra": "iterations: 62\ncpu: 0.004502218677419362 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0034880280494689942,
"extra": "iterations: 80\ncpu: 0.0034879252999999943 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005661779520462971,
"extra": "iterations: 49\ncpu: 0.005661631530612247 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004675667164689403,
"extra": "iterations: 59\ncpu: 0.004675724881355939 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.00121828198207574,
"extra": "iterations: 529\ncpu: 0.0005571565595463132 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016138977474636502,
"extra": "iterations: 360\ncpu: 0.000726475186111111 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.118132269103392e-05,
"extra": "iterations: 8970\ncpu: 3.118175195094761e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.510707212159818e-05,
"extra": "iterations: 6199\ncpu: 4.510612114857235e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0007084683146825066,
"extra": "iterations: 397\ncpu: 0.0007084801612090681 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015127787718901763,
"extra": "iterations: 185\ncpu: 0.0015128020270270268 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.036772980529894e-05,
"extra": "iterations: 3961\ncpu: 7.037969957081568e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.000548382794553753,
"extra": "iterations: 511\ncpu: 0.0005483660958904132 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0032465790593346885,
"extra": "iterations: 86\ncpu: 0.0032469097790697684 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.606702534657605e-05,
"extra": "iterations: 4240\ncpu: 6.609054127358596e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005272615609412383,
"extra": "iterations: 529\ncpu: 0.0005272284801512192 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0031518829002808993,
"extra": "iterations: 89\ncpu: 0.0031515853932584208 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00027962919470068216,
"extra": "iterations: 999\ncpu: 0.0002795990170170171 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0012028981477786333,
"extra": "iterations: 234\ncpu: 0.0012027971025641035 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.004943973139712685,
"extra": "iterations: 57\ncpu: 0.004944053017543858 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.01116532325744629,
"extra": "iterations: 25\ncpu: 0.011163628799999995 seconds\nthreads: 1"
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
"date": 1783553392894,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002641691377347719,
"extra": "iterations: 1058\ncpu: 0.0002638868241965973 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020863528041853823,
"extra": "iterations: 1341\ncpu: 0.0002086320111856824 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002868311745779855,
"extra": "iterations: 98\ncpu: 0.002868241816326532 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.001845736252634149,
"extra": "iterations: 152\ncpu: 0.0018456640460526325 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.0064169060100208635,
"extra": "iterations: 44\ncpu: 0.006416990931818185 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004013375959534576,
"extra": "iterations: 69\ncpu: 0.004013272985507246 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028765988104122206,
"extra": "iterations: 97\ncpu: 0.0028764603092783507 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018936315098324339,
"extra": "iterations: 148\ncpu: 0.001893449344594594 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005749011526302417,
"extra": "iterations: 49\ncpu: 0.005748844877551018 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008659079670906067,
"extra": "iterations: 32\ncpu: 0.008659190031250014 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011772135893503824,
"extra": "iterations: 24\ncpu: 0.011771819250000022 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002893703500020135,
"extra": "iterations: 97\ncpu: 0.002893652350515462 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018779575424706376,
"extra": "iterations: 149\ncpu: 0.0018779050671140914 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003912042564069721,
"extra": "iterations: 71\ncpu: 0.003911710267605635 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028358478935397403,
"extra": "iterations: 98\ncpu: 0.002835895979591842 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004506980219194967,
"extra": "iterations: 62\ncpu: 0.004506878354838704 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003512898087501526,
"extra": "iterations: 80\ncpu: 0.0035125556874999967 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00592109431391177,
"extra": "iterations: 46\ncpu: 0.0059206532173913 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004779253687177385,
"extra": "iterations: 56\ncpu: 0.004778596607142861 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0010700687285392515,
"extra": "iterations: 527\ncpu: 0.000517084305502847 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016671056599961115,
"extra": "iterations: 388\ncpu: 0.0007629255103092779 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.107953614122784e-05,
"extra": "iterations: 8921\ncpu: 3.1077801367559694e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.47312305713522e-05,
"extra": "iterations: 6264\ncpu: 4.473197477650064e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.000679719216615251,
"extra": "iterations: 412\ncpu: 0.0006797292063106798 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015121472848428263,
"extra": "iterations: 185\ncpu: 0.0015121704432432442 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.026552375628684e-05,
"extra": "iterations: 3982\ncpu: 7.028068131592171e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.000540328440859885,
"extra": "iterations: 517\ncpu: 0.0005400693056092827 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0032227916279058346,
"extra": "iterations: 87\ncpu: 0.0032229010689655223 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.617498972601317e-05,
"extra": "iterations: 4231\ncpu: 6.619231032852628e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005222067850805847,
"extra": "iterations: 534\ncpu: 0.0005222527528089908 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0031720118576221254,
"extra": "iterations: 89\ncpu: 0.003172240977528095 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002827536194555221,
"extra": "iterations: 992\ncpu: 0.0002827455332661289 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011951464873093825,
"extra": "iterations: 234\ncpu: 0.001195105542735042 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.004609037618168064,
"extra": "iterations: 61\ncpu: 0.004609093393442628 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.01116201400756836,
"extra": "iterations: 25\ncpu: 0.011158487719999997 seconds\nthreads: 1"
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
"date": 1783553392958,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026612363777051343,
"extra": "iterations: 1047\ncpu: 0.0002660858328557785 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00021007280342598506,
"extra": "iterations: 1337\ncpu: 0.00021006025804038895 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.002857412610735212,
"extra": "iterations: 98\ncpu: 0.0028573697346938784 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018478616287833768,
"extra": "iterations: 152\ncpu: 0.0018474275197368421 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006426118140996888,
"extra": "iterations: 43\ncpu: 0.006425928186046511 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.003951824886698119,
"extra": "iterations: 71\ncpu: 0.003952034154929576 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002860460962568011,
"extra": "iterations: 98\ncpu: 0.002859838520408165 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018473382027733407,
"extra": "iterations: 151\ncpu: 0.0018472815496688754 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005750247410365514,
"extra": "iterations: 49\ncpu: 0.005749095408163265 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008663833141326904,
"extra": "iterations: 32\ncpu: 0.008663967750000003 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011630316575368246,
"extra": "iterations: 24\ncpu: 0.011630479166666671 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.00286368447907117,
"extra": "iterations: 98\ncpu: 0.002863550142857144 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.001863071123758952,
"extra": "iterations: 150\ncpu: 0.0018630147733333282 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0038555363814036054,
"extra": "iterations: 72\ncpu: 0.003854689027777771 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0028429007289385556,
"extra": "iterations: 99\ncpu: 0.002842944020202023 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004546769734086662,
"extra": "iterations: 58\ncpu: 0.004546846482758615 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0035003989934921267,
"extra": "iterations: 80\ncpu: 0.0034996432250000045 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005746267279800104,
"extra": "iterations: 49\ncpu: 0.005746072775510205 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.004816661446781482,
"extra": "iterations: 59\ncpu: 0.004816732627118648 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0012564624778527783,
"extra": "iterations: 486\ncpu: 0.0005718391748971193 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0016695547625015343,
"extra": "iterations: 366\ncpu: 0.0007677635628415302 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.106128975337984e-05,
"extra": "iterations: 9027\ncpu: 3.106103434142019e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.4773949686523594e-05,
"extra": "iterations: 6256\ncpu: 4.477468094629156e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0006835355991270484,
"extra": "iterations: 410\ncpu: 0.0006834103073170729 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015079334218014954,
"extra": "iterations: 186\ncpu: 0.0015079508709677425 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.381118029566515e-05,
"extra": "iterations: 3964\ncpu: 7.381578456104957e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0005397386992760147,
"extra": "iterations: 518\ncpu: 0.0005398695328185338 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.003220708890892994,
"extra": "iterations: 87\ncpu: 0.003221126988505745 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.59638990735794e-05,
"extra": "iterations: 4237\ncpu: 6.598544323814119e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005214992743819508,
"extra": "iterations: 536\ncpu: 0.0005215692555970161 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0031530964240599215,
"extra": "iterations: 89\ncpu: 0.0031529325617977704 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00027972424101686767,
"extra": "iterations: 1002\ncpu: 0.00027971924650698637 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.001210087332232245,
"extra": "iterations: 232\ncpu: 0.00121011172413793 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.004753132661183675,
"extra": "iterations: 60\ncpu: 0.004752944433333333 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.011253557205200196,
"extra": "iterations: 25\ncpu: 0.011250735040000005 seconds\nthreads: 1"
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
"date": 1783553392914,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026310594410887286,
"extra": "iterations: 1058\ncpu: 0.00026310996030245754 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020817989568078333,
"extra": "iterations: 1343\ncpu: 0.00020818350111690247 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0028371664942527302,
"extra": "iterations: 98\ncpu: 0.002837003224489797 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018520828903905604,
"extra": "iterations: 151\ncpu: 0.0018521193377483442 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006529259127239849,
"extra": "iterations: 43\ncpu: 0.006520616860465121 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004150729248489159,
"extra": "iterations: 69\ncpu: 0.004150666782608697 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0028417736592918938,
"extra": "iterations: 99\ncpu: 0.0028418188080808074 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018732293446858724,
"extra": "iterations: 150\ncpu: 0.0018730074333333313 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005733431602010922,
"extra": "iterations: 49\ncpu: 0.005733538020408163 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008699856698513031,
"extra": "iterations: 32\ncpu: 0.008697501968750007 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011456221342086792,
"extra": "iterations: 24\ncpu: 0.011455657999999999 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028325861150568185,
"extra": "iterations: 99\ncpu: 0.0028326408282828308 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018498536787535016,
"extra": "iterations: 152\ncpu: 0.0018497653355263174 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0039088692930009635,
"extra": "iterations: 72\ncpu: 0.003908636861111104 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.0029247204462687173,
"extra": "iterations: 96\ncpu: 0.0029240280312499978 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0047172546386718755,
"extra": "iterations: 60\ncpu: 0.004717195133333323 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003766649478190654,
"extra": "iterations: 74\ncpu: 0.00376671152702703 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.006041599356609842,
"extra": "iterations: 46\ncpu: 0.006040064434782618 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.00511267050257269,
"extra": "iterations: 53\ncpu: 0.005112508811320755 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0013633926316063004,
"extra": "iterations: 505\ncpu: 0.00057307010891089 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0019797329618639337,
"extra": "iterations: 319\ncpu: 0.0008662640626959217 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.107514947685894e-05,
"extra": "iterations: 9010\ncpu: 3.107429966703663e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.3677344331150875e-05,
"extra": "iterations: 6396\ncpu: 4.367799499687304e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0006395556611013195,
"extra": "iterations: 438\ncpu: 0.000639534803652968 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.001531615283319859,
"extra": "iterations: 183\ncpu: 0.001531639683060109 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.034173309578763e-05,
"extra": "iterations: 3987\ncpu: 7.036207148231754e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0005388457421392834,
"extra": "iterations: 519\ncpu: 0.0005389819402697498 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0032461971044540406,
"extra": "iterations: 80\ncpu: 0.0032462811624999898 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.603199121214697e-05,
"extra": "iterations: 4239\ncpu: 6.604616843595264e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005212429515476333,
"extra": "iterations: 537\ncpu: 0.0005213080297951518 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.003150037165438192,
"extra": "iterations: 89\ncpu: 0.003149634123595532 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00027857277523818895,
"extra": "iterations: 1005\ncpu: 0.00027855377213930325 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011914963417864863,
"extra": "iterations: 235\ncpu: 0.001191300991489362 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.004611546876000577,
"extra": "iterations: 61\ncpu: 0.00461071498360656 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.011203517516454061,
"extra": "iterations: 24\ncpu: 0.011203620041666668 seconds\nthreads: 1"
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
"date": 1783553392904,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.0002657490684872582,
"extra": "iterations: 1050\ncpu: 0.00026573278285714285 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.00020732535021846627,
"extra": "iterations: 1356\ncpu: 0.00020732833407079648 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.00282082022452841,
"extra": "iterations: 98\ncpu: 0.0028205941326530613 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018558439040026128,
"extra": "iterations: 151\ncpu: 0.0018557515562913909 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006503332492917084,
"extra": "iterations: 43\ncpu: 0.006502931511627909 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004062884095786275,
"extra": "iterations: 69\ncpu: 0.0040620021159420245 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002823429878311928,
"extra": "iterations: 99\ncpu: 0.0028233056363636383 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.001855164963678019,
"extra": "iterations: 151\ncpu: 0.0018547093178807941 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.00571050449293487,
"extra": "iterations: 49\ncpu: 0.005709516795918369 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.00864251454671224,
"extra": "iterations: 33\ncpu: 0.008642123181818185 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011557837327321371,
"extra": "iterations: 24\ncpu: 0.01155669316666667 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0028516114360154277,
"extra": "iterations: 99\ncpu: 0.0028516547474747448 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018722852071126302,
"extra": "iterations: 150\ncpu: 0.0018722026399999989 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003947660956584232,
"extra": "iterations: 71\ncpu: 0.003947457746478864 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002935066327943907,
"extra": "iterations: 91\ncpu: 0.002935120637362642 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004729012648264567,
"extra": "iterations: 60\ncpu: 0.004728722083333326 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.003795704326114139,
"extra": "iterations: 74\ncpu: 0.0037955293648648664 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.006088619646818742,
"extra": "iterations: 46\ncpu: 0.006087983608695644 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005125615331861708,
"extra": "iterations: 54\ncpu: 0.005125239592592593 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011768409603675044,
"extra": "iterations: 487\ncpu: 0.0005224430410677616 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0017712229176571495,
"extra": "iterations: 380\ncpu: 0.0007874645500000021 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.1123949577545006e-05,
"extra": "iterations: 8971\ncpu: 3.11243803366403e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.4123171860882504e-05,
"extra": "iterations: 6364\ncpu: 4.412202388434947e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0006414432130101639,
"extra": "iterations: 434\ncpu: 0.0006414260368663594 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015310107684526288,
"extra": "iterations: 183\ncpu: 0.001531035448087432 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.026368518765176e-05,
"extra": "iterations: 3983\ncpu: 7.027511046949582e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0005392452924868316,
"extra": "iterations: 518\ncpu: 0.0005393130888030889 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0032346824119830957,
"extra": "iterations: 87\ncpu: 0.003234746080459772 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.611447922880143e-05,
"extra": "iterations: 4237\ncpu: 6.612353575642966e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005218716307059347,
"extra": "iterations: 537\ncpu: 0.000521901737430157 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.0031553895285959996,
"extra": "iterations: 89\ncpu: 0.0031554416629213282 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.00028552790077365173,
"extra": "iterations: 980\ncpu: 0.0002854954397959183 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0011924408851785864,
"extra": "iterations: 235\ncpu: 0.0011922756425531914 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.004635756133032627,
"extra": "iterations: 61\ncpu: 0.004635426327868851 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.0111484432220459,
"extra": "iterations: 25\ncpu: 0.011144888439999993 seconds\nthreads: 1"
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
"date": 1783553392948,
"tool": "customSmallerIsBetter",
"benches": [
{
"name": "DftFourierToReal/4x4",
"unit": "seconds",
"value": 0.00026519528844139796,
"extra": "iterations: 1056\ncpu: 0.00026518424431818186 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/7x1",
"unit": "seconds",
"value": 0.0002092013198338198,
"extra": "iterations: 1335\ncpu: 0.00020919843370786518 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0028326487300371884,
"extra": "iterations: 99\ncpu: 0.0028326915050505054 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/12x12",
"unit": "seconds",
"value": 0.0018599828084309897,
"extra": "iterations: 150\ncpu: 0.0018596933600000004 seconds\nthreads: 1"
},
{
"name": "DftFourierToReal/16x18",
"unit": "seconds",
"value": 0.006529309028802917,
"extra": "iterations: 43\ncpu: 0.006529031720930236 seconds\nthreads: 1"
},
{
"name": "FftFourierToReal/16x18",
"unit": "seconds",
"value": 0.004063917242962381,
"extra": "iterations: 69\ncpu: 0.0040637324927536205 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.002825462456905481,
"extra": "iterations: 99\ncpu: 0.0028252098787878814 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_nzeta-28",
"unit": "seconds",
"value": 0.0018628907683711725,
"extra": "iterations: 149\ncpu: 0.0018628046375838915 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-56",
"unit": "seconds",
"value": 0.005723028766865634,
"extra": "iterations: 49\ncpu: 0.005722003530612243 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-84",
"unit": "seconds",
"value": 0.008611231139211944,
"extra": "iterations: 33\ncpu: 0.008609785454545462 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_nzeta-112",
"unit": "seconds",
"value": 0.011603236198425293,
"extra": "iterations: 24\ncpu: 0.011602972833333336 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.002812142372131348,
"extra": "iterations: 100\ncpu: 0.0028110239100000013 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-30",
"unit": "seconds",
"value": 0.0018859197629378147,
"extra": "iterations: 149\ncpu: 0.001885957174496647 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.003912969374320877,
"extra": "iterations: 71\ncpu: 0.003912789478873243 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-60",
"unit": "seconds",
"value": 0.002922938999376799,
"extra": "iterations: 95\ncpu: 0.002922776189473687 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.004704574743906657,
"extra": "iterations: 60\ncpu: 0.00470433766666667 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-90",
"unit": "seconds",
"value": 0.0037583859761555994,
"extra": "iterations: 75\ncpu: 0.003757748986666663 seconds\nthreads: 1"
},
{
"name": "Dft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.006062849708225416,
"extra": "iterations: 46\ncpu: 0.006062402869565198 seconds\nthreads: 1"
},
{
"name": "Fft/12x12_ntheta-120",
"unit": "seconds",
"value": 0.005060438676313921,
"extra": "iterations: 55\ncpu: 0.005060518836363638 seconds\nthreads: 1"
},
{
"name": "BM_FourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0011407412380811228,
"extra": "iterations: 502\ncpu: 0.0005124248147410356 seconds\nthreads: 1"
},
{
"name": "BM_DftFourierToReal_Parallel_W7x_4t",
"unit": "seconds",
"value": 0.0017740378777186079,
"extra": "iterations: 336\ncpu: 0.000814537848214286 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/4x4",
"unit": "seconds",
"value": 3.107318257983895e-05,
"extra": "iterations: 9011\ncpu: 3.107359971146377e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/7x1",
"unit": "seconds",
"value": 4.374810903539921e-05,
"extra": "iterations: 6170\ncpu: 4.374719416531604e-05 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/12x12",
"unit": "seconds",
"value": 0.0006469328094411781,
"extra": "iterations: 432\ncpu: 0.0006469416574074073 seconds\nthreads: 1"
},
{
"name": "DeAliasConstraintForce/16x18",
"unit": "seconds",
"value": 0.0015258853850157366,
"extra": "iterations: 184\ncpu: 0.0015259083750000008 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/5x4",
"unit": "seconds",
"value": 7.039489621586936e-05,
"extra": "iterations: 3983\ncpu: 7.040927215666622e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/8x6",
"unit": "seconds",
"value": 0.0005391033796163705,
"extra": "iterations: 520\ncpu: 0.0005391284673076907 seconds\nthreads: 1"
},
{
"name": "LaplaceSolve/12x8",
"unit": "seconds",
"value": 0.0032165461573107493,
"extra": "iterations: 87\ncpu: 0.0032167976321839102 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/5x4",
"unit": "seconds",
"value": 6.608247081551268e-05,
"extra": "iterations: 4236\ncpu: 6.60929346081219e-05 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/8x6",
"unit": "seconds",
"value": 0.0005217192324806242,
"extra": "iterations: 534\ncpu: 0.0005217553333333251 seconds\nthreads: 1"
},
{
"name": "LaplaceDecompose/12x8",
"unit": "seconds",
"value": 0.003152919619270925,
"extra": "iterations: 89\ncpu: 0.0031530930674157054 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/5x4",
"unit": "seconds",
"value": 0.0002813341634391178,
"extra": "iterations: 993\ncpu: 0.0002813127139979862 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/8x6",
"unit": "seconds",
"value": 0.0012016449875074395,
"extra": "iterations: 233\ncpu: 0.0012014266051502155 seconds\nthreads: 1"
},
{
"name": "TransformGreensFunctionDerivative/12x8",
"unit": "seconds",
"value": 0.004676500956217448,
"extra": "iterations: 60\ncpu: 0.0046763333833333386 seconds\nthreads: 1"
},
{
"name": "ComputeOutputQuantities/cma",
"unit": "seconds",
"value": 0.011138598124186197,
"extra": "iterations: 24\ncpu: 0.011135343541666676 seconds\nthreads: 1"
}
]
}
]
}
}
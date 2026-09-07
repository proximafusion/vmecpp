# Vacuum-field interpolation

VMEC++ uses tensor cubic interpolation of the external magnetic field in R and
Z. Each evaluation uses four nodes in each direction, with a one-sided stencil
near the grid edges. Tables with fewer than four R or Z nodes use bilinear
interpolation. The toroidal plane selection is unchanged; interpolation does
not extend the field outside its supplied domain.

`VmecInput.mgrid_interpolation` selects the numerical method. Cubic is the solver
default. Use `vmecpp.MGridInterpolation.LINEAR` to reproduce calculations made
with historical bilinear interpolation. The selection is stored with the full
input/output data. Historical full outputs lacking this field are read as
linear; a newly loaded input file without the field uses the current default.

## Accuracy and cost

A frozen study compared the two methods at the same plasma resolution, profiles,
coil currents and final force tolerances. The symmetric cases used 1e-14;
asymmetric CTH used its original 1e-8 tolerance and the W7-X timing comparison
used its original 1e-12 tolerance. The results below were confirmed against
upstream `98aaaea`, following an initial study against `4e0eefb`.

For cases with source coils, all table sizes were derived from one finer table
computed from identical polygonal/circular filaments. An independent
straight-segment Biot-Savart calculation and refined circular-filament quadrature
agreed with table nodes to 3.1e-13 or better. Equilibrium comparisons used a
401-by-401 cubic-table reference and matched physical positions before comparing
B, removing differences due solely to flux-coordinate parametrisation.

| Case | Relative B difference, linear 101 | Relative B difference, cubic 51 | Cubic/linear runtime at the original table size |
|---|---:|---:|---:|
| CTH, finite pressure/current | 2.96e-5 | 4.35e-7 | 1.003 |
| Free-boundary Solovev | 1.41e-4 | 1.42e-6 | 1.061 |
| QUASR 954 | 7.62e-5 | 4.85e-6 | 1.037 |
| QUASR 65579 | 6.88e-5 | 4.21e-6 | 1.010 |
| Asymmetric CTH | See below | See below | 1.014 |
| W7-X, finite pressure, held out | See below | See below | 1.004 |

The timing ratios are medians of six adjacent, process-isolated pairs on an
Apple M4 Mac (10 CPU cores, 16 GB RAM) with four OpenMP threads. All 120 current-upstream
confirmation solves converged; the table includes the original-size comparisons.
The largest median slowdown was 6.1%. Bootstrap intervals from these small samples
are estimates, not hard bounds on future runtime. An earlier eight-pair study also
found similar solver cost, with wider uncertainty in some cases. The improvement
is in field-table accuracy; these measurements do not establish a solver speedup.

A separate regression matrix used all 12 bundled QUASR configurations with their
original resolution and stage budgets, in vacuum and with two prescribed
pressure/current profiles. Both methods converged on the same five of 36 problems;
neither converged on any of the finite-pressure variants at these budgets. All
failures were retained. The converged controls received independent bulk-field
checks; finite-pressure accuracy is instead supported by CTH and W7-X.

The supplied-table coarsening study, run against `4e0eefb`, used asymmetric CTH
and W7-X tables without independent finer coil tables.
At original table nodes withheld from a coarser 61-by-61 W7-X table, cubic
interpolation reduced field error by about 80 times compared with bilinear at
that same resolution. Complete equilibria also approached the finest supplied
cubic-table result more closely. That result is a refinement reference, not an
exact coil-field certificate.

The gain concerns field-table error. Plasma Fourier/radial truncation and the
polygonal approximation of a smooth coil remain separate error sources. Sampled
bulk force-balance error was mostly unchanged when plasma resolution dominated.
Both methods stalled when asymmetric CTH was pushed from its supplied 1e-8
force tolerance to 1e-14; that stricter failure is not counted as convergence.

The [recorded observations](../benchmarks/mgrid_interpolation_results.json) include
paired timings, binary/input hashes, physical checks and all QUASR outcomes.

## Reproduce a comparison

The benchmark retains source hashes, actual inputs, method selections, raw
outputs, convergence failures and paired solve timings. Run from the repository
root, with VMEC++ installed:

```bash
OMP_NUM_THREADS=4 python benchmarks/mgrid_interpolation.py \
  --input src/vmecpp/cpp/vmecpp/test_data/cth_like_free_bdy.json \
  --coils src/vmecpp/cpp/vmecpp/test_data/coils.cth_like \
  --parameters src/vmecpp/cpp/vmecpp/test_data/makegrid_parameters_cth_like.json \
  --finest 401 --grids 51 101 --repeats 8 --output mgrid-cth-study
```

Replace `cth_like` with `solovev` for the axisymmetric case. A supplied MGRID
file can be used with `--mgrid` instead of `--coils`/`--parameters`; choose R-grid
counts whose spacings divide the original grid, and an appropriate `--ftol`.
Z is reduced by the same stride, including for rectangular grids. In this mode
the reference uses the original supplied table. Table preparation and loading
are reported separately from the measured solver call. The script reports
Fourier-coefficient differences; independent physical-point reconstruction was
an additional validation in the study above.

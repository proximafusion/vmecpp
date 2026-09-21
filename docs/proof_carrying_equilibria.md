# Proof-carrying equilibria

Rocq, formerly Coq, is a proof assistant: a program that checks mathematical proofs down to every inference step. When Rocq accepts a theorem, what remains to be trusted is its small proof-checking kernel and the stated assumptions, not the author of the proof and not the code that produced the numbers.

[Stellarocq](https://github.com/CharlesCNorton/stellarocq) applies it to VMEC++ in three ways: it certifies the force balance of a wout, it proves and encloses physical properties of the reconstructed field, and it serves as an oracle for the solver.

## Certificates

A converged wout ships with a small file, and an independent checker, extracted from a Rocq proof, validates it. The theorem behind the checker says: if the verdict is VALID, then the ideal-MHD force residual of the field reconstructed from these exact coefficients, by VMEC's own half-grid rule, is a genuine real number at every certified point, with magnitude below the stated bound. A division by zero, an invalid square root, or an unlucky rounding anywhere in that evaluation makes the verdict INVALID, because the arithmetic is verified interval arithmetic (CoqInterval, proven against the Flocq formalization of IEEE-754).

The value is independence. "This wout satisfies force balance" normally means "VMEC++ says so, and VMEC++ agrees with Fortran VMEC." The checker shares no code with either and does not trust the generator that wrote the certificate: it recomputes everything from the coefficients with proven-sound arithmetic. A wout that passes cannot misstate its residual, whatever bug either solver might contain.

What is certified: the mu0-scaled residual `J x B - grad p` of the field defined by the wout Fourier coefficients under VMEC's half-grid conventions (parity-aware averages of R and Z onto the half grid, the wout's half-grid lambda and iota, centered differences of the covariant field across the node), at a stated set of full-grid nodes, bounded per component. Both symmetry classes are covered. A wout does not store `PRES_SCALE`, so the scale is recovered from the wout's own `pres` and put back into the pressure coefficients, and the certificate is about the pressure the equilibrium balances. What is not certified: anything about the solver's internals. The trust base is the Rocq kernel with the classical real axioms and the primitive float and integer specifications of its standard library, extraction and the OCaml compiler, and a parsing driver; `make audit` in Stellarocq prints the axioms behind every theorem.

The angles of a certificate can be certified two ways. A point certificate bounds the residual at the angles it lists. A cell certificate bounds it over cells of angles that abut, so its verdict speaks for a continuum of angles and not for a sample of them: the theorem behind it (`check_ccert_correct`) walks from the centre of a cell to any point of it by a mean-value step in each angle, with the derivative enclosed over the whole cell.

The bounds of a cell certificate are written by the checker, not by the generator. Interval arithmetic over a box loses the cancellation that makes an equilibrium residual small, and how much it loses is a property of the arithmetic that no float sample predicts, so `--tighten` reads back the enclosure the verified code computes for each cell and writes the smallest claim that code accepts. The result is an ordinary certificate, and an ordinary run establishes it.

## Physics

Stellarocq proves identities the reconstruction satisfies rather than assumes. `divergence_free` states that the divergence of the reconstructed field is exactly zero, and `pressure_is_a_flux_function` that dp/ds reads neither angle. The cells that bound the residual also enclose flux-surface integrals, so `dV/ds`, the enclosed currents and the terms of the Mercier criterion come out as intervals, and `mercier_geodesic_nonpositive` proves the sign of the geodesic term by Cauchy-Schwarz where no enclosure decides it. Departure from quasisymmetry is bounded from the field, without Boozer coordinates. A residual component proven bounded away from zero over a cell shows that no field of the certified form is in force balance there. The physical assumptions a statement needs are propositions in `theories/Hypotheses.v`, each with what would falsify it.

## Solver oracle

The checker shares no code with VMEC++ or Fortran VMEC, so a certificate that fails where one has to hold points at the solver. A stellarator-symmetric equilibrium certified through the non-stellarator-symmetric reconstruction, with its antisymmetric coefficients set to zero, has to hold at the same bounds, which is the reduction `tests/test_lasym.py` requires of the solver. Checking that reduction found [#788](https://github.com/proximafusion/vmecpp/issues/788), fixed in [#789](https://github.com/proximafusion/vmecpp/pull/789), and the same defect in educational_VMEC ([#27](https://github.com/jonathanschilling/educational_VMEC/issues/27) there).

## Usage

```sh
git clone https://github.com/CharlesCNorton/stellarocq
python examples/make_equilibrium_certificate.py --stellarocq stellarocq
python examples/make_equilibrium_certificate.py --stellarocq stellarocq --cells --nodes 6 --nu 8192
```

The example runs VMEC++ on `examples/data/solovev.json`, or on the input file it is given, and saves the wout. It then writes a certificate with `gen/make_cert.py` of the checkout, checks it, and compares it with the wout through `gen/verify_cert.py`. With `--cells` the checker first writes the cell bounds (`--tighten`). `--mpol`, `--ntor`, `--ns` and `--ftol` override the resolution and the force tolerance of the input before the run, and `--project` asks the checker for the harmonics of the residual at the modes of the run. The checker is `extract/_build/default/main.exe` of the checkout after `make all` there, or the statically linked x86_64 Linux build on the [Stellarocq releases](https://github.com/CharlesCNorton/stellarocq/releases), each named after the commit it was built from and passed with `--checker`. `STELLAROCQ_JOBS=n` sets the number of worker processes, and the generator needs `numpy` and `netCDF4`.

## Results

Six nodes per case, 20 worker processes. The bound is the largest of the three component bounds, over the reference `B^2` scale the generator prints for the point certificate.

| case | points | largest bound, of the field scale | verdict |
|---|---|---|---|
| `wout_solovev` (axisymmetric, ns=55, 6 modes, power series) | 48 | 7.3e-5 | VALID, under 0.1 s |
| `examples/data/w7x.json` as shipped (3D, nfp=5, 288 modes, ns=99) | 192 | 1.3e-2 | VALID, 3.7 s |
| `wout_cth_like_fixed_bdy` (3D, nfp=5, 41 modes, ns=25, two-power pressure) | 192 | 1.0e-2 | VALID, 0.2 s |
| `up_down_asym.json` (non-stellarator-symmetric, ns=17) | 48 | 1.2e-2 | VALID, under 0.1 s |
| `wout_cma` (3D, nfp=2, 59 modes, ns=51) | 192 | 5.2e-2 | VALID, 0.3 s |
| solovev with one `rmnc` coefficient of a certified stencil perturbed by 0.1% | 48 | same claim | INVALID, under 0.1 s |

Every stellarator-symmetric case above also certifies through the non-stellarator-symmetric reconstruction with its antisymmetric coefficients set to zero, at the same bounds (`gen/make_cert.py --force-lasym`).

For a three-dimensional equilibrium the bound follows the resolution and the force tolerance of the run. `input.li383_low_res`, which ships with 25 modes, 16 surfaces and a tolerance of 1e-6, run by the example with `--ns 31 --ftol 1e-14` and the Fourier resolution raised:

| `--mpol`, `--ntor` | modes | largest bound, of the field scale |
|---|---|---|
| 4, 3 | 25 | 3.0e-1 |
| 6, 4 | 50 | 4.1e-2 |
| 8, 6 | 98 | 1.0e-2 |
| 10, 8 | 162 | 6.8e-3 |
| 12, 10 | 242 | 5.6e-3 |

With the input's own tolerance the last row is 3.1e-2. Past about two hundred modes the bound stays near 6e-3 at this radial resolution.

A spectral solution balances forces mode by mode over the modes it retains, and what it leaves pointwise is the truncation. `--project` follows a point certificate with the largest mean harmonic of each component over the modes of the run and the angles of a node, a finite sum that `harm_encloses` of Stellarocq encloses. Over 48 by 48 angles per node, of the field scale:

| case | largest pointwise bound | `r_s` harmonic | `r_u` harmonic | `r_v` harmonic |
|---|---|---|---|---|
| `input.li383_low_res` at `--mpol 12 --ntor 10 --ns 31 --ftol 1e-14` | 6.7e-3 | 9.0e-4 | 9.7e-5 | 6.1e-5 |
| `examples/data/w7x.json` as shipped | 1.9e-2 | 2.1e-3 | 3.9e-5 | 7.1e-5 |

| case | cells | worst cell bound | of the field scale | verdict |
|---|---|---|---|---|
| `wout_solovev` (axisymmetric, ns=55) | 49152 | 1.3e-5 | 1.9e-4 | VALID, 41 s |
| `wout_circular_tokamak_reference` (axisymmetric, ns=17) | 49152 | 1.5e-2 | 2.5e-4 | VALID, 48 s |
| `wout_cma` (3D, nfp=2, 59 modes, ns=51) | 24576 | 1.3e-2 | 4.3e-2 | VALID, 90 s |
| `wout_li383_low_res_reference` (3D, nfp=3, 25 modes, ns=16) | 24576 | 8.7e-1 | 2.4e-1 | VALID, 39 s |
| `input.li383_low_res` at `--mpol 12 --ntor 10 --ns 31 --ftol 1e-14` (242 modes) | 24576 | 2.7e-2 | 7.4e-3 | VALID, 628 s |

Tightening the five took 30 s, 33 s, 50 s, 24 s and 254 s. The two axisymmetric cases carry 6 nodes of 8192 poloidal cells covering the whole angular torus; the three-dimensional ones carry 3 nodes of 4096 poloidal cells at each of 2 toroidal angles.

An axisymmetric equilibrium has every `n` zero, so its toroidal derivative encloses to zero and one cell spans the whole toroidal angle; the covering of the angular torus is one-dimensional. A three-dimensional one has to resolve the toroidal direction as finely as the poloidal, which squares the cell count, so its cells cover the poloidal angle at each of a few toroidal angles instead.

Certification is offline and per-wout; nothing runs in CI.

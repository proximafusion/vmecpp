# Proof-carrying equilibria

Rocq, formerly Coq, is a proof assistant: a program that checks mathematical proofs down to every inference step. When Rocq accepts a theorem, what remains to be trusted is its small proof-checking kernel and the stated assumptions, not the author of the proof and not the code that produced the numbers.

[Stellarocq](https://github.com/CharlesCNorton/stellarocq) uses that machinery to give a VMEC++ equilibrium a certificate. A converged wout ships with a small file, and an independent checker, extracted from a Rocq proof, validates it. The theorem behind the checker says: if the verdict is VALID, then the ideal-MHD force residual of the field reconstructed from these exact coefficients, by VMEC's own half-grid rule, is a genuine real number at every certified point, with magnitude below the stated bound. A division by zero, an invalid square root, or an unlucky rounding anywhere in that evaluation makes the verdict INVALID, because the arithmetic is verified interval arithmetic (CoqInterval, proven against the Flocq formalization of IEEE-754).

The value is independence. "This wout satisfies force balance" normally means "VMEC++ says so, and VMEC++ agrees with Fortran VMEC." The checker shares no code with either and does not trust the generator that wrote the certificate: it recomputes everything from the coefficients with proven-sound arithmetic. A wout that passes cannot misstate its residual, whatever bug either solver might contain.

What is certified: the mu0-scaled residual `J x B - grad p` of the field defined by the wout Fourier coefficients under VMEC's half-grid conventions (parity-aware averages of R and Z onto the half grid, the wout's half-grid lambda and iota, centered differences of the covariant field across the node), at a stated set of full-grid nodes, bounded per component. Both symmetry classes are covered. A wout does not store `PRES_SCALE`, so the scale is recovered from the wout's own `pres` and put back into the pressure coefficients, and the certificate is about the pressure the equilibrium balances. The example script reads a power-series or a two-power pressure, and `gen/make_cert.py` of Stellarocq reads the other parameterizations. What is not certified: anything about the solver's internals. The trust base is the Rocq kernel with the classical real axioms and the primitive float and integer specifications of its standard library, extraction and the OCaml compiler, and a parsing driver; `make audit` in Stellarocq prints the axioms behind every theorem.

Beside the certificates, Stellarocq proves identities the reconstruction satisfies rather than assumes. `divergence_free` states that the divergence of the reconstructed field is exactly zero, and `pressure_is_a_flux_function` that dp/ds reads neither angle.

## Points and cells

The angles of a certificate can be certified two ways. A point certificate bounds the residual at the angles it lists. A cell certificate bounds it over cells of angles that abut, so its verdict speaks for a continuum of angles and not for a sample of them: the theorem behind it (`check_ccert_correct`) walks from the centre of a cell to any point of it by a mean-value step in each angle, with the derivative enclosed over the whole cell.

The bounds of a cell certificate are written by the checker, not by the generator. Interval arithmetic over a box loses the cancellation that makes an equilibrium residual small, and how much it loses is a property of the arithmetic that no float sample predicts, so `--tighten` reads back the enclosure the verified code computes for each cell and writes the smallest claim that code accepts. The result is an ordinary certificate, and an ordinary run establishes it.

## Usage

```sh
python examples/make_equilibrium_certificate.py wout_solovev.nc cert.txt   # --nodes 6 --nu 8 --nv 4 --prec 53
stellarocq-check cert.txt        # extract/_build/default/main.exe of a Stellarocq build; STELLAROCQ_JOBS=n workers
python gen/verify_cert.py wout_solovev.nc cert.txt   # in a Stellarocq checkout: the certificate against the wout
```

```sh
python examples/make_equilibrium_certificate.py wout_solovev.nc cells.txt --cells --nodes 6 --nu 8192
stellarocq-check --tighten cells.txt cert.txt
stellarocq-check cert.txt
```

## Results

Six nodes per case, 20 worker processes. The field scale is the reference `B^2` scale the script prints for the point certificate.

| case | points | bound on `r_s`, of the field scale | verdict |
|---|---|---|---|
| `wout_solovev` (axisymmetric, ns=55, power series) | 48 | 7.3e-5 | VALID, under 0.1 s |
| `wout_cma` (3D, nfp=2, 59 modes, ns=51) | 192 | 3.3e-2 | VALID, 0.3 s |
| `wout_cth_like_fixed_bdy` (3D, nfp=5, 41 modes, ns=25, two-power pressure) | 192 | 7.9e-3 | VALID, 0.2 s |
| `up_down_asym` (non-stellarator-symmetric, ns=17) | 48 | 6.0e-3 | VALID, under 0.1 s |
| solovev with one `rmnc` coefficient of a certified stencil perturbed by 0.1% | 48 | same claim | INVALID, under 0.1 s |

Every stellarator-symmetric case above also certifies through the non-stellarator-symmetric reconstruction with its antisymmetric coefficients set to zero, at the same bounds (`gen/make_cert.py --force-lasym` of Stellarocq), which is the reduction `tests/test_lasym.py` requires of the solver.

| case | cells | worst cell bound | of the field scale | verdict |
|---|---|---|---|---|
| `wout_solovev` (axisymmetric, ns=55) | 49152 | 1.3e-5 | 1.9e-4 | VALID, 41 s |
| `wout_circular_tokamak_reference` (axisymmetric, ns=17) | 49152 | 1.5e-2 | 2.5e-4 | VALID, 48 s |
| `wout_cma` (3D, nfp=2, 59 modes, ns=51) | 24576 | 1.3e-2 | 4.3e-2 | VALID, 90 s |
| `wout_li383_low_res_reference` (3D, nfp=3, 25 modes, ns=16) | 24576 | 8.7e-1 | 2.4e-1 | VALID, 39 s |

Tightening the four took 30 s, 33 s, 50 s and 24 s. The two axisymmetric cases carry 6 nodes of 8192 poloidal cells covering the whole angular torus; the two three-dimensional ones carry 3 nodes of 4096 poloidal cells at each of 2 toroidal angles.

An axisymmetric equilibrium has every `n` zero, so its toroidal derivative encloses to zero and one cell spans the whole toroidal angle; the covering of the angular torus is one-dimensional. A three-dimensional one has to resolve the toroidal direction as finely as the poloidal, which squares the cell count, so its cells cover the poloidal angle at each of a few toroidal angles instead.

Certification is offline and per-wout; nothing runs in CI.

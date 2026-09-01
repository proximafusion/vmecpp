# Proof-carrying equilibria

Rocq is a proof assistant: a program that checks mathematical proofs down to every inference step. It has been developed at Inria since 1984 under its former name, Coq, and received the ACM Software System Award in 2013. It is the system in which the four-color theorem was verified and in which CompCert, the first formally verified C compiler, was built. When Rocq accepts a theorem, what remains to be trusted is its small proof-checking kernel and the stated assumptions, not the author of the proof and not the code that produced the numbers.

[Stellarocq](https://github.com/CharlesCNorton/stellarocq) uses that machinery to give a VMEC++ equilibrium a certificate. A converged wout ships with a small file, and an independent checker, extracted from a Rocq proof, validates it. The theorem behind the checker says: if the verdict is VALID, then the ideal-MHD force residual of the field reconstructed from these exact coefficients, by a reconstruction rule stated in the certificate, is a genuine real number at every certified point, with magnitude below the stated bound. A division by zero, an invalid square root, or an unlucky rounding anywhere in that evaluation makes the verdict INVALID, because the arithmetic is verified interval arithmetic (CoqInterval, proven against the Flocq formalization of IEEE-754).

The value is independence. "This wout satisfies force balance" normally means "VMEC++ says so, and VMEC++ agrees with Fortran VMEC." The checker shares no code with either and does not trust the generator that wrote the certificate: it recomputes everything from the coefficients with proven-sound arithmetic. A wout that passes cannot misstate its residual, whatever bug either solver might contain.

What is certified: the mu0-scaled residual `J x B - grad p` of the smooth field defined by the wout Fourier coefficients under a fixed interpolation rule, evaluated at a stated grid of points, bounded per component. What is not: the continuum between the points, or anything about the solver's internals. The trust base is the Rocq kernel, the classical real axioms of its standard library, the primitive float and integer specifications, and a thin parsing driver, all listed in the Stellarocq README.

## Usage

```sh
python examples/make_equilibrium_certificate.py wout_solovev.nc cert.txt
stellarocq-check cert.txt        # the checker binary, built from Stellarocq
```

## Results

| case | points | verdict |
|---|---|---|
| `wout_solovev` (axisymmetric, ns=55) | 48 | VALID, 146 s |
| `wout_cma` (3D, nfp=2, 59 modes, ns=51) | 72 | VALID, 2025 s |
| solovev with one `rmnc` coefficient perturbed by 0.1% | 48 | INVALID, 0.6 s |

Certification is offline and per-wout; nothing runs in CI.

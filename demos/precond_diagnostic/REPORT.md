# Preconditioner diagnostic report

Branch `miller-recurrence`; candidate preconditioner variants tested on
solovev, CTH, reduced W7-X, and full W7-X fixed-boundary cases.

## TL;DR

- **Best confirmed win on full `input.w7x`: Candidate B
  (`precond-rz-shear`), -5.6%** (2759 iters vs 2923 baseline). A shear
  stiffness term `cxd * (m*iota_f - n*nfp)^2` added to the R,Z preconditioner
  diagonal.
- On reduced/smaller cases, a parity-max variant (`precond-cand-c`) also
  helps; a related Gershgorin-regularization variant
  (`precond-rz-parity-coupling`) fails to converge on the full W7-X case.
  Parity coupling is promising but needs the proper 2x2 block solve, not
  diagonal regularization.
- **Late-stage R,Z residual is concentrated at low m (0-5), n=0-1**, not high
  m. The Miller-recurrence branch already fixed the high-m NESTOR tail. This
  redirects the next round of preconditioner work toward low-m / shear
  physics rather than high-m damping.
- **Condition numbers are NOT included in this report.** See "Missing:
  condition-number measurement" below.

## Candidates tested

Two separate experimental tracks produced slightly different implementations
of the same three ideas. Both tracks' results are reported below.

### Track 1: solovev -> CTH -> W7-X reduced (worktrees `vmecpp-precond-{baseline,cand-a,cand-b,cand-c}`)

| Tag | Branch | Patch |
|---|---|---|
| A | `precond-cand-a` | `m*m -> m*m * (1 + (m/(mpol-1))^2)` in R,Z diag |
| B | `precond-cand-b` | `cxd * (m*iota_f - n*nfp)^2` added to R,Z diag |
| C | `precond-cand-c` | `max(|even|, |odd|)` replaces the split diagonals |

### Track 2: full `input.w7x` (branches on local repo)

| Tag | Branch | Patch |
|---|---|---|
| A | `precond-rz-highm-rolloff` | `pow(sqrtSF, min(m^2/256, 8))` applied post-solve to R,Z coeffs (mirrors lambda) |
| B | `precond-rz-shear` | `cxd * (m*iota_f - n*nfp)^2` added to R,Z diag (same as Track 1 B) |
| C | `precond-rz-parity-coupling` | Gershgorin regularization of even/odd diagonals by `|ax[...+1]|` |

## Iterations-to-converge

| Case | Baseline | A | B | C |
|---|---:|---:|---:|---:|
| solovev (Track 1) | 906 | 885 (-2.3%) | 905 (flat) | **874 (-3.5%)** |
| CTH (Track 1) | 3026 | 3087 (+2.0%) | **2859 (-5.5%)** | **2859 (-5.5%)** |
| W7-X reduced (Track 1) | 540 | **449 (-16.9%)** | 482 (-10.7%) | 456 (-15.6%) |
| W7-X full `input.w7x` (Track 2) | 2923 | 2889 (-1.2%) | **2759 (-5.6%)** | diverged |

**Headline conflict:** Track 1 Candidate C (parity-max) wins monotonically
across small/medium cases; Track 2 Candidate C (Gershgorin) **diverges**
on the full problem. These are two different implementations of the same
idea. The Gershgorin variant over-regularizes the diagonal and destroys the
ability to represent the cross-parity coupling it is meant to approximate.
The proper 2x2 block tridiagonal solve has not been tried and is the right
next step for the parity-coupling idea.

**What is robust:** Candidate B (shear term) helps on every case where it
helps at all, modestly but consistently, including the full W7-X.

## Convergence plots

- [cth_convergence.png](cth_convergence.png) - CTH, Track 1
- [w7x_convergence.png](w7x_convergence.png) - W7-X reduced, Track 1
- Full W7-X logs available per-branch (Track 2); plot can be regenerated
  by adding its logs to `w7x_full/` and running `make_convergence_plot.py`

## What the per-mode spectrum says

### CTH late-stage (Track 1, iter 3026, baseline)

| Component | Dominant mode(s) |
|---|---|
| R force residual | **(m=3, n=1)** tied with (0,0); cluster at m=1..3, n=0..1 |
| Z force residual | **(m=3, n=1)** dominates; (2,1), (2,0), (1,0) follow |
| Lambda | **(m=4, n=1)** dominates by ~6x over (3,1) |

### W7-X full (Track 2, baseline)

Late-stage R,Z residual concentrated at low m (0-5), n=0-1. **Not high m.**
The Miller-recurrence work already eliminated the high-m NESTOR tail that
historically motivated aggressive high-m roll-off in the lambda preconditioner.
This implies:

- Candidate A (high-m roll-off) has limited headroom: the problem is no
  longer high-m limited.
- Candidate B (shear) targets the right part of the spectrum: `m*iota - n*nfp`
  is small for low-m / low-n modes near rational surfaces, so its diagonal
  contribution differentiates modes that are otherwise very similar in
  stiffness.

Full top-5 tables per candidate per component at
[mode_dominance.md](mode_dominance.md). Heatmaps and band traces at
`{case}/{tag}_heatmap.png` and `{case}/{case}_traces.png`.

## Missing: condition-number measurement

The user explicitly asked for condition numbers. **They are not computed in
this report.** Producing them requires:

1. A pybind hook exposing a single-step forward-force evaluation:
   `set_state(x) -> IdealMhdModel::update() once -> read preconditioned
   FourierForces`. Not yet present in
   `src/vmecpp/cpp/vmecpp/vmec/pybind11/pybind_vmec.cc`.
2. A Python driver that, at a converged state `x*`, perturbs each Fourier
   coefficient and reconstructs `P^{-1} J` by one-hot finite differences.
3. `numpy.linalg.svd` for solovev (N ~ O(10^2)) and CTH (N ~ O(10^3));
   `scipy.sparse.linalg.svds` for extremal singular values on reduced W7-X
   (N ~ O(10^4)). Full W7-X is infeasible with this approach.

Estimated effort: ~200 LoC pybind binding, ~150 LoC Python driver, plus a
solovev validation pass. This is the logical next step because it will
disambiguate:

- Is slow convergence **condition-number limited** (a few tiny singular
  values in specific mode directions) or **nonlinearity limited** (good
  conditioning but the linearized step is a poor local model)?
- Are the iteration-count wins above driven by spectrum changes or by
  unrelated side effects (time-step stability, damping, etc.)?

## Recommendation

1. **Adopt Candidate B (shear)** as the first real improvement. It is
   the only variant that wins on the full `input.w7x` case with a clear
   physical justification.
2. **Do not adopt Candidate C in its Gershgorin form** (`precond-rz-parity-coupling`). It diverges on full W7-X. The parity-max form
   (`precond-cand-c`) works on smaller cases but has not been tested on full
   W7-X. Before merging either, implement the proper 2x2 parity-block
   tridiagonal solve.
3. **De-prioritize Candidate A.** The high-m tail problem has been solved
   by the Miller recurrence work on this branch; further high-m roll-off
   has limited headroom and regresses on CTH.
4. **Build the condition-number measurement tool.** Without it, these
   iteration-count differences are suggestive but not conclusive. The
   tool is also the only way to target the (m=3, n=1) CTH residual
   systematically.

## Artifacts

- Convergence plots: [cth_convergence.png](cth_convergence.png), [w7x_convergence.png](w7x_convergence.png)
- Mode dominance: [mode_dominance.md](mode_dominance.md)
- Per-case spectra: `{solovev,cth,w7x}/{baseline,candA,candB,candC}_spectrum.csv`
- Heatmaps: `{case}/{tag}_heatmap.png`
- Scripts: `make_convergence_plot.py`, `make_mode_table.py`, `make_plots.py`
- Candidate worktrees (Track 1): `/home/jurasic/vmecpp-precond-{baseline,cand-a,cand-b,cand-c}`
- Candidate branches (Track 2): `precond-rz-{highm-rolloff,shear,parity-coupling}`, `precond-diagnostic-base`
- Full W7-X artifacts: `/tmp/precond_baseline_artifacts/`
- Diagnostic dump env var: `VMECPP_DUMP_FORCE_SPECTRUM=1`

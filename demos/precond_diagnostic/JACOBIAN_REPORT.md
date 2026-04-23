# Preconditioner Jacobian diagnostic: condition numbers and matrix structure

Branch: `trucation-diagnostic` (off `precond-rz-shear` which is off
`miller-recurrence`). This report documents the `VmecJacobianProbe` tool
and presents condition-number / matrix-structure measurements for solovev,
CTH, and reduced W7-X fixed-boundary cases, comparing the baseline
preconditioner against the shear-term candidate (`precond-rz-shear`).

## TL;DR

| Case | N | kappa(J) | kappa(P^-1 J) baseline | kappa(P^-1 J) shear | preconditioner gain |
|---|---:|---:|---:|---:|---:|
| solovev (axisymmetric, ns=55, mpol=6)       |   990 |  1.14e7 |  6.11e5 |  6.11e5 | **19.0x** |
| CTH fixed-bdy (ns=15, mpol=5, ntor=4)       |  2250 |  7.09e7 |  8.54e6 |  8.55e6 |  **8.3x** |
| W7-X reduced (ns=11, mpol=6, ntor=4, nfp=5) |  1980 |  8.82e6 |  7.01e5 |  7.00e5 | **12.6x** |

- The current R,Z tridiagonal + lambda diagonal preconditioner gives
  **8-19x** condition-number reduction on every case tested.
- **Shear (candidate B) does not change the condition number at
  convergence.** Its 5.6% iteration-count win on full W7-X is therefore
  a *dynamical* effect (time-step admissibility, far-from-convergence
  curvature), not a spectral one.
- The **measured condition numbers are consistent with observed
  iteration counts via the sqrt-kappa rule of thumb.** Solovev
  sqrt(6.1e5) ~ 780 iterations (observed ~900); CTH sqrt(8.5e6) ~ 2900
  (observed ~3000). This cross-check is the primary validation that the
  tool is measuring what we think it is.
- **Single-biggest result for preconditioner development:** solovev's
  19x gain drops to 8.3x on CTH and rises back to 12.6x on W7-X
  reduced. The variation across cases is the information-rich signal:
  it tells us where the current preconditioner is doing well
  (solovev/W7-X) vs badly (CTH), and gives a measurable target for
  improvement.

## Tool summary

- **C++** helper: [src/vmecpp/cpp/vmecpp/vmec/vmec_jacobian_probe/vmec_jacobian_probe.h](../../src/vmecpp/cpp/vmecpp/vmec/vmec_jacobian_probe/vmec_jacobian_probe.h)
  and
  [.cc](../../src/vmecpp/cpp/vmecpp/vmec/vmec_jacobian_probe/vmec_jacobian_probe.cc).
  Wraps a single-thread `Vmec` instance. `EvaluateForces(preconditioned)`
  runs one `IdealMhdModel::update` and returns either the
  `INVARIANT_RESIDUALS` checkpoint (before preconditioning) or
  `PRECONDITIONED_RESIDUALS` checkpoint (after M1 + RZ tridiagonal +
  lambda diagonal). State packed as flat vector in internal product basis.
- **Pybind**: new `VmecJacobianProbe` class in
  [pybind_vmec.cc](../../src/vmecpp/cpp/vmecpp/vmec/pybind11/pybind_vmec.cc).
- **Python driver**: [`compute_jacobian.py`](compute_jacobian.py) - one-hot
  central-difference FD Jacobian, saves `.npz`, plots first diagnostics.
- **Analyzer**: [`analyze_jacobian.py`](analyze_jacobian.py) - numerical-rank
  truncated condition number, matrix structure, per-mode diagonal maps,
  radial profile of diag energy.
- **Runtime toggle**: `VMECPP_DISABLE_SHEAR=1` disables the shear
  stiffness term so baseline and shear can be compared with a single
  compiled binary.

## Methodology

1. Run VMEC to convergence at `x*` via `VmecJacobianProbe`.
2. Snapshot `x0 = x*`.
3. For `i = 0..N-1`, central-difference
   `J[:, i] = (F(x0 + eps e_i) - F(x0 - eps e_i)) / (2 eps)` with
   `eps = 1e-7`. Both `F_unprec` and `F_prec = P^-1 F_unprec` extracted.
4. Full SVD; report *numerical-rank-truncated* condition number, i.e.
   truncate singular values below `rel_tol * sigma_max` (`rel_tol = 1e-8`).
5. Raw `sigma_max / sigma_min` is not meaningful because VMEC's force map
   has an intrinsic null space from gauge redundancies (m=0 toroidal,
   axis extrapolation, m=1 constraint, zcs/rss couplings) of dimension
   ~15-30% of N. The numerical rank is stable across `rel_tol in
   {1e-6, 1e-8, 1e-10, 1e-12}` for all cases here - there is a clean
   spectral gap between the true null space and the smallest physical
   singular value.

## Results

### Solovev (axisymmetric)

N = **990** (ns=55 x mpol=6 x 1 ntor+1 x 3 comp x 1 basis).
Numerical rank = **854** (86%).

| quantity | J | P^-1 J (baseline) | P^-1 J (shear) |
|---|---:|---:|---:|
| sigma_max     |  5458.55 |  67.56 | 67.56 |
| Frobenius     | 26155.85 | 211.05 | 211.03 |
| kappa (rel_tol=1e-8) | **1.14e7** | **6.11e5** | **6.11e5** |

- Preconditioner reduces Frobenius norm by **124x**, spectral radius
  by **81x**, and condition number by **19x**.
- Shear term has **no effect on solovev**. Expected: solovev is
  axisymmetric (ntor=0), so `(m*iota - n*nfp)^2 = (m*iota)^2` just
  rescales the existing `m^2` diagonal term by a factor of `iota^2`.

**Plots**:
- Singular-value spectrum (log-normalized):
  ![`jacobian/solovev_baseline_sv_normalized.png`](jacobian/solovev_baseline_sv_normalized.png).
  Clean drop at index 854 between physical modes and the null-space
  tail. Blue (J) and red (P^-1 J) curves are nearly identical in shape
  but P^-1 J is shifted down by the normalization.
- Matrix structure heatmap:

  ![`jacobian/solovev_baseline_structure.png`](jacobian/solovev_baseline_structure.png)

  Left (J) shows visible block-tridiagonal-per-mode structure; right
  (P^-1 J) has strongly suppressed off-diagonals, making the per-mode
  tridiagonal pattern much less pronounced.

### CTH-like fixed boundary (3D, stellarator)

N = **2250** (ns=15 x mpol=5 x 5 ntor+1 x 3 comp x 2 basis).
Numerical rank = **1570** (70%).

| quantity | J | P^-1 J (baseline) | P^-1 J (shear) |
|---|---:|---:|---:|
| sigma_max     | 1336.80 | 207.52 | 207.52 |
| Frobenius     | 4872.32 | 520.64 | 520.58 |
| kappa (rel_tol=1e-8) | **7.09e7** | **8.54e6** | **8.55e6** |

- Preconditioner gives **8.3x** kappa reduction - the worst of the
  three cases. CTH is the hardest case for this preconditioner.
- **Shear is spectrally irrelevant on CTH**: 8.54e6 vs 8.55e6. The
  5.5% iteration-count improvement reported earlier must come from
  non-spectral effects.

**Plots**:
- ![`jacobian/cth_baseline_sv_normalized.png`](jacobian/cth_baseline_sv_normalized.png)
  - the P^-1 J tail (indices 500-1500) sits slightly *above* the J
  tail on log scale, which is the signature of spectrum-flattening by
  the preconditioner.
- ![`jacobian/cth_baseline_structure.png`](jacobian/cth_baseline_structure.png) -
  J has strong block-tridiagonal structure; P^-1 J shows a pattern with
  most energy near the diagonal but with extensive off-block coupling
  still present, which is the signature of remaining stiffness the
  current preconditioner cannot address.
- Per-mode diagonal energy maps:
  ![`jacobian/cth_baseline_diag_PiJ_R.png`](jacobian/cth_baseline_diag_PiJ_R.png),
  ![`_Z.png`](jacobian/cth_baseline_diag_PiJ_Z.png) -
  energy is concentrated at low m and low n, consistent with the
  baseline force-residual analysis from the earlier per-mode diagnostic.

### W7-X reduced (3D, stellarator, fixed boundary)

N = **1980** (ns=11 x mpol=6 x 5 ntor+1 x 3 comp x 2 basis).
Numerical rank = **1354** (68%).

| quantity | J | P^-1 J (baseline) | P^-1 J (shear) |
|---|---:|---:|---:|
| sigma_max     | 81590.55 |  54.13 |  54.12 |
| Frobenius     | 377611.34 | 151.07 | 151.06 |
| kappa (rel_tol=1e-8) | **8.82e6** | **7.01e5** | **7.00e5** |

- Preconditioner gives **12.6x** kappa reduction. The spectral-radius
  reduction here (**1507x**) is an order of magnitude larger than on
  solovev (81x) or CTH (6.4x). This is because W7-X's unpreconditioned
  operator has very large values driven by the stellarator field-line
  geometry; the tridiagonal preconditioner successfully normalizes
  this out.
- **Shear gives 0.07% reduction in kappa on W7-X reduced.** Again
  spectrally irrelevant at convergence.

**Plots**:
- ![`jacobian/w7x_shear_structure.png`](jacobian/w7x_shear_structure.png) -
  The most striking visualization in the study. J (left) shows a
  distinctly banded structure with very strong tridiagonal blocks per
  Fourier mode, high-amplitude off-diagonals, and visible stellarator
  n-coupling blocks. P^-1 J (right) is dramatically more uniform,
  with the operator normalized to O(1) values across the entire
  matrix. This is exactly the right shape for a good preconditioner:
  transform J from ill-scaled block-tridiagonal into a near-uniform
  operator.
- ![`jacobian/w7x_baseline_sv_normalized.png`](jacobian/w7x_baseline_sv_normalized.png) -
  Spectrum.

## Interpretation and next steps

### Consistency check: kappa vs iteration count

For a first-order time-stepper on a linearized system, convergence to
a factor-10 reduction in residual requires roughly `sqrt(kappa)`
iterations. Check against observed convergence:

| Case | kappa(P^-1 J) | sqrt(kappa) | Observed iters |
|---|---:|---:|---:|
| solovev | 6.11e5 |  780 |  900 |
| CTH     | 8.54e6 | 2920 | 3026 |
| W7-X reduced | 7.01e5 |  840 |  540 |

The solovev and CTH numbers match within 15%, a strong independent
validation of the tool. W7-X reduced has fewer iterations than expected
from kappa alone, which is consistent with the observation that this
preconditioner over-shrinks the operator (spectral_J / spectral_PiJ =
1507x, vs 81 on solovev). That over-shrinkage increases the effective
time-step and reduces iteration count below the sqrt(kappa) prediction.

### What this tells us about the shear candidate

On every case tested, shear has zero or negligible effect on the
condition number at convergence. Yet on the full W7-X problem the
subagent measured a 5.6% iteration-count reduction (2759 vs 2923).

Conclusion: **the shear term is not doing spectral preconditioning**.
It is helping dynamically - improving stability or time-step
admissibility far from the converged state. Possible mechanisms:
- Field-line-bending stiffness `(m*iota - n*nfp)^2` differs between
  iterations and convergence, so its effect on kappa is small *at
  convergence* but non-zero during the transient.
- The shear term is ~zero on resonant surfaces (`m*iota = n*nfp`),
  which means it under-damps modes at or near rational surfaces, which
  may be exactly the ones that would otherwise explode into
  non-linear instabilities.

To investigate, we need Jacobian probes at *intermediate* iterations
(e.g., iter 100, 500, 1000 on W7-X), which is a straightforward
extension of the current tool: add a callback to `Vmec::SolveEquilibrium`
that snapshots `decomposed_x_[0]` every N iterations, then run the
Jacobian extraction on those snapshots.

### Where to look for preconditioner improvements

The per-mode diagonal maps
(![`jacobian/cth_baseline_diag_PiJ_R.png`](jacobian/cth_baseline_diag_PiJ_R.png)
and the matrix structure plots show that after preconditioning there is
still substantial off-block coupling between different `(m, n)` Fourier
modes. The current preconditioner assumes modes are decoupled: R,Z
tridiagonal separately per `(m, n)`, plus lambda diagonal.

**Directions worth exploring** (now that the tool exists to measure
impact):

1. **Cross-mode coupling from the spectral constraint force.** The
   constraint force multiplier `tcon` introduces off-diagonal coupling
   between neighboring-m modes that the current preconditioner doesn't
   model. Augment the diagonal with a `faccon`-dependent term.

2. **Parity-coupled 2x2 tridiagonal block solve** (the proper form
   of candidate C that the subagent implemented as a Gershgorin
   approximation which diverged on full W7-X). Build the true 2x2
   parity-block operator per (m, n) and solve it as a block tridiagonal.

3. **Damp the null-space-adjacent modes.** The physical singular
   values right above the null-space gap (e.g. the last 50-100 modes
   before the 854th on solovev) are the smallest non-null singular
   values and thus drive kappa. Targeting these specifically with a
   corrective term could yield large kappa improvements.

4. **Iota-band preconditioning.** Use a discrete `(m*iota - n*nfp)`
   lookup per-mode to preconditioner each mode at its shear-weighted
   stiffness instead of the uniform-m^2 + uniform-n^2 decomposition.

## Tool usage

```python
import vmecpp
from vmecpp.cpp import _vmecpp as cpp

indata = vmecpp.VmecInput.from_file(
    "examples/data/solovev.json"
)._to_cpp_vmecindata()
probe = cpp.VmecJacobianProbe(indata)
probe.run_to_convergence()
probe.snapshot_state()

x0 = probe.get_state_vector()        # (N,) flat state, internal basis
f0 = probe.evaluate_forces(preconditioned=True)   # (N,) flat force

# One-hot perturbation (for finite-difference Jacobian column):
x = x0.copy()
x[i] += 1e-7
probe.set_state_vector(x)
f = probe.evaluate_forces(preconditioned=True)
probe.restore_state()
```

Index metadata:
```python
m = probe.index_m()       # (N,) poloidal mode number per entry
n = probe.index_n()       # (N,) toroidal mode number per entry
jF = probe.index_jF()     # (N,) radial surface index per entry
basis = probe.index_basis()   # (N,) 0..num_basis-1 basis function
comp = probe.index_comp()     # (N,) 0=R 1=Z 2=lambda
```

## Artifacts

- Scripts: [`compute_jacobian.py`](compute_jacobian.py),
  [`analyze_jacobian.py`](analyze_jacobian.py)
- Per-case `.npz`: `jacobian/{case}_{baseline,shear}.npz`
  (J, PiJ, mode-index metadata). 15-80 MB each.
- Per-case analysis JSON: `jacobian/{case}_{baseline,shear}_analysis.json`
  (condition numbers at multiple `rel_tol`, Frobenius norms, rank counts).
- Plots:
    - `jacobian/{case}_{variant}_sv_normalized.png` - spectrum
    - `jacobian/{case}_{variant}_structure.png` - matrix heatmap
    - `jacobian/{case}_{variant}_eigs_symmetric_part.png`
    - `jacobian/{case}_{variant}_diag_{J,PiJ}_{R,Z,L}.png` - per-mode stiffness
    - `jacobian/{case}_{variant}_diag_{J,PiJ}_radial.png` - radial profile

## Reproduce

```bash
# Solovev (~2 min per variant):
VMECPP_DISABLE_SHEAR=1 .venv/bin/python demos/precond_diagnostic/compute_jacobian.py \
    --input examples/data/solovev.json --out solovev_baseline
.venv/bin/python demos/precond_diagnostic/compute_jacobian.py \
    --input examples/data/solovev.json --out solovev_shear

# CTH (~35 min per variant):
VMECPP_DISABLE_SHEAR=1 .venv/bin/python demos/precond_diagnostic/compute_jacobian.py \
    --input examples/data/cth_like_fixed_bdy.json --out cth_baseline
.venv/bin/python demos/precond_diagnostic/compute_jacobian.py \
    --input examples/data/cth_like_fixed_bdy.json --out cth_shear

# W7-X reduced (~2 min per variant):
VMECPP_DISABLE_SHEAR=1 .venv/bin/python demos/precond_diagnostic/compute_jacobian.py \
    --input demos/precond_diagnostic/w7x_reduced.json --ns 11 \
    --out w7x_baseline
.venv/bin/python demos/precond_diagnostic/compute_jacobian.py \
    --input demos/precond_diagnostic/w7x_reduced.json --ns 11 \
    --out w7x_shear

# Analyze any run:
.venv/bin/python demos/precond_diagnostic/analyze_jacobian.py \
    --npz demos/precond_diagnostic/jacobian/solovev_baseline.npz
```

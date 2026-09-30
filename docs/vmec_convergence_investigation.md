# What limits VMEC++ convergence: diagnosis, attempts, results

Setup for everything below: vmecpp 0.7.5 from PyPI, driven from Python through
`VmecModel`, single thread, single grid (final `ns`). Test cases: `cth_like_fixed_bdy`
(and its prescribed-iota variant), `li383_low_res`, `solovev`, `cma`,
`near_axis_iota_nfp4`, `w7x`, and the ConStellaration boundary from PR #878
(`constellaration_nfp5`) plus ±15% random perturbations of it. Numbers are single
runs; the transient is chaotic, so differences of a few percent are not significant.

## 1. Summary

| Finding | Evidence |
|---|---|
| The user `delt` is a hand-tuned stability (CFL-like) limit: stable iff `delt² λ_max(P⁻¹H) < 4`, independent of damping | ConStellaration: 2/√λ_max = 0.63–0.65, constant along the trajectory; main stops converging at 0.65. W7-X: 2/√λ_max ≈ 0.58, and the restarts ratchet `delt` from 1.0 to ≈0.55. |
| The Δt-dependence of the converged state (PR #878) is the m = 1 gauge freezing at fsqz < 1e-6 | Fixing the gauge from iteration 1: 0.6-vs-0.35 difference in `z_cs` drops from 4.4e-3 to 1.7e-7 at ftol 1e-14 (upstream now pins the gauge by default, #849). |
| **The slow tail is a smooth, low-(m, n) mode, ~99% λ, coupled to R/Z** | Tail error decomposition; dense spectra (section 2). |
| The smallest eigenvalues of P⁻¹H are radial odd-even ("checkerboard") λ modes, but they carry only ~1% of the tail error | Section 2.2. |
| VMEC's diagonal-in-(m, n) preconditioner misses the (m, n) coupling on a surface; an exact {R/Z, λ} block cuts √κ by 3–6× | Section 2.3. |
| VMEC's raw force Jacobian = diag(d) × Hessian of the discrete energy, to 0.18% | Section 4. |
| Near-null directions: poloidal relabeling + near-gauge R/Z modes; spectral condensation makes VMEC's Hessian definite, but only weakly | Section 4. |

## 2. Which modes limit the tail

### 2.1 VMEC's damping is critical damping of the dominant mode

The `|log(fsq_k/fsq_{k-1})|` average in the damping detects oscillation. Its fixed
point is critical damping of whichever mode currently dominates, dτ ≈ h√λ_dom. The
`0.15` cap limits the damping during transients, and `NDAMP = 10` is a low-pass filter.
Tail iterations therefore scale like √(λ_max/λ_dom).

The cth_like tail rate (×0.970 per iteration, h = 1.05) implies λ_dom ≈ 2e-4, which
lies among the smallest eigenvalues of P⁻¹H. With the true (λ_min, λ_max), a fixed
optimal heavy-ball momentum was **2–8× slower** than VMEC's damping: 1465–6081 against
503 iterations on cth_like, and 3822–8844 against 2213 on ConStellaration.

### 2.2 The tail error is smooth λ; the smallest eigenvalues are checkerboard modes

Dense finite-difference P⁻¹H at the converged state (2740 unknowns for cth_like, 4868
for ConStellaration):

- **Tail error.** At fsq ≈ 1e-9 the error is 99% λ, radially smooth, and in low (m, n),
  on cth_like and near_axis. On near_axis it is λ at m = 1, n = 0, near the axis.
- **Smallest eigenvalues (~1e-4 relative).** These are λ_j alternating in j. The
  half-grid average (λ_j + λ_{j+1})/2 annihilates them, and VMEC's λ preconditioner,
  which has no radial coupling, cannot represent that. They hold only ~1% of the tail
  error, so they do not limit the damped scheme, but they make Newton-type steps
  ill-posed.
- **No symmetric higher-order radial stencil removes them.** Any midpoint-symmetric
  interpolation of any order is zero on (−1)^j. What does lift them: a second-difference
  (Rhie–Chow-type) stabilization, or storing λ on the half grid.

![spectra and checkerboard](lambda_spectra_checkerboard.png)

Left: smallest eigenvalues of the preconditioned operators. Middle: the slowest
eigenvector (checkerboard) against the tail error (smooth hump). Right: radial λ
discretizations. Wider symmetric stencils don't help (4.4e-6 → 6e-6 relative); a
1e-2 second-difference term (6.1e-4) and λ on the half grid (7.1e-4) lift the null.

### 2.3 What the missing coupling costs (dense spectra, √κ)

| Preconditioner | cth_like | ConStellaration |
|---|---|---|
| VMEC | 179 | 362 |
| Exact per-(m,n) radial blocks (VMEC's structure) | 223 | 407 |
| Exact per-surface blocks (no radial coupling) | 326 | 354 |
| Exact R/Z block + VMEC λ | 133 | 153 |
| VMEC R/Z + exact λ block | 91 | 356 |
| **Exact {R/Z, λ} block Jacobi** | **54** | **63** |
| λ solved exactly (Schur), VMEC R/Z | 61 | 315 |
| λ exact + exact R/Z block | 27 | 31 |

- **What VMEC misses is the (m, n) coupling on a surface, not the radial structure.**
  Keeping VMEC's per-(m, n) radial structure with exact entries is *worse* than what
  VMEC has.
- **The raw Hessian H is exactly radially tridiagonal.** P⁻¹H is not, because VMEC's
  tridiagonal preconditioner has a dense inverse.

## 3. Attempts and outcomes

### 3.1 Iteration schemes

| Attempt | Outcome |
|---|---|
| FIRE 2.0 | Fails on every stiff case: uphill resets every ~20 evaluations. The preconditioned power F·v doesn't track the residual. |
| Nesterov with adaptive restart | Wins on mild cases (cth_like 83 against 120 evaluations); fails or is slower on cma, near_axis and W7-X. The m = 1 gauge switch at fsqz < 1e-6 causes a limit cycle; with the gauge fixed, near_axis converges in 2804. |
| Heavy ball with spectral β | Worse than VMEC; the Rayleigh estimate of μ goes ≤ 0 on W7-X and ConStellaration. |
| Stability-limited dt + τ guard (exact quadratic of the Jacobian, pointwise ½ bound + momentum reset) | 0 bad Jacobians; geometric mean 0.96 of evaluations against main + backup fix. [^1] |
| Newton–Krylov / pseudo-transient continuation in the tail | Stalls: near-null directions make the Newton step ~3% of \|x\|. |

[^1]: That suite used a driver with an extra 1/(1+dτ) factor in the damping, i.e.
over-damped. All later drivers use VMEC's exact form and reproduce native iteration
counts to within 1–3%.

![momentum methods](momentum_comparison_gauge.png)

### 3.2 Validated backup (robustness)

![delt scan](constellaration_delt_scan.png)
![resolution scan](backup_fix_resolution_scan.png)

- **ConStellaration.** `delt` 0.66–0.9 converges in 895–1175 iterations, against
  943–3580 or failure on main.
- **Perturbed boundaries at mpol = ntor = 8.** 12–41% fewer evaluations.
- **W7-X.** 3–8% fewer across resolutions (6,6) to (20,16) and `delt` 0.8–1.2; W7-X
  only restarts during the transient.

### 3.3 Solving λ exactly

- **λ enters linearly.** F_λ is linear in λ at fixed R, Z to 1e-10, so one block solve
  gives λ*(R, Z).
- **Cost** (single thread), in force evaluations: the solve costs ≈1–2 per step; a
  rebuild costs 19 (cth_like) to ≈450 (ns = 1000, 18×18). The Galerkin half of the
  rebuild could be done as a Fourier convolution and become negligible.
- **It makes the tail slower.** Eliminating λ softens R/Z: the coupling term is ~½‖H_RR‖,
  and on ConStellaration the reduced problem's softest mode (8e-5) is exposed.

| Tail fsq 1e-5 → 1e-12 | cth_like | ConStellaration |
|---|---|---|
| VMEC | 412 | 1655 |
| λ exact, 1 evaluation/step | 367 | 7878 |
| λ exact, 2 evaluations/step | 647 | 23042 |
| Exact {R/Z, λ} block Jacobi (finite differences) | **231** | **961** |

![exact lambda tails](exact_lambda_tails.png)
![exact lambda cost](exact_lambda_cost.png)

### 3.4 Block preconditioners in the tail

- **Finite-difference assembly by radial coloring.** Perturbing one (family, m, n) on
  every third surface gives all three bands: 3 × dofs-per-surface evaluations,
  independent of ns.
- **Full {R/Z, λ} blocks.** Tail 1.7–2.4× shorter.
- **Low modes only (m < 4, n ≤ 4, the rest VMEC).** Tail 2.4× shorter on ConStellaration
  (assembly 271 evaluations), −22% on W7-X. Mixed at larger mc, because the low/high
  coupling is dropped.

![block preconditioner tails](block_preconditioner_tail.png)

### 3.5 Analytic preconditioners (no finite differencing)

- **Analytic λ–λ block.** Metric resolved on the surface plus VMEC's radial averaging:
  4% error against the exact block (27% for the surface-averaged, (m,n)-diagonal
  version). Replacing only VMEC's λ preconditioner by the regularized version changes
  iterations by −8% to +3% at 1.4–5× the wall time. Without stabilization it stalls on
  bad Jacobians.
  ![lambda-only preconditioner](vmec_lambda_regularized_preconditioner.png)
- **Analytic low-mode coupled block.** Per-point 28×28 Hessians of the energy density
  (JAX) combined with a low-mode Galerkin: 1.05% error against finite differences.
  Rebuild ≈0.4 s on cth_like (≈250 evaluations there; estimated ≈100–150 at W7-X scale).
  Made positive definite by a block modified Cholesky. **It does not help yet**:
  - small floor: eig(M⁻¹H) up to ~900, i.e. directions where M is too soft;
  - large floor: soft physical modes over-stiffened to 2e-5, and the tail takes 50k iterations.
  ![analytic low-mode block](lowmode_analytic_preconditioner.png)

## 4. Near-null directions (prescribed-ι cth_like)

- **VMEC's force is a row-scaled energy gradient.** Without spectral condensation,
  VMEC's force Jacobian equals diag(d) × analytic energy Hessian to 0.18%
  (d = 0.71–1.07; the largest deviations are in the m = 1 r_ss rows). The apparent 9.8%
  asymmetry is exactly this row scaling.
- **Analytic Hessian: 136 nonpositive eigenvalues.** Their eigenvectors lie median 99%
  in the span of the poloidal relabeling directions
  (δR, δZ, δλ) = (R_θ, Z_θ, 1 + λ_θ)·φ. After projecting those out, 31 near-gauge R/Z
  directions remain at about −5e-4 of the mean diagonal. They are not pressure-driven
  (30 with the pressure off).
- **Spectral condensation.** Small (1.3% of ‖H‖, symmetric, R/Z only), but it makes
  VMEC's Hessian definite: 0 nonpositive eigenvalues, against 6 without it. It gives the
  near-gauge directions about 24× the energy's own stiffness. The softest remaining
  directions are m = 1 relabelings next to the axis (3e-6), where the constraint weight
  m(m−1) vanishes.
- **Why this blocks the analytic preconditioner.** The physically soft modes and the
  near-gauge modes have similar eigenvalue magnitudes, so no scalar floor separates them.
  The spectral-condensation Hessian must be included in closed form. Its form in VMEC++:
  xmpq = m(m−1) (×√s for odd m); gcon = (rcon − rcon₀)R_θ + (zcon − zcon₀)Z_θ;
  de-aliased with faccon ∝ 1/xmpq²; profile tcon from the preconditioner diagonal
  × (32Δs)².

![indefinite directions](indefinite_directions.png)

## 5. End-to-end result (current best)

Validated backup, then the finite-difference low-mode block tail from fsq 1e-5
(0.8 × the stability limit), against native VMEC++. ftol 1e-12; wall time single-threaded.

| Case | native | backup fix | block tail |
|---|---|---|---|
| cth_like | 666 ev / 0.91 s | 666 / 1.01 s | 687 / 1.71 s |
| li383 | 393 / 0.10 s | 393 / 0.09 s | 475 / 0.27 s |
| ConStellaration (`delt` 0.7) | fails | 2512 / 2.35 s | **1230 / 1.92 s** |
| pert0 | 1684 / 1.48 s | **1539 / 1.37 s** | 1325 / 2.17 s |
| pert3 | 3486 / 3.14 s | **1603 / 1.67 s** | 1311 / 2.15 s |
| pert5 | 3081 / 2.71 s | 3628 / 3.37 s | **1467 / 2.15 s** |
| pert7 | 3251 / 2.77 s | 3092 / 2.76 s | **1483 / 2.25 s** |
| mpol = ntor = 10 | 6577 / 17.6 s | 3092 / 8.7 s | **2264 / 8.3 s** |
| W7-X | 2962 / 53 s | **2791 / 50 s** | 5623 / 122 s |

"ev" is force evaluations, including the 271 for block assembly; bold is the fastest
wall time.

- **Robustness.** No failures; ConStellaration no longer deadlocks.
- **Speed.** 1.2–2.2× faster in wall time than native on base ConStellaration, pert5,
  pert7 and mpol = ntor = 10. On pert0 and pert3 it saves evaluations but not wall time,
  because the Python tail driver adds ≈0.6 ms per step.
- **Regressions.**
  - Small cases, where the assembly exceeds the whole tail.
  - W7-X, where m < 4, n ≤ 4 is too small for mpol = ntor = 12.
  - An automatic keep-or-switch-back rule over a fixed 400-evaluation window was too
    conservative: it switched back on cases where the block was winning.

![end to end](end_to_end_comparison.png)

## 6. Recommendations

1. **Replace the user `delt` with the tracked stability limit** 2/√λ_max: power iteration, ≈4% overhead.
2. **Build the analytic tail preconditioner** as an {R/Z, λ} block over low modes, with
   mc and nc scaled to mpol and ntor. Include the closed-form spectral-condensation
   Hessian, pin the m = 1 gauge, handle the λ odd-even modes (radial averaging in the
   block plus stabilization), and assemble it in C++ from the pointwise Hessian formulas.
4. **Add a safeguard that works.** Decide whether to keep the block from the post-switch
   decay rate and the expected tail length compared with the assembly cost.

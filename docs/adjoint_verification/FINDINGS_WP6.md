# WP6: reference gradients and end-to-end verification

## Reference validity

WP5 has established Newton convergence on the small CTH case only. Its
`reference_equilibrium` helper is not published, so WP6 uses the requested
interim VMEC reference at `ftol=1e-16`. This is a normalized squared-force
termination threshold, not a bound of `1e-16` on geometry or gradient error.
Finite-difference step studies and repeat-solve differences are needed to
separate truncation error from the reference's convergence floor.

All cases and objective definitions come from the WP0 harness. They are
stellarator symmetric, fixed boundary, and use `always_fix_m1_gauge=True`.
Random-linear objective weights must remain fixed when the boundary changes.
The intermediate reference does not establish WP3's analytic-mask correctness
or WP4's relabeling-invariance claims.

The restored runtime uses Enzyme v0.0.264 with Clang 18. Missing Clang development
headers initially caused Enzyme to build only `LLVMEnzyme-18.so`. That plugin
does not implement the repository's Clang frontend integration: linking left
`__enzyme_autodiff` and `__enzyme_fwddiff` unresolved. Installing the matching
headers and building `ClangEnzyme-18.so` restored both the native Enzyme flag and
the exact-Jacobian capability. This was a build failure, not evidence of a
numerical adjoint failure. No production source changes were needed.

## N7: Taylor studies

The study compares the absolute remainder
`abs(J(b + h*d) - J(b) - h*dot(g, d))` with step size. A slope near two must
persist over a contiguous range above the measured reference noise floor.
A slope from a single selected pair or an identically zero remainder is not
evidence for second-order accuracy.

Prescribed-iota cases (`ncurr=0`) have a boundary-independent iota objective.
They require a separate zero-derivative check, not a quadratic Taylor slope.
Treating their zero remainder as a failed adjoint test would confuse an
analytically constant objective with insufficient reference convergence.

Numerical results are reported below when the Enzyme-backed study completes.
An unavailable exact derivative is a blocked study, not a passing test.

The initial native-reference check reached maximum squared/normalized force
residuals of `9.81e-17` (small CTH), `9.82e-17` (CTH), and `9.48e-17`
(li383). Restarting the identical converged state changed it by at most
`5.56e-17`, but can immediately accept the same state and therefore does not
measure reference accuracy. The study instead estimates its noise floor by
solving a nearby boundary independently and restarting back to the base.

The nearby-boundary return gives the following differences despite both solves
meeting `ftol=1e-16`:

| Case | Max state difference | Iota difference | Well difference | QS difference | Random-linear difference |
| --- | --- | --- | --- | --- | --- |
| Small CTH | 3.220e-7 | 1.034e-8 | 4.517e-5 | 7.406e-8 | 1.091e-6 |
| CTH | 1.514e-6 | 6.430e-9 | 4.589e-5 | 8.907e-7 | 1.854e-5 |
| li383 | 3.512e-6 | 5.160e-9 | 4.408e-6 | 4.521e-9 | 3.314e-6 |

Volume, aspect, and the boundary objective were unchanged in this probe. This
does not establish accuracy for objectives depending on the interior state.
For example, CTH's measured random-linear value uncertainty alone gives a
central-difference uncertainty scale of `1.85e-2` at a step of `1e-3`.
The interim reference therefore cannot certify every requested component to
`1e-6` relative accuracy. WP5's all-case Newton reference remains a scientific
dependency, not merely a performance improvement. The raw measurements and
input provenance are in `plots_wp6/native_reference_noise.json`.

### Small-case measured Taylor subset

Using eleven steps from `1e-2` to `1e-7`, volume and the boundary objective
pass for m=1, highest-m, and highest-|n| directions. Their six fitted slopes
range from `1.9999805` to `2.0000361`, with ten or eleven usable points.

The random-linear objective is not certified:

| Direction | Fitted slope | Usable points | Smallest absolute AD/FD difference | Status |
| --- | --- | --- | --- | --- |
| m=1 | 2.021947 | 3 | 2.566e-4 | Noise floor |
| Highest m | 1.894059 | 4 | 2.859e-4 | Unresolved derivative |
| Highest absolute n | 2.002860 | 4 | 3.653e-4 | Unresolved derivative |

A nearly quadratic remainder is not sufficient when the directional derivative
cannot be resolved to the requested tolerance. These results do not isolate an
adjoint bug: the independent reference uncertainty already exceeds the target
at the tested derivative steps. They also do not excuse an incorrect gradient.
Repeat with WP5's validated reference before assigning the discrepancy to
reference convergence, structural masks, or relabeling dependence.

The saved directional data can be replayed without any equilibrium solves:
`scripts/plot_reference_step_study.py --data-dir plots_wp6 --output-dir plots_wp6`.
The resulting `plot_02_fd_step_study_small_random_linear` PNG/NPZ pair compares
the measured absolute AD/FD discrepancy, the value-noise scale divided by the
FD step, and the unchanged derivative target. Each source NPZ's hash and
original provenance are retained. Decreasing the FD step does not monotonically
improve this comparison: reference noise is amplified by division by the step.
Replay rejects mixed source provenance, including files left over from a
different seed or input configuration in the same output directory.

## Component finite differences

The component study retains the dense signed-toroidal-mode boundary layout,
including redundant parameter slots. Genuinely inactive coefficients and
near-zero derivatives are reported separately; relative errors alone are not
meaningful near zero. Agreement with one finite-difference step does not
establish the requested `1e-6` relative acceptance.

The absolute tolerance applies only when both derivatives are near zero.
Adding it to every relative comparison can accept a `5e-5` relative error
for a derivative near `1e-4`; that is fifty times the requested tolerance.
Redundant negative-n slots at m=0 are scanned, not assigned zero derivatives:
redundancy does not imply that an individual parameter has no effect.

### Timeout checkpoint

The interrupted run committed its small-case Taylor plots and reference-noise
measurements. It did not commit any component results:
`plots_wp6/wp6_components.log` is empty. An empty log is not a numerical failure
or evidence that any signed boundary component has passed.

The restored sandbox initially had `VMECPP_ENABLE_ENZYME=OFF`. Saved NPZ data
remain evidence from the recorded Enzyme binary, but new exact-gradient
measurements require restoring and verifying an Enzyme-enabled runtime.
No missing measurements are inferred from the saved Taylor slopes.
The continuation rebuilt `ClangEnzyme-18.so` from Enzyme v0.0.264 with
LLVM/Clang 18 and restored the editable extension without dependency or
production-source changes. Both `_vmecpp.VMECPP_ENABLE_ENZYME` and
`vmecpp.has_exact_force_jacobian()` then returned `True`.

### Durable component scans

The component study now writes `components_CASE.checkpoint.npz` after each
complete plus/minus pair, using an atomic replacement and file/directory
synchronization. An interruption during a pair repeats that pair on resume.
Completed failed native solves remain failures, not missing work or passes.

Run `scripts/plot_reference_gradients.py` with
`--cases cth_like_fixed_bdy li383_low_res --objectives volume boundary random_linear
--studies components --hot-restart --output-dir plots_wp6`, and add `--resume`
to continue existing checkpoints without rerunning Taylor studies. The initial
run also writes an empty checkpoint before its first pair.
Starting a fresh scan replaces an existing checkpoint; continuing it requires
`--resume`. Use only one writer per case/checkpoint path.

Resume validates checksums, steps, ordered objectives and random weights,
input, base state, exact gradients, reference noise, and runtime/helper hashes.
It rejects incompatible or corrupt data instead of combining runs. Exact
floating-point comparisons are conservative: a changed binary or even a
non-bitwise-reproducible reference solve requires a new scan, not silently
accepted reuse. Native resume reproducibility must be checked per case.
Analytic-oracle interruption/resume and rejection tests passed (18 tests);
saved-data replay and mixed-provenance rejection passed (3 tests).
The replay uses the saved per-step derivative acceptance thresholds rather
than inferring a different near-zero rule from the AD derivative alone.

### li383 measured signed-mode component scan

The continuation completed all 168 central-difference pairs: 56 signed
`rbc`/`zbs` slots at steps `1e-3`, `3e-4`, and `1e-4`, including redundant and
inactive slots. All 336 perturbed native solves converged; the checkpoint
contains 338 records including the two reference records.

| Objective | Passing slots | Noise-limited slots | Largest relative error on resolved nonzero samples |
| --- | --- | --- | --- |
| Volume | 56/56 | 0 | 5.533e-9 |
| Boundary | 56/56 | 0 | 7.418e-9 |
| Random linear | 1/56 | 55 | Not resolved |

Counts include near-zero and inactive-slot checks. Maxima use every resolved
nonzero sample at all three steps, not an error-minimizing step choice.
The random-linear value noise is `3.314114e-6`, giving derivative-noise scales
from `3.314e-3` to `3.314e-2`; none of its nonzero component derivatives is
resolved at the required target. This is an unresolved reference comparison,
not demonstrated agreement or a localized adjoint bug. The passing volume
and boundary checks do not certify interior-sensitive objectives.

The complete data and fixed-middle-step plots are saved as
`plot_03_components_li383_low_res_{volume,boundary,random_linear}` PNG/NPZ
pairs. A fresh native `--resume` accepted the exact provenance, with completed
pairs unchanged at 168 and records unchanged at 338.

### CTH measured signed-mode component scan

A later bounded continuation completed all 270 pairs: 90 signed slots at the
same three steps. The complete checkpoint contains 542 records with no failed
native solves. Native `--resume` accepted the exact reference provenance and
continued the partial scan without discarding its completed pairs.

| Objective | Passing slots | Noise-limited slots | Largest relative error on resolved nonzero samples |
| --- | --- | --- | --- |
| Volume | 88/90 | 2 | 1.841e-8 |
| Boundary | 83/90 | 7 | 1.161e-8 |
| Random linear | 1/90 | 89 | Not resolved |

Volume's unresolved slots are `rbc(m=0,n=+/-4)`. Their AD derivatives are
`8.809697e-6`, and measured relative errors are below `6.877e-8`, but only the
largest step resolves the derivative relative to the roundoff-noise estimate.
The boundary objective's seven unresolved slots are
`rbc(2,-4)`, `rbc(3,-4)`, `rbc(3,-3)`, `zbs(2,-4)`, `zbs(3,-1)`,
`zbs(4,-4)`, and `zbs(4,-3)`. Their measured relative errors are below
`1.401e-7`, but none has two adjacent resolved steps. For example,
`zbs(3,-1)` has a derivative of `-2.972212e-6`: its relative target is
`2.972e-12`, below the `3.553e-12` derivative-noise estimate even at `h=1e-3`.

These volume/boundary labels are conservative roundoff-resolution failures,
not evidence of errors exceeding the relative target. A larger-step study
could resolve these small coefficients; decreasing the FD step makes the
reference-noise comparison worse. No acceptance criterion is weakened here.
For random linear, the measured value noise is `1.853807e-5`, and all 89
nonzero slots remain unresolved.

Complete data and plots are saved as
`plot_03_components_cth_like_fixed_bdy_{volume,boundary,random_linear}` PNG/NPZ
pairs, alongside `components_cth_like_fixed_bdy.checkpoint.npz`.
Fresh invocations without `--resume` replaced earlier progress during
execution; duplicate writers were stopped before serial continuation.
Future runs must be serialized per case and explicitly resumed.

## N10: one-shot PR cross-check

Blocked: no one-shot PR branch or converged design state was supplied, and WP4's
slice helper is not implemented. Repository PR searches for "one-shot" and
"one shot optimization" did not identify the requested design-optimization
branch. No branch is substituted or checked out, and no agreement is claimed.

## Handoffs and conclusions

The results in this package use an interim reference and must be repeated with
WP5's validated reference equilibria before production integration. Numerical
failures, if observed, must be recorded with their steps, directions, objective,
force residuals, and reference method rather than hidden by weakening the
acceptance tolerance.

The continuation's focused checks passed: 18 checkpoint tests and 3 saved-data
replay tests. Formatting hooks passed. The isolated default Pyright hook cannot
resolve this sandbox's globally installed dependencies; the same checker with
the explicit Python interpreter reported zero errors. CodeQL reported zero
Python alerts. The automated review executable was unavailable; a read-only
review identified mixed-source provenance and a reconstructed plotting
threshold, both corrected with regression coverage.

# Boundary optimization under force balance

`vmecpp.joint` optimizes a fixed boundary and its VMEC state without converging VMEC between steps. Hessian products of the force Jacobian are exact in an Enzyme build (`vmecpp.has_exact_force_jacobian()`) and central differences of the force otherwise.

```python
import jax

from vmecpp.joint import QuasisymmetryProblem, optimize

jax.config.update("jax_enable_x64", True)

problem = QuasisymmetryProblem(vmec_input, aspect=6.0, mean_iota=0.42, max_mode=3)
result = optimize(problem, stages=(1, 2, 3))
```

`examples/joint_optimization.py` runs a problem by both methods and checks each final boundary with a cold run read by SIMSOPT.

| Problem | Options |
|---|---|
| Quasi-axisymmetry, `ns` 25 | none |
| Quasi-axisymmetry, `ns` 51 | `--ns 51 --mpol 7 --stages 1 2 3 4 5` |
| Quasi-helical symmetry | add `--nfp 4 --helicity-n 1 --aspect 8 --iota 1.2` |
| Finite pressure | add `--pressure 10000` |
| Stages on fewer modes | add `--margin 2` |

## Method

Each Levenberg-Marquardt step solves

```
min |r + J_x dx + J_b db|^2 + damping |db|^2   subject to   F + H dx + F_b db = 0
```

for the state `x` and the boundary coefficients `b`, where `r` are the residuals, `F` is the force on the interior of the state and `H` is its Jacobian. Eliminating `dx`:

```
dx = dn + T db,   dn = -H^-1 F,   T = -H^-1 F_b
r_b = r + J_x dn,   J = J_x T + J_b
(J^T J + damping D) db = -J^T r_b
```

The boundary moves by `db` and the state by `dx`. `dn` is limited to 5% of the major radius `R`, in the step and in `r_b`. VMEC's descent then relaxes the state to the force tolerance, which takes no iterations when the step already meets it. A trial step on which VMEC raises is repeated without `dn`.

The merit is `|r_b|^2`. Within the limit its error is of second order in the distance from force balance. The error is estimated as `2 |r_b| |J_x dn| |dn| / R` for the whole of `dn`. A state is relaxed until this is below `error_fraction` times the larger of the predicted drop and 1% of the merit, between `NESTED_TOLERANCE` and `LOOSEST_TOLERANCE`. A trial step is refused if `|r|^2` rises by more than `abs(|r|^2 - |r_b|^2)` plus a quarter of the predicted drop, and has `H` assembled at its state at once if `|r|^2` lies below the merit by more than half the predicted drop. Otherwise its merit takes `dn` from GMRES on Hessian products at the trial state, preconditioned by the factors of the last point, until the preconditioned residual is 1% of `dn` or for at most 64 iterations, and `H` is assembled only if this merit falls. The merit on the trial state's own factors decides the step. A trial step refused on its merit tightens the force tolerance by a decade for the rest of the stage. The last stage starts from a state restored to force balance and relaxes every step to `NESTED_TOLERANCE`.

`nested=True` converges each step to `NESTED_TOLERANCE` with the merit `|r|^2`. A stage ends after three steps in a row with a drop below 1%. A step whose gain exceeds 0.5 at a damping above its value at the start of the stage does not count, and the damping then falls ten times faster.

`margin` runs every stage but the last on a state whose `mpol` and `ntor` are that many above the highest mode of its boundary, and the last stage on the input's. Between stages the state takes zeros in the modes it gains and is relaxed on them. The residuals are those of the input's modes, with zeros in the modes a state lacks.

## Test problem

Quasi-axisymmetry after Landreman and Paul (PRL 2022).

| Choice | Value |
|---|---|
| Field periods | 2 |
| Helicity | (1, 0) |
| Objective | SIMSOPT's `QuasisymmetryRatioResidual` on s = 0.1, 0.2, ..., 1.0 |
| Targets | aspect ratio 6, mean iota 0.42, as residuals of unit weight |
| Profiles | no pressure, no net current (`ncurr` 1, `curtor` 0) |
| Flux | `phiedge` 0.08 |
| Start | `rbc(0,0)` 1, `rbc(1,0)` = `zbs(1,0)` = 0.16, `rbc(1,1)` = -`zbs(1,1)` = 0.05 |
| Free modes | `rbc`, `zbs` with `m` from 0 to `max_mode` and `abs(n)` up to `max_mode`, except `m` = 0 with `n` <= 0 |
| Bounds | none |

## Other objectives

A subclass of `JointProblem` defines `residuals`, differentiable by JAX:

```python
import jax.numpy as jnp

from vmecpp.joint import JointProblem


class MeanIota(JointProblem):
    def residuals(self, solved, boundary):
        mean_iota = (solved.poloidal_flux[-1] - solved.poloidal_flux[0]) / (
            solved.toroidal_flux[-1] - solved.toroidal_flux[0]
        )
        return jnp.atleast_1d(mean_iota - 0.3)
```

## Constraints and bounds

A subclass that defines `constraints` keeps each of their values between `lower` and `upper`. `boundary_lower` and `boundary_upper` bound the boundary coefficients, as arrays of `rbc` and then `zbs` of shape `(2, mpol, 2 ntor + 1)`:

```python
class QuasisymmetryAboveIota(QuasisymmetryProblem):
    def constraints(self, solved, boundary):
        mean_iota = (solved.poloidal_flux[-1] - solved.poloidal_flux[0]) / (
            solved.toroidal_flux[-1] - solved.toroidal_flux[0]
        )
        return jnp.atleast_1d(mean_iota)


problem = QuasisymmetryAboveIota(vmec_input, lower=0.45, max_mode=3)
result = optimize(problem, stages=(1, 2, 3))
result.evaluation.constraints, result.multipliers
```

A constraint `c` joins the residuals as `weight (c + shift - z)`, with `z` the value within its bounds closest to `c + shift`, so it adds nothing while `c + shift` lies within them. Each step solves its least-squares problem for `db` within the bounds of the coefficients and for every `z` within the bounds of its constraint, by SciPy's bounded-variable least squares, so the linearized problem meets a bound before a step crosses it.

The shifts start at zero. Unless `weights` gives the weights, each stage starts every constraint at the weight for which it adds ten times the curvature of the residuals along its own gradient, `weight^2 a (J^T J + damping D)^-1 a^T = 10` with `a` the derivatives of the constraint, so neither the unit of a constraint nor that of the residuals matters. The stages before the last hold the constraints with their weights alone. When the last stage has stalled, each shift moves to `c + shift - z`, which is the multiplier update of the augmented Lagrangian, and the stage goes on until no shift would move by more than `constraint_tolerance`, 1e-4 by default, times the size of the bounds of its constraint. A constraint outside that tolerance takes ten times its weight at the same multiplier when the update before has not brought it to a quarter of its distance from its bounds, or when its weighted distance is smaller than the residuals, so that a step does not resolve it. The shifts move at most eight times, and not in a last stage that ends at its step limit. A stage counts as stalled by the drop of the merit relative to the merit without its shifts.

`Result.multipliers` holds `2 weight^2 shift` for each constraint, its Lagrange multiplier for `|r|^2` as of the last update: positive at an upper bound, negative at a lower one and zero between them. In the tests it lies within 10% of the multiplier that balances the gradient of the residuals.

## Quasi-isodynamic fields

`examples/joint_quasi_isodynamic.py` defines `QuasiIsodynamicProblem`, a subclass of `QuasisymmetryProblem`, runs it by both methods and checks each final boundary with a cold run on more modes.

| Choice | Value |
|---|---|
| Field periods | 3 |
| Start | the boundary of ConStellaration's baseline: its near-axis construction for aspect ratio 10, elongation 5, iota 0.75 and mirror ratio 0.25, on modes up to 1 |
| Objective | the residuals of quasi-isodynamicity on the outermost half-grid surface |
| Targets | aspect ratio 10, mean iota 0.82, as residuals of unit weight |
| Limits | mirror ratio at the edge 0.2 less a reserve of 3%, elongation of each cross-section 5, as constraints |
| Profiles | no pressure, no net current |
| Free modes | as in the test problem, in stages 1 to 4 |

The aspect ratio and the limits are those of ConStellaration's `SimpleToBuildQIStellarator` problem, and the mean iota puts iota at the edge above its 0.25 per field period.

| Problem | Options |
|---|---|
| `ns` 25, `mpol` 5 | none |
| More modes in the state | `--mpol 6` |
| Other targets and limits | `--aspect`, `--iota`, `--mirror`, `--elongation`, `--reserve` |
| Modes of the cold run | `--check-mpol 10` |

The field strength of the surface is transformed to Boozer coordinates as booz_xform does and followed along 25 field lines, each over one field period from the toroidal angle of the maximum, scaled to span 0 to 1 over all of them. A quasi-isodynamic field lies below each level on the same fraction of every field line, and has one well per line, its maximum at both ends of each line and its minimum on each line. The residuals are, for 101 levels, the fraction of each line below the level less its mean over the lines, then for each line its variation beyond that of one well, the distance of its ends from the maximum and that of its lowest point from the minimum. They are continuous in the field strength, with kinks.

The mirror ratio is `(max - min) / (max + min)` of the field strength extrapolated to the edge, and the elongation is that of the cross-sections in planes perpendicular to the magnetic axis, the ratio of the semi-axes of the ellipse with the perimeter and the area of each, both as ConStellaration defines them. The constraints are the mirror ratio and the elongation of each of the 23 cross-sections of a field period. The derivative of the mirror ratio is that of the same ratio with each extreme as a mean weighted by `exp(+-200 value / mean)`, which follows an extreme that several grid points share. The mirror ratio stays below its limit less `reserve`, since a run on more modes and surfaces lies above this one.

## Result

`Result` holds the final boundary, its state at `NESTED_TOLERANCE` with the residuals, the values of the constraints and the force residual there, the accepted steps, one `Stage` per stage and the multipliers of the constraints.

| `Stage` field | Meaning |
|---|---|
| `stopped` | `"stalled"`: three steps below a drop of 1% that the damping did not limit. `"no_step"`: no trial step predicted a drop, or none held, also from the state relaxed to `NESTED_TOLERANCE`. `"max_steps"`: step limit. |
| `rejected_steps` | Trial steps refused for their merit or force residual |
| `failed_solves` | Trial steps on which VMEC raised |
| `tightened` | Further relaxations for the accuracy of a merit |
| `shift_updates` | Updates of the shifts of the constraints |
| `mpol`, `ntor` | Modes of the state the stage ran on |

Residuals that are not finite at the first state raise a `ValueError`.

## Limitations

- One `H` takes `3 * 6 * mpol * (ntor + 1)` Hessian products, about 1000 force evaluations at `ns` 51, `mpol` 7, or 2000 with differenced products. `workers` runs them on one model per thread.
- Slow modes next to the axis persist in VMEC's descent after the force tolerance is met. At `NESTED_TOLERANCE`, `|r|^2` of the nested baseline can differ from a cold run by 10%.
- `dn` converges from states along a run. Far from force balance, such as the rotating ellipse with pressure, it does not, and VMEC's descent starts the run.
- Where `H` is nearly singular, as at a start beyond an equilibrium limit, `dn` lies far outside its limit and the merit stays close to `|r|^2` until a step moves the boundary away.
- The final merit depends on the path. Quasi-axisymmetry at `ns` 25 ends near 6.4e-6 or 8.4e-6 by either method.
- `always_fix_m1_gauge` is on. Quasisymmetry from a free-gauge run depends on where the iteration froze the gauge, and differs by more than a factor of ten at low error.
- Quasisymmetry residuals are not finite on a surface where the Jacobian changes sign. The rotating ellipse reaches this on its boundary between beta 0.66% and 1.1%.
- Differenced products use a relative step of 3e-9, where truncation and roundoff balance along the soft modes. Against exact products at `ns` 51, the state tangents agree to 4e-6 and the residual derivatives to 1.5e-4.
- Fixed boundary and stellarator symmetry only. A multigrid input runs on the last entry of `ns_array`.
- A constraint holds on the modes and surfaces of its problem. A cold run of `examples/joint_quasi_isodynamic.py` on more modes has a mirror ratio up to 1.6% above that of the run, which the reserve of the example covers. Without its limits the example goes towards a field strength that depends on the toroidal angle alone, at a mirror ratio of 0.91 and an elongation of 16.6 from its start.
- A step takes the constraints through their first derivatives, so its steps along a curved bound are short. A weight far above the balanced one can stall a stage there: with unit weights, at which the elongation of the quasi-isodynamic example is over a thousand times stiffer than its residuals, three of four nested runs end with residuals thirty to a hundred times those of the fourth.
- Quasi-isodynamic runs depend on the path as well. With a mirror limit of 0.25 and a mean iota of 0.8 they end in one of two minima, the higher above the mirror limit.

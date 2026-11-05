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

The boundary moves by `db` and the state by `dx`, with `dn` limited to 5% of the major radius `R`. VMEC's descent then relaxes the state to the force tolerance, which takes no iterations when the step already meets it. A trial step on which VMEC raises is repeated without `dn`.

The merit is `|r_b|^2`, with an error of second order in the distance from force balance. The error is estimated as `2 |r_b| |r_b - r| |dn| / R`. A state is relaxed until this is below `error_fraction` times the larger of the predicted drop and 1% of the merit, between `NESTED_TOLERANCE` and `LOOSEST_TOLERANCE`. A trial step is refused before `H` is assembled if `|r|^2` rises by more than `abs(|r|^2 - |r_b|^2)` plus a quarter of the predicted drop. The last state is relaxed to `NESTED_TOLERANCE`.

`nested=True` converges each step to `NESTED_TOLERANCE` with the merit `|r|^2`. A stage ends after three steps in a row with a drop below 1%.

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
class MeanIota(JointProblem):
    def residuals(self, solved, boundary):
        mean_iota = (solved.poloidal_flux[-1] - solved.poloidal_flux[0]) / (
            solved.toroidal_flux[-1] - solved.toroidal_flux[0]
        )
        return jnp.atleast_1d(mean_iota - 0.3)
```

## Result

`Result` holds the final boundary, its state at `NESTED_TOLERANCE` with the residuals and force residual there, the accepted steps and one `Stage` per stage.

| `Stage` field | Meaning |
|---|---|
| `stopped` | `"stalled"`: three steps below a drop of 1%. `"no_step"`: no trial step held, also from the state relaxed to `NESTED_TOLERANCE`. `"max_steps"`: step limit. |
| `rejected_steps` | Trial steps refused for their merit or force residual |
| `failed_solves` | Trial steps on which VMEC raised |
| `tightened` | Further relaxations for the accuracy of a merit |

Residuals that are not finite at the first state raise a `ValueError`.

## Limitations

- One `H` takes `3 * 6 * mpol * (ntor + 1)` Hessian products, about 2000 force evaluations at `ns` 51, `mpol` 7. `workers` runs them on one model per thread.
- A trial step refused after the screen costs one `H`.
- Slow modes next to the axis persist in VMEC's descent after the force tolerance is met. At `NESTED_TOLERANCE`, `|r|^2` of the nested baseline can differ from a cold run by 10%.
- `dn` converges from states along a run. Far from force balance, such as the rotating ellipse with pressure, it does not, and VMEC's descent starts the run.
- The final merit depends on the path. Quasi-axisymmetry at `ns` 25 ends near 6.4e-6 or 7.9e-6 by either method.
- `always_fix_m1_gauge` is on. Quasisymmetry from a free-gauge run depends on where the iteration froze the gauge, and differs by more than a factor of ten at low error.
- Quasisymmetry residuals are not finite on a surface where the Jacobian changes sign. The rotating ellipse reaches this on its boundary between beta 0.66% and 1.1%.
- Differenced products use a relative step of 3e-9, where truncation and roundoff balance along the soft modes. Against exact products at `ns` 51, the state tangents agree to 4e-6 and the residual derivatives to 1.5e-4.
- Fixed boundary and stellarator symmetry only. A multigrid input runs on the last entry of `ns_array`.

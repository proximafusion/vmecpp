# Boundary optimization without converging every step

Each step of `vmecpp.joint` moves a fixed boundary and carries the VMEC state along with it, and VMEC's own descent then relaxes the state only as far as that step needs. It runs in any build. In a build with Enzyme, `vmecpp.has_exact_force_jacobian()`, the Hessian products of the force Jacobian are exact; otherwise they are central differences of the force, at the same cost.

```python
import jax

from vmecpp.joint import QuasisymmetryProblem, optimize

jax.config.update("jax_enable_x64", True)

problem = QuasisymmetryProblem(vmec_input, aspect=6.0, mean_iota=0.42, max_mode=3)
result = optimize(problem, stages=(1, 2, 3))
```

`python examples/joint_optimization.py` runs the test problem below by this method and by the nested baseline, and checks each final boundary with a cold, fully converged run read by SIMSOPT. `workers` sets the threads for the force Jacobian and the residual derivatives, by default one per processor. A multigrid input runs on the last entry of its `ns_array`.

## The step

Each step of Levenberg-Marquardt on the boundary coefficients:

1. factors the force Jacobian on the interior of the state,
2. solves for the state tangent of every free boundary mode,
3. differentiates the residuals along those tangents,
4. takes the step in the boundary and moves the state along its tangent,
5. relaxes the state by VMEC's own descent only as far as the step needs.

The error of the merit grows with the square root of the force residual, with a coefficient that differs by orders of magnitude between problems and along a run. The driver measures that coefficient `K` by relaxing the state ten times further after each step, and relaxes each trial step to the force tolerance `(noise_fraction * predicted_drop / K)^2`, between `NESTED_TOLERANCE` and the smaller of `LOOSEST_TOLERANCE` and `MERIT_TOLERANCE` times the merit. Before a trial step that needs a tighter tolerance than the current state has, it relaxes the current state too, so that both sides of the comparison of merits have the accuracy of the step. `noise_fraction=0` converges every step to `NESTED_TOLERANCE`, which is the nested baseline with the same derivatives.

A stage ends after three steps in a row that lower the merit by less than 1%.

## The test problem

The example is the quasi-axisymmetric problem of Landreman and Paul (PRL 2022).

| Choice | Value |
|---|---|
| Field periods | 2 |
| Helicity | (1, 0) |
| Objective | SIMSOPT's `QuasisymmetryRatioResidual` on s = 0.1, 0.2, ..., 1.0 |
| Targets | aspect ratio 6, mean iota 0.42 |
| Profiles | no pressure, no net current (`ncurr` 1, `curtor` 0) |
| Flux | `phiedge` 0.08 |
| Start | rotating ellipse: `rbc(0,0)` 1, `rbc(1,0)` = `zbs(1,0)` = 0.16, `rbc(1,1)` = -`zbs(1,1)` = 0.05 |
| Free modes | `rbc`, `zbs` with `m` from 0 to `max_mode` and `abs(n)` up to `max_mode`, except `m` = 0 with `n` <= 0 |

The coefficients carry no bounds. The aspect ratio and the mean iota enter as residuals of unit weight, so they are met to the accuracy of the least-squares solution, and the example names any that the cold run misses by more than 1e-3.

## Other objectives

A subclass of `JointProblem` defines `residuals`, a function of the geometry and the boundary that JAX can differentiate, and runs on the same driver:

```python
class MeanIota(JointProblem):
    def residuals(self, solved, boundary):
        mean_iota = (solved.poloidal_flux[-1] - solved.poloidal_flux[0]) / (
            solved.toroidal_flux[-1] - solved.toroidal_flux[0]
        )
        return jnp.atleast_1d(mean_iota - 0.3)
```

## How a run ends

`Result.stages` records how each stage of boundary modes ended. `stopped` is `"stalled"` after three steps in a row below a drop of 1%, `"no_step"` when twelve trial steps in a row fail, and `"max_steps"` at the step limit. `rejected_steps` counts the trial steps refused for their merit or force residual, `failed_solves` those on which VMEC raised, and `tightened` the times the current state was relaxed further before a step. Residuals that are not finite at the state of a step raise a `ValueError`.

## Numerical limitations

- One force Jacobian takes `3 * 6 * mpol * (ntor + 1)` Hessian products, as much work as about 2000 force evaluations at `ns` 51 and `mpol` 7, so a step costs far more than its relaxation. The products are independent, and `workers` runs them on one thread per model.
- Slow modes next to the axis move the quasisymmetry residuals until the force residual is near 1e-10, and below a merit of 1e-6 further still, so the last steps of a run need a tight tolerance and save few force evaluations.
- The final merit depends on the path through local minima: quasi-axisymmetry at `ns` 25 ends near 6.4e-6 or 7.9e-6, by either method, depending on small changes of the settings.
- The state is solved with `always_fix_m1_gauge`. Quasisymmetry computed from a run with a free gauge depends on where the iteration froze the gauge, and at low error the two differ by more than a factor of ten. Compare with a pinned run, and state the gauge of any independent check.
- The quasisymmetry measure takes the square root of `sqrt(g)`, so its residuals are not finite on a surface where the Jacobian changes sign. The rotating ellipse reaches that point on its boundary between beta 0.66% and 1.1%. A trial step without a finite merit is refused, and a run that starts there raises.
- Differenced products lose accuracy along the soft modes. With pressure, a derivative of the mean iota can be off by 20%, against 1e-4 of the norm of the residual derivatives with exact products. The descent still converges, because the relaxation corrects the state after each step.
- Only fixed-boundary, stellarator-symmetric inputs are supported.

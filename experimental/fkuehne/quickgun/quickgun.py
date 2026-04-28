"""JAX-compatible quickgun (pellet injection) physics model for stellarators.

Models a frozen hydrogen pellet fired by a quickgun along a chord through
the stellarator plasma using the Neutral Gas Shielding (NGS) ablation model
(Parks & Turnbull, Phys. Fluids 21, 1735, 1978).

Design principles
-----------------
* All computation is JAX-compatible and fully differentiable w.r.t. every
  physical parameter.
* ODE integration uses ``diffrax.Tsit5`` with an adaptive (PID) step
  controller.
* Conditional branches (pellet alive vs. fully ablated) use
  ``jax.lax.cond`` so the computation graph stays differentiable.
* Results and parameter objects are stored as *dapper data*: plain Python
  ``dataclasses`` that are registered as JAX pytrees via
  ``@pytree.register_dapper_data``.  This makes every result leaf
  directly accessible to ``jax.grad`` / ``jax.vmap`` / ``jax.jit``.
* Parameter optimisation (find the injection velocity that delivers the
  pellet to a target deposition depth) is done with
  ``optimistix.LevenbergMarquardt``, which internally differentiates the
  simulation, confirming end-to-end differentiability.

Physical model
--------------
The pellet travels along a straight chord through the plasma.  Taking the
chord entry point as x = 0, the plasma minor radius at chord position x is

    r(x) = |x - a|

where a = ``plasma.minor_radius`` is the minor radius of the plasma.
Parabolic density and temperature profiles are assumed:

    n_e(r) = n_e0 * max(0, 1 - (r/a)^2)
    T_e(r) = T_e0 * max(0, 1 - (r/a)^2)

The NGS ablation rate is

    dm/dt = -C * n_e^{1/3} * T_e^{5/3} * r_p^2

where r_p is the current pellet radius, and C is an empirical coefficient.
Velocity is held constant (the rocket and drag forces are small compared
with the initial momentum for typical stellarator pellet parameters).

Usage
-----
Run as a script to see the forward simulation, gradient check, and
optimisation demo::

    python quickgun.py

or import and call ``simulate`` / ``find_velocity_for_target_depth``
directly.
"""

from __future__ import annotations

import dataclasses

import diffrax
import jax
import jax.numpy as jnp
import optimistix as optx

# Enable 64-bit floating-point arithmetic for physical accuracy.
# This must be set before any JAX computation.
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# pytree registration: @pytree.register_dapper_data
# ---------------------------------------------------------------------------


class _PytreeRegistry:
    """Namespace that provides the ``register_dapper_data`` decorator.

    Usage::

        @pytree.register_dapper_data
        @dataclasses.dataclass
        class MyData:
            x: float
            y: float

    After decoration ``MyData`` is a valid JAX pytree: its instances can be
    passed through ``jax.jit``, ``jax.grad``, ``jax.vmap``, etc.
    """

    @staticmethod
    def register_dapper_data(nodetype: type) -> type:
        """Register a ``@dataclasses.dataclass`` as a JAX pytree.

        All dataclass fields become pytree leaves (differentiable data).
        The class structure (field names) is the static treedef.

        Args:
            nodetype: A class that has already been decorated with
                ``@dataclasses.dataclass``.

        Returns:
            The same class, now registered in JAX's pytree registry.

        Raises:
            TypeError: If ``nodetype`` is not a dataclass.
        """
        if not dataclasses.is_dataclass(nodetype):
            msg = (
                f"{nodetype.__name__} must be decorated with @dataclasses.dataclass"
                " before @pytree.register_dapper_data"
            )
            raise TypeError(msg)
        return jax.tree_util.register_dataclass(nodetype)


pytree = _PytreeRegistry()


# ---------------------------------------------------------------------------
# Data structures (dapper pytrees)
# ---------------------------------------------------------------------------


@pytree.register_dapper_data
@dataclasses.dataclass
class PelletParams:
    """Physical parameters of the pellet and quickgun injector.

    All fields are JAX-differentiable pytree leaves.

    Attributes:
        initial_mass: Initial pellet mass [kg].
        initial_velocity: Injection velocity [m/s] (positive = inward).
        ablation_coeff: NGS ablation coefficient C in
            ``dm/dt = -C * n_e^{1/3} * T_e^{5/3} * r_p^2`` [SI].
        pellet_density: Density of the frozen hydrogen pellet [kg/m^3].
    """

    initial_mass: float
    initial_velocity: float
    ablation_coeff: float
    pellet_density: float


@pytree.register_dapper_data
@dataclasses.dataclass
class PlasmaProfile:
    """Parabolic plasma profiles for the stellarator.

    Attributes:
        n_e0: Peak electron density [m^{-3}].
        T_e0: Peak electron temperature [eV].
        minor_radius: Plasma minor radius [m].
    """

    n_e0: float
    T_e0: float
    minor_radius: float


@pytree.register_dapper_data
@dataclasses.dataclass
class SimulationResult:
    """Time-resolved trajectory and summary of a pellet injection event.

    All array fields are JAX arrays and support differentiation through
    the simulation.

    Attributes:
        times: Saved time points [s], shape ``(n_save,)``.
        masses: Pellet mass at each saved time [kg], shape ``(n_save,)``.
        positions: Pellet position along the chord [m], shape ``(n_save,)``.
        velocities: Pellet velocity [m/s], shape ``(n_save,)``.
        final_mass: Pellet mass at ``t_max`` [kg], scalar.
        final_position: Pellet position at ``t_max`` [m], scalar.
        mean_deposition_depth: Mass-weighted mean position at which the
            pellet material is deposited [m], scalar.  This is the primary
            figure of merit used in the optimisation demo.
    """

    times: jax.Array
    masses: jax.Array
    positions: jax.Array
    velocities: jax.Array
    final_mass: jax.Array
    final_position: jax.Array
    mean_deposition_depth: jax.Array


# ---------------------------------------------------------------------------
# Physics helper functions
# ---------------------------------------------------------------------------


def pellet_radius(mass: jax.Array, pellet_density: jax.Array) -> jax.Array:
    """Radius of a spherical pellet from its mass.

    r_p = (3 m / (4 pi rho))^{1/3}

    Args:
        mass: Pellet mass [kg].  Clamped to zero before taking the cube
            root so the function remains well-defined when mass < 0.
        pellet_density: Pellet material density [kg/m^3].

    Returns:
        Pellet radius [m].
    """
    return (3.0 * jnp.maximum(mass, 0.0) / (4.0 * jnp.pi * pellet_density)) ** (
        1.0 / 3.0
    )


def chord_minor_radius(position: jax.Array, plasma: PlasmaProfile) -> jax.Array:
    """Plasma minor radius at a given chord position.

    For a chord through the plasma centre:
        r(x) = |x - a|
    where a is the plasma minor radius and x is the position along the
    chord measured from the entry point.

    Args:
        position: Pellet position along the chord [m].
        plasma: Plasma profile parameters.

    Returns:
        Local minor radius [m].
    """
    return jnp.abs(position - plasma.minor_radius)


def n_e_profile(minor_r: jax.Array, plasma: PlasmaProfile) -> jax.Array:
    """Parabolic electron density: n_e(r) = n_e0 * max(0, 1 - (r/a)^2).

    Args:
        minor_r: Local minor radius [m].
        plasma: Plasma profile parameters.

    Returns:
        Electron density [m^{-3}].
    """
    return plasma.n_e0 * jnp.maximum(0.0, 1.0 - (minor_r / plasma.minor_radius) ** 2)


def T_e_profile(minor_r: jax.Array, plasma: PlasmaProfile) -> jax.Array:
    """Parabolic electron temperature: T_e(r) = T_e0 * max(0, 1 - (r/a)^2).

    Args:
        minor_r: Local minor radius [m].
        plasma: Plasma profile parameters.

    Returns:
        Electron temperature [eV].
    """
    return plasma.T_e0 * jnp.maximum(0.0, 1.0 - (minor_r / plasma.minor_radius) ** 2)


def ngs_ablation_rate(
    mass: jax.Array,
    position: jax.Array,
    params: PelletParams,
    plasma: PlasmaProfile,
) -> jax.Array:
    """Parks-Turnbull Neutral Gas Shielding (NGS) ablation rate.

    dm/dt = -C * n_e^{1/3} * T_e^{5/3} * r_p^2

    Reference: Parks & Turnbull, Phys. Fluids 21 (1978) 1735.

    The density and temperature are clamped to a small positive floor before
    the fractional power so that the gradient stays finite at the plasma edge
    where the profiles vanish (avoids 0^{-2/3} singularities during autodiff).

    Args:
        mass: Current pellet mass [kg].
        position: Pellet position along the chord [m].
        params: Pellet and injector parameters.
        plasma: Plasma profile parameters.

    Returns:
        Ablation rate dm/dt [kg/s] (non-positive).
    """
    _eps = 1e-30  # gradient regularisation floor
    r_minor = chord_minor_radius(position, plasma)
    r_p = pellet_radius(mass, params.pellet_density)
    n_e = jnp.maximum(n_e_profile(r_minor, plasma), _eps)
    T_e = jnp.maximum(T_e_profile(r_minor, plasma), _eps)
    return -(params.ablation_coeff * n_e ** (1.0 / 3.0) * T_e ** (5.0 / 3.0) * r_p**2)


# ---------------------------------------------------------------------------
# ODE vector field
# ---------------------------------------------------------------------------


def _pellet_ode(
    _t: jax.Array,
    state: tuple[jax.Array, jax.Array, jax.Array],
    args: tuple[PelletParams, PlasmaProfile],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Vector field for the pellet injection ODE.

    State: ``(mass [kg], position [m], velocity [m/s])``

    The velocity is held constant (rocket and drag forces are negligible for
    typical stellarator pellet parameters).  The ablation branch uses
    ``jax.lax.cond`` to keep the computation graph fully differentiable while
    correctly zeroing the ablation rate once the pellet is gone.

    Args:
        t: Current time [s] (not used; the ODE is autonomous).  Named ``_t``
            to silence the unused-argument linter warning.
        state: Current ``(mass, position, velocity)`` tuple.
        args: ``(PelletParams, PlasmaProfile)`` tuple.

    Returns:
        Time derivatives ``(d_mass, d_position, d_velocity)``.
    """
    mass, position, velocity = state
    params, plasma = args

    rate = ngs_ablation_rate(mass, position, params, plasma)

    # Branch: ablate only while the pellet still has mass.
    # jax.lax.cond evaluates both branches symbolically but selects
    # the result differentiably, so gradients flow through this branch.
    d_mass = jax.lax.cond(
        mass > 0.0,
        lambda: rate,
        lambda: jnp.zeros_like(rate),
    )

    d_position = velocity
    d_velocity = jnp.zeros_like(velocity)

    return d_mass, d_position, d_velocity


# ---------------------------------------------------------------------------
# Forward simulation
# ---------------------------------------------------------------------------


def simulate(
    params: PelletParams,
    plasma: PlasmaProfile,
    t_max: float = 2.0e-3,
    n_save: int = 500,
) -> SimulationResult:
    """Simulate a pellet injection event using diffrax Tsit5.

    The pellet enters the plasma from the edge (x = 0) and travels along a
    chord of length 2 * ``plasma.minor_radius`` at constant velocity.

    Args:
        params: Pellet and injector parameters.
        plasma: Plasma profile.
        t_max: Maximum simulation time [s].
        n_save: Number of uniformly-spaced time points to save.

    Returns:
        A :class:`SimulationResult` pytree containing the full trajectory
        and scalar summary statistics.
    """
    t0 = jnp.float64(0.0)
    t1 = jnp.float64(t_max)

    y0 = (
        jnp.float64(params.initial_mass),
        jnp.float64(0.0),  # pellet starts at the plasma edge (x = 0)
        jnp.float64(params.initial_velocity),
    )

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_pellet_ode),
        diffrax.Tsit5(),
        t0=t0,
        t1=t1,
        dt0=(t1 - t0) / n_save,
        y0=y0,
        args=(params, plasma),
        saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, n_save)),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        max_steps=16 * n_save,
    )

    masses = sol.ys[0]
    positions = sol.ys[1]
    velocities = sol.ys[2]

    # Mass-weighted mean deposition depth (differentiable).
    # dm[i] = mass lost in the i-th interval, approximated from the
    # saved mass values.  Positions are evaluated at the same grid.
    m_prev = jnp.concatenate([masses[:1], masses[:-1]])
    ablation = jnp.maximum(m_prev - masses, 0.0)
    total_ablated = jnp.sum(ablation) + 1e-30  # guard against division by zero
    mean_depth = jnp.sum(positions * ablation) / total_ablated

    return SimulationResult(
        times=sol.ts,
        masses=masses,
        positions=positions,
        velocities=velocities,
        final_mass=masses[-1],
        final_position=positions[-1],
        mean_deposition_depth=mean_depth,
    )


# ---------------------------------------------------------------------------
# Optimisation: find the injection velocity for a target deposition depth
# ---------------------------------------------------------------------------


def find_velocity_for_target_depth(
    target_depth: float,
    plasma: PlasmaProfile,
    base_params: PelletParams,
    v_initial: float | None = None,
    t_max: float = 2.0e-3,
) -> tuple[float, SimulationResult]:
    """Find the injection velocity that places the pellet at a target depth.

    Solves the nonlinear least-squares problem

        min_v  (mean_deposition_depth(v) - target_depth)^2

    using :class:`optimistix.LevenbergMarquardt`.  The Jacobian is computed
    with reverse-mode autodiff (``jac="bwd"``), because the simulation uses
    diffrax's :class:`~diffrax.RecursiveCheckpointAdjoint` (a ``custom_vjp``
    function) which is incompatible with forward-mode JVP.  Using
    ``jac="bwd"`` proves that the simulation is end-to-end
    reverse-differentiable.

    The mean deposition depth is monotone in the injection velocity only for
    velocities below ~500 m/s (where the pellet is fully consumed inside the
    plasma).  A sensible initial guess ``v_initial`` in this regime leads to
    reliable convergence in a handful of iterations.

    Args:
        target_depth: Desired mass-weighted mean deposition depth [m].
        plasma: Plasma profile.
        base_params: Base parameters; only ``initial_velocity`` is
            optimised.  All other fields are held fixed.
        v_initial: Starting velocity for the optimiser [m/s].  Defaults to
            ``base_params.initial_velocity``.
        t_max: Simulation duration [s].

    Returns:
        A ``(optimal_velocity [m/s], SimulationResult)`` tuple.
    """
    if v_initial is None:
        v_initial = base_params.initial_velocity

    def residual(velocity: jax.Array, args: None) -> jax.Array:  # noqa: ARG001
        params = PelletParams(
            initial_mass=base_params.initial_mass,
            initial_velocity=velocity[0],
            ablation_coeff=base_params.ablation_coeff,
            pellet_density=base_params.pellet_density,
        )
        result = simulate(params, plasma, t_max=t_max)
        return jnp.array([result.mean_deposition_depth - target_depth])

    v0 = jnp.array([v_initial])
    # LevenbergMarquardt uses the Jacobian; we request reverse-mode ("bwd")
    # because the diffrax adjoint is a custom_vjp (no JVP support).
    solution = optx.least_squares(
        residual,
        optx.LevenbergMarquardt(rtol=1e-6, atol=1e-9),
        v0,
        args=None,
        options={"jac": "bwd"},
        max_steps=50,
        throw=False,
    )

    opt_v = float(solution.value[0])
    opt_params = PelletParams(
        initial_mass=base_params.initial_mass,
        initial_velocity=opt_v,
        ablation_coeff=base_params.ablation_coeff,
        pellet_density=base_params.pellet_density,
    )
    return opt_v, simulate(opt_params, plasma, t_max=t_max)


# ---------------------------------------------------------------------------
# Demo / main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the quickgun demo: forward simulation, gradient check, optimisation.

    Demonstrates:
    1. Forward simulation of the NGS ablation model.
    2. Exact gradient ``d(mean_deposition_depth)/d(initial_velocity)``
       computed via ``jax.grad`` -- confirms differentiability.
    3. Optimisation with ``optimistix.LevenbergMarquardt`` to find the
       injection velocity that places the mean deposition depth at a
       prescribed target, further confirming differentiability.
    """
    print("=== quickgun: JAX pellet injection model ===\n")

    # ------------------------------------------------------------------
    # Physical setup: a 1.5-mm frozen H2 pellet in a W7-X-like plasma
    # ------------------------------------------------------------------
    rho_H2_ice = 88.0  # kg/m^3  (solid hydrogen ice density)
    r0 = 1.5e-3  # m        (initial pellet radius)
    m0 = (4.0 / 3.0) * float(jnp.pi) * r0**3 * rho_H2_ice  # ~ 1.24e-6 kg

    params = PelletParams(
        initial_mass=m0,
        initial_velocity=1200.0,  # m/s  (typical quickgun muzzle velocity)
        ablation_coeff=3.0e-9,  # SI   (calibrated: ~1 ms ablation at core)
        pellet_density=rho_H2_ice,
    )

    plasma = PlasmaProfile(
        n_e0=3.0e19,  # m^-3  (W7-X standard plasma)
        T_e0=800.0,  # eV
        minor_radius=0.5,  # m
    )

    # ------------------------------------------------------------------
    # 1. Forward simulation
    # ------------------------------------------------------------------
    print("--- 1. Forward simulation ---")
    result = simulate(params, plasma)
    mass_frac = float(result.final_mass) / m0
    print(f"  Initial mass          : {m0:.3e} kg")
    print(
        f"  Final mass            : {float(result.final_mass):.3e} kg"
        f"  ({100 * mass_frac:.1f} % remaining)"
    )
    print(f"  Final position        : {float(result.final_position) * 100:.1f} cm")
    print(
        f"  Mean deposition depth : {float(result.mean_deposition_depth) * 100:.1f} cm"
        f"  (plasma centre = {plasma.minor_radius * 100:.0f} cm)"
    )

    # ------------------------------------------------------------------
    # 2. Gradient via jax.grad -- proves end-to-end differentiability
    # ------------------------------------------------------------------
    print("\n--- 2. Gradient check via jax.grad ---")

    def depth_fn(v: jax.Array) -> jax.Array:
        p = PelletParams(
            initial_mass=params.initial_mass,
            initial_velocity=v,
            ablation_coeff=params.ablation_coeff,
            pellet_density=params.pellet_density,
        )
        return simulate(p, plasma).mean_deposition_depth

    grad_v = jax.grad(depth_fn)(jnp.float64(params.initial_velocity))
    print(
        f"  d(mean_depth)/d(v) = {float(grad_v):.3e} m / (m/s)"
        "  [non-zero => differentiable]"
    )

    # ------------------------------------------------------------------
    # 3. Optimisation: tune velocity to hit a target deposition depth
    #
    # The mean deposition depth is monotone in the injection velocity for
    # v < ~500 m/s (where the pellet ablates fully inside the plasma).
    # We start the optimiser at v=200 m/s (shallow deposition) and drive
    # it to a target of 30 cm, demonstrating that optimistix can call
    # jax.grad through the full ODE simulation.
    # ------------------------------------------------------------------
    target_depth = 0.30  # m  (30 cm, deeper than the v=200 baseline)
    print(
        f"\n--- 3. Optimise velocity for target depth {target_depth * 100:.0f} cm ---"
    )

    # Baseline at the optimizer's starting velocity
    v_opt_start = 200.0  # m/s
    baseline_params = PelletParams(
        initial_mass=m0,
        initial_velocity=v_opt_start,
        ablation_coeff=params.ablation_coeff,
        pellet_density=rho_H2_ice,
    )
    baseline_result = simulate(baseline_params, plasma)
    print(
        f"  Baseline velocity     : {v_opt_start:.0f} m/s"
        f"  ->  depth = {float(baseline_result.mean_deposition_depth) * 100:.1f} cm"
    )

    opt_v, opt_result = find_velocity_for_target_depth(
        target_depth, plasma, baseline_params, v_initial=v_opt_start
    )
    achieved = float(opt_result.mean_deposition_depth)
    print(f"  Target depth          : {target_depth * 100:.1f} cm")
    print(f"  Optimal velocity      : {opt_v:.1f} m/s")
    print(f"  Achieved depth        : {achieved * 100:.1f} cm")
    print(
        f"  Remaining mass        : {float(opt_result.final_mass):.3e} kg"
        f"  ({100 * float(opt_result.final_mass) / m0:.1f} %)"
    )
    print(f"  Depth error           : {abs(achieved - target_depth) * 100:.3f} cm")


if __name__ == "__main__":
    main()

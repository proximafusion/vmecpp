# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Quasi-isodynamicity from a near-axis boundary by vmecpp.joint and by the nested
baseline.

The residuals measure how far the wells of the field strength along the field lines are
from those of a quasi-isodynamic field, in Boozer coordinates by the transformation of
booz_xform. The mirror ratio and the elongation are ConStellaration's. All are in JAX. A
cold, fully converged run on more modes checks each result.
"""

import argparse
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

import vmecpp
from vmecpp import autodiff_wout
from vmecpp.joint import QuasisymmetryProblem, converged_input, optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from joint_optimization import rotating_ellipse  # type: ignore

jax.config.update("jax_enable_x64", True)

NEAR_AXIS = {
    (0, 0): (0.9995178918749238, 0.0),
    (0, 1): (-0.03397756446187697, -0.12522899738111876),
    (1, -1): (-0.01970654337375076, -0.06026412218125497),
    (1, 0): (0.09979103126013728, -0.13433616571003834),
    (1, 1): (-0.02141878530027218, 0.05929507897750279),
}
"""The boundary that ConStellaration's baseline starts from, as rbc and zbs by (m, n).

It is its near-axis construction for three field periods, aspect ratio 10, elongation 5,
iota 0.75 and mirror ratio 0.25, on modes up to 1.
"""

SHARPNESS = 200.0
"""Sharpness of the weights of the extremes of a field strength or an elongation."""

SOFTNESS = 0.02
"""Fraction of a limit over which its excess sets in."""

CONSTRAINT_TOLERANCE = 1.0e-2
"""Largest relative miss of a target or a limit that counts as met."""

_WOUT_QUANTITIES = (
    "bmnc",
    "bsubumnc",
    "bsubvmnc",
    "lmns",
    "iotas",
    "rmnc",
    "zmns",
    "raxis_cc",
    "zaxis_cs",
)


def near_axis_start(mpol: int, ntor: int, ns: int) -> vmecpp.VmecInput:
    """A vacuum-like input inside the boundary of NEAR_AXIS."""
    vmec_input = rotating_ellipse(mpol, ntor, ns, nfp=3)
    rbc, zbs = np.zeros((mpol, 2 * ntor + 1)), np.zeros((mpol, 2 * ntor + 1))
    for (m, n), (r, z) in NEAR_AXIS.items():
        rbc[m, ntor + n], zbs[m, ntor + n] = r, z
    vmec_input.rbc, vmec_input.zbs = rbc, zbs
    vmec_input.raxis_c = np.zeros(ntor + 1)
    vmec_input.phiedge = 0.03
    return vmec_input


@dataclasses.dataclass(frozen=True)
class BoozerTables:
    """The angles of the grid of a Boozer transformation and the Fourier modes of a wout
    on it.

    ``mboz`` poloidal and ``nboz`` toroidal modes take ``2 (2 mboz + 1)`` and
    ``2 (2 nboz + 1)`` points over one field period, as in booz_xform.
    """

    nfp: int
    mboz: int
    nboz: int
    theta: np.ndarray
    zeta: np.ndarray
    xm: np.ndarray
    xn: np.ndarray
    xm_nyq: np.ndarray
    xn_nyq: np.ndarray
    sines: np.ndarray
    cosines: np.ndarray
    sines_nyq: np.ndarray
    cosines_nyq: np.ndarray

    @classmethod
    def of(cls, wout, mboz: int, nboz: int) -> "BoozerTables":
        nfp = int(wout.nfp)
        ntheta, nzeta = 2 * (2 * mboz + 1), 2 * (2 * nboz + 1)
        theta = np.repeat(2.0 * np.pi * np.arange(ntheta) / ntheta, nzeta)
        zeta = np.tile(2.0 * np.pi * np.arange(nzeta) / (nfp * nzeta), ntheta)
        tables = {}
        for suffix in ("", "_nyq"):
            xm = np.asarray(getattr(wout, "xm" + suffix), dtype=float)
            xn = np.asarray(getattr(wout, "xn" + suffix), dtype=float)
            angle = xm[:, None] * theta[None, :] - xn[:, None] * zeta[None, :]
            tables["xm" + suffix], tables["xn" + suffix] = xm, xn
            tables["sines" + suffix] = np.sin(angle)
            tables["cosines" + suffix] = np.cos(angle)
        return cls(nfp=nfp, mboz=mboz, nboz=nboz, theta=theta, zeta=zeta, **tables)


def boozer_angles(tables: BoozerTables, bsubumnc, bsubvmnc, lmns, iota):
    """The Boozer angles of the grid of the tables on one half-grid surface and the
    Jacobian of the angles with respect to those of the grid, from the Fourier
    coefficients of its wout.

    The transformation is that of booz_xform for a stellarator-symmetric field.
    """
    xm, xn = jnp.asarray(tables.xm), jnp.asarray(tables.xn)
    xm_nyq, xn_nyq = jnp.asarray(tables.xm_nyq), jnp.asarray(tables.xn_nyq)
    current_i, current_g = bsubumnc[0], bsubvmnc[0]
    # the covariant components are the derivatives of w beside the currents
    w_mn = jnp.where(
        xm_nyq != 0,
        bsubumnc / jnp.where(xm_nyq != 0, xm_nyq, 1.0),
        jnp.where(xn_nyq != 0, -bsubvmnc / jnp.where(xn_nyq != 0, xn_nyq, 1.0), 0.0),
    )
    lam = lmns @ tables.sines
    lam_theta = (lmns * xm) @ tables.cosines
    lam_zeta = -(lmns * xn) @ tables.cosines
    w = w_mn @ tables.sines_nyq
    w_theta = (w_mn * xm_nyq) @ tables.cosines_nyq
    w_zeta = -(w_mn * xn_nyq) @ tables.cosines_nyq
    scale = 1.0 / (current_g + iota * current_i)
    nu = scale * (w - current_i * lam)
    nu_theta = scale * (w_theta - current_i * lam_theta)
    nu_zeta = scale * (w_zeta - current_i * lam_zeta)
    return (
        tables.theta + lam + iota * nu,
        tables.zeta + nu,
        (1.0 + lam_theta) * (1.0 + nu_zeta) + (iota - lam_zeta) * nu_theta,
    )


def boozer_strength(tables: BoozerTables, bmnc, bsubumnc, bsubvmnc, lmns, iota):
    """The field strength of one half-grid surface in Boozer coordinates, from the
    Fourier coefficients of its wout.

    The result holds ``a`` and ``d`` of ``|B| = sum a[m, n] cos(m theta) cos(n nfp
    zeta) + d[m, n] sin(m theta) sin(n nfp zeta)`` in the Boozer angles, for ``m`` below
    ``mboz`` and ``n`` up to ``nboz``.
    """
    theta, zeta, jacobian = boozer_angles(tables, bsubumnc, bsubvmnc, lmns, iota)
    weighted = (bmnc @ tables.cosines_nyq) * jacobian
    m = jnp.arange(tables.mboz, dtype=jnp.float64)
    n = jnp.arange(tables.nboz + 1, dtype=jnp.float64) * tables.nfp
    m_theta, n_zeta = jnp.outer(m, theta), jnp.outer(n, zeta)
    factor = 2.0 / theta.size
    # 1/2 for the mean, 1 for the modes of one angle, 2 for those of both
    weight = (jnp.where(m > 0, 2.0, 1.0)[:, None] * jnp.where(n > 0, 2.0, 1.0)) / 2.0
    return (
        weight * factor * ((jnp.cos(m_theta) * weighted) @ jnp.cos(n_zeta).T),
        2.0 * factor * ((jnp.sin(m_theta) * weighted) @ jnp.sin(n_zeta).T),
    )


def quasi_isodynamic_residuals(
    a,
    d,
    iota,
    nfp: int,
    *,
    field_lines: int = 25,
    toroidal_points: int = 201,
    levels: int = 101,
):
    """Residuals that vanish for a quasi-isodynamic field and follow the field
    continuously, of the field strength ``a``, ``d`` of :func:`boozer_strength` on a
    surface of rotational transform ``iota``.

    Each field line runs over one field period from the toroidal angle of the maximum of
    the field strength, which is scaled to span 0 to 1 over all of them. On every field
    line, a quasi-isodynamic field lies below each level on the same fraction of the
    line, and has one well, its maximum at both ends and its minimum between them. The
    residuals are the differences of those fractions from their mean over the field
    lines, the variation of each line beyond that of one well, the distance of its ends
    from the maximum and that of its lowest point from the minimum.
    """
    period = 2.0 * jnp.pi / nfp
    start = jnp.where(a[0, 1] < 0, 0.5 * period, 0.0)
    phi = start + period * jnp.linspace(0.0, 1.0, toroidal_points)
    alpha = 2.0 * jnp.pi * jnp.arange(field_lines) / field_lines
    theta = alpha[:, None] + iota * (phi - start)[None, :]
    m = jnp.arange(a.shape[0], dtype=jnp.float64)
    n = jnp.arange(a.shape[1], dtype=jnp.float64) * nfp
    cos_n, sin_n = jnp.cos(jnp.outer(n, phi)), jnp.sin(jnp.outer(n, phi))
    strength = jnp.einsum(
        "mlp,mp->lp", jnp.cos(m[:, None, None] * theta), a @ cos_n
    ) + jnp.einsum("mlp,mp->lp", jnp.sin(m[:, None, None] * theta), d @ sin_n)
    strength = (strength - strength.min()) / (strength.max() - strength.min())
    lower = jnp.minimum(strength[:, :-1], strength[:, 1:])[:, :, None]
    span = jnp.abs(jnp.diff(strength, axis=1))[:, :, None]
    level = (jnp.arange(levels) + 0.5) / levels
    # the fraction of each segment of a line below each level, then of the line
    below = jnp.where(
        span > 0,
        jnp.clip((level - lower) / jnp.where(span > 0, span, 1.0), 0.0, 1.0),
        level > lower,
    ).mean(axis=1)
    spread = below - below.mean(axis=0)
    wells = 0.5 * span.sum(axis=(1, 2)) - (strength.max(axis=1) - strength.min(axis=1))
    ends = 1.0 - 0.5 * (strength[:, 0] + strength[:, -1])
    return jnp.concatenate(
        [
            spread.ravel() / jnp.sqrt(field_lines * levels),
            wells / jnp.sqrt(field_lines),
            ends / jnp.sqrt(field_lines),
            strength.min(axis=1) / jnp.sqrt(field_lines),
        ]
    )


def _extreme(values, sharpness: float):
    """The mean of the values weighted by ``exp(sharpness values / mean)``: their
    maximum as the sharpness grows, their minimum as it falls."""
    return values @ jax.nn.softmax(sharpness * values / values.mean())


def _excess(value, limit: float):
    """The amount by which a value exceeds a limit, setting in smoothly over SOFTNESS
    times the limit."""
    width = SOFTNESS * limit
    return width * jax.nn.softplus((value - limit) / width)


def mirror_ratio(strength, sharpness: float = SHARPNESS):
    """``(max - min) / (max + min)`` of a field strength with the extremes of
    :func:`_extreme`."""
    highest, lowest = _extreme(strength, sharpness), _extreme(strength, -sharpness)
    return (highest - lowest) / (highest + lowest)


def _ellipse_shape(elongation):
    """The squared perimeter over the area of an ellipse whose semi-axes have this
    ratio."""
    nodes, weights = np.polynomial.legendre.leggauss(64)
    angle = 0.25 * np.pi * (nodes + 1.0)
    second_kind = (0.25 * np.pi * weights) @ jnp.sqrt(
        1.0 - (1.0 - elongation**-2) * np.sin(angle) ** 2
    )
    return 16.0 / jnp.pi * elongation * second_kind**2


def elongations(
    xm, xn, rmnc, zmns, raxis_cc, zaxis_cs, nfp: int, *, planes: int, points: int
):
    """The elongation of the cross-sections of a surface in planes perpendicular to its
    magnetic axis, as ConStellaration computes it.

    Each of ``planes`` planes over one field period, both ends included, cuts the
    surface in a polygon through ``points`` poloidal angles, the first counted twice.
    Its elongation is the ratio of the semi-axes of the ellipse with its perimeter and
    its area.
    """
    phi = jnp.linspace(0.0, 2.0 * jnp.pi / nfp, planes)
    theta = 2.0 * jnp.pi * jnp.arange(points - 1) / (points - 1)
    n = nfp * jnp.arange(raxis_cc.size)

    def axis(angle):
        radius = raxis_cc @ jnp.cos(jnp.outer(n, angle))
        height = -zaxis_cs @ jnp.sin(jnp.outer(n, angle))
        return jnp.stack(
            [radius * jnp.cos(angle), radius * jnp.sin(angle), height], axis=-1
        )

    def surface(angle):
        phase = xm[:, None, None] * theta - xn[:, None, None] * angle[None]
        radius = jnp.tensordot(rmnc, jnp.cos(phase), 1)
        height = jnp.tensordot(zmns, jnp.sin(phase), 1)
        return jnp.stack(
            [radius * jnp.cos(angle), radius * jnp.sin(angle), height], axis=-1
        )

    centre, tangent = jax.jvp(axis, (phi,), (jnp.ones_like(phi),))
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)

    def into_plane(angle):
        point, slope = jax.jvp(surface, (angle,), (jnp.ones_like(angle),))
        return angle - jnp.einsum(
            "ijk,ik->ij", point - centre[:, None], tangent
        ) / jnp.einsum("ijk,ik->ij", slope, tangent)

    # Newton steps of each point into the plane of its axis point, of which the last
    # carries the derivatives
    angle = jnp.broadcast_to(phi[:, None], (planes, points - 1))
    for _ in range(7):
        angle = jax.lax.stop_gradient(into_plane(angle))
    section = surface(into_plane(angle)) - centre[:, None]
    following = jnp.roll(section, -1, axis=1)
    perimeter = jnp.linalg.norm(following - section, axis=-1).sum(axis=1)
    area = 0.5 * jnp.abs(
        jnp.einsum("ijk,ik->i", jnp.cross(section, following), tangent)
    )
    shape = perimeter**2 / area

    def to_shape(elongation):
        value, slope = jax.vmap(jax.value_and_grad(_ellipse_shape))(elongation)
        return elongation - (value - shape) / slope

    # likewise from the elongation of a long ellipse of this shape
    elongation = jnp.pi * shape / 16.0
    for _ in range(7):
        elongation = jax.lax.stop_gradient(to_shape(elongation))
    return to_shape(elongation)


@dataclasses.dataclass(kw_only=True)
class QuasiIsodynamicProblem(QuasisymmetryProblem):
    """The aspect-ratio and mean-iota targets of the base class, the excess of the
    mirror ratio at the edge over ``mirror`` and of the largest elongation over
    ``elongation``, and the residuals of quasi-isodynamicity on the outermost half-grid
    surface.

    The mirror ratio is that of the field strength extrapolated to the edge on the grid
    of the Boozer transformation. The elongation is that of :func:`elongations` in
    ``planes`` planes through ``points`` poloidal angles.
    """

    surfaces: tuple[float, ...] = ()
    aspect: float = 10.0
    mean_iota: float = 0.82
    mirror: float = 0.2
    elongation: float = 5.0
    planes: int = 24
    points: int = 26
    field_lines: int = 25
    toroidal_points: int = 201
    levels: int = 101

    def __post_init__(self) -> None:
        super().__post_init__()
        modes = SimpleNamespace(**autodiff_wout.static_fields(self.vmec_input))
        self._tables = BoozerTables.of(modes, 6 * self.mpol, 2 * self.ntor - 1)

    def edge(self, wout, tables: BoozerTables):
        """The residuals of quasi-isodynamicity on the outermost half-grid surface, the
        field strength extrapolated to the edge on the grid of the tables and the
        elongations of the boundary, from the quantities of a wout."""
        edge = wout["iotas"].shape[0] - 1
        iota = wout["iotas"][edge]
        a, d = boozer_strength(
            tables,
            *(wout[name][:, edge] for name in ("bmnc", "bsubumnc", "bsubvmnc", "lmns")),
            iota,
        )
        return (
            quasi_isodynamic_residuals(
                a,
                d,
                iota,
                tables.nfp,
                field_lines=self.field_lines,
                toroidal_points=self.toroidal_points,
                levels=self.levels,
            ),
            (1.5 * wout["bmnc"][:, edge] - 0.5 * wout["bmnc"][:, edge - 1])
            @ tables.cosines_nyq,
            elongations(
                jnp.asarray(tables.xm),
                jnp.asarray(tables.xn),
                wout["rmnc"][:, edge],
                wout["zmns"][:, edge],
                wout["raxis_cc"],
                wout["zaxis_cs"],
                tables.nfp,
                planes=self.planes,
                points=self.points,
            ),
        )

    def residuals(self, solved, boundary):
        quasi_isodynamic, strength, elongation = self.edge(
            autodiff_wout.wout_quantities(solved, self.vmec_input), self._tables
        )
        limits = jnp.stack(
            [
                _excess(mirror_ratio(strength), self.mirror),
                _excess(_extreme(elongation, SHARPNESS), self.elongation),
            ]
        )
        return jnp.concatenate(
            [super().residuals(solved, boundary), limits, quasi_isodynamic]
        )

    def parts(self, residuals: np.ndarray) -> dict[str, float]:
        return {
            "aspect": float(residuals[0]) + self.aspect,
            "mean_iota": float(residuals[1]) + self.mean_iota,
            "mirror_excess": float(residuals[2]),
            "elongation_excess": float(residuals[3]),
            "quasi_isodynamicity": float(residuals[4:] @ residuals[4:]),
        }


def measures(problem: QuasiIsodynamicProblem, wout) -> dict[str, float]:
    """The sum of the squares of the residuals of quasi-isodynamicity of a wout, its
    mirror ratio at the edge, its largest elongation, its aspect ratio and its mean
    iota.

    The mirror ratio and the elongation take their extremes as they are.
    """
    quasi_isodynamic, strength, elongation = problem.edge(
        {name: np.asarray(getattr(wout, name)) for name in _WOUT_QUANTITIES},
        BoozerTables.of(wout, 6 * wout.mpol, 2 * wout.ntor - 1),
    )
    return {
        "quasi_isodynamicity": float(quasi_isodynamic @ quasi_isodynamic),
        "mirror": float(
            (strength.max() - strength.min()) / (strength.max() + strength.min())
        ),
        "elongation": float(elongation.max()),
        "aspect": float(wout.aspect),
        "mean_iota": float(np.mean(np.asarray(wout.iotas)[1:])),
    }


def converged_wout(problem: QuasiIsodynamicProblem, boundary: np.ndarray, mpol: int):
    """The wout of a cold, fully converged run of a boundary on ``mpol`` poloidal modes
    and as many toroidal ones.

    A run that does not converge raises a RuntimeError.
    """
    vmec_input = converged_input(problem, boundary, pin_gauge=True).model_copy(
        update={
            "mpol": mpol,
            "ntor": mpol,
            "rbc": vmecpp.VmecInput.resize_2d_coeff(boundary[0], mpol, mpol),
            "zbs": vmecpp.VmecInput.resize_2d_coeff(boundary[1], mpol, mpol),
            "raxis_c": np.zeros(mpol + 1),
            "zaxis_s": np.zeros(mpol + 1),
            "return_outputs_even_if_not_converged": False,
        }
    )
    return vmecpp.run(vmec_input, verbose=False, max_threads=1).wout


def check(problem: QuasiIsodynamicProblem, boundary: np.ndarray, mpol: int) -> None:
    """Print the measures and any unmet target or limit of a cold run on ``mpol``
    poloidal modes."""
    try:
        wout = converged_wout(problem, boundary, mpol)
    except RuntimeError:
        print(f"  converged run on mpol {mpol}: VMEC++ did not converge")
        return
    found = measures(problem, wout)
    misses = {
        "aspect": abs(found["aspect"] / problem.aspect - 1.0),
        "mean iota": abs(found["mean_iota"] / problem.mean_iota - 1.0),
        "mirror ratio": found["mirror"] / problem.mirror - 1.0,
        "elongation": found["elongation"] / problem.elongation - 1.0,
    }
    unmet = [name for name, miss in misses.items() if miss > CONSTRAINT_TOLERANCE]
    print(
        f"  converged run on mpol {mpol}: force residual"
        f" {max(wout.fsqr, wout.fsqz, wout.fsql):.1e}, quasi-isodynamicity"
        f" {found['quasi_isodynamicity']:.3e}, aspect {found['aspect']:.5f},"
        f" mean iota {found['mean_iota']:.5f}, mirror ratio {found['mirror']:.4f},"
        f" elongation {found['elongation']:.3f}, unmet: {', '.join(unmet) or 'none'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ns", type=int, default=25)
    parser.add_argument("--mpol", type=int, default=5)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--aspect", type=float, default=10.0)
    parser.add_argument("--iota", type=float, default=0.82)
    parser.add_argument("--mirror", type=float, default=0.2, help="largest at the edge")
    parser.add_argument("--elongation", type=float, default=5.0, help="largest")
    parser.add_argument(
        "--check-mpol",
        type=int,
        default=8,
        help="mpol and ntor of the converged run that checks a result",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=None,
        help="mpol and ntor of every stage but the last above its boundary modes",
    )
    args = parser.parse_args()
    for name, nested in (("nested", True), ("joint", False)):
        problem = QuasiIsodynamicProblem(
            near_axis_start(args.mpol, args.mpol, args.ns),
            aspect=args.aspect,
            mean_iota=args.iota,
            mirror=args.mirror,
            elongation=args.elongation,
            max_mode=max(args.stages),
        )
        result = optimize(
            problem, tuple(args.stages), nested=nested, margin=args.margin, log=print
        )
        costs = problem.costs
        print(
            f"{name}: {len(result.steps)} steps, {costs.force_evaluations} force"
            f" evaluations, {result.steps[-1].seconds:.1f} s, of which solves"
            f" {costs.solve_seconds:.1f} s, {costs.assemblies} assemblies"
            f" {costs.assembly_seconds:.1f} s and the objective"
            f" {costs.objective_seconds:.1f} s"
        )
        check(problem, result.boundary, args.check_mpol)


if __name__ == "__main__":
    main()

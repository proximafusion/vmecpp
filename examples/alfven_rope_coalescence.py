# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""The inductive energy of Alfven's rope and of the single strand its strands coalesce
into, from Biot-Savart over VMEC++'s current density and in closed form."""

import argparse
import dataclasses
import functools
import json
import os
from pathlib import Path

import alfven_rope as rope_example
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, jv

import vmecpp

MU0 = rope_example.MU0
DATA = Path(__file__).resolve().parent / "data" / "alfven_rope.json"


@dataclasses.dataclass(frozen=True)
class Merged(rope_example.Rope):
    """One Taylor strand on the axis of the rope, with the current and toroidal flux of
    all its strands at the same k."""

    strands: int = 1
    merged: int = 2

    @functools.cached_property
    def source(self):
        return rope_example.Rope(angle=self.angle, spacing=self.spacing, strands=2)

    @functools.cached_property
    def helix_radius(self):
        return 0.0

    @functools.cached_property
    def radius(self):
        return np.sqrt(self.merged) * self.source.radius

    @functools.cached_property
    def ka(self):
        return np.sqrt(self.merged) * self.source.ka

    @functools.cached_property
    def phiedge(self):
        return self.merged * self.source.phiedge

    @functools.cached_property
    def current(self):
        return self.merged * self.source.current


def surface(wout, ntheta=96, nzeta=288):
    """Points and outward area elements n dS of the boundary of a strand."""
    theta = 2.0 * np.pi * np.arange(ntheta) / ntheta
    zeta = 2.0 * np.pi * np.arange(nzeta) / nzeta
    points, e_theta, e_zeta, *_ = rope_example.geometry(
        wout, np.array([int(wout.ns) - 1]), theta, zeta
    )
    area = np.cross(e_theta, e_zeta)[0] * (2.0 * np.pi) ** 2 / (ntheta * nzeta)
    points = points[0]
    # the orientation for which the divergence theorem gives a positive volume
    area *= np.sign(np.sum(points * area))
    return points.reshape(-1, 3), area.reshape(-1, 3)


def potential(targets, boundary):
    """The integral of 1 / |r - r'| over the strand's volume at the points r, as half
    the boundary integral of the unit vector from r times n dS."""
    points, area = boundary
    out = np.empty(len(targets))
    for k in range(0, len(targets), 64):
        d = points[None] - targets[k : k + 64, None]
        norm = np.linalg.norm(d, axis=-1)
        unit = np.divide(
            d, norm[..., None], out=np.zeros_like(d), where=norm[..., None] > 0
        )
        out[k : k + 64] = 0.5 * np.sum(unit * area[None], axis=(1, 2))
    return out


def energy(strand, boundary, turns, nfp, chunk=128):
    """The energy of the strand and its images turned by turns about Z, with J(r) taken
    out at each target and its integral of 1 / |r - r'| taken on the boundary."""
    points, density, volume = strand
    flat_p, flat_j, flat_v = (
        points.reshape(-1, 3),
        density.reshape(-1, 3),
        volume.ravel(),
    )
    flat_jdv = flat_j * flat_v[:, None]
    period = slice(0, points.shape[2] // nfp)
    targets = points[:, :, period].reshape(-1, 3)
    target_j = density[:, :, period].reshape(-1, 3)
    target_v = volume[:, :, period].ravel()
    images = [(flat_p @ t.T, flat_jdv @ t.T) for t in turns[1:]]
    total = 0.0
    for k in range(0, len(targets), chunk):
        t, j = targets[k : k + chunk], target_j[k : k + chunk]
        distance = np.linalg.norm(t[:, None] - flat_p[None], axis=-1)
        inverse = np.divide(
            1.0, distance, out=np.zeros_like(distance), where=distance > 0
        )
        regular = inverse @ flat_jdv - j * (inverse @ flat_v)[:, None]
        own = regular + j * potential(t, boundary)[:, None]
        for image_p, image_jdv in images:
            far = 1.0 / np.linalg.norm(t[:, None] - image_p[None], axis=-1)
            own += far @ image_jdv
        total += np.sum(np.sum(j * own, axis=1) * target_v[k : k + chunk])
    return MU0 / (8.0 * np.pi) * nfp * len(turns) * total


def turn(angle):
    """The rotation by angle about Z."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fixed(rope, x, ns):
    """The fixed-boundary strand inside the boundary of x on ns surfaces, reached
    through the grids of 25 and 49 surfaces below it."""
    grids = np.array([g for g in (25, 49) if g < ns] + [ns])
    rb, zb = rope_example.boundary(rope, x)
    return vmecpp.run(
        rope_example.strand_input(
            rope,
            rb,
            zb,
            ns_array=grids,
            ftol_array=np.full(grids.size, 1e-13),
            niter_array=np.full(grids.size, 20000),
        ),
        verbose=False,
    )


def lundquist(x):
    """J0 J2 / J1^2 at k a = x, which sets the energy of the interior of a Lundquist
    strand."""
    return j0(x) * jv(2, x) / j1(x) ** 2


def braid_energy(rope, d):
    """The energy of two thin Lundquist strands wound at helix radius d: their rings, their
    mutual inductance and the winding."""
    x, b, r0 = rope.ka, rope.radius, rope_example.R0
    t = rope_example.N * d / r0
    # each strand as a ring with a Lundquist interior
    ring = np.log(8.0 * r0 / b) - 1.0 - 1.5 * lundquist(x)
    # the pair's mutual inductance, to order (d / R0)^2
    mutual = (
        np.log(4.0 * r0 / d)
        - 2.0
        + (d / r0) ** 2 * (0.75 * np.log(4.0 * r0 / d) - 0.25)
    )
    # the winding's solenoid, tan psi = t, and its overlap with the strands' interiors
    winding = 0.75 * t**2 + t * b / d * (2.0 / x - j0(x) / j1(x))
    # each strand deformed by its partner's tidal field
    tidal = (b / d) ** 4 / 32.0 * (j1(x) * jv(3, x) + j0(x) * jv(2, x))
    tidal /= j1(x) ** 2 - j0(x) * jv(2, x)
    return MU0 * r0 * rope.current**2 * (ring + mutual + winding + tidal)


def merged_energy(rope):
    """The energy of the thin Lundquist ring with the flux and current of both strands
    at their k, at the radius where its edge field is the guide field."""
    x, r0 = rope.ka, rope_example.R0
    y = brentq(lambda v: v * j1(v) / j0(v) - 2.0 * x * j1(x) / j0(x), 1e-6, 2.404)
    ring = np.log(8.0 * r0 * x / (y * rope.radius)) - 1.0 - 1.5 * lundquist(y)
    return 2.0 * MU0 * r0 * rope.current**2 * ring


def main():
    parser = argparse.ArgumentParser(description="coalescence energy of Alfven's rope")
    parser.add_argument("--rope", default=str(DATA), help="matched shape of the rope")
    parser.add_argument(
        "--merged",
        help="matched shape of the merged strand, matched and written when absent; by "
        "default the rope's file with _merged added to its name",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="the energies on 25, 49 and 97 surfaces and at 24 x 108 and 32 x 144 "
        "angles, instead of 25 surfaces and 24 x 108 angles",
    )
    args = parser.parse_args()
    data = json.loads(Path(args.rope).read_text())
    rope = rope_example.Rope(angle=data["angle"], spacing=data["spacing"])
    merged = Merged(angle=rope.angle, spacing=rope.spacing)
    print(
        f"rope: 2 strands of {rope.current / 1e3:.2f} kA and {rope.phiedge:.6f} Wb; "
        f"merged strand of {merged.current / 1e3:.2f} kA and {merged.phiedge:.6f} Wb",
        flush=True,
    )
    path = Path(args.rope)
    path = Path(args.merged) if args.merged else path.with_stem(path.stem + "_merged")
    if path.exists():
        matched = json.loads(path.read_text())
    else:
        x, vertical, _ = rope_example.match(
            merged,
            args.workers,
            lambda step, f, field: print(
                f"  step {step}: {np.abs(f).max():.2e}, {field:+.5f} T"
            ),
        )
        matched = {
            "angle": rope.angle,
            "spacing": rope.spacing,
            "vertical": vertical,
            "modes": merged.modes,
            "x": x.tolist(),
        }
        path.write_text(json.dumps(matched, indent=1))
    pair_turns = [np.eye(3), turn(np.pi / rope_example.N)]
    total = 2.0 * rope.current
    # surfaces of the solves and of Simpson's rule, and angles of the sums
    grids = [(25, 25, 24, 108)]
    if args.convergence:
        grids = [
            (25, 25, 32, 144),
            (49, 25, 32, 144),
            (97, 25, 32, 144),
            (97, 49, 24, 108),
            (97, 49, 32, 144),
        ]
    runs = {}
    for ns, surfaces, ntheta, nzeta in grids:
        if ns not in runs:
            runs[ns] = (
                fixed(rope, np.array(data["x"]), ns),
                fixed(merged, np.array(matched["x"]), ns),
            )
        braid, coalesced = runs[ns]
        step = (ns - 1) // (surfaces - 1)
        *strand, i_braid = rope_example.currents(braid.wout, step, ntheta, nzeta)
        w_braid = energy(strand, surface(braid.wout), pair_turns, rope_example.N)
        *strand, i_merged = rope_example.currents(coalesced.wout, step, ntheta, nzeta)
        w_merged = energy(strand, surface(coalesced.wout), [np.eye(3)], rope_example.N)
        print(
            f"ns {ns}, {surfaces} surfaces, {ntheta} x {nzeta}: strand current "
            f"{i_braid / 1e3:.2f} kA, merged {i_merged / 1e3:.2f} kA; energy of the "
            f"braid {w_braid / 1e3:.3f} kJ, merged {w_merged / 1e3:.3f} kJ, "
            f"coalescence {(w_merged - w_braid) / 1e3:.3f} kJ; inductance "
            f"{2.0 * w_braid / total**2 * 1e6:.4f} and "
            f"{2.0 * w_merged / total**2 * 1e6:.4f} uH",
            flush=True,
        )
    _, d = rope_example.helix_of(rope_example.axis_points(runs[25][0].wout))
    vertical = matched["vertical"]
    # straight wires: two of radius b at the helix radius, one of radius sqrt(2) b
    wire = (
        2.0
        * np.pi
        * rope_example.R0
        * MU0
        * rope.current**2
        / (4.0 * np.pi)
        * (0.5 + 2.0 * np.log(d / rope.radius))
    )
    print(
        f"thin straight wires: coalescence {wire / 1e3:.3f} kJ; vertical field of the "
        f"merged strand {vertical:+.5f} T"
    )
    w_braid, w_merged = braid_energy(rope, d), merged_energy(rope)
    print(
        f"closed form: braid {w_braid / 1e3:.3f} kJ, merged {w_merged / 1e3:.3f} kJ, "
        f"coalescence {(w_merged - w_braid) / 1e3:.3f} kJ"
    )


if __name__ == "__main__":
    main()

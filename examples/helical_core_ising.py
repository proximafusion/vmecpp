# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Solve two-spin Ising problems with the orientation of the helical core of a spherical
tokamak.

Without stellarator symmetry the core can turn about the torus. A ripple of R with four
periods holds it at the four states of two spins. The boundary also sets their coupling
and their fields.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np

import vmecpp

# the spherical tokamak with q = 1 on the axis
SPHERICAL_TOKAMAK = vmecpp.VmecInput.from_file(
    Path(__file__).parent / "data" / "spherical_tokamak.json"
)
NTOR = SPHERICAL_TOKAMAK.ntor_max
# pres_scale above the kink threshold of 2.74e4 under the pinning ripple
PRESSURE = 3.0e4
# pres_scale below and above the kink threshold of 2.54e4 without the pinning ripple
BELOW, ABOVE = 2.45e4, 2.6e4
# the n = 4 ripple of R that holds the core at the four diagonal orientations
PIN = 0.03
# the axis shift of the initial guess of a held core
HELD = 0.3
# the ripples at which the coupling and the fields are measured
COUPLING, FIELD = 5e-3, 2e-5


class Boundary(NamedTuple):
    """The ripples of the boundary that set an Ising problem."""

    coupling: float  # the n = 2 sine ripple of R
    field_1: float  # the m = 1, n = 1 cosine ripple of R and Z
    field_2: float  # the m = 1, n = 1 sine ripple of R and Z
    pin: float = PIN


# Every sign of the coupling and of both fields. The coupling outweighs the second field.
PROBLEMS = [
    Boundary(*signs)
    for signs in itertools.product((5e-3, -5e-3), (1e-4, -1e-4), (5e-5, -5e-5))
]


def boundary_coefficients(boundary):
    """The coefficients rbc, rbs, zbc and zbs of the boundary with its ripples."""
    rbc, zbs = SPHERICAL_TOKAMAK.rbc.copy(), SPHERICAL_TOKAMAK.zbs.copy()
    rbs, zbc = np.zeros_like(rbc), np.zeros_like(zbs)
    rbc[0, NTOR + 4] = boundary.pin
    rbs[0, NTOR + 2] = boundary.coupling
    rbc[1, NTOR + 1] = zbs[1, NTOR + 1] = boundary.field_1
    rbs[1, NTOR + 1] = boundary.field_2
    zbc[1, NTOR + 1] = -boundary.field_2
    return rbc, rbs, zbc, zbs


def turned(coefficients, degrees):
    """The coefficients rbc, rbs, zbc and zbs in a frame turned by ``degrees``
    toroidally."""
    rbc, rbs, zbc, zbs = coefficients
    angle = np.radians(degrees) * np.arange(-NTOR, NTOR + 1)
    c, s = np.cos(angle), np.sin(angle)
    return rbc * c - rbs * s, rbc * s + rbs * c, zbc * c - zbs * s, zbs * c + zbc * s


def spherical_tokamak(pres_scale, coefficients, orientation=0.0, core=0.0, lasym=True):
    """The input of the spherical tokamak with the boundary ``coefficients`` and its
    initial axis shifted by ``core`` towards ``orientation`` in degrees."""
    phi = np.radians(orientation)
    rbc, rbs, zbc, zbs = coefficients
    vmec_input = SPHERICAL_TOKAMAK.model_copy(deep=True)
    vmec_input.niter_array = np.array([200000])
    vmec_input.pres_scale = pres_scale
    vmec_input.rbc, vmec_input.zbs = rbc, zbs
    vmec_input.raxis_c[1] = core * np.cos(phi)
    vmec_input.zaxis_s[1] = -core * np.cos(phi)
    if lasym:
        sine = np.zeros(NTOR + 1)
        sine[1] = -core * np.sin(phi)
        vmec_input.lasym = True
        vmec_input.rbs, vmec_input.zbc = rbs, zbc
        vmec_input.raxis_s, vmec_input.zaxis_c = sine, sine.copy()
    return vmec_input


def solve(vmec_input, restart_from=None):
    """The equilibrium of the input on one thread."""
    return vmecpp.run(
        vmec_input, verbose=False, restart_from=restart_from, max_threads=1
    )


def energy(output):
    """The MHD energy of an equilibrium."""
    return float(output.wout.wb) - float(output.wout.wp)


def core_of(output):
    """The axis shift of the helical core of an equilibrium and its orientation in
    degrees."""
    xm, xn = np.asarray(output.wout.xm), np.asarray(output.wout.xn)
    mode = (xm == 0) & (xn == 1)
    c = float(np.asarray(output.wout.rmnc)[mode, 0][0])
    s = float(np.asarray(output.wout.rmns)[mode, 0][0]) if output.wout.lasym else 0.0
    return np.hypot(c, s), np.degrees(np.arctan2(-s, c)) % 360.0


def spins(orientation):
    """The two spins of a core at ``orientation`` in degrees."""
    phi = np.radians(orientation)
    return int(np.sign(np.cos(phi))), int(np.sign(np.sin(phi)))


def held_energy(boundary, diagonal, sign):
    """The MHD energy of the core the boundary holds at ``diagonal`` degrees for a
    positive ``sign`` and opposite it for a negative one.

    The solve is stellarator-symmetric in the frame of the diagonal.
    """
    coefficients = turned(boundary_coefficients(boundary), diagonal)
    vmec_input = spherical_tokamak(
        PRESSURE, coefficients, 0.0, sign * HELD, lasym=False
    )
    vmec_input.ftol_array = np.array([1e-16])
    return energy(solve(vmec_input))


# The held cores of a coupling at 45 and 135 degrees, and along and against a field
# along the diagonal at 45 degrees.
HOLDS = [
    (Boundary(COUPLING, 0.0, 0.0), 45.0, 1),
    (Boundary(COUPLING, 0.0, 0.0), 135.0, 1),
    (Boundary(0.0, FIELD, -FIELD), 45.0, 1),
    (Boundary(0.0, FIELD, -FIELD), 45.0, -1),
]


def ising_coefficients(aligned, opposed, along, against):
    """J per unit coupling ripple and h per unit field ripple from the energies of the
    HOLDS."""
    return (opposed - aligned) / (2.0 * COUPLING), (against - along) / (4.0 * FIELD)


def ising(boundary, per_coupling, per_field):
    """J, h1 and h2 of E = E0 - J s1 s2 - h1 s1 - h2 s2 for the ripples of the
    boundary."""
    return (
        per_coupling * boundary.coupling,
        per_field * boundary.field_1,
        -per_field * boundary.field_2,
    )


def ground_state(coupling, field_1, field_2):
    """The two spins of least energy."""
    return min(
        itertools.product((1, -1), repeat=2),
        key=lambda s: -coupling * s[0] * s[1] - field_1 * s[0] - field_2 * s[1],
    )


def anneal(boundary):
    """The orientation of the helical core the plasma reaches from below the kink
    threshold."""
    free = boundary_coefficients(boundary._replace(pin=0.0))
    state = None
    # Each step restarts from the equilibrium of the step before. The tolerance above
    # the threshold is low enough for the kink to grow from the response to the fields.
    for pres_scale, coefficients, ftol in (
        (BELOW, free, 1e-13),
        (ABOVE, free, 1e-14),
        (PRESSURE, boundary_coefficients(boundary), 1e-13),
    ):
        amplitude, orientation = (0.0, 0.0) if state is None else core_of(state)
        vmec_input = spherical_tokamak(pres_scale, coefficients, orientation, amplitude)
        vmec_input.ftol_array = np.array([ftol])
        state = solve(vmec_input, restart_from=state)
    return core_of(state)[1]


def main():
    with ProcessPoolExecutor() as pool:
        energies = list(pool.map(held_energy, *zip(*HOLDS, strict=True)))
        orientations = list(pool.map(anneal, PROBLEMS))
    per_coupling, per_field = ising_coefficients(*energies)
    print(
        f"J {per_coupling:+.3e} per coupling ripple, h {per_field:+.3e} per field ripple"
    )
    wrong = 0
    for boundary, orientation in zip(PROBLEMS, orientations, strict=True):
        coefficients = ising(boundary, per_coupling, per_field)
        ground = ground_state(*coefficients)
        wrong += spins(orientation) != ground
        print(
            "J {:+.2e}, h1 {:+.2e}, h2 {:+.2e}: ground state {}, the plasma reaches {}"
            " at {:.1f} deg".format(
                *coefficients, ground, spins(orientation), orientation
            )
        )
    if wrong:
        message = f"the plasma misses the ground state of {wrong} problems"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()

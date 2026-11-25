# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A computer of kink parametrons that runs Rule 110.

Above a critical pressure the axisymmetric equilibrium of a spherical tokamak with q = 1
on its axis is unstable to the m = 1, n = 1 internal kink. Its stable equilibria are two
helical cores, mirror images of each other. A helical ripple of the boundary above a
threshold leaves only the core whose magnetic axis shifts with the sign of the ripple,
and below it the plasma keeps the core it has. Such a bistable element is a parametron,
and the sum of the ripples of its inputs sets it. It computes the majority of three
inputs, and a reversed coupling negates an input. This example certifies the majority
with VMEC++ equilibria started toward either core and evolves Rule 110 from it. M. Cook,
Complex Systems 15, 1 (2004), proved this elementary cellular automaton universal.
"""

import numpy as np

import vmecpp

NS, MPOL, NTOR = 49, 6, 1
# pres_scale above the kink threshold. The threshold is 2.53e4 on 25 surfaces and falls
# with the number of surfaces.
PRESSURE = 2.8e4
# The m = 1, n = 1 ripple of the boundary that one input adds.
UNIT = 1.0e-2
# The n = 1 shifts of the initial magnetic axis toward the two cores.
SEEDS = (0.05, -0.05)


def spherical_tokamak(ripple, seed):
    """The VMEC++ input of the spherical tokamak with the boundary ripple ``ripple``.

    The boundary has major radius 1.5, minor radius 1 and elongation 1.6. The profiles
    are \\iota = 1 - 2/3 s^4 and p = p_0 (1 - s)^4. The initial magnetic axis shifts
    helically by ``seed``.
    """
    vmec_input = vmecpp.VmecInput(
        lasym=False,
        nfp=1,
        mpol=MPOL,
        ntor=NTOR,
        ns_array=np.array([NS]),
        ftol_array=np.array([1e-12]),
        niter_array=np.array([30000]),
        phiedge=5.0,
        ncurr=0,
        ai=np.array([1.0, 0.0, 0.0, 0.0, -2.0 / 3.0]),
        pmass_type="power_series",
        am=np.array([1.0, -4.0, 6.0, -4.0, 1.0]),
        pres_scale=PRESSURE,
        gamma=0.0,
        always_fix_m1_gauge=True,
        raxis_c=np.array([1.5, seed]),
        zaxis_s=np.array([0.0, -seed]),
    )
    vmec_input.rbc = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[1.5], [1.0]]), MPOL, NTOR
    )
    vmec_input.zbs = vmecpp.VmecInput.resize_2d_coeff(
        np.array([[0.0], [1.6]]), MPOL, NTOR
    )
    vmec_input.rbc[1, NTOR + 1] = ripple
    vmec_input.zbs[1, NTOR + 1] = ripple
    return vmec_input


def axis_shift(output):
    """The n = 1 cosine coefficient of R on the magnetic axis of an equilibrium."""
    xm, xn = np.asarray(output.wout.xm), np.asarray(output.wout.xn)
    return float(np.asarray(output.wout.rmnc)[(xm == 0) & (xn == 1), 0][0])


def equilibrium(ripple, seed=0.0, restart_from=None):
    """The VMEC++ equilibrium of the spherical tokamak with the boundary ripple
    ``ripple``, from the initial axis ``seed`` or the state ``restart_from``."""
    vmec_input = spherical_tokamak(ripple, seed)
    return vmecpp.run(vmec_input, verbose=False, restart_from=restart_from)


def certify():
    """The bit that each sum of three inputs of +1 or -1 sets, and the axis shifts.

    The bit is the sign of the axis shift. Every sum is solved from initial axes shifted
    toward both cores, and a sum whose two equilibria differ in sign raises.
    """
    gate, shifts = {}, {}
    for total in (-3, -1, 1, 3):
        shifts[total] = [axis_shift(equilibrium(total * UNIT, seed)) for seed in SEEDS]
        signs = set(np.sign(shifts[total]))
        if len(signs) != 1:
            message = f"the ripple of {total} inputs leaves both cores stable"
            raise RuntimeError(message)
        gate[total] = int(signs.pop())
    return gate, shifts


def majority(gate, a, b, c):
    """The certified gate on bits of +1 and -1."""
    lookup = np.array([gate[-3], gate[-1], gate[1], gate[3]])
    return lookup[(np.asarray(a) + b + c + 3) // 2]


def rule_110(gate, left, centre, right):
    """The next state of cells of Rule 110 from four majority gates.

    The rule is (centre or right) and not (left and centre and right). A constant input
    of +1 makes a gate an or, and one of -1 an and.
    """
    either = majority(gate, centre, right, 1)
    both = majority(gate, left, centre, -1)
    all_three = majority(gate, both, right, -1)
    return majority(gate, either, -all_three, -1)


def by_number(bits):
    """The next generation of rows of cells of 0 and 1 by the rule number 110."""
    index = 4 * np.roll(bits, 1, -1) + 2 * bits + np.roll(bits, -1, -1)
    return (110 >> index) & 1


def evolve(gate, cells=64, generations=32):
    """The generations of a ring of cells from one live cell, as [generation, cell]."""
    state = -np.ones(cells, dtype=int)
    state[-1] = 1
    rows = [state]
    for _ in range(generations):
        state = rule_110(gate, np.roll(state, 1), state, np.roll(state, -1))
        rows.append(state)
    return np.array(rows)


def main():
    gate, shifts = certify()
    for total, (from_plus, from_minus) in shifts.items():
        print(
            f"ripple of {total:+d} inputs: axis shift {from_plus:+.4f} from an axis"
            f" started at {SEEDS[0]:+}, {from_minus:+.4f} from one at {SEEDS[1]:+}"
        )
    for bit in (1, -1):
        written = equilibrium(bit * UNIT)
        held = equilibrium(0.0, restart_from=written)
        print(
            f"written with {bit:+d} input: axis shift {axis_shift(written):+.4f},"
            f" {axis_shift(held):+.4f} once the input is gone"
        )
    rows = evolve(gate)
    for row in rows:
        print("".join("#" if cell > 0 else "." for cell in row))
    bits = (rows > 0).astype(int)
    agree = np.array_equal(bits[1:], by_number(bits[:-1]))
    print(f"{len(rows) - 1} generations of Rule 110 from the gate: {agree}")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Add numbers and run Rule 110 on kink parametrons.

A kink parametron is a VMEC++ equilibrium of a spherical tokamak that holds one of its
two helical cores.
"""

import numpy as np

import vmecpp

NS, MPOL, NTOR = 49, 6, 1
# pres_scale above the kink threshold of 2.53e4 on 25 surfaces
PRESSURE = 2.8e4
# the m = 1, n = 1 boundary ripple of one input
UNIT = 2.0e-2
# a ring of eight cells with each neighbourhood of three cells once
DE_BRUIJN = (0, 0, 0, 1, 0, 1, 1, 1)


def spherical_tokamak(ripple):
    """The VMEC++ input of the spherical tokamak with the boundary ripple ``ripple``."""
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
        raxis_c=np.array([1.5, 0.0]),
        zaxis_s=np.array([0.0, 0.0]),
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


def equilibrium(ripple, restart_from=None):
    """The VMEC++ equilibrium of the spherical tokamak with the boundary ripple
    ``ripple``."""
    return vmecpp.run(
        spherical_tokamak(ripple), verbose=False, restart_from=restart_from
    )


class Parametron:
    """A kink parametron and the equilibrium of the core it holds."""

    def __init__(self):
        self.state = None

    def write(self, ripple):
        """Write the core that ``ripple`` selects, hold it and return its axis shift."""
        written = equilibrium(ripple, restart_from=self.state)
        self.state = equilibrium(0.0, restart_from=written)
        return axis_shift(self.state)


def certify():
    """The held level and the axis shifts that each sum of three inputs writes over both
    held cores."""
    held = {}
    for bit in (1, -1):
        held[bit] = Parametron()
        held[bit].write(bit * UNIT)
    writes = {}
    for total in (-3, -1, 1, 3):
        writes[total] = [
            axis_shift(equilibrium(total * UNIT, restart_from=held[bit].state))
            for bit in (1, -1)
        ]
        if len(set(np.sign(writes[total]))) != 1:
            message = f"the ripple of {total} inputs leaves both cores stable"
            raise RuntimeError(message)
    return axis_shift(held[1].state), writes


def majority(gate, level, a, b, c):
    """The axis shift that the inputs ``a``, ``b`` and ``c`` write in ``gate``."""
    return gate.write(UNIT / level * (a + b + c))


def full_adder(gates, level, a, b, carry):
    """The axis shifts of the sum and the carry of three inputs on ``gates``."""
    out = majority(gates[0], level, a, b, carry)
    inner = majority(gates[1], level, a, b, -carry)
    return majority(gates[2], level, -out, carry, inner), out


def add(level, x, y, bits):
    """The sum of ``x`` and ``y`` on ``bits`` full adders."""
    carry, total = -level, 0
    for k in range(bits):
        a, b = (level * (2 * (number >> k & 1) - 1) for number in (x, y))
        s, carry = full_adder([Parametron() for _ in range(3)], level, a, b, carry)
        total |= int(s > 0) << k
    return total | int(carry > 0) << bits


def rule_110(cells, level, shifts):
    """The axis shifts of the next generation of Rule 110 on four parametrons a cell."""
    new = np.empty(len(shifts))
    for k, (either, both, all_three, cell) in enumerate(cells):
        left, centre, right = shifts[k - 1], shifts[k], shifts[(k + 1) % len(shifts)]
        any_of_two = majority(either, level, centre, right, level)
        first_two = majority(both, level, left, centre, -level)
        all_of_three = majority(all_three, level, first_two, right, -level)
        new[k] = majority(cell, level, any_of_two, -all_of_three, -level)
    return new


def by_number(bits):
    """The next generation of rows of cells of 0 and 1 by the rule number 110."""
    index = 4 * np.roll(bits, 1, -1) + 2 * bits + np.roll(bits, -1, -1)
    return (110 >> index) & 1


def evolve(level, bits, generations):
    """The [generation, cell] axis shifts of a ring of Rule 110 from ``bits``."""
    cells = [[Parametron() for _ in range(4)] for _ in bits]
    shifts = np.array(
        [
            cell[3].write(UNIT * (2 * bit - 1))
            for cell, bit in zip(cells, bits, strict=True)
        ]
    )
    rows = [shifts]
    for _ in range(generations):
        shifts = rule_110(cells, level, shifts)
        rows.append(shifts)
    return np.array(rows)


def main():
    level, writes = certify()
    for total, (over_plus, over_minus) in writes.items():
        print(
            f"ripple of {total:+d} inputs: axis shift {over_plus:+.4f} written over the"
            f" held core at {level:+.4f}, {over_minus:+.4f} over the one at"
            f" {-level:+.4f}"
        )
    total = add(level, 7, 9, 4)
    print(f"7 + 9 on four bits of parametrons: {total}")
    rows = evolve(level, DE_BRUIJN, 2)
    for row in rows:
        print("".join("#" if shift > 0 else "." for shift in row))
    bits = (rows > 0).astype(int)
    if total != 7 + 9 or not np.array_equal(bits[1:], by_number(bits[:-1])):
        message = "the parametrons disagree with the sum or the rule number"
        raise RuntimeError(message)
    print(
        f"{len(rows) - 1} generations of Rule 110 on {4 * len(DE_BRUIJN)} parametrons"
        " agree with the rule number, and their cells hold axis shifts of"
        f" {np.abs(rows).min():.5f} to {np.abs(rows).max():.5f}"
    )


if __name__ == "__main__":
    main()

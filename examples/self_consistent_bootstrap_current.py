# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""A bootstrap current made self-consistent during the solve, from Python.

``vmecpp.solve_equilibrium`` runs the force-balance iteration in Python and
hands every iteration to a callback. Here the callback replaces the enclosed
toroidal current, every 25 iterations, by the bootstrap current of SIMSOPT's
Redl closure on the iterating field, read through
``VmecModel.half_grid_fields``, and prescribes it through ``VmecModel.curr_h``.
The solve ends when the forces are balanced and the closure no longer changes
the current.

The Redl closure holds for tokamaks and quasi-symmetric fields. The
configuration is the quasi-helical warm start of the SIMSOPT examples, whose
pressure is set here from the kinetic profiles.
"""

from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial
from simsopt.mhd.bootstrap import compute_trapped_fraction, j_dot_B_Redl
from simsopt.mhd.profiles import ProfilePolynomial

import vmecpp

MU_0 = 4.0e-7 * np.pi
ELEMENTARY_CHARGE = 1.602176634e-19


def kinetic_pressure(ne, te, ti, zeff):
    """Power-series coefficients in s of p = e (n_e T_e + n_i T_i), n_i = n_e / Z_eff,
    in Pa, from the power-series coefficients of n_e in m^-3 and T_e, T_i in eV."""
    return ELEMENTARY_CHARGE * polynomial.polyadd(
        polynomial.polymul(ne, te), polynomial.polymul(ne, ti) / zeff
    )


def full_surfaces(values, fields):
    """Half-grid values, [ns - 1, nzeta * ntheta_eff], on the full poloidal grid as
    [ntheta, nzeta, ns - 1], the layout of SIMSOPT's compute_trapped_fraction.

    Without lasym the solver stores theta in [0, pi], and the rest of each surface
    follows from f(theta, zeta) = f(-theta, -zeta).
    """
    stored = np.asarray(values).reshape(-1, fields.nzeta, fields.ntheta_eff)
    if fields.ntheta_eff == fields.ntheta_even:
        full = stored
    else:
        theta = np.arange(fields.ntheta_even)
        mirrored = theta >= fields.ntheta_eff
        full = np.empty((stored.shape[0], fields.nzeta, fields.ntheta_even))
        full[:, :, ~mirrored] = stored[:, :, theta[~mirrored]]
        zeta_mirror = (fields.nzeta - np.arange(fields.nzeta)) % fields.nzeta
        theta_mirror = fields.ntheta_even - theta[mirrored]
        full[:, :, mirrored] = stored[:, zeta_mirror][:, :, theta_mirror]
    return np.transpose(full, (2, 1, 0))


def redl_current(fields, ne, te, ti, zeff, helicity_n, psi_edge):
    """The enclosed current, in the units of wout buco, that the Redl bootstrap current
    of the fields drives, and <J.B> on the half grid.

    mu_0 <J.B> = signgs (G I' - I G') / vp with G = bvco and I = buco, so
    (I / G)' = signgs mu_0 <J.B> vp / G^2, integrated from I(0) = 0.
    """
    modb = np.sqrt(
        np.asarray(fields.bsupu) * np.asarray(fields.bsubu)
        + np.asarray(fields.bsupv) * np.asarray(fields.bsubv)
    )
    _, _, epsilon, _, fsa_1overb, f_t = compute_trapped_fraction(
        full_surfaces(modb, fields), full_surfaces(fields.gsqrt, fields)
    )
    g = np.asarray(fields.bvco)
    i = np.asarray(fields.buco)
    iota = np.asarray(fields.iota)
    num_half = g.size
    s = (np.arange(num_half) + 0.5) / num_half
    j_dot_b, _ = j_dot_B_Redl(
        ne,
        te,
        ti,
        zeff,
        helicity_n,
        s=s,
        G=g,
        R=(g + iota * i) * fsa_1overb,
        iota=iota,
        epsilon=epsilon,
        f_t=f_t,
        psi_edge=psi_edge,
        nfp=fields.nfp,
    )
    integrand = fields.signgs * MU_0 * j_dot_b * np.asarray(fields.vp) / g**2
    delta_s = 1.0 / num_half
    i_over_g = 0.5 * delta_s * integrand[0] + np.concatenate(
        ([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * delta_s))
    )
    return g * i_over_g, j_dot_b


def solve_with_bootstrap_current(
    vmec_input,
    ne,
    te,
    ti,
    zeff,
    helicity_n,
    *,
    interval=25,
    relaxation=0.5,
    tolerance=1.0e-3,
    max_rounds=50,
):
    """Solve ``vmec_input`` with the enclosed current made equal to the Redl bootstrap
    current of the equilibrium.

    The multigrid solve with the input's current gives the starting state. On the
    final grid, the callback moves the current a fraction ``relaxation`` towards the
    closure every ``interval`` iterations once the invariant residual is below 1e-2,
    and a new round starts whenever the forces converge while the closure still
    changes the current by more than ``tolerance`` of its maximum. ``vmec_input``
    needs ncurr = 1. Returns the model and the number of force iterations on the
    final grid.
    """
    psi_edge = -vmec_input.phiedge / (2.0 * np.pi)
    model, results = vmecpp.solve_multigrid(vmec_input)
    if not results[-1].converged:
        msg = "the starting equilibrium did not converge"
        raise RuntimeError(msg)

    def relaxed_current():
        target, _ = redl_current(
            model.half_grid_fields(), ne, te, ti, zeff, helicity_n, psi_edge
        )
        current = np.asarray(model.curr_h)
        change = np.abs(target - current).max() / np.abs(target).max()
        return current + relaxation * (target - current), change

    def callback(state):
        if state.iteration % interval == 0 and state.fsq_invariant < 1.0e-2:
            model.curr_h, _ = relaxed_current()

    iterations = 0
    for _ in range(max_rounds):
        relaxed, change = relaxed_current()
        if change < tolerance:
            return model, iterations
        model.curr_h = relaxed
        result = vmecpp.solve_equilibrium(model, callback=callback)
        iterations += result.num_iterations
        if not result.converged:
            msg = "the equilibrium did not converge with the bootstrap current"
            raise RuntimeError(msg)
    msg = f"the bootstrap current did not settle in {max_rounds} rounds"
    raise RuntimeError(msg)


def with_current_profile(vmec_input, buco):
    """``vmec_input`` with the enclosed current prescribed as the piecewise-linear
    profile through the half-grid values ``buco``, which a solve reproduces."""
    num_half = buco.size
    s = (np.arange(num_half) + 0.5) / num_half
    current = 2.0 * np.pi * vmec_input.signgs * buco / MU_0
    edge = 1.5 * current[-1] - 0.5 * current[-2]
    return vmec_input.model_copy(
        update={
            "pcurr_type": "line_segment_i",
            "ac_aux_s": np.concatenate(([0.0], s, [1.0])),
            "ac_aux_f": np.concatenate(([0.0], current, [edge])),
            "curtor": edge,
        }
    )


def redl_mismatch(output, ne, te, ti, zeff, helicity_n, ntheta=64, nphi=65):
    """SIMSOPT's measure of how far an equilibrium is from carrying the Redl bootstrap
    current, as VmecRedlBootstrapMismatch forms it: the wout <J.B> against the Redl
    <J.B> of the Boozer-independent quantities it computes from the wout spectrum,
    normalized by sqrt(sum((<J.B>_vmec + <J.B>_Redl)^2))."""
    wout = output.wout
    ns = wout.ns
    s_half = (np.arange(1, ns) - 0.5) / (ns - 1)
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0, 2 * np.pi / wout.nfp, nphi, endpoint=False)
    phi2d, theta2d = np.meshgrid(phi, theta)
    angle = (
        np.asarray(wout.xm_nyq)[:, None, None] * theta2d[None]
        - np.asarray(wout.xn_nyq)[:, None, None] * phi2d[None]
    )
    cosangle = np.cos(angle)
    modb = np.einsum("mj,mtp->tpj", np.asarray(wout.bmnc)[:, 1:], cosangle)
    sqrtg = np.einsum("mj,mtp->tpj", np.asarray(wout.gmnc)[:, 1:], cosangle)
    _, _, epsilon, _, fsa_1overb, f_t = compute_trapped_fraction(modb, sqrtg)
    g = np.asarray(wout.bvco)[1:]
    i = np.asarray(wout.buco)[1:]
    iota = np.asarray(wout.iotas)[1:]
    j_dot_b_redl, _ = j_dot_B_Redl(
        ne,
        te,
        ti,
        zeff,
        helicity_n,
        s=s_half,
        G=g,
        R=(g + iota * i) * fsa_1overb,
        iota=iota,
        epsilon=epsilon,
        f_t=f_t,
        psi_edge=-np.asarray(wout.phi)[-1] / (2 * np.pi),
        nfp=wout.nfp,
    )
    j_dot_b_vmec = np.interp(s_half, np.linspace(0, 1, ns), np.asarray(wout.jdotb))
    denominator = np.sqrt(np.sum((j_dot_b_vmec + j_dot_b_redl) ** 2))
    return (j_dot_b_vmec - j_dot_b_redl) / denominator


def main():
    vmec_input = vmecpp.VmecInput.from_file(
        Path(__file__).parent / "data" / "input.nfp4_QH_warm_start"
    )
    ne_coefficients = 1.0e19 * np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    te_coefficients = 4.0e3 * np.array([1.0, -1.0])
    ti_coefficients = te_coefficients
    zeff = 1.0
    # quasi-helical with |B| a function of theta - helicity_n nfp zeta
    helicity_n = -1
    vmec_input = vmec_input.model_copy(
        update={
            "pmass_type": "power_series",
            "am": kinetic_pressure(
                ne_coefficients, te_coefficients, ti_coefficients, zeff
            ),
            "pres_scale": 1.0,
        }
    )
    ne = ProfilePolynomial(ne_coefficients)
    te = ProfilePolynomial(te_coefficients)
    ti = ProfilePolynomial(ti_coefficients)

    zero_current = vmecpp.run(vmec_input, verbose=False)
    model, iterations = solve_with_bootstrap_current(
        vmec_input, ne, te, ti, zeff, helicity_n
    )
    buco = np.asarray(model.curr_h)
    self_consistent = vmecpp.run(with_current_profile(vmec_input, buco), verbose=False)

    print(f"force iterations with the closure: {iterations}")
    print(f"volume-averaged beta: {self_consistent.wout.betatotal:.4f}")
    print(f"net bootstrap current: {self_consistent.wout.ctor:.1f} A")
    for name, output in [
        ("zero current", zero_current),
        ("bootstrap current", self_consistent),
    ]:
        iotaf = np.asarray(output.wout.iotaf)
        mismatch = redl_mismatch(output, ne, te, ti, zeff, helicity_n)
        print(
            f"{name}: iota axis {iotaf[0]:.4f}, mid {iotaf[iotaf.size // 2]:.4f}, "
            f"edge {iotaf[-1]:.4f}, Redl mismatch {np.sqrt(np.sum(mismatch**2)):.2e}"
        )


if __name__ == "__main__":
    main()

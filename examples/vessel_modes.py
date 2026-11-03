# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
r"""The electromagnetic modes of the vessel of a magnetic confinement device, from a
VMEC++ solve inside its wall, and their reach as a haloscope.

The cavity modes solve \nabla \times \nabla \times E + (\omega_{pe}^2 / c^2) b (b \cdot
E) = k^2 E with n \times E = 0 on the wall, for a thin plasma of electrons free along
the field, in covariant components on edge elements in x = \sqrt{s} and harmonics in the
angles. The fast wave is the parallel magnetic field \Psi of the compressional Alfven
wave in a hydrogen plasma, -\nabla \cdot (\epsilon^{-1} \nabla \Psi) = k^2 \Psi with
\epsilon = 1 + c^2 / v_A^2 and \Psi free on the wall. Reach is at signal to noise one
against a thermal line.
"""

import argparse
import functools
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import constants
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import eigsh

import vmecpp

MU0, C, EPS0 = constants.mu_0, constants.c, constants.epsilon_0
E, ME, MI, K_B = constants.e, constants.m_e, constants.m_p, constants.k
ETA0 = MU0 * C  # impedance of free space, ohm
HBAR_EV = constants.hbar / E  # eV s
HBARC_GEV_M = constants.hbar * C / (E * 1.0e9)  # GeV m
PE2_PER_DENSITY = E**2 / (ME * EPS0 * C**2)  # \omega_{pe}^2 / c^2 per density, 1 / m
RHO_GEV_CM3 = 0.45  # local density of the galactic halo
RHO = RHO_GEV_CM3 * 1.0e9 * E * 1.0e6  # the same in J / m^3
Q_HALO = 1.0e6  # frequency over the linewidth of the halo
CELLS = 48  # radial cells
HARMONICS = (6, 3)  # largest |m| and largest number of field periods in n
MODES = 12  # modes solved near a shift
EDGE_DENSITY = 1.0e-4  # density outside the plasma over its peak
FAST_BAND = 1.5  # fast-wave modes compared, up to this multiple of the lowest k
POLARIZATION = np.sqrt(1.0 / 3.0)  # rms of a random polarization along a direction
B_HALO = np.sqrt(2.0 * RHO * MU0)  # field with the energy density of the halo, T
# \kappa / g for the axion current \kappa B, with \kappa = g \sqrt{2 \rho}, in GeV / m
KAPPA_PER_G = np.sqrt(2.0 * RHO_GEV_CM3 * (HBARC_GEV_M * 100.0) ** 3) / HBARC_GEV_M
WALL_ABSORPTION = 0.026  # mean absorption of diffuse waves in the empty W7-X vessel
WALL_FREQUENCY = 140.0e9  # the frequency of that measurement, Hz
Q_MARGIN = 0.5  # share of the wall's Q left by in-vessel parts at long wavelengths
T_WALL = 310.0  # K
T_AMPLIFIER = 59.0  # K, of a 0.8 dB amplifier
T_THIN = 5.0 * E / K_B  # electrons of a thin plasma, K
T_HOT = 1000.0 * E / K_B  # electrons of a hot plasma, K
COLLISIONS = 1.0e5  # electron collisions of a thin plasma with neutrals at 5e-3 Pa, 1/s
TRACK_START = 1.0e13  # the density from which the mode is followed by doubling, 1/m^3


def vessel_input(plasma, gap):
    """The current-free input whose boundary is the plasma boundary moved out by gap
    equivalent minor radii, along the normal in each cross-section, at the field
    strength of the plasma; an axisymmetric vessel takes a small transform that nests
    its surfaces."""
    nfp, mpol, ntor = int(plasma.nfp), int(plasma.mpol), int(plasma.ntor)
    lasym = plasma.lasym
    points_v = 4 * ntor + 16 if ntor else 1
    theta = 2.0 * np.pi * np.arange(4 * mpol + 16) / (4 * mpol + 16)
    zeta = 2.0 * np.pi / nfp * np.arange(points_v) / points_v
    m = np.arange(mpol)[:, None, None, None]
    n = np.arange(-ntor, ntor + 1)[None, :, None, None]
    angle = m * theta[None, None, None, :] - n * nfp * zeta[None, None, :, None]
    cos, sin = np.cos(angle), np.sin(angle)
    rbc, zbs = plasma.rbc[..., None, None], plasma.zbs[..., None, None]
    rbs = plasma.rbs[..., None, None] if lasym else 0.0 * rbc
    zbc = plasma.zbc[..., None, None] if lasym else 0.0 * zbs
    r = np.sum(rbc * cos + rbs * sin, (0, 1))
    z = np.sum(zbs * sin + zbc * cos, (0, 1))
    r_t = np.sum(m * (rbs * cos - rbc * sin), (0, 1))
    z_t = np.sum(m * (zbs * cos - zbc * sin), (0, 1))
    area = np.mean(r * z_t - z * r_t, axis=1) * np.pi
    minor = np.sqrt(np.mean(np.abs(area)) / np.pi)
    length = np.sign(area)[:, None] * np.hypot(r_t, z_t)
    r, z = r + gap * minor * z_t / length, z - gap * minor * r_t / length
    weight = np.where((m == 0) & (n == 0), 1.0, 2.0) * ((m > 0) | (n >= 0))

    def fourier(values, basis):
        return np.sum(weight * values * basis, (2, 3)) / r.size

    def axis(coefficients):
        return np.eye(1, ntor + 1)[0] * coefficients[0, ntor]

    return vmecpp.VmecInput(
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        lasym=lasym,
        ns_array=np.array([13, 25, 49]),
        ftol_array=np.array([1e-9, 1e-10, 1e-11]),
        niter_array=np.array([10000, 20000, 40000]),
        phiedge=float(plasma.phiedge) * (1.0 + gap) ** 2,
        am=np.array([0.0]),
        ncurr=int(ntor > 0),
        ai=np.array([0.05]),
        ac=np.array([0.0]),
        rbc=fourier(r, cos),
        zbs=fourier(z, sin),
        rbs=fourier(r, sin) if lasym else None,
        zbc=fourier(z, cos) if lasym else None,
        raxis_c=axis(fourier(r, cos)),
        zaxis_s=np.zeros(ntor + 1),
        raxis_s=np.zeros(ntor + 1) if lasym else None,
        zaxis_c=axis(fourier(z, cos)) if lasym else None,
    )


class Vessel:
    r"""Geometry and field of a VMEC++ solve inside the wall, scaled to a volume and
    mean field, on cells in x = \sqrt{s} and a grid of its angles over one field
    period."""

    def __init__(
        self, wout, edge, volume=None, field=None, cells=CELLS, kept=HARMONICS
    ):
        self.nfp = int(wout.nfp)
        self.axisymmetric = int(wout.ntor) == 0
        self.xm, self.xn = np.asarray(wout.xm, float), np.asarray(wout.xn, float)
        self.xm_nyq = np.asarray(wout.xm_nyq, float)
        self.xn_nyq = np.asarray(wout.xn_nyq, float)
        self.plasma_edge = edge
        self.cells, self.dx, self.kept = cells, 1.0 / cells, kept
        self.xc = self.dx * (np.arange(cells) + 0.5)
        scale = (volume / float(wout.volume_p)) ** (1.0 / 3.0) if volume else 1.0
        self.volume = float(wout.volume_p) * scale**3
        self.major = float(wout.Rmajor_p) * scale
        self.radius = np.sqrt(self.volume / (2.0 * np.pi**2 * self.major))
        field_scale = field / float(wout.volavgB) if field else 1.0
        ns = int(wout.ns)
        full = np.sqrt(np.arange(ns) / (ns - 1))
        half = np.sqrt((np.arange(1, ns) - 0.5) / (ns - 1))

        def splines(name, nodes, first, factor):
            """The cosine and sine series of a quantity in x, one zero without lasym."""
            c, s = (getattr(wout, f"{name}mn{kind}") for kind in "cs")
            zero = 0.0 * np.asarray(s if c is None else c)
            return [
                CubicSpline(nodes, np.asarray(part)[:, first:] * factor, axis=1)
                for part in (zero if c is None else c, zero if s is None else s)
            ]

        self._r = splines("r", full, 0, scale)
        self._z = splines("z", full, 0, scale)
        self._b = splines("b", half, 1, field_scale)
        # the contravariant components fall as the lengths grow, at a given |B|
        self._bsupu = splines("bsupu", half, 1, field_scale / scale)
        self._bsupv = splines("bsupv", half, 1, field_scale / scale)
        self.ntheta = 4 * kept[0] + 8
        self.nzeta = 8 if self.axisymmetric else 4 * kept[1] + 8
        self.per_point = 1.0 / (self.ntheta * self.nzeta)
        self.theta = 2.0 * np.pi * np.arange(self.ntheta) / self.ntheta
        self.zeta = 2.0 * np.pi / self.nfp * np.arange(self.nzeta) / self.nzeta
        self.grids = [self._grid(x) for x in self.xc]
        self.wall = self._grid(1.0)
        # the volume over 4 \pi^2, the measure of the mass matrix
        self.measure = sum(np.sum(g["w"]) for g in self.grids) * self.dx
        weighted = sum(np.sum(g["w"] * g["b"]) for g in self.grids) * self.dx
        self.mean_field = weighted / self.measure
        self.area = 4.0 * np.pi**2 * np.mean(self.wall["surface"])

    def _grid(self, x):
        basis = self._trig(self.xm, self.xn)
        nyquist = self._trig(self.xm_nyq, self.xn_nyq)

        def total(cosine, sine, trig=basis):
            return np.tensordot(cosine, trig[0], 1) + np.tensordot(sine, trig[1], 1)

        rc, rs = self._r[0](x), self._r[1](x)
        zc, zs = self._z[0](x), self._z[1](x)
        m, n = self.xm, self.xn
        r = total(rc, rs)
        r_x = total(self._r[0](x, 1), self._r[1](x, 1))
        z_x = total(self._z[0](x, 1), self._z[1](x, 1))
        e = np.stack(
            [
                np.stack([r_x, 0.0 * r, z_x]),
                np.stack([total(rs * m, -rc * m), 0.0 * r, total(zs * m, -zc * m)]),
                np.stack([total(-rs * n, rc * n), r, total(-zs * n, zc * n)]),
            ]
        )
        b, bsupu, bsupv = (
            total(pair[0](x), pair[1](x), nyquist)
            for pair in (self._b, self._bsupu, self._bsupv)
        )
        # e_\theta x e_\zeta, e_\zeta x e_x and e_x x e_\theta
        cross = np.cross(np.roll(e, -1, axis=0), np.roll(e, -2, axis=0), axis=1)
        jacobian = np.sum(e[0] * cross[0], axis=0)
        return {
            "w": np.abs(jacobian) * self.per_point,
            "jacobian": np.abs(jacobian),
            "surface": np.linalg.norm(cross[0], axis=0),
            "e": e,
            "grad": cross / jacobian,
            "b": np.abs(b),
            "unit": (bsupu * e[1] + bsupv * e[2]) / np.abs(b),
            "phi": np.broadcast_to(self.zeta[:, None], r.shape),
        }

    def _trig(self, xm, xn):
        angle = (
            xm[:, None, None] * self.theta[None, None, :]
            - xn[:, None, None] * self.zeta[None, :, None]
        )
        return np.cos(angle), np.sin(angle)

    def density(self, x):
        """Density over its peak: 1 - x^4 inside the plasma, little outside it."""
        return np.maximum(1.0 - (x / self.plasma_edge) ** 4, EDGE_DENSITY)

    def harmonics(self, nu0):
        """(m, n) of a family with toroidal number nu0 modulo the field periods."""
        m_max, spans = self.kept[0], 0 if self.axisymmetric else self.kept[1]
        toroidal = nu0 + self.nfp * np.arange(-spans, spans + 1)
        return np.array([(m, n) for m in range(-m_max, m_max + 1) for n in toroidal])

    def family(self, key):
        """The toroidal number modulo the field periods of a drive along z or x."""
        if key == "z":
            return 0
        return 1 if self.axisymmetric else 1 % self.nfp

    def direction(self, key, grid):
        r"""The drive along z, or along x in cylindrical components.

        With three periods or more x splits between the families of n = 1 and n = -1,
        two modes that make one real standing mode, whose overlap is \sqrt{2} that of
        the n = 1 phasor.
        """
        phi = grid["phi"]
        if key == "z":
            return np.stack([0.0 * phi, 0.0 * phi, 1.0 + 0.0 * phi])
        if self.axisymmetric or self.nfp > 2:
            phasor = np.exp(-1j * phi) / np.sqrt(2.0)
            return np.stack([phasor, -1j * phasor, 0.0 * phi])
        return np.stack([np.cos(phi), -np.sin(phi), 0.0 * phi])


class Forms:
    """Hermitian forms and sources in the harmonics of a family, added cell by cell as
    blocks between groups of unknowns, each one component at one radial position."""

    def __init__(self, vessel, nu0, total):
        modes = vessel.harmonics(nu0)
        self.m, n = modes[:, 0], modes[:, 1]
        phase = (
            self.m[None, None, :] * vessel.theta[None, :, None]
            - n[None, None, :] * vessel.zeta[:, None, None]
        )
        self.e = np.exp(1j * phase).reshape(-1, len(modes))
        self.h = self.e.conj().T
        self.d_t, self.d_v = 1j * self.m + 0j, -1j * n + 0j
        self.one = np.ones(len(modes)) + 0j
        self.dx, self.total, self.size = vessel.dx, total, len(modes)
        self.blocks, self.sources = {}, {}
        self.keep = np.ones(total * self.size, dtype=bool)

    def nodal(self, component, left, right, factor):
        """Table rows of a component linear between the unknowns at two nodes: component,
        unknown, value at the midpoint of the cell, and slope in x."""
        return [
            (component, left, 0.5 * factor, -factor / self.dx),
            (component, right, 0.5 * factor, factor / self.dx),
        ]

    def radial(self, component, left, right, factor):
        """Table rows of a component that is the x derivative between two nodes."""
        return [
            (component, left, -factor / self.dx, 0.0 * factor),
            (component, right, factor / self.dx, 0.0 * factor),
        ]

    def add(self, name, groups, table, tensor, dx=None):
        """Add the form of a tensor over a cell, its linear profiles integrated exactly
        at the tensor frozen on the cell."""
        dx = dx or self.dx
        store = self.blocks.setdefault(name, {})
        metric = {}
        for i, p, value_p, slope_p in table:
            for j, q, value_q, slope_q in table:
                pair = (min(i, j), max(i, j))
                if pair not in metric:
                    metric[pair] = (self.h * np.ravel(tensor[pair])) @ self.e
                profile = dx * np.outer(value_p.conj(), value_q)
                profile = profile + dx**3 / 12.0 * np.outer(slope_p.conj(), slope_q)
                key = (groups[p], groups[q])
                store[key] = store.get(key, 0.0) + profile * metric[pair]

    def add_source(self, name, groups, table, weights):
        """Add a source with the given weight on the components of the table."""
        shape = (self.total, self.size)
        target = self.sources.setdefault(name, np.zeros(shape, dtype=complex))
        parts = [self.h @ np.ravel(weight) for weight in weights]
        for i, p, value, _ in table:
            target[groups[p]] += value.conj() * parts[i] * self.dx

    def results(self):
        """The sparse forms and the sources on the unknowns kept."""
        zero = sp.csc_matrix((self.size, self.size))
        span = range(self.total)
        matrices = {
            name: sp.bmat(
                [
                    [store.get((i, j), zero if i == j else None) for j in span]
                    for i in span
                ],
                format="csc",
            )[self.keep][:, self.keep]
            for name, store in self.blocks.items()
        }
        return matrices, {k: v.ravel()[self.keep] for k, v in self.sources.items()}


def close_edges(forms, cells):
    r"""Keep E_\theta = 0 and m = 0 of E_\zeta on the axis, and no tangential E on the
    wall, on unknowns that run E_\theta, E_\zeta at node j as 3 j, 3 j + 1, and E_x in
    cell c as 3 c + 2."""
    forms.keep[: 2 * forms.size] = False
    forms.keep[forms.size + np.flatnonzero(forms.m == 0)] = True
    forms.keep[3 * cells * forms.size :] = False


@functools.cache
def assemble_cavity(vessel, nu0):
    r"""The curl-curl form a, mass m, density-weighted form d of E along the field and
    form w of the tangential \nabla \times E on the wall, with the sources along x, z
    and B, on the five unknowns of a cell: E_x, E_\theta at its nodes, E_\zeta at its
    nodes."""
    cells, per_point = vessel.cells, vessel.per_point
    forms = Forms(vessel, nu0, 3 * cells + 2)
    d_t, d_v, one = forms.d_t, forms.d_v, forms.one
    nodal, radial, none = forms.nodal, forms.radial, 0.0 * one
    field = [(0, 0, one, none), *nodal(1, 1, 2, one), *nodal(2, 3, 4, one)]
    curl = [*nodal(0, 3, 4, d_t), *nodal(0, 1, 2, -d_v), (1, 0, d_v, none)]
    curl += [*radial(1, 3, 4, -one), (2, 0, -d_t, none), *radial(2, 1, 2, one)]

    def cell_groups(c):
        return (3 * c + 2, 3 * c, 3 * c + 3, 3 * c + 1, 3 * c + 4)

    def covariant(grid):
        return np.sum(grid["e"][:, None] * grid["e"][None, :], axis=2)

    for c, grid in enumerate(vessel.grids):
        groups, w, grad = cell_groups(c), grid["w"], grid["grad"]
        along = np.sum(grid["unit"] * grad, axis=1)
        forms.add("a", groups, curl, covariant(grid) / grid["jacobian"] * per_point)
        forms.add("m", groups, field, np.sum(grad[:, None] * grad[None, :], axis=2) * w)
        shape = vessel.density(vessel.xc[c])
        forms.add("d", groups, field, along[:, None] * along[None] * w * shape)
        forms.add_source("b", groups, field, along * grid["b"] * w)
        for key in ("x", "z"):
            drive = np.sum(vessel.direction(key, grid) * grad, axis=1)
            forms.add_source(key, groups, field, drive * w)
    wall = vessel.wall
    forms.add(
        "w",
        cell_groups(cells - 1),
        [row for row in curl if row[0] != 0],
        covariant(wall) / wall["jacobian"] ** 2 * wall["surface"] * per_point,
        dx=1.0,
    )
    close_edges(forms, cells)
    return forms.results()


def assemble_fast(vessel, key, peak_density):
    """The stiffness a with the inverse permittivity of the plasma, its part across the
    surfaces p, mass m and the source along key, on piecewise linear radial elements
    with m = 0 alone on the axis and the wall node free."""
    forms = Forms(vessel, vessel.family(key), vessel.cells + 1)
    value = forms.nodal(0, 0, 1, forms.one)
    across = [*forms.radial(0, 0, 1, forms.one), *forms.nodal(1, 0, 1, forms.d_t)]
    gradient = [*across, *forms.nodal(2, 0, 1, forms.d_v)]
    for c, grid in enumerate(vessel.grids):
        groups, grad = (c, c + 1), grid["grad"]
        shape = vessel.density(vessel.xc[c])
        inverse = 1.0 / (1.0 + C**2 * MU0 * peak_density * shape * MI / grid["b"] ** 2)
        weight = grid["w"] * inverse
        tensor = np.sum(grad[:, None] * grad[None, :], axis=2) * weight
        forms.add("a", groups, gradient, tensor)
        forms.add("p", groups, across, tensor)
        forms.add("m", groups, value, grid["w"][None, None])
        across_b = np.cross(grid["unit"], grad, axisa=0, axisb=1, axisc=1)
        drive = vessel.direction(key, grid) * across_b
        forms.add_source("drive", groups, gradient, np.sum(drive, axis=1) * weight)
    forms.keep[np.flatnonzero(forms.m != 0)] = False
    return forms.results()


def modes_near(stiffness, mass, count, shift):
    """The k^2 nearest a shift, ascending, with their vectors of unit norm in the
    mass."""
    values, vectors = eigsh(
        stiffness, k=count, M=mass, sigma=shift, which="LM", tol=1e-8
    )
    order = np.argsort(values)
    return values[order], vectors[:, order]


def quadratic(matrix, vectors):
    """The value of a Hermitian form on each vector."""
    return np.sum(vectors.conj() * (matrix @ vectors), axis=0).real


def noise_field(f, q, temperature, seconds):
    r"""Peak amplitude of a mode of unit norm in the mass at signal to noise one: thermal
    energy k T in a Lorentzian of width f / q, read over the linewidth of the halo for a
    time. It gives the haloscope equation g^2 (\rho / m) B^2 V C q / 4 = k T
    \sqrt{\Delta\nu / t}, for a receiver at critical coupling with q unloaded."""
    density = MU0 * K_B * temperature / np.pi**3
    return np.sqrt(density * q / f) * (f / Q_HALO) ** 0.25 / seconds**0.25


def wall_resistance(f):
    r"""Surface resistance of the first wall in ohm, from the absorption 16 R_s / (3
    Z_0) of diffuse waves on a good conductor, scaled with the skin depth as
    \sqrt{f}."""
    return 3.0 / 16.0 * ETA0 * WALL_ABSORPTION * np.sqrt(f / WALL_FREQUENCY)


def cavity_modes(
    vessel, nu0, peak_density, shift, seconds=1800.0, margin=Q_MARGIN, count=MODES
):
    """The modes of a family nearest a shift in k^2: wave number, frequency, the Q from
    the wall and the plasma, the noise temperature and field, and the overlaps."""
    matrices, sources = assemble_cavity(vessel, nu0)
    pe2 = peak_density * PE2_PER_DENSITY
    stiffness = matrices["a"] + pe2 * matrices["d"]
    values, vectors = modes_near(stiffness, matrices["m"], count, shift)
    # the zero modes of the curl-curl form are gradients
    physical = values > 1.0e-3 * shift
    values, vectors = values[physical], vectors[:, physical]
    k = np.sqrt(values)
    f = C * k / (2.0 * np.pi)
    loss_wall = wall_resistance(f) * quadratic(matrices["w"], vectors)
    loss_wall = loss_wall / (margin * ETA0 * k**3)
    parallel = quadratic(matrices["d"], vectors)
    loss_plasma = COLLISIONS / (2.0 * np.pi * f) * pe2 * parallel / values
    q = 1.0 / (loss_wall + loss_plasma)
    temperature = (T_WALL * loss_wall + T_THIN * loss_plasma) * q + T_AMPLIFIER
    return {
        "k": k,
        "f": f,
        "q": q,
        "temperature": temperature,
        "parallel": parallel,
        "noise": noise_field(f, q, temperature, seconds),
        "overlaps": {n: np.abs(s.conj() @ vectors) for n, s in sources.items()},
    }


@functools.cache
def tracked_mode(vessel, peak_density):
    """The k^2 and the parallel share of the mode that couples best to an axion,
    followed from the empty vessel through densities that double from TRACK_START."""
    if peak_density <= 0.0:
        shift = 1.2 * (2.405 / vessel.radius) ** 2
    else:
        lower = 0.0
        if peak_density > TRACK_START:
            doublings = np.ceil(np.log2(peak_density / TRACK_START)) - 1.0
            lower = TRACK_START * 2.0**doublings
        k2, parallel = tracked_mode(vessel, lower)
        shift = 1.02 * k2 + (peak_density - lower) * PE2_PER_DENSITY * parallel
    rows = cavity_modes(vessel, 0, peak_density, shift)
    i = int(np.argmax(rows["overlaps"]["b"]))
    return float(rows["k"][i] ** 2), float(rows["parallel"][i])


def cavity_mode(vessel, peak_density=0.0, seconds=1800.0, margin=Q_MARGIN):
    """The mode that couples best to an axion at a peak density of a thin plasma, with
    its form factor and its reach in g, 1/GeV, then the modes near it that couple best
    to a dark photon along z and along x, their reach the mixing eps."""
    shift = 1.02 * tracked_mode(vessel, peak_density)[0]
    axion = cavity_modes(vessel, 0, peak_density, shift, seconds, margin)
    photon = cavity_modes(
        vessel, vessel.family("x"), peak_density, shift, seconds, margin
    )

    def best(rows, reach):
        i = int(np.argmin(reach))
        fields = ("k", "f", "q", "temperature", "parallel")
        return {name: float(rows[name][i]) for name in fields} | {
            "reach": float(reach[i])
        }

    def mixing(rows, name):
        return rows["noise"] / (
            rows["overlaps"][name] * B_HALO * POLARIZATION * rows["q"]
        )

    overlap = axion["overlaps"]["b"]
    coupling = axion["noise"] * axion["k"] / (overlap * KAPPA_PER_G * axion["q"])
    form = overlap[np.argmin(coupling)] ** 2 / (vessel.mean_field**2 * vessel.measure)
    return (
        best(axion, coupling) | {"form": float(form)},
        best(axion, mixing(axion, "z")),
        best(photon, mixing(photon, "x")),
    )


def fast_wave(vessel, key, peak_density, q=100.0, temperature=T_HOT, seconds=1800.0):
    """The frequency of the lowest compressional mode of the plasma that closes across
    the surfaces, and the frequency and reach for a dark photon along key of the best
    mode within FAST_BAND of it."""
    matrices, sources = assemble_fast(vessel, key, peak_density)
    speed = 1.0 / np.sqrt(1.0 + C**2 * MU0 * peak_density * MI / vessel.mean_field**2)
    # below every mode but the constant
    shift = 0.1 * (speed / vessel.radius) ** 2
    values, vectors = modes_near(matrices["a"], matrices["m"], 40, shift)
    across = quadratic(matrices["p"], vectors) / values
    closes = (values > shift) & (across > 0.5)
    k = np.sqrt(values[closes])
    drive = np.abs(sources["drive"].conj() @ vectors[:, closes]) / k
    f = C * k / (2.0 * np.pi)
    eps = noise_field(f, q, temperature, seconds) / (drive * B_HALO * POLARIZATION * q)
    i = int(np.argmin(np.where(k < FAST_BAND * k[0], eps, np.inf)))
    return {
        "lowest": float(f[0]),
        "f": float(f[i]),
        "k": float(k[i]),
        "eps": float(eps[i]),
        "across": float(across[closes][i]),
    }


def solve_vessel(plasma, gap=0.9, volume=None, field=None, cells=CELLS, kept=HARMONICS):
    wout = vmecpp.run(vessel_input(plasma, gap), verbose=False).wout
    return Vessel(wout, 1.0 / (1.0 + gap), volume, field, cells, kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    w7x = Path(__file__).parent / "data" / "w7x.json"
    parser.add_argument("input_file", nargs="?", type=Path, default=w7x)
    parser.add_argument("--gap", type=float, default=0.9, help="wall, in minor radii")
    parser.add_argument("--volume", type=float, help="chamber volume in m^3")
    parser.add_argument("--field", type=float, help="volavgB in T")
    parser.add_argument("--cells", type=int, default=CELLS, help="radial cells")
    parser.add_argument("--harmonics", type=int, nargs=2, default=HARMONICS)
    args = parser.parse_args()
    plasma = vmecpp.VmecInput.from_file(args.input_file)
    vessel = solve_vessel(
        plasma, args.gap, args.volume, args.field, args.cells, tuple(args.harmonics)
    )
    print(
        f"vessel {vessel.volume:.1f} m^3, wall {vessel.area:.0f} m^2, "
        f"R {vessel.major:.2f} m, mean field {vessel.mean_field:.2f} T"
    )
    print(
        "the cavity mode, by peak density: f MHz, m neV, form factor, Q, T_sys K, "
        "g 1/GeV; best along z: f MHz, eps; best along x: f MHz, eps"
    )
    for n0 in (0.0, 1e14, 3e14, 1e15):
        row, along_z, along_x = cavity_mode(vessel, n0)
        mass = HBAR_EV * 2.0 * np.pi * row["f"] * 1e9
        print(
            f"  {n0:.0e}  {row['f'] / 1e6:7.1f}  {mass:7.1f}  {row['form']:.3f}  "
            f"{row['q']:6.0f}  {row['temperature']:6.0f}  {row['reach']:.1e}  "
            f"{along_z['f'] / 1e6:7.1f} {along_z['reach']:.1e}  "
            f"{along_x['f'] / 1e6:7.1f} {along_x['reach']:.1e}"
        )
    print("the fast wave, by peak density and drive: lowest f MHz, best f MHz, reach")
    for n0, te in ((1e20, 1000.0), (1e19, 1000.0), (1e18, 300.0)):
        for key in ("x", "z"):
            row = fast_wave(vessel, key, n0, temperature=te * E / K_B)
            print(
                f"  {n0:.0e}  {key}  {row['lowest'] / 1e6:6.2f}  "
                f"{row['f'] / 1e6:6.2f}  {row['eps']:.1e}"
            )


if __name__ == "__main__":
    main()

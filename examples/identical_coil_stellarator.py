# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Solve a quasi-isodynamic stellarator of 80 identical coils with VMEC++ and compare it
with W7-X."""

import dataclasses

import numpy as np
from scipy.optimize import fminbound, root_scalar
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier

import vmecpp

NFP = 5
RADIUS = 1.372835450  # coil radius in m
# Every coil is one shape: a circle of radius RADIUS lifted along its normal by
# WAVE \sin(3 t - phase) at the angle t from its first axis. A phase turns the
# shape about the normal by a third of it.
WAVE = 0.461500738  # depth of the wave in m
CURRENT = 1.0e6  # coil current in A
# Centres in m of the coils from \phi = 0 to the middle of a field period. The first and
# the last coil are their own images under stellarator symmetry.
CENTRES = np.array(
    [
        [10.045640553092, 0.0, 0.0],
        [9.982198910317, 0.617337419242, 0.064038204647],
        [9.799339028323, 1.001089517756, 0.235402774567],
        [9.446929978235, 1.206716818925, 0.483418657874],
        [8.976535135923, 1.339372477461, 0.708685626304],
        [8.358888495586, 1.672378696316, 0.921633517025],
        [7.761242012231, 2.476724573269, 0.733758452002],
        [7.157959849666, 3.388217536496, 0.486969037413],
        [6.367517334004, 4.626272140965, 0.0],
    ]
)
# Unit normals of the same coils. A positive current drives the field at a centre along
# the normal.
NORMALS = np.array(
    [
        [0.0, 0.774361840642, 0.632743028218],
        [-0.172478640313, 0.784960511865, 0.595053034148],
        [-0.361828111818, 0.780022003146, 0.510535103697],
        [-0.530000358393, 0.712530451796, 0.459782530515],
        [-0.681140207913, 0.697184291476, 0.223566725798],
        [-0.380170668171, 0.904850481735, -0.191613853275],
        [-0.541073987613, 0.824093283178, -0.167657986835],
        [-0.587837157056, 0.776616626968, -0.226526138670],
        [-0.568042378750, 0.781843260169, -0.256999946444],
    ]
)
# Unit vectors in the planes of the same coils along which t = 0.
AXES = np.array(
    [
        [1.000000000000, 0.0, 0.0],
        [0.983166731200, 0.174163916857, 0.055227789436],
        [0.918963293280, 0.206321105502, 0.336062593915],
        [0.700669051610, 0.062550813484, 0.710739246031],
        [0.387362957967, 0.084044418171, 0.918088489509],
        [-0.213895884557, 0.115542132990, 0.969999260862],
        [-0.590422756497, -0.230278930965, 0.773545462506],
        [-0.752839753712, -0.422667012910, 0.504564070687],
        [-0.809016994374, -0.587785252294, 0.000000000001],
    ]
)
# Phases of the wave on the same coils in rad.
PHASES = np.array(
    [
        3.141592653590,
        2.730298969081,
        -2.866689897770,
        -2.209743506250,
        -2.518766340579,
        -2.134449652307,
        -1.946208605953,
        -1.808438546206,
        0.0,
    ]
)
PRESSURES = (8000.0, 16000.0, 24000.0)  # p_0 in Pa of the profile p_0 (1 - s)


@dataclasses.dataclass(frozen=True)
class Interior:
    """The tracing of a flux surface and the resolution of the VMEC++ run of its
    interior."""

    nfp: int
    axis: tuple[float, float]  # bracket of R of the magnetic axis at \phi = 0, Z = 0
    steps: int  # multiplies the RK4 steps of every trace
    planes: int  # planes per field period at which the surface is traced
    offset: float  # start of the traced surface outboard of the axis, in m
    mpol: int
    ntor: int

    @property
    def period(self):
        """Toroidal angle of a field period."""
        return 2.0 * np.pi / self.nfp


DESIGN = Interior(
    nfp=NFP, axis=(9.32, 9.44), steps=2, planes=96, offset=0.3, mpol=12, ntor=24
)
# W7-X in the standard configuration. Its boundary lies inside the 5/5 islands.
W7X = Interior(
    nfp=5, axis=(5.90, 6.00), steps=1, planes=48, offset=0.22, mpol=12, ntor=12
)


def coil(centre, normal, axis, phase):
    """The coil at centre about normal: a circle of radius RADIUS from its first axis
    along axis, lifted along normal by its wave at phase; a positive current turns about
    normal."""
    second = np.cross(normal, axis)
    t = 2.0 * np.pi * np.arange(64) / 64
    points = (
        centre
        + RADIUS * (np.cos(t)[:, None] * axis + np.sin(t)[:, None] * second)
        + WAVE * np.sin(3 * t - phase)[:, None] * normal
    )
    harmonics = np.fft.rfft(points, axis=0) / len(t)
    curve = CurveXYZFourier(64, 3)
    # c_0 and then s_j, c_j of \sin(2 \pi j t) and \cos(2 \pi j t) in each coordinate
    curve.x = np.concatenate(
        [
            [harmonics[0, k].real]
            + [
                v
                for j in range(1, 4)
                for v in (-2.0 * harmonics[j, k].imag, 2.0 * harmonics[j, k].real)
            ]
            for k in range(3)
        ]
    )
    return Coil(curve, Current(CURRENT))


def coils():
    """The 80 coils of the design."""
    # Stellarator symmetry maps (x, y, z) to (x, -y, -z) and n to (-n_x, n_y, n_z).
    flip = np.array([1.0, -1.0, -1.0])
    centres = np.concatenate([CENTRES, flip * CENTRES[1:-1]])
    normals = np.concatenate([NORMALS, -flip * NORMALS[1:-1]])
    axes = np.concatenate([AXES, flip * AXES[1:-1]])
    # The image of a coil carries its wave at the opposite phase.
    phases = np.concatenate([PHASES, -PHASES[1:-1]])
    out = []
    for k in range(NFP):
        cos, sin = np.cos(2.0 * np.pi * k / NFP), np.sin(2.0 * np.pi * k / NFP)
        turn = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        out += [
            coil(turn @ c, turn @ n, turn @ a, p)
            for c, n, a, p in zip(centres, normals, axes, phases, strict=True)
        ]
    return out


def w7x_coils():
    """The 50 coils of W7-X in the standard configuration."""
    try:  # SIMSOPT 1.11 and later
        from simsopt.configs import get_data  # noqa: PLC0415

        curves, currents = get_data("w7x")[:2]
    except ImportError:
        from simsopt.configs import get_w7x_data  # noqa: PLC0415

        curves, currents, _ = get_w7x_data()
    return coils_via_symmetries(curves, currents, 5, True)


class FieldLines:
    """Field lines of a field traced by RK4 in the toroidal angle."""

    def __init__(self, field, period):
        self.field, self.period = field, period

    def b(self, r, phi, z):
        """Cylindrical components of B at the points (r, phi, z)."""
        cos, sin = np.cos(phi), np.sin(phi)
        self.field.set_points(np.ascontiguousarray(np.stack([r * cos, r * sin, z], -1)))
        b = np.asarray(self.field.B())
        return b[:, 0] * cos + b[:, 1] * sin, -b[:, 0] * sin + b[:, 1] * cos, b[:, 2]

    def trace(self, r0, z0, planes, periods, steps):
        """R and Z of the line through (r0, 0, z0) at planes planes a period for periods
        periods."""
        r, z, phi = np.atleast_1d(float(r0)), np.atleast_1d(float(z0)), 0.0
        out = [(r[0], z[0])]
        h = self.period / planes / steps  # steps RK4 steps between two planes

        def slope(r, phi, z):
            b_r, b_phi, b_z = self.b(r, np.full_like(r, phi), z)
            return r * b_r / b_phi, r * b_z / b_phi

        for _ in range(periods * planes):
            for _ in range(steps):
                k1 = slope(r, phi, z)
                k2 = slope(r + 0.5 * h * k1[0], phi + 0.5 * h, z + 0.5 * h * k1[1])
                k3 = slope(r + 0.5 * h * k2[0], phi + 0.5 * h, z + 0.5 * h * k2[1])
                k4 = slope(r + h * k3[0], phi + h, z + h * k3[1])
                r = r + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
                z = z + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
                phi += h
            out.append((r[0], z[0]))
        out = np.asarray(out)
        return out[:, 0], out[:, 1]


def magnetic_axis(interior, lines):
    """R of the magnetic axis at \\phi = 0, Z = 0, its crossings of the planes of a
    period, and the rotational transform next to it."""
    steps = interior.steps
    # Stellarator symmetry puts the axis at Z = 0 in the middle of the period.
    r_axis = root_scalar(
        lambda r: lines.trace(r, 0.0, 2, 1, 80 * steps)[1][1],
        bracket=interior.axis,
        method="brentq",
        xtol=1e-12,
    ).root
    r, z = lines.trace(r_axis, 0.0, interior.planes, 1, 8 * steps)
    axis = np.stack([r[: interior.planes], z[: interior.planes]], -1)
    periods = 150
    r, z = lines.trace(r_axis + 1e-3, 0.0, 1, periods, 240 * steps)
    angle = np.unwrap(np.arctan2(z, r - r_axis))
    return r_axis, axis, (angle[-1] - angle[0]) / (periods * interior.period)


def flux_surface(interior, lines, r_axis, periods=120, harmonics=10):
    """R and Z of the surface traced from offset outboard of the axis on a uniform grid
    of its straight-field-line angle, its rotational transform and its misfit."""
    planes = interior.planes
    r, z = lines.trace(
        r_axis + interior.offset, 0.0, planes, periods, 4 * interior.steps
    )
    plane = np.arange(len(r)) % planes
    phi = np.arange(len(r)) * interior.period / planes

    def fit(iota, harmonics):
        theta = iota * phi
        coef = np.empty((2, planes, 2 * harmonics - 1))
        worst = 0.0
        for k in range(planes):
            sel = plane == k
            basis = np.column_stack(
                [np.ones(sel.sum())]
                + [
                    f(m * theta[sel])
                    for m in range(1, harmonics)
                    for f in (np.cos, np.sin)
                ]
            )
            for i, target in enumerate((r[sel], z[sel])):
                coef[i, k], *_ = np.linalg.lstsq(basis, target, rcond=None)
                worst = max(worst, np.abs(basis @ coef[i, k] - target).max())
        return coef, worst

    # The angle about the axis in the planes a period apart gives a first estimate. A
    # scan around it and a bounded search refine it.
    first = plane == 0
    angle = np.unwrap(np.arctan2(z[first], r[first] - r_axis))
    iota0 = (angle[-1] - angle[0]) / phi[first][-1]
    scan = iota0 + np.linspace(-0.02, 0.02, 41)
    best = int(np.argmin([fit(x, 4)[1] for x in scan]))
    step = scan[1] - scan[0]
    iota = fminbound(
        lambda x: fit(x, 4)[1], scan[best] - step, scan[best] + step, xtol=1e-10, disp=0
    )
    coef, worst = fit(iota, harmonics)
    theta = 2.0 * np.pi * np.arange(96) / 96
    basis = np.column_stack(
        [np.ones_like(theta)]
        + [f(m * theta) for m in range(1, harmonics) for f in (np.cos, np.sin)]
    )
    return coef[0] @ basis.T, coef[1] @ basis.T, iota, worst


def toroidal_flux(lines, rb, zb, axis, planes):
    """The toroidal flux through the cross-section of the surface in each plane."""
    flux = np.empty(len(planes))
    for k, phi in enumerate(planes):
        p1 = np.stack([rb[k], zb[k]], -1)
        p2 = np.roll(p1, -1, axis=0)
        d1, d2 = p1 - axis[k], p2 - axis[k]
        area = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        area *= np.sign(area.sum())
        # B_\phi at the edge midpoints of each triangle of the fan from the axis
        mid = np.concatenate([axis[k] + 0.5 * d1, axis[k] + 0.5 * d2, 0.5 * (p1 + p2)])
        b_phi = lines.b(mid[:, 0], np.full(len(mid), phi), mid[:, 1])[1]
        flux[k] = np.sum(area * b_phi.reshape(3, -1).mean(axis=0))
    return flux


def fourier(values, zeta, theta, mpol, ntor, parity):
    """Coefficients [m, ntor + n] of values[plane, theta] in \\cos (parity 0) or
    \\sin (parity 1) of m \\theta - n \\zeta."""
    out = np.zeros((mpol, 2 * ntor + 1))
    basis = np.cos if parity == 0 else np.sin
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n <= 0 and not (n == 0 and parity == 0):
                continue
            weight = (1.0 if (m == 0 and n == 0) else 2.0) / values.size
            out[m, ntor + n] = weight * np.sum(
                values * basis(m * theta[None, :] - n * zeta[:, None])
            )
    return out


def vmec_input(interior, rb, zb, axis, flux, planes):
    """The fixed-boundary input of the interior of the traced surface."""
    mpol, ntor = interior.mpol, interior.ntor
    zeta = planes * interior.nfp
    theta = 2.0 * np.pi * np.arange(rb.shape[1]) / rb.shape[1]
    return vmecpp.VmecInput(
        nfp=interior.nfp,
        lasym=False,
        mpol=mpol,
        ntor=ntor,
        nzeta=2 * ntor + 8,
        ns_array=np.array([13, 25]),
        ftol_array=np.array([1e-10, 1e-12]),
        niter_array=np.array([30000, 60000]),
        phiedge=float(abs(flux.mean())),
        ncurr=1,
        pcurr_type="power_series",
        ac=np.array([0.0]),
        curtor=0.0,
        pmass_type="power_series",
        am=np.array([1.0, -1.0]),
        pres_scale=0.0,
        gamma=0.0,
        rbc=fourier(rb, zeta, theta, mpol, ntor, 0),
        zbs=fourier(zb, zeta, theta, mpol, ntor, 1),
        # The axis is R = \sum raxis_c \cos(n \zeta), Z = -\sum zaxis_s \sin(n \zeta).
        raxis_c=fourier(axis[:, :1], zeta, np.zeros(1), 1, ntor, 0)[0, ntor:],
        zaxis_s=fourier(axis[:, 1:], zeta, np.zeros(1), 1, ntor, 1)[0, ntor:],
    )


def interior(field, settings):
    """The traced magnetic axis and boundary, their rotational transform and the input
    of VMEC++ for the interior of the boundary."""
    lines = FieldLines(field, settings.period)
    r_axis, axis, iota_axis = magnetic_axis(settings, lines)
    planes = np.arange(settings.planes) * settings.period / settings.planes
    rb, zb, iota_surface, misfit = flux_surface(settings, lines, r_axis)
    flux = toroidal_flux(lines, rb, zb, axis, planes)
    return {
        "r_axis": r_axis,
        "axis": axis,
        "iota_axis": iota_axis,
        "planes": planes,
        "iota_surface": iota_surface,
        "misfit": misfit,
        "input": vmec_input(settings, rb, zb, axis, flux, planes),
    }


def axis_curve(wout, phi):
    """R and Z [phi, 2] of the magnetic axis of a wout at the toroidal angles phi."""
    angle = -np.asarray(wout.xn, dtype=float)[:, None] * np.asarray(phi)[None, :]
    return np.stack(
        [
            np.asarray(wout.rmnc)[:, 0] @ np.cos(angle),
            np.asarray(wout.zmns)[:, 0] @ np.sin(angle),
        ],
        -1,
    )


def effective_ripple(strength, size, drift, weight, major_radius, levels=4000):
    """The effective ripple of a field line from B, |\\nabla \\psi|, the radial drift
    and ds / B at its samples."""
    # Nemov et al., Phys. Plasmas 6, 4622 (1999). B_0 in b = B / B_0 is the mean of B
    # weighted by ds / B.
    eta = strength * np.sum(weight) / np.sum(weight * strength)
    step_weight = 0.5 * (weight[:-1] + weight[1:])
    step_size = 0.5 * (weight[:-1] * size[:-1] + weight[1:] * size[1:])
    scale = (
        np.pi
        * major_radius**2
        / (8.0 * np.sqrt(2.0))
        * np.sum(step_weight)
        / np.sum(step_size) ** 2
    )
    change = -np.diff(eta)
    flat = np.abs(change) < 1.0e-14
    width = (eta.max() - eta.min()) / levels
    total = 0.0
    # a midpoint sum over the levels b'
    for level in eta.min() + width * (np.arange(levels) + 0.5):
        below = np.maximum(level - eta, 0.0)
        # the mean of \sqrt{\max(b' - b, 0)} over each step
        root = np.where(
            flat,
            np.sqrt(below[:-1]),
            2.0 / 3.0 * np.diff(below**1.5) / np.where(flat, 1.0, change),
        )
        wet = root > 0.0
        starts = np.flatnonzero(wet[1:] & ~wet[:-1]) + 1
        stops = np.flatnonzero(~wet[1:] & wet[:-1]) + 1
        # Wells cut by an end of the line are left out.
        if wet[0]:
            stops = stops[1:]
        count = min(len(starts), len(stops))
        if count == 0:
            continue
        factor = weight * (4.0 / eta - 1.0 / level) * drift
        h = np.concatenate([[0.0], np.cumsum(root * 0.5 * (factor[:-1] + factor[1:]))])
        i = np.concatenate([[0.0], np.cumsum(root * step_weight)])
        a, b = starts[:count], stops[:count]
        well_h, well_i = h[b] - h[a], (i[b] - i[a]) / np.sqrt(level)
        total += width * np.sum(well_h**2 / (level**2 * well_i))
    return float((scale * total) ** (2.0 / 3.0))


def wout_ripple(wout, js, periods=120, steps=96):
    """The effective ripple of the half-grid surface js of a wout along the field line
    from \\theta = 0, \\phi = 0."""
    period = 2.0 * np.pi / int(wout.nfp)
    m = np.asarray(wout.xm_nyq, dtype=float)
    n = np.asarray(wout.xn_nyq, dtype=float)
    b_theta_up = np.asarray(wout.bsupumnc)[:, js]
    b_phi_up = np.asarray(wout.bsupvmnc)[:, js]

    def slope(theta, phi):
        c = np.cos(m * theta - n * phi)
        return (b_theta_up @ c) / (b_phi_up @ c)

    count = periods * steps
    h = period / steps
    theta = np.zeros(count + 1)
    for k in range(count):
        phi = k * h
        k1 = slope(theta[k], phi)
        k2 = slope(theta[k] + 0.5 * h * k1, phi + 0.5 * h)
        k3 = slope(theta[k] + 0.5 * h * k2, phi + 0.5 * h)
        k4 = slope(theta[k] + h * k3, phi + h)
        theta[k + 1] = theta[k] + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    phi = h * np.arange(count + 1)
    angle = m[:, None] * theta[None] - n[:, None] * phi[None]
    cos, sin = np.cos(angle), np.sin(angle)
    bmnc = np.asarray(wout.bmnc)[:, js]
    strength = bmnc @ cos
    jacobian = np.asarray(wout.gmnc)[:, js] @ cos
    # (B \times \nabla B) \cdot \nabla s / B^2
    #     = (B_\theta \partial_\phi B - B_\phi \partial_\theta B) / (\sqrt{g} B^2)
    drift = (
        (np.asarray(wout.bsubumnc)[:, js] @ cos) * ((n * bmnc) @ sin)
        + (np.asarray(wout.bsubvmnc)[:, js] @ cos) * ((m * bmnc) @ sin)
    ) / (jacobian * strength**2)
    # |\nabla s| from R and Z on the full-grid surfaces around js
    m = np.asarray(wout.xm, dtype=float)
    n = np.asarray(wout.xn, dtype=float)
    angle = m[:, None] * theta[None] - n[:, None] * phi[None]
    cos_full, sin_full = np.cos(angle), np.sin(angle)
    rmnc = np.asarray(wout.rmnc)[:, js - 1 : js + 1].mean(axis=1)
    zmns = np.asarray(wout.zmns)[:, js - 1 : js + 1].mean(axis=1)
    r = rmnc @ cos_full
    r_theta, r_phi = -(m * rmnc) @ sin_full, (n * rmnc) @ sin_full
    z_theta, z_phi = (m * zmns) @ cos_full, -(n * zmns) @ cos_full
    size = np.sqrt(
        (r_theta**2 + z_theta**2) * (r_phi**2 + z_phi**2 + r**2)
        - (r_theta * r_phi + z_theta * z_phi) ** 2
    ) / np.abs(jacobian)
    weight = h / np.abs(b_phi_up @ cos)
    # R_0 is the length of the magnetic axis over 2 \pi.
    turn = 2.0 * np.pi * np.arange(4097) / 4096
    axis = axis_curve(wout, turn)
    axis = np.stack(
        [axis[:, 0] * np.cos(turn), axis[:, 0] * np.sin(turn), axis[:, 1]], -1
    )
    major = np.linalg.norm(np.diff(axis, axis=0), axis=1).sum() / (2.0 * np.pi)
    return effective_ripple(strength, size, drift, weight, major)


def closure(strength):
    """Share of the range of B[\\theta, \\phi] whose contours close poloidally."""
    # Helical contours carry the minimum of B in \phi once through the period as \theta
    # goes around. Otherwise the contours close poloidally from the largest minimum of B
    # in \phi to the smallest maximum.
    count = strength.shape[1]
    lowest = strength.argmin(axis=1)
    step = np.diff(np.append(lowest, lowest[0]))
    if abs(np.sum((step + count // 2) % count - count // 2)) >= count // 2:
        return 0.0
    lowest_ridge = strength.max(axis=1).min()
    highest_valley = strength.min(axis=1).max()
    return float((lowest_ridge - highest_valley) / (strength.max() - strength.min()))


def wout_closure(wout, js, angles=128):
    """:func:`closure` of the half-grid surface js of a wout."""
    theta = 2.0 * np.pi * np.arange(angles) / angles
    phi = 2.0 * np.pi / int(wout.nfp) * np.arange(angles) / angles
    m = np.asarray(wout.xm_nyq, dtype=float)
    n = np.asarray(wout.xn_nyq, dtype=float)
    bmnc = np.asarray(wout.bmnc)[:, js]
    # \cos(m \theta - n \phi) = \cos m\theta \cos n\phi + \sin m\theta \sin n\phi
    poloidal, toroidal = m[None, :] * theta[:, None], n[None, :] * phi[:, None]
    strength = (np.cos(poloidal) * bmnc) @ np.cos(toroidal).T + (
        np.sin(poloidal) * bmnc
    ) @ np.sin(toroidal).T
    return closure(strength)


def measures(wout):
    """The values of s, the effective ripple and :func:`closure` on the half-grid
    surfaces ns // 4, ns // 2 and ns - 1 of a wout."""
    ns = int(wout.ns)
    surfaces = (ns // 4, ns // 2, ns - 1)
    return (
        [(js - 0.5) / (ns - 1) for js in surfaces],
        [wout_ripple(wout, js) for js in surfaces],
        [wout_closure(wout, js) for js in surfaces],
    )


def describe(s, ripple, closed):
    """:func:`measures` as text."""
    return (
        "effective ripple "
        + ", ".join(f"{100 * value:.2f}%" for value in ripple)
        + ", contours of B closing poloidally over "
        + ", ".join(f"{value:.2f}" for value in closed)
        + " of its range, at s = "
        + ", ".join(f"{value:.2f}" for value in s)
    )


def main():
    field = BiotSavart(coils())
    k = interior(field, DESIGN)
    print(
        f"80 coils of radius {RADIUS:.3f} m: magnetic axis through R = "
        f"{k['r_axis']:.5f} m on the midplane of phi = 0 with iota "
        f"{abs(k['iota_axis']):.4f} next to it, boundary traced from 30 cm outboard "
        f"of it with iota {abs(k['iota_surface']):.4f}, fitted within "
        f"{k['misfit']:.1e} m"
    )
    last = None
    for p0 in (0.0, *PRESSURES):
        step = k["input"].model_copy(update={"pres_scale": p0})
        if last is not None:
            # A hot restart takes one multigrid step.
            step = step.model_copy(
                update={
                    "ns_array": step.ns_array[-1:],
                    "ftol_array": np.array([1.0e-10]),
                    "niter_array": step.niter_array[-1:],
                }
            )
        last = vmecpp.run(step, max_threads=4, verbose=False, restart_from=last)
        wout = last.wout
        shift = np.hypot(*(axis_curve(wout, k["planes"]) - k["axis"]).T).max()
        iota = np.asarray(wout.iotaf)
        print(
            f"p0 = {p0:5.0f} Pa, beta {wout.betatotal:.1e}: {wout.itfsq} iterations, "
            f"magnetic axis {shift:.1e} m from the traced one, iota "
            f"{abs(iota[0]):.4f} on the axis and {abs(iota[-1]):.4f} on the boundary, "
            + describe(*measures(wout))
        )
    w7x = interior(BiotSavart(w7x_coils()), W7X)
    wout = vmecpp.run(w7x["input"], max_threads=4, verbose=False).wout
    print(
        f"W7-X, boundary traced from 22 cm outboard of its axis: {wout.itfsq} "
        "iterations, " + describe(*measures(wout))
    )


if __name__ == "__main__":
    main()

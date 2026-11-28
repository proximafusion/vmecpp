# SPDX-FileCopyrightText: 2026-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Solve a quasi-isodynamic stellarator of 50 identical coils with VMEC++ and compare it
with W7-X."""

import dataclasses

import numpy as np
from scipy.optimize import fminbound, root_scalar
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier

import vmecpp

NFP = 5
# Every coil is one 3D shape placed by rigid rotations and translations: at the angle t
# around the loop, the conductor lies at u(t) axis + v(t) (normal x axis) + w(t) normal
# in m from the centre, with Fourier coefficients U_COS, U_SIN, V_COS, V_SIN, W_COS,
# W_SIN for m = 1..6.
U_COS = (
    1.202136654,
    -0.041864268,
    0.029180006,
    -0.040242747,
    -0.024642407,
    -0.001061615,
)
U_SIN = (
    0.260231319,
    -0.071748680,
    0.079364237,
    0.029102220,
    0.033257913,
    -0.042465491,
)
V_COS = (
    0.004762313,
    0.190246166,
    -0.098843801,
    -0.051272167,
    0.006500505,
    0.008455956,
)
V_SIN = (
    1.078585540,
    0.177422666,
    0.037778199,
    -0.106602181,
    -0.021095647,
    0.006649826,
)
W_COS = (
    -0.013079373,
    -0.077737370,
    -0.114787891,
    0.080510233,
    -0.067035774,
    -0.027053472,
)
W_SIN = (
    -0.022317475,
    0.065512828,
    -0.069191181,
    -0.047063250,
    0.048316482,
    0.014272521,
)
CURRENT = 1.0e6  # coil current magnitude in A
# Centres in m of the 5 coils in the first half of a field period.
CENTRES = np.array(
    [
        [5.562399196168, 0.325275109451, -0.133873990556],
        [5.734748186238, 0.634784019040, 0.485108133957],
        [5.626083079389, 1.466512836823, 0.375488610382],
        [4.980522284795, 1.998617649453, 0.170164306079],
        [4.542287971878, 2.741943762784, -0.059433386818],
    ]
)
# Unit normals of the same coils.
NORMALS = np.array(
    [
        [-0.108925959126, 0.959522003054, 0.259716501371],
        [-0.393163247625, 0.882992738684, 0.256410772294],
        [-0.517044292001, 0.855698284828, -0.021110316216],
        [-0.599563167031, 0.777687304838, -0.189014456152],
        [-0.565706273545, 0.787710584237, -0.243902536993],
    ]
)
# Unit vectors in the planes of the same coils along which t = 0.
AXES = np.array(
    [
        [0.909767489318, 0.201512214734, -0.362926910951],
        [0.719382840786, 0.469080131513, -0.512300847748],
        [0.697264046804, 0.406751274208, -0.590234063711],
        [0.781972767688, 0.619530500390, 0.068560554839],
        [0.630794502043, 0.603886352917, 0.487257189739],
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
    nfp=NFP, axis=(5.85, 5.98), steps=2, planes=96, offset=0.185, mpol=12, ntor=24
)
# W7-X in the standard configuration. Its boundary lies inside the 5/5 islands.
W7X = Interior(
    nfp=5, axis=(5.90, 6.00), steps=1, planes=48, offset=0.22, mpol=12, ntor=12
)


def coil(centre, normal, axis, current=CURRENT):
    """The coil of the common 3D shape placed by ``centre``, ``normal`` and ``axis``."""
    second = np.cross(normal, axis)
    t = 2.0 * np.pi * np.arange(64) / 64
    orders = np.arange(1, len(U_COS) + 1)
    cos_mt = np.cos(np.outer(t, orders))
    sin_mt = np.sin(np.outer(t, orders))
    u = cos_mt @ np.array(U_COS) + sin_mt @ np.array(U_SIN)
    v = cos_mt @ np.array(V_COS) + sin_mt @ np.array(V_SIN)
    w = cos_mt @ np.array(W_COS) + sin_mt @ np.array(W_SIN)
    points = centre + u[:, None] * axis + v[:, None] * second + w[:, None] * normal
    order = len(U_COS)
    harmonics = np.fft.rfft(points, axis=0) / len(t)
    curve = CurveXYZFourier(64, order)
    # c_0 and then s_j, c_j of \sin(2 \pi j t) and \cos(2 \pi j t) in each coordinate
    curve.x = np.concatenate(
        [
            [harmonics[0, k].real]
            + [
                val
                for j in range(1, order + 1)
                for val in (-2.0 * harmonics[j, k].imag, 2.0 * harmonics[j, k].real)
            ]
            for k in range(3)
        ]
    )
    return Coil(curve, Current(current))


def coils():
    """The 50 identical coils of the design."""
    # Stellarator symmetry S = diag(1, -1, -1) is a 180-degree rigid rotation about the
    # x-axis (det S = +1), carrying each coil of the first half-period into a rigidly
    # congruent coil in the second half-period with reversed current.
    flip = np.array([1.0, -1.0, -1.0])
    centres = np.concatenate([CENTRES, flip * CENTRES])
    normals = np.concatenate([NORMALS, flip * NORMALS])
    axes = np.concatenate([AXES, flip * AXES])
    currents = np.concatenate(
        [np.full(len(CENTRES), CURRENT), np.full(len(CENTRES), -CURRENT)]
    )
    out = []
    for k in range(NFP):
        cos, sin = np.cos(2.0 * np.pi * k / NFP), np.sin(2.0 * np.pi * k / NFP)
        turn = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        out += [
            coil(turn @ c, turn @ n, turn @ a, curr)
            for c, n, a, curr in zip(centres, normals, axes, currents, strict=True)
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
        ns_array=np.array([13, 25, 51]),
        ftol_array=np.array([1e-10, 1e-12, 1e-12]),
        niter_array=np.array([30000, 60000, 100000]),
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
    # Nemov et al., Phys. Plasmas 6, 4622 (1999)
    # b = B / B_0 with B_0 the mean of B weighted by ds / B
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
    # the strongest B of the line as the reference field
    return float((scale * total * eta.max() ** 2) ** (2.0 / 3.0))


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
    # R_0 = V / (2 \pi A) with A the mean area of the cross-sections
    return effective_ripple(strength, size, drift, weight, float(wout.Rmajor_p))


def closure(strength):
    """Share of the range of B[\\theta, \\phi] whose contours close poloidally."""
    count = strength.shape[1]
    lowest = strength.argmin(axis=1)
    step = np.diff(np.append(lowest, lowest[0]))
    # helical contours: the minimum of B in \phi winds once through the period
    if abs(np.sum((step + count // 2) % count - count // 2)) >= count // 2:
        return 0.0
    # poloidal contours from the largest minimum of B in \phi to the smallest maximum
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
    surfaces of a wout nearest s = 0.5 and 0.75 and its outermost one."""
    ns = int(wout.ns)
    surfaces = (round(0.5 * (ns - 1) + 0.5), round(0.75 * (ns - 1) + 0.5), ns - 1)
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
    every = coils()
    field = BiotSavart(every)
    k = interior(field, DESIGN)
    print(
        f"{len(every)} coils of semi-axes {U_COS[0]:.3f} m and {V_SIN[0]:.3f} m: "
        "magnetic axis through R = "
        f"{k['r_axis']:.5f} m on the midplane of phi = 0 with iota "
        f"{abs(k['iota_axis']):.4f} next to it, boundary traced from 18.5 cm outboard "
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

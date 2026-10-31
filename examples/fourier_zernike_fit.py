# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Fit the flux surfaces of a wout file with a Fourier-Zernike basis.

R = \\sum R_lmn R_l^|m|(rho) T_m(theta) F_n(zeta), and likewise for Z, in DESC's
conventions: T_m is cos(m theta) for m >= 0 and sin(|m| theta) otherwise, F_n is
cos(n nfp zeta) for n >= 0 and sin(|n| nfp zeta) otherwise, rho = sqrt(s). The
radial degree of each (m, n) term is chosen by two-fold cross-validation on
alternate surfaces.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import eval_jacobi

import vmecpp


def zernike_radial(rho, ell, m):
    k = (ell - m) // 2
    return (-1) ** k * rho**m * eval_jacobi(k, m, 0, 1 - 2 * rho**2)


def fit(rho, values, m, max_degree=20):
    """Coefficients of R_l^m, l = m, m + 2, ..., and the residual."""

    def design(r, degree):
        return np.column_stack(
            [zernike_radial(r, m + 2 * i, m) for i in range(degree + 1)]
        )

    def lstsq(idx, degree):
        return np.linalg.lstsq(design(rho[idx], degree), values[idx], rcond=None)[0]

    folds = (np.arange(len(rho))[::2], np.arange(len(rho))[1::2])
    degree = min(
        range(min(max_degree, len(folds[1]) - 2) + 1),
        key=lambda d: max(
            np.abs(design(rho[b], d) @ lstsq(a, d) - values[b]).max()
            for a, b in (folds, folds[::-1])
        ),
    )
    c = lstsq(slice(None), degree)
    return [m + 2 * i for i in range(degree + 1)], c, design(rho, degree) @ c - values


def fourier_zernike(wout):
    rho = np.sqrt(np.linspace(0.0, 1.0, wout.ns))
    result = {"num_field_periods": int(wout.nfp)}
    for name, cos, sin in (
        ("r", wout.rmnc, wout.rmns if wout.lasym else None),
        ("z", wout.zmnc if wout.lasym else None, wout.zmns),
    ):
        # cos(a - b) = cos a cos b + sin a sin b, sin(a - b) = sin a cos b - cos a sin b
        terms = defaultdict(float)
        for j, (m, xn) in enumerate(
            zip(wout.xm.astype(int), wout.xn.astype(int), strict=True)
        ):
            c = 0.0 if cos is None else cos[j]
            s = 0.0 if sin is None else sin[j]
            sign, n = np.sign(xn), abs(xn) // wout.nfp
            terms[m, n] = terms[m, n] + c
            terms[m, -n] = terms[m, -n] - sign * s
            if m > 0:
                terms[-m, -n] = terms[-m, -n] + sign * c
                terms[-m, n] = terms[-m, n] + s
        modes, lmn, residual = [], [], 0.0
        for (m, n), values in terms.items():
            if np.any(values != 0):
                ells, c, error = fit(rho, values, abs(m))
                modes += [(int(ell), int(m), int(n)) for ell in ells]
                lmn += c.tolist()
                residual = residual + np.abs(error)
        result |= {f"{name}_modes": modes, f"{name}_lmn": lmn}
        result[f"{name}_residual"] = float(np.max(residual))
    return result


if __name__ == "__main__":
    data = Path(__file__).parent / "data" / "wout_cth_like_fixed_bdy.nc"
    result = fourier_zernike(vmecpp.VmecWOut.from_wout_file(data))
    print(
        f"largest residual {1e3 * result['r_residual']:.3f} mm in R, "
        f"{1e3 * result['z_residual']:.3f} mm in Z"
    )
    Path("fourier_zernike.json").write_text(json.dumps(result))

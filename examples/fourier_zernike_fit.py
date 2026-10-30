# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Fit the flux surfaces of a wout file with a Fourier-Zernike basis.

The fit gives the geometry in closed form, regular on the magnetic axis:

    R(rho, theta, zeta) = \\sum R_lmn Z_l^m(rho, theta) F_n(zeta)

and likewise for Z, in the conventions of DESC: Z_l^m is the Zernike radial
polynomial R_l^|m|(rho) times cos(m theta) for m >= 0 or sin(|m| theta) for
m < 0, F_n is cos(n nfp zeta) for n >= 0 or sin(|n| nfp zeta) for n < 0, and
rho = sqrt(s). Each product-basis coefficient is fitted over the full-grid
surfaces with the radial degree chosen by two-fold cross-validation on
alternate surfaces. The script prints the largest position residual and writes
the coefficients (m) to fourier_zernike.json.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import vmecpp


def zernike_radial(rho, ell, m):
    """R_l^m(rho) = (-1)^k rho^m P_k^(m, 0)(1 - 2 rho^2), k = (l - m) / 2."""
    k = (ell - m) // 2
    x = 1.0 - 2.0 * rho**2
    p_prev, p = np.zeros_like(rho), np.ones_like(rho)
    for j in range(1, k + 1):
        if j == 1:
            p_prev, p = p, (m + 1) + 0.5 * (m + 2) * (x - 1.0)
            continue
        s = 2 * j + m
        p_prev, p = (
            p,
            (
                (s - 1) * (s * (s - 2) * x + m**2) * p
                - 2 * (j + m - 1) * (j - 1) * s * p_prev
            )
            / (2 * j * (j + m) * (s - 2)),
        )
    return (-1) ** k * rho**m * p


def product_basis(xm, xn, nfp, cos_coefficients, sin_coefficients):
    """Split cos/sin(m theta - n zeta) into DESC's product-basis (m, n) columns."""
    columns = defaultdict(float)
    for j, (m, n) in enumerate(zip(xm, xn // nfp, strict=True)):
        c = cos_coefficients[j] if cos_coefficients is not None else 0.0
        s = sin_coefficients[j] if sin_coefficients is not None else 0.0
        sign = np.sign(n)
        # cos(a - b) = cos a cos b + sin a sin b, sin(a - b) = sin a cos b - cos a sin b
        columns[m, abs(n)] = columns[m, abs(n)] + c
        columns[m, -abs(n)] = columns[m, -abs(n)] - sign * s
        if m > 0:  # sin(0 theta) vanishes
            columns[-m, -abs(n)] = columns[-m, -abs(n)] + sign * c
            columns[-m, abs(n)] = columns[-m, abs(n)] + s
    return {mode: value for mode, value in columns.items() if np.any(value != 0)}


def fit_radial(rho, values, m, max_degree=20):
    """Fit values(rho) with R_l^|m|, l = |m|, |m| + 2, ..., cross-validating the degree."""
    m = abs(m)

    def design(r, degree):
        return np.column_stack(
            [zernike_radial(r, m + 2 * i, m) for i in range(degree + 1)]
        )

    folds = [np.arange(len(rho))[i::2] for i in (0, 1)]
    errors = []
    for degree in range(min(max_degree, len(folds[1]) - 2) + 1):
        error = 0.0
        for train, test in (folds, folds[::-1]):
            coefficients = np.linalg.lstsq(
                design(rho[train], degree), values[train], rcond=None
            )[0]
            error = max(
                error,
                np.abs(design(rho[test], degree) @ coefficients - values[test]).max(),
            )
        errors.append(error)
    degree = int(np.argmin(errors))
    coefficients = np.linalg.lstsq(design(rho, degree), values, rcond=None)[0]
    residual = design(rho, degree) @ coefficients - values
    return [m + 2 * i for i in range(degree + 1)], coefficients, residual


def fourier_zernike(wout):
    rho = np.sqrt(np.linspace(0.0, 1.0, wout.ns))
    xm, xn = np.asarray(wout.xm, dtype=int), np.asarray(wout.xn, dtype=int)
    result = {"num_field_periods": int(wout.nfp)}
    for name, cos_coefficients, sin_coefficients in (
        ("r", wout.rmnc, wout.rmns if wout.lasym else None),
        ("z", wout.zmnc if wout.lasym else None, wout.zmns),
    ):
        columns = product_basis(xm, xn, wout.nfp, cos_coefficients, sin_coefficients)
        modes, lmn, residual = [], [], np.zeros(wout.ns)
        for (m, n), values in columns.items():
            ells, coefficients, error = fit_radial(rho, values, m)
            modes += [(int(ell), int(m), int(n)) for ell in ells]
            lmn += coefficients.tolist()
            # Bounds the position error: |F_n| <= 1 and |cos|, |sin| <= 1.
            residual += np.abs(error)
        result[f"{name}_modes"], result[f"{name}_lmn"] = modes, lmn
        result[f"{name}_residual"] = float(residual.max())
    return result


def main():
    wout = vmecpp.VmecWOut.from_wout_file(
        Path(__file__).parent / "data" / "wout_cth_like_fixed_bdy.nc"
    )
    result = fourier_zernike(wout)
    print(
        f"{len(result['r_modes'])} R and {len(result['z_modes'])} Z modes; "
        f"largest residual {1e3 * result['r_residual']:.3f} mm in R, "
        f"{1e3 * result['z_residual']:.3f} mm in Z"
    )
    Path("fourier_zernike.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()

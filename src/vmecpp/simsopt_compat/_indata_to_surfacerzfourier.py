# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
import json
from pathlib import Path

import numpy as np
from simsopt import geo


def surfacerzfourier_from_any_vmec_indata(
    input_file: Path,
    ntheta: int | None = None,
    nphi: int | None = None,
    range: str | None = None,
) -> geo.SurfaceRZFourier:
    """An alternative to simsopt.geo.SurfaceRZFourier.from_vmec_indata that works with
    both Fortran VMEC indata files and VMEC++ JSON input files.

    It falls back to SurfaceRZFourier.from_vmec_indata for Fortran VMEC indata files.

    Args:
        input_file: a Fortran VMEC indata file or a VMEC++ JSON input file
        ntheta: Number of grid points in the poloidal angle
                same as in SIMSOPT's Surface.from_nphi_ntheta
        nphi: Number of grid points in the toroidal angle,
              same as in SIMSOPT's Surface.from_nphi_ntheta
        range: Toroidal extent of the toroidal (phi) grid, same as in SIMSOPT's
               Surface.from_nphi_ntheta. Possible options are "full torus",
               "field period" or "half period" (see SIMSOPT's docs)

    Returns:
        a SIMSOPT SurfaceRZFourier object that represents the boundary described in
        the input file.

    IMPORTANT NOTE:
    If the indata file contains RBC/ZBS coefficients for poloidal modes m >= mpol,
    the SurfaceRZFourier obtained from a VMEC++ input file might differ slightly
    from the one that would be obtained by calling SurfaceRZFourier.from_vmec_indata
    on the corresponding Fortran indata file: it will be missing those m >= mpol modes.

    See surfacerzfourier_from_vmecppindata for more information.
    """
    try:
        # we try to open vmec_input_filepath as a VMEC++ JSON input first...
        boundary = surfacerzfourier_from_vmecppindata(
            input_file, ntheta=ntheta, nphi=nphi, range=range
        )
    except json.JSONDecodeError:
        # ...and fall back to opening it as a Fortran INDATA file in case of a
        # JSONDecodeError. This is a bit indirect, but we cannot do something
        # simpler like checking whether the filename extension is .json because
        # in general even a JSON indata file could not have a .json extension
        # (e.g. because ScaleVmec's output file name is a configuration parameter
        # and we cannot expect users to update it every time they switch between
        # VMEC2000 and VMECPP).
        boundary = geo.SurfaceRZFourier.from_vmec_input(
            str(input_file), ntheta=ntheta, nphi=nphi, range=range
        )

    return boundary


def surfacerzfourier_from_vmecppindata(
    input_file: Path,
    ntheta: int | None = None,
    nphi: int | None = None,
    range: str | None = None,
) -> geo.SurfaceRZFourier:
    """A VMEC++-compatible alternative to simsopt.geo.SurfaceRZFourier.from_vmec_indata.

    Args:
        input_file: a VMEC++ JSON input file
        ntheta: Number of grid points in the poloidal angle
                same as in SIMSOPT's Surface.from_nphi_ntheta
        nphi: Number of grid points in the toroidal angle,
              same as in SIMSOPT's Surface.from_nphi_ntheta
        range: Toroidal extent of the toroidal (phi) grid, same as in SIMSOPT's
               Surface.from_nphi_ntheta. Possible options are "full torus",
               "field period" or "half period" (see SIMSOPT's docs)

    Returns:
        a SIMSOPT SurfaceRZFourier object that represents the boundary described in
        the input file.

    IMPORTANT NOTE:
    If the indata file contains RBC/ZBS coefficients for poloidal modes m >= mpol,
    the SurfaceRZFourier obtained from a VMEC++ input file might differ slightly
    from the one that would be obtained via SurfaceRZFourier.from_vmec_indata:
    it will be missing poloidal modes m >= mpol.

    The reason is that SIMSOPT's SurfaceRZFourier implementation takes into account
    poloidal modes m <= mpol, although VMEC2000 and VMEC++ only care about m < mpol.
    Indata files sometimes do contain Fourier coefficients for m == mpol, as it is not
    an error to include coefficients that VMEC will ignore.
    VMEC++ JSON input files, however, never contain Fourier coefficients with poloidal
    modes m >= mpol, and automatic conversion between Fortran indata files and VMEC++
    JSON input files actually discards those coefficients.

    We think that omitting modes m == mpol when constructing SurfaceRZFourier is the
    correct choice, as it is consistent with the behavior of VMEC, but this causes
    an inconsistency with the original SurfaceRZFourier.from_vmec_indata.

    See also https://github.com/hiddenSymmetries/simsopt/pull/437.
    """

    with open(input_file, encoding="utf-8") as indata_file:
        indata = json.load(indata_file)

    if indata["lasym"]:
        msg = "Inputs without stellarator symmetry are not supported."
        raise NotImplementedError(msg)

    return surfacerzfourier_from_fourier_coeffs(
        mpol=indata["mpol"],
        ntor=indata["ntor"],
        rbc=indata["rbc"],
        zbs=indata["zbs"],
        nfp=indata["nfp"],
        ntheta=ntheta,
        nphi=nphi,
        range=range,
    )


def surfacerzfourier_from_fourier_coeffs(
    *,
    mpol: int,
    ntor: int,
    rbc: list[dict],
    zbs: list[dict],
    nfp: int,
    ntheta: int | None = None,
    nphi: int | None = None,
    range: str | None = None,
) -> geo.SurfaceRZFourier:
    """The fourier coefficient are expected to be in the form of a list of dictionaries
    with entries {"m": m, "n": n, "value": value} which is the VMEC++ input data format.

    mpol uses the VMEC++ convention here: it is the end of the exclusive range:
    mpol==N means that the coefficients will have highest poloidal mode equal to N - 1.

    If one of ntheta or nphi is present, both must be present.

    Non-stellarator-symmetric inputs are not supported.
    """

    if ntheta is not None or nphi is not None:
        # TODO(eguiraud): just use geo.Surface.get_quadpoints when we switch to
        # SIMSOPT>=1.6.3. The current SIMSOPT version has a bug that prevents its usage,
        # fixed by https://github.com/hiddenSymmetries/simsopt/pull/418/commits/59877c146a37fe815480e62433b7342ddfac2c30
        # quadpoints_phi, quadpoints_theta = geo.Surface.get_quadpoints(
        #     nfp=nfp,
        #     ntheta=ntheta,
        #     nphi=nphi,
        #     range=range,
        # )
        quadpoints_phi, quadpoints_theta = _get_simsopt_surface_quadpoints(
            nfp, ntheta, nphi, range
        )
    else:
        quadpoints_phi, quadpoints_theta = None, None

    surface = geo.SurfaceRZFourier(
        mpol=mpol - 1,
        ntor=ntor,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )

    # set Fourier coefficients
    # RBC
    for d in rbc:
        m, n, value = d["m"], d["n"], d["value"]
        surface.set_rc(m, n, value)

    # ZBS
    for d in zbs:
        m, n, value = d["m"], d["n"], d["value"]
        surface.set_zs(m, n, value)

    return surface


def _get_simsopt_surface_quadpoints(
    nfp: int,
    ntheta: int | None = None,
    nphi: int | None = None,
    range: str | None = None,
) -> tuple[list[float], list[float]]:
    """Mirror the logic of simsopt.geo.surface.Surface.get_quadpoints.

    NOTE: we only need this function until we switch to SIMSOPT 1.6.3,
    which fixes a fatal bug in SIMSOPT's get_quadpoints implementation.

    Returns:
        (phi_quadpoints, theta_quadpoints)
    """
    if ntheta is None:
        ntheta = 62

    theta_quadpoints = list(np.linspace(0.0, 1.0, ntheta, endpoint=False))

    if range is None:
        range = "full torus"
    assert range in ("full torus", "half period", "field period")

    div = 1 if range == "full torus" else nfp
    end_val = 0.5 if range == "half period" else 1.0

    if nphi is None:
        nphi = 61

    phi_quadpoints = list(np.linspace(0.0, end_val / div, nphi, endpoint=False))

    if range == "half period":
        # shift by half of the grid spacing:
        dphi = phi_quadpoints[1] - phi_quadpoints[0]
        phi_quadpoints += 0.5 * dphi

    return phi_quadpoints, theta_quadpoints

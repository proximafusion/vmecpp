# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
""":class:`VmecOutput`, the container bundling every output of a VMEC++ run."""

from __future__ import annotations

from vmecpp._indata import VmecInput
from vmecpp._pydantic_numpy import BaseModelWithNumpy
from vmecpp._wout import (
    JxBOut,
    Mercier,
    Threed1AxisGeometry,
    Threed1Betas,
    Threed1FirstTable,
    Threed1GeometricAndMagneticQuantities,
    Threed1ShafranovIntegrals,
    Threed1Volumetrics,
    VmecWOut,
)


class VmecOutput(BaseModelWithNumpy):
    """Container for the full output of a VMEC run."""

    input: VmecInput
    """The input to the VMEC run that produced this output."""

    jxbout: JxBOut
    """Python equivalent of VMEC's "jxbout" file."""

    mercier: Mercier
    """Python equivalent of VMEC's "mercier" file.

    Contains radial profiles and stability criteria relevant for Mercier stability
    analysis, including the Mercier criterion and its decomposition into shear, well,
    current, and geodesic contributions. Also includes profiles of rotational transform,
    toroidal flux, pressure, and their derivatives.
    """

    threed1_volumetrics: Threed1Volumetrics
    """Python equivalent of VMEC's volumetrics section in the "threed1" file.

    Contains global and flux-surface-averaged quantities such as total and average
    pressure, poloidal and toroidal magnetic field energies, kinetic energy, and related
    integrals. Useful for postprocessing and global equilibrium characterization.
    """

    threed1_first_table: Threed1FirstTable
    """Python equivalent of the first table in VMEC's "threed1" file: radial
    profiles of s, iota, currents, pressure, and force balance."""

    threed1_geometric_magnetic: Threed1GeometricAndMagneticQuantities
    """Python equivalent of the geometric and magnetic quantities in VMEC's
    "threed1" file: global geometry, beta values, currents, and geometric profiles."""

    threed1_axis: Threed1AxisGeometry
    """Python equivalent of the magnetic-axis geometry in VMEC's "threed1" file."""

    threed1_betas: Threed1Betas
    """Python equivalent of the beta values in VMEC's "threed1" file."""

    threed1_shafranov_integrals: Threed1ShafranovIntegrals
    """Python equivalent of the Shafranov surface integrals in VMEC's "threed1" file."""

    wout: VmecWOut
    """Python equivalent of VMEC's "wout" file."""

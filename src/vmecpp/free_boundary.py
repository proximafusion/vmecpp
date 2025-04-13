# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
from pathlib import Path

import numpydantic as npyd
import pydantic

from vmecpp.cpp import _vmecpp

logger = logging.getLogger(__name__)


class MakegridParameters(pydantic.BaseModel):
    """
    Pydantic model mirroring the C++ makegrid::MakegridParameters struct.

    Represents the parameters used to define the grid for external field
    calculations (mgrid file).
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    normalize_by_currents: bool
    """If true, normalize the magnetic field by coil currents and windings."""
    assume_stellarator_symmetry: bool
    """If true, compute on half-period and mirror."""
    number_of_field_periods: int
    """Number of toroidal field periods."""
    r_grid_minimum: float
    """Radial coordinate of the first grid point."""
    r_grid_maximum: float
    """Radial coordinate of the last grid point."""
    number_of_r_grid_points: int
    """Number of radial grid points."""
    z_grid_minimum: float
    """Vertical coordinate of the first grid point."""
    z_grid_maximum: float
    """Vertical coordinate of the last grid point."""
    number_of_z_grid_points: int
    """Number of vertical grid points."""
    number_of_phi_grid_points: int
    """Number of toroidal grid points per field period."""

    @staticmethod
    def _from_cpp_makegrid_parameters(
        cpp_obj: _vmecpp.MakegridParameters,
    ) -> MakegridParameters:
        makegrid_parameters = MakegridParameters(
            **{attr: getattr(cpp_obj, attr) for attr in MakegridParameters.model_fields}
        )
        return makegrid_parameters

    def _to_cpp_makegrid_parameters(self) -> _vmecpp.MakegridParameters:
        return _vmecpp.MakegridParameters()

    @staticmethod
    def from_file(input_file: str | Path) -> MakegridParameters:
        return MakegridParameters._from_cpp_makegrid_parameters(
            _vmecpp.MakegridParameters.from_file(input_file)
        )


class MagneticConfiguration(pydantic.BaseModel):
    @staticmethod
    def from_file(input_file: str | Path) -> MagneticConfiguration:
        return _vmecpp.MagneticConfiguration.from_file(input_file)

    def _to_cpp_magnetic_configuration(self):
        return _vmecpp.MagneticConfiguration()


class MagneticFieldResponseTable(pydantic.BaseModel):
    """
    Pydantic model mirroring the C++ makegrid::MagneticFieldResponseTable struct.

    Holds the precomputed magnetic field response on a grid, separated by
    coil circuit. Each field component (b_r, b_p, b_z) is a list where each
    element corresponds to a circuit and contains the flattened 1D array of
    field values on the grid for that circuit carrying unit current.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    parameters: MakegridParameters
    """The grid parameters used to compute this table."""

    # List of 1D arrays (flattened grid), one array per circuit.
    # Shape of each inner array: (number_of_phi_grid_points * number_of_z_grid_points * number_of_r_grid_points)
    b_r: npyd.NDArray[npyd.Shape["* num_coils,* num_mgrid_cells"], float]
    """Cylindrical R components of magnetic field per circuit."""
    b_p: npyd.NDArray[npyd.Shape["* num_coils,* num_mgrid_cells"], float]
    """Cylindrical Phi components of magnetic field per circuit."""
    b_z: npyd.NDArray[npyd.Shape["* num_coils,* num_mgrid_cells"], float]
    """Cylindrical Z components of magnetic field per circuit."""

    @staticmethod
    def _from_cpp_magnetic_field_response_table(
        cpp_obj: _vmecpp.MagneticFieldResponseTable,
    ) -> MagneticFieldResponseTable:
        magnetic_field_response_table = MagneticFieldResponseTable(
            **{
                attr: getattr(cpp_obj, attr)
                for attr in MagneticFieldResponseTable.model_fields
            }
        )

        return magnetic_field_response_table

    @staticmethod
    def from_file(input_file: str | Path) -> MagneticFieldResponseTable:
        return MagneticFieldResponseTable._from_cpp_magnetic_field_response_table(
            _vmecpp.MagneticFieldResponseTable.from_file(input_file)
        )


def compute_magnetic_field_response_table(
    makegrid_parameters: MakegridParameters,
    magnetic_configuration: MagneticConfiguration,
) -> MagneticFieldResponseTable:
    return _vmecpp.compute_magnetic_field_response_table(
        makegrid_parameters._to_cpp_makegrid_parameters(),
        magnetic_configuration._to_cpp_magnetic_configuration(),
    )


__all__ = [
    "MagneticFieldResponseTable",
    "MakegridParameters",
    "compute_magnetic_field_response_table",
]

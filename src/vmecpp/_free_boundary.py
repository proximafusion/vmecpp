# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import jaxtyping as jt
import numpy as np
import pydantic

from vmecpp._pydantic_numpy import BaseModelWithNumpy
from vmecpp.cpp import _vmecpp  # type: ignore


class MakegridParameters(BaseModelWithNumpy):
    """
    Pydantic model mirroring the C++ makegrid::MakegridParameters struct.

    Parameters defining the grid for external magnetic field calculations (mgrid file).
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
        return _vmecpp.MakegridParameters(
            *[getattr(self, attr) for attr in MakegridParameters.model_fields]
        )

    @staticmethod
    def from_file(input_file: str | Path) -> MakegridParameters:
        return MakegridParameters._from_cpp_makegrid_parameters(
            _vmecpp.MakegridParameters.from_file(input_file)
        )


class MagneticFieldResponseTable(BaseModelWithNumpy):
    """
    Pydantic model mirroring the C++ makegrid::MagneticFieldResponseTable struct.

    Holds the precomputed magnetic field response on a grid, separated by
    coil circuit. Each field component (b_r, b_p, b_z) is a list where each
    element corresponds to a circuit and contains the flattened 1D array of
    field values on the grid for that circuit carrying unit current.
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    parameters: MakegridParameters

    # List of 1D arrays (flattened grid), one array per circuit.
    # Shape of each inner array: (number_of_phi_grid_points * number_of_z_grid_points * number_of_r_grid_points)
    b_r: jt.Float[np.ndarray, "num_coils num_mgrid_cells"]
    """Cylindrical R components of magnetic field per circuit."""
    b_p: jt.Float[np.ndarray, "num_coils num_mgrid_cells"]
    """Cylindrical Phi components of magnetic field per circuit."""
    b_z: jt.Float[np.ndarray, "num_coils num_mgrid_cells"]
    """Cylindrical Z components of magnetic field per circuit."""

    @staticmethod
    def _from_cpp_magnetic_field_response_table(
        cpp_obj: _vmecpp.MagneticFieldResponseTable,
    ) -> MagneticFieldResponseTable:
        magnetic_field_response_table = MagneticFieldResponseTable(
            parameters=MakegridParameters._from_cpp_makegrid_parameters(
                cpp_obj.parameters
            ),
            b_r=cpp_obj.b_r,
            b_p=cpp_obj.b_p,
            b_z=cpp_obj.b_z,
        )

        return magnetic_field_response_table

    @staticmethod
    def from_coils_file(
        coils_path: str | Path,
        makegrid_parameters: MakegridParameters,
    ) -> MagneticFieldResponseTable:
        magnetic_configuration = _vmecpp.MagneticConfiguration.from_file(coils_path)
        cpp_response_table = _vmecpp.compute_magnetic_field_response_table(
            makegrid_parameters._to_cpp_makegrid_parameters(),
            magnetic_configuration,
        )
        return MagneticFieldResponseTable._from_cpp_magnetic_field_response_table(
            cpp_response_table
        )

    def _to_cpp_magnetic_field_response_table(
        self,
    ) -> _vmecpp.MagneticFieldResponseTable:
        """Convert the Pydantic object to a C++ MagneticFieldResponseTable object,
        avoiding a copy if possible."""
        # If vmecpp.MagneticFieldResponseTable was created from a C++ object, the
        # arrays be views to the memory of the C++ object. We don't need to create
        # a new object and just return the underling one.
        underlying = self.b_r.base
        if (
            isinstance(underlying, _vmecpp.MagneticFieldResponseTable)
            and underlying == self.b_p.base
            and underlying == self.b_z.base
        ):
            return underlying
        # Otherwise, create a new C++ object
        return _vmecpp.MagneticFieldResponseTable(
            self.parameters._to_cpp_makegrid_parameters(),
            self.b_r,
            self.b_p,
            self.b_z,
        )


__all__ = [
    "MagneticFieldResponseTable",
    "MakegridParameters",
]

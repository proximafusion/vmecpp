# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""SIMSOPT compatibility layer for VMEC++."""

# This is just a re-export
# TODO(eguiraud): once we internally migrate to the standalone VMEC++ repo,
# refactor things so that the contents of simsopt_vmecpp are moved here
from vmecpp.cpp.vmecpp.vmec.pybind11.simsopt_vmecpp import Vmec

__all__ = ["Vmec"]

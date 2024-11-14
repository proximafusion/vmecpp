import math
from pathlib import Path

import numpy as np
from starfinder.mhd import vmec2000_wrapper, vmecpp_wrapper
from util import workspace

from vmecpp import simsopt_compat


def test_is_vmec2000_input():
    repo_root = Path(workspace.workspace_root())
    vmec2000_input_file = repo_root / "vmecpp/test_data/input.cma"
    vmecpp_input_file = repo_root / "vmecpp/test_data/cma.json"

    assert simsopt_compat.is_vmec2000_input(vmec2000_input_file)
    assert not simsopt_compat.is_vmec2000_input(vmecpp_input_file)


def test_ensure_vmec2000_input_noop():
    repo_root = Path(workspace.workspace_root())
    vmec2000_input_file = repo_root / "vmecpp/test_data/input.cma"

    with simsopt_compat.ensure_vmec2000_input(vmec2000_input_file) as indata_file:
        assert indata_file == vmec2000_input_file


def test_ensure_vmec2000_input_from_vmecpp_input():
    repo_root = Path(workspace.workspace_root())
    vmecpp_input_file = repo_root / "vmecpp/test_data/cma.json"

    with simsopt_compat.ensure_vmec2000_input(
        vmecpp_input_file
    ) as converted_indata_file:
        vmec2000 = vmec2000_wrapper.Vmec(str(converted_indata_file))

    vmecpp = vmecpp_wrapper.Vmec(str(vmecpp_input_file))

    # vmec2000.indata has way many more variables than vmecpp.indata, so we test
    # the common subset.
    for varname in dir(vmecpp.indata):
        if varname.startswith("_") or varname in ["free_boundary_method"]:
            continue

        vmecpp_var = getattr(vmecpp.indata, varname)
        if callable(vmecpp_var):
            continue  # this is a method, not a variable

        varname_vmec2000 = varname
        if varname[1:-1] == "axis_":
            # these are called differently in VMEC2000, e.g. raxis_c -> raxis_cc
            varname_vmec2000 = f"{varname[:-1]}c{varname[-1]}"
        vmec2000_var = getattr(vmec2000.indata, varname_vmec2000)

        if isinstance(vmecpp_var, (str, int, bool)):
            if isinstance(vmec2000_var, bytes):
                vmec2000_var = vmec2000_var.decode().strip()
            elif varname in ["ntheta", "nzeta"]:
                assert vmecpp_var == 0  # like in the input file
                assert vmec2000_var == 16  # the default VMEC2000 sets if it's == 0
            else:
                assert vmecpp_var == vmec2000_var
        elif isinstance(vmecpp_var, float):
            assert math.isclose(vmecpp_var, vmec2000_var)
        else:
            assert isinstance(vmecpp_var, np.ndarray)

            # NOTE: these are differences in behavior between VMEC++ and VMEC2000,
            # not an issue with the file format conversion.
            if varname == "ac_aux_f":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 101))
            elif varname == "ac_aux_s":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([-1.0] * 101))
            elif varname == "ai":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 21))
            elif varname == "ai_aux_f":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 101))
            elif varname == "ai_aux_s":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([-1.0] * 101))
            elif varname == "am_aux_f":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 101))
            elif varname == "am_aux_s":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([-1.0] * 101))
            elif varname == "extcur":
                assert vmecpp_var.shape == (0,)
                assert all(vmec2000_var == np.array([0.0] * 300))
            else:
                # VMEC2000 pads the arrays with zeros, VMEC++ instantiates them
                # with the right length
                if len(vmecpp_var.shape) == 1:
                    vmec2000_var_truncated = vmec2000_var[: len(vmecpp_var)]
                else:
                    assert vmecpp.indata is not None  # for pyright
                    # RBS and ZBC might be just empty
                    if varname in ["rbs", "zbc"] and not vmecpp.indata.lasym:
                        vmec2000_var_truncated = np.zeros(shape=(0, 0))
                    else:
                        # must be 2D RBC, ZBS. here there is a triple mismatch:
                        # 1. VMEC2000 uses layout (n, m) while VMEC++ uses (m, n)
                        # 2. VMEC2000 pre-allocates an array with shape (203, 101)
                        #    while VMEC++ allocates according to mpol, ntor
                        # 3. for the `n` index values are laid out as
                        #    [-ntor, ..., 0, ..., ntor] (with ntor being the one from
                        #    the file for VMEC++, and 101 for VMEC2000), so we need to
                        #    truncate the entries in vmec2000_var symmetrically around
                        #    the center
                        # First we transpose and truncate the rows:
                        vmec2000_var_truncated = vmec2000_var.T[
                            : vmecpp_var.shape[0], :
                        ]
                        # Now we truncate the columns symmetrically around the center:
                        ntor = vmecpp.indata.ntor
                        vmec2000_var_truncated = vmec2000_var_truncated[
                            :, 101 - ntor : 101 + ntor + 1
                        ]
                np.testing.assert_allclose(vmecpp_var, vmec2000_var_truncated)

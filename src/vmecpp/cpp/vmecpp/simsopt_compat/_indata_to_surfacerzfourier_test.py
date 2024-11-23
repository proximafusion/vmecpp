import math
import tempfile
from pathlib import Path

from simsopt import geo

from vmecpp import _util
from vmecpp.cpp.third_party.indata2json import indata_to_json
from vmecpp.cpp.vmecpp import simsopt_compat


def test_surfacerzfourier_from_vmecppindata_no_ntheta_nphi():
    """Best way I could think of testing this: compare a surface obtained with
    simsopt.geo.SurfaceRZFourier.from_vmec_input with one obtained with
    surfacerzfourier_from_vmecppindata after converting the Fortran indata file
    to the VMEC++ indata file using indata2json."""
    fortran_indata_file = Path(
        _util.package_root(), "cpp", "vmecpp", "test_data", "input.cma"
    )

    reference_surface = geo.SurfaceRZFourier.from_vmec_input(str(fortran_indata_file))

    with tempfile.TemporaryDirectory():
        vmecpp_indata_file = indata_to_json.indata_to_json(fortran_indata_file)
        test_surface = simsopt_compat.surfacerzfourier_from_vmecppindata(
            Path(vmecpp_indata_file)
        )

    assert reference_surface.stellsym
    assert test_surface.stellsym

    # NOTE: SIMSOPT's SurfaceRZFourier implementation takes into account
    # poloidal modes up to m == mpol, although VMEC2000 and VMEC++ only
    # care about m < mpol.
    # `indata_to_json` cuts off Fourier coefficients at m >= mpol because
    # they are not ignored by VMEC.
    #
    # As a consequence it is impossible to reconstruct the exact same SurfaceRZFourier
    # from the VMEC++ input as from the original VMEC2000 input: it is missing the
    # information on the highest poloidal modes.
    #
    # So here we only test poloidal modes up to m == mpol - 1 and cannot compare
    # the outputs of reference_surface.gamma() and test_surface.gamma(), they will
    # be slightly different.
    mpol = reference_surface.mpol
    ntor = reference_surface.ntor
    for m in range(mpol):
        for n in range(-ntor, ntor + 1):
            assert math.isclose(
                reference_surface.get_rc(m=m, n=n), test_surface.get_rc(m=m, n=n)
            )
            assert math.isclose(
                reference_surface.get_rc(m=m, n=n), test_surface.get_rc(m=m, n=n)
            )

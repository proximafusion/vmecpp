"""Function-scoped independent inputs for adjoint verification."""

import jax
import pytest

import vmecpp

from .harness import (
    INPUT_NAMES,
    BoundaryObjective,
    InteriorLambdaObjective,
    InteriorM2RObjective,
    InvariantObjective,
    QuasisymmetryObjective,
    RandomLinearObjective,
    load_input,
    ncurr1_input,
    resolution_input,
    small_input,
)

jax.config.update("jax_enable_x64", True)

requires_enzyme = pytest.mark.skipif(
    not vmecpp.has_exact_force_jacobian(),
    reason="requires VMECPP_ENABLE_ENZYME=ON",
)


@pytest.fixture
def input_factory():
    return load_input


@pytest.fixture
def resolution_factory():
    return resolution_input


@pytest.fixture(params=INPUT_NAMES)
def vmec_input(request):
    return load_input(request.param)


@pytest.fixture
def solovev_input():
    return load_input("solovev")


@pytest.fixture
def solovev_analytical_input():
    return load_input("solovev_analytical")


@pytest.fixture
def circular_tokamak_input():
    return load_input("circular_tokamak")


@pytest.fixture
def cth_like_fixed_bdy_input():
    return load_input("cth_like_fixed_bdy")


@pytest.fixture
def cth_like_fixed_bdy_iota_input():
    return load_input("cth_like_fixed_bdy_iota")


@pytest.fixture
def li383_low_res_input():
    return load_input("li383_low_res")


@pytest.fixture
def near_axis_iota_nfp4_input():
    return load_input("near_axis_iota_nfp4")


@pytest.fixture
def cth_ncurr1_input():
    return ncurr1_input()


@pytest.fixture
def small_cth_input():
    return small_input()


@pytest.fixture(name="solovev")
def solovev_fixture(solovev_input):
    return solovev_input.model_copy(deep=True)


@pytest.fixture(name="solovev_analytical")
def solovev_analytical_fixture(solovev_analytical_input):
    return solovev_analytical_input.model_copy(deep=True)


@pytest.fixture(name="circular_tokamak")
def circular_tokamak_fixture(circular_tokamak_input):
    return circular_tokamak_input.model_copy(deep=True)


@pytest.fixture(name="cth_like_fixed_bdy")
def cth_like_fixed_bdy_fixture(cth_like_fixed_bdy_input):
    return cth_like_fixed_bdy_input.model_copy(deep=True)


@pytest.fixture(name="cth_like_fixed_bdy_iota")
def cth_like_fixed_bdy_iota_fixture(cth_like_fixed_bdy_iota_input):
    return cth_like_fixed_bdy_iota_input.model_copy(deep=True)


@pytest.fixture(name="li383_low_res")
def li383_low_res_fixture(li383_low_res_input):
    return li383_low_res_input.model_copy(deep=True)


@pytest.fixture(name="near_axis_iota_nfp4")
def near_axis_iota_nfp4_fixture(near_axis_iota_nfp4_input):
    return near_axis_iota_nfp4_input.model_copy(deep=True)


@pytest.fixture(name="small")
def small_fixture(small_cth_input):
    return small_cth_input.model_copy(deep=True)


@pytest.fixture
def invariant_objectives(vmec_input):
    return {
        **{
            name: InvariantObjective(vmec_input, name)
            for name in ("volume", "aspect", "iota", "well")
        },
        "qs": QuasisymmetryObjective(),
    }


@pytest.fixture
def boundary_objective():
    return BoundaryObjective()


@pytest.fixture
def relabeling_objectives():
    return {
        "interior_lambda": InteriorLambdaObjective(),
        "interior_m2_r": InteriorM2RObjective(),
    }


@pytest.fixture
def random_linear_objective_factory():
    return RandomLinearObjective.from_geometry

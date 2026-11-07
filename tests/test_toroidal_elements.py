# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Toroidal spectral elements in place of the toroidal Fourier basis."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vmecpp

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"

ELEMENT_FIELDS = (
    "toroidal_element_degree",
    "toroidal_element_breaks",
    "rbc_nodes",
    "rbs_nodes",
    "zbc_nodes",
    "zbs_nodes",
    "raxis_nodes",
    "zaxis_nodes",
)


def _with_elements(vmec_input, num_elements, degree, **update):
    """`vmec_input` on uniform elements, with boundary and axis at the nodes."""
    breaks = np.linspace(0.0, np.pi, num_elements + 1)
    v = vmecpp.toroidal_element_nodes(breaks, degree, vmec_input.nfp)
    ntor = vmec_input.ntor
    n = np.arange(-ntor, ntor + 1)
    cos, sin = np.cos(np.outer(n, v)), np.sin(np.outer(n, v))
    rbc, zbs = np.asarray(vmec_input.rbc), np.asarray(vmec_input.zbs)
    rbs = np.asarray(vmec_input.rbs) if vmec_input.lasym else np.zeros_like(rbc)
    zbc = np.asarray(vmec_input.zbc) if vmec_input.lasym else np.zeros_like(zbs)
    na = np.arange(ntor + 1)
    cos_axis, sin_axis = np.cos(np.outer(na, v)), np.sin(np.outer(na, v))
    raxis_s = np.asarray(vmec_input.raxis_s) if vmec_input.lasym else np.zeros(ntor + 1)
    zaxis_c = np.asarray(vmec_input.zaxis_c) if vmec_input.lasym else np.zeros(ntor + 1)
    element_ntor = num_elements * degree
    mpol = vmec_input.mpol
    return vmec_input.model_copy(
        update={
            "ntor": element_ntor,
            "rbc": np.zeros((mpol, 2 * element_ntor + 1)),
            "zbs": np.zeros((mpol, 2 * element_ntor + 1)),
            "rbs": np.zeros((mpol, 2 * element_ntor + 1)) if vmec_input.lasym else None,
            "zbc": np.zeros((mpol, 2 * element_ntor + 1)) if vmec_input.lasym else None,
            "raxis_c": np.zeros(element_ntor + 1),
            "zaxis_s": np.zeros(element_ntor + 1),
            "raxis_s": np.zeros(element_ntor + 1) if vmec_input.lasym else None,
            "zaxis_c": np.zeros(element_ntor + 1) if vmec_input.lasym else None,
            "toroidal_element_degree": degree,
            "toroidal_element_breaks": breaks,
            "rbc_nodes": rbc @ cos - rbs @ sin,
            "rbs_nodes": rbc @ sin + rbs @ cos,
            "zbs_nodes": zbs @ cos + zbc @ sin,
            "zbc_nodes": zbc @ cos - zbs @ sin,
            "raxis_nodes": np.asarray(vmec_input.raxis_c) @ cos_axis
            - raxis_s @ sin_axis,
            "zaxis_nodes": zaxis_c @ cos_axis
            - np.asarray(vmec_input.zaxis_s) @ sin_axis,
            **update,
        }
    )


def _modes(wout, name, nyquist=False):
    xm = np.asarray(wout.xm_nyq if nyquist else wout.xm)
    xn = np.asarray(wout.xn_nyq if nyquist else wout.xn)
    values = np.asarray(getattr(wout, name))
    return {
        (int(m), int(n)): values[k] for k, (m, n) in enumerate(zip(xm, xn, strict=True))
    }


def _assert_common_modes_close(reference, actual, name, atol, nyquist=False):
    """Compares the Fourier modes that both wouts carry."""
    expected_modes = _modes(reference, name, nyquist)
    actual_modes = _modes(actual, name, nyquist)
    common = sorted(set(expected_modes) & set(actual_modes))
    np.testing.assert_allclose(
        np.array([actual_modes[k] for k in common]),
        np.array([expected_modes[k] for k in common]),
        atol=atol,
    )


@pytest.fixture(scope="module")
def cth_input():
    vmec_input = vmecpp.VmecInput.from_file(TEST_DATA_DIR / "cth_like_fixed_bdy.json")
    return vmec_input.model_copy(
        update={
            "ns_array": np.array([13, 25]),
            "ftol_array": np.array([1.0e-10, 1.0e-14]),
            "niter_array": np.array([20000, 40000]),
            "always_fix_m1_gauge": True,
        }
    )


@pytest.fixture(scope="module")
def cth_elements_input(cth_input):
    return _with_elements(cth_input, num_elements=4, degree=6)


@pytest.fixture(scope="module")
def cth_elements_output(cth_elements_input):
    return vmecpp.run(cth_elements_input, max_threads=1, verbose=False)


def test_elements_converge_to_the_fourier_equilibrium(cth_input, cth_elements_output):
    """Four elements of degree 6 reproduce Fourier with ntor = 12."""
    ntor = 12
    pad = ntor - cth_input.ntor
    fourier = cth_input.model_copy(
        update={
            "ntor": ntor,
            "nzeta": 2 * ntor + 4,
            "rbc": np.pad(np.asarray(cth_input.rbc), ((0, 0), (pad, pad))),
            "zbs": np.pad(np.asarray(cth_input.zbs), ((0, 0), (pad, pad))),
            "raxis_c": np.pad(np.asarray(cth_input.raxis_c), (0, pad)),
            "zaxis_s": np.pad(np.asarray(cth_input.zaxis_s), (0, pad)),
        }
    )
    reference = vmecpp.run(fourier, max_threads=1, verbose=False).wout
    wout = cth_elements_output.wout
    assert wout.ier_flag == 0
    _assert_common_modes_close(reference, wout, "rmnc", atol=1.0e-7)
    _assert_common_modes_close(reference, wout, "zmns", atol=1.0e-7)
    _assert_common_modes_close(reference, wout, "bmnc", atol=1.0e-7, nyquist=True)
    np.testing.assert_allclose(wout.iotaf, reference.iotaf, atol=1.0e-7)
    np.testing.assert_allclose(wout.volume_p, reference.volume_p, rtol=1.0e-12)


def test_symmetric_elements_through_the_asymmetric_paths(cth_input):
    """A lasym run with zero asymmetric content reproduces the symmetric element run."""
    symmetric = _with_elements(cth_input, num_elements=2, degree=4)
    asymmetric = _with_elements(
        cth_input.model_copy(
            update={
                "lasym": True,
                "rbs": np.zeros_like(np.asarray(cth_input.rbc)),
                "zbc": np.zeros_like(np.asarray(cth_input.zbs)),
                "raxis_s": np.zeros(cth_input.ntor + 1),
                "zaxis_c": np.zeros(cth_input.ntor + 1),
            }
        ),
        num_elements=2,
        degree=4,
    )
    expected = vmecpp.run(symmetric, max_threads=1, verbose=False).wout
    actual = vmecpp.run(asymmetric, max_threads=1, verbose=False).wout
    assert actual.niter == expected.niter
    for name in ("rmnc", "zmns", "bmnc"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), atol=1.0e-12
        )
    for name in ("lmns", "iotaf"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), atol=1.0e-11
        )
    for name in ("rmns", "zmnc", "lmnc", "bmns"):
        np.testing.assert_allclose(getattr(actual, name), 0.0, atol=1.0e-12)


def _racetrack_axis(v, half_length, rho):
    """R of a stadium axis at v and the stretch of its tube cut at phi = const."""
    phi = np.mod(v / 2.0, np.pi)
    phi = np.minimum(phi, np.pi - phi)
    end = phi <= np.arctan2(rho, half_length)
    r = np.empty_like(phi)
    stretch = np.empty_like(phi)
    r[end] = half_length * np.cos(phi[end]) + np.sqrt(
        rho**2 - (half_length * np.sin(phi[end])) ** 2
    )
    psi = np.arctan2(r[end] * np.sin(phi[end]), r[end] * np.cos(phi[end]) - half_length)
    stretch[end] = 1.0 / np.abs(np.cos(psi - phi[end]))
    r[~end] = rho / np.sin(phi[~end])
    stretch[~end] = 1.0 / np.abs(np.sin(phi[~end]))
    return r, stretch


def test_one_element_per_racetrack_section_converges(cth_input):
    """One element per racetrack section converges with the default quadrature."""
    degree, minor_radius = 6, 0.15
    breaks = np.array([0.0, 2.0 * np.arctan2(1.0, 1.0), np.pi])
    ntor = (len(breaks) - 1) * degree
    v = vmecpp.toroidal_element_nodes(breaks, degree, nfp=2)
    r_axis, stretch = _racetrack_axis(v, half_length=1.0, rho=1.0)
    mpol = 6
    rbc_nodes = np.zeros((mpol, 2 * ntor))
    zbs_nodes = np.zeros((mpol, 2 * ntor))
    rbc_nodes[0] = r_axis
    rbc_nodes[1] = minor_radius * stretch
    zbs_nodes[1] = minor_radius
    vmec_input = cth_input.model_copy(
        update={
            "nfp": 2,
            "mpol": mpol,
            "ntor": ntor,
            "ntheta": 0,
            "nzeta": 0,
            "rbc": np.zeros((mpol, 2 * ntor + 1)),
            "zbs": np.zeros((mpol, 2 * ntor + 1)),
            "raxis_c": np.zeros(ntor + 1),
            "zaxis_s": np.zeros(ntor + 1),
            "ncurr": 0,
            "piota_type": "power_series",
            "ai": np.array([0.3, 0.1]),
            "pres_scale": 0.0,
            "phiedge": np.pi * minor_radius**2,
            "toroidal_element_degree": degree,
            "toroidal_element_breaks": breaks,
            "rbc_nodes": rbc_nodes,
            "rbs_nodes": np.zeros((mpol, 2 * ntor)),
            "zbc_nodes": np.zeros((mpol, 2 * ntor)),
            "zbs_nodes": zbs_nodes,
            "raxis_nodes": r_axis,
            "zaxis_nodes": np.zeros(2 * ntor),
        }
    )
    wout = vmecpp.run(vmec_input, max_threads=1, verbose=False).wout
    assert wout.ier_flag == 0
    assert wout.niter < 5000


def test_element_fields_round_trip_through_json(cth_elements_input, tmp_path):
    path = tmp_path / "elements.json"
    cth_elements_input.save(path)
    reloaded = vmecpp.VmecInput.from_file(path)
    for name in ELEMENT_FIELDS:
        np.testing.assert_array_equal(
            getattr(reloaded, name), getattr(cth_elements_input, name)
        )


def test_hot_restart_with_elements_is_rejected(cth_elements_input, cth_elements_output):
    single_step = cth_elements_input.model_copy(
        update={
            "ns_array": cth_elements_input.ns_array[-1:],
            "ftol_array": cth_elements_input.ftol_array[-1:],
            "niter_array": cth_elements_input.niter_array[-1:],
        }
    )
    with pytest.raises(ValueError, match="hot restart is not available"):
        vmecpp.run(single_step, restart_from=cth_elements_output, verbose=False)


def test_traced_runs_with_elements_are_rejected(cth_input):
    vmec_input = _with_elements(cth_input, num_elements=2, degree=4)

    def aspect(rbc):
        update = {"rbc": rbc}
        return vmecpp.run(
            vmec_input.model_copy(update=update), verbose=False
        ).wout.aspect

    with pytest.raises(NotImplementedError, match="toroidal elements"):
        jax.grad(aspect)(jnp.asarray(vmec_input.rbc))


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"ntor": 6}, "ntor has to be"),
        ({"toroidal_element_breaks": np.array([0.0, 1.0, 3.0])}, "from 0 to pi"),
        ({"lbsubs": True}, "lbsubs"),
        ({"rbc_nodes": np.zeros((5, 3))}, "nodes needs shape"),
    ],
)
def test_inconsistent_element_inputs_are_rejected(cth_input, update, message):
    vmec_input = _with_elements(cth_input, num_elements=2, degree=4)
    with pytest.raises(ValueError, match=message):
        vmecpp.run(vmec_input.model_copy(update=update), verbose=False)

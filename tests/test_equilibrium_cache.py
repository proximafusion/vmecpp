# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for the content-addressed equilibrium cache."""

import copy
from pathlib import Path

import numpy as np
import pytest

import vmecpp
from vmecpp._cache import _final_step_input

REPO_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"


@pytest.fixture(scope="module")
def solovev_input() -> vmecpp.VmecInput:
    return vmecpp.VmecInput.from_file(TEST_DATA_DIR / "solovev.json")


@pytest.fixture(scope="module")
def populated_cache(
    solovev_input: vmecpp.VmecInput, tmp_path_factory: pytest.TempPathFactory
) -> tuple[vmecpp.EquilibriumCache, vmecpp.VmecOutput]:
    """One cache holding one cold solovev solve, shared across tests."""
    cache = vmecpp.EquilibriumCache(tmp_path_factory.mktemp("cache"))
    cold = cache.solve(solovev_input, verbose=False, max_threads=1)
    return cache, cold


# --- keys --------------------------------------------------------------------


def test_key_is_stable(solovev_input: vmecpp.VmecInput):
    key = vmecpp.cache_key(solovev_input)
    assert key == vmecpp.cache_key(solovev_input)
    assert key == vmecpp.cache_key(copy.deepcopy(solovev_input))
    # A round trip through validation does not move the key.
    revalidated = vmecpp.VmecInput.model_validate_json(solovev_input.model_dump_json())
    assert key == vmecpp.cache_key(revalidated)


def test_key_tracks_the_physics(solovev_input: vmecpp.VmecInput):
    perturbed = solovev_input.model_copy(deep=True)
    perturbed.rbc[1, 0] *= 1.0 + 1.0e-12
    assert vmecpp.cache_key(perturbed) != vmecpp.cache_key(solovev_input)

    coarser = solovev_input.model_copy(deep=True)
    coarser.ns_array = np.asarray(coarser.ns_array[:-1])
    coarser.ftol_array = np.asarray(coarser.ftol_array[:-1])
    coarser.niter_array = np.asarray(coarser.niter_array[:-1])
    assert vmecpp.cache_key(coarser) != vmecpp.cache_key(solovev_input)


def test_key_ignores_the_mgrid_path(tmp_path: Path):
    free_boundary_input = vmecpp.VmecInput.from_file(
        TEST_DATA_DIR / "solovev_free_bdy.json"
    )
    mgrid = REPO_ROOT / "src" / "vmecpp" / "cpp" / free_boundary_input.mgrid_file
    key = vmecpp.cache_key(
        free_boundary_input.model_copy(update={"mgrid_file": str(mgrid)})
    )

    moved = tmp_path / "renamed_mgrid.nc"
    moved.write_bytes(mgrid.read_bytes())
    relocated = free_boundary_input.model_copy(update={"mgrid_file": str(moved)})
    assert vmecpp.cache_key(relocated) == key

    moved.write_bytes(mgrid.read_bytes() + b"tampered")
    assert vmecpp.cache_key(relocated) != key


# --- hits --------------------------------------------------------------------


def test_unverified_hit_returns_the_stored_output(
    solovev_input: vmecpp.VmecInput,
    populated_cache: tuple[vmecpp.EquilibriumCache, vmecpp.VmecOutput],
):
    cache, cold = populated_cache
    hit = cache.solve(solovev_input, verify=False, verbose=False, max_threads=1)
    assert hit.wout.niter == cold.wout.niter
    np.testing.assert_array_equal(hit.wout.rmnc, cold.wout.rmnc)
    np.testing.assert_array_equal(hit.wout.lmns, cold.wout.lmns)
    assert hit.wout.b0 == cold.wout.b0


def test_verified_hit_reconverges_from_the_stored_state(
    solovev_input: vmecpp.VmecInput,
    populated_cache: tuple[vmecpp.EquilibriumCache, vmecpp.VmecOutput],
):
    cache, cold = populated_cache
    hit = cache.solve(solovev_input, verbose=False, max_threads=1)
    # Starting from the converged state, force balance is re-established in a
    # couple of iterations rather than the cold count.
    assert hit.wout.niter < cold.wout.niter / 5
    assert hit.wout.b0 == pytest.approx(cold.wout.b0, rel=1e-8)
    assert hit.wout.volume_p == pytest.approx(cold.wout.volume_p, rel=1e-8)


# --- misses ------------------------------------------------------------------


def test_warm_start_from_the_nearest_neighbor(
    solovev_input: vmecpp.VmecInput,
    populated_cache: tuple[vmecpp.EquilibriumCache, vmecpp.VmecOutput],
):
    cache, _ = populated_cache
    perturbed = solovev_input.model_copy(deep=True)
    perturbed.rbc[1, 0] *= 1.0 + 1.0e-5

    cold = vmecpp.run(perturbed, verbose=False, max_threads=1)
    cold_single_grid = vmecpp.run(
        _final_step_input(perturbed), verbose=False, max_threads=1
    )
    warm = cache.solve(perturbed, verbose=False, max_threads=1)

    # The warm solve runs the final grid only, seeded by the stored neighbor,
    # and beats the cold solve of that same grid.
    assert warm.wout.niter < cold_single_grid.wout.niter
    # Two convergence paths into the same minimum agree to the hot restart
    # spread, not to ftol.
    assert warm.wout.b0 == pytest.approx(cold.wout.b0, rel=1e-4)
    assert warm.wout.volume_p == pytest.approx(cold.wout.volume_p, rel=1e-4)

    # The perturbed equilibrium is now stored under its own key and answers
    # a repeat without warm-starting.
    assert vmecpp.cache_key(perturbed) in cache
    assert len(cache) == 2


def test_a_fresh_cache_reads_entries_from_disk(
    solovev_input: vmecpp.VmecInput,
    populated_cache: tuple[vmecpp.EquilibriumCache, vmecpp.VmecOutput],
):
    cache, cold = populated_cache
    reopened = vmecpp.EquilibriumCache(cache.path)
    assert vmecpp.cache_key(solovev_input) in reopened
    hit = reopened.solve(solovev_input, verify=False, verbose=False, max_threads=1)
    assert hit.wout.b0 == cold.wout.b0


# --- async -------------------------------------------------------------------


def test_submit_resolves_to_the_same_result(
    solovev_input: vmecpp.VmecInput,
    populated_cache: tuple[vmecpp.EquilibriumCache, vmecpp.VmecOutput],
):
    cache, cold = populated_cache
    with cache:
        future = cache.submit(solovev_input, verify=False, verbose=False, max_threads=1)
        hit = future.result(timeout=300)
    assert hit.wout.b0 == cold.wout.b0

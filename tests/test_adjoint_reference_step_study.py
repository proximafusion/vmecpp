"""Replay saved WP6 directional studies without an equilibrium solve."""

import hashlib
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = (
    Path(__file__).resolve().parents[1] / "docs" / "adjoint_verification" / "scripts"
)


@pytest.mark.parametrize("ad", [2.0, 1e-9])
def test_saved_directional_errors_are_replayed(tmp_path, monkeypatch, ad):
    pytest.importorskip("matplotlib")
    source = tmp_path / "plot_02_taylor_small_random_linear_m1.npz"
    h = np.geomspace(1e-2, 1e-5, 4)
    fd = 2 + h**2
    np.savez(
        source,
        steps=h,
        central_fd=fd,
        directional_ad=np.asarray(ad),
        noise_floor=np.asarray(1e-8),
        direction_name=np.asarray("m1"),
        status=np.asarray("unresolved"),
        provenance=np.asarray('{"reference":"test oracle"}'),
        acceptance_threshold=1e-6 * fd,
    )
    output = tmp_path / "output"
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot", "--data-dir", str(tmp_path), "--output-dir", str(output)],
    )
    runpy.run_path(str(SCRIPTS / "plot_reference_step_study.py"), run_name="__main__")
    stem = "plot_02_fd_step_study_small_random_linear"
    assert (output / f"{stem}.png").is_file()
    with np.load(output / f"{stem}.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["steps_0"], h)
        np.testing.assert_array_equal(saved["central_fd_0"], fd)
        np.testing.assert_allclose(saved["noise_scale_0"], 1e-8 / h)
        np.testing.assert_array_equal(saved["acceptance_threshold_0"], 1e-6 * fd)
        assert saved["status_0"].item() == "unresolved"
        assert (
            saved["source_sha256_0"].item()
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )


def test_incompatible_directional_studies_are_rejected(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    for seed, direction in enumerate(("m1", "highm")):
        np.savez(
            tmp_path / f"plot_02_taylor_small_random_linear_{direction}.npz",
            steps=np.asarray([1e-2, 1e-3]),
            central_fd=np.asarray([2.01, 2.001]),
            directional_ad=np.asarray(2.0),
            noise_floor=np.asarray(1e-8),
            direction_name=np.asarray(direction),
            status=np.asarray("unresolved"),
            provenance=np.asarray(f'{{"reference":"test oracle","seed":{seed}}}'),
            acceptance_threshold=np.asarray([2.01e-6, 2.001e-6]),
        )
    output = tmp_path / "output"
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot", "--data-dir", str(tmp_path), "--output-dir", str(output)],
    )
    with pytest.raises(ValueError, match="incompatible saved study provenance"):
        runpy.run_path(
            str(SCRIPTS / "plot_reference_step_study.py"), run_name="__main__"
        )
    assert not output.exists()

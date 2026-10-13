"""Deterministic rendering of synthetic WP7 data, without an Enzyme runtime."""

import runpy
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import matplotlib.figure
import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "adjoint_verification"
    / "scripts"
    / "plot_mercier_corrections.py"
)
PLOT = runpy.run_path(str(SCRIPT))


def test_scientific_measurement_enables_x64_and_rejects_float32():
    requests = []
    config = SimpleNamespace(x64_enabled=False)

    def enable(name, value):
        requests.append((name, value))
        config.x64_enabled = value

    config.update = enable
    PLOT["enable_x64"](config)
    assert requests == [("jax_enable_x64", True)]
    assert config.x64_enabled
    config.x64_enabled = False
    config.update = lambda *_args: None
    with pytest.raises(RuntimeError, match="no float32 result is accepted"):
        PLOT["enable_x64"](config)


def test_synthetic_mercier_plot_preserves_named_records(monkeypatch):
    data = PLOT["initial_data"]("small")
    data["status"] = "synthetic_not_measured"
    data["fd_relative_error"] = np.asarray(
        [[1e-4, 2e-4, 3e-4, 4e-4], [1e-5, 2e-5, 3e-5, 4e-5], [1e-6] * 4]
    )
    data["correction_residual"] = np.broadcast_to(
        np.asarray([[1e-4], [1e-8]]), (2, 4)
    ).copy()
    data["gradient_relative_error"] = np.broadcast_to(
        np.asarray([[[1e-3, 1e-5]], [[1e-7, 1e-9]]]), (2, 4, 2)
    ).copy()
    data["value_absolute_error"] = data["gradient_relative_error"] * 0.1
    data["reference_residual_history"] = np.asarray([1e-4, 1e-8, 1e-13])
    output = Path("plot_08_mercier_corrections_synthetic_memory")
    arrays, image = BytesIO(), BytesIO()
    replacements = []
    savefig = matplotlib.figure.Figure.savefig

    def save_image(figure, destination, **kwargs):
        assert destination == output / "plot_08_mercier_corrections_small.png.part"
        savefig(figure, image, **kwargs)

    def open_arrays(path, mode):
        assert path == output / "plot_08_mercier_corrections_small.npz.part"
        assert mode == "wb"
        return nullcontext(arrays)

    with monkeypatch.context() as patches:
        patches.setattr(Path, "mkdir", lambda *_args, **_kwargs: None)
        patches.setattr(Path, "open", open_arrays)
        patches.setattr(
            Path,
            "replace",
            lambda source, target: replacements.append((source, target)),
        )
        patches.setattr(matplotlib.figure.Figure, "savefig", save_image)
        PLOT["checkpoint"](data, output, "small")
        PLOT["render"](data, output, "small")
    assert image.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert [(source.suffix, target.suffix) for source, target in replacements] == [
        (".part", ".npz"),
        (".part", ".png"),
    ]
    arrays.seek(0)
    with np.load(arrays, allow_pickle=False) as saved:
        assert saved["status"].item() == "synthetic_not_measured"
        assert saved["solve_status"].tolist() == ["pending"] * 7
        assert saved["terms"].tolist() == ["DMerc", "DCurr", "DGeod", "DWell"]
        for name in (
            "steps",
            "fd_relative_error",
            "correction_residual",
            "gradient_relative_error",
            "value_absolute_error",
            "reference_residual_history",
        ):
            np.testing.assert_array_equal(saved[name], data[name])

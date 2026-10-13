"""WP0 plot naming and reusable numerical data."""

import runpy
from pathlib import Path

import numpy as np
import pytest

STYLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "adjoint_verification"
    / "scripts"
    / "plot_style.py"
)
PLOT_STYLE = runpy.run_path(str(STYLE_PATH))


def test_plot_and_named_data_are_saved_together(tmp_path):
    pyplot = pytest.importorskip("matplotlib.pyplot")
    figure, axes = pyplot.subplots()
    x = np.arange(3.0)
    y = x**2
    axes.plot(x, y)
    try:
        PLOT_STYLE["save_plot"](figure, tmp_path, "plot_00_example_small", x=x, y=y)
    finally:
        pyplot.close(figure)
    assert (tmp_path / "plot_00_example_small.png").is_file()
    with np.load(tmp_path / "plot_00_example_small.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["x"], x)
        np.testing.assert_array_equal(data["y"], y)


@pytest.mark.parametrize("stem", ["../plot_escape", "example"])
def test_invalid_plot_names_are_rejected(tmp_path, stem):
    with pytest.raises(ValueError, match="plot stem"):
        PLOT_STYLE["save_plot"](None, tmp_path, stem, x=np.ones(1))


def test_object_data_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="without object dtype"):
        PLOT_STYLE["save_plot"](
            None, tmp_path, "plot_00_objects", x=np.asarray([{}], dtype=object)
        )


def test_mode_colors_are_stable():
    pytest.importorskip("matplotlib")
    mode_color = PLOT_STYLE["mode_color"]
    assert mode_color(4) == mode_color(4)
    assert mode_color(4) != mode_color(8)

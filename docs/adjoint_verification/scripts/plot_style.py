"""Plot conventions for reusable adjoint-verification PNG/NPZ pairs.

Use ``plot_<number:02d>_<topic>_<case>`` as the file stem. Data arrays must be
named, and include the abscissa and every plotted ordinate. Mode colors depend
only on m, so comparisons across resolutions retain the same color assignment.
Matplotlib is imported lazily; it is not a solver dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

STYLE: dict[Any, Any] = {
    "figure.figsize": (7.0, 4.5),
    "figure.dpi": 120,
    "savefig.dpi": 160,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
}


def mode_color(m: int):
    """Stable tab20 color for a nonnegative poloidal mode."""
    import matplotlib as mpl  # noqa: PLC0415 - optional plotting dependency

    if m < 0:
        error_message = "poloidal mode m must be nonnegative"
        raise ValueError(error_message)
    return mpl.colormaps["tab20"](m % 20)


def save_plot(figure, directory: Path, stem: str, **data: np.ndarray) -> None:
    """Save the plot and its named numerical data without pickled objects."""
    if Path(stem).name != stem or not stem.startswith("plot_"):
        error_message = "plot stem must be a plain filename starting with plot_"
        raise ValueError(error_message)
    arrays = {name: np.asarray(value) for name, value in data.items()}
    if (
        not arrays
        or "allow_pickle" in arrays
        or any(array.dtype.hasobject for array in arrays.values())
    ):
        error_message = "plot data must contain named arrays without object dtype"
        raise ValueError(error_message)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"{stem}.npz", allow_pickle=False, **arrays)
    figure.savefig(directory / f"{stem}.png", bbox_inches="tight")

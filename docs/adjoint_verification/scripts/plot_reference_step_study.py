"""Reuse WP6 Taylor data to plot the directional FD error and reference noise.

This script does not solve an equilibrium or require Enzyme. It shows why a quadratic
remainder alone does not certify a directional derivative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plot_style import STYLE, save_plot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", default="small")
    parser.add_argument("--objective", default="random_linear")
    args = parser.parse_args()
    paths = sorted(
        args.data_dir.glob(f"plot_02_taylor_{args.case}_{args.objective}_*.npz")
    )
    if not paths:
        message = "no saved Taylor NPZ data match the requested case and objective"
        raise ValueError(message)
    arrays = {}
    study_provenance = None
    with plt.rc_context(STYLE):
        figure, axes = plt.subplots(
            1, len(paths), figsize=(4 * len(paths), 3.5), squeeze=False
        )
        try:
            for index, path in enumerate(paths):
                with np.load(path, allow_pickle=False) as saved:
                    h = saved["steps"]
                    fd = saved["central_fd"]
                    ad = float(saved["directional_ad"])
                    noise = float(saved["noise_floor"])
                    direction = str(saved["direction_name"].item())
                    status = str(saved["status"].item())
                    provenance = str(saved["provenance"].item())
                source_provenance = json.loads(provenance)
                if study_provenance is None:
                    study_provenance = source_provenance
                elif source_provenance != study_provenance:
                    message = f"incompatible saved study provenance: {path}"
                    raise ValueError(message)
                if (
                    h.ndim != 1
                    or fd.shape != h.shape
                    or not np.isfinite(h).all()
                    or np.any(h <= 0)
                    or not np.isfinite(ad)
                    or not np.isfinite(noise)
                    or noise < 0
                ):
                    message = f"invalid saved directional study: {path}"
                    raise ValueError(message)
                error = np.abs(fd - ad)
                noise_scale = noise / h
                axis = axes[0, index]
                axis.loglog(
                    h, np.where(error > 0, error, np.nan), "o-", label="|FD - AD|"
                )
                axis.loglog(
                    h,
                    np.where(noise_scale > 0, noise_scale, np.nan),
                    "--",
                    label="measured value noise / h",
                )
                axis.axhline(
                    1e-6 * abs(ad) if abs(ad) > 1e-8 else 1e-8,
                    color="black",
                    linestyle=":",
                    label="derivative target",
                )
                axis.set(
                    xlabel="central-difference step h",
                    ylabel="absolute directional-derivative error",
                    title=f"{direction}: {status}",
                )
                axis.legend(fontsize=7)
                arrays[f"steps_{index}"] = h
                arrays[f"central_fd_{index}"] = fd
                arrays[f"directional_ad_{index}"] = np.asarray(ad)
                arrays[f"noise_scale_{index}"] = noise_scale
                arrays[f"direction_{index}"] = np.asarray(direction)
                arrays[f"status_{index}"] = np.asarray(status)
                arrays[f"provenance_{index}"] = np.asarray(provenance)
                arrays[f"source_sha256_{index}"] = np.asarray(
                    hashlib.sha256(path.read_bytes()).hexdigest()
                )
            figure.tight_layout()
            save_plot(
                figure,
                args.output_dir,
                f"plot_02_fd_step_study_{args.case}_{args.objective}",
                **arrays,
            )
        finally:
            plt.close(figure)


if __name__ == "__main__":
    main()

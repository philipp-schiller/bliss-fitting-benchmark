# scripts/Benchmarking/analysis/plot_ellipsoid_sweep.py

from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import trim_mean

from core.sweep_result import SweepResult
from metrics.ellipsoid_metrics import (
    compute_ellipsoid_metrics,
)


MetricReducer = Callable[[np.ndarray], float]

# Distinguishable in both color and grayscale.
MARKERS = [
    "o",   # circle
    "s",   # square
    "^",   # triangle up
    "D",   # diamond
    "v",   # triangle down
    "P",   # filled plus
    "X",   # filled x
    "*",   # star
]

LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1)),
    (0, (5, 2, 1, 2)),
]


def trimmed_nanmean(
    values: np.ndarray,
    proportion_to_cut: float = 0.0,
) -> float:
    """
    Compute a trimmed mean after removing NaN and infinite values.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan

    return float(
        trim_mean(
            values,
            proportiontocut=proportion_to_cut,
        )
    )


def collect_ellipsoid_sweep(
    sweep_values: Sequence[float],
    result_directories: Sequence[str | Path],
    reducer: MetricReducer = trimmed_nanmean,
) -> dict:
    """
    Load and aggregate an ellipsoid sweep.

    Each result directory must contain a sweep_result.npz file.
    """

    if len(sweep_values) != len(result_directories):
        raise ValueError(
            "sweep_values and result_directories must have "
            "the same length."
        )

    sweep_values = np.asarray(sweep_values, dtype=float)

    center_errors: dict[str, list[float]] = {}
    volume_errors: dict[str, list[float]] = {}
    shape_errors: dict[str, list[float]] = {}
    fitting_times: dict[str, list[float]] = {}

    expected_labels = None

    for result_directory in result_directories:
        result = SweepResult.load(result_directory)

        if expected_labels is None:
            expected_labels = result.labels
        elif result.labels != expected_labels:
            raise ValueError(
                "The fitter labels are not identical for all "
                f"sweep points. Mismatch at {result_directory}."
            )

        metrics = compute_ellipsoid_metrics(result)

        for fitter_index, method in enumerate(result.labels):
            center_values = metrics["Ec"][:, fitter_index]
            volume_values = metrics["Evolume"][:, fitter_index]
            shape_values = metrics["Ea"][:, fitter_index]
            time_values = result.fitting_times[:, fitter_index]

            center_errors.setdefault(method, []).append(
                reducer(center_values)
            )

            volume_errors.setdefault(method, []).append(
                reducer(volume_values)
            )

            shape_errors.setdefault(method, []).append(
                reducer(shape_values)
            )

            fitting_times.setdefault(method, []).append(
                reducer(time_values)
            )

    return {
        "values": sweep_values,
        "labels": expected_labels or [],
        "center_errors": center_errors,
        "volume_errors": volume_errors,
        "shape_errors": shape_errors,
        "fitting_times": fitting_times,
    }


def plot_ellipsoid_sweep(
    sweep_data: dict,
    x_label: str,
    save_path: str | Path,
    invert_x_axis: bool = False,
    log_error_axes: bool = True,
    show: bool = False,
) -> Path:
    """
    Plot volume, center, and shape errors together with fitting time.
    """

    sweep_values = sweep_data["values"]

    methods = sweep_data["labels"]

    method_styles = {
        method: {
            "marker": MARKERS[index % len(MARKERS)],
            "linestyle": LINESTYLES[index % len(LINESTYLES)],
        }
        for index, method in enumerate(methods)
    }

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12.5, 3.8),
    )

    panels = [
        (
            sweep_data["volume_errors"],
            r"Avg. volume deviation $E_{\mathrm{volume}}$",
        ),
        (
            sweep_data["center_errors"],
            r"Avg. centre deviation $E_c$",
        ),
        (
            sweep_data["shape_errors"],
            r"Avg. shape deviation $E_{\mathrm{shape}}$",
        ),
        (
            sweep_data["fitting_times"],
            "Avg. computation time (s)",
        ),
    ]

    for panel_index, (values_by_method, y_label) in enumerate(
        panels
    ):
        ax = axes[panel_index]

        # ----------------------------
        # Plot all methods
        # ----------------------------
        for method, values in values_by_method.items():
            style = method_styles[method]

            plot_values = np.asarray(
                values,
                dtype=float,
            ).copy()

            plot_values[plot_values > 1e3] = np.nan

            ax.plot(
                sweep_values,
                plot_values,
                marker=style["marker"],
                # linestyle=style["linestyle"],
                linewidth=2,
                label=method,
            )

        ax.set_xlabel(
            x_label,
            fontsize=12,
        )

        ax.set_ylabel(
            y_label,
            fontsize=12,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        if log_error_axes and panel_index < 3:
            ax.set_yscale("log")

        if invert_x_axis:
            ax.invert_xaxis()

    handles, legend_labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        fontsize=12,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)

    save_path = Path(save_path)

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )

    vector_path = save_path.with_suffix(".pdf")

    fig.savefig(
        vector_path,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved plot: {save_path}")
    print(f"Saved plot: {vector_path}")

    return save_path
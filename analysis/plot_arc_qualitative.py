# scripts/Benchmarking/analysis/plot_arc_qualitative.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayes_shape_fitting.visualization.plotting_2d import (
    ShapePlotter2D,
)

from core.sweep_result import SweepResult
from experiments.exp_2_varying_arc_ellipse import make_arc_length_experiment

from methods.ellipse_fitters.ellipse_utils import canonicalize_ellipse_params


def plot_arc_qualitative(
    repository_root: Path,
    arc_half_widths: list[float],
    iteration_indices: list[int],
    save_directory: Path,
    show: bool = False,
) -> Path:
    """
    Plot individual ellipse fits for selected arc coverages.

    One panel is generated for each combination of arc half-width and
    selected Monte Carlo iteration.
    """

    if len(arc_half_widths) != len(iteration_indices):
        raise ValueError(
            "arc_half_widths and iteration_indices must have "
            "the same length."
        )

    n_panels = len(arc_half_widths)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(3.3 * n_panels, 3.3),
        squeeze=False,
    )

    axes = axes.ravel()

    plotter = ShapePlotter2D()

    color_cycle = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    for panel_index, (w, iteration) in enumerate(
        zip(arc_half_widths, iteration_indices)
    ):
        # The config is needed for the shape object.
        # Importing or constructing it does not run the experiment.
        config = make_arc_length_experiment(
            w=float(w),
            iterations=1,
        )

        result_directory = (
            repository_root
            / "results"
            / "ellipse"
            / "arc_length_sweep"
            / f"w_{w:.2f}"
        )

        pointcloud_directory = (
            result_directory / "pointclouds"
        )

        result = SweepResult.load(result_directory)

        if not 0 <= iteration < result.fitting_params.shape[0]:
            raise IndexError(
                f"Iteration {iteration} is unavailable for "
                f"w={w:.2f}. The result contains "
                f"{result.fitting_params.shape[0]} iterations."
            )

        shape = config.shape
        shape_gt = result.shape_gt[iteration]
        fitting_params = result.fitting_params
        labels = result.labels

        ax = axes[panel_index]

        ax.set_aspect(
            "equal",
            adjustable="box",
        )
        ax.set_box_aspect(1)

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")

        # Plot limits based on the ground-truth ellipse.
        margin_factor_x = 1.7
        margin_factor_y = 1.9

        x_radius = max(
            abs(shape_gt[2]),
            abs(shape_gt[3]),
        )
        y_radius = x_radius

        ax.set_xlim(
            shape_gt[0] - margin_factor_x * x_radius,
            shape_gt[0] + margin_factor_x * x_radius,
        )

        ax.set_ylim(
            shape_gt[1] - margin_factor_y * y_radius,
            shape_gt[1] + margin_factor_y * y_radius,
        )

        # --------------------------------------------------------------
        # Ground truth
        # --------------------------------------------------------------

        canonical_ground_truth = canonicalize_ellipse_params(
            shape_gt
        )

        plotter.plot_shape(
            shape,
            canonical_ground_truth,
            color="black",
            ax=ax,
            linewidth=2,
            linestyle="dashed",
            label="Ground Truth",
        )

        # --------------------------------------------------------------
        # Individual fitted ellipses
        # --------------------------------------------------------------

        for fitter_index, label in enumerate(labels):
            fitted_params = fitting_params[
                iteration,
                fitter_index,
                :,
            ]

            if not np.all(np.isfinite(fitted_params)):
                continue


            color = color_cycle[
                fitter_index % len(color_cycle)
            ]

            plotter.plot_shape(
                shape,
                fitted_params,
                color=color,
                ax=ax,
                linewidth=2.0,
                label=label,
            )

        # --------------------------------------------------------------
        # Load the exact point cloud used for this iteration
        # --------------------------------------------------------------

        pointcloud_path = (
            pointcloud_directory
            / (
                f"pointcloud_{config.name}_"
                f"{iteration:04d}.npz"
            )
        )

        if not pointcloud_path.exists():
            raise FileNotFoundError(
                "Point-cloud file not found:\n"
                f"{pointcloud_path}"
            )

        with np.load(
            pointcloud_path,
            allow_pickle=False,
        ) as pointcloud_data:
            observations = pointcloud_data["points"].copy()

        ax.scatter(
            observations[:, 0],
            observations[:, 1],
            s=10,
            alpha=0.35,
            color="gray",
            label="Observations",
            zorder=1,
        )

        coverage = int(round(200 * w))

        ax.set_xlabel(
            f"Arc coverage: {coverage}%",
            fontsize=10,
        )

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------

    legend_entries = {}

    for ax in axes:
        handles, legend_labels = (
            ax.get_legend_handles_labels()
        )

        for handle, label in zip(
            handles,
            legend_labels,
        ):
            if label and label not in legend_entries:
                legend_entries[label] = handle

    fig.legend(
        legend_entries.values(),
        legend_entries.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(legend_entries),
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout(
        rect=[0, 0.12, 1, 1]
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        save_directory
        / "arc_qualitative.png"
    )

    pdf_path = (
        save_directory
        / "arc_qualitative.pdf"
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")

    return png_path


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]

    # One selected iteration for each arc half-width.
    arc_half_widths = [
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10
    ]

    iteration_indices = [
        4,5,6,7,8,9,
    ]

    save_directory = (
        repository_root
        / "results"
        / "ellipse"
        / "arc_length_sweep"
        / "plots"
    )

    plot_arc_qualitative(
        repository_root=repository_root,
        arc_half_widths=arc_half_widths,
        iteration_indices=iteration_indices,
        save_directory=save_directory,
        show=True,
    )
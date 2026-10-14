# scripts/Benchmarking/analysis/plot_arc_qualitative.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayes_shape_fitting.visualization.plotting_2d import (
    ShapePlotter2D,
)
from networkx.algorithms.smallworld import sigma

from core.sweep_result import SweepResult
from experiments.exp_2_varying_arc_ellipse import make_arc_length_experiment
from experiments.exp_4_laser_noise import make_laser_noise, make_laser_experiment

from methods.ellipse_fitters.ellipse_utils import canonicalize_ellipse_params


def plot_laser_qualitative(
    repository_root: Path,
    noises: list[float],
    iteration_indices: list[int],
    save_directory: Path,
    show: bool = False,
    png_name: str = None
) -> Path:
    """
    Plot individual ellipse fits for selected arc coverages.

    One panel is generated for each combination of arc half-width and
    selected Monte Carlo iteration.
    """

    if len(noises) != len(iteration_indices):
        raise ValueError(
            "noises and iteration_indices must have "
            "the same length."
        )

    n_panels = len(noises)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(8.2, 3.4),

        #figsize=(3.3 * n_panels, 3.3),
        squeeze=False,
    )
    fig.subplots_adjust(wspace=0.05)

    axes = axes.ravel()

    plotter = ShapePlotter2D()

    color_cycle = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    for panel_index, (sigma_0, iteration) in enumerate(
        zip(noises, iteration_indices)
    ):
        # The config is needed for the shape object.
        # Importing or constructing it does not run the experiment.
        config = make_laser_experiment(sigma_0=sigma_0)


        result_directory = (
            repository_root
            / "results"
            / "ellipse"
            / "laser_noise_sweep"
            / f"sigma_0_{sigma_0:.3f}"
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
        margin_factor_x = 1.3
        margin_factor_y = 1.15

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

        canonical_ground_truth = shape_gt

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
            alpha=0.55,
            color="black",
            #label="Observations",
            zorder=1,
        )

        ax.set_xlabel(
            rf"$\sigma_0 = {sigma_0:.2f}$",
            fontsize=14,
        )

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------

    reorder = lambda hl, nc: (sum((lis[i::nc] for i in range(nc)), []) for lis in hl)

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

    handles, labels = reorder(
        (
            list(legend_entries.values()),
            list(legend_entries.keys()),
        ),
        5,
    )

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol = 5,
        frameon=True,
        fontsize=14,
    )

    fig.tight_layout(
        rect=[0, 0.25, 1, 1]
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    save_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if png_name is None:
        png_name = "laser_qualitative.png"

    png_path = (
        save_directory
        / png_name
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved plot: {png_path}")

    return png_path


if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]

    for i in range(1):

        # One selected iteration for each arc half-width.
        noises = [0.01,0.03,0.05]

        iteration_indices = [
         19,57,5
        ]

        save_directory = (
            repository_root
            / "results"
            / "ellipse"
            / "laser_noise_sweep"
            / "plots"
        )

        plot_laser_qualitative(
            repository_root=repository_root,
            noises=noises,
            iteration_indices=iteration_indices,
            save_directory=save_directory,
            show=False,
            png_name=f"laser_individual.eps"
        )
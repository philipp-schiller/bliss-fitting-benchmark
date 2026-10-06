# analysis/plot_noise_qualitative.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bayes_shape_fitting.visualization.plotting_2d import (
    ShapePlotter2D,
)

from core.sweep_result import SweepResult
from experiments.exp_3_varying_noise_ellipse import (
    make_noise_experiment,
)
from methods.ellipse_fitters.ellipse_utils import (
    canonicalize_ellipse_params,
)


def plot_noise_qualitative(
    repository_root: Path,
    noise_variances: list[float],
    iteration_indices: list[int],
    save_directory: Path,
    show: bool = False,
    png_name = None
) -> Path:
    """
    Plot individual ellipse fits for selected noise variances.

    One panel is generated for each combination of noise variance
    and selected Monte Carlo iteration.
    """

    if len(noise_variances) != len(iteration_indices):
        raise ValueError(
            "noise_variances and iteration_indices must have "
            "the same length."
        )

    n_panels = len(noise_variances)

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

    for panel_index, (sigma_sq, iteration) in enumerate(
        zip(noise_variances, iteration_indices)
    ):
        # The config provides the shape and experiment name.
        # Constructing the config does not execute the experiment.
        config = make_noise_experiment(
            sigma_sq=float(sigma_sq),
            iterations=1,
        )

        result_directory = (
            repository_root
            / "results"
            / "ellipse"
            / "noise_sweep"
            / f"sigma_sq_{sigma_sq:.5f}"
        )

        pointcloud_directory = (
            result_directory
            / "pointclouds"
        )

        result = SweepResult.load(result_directory)

        n_iterations = result.fitting_params.shape[0]

        if not 0 <= iteration < n_iterations:
            raise IndexError(
                f"Iteration {iteration} is unavailable for "
                f"sigma squared={sigma_sq:.5f}. "
                f"The result contains {n_iterations} iterations."
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

        # --------------------------------------------------------------
        # Plot limits based only on the ground-truth ellipse
        # --------------------------------------------------------------

        canonical_ground_truth = canonicalize_ellipse_params(
            shape_gt
        )

        plot_radius = max(
            abs(canonical_ground_truth[2]),
            abs(canonical_ground_truth[3]),
        )

        margin_factor_x = 1.7
        margin_factor_y = 1.9

        ax.set_xlim(
            canonical_ground_truth[0]
            - margin_factor_x * plot_radius,
            canonical_ground_truth[0]
            + margin_factor_x * plot_radius,
        )

        ax.set_ylim(
            canonical_ground_truth[1]
            - margin_factor_y * plot_radius,
            canonical_ground_truth[1]
            + margin_factor_y * plot_radius,
        )

        # --------------------------------------------------------------
        # Ground truth
        # --------------------------------------------------------------

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
                print(
                    f"Skipping failed fit: "
                    f"sigma squared={sigma_sq:.5f}, "
                    f"iteration={iteration}, "
                    f"method={label}"
                )
                continue

            canonical_fitted_params = (
                canonicalize_ellipse_params(
                    fitted_params
                )
            )

            color = color_cycle[
                fitter_index % len(color_cycle)
            ]

            plotter.plot_shape(
                shape,
                canonical_fitted_params,
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

        ax.set_xlabel(
            rf"$\sigma^2 = {sigma_sq:.3f}$",
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

    if png_name is None:
        png_name = "noise_qualitative.png"

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

    for i in range(100):
        # One selected iteration for each fixed noise variance.
        noise_variances = [
            0.010,
            0.050,
            0.090,
        ]

        iteration_indices = [
            i, i, i
        ]

        save_directory = (
            repository_root
            / "results"
            / "ellipse"
            / "noise_sweep"
            / "plots"
        )

        plot_noise_qualitative(
            repository_root=repository_root,
            noise_variances=noise_variances,
            iteration_indices=iteration_indices,
            save_directory=save_directory,
            show=False,
            png_name=f"noise_{i}"
        )
"""
Here we define plotting functions for a Sweep Result
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
from bliss.visualization.plotting_2d import ShapePlotter2D
from matplotlib import patches
import numpy as np

from core.sweep_result import SweepResult
from core.experiment_config import ExperimentConfig

def plot_sweep_result(
    result: SweepResult,
    config: ExperimentConfig,
    *,
    iteration: int = 0,
    pointcloud_dir: str | Path,
    save_dir: str | Path = "plots",
    ) -> Path:
    """
    Plot fitted shapes and*the stored point cloud for a sweep*result.

    Parameters
    ------*---
    result:
        Numerical output returned by run_experiment()*

    config:
        Configuration used to produce the result.

    *teration:
        Iteration used for the displayed ground truth and point cloud.

    pointcloud_dir:
  *     Directory containing the stor*d point-cloud files.

    save_dir:
        Directory in which the figure is saved.

    Returns
    -------
    Path
        Path of the saved figure.
    """

    if not 0 <= iteration < result.fitting_params.shape[0]:
        raise IndexError(
            f"iteration must be between 0 and "
            f"{result.fitting_params.shape[0] - 1}, "
            f"but received {iteration}."
        )

    if result.fitting_params.shape[1] != len(config.fitters):
        raise ValueError(
            "The number of fitters in the result does not match "
            "the number of fitters in the configuration."
        )

    if result.labels and result.labels != config.labels:
        raise ValueError(
            "The labels stored in SweepResult do not match "
            "the labels in ExperimentConfig."
        )

    shape = config.shape
    labels = config.labels
    fitting_params = result.fitting_params
    shape_gt = result.shape_gt[iteration]

    n_fitters = len(labels)

    # One panel for each fitter and one panel for the point-cloud example.
    n_panels = n_fitters
    n_rows = 1
    n_cols = math.ceil(n_panels / n_rows)

    plotter = ShapePlotter2D()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4), sharex=True, sharey=True)
    fig.set_size_inches(19.2, 5.8)
    fig.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.1)

    axes = axes.ravel()

    # These limits are currently ellipse-specific.
    x_range = shape_gt[2] * 1.5
    y_range = shape_gt[3] * 2.0

    x_limits = (
        shape_gt[0] - x_range,
        shape_gt[0] + x_range,
    )
    y_limits = (
        shape_gt[1] - y_range,
        shape_gt[1] + y_range,
    )

    # ------------------------------------------------------------------
    # Fitting result panels
    # ------------------------------------------------------------------

    for fitter_index, label in enumerate(labels):
        ax = axes[fitter_index]

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=20)
        ax.grid(True, alpha=1.0)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)

        valid = ~np.isnan(
            fitting_params[:, fitter_index, :]
        ).any(axis=1)

        # Plot all successful fits across the sweep.
        first_fit = True

        for fitted_params in fitting_params[
            valid,
            fitter_index,
            :,
        ]:
            plotter.plot_shape(
                shape,
                fitted_params,
                label=(
                    "Fitted Ellipses"
                    if first_fit
                    else None
                ),
                color="red",
                ax=ax,
            )
            first_fit = False

        plotter.plot_shape(
            shape,
            shape_gt,
            label="Ground Truth",
            color="black",
            ax=ax,
        )

        ax.plot(
            shape_gt[0],
            shape_gt[1],
            marker="o",
            linestyle="none",
            color="black",
            label="True Center",
        )

        if np.any(valid):
            average_center = np.nanmean(
                fitting_params[
                    valid,
                    fitter_index,
                    :2,
                ],
                axis=0,
            )

            ax.plot(
                average_center[0],
                average_center[1],
                marker="*",
                markersize=12,
                linestyle="none",
                color="blue",
                label="Average Center",
            )

    # ------------------------------------------------------------------
    # Point-cloud panel
    # ------------------------------------------------------------------

    """example_ax = axes[n_fitters]

    example_ax.set_aspect("equal", adjustable="box")
    example_ax.set_title(
        f"Example Point Sample, Iteration {iteration}",
        fontsize=20,
    )
    example_ax.grid(True, alpha=1.0)
    example_ax.set_xlabel("")
    example_ax.set_ylabel("")
    example_ax.set_xlim(*x_limits)
    example_ax.set_ylim(*y_limits)

    plotter.plot_shape(
        shape,
        shape_gt,
        label="Ground Truth",
        color="black",
        ax=example_ax,
    )

    example_ax.plot(
        shape_gt[0],
        shape_gt[1],
        marker="o",
        linestyle="none",
        color="black",
        label="True Center",
    )"""

    pointcloud_path = (
        Path(pointcloud_dir)
        / f"pointcloud_{config.name}_{iteration:04d}.npz"
    )

    if not pointcloud_path.exists():
        raise FileNotFoundError(
            f"Point-cloud file does not exist: {pointcloud_path}"
        )

    pointcloud_data = np.load(pointcloud_path)

    observations = pointcloud_data["points"]
    parameter_samples = pointcloud_data["parameter_samples"]

    """example_ax.scatter(
        observations[:, 0],
        observations[:, 1],
        s=8,
        alpha=0.8,
        label="Sampled Points",
    )"""

    # ------------------------------------------------------------------
    # Covariance ellipses for this exact iteration
    # ------------------------------------------------------------------

    noise_params = result.noise_gt[iteration]

    covariances = config.noise_function.evaluate(
        parameter_samples,
        params=noise_params,
    )

    source_points = shape.parametrization(
        parameter_samples,
        shape_gt,
    )

    for covariance_index, (center, covariance) in enumerate(
        zip(source_points, covariances)
    ):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        eigenvalues = np.maximum(eigenvalues, 0.0)

        angle = np.degrees(
            np.arctan2(
                eigenvectors[1, 0],
                eigenvectors[0, 0],
            )
        )

        width, height = 2.0 * np.sqrt(eigenvalues)

        covariance_ellipse = patches.Ellipse(
            xy=center,
            width=width,
            height=height,
            angle=angle,
            edgecolor="green",
            facecolor="green",
            linewidth=0.1,
            alpha=0.1,
            label=(
                "Uncertainty"
                if covariance_index == 0
                else None
            ),
        )

        #example_ax.add_patch(covariance_ellipse)

    # Hide unused subplot positions.
    for unused_ax in axes[n_panels:]:
        unused_ax.set_visible(False)

    # ------------------------------------------------------------------
    # Shared legend
    # ------------------------------------------------------------------

    unique_entries = {}

    for ax in fig.axes:
        handles, legend_labels = ax.get_legend_handles_labels()

        for handle, legend_label in zip(handles, legend_labels):
            if legend_label and legend_label not in unique_entries:
                unique_entries[legend_label] = handle

    fig.legend(
        unique_entries.values(),
        unique_entries.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),  # vertical position below figure
        ncol=n_cols,  # <-- one row!
        fontsize=20,
        handletextpad=0.8
    )

    # Add bottom margin so the legend is not cut off
    fig.subplots_adjust(bottom=0.18, top=0.88, hspace=0.35, wspace=0.3)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])

    """fig.legend(
        unique_entries.values(),
        unique_entries.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(unique_entries),
        fontsize=20,
        handletextpad=0.8,
    )

    fig.subplots_adjust(
        bottom=0.18,
        top=0.88,
        hspace=0.35,
        wspace=0.3,
    )"""

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = (
        save_dir
        / f"{config.name}_iteration_{iteration:04d}.png"
    )

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved plot: {save_path}")

    return save_path
# analysis/plot_ellipsoid_qualitative.py

from pathlib import Path
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from core.sweep_result import SweepResult


# ---------------------------------------------------------------------------
# Publication style
# ---------------------------------------------------------------------------

PUB_RC = {
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.3,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "ps.fonttype": 42,
    "pdf.fonttype": 42,
}

PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#999999",
    "#F0E442",
]

GT_COLOR = "black"
POINT_COLOR = "dimgray"
POINT_EDGE_COLOR = "#888888"


# ---------------------------------------------------------------------------
# Parameter conversion
# ---------------------------------------------------------------------------

def ellipsoid_params_to_geometry(
    params: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Convert the nine-parameter Ellipsoid_R representation:

        [x_c, y_c, z_c, a, r1, r2, alpha, beta, gamma]

    into center, semiaxes, and Euler angles.

    The convention used here is:

        r1 = b / a
        r2 = c / a
    """

    params = np.asarray(
        params,
        dtype=float,
    )

    if params.shape != (9,):
        raise ValueError(
            "Expected ellipsoid parameters with shape (9,), "
            f"but received {params.shape}."
        )

    if not np.all(np.isfinite(params)):
        raise ValueError(
            "Ellipsoid parameters contain NaN or infinite values."
        )

    (
        x_center,
        y_center,
        z_center,
        a,
        r1,
        r2,
        alpha,
        beta,
        gamma,
    ) = params

    b = a * r1
    c = a * r2

    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(
            "The fitted ellipsoid semiaxes must be positive."
        )

    return {
        "center": np.array([
            x_center,
            y_center,
            z_center,
        ]),
        "axes": np.array([
            a,
            b,
            c,
        ]),
        "angles": np.array([
            alpha,
            beta,
            gamma,
        ]),
    }


# ---------------------------------------------------------------------------
# Surface generation
# ---------------------------------------------------------------------------

def ellipsoid_surface(
    params: dict[str, np.ndarray],
    n: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a rotated ellipsoid surface.

    Euler-angle convention:

        R = Rz(alpha) @ Ry(beta) @ Rx(gamma)
    """

    center = np.asarray(
        params["center"],
        dtype=float,
    )

    axes = np.asarray(
        params["axes"],
        dtype=float,
    )

    alpha, beta, gamma = np.asarray(
        params["angles"],
        dtype=float,
    )

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)

    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    rotation_z = np.array([
        [cos_alpha, -sin_alpha, 0.0],
        [sin_alpha, cos_alpha, 0.0],
        [0.0, 0.0, 1.0],
    ])

    rotation_y = np.array([
        [cos_beta, 0.0, sin_beta],
        [0.0, 1.0, 0.0],
        [-sin_beta, 0.0, cos_beta],
    ])

    rotation_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_gamma, -sin_gamma],
        [0.0, sin_gamma, cos_gamma],
    ])

    rotation = (
        rotation_z
        @ rotation_y
        @ rotation_x
    )

    azimuth = np.linspace(
        0.0,
        2.0 * np.pi,
        n,
    )

    polar_angle = np.linspace(
        0.0,
        np.pi,
        n,
    )

    azimuth_grid, polar_grid = np.meshgrid(
        azimuth,
        polar_angle,
    )

    x_local = (
        axes[0]
        * np.cos(azimuth_grid)
        * np.sin(polar_grid)
    )

    y_local = (
        axes[1]
        * np.sin(azimuth_grid)
        * np.sin(polar_grid)
    )

    z_local = (
        axes[2]
        * np.cos(polar_grid)
    )

    local_points = np.stack(
        [
            x_local.ravel(),
            y_local.ravel(),
            z_local.ravel(),
        ],
        axis=1,
    )

    world_points = (
        local_points @ rotation.T
        + center
    )

    surface_shape = x_local.shape

    return (
        world_points[:, 0].reshape(surface_shape),
        world_points[:, 1].reshape(surface_shape),
        world_points[:, 2].reshape(surface_shape),
    )

def ellipsoid_cap_surface(
    params: dict[str, np.ndarray],
    coverage: float,
    coverage_pole="south",
    n: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the covered polar-cap region of an ellipsoid.

    Parameters
    ----------
    params:
        Dictionary containing:

            center
            axes
            angles

    coverage:
        Fraction of the full sphere covered by the cap.

        coverage=0.5 gives a hemisphere.
        coverage=1.0 gives the complete ellipsoid.

    coverage_pole:
        Cap direction in canonical ellipsoid coordinates. Supported
        values are "south", "north", or a three-dimensional vector.

    n:
        Surface resolution.

    Returns
    -------
    X, Y, Z:
        Ellipsoid surface arrays with points outside the cap replaced
        by NaN.
    """

    coverage = float(coverage)

    if not 0.0 < coverage <= 1.0:
        raise ValueError(
            "coverage must satisfy 0 < coverage <= 1."
        )

    center = np.asarray(
        params["center"],
        dtype=float,
    )

    axes = np.asarray(
        params["axes"],
        dtype=float,
    )

    alpha, beta, gamma = np.asarray(
        params["angles"],
        dtype=float,
    )

    if isinstance(coverage_pole, str):
        pole_name = coverage_pole.lower()

        if pole_name == "south":
            pole = np.array([
                0.0,
                0.0,
                -1.0,
            ])
        elif pole_name == "north":
            pole = np.array([
                0.0,
                0.0,
                1.0,
            ])
        else:
            raise ValueError(
                "For plotting, coverage_pole must be 'south', "
                "'north', or a three-dimensional vector."
            )
    else:
        pole = np.asarray(
            coverage_pole,
            dtype=float,
        )

        if pole.shape != (3,):
            raise ValueError(
                "A custom coverage pole must have shape (3,)."
            )

        pole_norm = np.linalg.norm(pole)

        if pole_norm < 1e-12:
            raise ValueError(
                "A custom coverage pole must be nonzero."
            )

        pole = pole / pole_norm

    # ZYX rotation convention.
    rotation_z = np.array([
        [
            np.cos(alpha),
            -np.sin(alpha),
            0.0,
        ],
        [
            np.sin(alpha),
            np.cos(alpha),
            0.0,
        ],
        [
            0.0,
            0.0,
            1.0,
        ],
    ])

    rotation_y = np.array([
        [
            np.cos(beta),
            0.0,
            np.sin(beta),
        ],
        [
            0.0,
            1.0,
            0.0,
        ],
        [
            -np.sin(beta),
            0.0,
            np.cos(beta),
        ],
    ])

    rotation_x = np.array([
        [
            1.0,
            0.0,
            0.0,
        ],
        [
            0.0,
            np.cos(gamma),
            -np.sin(gamma),
        ],
        [
            0.0,
            np.sin(gamma),
            np.cos(gamma),
        ],
    ])

    rotation = (
        rotation_z
        @ rotation_y
        @ rotation_x
    )

    azimuth = np.linspace(
        0.0,
        2.0 * np.pi,
        n,
    )

    polar_angle = np.linspace(
        0.0,
        np.pi,
        n,
    )

    azimuth_grid, polar_grid = np.meshgrid(
        azimuth,
        polar_angle,
    )

    # Unit-sphere coordinates in the canonical ellipsoid frame.
    unit_x = (
        np.cos(azimuth_grid)
        * np.sin(polar_grid)
    )

    unit_y = (
        np.sin(azimuth_grid)
        * np.sin(polar_grid)
    )

    unit_z = np.cos(
        polar_grid
    )

    unit_points = np.stack(
        [
            unit_x,
            unit_y,
            unit_z,
        ],
        axis=-1,
    )

    # For a cap with surface fraction coverage:
    #
    #     dot(unit_point, pole) >= 1 - 2 * coverage
    cap_threshold = (
        1.0 - 2.0 * coverage
    )

    pole_projection = (
        unit_points[..., 0] * pole[0]
        + unit_points[..., 1] * pole[1]
        + unit_points[..., 2] * pole[2]
    )

    cap_mask = (
        pole_projection >= cap_threshold
    )

    # Map the canonical unit sphere onto the ellipsoid.
    local_points = unit_points * axes

    world_points = (
        local_points.reshape(-1, 3)
        @ rotation.T
        + center
    ).reshape(
        unit_points.shape
    )

    x_cap = np.where(
        cap_mask,
        world_points[..., 0],
        np.nan,
    )

    y_cap = np.where(
        cap_mask,
        world_points[..., 1],
        np.nan,
    )

    z_cap = np.where(
        cap_mask,
        world_points[..., 2],
        np.nan,
    )

    return x_cap, y_cap, z_cap

# ---------------------------------------------------------------------------
# Measurement loading
# ---------------------------------------------------------------------------

def load_saved_measurement(
    result_directory: str | Path,
    experiment_name: str,
    iteration: int,
) -> np.ndarray:
    """
    Load the exact point cloud saved by run_experiment().
    """

    result_directory = Path(
        result_directory
    )

    pointcloud_path = (
        result_directory
        / "pointclouds"
        / (
            f"pointcloud_{experiment_name}_"
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
    ) as measurement:
        if "points" not in measurement.files:
            raise ValueError(
                f"The measurement does not contain 'points': "
                f"{pointcloud_path}"
            )

        points = measurement["points"].copy()

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "Expected point cloud with shape (N, 3), "
            f"but received {points.shape}."
        )

    return points


# ---------------------------------------------------------------------------
# Surface clipping
# ---------------------------------------------------------------------------

def clip_surface_to_limits(
    x_surface: np.ndarray,
    y_surface: np.ndarray,
    z_surface: np.ndarray,
    limits,
    factor: float = 1.0,
):
    """
    Replace surface coordinates lying outside the plot region with NaN.
    """

    mask = np.zeros(
        x_surface.shape,
        dtype=bool,
    )

    for coordinate, (lower, upper) in zip(
        (x_surface, y_surface, z_surface),
        limits,
    ):
        center = 0.5 * (
            lower + upper
        )

        half_extent = 0.5 * (
            upper - lower
        )

        mask |= (
            np.abs(coordinate - center)
            > factor * half_extent
        )

    clipped_x = x_surface.copy()
    clipped_y = y_surface.copy()
    clipped_z = z_surface.copy()

    clipped_x[mask] = np.nan
    clipped_y[mask] = np.nan
    clipped_z[mask] = np.nan

    return (
        clipped_x,
        clipped_y,
        clipped_z,
        bool(mask.all()),
    )


# ---------------------------------------------------------------------------
# Qualitative grid
# ---------------------------------------------------------------------------

def plot_ellipsoid_qualitative_grid(
        repository_root: str | Path,
        trial: int,
        coverages: Sequence[float] = (
                0.50,
                0.40,
                0.30,
                0.20,
        ),
        methods: Sequence[str] | None = None,
        output_path: str | Path | None = None,
        coverage_pole="south",
        panel_size: float = 3.14,
        elev: float = 18.0,
        azim: float = -60.0,
        wireframe_n: int = 18,
        ground_truth_surface_n: int = 60,
        ground_truth_alpha: float = 0.1,
        covered_region_alpha: float = 0.50,
        point_size: float = 10.0,
        zoom: float = 1.7,
        show_axes: bool = False,
        show_ground_truth_row: bool = True,
        show: bool = False,
) -> Path:
    """
    Plot one trial across several polar-cap coverages.

    Columns correspond to the requested coverage levels.
    Rows correspond to fitting methods. An optional first row shows
    only the ground truth, observed cap, and observed point cloud.
    """

    repository_root = Path(
        repository_root
    )

    coverages = [
        float(coverage)
        for coverage in coverages
    ]

    if not coverages:
        raise ValueError(
            "At least one coverage must be provided."
        )

    coverage_data = {}

    expected_labels = None

    # ------------------------------------------------------------------
    # Load results and measurements
    # ------------------------------------------------------------------

    for coverage in coverages:
        result_directory = (
            repository_root
            / "results"
            / "ellipsoid"
            / "coverage_sweep"
            / f"coverage_{coverage:.2f}"
        )

        result = SweepResult.load(
            result_directory
        )

        if expected_labels is None:
            expected_labels = list(
                result.labels
            )
        elif result.labels != expected_labels:
            raise ValueError(
                "The fitter labels differ between coverage results. "
                f"Mismatch in {result_directory}."
            )

        if not 0 <= trial < result.fitting_params.shape[0]:
            raise IndexError(
                f"Trial {trial} is unavailable for coverage "
                f"{coverage:.2f}. The result has "
                f"{result.fitting_params.shape[0]} iterations."
            )

        points = load_saved_measurement(
            result_directory=result_directory,
            experiment_name=result.experiment_name,
            iteration=trial,
        )

        true_parameter_vector = (
            result.shape_gt[trial]
        )

        true_geometry = (
            ellipsoid_params_to_geometry(
                true_parameter_vector
            )
        )

        fitted_geometries = {}

        for method_index, method_label in enumerate(
            result.labels
        ):
            fitted_parameter_vector = (
                result.fitting_params[
                    trial,
                    method_index,
                    :,
                ]
            )

            if not np.all(
                np.isfinite(
                    fitted_parameter_vector
                )
            ):
                continue

            try:
                fitted_geometries[method_label] = (
                    ellipsoid_params_to_geometry(
                        fitted_parameter_vector
                    )
                )
            except ValueError as error:
                print(
                    f"Skipping {method_label} for "
                    f"coverage={coverage:.2f}, "
                    f"trial={trial}: {error}"
                )

        coverage_data[coverage] = {
            "points": points,
            "true_geometry": true_geometry,
            "fitted_geometries": fitted_geometries,
        }

    if methods is None:
        methods = expected_labels
    else:
        methods = list(methods)

    if not methods:
        raise ValueError(
            "No fitting methods were selected."
        )

    unknown_methods = (
        set(methods)
        - set(expected_labels)
    )

    if unknown_methods:
        raise ValueError(
            "Unknown fitting methods: "
            + ", ".join(
                sorted(unknown_methods)
            )
        )

    # ------------------------------------------------------------------
    # Calculate uniform limits from observations and ground truth only
    # ------------------------------------------------------------------

    extent_minimum = np.full(
        3,
        np.inf,
    )

    extent_maximum = np.full(
        3,
        -np.inf,
    )

    for coverage in coverages:
        data = coverage_data[coverage]

        points = data["points"]

        extent_minimum = np.minimum(
            extent_minimum,
            np.nanmin(points, axis=0),
        )

        extent_maximum = np.maximum(
            extent_maximum,
            np.nanmax(points, axis=0),
        )

        ground_truth_surface = ellipsoid_surface(
            data["true_geometry"],
            n=ground_truth_surface_n,
        )

        for dimension, coordinates in enumerate(
            ground_truth_surface
        ):
            extent_minimum[dimension] = min(
                extent_minimum[dimension],
                np.nanmin(coordinates),
            )

            extent_maximum[dimension] = max(
                extent_maximum[dimension],
                np.nanmax(coordinates),
            )

    extent_center = 0.5 * (
        extent_maximum + extent_minimum
    )

    maximum_span = np.max(
        extent_maximum - extent_minimum
    )

    half_extent = (
        0.58 * maximum_span
        if maximum_span > 0
        else 1.0
    )

    limits = [
        (
            center_coordinate - half_extent,
            center_coordinate + half_extent,
        )
        for center_coordinate in extent_center
    ]

    # ------------------------------------------------------------------
    # Figure setup
    # ------------------------------------------------------------------

    number_of_columns = len(
        coverages
    )

    number_of_rows = (
        len(methods)
        + int(show_ground_truth_row)
    )

    axes = np.full(
        (
            number_of_rows,
            number_of_columns,
        ),
        None,
        dtype=object,
    )

    def setup_axis(
        axis,
        points: np.ndarray,
        true_geometry: dict,
        fitted_surface=None,
        fitted_color=None,
    ) -> None:
        ground_truth_surface = ellipsoid_surface(
            true_geometry,
            n=ground_truth_surface_n,
        )

        axis.plot_surface(
            *ground_truth_surface,
            color=GT_COLOR,
            alpha=ground_truth_alpha,
            edgecolor="none",
            linewidth=0,
            antialiased=True,
            shade=False,
            zorder=1,
        )

        axis.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=point_size,
            facecolor=POINT_COLOR,
            edgecolor=POINT_EDGE_COLOR,
            linewidth=0.2,
            depthshade=True,
            rasterized=True,
            zorder=2,
        )

        if fitted_surface is not None:
            (
                clipped_x,
                clipped_y,
                clipped_z,
                fully_outside,
            ) = clip_surface_to_limits(
                *fitted_surface,
                limits=limits,
                factor=1.0,
            )

            if not fully_outside:
                axis.plot_wireframe(
                    clipped_x,
                    clipped_y,
                    clipped_z,
                    color=fitted_color,
                    linewidth=0.6,
                    rcount=wireframe_n,
                    ccount=wireframe_n,
                    zorder=3,
                )

        axis.view_init(
            elev=elev,
            azim=azim,
        )

        axis.set_autoscale_on(
            False
        )

        axis.set_xlim3d(
            limits[0]
        )
        axis.set_ylim3d(
            limits[1]
        )
        axis.set_zlim3d(
            limits[2]
        )

        try:
            axis.set_box_aspect(
                (1, 1, 1),
                zoom=zoom,
            )
        except TypeError:
            axis.set_box_aspect(
                (1, 1, 1)
            )

        if not show_axes:
            axis.set_axis_off()

        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_zticklabels([])

    # ------------------------------------------------------------------
    # Draw grid
    # ------------------------------------------------------------------

    with plt.rc_context(PUB_RC):
        figure = plt.figure(
            figsize=(
                panel_size * number_of_columns,
                panel_size * number_of_rows,
            )
        )

        row_offset = int(
            show_ground_truth_row
        )

        if show_ground_truth_row:
            for column_index, coverage in enumerate(
                    coverages
            ):
                axis = figure.add_subplot(
                    number_of_rows,
                    number_of_columns,
                    column_index + 1,
                    projection="3d",
                )

                axes[
                    0,
                    column_index,
                ] = axis

                data = coverage_data[
                    coverage
                ]

                setup_axis(
                    axis=axis,
                    points=data["points"],
                    true_geometry=data["true_geometry"],
                )

                # ----------------------------------------------------------
                # Gray polar-cap region
                # ----------------------------------------------------------

                x_cap, y_cap, z_cap = ellipsoid_cap_surface(
                    params=data["true_geometry"],
                    coverage=coverage,
                    coverage_pole=coverage_pole,
                    n=ground_truth_surface_n,
                )

                axis.plot_surface(
                    x_cap,
                    y_cap,
                    z_cap,
                    color="#8F8F8F",
                    alpha=covered_region_alpha,
                    edgecolor="none",
                    linewidth=0,
                    antialiased=True,
                    shade=False,
                    zorder=2,
                )

                axis.set_title(
                    f"{100.0 * coverage:g}%",
                    fontsize=20,
                    pad=6,
                )

        for method_index, method in enumerate(
            methods
        ):
            grid_row = (
                method_index
                + row_offset
            )

            method_color = PALETTE[
                method_index % len(PALETTE)
            ]

            for column_index, coverage in enumerate(
                coverages
            ):
                subplot_number = (
                    grid_row
                    * number_of_columns
                    + column_index
                    + 1
                )

                axis = figure.add_subplot(
                    number_of_rows,
                    number_of_columns,
                    subplot_number,
                    projection="3d",
                )

                axes[
                    grid_row,
                    column_index,
                ] = axis

                data = coverage_data[
                    coverage
                ]

                fitted_geometry = (
                    data["fitted_geometries"]
                    .get(method)
                )

                fitted_surface = None

                if fitted_geometry is not None:
                    fitted_surface = (
                        ellipsoid_surface(
                            fitted_geometry,
                            n=wireframe_n,
                        )
                    )

                setup_axis(
                    axis=axis,
                    points=data["points"],
                    true_geometry=data["true_geometry"],
                    fitted_surface=fitted_surface,
                    fitted_color=method_color,
                )

                if fitted_geometry is None:
                    axis.text2D(
                        0.5,
                        0.5,
                        "Fit unavailable",
                        transform=axis.transAxes,
                        ha="center",
                        va="center",
                        fontsize=10,
                        color="#777777",
                    )

                if not show_ground_truth_row:
                    axis.set_title(
                        f"{100.0 * coverage:g}%",
                        fontsize=20,
                        pad=6,
                    )

        # ------------------------------------------------------------------
        # Row labels
        # ------------------------------------------------------------------

        row_labels = []

        if show_ground_truth_row:
            row_labels.append(
                "Ground Truth\n+ Observations"
            )

        row_labels.extend(
            methods
        )

        for row_index, row_label in enumerate(
            row_labels
        ):
            row_axis = axes[
                row_index,
                0,
            ]

            row_axis.text2D(
                -0.08,
                0.5,
                row_label,
                transform=row_axis.transAxes,
                ha="center",
                va="center",
                rotation=90,
                fontsize=18,
            )

        figure.text(
            0.01,
            0.96,
            "Coverage level:",
            ha="left",
            va="top",
            fontsize=18,
        )

        figure.subplots_adjust(
            wspace=0.01,
            hspace=0.03,
            left=0.12,
            right=0.99,
            top=0.94,
            bottom=0.08,
        )

        if output_path is None:
            output_path = (
                repository_root
                / "results"
                / "ellipsoid"
                / "coverage_sweep"
                / "plots"
                / (
                    f"ellipsoid_qualitative_"
                    f"trial_{trial:04d}.pdf"
                )
            )
        else:
            output_path = Path(
                output_path
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            bbox_inches="tight",
            dpi=600,
        )

        # Also save a PNG preview.
        png_path = output_path.with_suffix(
            ".png"
        )

        figure.savefig(
            png_path,
            bbox_inches="tight",
            dpi=600,
        )

        if show:
            # Agg does not normally open an interactive window, but this
            # still supports environments with an interactive backend.
            plt.show()

        plt.close(
            figure
        )

    print(
        f"Saved plot: {output_path}"
    )
    print(
        f"Saved plot: {png_path}"
    )

    return output_path


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repository_root = (
        Path(__file__).resolve().parents[1]
    )

    for i in range(100):

        plot_ellipsoid_qualitative_grid(
            repository_root=repository_root,
            trial=i,
            coverages=[
                0.2
            ],
            methods=[
                "Calafiore",
                r"$\mathbf{BLISS}$",
            ],
            output_path=(
                repository_root
                / "results"
                / "ellipsoid"
                / "coverage_sweep"
                / "plots"
                / f"qualitative_grid_{i}.pdf"
            ),
            show_axes=False,
            panel_size=3.14,
            zoom=1.7,
            point_size=10,
            ground_truth_alpha=0.1,
            show_ground_truth_row=True,
            show=False,
        )
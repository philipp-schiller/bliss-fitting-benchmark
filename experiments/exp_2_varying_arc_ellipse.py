# experiments/ellipse/arc_length_sweep.py

from pathlib import Path

import numpy as np

from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.ellipse import Ellipse

from core.experiment_config import ExperimentConfig
from core.run_experiment import run_experiment
from core.sweep_result import SweepResult

from methods.ellipse_fitters.bayes_fitter import BayesFitter
from methods.ellipse_fitters.hyper_renormalization import HyperRenormalizationFitter
from methods.ellipse_fitters.zhao_bayes import BayesianEllipseFitterZhao
from methods.ellipse_fitters.geometric_ls import GeometricEllipseFitter


from analysis.plot_ellipse_sweep import (
    collect_ellipse_sweep,
    plot_ellipse_sweep, trimmed_nanmean, nanmedian,
)

from analysis.plot_ellipse_sweep_boxplot import plot_ellipse_boxplot_summary, plot_ellipse_bias_sweep

shape = Ellipse()

def sample_ellipse_shape(template) -> np.ndarray:
    """
    Generate random ellipse parameters:
    (cx, cy, a, b, phi).
    """

    return np.array([
        np.random.uniform(-1.0, 1.0),
        np.random.uniform(-1.0, 1.0),
        np.random.uniform(2.0, 2.5),
        np.random.uniform(1.5, 2.0),
        np.random.uniform(-np.pi / 4, np.pi / 4),
    ])


def sample_source_params(template) -> np.ndarray:
    """
    Resample entries represented by None.

    For this experiment:
        center is random;
        width w is fixed.
    """

    center, width = template

    if center is None:
        center = np.random.uniform(0.25, 0.5)

    if width is None:
        raise ValueError(
            "The arc width must be fixed for the arc-length sweep."
        )

    return np.array([center, width])

def make_arc_length_experiment(
    w: float,
    *,
    w_prior: float | None = None,
    iterations: int = 100,
) -> ExperimentConfig:
    """
    Construct an experiment for one fixed arc width.

    Parameters
    ----------
    w:
        Ground-truth arc width.

    w_prior:
        Prior mean used by BLISS. If omitted, the prior is centered
        at the ground-truth value.

    iterations:
        Number of Monte Carlo repetitions for this value of w.
    """

    if w_prior is None:
        w_prior = w

    #experiment_name = f"arc_length_w_{w:.2f}"
    experiment_name = f"arc_length_bounds_w_{w:.2f}"

    if not np.isclose(w_prior, w):
        experiment_name += f"_prior_{w_prior:.2f}"

    # ---------------------------------------------------------------
    # Ground-truth model
    # ---------------------------------------------------------------

    ground_truth_source = UniformSourceDensity(
        n_dim=1,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        periodic=shape.periodic,
    )

    ground_truth_noise = IsotropicNoise(
        dim_param=shape.parameter_dim,
        dim_ambient=shape.ambient_dim,
    )

    # ---------------------------------------------------------------
    # Priors
    # ---------------------------------------------------------------

    source_prior_weak = GaussianPrior(
        mean=np.array([0.1, w_prior]),
        cov=np.diag([1, 0.00065]),
    )

    source_prior_strong = GaussianPrior(
        mean=np.array([0.1, w_prior]),
        cov=np.diag([1, 0.00016]),
    )

    source_bounds_weak = [
        (0, 1),
        (0.1, 0.5)
    ]

    source_bounds_strong = [
        (0, 1),
        (0.1,0.5)
        # (max(0.0, w - 0.05), w + 0.05),
    ]

    noise_bounds = [
        (0.001, 0.04),
    ]

    source_init = np.array([0.375, w_prior])
    noise_init = np.array([0.0225])

    shape_bounds = [
        (-10, 10),  # x_c
        (-10, 10),  # y_c
        (0.3, np.inf),  # a
        (0.3, np.inf),  # b
        (-np.inf, np.inf),  # theta
    ]

    # ---------------------------------------------------------------
    # BLISS with weak priors
    # ---------------------------------------------------------------

    bliss_weak = BayesFitter(
        name="BLISS_weak",
        source_density=UniformSourceDensity(
            n_dim=1,
            domain_low=shape.domain_low,
            domain_high=shape.domain_high,
            periodic=shape.periodic,
            prior=source_prior_weak,
        ),
        noise_function=IsotropicNoise(
            dim_param=shape.parameter_dim,
            dim_ambient=shape.ambient_dim,
        ),
        source_init=source_init.copy(),
        noise_init=noise_init.copy(),
        source_bounds=source_bounds_weak,
        noise_bounds=noise_bounds,
        shape_bounds=shape_bounds
    )

    # ---------------------------------------------------------------
    # BLISS with strong priors
    # ---------------------------------------------------------------

    bliss_strong = BayesFitter(
        name="BLISS_strong",
        source_density=UniformSourceDensity(
            n_dim=1,
            domain_low=shape.domain_low,
            domain_high=shape.domain_high,
            periodic=shape.periodic,
            prior=source_prior_strong,
        ),
        noise_function=IsotropicNoise(
            dim_param=shape.parameter_dim,
            dim_ambient=shape.ambient_dim,
        ),
        source_init=source_init.copy(),
        noise_init=noise_init.copy(),
        source_bounds=source_bounds_strong,
        noise_bounds=noise_bounds,
        shape_bounds=shape_bounds
    )

    # ---------------------------------------------------------------
    # BLISS with exact source and noise parameters
    # ---------------------------------------------------------------

    exact_source = UniformSourceDensity(
        n_dim=1,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        periodic=shape.periodic,
    )

    exact_noise = IsotropicNoise(
        dim_param=shape.parameter_dim,
        dim_ambient=shape.ambient_dim,
    )

    # This fitter requires special handling if the random source center
    # changes in every iteration. See the note below.
    bliss_exact = BayesFitter(
        name="BLISS_exact",
        source_density=exact_source,
        noise_function=exact_noise,
        shape_bounds=shape_bounds,
        source_init=np.array([0.375]),
        source_bounds=[(0, 1)]  # the arc length
    )

    return ExperimentConfig(
        name=experiment_name,
        iterations=iterations,
        num_points=100,

        shape=shape,
        source_density=ground_truth_source,
        noise_function=ground_truth_noise,

        # Entire ellipse is resampled each iteration.
        shape_params=None,
        shape_sampler=sample_ellipse_shape,

        # Center is random, but w is fixed for this experiment.
        source_params=np.array(
            [None, w],
            dtype=object,
        ),
        source_sampler=sample_source_params,

        # Noise is fixed.
        noise_params=np.array([0.0225]),

        fitters=[
            GeometricEllipseFitter(),
            HyperRenormalizationFitter(),
            BayesianEllipseFitterZhao(),
            bliss_weak,
            bliss_strong,
            bliss_exact,
        ],

        labels=[
            "GLS",
            "HR",
            "Zhao",
            r"$\mathbf{BLISS\text{-}weak}$",
            r"$\mathbf{BLISS\text{-}strong}$",
            r"$\mathbf{BLISS\text{-}exact}$",
        ],
    )


RUN_EXPERIMENT = False


if __name__ == "__main__":

    np.random.seed(42)

    repository_root = Path(__file__).resolve().parents[1]

    arc_half_widths = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    ]

    result_directories = []

    for w in arc_half_widths:
        config = make_arc_length_experiment(
            w=float(w),
            iterations=100,
        )

        result_directory = (
            repository_root
            / "results"
            / "ellipse"
            / "arc_length_sweep"
            / f"w_{w:.2f}"
        )

        pointcloud_directory = (
            result_directory
            / "pointclouds"
        )

        result_directories.append(result_directory)

        if RUN_EXPERIMENT:
            print(
                f"\nRunning {config.name}: "
                f"w={w:.2f}, "
                f"iterations={config.iterations}"
            )

            result = run_experiment(
                config,
                save_pointclouds=True,
                pointcloud_directory=pointcloud_directory,
            )

            result.save(result_directory)

        else:
            print(
                f"Using saved result for w={w:.2f}: "
                f"{result_directory}"
            )
            SweepResult.load(result_directory)

    # Arc coverage is twice the half-width, expressed as a percentage.
    arc_coverages = [
        200.0 * w
        for w in arc_half_widths
    ]

    sweep_data = collect_ellipse_sweep(
        sweep_values=arc_coverages,
        result_directories=result_directories,
        reducer= nanmedian,
    )

    plot_directory = (
        repository_root
        / "results"
        / "ellipse"
        / "arc_length_sweep"
        / "plots"
    )

    plot_ellipse_sweep(
        sweep_data=sweep_data,
        x_label="Arc Coverage (%)",
        save_path=plot_directory / "arc_coverage_sweep_median.png",
        invert_x_axis=True,
        log_error_axes=True,
        show=True,
    )

    plot_ellipse_boxplot_summary(
        sweep_data,
        coverage_indices=[4, 2, 0],  # 60%,40%,20%
        x_label_template="Arc coverage {value:.0f}%",
        value_transform=lambda w: w,
        save_path= plot_directory / "coverage_boxplots_median.png",
        show=True
    )

    plot_ellipse_bias_sweep(
        sweep_data,
        x_label="Arc coverage (%)",
        invert_x_axis=True,
        save_path="coverage_bias_sweep_median.png",
        show=True,
    )
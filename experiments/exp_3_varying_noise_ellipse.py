from pathlib import Path

import numpy as np

from bliss.inference.noise_functions import IsotropicNoise
from bliss.inference.priors import GaussianPrior
from bliss.inference.samplers.map_estimation import MAPOptimizer
from bliss.inference.source_densitites import UniformSourceDensity
from bliss.models.shapes.ellipse import Ellipse

from core.experiment_config import ExperimentConfig
from core.run_experiment import run_experiment
from core.sweep_result import SweepResult

from methods.ellipse_fitters.bayes_fitter import BayesFitter
from methods.ellipse_fitters.hyper_renormalization import HyperRenormalizationFitter
from methods.ellipse_fitters.zhao_bayes import BayesianEllipseFitterZhao
from methods.ellipse_fitters.geometric_ls import GeometricEllipseFitter

from analysis.plot_ellipse_sweep import (
    collect_ellipse_sweep,
    plot_ellipse_sweep, nanmedian, trimmed_nanmean
)
from analysis.plot_ellipse_sweep_boxplot import plot_ellipse_boxplot_summary, plot_ellipse_bias_sweep

shape = Ellipse()

# Arc half-width 0.3 corresponds to 60% coverage in your convention.
ARC_HALF_WIDTH = 0.3


# ---------------------------------------------------------------------------
# Ellipse-specific parameter samplers
# ---------------------------------------------------------------------------

def sample_ellipse_shape(template) -> np.ndarray:
    """
    Sample random ellipse parameters:

        [cx, cy, a, b, phi]
    """

    cx = np.random.uniform(-1.0, 1.0)
    cy = np.random.uniform(-1.0, 1.0)
    a = np.random.uniform(2.0, 2.5)
    b = np.random.uniform(1.5, 2.0)
    phi = np.random.uniform(
        -np.pi / 4,
        np.pi / 4,
    )

    return np.array(
        [cx, cy, a, b, phi],
        dtype=float,
    )


def sample_source_params(template) -> np.ndarray:
    """
    Resample only source parameters represented by None.

    For this experiment, the source center is random and the arc
    half-width is fixed.
    """

    center, width = template

    if center is None:
        center = np.random.uniform(0.25, 0.5)

    if width is None:
        raise ValueError(
            "The arc half-width must be fixed for the noise sweep."
        )

    return np.array(
        [center, width],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Experiment factory
# ---------------------------------------------------------------------------

def make_noise_experiment(
    sigma_sq: float = 0.05,
    sigma_sq_prior: float | None = None,
    iterations: int = 100,
) -> ExperimentConfig:
    """
    Construct one experiment for a fixed isotropic noise variance.

    Parameters
    ----------
    sigma_sq:
        Ground-truth isotropic noise variance.

    sigma_sq_prior:
        Prior mean used by BLISS. If None, the prior is centered on
        the ground-truth value.

    iterations:
        Number of Monte Carlo repetitions at this noise level.
    """

    sigma_sq = float(sigma_sq)

    if sigma_sq <= 0:
        raise ValueError(
            "sigma_sq must be positive."
        )

    if sigma_sq_prior is None:
        sigma_sq_prior = sigma_sq

    sigma_sq_prior = float(sigma_sq_prior)

    if sigma_sq_prior <= 0:
        raise ValueError(
            "sigma_sq_prior must be positive."
        )

    if np.isclose(sigma_sq, sigma_sq_prior):
        experiment_name = (
            f"ellipse_noise_variance_{sigma_sq:.5f}"
        )
    else:
        experiment_name = (
            f"ellipse_noise_variance_{sigma_sq:.5f}"
            f"_prior_{sigma_sq_prior:.5f}"
        )

    # ------------------------------------------------------------------
    # Ground-truth source and noise models
    # ------------------------------------------------------------------

    ground_truth_source_density = UniformSourceDensity(
        n_dim=1,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        periodic=shape.periodic,
    )

    ground_truth_noise_function = IsotropicNoise(
        dim_param=shape.parameter_dim,
        dim_ambient=shape.ambient_dim,
    )

    # ------------------------------------------------------------------
    # Source bounds
    # ------------------------------------------------------------------

    source_bounds_weak = [
        (0.0, 1.0),
        (0.1, 0.5)
    ]

    source_bounds_strong = [
        (0.0, 1.0),
        (0.1, 0.5)
    ]

    # ------------------------------------------------------------------
    # Noise priors and bounds
    # ------------------------------------------------------------------

    strong_noise_prior_variance = 6.0e-6

    noise_prior_strong = GaussianPrior(
        mean=np.array([sigma_sq_prior]),
        cov=np.array([
            [strong_noise_prior_variance]
        ]),
    )

    noise_bounds = [(0.0001, np.inf)]

    source_init = np.array([
        0.375,
        ARC_HALF_WIDTH,
    ])

    noise_init = np.array([
        sigma_sq,
    ])

    # ------------------------------------------------------------------
    # BLISS with weak noise prior
    # ------------------------------------------------------------------

    bliss_weak = BayesFitter(
        name="BLISS_weak",

        source_density=UniformSourceDensity(
            n_dim=1,
            domain_low=shape.domain_low,
            domain_high=shape.domain_high,
            periodic=shape.periodic,
        ),

        noise_function=IsotropicNoise(
            dim_param=shape.parameter_dim,
            dim_ambient=shape.ambient_dim,
        ),

        source_init=source_init.copy(),
        noise_init=noise_init.copy(),

        source_bounds=source_bounds_weak,
        noise_bounds=noise_bounds,
    )

    # ------------------------------------------------------------------
    # BLISS with strong noise prior
    # ------------------------------------------------------------------

    bliss_strong = BayesFitter(
        name="BLISS_strong",

        source_density=UniformSourceDensity(
            n_dim=1,
            domain_low=shape.domain_low,
            domain_high=shape.domain_high,
            periodic=shape.periodic,
        ),

        noise_function=IsotropicNoise(
            dim_param=shape.parameter_dim,
            dim_ambient=shape.ambient_dim,
            prior=noise_prior_strong,
        ),

        source_init=source_init.copy(),
        noise_init=noise_init.copy(),

        source_bounds=source_bounds_strong,
        noise_bounds=noise_bounds,
    )

    # ------------------------------------------------------------------
    # BLISS with exact source and noise parameters
    # ------------------------------------------------------------------

    bliss_exact = BayesFitter(
        name="BLISS_exact",

        source_density=UniformSourceDensity(
            n_dim=1,
            domain_low=shape.domain_low,
            domain_high=shape.domain_high,
            periodic=shape.periodic,
        ),

        noise_function=IsotropicNoise(
            dim_param=shape.parameter_dim,
            dim_ambient=shape.ambient_dim,
        ),
        source_init=np.array([0.375]),
        source_bounds=[(0, 1)]  # the arc length
    )

    # The runner updates BLISS_exact.fixed_params after each resampling.

    return ExperimentConfig(
        name=experiment_name,
        iterations=iterations,
        num_points=100,

        shape=shape,

        source_density=ground_truth_source_density,
        noise_function=ground_truth_noise_function,

        # Fully random ellipse.
        shape_params=None,
        shape_sampler=sample_ellipse_shape,

        # Random source center with fixed 40% arc coverage.
        source_params=np.array(
            [None, ARC_HALF_WIDTH],
            dtype=object,
        ),
        source_sampler=sample_source_params,

        # Fixed noise variance within this experiment.
        noise_params=np.array([
            sigma_sq,
        ]),

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


# ---------------------------------------------------------------------------
# Run or load the complete noise sweep
# ---------------------------------------------------------------------------

RUN_EXPERIMENT = False

if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]

    # These are variances, not standard deviations.
    noise_variances = [
        0.010,
        0.020,
        0.030,
        0.040,
        0.050,
        0.060,
        0.070,
        0.080,
        0.090,
        0.100
    ]

    result_directories = []

    for sigma_sq in noise_variances:
        config = make_noise_experiment(
            sigma_sq=sigma_sq,
            iterations=100,
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

        result_directories.append(
            result_directory
        )

        if RUN_EXPERIMENT:
            print(
                f"\nRunning {config.name}: "
                f"sigma squared={sigma_sq:.5f}, "
                f"iterations={config.iterations}"
            )

            result = run_experiment(
                config,
                save_pointclouds=True,
                pointcloud_directory=pointcloud_directory,
            )

            save_path = result.save(
                result_directory
            )

            print(f"Saved result: {save_path}")

        else:
            print(
                f"Using saved result for "
                f"sigma squared={sigma_sq:.5f}: "
                f"{result_directory}"
            )

            # Optional immediate validation.
            SweepResult.load(result_directory)

    # ------------------------------------------------------------------
    # Aggregate and plot the sweep
    # ------------------------------------------------------------------

    sweep_data = collect_ellipse_sweep(
        sweep_values=noise_variances,
        result_directories=result_directories,
        reducer=nanmedian
    )

    plot_directory = (
        repository_root
        / "results"
        / "ellipse"
        / "noise_sweep"
        / "plots"
    )

    plot_ellipse_sweep(
        sweep_data=sweep_data,
        x_label=r"Noise variance $\sigma^2$",
        save_path=(
            plot_directory
            / "noise_variance_sweep.png"
        ),
        invert_x_axis=False,
        log_error_axes=True,
        show=True,
    )

    plot_ellipse_boxplot_summary(
        sweep_data,
        coverage_indices=[0, 5, 9],
        x_label_template=r"$\sigma^2 = {value:g}$",
        save_path=(
            plot_directory
            / "noise_boxplots.png"
        ),
        show=True
    )

    plot_ellipse_bias_sweep(
        sweep_data,
        x_label=r"Noise variance $\sigma^2$",
        save_path=(
                plot_directory
                / "noise_variance_bias_sweep.png"
        ),
        show=True,
    )
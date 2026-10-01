# experiments/exp_4_laser_noise_ellipse.py

from pathlib import Path

import numpy as np

from bayes_shape_fitting.inference.noise_functions import (
    LaserDependentNoise,
    PartiallyFixedNoise,
)
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.source_densitites import (
    UniformSourceDensity,
)
from bayes_shape_fitting.models.shapes.ellipse import Ellipse

from analysis.plot_ellipse_sweep import (
    collect_ellipse_sweep,
    plot_ellipse_sweep,
)
from core.experiment_config import ExperimentConfig
from core.run_experiment import run_experiment
from core.sweep_result import SweepResult

from methods.ellipse_fitters.bayes_fitter import BayesFitter
from methods.ellipse_fitters.geometric_ls import (
    GeometricEllipseFitter,
)
from methods.ellipse_fitters.hyper_renormalization import (
    HyperRenormalizationFitter,
)
from methods.ellipse_fitters.zhao_bayes import (
    BayesianEllipseFitterZhao,
)

RUN_EXPERIMENT = False

# ---------------------------------------------------------------------------
# Fixed experiment settings
# ---------------------------------------------------------------------------

SHAPE_PARAMS = np.array([
    -1.0,
    0.0,
    3.0,
    2.0,
    np.pi / 4,
])

SOURCE_PARAMS = np.array([
    0.30,
    0.25,
])

SCANNER_POSITION = np.array([
    -6.0,
    2.0,
])

# The second laser-noise parameter remains fixed.
SIGMA_1 = 0.08

shape = Ellipse()


# ---------------------------------------------------------------------------
# Noise-model construction
# ---------------------------------------------------------------------------

def make_laser_noise(
        sigma_0_fixed: float | None = None,
        prior: GaussianPrior | None = None,
) -> PartiallyFixedNoise:
    """
    Construct the partially fixed laser-dependent noise model.

    The base-noise parameter ordering is assumed to be:

        [cx, cy, a, b, phi, sigma_0, sigma_1]

    The ellipse parameters and sigma_1 are fixed. sigma_0 is either
    supplied through noise_params or fixed explicitly.
    """

    base_noise = LaserDependentNoise(
        dim_param=1,
        dim_ambient=2,
        shape=shape,
        scanner=SCANNER_POSITION.copy(),
        prior = prior
    )

    fixed_params = {
        0: SHAPE_PARAMS[0],
        1: SHAPE_PARAMS[1],
        2: SHAPE_PARAMS[2],
        3: SHAPE_PARAMS[3],
        4: SHAPE_PARAMS[4],
        6: SIGMA_1,
    }

    if sigma_0_fixed is not None:
        fixed_params[5] = float(sigma_0_fixed)

    return PartiallyFixedNoise(
        base_noise,
        fixed_params=fixed_params,
    )


# ---------------------------------------------------------------------------
# Experiment factory
# ---------------------------------------------------------------------------

def make_laser_experiment(
        sigma_0: float = 0.01,
        sigma_0_prior: float | None = None,
        iterations: int = 100,
) -> ExperimentConfig:
    """
    Construct one laser-dependent-noise experiment.

    Parameters
    ----------
    sigma_0:
        Ground-truth value of the variable laser-noise parameter.

    sigma_0_prior:
        Prior mean used by BLISS. If omitted, the prior is centered
        on the ground-truth value.

    iterations:
        Number of Monte Carlo repetitions.
    """

    sigma_0 = float(sigma_0)

    if sigma_0 <= 0:
        raise ValueError("sigma_0 must be positive.")

    if sigma_0_prior is None:
        sigma_0_prior = sigma_0

    sigma_0_prior = float(sigma_0_prior)

    if sigma_0_prior <= 0:
        raise ValueError("sigma_0_prior must be positive.")

    if np.isclose(sigma_0, sigma_0_prior):
        experiment_name = (
            f"laser_noise_sigma_0_{sigma_0:.3f}"
        )
    else:
        experiment_name = (
            f"laser_noise_sigma_0_{sigma_0:.3f}"
            f"_prior_{sigma_0_prior:.3f}"
        )

    # ------------------------------------------------------------------
    # Ground-truth model
    # ------------------------------------------------------------------

    ground_truth_source_density = UniformSourceDensity(
        n_dim=1,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        periodic=shape.periodic,
    )

    # sigma_0 is supplied through config.noise_params.
    ground_truth_noise_function = make_laser_noise()

    # ------------------------------------------------------------------
    # BLISS priors
    # ------------------------------------------------------------------

    source_prior = GaussianPrior(
        mean=SOURCE_PARAMS.copy(),
        cov=np.diag([
            1.0,
            0.00065,
        ]),
    )

    noise_prior_variance = 0.0001

    noise_prior = GaussianPrior(
        mean=np.array([
            sigma_0_prior,
        ]),
        cov=np.array([
            [noise_prior_variance],
        ]),
    )

    source_bounds = [
        (0.0, 1.0),
        (0.05, 0.40),
    ]

    noise_radius = (
            1.96 * np.sqrt(noise_prior_variance)
    )

    noise_bounds = [
        (
            max(
                0.0,
                sigma_0_prior - noise_radius,
            ),
            sigma_0_prior + noise_radius,
        )
    ]

    source_init = SOURCE_PARAMS.copy()

    noise_init = np.array([
        sigma_0_prior,
    ])

    # ------------------------------------------------------------------
    # BLISS fitter
    # ------------------------------------------------------------------

    bliss_source_density = UniformSourceDensity(
        n_dim=1,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        periodic=shape.periodic,
        prior=source_prior,
    )

    bliss_noise_function = make_laser_noise(
        prior=noise_prior,
    )

    bliss = BayesFitter(
        name="BLISS",
        source_density=bliss_source_density,
        noise_function=bliss_noise_function,
        source_init=source_init,
        noise_init=noise_init,
        source_bounds=source_bounds,
        noise_bounds=noise_bounds,
    )

    # ------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------

    return ExperimentConfig(
        name=experiment_name,
        iterations=iterations,
        num_points=100,

        shape=shape,

        source_density=ground_truth_source_density,
        noise_function=ground_truth_noise_function,

        # All ground-truth values are fixed within this experiment.
        shape_params=SHAPE_PARAMS.copy(),
        source_params=SOURCE_PARAMS.copy(),
        noise_params=np.array([
            sigma_0,
        ]),

        fitters=[
            GeometricEllipseFitter(),
            HyperRenormalizationFitter(),
            BayesianEllipseFitterZhao(),
            bliss,
        ],

        labels=[
            "GLS",
            "HR",
            "Zhao",
            r"$\mathbf{BLISS}$",
        ],
    )


# ---------------------------------------------------------------------------
# Run or load the sigma_0 sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]

    sigma_0_values = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ]

    result_directories = []

    for sigma_0 in sigma_0_values:
        config = make_laser_experiment(
            sigma_0=float(sigma_0),
            iterations=100,
        )

        result_directory = (
            repository_root
            / "results"
            / "ellipse"
            / "laser_noise_sweep"
            / f"sigma_0_{sigma_0:.3f}"
        )

        pointcloud_directory = (
            result_directory
            / "pointclouds"
        )

        result_directories.append(result_directory)

        if RUN_EXPERIMENT:
            print(
                f"\nRunning {config.name}: "
                f"sigma_0={sigma_0:.3f}, "
                f"iterations={config.iterations}"
            )

            result = run_experiment(
                config,
                save_pointclouds=True,
                pointcloud_directory=pointcloud_directory,
            )

            save_path = result.save(result_directory)

            print(f"Saved result: {save_path}")

        else:
            print(
                f"Using saved result for "
                f"sigma_0={sigma_0:.3f}: "
                f"{result_directory}"
            )

            # Check that the saved result exists and can be loaded.
            SweepResult.load(result_directory)

    # ------------------------------------------------------------------
    # Collect metrics across the sigma_0 sweep
    # ------------------------------------------------------------------

    sweep_data = collect_ellipse_sweep(
        sweep_values=sigma_0_values,
        result_directories=result_directories,
    )

    # ------------------------------------------------------------------
    # Plot the complete sweep
    # ------------------------------------------------------------------

    plot_directory = (
        repository_root
        / "results"
        / "ellipse"
        / "laser_noise_sweep"
        / "plots"
    )

    plot_ellipse_sweep(
        sweep_data=sweep_data,
        x_label=r"Laser-noise parameter $\sigma_0$",
        save_path=(
            plot_directory
            / "laser_noise_sweep.png"
        ),
        invert_x_axis=False,
        log_error_axes=True,
        show=True,
    )
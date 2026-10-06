# experiments/exp_5_varying_coverage_ellipsoid.py

from pathlib import Path

import numpy as np

from bayes_shape_fitting.inference.noise_functions import (
    IsotropicNoise,
)
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.source_densitites import (
    PartiallyFixedDensity,
    UniformSourceDensity,
)
from bayes_shape_fitting.models.shapes.ellipsoid import (
    Ellipsoid_R,
)

from core.experiment_config import ExperimentConfig
from core.measurement import Measurement
from core.run_experiment import run_experiment
from core.sweep_result import SweepResult

from methods.ellipsoid_fitters.bayes_fitter import (
    BayesFitter,
)
from methods.ellipsoid_fitters.calafiore import (
    CalafioreEllipsoidFitter,
)


RUN_EXPERIMENT = True


# ---------------------------------------------------------------------------
# Fixed experiment settings
# ---------------------------------------------------------------------------

N_POINTS = 100

CENTER = np.array([
    0.3,
    -0.2,
    0.1,
])

# The old experiment used noise_std=0.05.
NOISE_STANDARD_DEVIATION = 0.05
NOISE_VARIANCE = NOISE_STANDARD_DEVIATION**2

shape = Ellipsoid_R()


# ---------------------------------------------------------------------------
# Random ellipsoid parameters
# ---------------------------------------------------------------------------

def sample_ellipsoid_shape_params(
    template,
) -> np.ndarray:
    """
    Sample one random ellipsoid.

    Parameter representation:

        [x_c, y_c, z_c, a, r1, r2, alpha, beta, gamma]

    with:

        r1 = b / a
        r2 = c / a
    """

    a = np.random.uniform(
        2.0,
        3.0,
    )

    r1 = np.random.uniform(
        0.5,
        1.0,
    )

    r2 = np.random.uniform(
        0.3,
        r1,
    )

    alpha = np.random.uniform(
        0.0,
        np.pi / 2,
    )

    beta = np.random.uniform(
        0.0,
        np.pi / 2,
    )

    gamma = np.random.uniform(
        0.0,
        np.pi / 2,
    )

    return np.array([
        CENTER[0],
        CENTER[1],
        CENTER[2],
        a,
        r1,
        r2,
        alpha,
        beta,
        gamma,
    ])


# ---------------------------------------------------------------------------
# Polar-cap sampling
# ---------------------------------------------------------------------------

def resolve_coverage_pole(
    coverage_pole,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a normalized polar-cap direction.
    """

    if isinstance(coverage_pole, str):
        key = coverage_pole.lower()

        if key == "south":
            pole = np.array([
                0.0,
                0.0,
                -1.0,
            ])

        elif key == "north":
            pole = np.array([
                0.0,
                0.0,
                1.0,
            ])

        elif key == "random":
            pole = rng.normal(
                size=3
            )
            pole /= np.linalg.norm(pole)

        else:
            raise ValueError(
                "coverage_pole must be 'south', 'north', "
                "'random', or a three-dimensional vector."
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

        norm = np.linalg.norm(pole)

        if norm < 1e-12:
            raise ValueError(
                "A custom coverage pole must be nonzero."
            )

        pole = pole / norm

    return pole


def sample_spherical_cap(
    n_points: int,
    coverage: float,
    coverage_pole,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample a spherical cap.

    Parameters
    ----------
    coverage:
        Fraction of the complete sphere surface.

        coverage=1.0 gives the full sphere.
        coverage=0.5 gives a hemisphere.
        coverage=0.2 gives 20 percent of the sphere.

    Returns
    -------
    unit_points:
        Unit-sphere points with shape (n_points, 3).

    parameter_samples:
        Surface parameters with shape (n_points, 2).

        The columns are:

            [azimuth, polar_angle]
    """

    coverage = float(coverage)

    if not 0.0 < coverage <= 1.0:
        raise ValueError(
            "coverage must satisfy 0 < coverage <= 1."
        )

    pole = resolve_coverage_pole(
        coverage_pole=coverage_pole,
        rng=rng,
    )

    # The area fraction of a spherical cap is:
    #
    # coverage = (1 - cos(theta_max)) / 2
    #
    # Therefore:
    #
    # cos(theta_max) = 1 - 2 * coverage

    cos_theta_min = 1.0 - 2.0 * coverage

    cos_theta = rng.uniform(
        cos_theta_min,
        1.0,
        size=n_points,
    )

    cap_azimuth = rng.uniform(
        0.0,
        2.0 * np.pi,
        size=n_points,
    )

    sin_theta = np.sqrt(
        np.maximum(
            0.0,
            1.0 - cos_theta**2,
        )
    )

    # Build an orthonormal basis around the selected cap pole.

    if abs(pole[0]) < 0.9:
        reference = np.array([
            1.0,
            0.0,
            0.0,
        ])
    else:
        reference = np.array([
            0.0,
            1.0,
            0.0,
        ])

    basis_u = np.cross(
        pole,
        reference,
    )
    basis_u /= np.linalg.norm(
        basis_u
    )

    basis_v = np.cross(
        pole,
        basis_u,
    )

    unit_points = (
        cos_theta[:, None] * pole
        + sin_theta[:, None]
        * (
            np.cos(cap_azimuth)[:, None] * basis_u
            + np.sin(cap_azimuth)[:, None] * basis_v
        )
    )

    # Convert the unit-sphere points into spherical coordinates.

    azimuth = np.mod(
        np.arctan2(
            unit_points[:, 1],
            unit_points[:, 0],
        ),
        2.0 * np.pi,
    )

    polar_angle = np.arccos(
        np.clip(
            unit_points[:, 2],
            -1.0,
            1.0,
        )
    )

    parameter_samples = np.column_stack([
        azimuth,
        polar_angle,
    ])

    return unit_points, parameter_samples


# ---------------------------------------------------------------------------
# Custom ellipsoid measurement generation
# ---------------------------------------------------------------------------

def ellipsoid_rotation_matrix(
    shape_params: np.ndarray,
) -> np.ndarray:
    """
    Return the ZYX rotation matrix for Ellipsoid_R parameters.

    shape_params:
        [x_c, y_c, z_c, a, r1, r2, alpha, beta, gamma]
    """

    shape_params = np.asarray(
        shape_params,
        dtype=float,
    )

    alpha = shape_params[6]
    beta = shape_params[7]
    gamma = shape_params[8]

    rotation_z = np.array([
        [np.cos(alpha), -np.sin(alpha), 0.0],
        [np.sin(alpha),  np.cos(alpha), 0.0],
        [0.0,            0.0,           1.0],
    ])

    rotation_y = np.array([
        [ np.cos(beta), 0.0, np.sin(beta)],
        [0.0,           1.0, 0.0],
        [-np.sin(beta), 0.0, np.cos(beta)],
    ])

    rotation_x = np.array([
        [1.0, 0.0,            0.0],
        [0.0, np.cos(gamma), -np.sin(gamma)],
        [0.0, np.sin(gamma),  np.cos(gamma)],
    ])

    return rotation_z @ rotation_y @ rotation_x


def map_unit_sphere_to_ellipsoid(
    unit_points: np.ndarray,
    shape_params: np.ndarray,
) -> np.ndarray:
    """
    Map canonical unit-sphere points to Ellipsoid_R.

    Ellipsoid_R representation:

        [x_c, y_c, z_c, a, r1, r2, alpha, beta, gamma]

    with:

        b = a * r1
        c = a * r2
    """

    unit_points = np.asarray(
        unit_points,
        dtype=float,
    )

    shape_params = np.asarray(
        shape_params,
        dtype=float,
    )

    if unit_points.ndim != 2 or unit_points.shape[1] != 3:
        raise ValueError(
            "unit_points must have shape (n_points, 3)."
        )

    if shape_params.shape != (9,):
        raise ValueError(
            "shape_params must have shape (9,)."
        )

    center = shape_params[:3]

    a = shape_params[3]
    r1 = shape_params[4]
    r2 = shape_params[5]

    b = a * r1
    c = a * r2

    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError(
            "All ellipsoid semiaxes must be positive."
        )

    semiaxes = np.array([
        a,
        b,
        c,
    ])

    rotation = ellipsoid_rotation_matrix(
        shape_params
    )

    scaled_points = (
        unit_points * semiaxes
    )

    return (
        scaled_points @ rotation.T
        + center
    )

def make_ellipsoid_measurement_function(
    coverage: float,
    coverage_pole="south",
):
    """
    Construct a measurement function for one fixed coverage.
    """

    coverage = float(coverage)

    def generate_measurement(
        config: ExperimentConfig,
        iteration: int,
    ) -> Measurement:
        """
        Generate one noisy polar-cap measurement.

        config.resample() is called by the runner before this function,
        so config.shape_params contains the ground-truth ellipsoid for
        the current iteration.
        """

        rng = np.random.default_rng(
            iteration
        )

        unit_points, parameter_samples = (
            sample_spherical_cap(
                n_points=config.num_points,
                coverage=coverage,
                coverage_pole=coverage_pole,
                rng=rng,
            )
        )

        source_points = map_unit_sphere_to_ellipsoid(
            unit_points=unit_points,
            shape_params=config.shape_params,
        )

        uncertainties = config.noise_function.evaluate(
            parameter_samples,
            params=config.noise_params,
        )

        errors = np.array([
            rng.multivariate_normal(
                mean=np.zeros(
                    config.shape.ambient_dim
                ),
                cov=covariance,
            )
            for covariance in uncertainties
        ])

        points = source_points + errors

        return Measurement(
            points=points,
            parameter_samples=parameter_samples,
            uncertainties=uncertainties,
            source_points=source_points,
            metadata={
                "iteration": iteration,
                "synthetic": True,
                "primitive": "ellipsoid",
                "coverage": coverage,
                "coverage_pole": str(
                    coverage_pole
                ),
            },
        )

    return generate_measurement


# ---------------------------------------------------------------------------
# Experiment factory
# ---------------------------------------------------------------------------

def make_ellipsoid_coverage_experiment(
    coverage: float,
    iterations: int = 50,
    coverage_pole="south",
) -> ExperimentConfig:
    """
    Construct an ellipsoid experiment for one fixed cap coverage.
    """

    coverage = float(coverage)

    if not 0.0 < coverage <= 1.0:
        raise ValueError(
            "coverage must satisfy 0 < coverage <= 1."
        )

    experiment_name = (
        f"ellipsoid_coverage_{coverage:.2f}"
    )

    def spherical_cap_coverage_to_polar_fraction(
            coverage: float,
    ) -> float:
        return np.arccos(1.0 - 2.0 * coverage) / np.pi

    spherical_coverage = spherical_cap_coverage_to_polar_fraction(
        coverage
    )


    # ------------------------------------------------------------------
    # Synthetic ground-truth models
    # ------------------------------------------------------------------

    ground_truth_source_density = UniformSourceDensity(
        n_dim=2,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
    )

    ground_truth_noise_function = IsotropicNoise(
        dim_param=shape.parameter_dim,
        dim_ambient=shape.ambient_dim,
    )
    # ------------------------------------------------------------------
    # BLISS source density
    # ------------------------------------------------------------------

    source_prior = GaussianPrior(
        mean=np.array([
            spherical_coverage,
        ]),
        cov=np.diag([
            0.00016,
        ]),
    )

    full_source_density = UniformSourceDensity(
        n_dim=2,
        domain_low=shape.domain_low,
        domain_high=shape.domain_high,
        prior=source_prior,
    )

    bliss_source_density = PartiallyFixedDensity(
        full_source_density,
        fixed_params={
            0: 0.5,
            1: 0.5,
            2: 0
        },
    )
    # ------------------------------------------------------------------
    # BLISS noise model
    # ------------------------------------------------------------------

    bliss_noise_function = IsotropicNoise(
        dim_param=shape.parameter_dim,
        dim_ambient=shape.ambient_dim,
        fixed_params=NOISE_VARIANCE,
    )

    source_init = np.array([
        spherical_coverage,
    ])

    source_bounds = [
        (spherical_coverage - 0.1, spherical_coverage + 0.1)
    ]

    angle_eps = 1e-6
    shape_bounds = [
        (-5, 5),  # x_c
        (-5, 5),  # y_c
        (-5, 5),  # z_c
        (0.1, 5),  # a
        (0.3, 1.0),  # r1
        (0.3, 1.0),  # r2
        (-np.pi, np.pi),  # α
        (-np.pi / 2 + angle_eps,
         np.pi / 2 - angle_eps),  # β
        (-np.pi, np.pi)  # γ
    ]

    # Since the noise parameter is fixed, no noise initialization
    # or noise bounds are required.
    bliss = BayesFitter(
        name="BLISS",
        source_density=bliss_source_density,
        noise_function=bliss_noise_function,
        shape_bounds=shape_bounds,
        source_init=source_init,
        source_bounds=source_bounds,
    )

    # ------------------------------------------------------------------
    # Custom polar-cap measurement generator
    # ------------------------------------------------------------------

    measurement_function = (
        make_ellipsoid_measurement_function(
            coverage=coverage,
            coverage_pole=coverage_pole,
        )
    )

    # ------------------------------------------------------------------
    # Experiment configuration
    # ------------------------------------------------------------------

    return ExperimentConfig(
        name=experiment_name,
        iterations=iterations,
        num_points=N_POINTS,

        shape=shape,

        # These describe the ground-truth models. The custom measurement
        # function handles the actual polar-cap sampling.
        source_density=ground_truth_source_density,
        noise_function=ground_truth_noise_function,

        # Generate a different random ellipsoid in every iteration.
        shape_params=None,
        shape_sampler=sample_ellipsoid_shape_params,

        # Store the fixed coverage as source-related ground truth.
        source_params=np.array([
            coverage,
        ]),

        # Isotropic noise variance.
        noise_params=np.array([
            NOISE_VARIANCE,
        ]),

        measurement_function=measurement_function,

        fitters=[
            CalafioreEllipsoidFitter(),
            bliss,
        ],

        labels=[
            "Calafiore",
            r"$\mathbf{BLISS}$",
        ],
    )


# ---------------------------------------------------------------------------
# Run or load the ellipsoid coverage sweep
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repository_root = Path(__file__).resolve().parents[1]

    coverages = [
        0.2,
    ]

    result_directories = []

    for coverage in coverages:
        config = make_ellipsoid_coverage_experiment(
            coverage=coverage,
            iterations=10,
            coverage_pole="south",
        )

        result_directory = (
            repository_root
            / "results"
            / "ellipsoid"
            / "coverage_sweep"
            / f"coverage_{coverage:.2f}"
        )

        pointcloud_directory = (
            result_directory
            / "pointclouds"
        )

        result_directories.append(result_directory)

        if RUN_EXPERIMENT:
            print(
                f"\nRunning {config.name}: "
                f"coverage={100.0 * coverage:.0f}%, "
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
                f"coverage={100.0 * coverage:.0f}%: "
                f"{result_directory}"
            )

            # Validate that the result exists and can be loaded.
            SweepResult.load(result_directory)
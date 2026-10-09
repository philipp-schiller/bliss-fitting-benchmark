import numpy as np

from methods.ellipse_fitters.bayes_fitter import BayesFitter
from methods.ellipse_fitters.hyper_renormalization import HyperRenormalizationFitter
from methods.ellipse_fitters.zhao_bayes import BayesianEllipseFitterZhao
from methods.ellipse_fitters.geometric_ls import GeometricEllipseFitter

from bayes_shape_fitting.inference.noise_functions import IsotropicNoise
from bayes_shape_fitting.inference.priors import GaussianPrior
from bayes_shape_fitting.inference.source_densitites import UniformSourceDensity
from bayes_shape_fitting.models.shapes.ellipse import Ellipse

from core.experiment_config import ExperimentConfig
from core.run_experiment import run_experiment
from core.plotting import plot_sweep_result
from core.sweep_result import SweepResult

from metrics.ellipse_metrics import compute_ellipse_metrics


shape = Ellipse()

# we define BLISS fitters here:
noise_prior =  GaussianPrior(mean=np.array([0.0225]), cov=np.array([0.0001]))
noise_bounds = [(0.001, 0.05)]
source_prior = GaussianPrior(mean=np.array([0.3, 0.20]), cov=np.diag([10, 0.00065]))
source_bounds = [(0, 1), (0.175, 0.225)]

# now we define init_values for source_density and noise function
source_init = np.array([0.3, 0.20])
noise_init = np.array(0.0225)

BLISS = BayesFitter(
    source_density = UniformSourceDensity(n_dim=1, domain_low=shape.domain_low, domain_high=shape.domain_high, periodic=shape.periodic, prior=source_prior),
    noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim, prior = noise_prior),
    source_init = source_init,
    noise_init = noise_init,
    source_bounds = source_bounds,
    noise_bounds = noise_bounds
)


IsotropicIllustrative = ExperimentConfig(
        name = "Isotropic_illustrative",

        iterations = 50,
        num_points = 100,

        shape = shape,

        # now define source density and noise function of the ground truth
        source_density = UniformSourceDensity(n_dim=1, domain_low=shape.domain_low,
                                               domain_high=shape.domain_high, periodic=shape.periodic),

        noise_function = IsotropicNoise(dim_param=shape.parameter_dim, dim_ambient=shape.ambient_dim),


        # define the ground-truth-params
        shape_params = np.array([-1, 0, 3, 2, np.pi / 4]),
        source_params = np.array([0.3, 0.20]),
        noise_params = 0.0225,


        # specify the baseline fitters
        fitters = [
            # we define standard fitters
            GeometricEllipseFitter(),
            HyperRenormalizationFitter(),
            BayesianEllipseFitterZhao(),
            # and the BLISS fitters
            BLISS
        ],

        # and their labels
        labels = [
            "GLS",
            "HR",
            "Zhao",
            r"$\mathbf{BLISS}$",
        ]
)

if __name__ == "__main__":

    np.random.seed(100)

    pointcloud_directory = f"../results/isotropic_illustrative/pointclouds"
    result_directory = f"../results/isotropic_illustrative"

    RUN_EXPERIMENT = False

    if RUN_EXPERIMENT:
        result = run_experiment(
            IsotropicIllustrative,
            save_pointclouds=True,
            pointcloud_directory=pointcloud_directory,
        )
        result.save(result_directory)
    else:
        result = SweepResult.load(result_directory)

    plot_sweep_result(
        result,
        IsotropicIllustrative,
        iteration=10,
        pointcloud_dir=pointcloud_directory,
        save_dir=result_directory + "/plots",
    )

    metrics = compute_ellipse_metrics(result)
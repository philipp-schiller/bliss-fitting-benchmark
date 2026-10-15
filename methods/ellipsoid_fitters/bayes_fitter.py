import numpy as np
from bayes_shape_fitting.visualization.animation_3d import ShapePlotterAnimations3D
from bayes_shape_fitting.visualization.plotting_3d import ShapePlotter3D
import matplotlib.pyplot as plt

from methods.ellipsoid_fitters.base import BaseEllipsoidFitter
from bayes_shape_fitting.inference.samplers.ecm_estimation import ECMOptimizer
from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.noise_functions import NoiseFunction
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.source_densitites import SourceDensity
from bayes_shape_fitting.models.shapes.ellipsoid import Ellipsoid_R

from bayes_shape_fitting.inference.fit import fit

from methods.ellipsoid_fitters.ellipsoid_utils import orient_south_pole_towards_scanner


class BayesFitter(BaseEllipsoidFitter):

    def __init__(
            self,
            name: str = "BayesFitter",
            source_density: SourceDensity = None,
            noise_function: NoiseFunction = None,
            shape_init: np.ndarray = None,
            source_init: np.ndarray = None,
            noise_init: np.ndarray = None,
            shape_bounds=None,
            source_bounds=None,
            noise_bounds=None,
            optimizer=None
    ):
        self.name = name
        self.source_density = source_density
        self.noise_function = noise_function
        self.shape_init = shape_init
        self.source_init = source_init
        self.noise_init = noise_init
        self.shape_bounds = shape_bounds
        self.source_bounds = source_bounds
        self.noise_bounds = noise_bounds
        self.optimizer = optimizer

    def fit(self, points: np.ndarray, source_density: SourceDensity = None, noise_function: NoiseFunction = None,
        shape_init: np.ndarray = None, source_init: np.ndarray = None, noise_init: np.ndarray = None,
        shape_bounds = None, source_bounds = None, noise_bounds = None, optimizer = None, minimize_kwargs = None, *args, ** kwargs) -> np.ndarray:

        # use instance defaults, but allow call-time override
        source_density = source_density or self.source_density
        noise_function = noise_function or self.noise_function
        source_init = source_init if source_init is not None else self.source_init
        noise_init = noise_init if noise_init is not None else self.noise_init
        shape_bounds = shape_bounds if shape_bounds is not None else self.shape_bounds
        source_bounds = source_bounds if source_bounds is not None else self.source_bounds
        noise_bounds = noise_bounds if noise_bounds is not None else self.noise_bounds
        optimizer = optimizer or self.optimizer

        if optimizer is None:
            optimizer = ECMOptimizer(verbose=False, scale_factor=1, max_iter=7, inner_max_iter=15, methods={
                'shape': 'L-BFGS-B',
                'source': "Powell",
                'noise': 'Powell'
            })

        shape = Ellipsoid_R()

        init = shape.initial_fit(points)

        scanner = np.array([0, 0, 10])
        init = orient_south_pole_towards_scanner(init, scanner)

        likelihood = MarginalLikelihood(
            shape=shape,
            source_density=source_density,
            noise_function=noise_function,
            num_samples=60,
        )

        fit_result = fit(points=points, shape=shape, likelihood=likelihood, shape_init=init, noise_init=noise_init, source_init=source_init, shape_bounds=shape_bounds, source_bounds=source_bounds, noise_bounds=noise_bounds, optimizer=optimizer)


        return fit_result.shape_params


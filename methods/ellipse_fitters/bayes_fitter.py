import copy

import numpy as np

from methods.ellipse_fitters.base import BaseEllipseFitter
from methods.ellipse_fitters.geometric_ls import _algebraic_ellipse_fit
from methods.ellipse_fitters.ellipse_utils import get_ellipse_params, canonicalize_ellipse_params
from methods.ellipse_fitters.self_normalized import SelfNormalizedFitter

from bayes_shape_fitting.visualization.plotting_2d import ShapePlotter2D
from bayes_shape_fitting.visualization.animation_2d import ShapePlotterAnimations
import matplotlib.pyplot as plt

from bayes_shape_fitting.inference.samplers.ecm_estimation import ECMOptimizer
from bayes_shape_fitting.inference.samplers.map_estimation import MAPOptimizer
from bayes_shape_fitting.inference.likelihoods import MarginalLikelihood
from bayes_shape_fitting.inference.noise_functions import NoiseFunction
from bayes_shape_fitting.inference.posterior import MarginalPosterior
from bayes_shape_fitting.inference.source_densitites import SourceDensity
from bayes_shape_fitting.models.shapes.ellipse import Ellipse
from bayes_shape_fitting.models.shapes.ellipse_reparametrized import Ellipse_R
from bayes_shape_fitting.inference.fit import fit as standard_fit



def convert_params_from_standard_to_reparametrized(p):
    cx, cy, a, b, angle = p
    return np.array([cx, cy, a, b / a, angle])

def convert_params_from_reparametrized_to_standard(p):
    cx, cy, a, r, angle = p
    return np.array([cx, cy, a, a*r, angle])


class BayesFitter(BaseEllipseFitter):

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
        source_bounds = None, noise_bounds = None, shape_bounds = None, optimizer=None, minimize_kwargs = None, plot: bool = False, *args, ** kwargs) -> np.ndarray:

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
            optimizer = ECMOptimizer(verbose=False, scale_factor=1, max_iter=15, inner_max_iter=100, methods={
                'shape': 'L-BFGS-B',
                'source': 'Powell',
                'noise': 'Powell'
            })

        shape = Ellipse()

        if shape_bounds is None:
            shape_bounds = [
                (-10, 10),  # x_c
                (-10, 10),  # y_c
                (0.3, np.inf),  # a
                (0.3, np.inf),  # b
                (-np.inf, np.inf),  # theta
            ]

        try:
            cx0, cy0, a0, b0, theta0 = SelfNormalizedFitter().fit(points)

            init_params = np.array(
                [cx0, cy0, a0, b0, theta0],
                dtype=float,
            )

            in_bounds = all(
                lower <= value <= upper
                for value, (lower, upper)
                in zip(init_params, shape_bounds)
            )

            if not in_bounds:
                raise ValueError(
                    f"Initial fit outside bounds: {init_params}"
                )

        except Exception:
            # --- 1. Initial fit  ---
            A, B, C, D, E, F = _algebraic_ellipse_fit(points)
            cx0, cy0, a0, b0, theta0 = get_ellipse_params(np.array([A, B, C, D, E, F]))

        # we flip the source-density also:
        shape_init, t_shift = canonicalize_ellipse_params(
            np.array([cx0, cy0, a0, b0, theta0])
        )

        likelihood = MarginalLikelihood(
            shape=shape,
            source_density=source_density,
            noise_function=noise_function,
            num_samples=300,
        )

        fit_result = standard_fit(points=points, shape=shape, likelihood=likelihood, shape_init=shape_init,
                         source_init=source_init, noise_init = noise_init, shape_bounds=shape_bounds,
                         source_bounds=source_bounds, noise_bounds = noise_bounds,
                         optimizer=optimizer)

        if plot:
            plotter = ShapePlotter2D()
            fig, ax, _ =  plotter.plot_shape(shape, fit_result.shape_params,  label="Fitted Surface", color="green")
            plotter.plot_shape(shape, shape_init, ax=ax, label="Initial Fit", color="orange")
            plotter.plot_observations(observations=points, ax=ax)

            animator = ShapePlotterAnimations()
            ani = animator.animate_optimization_result(fit_result, shape, source_density, noise_function, source_prior=source_density.get_prior(), noise_prior=noise_function.get_prior(), source_bounds=source_bounds, noise_bounds=noise_bounds, observations = points, interval=1000)
            plt.show()

        return canonicalize_ellipse_params(fit_result.shape_params)[0]

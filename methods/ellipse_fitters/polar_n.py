from methods.ellipse_fitters.base import BaseEllipseFitter
from methods.ellipse_fitters.self_normalized import SelfNormalizedFitter
import numpy as np
from methods.ellipse_fitters.ellipse_utils import get_ellipse_params, get_homogenous_coords_from_ellipse
from scipy.linalg import eig
from numpy.linalg import norm
from scipy.optimize import least_squares


class Polar_n_Fitter(BaseEllipseFitter):

    def fit(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param points:
        :param **kwargs:
        """

        def conic_vector_to_matrix(params):
            A, B, C_, D, E, F = params
            return np.array([
                [A, B / 2, D / 2],
                [B / 2, C_, E / 2],
                [D / 2, E / 2, F]
            ])

        def polar_n_direction_distance(point, C):
            x, y = point
            m = np.array([x, y, 1.0])  # homogeneous coordinates

            mCm = m @ C @ m

            # Extract coefficients from C
            A = C[0, 0]
            B = 2 * C[0, 1]
            C_ = C[1, 1]
            D = 2 * C[0, 2]
            E = 2 * C[1, 2]

            fx = 2 * A * x + B * y + D
            fy = 2 * C_ * y + B * x + E

            mGm = fx ** 2 + fy ** 2
            mWm = m @ m

            mGm2 = mGm ** 2
            mCm_mWm = mCm * mWm

            # Guard against sqrt of negative or zero
            if mGm <= 0 or mGm2 <= mCm_mWm:
                return 1e6  # large penalty

            denom = np.sqrt(mGm) * np.sqrt(1 + (mGm2 - mCm_mWm) / mGm2)
            if denom == 0:
                return 1e6

            return np.abs(mCm) / denom

        def residuals(params):
            C = conic_vector_to_matrix(params)
            return np.array([
                polar_n_direction_distance(pt, C) for pt in points
            ])

        # compute the initial guess
        L = SelfNormalizedFitter()
        param_ini = L.fit(points)
        initial_params = get_homogenous_coords_from_ellipse(param_ini)

        try:
            result = least_squares(
                residuals,
                initial_params,
                method='lm',
                max_nfev=5000,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )

            if not result.success:
                return np.full(5, np.nan)

            return get_ellipse_params(result.x)

        except Exception as e:
            print(e)
            return np.full(5, np.nan)

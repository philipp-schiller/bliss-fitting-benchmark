from methods.ellipse_fitters.base import BaseEllipseFitter
import numpy as np
from utils.ellipse_utils import get_ellipse_params
from scipy.linalg import eig
from numpy.linalg import norm


class ElliFitFitter(BaseEllipseFitter):

    def fit(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param points:
        :param **kwargs:
        """

        N = len(points)
        if N < 5:
            return np.full(5, np.nan)

        # Step 1: Compute bounding box and center
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        mean_x = (x_min + x_max) / 2.0
        mean_y = (y_min + y_max) / 2.0

        # Step 2: Build design matrix X and vector Yb
        X = np.zeros((N, 5))
        Yb = np.zeros(N)

        for i, (x, y) in enumerate(points):
            xb = x - mean_x
            yb = y - mean_y
            X[i] = [xb * xb, 2 * xb * yb, -2 * xb, -2 * yb, -1]
            Yb[i] = -yb * yb

        # Step 3: Solve linear system using least squares (QR decomposition)
        phi, *_ = np.linalg.lstsq(X, Yb, rcond=None)
        a, b, c, d, e = phi

        # Step 4: Compute ellipse parameters
        denom = a - b * b
        if np.abs(denom) < 1e-12:
            return np.full(5, np.nan)

        x = (c - d * b) / denom
        y = (a * d - c * b) / denom

        ab_nom = 2 * (e + a * x * x + 2 * b * x * y + y * y)
        first_denom = 1 + a
        second_denom = np.sqrt((1 - a) ** 2 + 4 * b * b)

        try:
            axis1 = np.sqrt(ab_nom / (first_denom - second_denom))
            axis2 = np.sqrt(ab_nom / (first_denom + second_denom))
        except:
            return np.full(5, np.nan)

        theta = -0.5 * np.arctan2(2 * b, 1 - a)
        cx = x + mean_x
        cy = y + mean_y

        return np.array([axis1, axis2, cx, cy, theta])
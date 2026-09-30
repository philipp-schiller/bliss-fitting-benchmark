from methods.ellipse_fitters.base import BaseEllipseFitter
import numpy as np
from scipy.linalg import eig
from methods.ellipse_fitters.ellipse_utils import get_ellipse_params


class TaubinFitter(BaseEllipseFitter):
    """
    Taubin Normalization for algebraic ellipse fit.
    Reference:
    G. Taubin, "Estimation Of Planar Curves, Surfaces, And Nonplanar
    Space Curves Defined By Implicit Equations, With Applications
    To Edge And Range Image Segmentation", PAMI 1991.
    """

    def fit(self, points: np.ndarray, **kwargs) -> np.ndarray:
        x = points[:, 0]
        y = points[:, 1]

        # ----------------------------
        # Design matrix (algebraic form)
        # ----------------------------
        D = np.vstack([
            x**2,
            x * y,
            y**2,
            x,
            y,
            np.ones_like(x)
        ]).T

        # Scatter matrix
        S = D.T @ D

        # ----------------------------------------------------
        # Taubin Normalization Matrix (K)
        # Derived from constraint: ||grad F||^2 normalized
        # ----------------------------------------------------
        K = np.zeros((6, 6))
        K[0, 0] = 4 * np.mean(x**2)
        K[1, 1] = np.mean(x**2 + y**2)
        K[2, 2] = 4 * np.mean(y**2)
        K[0, 1] = K[1, 0] = 2 * np.mean(x*y)
        K[1, 2] = K[2, 1] = 2 * np.mean(x*y)
        K[0, 3] = K[3, 0] = 2 * np.mean(x)
        K[2, 4] = K[4, 2] = 2 * np.mean(y)
        K[3, 3] = 1
        K[4, 4] = 1
        K[5, 5] = 0  # constant term has no derivative

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = eig(S, K)

        # Select valid eigenvalues
        mask = (
            np.isfinite(eigenvalues)
            & ~np.isnan(eigenvalues)
            & np.isreal(eigenvalues)
        )
        eigenvalues = np.real(eigenvalues[mask])
        eigenvectors = np.real(eigenvectors[:, mask])

        # Choose eigenvector with smallest positive eigenvalue
        idx = np.argmin(eigenvalues)
        solution = eigenvectors[:, idx]

        # Convert algebraic ellipse params → geometric params
        return get_ellipse_params(solution)

import numpy as np
from numpy.linalg import eig, norm, inv

from methods.ellipse_fitters.base import BaseEllipseFitter
from methods.ellipse_fitters.ellipse_utils import get_ellipse_params


class DirectEllipseFit(BaseEllipseFitter):

    def fit(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fits an ellipse to 2D points.

        Parameters
        ----------
        points : (N, 2) np.ndarray
            Input 2D data points.

        Returns
        -------
        theta : (6,) np.ndarray
            Ellipse parameters [a, b, c, d, e, f] for
            a*x^2 + b*x*y + c*y^2 + d*x + e*y + f = 0,
            with b^2 - 4ac < 0.
        """

        n_pts = len(points)
        if n_pts < 5:
            raise ValueError("At least 5 points are required to fit an ellipse.")

        # --- 1. Normalize the data (translate and scale isotropically)
        normalized_points, T = self._normalize_data_isotropically(points)

        # Add column of ones as homogeneous coordinates
        X = np.hstack([normalized_points, np.ones((n_pts, 1))])

        # --- 2. Fit ellipse in normalized space
        theta = self._direct_ellipse_fit(X.T)
        theta /= norm(theta)

        # --- 3. Build conic matrix C
        a, b, c, d, e, f = theta
        C = np.array([
            [a, b / 2, d / 2],
            [b / 2, c, e / 2],
            [d / 2, e / 2, f]
        ])

        # --- 4. Denormalize C
        C_denorm = T.T @ C @ T

        aa = C_denorm[0, 0]
        bb = 2 * C_denorm[0, 1]
        cc = C_denorm[1, 1]
        dd = 2 * C_denorm[0, 2]
        ee = 2 * C_denorm[1, 2]
        ff = C_denorm[2, 2]

        theta = np.array([aa, bb, cc, dd, ee, ff])
        theta /= norm(theta)
        return get_ellipse_params(theta)

    # ----------------------------------------------------------------------

    def _normalize_data_isotropically(self, pts: np.ndarray):
        """Shift and scale data so it lies roughly in [-1, 1] range."""
        centroid = np.mean(pts, axis=0)
        shifted = pts - centroid
        scale = np.sqrt(2) / np.mean(np.linalg.norm(shifted, axis=1))
        normalized = shifted * scale

        # Homogeneous transformation matrix
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])

        return normalized, T

    def _direct_ellipse_fit(self, X: np.ndarray) -> np.ndarray:
        """
        Numerically stable direct least squares ellipse fit (Halír & Flusser).
        X is 3xN array of homogeneous coordinates.
        """

        x = X[0, :]
        y = X[1, :]

        # Design matrices
        D1 = np.vstack([x ** 2, x * y, y ** 2]).T
        D2 = np.vstack([x, y, np.ones_like(x)]).T

        # Scatter matrices
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2

        # Reduced scatter matrix
        T = -inv(S3) @ S2.T
        M = S1 + S2 @ T

        # Constraint matrix for ellipse
        C = np.zeros((3, 3))
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1

        # Solve generalized eigenvalue problem
        eigvals, eigvecs = eig(np.linalg.inv(C) @ M)

        # Select the eigenvector satisfying 4ac - b^2 > 0
        cond = 4 * eigvecs[0, :] * eigvecs[2, :] - eigvecs[1, :] ** 2
        valid = np.where(cond > 0)[0]

        if len(valid) == 0:
            raise RuntimeError("No valid ellipse solution found.")

        a1 = eigvecs[:, valid[0]]

        # Recover full parameter vector
        a2 = T @ a1
        theta = np.concatenate([a1, a2])
        return theta
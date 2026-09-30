from methods.ellipse_fitters.base import BaseEllipseFitter
import numpy as np
from methods.ellipse_fitters.ellipse_utils import get_ellipse_params
from scipy.linalg import eig


class LeastSquaresFitter(BaseEllipseFitter):

    def fit(self, points: np.ndarray, constraint_matrix=np.eye(6), **kwargs) -> np.ndarray:
        """
        :param points:
        :param constraint_matrix: Constraint matrix of the eigenvalue problem
        :param **kwargs:
        """

        # we use an elliptic constraint

        C = np.zeros((6, 6))
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1

        x = points[:, 0]
        y = points[:, 1]

        # Design matrix
        D = np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T
        M = D.T @ D
        eigenvalues, eigenvectors = eig(M, C)

        mask = ~np.isnan(eigenvalues) & (eigenvalues > 0) & ~np.isinf(eigenvalues) & np.isreal(eigenvalues)
        eigenvalues = np.real(eigenvalues[mask])
        eigenvectors = eigenvectors.T[mask].T

        n = len(eigenvalues)
        # scale the solution such that it fulfills the constraint
        mu = np.zeros(n)
        residuals = np.zeros(n)
        for i in range(n):
            mu[i] = np.sqrt(eigenvalues[i] / (eigenvectors[:, i] @ M @ eigenvectors[:, i]))
            eigenvectors[:, i] *= mu[i]
            residuals[i] = eigenvectors[:, i] @ M @ eigenvectors[:, i]

        # get the index of the smallest eigenvalue

        min_ev_idx = np.argmin(eigenvalues)

        solution = eigenvectors[:, min_ev_idx]

        return get_ellipse_params(solution)

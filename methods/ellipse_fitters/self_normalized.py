from methods.ellipse_fitters.base import BaseEllipseFitter
import numpy as np
from methods.ellipse_fitters.ellipse_utils import get_ellipse_params
from scipy.linalg import eig


class SelfNormalizedFitter(BaseEllipseFitter):
    def fit(self, points: np.ndarray, bias_correction=True, **kwargs) -> np.ndarray:
        """
        Fits an ellipse to data
        :param bias_correction:
        :param points:
        :param kwargs:
        :return:
        """

        x = points[:, 0]
        y = points[:, 1]

        # Design matrix
        D = np.vstack([x ** 2, x * y, y ** 2, x, y, np.ones_like(x)]).T
        S = D.T @ D  # Shape: [N, 6]

        S12 = S[5, :5]
        S22 = S[5, 5]
        Sr = S[:5, :5] - np.outer(S12, S12) / S22

        C = np.array([
            [4 * S12[0], 2 * S12[1], 0, 2 * S12[3], 0],
            [2 * S12[1], S12[0] + S12[2], 2 * S12[1], S12[4], S12[3]],
            [0, 2 * S12[1], 4 * S12[2], 0, 2 * S12[4]],
            [2 * S12[3], S12[4], 0, S22, 0],
            [0, S12[3], 2 * S12[4], 0, S22]
        ])
        L = np.linalg.inv(np.linalg.cholesky(C))
        evals, evects = np.linalg.eigh(np.inner(np.dot(L, Sr), L))
        j = np.argmin(evals)
        G1 = np.dot(evects[:, j], L)
        G2 = -np.dot(G1, S12) / S22

        # to the debias:
        if bias_correction:
            G2 += (G1[0] + G1[2]) * evals[j]
        sol = np.r_[G1, G2]

        return get_ellipse_params(sol)

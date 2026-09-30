import numpy as np
from base import BaseEllipseFitter
from least_squares import LeastSquaresFitter
from ellipse_utils import get_ellipse_params, get_homogenous_coords_from_ellipse

import copy


class FundamentalSchemeFitter(BaseEllipseFitter):

    def fit(self, points, uncertainties: np.ndarray = None, iterations: int = 100, *args, **kwargs):


        noise_x = points[:, 0]
        noise_y = points[:, 1]
        f = 1

        W = np.ones(len(noise_x), dtype="float64")

        # first, we compute the initial fit with least squares
        L = LeastSquaresFitter()
        alg_fit = L.fit(points)
        # convert it to homogenous coordinates
        theta_zero = get_homogenous_coords_from_ellipse(alg_fit)
        theta_zero[[1, 3, 4]] /= 2

        diff = 10000.0

        j = 0

        grad_u = lambda x, y: np.array([[2 * x, 2 * y, 0, 2, 0, 0], [0, 2 * x, 2 * y, 0, 2, 0]])

        while diff > 1e-10 and j < iterations:
            xi_sum = np.zeros((6, 6))
            L_sum = np.zeros((6, 6))
            V0_xi_list = []
            for i in range(len(noise_x)):
                x = noise_x[i]
                y = noise_y[i]
                xi = np.array([[x ** 2, 2 * x * y, y ** 2, 2 * f * x, 2 * f * y, f * f]])
                xi_sum += np.dot(W[i], np.dot(xi.T, xi))

                if uncertainties is None:
                    V0_xi = 4 * np.array(
                        [
                            [x ** 2, x * y, 0, f * x, 0, 0],
                            [x * y, x ** 2 + y ** 2, x * y, f * y, f * x, 0],
                            [0, x * y, y ** 2, 0, f * y, 0],
                            [f * x, f * y, 0, f ** 2, 0, 0],
                            [0, f * x, f * y, 0, f ** 2, 0],
                            [0, 0, 0, 0, 0, 0],
                        ]
                    )
                else:
                    V0_xi = grad_u(x, y).T @ uncertainties[i] @ grad_u(x, y)
                V0_xi_list.append(V0_xi)
                L_sum += np.dot(
                    np.dot(W[i] ** 2, np.dot(xi.T[:, 0], theta_zero) ** 2), V0_xi
                )


            M = xi_sum / len(noise_x)
            L = L_sum / len(noise_x)
            X = M - L
            eig_val, eig_vec = np.linalg.eig(X)
            theta = eig_vec[:, np.argmin(np.abs(eig_val))]

            if np.dot(theta, theta_zero) < 0:
                theta = np.dot(-1, theta)
            diff = np.sum(np.abs(theta_zero - theta))
            if diff <= 1e-10:
                break

            theta /= np.linalg.norm(theta)

            # Compute all weights at once
            W = np.array([1.0 / (theta.T @ V0_xi_list[i] @ theta + 1e-6) for i in range(len(noise_x))])
            W /= np.mean(W)

            theta_zero = theta
            j = j + 1
            if j == iterations - 1:
                print("No convergence")
                return [np.nan]*5
           
        theta[[1, 3, 4]] *= 2

        return get_ellipse_params(theta)

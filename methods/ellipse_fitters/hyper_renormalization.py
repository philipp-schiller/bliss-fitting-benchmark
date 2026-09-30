import numpy as np
from numpy.linalg import norm
from scipy.linalg import eig
from methods.ellipse_fitters.base import BaseEllipseFitter
from methods.ellipse_fitters.ellipse_utils import get_ellipse_params


class HyperRenormalizationFitter(BaseEllipseFitter):
    """
    Kanatani Hyper-Renormalization for conic/ellipse fitting.
    This is the 2nd-order extension of renormalization,
    first proposed by Kanatani (2010), sometimes called the 'hyper-core' method.
    """

    def fit(self, points, uncertainties=None, iterations=30, *args, **kwargs):

        x = points[:, 0]
        y = points[:, 1]
        Np = len(points)

        # -------------------------
        # xi(x,y) = [x^2, xy, y^2, x, y, 1]
        # -------------------------
        def xi_vec(x, y):
            return np.array([x*x, x*y, y*y, x, y, 1.0])

        # ∂F/∂(x,y)
        def grad_F(x, y, th):
            a, b, c, d, e, _ = th
            Fx = 2*a*x + b*y + d
            Fy = b*x + 2*c*y + e
            return np.array([Fx, Fy])

        # Jacobian of xi wrt (x,y)
        def J_xi(x, y):
            return np.array([
                [2*x,     y,   0,   1, 0, 0],
                [0,       x, 2*y,   0, 1, 0]
            ])   # shape = (2×6)

        # default uncertainty = isotropic identity
        if uncertainties is None:
            uncertainties = [np.eye(2)] * Np

        # ------------------------------------------
        # Initial estimate: DLT
        # ------------------------------------------
        X = np.stack([x*x, x*y, y*y, x, y, np.ones(Np)], axis=1)
        _, _, Vt = np.linalg.svd(X)
        theta = Vt[-1, :]
        theta /= norm(theta)

        # ------------------------------------------
        # Hyper-Renormalization Iteration
        # ------------------------------------------
        for _ in range(iterations):

            M = np.zeros((6, 6))
            N = np.zeros((6, 6))
            H = np.zeros((6, 6))    # Hyper correction

            for i in range(Np):

                xi = xi_vec(x[i], y[i])
                M += np.outer(xi, xi)

                # local covariance propagation
                J   = J_xi(x[i], y[i])
                Sig = uncertainties[i]                   # 2×2
                V0  = J.T @ Sig @ J                      # 6×6

                N += V0

                # Hyper correction term:
                g = grad_F(x[i], y[i], theta)            # gradient of F
                sigma2 = g @ Sig @ g                     # scalar
                H += sigma2 * V0                         # weight second-order term

            # Normalize sums
            M /= Np
            N /= Np
            H /= Np

            # Hyper-renormalization core matrix:
            #   (N + H) θ = λ M θ
            #
            # N:  1st-order term (renormalization)
            # H:  2nd-order hyper-correction
            #
            A = N + H

            eigvals, eigvecs = eig(A, M)

            # renormalization chooses the largest eigenvalue (best conditioned estimator)
            idx = np.argmax(eigvals.real)
            new_theta = eigvecs[:, idx].real

            # normalize using M (geometrically consistent)
            new_theta /= np.sqrt(new_theta @ M @ new_theta)

            # convergence
            if norm(new_theta - theta) < 1e-12:
                theta = new_theta
                break

            theta = new_theta

        # final output: convert coefficients → geometric ellipse parameters
        return get_ellipse_params(theta)

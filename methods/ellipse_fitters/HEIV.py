import numpy as np
from numpy.linalg import norm
from scipy.linalg import eig, eigh

from methods.ellipse_fitters.base import BaseEllipseFitter
from methods.ellipse_fitters.direct_ellipse_fit import DirectEllipseFit
from methods.ellipse_fitters.ellipse_utils import get_ellipse_params, \
    get_homogenous_coords_from_ellipse
from methods.ellipse_fitters.least_squares import LeastSquaresFitter



class HEIVFitter(BaseEllipseFitter):

    """
    Full Heteroscedastic Errors-In-Variables (HEIV) ellipse fitting.

    Uses the full algebraic conic representation:
        ax^2 + bxy + cy^2 + dx + ey + f = 0

    Assumes:
        - uncertainties[i] is a 2×2 covariance matrix
        - generalized eigenvalue formulation (full HEIV)
    """

    # ------------------------------------------------------------------
    # Algebraic embedding and Jacobian
    # ------------------------------------------------------------------

    @staticmethod
    def u(x, y):
        """Full algebraic embedding."""
        return np.array([x * x, x * y, y * y, x, y, 1.0])

    @staticmethod
    def grad_u(x, y):
        """Jacobian of u(x,y) with respect to (x,y)."""
        return np.array([
            [2 * x, y, 0, 1, 0, 0],
            [0, x, 2 * y, 0, 1, 0],
        ])

    def algebraic_ls_initialization(self, points):
        """
        Algebraic least-squares initialization for full conic.

        Fits: ax^2 + bxy + cy^2 + dx + ey + f = 0
        """
        x = points[:, 0]
        y = points[:, 1]

        # Design matrix
        D = np.column_stack([
            x * x,
            x * y,
            y * y,
            x,
            y,
            np.ones_like(x)
        ])

        # Solve D theta = 0 via SVD
        _, _, Vt = np.linalg.svd(D, full_matrices=False)
        theta = Vt[-1]  # smallest singular value

        # Normalize for numerical stability
        theta /= np.linalg.norm(theta)

        return theta

    # ------------------------------------------------------------------
    # Main fitting routine
    # ------------------------------------------------------------------

    def fit_(self, points, uncertainties, max_iter=50, tol=1e-6,  *args, **kwargs):
        if uncertainties is None:
            raise ValueError("HEIV requires per-point covariance matrices.")

        x = points[:, 0]
        y = points[:, 1]
        Np = len(points)

        # --------------------------------------------------------------
        # Initialization via algebraic least squares
        # --------------------------------------------------------------

        #ls = DirectEllipseFit()
        #theta = get_homogenous_coords_from_ellipse(ls.fit(points))
        #theta /= np.linalg.norm(theta)

        theta = self.algebraic_ls_initialization(points)

        # --------------------------------------------------------------
        # Iterative HEIV refinement
        # --------------------------------------------------------------

        for _ in range(max_iter):
            M = np.zeros((6, 6))
            N = np.zeros((6, 6))

            converged = True

            for i in range(Np):
                ui = self.u(x[i], y[i])
                Ji = self.grad_u(x[i], y[i])
                Ci = uncertainties[i]

                # Propagated covariance in algebraic space

                eps = 1e-8
                Bi = Ji.T @ Ci @ Ji + eps * np.eye(6)

                denom = theta @ Bi @ theta
                if denom <= 1e-12:
                    raise ValueError("Degenerate propagated uncertainty encountered.")

                Ai = np.outer(ui, ui)

                M += Ai / denom
                N += (theta @ Ai @ theta) * Bi / (denom ** 2)

            # Solve generalized eigenproblem
            eigvals, eigvecs = eig(M, N)
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)

            idx = np.argmin(eigvals)
            theta_new = eigvecs[:, idx]

            # Normalize with respect to N
            theta_new /= np.sqrt(theta_new @ N @ theta_new)

            # Sign-invariant convergence
            delta = min(
                np.linalg.norm(theta - theta_new),
                np.linalg.norm(theta + theta_new),
            )

            theta = theta_new
            if delta < tol:
                break

        # --------------------------------------------------------------
        # Ellipse validity check: 4ac − b^2 > 0
        # --------------------------------------------------------------

        a, b, c = theta[0], theta[1], theta[2]
        if 4 * a * c - b * b <= 0:
            return [np.nan] * 5

        print("HEIV + "+  str(get_ellipse_params(theta)))

        return get_ellipse_params(theta)

    def fit(self, points, uncertainties=None, iterations=100, tol=1e-3, *args, **kwargs):

        """
        WE try the reduced approach
        :param points:
        :param uncertainties:
        :param iterations:
        :param tol:
        :param args:
        :param kwargs:
        :return:
        """

        x = points[:, 0]
        y = points[:, 1]

        u = lambda x,y: np.array([[x**2,x*y,y**2,x,y]])

        grad_u = lambda x, y: np.array([[2 * x, y, 0, 1, 0], [0, x, 2 * y, 0, 1]])

        # first, we compute the initial fit with least squares
        #L = LeastSquaresFitter()
        L = DirectEllipseFit()
        alg_fit = L.fit(points)
        # convert it to homogenous coordinates
        theta_zero = get_homogenous_coords_from_ellipse(alg_fit)
        eta_zero = theta_zero[:5]
        eta_zero /= np.linalg.norm(eta_zero)

        diff = 10000.0
        j = 0

        while j<iterations and diff > tol:

            # Compute weights and z once per iteration
            weights = np.zeros(len(points))
            for i in range(len(points)):
                B_i = grad_u(x[i], y[i]).T @ uncertainties[i] @ grad_u(x[i], y[i])
                weights[i] = 1 / ((eta_zero.T @ B_i @ eta_zero) + 1e-6)

            z = np.zeros(5)
            for i in range(len(points)):
                z += weights[i] * u(x[i], y[i])[0]
            z /= np.sum(weights)

            # Now compute M_i and N_i
            M_i = np.zeros((5, 5))
            N_i = np.zeros((5, 5))
            for i in range(len(points)):
                A_i = (u(x[i], y[i]) - z).T @ (u(x[i], y[i]) - z)
                B_i = grad_u(x[i], y[i]).T @ uncertainties[i] @ grad_u(x[i], y[i])
                denom = eta_zero.T @ B_i @ eta_zero
                if denom < 1e-6:  # avoid division by zero
                    denom = 1e-6
                M_i += A_i / denom
                N_i += ((eta_zero.T @ A_i @ eta_zero) / (denom ** 2)) * B_i

            eig_val, eig_vec = eig(M_i, N_i)

            # eigenvector to the smallest eigenvalue
            eig_val_real = np.real(eig_val)  # Ensure real part
            idx = np.argmin(eig_val_real)  # Index of smallest eigenvalue
            #idx = np.argmin(np.abs(eig_val_real-1))  # Index closest to one
            eta = np.real(eig_vec[:, idx])

            # Normalize w.r.t. N_i for stability
            eta /= np.linalg.norm(eta)

            diff = min(
                np.linalg.norm(eta_zero - eta),
                np.linalg.norm(eta_zero + eta))

            eta_zero = eta
            j = j + 1

            if j == iterations - 1:
                print("No convergence")
                #return [np.nan]*5

        print("Convergence after "+ str(j))

        # now we compute the centroid:
        weights = np.zeros(len(points))
        for i in range(len(points)):
            B_i = grad_u(x[i],y[i]).T @ uncertainties[i] @ grad_u(x[i],y[i])
            weights[i] = 1 / (eta.T @ B_i @ eta)

        z = np.zeros(5)
        for i in range(len(points)):
            z += weights[i]*u(x[i],y[i])[0]
        # normalize
        z /= np.sum(weights)

        alpha = -z.T @ eta

        theta = np.hstack([eta,alpha])

        #theta[[1, 3, 4]] /= 2
        print(get_ellipse_params(theta))
        return get_ellipse_params(theta)





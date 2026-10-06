
import numpy as np
from scipy.spatial.transform import Rotation

from methods.ellipsoid_fitters.base import (
    BaseEllipsoidFitter,
)


class CalafioreEllipsoidFitter(BaseEllipsoidFitter):
    """
    Algebraic ellipsoid fitter based on quadric coefficients.

    Returns parameters in the form:

        [x_c, y_c, z_c, a, r_1, r_2, alpha, beta, gamma]

    where:

        r_1 = a / b
        r_2 = a / c

    and alpha, beta, gamma are ZYX Euler angles in radians.
    """

    def __init__(self, name: str = "Calafiore"):
        self.name = name

    def fit(
        self,
        points: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Fit an ellipsoid to three-dimensional points.

        Parameters
        ----------
        points:
            Point cloud with shape (n_points, 3).

        Returns
        -------
        np.ndarray
            Ellipsoid parameters:

            [x_c, y_c, z_c, a, r_1, r_2, alpha, beta, gamma]

        Raises
        ------
        ValueError
            If the points are invalid or no valid ellipsoid can be
            recovered.
        """

        points = np.asarray(points, dtype=float)

        self._validate_points(points)

        coefficients = self._fit_quadric_coefficients(points)

        return self._quadric_to_parameters(coefficients)

    @staticmethod
    def _validate_points(points: np.ndarray) -> None:
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "Expected points with shape (n_points, 3), "
                f"but received {points.shape}."
            )

        if points.shape[0] < 9:
            raise ValueError(
                "At least nine points are required to fit an "
                "ellipsoid."
            )

        if not np.all(np.isfinite(points)):
            raise ValueError(
                "The point cloud contains NaN or infinite values."
            )

    def _fit_quadric_coefficients(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate algebraic quadric coefficients.

        The quadric is represented as:

            x.T @ Q @ x + 2 q.T @ x + d = 0

        with coefficient ordering:

            [A, B, C, D, E, F, G, H, I, J]

        corresponding to:

            A x² + B y² + C z²
            + 2D xy + 2E xz + 2F yz
            + 2G x + 2H y + 2I z + J = 0.
        """

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        design_matrix = np.column_stack([
            x**2,
            y**2,
            z**2,
            2.0 * x * y,
            2.0 * x * z,
            2.0 * y * z,
            2.0 * x,
            2.0 * y,
            2.0 * z,
            np.ones(len(points)),
        ])

        _, _, right_singular_vectors = np.linalg.svd(
            design_matrix,
            full_matrices=False,
        )

        # Start with the smallest-singular-value solution, then inspect
        # additional candidates if the first one is not an ellipsoid.
        n_candidates = min(
            6,
            len(right_singular_vectors),
        )

        for candidate_index in range(1, n_candidates + 1):
            coefficients = right_singular_vectors[
                -candidate_index
            ]

            for sign in (1.0, -1.0):
                candidate = sign * coefficients

                if self._is_valid_ellipsoid(candidate):
                    return candidate

        raise ValueError(
            "The algebraic fit did not produce a valid ellipsoid."
        )

    @staticmethod
    def _coefficient_matrices(
        coefficients: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Convert the coefficient vector into Q, q, and d.
        """

        (
            A,
            B,
            C,
            D,
            E,
            F,
            G,
            H,
            I,
            J,
        ) = coefficients

        quadratic_matrix = np.array([
            [A, D, E],
            [D, B, F],
            [E, F, C],
        ])

        linear_vector = np.array([
            G,
            H,
            I,
        ])

        constant = float(J)

        return (
            quadratic_matrix,
            linear_vector,
            constant,
        )

    @classmethod
    def _is_valid_ellipsoid(
        cls,
        coefficients: np.ndarray,
    ) -> bool:
        """
        Check whether the quadric defines a real, bounded ellipsoid.
        """

        try:
            (
                quadratic_matrix,
                linear_vector,
                constant,
            ) = cls._coefficient_matrices(coefficients)

            eigenvalues = np.linalg.eigvalsh(
                quadratic_matrix
            )

            if not np.all(eigenvalues > 0.0):
                return False

            center = -np.linalg.solve(
                quadratic_matrix,
                linear_vector,
            )

            translated_constant = (
                constant
                + 2.0 * linear_vector @ center
                + center @ quadratic_matrix @ center
            )

            # We need:
            #
            # (x - center).T Q (x - center)
            #     = -translated_constant > 0
            return bool(
                np.isfinite(translated_constant)
                and translated_constant < 0.0
            )

        except np.linalg.LinAlgError:
            return False

    @classmethod
    def _quadric_to_parameters(
        cls,
        coefficients: np.ndarray,
    ) -> np.ndarray:
        """
        Convert algebraic quadric coefficients to geometric parameters.
        """

        (
            quadratic_matrix,
            linear_vector,
            constant,
        ) = cls._coefficient_matrices(coefficients)

        center = -np.linalg.solve(
            quadratic_matrix,
            linear_vector,
        )

        translated_constant = (
            constant
            + 2.0 * linear_vector @ center
            + center @ quadratic_matrix @ center
        )

        scale = -translated_constant

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "The fitted quadric does not define a real ellipsoid."
            )

        normalized_matrix = (
            quadratic_matrix / scale
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            normalized_matrix
        )

        if not np.all(eigenvalues > 0.0):
            raise ValueError(
                "The fitted quadric is not positive definite."
            )

        semiaxes = 1.0 / np.sqrt(eigenvalues)

        # Sort axes so that:
        #
        #     a >= b >= c
        #
        # The eigenvector columns must be reordered in the same way.
        order = np.argsort(semiaxes)[::-1]

        semiaxes = semiaxes[order]
        rotation_matrix = eigenvectors[:, order]

        # Eigenvector matrices can have determinant -1. Euler-angle
        # conversion requires a proper rotation with determinant +1.
        if np.linalg.det(rotation_matrix) < 0.0:
            rotation_matrix[:, -1] *= -1.0

        a, b, c = semiaxes

        if b <= 0.0 or c <= 0.0:
            raise ValueError(
                "The fitted semiaxes must be positive."
            )

        r_1 = b / a
        r_2 = c / a

        alpha, beta, gamma = (
            Rotation.from_matrix(rotation_matrix)
            .as_euler("ZYX", degrees=False)
        )

        return np.array([
            center[0],
            center[1],
            center[2],
            a,
            r_1,
            r_2,
            alpha,
            beta,
            gamma,
        ])


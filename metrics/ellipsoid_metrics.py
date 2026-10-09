# metrics/ellipsoid_metrics.py

import numpy as np

from core.sweep_result import SweepResult


def _get_fitter_params(
    result: SweepResult,
    fitter_label: str,
) -> np.ndarray:
    """
    Return all fitted parameters for one fitter.

    Returns
    -------
    np.ndarray
        Array with shape (n_iterations, 9).
    """

    try:
        fitter_index = result.labels.index(fitter_label)
    except ValueError as error:
        raise ValueError(
            f"Unknown fitter label {fitter_label!r}. "
            f"Available labels: {result.labels}"
        ) from error

    return result.fitting_params[:, fitter_index, :]


def _valid_fits(
    result: SweepResult,
    fitter_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return valid fitted parameters and their corresponding ground truth.

    Returns
    -------
    fitted_params:
        Shape (n_valid_iterations, 9).

    ground_truth:
        Shape (n_valid_iterations, 9).
    """

    fitted_params = _get_fitter_params(
        result,
        fitter_label,
    )

    valid = np.all(np.isfinite(fitted_params), axis=1)

    return (
        fitted_params[valid],
        result.shape_gt[valid],
    )


def compute_Ec(
    result: SweepResult,
    fitter_label: str,
) -> np.ndarray:
    """
    Compute the center-offset error for every valid iteration.

    Ec = ||c_hat - c_true||_2

    Returns
    -------
    np.ndarray
        Center errors with shape (n_valid_iterations,).
    """

    fitted_params, ground_truth = _valid_fits(
        result,
        fitter_label,
    )

    center_difference = (
        fitted_params[:, :3] - ground_truth[:, :3]
    )

    return np.linalg.norm(
        center_difference,
        axis=1,
    )


def _rotation_matrix_zyx(
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """
    Construct a rotation matrix using ZYX Euler angles.

    R = Rz(alpha) @ Ry(beta) @ Rx(gamma)
    """

    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cos_beta = np.cos(beta)
    sin_beta = np.sin(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    rotation_z = np.array([
        [cos_alpha, -sin_alpha, 0.0],
        [sin_alpha, cos_alpha, 0.0],
        [0.0, 0.0, 1.0],
    ])

    rotation_y = np.array([
        [cos_beta, 0.0, sin_beta],
        [0.0, 1.0, 0.0],
        [-sin_beta, 0.0, cos_beta],
    ])

    rotation_x = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_gamma, -sin_gamma],
        [0.0, sin_gamma, cos_gamma],
    ])

    return rotation_z @ rotation_y @ rotation_x


def _ellipsoid_shape_matrix(
    params: np.ndarray,
) -> np.ndarray:
    """
    Construct the 3 x 3 ellipsoid shape matrix.

    Parameters
    ----------
    params:
        Ellipsoid parameters
        (cx, cy, cz, a, r1, r2, alpha, beta, gamma),

        where

        r1 = a / b
        r2 = a / c

        and the angles use the ZYX convention.
    """

    (
        _,
        _,
        _,
        a,
        r1,
        r2,
        alpha,
        beta,
        gamma,
    ) = params

    b = a / r1
    c = a / r2

    rotation = _rotation_matrix_zyx(
        alpha,
        beta,
        gamma,
    )

    return (
        rotation
        @ np.diag([a, b, c])
        @ rotation.T
    )


def compute_Ea(
    result: SweepResult,
    fitter_label: str,
) -> np.ndarray:
    """
    Compute the ellipsoid-shape deviation for every valid iteration.

    Ea = sigma_max(inv(A_hat) A_true)
         / sigma_min(inv(A_hat) A_true) - 1

    Returns
    -------
    np.ndarray
        Shape errors with shape (n_valid_iterations,).
        Invalid or singular estimates are represented by NaN.
    """

    fitted_params, ground_truth = _valid_fits(
        result,
        fitter_label,
    )

    errors = np.full(
        len(fitted_params),
        np.nan,
        dtype=float,
    )

    for index, (fitted, true) in enumerate(
        zip(fitted_params, ground_truth)
    ):
        try:
            fitted_matrix = _ellipsoid_shape_matrix(fitted)
            true_matrix = _ellipsoid_shape_matrix(true)

            # More stable than explicitly computing inv(A_hat).
            comparison = np.linalg.solve(
                fitted_matrix,
                true_matrix,
            )

            singular_values = np.linalg.svd(
                comparison,
                compute_uv=False,
            )

            smallest = singular_values.min()

            if smallest <= 0 or not np.isfinite(smallest):
                continue

            errors[index] = (
                singular_values.max() / smallest - 1.0
            )

        except (np.linalg.LinAlgError, ValueError):
            # Keep this iteration as NaN.
            continue

    return errors


def compute_Evolume(
    result: SweepResult,
    fitter_label: str,
    *,
    absolute: bool = True,
) -> np.ndarray:
    """
    Compute the relative ellipsoid-volume error.

    Signed error:
        Evolume = (a_hat * b_hat * c_hat)
                  / (a_true * b_true * c_true) - 1

    Absolute error:
        Evolume = abs(
            (a_hat * b_hat * c_hat)
            / (a_true * b_true * c_true) - 1
        )

    The constant factor 4*pi/3 cancels in the volume ratio.

    Parameters
    ----------
    absolute:
        If True, return the absolute relative error.
        If False, preserve the sign of the volume error.

    Returns
    -------
    np.ndarray
        Volume errors with shape (n_valid_iterations,).
        Invalid estimates are represented by NaN.
    """

    fitted_params, ground_truth = _valid_fits(
        result,
        fitter_label,
    )

    fitted_a = fitted_params[:, 3]
    fitted_r1 = fitted_params[:, 4]
    fitted_r2 = fitted_params[:, 5]

    true_a = ground_truth[:, 3]
    true_r1 = ground_truth[:, 4]
    true_r2 = ground_truth[:, 5]

    with np.errstate(divide="ignore", invalid="ignore"):
        fitted_b = fitted_a / fitted_r1
        fitted_c = fitted_a / fitted_r2

        true_b = true_a / true_r1
        true_c = true_a / true_r2

        fitted_volume_factor = (
            fitted_a * fitted_b * fitted_c
        )

        true_volume_factor = (
            true_a * true_b * true_c
        )

        volume_error = (
            fitted_volume_factor / true_volume_factor - 1.0
        )

    volume_error[~np.isfinite(volume_error)] = np.nan

    if absolute:
        volume_error = np.abs(volume_error)

    return volume_error


def compute_ellipsoid_metrics(
    result: SweepResult,
) -> dict[str, np.ndarray]:
    """
    Compute all ellipsoid metrics for all fitters.

    Each output array has shape:
        (n_iterations, n_fitters)

    Failed fits are represented by NaN.
    """

    n_iterations = result.fitting_params.shape[0]
    n_fitters = result.fitting_params.shape[1]

    center_errors = np.full(
        (n_iterations, n_fitters),
        np.nan,
    )
    shape_errors = np.full(
        (n_iterations, n_fitters),
        np.nan,
    )
    volume_errors = np.full(
        (n_iterations, n_fitters),
        np.nan,
    )

    for fitter_index, fitter_label in enumerate(result.labels):
        fitted_params = result.fitting_params[
            :,
            fitter_index,
            :,
        ]

        valid = np.all(
            np.isfinite(fitted_params),
            axis=1,
        )

        center_errors[
            valid,
            fitter_index,
        ] = compute_Ec(
            result,
            fitter_label,
        )

        shape_errors[
            valid,
            fitter_index,
        ] = compute_Ea(
            result,
            fitter_label,
        )

        volume_errors[
            valid,
            fitter_index,
        ] = compute_Evolume(
            result,
            fitter_label,
            absolute=True,
        )

    return {
        "Ec": center_errors,
        "Ea": shape_errors,
        "Evolume": volume_errors,
    }
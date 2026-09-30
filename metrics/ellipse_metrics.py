# metrics/ellipse_metrics.py

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
        Array with shape (n_iterations, 5).
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
        Shape (n_valid_iterations, 5).

    ground_truth:
        Shape (n_valid_iterations, 5).
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
        fitted_params[:, :2] - ground_truth[:, :2]
    )

    return np.linalg.norm(
        center_difference,
        axis=1,
    )


def _ellipse_shape_matrix(
    params: np.ndarray,
) -> np.ndarray:
    """
    Construct the 2 x 2 ellipse shape matrix.

    Parameters
    ----------
    params:
        Ellipse parameters (cx, cy, a, b, phi).
    """

    _, _, a, b, phi = params

    rotation = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)],
    ])

    return (
        rotation
        @ np.diag([a, b])
        @ rotation.T
    )


def compute_Ea(
    result: SweepResult,
    fitter_label: str,
) -> np.ndarray:
    """
    Compute the ellipse-shape deviation for every valid iteration.

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
        fitted_matrix = _ellipse_shape_matrix(fitted)
        true_matrix = _ellipse_shape_matrix(true)

        try:
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

        except np.linalg.LinAlgError:
            # Keep this iteration as NaN.
            continue

    return errors


def compute_Earea(
    result: SweepResult,
    fitter_label: str,
    *,
    absolute: bool = True,
) -> np.ndarray:
    """
    Compute the relative ellipse-area error.

    Signed error:
        Earea = (a_hat * b_hat) / (a_true * b_true) - 1

    Absolute error:
        Earea = abs(
            (a_hat * b_hat) / (a_true * b_true) - 1
        )

    Parameters
    ----------
    absolute:
        If True, return the absolute relative error.
        If False, preserve the sign of the area error.

    Returns
    -------
    np.ndarray
        Area errors with shape (n_valid_iterations,).
    """

    fitted_params, ground_truth = _valid_fits(
        result,
        fitter_label,
    )

    fitted_area_factor = (
        fitted_params[:, 2] * fitted_params[:, 3]
    )

    true_area_factor = (
        ground_truth[:, 2] * ground_truth[:, 3]
    )

    area_error = (
        fitted_area_factor / true_area_factor - 1.0
    )

    if absolute:
        area_error = np.abs(area_error)

    return area_error


def compute_ellipse_metrics(
    result: SweepResult,
) -> dict[str, np.ndarray]:
    """
    Compute all ellipse metrics for all fitters.

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
    area_errors = np.full(
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

        area_errors[
            valid,
            fitter_index,
        ] = compute_Earea(
            result,
            fitter_label,
            absolute=True,
        )

    return {
        "Ec": center_errors,
        "Ea": shape_errors,
        "Earea": area_errors,
    }
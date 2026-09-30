"""
This file contains useful utils around ellipses.

We parametrize an ellipse via params = (a,b,x_c,y_c,theta), where a and b are the axis lengths, (x_c,y_c) is the
center point and theta the rotation angle


This gives rise to a parametrization phi(t), t \in [0,2pi]
"""

import numpy as np
from scipy.integrate import quad
from typing import Union


def convert_ellipse_params(params: np.ndarray) -> np.ndarray:
    """
    Convert ellipse parameters (x_c, y_c, a, b, theta)
    to (x_c, y_c, a, r, theta) such that:
        a >= b
        r = b/a ∈ [0, 1]
    If b > a, swap axes and rotate theta by +pi/2.

    Parameters
    ----------
    params : np.ndarray shape (5,)
        [x_c, y_c, a, b, theta]

    Returns
    -------
    new_params : np.ndarray shape (5,)
        [x_c, y_c, a, r, theta]
    """

    x_c, y_c, a, b, theta = params

    # if minor axis > major axis → flip convention
    if b > a:
        a, b = b, a
        theta += np.pi / 2.0

    # Normalize angle to [-pi/2, pi/2]
    theta = (theta + np.pi / 2) % np.pi - np.pi / 2

    # axis ratio r ∈ [0,1]
    r = b / a

    return np.array([x_c, y_c, a, r, theta])

def get_ellipse_params_(homogenous_coords: np.ndarray) -> np.ndarray:
    A, B, C, D, E, F = homogenous_coords

    if B**2 - 4*A*C >= 0:
        raise ValueError("Not an ellipse: discriminant >= 0")

    # Center
    denom = B**2 - 4*A*C
    x_c = (2*C*D - B*E) / denom
    y_c = (2*A*E - B*D) / denom

    # Value at center
    F_bar = (
        A*x_c**2 + B*x_c*y_c + C*y_c**2 +
        D*x_c + E*y_c + F
    )

    # Ensure interior is negative
    if F_bar > 0:
        A, B, C, D, E, F = -A, -B, -C, -D, -E, -F
        F_bar = (
            A*x_c**2 + B*x_c*y_c + C*y_c**2 +
            D*x_c + E*y_c + F
        )

    # Rotation
    theta = 0.5 * np.arctan2(B, A - C)

    # Normalize angle
    theta = ((theta + np.pi/4) % (np.pi/2)) - np.pi/4

    # Rotated coefficients
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A_rot = A*cos_t**2 + B*cos_t*sin_t + C*sin_t**2
    C_rot = A*sin_t**2 - B*cos_t*sin_t + C*cos_t**2

    if A_rot <= 0 or C_rot <= 0 or abs(F_bar) < 1e-12:
        raise ValueError("Degenerate ellipse.")

    # Raw axes
    a_raw = np.sqrt(-F_bar / A_rot)
    b_raw = np.sqrt(-F_bar / C_rot)

    # ✅ Fix orientation & enforce a >= b
    if b_raw > a_raw:
        a_raw, b_raw = b_raw, a_raw
        theta += np.pi/2

    # Normalize θ back to [-π/4, π/4]
    theta = ((theta + np.pi/4) % (np.pi/2)) - np.pi/4

    # Axis ratio r = b/a ∈ [0,1]
    r = b_raw / a_raw

    return np.array([x_c, y_c, a_raw, r, theta])

def get_ellipse_params(homogenous_coords: np.ndarray) -> np.ndarray:
    """
    This function converts an ellipse in homogenous coordinates,
    given by A, B, C, D, E, F, the coefficients of the general conic equation
                         Ax² + Bxy + Cy² + Dx + Ey + F = 0
    :param homogenous_coords: A,B,C,D,E,F
    :return: (x_c,y_c,a,b,theta)
    """
    A, B, C, D, E, F = homogenous_coords
    if B ** 2 - 4 * A * C >= 0:
        raise ValueError("The conic does not represent an ellipse (discriminant >= 0).")

        # Compute the center (x0, y0)
    denom = B ** 2 - 4 * A * C
    x_c = (2 * C * D - B * E) / denom
    y_c = (2 * A * E - B * D) / denom

    # Translate F to the center
    F_bar = (
            A * x_c ** 2 + B * x_c * y_c + C * y_c ** 2 + D * x_c + E * y_c + F
    )

    # Normalize sign so F_bar < 0
    if F_bar > 0:
        A, B, C, D, E, F = -A, -B, -C, -D, -E, -F
        F_bar = A * x_c ** 2 + B * x_c * y_c + C * y_c ** 2 + D * x_c + E * y_c + F

    theta = 0.5 * np.arctan2(B, A - C)

    # Normalize angle to [-pi/2, pi/2]
    theta = (theta + np.pi / 2) % np.pi - np.pi/2


    # Compute rotated coefficients
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A_rot = A * cos_t ** 2 + B * cos_t * sin_t + C * sin_t ** 2
    C_rot = A * sin_t ** 2 - B * cos_t * sin_t + C * cos_t ** 2

    if abs(F_bar) < 1e-12 or A_rot <= 0 or C_rot <= 0:
        raise ValueError("Degenerate ellipse or invalid coefficients.")

    # Compute semi-axes
    a = np.sqrt(-F_bar / A_rot)
    b = np.sqrt(-F_bar / C_rot)

    # now we enforce that a > b (and rotate if necessary)
    if b > a:
        a,b = b,a
        theta += np.pi/2
        # Normalize angle to [-pi/2, pi/2]
        theta = (theta + np.pi / 2) % np.pi - np.pi / 2

    return np.array([x_c, y_c, a, b,  theta])


def get_homogenous_coords_from_ellipse(params: np.ndarray) -> np.ndarray:
    """
    Converts ellipse parameters (a,b,x_c,y_c,theta) to conic coefficients.

    :param params: ellipse parameters (a,b,x_c,y_c,theta)

    Returns:
        A, B, C, D, E, F: coefficients of the general conic equation
                         Ax² + Bxy + Cy² + Dx + Ey + F = 0
    """

    a, b, xc, yc, theta = params

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Rotation matrix components
    a2 = a ** 2
    b2 = b ** 2

    # Coefficients of the rotated ellipse at origin
    A = (cos_t ** 2 / a2) + (sin_t ** 2 / b2)
    B = 2 * cos_t * sin_t * (1 / a2 - 1 / b2)
    C = (sin_t ** 2 / a2) + (cos_t ** 2 / b2)

    # Linear terms due to translation
    D = -2 * A * xc - B * yc
    E = -2 * C * yc - B * xc

    # Constant term
    F = A * xc ** 2 + B * xc * yc + C * yc ** 2 - 1

    return np.array([A, B, C, D, E, F])


def compute_visible_part(params: np.ndarray, laser: np.ndarray) -> (float, float):
    """
    Computes the visible part of the ellipse, given a scanner position
    :param params: np.array([a, b, x_c, y_c, theta])
    :param laser: np.array([x, y]) position of the scanner
    :return: t1, t2 (angles in radians) of the visible (tangent) points on the ellipse
    """
    a, b, xc, yc, theta = params
    x0, y0 = laser

    # Step 1: Transform laser point into canonical frame
    dx = x0 - xc
    dy = y0 - yc

    cos_t = np.cos(-theta)
    sin_t = np.sin(-theta)

    x0_ = cos_t * dx - sin_t * dy
    y0_ = sin_t * dx + cos_t * dy

    # Step 2: Check if laser is outside the ellipse
    value = (x0_ ** 2) / a ** 2 + (y0_ ** 2) / b ** 2
    if value <= 1.0:
        raise ValueError("Laser is inside or on the ellipse — no visible tangent points.")

    # Step 3: Solve the polar line equation:
    # (x0_/a) * cos(t) + (y0_/b) * sin(t) = 1
    A = x0_ / a
    B = y0_ / b

    # Let:
    # R = sqrt(A^2 + B^2)
    # φ = atan2(B, A)
    R = np.hypot(A, B)
    phi = np.arctan2(B, A)

    # Now solve: R * cos(t - phi) = 1 => cos(t - phi) = 1 / R
    if np.abs(1 / R) > 1:
        raise ValueError("Numerical issue: 1/R > 1, no real solution.")

    delta = np.arccos(1 / R)

    t1_canonical = phi - delta
    t2_canonical = phi + delta

    # Wrap angles to [0, 2pi]
    t1_canonical = t1_canonical % (2 * np.pi)
    t2_canonical = t2_canonical % (2 * np.pi)

    return t1_canonical, t2_canonical


def is_visible(t_vals: float | np.ndarray, params: np.array, laser: np.ndarray) -> Union[bool, np.ndarray]:
    """
    :param t_vals:
    :param params:
    :param laser:
    :return:
    """
    t_1, t_2 = compute_visible_part(params=params, laser=laser)

    assert 0 <= t_1 <= 2 * np.pi
    assert 0 <= t_2 <= 2 * np.pi

    t_arr = np.asarray(t_vals)
    result = np.array(t_arr, dtype=bool)

    # we have to distinguish the cases
    if t_1 < t_2:

        result = (t_arr >= t_1) & (t_arr <= t_2)

    else:
        result = ((t_arr >= t_1) & (t_arr <= 2 * np.pi)) | ((t_arr >= 0) & (t_arr <= t_2))

    return result

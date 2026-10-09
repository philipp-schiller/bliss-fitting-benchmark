import numpy as np


def orient_south_pole_towards_scanner(
    theta: np.ndarray,
    scanner_position: np.ndarray,
) -> np.ndarray:
    """
    Choose an equivalent ellipsoid orientation such that the south pole
    is closer to the scanner than the north pole.

    Parameters
    ----------
    theta:
        [x_c, y_c, z_c, a, r1, r2, alpha, beta, gamma]

    scanner_position:
        Scanner coordinates [x, y, z].

    Returns
    -------
    np.ndarray
        Ellipsoid parameters with a consistently oriented south pole.
    """

    theta = np.asarray(theta, dtype=float).copy()
    scanner_position = np.asarray(scanner_position, dtype=float)

    if theta.shape != (9,):
        raise ValueError(
            f"theta must have shape (9,), got {theta.shape}"
        )

    if scanner_position.shape != (3,):
        raise ValueError(
            "scanner_position must have shape (3,), "
            f"got {scanner_position.shape}"
        )

    center = theta[:3]
    alpha, beta, gamma = theta[6:9]

    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    rotation = np.array([
        [
            ca * cb,
            ca * sb * sg - sa * cg,
            ca * sb * cg + sa * sg,
        ],
        [
            sa * cb,
            sa * sb * sg + ca * cg,
            sa * sb * cg - ca * sg,
        ],
        [
            -sb,
            cb * sg,
            cb * cg,
        ],
    ])

    local_z = rotation[:, 2]
    scanner_direction = scanner_position - center

    # The south pole points along -local_z. If local_z points towards
    # the scanner, north is currently closer, so exchange both poles.
    if np.dot(local_z, scanner_direction) > 0:
        rotation[:, 0] = -rotation[:, 0]
        rotation[:, 2] = -rotation[:, 2]

        beta = np.arctan2(
            -rotation[2, 0],
            np.hypot(rotation[0, 0], rotation[1, 0]),
        )

        alpha = np.arctan2(
            rotation[1, 0],
            rotation[0, 0],
        )

        gamma = np.arctan2(
            rotation[2, 1],
            rotation[2, 2],
        )

        theta[6:9] = [alpha, beta, gamma]

    return theta
"""
geometric_ellipse_fitter.py
============================

Ellipse fitting that minimizes true GEOMETRIC (orthogonal) distance from
each point to the ellipse boundary, as opposed to algebraic residual.

Background
----------
Algebraic fits (e.g. Fitzgibbon, Pilu & Fisher, "Direct Least Square
Fitting of Ellipses", IEEE TPAMI 21(5), 1999) minimize the residual of the
conic equation A x^2 + Bxy + Cy^2 + Dx + Ey + F = 0 directly. This is fast,
non-iterative (a generalized eigenvalue problem), and numerically robust,
but it is a biased estimator of the true shape, particularly for partial
arcs or elongated ellipses -- points are effectively weighted unevenly
depending on the local slope of the conic.

Geometric / orthogonal distance fitting instead minimizes

    sum_i  min_{q on ellipse} || p_i - q ||^2

which is the maximum-likelihood fit under isotropic Gaussian noise on the
point positions. This is the classic approach of:

  - Gander, Golub & Strebel, "Least-Squares Fitting of Circles and
    Ellipses", BIT Numerical Mathematics 34, 1994.
  - Ahn, Rauh & Warnecke, "Least-squares orthogonal distances fitting of
    circle, sphere, ellipse, hyperbola, and parabola", Pattern Recognition
    34(12), 2001.

Method
------
1. Initialize with the algebraic direct-least-squares ellipse fit
   (Halir & Flusser's numerically stable formulation of Fitzgibbon's
   method), which always returns a genuine ellipse.
2. Refine with nonlinear least squares over (cx, cy, a, b, theta). For each
   trial set of parameters, the residual for point i is the vector from
   p_i to its closest point on that ellipse. The closest point is found
   exactly (not approximated) via Eberly's robust root-finding algorithm
   ("Distance from a Point to an Ellipse, an Ellipsoid, or a
   Hyperellipsoid", Geometric Tools, 2013).

   Differentiating the objective while treating the (re-solved) closest
   point as locally constant is valid by the envelope theorem, since the
   closest point is itself a minimizer of the inner distance problem --
   this is the standard formulation used by Ahn et al. and others.

"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from methods.ellipse_fitters.base import BaseEllipseFitter


# --------------------------------------------------------------------------
# Exact closest point on an axis-aligned ellipse (Eberly's robust method)
# --------------------------------------------------------------------------

def closest_point_on_ellipse(x: float, y: float, a: float, b: float, eps: float = 1e-12):
    """Closest point on the axis-aligned ellipse (x/a)^2+(y/b)^2=1, centered
    at the origin, to the point (x, y). Exact, robust for all cases
    (interior/exterior points, points on an axis, circles)."""
    sx = 1.0 if x >= 0 else -1.0
    sy = 1.0 if y >= 0 else -1.0
    x, y = abs(x), abs(y)

    swapped = False
    if a < b:
        a, b, x, y = b, a, y, x
        swapped = True

    if y > eps:
        if x > eps:
            def G(t):
                return (a * x / (t + a * a)) ** 2 + (b * y / (t + b * b)) ** 2 - 1.0

            t_lo = -b * b + eps
            t_hi = max(np.hypot(a * x, b * y) - b * b, t_lo + 1.0)
            while G(t_hi) > 0:
                t_hi *= 2.0
            for _ in range(100):
                t_mid = 0.5 * (t_lo + t_hi)
                if G(t_mid) > 0:
                    t_lo = t_mid
                else:
                    t_hi = t_mid
            t = 0.5 * (t_lo + t_hi)
            ex = a * a * x / (t + a * a)
            ey = b * b * y / (t + b * b)
        else:
            ex, ey = 0.0, b
    else:
        numer0 = a * x
        denom0 = a * a - b * b
        if denom0 > eps and numer0 < denom0:
            xde = numer0 / denom0
            ex = a * xde
            ey = b * np.sqrt(max(0.0, 1.0 - xde * xde))
        else:
            ex, ey = a, 0.0

    if swapped:
        ex, ey = ey, ex
    return sx * ex, sy * ey


def _geometric_residuals(params, points):
    cx, cy, a, b, theta = params
    a, b = abs(a), abs(b)
    ct, st = np.cos(theta), np.sin(theta)
    dx = points[:, 0] - cx
    dy = points[:, 1] - cy
    lx = ct * dx + st * dy
    ly = -st * dx + ct * dy

    res = np.empty((len(points), 2))
    for i in range(len(points)):
        ex, ey = closest_point_on_ellipse(lx[i], ly[i], a, b)
        res[i, 0] = lx[i] - ex
        res[i, 1] = ly[i] - ey
    return res.ravel()


# --------------------------------------------------------------------------
# Algebraic initializer: Halir & Flusser's stable direct ellipse fit
# --------------------------------------------------------------------------

def _algebraic_ellipse_fit(points: np.ndarray):
    """Halir & Flusser, "Numerically stable direct least squares fitting of
    ellipses" (1998), a numerically robust reformulation of Fitzgibbon,
    Pilu & Fisher's direct method (1999). Always returns a genuine ellipse.
    Returns conic coefficients (A, B, C, D, E, F)."""
    x = points[:, 0]
    y = points[:, 1]
    # centering/scaling improves conditioning; undo it on the coefficients after
    mx, my = x.mean(), y.mean()
    scale = np.sqrt(np.mean((x - mx) ** 2 + (y - my) ** 2)) or 1.0
    xs = (x - mx) / scale
    ys = (y - my) / scale

    D1 = np.column_stack([xs ** 2, xs * ys, ys ** 2])
    D2 = np.column_stack([xs, ys, np.ones_like(xs)])
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.array([M[2, :] / 2.0, -M[1, :], M[0, :] / 2.0])
    eigval, eigvec = np.linalg.eig(M)
    cond = 4 * eigvec[0, :] * eigvec[2, :] - eigvec[1, :] ** 2
    valid = np.where(cond > 0)[0]
    if len(valid) == 0:
        raise RuntimeError("algebraic ellipse fit failed to find a valid ellipse solution")
    a1 = np.real(eigvec[:, valid[0]])
    a2 = T @ a1
    A, B, C = a1
    D, E, F = a2

    # undo the centering/scale normalization: substitute x=(x'-mx)/s, y=(y'-my)/s
    A2 = A / scale ** 2
    B2 = B / scale ** 2
    C2 = C / scale ** 2
    D2c = (D - 2 * A * mx / scale - B * my / scale) / scale
    E2 = (E - 2 * C * my / scale - B * mx / scale) / scale
    F2 = (F + A * mx ** 2 / scale ** 2 + C * my ** 2 / scale ** 2 + B * mx * my / scale ** 2
          - D * mx / scale - E * my / scale)
    return A2, B2, C2, D2c, E2, F2


def _conic_to_params(A, B, C, D, E, F):
    """Convert conic coefficients to (cx, cy, a, b, theta) via eigendecomposition."""
    Q = np.array([[A, B / 2.0], [B / 2.0, C]])
    center = np.linalg.solve(np.array([[2 * A, B], [B, 2 * C]]), np.array([-D, -E]))
    cx, cy = center
    F0 = A * cx * cx + B * cx * cy + C * cy * cy + D * cx + E * cy + F
    eigvals, eigvecs = np.linalg.eigh(Q)
    axes = np.sqrt(np.clip(-F0 / eigvals, 1e-12, None))
    theta = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    return cx, cy, axes[0], axes[1], theta


# --------------------------------------------------------------------------
# Public fitter class
# --------------------------------------------------------------------------

class GeometricEllipseFitter(BaseEllipseFitter):
    """
    Ellipse fitting by minimizing true geometric (orthogonal) distance to
    the ellipse boundary -- the maximum-likelihood fit under isotropic
    Gaussian point noise. See module docstring for references.

    Not robust to outliers by design (every point is assumed to genuinely
    lie near the ellipse). For contaminated data, either pre-filter
    outliers or use BayesianEllipseFitter instead.
    """

    def __init__(self, max_nfev: int = 200, ftol: float = 1e-12, xtol: float = 1e-12):
        self.max_nfev = max_nfev
        self.ftol = ftol
        self.xtol = xtol
        self.result_ = None  # scipy OptimizeResult from the last fit() call

    def fit(self, points: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        :param points: (N, 2) array of x, y coordinates to fit
        :return: np.ndarray [a, b, c_x, c_y, theta]
        """
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must be an (N, 2) array")

        A, B, C, D, E, F = _algebraic_ellipse_fit(points)
        cx0, cy0, a0, b0, theta0 = _conic_to_params(A, B, C, D, E, F)

        x0 = np.array([cx0, cy0, a0, b0, theta0])
        res = least_squares(
            _geometric_residuals, x0, args=(points,),
            method="lm", max_nfev=self.max_nfev, ftol=self.ftol, xtol=self.xtol,
        )
        self.result_ = res

        cx, cy, a, b, theta = res.x
        a, b = abs(a), abs(b)
        return np.array([cx, cy, a, b, theta])
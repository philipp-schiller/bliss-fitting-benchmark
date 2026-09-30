from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass

from methods.ellipse_fitters.base import BaseEllipseFitter

import numpy as np

# --------------------------------------------------------------------------
# Data normalization  (data_normalize_input.m)
# --------------------------------------------------------------------------

def normalize_points(x: np.ndarray):
    """Zero-mean, unit-RMS-distance normalization (matches data_normalize_input.m)."""
    n = x.shape[0]
    xd = x.mean(axis=0)
    xc = x - xd
    xscale = np.sqrt(np.sum(xc ** 2) / n)
    X = xc / xscale
    return X, xd, xscale


def rdos_scores(data: np.ndarray, K: int | None = None, h: float = 0.1) -> np.ndarray:
    """Generalized-dimension re-implementation of knn_cpp.cpp's RDOS score."""
    n, d = data.shape
    if K is None:
        K = min(100, n - 1)
    K = max(1, K)

    tree = cKDTree(data)
    _, idx = tree.query(data, k=K + 1)
    if idx.ndim == 1:
        idx = idx[:, None]
    idx = idx[:, 1:]  # drop self-match

    # reverse neighbors: who lists i as one of their K nearest neighbors
    reverse = [[] for _ in range(n)]
    for m in range(n):
        for j in idx[m]:
            reverse[j].append(m)

    # extended neighborhood per point: reverse-KNN U KNN U KNN-of-KNN
    neighborhoods = []
    for i in range(n):
        s = set(reverse[i])
        s.update(idx[i].tolist())
        for l in idx[i]:
            s.update(idx[l].tolist())
        neighborhoods.append(sorted(s))

    const = 1.0 / (2.0 * np.pi * h * h) ** (d / 2.0)

    # local kernel density estimate at each point over its own neighborhood
    point_pdf = np.empty(n)
    for i in range(n):
        nb = neighborhoods[i]
        pts = data[nb]
        sq = np.sum((pts - data[i]) ** 2, axis=1)
        s = np.exp(-0.5 * sq / (h * h)).sum() + 1.0
        point_pdf[i] = const * s / (len(nb) + 1)

    # RDOS: ratio of the neighborhood's average density to the point's own density
    scores = np.empty(n)
    for i in range(n):
        nb = neighborhoods[i]
        if len(nb) <= 1:
            scores[i] = 1.0
            continue
        ave = point_pdf[nb].sum() / (len(nb) - 1)
        scores[i] = ave / point_pdf[i]
    return scores


def robust_init(data: np.ndarray, rdos_threshold: float = 2.0):
    """Normalize data, score outlierness, and return a robust initial center."""
    X, xd, xscale = normalize_points(data)
    scores = rdos_scores(X)
    inlier_mask = scores <= rdos_threshold
    if inlier_mask.sum() < 3:  # degrade gracefully if the threshold is too strict
        inlier_mask = np.ones(len(data), dtype=bool)
    inliers = data[inlier_mask]
    init_center = inliers.mean(axis=0)
    outlierness = 1.0 - inlier_mask.sum() / len(data)
    return X, xd, xscale, init_center, outlierness, inlier_mask


# --------------------------------------------------------------------------
# EM core  (precompute.c -> vectorized numpy;  parameter_solving.m -> Python)
# --------------------------------------------------------------------------
#
# This is a straight, dimension-generic re-derivation of the CPD-style E/M
# steps in precompute.c and parameter_solving.m. It fits an affine map
# T(y) = B y + t that best explains X as noisy, affinely-transformed copies
# of Y (points on a unit circle/sphere), with an extra uniform "outlier"
# component competing for probability mass -- this is what makes it robust
# to outliers, and Bayesian in the sense of maximizing a posterior over the
# correspondence + parameters jointly (an EM / variational treatment).

def _e_step(X, T, sigma2, outlier, vol):
    """Vectorized equivalent of cpd_comp() in precompute.c."""
    n, d = X.shape
    m = T.shape[0]
    ksig = -2.0 * sigma2

    diff2 = ((X[:, None, :] - T[None, :, :]) ** 2).sum(axis=2)  # N x M
    P = np.exp(diff2 / ksig)
    sp_inlier = P.sum(axis=1)  # N

    outlier_tmp = (outlier * m * (-ksig * np.pi) ** (0.5 * d)) / ((1 - outlier) * vol)
    outlier_tmp2 = (vol * (1 - outlier)) / (outlier * m)

    PM = 1.0 / (1.0 + outlier_tmp2 * sp_inlier)
    sp_total = sp_inlier + outlier_tmp
    Pt1 = 1.0 - outlier_tmp / sp_total

    W = P / sp_total[:, None]
    P1 = W.sum(axis=0)          # M
    Px = W.T @ X                # M x D

    E = -np.log(sp_total).sum() + d * n * np.log(sigma2) / 2.0 - n * np.log(1.0 - outlier)
    return P1, Pt1, Px, PM, E


def solve_affine_em(X, Y, tol=1e-8, outliers=1e-6, vol=1.0, max_it=200):
    """Direct port of parameter_solving.m. Returns (B, t, n_iter)."""
    n, d = X.shape
    m = Y.shape[0]

    sigma2 = (m * np.trace(X.T @ X) + n * np.trace(Y.T @ Y)
              - 2 * (X.sum(axis=0) @ Y.sum(axis=0))) / (m * n * d)

    T = Y.copy()
    it = 0
    ntol = tol + 10.0
    L = 1.0

    while ntol > tol and it < max_it:
        L_old = L
        P1, Pt1, Px, PM, L = _e_step(X, T, sigma2, outliers, vol)
        ntol = abs((L - L_old) / L)

        Np = P1.sum()
        mu_x = X.T @ Pt1 / Np
        mu_y = Y.T @ P1 / Np

        B1 = Px.T @ Y - Np * np.outer(mu_x, mu_y)
        B2 = (Y * P1[:, None]).T @ Y - Np * np.outer(mu_y, mu_y)
        B = B1 @ np.linalg.inv(B2)

        t = mu_x - B @ mu_y

        sigma2 = abs((X ** 2 * Pt1[:, None]).sum() - Np * (mu_x @ mu_x)
                     - np.trace(B1 @ B.T)) / (Np * d)

        outliers = PM.sum() / (PM.sum() + Np)
        it += 1
        T = Y @ B.T + t

    return B, t, it


# --------------------------------------------------------------------------
# 2D-specific pieces (the part that was hard-wired to 3D in the original repo)
# --------------------------------------------------------------------------

def unit_circle_points(center: np.ndarray, m: int = 100) -> np.ndarray:
    """Points on a unit circle around `center` -- the 2D analogue of the
    unit-sphere `drawEllipsoid(ellParInit, 1, sqrt(inlier_num))` call used
    to seed Y in demo.m."""
    ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
    return center + np.column_stack([np.cos(ang), np.sin(ang)])


@dataclass
class EllipseFit:
    center: np.ndarray      # (2,)
    axes: np.ndarray        # (2,) semi-axis lengths [a, b]
    angle: float            # radians, rotation of the first (a) axis
    n_iter: int
    inlier_mask: np.ndarray
    outlierness: float

    def as_tuple(self):
        return (self.center[0], self.center[1], self.axes[0], self.axes[1], self.angle)


def ellipse_params_from_affine(init_center, R, t) -> tuple[np.ndarray, np.ndarray, float]:
    """2D re-derivation of ellipsoid_par.m: recover (center, semi-axes, angle)
    from the fitted affine map (R, t) applied to the unit circle."""
    center = t + R @ init_center
    B = R @ R.T
    eigvals, eigvecs = np.linalg.eigh(B)  # ascending order, matches MATLAB eig() convention
    eigvals = np.clip(eigvals, 0, None)
    axes = np.sqrt(eigvals)
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    return center, axes, angle


# --------------------------------------------------------------------------
# Top-level driver  (demo.m, specialized to 2D)
# --------------------------------------------------------------------------

def fit_ellipse(points: np.ndarray, m_init: int = 100, rdos_threshold: float = 2.0,
                 tol: float = 1e-8, max_it: int = 200) -> EllipseFit:
    """
    Fit an ellipse to 2D scattered points, robust to outliers, using the
    Bayesian/EM approach of BayFit (Zhao et al., TPAMI 2024).

    Parameters
    ----------
    points : (N, 2) array of x, y coordinates.
    m_init : number of points used to seed the unit-circle model Y.
    rdos_threshold : outlier score cutoff used only for robust initialization
        of the starting center (the EM itself handles outliers via its
        uniform-noise component regardless of this choice).
    tol, max_it : EM convergence tolerance / iteration cap.

    Returns
    -------
    EllipseFit(center, axes, angle, n_iter, inlier_mask, outlierness)
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (N, 2) array")

    # 1. Robust init: normalize, score outlierness, get an initial center
    X, xd, xscale, init_center, outlierness, inlier_mask = robust_init(points, rdos_threshold)

    # 2. Seed model points Y on a unit circle around the (unnormalized) init center,
    #    then normalize Y the same way X was normalized.
    Y_raw = unit_circle_points(init_center, m_init)
    Y, yd, yscale = normalize_points(Y_raw)

    # 3. Area of the (normalized) data's bounding box -- the 2D analogue of the
    #    bounding-box *volume* used in ellipsoid_fit_EM.m for the outlier term.
    L = X.max(axis=0) - X.min(axis=0)
    area = max(L[0] * L[1], 1e-12)

    # 4. EM in normalized space
    B, t, n_iter = solve_affine_em(X, Y, tol=tol, outliers=1e-6, vol=area, max_it=max_it)

    # 5. De-normalize the affine map (ellipsoid_fit_EM.m's Transform composition)
    s = xscale / yscale
    R = s * B
    t_full = xscale * t + xd - s * (B @ yd)

    # 6. Recover ellipse geometric parameters
    center, axes, angle = ellipse_params_from_affine(init_center, R, t_full)

    return EllipseFit(center=center, axes=axes, angle=angle, n_iter=n_iter,
                       inlier_mask=inlier_mask, outlierness=outlierness)


class BayesianEllipseFitterZhao(BaseEllipseFitter):
    """
    Robust, outlier-tolerant ellipse fitting via the Bayesian EM approach of
    Zhao et al., "A Bayesian Approach Toward Robust Multidimensional
    Ellipsoid-Specific Fitting" (IEEE TPAMI 2024), ported from the official
    MATLAB/C++ implementation (https://github.com/zikai1/BayFit).
    """

    def __init__(self, m_init: int = 100, rdos_threshold: float = 0.001,
                 tol: float = 1e-8, max_it: int = 200):
        """
        :param m_init: number of points used to seed the unit-circle model.
        :param rdos_threshold: outlier score cutoff used only to seed a
            robust initial center; the EM itself is robust to outliers
            regardless of this value.
        :param tol: EM convergence tolerance.
        :param max_it: max EM iterations.
        """
        self.m_init = m_init
        self.rdos_threshold = rdos_threshold
        self.tol = tol
        self.max_it = max_it
        self.last_fit_ = None  # holds the full EllipseFit result after fit()

    def fit(self, points: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        :param points: (N, 2) array of x, y coordinates to fit
        :param args:
        :param kwargs:
        :return: np.ndarray [a, b, c_x, c_y, theta] -- semi-axes a (major
            convention not enforced, see note below), center (c_x, c_y),
            and rotation angle theta in radians.
        """
        result = fit_ellipse(
            points,
            m_init=kwargs.pop("m_init", self.m_init),
            rdos_threshold=kwargs.pop("rdos_threshold", self.rdos_threshold),
            tol=kwargs.pop("tol", self.tol),
            max_it=kwargs.pop("max_it", self.max_it),
        )
        self.last_fit_ = result

        a, b = result.axes
        c_x, c_y = result.center
        theta = result.angle
        return np.array([c_x, c_y, a, b, theta])
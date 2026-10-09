"""
BayesianZhaoFitter
==================

Python port of the MATLAB/C++ code in https://github.com/zikai1/BayFit
(M. Zhao et al., "A Bayesian Approach Toward Robust and Multidimensional
Ellipsoid Fitting", TPAMI 2024; EM version: BMVC 2022).

Pipeline (mirrors ``demo.m``):
    1. ``outlier_det``          -> normalise points, RDOS outlier score (kNN + KDE)
    2. inliers (score <= 2)     -> initial centre = mean of inliers
    3. ``drawEllipsoid``        -> unit-sphere template (floor(sqrt(K)+1)^2 points)
    4. ``ellipsoid_fit_EM``     -> EM (CPD-style) estimation of the affine map
                                   sphere -> data, with a uniform outlier component
    5. ``ellipsoid_par``        -> centre, semi-axes, Euler angles from the affine map

Only numpy is required.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from methods.ellipsoid_fitters.base import BaseEllipsoidFitter

# Upper bound on the number of elements of the temporary (rows, M) arrays used
# in the E-step; only affects memory use, not the result.
_CHUNK_ELEMS = 2_000_000

# --------------------------------------------------------------------------- #
# data_normalize_input.m
# --------------------------------------------------------------------------- #
def _normalize(x: np.ndarray):
    """Zero mean, unit RMS radius. Returns (normalised, mean, scale)."""
    n = x.shape[0]
    mean = x.mean(axis=0)
    xc = x - mean
    scale = np.sqrt(np.sum(xc ** 2) / n)
    return xc / scale, mean, scale


# --------------------------------------------------------------------------- #
# outlier_det.m + knn_cpp.cpp (RDOS score)
# --------------------------------------------------------------------------- #
def _rdos_scores(data: np.ndarray, k_max: int = 100, h: float = 0.1) -> np.ndarray:
    """
    ``data`` is the already normalised Nx3 array. Indices are 0-based here.

    The original C++ (``score_comp``) contains a few quirks which are reproduced
    on purpose so that results match the MATLAB code:

    * The inner ``for(k...)`` loop re-uses (and exhausts) the outer ``k``, so the
      outer loop runs once: the neighbourhood of point i is
      {points having i in their kNN} U {1st NN l of i} U {kNN of l}.
    * Neighbour densities are read with an additional ``-1`` index shift
      (``point_pdf[idx-1]``). For idx == 0 this is an out-of-bounds read
      (undefined behaviour in C++); it is treated as density 0 here.
    """
    n = data.shape[0]
    K = min(k_max, n - 1)

    # --- kNN graph (outlier_det.m); zero distances (self / duplicates) -> 999999
    knn = np.empty((n, K), dtype=np.intp)
    for i in range(n):
        dis = np.sqrt(np.sum((data - data[i]) ** 2, axis=1))
        dis[dis == 0] = 999999.0
        knn[i] = np.argsort(dis, kind="stable")[:K]

    # --- reverse neighbours: rev(i) = {m : i in knn[m]}
    cols = knn.ravel()
    rows = np.repeat(np.arange(n), K)
    order = np.argsort(cols, kind="stable")
    rev_rows = rows[order]
    starts = np.searchsorted(cols[order], np.arange(n + 1))

    # --- neighbour sets and KDE density (kernel_comp)
    neigh = []
    pdf = np.empty(n)
    for i in range(n):
        first = knn[i, 0]
        idx = np.unique(np.concatenate((rev_rows[starts[i]:starts[i + 1]], [first], knn[first])))
        neigh.append(idx)
        d2 = np.sum((data[idx] - data[i]) ** 2, axis=1)
        total = np.sum(np.exp(-0.5 * d2 / (h * h))) + 1.0
        pdf[i] = total / ((idx.size + 1) * (2.0 * np.pi * h * h) ** 1.5)

    # --- relative density score
    shifted = np.concatenate(([0.0], pdf[:-1]))  # shifted[j] = pdf[j-1], shifted[0] = 0
    score = np.empty(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n):
            ave = np.float64(shifted[neigh[i]].sum()) / np.float64(neigh[i].size - 1)
            score[i] = ave / pdf[i]
    return score


# --------------------------------------------------------------------------- #
# drawEllipsoid.m (type = 1, unit sphere, no rotation)
# --------------------------------------------------------------------------- #
def _sphere_template(center: np.ndarray, num: float) -> np.ndarray:
    """Unit sphere sampled on a (phi, theta) grid, as in ``drawEllipsoid``."""
    n = int(np.floor(num + 1))  # MATLAB linspace floors a non-integer count
    theta = np.linspace(0.0, np.pi, n)
    phi = np.linspace(0.0, 2.0 * np.pi, n)
    sin_t = np.sin(theta)
    x = np.outer(np.cos(phi), sin_t)
    y = np.outer(np.sin(phi), sin_t)
    z = np.outer(np.ones(n), np.cos(theta))
    pts = np.stack([x.ravel("F"), y.ravel("F"), z.ravel("F")], axis=1)
    return pts + center


# --------------------------------------------------------------------------- #
# precompute.c (E-step)
# --------------------------------------------------------------------------- #
def _precompute(X, T, sigma2, w, vol):
    N, D = X.shape
    M = T.shape[0]
    ksig = -2.0 * sigma2
    outlier_tmp = (w * M * (-ksig * np.pi) ** (0.5 * D)) / ((1.0 - w) * vol)
    outlier_tmp2 = (vol * (1.0 - w)) / (w * M)

    P1 = np.zeros(M)
    Pt1 = np.empty(N)
    PX = np.zeros((M, D))
    PM = np.empty(N)
    E = 0.0

    t2 = np.sum(T * T, axis=1)
    step = max(1, _CHUNK_ELEMS // M)
    for s in range(0, N, step):
        xs = X[s:s + step]
        # squared distances via |x|^2 + |t|^2 - 2 x.t (BLAS); clipped against round-off
        d2 = np.maximum(np.sum(xs * xs, axis=1)[:, None] + t2[None, :] - 2.0 * (xs @ T.T), 0.0)
        P = np.exp(d2 / ksig)
        sp = P.sum(axis=1)
        PM[s:s + step] = 1.0 / (1.0 + outlier_tmp2 * sp)
        sp = sp + outlier_tmp
        Pt1[s:s + step] = 1.0 - outlier_tmp / sp
        Pn = P / sp[:, None]
        P1 += Pn.sum(axis=0)
        PX += Pn.T @ xs
        E += -np.sum(np.log(sp))
    E += D * N * np.log(sigma2) / 2.0 - N * np.log(1.0 - w)
    return P1, Pt1, PX, PM, E


# --------------------------------------------------------------------------- #
# parameter_solving.m (EM loop)
# --------------------------------------------------------------------------- #
def _parameter_solving(X, Y, tol, w, vol, max_iter):
    N, D = X.shape
    M = Y.shape[0]

    sigma2 = (M * np.sum(X * X) + N * np.sum(Y * Y) - 2.0 * (X.sum(axis=0) @ Y.sum(axis=0))) / (M * N * D)
    T = Y
    ntol = tol + 10.0
    L = 1.0
    it = 0

    while ntol > tol and it < max_iter:
        L_old = L
        P1, Pt1, PX, PM, L = _precompute(X, T, sigma2, w, vol)
        ntol = abs((L - L_old) / L)

        Np = P1.sum()
        mu_x = X.T @ Pt1 / Np
        mu_y = Y.T @ P1 / Np

        B1 = PX.T @ Y - Np * np.outer(mu_x, mu_y)
        B2 = (Y * P1[:, None]).T @ Y - Np * np.outer(mu_y, mu_y)
        B = np.linalg.solve(B2.T, B1.T).T  # B = B1 / B2
        t = mu_x - B @ mu_y

        sigma2 = abs(np.sum(X ** 2 * Pt1[:, None]) - Np * (mu_x @ mu_x) - np.trace(B1 @ B.T)) / (Np * D)

        temp = PM.sum()
        w = temp / (temp + Np)

        it += 1
        T = Y @ B.T + t
    return B, t, it


# --------------------------------------------------------------------------- #
# ellipsoid_par.m
# --------------------------------------------------------------------------- #
def _ellipsoid_par(init_center, R, t) -> np.ndarray:
    """-> [xc, yc, zc, a, r1=a/b, r2=a/c, alpha(Z), beta(Y), gamma(X)]"""
    center = t + R @ init_center

    Bm = R @ R.T
    evals, V = np.linalg.eigh(0.5 * (Bm + Bm.T))  # ascending, like MATLAB eig on symmetric input
    a, b, c = np.sqrt(np.maximum(evals, 0.0))

    # An ellipsoid is invariant to the sign of its axes. The Euler extraction below
    # needs a proper rotation (det = +1); flip the 3rd axis if necessary.
    if np.linalg.det(V) < 0:
        V[:, 2] = -V[:, 2]

    # R = Rz(alpha) @ Ry(beta) @ Rx(gamma)
    alpha = np.arctan2(V[1, 0], V[0, 0])
    beta = np.arctan2(-V[2, 0], np.hypot(V[0, 0], V[1, 0]))
    gamma = np.arctan2(V[2, 1], V[2, 2])

    return np.array([center[0], center[1], center[2], a, b / a, c / a, alpha, beta, gamma])


# --------------------------------------------------------------------------- #
# Fitter
# --------------------------------------------------------------------------- #
class BayesianZhaoFitter(BaseEllipsoidFitter):
    """
    Robust ellipsoid-specific fitting via EM (Zhao et al.).

    Output of :meth:`fit`: ``[x_c, y_c, z_c, a, r1, r2, alpha, beta, gamma]``

    * ``a, b, c`` are the semi-axis lengths in ascending order (``a <= b <= c``),
      as returned by the original ``ellipsoid_par.m``; ``r1 = a/b``, ``r2 = a/c``.
    * ``alpha, beta, gamma`` are ZYX Euler angles in radians, i.e. the axes of the
      ellipsoid (a, b, c along the columns) are given by
      ``R = Rz(alpha) @ Ry(beta) @ Rx(gamma)``
      (``scipy.spatial.transform.Rotation.from_euler("ZYX", [alpha, beta, gamma])``).

    Hyper-parameters (defaults = values hard-coded in the MATLAB code); they can
    also be overridden per call: ``fitter.fit(points, tol=1e-6)``.

    :param outlier_weight: initial weight of the uniform outlier component (``opt.outliers``)
    :param tol: relative tolerance on the EM objective (``opt.tol``)
    :param max_iter: iteration cap; ``opt.max_it = 200`` is declared in the MATLAB code
                     but never used there (the loop runs until convergence).
                     Use ``None`` for no cap.
    :param rdos_k: max. number of nearest neighbours in the RDOS outlier score
    :param rdos_bandwidth: KDE bandwidth ``h`` of the RDOS score (data normalised to unit RMS radius)
    :param rdos_threshold: points with RDOS score <= threshold are used as inliers for initialisation
    """

    _OPTIONS = ("outlier_weight", "tol", "max_iter", "rdos_k", "rdos_bandwidth", "rdos_threshold")

    def __init__(
        self,
        outlier_weight: float = 0,
        tol: float = 1e-8,
        max_iter: int | None = None,
        rdos_k: int = 100,
        rdos_bandwidth: float = 0.1,
        rdos_threshold: float = np.inf,
    ):
        self.outlier_weight = outlier_weight
        self.tol = tol
        self.max_iter = max_iter
        self.rdos_k = rdos_k
        self.rdos_bandwidth = rdos_bandwidth
        self.rdos_threshold = rdos_threshold

        # diagnostics of the last fit
        self.n_iter_: int | None = None
        self.rdos_scores_: np.ndarray | None = None

    def fit(self, points: np.ndarray, **kwargs) -> np.ndarray:
        opts = {name: kwargs.pop(name, getattr(self, name)) for name in self._OPTIONS}


        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
        if pts.shape[0] < 9:
            raise ValueError("At least 9 points are required to fit an ellipsoid")

        # --- outlier_det.m: normalise, RDOS score
        X, x_mean, x_scale = _normalize(pts)
        score = _rdos_scores(X, opts["rdos_k"], opts["rdos_bandwidth"])
        self.rdos_scores_ = score

        # --- demo.m: initial centre from inliers, spherical template
        inlier = pts[score <= opts["rdos_threshold"]]
        if inlier.shape[0] == 0:  # not in the original (would give NaN); fall back to all points
            inlier = pts
        inlier_num = inlier.shape[0]
        init_center = inlier.mean(axis=0)

        template = _sphere_template(init_center, np.sqrt(inlier_num))
        Y, y_mean, y_scale = _normalize(template)

        # --- ellipsoid_fit_EM.m
        extent = X.max(axis=0) - X.min(axis=0)
        vol = extent[0] * extent[1] * extent[2]
        max_iter = np.inf if opts["max_iter"] is None else opts["max_iter"]
        B, t, n_iter = _parameter_solving(X, Y, opts["tol"], opts["outlier_weight"], vol, max_iter)
        self.n_iter_ = n_iter

        s = x_scale / y_scale
        t_final = x_scale * t + x_mean - s * (B @ y_mean)
        R_final = s * B

        # --- ellipsoid_par.m
        return _ellipsoid_par(init_center, R_final, t_final)
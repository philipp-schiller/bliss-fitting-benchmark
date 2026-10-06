# core/measurement.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np


if TYPE_CHECKING:
    from core.experiment_config import ExperimentConfig


@dataclass
class Measurement:
    """
    Data supplied to the fitters for one experiment iteration.

    Parameters
    ----------
    points:
        Observed point cloud with shape:

            (n_points, ambient_dimension)

    parameter_samples:
        Latent shape parameters used to generate synthetic points.

        For an ellipse, this will usually have shape:

            (n_points, 1)

        For an ellipsoid, this may have shape:

            (n_points, 2)

        This field may be None for real-world data.

    uncertainties:
        Point-wise covariance matrices with shape:

            (n_points, ambient_dimension, ambient_dimension)

        This field may be None when uncertainties are unknown.

    source_points:
        Noise-free points corresponding to the observations.

        This field will normally be available for synthetic data but
        not for real-world data.

    metadata:
        Additional information about the measurement, such as the
        source filename or whether the measurement is synthetic.
    """

    points: np.ndarray

    parameter_samples: np.ndarray | None = None
    uncertainties: np.ndarray | None = None
    source_points: np.ndarray | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.points = np.asarray(
            self.points,
            dtype=float,
        )

        if self.points.ndim != 2:
            raise ValueError(
                "points must be a two-dimensional array with shape "
                "(n_points, ambient_dimension)."
            )

        n_points = self.points.shape[0]

        if self.parameter_samples is not None:
            self.parameter_samples = np.asarray(
                self.parameter_samples,
                dtype=float,
            )

            if self.parameter_samples.shape[0] != n_points:
                raise ValueError(
                    "parameter_samples and points must contain the "
                    "same number of samples."
                )

        if self.uncertainties is not None:
            self.uncertainties = np.asarray(
                self.uncertainties,
                dtype=float,
            )

            expected_shape = (
                n_points,
                self.points.shape[1],
                self.points.shape[1],
            )

            if self.uncertainties.shape != expected_shape:
                raise ValueError(
                    "uncertainties must have shape "
                    f"{expected_shape}, but received "
                    f"{self.uncertainties.shape}."
                )

        if self.source_points is not None:
            self.source_points = np.asarray(
                self.source_points,
                dtype=float,
            )

            if self.source_points.shape != self.points.shape:
                raise ValueError(
                    "source_points must have the same shape as points."
                )

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def ambient_dimension(self) -> int:
        return self.points.shape[1]

    @property
    def has_uncertainties(self) -> bool:
        return self.uncertainties is not None

    @property
    def is_synthetic(self) -> bool:
        return bool(
            self.metadata.get("synthetic", False)
        )


MeasurementFunction = Callable[
    ["ExperimentConfig", int],
    Measurement,
]


def generate_synthetic_measurement(
    config: ExperimentConfig,
    iteration: int,
) -> Measurement:
    """
    Generate a standard synthetic measurement.

    This function implements the generation procedure currently used
    by the ellipse experiments:

    1. Sample latent parameters from the source density.
    2. Evaluate the point-wise noise covariances.
    3. Generate the noise-free source points.
    4. Sample independent Gaussian errors.
    5. Add the errors to the source points.
    """

    parameter_samples = config.source_density.sample(
        n=config.num_points,
        params=config.source_params,
    )

    uncertainties = config.noise_function.evaluate(
        parameter_samples,
        params=config.noise_params,
    )

    source_points = config.shape.parametrization(
        parameter_samples,
        config.shape_params,
    )

    errors = np.array([
        np.random.multivariate_normal(
            mean=np.zeros(
                config.shape.ambient_dim
            ),
            cov=covariance,
        )
        for covariance in uncertainties
    ])

    points = source_points + errors

    return Measurement(
        points=points,
        parameter_samples=parameter_samples,
        uncertainties=uncertainties,
        source_points=source_points,
        metadata={
            "iteration": iteration,
            "synthetic": True,
            "primitive": type(config.shape).__name__,
        },
    )


def load_numpy_measurement(
    file_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Measurement:
    """
    Load a real-world measurement from an NPY or NPZ file.

    Supported NPY format
    --------------------
    The file directly contains the point array.

    Supported NPZ format
    --------------------
    The archive must contain a `points` array. It may additionally
    contain `parameter_samples`, `uncertainties`, or `source_points`.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Measurement file not found: {file_path}"
        )

    measurement_metadata = {
        "synthetic": False,
        "source_file": str(file_path),
    }

    if metadata is not None:
        measurement_metadata.update(metadata)

    if file_path.suffix == ".npy":
        points = np.load(
            file_path,
            allow_pickle=False,
        )

        return Measurement(
            points=points,
            metadata=measurement_metadata,
        )

    if file_path.suffix == ".npz":
        with np.load(
            file_path,
            allow_pickle=False,
        ) as data:
            if "points" not in data.files:
                raise ValueError(
                    f"The NPZ file does not contain a 'points' "
                    f"array: {file_path}"
                )

            points = data["points"].copy()

            parameter_samples = (
                data["parameter_samples"].copy()
                if "parameter_samples" in data.files
                else None
            )

            uncertainties = (
                data["uncertainties"].copy()
                if "uncertainties" in data.files
                else None
            )

            source_points = (
                data["source_points"].copy()
                if "source_points" in data.files
                else None
            )

        return Measurement(
            points=points,
            parameter_samples=parameter_samples,
            uncertainties=uncertainties,
            source_points=source_points,
            metadata=measurement_metadata,
        )

    raise ValueError(
        "Unsupported measurement format. Expected an NPY or NPZ "
        f"file, but received: {file_path.suffix}"
    )


def save_measurement(
    measurement: Measurement,
    file_path: str | Path,
) -> Path:
    """
    Save a measurement as a compressed NPZ archive.

    Only available arrays are included in the archive. Metadata is
    intentionally not stored because it may contain non-array objects.
    """

    file_path = Path(file_path)

    if file_path.suffix != ".npz":
        file_path = file_path.with_suffix(".npz")

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arrays = {
        "points": measurement.points,
    }

    if measurement.parameter_samples is not None:
        arrays["parameter_samples"] = (
            measurement.parameter_samples
        )

    if measurement.uncertainties is not None:
        arrays["uncertainties"] = (
            measurement.uncertainties
        )

    if measurement.source_points is not None:
        arrays["source_points"] = (
            measurement.source_points
        )

    np.savez_compressed(
        file_path,
        **arrays,
    )

    return file_path
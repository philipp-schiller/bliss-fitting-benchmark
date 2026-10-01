from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

from core.experiment_config import ExperimentConfig
from core.sweep_result import SweepResult


def run_experiment(
    config: ExperimentConfig,
    *,
    save_pointclouds: bool = False,
    pointcloud_directory: str | Path = "Pointclouds",
) -> SweepResult:
    """Run all fitters defined in an experiment configuration."""

    n_iterations = config.iterations
    n_fitters = len(config.fitters)
    n_shape_params = config.shape.get_n_params()

    # NaN means that a result has not been produced successfully.
    fitting_params = np.full(
        (n_iterations, n_fitters, n_shape_params),
        np.nan,
    )

    fitting_times = np.full(
        (n_iterations, n_fitters),
        np.nan,
    )

    # __post_init__ has already generated the first realization,
    # so the dimensions are known here.
    n_source_params = len(np.atleast_1d(config.source_params))
    n_noise_params = len(np.atleast_1d(config.noise_params))

    shape_gt = np.empty(
        (n_iterations, n_shape_params),
    )

    source_gt = np.empty(
        (n_iterations, n_source_params),
    )

    noise_gt = np.empty(
        (n_iterations, n_noise_params),
    )

    if save_pointclouds:
        pointcloud_directory = Path(pointcloud_directory)
        pointcloud_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    for iteration in tqdm(
        range(n_iterations),
        desc=config.name,
    ):
        # Fixed parameters stay fixed.
        # Parameters represented by None in their templates are resampled.
        config.resample(seed=iteration)

        shape_gt[iteration] = np.asarray(
            config.shape_params,
            dtype=float,
        )

        source_gt[iteration] = np.atleast_1d(
            config.source_params
        ).astype(float)

        noise_gt[iteration] = np.atleast_1d(
            config.noise_params
        ).astype(float)

        points, parameter_samples = config.get_measurement(
            n_points=config.num_points
        )

        # Compute this once because every fitter receives the same
        # measurement uncertainties.
        uncertainties = config.noise_function.evaluate(
            parameter_samples,
            params=config.noise_params,
        )

        if save_pointclouds:
            filename = (
                    pointcloud_directory
                    / f"pointcloud_{config.name}_{iteration:04d}.npz"
            )
            np.savez_compressed(
            filename,
            points=points,
            parameter_samples=parameter_samples,
            )

        for fitter_index, fitter in enumerate(config.fitters):
            try:

                # we give the exact parameters to the exact BLISS version if there
                if getattr(fitter, "name", None) == "BLISS_exact":
                    fitter.source_density.fixed_params = np.asarray(
                        config.source_params,
                        dtype=float,
                    ).copy()

                    noise_params = np.asarray(
                        config.noise_params,
                        dtype=float,
                    )

                    if noise_params.size == 1:
                        noise_params = float(noise_params.item())

                    fitter.noise_function.fixed_params = noise_params

                start = time.perf_counter()

                fitted_params = fitter.fit(
                    points,
                    uncertainties=uncertainties,
                    shape_init=None,
                )

                elapsed_time = time.perf_counter() - start

                fitted_params = np.asarray(
                    fitted_params,
                    dtype=float,
                )

                if fitted_params.shape != (n_shape_params,):
                    raise ValueError(
                        f"Expected {n_shape_params} fitted parameters, "
                        f"but received shape {fitted_params.shape}."
                    )

                fitting_params[
                    iteration,
                    fitter_index,
                ] = fitted_params

                fitting_times[
                    iteration,
                    fitter_index,
                ] = elapsed_time

            except Exception as error:
                fitter_name = getattr(
                    fitter,
                    "name",
                    fitter.__class__.__name__,
                )

                print(
                    f"[{config.name}] "
                    f"iteration {iteration}, "
                    f"fitter '{fitter_name}' failed: {error}"
                )

    return SweepResult(
        experiment_name=config.name,
        labels=list(config.labels),
        fitting_params=fitting_params,
        fitting_times=fitting_times,
        shape_gt=shape_gt,
        source_gt=source_gt,
        noise_gt=noise_gt,
    )
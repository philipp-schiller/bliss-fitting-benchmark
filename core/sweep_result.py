from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
import numpy as np


@dataclass(slots=True)
class SweepResult:
    experiment_name: str
    labels: list[str]

    fitting_params: np.ndarray
    fitting_times: np.ndarray

    shape_gt: np.ndarray
    source_gt: np.ndarray
    noise_gt: np.ndarray

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_iterations(self) -> int:
        return self.fitting_params.shape[0]

    @property
    def n_fitters(self) -> int:
        return self.fitting_params.shape[1]

    @property
    def n_shape_params(self) -> int:
        return self.fitting_params.shape[2]

    def get_fitter_results(self, index: int) -> np.ndarray:
        return self.fitting_params[:, index, :]

    def get_fitter_times(self, index: int) -> np.ndarray:
        return self.fitting_times[:, index]

    def mean_runtime(self, index: int) -> float:
        return float(np.nanmean(self.get_fitter_times(index)))

    def std_runtime(self, index: int) -> float:
        return float(np.nanstd(self.get_fitter_times(index)))

    def success_rate(self, index: int) -> float:
        estimates = self.get_fitter_results(index)
        successful = np.all(np.isfinite(estimates), axis=1)
        return float(np.mean(successful))

    def save(self, directory: str | Path) -> None:
        """Save the numerical experiment results to a compressed NPZ file."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            directory / "sweep_result.npz",
            experiment_name=self.experiment_name,
            labels=np.asarray(self.labels),
            fitting_params=self.fitting_params,
            fitting_times=self.fitting_times,
            shape_gt=self.shape_gt,
            source_gt=self.source_gt,
            noise_gt=self.noise_gt,
        )
        print(f"Result saved to {directory}\sweep_result.npz")

    @classmethod
    def load(
            cls,
            path: str | Path,
    ) -> "SweepResult":
        path = Path(path)

        if path.is_dir():
            path = path / "sweep_result.npz"

        if not path.exists():
            raise FileNotFoundError(
                f"Sweep result not found: {path}"
            )

        with np.load(path, allow_pickle=False) as data:
            return cls(
                experiment_name=str(
                    data["experiment_name"].item()
                ),
                labels=data["labels"].tolist(),
                fitting_params=data["fitting_params"].copy(),
                fitting_times=data["fitting_times"].copy(),
                shape_gt=data["shape_gt"].copy(),
                source_gt=data["source_gt"].copy(),
                noise_gt=data["noise_gt"].copy(),
            )
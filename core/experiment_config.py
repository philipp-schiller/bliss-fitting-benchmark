from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Type, TypeVar
import copy

import numpy as np


T = TypeVar("T", bound="ExperimentConfig")
Sampler = Callable[[Any], np.ndarray]


@dataclass
class ExperimentConfig:
    # --- basic metadata ---
    name: str | None = None
    iterations: int = 100
    num_points: int = 100

    # --- geometry and ground-truth model ---
    shape: object | None = None
    source_density: object | None = None
    noise_function: object | None = None

    # --- fixed or partially random ground-truth parameters ---
    shape_params: np.ndarray | None = None
    source_params: np.ndarray | None = None
    noise_params: np.ndarray | float | None = None

    # Used only when the corresponding parameters contain None
    shape_sampler: Sampler | None = None
    source_sampler: Sampler | None = None
    noise_sampler: Sampler | None = None

    # --- fitters and labels ---
    fitters: list = field(default_factory=list)
    labels: list[str] = field(default_factory=list)

    # Internal sampling templates, not constructor arguments
    shape_template: Any = field(init=False, repr=False)
    source_template: Any = field(init=False, repr=False)
    noise_template: Any = field(init=False, repr=False)

    _shape_is_random: bool = field(init=False, repr=False)
    _source_is_random: bool = field(init=False, repr=False)
    _noise_is_random: bool = field(init=False, repr=False)

    def __post_init__(self):
        self._validate()

        self.shape_template = copy.deepcopy(self.shape_params)
        self.source_template = copy.deepcopy(self.source_params)
        self.noise_template = copy.deepcopy(self.noise_params)

        self._shape_is_random = self._contains_none(
            self.shape_template
        )
        self._source_is_random = self._contains_none(
            self.source_template
        )
        self._noise_is_random = self._contains_none(
            self.noise_template
        )

        self._validate_samplers()

        # Replace initially unspecified values with the first realization.
        self.resample(seed=0)

    @staticmethod
    def _contains_none(value: Any) -> bool:
        if value is None:
            return True

        array = np.asarray(value, dtype=object)
        return any(item is None for item in array.flat)

    def _validate(self) -> None:
        if self.shape is None:
            raise ValueError("No shape was provided.")

        if self.source_density is None:
            raise ValueError("No source_density was provided.")

        if self.noise_function is None:
            raise ValueError("No noise_function was provided.")

        if self.iterations <= 0:
            raise ValueError("iterations must be positive.")

        if self.num_points <= 0:
            raise ValueError("num_points must be positive.")

        if self.labels and len(self.labels) != len(self.fitters):
            raise ValueError(
                "The number of labels must equal the number of fitters."
            )

    def _validate_samplers(self) -> None:
        requirements = (
            (
                "shape",
                self._shape_is_random,
                self.shape_sampler,
            ),
            (
                "source",
                self._source_is_random,
                self.source_sampler,
            ),
            (
                "noise",
                self._noise_is_random,
                self.noise_sampler,
            ),
        )

        for parameter_name, is_random, sampler in requirements:
            if is_random and sampler is None:
                raise ValueError(
                    f"{parameter_name}_params contains None, but "
                    f"no {parameter_name}_sampler was provided."
                )

    def resample(self, seed: int | None = None) -> None:
        """Resample parameters that were unspecified in their templates."""

        if seed is not None:
            np.random.seed(seed)

        if self._shape_is_random:
            self.shape_params = np.asarray(
                self.shape_sampler(
                    copy.deepcopy(self.shape_template)
                ),
                dtype=float,
            )

        if self._source_is_random:
            self.source_params = np.asarray(
                self.source_sampler(
                    copy.deepcopy(self.source_template)
                ),
                dtype=float,
            )

        if self._noise_is_random:
            self.noise_params = np.asarray(
                self.noise_sampler(
                    copy.deepcopy(self.noise_template)
                ),
                dtype=float,
            )

    @classmethod
    def get_from_dict(
        cls: Type[T],
        cfg_dict: Dict[str, Any],
    ) -> T:
        """Construct an ExperimentConfig from a dictionary."""

        valid_fields = {
            name
            for name, definition in cls.__dataclass_fields__.items()
            if definition.init
        }

        filtered = {
            key: value
            for key, value in cfg_dict.items()
            if key in valid_fields
        }

        unused = set(cfg_dict) - valid_fields

        if unused:
            print(
                "[ExperimentConfig] Warning: ignoring unknown "
                f"fields: {sorted(unused)}"
            )

        return cls(**filtered)

    def get_measurement(
        self,
        n_points: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate one noisy measurement from the current ground truth."""

        if n_points is None:
            n_points = self.num_points

        parameter_samples = self.source_density.sample(
            n=n_points,
            params=self.source_params,
        )

        covariances = self.noise_function.evaluate(
            parameter_samples,
            params=self.noise_params,
        )

        source_points = self.shape.parametrization(
            parameter_samples,
            self.shape_params,
        )

        errors = np.array([
            np.random.multivariate_normal(
                mean=np.zeros(self.shape.ambient_dim),
                cov=covariance,
            )
            for covariance in covariances
        ])

        measured_points = source_points + errors

        return measured_points, parameter_samples
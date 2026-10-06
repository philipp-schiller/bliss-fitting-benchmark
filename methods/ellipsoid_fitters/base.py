from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


class BaseEllipsoidFitter(ABC):
    @abstractmethod
    def fit(self, points: np.ndarray, **kwargs) -> np.ndarray:
        """
        :param points: Points to fit
        :param args:
        :param kwargs:
        :return: parameters of the fitted ellipsoid in form [x_c, y_c, z_c, a, r1, r2, α, β, γ]
                 where r_1 = a/b and r_2 ) a/c
        """
        pass

from abc import ABC, abstractmethod
import numpy as np
from typing import Callable


class BaseEllipseFitter(ABC):
    @abstractmethod
    def fit(self, points: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        :param points: Points to fit
        :param args:
        :param kwargs:
        :return: parameters of the fitted ellipse (c_x,c_y, a, b, theta)
        """
        pass

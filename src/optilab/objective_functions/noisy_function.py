"""
Class that makes an objective function noisy using random normal distribution. 
"""

# pylint: disable=too-few-public-methods

from typing import List

import numpy as np

from .objective_function import ObjectiveFunction


class NoisyFunction(ObjectiveFunction):
    """
    Class that makes an objective function noisy using random normal distribution.
    """

    def __init__(self, function: ObjectiveFunction, noise: float, dim: int):
        """
        Class constructor.

        :param function: objective function to add noise to.
        :param noise: float multiplier of noise.
        :param dim: dimensionality of the function.
        """
        super().__init__(f"noisy_{function.name}_{noise}", dim)
        self.function = function
        self.noise = noise

    def __call__(self, x: List[float]) -> float:
        """
        Evaluate a single point with the objective function.

        :param x: point to be evaluated
        :raises ValueError: if dimensionality of x doesn't match self.dim
        :return: value of the function in the provided point
        """
        super().__call__(x)
        return self.function(x) * (1 + self.noise * np.random.normal(0, 1))

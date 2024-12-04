"""
The noisy sphere function objective function
"""

# pylint: disable=too-few-public-methods

from typing import List

import numpy as np

from .objective_function import ObjectiveFunction


class NoisySphereFunction(ObjectiveFunction):
    """
    Noisy sphere objective function.
    """

    def __init__(self, epsilon: float, dim: int):
        """
        Class constructor.

        :raises ValueError: when the number of function is invalid.
        :param dim: dimensionality of the function.
        """
        super().__init__(f"noisy_sphere_{epsilon}", dim)
        self.epsilon = epsilon

    def __call__(self, x: List[float]) -> float:
        """
        Evaluate a single point with the objective function.

        :param x: point to be evaluated
        :raises ValueError: if dimensionality of x doesn't match self.dim
        :return: value of the function in the provided point
        """
        super().__call__(x)
        return sum(x_i**2 for x_i in x) * (1 + np.random.normal(0, 1))

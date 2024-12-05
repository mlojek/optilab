"""
The rastrigin objective function
"""

# pylint: disable=too-few-public-methods

from typing import List

import numpy as np

from ..objective_function import ObjectiveFunction


class RastriginFunction(ObjectiveFunction):
    """
    Rastrigin objective function.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        :raises ValueError: when the number of function is invalid.
        :param dim: dimensionality of the function.
        """
        super().__init__("rastrigin", dim)

    def __call__(self, x: List[float]) -> float:
        """
        Evaluate a single point with the objective function.

        :param x: point to be evaluated
        :raises ValueError: if dimensionality of x doesn't match self.dim
        :return: value of the function in the provided point
        """
        super().__call__(x)
        return sum(x_i**2 - 10 * np.cos(2 * np.pi * x_i) + 10 for x_i in x)

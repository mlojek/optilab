"""
Sphere function. y is the sum of squares of elements of x vector.
"""

# pylint: disable=too-few-public-methods

from typing import List

import numpy as np

from ..objective_function import ObjectiveFunction


class SphereFunction(ObjectiveFunction):
    """
    Sphere function. y is the sum of squares of elements of x vector.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim (int): Dimensionality of the function.
        """
        super().__init__("sphere", dim)

    def __call__(self, x: List[float]) -> float:
        """
        Evaluate a single point with the objective function.

        Args:
            x (List[float]): Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Returns:
            float: Value of the function in the provided point.
        """
        super().__call__(x)
        return sum(np.asarray(x) ** 2)

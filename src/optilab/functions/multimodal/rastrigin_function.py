"""
The rastrigin objective function
"""

import numpy as np

from ...data_classes import Point
from ..objective_function import ObjectiveFunction


class RastriginFunction(ObjectiveFunction):
    """
    Rastrigin objective function.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim (int): Dimensionality of the function.
        """
        super().__init__("rastrigin", dim)

    def __call__(self, point: Point) -> Point:
        """
        Evaluate a single point with the objective function.

        Args:
            point (Point): Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Returns:
            Point: Evaluated point.
        """
        super().__call__(point)

        return Point(
            x=point.x,
            y=sum(x_i**2 - 10 * np.cos(2 * np.pi * x_i) + 10 for x_i in point.x),
            is_evaluated=True,
        )

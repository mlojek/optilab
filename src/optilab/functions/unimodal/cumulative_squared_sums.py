"""
Cumulative squared sums function.
"""

# pylint: disable=too-few-public-methods

from typing import List

from ..objective_function import ObjectiveFunction


class CumulativeSquaredSums(ObjectiveFunction):
    """
    Cumulative squared sums function.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim (int): Dimensionality of the function.
        """
        super().__init__("cumulative_squared_sums", dim)

    def __call__(self, x: List[float]) -> float:
        """
        Evaluate a single point with the objective function.

        Args:
            x (List[float]): Point to be evaluated.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Returns:
            float: Value of the function in the provided point.
        """
        super().__call__(x)
        return sum(sum(x[:i]) ** 2 for i in range(len(x)))

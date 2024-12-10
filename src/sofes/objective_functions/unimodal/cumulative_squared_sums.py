"""
TODO
"""

# pylint: disable=too-few-public-methods

from typing import List

from ..objective_function import ObjectiveFunction


class CumulativeSquaredSums(ObjectiveFunction):
    """
    TODO
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        :param dim: dimensionality of the function.
        """
        super().__init__("sphere", dim)

    def __call__(self, x: List[float]) -> float:
        """
        Evaluate a single point with the objective function.

        :param x: point to be evaluated
        :raises ValueError: if dimensionality of x doesn't match self.dim
        :return: value of the function in the provided point
        """
        super().__call__(x)
        return sum(sum(x[:i]) ** 2 for i in range(len(x)))

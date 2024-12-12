"""
Bent Cigar objective function. 
"""

# pylint: disable=too-few-public-methods

from typing import List

from ..objective_function import ObjectiveFunction


class BentCigarFunction(ObjectiveFunction):
    """
    Bent Cigar objective function.
    """

    def __init__(self, dim: int):
        """
        Class constructor.

        Args:
            dim (int): Dimensionality of the function.
        """
        super().__init__("bent_cigar", dim)

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
        return x[0] ** 2 + sum(x_i**2 for x_i in x) * (10**6)

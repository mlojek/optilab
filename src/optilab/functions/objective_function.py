"""
Abstract base class representing a callable objective function.
"""

# pylint: disable=too-few-public-methods

from typing import List


class ObjectiveFunction:
    """
    Abstract base class representing a callable objective function.
    """

    def __init__(self, name: str, dim: int) -> None:
        """
        Class constructor.

        Args:
            name (str): Name of the objective function.
            dim (int): Dimensionality of the function.
        """
        self.name = name
        self.dim = dim
        self.num_calls = 0

    def __call__(self, x: List[float]) -> float:
        """
        Evaluate a single point with the objective function.

        Args:
            x (List[float]): Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim

        Returns:
            float: Value of the function in the provided point.
        """
        if not len(x) == self.dim:
            raise ValueError(
                f"The dimensionality of the provided point is not matching the dimensionality"
                f"of the function. Expected {self.dim}, got {len(x)}"
            )
        self.num_calls += 1

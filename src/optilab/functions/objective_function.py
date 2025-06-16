"""
Base class representing a callable objective function.
"""

from typing import Any, Dict

from ..data_classes import FunctionMetadata, Point


class ObjectiveFunction:
    """
    Base class representing a callable objective function.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        hyperparameters: Dict[str, Any] = None,
    ) -> None:
        """
        Class constructor.

        Args:
            name (str): Name of the objective function.
            dim (int): Dimensionality of the function.
            hyperparameters (Dict[str, Any]): Dictionary with hyperparameters of the function.
        """
        if not hyperparameters:
            hyperparameters = {}
        self.metadata = FunctionMetadata(name, dim, hyperparameters)
        self.num_calls = 0

    def __call__(self, point: Point) -> Point:
        """
        Evaluate a single point with the objective function.

        Args:
            point (Point): Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim

        Returns:
            Point: Evaluated point.
        """
        if not len(point.x) == self.metadata.dim:
            raise ValueError(
                f"The dimensionality of the provided point is not matching the dimensionality"
                f"of the function. Expected {self.metadata.dim}, got {len(point.x)}"
            )
        self.num_calls += 1

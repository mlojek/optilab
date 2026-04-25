"""
Base class representing a callable objective function.
"""

from typing import Any

from ..data_classes import FunctionMetadata, Point


class ObjectiveFunction:
    """
    Base class representing a callable objective function.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Class constructor.

        Args:
            name: Name of the objective function.
            dim: Dimensionality of the function.
            hyperparameters: Dictionary with hyperparameters of the function.
        """
        if not hyperparameters:
            hyperparameters = {}
        self.metadata = FunctionMetadata(
            name=name, dim=dim, hyperparameters=hyperparameters
        )
        self.num_calls = 0

    def __call__(self, point: Point) -> Point:
        """
        Validate point dimensionality and increment call counter.

        Args:
            point: Point to evaluate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim

        Returns:
            The validated point (subclasses return a new evaluated Point).
        """
        assert point.x is not None
        if not len(point.x) == self.metadata.dim:
            raise ValueError(
                f"The dimensionality of the provided point is not matching the dimensionality"
                f"of the function. Expected {self.metadata.dim}, got {len(point.x)}"
            )
        self.num_calls += 1
        return point

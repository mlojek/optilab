"""
Abstract base class for surrogate objective functions.
"""

from typing import Any

from ...data_classes import Point, PointList
from ..objective_function import ObjectiveFunction


class SurrogateObjectiveFunction(ObjectiveFunction):
    """
    Abstract base class for surrogate objective functions.
    """

    def __init__(
        self,
        name: str,
        train_set: PointList | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        """
        Class constructor. The dimensionality is deduced from the training points.

        Args:
            name: Name of the surrogate function.
            train_set: Training data for the model.
            hyperparameters: Dictionary with hyperparameters of the function.
        """
        self.is_ready = False
        super().__init__(
            name,
            1,
            hyperparameters,
        )

        if train_set:
            self.train(train_set)

    def train(self, train_set: PointList) -> None:
        """
        Train the Surrogate function with provided data.

        Args:
            train_set: Training data for the model.

        Raises:
            ValueError: If not all points are evaluated.
        """
        if not all((train_point.is_evaluated for train_point in train_set.points)):
            raise ValueError("Not all points in the training set are evaluated!")

        self.is_ready = True

        dim_set = {point.dim() for point in train_set.points}
        if not len(dim_set) == 1:
            raise ValueError(
                "Provided train set has x-es with different dimensionalities."
            )

        if 0 in dim_set:
            raise ValueError("0-dim x values found in train set.")

        self.metadata.dim = list(dim_set)[0]

        self.train_set = train_set

    def __call__(self, point: Point) -> Point:
        """
        Validate the point before estimation. Concrete subclasses return a new evaluated Point.

        Args:
            point: Point to estimate.

        Raises:
            NotImplementedError: If the surrogate function is not trained.
            ValueError: If dimensionality of x doesn't match self.dim.

        Returns:
            The validated point (concrete subclasses return a new estimated Point).
        """
        if not self.is_ready:
            raise NotImplementedError("The surrogate function is not trained!")
        return super().__call__(point)

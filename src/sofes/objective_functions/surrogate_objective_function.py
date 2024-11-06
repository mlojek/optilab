"""
Abstract base class for surrogate objective functions.
"""

# pylint: disable=too-few-public-methods

from typing import List, Tuple

from .objective_function import ObjectiveFunction


class SurrogateObjectiveFunction(ObjectiveFunction):
    """
    Abstract base class for surrogate objective functions.
    """

    def __init__(
        self, name: str, dim: int, train_set: List[Tuple[List[float], float]]
    ) -> None:
        """
        Class constructor

        :param name: name of the surrogate function
        :param dim: dimensionality of the function
        :param_train_set: training data for the model
        """
        super().__init__(name, dim)
        self.train(train_set)

    def train(self, train_set: List[Tuple[List[float], float]]) -> None:
        """
        Train the Surrogate function with provided data

        :param train_set: train data expressed as list of tuples of x, y values
        """
        raise NotImplementedError

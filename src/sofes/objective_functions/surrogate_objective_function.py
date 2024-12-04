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
        self, name: str, train_set: List[Tuple[List[float], float]] = None
    ) -> None:
        """
        Class constructor

        :param name: name of the surrogate function
        :param dim: dimensionality of the function
        :param_train_set: training data for the model
        """
        self.is_ready = False
        super().__init__(name, 1)

        if train_set:
            self.train(train_set)

    def train(self, train_set: List[Tuple[List[float], float]]) -> None:
        """
        Train the Surrogate function with provided data

        :param train_set: train data expressed as list of tuples of x, y values
        """
        self.is_ready = True
        dim_set = {len(x) for x, _ in train_set}
        if not len(dim_set) == 1:
            raise ValueError(
                "Provided train set has x-es with different dimensionalities."
            )
        self.dim = list(dim_set)[0]
        self.train_set = train_set

    def __call__(self, x: List[float]) -> float:
        """
        Predict the value of x with the surrogate function.

        :param x: point to predict the function value of
        :return: predicted function value
        """
        if not self.is_ready:
            raise NotImplementedError("The surrogate function is not trained!")
        super().__call__(x)

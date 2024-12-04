"""
TODO
"""

from typing import Callable, List, Tuple

import numpy as np

from .surrogate_objective_function import SurrogateObjectiveFunction


def biquadratic_kernel_function(x: float) -> float:
    """
    TODO
    """
    if np.abs(x) >= 1:
        return 0
    return (1 - x**2) ** 2


class LocallyWeightedRegression(SurrogateObjectiveFunction):
    """
    TODO
    """

    def __init__(
        self,
        num_neighbours: float,
        train_set: List[Tuple[List[float], float]] = None,
        covariance_matrix: List[List[float]] = None,  # Have to be validated,
        kernel_function: Callable[[float], float] = biquadratic_kernel_function,
    ) -> None:
        """
        Class constructor
        TODO
        TODO always has interactions

        :param name: name of the surrogate function
        :param dim: dimensionality of the function
        :param_train_set: training data for the model
        """
        self.is_ready = False
        super().__init__("locally_weighted_regression", train_set)

        if train_set:
            self.train(train_set)

        self.num_neighbours = num_neighbours

        if covariance_matrix:
            self.set_covariance_matrix(covariance_matrix)
        else:
            self.set_covariance_matrix(np.eye(self.dim))

        self.kernel_function = kernel_function

    def set_covariance_matrix(self, new_covariance_matrix: List[List[float]]) -> None:
        """
        TODO
        """
        # validate new matrix
        new_covariance_matrix = np.array(new_covariance_matrix)
        assert len(new_covariance_matrix.shape) == 2
        assert new_covariance_matrix.shape[0] == new_covariance_matrix.shape[1]

        self.reversed_covariance_matrix = np.invert(np.array(new_covariance_matrix))

    def __call__(self, x: List[float], regressor: SurrogateObjectiveFunction) -> float:
        """
        Predict the value of x with the surrogate function.

        :param x: point to predict the function value of
        :return: predicted function value


        # TODO regressor should have train and __call__ methods implemented
        """
        super().__call__(x)

        # calculate distances to all points
        distance_points = [
            (
                np.sqrt(
                    np.transpose(np.array(x_t) - np.array(x))
                    * self.reversed_covariance_matrix
                    * (np.array(x_t) - np.array(x))
                ),
                np.array(x_t),
                y_t,
            )
            for x_t, y_t in self.train_set
        ].sort(key=lambda i: i[0])

        # select KNN
        knn_points = distance_points[: self.num_neighbours]

        bandwidth = knn_points[-1][0]

        weighted_points = [
            (np.sqrt(self.kernel_function(d / bandwidth)) * x_t, y_t)
            for d, x_t, y_t in knn_points
        ]

        regressor.train(weighted_points)
        return regressor(x)

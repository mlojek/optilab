"""
TODO
"""

from typing import Callable, List, Tuple

import numpy as np
from scipy.spatial.distance import mahalanobis

from .surrogate_objective_function import SurrogateObjectiveFunction

# pylint: disable=too-many-arguments,too-many-positional-arguments


def biquadratic_kernel_function(x: float) -> float:
    """
    TODO
    """
    if np.abs(x) >= 1:
        return 0
    return (1 - x**2) ** 2


class LocallyWeightedRegression(SurrogateObjectiveFunction):
    """
    # TODO
    """

    def __init__(
        self,
        num_neighbours: float,
        regressor: SurrogateObjectiveFunction,
        train_set: List[Tuple[List[float], float]] = None,
        covariance_matrix: List[List[float]] = None,  # Have to be validated,
        kernel_function: Callable[[float], float] = biquadratic_kernel_function,
    ) -> None:
        """
        # TODO
        """
        self.is_ready = False
        super().__init__(f"locally_weighted_{regressor.name}", train_set)

        if train_set:
            self.train(train_set)

        self.num_neighbours = num_neighbours

        if covariance_matrix:
            self.set_covariance_matrix(covariance_matrix)
        else:
            self.set_covariance_matrix(np.eye(self.dim))

        self.kernel_function = kernel_function
        self.regressor = regressor

    def set_covariance_matrix(self, new_covariance_matrix: List[List[float]]) -> None:
        """
        TODO
        """
        # validate new matrix
        new_covariance_matrix = np.array(new_covariance_matrix)
        assert len(new_covariance_matrix.shape) == 2
        assert new_covariance_matrix.shape[0] == new_covariance_matrix.shape[1]

        self.reversed_covariance_matrix = np.linalg.inv(np.array(new_covariance_matrix))

    def __call__(self, x: List[float]) -> float:
        """
        TODO
        """
        super().__call__(x)

        # calculate distances to all points
        distance_points = [
            (
                mahalanobis(x_t, x, self.reversed_covariance_matrix),
                np.array(x_t),
                y_t,
            )
            for x_t, y_t in self.train_set
        ]

        distance_points.sort(key=lambda i: i[0])

        # select KNN
        knn_points = distance_points[: self.num_neighbours]

        bandwidth = knn_points[-1][0]

        weighted_points = [
            (np.sqrt(self.kernel_function(d / bandwidth)) * x_t, y_t)
            for d, x_t, y_t in knn_points
        ]

        self.regressor.train(weighted_points)
        return self.regressor(x)

"""
Surrogate function which estimates the objective function with polynomial regression.
Points are weighted based on mahalanobis distance from query points.
"""

from typing import Callable

import numpy as np
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import PolynomialFeatures

from ...data_classes import Point, PointList
from .surrogate_objective_function import SurrogateObjectiveFunction


def biquadratic_kernel_function(x: float) -> float:
    """
    Biquadratic weighting function.

    Args:
        x (float): Distance between points.

    Returns:
        float: Weight value.
    """
    if np.abs(x) >= 1:
        return 0

    return (1 - x**2) ** 2


class LocallyWeightedPolynomialRegression(SurrogateObjectiveFunction):
    """
    Surrogate function which estimates the objective function with polynomial regression.
    Points are weighted based on mahalanobis distance from query points.
    """

    def __init__(
        self,
        degree: int,
        num_neighbors: int,
        train_set: PointList = None,
        covariance_matrix: np.ndarray = None,
        kernel_function: Callable[[float], float] = biquadratic_kernel_function,
    ) -> None:
        """
        Class constructor.

        Args:
            degree (int): Degree of the polynomial used to approximate function.
            num_neighbors (float): Number of closest points to use in function approximation.
            train_set (PointList): Training set for the regressor, optional.
            covariance_matrix (np.ndarray): Covariance class used in mahalanobis distance,
                optional. When no such matrix is provided an identity matrix is used.
            kernel_function (Callable[[float], float]): Function used to assign weights to points.
        """
        self.is_ready = False
        super().__init__(
            f"locally_weighted_polynomial_regression_{degree}_degree",
            train_set,
            {"degree": degree, "num_neighbors": num_neighbors},
        )

        if train_set:
            self.train(train_set)

        if covariance_matrix:
            self.set_covariance_matrix(covariance_matrix)
        else:
            self.set_covariance_matrix(np.eye(self.metadata.dim))

        self.kernel_function = kernel_function
        self.preprocessor = PolynomialFeatures(degree=degree)
        self.weights = None

    def set_covariance_matrix(self, new_covariance_matrix: np.ndarray) -> None:
        """
        Setter for the covariance matrix.

        Args:
            new_covariance_matrix (np.ndarray): New covariance matrix to use for mahalanobis
                distance.
        """
        self.reversed_covariance_matrix = np.linalg.inv(new_covariance_matrix)

    def __call__(self, point: Point) -> Point:
        """
        Estimate the value of a single point with the surrogate function. Since the surrogate model
        is built for each point independently, this is where the regressor is trained.

        Args:
            x (Point): Point to estimate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Return:
            Point: Estimated point.
        """
        super().__call__(point)

        distance_points = [
            (
                mahalanobis(train_point.x, point.x, self.reversed_covariance_matrix),
                np.array(train_point.x),
                train_point.y,
            )
            for train_point in self.train_set
        ]

        distance_points.sort(key=lambda i: i[0])

        knn_points = distance_points[: self.metadata.hyperparameters["num_neighbours"]]

        bandwidth = knn_points[-1][0]

        weights = [
            (np.sqrt(self.kernel_function(d / bandwidth)), x_i, y_i)
            for d, x_i, y_i in knn_points
        ]

        weighted_x, weighted_y = zip(
            *[
                (
                    w * np.array(self.preprocessor.fit_transform([x_i])[0]),
                    w * np.array(y_i),
                )
                for w, x_i, y_i in weights
            ]
        )

        self.weights = np.linalg.lstsq(weighted_x, weighted_y)[0]

        return Point(
            x=point.x,
            y=sum(self.weights * self.preprocessor.fit_transform([point.x])[0]),
            is_evaluated=False,
        )

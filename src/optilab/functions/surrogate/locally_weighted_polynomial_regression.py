"""
Surrogate function which estimates the objective function with polynomial regression.
Points are weighted based on mahalanobis distance from query points.
"""

from typing import Callable

import faiss
import numpy as np
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

        if covariance_matrix:
            self.set_covariance_matrix(covariance_matrix)
        else:
            self.set_covariance_matrix(np.eye(self.metadata.dim))

        if train_set:
            self.train(train_set)

        self.kernel_function = kernel_function
        self.preprocessor = PolynomialFeatures(degree=degree)

        self.weights = None
        self.index = None

    def set_covariance_matrix(self, new_covariance_matrix: np.ndarray) -> None:
        """
        Setter for the covariance matrix.

        Args:
            new_covariance_matrix (np.ndarray): New covariance matrix to use for mahalanobis
                distance.
        """
        self.inverse_sqrt_covariance = np.linalg.inv(
            np.linalg.cholesky(new_covariance_matrix)
        ).T

    def train(self, train_set: PointList) -> None:
        """
        Build FAISS index and preprocess data to use Mahalanobis distance.

        Args:
            train_set (PointList): Training set for the function
        """
        super().train(train_set)

        x_train, y_train = self.train_set.pairs()
        x_train = np.array(x_train, dtype=np.float32) @ self.inverse_sqrt_covariance
        self.y_train = np.array(y_train, dtype=np.float32)

        self.index = faiss.IndexFlatL2(x_train.shape[1])
        self.index.add(x_train)  # pylint: disable=no-value-for-parameter

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

        x_query = np.array([point.x], dtype=np.float32) @ self.inverse_sqrt_covariance
        distances, indices = (
            self.index.search(  # pylint: disable=no-value-for-parameter
                x_query,
                self.metadata.hyperparameters["num_neighbors"],
            )
        )
        distances = np.sqrt(distances)

        knn_x = np.array([self.train_set[i].x for i in indices[0]])
        knn_y = np.array([self.train_set[i].y for i in indices[0]])

        bandwidth = distances[0][-1]

        weights = np.array(
            [np.sqrt(self.kernel_function(d / bandwidth)) for d in distances[0]]
        )

        weighted_x = weights[:, None] * self.preprocessor.fit_transform(knn_x)
        weighted_y = weights * knn_y

        self.weights = np.linalg.lstsq(weighted_x, weighted_y, rcond=None)[0]

        y_pred = float(
            np.dot(self.preprocessor.fit_transform([point.x])[0], self.weights)
        )

        return Point(x=point.x, y=y_pred, is_evaluated=False)

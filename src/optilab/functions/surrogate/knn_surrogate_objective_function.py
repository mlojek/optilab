"""
Surrogate Objective function using FAISS for fast KNN-based regression.
"""

import faiss
import numpy as np

from ...data_classes import Point, PointList
from .surrogate_objective_function import SurrogateObjectiveFunction


class KNNSurrogateObjectiveFunction(SurrogateObjectiveFunction):
    """
    Surrogate objective function using FAISS for fast KNN-based regression.
    """

    def __init__(self, num_neighbors: int, train_set: PointList = None) -> None:
        """
        Class constructor.

        Args:
            num_neighbors (int): Number of closest neighbors to use in regression.
            train_set (PointList): Training data for the model.
        """
        self.num_neighbors = num_neighbors

        self.faiss_index = None
        self.y_train = None

        super().__init__(
            f"FastKNN{num_neighbors}", train_set, {"num_neighbors": num_neighbors}
        )

    def train(self, train_set: PointList) -> None:
        """
        Train the FAISS-based KNN Surrogate function with provided data.

        Args:
            train_set (PointList): Training data for the model.
        """
        super().train(train_set)

        x_train, y_train = self.train_set.pairs()
        x_train = np.array(x_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        self.faiss_index = faiss.IndexFlatL2(x_train.shape[1])
        self.faiss_index.add(x_train)  # pylint: disable=no-value-for-parameter
        self.y_train = y_train

    def __call__(self, point: Point) -> Point:
        """
        Estimate the function value at a given point using kNN regression.

        Args:
            point (Point): Point to estimate.

        Returns:
            Point: Estimated value of the function at the given point.
        """
        super().__call__(point)

        if len(self.train_set) < self.num_neighbors:
            raise ValueError("Train set length is below number of neighbors.")

        x_query = np.array([point.x], dtype=np.float32)
        distances, indices = (
            self.faiss_index.search(  # pylint: disable=no-value-for-parameter
                x_query,
                self.num_neighbors,
            )
        )

        weights = 1 / (np.sqrt(distances) + 1e-8)  # avoid division by zero
        y_pred = np.sum(self.y_train[indices] * weights, axis=1)[0] / weights.sum()

        return Point(x=point.x, y=float(y_pred), is_evaluated=False)

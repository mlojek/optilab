"""
Surrogate objective function that uses K nearest neighbours model to estimate
function value. The neighbours are weighted by distance from x.
"""

from sklearn.neighbors import KNeighborsRegressor

from ...data_classes import FunctionMetadata, Point, PointList
from .surrogate_objective_function import SurrogateObjectiveFunction


class KNNSurrogateObjectiveFunction(SurrogateObjectiveFunction):
    """
    Surrogate objective function that uses K nearest neighbours model to estimate
    function value. The neighbours are weighted by distance from x.
    """

    def __init__(
        self,
        num_neighbors: int,
        train_set: PointList = None,
    ) -> None:
        """
        Class constructor.

        Args:
            num_neighbors (int): Number of closest neighbors to use in regression.
            train_set (PointList): Training data for the model.
        """
        self.num_neighbors = num_neighbors
        self.model = KNeighborsRegressor(n_neighbors=num_neighbors, weights="distance")
        super().__init__(f"KNN{num_neighbors}", train_set)

    def train(self, train_set: PointList) -> None:
        """
        Train the KNN Surrogate function with provided data.

        Args:
            train_set (PointList): Training data for the model.
        """
        super().train(train_set)
        self.model.fit(*train_set.pairs())

    def get_metadata(self) -> FunctionMetadata:
        """
        Get the metadata describing the function.

        Returns:
            FunctionMetadata: The metadata of the function.
        """
        metadata = super().get_metadata()
        metadata.hyperparameters["num_neighbors"] = self.num_neighbors
        return metadata

    def __call__(self, point: Point) -> Point:
        """
        Estimate the value of a single point with the surrogate function.

        Args:
            point (Point): Point to estimate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Returns:
            Point: Estimated value of the function in the provided point.
        """
        super().__call__(point)
        return Point(
            x=point.x, y=float(self.model.predict([point.x])[0]), is_evaluated=False
        )

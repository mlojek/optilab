"""
Surrogate objective function that uses K nearest neighbours model to estimate
function value. The neighbours are weighted by distance from x.
"""

from typing import List, Tuple

from sklearn.neighbors import KNeighborsRegressor

from .surrogate_objective_function import SurrogateObjectiveFunction


class KNNSurrogateObjectiveFunction(SurrogateObjectiveFunction):
    """
    Surrogate objective function that uses K nearest neighbours model to estimate
    function value. The neighbours are weighted by distance from x.
    """

    def __init__(
        self,
        num_neighbors: int,
        train_set: List[Tuple[List[float], float]] = None,
    ) -> None:
        """
        Class constructor.

        Args:
            num_neighbors (int): Number of closest neighbors to use in regression.
            train_set (List[Tuple[List[float], float]]): Training data for the model.
        """
        self.model = KNeighborsRegressor(n_neighbors=num_neighbors, weights="distance")
        super().__init__(f"KNN{num_neighbors}", train_set)

    def train(self, train_set: List[Tuple[List[float], float]]) -> None:
        """
        Train the KNN Surrogate function with provided data

        Args:
            train_set (List[Tuple[List[float], float]]): Train data expressed as list
                of tuples of x, y values.
        """
        super().train(train_set)
        x_train = [x for x, _ in train_set]
        y_train = [y for _, y in train_set]
        self.model.fit(x_train, y_train)

    def __call__(self, x: List[float]) -> float:
        """
        Estimate the value of a single point with the surrogate function.

        Args:
            x (List[float]): Point to estimate.

        Raises:
            ValueError: If dimensionality of x doesn't match self.dim.

        Return:
            float: Estimated value of the function in the provided point.
        """
        super().__call__(x)
        return float(self.model.predict([x])[0])

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
        name: str,
        dim: int,
        train_set: List[Tuple[List[float], float]],
        num_neighbors: int = 5,
    ) -> None:
        """"""
        super().__init__(name, dim, train_set)
        self.model = KNeighborsRegressor(n_neighbors=num_neighbors, weights="distance")
        self.train(train_set)

    def train(self, train_set: List[Tuple[List[float], float]]) -> None:
        """"""
        x_train = [x for x, _ in train_set]
        y_train = [y for _, y in train_set]
        self.model.fit(x_train, y_train)

    def __call__(self, x: List[float]) -> float:
        """"""
        super().__call__(x)
        return float(self.model.predict([x])[0])

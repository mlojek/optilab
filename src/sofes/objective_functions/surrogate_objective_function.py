""""""

# pylint: disable=too-few-public-methods

from typing import Any, List, Tuple

from .objective_function import ObjectiveFunction


class SurrogateObjectiveFunction(ObjectiveFunction):
    """"""

    def __init__(
        self, name: str, dim: int, train_set: List[Tuple[List[float], float]]
    ) -> None:
        super().__init__(name, dim)
        self.train(train_set)

    def train(self, train_set: List[Tuple[List[float], float]]) -> None:
        """"""
        raise NotImplementedError

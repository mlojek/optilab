"""
Locally weighted regression approximate ranking metamodel based on lmm-CMA-ES.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments

from typing import List, Tuple

from sofes.objective_functions import (
    LocallyWeightedRegression,
    ObjectiveFunction,
    SurrogateObjectiveFunction,
)

from .approximate_ranking_metamodel import ApproximateRankingMetamodel


class LocallyWeightedRegressionMetamodel(ApproximateRankingMetamodel):
    """
    TODO
    """

    def __init__(
        self,
        input_size: int,
        popsize: int,
        objective_function: ObjectiveFunction,
        regressor: SurrogateObjectiveFunction,
        num_neighbours: int,
        covariance_matrix: List[List[float]] = None,
    ) -> None:
        """
        TODO
        """
        super().__init__(
            input_size,
            popsize,
            objective_function,
            LocallyWeightedRegression(num_neighbours, covariance_matrix),
        )
        self.regressor = regressor

    def _update_covariance_matrix(self, new_covariance_matrix):
        """
        TODO
        """
        self.surrogate_function.set_covariance_matrix(new_covariance_matrix)

    def __call__(self, xs: List[List[float]]) -> List[Tuple[List[float], float]]:
        """
        TODO
        """
        return [(x, self.surrogate_function(x, self.regressor)) for x in xs]

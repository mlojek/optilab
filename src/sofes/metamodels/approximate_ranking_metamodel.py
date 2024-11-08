"""
Approximate ranking metamodel based on lmm-CMA-ES.
"""

from typing import List, Type

from sofes.objective_functions import ObjectiveFunction, SurrogateObjectiveFunction


class ApproximateRankingMetamodel:
    """Approximate ranking metamodel based on lmm-CMA-ES"""

    def __init__(
        self,
        input_size: int,
        popsize: int,
        objective_function: ObjectiveFunction,
        surrogate_function_type: Type[SurrogateObjectiveFunction],
    ) -> None:
        self.input_size = input_size
        self.popsize = popsize

        self.n_init = input_size
        self.n_step = max(1, input_size // 10)

        self.train_set = []

        self.objective_function = objective_function
        self.surrogate_function = None  # initialize SOF

    def __call__(self, x, train_set) -> List[float]:
        # assert lambda

        pass

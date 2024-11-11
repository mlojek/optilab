"""
Approximate ranking metamodel based on lmm-CMA-ES.
"""

from typing import List, Tuple, Type

from sofes.objective_functions import ObjectiveFunction, SurrogateObjectiveFunction


def rank_items(items: List[Tuple[List[float], float, bool]]) -> List[Tuple[List[float], float, bool]]:
    return zip(*sorted(items, key=lambda x: x[1]))


class ApproximateRankingMetamodel:
    """Approximate ranking metamodel based on lmm-CMA-ES"""

    def __init__(
        self,
        input_size: int,
        popsize: int,
        objective_function: ObjectiveFunction,
        surrogate_function: SurrogateObjectiveFunction,
    ) -> None:
        self.input_size = input_size
        self.popsize = popsize

        self.n_init = input_size
        self.n_step = max(1, input_size // 10)

        self.train_set = []

        self.objective_function = objective_function
        self.surrogate_function = surrogate_function

    def _update_n(self, num_iters: int) -> None:
        if num_iters > 2:
            self.n_init = min(self.n_init + self.n_step, self.input_size - self.n_step)
        elif num_iters < 2:
            self.n_init = max(self.n_step, self.n_init - self.n_step)

    def do_magic(self, xs: List[List[float]]) -> List[float]:
        # 1 approximate
        self.surrogate_function.train(self.train_set)

        items = [
            (x, self.surrogate_function(x), False)
            for x in xs
        ]

        # 2 rank
        all_items = rank_items(items)
        mu_items = all_items[:self.popsize]

        # 3 evaluate and add to train set
        for item in all_items[:self.n_init]:
            item = (item[0], self.objective_function(item[0]), True)
            # add to train set
            

        for i in range((self.input_size - self.n_init) / self.n_step):
            # 6 retrain and approximate
            # 7 rank
            # 8 if rank change
            if pass:
                self._update_n(i + 1)
                break
            else:
                # 12 evaluate and add to train set
                pass

        # return ys evaluated by real function
        return pass

    def __call__(self, xs: List[List[float]]) -> List[float]:
        # assert lambda
        if not len(xs) == self.input_size:
            raise ValueError(f'The number of provided points is different than expected. Expected {self.input_size}, got {len(xs)}.')

        if len(self.train_set) < self.input_size:
            ys = [self.objective_function(x) for x in xs]
            self.train_set.extend(list(zip(xs, ys)))
            return ys
        
        else:
            return self.do_magic(xs)

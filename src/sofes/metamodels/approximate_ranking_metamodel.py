"""
Approximate ranking metamodel based on lmm-CMA-ES.
"""

from typing import List, Tuple

from sofes.objective_functions import ObjectiveFunction, SurrogateObjectiveFunction


def rank_items(
    items: List[Tuple[List[float], float]]
) -> List[Tuple[List[float], float]]:
    return list(sorted(items, key=lambda x: x[1]))


class ApproximateRankingMetamodel:
    """Approximate ranking metamodel based on lmm-CMA-ES"""

    def __init__(
        self,
        input_size: int,
        popsize: int,
        objective_function: ObjectiveFunction,
        surrogate_function: SurrogateObjectiveFunction,
    ) -> None:
        """"""
        self.input_size = input_size
        self.popsize = popsize

        self.n_init = input_size
        self.n_step = max(1, input_size // 10)

        self.train_set = []

        self.objective_function = objective_function
        self.surrogate_function = surrogate_function

    def _update_n(self, num_iters: int) -> None:
        """"""
        if num_iters > 2:
            self.n_init = min(self.n_init + self.n_step, self.input_size - self.n_step)
        elif num_iters < 2:
            self.n_init = max(self.n_step, self.n_init - self.n_step)

    def approximate(self, xs: List[List[float]], retrain: bool=False) -> List[Tuple[List[float], float]]:
        """"""
        if retrain:
            self.surrogate_function.train(self.train_set)
        return [(x, self.surrogate_function(x)) for x in xs]
    
    def evaluate(self, xs: List[List[float]]) -> List[Tuple[List[float], float]]:
        ''''''
        result = [(x, self.objective_function(x)) for x in xs]
        self.train_set.extend(result)
        return result

    def get_log(self) -> List[float]:
        ''''''
        return [y for _, y in self.train_set]

    def do_call(self, xs: List[List[float]]) -> List[float]:
        """"""

        # 1 approximate
        items = self.approximate(xs, retrain=True)

        # 2 rank
        items_ranked = rank_items(items)
        items_mu_ranked = items_ranked[: self.popsize]

        # 3 evaluate and add to train set
        items_ranked[:self.n_init] = self.evaluate([x for x, _ in items_ranked[: self.n_init]])

        num_iter = 0
        for _ in range((self.input_size - self.n_init) // self.n_step):
            num_iter += 1
            # 6 retrain and approximate
            new_items = self.approximate(xs, retrain=True)

            # 7 rank
            new_items_ranked = rank_items(new_items)
            new_items_mu_ranked = new_items_ranked[: self.popsize]

            # 8 if rank change
            if [l[0] == r[0] for l, r in zip(new_items_mu_ranked, items_mu_ranked)]:
                break
            else:
                counter = 0
                to_eval = []
                for x in xs:
                    for tmp_x, _ in self.train_set:
                        if not x == tmp_x:
                            counter += 1
                            to_eval.append(x)
                            break
                    if counter >= self.n_step:
                        break
                items_mu_ranked = new_items_mu_ranked
                self.evaluate(to_eval)

        self._update_n(num_iter)

        return items_mu_ranked

    def __call__(self, xs: List[List[float]]) -> List[Tuple[List[float], float]]:
        """"""
        if not len(xs) == self.input_size:
            raise ValueError(
                f"The number of provided points is different than expected. Expected {self.input_size}, got {len(xs)}."
            )

        if len(self.train_set) < self.input_size:
            return self.evaluate(xs)
        else:
            return self.do_call(xs)

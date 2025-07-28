"""
KNN-IPOP-CMA-ES optimizer. IPOP-CMA-ES is enhanced with a KNN metamodel
similar to the one from LMM-CMA-ES.
"""

from ..data_classes import Bounds, PointList
from ..functions import ObjectiveFunction
from ..functions.surrogate import KNNSurrogateObjectiveFunction
from ..metamodels import ApproximateRankingMetamodel
from .cma_es import CmaEs
from .optimizer import Optimizer


class KnnIpopCmaEs(CmaEs):
    """
    KNN-IPOP-CMA-ES optimizer: CMA-ES with increasing population restarts and with KNN
    metamodel similar to LMM-CMA-ES.
    """

    def __init__(
        self,
        population_size: int,
        num_neighbors: int,
        buffer_size: int,
    ):
        """
        Class constructor.

        Args:
            population_size (int): Starting size of the population.
            num_neighbors (int): Number of neighbors used by KNN metamodel.
            buffer_size (int): Number of last evaluated points provided to KNN metamodel.
        """
        # Skipping super().__init__ and calling grandparent init instead.
        # pylint: disable=super-init-not-called, non-parent-init-called
        Optimizer.__init__(
            self,
            f"knn{num_neighbors}b{buffer_size}-ipop-cma-es",
            population_size,
            {
                "num_neighbors": num_neighbors,
                "buffer_size": buffer_size,
            },
        )

    # pylint: disable=duplicate-code
    def optimize(
        self,
        function: ObjectiveFunction,
        bounds: Bounds,
        call_budget: int,
        tolerance: float,
        target: float = 0.0,
    ) -> PointList:
        current_population_size = self.metadata.population_size

        metamodel = ApproximateRankingMetamodel(
            self.metadata.population_size,
            self.metadata.population_size // 2,
            function,
            KNNSurrogateObjectiveFunction(
                self.metadata.hyperparameters["num_neighbors"]
            ),
            buffer_size=self.metadata.hyperparameters["buffer_size"],
        )

        while not self._stop_external(
            metamodel.get_log(),
            current_population_size,
            call_budget,
            target,
            tolerance,
        ):
            es = self._spawn_cmaes(
                bounds,
                function.metadata.dim,
                current_population_size,
                len(bounds) / 2,
            )

            while not self._stop(
                es,
                metamodel.get_log(),
                current_population_size,
                call_budget,
                target,
                tolerance,
            ):
                solutions = PointList.from_list(es.ask())

                if (
                    len(metamodel.train_set)
                    < self.metadata.hyperparameters["num_neighbors"]
                ):
                    xy_pairs = metamodel.evaluate(solutions)
                else:
                    metamodel.adapt(solutions)
                    xy_pairs = metamodel(solutions)

                x, y = xy_pairs.pairs()
                es.tell(x, y)

            current_population_size *= 2
            metamodel.population_size *= 2
            metamodel.mu *= 2

        return metamodel.get_log()

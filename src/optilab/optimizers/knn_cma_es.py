"""
KNN-CMA-ES optimizer. CMA-ES is enhanced with a KNN metamodel similar to the one from LMM-CMA-ES.
"""

from ..data_classes import Bounds, PointList
from ..functions import ObjectiveFunction
from ..functions.surrogate import KNNSurrogateObjectiveFunction
from ..metamodels import ApproximateRankingMetamodel
from .cma_es import CmaEs
from .optimizer import Optimizer


class KnnCmaEs(CmaEs):
    """
    KNN-CMA-ES optimizer. CMA-ES is enhanced with a KNN metamodel similar
    to the one from LMM-CMA-ES.
    """

    def __init__(
        self,
        population_size: int,
        sigma0: float,
        num_neighbors: int,
        buffer_size: int,
    ):
        """
        Class constructor.

        Args:
            population_size (int): Size of the population.
            sigma0 (float): Starting value of the sigma,
            num_neighbors (int): Number of neighbors used by KNN metamodel.
            buffer_size (int): Number of last evaluated points provided to KNN metamodel.
        """
        # Skipping super().__init__ and calling grandparent init instead.
        # pylint: disable=super-init-not-called, non-parent-init-called
        Optimizer.__init__(
            self,
            f"knn{num_neighbors}b{buffer_size}-cma-es",
            population_size,
            {
                "sigma0": sigma0,
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
        """
        Run a single optimization of provided objective function.

        Args:
            function (ObjectiveFunction): Objective function to optimize.
            bounds (Bounds): Search space of the function.
            call_budget (int): Max number of calls to the objective function.
            tolerance (float): Tolerance of y value to count a solution as acceptable.
            target (float): Objective function value target, default 0.

        Returns:
            PointList: Results log from the optimization.
        """
        metamodel = ApproximateRankingMetamodel(
            self.metadata.population_size,
            self.metadata.population_size // 2,
            function,
            KNNSurrogateObjectiveFunction(
                self.metadata.hyperparameters["num_neighbors"]
            ),
            buffer_size=self.metadata.hyperparameters["buffer_size"],
        )

        es = self._spawn_cmaes(
            bounds,
            function.metadata.dim,
            self.metadata.population_size,
            self.metadata.hyperparameters["sigma0"],
        )

        while not self._stop(
            es,
            metamodel.get_log(),
            self.metadata.population_size,
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

        return metamodel.get_log()

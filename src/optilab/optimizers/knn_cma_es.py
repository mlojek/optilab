"""
KNN-CMA-ES optimizer. CMA-ES is enhanced with a KNN metamodel similar to the one from LMM-CMA-ES.
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments, duplicate-code

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

    def __init__(self, population_size: int, sigma0: float, num_neighbors: int):
        """
        Class constructor.

        Args:
            population_size (int): Size of the population.
            sigma0 (float): Starting value of the sigma,
            num_neighbors (int): Number of neighbors used by KNN metamodel.
        """
        # Skipping super().__init__ and calling grandparent init instead.
        # pylint: disable=super-init-not-called, non-parent-init-called
        Optimizer.__init__(
            self,
            "knn-cma-es",
            population_size,
            {"sigma0": sigma0, "num_neighbors": num_neighbors},
        )

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
        )

        es = self._spawn_cmaes(bounds, function.dim)

        while (
            metamodel.get_log().best_y() > tolerance
            and len(metamodel.get_log()) <= call_budget
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

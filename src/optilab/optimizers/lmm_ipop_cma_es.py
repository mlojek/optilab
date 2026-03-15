"""
LMM-IPOP-CMA-ES optimizer: IPOP-CMA-ES with local polynomial regression metamodel.
It's a fusion of IPOP-CMA-ES and LMM-CMA-ES.
"""

from ..data_classes import Bounds, PointList
from ..functions import ObjectiveFunction
from ..functions.surrogate import LocallyWeightedPolynomialRegression
from ..metamodels import ApproximateRankingMetamodel
from .cma_es import CmaEs
from .optimizer import Optimizer


class LmmIpopCmaEs(CmaEs):
    """
    LMM-IPOP-CMA-ES optimizer: IPOP-CMA-ES with local polynomial regression metamodel.
    It's a fusion of IPOP-CMA-ES and LMM-CMA-ES.
    """

    def __init__(
        self,
        population_size: int,
        polynomial_dim: int,
    ):
        """
        Class constructor.

        Args:
            population_size: Size of the population.
            polynomial_dim: Dimension of the polynomial regression.
        """
        # Skipping super().__init__ and calling grandparent init instead.
        # pylint: disable=super-init-not-called, non-parent-init-called
        Optimizer.__init__(
            self,
            "lmm-ipop-cma-es",
            population_size,
            {
                "polynomial_dim": polynomial_dim,
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
            function: Objective function to optimize.
            bounds: Search space of the function.
            call_budget: Max number of calls to the objective function.
            tolerance: Tolerance of y value to count a solution as acceptable.
            target: Objective function value target, default 0.

        Returns:
            Results log from the optimization.
        """
        current_population_size = self.metadata.population_size

        num_neighbors = function.metadata.dim * (function.metadata.dim + 3) + 2

        metamodel = ApproximateRankingMetamodel(
            self.metadata.population_size,
            self.metadata.population_size // 2,
            function,
            LocallyWeightedPolynomialRegression(
                self.metadata.hyperparameters["polynomial_dim"], num_neighbors
            ),
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
                metamodel.surrogate_function.set_covariance_matrix(es.C)
                metamodel.adapt(solutions)
                xy_pairs = metamodel(solutions)
                x, y = xy_pairs.pairs()
                es.tell(x, y)

            current_population_size *= 2
            metamodel.population_size *= 2
            metamodel.mu *= 2
            metamodel.init_n()

        return metamodel.get_log()

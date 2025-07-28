"""
CMA-ES optimizer: Covariance Matrix Adaptation Evolution Strategy.
"""

import cma

from ..data_classes import Bounds, PointList
from ..functions import ObjectiveFunction
from .optimizer import Optimizer


class CmaEs(Optimizer):
    """
    CMA-ES optimizer: Covariance Matrix Adaptation Evolution Strategy.
    """

    def __init__(
        self,
        population_size: int,
        sigma0: float,
    ):
        """
        Class constructor.

        Args:
            population_size (int): Size of the population.
            sigma0 (float): Starting value of the sigma,
        """
        super().__init__(
            "cma-es",
            population_size,
            {"sigma0": sigma0},
        )

    @staticmethod
    def _stop_internal(
        es: cma.CMAEvolutionStrategy,
    ) -> bool:
        """
        Check an instance of CMA-ES optimizer for internal stop criteria.

        Args:
            es (cma.CMAEvolutionStrategy): An instance of CMA-ES optimizer.

        Returns:
            bool: If true, the internal stop criteria of CMA-ES have been reached and optimization
                should be ended.
        """
        return es.stop()

    @staticmethod
    def _spawn_cmaes(
        bounds: Bounds,
        dim: int,
        population_size: int,
        sigma0: float,
    ) -> cma.CMAEvolutionStrategy:
        """
        Create a new instance of cma optimizer.

        Args:
            bounds (Bounds): The bounds of the search area.
            dim (int): The dimensionality of the search area.

        Returns:
            cma.CMAEvolutionStrategy: A new cma optimizer instance.
        """
        return cma.CMAEvolutionStrategy(
            bounds.random_point(dim).x,
            sigma0,
            {
                "popsize": population_size,
                "bounds": bounds.to_list(),
                "verbose": -9,
            },
        )

    def _stop(
        self,
        es: cma.CMAEvolutionStrategy,
        log: PointList,
        population_size: int,
        call_budget: int,
        target: float,
        tolerance: float,
    ) -> bool:
        """
        Decide if the optimization should be stopped.

        Args:
            es (cma.CMAEvolutionStrategy): CMA-ES instance.
            log (PointList): Results log.
            population_size (int): The size of the population in one generation.
            call_budget (int): Maximum number of optimized function calls.
            target (float): Global minimum value of the optimized function.
            tolerance (float): Tolerated error value of the optimization.

        Returns:
            bool: True if the optimization should be stopped.
        """
        return self._stop_internal(es) or self._stop_external(
            log,
            population_size,
            call_budget,
            target,
            tolerance,
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
        es = self._spawn_cmaes(
            bounds,
            function.metadata.dim,
            self.metadata.population_size,
            self.metadata.hyperparameters["sigma0"],
        )

        res_log = PointList(points=[])

        while not self._stop(
            es,
            res_log,
            self.metadata.population_size,
            call_budget,
            target,
            tolerance,
        ):
            solutions = PointList.from_list(es.ask())
            results = PointList(points=[function(x) for x in solutions.points])
            res_log.extend(results)
            x, y = results.pairs()
            es.tell(x, y)

        return res_log

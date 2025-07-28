"""
IPOP-CMA-ES optimizer: CMA-ES with increasing population restarts.
"""

from ..data_classes import Bounds, PointList
from ..functions import ObjectiveFunction
from .cma_es import CmaEs
from .optimizer import Optimizer


class IpopCmaEs(CmaEs):
    """
    IPOP-CMA-ES optimizer: CMA-ES with increasing population restarts.
    """

    def __init__(self, population_size: int):
        # Skipping super().__init__ and calling grandparent init instead.
        # pylint: disable=super-init-not-called, non-parent-init-called
        Optimizer.__init__(
            self,
            "ipop-cma-es",
            population_size,
            {},
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
        res_log = PointList(points=[])

        while not self._stop_external(
            res_log,
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
                res_log,
                current_population_size,
                call_budget,
                target,
                tolerance,
            ):
                solutions = PointList.from_list(es.ask())
                results = PointList(points=[function(x) for x in solutions.points])
                res_log.extend(results)
                x, y = results.pairs()
                es.tell(x, y)

            current_population_size *= 2

        return res_log

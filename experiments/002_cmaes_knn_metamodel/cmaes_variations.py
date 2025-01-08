"""
Variations of CMA-ES encapsulated in easy to call functions.
"""

# pylint: disable=too-many-arguments, too-many-locals
import cma

from optilab.data_classes import Bounds, PointList
from optilab.functions import ObjectiveFunction
from optilab.functions.surrogate import (
    LocallyWeightedPolynomialRegression,
    SurrogateObjectiveFunction,
    KNNSurrogateObjectiveFunction
)
from optilab.metamodels import ApproximateRankingMetamodel


def cma_es(
    function: ObjectiveFunction,
    population_size: int,
    call_budget: int,
    bounds: Bounds,
    *,
    sigma0: float = 1,
    target: float = 0.0,
    tolerance: float = 1e-8,
) -> PointList:
    """
    Run optimization with regular CMA-ES.
    """

    x0 = bounds.random_point(function.dim).x

    res_log = PointList(points=[])

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "popsize": population_size,
            "bounds": bounds.to_list(),
            "maxfevals": call_budget,
            "ftarget": target,
            "verbose": -9,
            "tolfun": tolerance,
        },
    )

    while not es.stop():
        solutions = PointList.from_list(es.ask())
        results = PointList(points=[function(x) for x in solutions.points])
        res_log.extend(results)
        x, y = results.pairs()
        es.tell(x, y)

    return res_log


def knn_cma_es(
    function: ObjectiveFunction,
    population_size: int,
    call_budget: int,
    bounds: Bounds,
    *,
    sigma0: float = 1,
    target: float = 0.0,
    tolerance: float = 1e-8,
    num_neighbors: int = 5,
) -> PointList:
    """
    Run optimization with KNN-CMA-ES
    """

    metamodel = ApproximateRankingMetamodel(
        population_size,
        population_size // 2,
        function,
        KNNSurrogateObjectiveFunction(num_neighbors),
    )

    x0 = bounds.random_point(function.dim).x

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "popsize": population_size,
            "bounds": bounds.to_list(),
            "maxfevals": call_budget,
            "ftarget": target,
            "verbose": -9,
            "tolfun": tolerance,
        },
    )

    while (
        metamodel.get_log().best_y() > tolerance and len(metamodel.get_log()) <= call_budget
    ):
        solutions = PointList.from_list(es.ask())
        metamodel.adapt(solutions)
        xy_pairs = metamodel(solutions)
        x, y = xy_pairs.pairs()
        es.tell(x, y)

    return metamodel.get_log()

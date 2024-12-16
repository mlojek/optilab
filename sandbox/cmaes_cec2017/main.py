"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

# pylint: disable=too-many-arguments
# pylint: disable=import-error
# pylint: disable=too-many-positional-arguments, too-many-locals

from typing import Tuple

import cma
import numpy as np
from tqdm import tqdm

from optilab.data_classes import PointList
from optilab.functions import ObjectiveFunction
from optilab.functions.benchmarks.cec2017_objective_function import (
    CEC2017ObjectiveFunction,
)
from optilab.functions.surrogate import KNNSurrogateObjectiveFunction
from optilab.metamodels import ApproximateRankingMetamodel
from optilab.plotting import plot_ecdf_curves


def run_cmaes_on_cec(
    function: ObjectiveFunction,
    dims: int,
    population_size: int,
    call_budget: int,
    bounds: Tuple[float, float],
    sigma0: float,
    armknn_metamodel: bool = False,
    num_neighbours: int = 5,
    debug: bool = False,
):
    """
    TODO
    """
    if armknn_metamodel:
        metamodel = ApproximateRankingMetamodel(
            population_size,
            population_size // 2,
            function,
            KNNSurrogateObjectiveFunction(num_neighbours),
        )

    x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=dims)

    res_log = PointList(points=[])

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "popsize": population_size,
            "bounds": [bounds[0], bounds[1]],
            "maxfevals": call_budget,
            "ftarget": 0,
            "verbose": -9,
            "tolfun": 1e-8,
        },
    )

    sigma_best_plot_data = []

    while not es.stop():
        if armknn_metamodel:
            solutions = PointList.from_list(es.ask())
            metamodel.adapt(solutions)
            xy_pairs = metamodel(solutions)
            x, y = xy_pairs.pairs()
            es.tell(x, y)
        else:
            solutions = PointList.from_list(es.ask())
            results = PointList(points=[function(x) for x in solutions.points])
            res_log.extend(results)
            x, y = results.pairs()
            es.tell(x, y)

        sigma_best_plot_data.append(
            (es.countevals, np.log10(es.best.f), np.log10(es.sigma))
        )

    if debug:
        print(dict(es.result._asdict()))
        # plt.clf()
        # _, axes = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column
        # axes[0].plot(
        #     [x[0] for x in sigma_best_plot_data], [x[1] for x in sigma_best_plot_data]
        # )
        # axes[0].set_title("bext_value")
        # axes[1].plot(
        #     [x[0] for x in sigma_best_plot_data], [x[2] for x in sigma_best_plot_data]
        # )
        # axes[1].set_title("sigma")
        # plt.tight_layout()
        # plt.show()

    if armknn_metamodel:
        return metamodel.get_log()
    return res_log


if __name__ == "__main__":
    # hyperparams:
    DIM = 10
    POPSIZE = 40
    NUM_RUNS = 5
    CALL_BUDGET = 1e6
    BOUNDS = (-100, 100)

    # func = SphereFunction(DIM)
    func = CEC2017ObjectiveFunction(1, DIM)
    # func = BentCigarFunction(DIM)
    logs_vanilla = [
        run_cmaes_on_cec(
            func,
            DIM,
            POPSIZE,
            CALL_BUDGET,
            BOUNDS,
            10,
            debug=False,
        )
        for _ in tqdm(range(NUM_RUNS), unit="runs")
    ]

    logs_armknn5 = [
        run_cmaes_on_cec(
            func,
            DIM,
            POPSIZE,
            CALL_BUDGET,
            BOUNDS,
            10,
            armknn_metamodel=True,
            debug=True,
        )
        for _ in tqdm(range(NUM_RUNS))
    ]

    plot_ecdf_curves({"vanilla": logs_vanilla, "knn": logs_armknn5}, n_dimensions=DIM)

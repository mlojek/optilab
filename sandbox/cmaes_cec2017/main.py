"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

# pylint: disable=too-many-arguments
# pylint: disable=import-error

from typing import List, Tuple

import cma
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from optilab.data_classes import ExperimentMetadata, ExperimentResults
from optilab.functions import ObjectiveFunction
from optilab.functions.benchmarks.cec2017_objective_function import (
    CEC2017ObjectiveFunction,
)
from optilab.functions.surrogate import KNNSurrogateObjectiveFunction
from optilab.metamodels import ApproximateRankingMetamodel

# from sofes.objective_functions.unimodal.sphere_function import SphereFunction
from optilab.plotting import plot_ecdf_curves


def run_cmaes_on_cec(  # pylint: disable=too-many-positional-arguments, too-many-locals
    function: ObjectiveFunction,
    dims: int,
    population_size: int,
    call_budget: int,
    bounds: Tuple[float, float],
    sigma0: float,
    armknn_metamodel: bool = False,
    num_neighbours: int = 5,
    debug: bool = False,
) -> List[float]:
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
    # x0 = np.random.uniform(-1e-3, 1e-3, size=dims)
    # x0 = np.zeros((10))

    res_log = []

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
            solutions = [x.tolist() for x in es.ask()]
            metamodel.adapt(solutions)
            xy_pairs = metamodel(solutions)
            x, y = zip(*xy_pairs)
            es.tell(x, y)
        else:
            solutions = es.ask()
            y = [function(x) for x in solutions]
            res_log.extend(y)
            es.tell(solutions, y)

        sigma_best_plot_data.append(
            (es.countevals, np.log10(es.best.f), np.log10(es.sigma))
        )

    if debug:
        print(dict(es.result._asdict()))
        plt.clf()
        _, axes = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column
        axes[0].plot(
            [x[0] for x in sigma_best_plot_data], [x[1] for x in sigma_best_plot_data]
        )
        axes[0].set_title("bext_value")
        axes[1].plot(
            [x[0] for x in sigma_best_plot_data], [x[2] for x in sigma_best_plot_data]
        )
        axes[1].set_title("sigma")
        plt.tight_layout()
        plt.show()

    if armknn_metamodel:
        return metamodel.get_log()
    return res_log


if __name__ == "__main__":
    # hyperparams:
    DIM = 10
    POPSIZE = 40
    NUM_RUNS = 5

    metadata = ExperimentMetadata(
        method_name="cmaes",
        method_hyperparameters={
            "sigma0": 10,
            "popsize": "4*dim",
            "call_budget": 1e6,
            "bounds": (-100, 100),
        },
        metamodel_name="approximate_ranking_metamodel_knn",
        metamodel_hyperparameters={"num_neighbours": 5},
        benchmark_name="cec2017",
    )
    results = ExperimentResults(metadata)

    # func = SphereFunction(DIM)
    func = CEC2017ObjectiveFunction(1, DIM)
    # func = BentCigarFunction(DIM)
    logs_vanilla = [
        run_cmaes_on_cec(
            func,
            DIM,
            POPSIZE,
            metadata.method_hyperparameters["call_budget"],
            metadata.method_hyperparameters["bounds"],
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
            metadata.method_hyperparameters["call_budget"],
            metadata.method_hyperparameters["bounds"],
            10,
            armknn_metamodel=True,
            debug=True,
        )
        for _ in tqdm(range(NUM_RUNS))
    ]

    plot_ecdf_curves(
        {"vanilla": logs_vanilla, "armknn5": logs_armknn5}, n_dimensions=DIM
    )

"""
CMA-ES on CEC2017 with metamodel from lmm-CMA-ES also known as Approximate Ranking Metamodel
"""

# pylint: disable=too-many-arguments, too-many-positional-arguments


from tqdm import tqdm

from sofes.data_classes import ExperimentMetadata, ExperimentResults
from sofes.objective_functions import (
    CEC2017ObjectiveFunction,
    KNNSurrogateObjectiveFunction
)
from sofes.metamodels import ApproximateRankingMetamodel

from typing import List, Tuple

import cma
import numpy as np


def run_cmaes_on_cec(
    cec_function: CEC2017ObjectiveFunction,
    dims: int,
    population_size: int,
    call_budget: int,
    bounds: Tuple[float, float],
    tolerance: float,
    target: float,
    sigma0: float,
) -> List[float]:
    """
    Runs the CMA-ES algorithm on a CEC function.

    :param cec_function: The CEC benchmark function to optimize.
    :param dims: The dimensionality of the problem.
    :param population_size: The population size for the CMA-ES algorithm.
    :param call_budget: The maximum number of function evaluations.
    :param bounds: The lower and upper bounds for the search space.
    :param sigma0: The starting value of the sigma parameter.
    :return: The list of error values achieved through the optimization
    """
    metamodel = ApproximateRankingMetamodel(population_size*2, population_size, cec_function, KNNSurrogateObjectiveFunction(5))

    x0 = np.random.uniform(low=bounds[0], high=bounds[1], size=dims)

    res_log = []

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        {
            "popsize": population_size,
            "bounds": [bounds[0], bounds[1]],
            "maxfevals": call_budget,
            "tolfun": tolerance,
            "ftarget": target,
            "verbose": -1,
        },
    )

    while not es.stop():
        solutions = es.ask(population_size*2)
        xy_pairs = metamodel(solutions)
        x, y = zip(*xy_pairs)
        res_log.extend(y)
        es.tell(x, y)

    return res_log


MAX_FES = 1e6
BOUNDS = [-100, 100]
SIGMA0 = 10
DIMS = [10]
POPSIZE_PER_DIM = 4
TOLERANCE = 1e-8
NUM_RUNS = 5
F_NUMS = [1, 2, 3, 4, 5]


if __name__ == "__main__":
    metadata = ExperimentMetadata(
        "cmaes",
        {"sigma0": SIGMA0, "popsize_per_dim": 4},
        "",
        {"dummy": 42},
        "cec2017",
        "",
        "",
    )
    results = ExperimentResults(metadata)

    for n in F_NUMS:
        for dimension in DIMS:
            print(f"{n} dim {dimension}")
            func = CEC2017ObjectiveFunction(n, dimension)
            maxes = [
                run_cmaes_on_cec(
                    func,
                    dimension,
                    POPSIZE_PER_DIM * dimension,
                    MAX_FES * dimension,
                    BOUNDS,
                    TOLERANCE,
                    0,
                    SIGMA0,
                )
                for _ in tqdm(range(NUM_RUNS))
            ]

            results.add_data(func.name, dimension, maxes)

    print(results.print_stats())
    results.save_to_json("cmaes_vanilla_cec2017.json")
    results.save_stats("cmaes_vanilla_cec2017.csv")
    results.plot_ecdf_curve(dim=10, savepath="cmaes_vanilla_cec2017_ecdf_10.png")
    results.plt_box_plot(savepath="cmaes_vanilla_cec2017_box.png")

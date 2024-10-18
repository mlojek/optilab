"""
Benchmarking CMA-ES algorithm on CEC 2017
"""
import json
from typing import Tuple

import cma
import numpy as np
from cec2017.functions import all_functions
from tqdm import tqdm
import matplotlib.pyplot as plt

from sofes.plotting.convergence_curve import plot_convergence_curve
from sofes.plotting.ecdf_curve import plot_ecdf_curves
from sofes.plotting.box_plot import plot_box_plot

MAX_FES = 1e4
BOUNDS = [-100, 100]
SIGMA0 = 10
DIMS = [10]
POPSIZE_PER_DIM = 4
TOLERANCE = 1e-8
NUM_RUNS = 10


def run_cmaes_on_cec(
    cec_function: callable,
    dims: int,
    population_size: int,
    call_budget: int,
    bounds: Tuple[float, float],
    tolerance: float,
    target: float,
    sigma0: float
):
    """
    Runs the CMA-ES algorithm on a CEC function.

    :param cec_function: The CEC benchmark function to optimize.
    :param dims: The dimensionality of the problem.
    :param population_size: The population size for the CMA-ES algorithm.
    :param call_budget: The maximum number of function evaluations.
    :param bounds: The lower and upper bounds for the search space.
    :param sigma0: The starting value of the sigma parameter.
    :return: The best solution found by the CMA-ES algorithm.
    """
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
        solutions = es.ask()
        fitness_values = [cec_function([x.tolist()])[0] for x in solutions]
        res_log.extend(fitness_values)
        es.tell(solutions, fitness_values)
        es.logger.add()

    res_log = [x - target for x in res_log]

    return res_log, es.result.fbest - target


if __name__ == "__main__":
    functions = [
        (f"f{i+1}", func, (i + 1) * 100) for i, func in enumerate(all_functions)
    ]

    results = []
    boxplot = {}

    for name, function, target in functions[:3]:
        for dimension in DIMS:
            print(f"{name} dim {dimension}")
            maxes = [
                run_cmaes_on_cec(
                    function,
                    dimension,
                    POPSIZE_PER_DIM * dimension,
                    MAX_FES * dimension,
                    BOUNDS,
                    TOLERANCE,
                    target,
                    SIGMA0
                )
                for _ in tqdm(range(NUM_RUNS))
            ]

            boxplot[name] = [x[1] for x in maxes]

            max_values = [x[1] for x in maxes]

            results.append(
                {
                    "name": name,
                    "dimensions": dimension,
                    "results": max_values,
                    "mean": np.mean(max_values),
                    "stdev": np.std(max_values),
                    "min": np.min(max_values),
                    "max": np.max(max_values),
                }
            )
    
    plot_box_plot(boxplot)

    with open("results.json", "w") as results_handle:
        json.dump(results, results_handle, indent=4)

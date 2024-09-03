from typing import Tuple
import json

import cma
import numpy as np
from tqdm import tqdm

from cec2017.functions import all_functions


functions = [
    (f'f{i}', func, i * 100)
    for i, func in zip(range(1, 30), all_functions)
]

MAX_FES = 1e4
BOUNDS = [-100, 100]
DIMS = [10, 30]
TOLERANCE = 1e-8
NUM_RUNS = 51


def run_cmaes_on_cec(
    cec_function: callable,
    dims: int,
    population_size: int,
    call_budget: int,
    constraints: Tuple[float, float],
    tolerance: float,
    target: float
):
    """
    Runs the CMA-ES algorithm on a CEC function.
    
    :param cec_function: The CEC benchmark function to optimize.
    :param dims: The dimensionality of the problem.
    :param population_size: The population size for the CMA-ES algorithm.
    :param call_budget: The maximum number of function evaluations.
    :param constraints: The lower and upper bounds for the search space.
    :return: The best solution found by the CMA-ES algorithm.
    """
    x0 = np.random.uniform(low=constraints[0], high=constraints[1], size=dims)
    sigma0 = 0.3
    
    es = cma.CMAEvolutionStrategy(
        x0, 
        sigma0, 
        {
            'popsize': population_size, 
            'bounds': [constraints[0], constraints[1]], 
            'maxfevals': call_budget,
            'tolfun': tolerance,
            'ftarget': target,
            'verbose': -1
        }
    )

    while not es.stop():
        solutions = es.ask()
        fitness_values = [cec_function([x.tolist()])[0] for x in solutions]

        if es.result.evaluations >= call_budget:
            break

        es.tell(solutions, fitness_values)

    return es.result.fbest


if __name__ == "__main__":
    results = []

    assert len(functions) == 29

    for name, function, target in functions:
        for dimension in DIMS:
            print(f"{name} dim {dimension}")
            maxes = [
                run_cmaes_on_cec(
                    function,
                    dimension,
                    4 * dimension,
                    MAX_FES * dimension,
                    BOUNDS,
                    TOLERANCE,
                    target
                )
                for _ in tqdm(range(NUM_RUNS))
            ]
            
            results.append({
                'name': name,
                'dimensions': dimension,
                'results': maxes,
                'mean': np.mean(maxes),
                'stdev': np.std(maxes),
                'min': np.min(maxes),
                'max': np.max(maxes)
            })
        break

    with open("results.json", "w") as results_handle:
        json.dump(results, results_handle, indent=4)

"""
Function running CMA-ES algorithm on CEC 2017 functions
"""

from typing import List, Tuple

import cma
import numpy as np


def run_cmaes_on_cec(
    cec_function: callable,
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
    assert min(res_log) == es.result.fbest - target

    return res_log

"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

# pylint: disable=import-error

from cmaes_variations import arm_cma_es, cma_es, lmm_cma_es
from tqdm import tqdm

from optilab.data_classes import Bounds
from optilab.functions.surrogate import (
    KNNSurrogateObjectiveFunction,
    PolynomialRegression,
)
from optilab.functions.unimodal import SphereFunction
from optilab.metamodels import IEPolationSurrogate
from optilab.plotting import plot_ecdf_curves

if __name__ == "__main__":
    # hyperparams:
    DIM = 2
    POPSIZE = 4 * DIM
    NUM_RUNS = 21
    CALL_BUDGET = 1e4 * DIM

    # optimized problem
    BOUNDS = Bounds(-100, 100)
    FUNC = SphereFunction(DIM)

    # perform optimization
    vanilla = [
        cma_es(FUNC, POPSIZE, CALL_BUDGET, BOUNDS)
        for _ in tqdm(range(NUM_RUNS), unit="runs")
    ]
    pr = [
        arm_cma_es(FUNC, POPSIZE, CALL_BUDGET, BOUNDS, PolynomialRegression(2))
        for _ in tqdm(range(NUM_RUNS), unit="runs")
    ]
    knn = [
        arm_cma_es(
            FUNC,
            POPSIZE,
            CALL_BUDGET,
            BOUNDS,
            KNNSurrogateObjectiveFunction(2 * DIM),
        )
        for _ in tqdm(range(NUM_RUNS), unit="runs")
    ]
    lmm = [
        lmm_cma_es(FUNC, POPSIZE, CALL_BUDGET, BOUNDS)
        for _ in tqdm(range(NUM_RUNS), unit="runs")
    ]
    ie = [
        arm_cma_es(
            FUNC,
            POPSIZE,
            CALL_BUDGET,
            BOUNDS,
            IEPolationSurrogate(
                KNNSurrogateObjectiveFunction(2 * DIM), PolynomialRegression(2)
            ),
        )
        for _ in tqdm(range(NUM_RUNS), unit="runs")
    ]

    # plot results
    plot_ecdf_curves(
        {
            "cma-es": vanilla,
            "pr2-cma-es": pr,
            "knn-cma-es": knn,
            "lmm-cma-es": lmm,
            "ie": ie,
        },
        n_dimensions=DIM,
        savepath="final_ecdf.png",
    )

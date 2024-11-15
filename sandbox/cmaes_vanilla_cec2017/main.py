"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

# pylint: disable=import-error

from run_cmaes_on_cec import run_cmaes_on_cec
from tqdm import tqdm

from sofes.data_classes import ExperimentMetadata, ExperimentResults
from sofes.objective_functions.cec2017_objective_function import (
    CEC2017ObjectiveFunction,
)

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

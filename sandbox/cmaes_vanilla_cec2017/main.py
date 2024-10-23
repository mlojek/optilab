"""
Benchmarking CMA-ES algorithm on CEC 2017
"""

import json

from cec2017.functions import all_functions
from run_cmaes_on_cec import run_cmaes_on_cec
from tqdm import tqdm

from sofes.data_classes.experiment_results import ExperimentResults
from sofes.plotting.box_plot import plot_box_plot
from sofes.plotting.convergence_curve import plot_convergence_curve
from sofes.plotting.ecdf_curve import plot_ecdf_curves

MAX_FES = 1e4
BOUNDS = [-100, 100]
SIGMA0 = 10
DIMS = [10]
POPSIZE_PER_DIM = 4
TOLERANCE = 1e-8
NUM_RUNS = 10


if __name__ == "__main__":
    functions = [
        (f"cec2017_f{i+1}", func, (i + 1) * 100) for i, func in enumerate(all_functions)
    ]

    results = ExperimentResults()

    for name, function, target in functions[:5]:
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
                    SIGMA0,
                )
                for _ in tqdm(range(NUM_RUNS))
            ]

            results.add_data(name, dimension, maxes)

    results.save_to_json("cmaes_vanilla_cec2017.json")
    results.plot_ecdf_curve(dim=10, savepath="cmaes_vanilla_cec2017_ecdf_10.png")
    results.plt_box_plot(savepath="cmaes_vanilla_cec2017_box.png")

"""
Plotting and calculating ECDF curves.
"""

# pylint: disable=too-many-locals

import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..data_classes import PointList
from .convergence_curve import convergence_curve


def _ecdf_thresholding(
    log: List[float],
    thresholds: List[float],
    n_dimensions: int,
    extend_to_len: int = None,
) -> Tuple[List[float], List[float]]:
    """
    Perform thresholding of a log with a given list of thresholds.
    The resulting y is the number of thresholds achieved by the log items.

    Args:
        log (PointList): Error log of a optimization function.
        thresholds (List[float]): ECDF value thresholds.
        n_dimensions (int): Dimensionality of optimized function.

    Returns:
        Tuple[List[float], List[float]]: x and y values for the curve.
    """
    y = [np.sum(thresholds >= item) / len(thresholds) for item in log]

    if extend_to_len:
        if extend_to_len < len(y):
            raise ValueError(
                "extend_to_len parameter is lower than lenght of the provided log"
            )
        while len(y) < extend_to_len:
            y.append(y[-1])

    x = [(i + 1) / n_dimensions for i in range(len(y))]

    return x, y


def ecdf_curve(
    data: Dict[str, List[PointList]],
    n_dimensions: int = 1,
    n_thresholds: int = 100,
    allowed_error: float = 1e-8,
) -> Dict[str, Tuple[List[float], List[float]]]:
    """
    Calculate ECDF curves.

    Args:
        data (Dict[str, PointList]): Lists of value logs indexed by method name.
        n_dimensions (int): Dimensionality of the solved problem.
        n_thresholds (int): Number of ECDF thresholds.
        allowed_error (float): Tolerable error value, used as the last threshold.

    Returns:
        Dict[str, Tuple[List[float], List[float]]]: x, y plot points for each method.
    """
    processed_logs = {}
    log_lengths = {}
    all_last_items = []

    for method, logs in data.items():
        processed_logs[method] = []
        max_len = 0
        for log in logs:
            new_log = convergence_curve(log)
            max_len = max(max_len, len(new_log))
            processed_log = [math.log10(v + 1e-10) for v in new_log]
            processed_logs[method].append(processed_log)
            all_last_items.append(processed_log[-1])
            log_lengths[method] = max_len

    low_value = math.log10(allowed_error)
    high_value = max(all_last_items)

    thresholds = np.linspace(low_value, high_value, n_thresholds + 1)[1:]

    ecdf_data = {}
    for method, logs in processed_logs.items():
        ecdf_ys = []
        ecdf_x = []

        for log in logs:
            x, y = _ecdf_thresholding(
                log, thresholds, n_dimensions, log_lengths[method]
            )
            ecdf_x = x
            ecdf_ys.append(y)

        ecdf_avg = np.mean(ecdf_ys, axis=0)
        ecdf_data[method] = (ecdf_x, ecdf_avg)

    return ecdf_data


def plot_ecdf_curves(
    data: Dict[str, PointList],
    n_dimensions: int = 1,
    n_thresholds: int = 100,
    allowed_error: float = 1e-8,
    savepath: str = None,
) -> None:
    """
    Calculate and plot ECDF curves.

    Args:
        data (Dict[str, List[List[float]]]): Lists of value logs for every method.
        n_dimensions (int): Dimensionality of the optimized function.
        n_thresholds (int): Number of ECDF thresholds.
        allowed_error (float): Tolerable error value, used as the last threshold.
        savepath (str): Path to save the plot, optional.
    """
    plt.clf()

    ecdf_data = ecdf_curve(data, n_dimensions, n_thresholds, allowed_error)

    for method, (x, y) in ecdf_data.items():
        plt.plot(x, y, label=method)

    plt.xlabel("Number of function evaluations divided by the number of dimensions.")
    plt.xscale("log")
    plt.ylabel("ECDF point pairs")
    plt.title("ECDF Curves")

    plt.legend()
    plt.grid(True)

    if savepath:
        plt.savefig(savepath)

    plt.show()

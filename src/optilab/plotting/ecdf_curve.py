"""
Plotting and calculating ECDF curves.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from ..data_classes import PointList
from .convergence_curve import convergence_curve


def _ecdf_thresholding(
    log: list[float],
    thresholds: np.ndarray,
    n_dimensions: int,
    extend_to_len: int | None = None,
) -> tuple[list[float], list[float]]:
    """
    Perform thresholding of a log with a given list of thresholds.
    The resulting y is the number of thresholds achieved by the log items.

    Args:
        log: Error log of a optimization function.
        thresholds: ECDF value thresholds.
        n_dimensions: Dimensionality of optimized function.

    Returns:
        x and y values for the curve.
    """
    y = [float(np.sum(thresholds >= item)) / len(thresholds) for item in log]

    if extend_to_len:
        if extend_to_len < len(y):
            raise ValueError(
                "extend_to_len parameter is lower than lenght of the provided log"
            )
        y.extend([y[-1]] * (extend_to_len - len(y)))

    x = [(i + 1) / n_dimensions for i in range(len(y))]

    return x, y


def ecdf_curve(
    data: dict[str, list[PointList]],
    n_dimensions: int,
    allowed_error: float,
    n_thresholds: int = 100,
) -> dict[str, tuple[list[float], list[float]]]:
    """
    Calculate ECDF curves.

    Args:
        data: Lists of value logs indexed by method name.
        n_dimensions: Dimensionality of the solved problem.
        allowed_error: Tolerable error value, used as the last threshold.
        n_thresholds: Number of ECDF thresholds.

    Returns:
        x, y plot points for each method.
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
            processed_log = [math.log10(max(v, allowed_error)) for v in new_log]
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
    data: dict[str, list[PointList]],
    n_dimensions: int,
    allowed_error: float,
    n_thresholds: int = 100,
    savepath: str | None = None,
    *,
    show: bool = True,
    function_name: str | None = None,
) -> None:
    """
    Calculate and plot ECDF curves.

    Args:
        data: Lists of value logs for every method.
        n_dimensions: Dimensionality of the optimized function.
        allowed_error: Tolerable error value, used as the last threshold.
        n_thresholds: Number of ECDF thresholds.
        savepath: Path to save the plot, optional.
        show: Wheather to show the plot, default True.
        function_name: Name of the optimized function, used in title.
    """
    plt.clf()

    ecdf_data = ecdf_curve(data, n_dimensions, allowed_error, n_thresholds)

    for method, (x, y) in ecdf_data.items():
        plt.plot(x, y, label=method)

    plt.xlabel("Number of function evaluations divided by the number of dimensions.")
    plt.xscale("log")
    plt.ylabel("ECDF point pairs")

    plt.legend()
    plt.grid(True)

    if function_name:
        plt.title(f"ECDF curves for function {function_name}")
    else:
        plt.title("ECDF curves")

    if savepath:
        plt.savefig(savepath)

    if show:
        plt.show()

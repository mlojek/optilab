"""
Calculating and plotting the convergence curve.
"""

from typing import Dict, List

from matplotlib import pyplot as plt


def convergence_curve(log: List[float]) -> List[float]:
    """
    For a given log return a convergence curve - the lowest value achieved
    so far.

    :param log: results log - the values of errors of the optimized function.
    :return: y values of the convergence curve.
    """
    min_so_far = float("inf")
    new_log = []

    for value in log:
        min_so_far = min(min_so_far, value)
        new_log.append(min_so_far)

    return new_log


def plot_convergence_curve(data: Dict[str, List[float]], savepath: str = None) -> None:
    """
    Plot the convergence curves of a few methods using pyplot.

    :param data: error logs of a few methods expressed as {method name: log}
    :param savepath: optional, path to save the plot.
    :return: None
    """
    plt.clf()

    for name, log in data.items():
        y = convergence_curve(log)
        plt.plot(y, label=name)

    plt.yscale("log")
    plt.xlabel("evaluations")
    plt.ylabel("value")
    plt.legend()
    plt.title("Convergence curves")

    if savepath:
        plt.savefig(savepath)

    plt.show()

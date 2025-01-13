"""
Calculating and plotting the convergence curve.
"""

from typing import Dict, List

from matplotlib import pyplot as plt

from ..data_classes import PointList


def convergence_curve(log: PointList) -> List[float]:
    """
    For a given log return a convergence curve - the lowest value achieved so far.

    Args:
        log (List[float]): Results log - the values of errors of the optimized function.

    Returns:
        List[float]: y values of the convergence curve.
    """
    min_so_far = float("inf")
    new_log = []

    for value in log:
        min_so_far = min(min_so_far, value.y)
        new_log.append(min_so_far)

    return new_log


def plot_convergence_curve(
    data: Dict[str, List[PointList]], savepath: str = None
) -> None:
    """
    Plot the convergence curves of a few methods using pyplot.

    Args:
        data (Dict[str, List[float]]): Error logs of a few methods expressed as {method name: log}.
        savepath (str): Path to save the plot, optional.
    """
    plt.clf()

    def average_lists(lists):
        max_length = max(len(lst) for lst in lists)
        padded_lists = [lst + [0] * (max_length - len(lst)) for lst in lists]
        averaged_list = [sum(values) / len(values) for values in zip(*padded_lists)]

        return averaged_list

    for name, loglist in data.items():
        ys = [convergence_curve(log) for log in loglist]
        averaged_ys = average_lists(ys)
        plt.plot(averaged_ys, label=name)

    plt.yscale("log")
    plt.xlabel("evaluations")
    plt.ylabel("value")
    plt.legend()
    plt.title("Convergence curves")

    if savepath:
        plt.savefig(savepath)

    plt.show()

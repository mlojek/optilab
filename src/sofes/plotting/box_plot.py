"""
Plotting optimization results with box plot.
"""

from typing import Dict, List

from matplotlib import pyplot as plt


def plot_box_plot(data: Dict[str, List[float]], savepath: str = None) -> None:
    """
    Plot box plots of optimization results.

    :param data: dictionary where keys are optimization method or function
    name and values are list of best values from each run.
    :param savepath: optional, path to save the plot
    :return: None
    """
    plt.clf()

    plot_values = []
    labels = []

    for name, values in data.items():
        plot_values.append(values)
        labels.append(name)

    plt.boxplot(plot_values)
    plt.xticks(range(1, len(labels) + 1), labels)

    if savepath:
        plt.savefig(savepath)

    plt.show()

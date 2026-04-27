"""
Plotting optimization results with box plot.
"""

from pathlib import Path

from matplotlib import pyplot as plt


def plot_box_plot(
    data: dict[str, list[float]],
    savepath: str | Path | None = None,
    *,
    show: bool = True,
    function_name: str | None = None,
    hide_outliers: bool = False,
) -> None:
    """
    Plot box plots of optimization results.

    Args:
        data: dictionary where keys are optimization method or function
            names and values are list of best values from each run.
        savepath: Path to save the plot, optional.
        show: Wheather to show the plot, default True.
        function_name: Name of the optimized function, used in title.
        hide_outliers: If true, outliers won't be shown on the box plot. Defualt is False.
    """
    plt.clf()

    plot_values = []
    labels = []

    for name, values in data.items():
        plot_values.append(values)
        labels.append(name)

    plt.boxplot(plot_values, showfliers=not hide_outliers)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
    plt.ylabel("value")
    plt.tight_layout(pad=2.0)

    if function_name:
        plt.title(f"Box plot for function {function_name}")
    else:
        plt.title("Box plot")

    if savepath:
        plt.savefig(savepath)

    if show:
        plt.show()

"""
Functions related to statistical testing.
"""

from typing import List

from scipy.stats import mannwhitneyu
from tabulate import tabulate


def mann_whitney_u_test_grid(data_lists: List[List[float]]) -> List[List[float]]:
    """
    Perform a grid run of Mann-Whitney U test on given list of data values and return
    a 2d array with results.

    Args:
        data_lists (List[List[float]]): List of lists of values to perform test on.

    Returns:
        List[List[float]]: Results as a 2d array with p-values.
    """
    n = len(data_lists)
    results_table = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                p_value = mannwhitneyu(
                    data_lists[i],
                    data_lists[j],
                    alternative="less",
                )[1]
                results_table[i][j] = p_value

    return results_table


def display_test_grid(test_results: List[List[float]]) -> str:
    """
    Given a grid of statistical test results, display them in a printable table.

    Args:
        test_results (List[List[float]]): Grid of p-values from the statistical test.

    Returns:
        str: Stat tests as a tabulated, ready to print table with p-values.
    """
    n = len(test_results)
    printable_results = [["-" for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                printable_results[i][j] = f"{test_results[i][j]:.4f}"

    header = list(range(n))
    return tabulate(
        printable_results,
        headers=header,
        showindex="always",
        tablefmt="github",
    )

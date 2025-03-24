"""
Aggregate pvalues for multiple algorithms and functions into one table.
"""

import pandas as pd


def aggregate_pvalues(pvalues_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pvalues for mutliple algorithms and function into one table.

    Args:
        pvalues_df (pd.DataFrame): Dataframe with columns: model, function, pvalue.

    Returns:
        pd.DataFrame: Dataframe with model names as columns, function names as row names
            and pvalues as values.
    """
    assert set(pvalues_df.columns) == {"model", "function", "pvalue"}

    model_list = pvalues_df.model.unique()
    function_list = sorted(pvalues_df.function.unique())

    aggregated_data = []

    for function in function_list:
        row = {"function": function}

        for model in model_list:
            value = pvalues_df.loc[
                (pvalues_df["model"] == model) & (pvalues_df["function"] == function),
                "pvalue",
            ]
            row[model] = value.values[0] if not value.empty else None

        aggregated_data.append(row)

    return pd.DataFrame(aggregated_data)

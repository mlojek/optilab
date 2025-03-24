"""
Aggregation of median and iqr stats for various functions and models.
"""

import pandas as pd


def aggregate_stats(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate median and iqr stats for various functions and models.

    Args:
        stats_df (pd.DataFrame): DataFrame with columns: model, function, y_median, y_iqr.

    Returns:
        pd.DataFrame: Dataframe with model names as columns, function names as rows,
            and median and iqr as values.
    """
    assert set(stats_df.columns) == {"model", "function", "y_median", "y_iqr"}

    model_list = stats_df.model.unique()
    function_list = sorted(stats_df.function.unique())

    aggregated_data = []

    for function in function_list:
        for stat in ["y_median", "y_iqr"]:
            row = {"function": function, "stat": stat}

            for model in model_list:
                value = stats_df.loc[
                    (stats_df["model"] == model) & (stats_df["function"] == function),
                    stat,
                ]
                row[model] = value.values[0] if not value.empty else None

            aggregated_data.append(row)

    return pd.DataFrame(aggregated_data)

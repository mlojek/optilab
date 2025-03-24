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

    aggregated_stats = pd.DataFrame()

    return aggregated_stats

"""
Aggregate pvalues for multiple algorithms and functions into one table.
"""

import pandas as pd


def aggregate_pvalues(pvalues_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate p-values for multiple algorithms and functions into one table.

    Args:
        pvalues_df (pd.DataFrame): DataFrame with columns: model, function, alternative, pvalue.

    Returns:
        pd.DataFrame: DataFrame with function and alternative as the first two columns,
                      model names as remaining columns, and p-values as values.
    """
    assert set(pvalues_df.columns) == {"model", "function", "alternative", "pvalue"}

    assert (pvalues_df["alternative"]).issubset({"better", "worse"})

    model_list = pvalues_df["model"].unique()
    func_dir_list = sorted(
        pvalues_df[["function", "alternative"]].drop_duplicates().values.tolist()
    )

    aggregated_data = []

    for function, alternative in func_dir_list:
        row = {"function": function, "alternative": alternative}

        for model in model_list:
            value = pvalues_df.loc[
                (pvalues_df["model"] == model)
                & (pvalues_df["function"] == function)
                & (pvalues_df["alternative"] == alternative),
                "pvalue",
            ]
            row[model] = value.values[0] if not value.empty else None

        aggregated_data.append(row)

    return pd.DataFrame(aggregated_data)

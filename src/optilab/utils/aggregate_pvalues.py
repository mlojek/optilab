"""
Aggregate pvalues for multiple algorithms and functions into one table.
"""

import pandas as pd


def aggregate_pvalues(pvalues_df: pd.DataFrame, significance: float) -> pd.DataFrame:
    """
    Aggregate p-values for multiple algorithms and functions into one table.

    Args:
        pvalues_df: DataFrame with columns: model, function, alternative, pvalue.
        significance: Statistical significance threshold for the tests.

    Returns:
        DataFrame with function and alternative as the first two columns,
                      model names as remaining columns, and p-values as values.
    """
    assert set(pvalues_df.columns) == {"model", "function", "alternative", "pvalue"}

    assert set(pvalues_df["alternative"]).issubset({"better", "worse"})

    model_list = pvalues_df["model"].unique()
    func_dir_list = sorted(
        pvalues_df[["function", "alternative"]].drop_duplicates().values.tolist()
    )

    aggregated_data = []

    better_counts: dict[str, int] = {model: 0 for model in model_list}
    worse_counts: dict[str, int] = {model: 0 for model in model_list}

    for function, alternative in func_dir_list:
        row = {"function": function, "alternative": alternative}

        for model in model_list:
            value = pvalues_df.loc[
                (pvalues_df["model"] == model)
                & (pvalues_df["function"] == function)
                & (pvalues_df["alternative"] == alternative),
                "pvalue",
            ]

            if not value.empty:
                row[model] = value.values[0]

                if value.values[0] < significance:
                    if alternative == "better":
                        better_counts[model] += 1
                    elif alternative == "worse":
                        worse_counts[model] += 1
            else:
                row[model] = None

        aggregated_data.append(row)

    aggregated_data.extend(
        [
            {"function": "total", "alternative": "better"} | better_counts,
            {"function": "total", "alternative": "worse"} | worse_counts,
        ]
    )

    return pd.DataFrame(aggregated_data)

"""
Entrypoint for CLI functionality of optilab.
"""

import argparse
from pathlib import Path

from .cli import OptilabCLI


def main() -> None:
    """
    Main function of the CLI utility. It's called when using optilab CLI command.
    """
    parser = argparse.ArgumentParser(description="Optilab CLI utility.", prog="optilab")
    parser.add_argument(
        "pickle_path",
        type=Path,
        help="Path to pickle file or directory with optimization runs.",
    )
    parser.add_argument(
        "--aggregate_pvalues",
        action="store_true",
        help="Aggregate pvalues of stat tests against run 0 in each pickle file into one table.",
    )
    parser.add_argument(
        "--aggregate_stats",
        action="store_true",
        help="Aggregate median and iqr for all processed runs into one table.",
    )
    parser.add_argument(
        "--entries",
        nargs="+",
        type=int,
        help="Space separated list of indexes of entries to include in analysis.",
    )
    parser.add_argument(
        "--hide_outliers",
        action="store_true",
        help="If specified, outliers will not be shown in the box plot.",
    )
    parser.add_argument(
        "--hide_plots",
        action="store_true",
        help="Hide plots when running the script.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="If specified, no artifacts will be saved.",
    )
    parser.add_argument(
        "--raw_values",
        action="store_true",
        help="If specified, y values below tolerance are not substituted by tolerance value.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path.cwd(),
        help="Path to directory to save the artifacts. Default is the user's working directory.",
    )
    parser.add_argument(
        "--significance",
        type=float,
        default=0.05,
        help="Statistical significance of the U tests. Default value is 0.05.",
    )
    parser.add_argument(
        "--test_evals",
        action="store_true",
        help="Perform Mann-Whitney U test on eval values.",
    )
    parser.add_argument(
        "--test_y",
        action="store_true",
        help="Perform Mann-Whitney U test on y values.",
    )
    OptilabCLI(parser.parse_args()).run()


if __name__ == "__main__":
    main()

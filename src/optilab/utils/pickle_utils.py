"""
Functions related to loading and dumping optimization results to pickle files.
"""

import pickle
from pathlib import Path
from typing import Any


def dump_to_pickle(data: Any, pickle_path: Path) -> None:
    """
    Dump data (such as List[OptimizationRun]) to a pickle file.

    Args:
        data (Any): Data to save to a pickle file.
        pickle_path (Path): Path to file to save the data.
    """
    with open(pickle_path, "wb") as pickle_handle:
        pickle.dump(data, pickle_handle)


def load_from_pickle(pickle_path: Path) -> Any:
    """
    Load data (such as List[OptimizationRun]) from a pickle file.

    Args:
        pickle_path (Path): Pickle file path to read from.

    Returns:
        Any: Data read from the pickle.
    """
    with open(pickle_path, "rb") as pickle_handle:
        data = pickle.load(pickle_handle)
    return data

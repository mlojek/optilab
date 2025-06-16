"""
Functions related to loading and dumping optimization results to pickle files.
"""

import pickle
from pathlib import Path
from typing import Any, List


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


def list_all_pickles(path: Path) -> List[Path]:
    """
    Given a path to either a file or directory return a list of all pickle files present there.

    Args:
        path (Path): Either a path to a pickle file or path to directory containing pickle files.

    Returns:
        List[Path]: List of paths to found pickle files.

    Raises:
        ValueError: If the path is a file and not a pickle, or when the path is a directory and
            contains no pickles.
    """
    file_path_list = []

    if path.is_file():
        if path.suffix == ".pkl":
            file_path_list.append(path)
        else:
            raise ValueError("Provided file path is not a pickle file.")
    elif path.is_dir():
        for file_path in sorted(path.iterdir()):
            if file_path.is_file() and file_path.suffix == ".pkl":
                file_path_list.append(file_path)
        if len(file_path_list) == 0:
            raise ValueError("No pickle file found in the provided directory.")

    return file_path_list

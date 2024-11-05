"""
Abstract base class representing a callable objective function.
"""

# pylint: disable=too-few-public-methods

from typing import List


class ObjectiveFunction:
    """
    Abstract base class representing a callable objective function.
    """

    def __init__(self, name: str, dim: int) -> None:
        """TODO"""
        self.name = name
        self.dim = dim
        self.num_calls = 0

    def __call__(self, x: List[float]) -> float:
        """TODO"""
        self.num_calls += 1

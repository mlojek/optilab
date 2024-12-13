"""
Class representing bounds of the search space.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Bounds:
    """
    Class representing bounds of the search space.
    """

    lower: float
    "The lower bound of search space."

    upper: float
    "The upper bound of search space."

    def to_list(self) -> List[float]:
        """
        Return the bounds as a list of two floats.

        Returns:
            List[float]: List containing the lower and upper bound.
        """
        return [self.lower, self.upper]

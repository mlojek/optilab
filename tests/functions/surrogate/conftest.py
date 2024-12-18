"""
Pytest fixtures used in surrogate objective functions unit tests. 
"""

import numpy as np
import pytest

from optilab.data_classes import Point, PointList


@pytest.fixture(name="train_set_2d_square")
def fixture_train_set_2d_square() -> PointList:
    """
    Simple 2D train set with four points making a square around point [0, 0].
    The expected value for point [0, 0] is 0.
    """
    return PointList(
        [
            Point(np.array([1, 1]), 3, True),
            Point(np.array([1, -1]), -3, True),
            Point(np.array([-1, -1]), 3, True),
            Point(np.array([-1, 1]), -3, True),
        ]
    )

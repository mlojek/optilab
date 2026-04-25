"""
Unit tests for Point dataclass.
"""

import numpy as np

from optilab.data_classes import Point


class TestPoint:
    """
    Unit tests for Point dataclass.
    """

    def test_dim(self):
        """
        Test if dim() method correctly returns the length of x.
        """
        point = Point(x=np.array([0, 1, 2]))
        assert point.dim() == 3

    def test_equal_with_only_x(self):
        """
        Check if equality operator works correctly with points that only have x values.
        """
        point0 = Point(x=np.array([0, 1, 2]))
        point1 = Point(x=np.array([0, 1, 2]))
        assert point0 == point1

    def test_equal_not_equal(self):
        """
        Check if equality operator correctly returns false when points have different x values.
        """
        point0 = Point(x=np.array([0, 1, 2]))
        point1 = Point(x=np.array([0, 1, 3]))
        assert point0 != point1

    def test_equal_no_empty_fields(self):
        """
        Check if equality operator works correctly with points with equal all fields.
        """
        point0 = Point(x=np.array([0, 1, 2]), y=5, is_evaluated=True)
        point1 = Point(x=np.array([0, 1, 2]), y=5, is_evaluated=True)
        assert point0 == point1

    def test_equal_one_has_empty_fields(self):
        """
        Check if equality operator works correctly when one point has all fields filled
        and other point only has x value.
        """
        point0 = Point(x=np.array([0, 1, 2]))
        point1 = Point(x=np.array([0, 1, 2]), y=5, is_evaluated=True)
        assert point0 == point1

    def test_equal_different_y(self):
        """
        Check if equality operator correctly returns True even when y values of points differ.
        """
        point0 = Point(x=np.array([0, 1, 2]), y=4, is_evaluated=True)
        point1 = Point(x=np.array([0, 1, 2]), y=5, is_evaluated=True)
        assert point0 == point1

    def test_equal_different_is_evaluated(self):
        """
        Check if equality operator correctly returns True even when is_evaluated values differ.
        """
        point0 = Point(x=np.array([0, 1, 2]), y=5, is_evaluated=True)
        point1 = Point(x=np.array([0, 1, 2]), y=5, is_evaluated=False)
        assert point0 == point1

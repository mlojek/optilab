"""
Unit tests for Bounds.handle_bounds method.
"""

import pytest


class TestBoundsHandleBounds:
    """
    Unit tests for Bounds.handle_bounds method.
    """

    def test_project(self, example_bounds, point_multidimensional):
        handled_point = example_bounds.handle_bounds(point_multidimensional, "project")
        assert handled_point.x == [14, 10, 20, 10, 20, 10, 20]

    def test_reflect(self, example_bounds, point_multidimensional):
        handled_point = example_bounds.handle_bounds(point_multidimensional, "reflect")
        assert handled_point.x == [14, 10, 20, 18, 17, 16, 12]

    def test_wrap(self, example_bounds, point_multidimensional):
        handled_point = example_bounds.handle_bounds(point_multidimensional, "wrap")
        assert handled_point.x == [14, 10, 20, 12, 13, 16, 12]

    def test_invalid_mode(self, example_bounds, point_multidimensional):
        with pytest.raises(ValueError, match="Invalid mode"):
            example_bounds.handle_bounds(point_multidimensional, "invalid")

"""
Unit tests for PointList dataclass.
"""

import numpy as np
import pytest

from optilab.data_classes import Point, PointList


@pytest.fixture(name="example_2d_pointlist")
def fixture_example_2d_pointlist():
    """
    An example PointList containing 2d points.
    """
    return PointList(
        [
            Point(np.array([0, 0]), 10, True),
            Point(np.array([1, 0]), 8, True),
            Point(np.array([0, 1]), 5, False),
            Point(np.array([-1, 0]), 1, False),
            Point(np.array([1, -1]), 2, True),
        ]
    )


@pytest.fixture(name="overlapping_2d_pointlist")
def fixture_overlapping_2d_pointlist():
    """
    An example PointList that overlaps with example_2d_pointlist fixture.
    """
    return PointList(
        [
            Point(np.array([0, 0]), 10, True),
            Point(np.array([0, 1]), -5, False),
            Point(np.array([1, -1]), 2, False),
            Point(np.array([2, 1]), 1, True),
        ]
    )


@pytest.fixture(name="example_3d_pointlist")
def fixture_example_3d_pointlist():
    """
    An example PointList containing 3d points.
    """
    return PointList(
        [
            Point(np.array([0, 0, 0]), 10, True),
            Point(np.array([1, 0, 1]), 8, True),
            Point(np.array([0, 1, 2]), 5, False),
        ]
    )


class TestPointList:
    """
    Unit tests for PointList dataclass.
    """

    # Point in PointList
    def test_point_in_pointlist(self, example_2d_pointlist):
        """
        Test if a point with the same x value is correctly classified to be in the PointList.
        """
        assert Point(np.array([0, 0])) in example_2d_pointlist

    def test_point_in_pointlist_different_y(self, example_2d_pointlist):
        """
        Test if a point with the same x value but different y value is correctly
        classified as belonging to a PointList.
        """
        assert Point(np.array([0, 0]), 19) in example_2d_pointlist

    def test_point_not_in_pointlist_different_y(self, example_2d_pointlist):
        """
        Test if a point is correctly classified as not belonging to a PointList.
        """
        assert Point(np.array([5, 0])) not in example_2d_pointlist

    # from_list
    def test_from_list(self):
        """
        Test if alternative constructor from_list() works as expected.
        """
        point_list = PointList.from_list(
            [
                np.array([0, 0]),
                np.array([0, 1]),
                np.array([0, 2]),
                np.array([1, 1]),
            ]
        )
        assert len(point_list) == 4

    def test_from_empty_list(self):
        """
        Test if alternative constructor from_list() works as expected
        when provided with an empty list.
        """
        empty_point_list = PointList.from_list([])
        assert len(empty_point_list) == 0

    # append
    def test_append(self, example_2d_pointlist):
        """
        Test if append() method works as expected.
        """
        example_2d_pointlist.append(Point(np.array([2, 2])))
        assert len(example_2d_pointlist) == 6

    def test_append_to_empty(self):
        """
        Test if append() method works as expected on an empty pointlist.
        """
        empty_pointlist = PointList([])
        empty_pointlist.append(Point(np.array([2, 2])))
        assert len(empty_pointlist) == 1

    # extend
    def test_extend(self, example_2d_pointlist, overlapping_2d_pointlist):
        """
        Test if extend() method works as expected.
        """
        example_2d_pointlist.extend(overlapping_2d_pointlist)
        assert len(example_2d_pointlist) == 9

    def test_extend_with_empty(self, example_2d_pointlist):
        """
        Test if extend() method works as expected when extending with an empty pointlist.
        """
        example_2d_pointlist.extend(PointList([]))
        assert len(example_2d_pointlist) == 5

    def test_extend_empty(self, example_2d_pointlist):
        """
        Test if extend() method works as expected on an empty pointlist.
        """
        empty_pointlist = PointList([])
        empty_pointlist.extend(example_2d_pointlist)
        assert len(empty_pointlist) == 5

    def test_extend_empty_with_empty(self):
        """
        Test if extend() method works as expected when extending an empty pointlist with
        another empty pointlist.
        """
        empty_pointlist = PointList([])
        empty_pointlist.extend(PointList([]))
        assert len(empty_pointlist) == 0

    # x
    def test_get_x(self, example_2d_pointlist):
        """
        Test if x getter method works correctly.
        """
        assert np.all(
            np.all(x, expected)
            for x, expected in zip(
                example_2d_pointlist.x(), [[0, 0], [1, 0], [0, 1], [-1, 0], [1, -1]]
            )
        )

    def test_get_x_empty(self):
        """
        Test if x getter method works correctly on an empty PointList.
        """
        assert PointList([]).x() == []

    # y
    def test_get_y(self, example_2d_pointlist):
        """
        Test if y getter method works correctly.
        """
        assert example_2d_pointlist.y() == [10, 8, 5, 1, 2]

    def test_get_y_empty(self):
        """
        Test if y getter method works correctly on an empty PointList.
        """
        assert PointList([]).y() == []

    # only_evaluated
    def test_only_evaluated(self, example_2d_pointlist):
        """
        Test if only_evaluated() method works as expected.
        """
        only_evaluated_points = example_2d_pointlist.only_evaluated()
        assert isinstance(only_evaluated_points, PointList)
        assert len(only_evaluated_points) == 3

    def test_only_evaluated_empty(self):
        """
        Test if only_evaluated() method works as expected on an empty PointList.
        """
        only_evaluated_empty = PointList([]).only_evaluated()
        assert isinstance(only_evaluated_empty, PointList)
        assert len(only_evaluated_empty) == 0

    # rank
    def test_rank(self, example_2d_pointlist):
        """
        Test if a PointList is correctly ranked by ascending y value.
        """
        example_2d_pointlist.rank()
        assert example_2d_pointlist.y() == [1, 2, 5, 8, 10]

    def test_rank_reverse(self, example_2d_pointlist):
        """
        Test if a PointList is correctly ranked by descending y value.
        """
        example_2d_pointlist.rank(reverse=True)
        assert example_2d_pointlist.y() == [10, 8, 5, 2, 1]

    def test_rank_empty(self):
        """
        Test if no error is raised when trying to rank an empty PointList.
        """
        empty_point_list = PointList([])
        empty_point_list.rank()
        assert len(empty_point_list) == 0

    # x_difference
    def test_x_difference(self, example_2d_pointlist, overlapping_2d_pointlist):
        """
        Test if x_difference() method works correctly with two overlapping pointlists.
        """
        difference = example_2d_pointlist.x_difference(overlapping_2d_pointlist)
        assert len(difference) == 2

    def test_x_difference_same(self, example_2d_pointlist):
        """
        Test if x_difference() method works correctly with two copies of the same pointlist.
        """
        difference = example_2d_pointlist.x_difference(example_2d_pointlist)
        assert len(difference) == 0

    def test_x_difference_empty(self, example_2d_pointlist):
        """
        Test if x_difference() method works correctly on an empty pointlist.
        """
        difference = PointList([]).x_difference(example_2d_pointlist)
        assert len(difference) == 0

    def test_x_difference_with_empty(self, example_2d_pointlist):
        """
        Test if x_difference() method works correctly when provided with an empty pointlist.
        """
        difference = example_2d_pointlist.x_difference(PointList([]))
        assert len(difference) == len(example_2d_pointlist)

    def test_x_difference_both_empty(self):
        """
        Test if x_difference() method works correctly with two empty pointlists.
        """
        difference = PointList([]).x_difference(PointList([]))
        assert len(difference) == 0

    def test_x_difference_different_dim(
        self, example_2d_pointlist, example_3d_pointlist
    ):
        """
        Test if x_difference() method works correctly with two pointlists
        of different dimensionalities.
        """
        difference = example_2d_pointlist.x_difference(example_3d_pointlist)
        assert len(difference) == len(example_2d_pointlist)

    # best
    def test_best(self, example_2d_pointlist):
        """
        Test if best() method works as expected.
        """
        best_point = example_2d_pointlist.best()
        assert best_point.y == 1
        assert np.all(best_point.x == [-1, 0])

    def test_best_empty_list(self):
        """
        Test if best() method works as expected on an empty PointList.
        """
        with pytest.raises(ValueError):
            PointList([]).best()

    # best_index
    def test_best_index(self, example_2d_pointlist):
        """
        Test if best_index() method correctly returns the index of the best point.
        """
        assert example_2d_pointlist.best_index() == 3

    def test_best_index_empty(self):
        """
        Test if best_index() method works as expected on an empty PointList.
        """
        with pytest.raises(ValueError):
            PointList([]).best_index()

    # best_y
    def test_best_y(self, example_2d_pointlist):
        """
        Test if best_y() method correctly returns the lowest y value.
        """
        assert example_2d_pointlist.best_y() == 1

    def test_best_y_empty(self):
        """
        Test if best_y() method works correctly on an empty PointList.
        """
        assert PointList([]).best_y() == np.inf

    # slice to best
    def test_slice_to_best(self, example_2d_pointlist):
        """
        Test if slice_to_best() method works as expected.
        """
        points_slice = example_2d_pointlist.slice_to_best()
        assert len(points_slice) == 4
        assert points_slice[-1].y == 1

    def test_slice_to_best_empty(self):
        """
        Test if slice_to_best() method works as expected on an empty PointList.
        """
        with pytest.raises(ValueError):
            PointList([]).slice_to_best()

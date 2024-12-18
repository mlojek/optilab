"""
Unit tests for KNNSurrogateObjectiveFunction.
"""

import numpy as np
import pytest

from optilab.data_classes import Point, PointList
from optilab.functions.surrogate.knn_surrogate_objective_function import (
    KNNSurrogateObjectiveFunction,
)


class TestKNNSurrogateObjectiveFunction:
    """
    Unit tests class for KNNSurrogateObjectiveFunction.
    """

    def test_empty_train_set(self):
        """
        Test if when the KNN surrogate function is provided with an empty train, it raises
        NotImplementedError when called.
        """
        train_set = PointList([])
        knn_sof = KNNSurrogateObjectiveFunction(3, train_set)
        with pytest.raises(NotImplementedError):
            knn_sof(Point(np.array([1])))

    def test_train_set_below_num_neighbours(self):
        """
        Test if when the KNN surrogate function is provided with a train set that is smaller than
        the number of neighbours it raises a ValueError as expected.
        """
        train_set = PointList([Point(np.array([0]), 0, True)])
        knn_sof = KNNSurrogateObjectiveFunction(3, train_set)
        with pytest.raises(ValueError):
            knn_sof(Point(np.array([2])))

    def test_x_dim_zero(self):
        """
        Test if when the KNN surrogate function is provided with a train set where the x-s are
        empty (or have the dimensionality of 0) the model raises a ValueError.
        """
        train_set = PointList(
            [
                Point(np.array([]), 0, True),
                Point(np.array([]), 1, True),
                Point(np.array([]), 2, True),
            ]
        )
        with pytest.raises(ValueError):
            KNNSurrogateObjectiveFunction(3, train_set)

    def test_simple_1d_case(self):
        """
        Test if the KNN surrogate function calculates the result as expected. The provided case
        has been calculated by hand and checked here if the KNN returns the expected result.
        """
        train_set = PointList(
            [
                Point(np.array([-5]), 2, True),
                Point(np.array([0]), -3, True),
                Point(np.array([5]), 3, True),
            ]
        )
        knn_sof = KNNSurrogateObjectiveFunction(3, train_set)
        assert np.isclose(-1.353, knn_sof(Point(np.array([1]))).y, atol=1e-3)

    def test_all_same(self):
        """
        Simple test case. All points have the same x and y and the result is expected to be
        the same y.
        """
        train_set = PointList([Point(np.array([0]), 1, True)] * 100)
        knn_sof = KNNSurrogateObjectiveFunction(5, train_set)
        y = knn_sof(Point(np.array([100]))).y
        assert y == 1
        assert knn_sof.num_calls == 1
        assert knn_sof.dim == 1

    def test_square(self, train_set_2d_square):
        """
        Test case with 4 point that make up a square. Two pairs of point have values 1 and 3.
        The x is the center of the square, so the expected y value is 2.
        """
        knn_sof = KNNSurrogateObjectiveFunction(4, train_set_2d_square)
        inference_result = knn_sof(Point(np.array([0, 0])))
        assert inference_result == Point(np.array([0, 0]))
        assert inference_result.y == 0
        assert not inference_result.is_evaluated
        assert knn_sof.num_calls == 1
        assert knn_sof.dim == 2

    def test_dim_not_constant(self):
        """
        Check if the initalization fails when x-s in train set have different dimensionality.
        """
        train_set = PointList(
            [
                Point(np.array([1, 1]), 1, True),
                Point(np.array([1, -1]), 3, True),
                Point(np.array([-1, -1]), 1, True),
                Point(np.array([-1]), 3, True),
            ]
        )
        with pytest.raises(ValueError):
            KNNSurrogateObjectiveFunction(4, train_set)

    def test_not_ready(self):
        """
        Check if when a KNNSurrogateObjectiveFunction model is created without providing training
        data it's labeled as not ready, and raises an error when called as expected.
        """
        knn_sof = KNNSurrogateObjectiveFunction(4)
        assert not knn_sof.is_ready
        with pytest.raises(NotImplementedError):
            knn_sof(Point(np.array([10])))

    def test_train_later(self, train_set_2d_square):
        """
        Check if creating a KNNSurrogateObjectiveFunction without training data and then training
        it post creation results in a working surrogate function.
        """
        knn_sof = KNNSurrogateObjectiveFunction(4)
        assert not knn_sof.is_ready
        knn_sof.train(train_set_2d_square)
        assert knn_sof(Point(np.array([0, 0]))).y == 0
        assert knn_sof.num_calls == 1
        assert knn_sof.dim == 2

    def test_knn_point_from_training_set(self, train_set_2d_square):
        """
        Check if when knn is provided with item from training set the surrogate returns its value.
        """
        knn_sof = KNNSurrogateObjectiveFunction(3, train_set_2d_square)
        assert knn_sof(Point(np.array([1, -1]))).y == -3

    def test_knn_duplicates_in_training_set(self):
        """
        Check if when the KNN accepts duplicates in it's training set. It's expected to do so.
        """
        train_set = PointList(
            [
                Point(np.array([1, 1]), 1, True),
                Point(np.array([1, -1]), 2, True),
                Point(np.array([1, -1]), 2, True),
                Point(np.array([-1, -1]), 3, True),
                Point(np.array([-1, 1]), 4, True),
                Point(np.array([0, 0]), 5, True),
            ]
        )
        knn_sof = KNNSurrogateObjectiveFunction(3, train_set)
        assert knn_sof(Point(np.array([3, -3]))).y == 2.75

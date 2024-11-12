"""
Unit tests for KNNSurrogateObjectiveFunction.
"""

import pytest

from sofes.objective_functions.knn_surrogate_objective_function import (
    KNNSurrogateObjectiveFunction,
)


class TestKNNSurrogateObjectiveFunction:
    """
    Unit tests class for KNNSurrogateObjectiveFunction.
    """

    def test_all_same(self):
        """
        Simple test case. All points have the same x and y and the result is expected to be
        the same y.
        """
        train_set = [([0], 1)] * 100
        knn_sof = KNNSurrogateObjectiveFunction(5, train_set)
        y = knn_sof([100])
        assert y == 1
        assert knn_sof.num_calls == 1
        assert knn_sof.dim == 1

    def test_square(self):
        """
        Test case with 4 point that make up a square. Two pairs of point have values 1 and 3.
        The x is the center of the square, so the expected y value is 2.
        """
        train_set = [([1, 1], 1), ([1, -1], 3), ([-1, -1], 1), ([-1, 1], 3)]
        knn_sof = KNNSurrogateObjectiveFunction(4, train_set)
        y = knn_sof([0, 0])
        assert y == 2
        assert knn_sof.num_calls == 1
        assert knn_sof.dim == 2

    def test_dim_not_constant(self):
        """
        Check if the initalization fails when x-s in train set have different dimensionality.
        """
        train_set = [([1, 1], 1), ([1, -1], 3), ([-1, -1], 1), ([-1], 3)]
        with pytest.raises(ValueError):
            KNNSurrogateObjectiveFunction(4, train_set)

    def test_not_ready(self):
        """"""
        knn_sof = KNNSurrogateObjectiveFunction(4)
        assert not knn_sof.is_ready
        with pytest.raises(NotImplementedError):
            knn_sof([10])

    def test_train_later(self):
        """"""
        train_set = [([1, 1], 1), ([1, -1], 3), ([-1, -1], 1), ([-1, 1], 3)]
        knn_sof = KNNSurrogateObjectiveFunction(4)
        assert not knn_sof.is_ready
        knn_sof.train(train_set)
        assert knn_sof([0, 0]) == 2
        assert knn_sof.num_calls == 1
        assert knn_sof.dim == 2

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
        """"""
        train_set = [([0], 1)] * 100
        knn_sof = KNNSurrogateObjectiveFunction(train_set, 5)
        y = knn_sof([100])
        assert y == 1
        assert knn_sof.num_calls == 1
        assert knn_sof.dim == 1

    def test_square(self):
        """"""
        train_set = [([1, 1], 1), ([1, -1], 3), ([-1, -1], 1), ([-1, 1], 3)]
        knn_sof = KNNSurrogateObjectiveFunction(train_set, 4)
        y = knn_sof([0, 0])
        assert y == 2
        assert knn_sof.num_calls == 1
        assert knn_sof.dim == 2

    def test_dim_not_constant(self):
        """"""
        train_set = [([1, 1], 1), ([1, -1], 3), ([-1, -1], 1), ([-1], 3)]
        with pytest.raises(ValueError):
            KNNSurrogateObjectiveFunction(train_set, 4)

"""
Top half metamodel. Estimates all points using the surrogate function, then
evaluates the top mu (typically half) points with the objective function.
"""

# pylint: disable=duplicate-code

from ..data_classes import PointList
from ..functions import ObjectiveFunction
from ..functions.surrogate import SurrogateObjectiveFunction


class TopHalfMetamodel:
    """
    Top-half metamodel.

    Estimates all points using the surrogate function, then calculates the
    real value for the top mu (typically half, hence the name) using the actual
    objective function. The remaining points get a value that's guaranteed
    to be worse than the rest of the points.
    """

    def __init__(
        self,
        population_size: int,
        mu: int,
        objective_function: ObjectiveFunction,
        surrogate_function: SurrogateObjectiveFunction,
        *,
        buffer_size: int | None = None,
    ) -> None:
        """
        Class constructor.

        Args:
            population_size: The population size (lambda).
            mu: Number of elite solutions (typically population_size // 2).
            objective_function: The real objective function.
            surrogate_function: Surrogate used to pre-screen candidates.
            buffer_size: If set, only the last *buffer_size* real evaluations
                are used as the surrogate training set.
        """
        self.population_size = population_size
        self.mu = mu

        self.train_set = PointList(points=[])

        self.objective_function = objective_function
        self.surrogate_function = surrogate_function

        self.buffer_size = buffer_size

        self._adapted_results: PointList | None = None

    def __call__(self, points: PointList) -> PointList:
        """
        Return evaluated / estimated values for *points*.

        If ``adapt()`` was called immediately before, returns the combined
        results (real values for top half, penalised for bottom half) and
        clears the internal cache.  Otherwise falls back to pure surrogate
        estimation.

        Args:
            points: Candidate solutions.

        Returns:
            PointList with y-values set for every point.
        """
        if self._adapted_results is not None:
            result = self._adapted_results
            self._adapted_results = None
            return result
        return PointList(points=[self.surrogate_function(x) for x in points])

    def adapt(self, xs: PointList) -> None:
        """
        Screen *xs* with the surrogate, evaluate the best half with the real
        objective, and cache a combined result for the next ``__call__()``.

        Args:
            xs: Candidate solutions (must have length == population_size).

        Raises:
            ValueError: If the number of points does not match population_size.
        """
        if len(xs) != self.population_size:
            raise ValueError(f"Expected {self.population_size} points, got {len(xs)}.")

        if len(self.train_set) < self.population_size:
            self._adapted_results = self.evaluate(xs)
            return

        estimated = PointList(points=[self.surrogate_function(x) for x in xs])
        estimated.rank()

        evaluated = self.evaluate(estimated[: self.mu])
        evaluated.rank()

        # penalize worst points
        penalty_y = evaluated[-1].y + 1
        not_evaluated = estimated[self.mu :]

        for point in not_evaluated:
            point.y = penalty_y

        evaluated.extend(not_evaluated)

        self._adapted_results = evaluated

    def train_surrogate(self) -> None:
        """
        Retrain the surrogate on the real evaluations.
        """
        if self.buffer_size:
            self.surrogate_function.train(self.train_set[-self.buffer_size :])
        else:
            self.surrogate_function.train(self.train_set)

    def evaluate(self, xs: PointList) -> PointList:
        """
        Evaluate *xs* with the real objective, extend the training set, and
        retrain the surrogate.

        Args:
            xs: Points to evaluate.

        Returns:
            PointList of evaluated points.
        """
        result = PointList(
            points=[self.objective_function(point) for point in xs.points]
        )
        self.train_set.extend(result)

        self.train_surrogate()

        return result

    def get_log(self) -> PointList:
        """
        Return all points evaluated with the real objective so far.
        """
        return self.train_set

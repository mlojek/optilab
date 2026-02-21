"""
Top-half metamodel.

Estimates all candidate solutions with a surrogate function, then evaluates
only the best half (top-mu) with the real objective function. The remaining
solutions receive penalised y-values so that CMA-ES — which only uses the
best mu = lambda/2 solutions for its state update — is guaranteed to operate
exclusively on real evaluations.
"""

# pylint: disable=duplicate-code

from typing import Optional

from ..data_classes import Point, PointList
from ..functions import ObjectiveFunction
from ..functions.surrogate import SurrogateObjectiveFunction


class TopHalfMetamodel:
    """
    Top-half metamodel.

    Workflow inside ``adapt()``:
    1. Estimate all *lambda* candidate points with the surrogate.
    2. Pick the best *mu* = *lambda* / 2 by surrogate estimate.
    3. Evaluate those *mu* points with the real objective function.
    4. Assign penalised y-values to the remaining *lambda* - *mu* points
       so that they are guaranteed to rank below the real evaluations.

    The next call to ``__call__()`` after ``adapt()`` returns the combined
    result (real values for the evaluated half, penalty for the rest).
    """

    def __init__(
        self,
        population_size: int,
        mu: int,
        objective_function: ObjectiveFunction,
        surrogate_function: SurrogateObjectiveFunction,
        *,
        buffer_size: Optional[int] = None,
    ) -> None:
        """
        Class constructor.

        Args:
            population_size: The population size (lambda).
            mu: Number of elite solutions (should be population_size // 2).
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

        # Stores combined results produced by the last ``adapt()`` call.
        self._adapted_results: Optional[PointList] = None

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

        # Not enough history → evaluate everything (bootstrap phase).
        if len(self.train_set) < self.population_size:
            self._adapted_results = self.evaluate(xs)
            return

        # --- Step 1: surrogate estimation of all candidates ---------------
        estimated = PointList(points=[self.surrogate_function(x) for x in xs])

        # --- Step 2: pick top-mu by surrogate value ----------------------
        indexed = [(i, estimated[i].y) for i in range(len(estimated))]
        indexed.sort(key=lambda pair: pair[1])
        top_indices = sorted(idx for idx, _ in indexed[: self.mu])
        top_indices_set = set(top_indices)

        # --- Step 3: evaluate top-mu with real objective ------------------
        top_points = PointList(points=[xs[i] for i in top_indices])
        evaluated = self.evaluate(top_points)

        # --- Step 4: build combined result --------------------------------
        eval_map = {idx: evaluated.points[j] for j, idx in enumerate(top_indices)}

        # Penalty: strictly worse than all real values.
        max_real_y = max(p.y for p in evaluated)
        penalty_y = max_real_y + abs(max_real_y) * 1e-6 + 1e-10

        combined = []
        for i, x in enumerate(xs):
            if i in top_indices_set:
                combined.append(eval_map[i])
            else:
                combined.append(Point(x=x.x, y=penalty_y, is_evaluated=False))

        self._adapted_results = PointList(points=combined)

    # ------------------------------------------------------------------
    # Training / evaluation helpers
    # ------------------------------------------------------------------

    def train_surrogate(self) -> None:
        """Retrain the surrogate on (optionally windowed) real evaluations."""
        if self.buffer_size:
            self.surrogate_function.train(
                PointList(self.train_set[-self.buffer_size :])
            )
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
        """Return all points evaluated with the real objective so far."""
        return self.train_set

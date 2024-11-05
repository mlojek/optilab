""""""

from typing import Any

from .objective_function import ObjectiveFunction


class SurrogateObjectiveFunction(ObjectiveFunction):
    """"""

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

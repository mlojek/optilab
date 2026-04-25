"""
Metadata of an optimizer model.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OptimizerMetadata:
    """
    Metadata of an optimizer model.
    """

    name: str
    "Name of the optimizer."

    population_size: int
    "Number of points generated in each generation."

    hyperparameters: dict[str, Any] = field(default_factory=dict)
    "Other hyperparameters of the optimizer, optional."

"""
Metadata of an optimizer model.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class OptimizerMetadata:
    """
    Metadata of an optimizer model.
    """

    name: str
    "Name of the optimizer."

    population_size: int
    "Number of points generated in each generation."

    hyperparameters: Dict[str, Any] = None
    "Other hyperparameters of the optimizer, optional."

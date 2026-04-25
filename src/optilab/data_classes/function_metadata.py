"""
Metadata of objective function.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class FunctionMetadata:
    """
    Metadata of objective function.
    """

    name: str
    "Name of the function."

    dim: int
    "Dimensionality of the function."

    hyperparameters: dict[str, Any]
    "Other hyperparameters of the function."

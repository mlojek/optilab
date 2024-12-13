"""
Metadata of objective function.
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class FunctionMetadata:
    """
    Metadata of objective function.
    """

    name: str
    "Name of the function."

    dim: int
    "Dimensionality of the function."

    hyperparameters: Dict[str, Any] = None
    "Other hyperparameters of the function, optional."

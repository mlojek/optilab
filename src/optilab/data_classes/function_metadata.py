"""
Metadata of objective function.
"""

from typing import Any

from pydantic import BaseModel


class FunctionMetadata(BaseModel):
    """
    Metadata of objective function.
    """

    name: str
    "Name of the function."

    dim: int
    "Dimensionality of the function."

    hyperparameters: dict[str, Any]
    "Other hyperparameters of the function."

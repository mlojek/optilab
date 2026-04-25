"""
Metadata of an optimizer model.
"""

from typing import Any

from pydantic import BaseModel, Field


class OptimizerMetadata(BaseModel):
    """
    Metadata of an optimizer model.
    """

    name: str
    "Name of the optimizer."

    population_size: int
    "Number of points generated in each generation."

    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    "Other hyperparameters of the optimizer, optional."

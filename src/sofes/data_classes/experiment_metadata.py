"""
Class containing metadata of an experiment.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ExperimentMetadata:
    """
    Data class representing the metadata of an experiment. This includes the information about
    the method used, metamodel used, and time of running.
    """

    method_name: str
    method_hyperparameters: dict
    metamodel_name: str
    metamodel_hyperparameters: dict
    benchmark_name: str
    time_begin: datetime
    time_end: datetime

    def begin_now(self) -> None:
        """
        Set time of the beggining to now.
        """
        self.time_begin = datetime.now()

    def end_now(self) -> None:
        """
        Set time of the ending to now.
        """
        self.time_end = datetime.now()

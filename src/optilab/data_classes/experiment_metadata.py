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
    "Name of the optimizer."

    method_hyperparameters: dict
    "Hyperparameters of the optimizer"

    metamodel_name: str
    "Name of the metamodel used."

    metamodel_hyperparameters: dict
    "Hyperparameters of the metamodel."

    benchmark_name: str
    "Name of the benchmark."

    time_begin: str = None
    "Timestamp of the beginning of the experiment."

    time_end: str = None
    "Timestamp of the end of the experiment."

    def begin_now(self) -> None:
        """
        Set time of the beggining to now.
        """
        self.time_begin = datetime.now().isoformat()

    def end_now(self) -> None:
        """
        Set time of the ending to now.
        """
        self.time_end = datetime.now().isoformat()

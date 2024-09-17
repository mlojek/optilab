'''
Classes dedicated to storing and visualizing experiment results.
'''
from typing import Dict, List, Union, Any


# TODO not singular run but rather a group of runs
class ExperimentResults:
    '''
    Class to store results of an experiment.
    '''
    def __init__(
        self,
        name: str,
        log: List[List[float]],
        metadata: Dict[str, Any]=None,
    ):
        '''
        Class constructor

        :param name: name of the experiment
        :param log: the log of values found in the experiment. list of list of values found in each run of the algorithm
        :param metadata: optional, the metadata of the experiment, e.g. model parameters
        '''
        if not metadata:
            metadata = {}

        self.name = name
        self.metadata = metadata
        self.log = log
        # TODO stats

    def calculate_stats(self, log) -> Dict[str, float]:
        '''
        TODO
        '''
        pass

    def plot_ecdf(self):
        '''
        TODO
        '''
        # TODO modes: average, all on one plot, all on separate plots, only best run
        pass

    def to_dict(self) -> Dict[str, Any]:
        '''
        TODO
        '''
        # include stats and num_runs in the dict
        pass


class ExperimentResultsCollection:
    '''
    TODO
    '''
    def __init__(self):
        '''
        TODO
        '''
        pass

    def save_to_file(self):
        '''
        TODO
        TODO determine csv or json by file extension
        '''
        pass

    def load_from_file(self):
        '''
        TODO
        '''
        pass

    def plot_box_plot(self):
        '''
        TODO
        '''
        # TODO specify how many boxes per plot and generate multiple plots
        pass

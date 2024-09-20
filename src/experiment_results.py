'''
Classes dedicated to storing and visualizing experiment results.
'''
import numpy as np
from typing import Dict, List, Union, Any, Tuple


class ExperimentResults:
    '''
    Class to store results of an experiment.
    '''
    def __init__(
        self,
        name: str,
        logs: List[List[float]],
        metadata: Dict[str, Any],
    ):
        '''
        Class constructor

        :param name: name of the experiment
        :param logs: lists of objective function values achieved in every run of the algorithm
        :param metadata: metadata of the experiment, e.g. model parameters 
        '''
        self.name = name
        self.metadata = metadata
        self.logs = logs
        self.num_runs = len(self.logs)
        self.results = [max(log) for log in self.logs]
        self.stats = self.calculate_stats(self.results)

    def calculate_stats(self, results: List[float]) -> Dict[str, float]:
        '''
        Calculates the stats of the experiment results (best values from each
        run). Returns a dict with min and max values, mean and median and
        standard deviation.

        :param results: list of experiment results - the best values from each run
        :return: a dictionary with stats of the data
        '''
        return {
            'min': min(results),
            'max': max(results),
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results)
        }

    def ecdf_points(self, log: List[float], dimensions: int, target: float) -> Tuple[List[float], List[float]]:
        '''
        TODO
        '''
        progressive_minimum = [log[0]]
        for item in log[1:]:
            if item < progressive_minimum[-1]:
                progressive_minimum.append(item)
            else:
                progressive_minimum.append(progressive_minimum[-1])

        # range is highest first value and target
        # calculate key levels
        # calculate ys
        y = []
        # calculate xs
        x = [i / dimensions for i in range(len(log))]
        return x, y

    def plot_ecdf(self):
        '''
        TODO
        '''
        # TODO modes: average, all on one plot, all on separate plots, only best run
        pass

    def to_dict(self, include_logs=False) -> Dict[str, Any]:
        '''
        Serializes the experiment results to a dictionary.

        :param include_logs: wheather to include full result logs, default False
        :return: the contents of the object saved to a dictionary
        '''
        result = {
            'name': self.name,
            'metadata': self.metadata,
            'num_runs': self.num_runs,
            'stats': self.stats,
            'results': self.results
        }

        if include_logs:
            result['logs'] = self.logs

        return result


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

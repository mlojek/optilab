'''
Data visualization functions - plotting etc.
'''
import numpy as np
from matplotlib import pyplot as plt
from typing import List


def ecdf_points(
        results_log: List[float],
        dimensions: int,
        target: float=None,
        targets_scale: str='lin',
        targets_num: int=100
    ):
    '''
    TODO
    important: only minimization, if you solved maximization problem do all * -1

    :param targets_scale: either lin or log
    '''
    values_range = (results_log[0], min(results_log))
    if target:
        values_range = (results_log[0], target)

    target_pairs = np.linspace(*values_range, targets_num)

    targets_reached = [1]
    current_targets = 1

    for result in results_log[1:]:
        while result <= target_pairs[current_targets]:
            if current_targets < targets_num - 1:
                current_targets += 1
            else:
                break
        targets_reached.append(current_targets)
    
    y = [item/targets_num for item in targets_reached]
    x = [i/dimensions for i in range(1, len(results_log)+1)]

    assert len(x) == len(y)
    return x, y


def ecdf_curve(
        results_log: List[float],
        dimensions: int,
        target: float=None,
        targets_scale: str='lin',
        targets_num: int=100
    ):
    '''
    TODO
    important: only minimization, if you solved maximization problem do all * -1

    :param targets_scale: either lin or log
    '''
    x, y = ecdf_points(results_log, dimensions, target, targets_scale, targets_num)
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    log = np.linspace(100000, 100, 1240)
    ecdf_curve(log, 10, 0)



import numpy as np
import matplotlib.pyplot as plt
import math

def plot_ecdf_curves(data: dict):
    """
    Plot ECDF curves based on the given data dictionary.
    
    :param data: Dictionary where keys are optimization method names and values are lists of lists,
                 where each list corresponds to values from consecutive evaluations.
    """
    processed_logs = {}
    all_last_items = []

    # Step 1: Transform each log to contain the lowest value from the start to current value
    for method, logs in data.items():
        processed_logs[method] = []
        for log in logs:
            min_so_far = float('inf')
            new_log = []
            for value in log:
                min_so_far = min(min_so_far, value)
                new_log.append(min_so_far)
            # Apply log10 transformation to the log
            processed_log = [math.log10(v) if v > 0 else -float('inf') for v in new_log]
            processed_logs[method].append(processed_log)
            all_last_items.append(processed_log[-1])

    # Step 2: Find the highest and lowest values from all the last items of all logs
    low_value = min(all_last_items)
    high_value = max(all_last_items)

    # Step 3: Create 100 thresholds as linear interpolation between the low and high values
    thresholds = np.linspace(low_value, high_value, 100)

    # Step 4: For each log, count how many thresholds have been achieved (<=)
    ecdf_data = {}
    for method, logs in processed_logs.items():
        ecdf_logs = []
        for log in logs:
            ecdf = [(np.sum(np.array(log) <= threshold)) for threshold in thresholds]
            ecdf_logs.append(ecdf)
        # Average the ECDFs across all logs for the current method
        ecdf_avg = np.mean(ecdf_logs, axis=0)
        ecdf_data[method] = ecdf_avg

    # Step 5: Plot one line for each optimization method
    plt.figure(figsize=(10, 6))
    for method, ecdf_avg in ecdf_data.items():
        plt.plot(thresholds, ecdf_avg, label=method)

    plt.xlabel('Logarithmic Thresholds (log10)')
    plt.ylabel('ECDF (Fraction of Logs ≤ Threshold)')
    plt.title('ECDF Curves for Optimization Methods')
    plt.legend()
    plt.grid(True)
    plt.show()

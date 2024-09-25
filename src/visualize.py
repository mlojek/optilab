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
    return x, targets_reached


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

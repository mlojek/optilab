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

    for method, logs in data.items():
        processed_logs[method] = []
        for log in logs:
            min_so_far = float('inf')
            new_log = []
            for value in log:
                min_so_far = min(min_so_far, value)
                new_log.append(min_so_far)
            processed_log = [math.log10(v) if v > 0 else -float('inf') for v in new_log]
            processed_logs[method].append(processed_log)
            all_last_items.append(processed_log[-1])

    low_value = min(all_last_items)
    high_value = max(all_last_items)

    thresholds = np.linspace(low_value, high_value, 101)[1:]

    plt.figure(figsize=(10, 6))
    ecdf_data = {}
    for method, logs in processed_logs.items():
        ecdf_logs = []
        for log in logs:
            ecdf = [(np.sum(thresholds >= item)) for item in log]
            
            ecdf_logs.append(ecdf)
        ecdf_avg = np.mean(ecdf_logs, axis=0)
        ecdf_data[method] = ecdf_avg
        ecdf_data[method] = ecdf_logs[0]

    for method, ecdf_avg in ecdf_data.items():
        plt.plot(ecdf_avg, label=method)

    plt.xlabel('Number of function evaluations')
    plt.xscale('log')
    plt.ylabel('ECDF point pairs')
    plt.title('ECDF Curves')

    plt.legend()
    plt.grid(True)
    plt.show()
